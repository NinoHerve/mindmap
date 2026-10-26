"""
Module Name: metric/results.py
Description: Typed, self-describing statistical results for
`MetricSuite` -- `PairwiseResult` (wilcoxon/mcnemar/ttest/mannwhitneyu/
paired_ttest/ttest_ess/bootstrap comparisons), `ClusterResult` (cluster-permutation
comparisons), `NormalityResult`, `AnovaResult`, and `EstimateResult`
(point estimates with confidence intervals). `PairwiseResult`/
`ClusterResult` operate on an explicit list of `(group_a, group_b)` pairs
rather than being anchored to one fixed reference method, so a "reference
vs many competitors" comparison and an arbitrary many-to-many one are the
same shape -- a single-pair result and a many-pair result are the same
class.

Every result is self-describing: it carries every setting needed to
reproduce it (test, offset/alternative, pairing, filters, correction/alpha)
plus a `data_key` fingerprint of the exact rows it was computed from, so
`MetricSuite.verify(result)` can tell whether it still matches a suite's
data.

Author: Nino Herve
"""

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mindmap.core.io import atomic_replace, json_safe, read_npz_fast
from mindmap.core.stats import (
    paired_test, independent_test, correct_pvalues, test_normality, repeated_measures_anova, estimate, resolve_alternative,
    resolve_effect_sizes, effective_sample_size, bootstrap_draws, bootstrap_pvalue, TEST_EFFECT_SIZES, GROUP_ESTIMATORS,
    DIFFERENCE_ESTIMATORS, PAIRED_EFFECT_SIZES, INDEPENDENT_EFFECT_SIZES, NULL_EFFECT_VALUES, CI_METHODS, DEFAULT_N_BOOT,
    ESTIMATORS_WITH_CI, bootstrap_interval, bootstrap_bias, jackknife_values, jackknife_acceleration,
    check_bootstrap_interval,
)
from mindmap.analysis.stats import cluster_permutation_mne


def _freeze_filter(val):
    """
    Normalizes a scalar-or-list-or-None filter/ordering argument into a
    hashable tuple, stored on a result as-is: `None` stays `None`, a
    lone scalar (str, number, bool, ...) becomes a 1-tuple, anything
    array-like becomes a tuple of JSON-safe scalars, order preserved (some
    callers, e.g. `subject_order`, care about order; others sort before
    calling this).
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return tuple(json_safe(v) for v in val)
    return (json_safe(val),)


def _sorted_filter(val):
    """Same as `_freeze_filter`, but sorted first -- for filters where order genuinely doesn't matter (e.g. a column's filter values)."""
    if val is None or not isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return _freeze_filter(val)
    return _freeze_filter(sorted(val))


def _freeze_filters(filters):
    """
    Normalizes an arbitrary `{column_name: value(s)}` filter dict (as passed
    to `MetricSuite.get_filtered_df(**filters)`) into a hashable,
    key-order-independent tuple stored on a result -- `None`/empty
    stays `None`, otherwise `((key, frozen_values), ...)` sorted by key.
    """
    if not filters:
        return None
    return tuple(sorted((k, _sorted_filter(v)) for k, v in filters.items()))


def _hashable_column(s):
    """
    One column normalized so its fingerprint doesn't depend on incidental
    dtype changes -- e.g. an int coord turning float, or a str coord turning
    object, once another metric registered with different coords is
    concatenated alongside it in `MetricSuite.to_df`.
    """
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() == s.notna().sum():
        return numeric.astype(float)
    return s.astype(str)


def _data_key(df):
    """
    A short fingerprint of exactly the rows a result was computed from (one
    metric's `to_df(metrics=..., filters=...)`) -- independent of row order,
    of column order, and of all-NaN columns other metrics' coords add to
    the long table, so it only changes when *this* data changes.
    """
    df = df.dropna(axis=1, how="all")
    columns = sorted(df.columns, key=str)
    frame = pd.DataFrame({c: _hashable_column(df[c]) for c in columns})
    row_hashes = np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy())
    digest = hashlib.sha1(json.dumps([str(c) for c in columns]).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()[:16]


#: Public `test=` name -> the `paired_test`/`independent_test` name it actually runs.
#: `"paired_ttest"` -> `"ttest"` because `paired_test`'s own "ttest" already means the
#: paired/related version; `"ttest"` (independent, public) -> `"ttest_ind"` for the
#: same reason on the `independent_test` side. Whichever dict a name is found in
#: determines the whole computation path (paired vs independent) in `compute()`.
_PAIRED_CORE_TEST = {"wilcoxon": "wilcoxon", "mcnemar": "mcnemar", "paired_ttest": "ttest", "ttest_ess": "ttest_ess"}
_INDEPENDENT_CORE_TEST = {"ttest": "ttest_ind", "mannwhitneyu": "mannwhitneyu"}


def _collapse(df, group_by, match_col, unit_col):
    """
    Collapses `df` down to one "Value" per `(group_by, match_col)`,
    averaging over `unit_col` first (if given) and then anything else --
    two-stage so each `unit_col`-level observation is weighted equally
    (e.g. each of several trials), not each raw row (which would
    over-weight whichever unit happened to have more raw repetitions).
    """
    if unit_col is not None:
        df = df.groupby([group_by, match_col, unit_col], as_index=False)["Value"].mean()
    return df.groupby([group_by, match_col], as_index=False)["Value"].mean()


def _paired_wide(df, group_by, pairing_col, unit_col, subject_order):
    """
    One row per `pairing_col` value, one column per group -- the collapsed,
    `subject_order`-reindexed table every paired test reads its pairs from
    (each pair then drops its own incomplete rows). Shared with
    `StatAccessor.paired_differences` so the differences it returns are
    exactly the ones the test saw.
    """
    collapsed = _collapse(df, group_by, pairing_col, unit_col)
    wide = collapsed.pivot(index=pairing_col, columns=group_by, values="Value")
    if subject_order is not None:
        wide = wide.reindex(list(subject_order))
    return wide


def _effect_size_ci_method(name, n_eff=None):
    """
    How a test's effect size gets its interval: `"analytic"` when the
    estimator has one (see `ESTIMATORS_WITH_CI`), else `"bootstrap"` -- or
    `None` (no interval) for a `ttest_ess` effect size without an
    autocorrelation-corrected interval, since a plain bootstrap would ignore
    the serial correlation.
    """
    if n_eff is not None:
        return "analytic" if name == "mean_diff" else None
    return "analytic" if name in ESTIMATORS_WITH_CI else "bootstrap"


def _ci_settings(ci, effect_sizes, ess, ci_seed):
    """`(ci_methods, ci_seed, rng)` for a test's intervals -- the seed is drawn only when some interval is bootstrapped."""
    if not ci:
        return None, None, None
    n_eff = 1.0 if ess else None  # any value: only whether the test corrects for autocorrelation matters here
    ci_methods = {name: _effect_size_ci_method(name, n_eff) for name in effect_sizes}
    if "bootstrap" not in ci_methods.values():
        return ci_methods, None, None
    if ci_seed is None:
        ci_seed = int(np.random.SeedSequence().entropy)
    return ci_methods, int(ci_seed), np.random.default_rng(ci_seed)


def _effect_size_cis(x, y, names, paired, ci_level, rng, n_eff=None, zero_method=None):
    """
    `{"<name>_ci_low": ..., "<name>_ci_high": ...}` for each effect size in
    `names`, computed by `mindmap.core.stats.estimate` exactly as
    `suite.estimate` would (`ci_level` coverage; bootstrap draws from `rng`).
    """
    out = {}
    for name in names:
        method = _effect_size_ci_method(name, n_eff)
        low = high = np.nan
        if method is not None:
            est = estimate(
                x, y, estimator=name, paired=paired, alpha=1 - ci_level, ci=True,
                n_eff=n_eff if method == "analytic" else None, zero_method=zero_method or "wilcox",
                ci_method=method, seed=rng,
            )
            low, high = est["ci_low"], est["ci_high"]
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = low, high
    return out


def _compute_paired_rows(
    df, group_by, pairing_col, unit_col, comparisons, core_test, subject_order, offset, alternative, effect_sizes,
    zero_method=None, ci_level=None, rng=None,
):
    wide = _paired_wide(df, group_by, pairing_col, unit_col, subject_order)

    rows = []
    for a, b in comparisons:
        missing = {a, b} - set(wide.columns)
        if missing:
            raise ValueError(f"Group(s) {sorted(missing)} not found in '{group_by}'.")
        complete = wide[[a, b]].dropna()
        if complete.empty:
            raise ValueError(f"No subjects with complete data for pair ('{a}', '{b}').")
        x, y = complete[a].to_numpy(), complete[b].to_numpy()
        result = paired_test(
            x, y, test=core_test, offset=offset, alternative=alternative, effect_sizes=effect_sizes, zero_method=zero_method,
        )
        cis = {} if ci_level is None else _effect_size_cis(
            x, y, effect_sizes, True, ci_level, rng, n_eff=result.get("n_eff"), zero_method=zero_method,
        )
        rows.append({
            "group_a": a, "group_b": b, "n_pairs": result["n"], "n_zero": result["n_zero"],
            "n_a": result["n"], "n_b": result["n"],
            "statistic": result["statistic"], "p_value": result["p_value"],
            "n_eff": result.get("n_eff", np.nan), "p_corrected": np.nan, "significant": None,
            "n_a_only": result.get("n_x_only", np.nan), "n_b_only": result.get("n_y_only", np.nan),
            **result["effect_sizes"], **cis,
        })
    return rows


def _collapse_with_coord(df, group_by, pairing_col, coord, unit_col):
    """Like `_collapse`, but keeps `coord` distinct throughout -- for `ClusterResult`, which
    must never average across the coordinate it's testing along."""
    if unit_col is not None:
        df = df.groupby([group_by, pairing_col, coord, unit_col], as_index=False)["Value"].mean()
    return df.groupby([group_by, pairing_col, coord], as_index=False)["Value"].mean()


def _compute_independent_rows(df, group_by, unit_col, comparisons, core_test, offset, alternative, effect_sizes, ci_level=None, rng=None):
    collapsed = df if unit_col is None else df.groupby([group_by, unit_col], as_index=False)["Value"].mean()

    rows = []
    for a, b in comparisons:
        vals_a = collapsed.loc[collapsed[group_by] == a, "Value"]
        vals_b = collapsed.loc[collapsed[group_by] == b, "Value"]
        if vals_a.empty:
            raise ValueError(f"Group '{a}' not found in '{group_by}'.")
        if vals_b.empty:
            raise ValueError(f"Group '{b}' not found in '{group_by}'.")
        x, y = vals_a.to_numpy(), vals_b.to_numpy()
        result = independent_test(x, y, test=core_test, offset=offset, alternative=alternative, effect_sizes=effect_sizes)
        cis = {} if ci_level is None else _effect_size_cis(x, y, effect_sizes, False, ci_level, rng)
        rows.append({
            "group_a": a, "group_b": b, "n_pairs": np.nan, "n_a": result["n_a"], "n_b": result["n_b"],
            "statistic": result["statistic"], "p_value": result["p_value"],
            "n_eff": np.nan, "p_corrected": np.nan, "significant": None,
            **result["effect_sizes"], **cis,
        })
    return rows


def _group_samples(df, group_by, pairing_col, unit_col, groups=None):
    """
    Each group's values, collapsed exactly like the tests collapse (one per
    `pairing_col` value, or per `unit_col` value) -- `group_by=None` treats
    the whole frame as one group. `groups` defaults to every group present,
    sorted. Returns `(groups, [values of each group])`.
    """
    column = group_by if group_by is not None else "_all"
    frame = df if group_by is not None else df.assign(_all="all")
    if pairing_col is not None:
        frame = _collapse(frame, column, pairing_col, unit_col)
    elif unit_col is not None:
        frame = frame.groupby([column, unit_col], as_index=False)["Value"].mean()
    if groups is None:
        groups = [None] if group_by is None else sorted(frame[column].unique(), key=str)
    values = []
    for g in groups:
        v = frame["Value"] if g is None else frame.loc[frame[column] == g, "Value"]
        if v.empty:
            raise ValueError(f"Group '{g}' not found in '{group_by}'.")
        values.append(v.to_numpy())
    return groups, values


def _pair_samples(df, group_by, pairing_col, unit_col, comparisons, subject_order=None):
    """
    Per `(group_a, group_b)` pair, the two arrays an estimate (or a
    bootstrap) reads: the matched complete cases with `pairing_col` (each
    pair its own), else each group's sample -- collapsed exactly like the
    tests collapse.
    """
    samples = []
    if pairing_col is not None:
        wide = _paired_wide(df, group_by, pairing_col, unit_col, subject_order)
        for a, b in comparisons:
            missing = {a, b} - set(wide.columns)
            if missing:
                raise ValueError(f"Group(s) {sorted(missing)} not found in '{group_by}'.")
            complete = wide[[a, b]].dropna()
            samples.append((complete[a].to_numpy(), complete[b].to_numpy()))
        return samples
    collapsed = df if unit_col is None else df.groupby([group_by, unit_col], as_index=False)["Value"].mean()
    for pair in comparisons:
        values = []
        for g in pair:
            v = collapsed.loc[collapsed[group_by] == g, "Value"]
            if v.empty:
                raise ValueError(f"Group '{g}' not found in '{group_by}'.")
            values.append(v.to_numpy())
        samples.append(tuple(values))
    return samples


def resolve_bootstrap(
    ci_method="bootstrap", n_boot=None, seed=None, replicates=None, replicate_col=None, interval="percentile",
    acceleration=None, jackknife=None, jackknife_col=None,
):
    """
    The bootstrap settings a result records -- `None` for an analytic
    interval (where none of them may be given).

    Two sources of draws:

    - resampled here (default): `n_boot` resamples (default 10,000) of
      the units the estimate reads, seeded by `seed` -- drawn at random
      and recorded when omitted, so every result can be reproduced;
    - registered (`replicates`, the metric holding one value per draw,
      and `replicate_col`, the column numbering the draws): for
      resamples made upstream, e.g. spikes re-averaged per draw.
      `n_boot`/`seed` don't apply.

    `interval` is how the interval is read from the draws ("percentile",
    "basic" or "bca" -- see `mindmap.core.stats.bootstrap_interval`). BCa
    needs an acceleration: computed from the units when resampled here;
    with registered replicates, from a registered jackknife (`jackknife`,
    the metric holding one value per leave-one-out, and `jackknife_col`,
    the column numbering them), or given as `acceleration` (0: the
    bias-corrected "BC" interval).

    Returns
    -------
    dict | None
        `{"n_boot", "seed", "replicates", "replicate_col", "interval",
        "acceleration", "jackknife", "jackknife_col"}`.
    """
    if ci_method not in CI_METHODS:
        raise ValueError(f"ci_method={ci_method!r} isn't supported. Choose from {list(CI_METHODS)}.")
    given = [k for k, v in dict(
        n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col, acceleration=acceleration,
        jackknife=jackknife, jackknife_col=jackknife_col,
    ).items() if v is not None]
    if ci_method == "analytic":
        if given or interval != "percentile":
            raise ValueError(f"{given + (['interval'] if interval != 'percentile' else [])} only apply to ci_method='bootstrap'.")
        return None
    check_bootstrap_interval(interval)
    if (replicates is None) != (replicate_col is None):
        raise ValueError(
            "`replicates` (the metric holding one value per bootstrap draw) and `replicate_col` (the column "
            "numbering the draws) go together."
        )
    if (jackknife is None) != (jackknife_col is None):
        raise ValueError(
            "`jackknife` (the metric holding one value per leave-one-out) and `jackknife_col` (the column "
            "numbering them) go together."
        )
    if interval != "bca" and (acceleration is not None or jackknife is not None):
        raise ValueError("`acceleration`/`jackknife` only apply to the 'bca' interval.")
    if jackknife is not None and acceleration is not None:
        raise ValueError("Pass either a registered `jackknife` or an `acceleration`, not both.")
    if jackknife is not None and replicates is None:
        raise ValueError(
            "A registered `jackknife` goes with registered `replicates` -- with draws resampled here, the "
            "acceleration comes from leaving out each unit in turn."
        )
    if acceleration is not None:
        if isinstance(acceleration, (bool, np.bool_)) or not np.isscalar(acceleration) or not np.isfinite(acceleration):
            raise ValueError(f"`acceleration` must be a finite number, got {acceleration!r}.")
        acceleration = float(acceleration)
    if interval == "bca" and replicates is not None and jackknife is None and acceleration is None:
        raise ValueError(
            "The BCa acceleration needs leave-one-out values, which registered replicates don't provide -- "
            "register them as their own metric and pass `jackknife`/`jackknife_col`, or pass `acceleration=0` "
            "for the bias-corrected (BC) interval."
        )
    extra = {"interval": interval, "acceleration": acceleration, "jackknife": jackknife, "jackknife_col": jackknife_col}
    if replicates is not None:
        if n_boot is not None or seed is not None:
            raise ValueError("With `replicates`, the draws are the registered ones -- `n_boot`/`seed` don't apply.")
        return {"n_boot": None, "seed": None, "replicates": replicates, "replicate_col": replicate_col, **extra}
    n_boot = DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be at least 1, got {n_boot}.")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    elif isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"`seed` must be an int (it is stored on the result), got {type(seed).__name__}.")
    return {"n_boot": n_boot, "seed": int(seed), "replicates": None, "replicate_col": None, **extra}


def bootstrap_metrics(metric, boot):
    """The metrics a result with bootstrap settings `boot` reads: `metric`, then its registered replicates and jackknife."""
    boot = boot or {}
    return [metric] + [boot[k] for k in ("replicates", "jackknife") if boot.get(k) is not None]


def _split_replicates(df, metric, boot):
    """
    `df` -> `(observed rows, replicate rows)`: with registered replicates,
    `df` holds both metrics (so one `data_key` covers both); `None` for the
    replicate rows otherwise.
    """
    if boot is None or boot["replicates"] is None:
        return df, None
    replicates, col = boot["replicates"], boot["replicate_col"]
    if replicates == metric:
        raise ValueError(
            f"`replicates` must name a separate metric from {metric!r} -- register the observed value under "
            f"{metric!r} and the bootstrap draws under their own metric (e.g. '{metric}_boot')."
        )
    draws = df[df["Metric"] == replicates]
    if draws.empty:
        raise ValueError(f"No rows registered for the replicates metric {replicates!r} (after filters).")
    if col not in draws.columns or draws[col].isna().any():
        raise ValueError(f"Every row of {replicates!r} needs a '{col}' value numbering its bootstrap draw.")
    return df[df["Metric"] == metric], draws


def _jackknife_rows(df, metric, boot):
    """The registered jackknife rows of `df` (see `resolve_bootstrap`), or `None`."""
    if boot is None or boot.get("jackknife") is None:
        return None
    jackknife, col = boot["jackknife"], boot["jackknife_col"]
    if jackknife in (metric, boot["replicates"]):
        raise ValueError(f"`jackknife` must name its own metric, separate from {metric!r} and the replicates.")
    rows = df[df["Metric"] == jackknife]
    if rows.empty:
        raise ValueError(f"No rows registered for the jackknife metric {jackknife!r} (after filters).")
    if col not in rows.columns or rows[col].isna().any():
        raise ValueError(f"Every row of {jackknife!r} needs a '{col}' value numbering its leave-one-out.")
    return rows


def _replicate_draws(draws, replicate_col, samples, point):
    """
    The bootstrap distribution read from registered replicates: each value
    of `replicate_col` is one resample; `samples(its rows)` rebuilds the
    arrays exactly as for the observed data, and `point(*arrays)` turns
    each into one value. Returns an `(n_draws, n_items)` array.
    """
    values = []
    for r, sub in draws.groupby(replicate_col, sort=True):
        try:
            values.append([point(*arrays) for arrays in samples(sub)])
        except ValueError as e:
            raise ValueError(f"Bootstrap draw {replicate_col}={r!r}: {e}") from None
    return np.asarray(values, dtype=float)


def comparison_draws(
    samples, draws_df, group_by, pairing_col, unit_col, comparisons, subject_order, boot, estimator, zero_method,
):
    """
    The bootstrap distribution of `estimator` for each comparison, as a
    list of arrays in `comparisons` order -- the one place draws are made,
    so a test, its `estimate.from_result` and its plot read the very same
    draws.

    Registered replicates (`draws_df`) are re-paired per draw exactly like
    the observed `samples`; otherwise each comparison's units are
    resampled `boot["n_boot"]` times from one generator seeded with
    `boot["seed"]`, comparison after comparison.
    """
    paired = pairing_col is not None
    if draws_df is not None:
        registered = _replicate_draws(
            draws_df, boot["replicate_col"],
            lambda sub: _pair_samples(sub, group_by, pairing_col, unit_col, comparisons, subject_order),
            lambda x, y: estimate(x, y, estimator=estimator, paired=paired, ci=False, zero_method=zero_method)["estimate"],
        )
        return [registered[:, i] for i in range(len(comparisons))]
    rng = np.random.default_rng(boot["seed"])
    return [bootstrap_draws(x, y, estimator, paired, boot["n_boot"], rng, zero_method) for x, y in samples]


def comparison_accelerations(
    samples, jackknife_df, group_by, pairing_col, unit_col, comparisons, subject_order, boot, estimator, zero_method,
):
    """
    The BCa acceleration of each comparison, in `comparisons` order -- the
    given `boot["acceleration"]`; else from the registered jackknife
    (`jackknife_df`, re-paired per leave-one-out like the replicates);
    else from leaving out each unit of `samples` in turn. `None` per
    comparison for the other intervals.
    """
    if boot.get("interval", "percentile") != "bca":
        return [None] * len(comparisons)
    if boot.get("acceleration") is not None:
        return [boot["acceleration"]] * len(comparisons)
    paired = pairing_col is not None
    if jackknife_df is not None:
        values = _replicate_draws(
            jackknife_df, boot["jackknife_col"],
            lambda sub: _pair_samples(sub, group_by, pairing_col, unit_col, comparisons, subject_order),
            lambda x, y: estimate(x, y, estimator=estimator, paired=paired, ci=False, zero_method=zero_method)["estimate"],
        )
        return [jackknife_acceleration(values[:, i]) for i in range(len(comparisons))]
    return [jackknife_acceleration(*jackknife_values(x, y, estimator, paired, zero_method)) for x, y in samples]


#: The bias diagnostics a bootstrap table holds per pair -- see `mindmap.core.stats.bootstrap_bias`.
BOOTSTRAP_BIAS_COLUMNS = ["boot_bias", "boot_se", "bias_se_ratio", "z0"]
#: |bias| / SE above which the percentile interval is noticeably off (Efron & Tibshirani 1993, ch. 10).
BIAS_SE_WARNING = 0.25


def _compute_bootstrap_rows(
    df, draws_df, group_by, pairing_col, unit_col, comparisons, subject_order, offset, alternative, boot, zero_method,
    ci_level=None, jackknife_df=None,
):
    estimator = boot["estimator"]
    interval = boot.get("interval", "percentile")
    paired = pairing_col is not None
    zero_method = zero_method or "wilcox"
    point = lambda x, y: estimate(x, y, estimator=estimator, paired=paired, ci=False, zero_method=zero_method)
    samples = _pair_samples(df, group_by, pairing_col, unit_col, comparisons, subject_order)
    all_draws = comparison_draws(
        samples, draws_df, group_by, pairing_col, unit_col, comparisons, subject_order, boot, estimator, zero_method,
    )
    accelerations = comparison_accelerations(
        samples, jackknife_df, group_by, pairing_col, unit_col, comparisons, subject_order, boot, estimator, zero_method,
    )
    null = NULL_EFFECT_VALUES.get(estimator, 0.0) + offset

    rows = []
    for i, ((a, b), (x, y)) in enumerate(zip(comparisons, samples)):
        if paired and not x.shape[0]:
            raise ValueError(f"No subjects with complete data for pair ('{a}', '{b}').")
        out = point(x, y)
        draws = all_draws[i]
        diagnostics = bootstrap_bias(draws, out["estimate"])
        if interval == "bca" and np.isinf(diagnostics["z0"]):
            raise ValueError(
                f"Every bootstrap draw of pair ('{a}', '{b}') lies on one side of its estimate, so the BCa interval "
                "is undefined -- use ci_method='basic'."
            )
        n_a, n_b = (out["n"], out["n"]) if paired else (out["n_a"], out["n_b"])
        row = {
            "group_a": a, "group_b": b, "n_pairs": n_a if paired else np.nan, "n_zero": np.nan, "n_a": n_a, "n_b": n_b,
            "statistic": out["estimate"],
            "p_value": bootstrap_pvalue(draws, null, alternative, interval, out["estimate"], accelerations[i] or 0.0),
            "n_eff": np.nan, "p_corrected": np.nan, "significant": None,
            "n_boot": int(np.sum(~np.isnan(draws))), **diagnostics,
            "acceleration": np.nan if accelerations[i] is None else accelerations[i], estimator: out["estimate"],
        }
        if ci_level is not None:
            row[f"{estimator}_ci_low"], row[f"{estimator}_ci_high"] = bootstrap_interval(
                draws, out["estimate"], 1 - ci_level, interval, accelerations[i] or 0.0,
            )
        rows.append(row)
    return rows


#: The unstandardized estimate that goes with each test -- `PairwiseResult.estimator`.
_NATURAL_ESTIMATOR = {
    "wilcoxon": "hodges_lehmann", "mannwhitneyu": "hodges_lehmann", "mcnemar": "prop_diff",
    "paired_ttest": "mean_diff", "ttest_ess": "mean_diff", "ttest": "mean_diff",
}

#: Which effect sizes each public `test=` may report -- the first is its default. Standardized
#: and unstandardized ("mean_diff", "hodges_lehmann") alike, all describing the raw difference
#: (never shifted by the offset) -- see `mindmap.core.stats.effect_size`.
_EFFECT_SIZES = {
    test: TEST_EFFECT_SIZES[core] for test, core in {**_PAIRED_CORE_TEST, **_INDEPENDENT_CORE_TEST}.items()
}
#: Effect sizes on a probability scale (0.5 = no effect): swapping the two groups maps
#: v -> 1 - v, not v -> -v like the others.
_PROBABILITY_EFFECT_SIZES = {"cles"}
#: Effect sizes on a ratio scale (1 = no effect): swapping the two groups maps v -> 1 / v.
_RATIO_EFFECT_SIZES = {"odds_ratio"}


def _mirror(name, values):
    """`values` of effect size `name` with the two groups swapped: 1 - v (probability), 1 / v (ratio), -v otherwise."""
    values = np.asarray(values, dtype=float)
    if name in _PROBABILITY_EFFECT_SIZES:
        return 1 - values
    if name in _RATIO_EFFECT_SIZES:
        with np.errstate(divide="ignore"):
            return 1 / values
    return -values

_PAIRWISE_BASE_COLUMNS = [
    "group_a", "group_b", "n_pairs", "n_zero", "n_a", "n_b", "statistic", "p_value", "n_eff", "p_corrected", "significant",
]
#: Extra columns some tests add after the base ones: McNemar's discordant counts (pairs where
#: only group_a's / only group_b's outcome is 1), and the bootstrap's number of valid draws,
#: bias diagnostics and BCa acceleration (NaN for the other intervals).
_TEST_COLUMNS = {"mcnemar": ["n_a_only", "n_b_only"], "bootstrap": ["n_boot", *BOOTSTRAP_BIAS_COLUMNS, "acceleration"]}


def _pairwise_columns(test, effect_sizes, ci=False):
    """A pairwise table's columns: the base ones, the test's own, then each effect size (with its interval when `ci`)."""
    effects = [c for name in effect_sizes for c in ((name, f"{name}_ci_low", f"{name}_ci_high") if ci else (name,))]
    return [*_PAIRWISE_BASE_COLUMNS, *_TEST_COLUMNS.get(test, []), *effects]

_FLIPPED_ALTERNATIVE = {"two-sided": "two-sided", "greater": "less", "less": "greater"}

#: Bumped whenever an effect size's definition changes. Version 1 (folders saved without the
#: field): wilcoxon's "rank_biserial" was the sign-based (n_pos - n_neg) / n, and
#: mannwhitneyu's had the opposite sign. Version 2: textbook definitions, but measured against
#: the offset. Version 3: effect sizes never include the offset; each result stores only the
#: effect sizes requested (`effect_sizes`).
_EFFECT_SIZE_VERSION = 3
#: Tests whose stored effect sizes differ between version 1 and the current definitions.
_TESTS_WITH_CHANGED_EFFECT_SIZES = {"wilcoxon", "mannwhitneyu"}


def _legacy_effect_sizes(table, test):
    """The effect sizes an older (version < 3) table holds: its non-empty named columns, or its single `effect_size`/`effect_size_type` pair."""
    names = tuple(n for n in _EFFECT_SIZES.get(test, ()) if n in table.columns and table[n].notna().any())
    if names:
        return names
    if "effect_size_type" in table.columns and not table.empty:
        name = table["effect_size_type"].iloc[0]
        table[name] = table["effect_size"]
        return (name,)
    return (_EFFECT_SIZES[test][0],)


class PairwiseResult:
    """
    The result of comparing two or more groups on one metric -- one row per
    pair compared, with the test statistic, p-value, and effect sizes.
    Returned by `MetricSuite.stat.wilcoxon`/`.mcnemar`/`.ttest`/
    `.mannwhitneyu`/`.paired_ttest`/`.ttest_ess`/`.bootstrap`; you'll
    rarely construct one directly.

    !!! note "Technical details"
        Covers two genuinely different comparison shapes with one class,
        distinguished by `pairing_col`:

        - Paired (`wilcoxon`/`mcnemar`/`paired_ttest`/`ttest_ess`, `pairing_col` set):
          rows are matched one-to-one across groups by `pairing_col` (e.g.
          the same region in every method) -- `table["n_pairs"]` is one
          shared count, since a paired test needs the exact same subjects
          on both sides. `table["n_zero"]` counts the pairs whose
          difference (minus the offset) is exactly 0: a Wilcoxon test with
          `zero_method="wilcox"` drops them from `n_pairs`, `"pratt"`/
          `"zsplit"` keep them (see `zero_method`).
        - Independent (`ttest`/`mannwhitneyu`, `pairing_col` is `None`): no
          correspondence between groups is assumed -- `table["n_a"]`/
          `"n_b"` can differ. `n_a`/`n_b` are also filled in for paired
          rows (both equal to `n_pairs` there), so either column is always
          meaningful regardless of which family produced a given row.

        Every pair is tested on its own complete-case subjects (paired) or
        its own collapsed sample (independent), independent of any other
        pair in the same object; `correct` treats all of them as one
        family.

        `test="bootstrap"` is either shape (paired with a `pairing_col`):
        its `statistic` and only effect size are the bootstrapped
        `estimator`, its p-value comes from the bootstrap distribution
        (see `mindmap.core.stats.bootstrap_pvalue`), and `bootstrap`
        records the settings (`estimator`, `n_boot`, `seed`, or the
        registered `replicates`/`replicate_col`); `table["n_boot"]`
        counts the valid draws. `bootstrap` is `None` for the other tests.

        With `ci` (the default from `suite.stat`), every effect size has a
        `ci_level` confidence interval in `<name>_ci_low`/`<name>_ci_high`,
        computed by the same functions as `suite.estimate`: the
        estimator's analytic interval when it has one, else a percentile
        bootstrap (`ci_seed` records its seed); a bootstrap test reads it
        from its own draws. `ttest_ess`'s standardized effect sizes get
        none (a plain bootstrap would ignore the autocorrelation).
        `ci_methods` records, per effect size, `"analytic"`,
        `"bootstrap"` or `None`. Like the effect sizes, the intervals
        describe the raw difference, never shifted by the offset.

        Self-describing: besides the test and its pairing settings, a
        result records the `offset`/`alternative` every pair was tested
        against, the Wilcoxon `zero_method` (`None` for the other tests),
        the `correction`/`alpha` applied (`None` until
        `correct`-ed), the `effect_sizes` it reports, and a `data_key`
        fingerprint of the rows it was computed from (see
        `MetricSuite.verify`). `EFFECT_SIZES` declares which effect sizes
        each test may report; each requested one is a column of `table`,
        the first being the default in `vs()` and the plots. Effect sizes
        describe the raw difference -- the offset only changes the
        p-value.
    """

    KIND = "pairwise"
    EFFECT_SIZES = _EFFECT_SIZES

    def __init__(
        self, metric, group_by, test, unit_col, pairing_col, subject_order, filters,
        table: pd.DataFrame, effect_sizes, computed_at=None, offset=0.0, alternative="two-sided",
        correction=None, alpha=None, data_key=None, effect_size_version=_EFFECT_SIZE_VERSION, zero_method=None,
        bootstrap=None, ci=False, ci_level=None, ci_methods=None, ci_seed=None,
    ):
        self.metric = metric
        self.group_by = group_by
        self.test = test
        self.zero_method = zero_method
        self.bootstrap = dict(bootstrap) if bootstrap is not None else None
        self.ci = bool(ci)
        self.ci_level = ci_level
        self.ci_methods = dict(ci_methods) if ci_methods is not None else None
        self.ci_seed = ci_seed
        self.unit_col = unit_col
        self.pairing_col = pairing_col
        self.subject_order = subject_order
        self.filters = filters
        self.effect_sizes = tuple(effect_sizes)
        self.offset = float(offset)
        self.alternative = alternative
        self.correction = correction
        self.alpha = alpha
        self.data_key = data_key
        self.effect_size_version = effect_size_version
        self.table = table.reset_index(drop=True)
        self.computed_at = computed_at or datetime.now()

    @property
    def estimator(self):
        """
        The unstandardized estimate that goes with this test --
        `"hodges_lehmann"` for wilcoxon/mannwhitneyu, `"mean_diff"` for the
        t-tests, `"prop_diff"` for mcnemar -- or, for a bootstrap test, the
        estimator it bootstrapped. What `suite.estimate.from_result`
        estimates by default.
        """
        if self.test == "bootstrap":
            return self.bootstrap["estimator"]
        return _NATURAL_ESTIMATOR[self.test]

    @property
    def has_outdated_effect_sizes(self):
        """Whether this result was saved under an older effect-size definition that changes its values -- see `_EFFECT_SIZE_VERSION`."""
        if self.effect_size_version < 2 and self.test in _TESTS_WITH_CHANGED_EFFECT_SIZES:
            return True
        return self.effect_size_version < 3 and self.offset != 0

    def _replace(self, **changes):
        """A copy of this result with some fields swapped -- every field carried over unless overridden."""
        fields = dict(
            metric=self.metric, group_by=self.group_by, test=self.test, unit_col=self.unit_col,
            pairing_col=self.pairing_col, subject_order=self.subject_order, filters=self.filters,
            table=self.table, effect_sizes=self.effect_sizes, computed_at=self.computed_at, offset=self.offset,
            alternative=self.alternative, correction=self.correction, alpha=self.alpha, data_key=self.data_key,
            effect_size_version=self.effect_size_version, zero_method=self.zero_method, bootstrap=self.bootstrap,
            ci=self.ci, ci_level=self.ci_level, ci_methods=self.ci_methods, ci_seed=self.ci_seed,
        )
        fields.update(changes)
        return PairwiseResult(**fields)

    @classmethod
    def compute(
        cls, df, metric, group_by, comparisons, test, unit_col=None, pairing_col=None, subject_order=None,
        filters=None, offset=0.0, alternative=None, effect_sizes=None, zero_method=None, bootstrap=None,
        ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Runs the actual tests and builds a `PairwiseResult` from scratch.
        Most users won't call this directly -- `MetricSuite.stat.wilcoxon`/
        etc. do -- but it's here if you want the raw computation without
        going through a suite.

        !!! note "Technical details"
            `test` alone decides paired vs. independent (see the class
            docstring) -- `pairing_col` is required for `"wilcoxon"`/
            `"paired_ttest"`/`"ttest_ess"`, and must be omitted for
            `"ttest"`/`"mannwhitneyu"` (there's nothing to align across
            groups for an independent-samples test). `unit_col`, when
            given, is a dedup axis -- rows are first collapsed to one
            value per `(group_by, unit_col)` (paired: per
            `(group_by, pairing_col, unit_col)`, then to
            `(group_by, pairing_col)`) before testing, so e.g. several
            repeated runs from one region don't get treated as several
            independent pieces of evidence. `filters` isn't used for
            computation here (the filtering already happened in `df`) --
            it's carried through purely as bookkeeping, and is expected to
            already be frozen (via `_freeze_filters`) by the caller.
            `data_key` is fingerprinted from `df` itself.

        Parameters
        ----------
        df: pd.DataFrame
            Long-format table (one metric's worth), at least
            `[group_by, "Value"]` (plus `unit_col`/`pairing_col` if given).

        metric, group_by: str
            Identity fields carried onto the returned result.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test.

        test: str
            `"wilcoxon"` | `"mcnemar"` | `"ttest"` | `"mannwhitneyu"` |
            `"paired_ttest"` | `"ttest_ess"` | `"bootstrap"`.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        pairing_col: str, optional
            Cross-group matching axis -- required for the three paired
            tests, must be omitted for the two independent ones.

        subject_order: list, optional
            Only matters for `test="ttest_ess"`.

        filters: optional
            Bookkeeping only -- see above.

        offset: float, optional
            Each pair tests `group_a - group_b` against this value
            (default 0) -- see `mindmap.core.stats.paired_test`.

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"`; defaults to
            `"two-sided"` when `offset` is 0, `"greater"` otherwise.

        effect_sizes: str | tuple[str, ...], optional
            Any of `EFFECT_SIZES[test]` (default: its first).

        zero_method: str, optional
            `"wilcoxon"` only: `"wilcox"` (default), `"pratt"` or
            `"zsplit"` -- see `mindmap.core.stats.paired_test`. Also a
            paired bootstrapped `"rank_biserial"`'s ranking of zeros.

        bootstrap: dict, optional
            `test="bootstrap"` only (required there): `resolve_bootstrap`'s
            settings plus `"estimator"`, the difference estimator
            bootstrapped. With registered `replicates`, `df` holds both
            metrics.

        ci: bool, optional
            Add a `ci_level` interval to every effect size (default True)
            -- see the class docstring.

        ci_level: float, optional
            The intervals' coverage (default 0.95). Independent of the
            significance `alpha`: a one-sided test at 0.05 matches a 90%
            interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals (for the effect sizes without an
            analytic one). Drawn at random when omitted and needed; stored
            as `ci_seed`.

        Returns
        -------
        PairwiseResult
        """
        if ci and not 0 < ci_level < 1:
            raise ValueError(f"`ci_level` must be in (0, 1), got {ci_level!r}.")
        if ci_seed is not None and (isinstance(ci_seed, (bool, np.bool_)) or not isinstance(ci_seed, (int, np.integer))):
            raise TypeError(f"`ci_seed` must be an int (it is stored on the result), got {type(ci_seed).__name__}.")
        level = ci_level if ci else None
        if test == "bootstrap":
            if bootstrap is None or bootstrap.get("estimator") is None:
                raise ValueError("test='bootstrap' needs its `bootstrap` settings, including the estimator (see suite.stat.bootstrap).")
            bootstrap = dict(bootstrap)
            estimator = bootstrap["estimator"]
            if estimator == "rank_biserial" and pairing_col is not None:
                zero_method = "wilcox" if zero_method is None else zero_method
            elif zero_method not in (None, "wilcox"):
                raise ValueError("`zero_method` only applies to a paired bootstrapped 'rank_biserial'.")
            else:
                zero_method = None
        elif bootstrap is not None:
            raise ValueError(f"`bootstrap` settings only apply to test='bootstrap', not {test!r}.")
        elif test == "wilcoxon":
            zero_method = "wilcox" if zero_method is None else zero_method
        elif zero_method is not None:
            raise ValueError(f"`zero_method` only applies to test='wilcoxon', not {test!r}.")
        offset = float(offset)
        alternative = resolve_alternative(offset, alternative)
        if test == "bootstrap":
            allowed = PAIRED_EFFECT_SIZES if pairing_col is not None else INDEPENDENT_EFFECT_SIZES
            if estimator not in allowed:
                kind = "paired (with a `pairing_col`)" if pairing_col is not None else "independent (no `pairing_col`)"
                raise ValueError(f"estimator={estimator!r} isn't defined for {kind} samples. Choose from {list(allowed)}.")
            if offset != 0 and estimator in NULL_EFFECT_VALUES:
                raise ValueError(
                    f"{estimator!r} is tested against its own null ({NULL_EFFECT_VALUES[estimator]:g}) -- it takes no `offset`."
                )
            if effect_sizes is not None and tuple([effect_sizes] if isinstance(effect_sizes, str) else effect_sizes) != (estimator,):
                raise ValueError("A bootstrap test reports the estimator it bootstrapped -- pass `estimator`, not `effect_sizes`.")
            effect_sizes = (estimator,)
            interval = bootstrap.get("interval", "percentile")
            check_bootstrap_interval(interval, estimator)
            observed, draws = _split_replicates(df, metric, bootstrap)
            rows = _compute_bootstrap_rows(
                observed, draws, group_by, pairing_col, unit_col, comparisons, subject_order, offset, alternative,
                bootstrap, zero_method, level, _jackknife_rows(df, metric, bootstrap),
            )
            ci_methods = {estimator: "bootstrap" if interval == "percentile" else f"bootstrap_{interval}"} if ci else None
            ci_seed = None
        elif test in _PAIRED_CORE_TEST:
            if pairing_col is None:
                raise ValueError(f"`pairing_col` is required for test={test!r} (a paired test).")
            core_test = _PAIRED_CORE_TEST[test]
            effect_sizes = resolve_effect_sizes(core_test, effect_sizes)
            ci_methods, ci_seed, rng = _ci_settings(ci, effect_sizes, test == "ttest_ess", ci_seed)
            rows = _compute_paired_rows(
                df, group_by, pairing_col, unit_col, comparisons, core_test, subject_order, offset, alternative, effect_sizes,
                zero_method, level, rng,
            )
        elif test in _INDEPENDENT_CORE_TEST:
            if pairing_col is not None:
                raise ValueError(f"test={test!r} is an independent-samples test and doesn't take `pairing_col`.")
            core_test = _INDEPENDENT_CORE_TEST[test]
            effect_sizes = resolve_effect_sizes(core_test, effect_sizes)
            ci_methods, ci_seed, rng = _ci_settings(ci, effect_sizes, False, ci_seed)
            rows = _compute_independent_rows(
                df, group_by, unit_col, comparisons, core_test, offset, alternative, effect_sizes, level, rng,
            )
        else:
            known = sorted({**_PAIRED_CORE_TEST, **_INDEPENDENT_CORE_TEST, "bootstrap": None})
            raise ValueError(f"Unknown test {test!r}. Choose from {known}.")

        table = pd.DataFrame(rows, columns=_pairwise_columns(test, effect_sizes, ci))
        return cls(
            metric=metric, group_by=group_by, test=test, unit_col=unit_col, pairing_col=pairing_col,
            subject_order=_freeze_filter(subject_order), filters=filters, table=table, effect_sizes=effect_sizes,
            offset=offset, alternative=alternative, data_key=_data_key(df), zero_method=zero_method, bootstrap=bootstrap,
            ci=ci, ci_level=level, ci_methods=ci_methods, ci_seed=ci_seed,
        )

    def correct(self, method="fdr_bh", alpha=0.05):
        """
        Adjusts p-values for multiple comparisons across every pair in this
        result, pooled into one family.

        !!! note "Technical details"
            Returns a **new** `PairwiseResult` -- never mutates `self` --
            so the same result can be re-corrected with a different
            `method`/`alpha` without recomputing any test. The new result
            records `method`/`alpha` as `correction`/`alpha`.

        Parameters
        ----------
        method: str, optional
            Correction method (default `"fdr_bh"`): `"fdr_bh"`,
            `"fdr_by"`, `"bonferroni"`, or `"holm"`.

        alpha: float, optional
            Significance threshold (default 0.05).

        Returns
        -------
        PairwiseResult
        """
        table = self.table.copy()
        if not table.empty:
            p_corrected, significant = correct_pvalues(table["p_value"].to_numpy(), method=method, alpha=alpha)
            table["p_corrected"] = p_corrected
            table["significant"] = significant
        return self._replace(table=table, correction=method, alpha=alpha)

    def vs(self, reference, higher_is_better=True):
        """
        One row per competitor of `reference` -- its p-values and every
        effect size this result reports, oriented so that **positive =
        `reference` better**, whichever order each pair was stored in, plus
        the offset/alternative each test used, expressed in that same
        orientation.

        !!! note "Technical details"
            Competitors are every group paired with `reference` in
            `self.table`, in table order; pairs not involving `reference`
            are ignored. Each pair was tested as `group_a - group_b`
            against `self.offset`; the result is re-expressed as
            `reference - competitor`: when `reference` is `group_b`, the
            effect sizes and the offset are negated and `"greater"`/
            `"less"` swap, and the same happens once more when
            `higher_is_better=False` (e.g. an error metric). Every effect
            size is antisymmetric in the two groups (`cles`, on a
            probability scale, maps to `1 - cles`; `odds_ratio`, on a ratio
            scale, to `1 / odds_ratio`), and a one-sided test
            of `a - b > offset` *is* the test of `b - a < -offset`, so the
            re-expression is exact -- the p-values never change. So
            `alternative="greater"` here always reads "tested whether
            `reference` is better by more than `offset`" -- e.g. compare a
            `hodges_lehmann` column with `offset` directly. Nothing is
            recomputed: the values come straight from `self.table`, so
            `p_corrected`/`significant` are `NaN`/`None` on a result that
            was never `correct`-ed.

        Parameters
        ----------
        reference: str
            The group every competitor is read against.

        higher_is_better: bool, optional
            Whether a larger metric value is better (default True).

        Returns
        -------
        pd.DataFrame
            Indexed by competitor (index named after `group_by`), columns
            `p_value`, `p_corrected`, `significant`, one column per
            effect size in `self.effect_sizes`, `offset`, `alternative`,
            `n_pairs`, `n_zero`/`n_eff` when any row has them, each effect
            size's `<name>_ci_low`/`<name>_ci_high` when the result has
            intervals (oriented the same way, so the bounds swap), for
            McNemar `n_reference_only`/`n_competitor_only` (the
            discordant pairs where only that side's outcome is 1), and for
            a bootstrap test `n_boot`, its bias diagnostics (`boot_bias`,
            `boot_se`, `bias_se_ratio`, `z0`) and `acceleration`.
        """
        if not isinstance(higher_is_better, bool):
            raise TypeError(f"`higher_is_better` must be a bool, got {type(higher_is_better).__name__}.")

        t = self.table
        as_a = t["group_a"] == reference
        as_b = t["group_b"] == reference
        if not (as_a | as_b).any():
            groups = sorted({*t["group_a"], *t["group_b"]}, key=str)
            raise ValueError(
                f"reference {reference!r} appears in no pair of this result "
                f"(groups compared: {groups})."
            )

        rows = t[as_a | as_b]
        flipped = (rows["group_a"] != reference).to_numpy() ^ (not higher_is_better)
        sign = np.where(flipped, -1.0, 1.0)

        out = pd.DataFrame({
            "p_value": rows["p_value"].to_numpy(),
            "p_corrected": rows["p_corrected"].to_numpy(),
            "significant": rows["significant"].to_numpy(),
        }, index=pd.Index(np.where(rows["group_a"] == reference, rows["group_b"], rows["group_a"]), name=self.group_by))
        for name in self.effect_sizes:
            raw = rows[name].to_numpy(dtype=float)
            out[name] = np.where(flipped, _mirror(name, raw), raw)
            if self.ci:
                low, high = rows[f"{name}_ci_low"].to_numpy(dtype=float), rows[f"{name}_ci_high"].to_numpy(dtype=float)
                out[f"{name}_ci_low"] = np.where(flipped, _mirror(name, high), low)
                out[f"{name}_ci_high"] = np.where(flipped, _mirror(name, low), high)
        out["offset"] = self.offset * sign + 0.0  # + 0.0: no "-0.0" display
        out["alternative"] = [_FLIPPED_ALTERNATIVE[self.alternative] if f else self.alternative for f in flipped]
        out["n_pairs"] = rows["n_pairs"].to_numpy()
        if rows["n_zero"].notna().any():
            out["n_zero"] = rows["n_zero"].to_numpy()
        if rows["n_eff"].notna().any():
            out["n_eff"] = rows["n_eff"].to_numpy()
        if "n_a_only" in rows.columns:
            as_a = (rows["group_a"] == reference).to_numpy()
            out["n_reference_only"] = np.where(as_a, rows["n_a_only"], rows["n_b_only"])
            out["n_competitor_only"] = np.where(as_a, rows["n_b_only"], rows["n_a_only"])
        if "n_boot" in rows.columns:
            out["n_boot"] = rows["n_boot"].to_numpy()
            # The bias, z0 and acceleration change sign with the orientation (the draws are negated), the SE doesn't.
            for name in ("boot_bias", "boot_se", "bias_se_ratio", "z0", "acceleration"):
                if name in rows.columns:
                    values = rows[name].to_numpy(dtype=float)
                    out[name] = values if name == "boot_se" else np.where(flipped, -values, values) + 0.0
        return out

    def __repr__(self):
        hypothesis = "" if self.offset == 0 and self.alternative == "two-sided" else f", offset={self.offset:g}, alternative={self.alternative!r}"
        if self.zero_method not in (None, "wilcox"):
            hypothesis += f", zero_method={self.zero_method!r}"
        if self.bootstrap is not None:
            hypothesis += f", estimator={self.bootstrap['estimator']!r}, {_bootstrap_label(self.bootstrap)}"
        correction = "" if self.correction is None else f", correction={self.correction!r}, alpha={self.alpha:g}"
        lines = [f"PairwiseResult(metric={self.metric!r}, test={self.test!r}{hypothesis}{correction}, n_comparisons={len(self.table)})"]
        for _, row in self.table.iterrows():
            has_corrected = pd.notna(row["p_corrected"])
            p = row["p_corrected"] if has_corrected else row["p_value"]
            tag = "*" if has_corrected and row["significant"] else ""
            n_desc = f"n={row['n_pairs']:.0f}" if pd.notna(row["n_pairs"]) else f"n_a={row['n_a']:.0f}, n_b={row['n_b']:.0f}"
            effects = ", ".join(f"{name}={row[name]:.3g}{self._ci_text(row, name)}" for name in self.effect_sizes)
            lines.append(f"  {row['group_a']} vs {row['group_b']}: p={p:.4g}{tag}, {effects}, {n_desc}")
        return "\n".join(lines)

    def _ci_text(self, row, name):
        """` [low, high]` for the repr, ` [no CI]` where the effect size has none, `""` without intervals."""
        if not self.ci:
            return ""
        low, high = row[f"{name}_ci_low"], row[f"{name}_ci_high"]
        return " [no CI]" if pd.isna(low) and pd.isna(high) else f" [{low:.3g}, {high:.3g}]"

    def save(self, folder: str | Path):
        """
        Saves this result to a folder, so it can be reloaded later with
        `load()` without recomputing anything.

        !!! note "Technical details"
            Writes `identity.json` (metadata) + `table.json` (the
            comparison table) -- mirrors the JSON-metadata convention in
            `mindmap/analysis/simulation/components.py`.

        Parameters
        ----------
        folder: str | Path
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        meta = {
            "kind": self.KIND, "metric": self.metric, "group_by": self.group_by, "test": self.test,
            "unit_col": self.unit_col, "pairing_col": self.pairing_col,
            "subject_order": list(self.subject_order) if self.subject_order is not None else None,
            "filters": _filters_to_json(self.filters),
            "offset": self.offset, "alternative": self.alternative, "effect_sizes": list(self.effect_sizes),
            "zero_method": self.zero_method, "bootstrap": self.bootstrap,
            "ci": self.ci, "ci_level": self.ci_level, "ci_methods": self.ci_methods, "ci_seed": self.ci_seed,
            "correction": self.correction, "alpha": self.alpha, "data_key": self.data_key,
            "effect_size_version": self.effect_size_version,
            "computed_at": self.computed_at.isoformat(),
        }

        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        atomic_replace(folder/"identity.json", _write_meta)

        records = self.table.to_dict(orient="records")

        def _write_table(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, default=json_safe)
        atomic_replace(folder/"table.json", _write_table)

    @classmethod
    def load(cls, folder: str | Path):
        """
        Reloads a result previously saved with `save()`.

        !!! note "Technical details"
            Folders saved by older versions still load: `offset=0`,
            `alternative="two-sided"` and `data_key`/`correction`/`alpha`
            `None` when missing, and `effect_sizes` read back from whichever
            effect-size columns the table holds. A result whose stored
            effect sizes differ from the current definitions (see
            `has_outdated_effect_sizes`: an old wilcoxon/mannwhitneyu
            rank-biserial, or effect sizes measured against a non-zero
            offset) loads with a warning -- recompute it.

        Parameters
        ----------
        folder: str | Path

        Returns
        -------
        PairwiseResult
        """
        folder = Path(folder)
        with open(folder/"identity.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(folder/"table.json", "r", encoding="utf-8") as f:
            records = json.load(f)

        table = pd.DataFrame(records)
        if "effect_sizes" in meta:
            effect_sizes = tuple(meta["effect_sizes"])
        else:
            effect_sizes = _legacy_effect_sizes(table, meta["test"])
        table = table.reindex(columns=_pairwise_columns(meta["test"], effect_sizes, meta.get("ci", False)))
        result = cls(
            metric=meta["metric"], group_by=meta["group_by"], test=meta["test"],
            unit_col=meta.get("unit_col"), pairing_col=meta.get("pairing_col"),
            subject_order=tuple(meta["subject_order"]) if meta["subject_order"] is not None else None,
            filters=_filters_from_json(meta.get("filters")),
            table=table, effect_sizes=effect_sizes, computed_at=datetime.fromisoformat(meta["computed_at"]),
            offset=meta.get("offset", 0.0), alternative=meta.get("alternative", "two-sided"),
            correction=meta.get("correction"), alpha=meta.get("alpha"), data_key=meta.get("data_key"),
            effect_size_version=meta.get("effect_size_version", 1),
            zero_method=meta.get("zero_method", "wilcox" if meta["test"] == "wilcoxon" else None),
            bootstrap=meta.get("bootstrap"), ci=meta.get("ci", False), ci_level=meta.get("ci_level"),
            ci_methods=meta.get("ci_methods"), ci_seed=meta.get("ci_seed"),
        )
        if result.has_outdated_effect_sizes:
            warnings.warn(
                f"{folder} holds a {result.test} result saved with an older effect-size definition "
                "(see mindmap.metric.results._EFFECT_SIZE_VERSION) -- its effect sizes are outdated; recompute it.",
                stacklevel=2,
            )
        return result


def _bootstrap_label(boot):
    """`n_boot=10000` or `replicates='rho_boot'` -- where a result's bootstrap draws came from."""
    return f"replicates={boot['replicates']!r}" if boot.get("replicates") else f"n_boot={boot['n_boot']}"


def _filters_to_json(filters):
    """`_freeze_filters`' output (`((key, (v1, v2, ...)), ...)` or `None`) -> JSON-safe nested lists."""
    if not filters:
        return None
    return [[k, list(v) if v is not None else None] for k, v in filters]


def _filters_from_json(filters):
    """Inverse of `_filters_to_json`."""
    if not filters:
        return None
    return tuple((k, tuple(v) if v is not None else None) for k, v in filters)


class ClusterResult:
    """
    The result of a cluster-permutation test between two or more groups
    along a coordinate (e.g. SNR, time) -- which ranges are significantly
    different, and by how much. Returned by
    `MetricSuite.cluster_permutation`; you'll rarely construct one
    directly.

    !!! note "Technical details"
        Each pair is computed completely independently via
        `mindmap.analysis.stats.cluster_permutation_mne` (unchanged, called
        once per pair) -- a single-pair result and a many-pair result are
        the same class. `correct` sets the multiple-comparison family:
        every cluster of every pair pooled (`family="cluster"`), or one
        p-value per pair, its smallest cluster p (`family="pair"`) -- see
        `correct`. `pair_table` gives the per-pair verdict either way.
    """

    KIND = "cluster"
    FAMILIES = ("cluster", "pair")

    def __init__(
        self, metric, group_by, coord, pairing_col, unit_col, higher_is_better, coord_order,
        cluster_alpha, n_permutations, seed, effect_size_summary, filters,
        clusters_by_pair: dict, t_obs_by_pair: dict, diff_by_pair: dict, computed_at=None,
        correction=None, alpha=None, data_key=None, family=None,
    ):
        self.metric = metric
        self.group_by = group_by
        self.coord = coord
        self.pairing_col = pairing_col
        self.unit_col = unit_col
        self.higher_is_better = higher_is_better
        self.coord_order = tuple(coord_order)
        self.cluster_alpha = cluster_alpha
        self.n_permutations = n_permutations
        self.seed = seed
        self.effect_size_summary = effect_size_summary
        self.filters = filters
        self.clusters_by_pair = clusters_by_pair
        self.t_obs_by_pair = t_obs_by_pair
        self.diff_by_pair = diff_by_pair
        self.correction = correction
        self.alpha = alpha
        self.family = family if family is not None or correction is None else "cluster"
        self.data_key = data_key
        self.computed_at = computed_at or datetime.now()

    def _replace(self, **changes):
        """A copy of this result with some fields swapped -- every field carried over unless overridden."""
        fields = dict(
            metric=self.metric, group_by=self.group_by, coord=self.coord, pairing_col=self.pairing_col,
            unit_col=self.unit_col, higher_is_better=self.higher_is_better, coord_order=self.coord_order,
            cluster_alpha=self.cluster_alpha, n_permutations=self.n_permutations, seed=self.seed,
            effect_size_summary=self.effect_size_summary, filters=self.filters,
            clusters_by_pair=self.clusters_by_pair, t_obs_by_pair=self.t_obs_by_pair, diff_by_pair=self.diff_by_pair,
            computed_at=self.computed_at, correction=self.correction, alpha=self.alpha, data_key=self.data_key,
            family=self.family,
        )
        fields.update(changes)
        return ClusterResult(**fields)


    @classmethod
    def compute(
        cls, df, metric, group_by, coord, comparisons, pairing_col, unit_col=None,
        higher_is_better=True, coord_order=None, cluster_alpha=0.05, n_permutations=3000,
        seed=0, effect_size_summary="cluster_mean", filters=None,
    ):
        """
        Runs the actual permutation tests and builds a `ClusterResult` from
        scratch. Most users won't call this directly --
        `MetricSuite.stat.cluster_permutation` does -- but it's here if you
        want the raw computation without going through a suite.

        !!! note "Technical details"
            Each `(reference, competitor)` pair gets its own, fully
            independent `cluster_permutation_mne` call -- this is the
            expensive step (`n_permutations` sign-flips per pair). When
            `unit_col` is given, `df` is pre-collapsed to one value per
            `(group_by, pairing_col, coord)` before that call -- first
            averaging over `unit_col` (e.g. several repeated runs), then
            over whatever's left -- so unequal repetition counts across
            `pairing_col` levels don't silently skew the per-coordinate
            mean `cluster_permutation_mne` tests. `cluster_permutation_mne`
            would collapse duplicate `(pairing_col, coord)` cells itself
            anyway if `unit_col` isn't given, but that's a flat one-stage
            average -- not equivalent when repetition counts are unequal.

        Parameters
        ----------
        df: pd.DataFrame
            Long-format table, pre-filtered to a single metric.

        metric, group_by, coord: str
            Identity fields carried onto the returned result.

        comparisons: list[tuple[str, str]]
            Which `(reference, competitor)` pairs to test.

        pairing_col: str
            The repeated-measures axis matched across groups (e.g.
            "region") -- required, since `cluster_permutation_mne` is
            inherently a paired test.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        higher_is_better, coord_order, cluster_alpha, n_permutations, seed, effect_size_summary
            Forwarded to `mindmap.analysis.stats.cluster_permutation_mne`
            -- see its docstring.

        filters: optional
            Identity/bookkeeping only.

        Returns
        -------
        ClusterResult
        """
        data_key = _data_key(df)
        if unit_col is not None:
            df = _collapse_with_coord(df, group_by, pairing_col, coord, unit_col)

        clusters_by_pair, t_obs_by_pair, diff_by_pair = {}, {}, {}
        for a, b in comparisons:
            result = cluster_permutation_mne(
                df, group_col=group_by, coord_col=coord, reference=a, competitor=b,
                subject_col=pairing_col, higher_is_better=higher_is_better, coord_order=coord_order,
                cluster_alpha=cluster_alpha, n_permutations=n_permutations, seed=seed,
                effect_size_summary=effect_size_summary,
            )
            clusters_by_pair[(a, b)] = result["clusters"]
            t_obs_by_pair[(a, b)] = result["t_obs"]
            diff_by_pair[(a, b)] = result["diff"]
            coord_order = result["coord_values"]  # resolved once, reused identically for every subsequent pair

        return cls(
            metric=metric, group_by=group_by, coord=coord, pairing_col=pairing_col, unit_col=unit_col,
            higher_is_better=higher_is_better, coord_order=coord_order, cluster_alpha=cluster_alpha,
            n_permutations=n_permutations, seed=seed, effect_size_summary=effect_size_summary,
            filters=filters, clusters_by_pair=clusters_by_pair,
            t_obs_by_pair=t_obs_by_pair, diff_by_pair=diff_by_pair, data_key=data_key,
        )

    def correct(self, method="fdr_bh", alpha=0.05, family="cluster"):
        """
        Adjusts the cluster p-values for testing several pairs. Returns a
        **new** `ClusterResult` -- never mutates `self` -- so a result can be
        re-corrected without rerunning the permutations.

        !!! note "Technical details"
            Within a pair, the permutation test already controls the
            family-wise error across the coordinate: each cluster's
            `p_value` is read against the null distribution of the
            *largest* cluster, so a pair's smallest cluster p answers "do
            these two groups differ anywhere along the coordinate?" What's
            left to correct for is the number of pairs. Two families:

            - `"cluster"` (default): every cluster of every pair, pooled
              into one family and corrected together. Pairs without
              clusters add nothing.
            - `"pair"`: one p-value per pair -- its smallest cluster p, or
              1 when the test found no cluster -- corrected across the
              pairs (e.g. Holm across 4 pairs). Every pair counts, with or
              without clusters. Each cluster's `p_corrected` is then
              `max(pair's corrected p, cluster's own p_value)`: a cluster
              is `significant` when its pair passes the correction and its
              own p-value is also within `alpha`. The pair's strongest
              cluster carries the pair's corrected p; the others locate
              further differences within a pair already shown to differ.

            `pair_table` reports the per-pair verdict for either family.

        Parameters
        ----------
        method: str, optional
            Correction method (default `"fdr_bh"`; e.g. `"holm"`) -- see
            `mindmap.core.stats.correct_pvalues`.

        alpha: float, optional
            Significance threshold (default 0.05).

        family: str, optional
            `"cluster"` (default) or `"pair"` -- see technical details.

        Returns
        -------
        ClusterResult
        """
        if family not in self.FAMILIES:
            raise ValueError(f"family={family!r} isn't supported. Choose from {list(self.FAMILIES)}.")
        corrected_by_pair = dict(self.clusters_by_pair)
        if family == "pair":
            pairs = list(self.clusters_by_pair)
            p_min = [float(cl["p_value"].min()) if not cl.empty else 1.0 for cl in self.clusters_by_pair.values()]
            if pairs:
                pair_corrected, _ = correct_pvalues(p_min, method=method, alpha=alpha)
                for pair, p_pair in zip(pairs, pair_corrected):
                    clusters = self.clusters_by_pair[pair].copy()
                    clusters["p_corrected"] = np.maximum(clusters["p_value"].to_numpy(dtype=float), p_pair)
                    clusters["significant"] = clusters["p_corrected"] <= alpha
                    corrected_by_pair[pair] = clusters
        else:
            non_empty = {pair: cl for pair, cl in self.clusters_by_pair.items() if not cl.empty}
            if non_empty:
                pairs = list(non_empty.keys())
                pooled = pd.concat(
                    [non_empty[p].assign(_pair_idx=i) for i, p in enumerate(pairs)], ignore_index=True,
                )
                p_corrected, significant = correct_pvalues(pooled["p_value"], method=method, alpha=alpha)
                pooled["p_corrected"] = p_corrected
                pooled["significant"] = significant
                for i, pair in enumerate(pairs):
                    corrected_by_pair[pair] = (
                        pooled[pooled["_pair_idx"] == i].drop(columns="_pair_idx").reset_index(drop=True)
                    )

        return self._replace(clusters_by_pair=corrected_by_pair, correction=method, alpha=alpha, family=family)

    @property
    def pair_table(self):
        """
        One row per pair: `group_a`, `group_b`, `n_clusters`, `p_min` (the
        smallest cluster `p_value`, 1 without any cluster), `p_corrected`
        and `significant` (any cluster significant). With
        `family="pair"`, `p_corrected` is the pair's own corrected p; with
        `family="cluster"`, the smallest of its clusters' (NaN for a pair
        without clusters -- it had nothing to correct). Both are
        NaN/`None` on an uncorrected result.
        """
        rows = []
        for (a, b), clusters in self.clusters_by_pair.items():
            corrected = np.nan
            if self.correction is not None:
                if not clusters.empty:
                    corrected = float(clusters["p_corrected"].min())
                elif self.family == "pair":
                    corrected = 1.0
            rows.append({
                "group_a": a, "group_b": b, "n_clusters": len(clusters),
                "p_min": float(clusters["p_value"].min()) if not clusters.empty else 1.0,
                "p_corrected": corrected,
                "significant": (bool(clusters["significant"].any()) if not clusters.empty else False)
                if self.correction is not None else None,
            })
        return pd.DataFrame(rows, columns=["group_a", "group_b", "n_clusters", "p_min", "p_corrected", "significant"])

    def __repr__(self):
        correction = f", correction={self.correction!r} ({self.family} family)" if self.correction else ""
        lines = [f"ClusterResult(metric={self.metric!r}, coord={self.coord!r}, n_pairs={len(self.clusters_by_pair)}{correction})"]
        for (a, b), clusters in self.clusters_by_pair.items():
            if clusters.empty:
                lines.append(f"  {a} vs {b}: no significant clusters")
                continue
            for _, row in clusters.iterrows():
                has_corrected = "p_corrected" in row and pd.notna(row["p_corrected"])
                p = row["p_corrected"] if has_corrected else row["p_value"]
                tag = "*" if row.get("significant") else ""
                lines.append(
                    f"  {a} vs {b}: [{row[f'{self.coord}_start']}, {row[f'{self.coord}_end']}] "
                    f"p={p:.4g}{tag}, d={row['effect_size']:.3g}"
                )
        return "\n".join(lines)

    def save(self, folder: str | Path):
        """
        Saves this result to a folder, so it can be reloaded later with
        `load()` without recomputing anything.

        !!! note "Technical details"
            Writes `identity.json` (metadata + per-pair cluster tables) +
            `arrays.npz` (per-pair `t_obs`/`diff`, numeric only).

        Parameters
        ----------
        folder: str | Path
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        pairs = list(self.clusters_by_pair.keys())

        meta = {
            "kind": self.KIND, "metric": self.metric, "group_by": self.group_by,
            "coord": self.coord, "pairing_col": self.pairing_col, "unit_col": self.unit_col,
            "higher_is_better": self.higher_is_better, "coord_order": list(self.coord_order),
            "cluster_alpha": self.cluster_alpha, "n_permutations": self.n_permutations,
            "seed": self.seed, "effect_size_summary": self.effect_size_summary,
            "filters": _filters_to_json(self.filters),
            "correction": self.correction, "alpha": self.alpha, "family": self.family, "data_key": self.data_key,
            "computed_at": self.computed_at.isoformat(),
            "pairs": [list(p) for p in pairs],
            "clusters_by_pair": {
                f"pair{i}": self.clusters_by_pair[p].to_dict(orient="records") for i, p in enumerate(pairs)
            },
            "diff_index_by_pair": {
                f"pair{i}": list(self.diff_by_pair[p].index) for i, p in enumerate(pairs)
            },
        }

        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=json_safe)
        atomic_replace(folder/"identity.json", _write_meta)

        arrays = {}
        for i, p in enumerate(pairs):
            arrays[f"pair{i}__t_obs"] = np.asarray(self.t_obs_by_pair[p], dtype=float)
            arrays[f"pair{i}__diff"] = self.diff_by_pair[p].to_numpy()
        if arrays:
            atomic_replace(folder/"arrays.npz", lambda p: np.savez(p, **arrays))

    @classmethod
    def load(cls, folder: str | Path):
        """
        Reloads a result previously saved with `save()`.

        Parameters
        ----------
        folder: str | Path

        Returns
        -------
        ClusterResult
        """
        folder = Path(folder)
        with open(folder/"identity.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        pairs = [tuple(p) for p in meta["pairs"]]
        arrays = read_npz_fast(folder/"arrays.npz") if pairs else {}

        clusters_by_pair, t_obs_by_pair, diff_by_pair = {}, {}, {}
        for i, pair in enumerate(pairs):
            clusters_by_pair[pair] = pd.DataFrame(meta["clusters_by_pair"][f"pair{i}"])
            t_obs_by_pair[pair] = arrays[f"pair{i}__t_obs"]
            diff_by_pair[pair] = pd.DataFrame(
                arrays[f"pair{i}__diff"], index=meta["diff_index_by_pair"][f"pair{i}"], columns=meta["coord_order"],
            )

        return cls(
            metric=meta["metric"], group_by=meta["group_by"], coord=meta["coord"],
            pairing_col=meta["pairing_col"], unit_col=meta.get("unit_col"), higher_is_better=meta["higher_is_better"],
            coord_order=meta["coord_order"], cluster_alpha=meta["cluster_alpha"],
            n_permutations=meta["n_permutations"], seed=meta["seed"],
            effect_size_summary=meta["effect_size_summary"],
            filters=_filters_from_json(meta.get("filters")),
            clusters_by_pair=clusters_by_pair, t_obs_by_pair=t_obs_by_pair, diff_by_pair=diff_by_pair,
            computed_at=datetime.fromisoformat(meta["computed_at"]),
            correction=meta.get("correction"), alpha=meta.get("alpha"), data_key=meta.get("data_key"),
            family=meta.get("family"),
        )


class NormalityResult:
    """
    The result of testing whether one metric's distribution is close to
    normal/Gaussian -- one row for the whole population, or one row per
    group if `group_by` was given. Returned by `MetricSuite.normality_test`;
    you'll rarely construct one directly.

    !!! note "Technical details"
        Each group is tested on its own independently, exactly like each
        pair in `PairwiseResult` -- a whole-population result and a
        many-group result are the same class.
    """

    KIND = "normality"

    def __init__(
        self, metric, group_by, test, filters, table: pd.DataFrame, computed_at=None,
        correction=None, alpha=None, data_key=None,
    ):
        self.metric = metric
        self.group_by = group_by
        self.test = test
        self.filters = filters
        self.table = table.reset_index(drop=True)
        self.correction = correction
        self.alpha = alpha
        self.data_key = data_key
        self.computed_at = computed_at or datetime.now()

    def _replace(self, **changes):
        """A copy of this result with some fields swapped -- every field carried over unless overridden."""
        fields = dict(
            metric=self.metric, group_by=self.group_by, test=self.test, filters=self.filters, table=self.table,
            computed_at=self.computed_at, correction=self.correction, alpha=self.alpha, data_key=self.data_key,
        )
        fields.update(changes)
        return NormalityResult(**fields)


    @classmethod
    def compute(cls, df, metric, group_by=None, test="shapiro", groups=None, filters=None):
        """
        Runs the actual normality test(s) and builds a `NormalityResult`
        from scratch. Most users won't call this directly --
        `MetricSuite.stat.normality_test` does -- but it's here if you want
        the raw computation without going through a suite.

        Parameters
        ----------
        df: pd.DataFrame
            Long-format table (one metric's worth), at least `["Value"]`
            (and `group_by`, if given).

        metric: str
            Identity field carried onto the returned result.

        group_by: str, optional
            Which registered column to test one group at a time. `None`
            (default) tests `df["Value"]` as a whole, producing a single
            row.

        test: str, optional
            Which normality test to run (default `"shapiro"`) -- see
            `mindmap.core.stats.test_normality`.

        groups: list, optional
            Which group values to test -- only meaningful when `group_by`
            is given; defaults to every group present in `df`. Ignored
            (a single whole-population row is produced) when `group_by` is
            `None`.

        filters: optional
            Identity/bookkeeping only -- not used for computation here (the
            filtering already happened in `df`), expected to already be
            frozen (via `_freeze_filters`) by the caller.

        Returns
        -------
        NormalityResult
        """
        rows = []
        if group_by is None:
            result = test_normality(df["Value"].to_numpy(), test=test)
            rows.append({
                "group": None, "n": result["n"], "statistic": result["statistic"],
                "p_value": result["p_value"], "p_corrected": np.nan, "significant": None,
            })
        else:
            if groups is None:
                groups = sorted(df[group_by].unique())
            for group in groups:
                sub = df[df[group_by] == group]
                if sub.empty:
                    raise ValueError(f"Group '{group}' not found in '{group_by}'.")
                result = test_normality(sub["Value"].to_numpy(), test=test)
                rows.append({
                    "group": group, "n": result["n"], "statistic": result["statistic"],
                    "p_value": result["p_value"], "p_corrected": np.nan, "significant": None,
                })

        table = pd.DataFrame(rows, columns=["group", "n", "statistic", "p_value", "p_corrected", "significant"])
        return cls(metric=metric, group_by=group_by, test=test, filters=filters, table=table, data_key=_data_key(df))

    def correct(self, method="fdr_bh", alpha=0.05):
        """
        Adjusts p-values for multiple comparisons across every group in
        this result, pooled into one family.

        !!! note "Technical details"
            Returns a **new** `NormalityResult` -- never mutates `self` --
            so the same result can be re-corrected with a different
            `alpha`/`method` without recomputing any test. A
            whole-population (single-row, no `group_by`) result still gets
            "corrected" here, but correcting a family of one is a no-op
            beyond copying the p-value across.

        Parameters
        ----------
        method: str, optional
            Correction method (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        Returns
        -------
        NormalityResult
        """
        table = self.table.copy()
        if not table.empty:
            p_corrected, significant = correct_pvalues(table["p_value"].to_numpy(), method=method, alpha=alpha)
            table["p_corrected"] = p_corrected
            table["significant"] = significant
        return self._replace(table=table, correction=method, alpha=alpha)

    def __repr__(self):
        lines = [f"NormalityResult(metric={self.metric!r}, test={self.test!r}, n_groups={len(self.table)})"]
        for _, row in self.table.iterrows():
            has_corrected = pd.notna(row["p_corrected"])
            p = row["p_corrected"] if has_corrected else row["p_value"]
            tag = "*" if has_corrected and row["significant"] else ""
            label = "(whole population)" if row["group"] is None else str(row["group"])
            lines.append(f"  {label}: p={p:.4g}{tag}, n={row['n']}")
        return "\n".join(lines)

    def save(self, folder: str | Path):
        """
        Saves this result to a folder, so it can be reloaded later with
        `load()` without recomputing anything.

        Parameters
        ----------
        folder: str | Path
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        meta = {
            "kind": self.KIND, "metric": self.metric, "group_by": self.group_by, "test": self.test,
            "filters": _filters_to_json(self.filters),
            "correction": self.correction, "alpha": self.alpha, "data_key": self.data_key,
            "computed_at": self.computed_at.isoformat(),
        }

        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        atomic_replace(folder/"identity.json", _write_meta)

        records = self.table.to_dict(orient="records")

        def _write_table(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, default=json_safe)
        atomic_replace(folder/"table.json", _write_table)

    @classmethod
    def load(cls, folder: str | Path):
        """
        Reloads a result previously saved with `save()`.

        Parameters
        ----------
        folder: str | Path

        Returns
        -------
        NormalityResult
        """
        folder = Path(folder)
        with open(folder/"identity.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(folder/"table.json", "r", encoding="utf-8") as f:
            records = json.load(f)

        table = pd.DataFrame(records, columns=["group", "n", "statistic", "p_value", "p_corrected", "significant"])
        return cls(
            metric=meta["metric"], group_by=meta["group_by"], test=meta["test"],
            filters=_filters_from_json(meta.get("filters")),
            table=table, computed_at=datetime.fromisoformat(meta["computed_at"]),
            correction=meta.get("correction"), alpha=meta.get("alpha"), data_key=meta.get("data_key"),
        )


class AnovaResult:
    """
    The result of one omnibus repeated-measures test across three or more
    matched groups on one metric -- "do these groups differ at all",
    complementary to `PairwiseResult`'s pairwise "does A differ from B".
    Returned by `MetricSuite.stat.anova`; you'll rarely construct one
    directly.

    !!! note "Technical details"
        Unlike `PairwiseResult`/`ClusterResult`/`NormalityResult`, this is
        a single test producing one p-value, not a family of pairs/groups
        -- there's nothing to multiple-comparison-correct here. A
        significant result is the usual green light to follow up with
        `MetricSuite.stat.wilcoxon`/`.paired_ttest`/`.ttest_ess` as
        post-hoc pairwise tests (each does its own FDR correction across
        the pairs it covers).
    """

    KIND = "anova"

    def __init__(
        self, metric, group_by, pairing_col, unit_col, test, groups, filters,
        statistic, p_value, n, k, effect_size, effect_size_type,
        alpha=None, significant=None, computed_at=None, data_key=None,
    ):
        self.metric = metric
        self.group_by = group_by
        self.pairing_col = pairing_col
        self.unit_col = unit_col
        self.test = test
        self.groups = groups
        self.filters = filters
        self.statistic = statistic
        self.p_value = p_value
        self.n = n
        self.k = k
        self.effect_size = effect_size
        self.effect_size_type = effect_size_type
        self.alpha = alpha
        self.significant = significant
        self.data_key = data_key
        self.computed_at = computed_at or datetime.now()


    @classmethod
    def compute(cls, df, metric, group_by, pairing_col, unit_col=None, test="friedman", groups=None, filters=None):
        """
        Runs the actual omnibus test and builds an `AnovaResult` from
        scratch. Most users won't call this directly --
        `MetricSuite.stat.anova` does -- but it's here if you want the raw
        computation without going through a suite.

        !!! note "Technical details"
            Complete-case only, across *every* included group at once: a
            subject missing from any one of them is dropped from the
            whole test -- unlike `PairwiseResult.compute`, which drops a
            subject only from the one pair it's actually missing from, an
            omnibus test needs one consistent subject set shared by every
            group simultaneously to be a meaningful single test. When
            `unit_col` is given, `df` is first collapsed to one value per
            `(group_by, pairing_col, unit_col)`, then to one per
            `(group_by, pairing_col)` -- same two-stage average as
            `PairwiseResult.compute`.

        Parameters
        ----------
        df: pd.DataFrame
            Long-format table (one metric's worth), at least
            `[group_by, pairing_col, "Value"]`.

        metric, group_by: str
            Identity fields carried onto the returned result.

        pairing_col: str
            The repeated-measures axis matched across groups (e.g.
            "region") -- required, since an omnibus RM test is inherently
            a paired/matched-groups test.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        test: str, optional
            Which omnibus test to run (default `"friedman"`) -- see
            `mindmap.core.stats.repeated_measures_anova`.

        groups: list, optional
            Which groups to include. Defaults to every group present in
            `df[group_by]`.

        filters: optional
            Identity/bookkeeping only -- not used for computation here (the
            filtering already happened in `df`), expected to already be
            frozen (via `_freeze_filters`) by the caller.

        Returns
        -------
        AnovaResult
        """
        collapsed = _collapse(df, group_by, pairing_col, unit_col)
        wide = collapsed.pivot(index=pairing_col, columns=group_by, values="Value")

        if groups is None:
            groups = sorted(wide.columns)
        else:
            missing = set(groups) - set(wide.columns)
            if missing:
                raise ValueError(f"Group(s) {sorted(missing)} not found in '{group_by}'.")

        complete = wide[list(groups)].dropna()
        if complete.empty:
            raise ValueError(f"No subjects with complete data across groups {list(groups)}.")

        result = repeated_measures_anova(complete.to_numpy(), test=test)
        return cls(
            metric=metric, group_by=group_by, pairing_col=pairing_col, unit_col=unit_col, test=test,
            groups=tuple(sorted(groups)), filters=filters,
            statistic=result["statistic"], p_value=result["p_value"],
            n=result["n"], k=result["k"],
            effect_size=result["effect_size"], effect_size_type=result["effect_size_type"],
            data_key=_data_key(df),
        )

    def correct(self, alpha=0.05):
        """
        Marks this result significant/not at `alpha`.

        !!! note "Technical details"
            No multiple-comparison correction happens here -- this is
            already a single omnibus p-value, not a family -- so this only
            (re-)sets `significant = p_value < alpha`. Returns a **new**
            `AnovaResult` -- never mutates `self` -- so the same result
            can be re-thresholded at a different `alpha` without rerunning
            the test.

        Parameters
        ----------
        alpha: float, optional
            Significance threshold (default 0.05).

        Returns
        -------
        AnovaResult
        """
        significant = (self.p_value < alpha) if self.p_value is not None and not np.isnan(self.p_value) else None
        return AnovaResult(
            metric=self.metric, group_by=self.group_by, pairing_col=self.pairing_col, unit_col=self.unit_col,
            test=self.test, groups=self.groups, filters=self.filters, statistic=self.statistic, p_value=self.p_value,
            n=self.n, k=self.k, effect_size=self.effect_size, effect_size_type=self.effect_size_type,
            alpha=alpha, significant=significant, computed_at=self.computed_at, data_key=self.data_key,
        )

    def __repr__(self):
        tag = "*" if self.significant else ""
        p = f"{self.p_value:.4g}" if self.p_value is not None and not np.isnan(self.p_value) else "nan"
        return (
            f"AnovaResult(metric={self.metric!r}, test={self.test!r}, groups={self.groups}): "
            f"p={p}{tag}, {self.effect_size_type}={self.effect_size:.3g}, n={self.n}, k={self.k}"
        )

    def save(self, folder: str | Path):
        """
        Saves this result to a folder, so it can be reloaded later with
        `load()` without recomputing anything.

        Parameters
        ----------
        folder: str | Path
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        meta = {
            "kind": self.KIND, "metric": self.metric, "group_by": self.group_by,
            "pairing_col": self.pairing_col, "unit_col": self.unit_col, "test": self.test,
            "groups": list(self.groups) if self.groups is not None else None,
            "filters": _filters_to_json(self.filters),
            "statistic": json_safe(self.statistic), "p_value": json_safe(self.p_value),
            "n": self.n, "k": self.k,
            "effect_size": json_safe(self.effect_size), "effect_size_type": self.effect_size_type,
            "alpha": self.alpha, "significant": self.significant, "data_key": self.data_key,
            "computed_at": self.computed_at.isoformat(),
        }

        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        atomic_replace(folder/"identity.json", _write_meta)

    @classmethod
    def load(cls, folder: str | Path):
        """
        Reloads a result previously saved with `save()`.

        Parameters
        ----------
        folder: str | Path

        Returns
        -------
        AnovaResult
        """
        folder = Path(folder)
        with open(folder/"identity.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        return cls(
            metric=meta["metric"], group_by=meta["group_by"], pairing_col=meta["pairing_col"],
            unit_col=meta.get("unit_col"), test=meta["test"],
            groups=tuple(meta["groups"]) if meta["groups"] is not None else None,
            filters=_filters_from_json(meta.get("filters")),
            statistic=meta["statistic"], p_value=meta["p_value"], n=meta["n"], k=meta["k"],
            effect_size=meta["effect_size"], effect_size_type=meta["effect_size_type"],
            alpha=meta.get("alpha"), significant=meta.get("significant"),
            computed_at=datetime.fromisoformat(meta["computed_at"]), data_key=meta.get("data_key"),
        )


class EstimateResult:
    """
    Point estimates, optionally with confidence intervals, for one metric
    -- either one row per group (its mean or median), or one row per
    `(group_a, group_b)` pair (a difference or effect size of `group_a`
    relative to `group_b`: `mean_diff`, `hodges_lehmann`, or any
    standardized effect size). Returned by `MetricSuite.estimate.*`;
    you'll rarely construct one directly.

    !!! note "Technical details"
        An estimate answers "how large" -- it has no hypothesis and no
        offset; read its interval against whatever offset matters. Every
        effect size a test can report is available here too, computed by
        the same function (`mindmap.core.stats.effect_size`), so the two
        always agree. Like the test results, it records every setting
        needed to reproduce it (estimator, alpha, `ci`, `ess`, pairing,
        filters, and with `ci_method="bootstrap"` the `bootstrap`
        settings: `n_boot` and `seed`, or the registered `replicates`/
        `replicate_col`) and a `data_key` fingerprint of its rows (both
        metrics' with registered replicates). Two table shapes, told
        apart by `is_difference`:

        - per group: columns `group`, `estimate`, `ci_low`, `ci_high`, `n`,
          `ci_method` (`group` is `None` for the whole population);
        - per pair: columns `group_a`, `group_b`, `estimate`, `ci_low`,
          `ci_high`, `n_a`, `n_b`, `ci_method` (+ `n_eff` with `ess`) --
          paired (`pairing_col` set: matched differences, `n_a == n_b`) or
          independent.

        With `ci=False`, `ci_low`/`ci_high` are NaN and `ci_method` is
        `None`. A bootstrap interval adds an `n_boot` column (the valid
        draws) and reads `ci_method == "bootstrap_percentile"`. `save`/
        `load` work on their own, so an estimate stays usable without the
        suite it came from.
    """

    KIND = "estimate"
    ESTIMATORS = GROUP_ESTIMATORS + DIFFERENCE_ESTIMATORS

    def __init__(
        self, metric, group_by, estimator, alpha, pairing_col, unit_col, subject_order, filters,
        table: pd.DataFrame, computed_at=None, data_key=None, ci=True, ess=False, zero_method=None, bootstrap=None,
    ):
        self.metric = metric
        self.group_by = group_by
        self.estimator = estimator
        self.alpha = alpha
        self.ci = ci
        self.ess = ess
        self.bootstrap = dict(bootstrap) if bootstrap is not None else None
        self.zero_method = zero_method
        self.pairing_col = pairing_col
        self.unit_col = unit_col
        self.subject_order = subject_order
        self.filters = filters
        self.data_key = data_key
        self.table = table.reset_index(drop=True)
        self.computed_at = computed_at or datetime.now()

    @property
    def ci_method(self):
        """`"bootstrap"` or `"analytic"` -- how the interval was built (`None` without one)."""
        if not self.ci:
            return None
        return "bootstrap" if self.bootstrap is not None else "analytic"

    @property
    def is_difference(self):
        """Whether rows are `(group_a, group_b)` differences rather than one group each."""
        return "group_a" in self.table.columns

    @property
    def comparisons(self):
        """The `(group_a, group_b)` pairs estimated, in table order -- `None` for per-group estimates."""
        if not self.is_difference:
            return None
        return tuple(zip(self.table["group_a"], self.table["group_b"]))

    @classmethod
    def compute(
        cls, df, metric, estimator, group_by=None, comparisons=None, pairing_col=None, unit_col=None,
        subject_order=None, alpha=0.05, ci=True, ess=False, filters=None, zero_method=None, bootstrap=None,
    ):
        """
        Computes the estimates and builds an `EstimateResult` from
        scratch -- what `MetricSuite.estimate.*` call.

        !!! note "Technical details"
            Collapsing matches the tests exactly, so an estimate describes
            the same units its test saw: with `pairing_col`, one value per
            `(group_by, pairing_col)` (first averaging over `unit_col`),
            and each pair keeps its own complete cases; without it, one
            value per `(group_by, unit_col)` when `unit_col` is given, raw
            rows otherwise. With `ess`, each pair's effective sample size
            is computed from its differences in `subject_order` -- exactly
            what `ttest_ess` uses. `filters` is bookkeeping only (the
            filtering already happened in `df`).

            A bootstrap interval resamples the units the estimate reads,
            after collapsing: the `pairing_col` values (whole pairs, for a
            difference), else the `unit_col` values, else the rows. One
            generator seeded by `bootstrap["seed"]` serves every row in
            table order, so `suite.stat.bootstrap` with the same seed
            draws exactly the same resamples. With registered
            `replicates`, each draw's rows are collapsed and paired
            exactly like the observed ones.

        Parameters
        ----------
        df: pd.DataFrame
            Long-format table (one metric's worth).

        metric: str
            Identity field carried onto the returned result.

        estimator: str
            Per group: `"mean"` | `"median"`. Per pair (needs
            `comparisons`): `"mean_diff"` | `"hodges_lehmann"` |
            `"rank_biserial"` | `"cohens_dz"` | `"hedges_gz"` (paired) |
            `"cles"` | `"cohens_d"` | `"hedges_g"` (independent) -- see
            `mindmap.core.stats.estimate`.

        group_by: str, optional
            Column holding the groups. Omit (per-group estimators only) to
            estimate the whole population as one group.

        comparisons: list[tuple[str, str]], optional
            The `(group_a, group_b)` pairs a difference estimator compares.

        pairing_col: str, optional
            Match pairs on this column (paired differences); omit for
            independent samples.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        subject_order: list, optional
            Restricts/reorders `pairing_col`'s values exactly like the
            test it accompanies -- the true order `ess` needs.

        alpha: float, optional
            The interval is `1 - alpha` (default 0.05, i.e. 95%).

        ci: bool, optional
            Whether to compute intervals (default True) -- raises
            `NotImplementedError` for an estimator without an analytic one
            unless `bootstrap` is given.

        ess: bool, optional
            Paired only: use each pair's effective sample size (for
            autocorrelated series, as in `ttest_ess`) -- in `mean_diff`'s
            interval and `hedges_gz`'s correction.

        filters: optional
            Bookkeeping only.

        zero_method: str, optional
            Paired `"rank_biserial"` only: how zero differences are ranked
            -- `"wilcox"` (default), `"pratt"` or `"zsplit"`, matching the
            Wilcoxon test's (see `mindmap.core.stats.effect_size`).

        bootstrap: dict, optional
            A percentile bootstrap interval instead of the analytic one --
            `resolve_bootstrap`'s settings. With registered `replicates`,
            `df` holds both metrics.

        Returns
        -------
        EstimateResult
        """
        if estimator not in cls.ESTIMATORS:
            raise ValueError(f"Unknown estimator {estimator!r}. Choose from {list(cls.ESTIMATORS)}.")
        if estimator in GROUP_ESTIMATORS and comparisons is not None:
            raise ValueError(
                f"estimator={estimator!r} describes each group on its own and takes no `comparisons` -- "
                "for a difference, use 'mean_diff' or 'hodges_lehmann'."
            )
        if estimator in DIFFERENCE_ESTIMATORS and comparisons is None:
            raise ValueError(f"estimator={estimator!r} describes a difference between two groups -- pass `comparisons`.")
        if comparisons is not None and group_by is None:
            raise ValueError("`group_by` is required with `comparisons` (it holds the groups being compared).")
        if ess and (comparisons is None or pairing_col is None):
            raise ValueError("`ess` only applies to paired differences (`comparisons` with a `pairing_col`).")
        if bootstrap is not None and not ci:
            raise ValueError("ci=False has no interval to bootstrap -- drop the bootstrap settings or pass ci=True.")
        if bootstrap is not None and ess:
            raise ValueError(
                "The bootstrap resamples units independently, so it can't honour `ess` (serial correlation) -- "
                "use the analytic interval with ess=True."
            )
        paired_rank_biserial = estimator == "rank_biserial" and comparisons is not None and pairing_col is not None
        if paired_rank_biserial:
            zero_method = "wilcox" if zero_method is None else zero_method
        elif zero_method not in (None, "wilcox"):
            raise ValueError("`zero_method` only applies to the paired rank-biserial (with a `pairing_col`).")
        else:
            zero_method = None

        data_key = _data_key(df)
        jackknife_df = _jackknife_rows(df, metric, bootstrap)
        df, draws_df = _split_replicates(df, metric, bootstrap)
        interval = (bootstrap or {}).get("interval", "percentile")
        if comparisons is None and (bootstrap or {}).get("jackknife") is not None:
            raise NotImplementedError("A registered jackknife is only supported for differences (`comparisons`).")
        boot_kwargs = {}
        if bootstrap is not None:
            boot_kwargs.update(ci_method="bootstrap", interval=interval)
            if interval == "bca" and bootstrap.get("acceleration") is not None:
                boot_kwargs["acceleration"] = bootstrap["acceleration"]
            if draws_df is None:
                boot_kwargs.update(n_boot=bootstrap["n_boot"], seed=np.random.default_rng(bootstrap["seed"]))
        extra = (["n_eff"] if ess else []) + (["n_boot"] if bootstrap is not None else [])

        rows = []
        if comparisons is None:
            groups, samples = _group_samples(df, group_by, pairing_col, unit_col)
            registered = None
            if draws_df is not None:
                registered = _replicate_draws(
                    draws_df, bootstrap["replicate_col"],
                    lambda sub: [(v,) for v in _group_samples(sub, group_by, pairing_col, unit_col, groups)[1]],
                    lambda v: estimate(v, estimator=estimator, ci=False)["estimate"],
                )
            for i, (g, values) in enumerate(zip(groups, samples)):
                draws = {"draws": registered[:, i]} if registered is not None else {}
                out = estimate(values, estimator=estimator, alpha=alpha, ci=ci, **boot_kwargs, **draws)
                rows.append({
                    "group": g, "estimate": out["estimate"], "ci_low": out["ci_low"], "ci_high": out["ci_high"],
                    "n": out["n"], "ci_method": out["ci_method"], "n_boot": out.get("n_boot"),
                })
            columns = ["group", "estimate", "ci_low", "ci_high", "n", "ci_method"]
        else:
            paired = pairing_col is not None
            zero_method_used = zero_method or "wilcox"
            samples = _pair_samples(df, group_by, pairing_col, unit_col, comparisons, subject_order)
            all_draws = accelerations = None
            if bootstrap is not None and ci:
                all_draws = comparison_draws(
                    samples, draws_df, group_by, pairing_col, unit_col, comparisons, subject_order, bootstrap,
                    estimator, zero_method_used,
                )
                accelerations = comparison_accelerations(
                    samples, jackknife_df, group_by, pairing_col, unit_col, comparisons, subject_order, bootstrap,
                    estimator, zero_method_used,
                )
            for i, ((a, b), (x, y)) in enumerate(zip(comparisons, samples)):
                draws = {"draws": all_draws[i]} if all_draws is not None else {}
                if accelerations is not None and accelerations[i] is not None:
                    draws["acceleration"] = accelerations[i]
                n_eff = effective_sample_size(x - y) if ess else None
                out = estimate(
                    x, y, estimator=estimator, paired=paired, alpha=alpha, ci=ci, n_eff=n_eff,
                    zero_method=zero_method_used, **boot_kwargs, **draws,
                )
                n_a, n_b = (out["n"], out["n"]) if paired else (out["n_a"], out["n_b"])
                rows.append({
                    "group_a": a, "group_b": b, "estimate": out["estimate"], "ci_low": out["ci_low"],
                    "ci_high": out["ci_high"], "n_a": n_a, "n_b": n_b, "ci_method": out["ci_method"],
                    "n_eff": out.get("n_eff"), "n_boot": out.get("n_boot"),
                })
            columns = ["group_a", "group_b", "estimate", "ci_low", "ci_high", "n_a", "n_b", "ci_method"]
        table = pd.DataFrame(rows, columns=columns + extra)

        return cls(
            metric=metric, group_by=group_by, estimator=estimator, alpha=alpha, pairing_col=pairing_col,
            unit_col=unit_col, subject_order=_freeze_filter(subject_order), filters=filters, table=table,
            data_key=data_key, ci=ci, ess=ess, zero_method=zero_method, bootstrap=bootstrap,
        )

    def vs(self, reference, higher_is_better=True):
        """
        One row per competitor of `reference`, the estimate re-expressed as
        `reference` relative to the competitor -- negated (interval bounds
        swapped) when `reference` is `group_b`, and once more when
        `higher_is_better=False` -- so positive always means `reference`
        better, matching `PairwiseResult.vs`. `cles`, on a probability
        scale, maps to `1 - cles` instead, and `odds_ratio`, on a ratio
        scale, to `1 / odds_ratio`.

        Parameters
        ----------
        reference: str
            The group every competitor is read against.

        higher_is_better: bool, optional
            Whether a larger metric value is better (default True).

        Returns
        -------
        pd.DataFrame
            Indexed by competitor (index named after `group_by`), columns
            `estimate`, `ci_low`, `ci_high`, `n_reference`,
            `n_competitor` (+ `n_eff` with `ess`, `n_boot` with a
            bootstrap interval).
        """
        if not self.is_difference:
            raise ValueError("vs() needs a difference result (one computed with `comparisons`), not per-group estimates.")
        if not isinstance(higher_is_better, bool):
            raise TypeError(f"`higher_is_better` must be a bool, got {type(higher_is_better).__name__}.")

        t = self.table
        rows = t[(t["group_a"] == reference) | (t["group_b"] == reference)]
        if rows.empty:
            groups = sorted({*t["group_a"], *t["group_b"]}, key=str)
            raise ValueError(f"reference {reference!r} appears in no pair of this result (groups compared: {groups}).")

        as_a = (rows["group_a"] == reference).to_numpy()
        flipped = (~as_a) ^ (not higher_is_better)
        value = rows["estimate"].to_numpy(dtype=float)
        low, high = rows["ci_low"].to_numpy(dtype=float), rows["ci_high"].to_numpy(dtype=float)
        mirror = lambda v: _mirror(self.estimator, v)
        out = pd.DataFrame({
            "estimate": np.where(flipped, mirror(value), value),
            "ci_low": np.where(flipped, mirror(high), low),
            "ci_high": np.where(flipped, mirror(low), high),
            "n_reference": np.where(as_a, rows["n_a"], rows["n_b"]),
            "n_competitor": np.where(as_a, rows["n_b"], rows["n_a"]),
        }, index=pd.Index(np.where(as_a, rows["group_b"], rows["group_a"]), name=self.group_by))
        for column in ("n_eff", "n_boot"):
            if column in rows.columns:
                out[column] = rows[column].to_numpy()
        return out

    def __repr__(self):
        level = f", ci={1 - self.alpha:.0%}" if self.ci else ""
        if self.bootstrap is not None:
            level += f" bootstrap, {_bootstrap_label(self.bootstrap)}"
        lines = [f"EstimateResult(metric={self.metric!r}, estimator={self.estimator!r}{level}, n_rows={len(self.table)})"]
        for _, row in self.table.iterrows():
            if self.is_difference:
                label = f"{row['group_a']} vs {row['group_b']}"
            else:
                label = "(whole population)" if row["group"] is None else str(row["group"])
            interval = f" [{row['ci_low']:.4g}, {row['ci_high']:.4g}]" if self.ci else ""
            lines.append(f"  {label}: {row['estimate']:.4g}{interval}")
        return "\n".join(lines)

    def save(self, folder: str | Path):
        """
        Saves this result to a folder, so it can be reloaded later with
        `load()` -- without the suite it was computed from.

        !!! note "Technical details"
            Writes `identity.json` (metadata, including the table's column
            layout, so an empty table reloads with the right shape) +
            `table.json`.

        Parameters
        ----------
        folder: str | Path
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        meta = {
            "kind": self.KIND, "metric": self.metric, "group_by": self.group_by, "estimator": self.estimator,
            "alpha": self.alpha, "ci": self.ci, "ess": self.ess, "zero_method": self.zero_method, "bootstrap": self.bootstrap,
            "pairing_col": self.pairing_col, "unit_col": self.unit_col,
            "subject_order": list(self.subject_order) if self.subject_order is not None else None,
            "filters": _filters_to_json(self.filters), "data_key": self.data_key,
            "columns": list(self.table.columns), "computed_at": self.computed_at.isoformat(),
        }

        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=json_safe)
        atomic_replace(folder/"identity.json", _write_meta)

        records = self.table.to_dict(orient="records")

        def _write_table(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, default=json_safe)
        atomic_replace(folder/"table.json", _write_table)

    @classmethod
    def load(cls, folder: str | Path):
        """
        Reloads a result previously saved with `save()`.

        Parameters
        ----------
        folder: str | Path

        Returns
        -------
        EstimateResult
        """
        folder = Path(folder)
        with open(folder/"identity.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(folder/"table.json", "r", encoding="utf-8") as f:
            records = json.load(f)

        return cls(
            metric=meta["metric"], group_by=meta["group_by"], estimator=meta["estimator"], alpha=meta["alpha"],
            pairing_col=meta["pairing_col"], unit_col=meta["unit_col"],
            subject_order=tuple(meta["subject_order"]) if meta["subject_order"] is not None else None,
            filters=_filters_from_json(meta["filters"]), table=pd.DataFrame(records, columns=meta["columns"]),
            computed_at=datetime.fromisoformat(meta["computed_at"]), data_key=meta["data_key"],
            ci=meta.get("ci", True), ess=meta.get("ess", False),
            zero_method=meta.get("zero_method", "wilcox" if meta["estimator"] == "rank_biserial" and meta.get("pairing_col") else None),
            bootstrap=meta.get("bootstrap"),
        )
