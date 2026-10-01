"""
Module Name: metric/results.py
Description: Typed, cacheable, mergeable statistical results for
`MetricSuite` -- `PairwiseResult` (wilcoxon/ttest/mannwhitneyu/
paired_ttest/ttest_ess comparisons), `ClusterResult` (cluster-permutation
comparisons), `NormalityResult`, and `AnovaResult`. `PairwiseResult`/
`ClusterResult` operate on an explicit list of `(group_a, group_b)` pairs
rather than being anchored to one fixed reference method, so a "reference
vs many competitors" comparison and an arbitrary many-to-many one are the
same shape -- a single-pair result and a many-pair result are the same
class, mergeable into one another.

Author: Nino Herve
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mindmap.core.io import atomic_replace, json_safe, read_npz_fast
from mindmap.core.stats import paired_test, independent_test, correct_pvalues, test_normality, repeated_measures_anova
from mindmap.analysis.stats import cluster_permutation_mne


def _freeze_filter(val):
    """
    Normalizes a scalar-or-list-or-None filter/ordering argument into a
    hashable tuple for use in an identity/cache key: `None` stays `None`, a
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
    key-order-independent tuple for identity/cache-key use -- `None`/empty
    stays `None`, otherwise `((key, frozen_values), ...)` sorted by key.
    """
    if not filters:
        return None
    return tuple(sorted((k, _sorted_filter(v)) for k, v in filters.items()))


#: Public `test=` name -> the `paired_test`/`independent_test` name it actually runs.
#: `"paired_ttest"` -> `"ttest"` because `paired_test`'s own "ttest" already means the
#: paired/related version; `"ttest"` (independent, public) -> `"ttest_ind"` for the
#: same reason on the `independent_test` side. Whichever dict a name is found in
#: determines the whole computation path (paired vs independent) in `compute()`.
_PAIRED_CORE_TEST = {"wilcoxon": "wilcoxon", "paired_ttest": "ttest", "ttest_ess": "ttest_ess"}
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


def _compute_paired_rows(df, group_by, pairing_col, unit_col, comparisons, core_test, subject_order):
    wide = _paired_wide(df, group_by, pairing_col, unit_col, subject_order)

    rows = []
    for a, b in comparisons:
        missing = {a, b} - set(wide.columns)
        if missing:
            raise ValueError(f"Group(s) {sorted(missing)} not found in '{group_by}'.")
        complete = wide[[a, b]].dropna()
        if complete.empty:
            raise ValueError(f"No subjects with complete data for pair ('{a}', '{b}').")
        result = paired_test(complete[a].to_numpy(), complete[b].to_numpy(), test=core_test)
        rows.append({
            "group_a": a, "group_b": b, "n_pairs": result["n"], "n_a": result["n"], "n_b": result["n"],
            "statistic": result["statistic"], "p_value": result["p_value"],
            "effect_size": result["effect_size"], "effect_size_type": result["effect_size_type"],
            "n_eff": result.get("n_eff", np.nan), "p_corrected": np.nan, "significant": None,
        })
    return rows


def _collapse_with_coord(df, group_by, pairing_col, coord, unit_col):
    """Like `_collapse`, but keeps `coord` distinct throughout -- for `ClusterResult`, which
    must never average across the coordinate it's testing along."""
    if unit_col is not None:
        df = df.groupby([group_by, pairing_col, coord, unit_col], as_index=False)["Value"].mean()
    return df.groupby([group_by, pairing_col, coord], as_index=False)["Value"].mean()


def _compute_independent_rows(df, group_by, unit_col, comparisons, core_test):
    collapsed = df if unit_col is None else df.groupby([group_by, unit_col], as_index=False)["Value"].mean()

    rows = []
    for a, b in comparisons:
        vals_a = collapsed.loc[collapsed[group_by] == a, "Value"]
        vals_b = collapsed.loc[collapsed[group_by] == b, "Value"]
        if vals_a.empty:
            raise ValueError(f"Group '{a}' not found in '{group_by}'.")
        if vals_b.empty:
            raise ValueError(f"Group '{b}' not found in '{group_by}'.")
        result = independent_test(vals_a.to_numpy(), vals_b.to_numpy(), test=core_test)
        rows.append({
            "group_a": a, "group_b": b, "n_pairs": np.nan, "n_a": result["n_a"], "n_b": result["n_b"],
            "statistic": result["statistic"], "p_value": result["p_value"],
            "effect_size": result["effect_size"], "effect_size_type": result["effect_size_type"],
            "n_eff": np.nan, "p_corrected": np.nan, "significant": None,
        })
    return rows


_PAIRWISE_TABLE_COLUMNS = [
    "group_a", "group_b", "n_pairs", "n_a", "n_b", "statistic", "p_value",
    "effect_size", "effect_size_type", "n_eff", "p_corrected", "significant",
]


class PairwiseResult:
    """
    The result of comparing two or more groups on one metric -- one row per
    pair compared, with the test statistic, p-value, and effect size.
    Returned by `MetricSuite.stat.wilcoxon`/`.ttest`/`.mannwhitneyu`/
    `.paired_ttest`/`.ttest_ess`; you'll rarely construct one directly.

    !!! note "Technical details"
        Covers two genuinely different comparison shapes with one class,
        distinguished by `pairing_col`:

        - Paired (`wilcoxon`/`paired_ttest`/`ttest_ess`, `pairing_col` set):
          rows are matched one-to-one across groups by `pairing_col` (e.g.
          the same region in every method) -- `table["n_pairs"]` is one
          shared count, since a paired test needs the exact same subjects
          on both sides.
        - Independent (`ttest`/`mannwhitneyu`, `pairing_col` is `None`): no
          correspondence between groups is assumed -- `table["n_a"]`/
          `"n_b"` can differ. `n_a`/`n_b` are also filled in for paired
          rows (both equal to `n_pairs` there), so either column is always
          meaningful regardless of which family produced a given row.

        Every pair is tested on its own complete-case subjects (paired) or
        its own collapsed sample (independent), independent of any other
        pair in the same object -- so a result covering many pairs is just
        as valid as one covering a single pair, and several can be
        `merge`d into a custom family before `correct`-ing.
    """

    KIND = "pairwise"

    def __init__(
        self, metric, group_by, test, unit_col, pairing_col, subject_order, filters,
        table: pd.DataFrame, computed_at=None,
    ):
        self.metric = metric
        self.group_by = group_by
        self.test = test
        self.unit_col = unit_col
        self.pairing_col = pairing_col
        self.subject_order = subject_order
        self.filters = filters
        self.table = table.reset_index(drop=True)
        self.computed_at = computed_at or datetime.now()

    @property
    def identity(self):
        """Hashable tuple describing exactly what this object contains -- metric/groups/method/params, used for `MetricSuite.list_cached_stats()`."""
        pairs = tuple(sorted(zip(self.table["group_a"], self.table["group_b"])))
        return (
            self.KIND, self.metric, self.group_by, self.test, self.unit_col, self.pairing_col,
            self.subject_order, self.filters, pairs,
        )

    @classmethod
    def compute(cls, df, metric, group_by, comparisons, test, unit_col=None, pairing_col=None, subject_order=None, filters=None):
        """
        Runs the actual tests and builds a `PairwiseResult` from scratch.
        Most users won't call this directly -- `MetricSuite.stat.wilcoxon`/
        etc. do, with caching -- but it's here if you want the raw
        computation without going through a suite.

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
            it's carried through purely as identity/bookkeeping, and is
            expected to already be frozen (via `_freeze_filters`) by the
            caller.

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
            `"wilcoxon"` | `"ttest"` | `"mannwhitneyu"` | `"paired_ttest"`
            | `"ttest_ess"`.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        pairing_col: str, optional
            Cross-group matching axis -- required for the three paired
            tests, must be omitted for the two independent ones.

        subject_order: list, optional
            Only matters for `test="ttest_ess"`.

        filters: optional
            Identity/bookkeeping only -- see above.

        Returns
        -------
        PairwiseResult
        """
        if test in _PAIRED_CORE_TEST:
            if pairing_col is None:
                raise ValueError(f"`pairing_col` is required for test={test!r} (a paired test).")
            rows = _compute_paired_rows(df, group_by, pairing_col, unit_col, comparisons, _PAIRED_CORE_TEST[test], subject_order)
        elif test in _INDEPENDENT_CORE_TEST:
            if pairing_col is not None:
                raise ValueError(f"test={test!r} is an independent-samples test and doesn't take `pairing_col`.")
            rows = _compute_independent_rows(df, group_by, unit_col, comparisons, _INDEPENDENT_CORE_TEST[test])
        else:
            known = sorted({**_PAIRED_CORE_TEST, **_INDEPENDENT_CORE_TEST})
            raise ValueError(f"Unknown test {test!r}. Choose from {known}.")

        table = pd.DataFrame(rows, columns=_PAIRWISE_TABLE_COLUMNS)
        return cls(
            metric=metric, group_by=group_by, test=test, unit_col=unit_col, pairing_col=pairing_col,
            subject_order=_freeze_filter(subject_order), filters=filters, table=table,
        )

    def correct(self, method="fdr_bh", alpha=0.05):
        """
        Adjusts p-values for multiple comparisons across every pair in this
        result, pooled into one family.

        !!! note "Technical details"
            Returns a **new** `PairwiseResult` -- never mutates `self` --
            so the cached raw object stays available for a different
            `alpha`/`method` later without recomputing any test.

        Parameters
        ----------
        method: str, optional
            Correction method (default `"fdr_bh"`).

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
        return PairwiseResult(
            metric=self.metric, group_by=self.group_by, test=self.test,
            unit_col=self.unit_col, pairing_col=self.pairing_col, subject_order=self.subject_order,
            filters=self.filters, table=table, computed_at=self.computed_at,
        )

    def vs(self, reference, higher_is_better=True):
        """
        One row per competitor of `reference` -- its p-values and an effect
        size oriented so that **positive = `reference` better**, whichever
        order each pair was stored in.

        !!! note "Technical details"
            Competitors are every group paired with `reference` in
            `self.table`, in table order; pairs not involving `reference`
            are ignored. `effect` is `effect_size` as stored when
            `reference` is `group_a` (`paired_test`'s `diff = group_a -
            group_b`), negated when it is `group_b`, and negated again when
            `higher_is_better=False` (e.g. an error metric). Both
            `"rank_biserial"` and `"cohens_dz"` are antisymmetric in the
            two groups, so the flip is exact. Nothing is recomputed: the
            values come straight from `self.table`, so `p_corrected`/
            `significant` are `NaN`/`None` on a result that was never
            `correct`-ed.

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
            `p_value`, `p_corrected`, `significant`, `effect`,
            `effect_size_type`, `n_pairs`, and `n_eff` when any row has it.
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
        sign = np.where(rows["group_a"] == reference, 1.0, -1.0) * (1.0 if higher_is_better else -1.0)
        out = pd.DataFrame({
            "p_value": rows["p_value"].to_numpy(),
            "p_corrected": rows["p_corrected"].to_numpy(),
            "significant": rows["significant"].to_numpy(),
            "effect": rows["effect_size"].to_numpy(dtype=float) * sign,
            "effect_size_type": rows["effect_size_type"].to_numpy(),
            "n_pairs": rows["n_pairs"].to_numpy(),
        }, index=pd.Index(np.where(rows["group_a"] == reference, rows["group_b"], rows["group_a"]), name=self.group_by))
        if rows["n_eff"].notna().any():
            out["n_eff"] = rows["n_eff"].to_numpy()
        return out

    @classmethod
    def merge(cls, results):
        """
        Combines several results into one -- e.g. to assemble a custom
        family of pairs from separate calls before `correct`-ing them
        together.

        !!! note "Technical details"
            All inputs must share the same
            metric/group_by/test/unit_col/pairing_col. Duplicate pairs are
            deduped, last one wins.

        Parameters
        ----------
        results: list[PairwiseResult]

        Returns
        -------
        PairwiseResult
        """
        if not results:
            raise ValueError("`results` must contain at least one PairwiseResult.")
        first = results[0]
        identity_fields = lambda r: (r.metric, r.group_by, r.test, r.unit_col, r.pairing_col)
        for r in results[1:]:
            if identity_fields(r) != identity_fields(first):
                raise ValueError("Cannot merge PairwiseResults with different metric/group_by/test/unit_col/pairing_col.")
        table = pd.concat([r.table for r in results], ignore_index=True)
        table = table.drop_duplicates(subset=["group_a", "group_b"], keep="last").reset_index(drop=True)
        return cls(
            metric=first.metric, group_by=first.group_by, test=first.test,
            unit_col=first.unit_col, pairing_col=first.pairing_col, subject_order=first.subject_order,
            filters=first.filters, table=table,
        )

    def __repr__(self):
        lines = [f"PairwiseResult(metric={self.metric!r}, test={self.test!r}, n_comparisons={len(self.table)})"]
        for _, row in self.table.iterrows():
            has_corrected = pd.notna(row["p_corrected"])
            p = row["p_corrected"] if has_corrected else row["p_value"]
            tag = "*" if has_corrected and row["significant"] else ""
            n_desc = f"n={row['n_pairs']:.0f}" if pd.notna(row["n_pairs"]) else f"n_a={row['n_a']:.0f}, n_b={row['n_b']:.0f}"
            lines.append(
                f"  {row['group_a']} vs {row['group_b']}: p={p:.4g}{tag}, "
                f"d={row['effect_size']:.3g} ({row['effect_size_type']}), {n_desc}"
            )
        return "\n".join(lines)

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
        PairwiseResult
        """
        folder = Path(folder)
        with open(folder/"identity.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(folder/"table.json", "r", encoding="utf-8") as f:
            records = json.load(f)

        table = pd.DataFrame(records, columns=_PAIRWISE_TABLE_COLUMNS)
        return cls(
            metric=meta["metric"], group_by=meta["group_by"], test=meta["test"],
            unit_col=meta.get("unit_col"), pairing_col=meta.get("pairing_col"),
            subject_order=tuple(meta["subject_order"]) if meta["subject_order"] is not None else None,
            filters=_filters_from_json(meta.get("filters")),
            table=table, computed_at=datetime.fromisoformat(meta["computed_at"]),
        )


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
        the same class and can be `merge`d/`correct`ed identically.
    """

    KIND = "cluster"

    def __init__(
        self, metric, group_by, coord, pairing_col, unit_col, higher_is_better, coord_order,
        cluster_alpha, n_permutations, seed, effect_size_summary, filters,
        clusters_by_pair: dict, t_obs_by_pair: dict, diff_by_pair: dict, computed_at=None,
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
        self.computed_at = computed_at or datetime.now()

    @property
    def identity(self):
        """Hashable tuple describing exactly what this object contains -- metric/groups/method/params, used for `MetricSuite.list_cached_stats()`."""
        pairs = tuple(sorted(self.clusters_by_pair.keys()))
        return (
            self.KIND, self.metric, self.group_by, self.coord, self.pairing_col, self.unit_col,
            self.higher_is_better, self.coord_order, self.cluster_alpha, self.n_permutations,
            self.seed, self.effect_size_summary, self.filters, pairs,
        )

    @classmethod
    def compute(
        cls, df, metric, group_by, coord, comparisons, pairing_col, unit_col=None,
        higher_is_better=True, coord_order=None, cluster_alpha=0.05, n_permutations=3000,
        seed=0, effect_size_summary="cluster_mean", filters=None,
    ):
        """
        Runs the actual permutation tests and builds a `ClusterResult` from
        scratch. Most users won't call this directly --
        `MetricSuite.cluster_permutation` does, with caching -- but it's
        here if you want the raw computation without going through a
        suite.

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
            t_obs_by_pair=t_obs_by_pair, diff_by_pair=diff_by_pair,
        )

    def correct(self, method="fdr_bh", alpha=0.05):
        """
        Adjusts p-values for multiple comparisons across every pair's
        clusters, pooled into one family.

        !!! note "Technical details"
            Pools every cluster's `p_value` across every pair in
            `clusters_by_pair` and corrects together -- mirrors
            `PairwiseResult.correct`. Pairs with no clusters at all are
            left untouched (nothing to correct). Returns a **new**
            `ClusterResult` -- never mutates `self`.

        Parameters
        ----------
        method: str, optional
            Correction method (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        Returns
        -------
        ClusterResult
        """
        non_empty = {pair: cl for pair, cl in self.clusters_by_pair.items() if not cl.empty}
        corrected_by_pair = dict(self.clusters_by_pair)

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

        return ClusterResult(
            metric=self.metric, group_by=self.group_by, coord=self.coord,
            pairing_col=self.pairing_col, unit_col=self.unit_col, higher_is_better=self.higher_is_better,
            coord_order=self.coord_order, cluster_alpha=self.cluster_alpha,
            n_permutations=self.n_permutations, seed=self.seed,
            effect_size_summary=self.effect_size_summary, filters=self.filters,
            clusters_by_pair=corrected_by_pair, t_obs_by_pair=self.t_obs_by_pair,
            diff_by_pair=self.diff_by_pair, computed_at=self.computed_at,
        )

    @classmethod
    def merge(cls, results):
        """
        Combines several results into one -- e.g. to assemble a custom
        family of pairs from separate calls before `correct`-ing them
        together.

        !!! note "Technical details"
            All inputs must share the same
            metric/group_by/coord/pairing_col/unit_col. Overlapping pairs
            are overwritten, last one wins.

        Parameters
        ----------
        results: list[ClusterResult]

        Returns
        -------
        ClusterResult
        """
        if not results:
            raise ValueError("`results` must contain at least one ClusterResult.")
        first = results[0]
        identity_fields = lambda r: (r.metric, r.group_by, r.coord, r.pairing_col, r.unit_col)
        for r in results[1:]:
            if identity_fields(r) != identity_fields(first):
                raise ValueError("Cannot merge ClusterResults with different metric/group_by/coord/pairing_col/unit_col.")
        clusters_by_pair, t_obs_by_pair, diff_by_pair = {}, {}, {}
        for r in results:
            clusters_by_pair.update(r.clusters_by_pair)
            t_obs_by_pair.update(r.t_obs_by_pair)
            diff_by_pair.update(r.diff_by_pair)
        return cls(
            metric=first.metric, group_by=first.group_by, coord=first.coord,
            pairing_col=first.pairing_col, unit_col=first.unit_col, higher_is_better=first.higher_is_better,
            coord_order=first.coord_order, cluster_alpha=first.cluster_alpha,
            n_permutations=first.n_permutations, seed=first.seed,
            effect_size_summary=first.effect_size_summary, filters=first.filters,
            clusters_by_pair=clusters_by_pair, t_obs_by_pair=t_obs_by_pair, diff_by_pair=diff_by_pair,
        )

    def __repr__(self):
        lines = [f"ClusterResult(metric={self.metric!r}, coord={self.coord!r}, n_pairs={len(self.clusters_by_pair)})"]
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
        many-group result are the same class, mergeable into one another.
    """

    KIND = "normality"

    def __init__(self, metric, group_by, test, filters, table: pd.DataFrame, computed_at=None):
        self.metric = metric
        self.group_by = group_by
        self.test = test
        self.filters = filters
        self.table = table.reset_index(drop=True)
        self.computed_at = computed_at or datetime.now()

    @property
    def identity(self):
        """Hashable tuple describing exactly what this object contains -- metric/group/method/params, used for `MetricSuite.list_cached_stats()`."""
        groups = tuple(sorted(json_safe(g) for g in self.table["group"]))
        return (self.KIND, self.metric, self.group_by, self.test, self.filters, groups)

    @classmethod
    def compute(cls, df, metric, group_by=None, test="shapiro", groups=None, filters=None):
        """
        Runs the actual normality test(s) and builds a `NormalityResult`
        from scratch. Most users won't call this directly --
        `MetricSuite.normality_test` does, with caching -- but it's here if
        you want the raw computation without going through a suite.

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
        return cls(metric=metric, group_by=group_by, test=test, filters=filters, table=table)

    def correct(self, method="fdr_bh", alpha=0.05):
        """
        Adjusts p-values for multiple comparisons across every group in
        this result, pooled into one family.

        !!! note "Technical details"
            Returns a **new** `NormalityResult` -- never mutates `self` --
            so the cached raw object stays available for a different
            `alpha`/`method` later without recomputing any test. A
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
        return NormalityResult(
            metric=self.metric, group_by=self.group_by, test=self.test,
            filters=self.filters, table=table, computed_at=self.computed_at,
        )

    @classmethod
    def merge(cls, results):
        """
        Combines several results into one -- e.g. to assemble a custom
        family of groups from separate calls before `correct`-ing them
        together.

        !!! note "Technical details"
            All inputs must share the same metric/group_by/test. Duplicate
            groups are deduped, last one wins.

        Parameters
        ----------
        results: list[NormalityResult]

        Returns
        -------
        NormalityResult
        """
        if not results:
            raise ValueError("`results` must contain at least one NormalityResult.")
        first = results[0]
        for r in results[1:]:
            if (r.metric, r.group_by, r.test) != (first.metric, first.group_by, first.test):
                raise ValueError("Cannot merge NormalityResults with different metric/group_by/test.")
        table = pd.concat([r.table for r in results], ignore_index=True)
        table = table.drop_duplicates(subset=["group"], keep="last").reset_index(drop=True)
        return cls(
            metric=first.metric, group_by=first.group_by, test=first.test,
            filters=first.filters, table=table,
        )

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
            "filters": _filters_to_json(self.filters), "computed_at": self.computed_at.isoformat(),
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
        alpha=None, significant=None, computed_at=None,
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
        self.computed_at = computed_at or datetime.now()

    @property
    def identity(self):
        """Hashable tuple describing exactly what this object contains -- metric/groups/method/params, used for `MetricSuite.list_cached_stats()`."""
        return (
            self.KIND, self.metric, self.group_by, self.pairing_col, self.unit_col,
            self.test, self.filters, self.groups,
        )

    @classmethod
    def compute(cls, df, metric, group_by, pairing_col, unit_col=None, test="friedman", groups=None, filters=None):
        """
        Runs the actual omnibus test and builds an `AnovaResult` from
        scratch. Most users won't call this directly --
        `MetricSuite.stat.anova` does, with caching -- but it's here if you
        want the raw computation without going through a suite.

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
        )

    def correct(self, alpha=0.05):
        """
        Marks this result significant/not at `alpha`.

        !!! note "Technical details"
            No multiple-comparison correction happens here -- this is
            already a single omnibus p-value, not a family -- so this only
            (re-)sets `significant = p_value < alpha`. Returns a **new**
            `AnovaResult` -- never mutates `self` -- so the cached raw
            object stays available for a different `alpha` later without
            rerunning the test.

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
            alpha=alpha, significant=significant, computed_at=self.computed_at,
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
            "alpha": self.alpha, "significant": self.significant,
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
            computed_at=datetime.fromisoformat(meta["computed_at"]),
        )
