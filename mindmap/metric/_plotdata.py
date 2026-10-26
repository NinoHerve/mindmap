"""
Module Name: metric/_plotdata.py
Description: The compute half of `MetricSuite.plot` -- everything a figure
shows, worked out before anything is drawn: the per-unit points (collapsed
the same way the tests collapse them), each group's center and interval
from `mindmap.core.stats`, and the annotation text read from a result.
`_draw.py` only turns these numbers into artists; it never computes a
statistic itself, so a figure always shows the numbers reported elsewhere.

Author: Nino Herve
"""

import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats

from mindmap.core.stats import estimate, _check_binary, NULL_EFFECT_VALUES


CENTERS = ("mean", "median", "proportion")
INTERVALS = ("iqr", "sd", "sem", "ci")


def check_summary(center, interval, level):
    """Validates a `center`/`interval`/`level` choice, with the allowed values in the error."""
    if center is not None and center not in CENTERS:
        raise ValueError(f"center={center!r} isn't supported. Choose from {list(CENTERS)} or None.")
    if interval is not None and interval not in INTERVALS:
        raise ValueError(f"interval={interval!r} isn't supported. Choose from {list(INTERVALS)} or None.")
    if interval in ("sd", "sem", "ci") and center is None:
        raise ValueError(f"interval={interval!r} is drawn around a center -- also pass center='mean' or 'median'.")
    if interval == "ci" and not 0 < level < 1:
        raise ValueError(f"`level` must be in (0, 1), got {level!r}.")


def summarize(values, center, interval, level=0.95):
    """
    One group's center and interval, as drawn.

    `center` is "mean", "median" or "proportion" (the share of 1s of a 0/1
    outcome). `"iqr"` is the 25th-75th percentile range, `"sd"`/`"sem"` are
    `center` +/- one standard deviation / standard error, and `"ci"` is the
    `level` confidence interval of `center` from `mindmap.core.stats.estimate`
    -- a t interval for the mean, the exact binomial interval for the
    median, the Wilson interval for a proportion -- the same interval
    `suite.estimate.mean`/`.median`/`.proportion` report. NaN where the
    sample is too small.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.shape[0]
    out = {"n": n, "center": np.nan, "low": np.nan, "high": np.nan}
    if n == 0:
        return out
    if center is not None:
        if center == "proportion":
            _check_binary(v)
        out["center"] = float(np.median(v) if center == "median" else np.mean(v))
    if interval == "iqr":
        out["low"], out["high"] = (float(q) for q in np.percentile(v, [25, 75]))
    elif interval in ("sd", "sem") and n > 1:
        spread = np.std(v, ddof=1) / (np.sqrt(n) if interval == "sem" else 1.0)
        out["low"], out["high"] = out["center"] - spread, out["center"] + spread
    elif interval == "ci":
        est = estimate(v, estimator=center, alpha=1 - level)
        out["low"], out["high"] = est["ci_low"], est["ci_high"]
    return out


def box_summary(values):
    """Matplotlib's box statistics (median, quartiles, 1.5 IQR whiskers, fliers) for one group -- what `ax.bxp` draws."""
    v = np.asarray(values, dtype=float)
    return boxplot_stats(v[np.isfinite(v)])[0]


def collapse(df, keys, unit=None):
    """
    Averages `df` to one "Value" per `keys` (+ `unit`, when given) -- e.g.
    one point per subject per group -- so a raw view shows independent
    units, not every repeated row. Unchanged when `unit` is None.
    """
    if unit is None:
        return df
    keys = [k for k in keys if k is not None]
    return df.groupby(keys + [unit], as_index=False, observed=True)["Value"].mean()


def collapse_for_result(df, group_by, pairing_col, unit_col):
    """
    Reproduces the collapse `PairwiseResult`/`AnovaResult.compute` consumed
    before testing, so a plot's points match what the test actually saw.

    Paired (`pairing_col` given): one point per `(group_by, pairing_col)`,
    first averaging over `unit_col` if given. Independent: one point per
    `(group_by, unit_col)` if `unit_col` is given, else `df` as-is.
    """
    metric_key = ["Metric"] if "Metric" in df.columns else []
    if pairing_col is not None:
        if unit_col is not None:
            df = df.groupby([group_by, pairing_col, unit_col] + metric_key, as_index=False)["Value"].mean()
        return df.groupby([group_by, pairing_col] + metric_key, as_index=False)["Value"].mean()
    if unit_col is not None:
        return df.groupby([group_by, unit_col] + metric_key, as_index=False)["Value"].mean()
    return df


def snap_span(xs, start, end):
    """
    `(start, end)` widened to whole samples of `xs`: from half-way to the
    sample before the first one inside the span, to half-way to the sample
    after the last one (at the ends of `xs`, half the neighbouring
    spacing) -- so a span over 5 samples at 200 Hz is 25 ms wide. Returned
    as given when no sample falls inside.
    """
    xs = np.unique(np.asarray(xs, dtype=float))
    xs = xs[np.isfinite(xs)]
    lo, hi = sorted((float(start), float(end)))
    inside = np.flatnonzero((xs >= lo) & (xs <= hi))
    if not inside.size:
        return lo, hi
    if xs.size < 2:
        return lo, hi
    edges = np.concatenate([[xs[0] - (xs[1] - xs[0]) / 2], (xs[1:] + xs[:-1]) / 2, [xs[-1] + (xs[-1] - xs[-2]) / 2]])
    return float(edges[inside[0]]), float(edges[inside[-1] + 1])


def moving_average(xs, values, width):
    """
    Centered moving average of `values` sampled at `xs`, over a window
    `width` wide in x units: each point is the mean of the finite values
    within `width / 2` of it. The window shrinks at the edges (no
    padding), so the curve covers the whole x range; uneven or unsorted
    `xs` are fine. NaN where the window holds no finite value.
    """
    xs, values = np.asarray(xs, dtype=float), np.asarray(values, dtype=float)
    order = np.argsort(xs, kind="stable")
    sx, sv = xs[order], values[order]
    ok = np.isfinite(sv)
    sums = np.concatenate([[0.0], np.cumsum(np.where(ok, sv, 0.0))])
    counts = np.concatenate([[0], np.cumsum(ok)])
    lo = np.searchsorted(sx, sx - width / 2, side="left")
    hi = np.searchsorted(sx, sx + width / 2, side="right")
    n = counts[hi] - counts[lo]
    smoothed = np.full(sx.shape, np.nan)
    np.divide(sums[hi] - sums[lo], n, out=smoothed, where=n > 0)
    out = np.empty_like(smoothed)
    out[order] = smoothed
    return out


def levels(series, order=None):
    """The levels of `series` to draw, in `order` if given (checked), else in order of first appearance."""
    present = list(pd.unique(series.dropna()))
    if order is None:
        return present
    order = list(order)
    missing = [v for v in order if v not in present]
    if missing:
        raise ValueError(f"`order` entries {missing} aren't in the data (available: {present}).")
    return order


def summary_table(points, by, center, interval, level=0.95, box=False):
    """
    One row per combination of the `by` columns (in the order the points
    are grouped): `n`, `center`, `low`, `high` (see `summarize`), plus the
    box statistics (`median`, `q1`, `q3`, `whislo`, `whishi`) when `box`.
    """
    rows = []
    for key, group in points.groupby(by, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(summarize(group["Value"], center, interval, level))
        if box:
            stats = box_summary(group["Value"])
            row.update({k: float(stats[k]) for k in ("med", "q1", "q3", "whislo", "whishi")})
            row["median"] = row.pop("med")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs.update(center=center, interval=interval, level=level if interval == "ci" else None)
    return table


# -- Annotation text read from a result ---------------------------------------

#: Keys every result view accepts in `annotate`, besides the result's own effect-size names.
ANNOTATION_KEYS = ("stars", "p", "effect", "n_eff")
#: Extra keys only `paired_differences` can draw -- they need the per-unit differences.
DIFFERENCE_ANNOTATION_KEYS = ("winrate", "counts")

_EFFECT_SYMBOLS = {
    "cohens_dz": "dz", "hedges_gz": "gz", "cohens_d": "d", "hedges_g": "g", "rank_biserial": "r", "cles": "CLES",
    "mean_diff": "Δmean", "hodges_lehmann": "HL", "prop_diff": "Δp", "odds_ratio": "OR",
}
#: Effect sizes in the metric's own units -- shown with significant digits, not 2 decimals.
_UNSTANDARDIZED_EFFECT_SIZES = {"mean_diff", "hodges_lehmann", "prop_diff"}
#: Where each effect size means "no difference" (0 for the rest), and those read on a log scale.
NULL_VALUES = NULL_EFFECT_VALUES
LOG_SCALE_EFFECT_SIZES = {"odds_ratio"}


def effect_label(name, value):
    """`r=+0.42`, `HL=+0.0421`, `OR=6` -- one effect size, as annotated on a plot."""
    if name in LOG_SCALE_EFFECT_SIZES:
        number = f"{value:.2g}"
    elif name in _UNSTANDARDIZED_EFFECT_SIZES:
        number = f"{value:+.3g}"
    else:
        number = f"{value:+.2f}"
    return f"{_EFFECT_SYMBOLS.get(name, name)}={number}"


def stars(p):
    """Significance stars at 0.001/0.01/0.05, `"ns"` otherwise (including a missing p-value)."""
    if pd.isna(p):
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def corrected_p(row):
    """`p_corrected`, falling back to `p_value` on an uncorrected result."""
    return row["p_corrected"] if pd.notna(row.get("p_corrected", np.nan)) else row["p_value"]


def check_annotate(annotate, result, allow_differences):
    """Normalizes `annotate` to a tuple and rejects unknown keys, listing what this result offers."""
    if annotate is None:
        return ()
    if isinstance(annotate, str):
        annotate = (annotate,)
    allowed = ANNOTATION_KEYS + (DIFFERENCE_ANNOTATION_KEYS if allow_differences else ())
    unknown = [k for k in annotate if k not in allowed and k not in result.effect_sizes]
    if unknown:
        hint = "" if allow_differences or not set(unknown) & set(DIFFERENCE_ANNOTATION_KEYS) else (
            " ('winrate'/'counts' need the per-unit differences -- use suite.plot.paired_differences)"
        )
        raise ValueError(
            f"Unknown annotation key(s) {unknown}{hint}. Choose from {list(allowed)} or one of this result's "
            f"effect sizes {list(result.effect_sizes)} (request more with the test's `effect_sizes=`)."
        )
    return tuple(annotate)


def check_effect_size(effect_size, result):
    """The effect size the `"effect"` key shows: `result.effect_sizes[0]` by default, validated otherwise."""
    if effect_size is None:
        return result.effect_sizes[0]
    if effect_size not in result.effect_sizes:
        raise ValueError(
            f"effect_size={effect_size!r} isn't in this result (it reports {list(result.effect_sizes)}) -- "
            "request it with the test's `effect_sizes=`."
        )
    return effect_size


def annotation_lines(annotate, row, effect_size, effect_sizes, diffs=None):
    """
    The text lines for one comparison, in `annotate` order, from one row of
    `PairwiseResult.vs(...)` (and the oriented per-unit differences for
    `"winrate"`/`"counts"`). `"n_eff"` adds nothing on a result without it.
    """
    lines = []
    for key in annotate:
        if key == "stars":
            lines.append(stars(corrected_p(row)))
        elif key == "p":
            lines.append(f"p={corrected_p(row):.2g}")
        elif key == "effect":
            lines.append(effect_label(effect_size, row[effect_size]))
        elif key in effect_sizes:
            lines.append(effect_label(key, row[key]))
        elif key == "winrate":
            lines.append(f"{(diffs > 0).mean() - 0.5:+.0%}")
        elif key == "counts":
            lines.append(f"{(diffs > 0).sum()}/{(diffs == 0).sum()}/{(diffs < 0).sum()}")
        elif key == "n_eff" and pd.notna(row.get("n_eff", np.nan)):
            lines.append(f"n_eff={row['n_eff']:.0f}")
    return lines


def infer_reference(pairs):
    """
    The one group common to every `(group_a, group_b)` pair, if there is
    one -- lets `reference` be omitted for the usual "every competitor vs.
    one baseline" result. `None` when the pairs share no group. A single
    pair returns its `group_a` (`StatAccessor.build_comparisons` always
    builds `(reference, competitor)` in that order).
    """
    if len(pairs) == 1:
        return pairs[0][0]
    candidates = None
    for a, b in pairs:
        candidates = {a, b} if candidates is None else (candidates & {a, b})
        if not candidates:
            return None
    return next(iter(candidates)) if candidates and len(candidates) == 1 else None
