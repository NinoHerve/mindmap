"""
Module Name: metric/plot.py
Description: `MetricSuite.plot`'s accessor -- one method per visual style,
bound to a `MetricSuite` instance. Raw views (`box`/`violin`/`bar`/`line`/
`scatter`/`distribution`/`histogram`) take a metric name and draw exactly
what's registered, no test involved. Stat views (`group_comparison`/
`cluster_permutation`/`normality_test`/`anova`) always take an
already-computed `PairwiseResult`/`ClusterResult`/`NormalityResult`/
`AnovaResult` -- never a metric name -- so the two families can never be
confused with each other at a call site. `group_comparison` accepts a
`PairwiseResult` from any of `suite.stat.wilcoxon`/`.ttest`/
`.mannwhitneyu`/`.paired_ttest`/`.ttest_ess`; `paired_differences` draws
the per-unit differences behind a paired one.

Author: Nino Herve
"""

import numpy as np
import pandas as pd

from mindmap.viz.core import (
    metric_box, metric_violin, metric_bar, metric_line, metric_scatter, metric_distribution, metric_histogram,
)
from mindmap.viz.stats import (
    metric_box_significance, metric_relational_clusters, metric_relational_cluster_summary, metric_paired_differences,
)

from mindmap.metric.results import PairwiseResult, ClusterResult, NormalityResult, AnovaResult


def _reference_view(table, group_by, reference):
    """
    Projects a general `PairwiseResult.table` (arbitrary `group_a`/`group_b`
    pairs) onto the shape `mindmap.viz.stats.metric_box_significance`
    expects: one row per competitor actually compared against `reference`,
    named after `group_by` -- pairs not involving `reference` at all are
    dropped (nothing to annotate against a fixed baseline).
    """
    rows = []
    for _, row in table.iterrows():
        if row["group_a"] == reference:
            competitor = row["group_b"]
        elif row["group_b"] == reference:
            competitor = row["group_a"]
        else:
            continue
        rows.append({**row.to_dict(), group_by: competitor})
    return pd.DataFrame(rows)


def _infer_reference(pairs):
    """
    The one group common to every `(group_a, group_b)` pair in `pairs`, if
    there is one -- lets `reference` be omitted for the common "every
    competitor vs. one fixed baseline" case, since the result already
    carries that baseline; returns `None` when the pairs don't share a
    common group (arbitrary many-to-many `comparisons`), in which case the
    caller must be told explicitly which group to draw against.

    A single pair is a special case: both of its groups are trivially
    "common" to it, which would otherwise read as ambiguous even though
    single-pair results are the most common case of all. `group_a` is used
    instead -- `StatAccessor.build_comparisons` always builds
    `(reference, competitor)` in that order, so this is right whenever the
    result's `comparisons` came from there; for a hand-written single-pair
    `comparisons=[(a, b)]`, it's whichever the caller listed first, a
    reasonable default rather than a hard error.
    """
    if len(pairs) == 1:
        return pairs[0][0]
    candidates = None
    for a, b in pairs:
        groups = {a, b}
        candidates = groups if candidates is None else (candidates & groups)
        if not candidates:
            return None
    return next(iter(candidates)) if candidates and len(candidates) == 1 else None


def _filters_to_dict(filters):
    """`PairwiseResult`/`ClusterResult.filters` is stored pre-frozen (`((key, (v1, v2, ...)), ...)`) -- `dict()` alone recovers a `to_df(filters=...)`-compatible mapping."""
    return dict(filters) if filters else {}


def _collapse_for_plot(df, group_by, pairing_col, unit_col):
    """
    Reproduces exactly the collapse `PairwiseResult`/`AnovaResult.compute`
    consumed before testing, so a plot's points always match what the test
    actually saw -- never the raw, uncollapsed rows.

    Mirrors `mindmap.metric.results._collapse`/`_compute_independent_rows`:
    paired (`pairing_col` given) collapses to one point per
    `(group_by, pairing_col)`, first averaging over `unit_col` if given;
    independent (`pairing_col` is `None`) collapses to one point per
    `(group_by, unit_col)` if `unit_col` is given, else leaves `df` as-is.
    """
    metric_key = ["Metric"] if "Metric" in df.columns else []
    if pairing_col is not None:
        if unit_col is not None:
            df = df.groupby([group_by, pairing_col, unit_col] + metric_key, as_index=False)["Value"].mean()
        return df.groupby([group_by, pairing_col] + metric_key, as_index=False)["Value"].mean()
    if unit_col is not None:
        return df.groupby([group_by, unit_col] + metric_key, as_index=False)["Value"].mean()
    return df


_ANNOTATION_KEYS = ("stars", "effect", "winrate", "counts", "n_eff")
_EFFECT_SYMBOLS = {"cohens_dz": "dz", "rank_biserial": "r"}


def _stars(p):
    """Significance stars at 0.001/0.01/0.05, `"ns"` otherwise (including a missing p-value)."""
    if pd.isna(p):
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


class PlotAccessor:
    """
    `MetricSuite.plot`'s namespace -- bound to one suite, accessed as
    `suite.plot.box(...)`, `suite.plot.group_comparison(...)`, etc. See
    `MetricSuite.plot`'s own docstring for the full list.
    """

    def __init__(self, suite):
        self._suite = suite

    def _metric_df(self, metric, filters):
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        return df

    # -- Raw views: metric name in, no test, one visual style each ----------

    def box(self, metric, group_by, filters=None, ax=None, **kwargs):
        """
        Box plot of `metric` per group -- raw data only, no test. See
        `metric_box` for `**kwargs` (e.g. `points=`, `order=`, `palette=`).

        Parameters
        ----------
        metric: str
            Which registered metric to plot.

        group_by: str
            Which registered column to group/color by.

        filters: dict, optional
            Only plot rows matching these `{column: value(s)}` pairs.

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        df = self._metric_df(metric, filters)
        return metric_box(df, group_name=group_by, ax=ax, **kwargs)

    def violin(self, metric, group_by, filters=None, ax=None, **kwargs):
        """Violin plot of `metric` per group -- raw data only, no test. See `metric_violin` for `**kwargs`."""
        df = self._metric_df(metric, filters)
        return metric_violin(df, group_name=group_by, ax=ax, **kwargs)

    def bar(self, metric, group_by, filters=None, ax=None, **kwargs):
        """Bar chart of `metric` per group -- raw data only, no test. See `metric_bar` for `**kwargs`."""
        df = self._metric_df(metric, filters)
        return metric_bar(df, group_name=group_by, ax=ax, **kwargs)

    def line(self, metric, group_by, x, filters=None, ax=None, **kwargs):
        """
        Line plot of `metric` against coordinate `x`, one line per group --
        raw data only, no test. See `metric_line` for `**kwargs` (e.g.
        `estimator=`, `errorbar=`).

        Parameters
        ----------
        metric, group_by, filters, ax
            See `box`.

        x: str
            Coordinate column to use as the x-axis (e.g. "snr", "times").
        """
        df = self._metric_df(metric, filters)
        return metric_line(df, x=x, group_name=group_by, ax=ax, **kwargs)

    def scatter(self, metric, group_by, x, filters=None, ax=None, **kwargs):
        """Scatter plot of `metric` against coordinate `x`, one color per group -- raw data only, no test. See `metric_scatter` for `**kwargs`."""
        df = self._metric_df(metric, filters)
        return metric_scatter(df, x=x, group_name=group_by, ax=ax, **kwargs)

    def distribution(self, metric, group_by=None, filters=None, ax=None, **kwargs):
        """
        Kernel-density curve of `metric`'s distribution -- raw data only, no
        test. See `metric_distribution` for `**kwargs`.

        !!! note "Technical details"
            Unlike `box`/`violin`/`bar`/`line`/`scatter`, `group_by` is
            optional here (default `None`) -- `metric`'s value is the x-axis
            itself, so a single ungrouped curve is already a meaningful
            plot. When given, groups are overlaid as translucent filled
            curves (`alpha=0.35` by default) rather than faceted into
            separate axes.

        Parameters
        ----------
        metric: str
            Which registered metric to plot.

        group_by: str, optional
            Which registered column to color-overlay by. Omit for one curve
            over the whole population.

        filters: dict, optional
            Only plot rows matching these `{column: value(s)}` pairs.

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        df = self._metric_df(metric, filters)
        return metric_distribution(df, group_name=group_by, ax=ax, **kwargs)

    def histogram(self, metric, group_by=None, filters=None, ax=None, **kwargs):
        """
        Histogram of `metric` -- raw data only, no test. See
        `metric_histogram` for `**kwargs`. `group_by` is optional (default
        `None`), same convention as `distribution` -- see its technical
        details.
        """
        df = self._metric_df(metric, filters)
        return metric_histogram(df, group_name=group_by, ax=ax, **kwargs)

    # -- Stat views: a computed result in, never a metric name ---------------

    def group_comparison(
        self, result, reference=None, alpha=0.05, ax=None, *, chance=None, chance_value=None,
        reference_lines=None, fill_alpha=None, tick_labels=None, reference_suffix=None, **kwargs,
    ):
        """
        Draws `result` (from `suite.stat.wilcoxon`/`.ttest`/
        `.mannwhitneyu`/`.paired_ttest`/`.ttest_ess`) as an annotated box
        plot -- boxes per group, with significance stars/brackets against
        `reference`.

        !!! note "Technical details"
            Everything about *what* was tested (`metric`, `group_by`,
            `pairing_col`, `unit_col`, `filters`) comes from `result`
            itself -- never passed here. `reference` is inferred
            automatically when every pair in `result.table` shares one
            common group (the usual "every competitor vs. one baseline"
            case); only an arbitrary many-to-many `comparisons` result,
            with no common group, needs it given explicitly. `alpha` is
            purely a plotting choice (which pairs get a star, tiered at
            0.05/0.01/0.001) -- independent of whatever `alpha`/
            `correction` were used to compute `result.table["p_corrected"]`
            in the first place.

        Parameters
        ----------
        result: PairwiseResult
            An already-computed result, e.g. from
            `suite.stat.wilcoxon(metric, group_by, comparisons=..., pairing_col=...)`.

        reference: str, optional
            The baseline every bracket is drawn against. Inferred from
            `result` when every pair shares one common group; required
            otherwise.

        alpha: float, optional
            Significance threshold for which pairs get annotated (default
            0.05) -- see technical details above.

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        chance: tuple[float, float], optional
            `(lo, hi)` range shaded in gray behind the boxes (e.g. a
            permutation chance range).

        chance_value: float, optional
            Dotted horizontal line behind the boxes (e.g. 0 or 0.5).

        reference_lines: dict, optional
            `{label: value}` or `{label: (value, color)}` -- dashed
            horizontal lines behind the boxes, labelled for `ax.legend()`
            (e.g. other methods' medians).

        fill_alpha: float, optional
            Box fill opacity in [0, 1], edges kept opaque. Default `None`
            keeps seaborn's opaque fills.

        tick_labels: dict, optional
            `{group: display name}` for the x tick labels.

        reference_suffix: str, optional
            Appended to `reference`'s tick label, e.g. `" (ref.)"`.

        **kwargs
            Passed straight through to `metric_box`.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if not isinstance(result, PairwiseResult):
            raise TypeError(
                f"suite.plot.group_comparison() expects a PairwiseResult (from suite.stat.wilcoxon(...)/.ttest(...)/"
                f".mannwhitneyu(...)/.paired_ttest(...)/.ttest_ess(...)), got {type(result).__name__}. "
                f"For a ClusterResult, use suite.plot.cluster_permutation() instead."
            )

        pairs = list(zip(result.table["group_a"], result.table["group_b"]))
        if reference is None:
            reference = _infer_reference(pairs)
        if reference is None:
            raise ValueError(
                "`reference` is required -- this result's pairs don't share one common group "
                "for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
            )

        df = self._suite.to_df(metrics=result.metric, filters=_filters_to_dict(result.filters))
        if df.empty:
            raise ValueError(f"No data registered for metric '{result.metric}'.")

        collapsed_df = _collapse_for_plot(df, result.group_by, result.pairing_col, result.unit_col)

        view = _reference_view(result.table, result.group_by, reference)
        if fill_alpha is not None:
            kwargs["fill_alpha"] = fill_alpha
        return metric_box_significance(
            collapsed_df, view, group_name=result.group_by, reference=reference, alpha=alpha, ax=ax,
            chance=chance, chance_value=chance_value, reference_lines=reference_lines,
            tick_labels=tick_labels, reference_suffix=reference_suffix, **kwargs,
        )

    def paired_differences(
        self, result, reference=None, higher_is_better=True, order=None, palette=None,
        annotate=("stars", "effect"), violin_min_n=30, fill_alpha=0.45, seed=0, ax=None, *,
        tick_labels=None, direction_labels=None,
    ):
        """
        Draws the per-unit paired differences behind `result` (from
        `suite.stat.wilcoxon`/`.paired_ttest`/`.ttest_ess`), `reference`
        minus each competitor, one column per competitor -- violins for
        large samples, points for small ones -- annotated with each test's
        significance, effect size and win rate.

        !!! note "Technical details"
            The differences are `suite.stat.paired_differences(result,
            reference, higher_is_better)` -- the exact units each test saw
            -- and every annotation is read from `result.vs(reference,
            higher_is_better)`, so positive always means `reference`
            better, in the plot and in the numbers. Nothing is recomputed.
            Annotation keys, stacked top to bottom in the order given:

            - `"stars"`: `***`/`**`/`*`/`ns` from `p_corrected` (0.001/
              0.01/0.05; falls back to `p_value` on an uncorrected result);
            - `"effect"`: `dz=+0.56` (Cohen's dz) or `r=+0.42`
              (rank-biserial), oriented as above;
            - `"winrate"`: share of units where `reference` is better,
              minus 50% (e.g. `+26%` = better at 76% of units);
            - `"counts"`: better/tied/worse unit counts, e.g. `61/0/39`;
            - `"n_eff"`: `n_eff=35`, only for `ttest_ess` results (no line
              otherwise).

            Win rate and counts come from the plotted differences, so ties
            are units with exactly equal values -- the ones Wilcoxon drops
            from its own `n_pairs`.

        Parameters
        ----------
        result: PairwiseResult
            A paired result, e.g. from
            `suite.stat.ttest_ess(metric, group_by, comparisons=..., pairing_col=...)`.

        reference: str, optional
            The group every difference is taken from. Inferred from
            `result` when every pair shares one common group; required
            otherwise.

        higher_is_better: bool, optional
            Whether a larger metric value is better (default True); when
            False, differences and effects are negated.

        order: list, optional
            Competitors to draw, left to right. Defaults to the order of
            `result.table`.

        palette: dict | str | list, optional
            `{group: color}`, or anything `sns.color_palette` accepts.

        annotate: tuple[str, ...], optional
            Which annotation lines to draw, top to bottom (default
            `("stars", "effect")`); see technical details. `()` for none.

        violin_min_n: int, optional
            Minimum number of units for a violin instead of points
            (default 30).

        fill_alpha: float, optional
            Violin fill opacity (default 0.45).

        seed: int, optional
            Seed for the point jitter (default 0).

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        tick_labels: dict, optional
            `{group: display name}` for the x tick labels.

        direction_labels: tuple[str, str], optional
            `(positive, negative)` labels written right of the axes at
            y = 0, e.g. `("wCSET better", "other method better")`.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if not isinstance(result, PairwiseResult):
            raise TypeError(
                f"suite.plot.paired_differences() expects a PairwiseResult (from suite.stat.wilcoxon(...)/"
                f".paired_ttest(...)/.ttest_ess(...)), got {type(result).__name__}."
            )
        if isinstance(annotate, str):
            annotate = (annotate,)
        unknown = [k for k in annotate if k not in _ANNOTATION_KEYS]
        if unknown:
            raise ValueError(f"Unknown annotation key(s) {unknown}. Choose from {list(_ANNOTATION_KEYS)}.")

        if reference is None:
            reference = _infer_reference(list(zip(result.table["group_a"], result.table["group_b"])))
        if reference is None:
            raise ValueError(
                "`reference` is required -- this result's pairs don't share one common group "
                "for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
            )

        summary = result.vs(reference, higher_is_better=higher_is_better)
        diffs = self._suite.stat.paired_differences(result, reference=reference, higher_is_better=higher_is_better)
        if order is None:
            order = list(summary.index)
        missing = [c for c in order if c not in summary.index]
        if missing:
            raise ValueError(f"`order` entries {missing} were not compared against reference {reference!r} in `result`.")

        annotations = {}
        for c in order:
            row = summary.loc[c]
            d = diffs[c].dropna().to_numpy(dtype=float)
            lines = []
            for key in annotate:
                if key == "stars":
                    p = row["p_corrected"] if pd.notna(row["p_corrected"]) else row["p_value"]
                    lines.append(_stars(p))
                elif key == "effect":
                    symbol = _EFFECT_SYMBOLS.get(row["effect_size_type"], row["effect_size_type"])
                    lines.append(f"{symbol}={row['effect']:+.2f}")
                elif key == "winrate":
                    lines.append(f"{(d > 0).mean() - 0.5:+.0%}")
                elif key == "counts":
                    lines.append(f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}")
                elif key == "n_eff" and pd.notna(row.get("n_eff", np.nan)):
                    lines.append(f"n_eff={row['n_eff']:.0f}")
            annotations[c] = lines

        return metric_paired_differences(
            diffs, order=order, palette=palette, annotations=annotations, violin_min_n=violin_min_n,
            fill_alpha=fill_alpha, seed=seed, tick_labels=tick_labels, direction_labels=direction_labels, ax=ax,
        )

    def cluster_permutation(self, result, reference=None, alpha=0.05, ax=None, **kwargs):
        """
        Draws `result` (from `MetricSuite.cluster_permutation`) as a curve
        plot with significant ranges shaded along the coordinate axis.

        !!! note "Technical details"
            Everything about *what* was tested comes from `result` itself.
            `reference` is inferred automatically when every pair shares
            one common group; only needed explicitly for an arbitrary
            many-to-many result with more than one pair. Auto-switches
            style based on how many pairs `result` holds:

            - exactly one pair: the detailed annotated-curve view
              (`mindmap.viz.stats.metric_relational_clusters`), shaded at
              that pair's own `result.cluster_alpha` (no plot-time `alpha`
              involved).
            - more than one pair: the compact multi-competitor overlay
              (`mindmap.viz.stats.metric_relational_cluster_summary`),
              styled from the pooled `p_corrected` against the plot-time
              `alpha` (default 0.05) -- independent of whatever `alpha`/
              `correction` were used when `result` was computed.

        Parameters
        ----------
        result: ClusterResult
            An already-computed result, e.g. from
            `suite.stat.cluster_permutation(metric, coord, group_by, comparisons=..., pairing_col=...)`.

        reference: str, optional
            The baseline every competitor is styled against in the
            multi-pair summary view. Inferred from `result` when every pair
            shares one common group; required otherwise. Unused for a
            single-pair result.

        alpha: float, optional
            Significance threshold for the multi-pair summary view's
            styling (default 0.05). Unused for a single-pair result (which
            uses `result.cluster_alpha` instead).

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        **kwargs
            Passed straight through to the underlying plotting call.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if not isinstance(result, ClusterResult):
            raise TypeError(
                f"suite.plot.cluster_permutation() expects a ClusterResult (from suite.stat.cluster_permutation(...)), "
                f"got {type(result).__name__}. For a PairwiseResult, use suite.plot.group_comparison() instead."
            )

        pairs = list(result.clusters_by_pair.keys())
        involved = sorted({g for pair in pairs for g in pair})
        df = self._suite.to_df(
            metrics=result.metric, filters={**_filters_to_dict(result.filters), result.group_by: involved},
        )
        if df.empty:
            raise ValueError(f"No data registered for metric '{result.metric}'.")

        if len(pairs) == 1:
            a, b = pairs[0]
            return metric_relational_clusters(
                df, result.clusters_by_pair[(a, b)], x=result.coord, group_name=result.group_by,
                alpha=result.cluster_alpha, ax=ax, **kwargs,
            )

        if reference is None:
            reference = _infer_reference(pairs)
        if reference is None:
            raise ValueError(
                "`reference` is required for the multi-pair summary plot -- this result's pairs don't "
                "share one common group for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
            )

        clusters_by_competitor = {}
        for a, b in pairs:
            if a == reference:
                competitor = b
            elif b == reference:
                competitor = a
            else:
                raise ValueError(
                    "The multi-pair summary plot requires every pair to include `reference` "
                    f"('{reference}') -- got ('{a}', '{b}')."
                )
            clusters_by_competitor[competitor] = result.clusters_by_pair[(a, b)]

        return metric_relational_cluster_summary(
            df, clusters_by_competitor, x=result.coord, reference=reference, coord_order=list(result.coord_order),
            group_name=result.group_by, p_col="p_corrected", alpha=alpha, ax=ax, **kwargs,
        )

    def normality_test(self, result, kind="distribution", ax=None, **kwargs):
        """
        Draws `result` (from `MetricSuite.normality_test`) as a
        distribution or histogram plot, annotated with each group's (or
        the whole population's) p-value and normal/non-normal verdict.

        !!! note "Technical details"
            Everything about *what* was tested (`metric`, `group_by`,
            `filters`) comes from `result` itself -- never passed here.
            Unlike `group_comparison`/`cluster_permutation`, there's no
            separate plot-time `alpha`: every group is always annotated
            (never selectively starred against a threshold), so the
            already-corrected `result.table["significant"]`/`"p_corrected"`
            are used directly, colored red when significant (i.e.
            significantly non-normal) and a neutral gray otherwise.

        Parameters
        ----------
        result: NormalityResult
            An already-computed result, e.g. from
            `suite.stat.normality_test(metric, group_by=...)`.

        kind: str, optional
            Which raw view to annotate: `"distribution"` (default, KDE
            curve, see `distribution`) or `"histogram"` (see `histogram`).

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        **kwargs
            Passed straight through to `metric_distribution`/`metric_histogram`.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if not isinstance(result, NormalityResult):
            raise TypeError(
                f"suite.plot.normality_test() expects a NormalityResult (from suite.stat.normality_test(...)), "
                f"got {type(result).__name__}."
            )

        plot_funcs = {"distribution": metric_distribution, "histogram": metric_histogram}
        if kind not in plot_funcs:
            raise ValueError(f"Kind '{kind}' not supported. Choose from {list(plot_funcs.keys())}.")

        df = self._suite.to_df(metrics=result.metric, filters=_filters_to_dict(result.filters))
        if df.empty:
            raise ValueError(f"No data registered for metric '{result.metric}'.")

        fig, ax = plot_funcs[kind](df, group_name=result.group_by, ax=ax, **kwargs)

        for i, (_, row) in enumerate(result.table.iterrows()):
            label = "whole population" if row["group"] is None else str(row["group"])
            p = row["p_corrected"] if pd.notna(row["p_corrected"]) else row["p_value"]
            verdict = "non-normal" if row["significant"] else "normal"
            color = "tab:red" if row["significant"] else "#333333"
            ax.text(
                0.98, 0.98 - 0.06 * i, f"{label}: p={p:.3g} ({verdict})",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=color,
            )

        return fig, ax

    def anova(self, result, kind="box", ax=None, **kwargs):
        """
        Draws `result` (from `MetricSuite.stat.anova`) as a box/violin/bar
        plot of every included group, annotated with the omnibus verdict
        (test, statistic, p-value, effect size).

        !!! note "Technical details"
            Deliberately draws no per-pair brackets, unlike
            `group_comparison` -- an omnibus test never compared individual
            pairs, only whether the groups differ *at all*. Follow up with
            `suite.plot.group_comparison` for the pairwise picture once a
            significant result here says it's worth asking "which ones".

            Collapsed to one point per `(group_by, pairing_col)` before
            plotting (mean across anything else, e.g. `unit_col`/
            repetitions) -- matching exactly what `AnovaResult.compute`
            itself averaged down to before running the omnibus test, same
            convention `group_comparison`'s plot uses. Plotting the raw,
            uncollapsed rows instead would show more spread than the test
            ever saw.

            Everything about *what* was tested (`metric`, `group_by`,
            `pairing_col`, `unit_col`, `groups`, `filters`) comes from
            `result` itself -- never passed here. No separate plot-time
            `alpha` (unlike `group_comparison`/`cluster_permutation`) --
            the already-set `result.significant`/`result.alpha` are used
            directly.

        Parameters
        ----------
        result: AnovaResult
            An already-computed result, e.g. from
            `suite.stat.anova(metric, group_by, ...)`.

        kind: str, optional
            Which raw view to annotate: `"box"` (default), `"violin"`, or
            `"bar"`.

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        **kwargs
            Passed straight through to `metric_box`/`metric_violin`/`metric_bar`.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if not isinstance(result, AnovaResult):
            raise TypeError(
                f"suite.plot.anova() expects an AnovaResult (from suite.stat.anova(...)), "
                f"got {type(result).__name__}."
            )

        plot_funcs = {"box": metric_box, "violin": metric_violin, "bar": metric_bar}
        if kind not in plot_funcs:
            raise ValueError(f"Kind '{kind}' not supported. Choose from {list(plot_funcs.keys())}.")

        df = self._suite.to_df(
            metrics=result.metric, filters={**_filters_to_dict(result.filters), result.group_by: list(result.groups)},
        )
        if df.empty:
            raise ValueError(f"No data registered for metric '{result.metric}'.")

        collapsed_df = _collapse_for_plot(df, result.group_by, result.pairing_col, result.unit_col)

        fig, ax = plot_funcs[kind](collapsed_df, group_name=result.group_by, ax=ax, **kwargs)

        test_labels = {"friedman": "Friedman", "anova": "Repeated-measures ANOVA"}
        stat_symbols = {"friedman": "χ²", "anova": "F"}
        verdict = "significant" if result.significant else "not significant"
        color = "tab:red" if result.significant else "#333333"
        label = test_labels.get(result.test, result.test)
        symbol = stat_symbols.get(result.test, "stat")
        text = (
            f"{label}: {symbol}={result.statistic:.3g}, p={result.p_value:.3g} ({verdict})\n"
            f"{result.effect_size_type}={result.effect_size:.3g}"
        )
        ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=color)

        return fig, ax
