"""
Module Name: metric/plot.py
Description: `MetricSuite.plot`'s accessor -- premade figures with the
statistics computed by the suite. Raw views (`box`/`violin`/`bar`/`point`/
`strip`/`line`/`scatter`/`distribution`/`histogram`) take a metric name;
stat views (`group_comparison`/`paired_differences`/`cluster_permutation`/
`normality_test`/`anova`/`estimate`) take an already-computed result, never
a metric name, so the two families can't be confused at a call site.

Every method computes first (`_plotdata.py`: per-unit points, centers and
intervals from `mindmap.core.stats`, annotation text from the result), then
draws with plain matplotlib (`_draw.py`), and returns a `Plot` -- the axes,
the numbers drawn and the artists by layer -- that still unpacks as
`fig, ax`. Layout is the caller's: pass `ax=` to draw into your own
subplots.

Author: Nino Herve
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mindmap.core.stats import BINARY_EFFECT_SIZES
from mindmap.metric import _draw
from mindmap.metric._plotdata import (
    check_summary, summary_table, box_summary, collapse, collapse_for_result, levels, check_annotate, check_effect_size,
    annotation_lines, infer_reference, corrected_p, moving_average, snap_span, NULL_VALUES, LOG_SCALE_EFFECT_SIZES,
)
from mindmap.metric.results import (
    PairwiseResult, ClusterResult, NormalityResult, AnovaResult, EstimateResult, _collapse_with_coord,
)


_AUTO = "auto"


def set_style(fontsize=None):
    """
    Sets the style every `MetricSuite.plot` method shares, for the rest of
    the session -- see `style` for a temporary one.

    Parameters
    ----------
    fontsize: float, optional
        One size for axis labels, tick labels, titles, legends and the
        annotations (effect sizes, p-values, direction labels; secondary
        annotation lines one point smaller), so panels from different
        methods match. None (default) keeps matplotlib's rcParams and the
        built-in annotation sizes.
    """
    if fontsize is not None and (isinstance(fontsize, (bool, np.bool_)) or not np.isscalar(fontsize) or fontsize <= 0):
        raise ValueError(f"`fontsize` must be a positive number of points, got {fontsize!r}.")
    _draw.STYLE["fontsize"] = fontsize


@contextmanager
def style(fontsize=None):
    """
    `set_style` for the plots drawn inside a `with` block only, e.g.
    `with mindmap.metric.plot.style(fontsize=8): suite.plot.line(...)`.
    """
    previous = dict(_draw.STYLE)
    set_style(fontsize)
    try:
        yield
    finally:
        _draw.STYLE.update(previous)

#: Per categorical kind: whether the per-unit points are overlaid by default.
_KIND_POINTS = {"box": True, "violin": False, "bar": False, "point": False, "strip": True}
#: The center that matches each test, drawn per group by the stat views with `center="auto"`.
_TEST_CENTER = {
    "wilcoxon": "median", "mannwhitneyu": "median", "friedman": "median",
    "paired_ttest": "mean", "ttest": "mean", "ttest_ess": "mean", "anova": "mean",
    "mcnemar": "proportion",
}
#: The group center that matches each bootstrapped estimator -- a bootstrap test's `"auto"` center.
_ESTIMATOR_CENTER = {
    "mean_diff": "mean", "cohens_dz": "mean", "hedges_gz": "mean", "cohens_d": "mean", "hedges_g": "mean",
    "hodges_lehmann": "median", "rank_biserial": "median", "cles": "median",
    "prop_diff": "proportion", "odds_ratio": "proportion",
}
#: Estimates in the metric's own units -- the only ones `paired_differences` can draw over the differences.
_UNIT_ESTIMATORS = {"mean_diff", "hodges_lehmann", "prop_diff"}


def _filters_to_dict(filters):
    """`PairwiseResult`/`ClusterResult.filters` is stored pre-frozen (`((key, (v1, v2, ...)), ...)`) -- `dict()` alone recovers a `to_df(filters=...)`-compatible mapping."""
    return dict(filters) if filters else {}


def _namer(labels):
    """`level -> display name`, from an optional `{level: name}` dict."""
    if labels is not None and not isinstance(labels, dict):
        raise TypeError(f"`labels` must be a dict {{level: display name}}, got {type(labels).__name__}.")
    return lambda v: str((labels or {}).get(v, v))


def _per_level_kws(line_kws, hue_levels, highlight, hue):
    """
    `(curve_kws, line_kws)` for `_draw.lines`: per-level styles from
    `highlight` and a `{level: kws}` `line_kws`, and the keywords left for
    every line.
    """
    per_level = (
        isinstance(line_kws, dict) and hue is not None and bool(line_kws)
        and all(k in hue_levels for k in line_kws) and all(isinstance(v, dict) for v in line_kws.values())
    )
    curve_kws = {}
    if highlight is not None:
        if hue is None:
            raise ValueError("`highlight` picks `hue` levels -- pass `hue`.")
        chosen = list(highlight) if isinstance(highlight, (list, tuple, set)) else [highlight]
        missing = [h for h in chosen if h not in hue_levels]
        if missing:
            raise ValueError(f"`highlight` levels {missing} aren't `hue` levels ({list(hue_levels)}).")
        curve_kws = {hv: {"linewidth": 2.2, "zorder": 3} if hv in chosen else {"linewidth": 1.5} for hv in hue_levels}
    if per_level:
        curve_kws = {hv: _draw.merge(curve_kws.get(hv, {}), line_kws.get(hv)) for hv in hue_levels}
        line_kws = None
    return curve_kws, line_kws


def _span_spec(span):
    """`(start, end[, label[, kws]])` -> `(start, end, label, kws)`."""
    if not isinstance(span, (tuple, list)) or not 2 <= len(span) <= 4:
        raise ValueError(f"Each span is `(start, end)`, `(start, end, label)` or `(start, end, label, kws)`, got {span!r}.")
    start, end, label, kws = (*span, None, None)[:4]
    return float(start), float(end), label, kws


def _vline_spec(vline):
    """`x` or `(x[, label[, kws]])` -> `(x, label, kws)`."""
    if np.isscalar(vline):
        return float(vline), None, None
    if not isinstance(vline, (tuple, list)) or not 1 <= len(vline) <= 3:
        raise ValueError(f"Each vline is `x`, `(x, label)` or `(x, label, kws)`, got {vline!r}.")
    x, label, kws = (*vline, None, None)[:3]
    return float(x), label, kws


def _test_center(result):
    """The group center that matches `result`'s test (for a bootstrap test, its estimator)."""
    if result.test == "bootstrap":
        return _ESTIMATOR_CENTER[result.estimator]
    return _TEST_CENTER[result.test]


def _stat_defaults(kind, result, center, interval, points):
    """
    Fills a stat view's `"auto"` center/interval/points: the center that
    matches `result`'s test with its confidence interval -- none on a box,
    which already shows median and IQR -- and `kind`'s default for the
    points.
    """
    if kind not in _KIND_POINTS:
        raise ValueError(f"kind={kind!r} isn't supported. Choose from {list(_KIND_POINTS)}.")
    return (
        (None if kind == "box" else _test_center(result)) if center == _AUTO else center,
        (None if kind == "box" else "ci") if interval == _AUTO else interval,
        _KIND_POINTS[kind] if points == _AUTO else points,
    )


def _has_ci(result, estimator, level):
    """Whether `result`'s table holds `estimator`'s interval at `level` (so nothing needs recomputing)."""
    return (
        result.ci and estimator in result.effect_sizes and result.ci_methods.get(estimator) is not None
        and np.isclose(result.ci_level, level)
    )


def _require_reference(reference, pairs):
    if reference is None:
        reference = infer_reference(pairs)
    if reference is None:
        raise ValueError(
            "`reference` is required -- this result's pairs don't share one common group "
            "for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
        )
    return reference


class Plot:
    """
    What every `suite.plot` method returns: the drawn axes, the numbers it
    shows, and its artists by layer, for restyling after the fact.

    Unpacks as `fig, ax = suite.plot.box(...)`.

    Attributes
    ----------
    ax: matplotlib.axes.Axes
    fig: matplotlib.figure.Figure
    kind: str
        Which plot this is (e.g. "box", "group_comparison").
    data: pd.DataFrame
        The summary drawn: one row per group (and hue level / coordinate)
        with `n`, `center`, `low`, `high` (plus the box statistics on a
        box plot, the annotation text on a stat view). `data.attrs` records
        `center`, `interval` and `level`.
    points: pd.DataFrame | None
        The per-unit values drawn (after collapsing to `unit`/the test's
        pairing), when the plot shows them.
    artists: dict
        Matplotlib artists by layer, e.g. `"boxes"`, `"points"`,
        `"centers"`, `"intervals"`, `"brackets"`, `"text"`.
    """

    def __init__(self, ax, kind, data=None, points=None, artists=None):
        self.ax = ax
        self.kind = kind
        self.data = data
        self.points = points
        self.artists = artists or {}

    @property
    def fig(self):
        return self.ax.figure

    def __iter__(self):
        yield self.fig
        yield self.ax

    def __repr__(self):
        layers = ", ".join(k for k, v in self.artists.items() if v)
        rows = 0 if self.data is None else len(self.data)
        return f"Plot(kind={self.kind!r}, rows={rows}, layers=[{layers}])"


class PlotAccessor:
    """
    `MetricSuite.plot`'s namespace -- premade figures with the suite's
    statistics, one method per figure type.

    !!! note "Common arguments"
        Every method takes the same names for the same things:

        - **Variables:** `metric` is the y-axis. `x` is the category (or
          coordinate) on the x-axis, `hue` an optional second grouping
          drawn side by side and colored. `unit` averages the rows to one
          value per unit (e.g. per subject) within each `x`/`hue` cell
          before anything is computed -- otherwise every row is a point and
          an interval treats repeated rows as independent. Stat views take
          all of this from the result.
        - **Estimates:** `center` is "mean", "median" or None. `interval`
          is "iqr" (25th-75th percentile), "sd", "sem", "ci" (the `level`
          confidence interval of `center`, the same one `suite.estimate`
          reports -- t for the mean, exact binomial for the median) or
          None. A box already shows median and IQR; `center`/`interval`
          there draw a marker with an error bar on top.
        - **Text:** `labels={level: "display name"}` renames groups in
          ticks and legends; `xlabel`, `ylabel`, `title` (default: the
          column/metric names, `""` for none); `legend` None (automatic),
          True, False or "outside". Stat views pick the numbers shown with
          `annotate`. The categorical views take `rotate_ticks` ("auto"
          by default: 45 degrees only when the tick labels would overlap,
          judged at the axes' size when drawn; or an angle, or False).
          One font size for every method: `mindmap.metric.plot.set_style
          (fontsize=8)`, or the `style(...)` context manager.
        - **Colors:** `palette` is a `{level: color}` dict, a list, or a
          colormap name; None uses the active matplotlib color cycle, so
          `plt.style.use(...)` themes apply. `fill_alpha` sets the opacity
          of fills (boxes, violins, bars, bands, densities).
        - **Fine styling:** one `*_kws` dict per layer, passed to exactly
          one matplotlib call over the defaults: `box_kws` (`ax.bxp`),
          `violin_kws` (`ax.violinplot`), `bar_kws` (`ax.bar`),
          `points_kws` (`ax.scatter`), `center_kws` (`ax.plot`, the center
          marker), `interval_kws` (`ax.errorbar`, or `ax.fill_between` for a
          band), `line_kws` (`ax.plot`), `fill_kws`, `hist_kws`,
          `bracket_kws`, `text_kws`, `span_kws`.
        - **Layout:** `ax=` draws into your own axes (e.g. one of
          `plt.subplots`); otherwise a new figure is made.

        Every method returns a `Plot` (axes, numbers drawn, artists),
        which unpacks as `fig, ax`.
    """

    def __init__(self, suite):
        self._suite = suite

    def _metric_df(self, metric, filters):
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        return df

    def _result_df(self, result, extra_filters=None):
        self._suite._require_origin(result)
        df = self._suite.to_df(metrics=result.metric, filters={**_filters_to_dict(result.filters), **(extra_filters or {})})
        if df.empty:
            raise ValueError(f"No data registered for metric '{result.metric}'.")
        return df

    # -- Shared categorical body ---------------------------------------------

    def _categorical(
        self, kind, points, x, hue=None, order=None, hue_order=None, center=None, interval=None, level=0.95,
        show_points=False, labels=None, palette=None, fill_alpha=None, seed=0, ax=None, kws=None, cell_kinds=None,
        overrides=None,
    ):
        """
        Draws `points` (one row per unit, "Value" column) as `kind` per `x`
        (and `hue`) level -- `cell_kinds={x level: kind}` overrides it per
        level, `overrides={x level: {"center"/"low"/"high": value}}` the
        summary drawn (e.g. a test's own estimate); returns the `Plot`,
        `{(x, hue): position}`, and the lowest/highest y drawn.
        """
        check_summary(center, interval, level)
        if kind == "bar" and center is None:
            raise ValueError("A bar's height is its center -- pass center='mean' or 'median'.")
        names = _namer(labels)
        x_levels = levels(points[x], order)
        hue_levels = levels(points[hue], hue_order) if hue is not None else [None]
        by = [x] + ([hue] if hue is not None else [])
        table = summary_table(points, by, center, interval, level, box=(kind == "box"))
        colors = _draw.resolve_colors(hue_levels if hue is not None else x_levels, palette)

        slot = 0.8 / len(hue_levels)
        rows = {tuple(r[c] for c in by): r for r in table.to_dict("records")}
        for xv, values in (overrides or {}).items():
            if (xv,) in rows:
                rows[(xv,)].update(values)
        draw_center = center is not None or any("center" in v for v in (overrides or {}).values())
        draw_interval = interval is not None or any("low" in v for v in (overrides or {}).values())
        cells, positions, ordered = [], {}, []
        for i, xv in enumerate(x_levels):
            for j, hv in enumerate(hue_levels):
                key = (xv,) if hue is None else (xv, hv)
                if key not in rows:
                    continue
                pos = i + (j - (len(hue_levels) - 1) / 2) * slot
                mask = points[x] == xv if hue is None else (points[x] == xv) & (points[hue] == hv)
                row = rows[key]
                positions[(xv, hv)] = pos
                ordered.append({**row, "position": pos})
                cells.append({
                    "pos": pos, "width": slot * 0.75, "color_key": xv if hue is None else hv,
                    "values": points.loc[mask, "Value"].to_numpy(dtype=float),
                    "center": row["center"], "low": row["low"], "high": row["high"],
                    "kind": (cell_kinds or {}).get(xv),
                })
        for c in cells:
            c["box"] = box_summary(c["values"]) if kind == "box" else None

        ax = _draw.get_ax(ax)
        artists, bottom, top = _draw.categorical(
            ax, cells, kind, colors, fill_alpha=fill_alpha, points=show_points, seed=seed,
            center=draw_center, interval=draw_interval, **(kws or {}),
        )
        _draw.tick_labels(ax, range(len(x_levels)), [names(v) for v in x_levels])
        ax.set_xlim(-0.5, len(x_levels) - 0.5)
        if hue is not None:
            artists["legend_handles"] = _draw.legend_handles(kind, hue_levels, colors, names, fill_alpha)

        data = pd.DataFrame(ordered)
        data.attrs.update(table.attrs)
        return Plot(ax, kind, data=data, points=points, artists=artists), positions, bottom, top

    def _raw_categorical(
        self, kind, metric, x, hue, unit, filters, order, hue_order, center, interval, level, points, labels,
        palette, fill_alpha, xlabel, ylabel, title, legend, seed, ax, kws, rotate_ticks="auto",
    ):
        if x is None:
            raise ValueError(f"suite.plot.{kind}() needs `x`, the column whose levels go on the x-axis.")
        df = collapse(self._metric_df(metric, filters), [x, hue], unit)
        plot, _, _, _ = self._categorical(
            kind, df, x, hue, order, hue_order, center, interval, level, points, labels, palette, fill_alpha, seed, ax, kws,
        )
        _draw.finish(
            plot.ax, x if xlabel is None else xlabel, metric if ylabel is None else ylabel, title,
            legend=(hue is not None) if legend is None else legend, handles=plot.artists.get("legend_handles"),
            legend_title=hue,
        )
        _draw.rotate_ticks(plot.ax, rotate_ticks)
        return plot

    # -- Raw views: metric name in, no test ------------------------------------

    def box(
        self, metric, x, hue=None, *, unit=None, filters=None, order=None, hue_order=None, center=None, interval=None,
        level=0.95, points=True, labels=None, palette=None, fill_alpha=None, xlabel=None, ylabel=None, title=None,
        legend=None, seed=0, ax=None, box_kws=None, points_kws=None, center_kws=None, interval_kws=None,
        rotate_ticks="auto",
    ):
        """
        Box plot of `metric` per `x` level: median, quartiles, 1.5 IQR
        whiskers, with the per-unit points on top (`points=False` to hide
        them; outliers are then drawn as fliers). See the class docstring
        for the common arguments.

        Parameters
        ----------
        metric: str
            Which registered metric to plot.

        x: str
            Column whose levels go on the x-axis (e.g. "method").

        hue: str, optional
            Second grouping, drawn side by side within each `x` level.

        unit: str, optional
            Average to one value per `unit` (e.g. "subject") per cell first.

        filters: dict, optional
            Only plot rows matching these `{column: value(s)}` pairs.

        center, interval: str, optional
            Overlay a marker with an error bar, e.g. `center="mean",
            interval="ci"` on top of the box. Default: none.

        Returns
        -------
        Plot
        """
        return self._raw_categorical(
            "box", metric, x, hue, unit, filters, order, hue_order, center, interval, level, points, labels, palette,
            fill_alpha, xlabel, ylabel, title, legend, seed, ax,
            dict(box_kws=box_kws, points_kws=points_kws, center_kws=center_kws, interval_kws=interval_kws),
            rotate_ticks,
        )

    def violin(
        self, metric, x, hue=None, *, unit=None, filters=None, order=None, hue_order=None, center="median",
        interval="iqr", level=0.95, points=False, labels=None, palette=None, fill_alpha=None, xlabel=None, ylabel=None,
        title=None, legend=None, seed=0, ax=None, violin_kws=None, points_kws=None, center_kws=None, interval_kws=None,
        rotate_ticks="auto",
    ):
        """
        Violin plot of `metric` per `x` level, with the median and IQR
        drawn inside by default. Same arguments as `box`.
        """
        return self._raw_categorical(
            "violin", metric, x, hue, unit, filters, order, hue_order, center, interval, level, points, labels, palette,
            fill_alpha, xlabel, ylabel, title, legend, seed, ax,
            dict(violin_kws=violin_kws, points_kws=points_kws, center_kws=center_kws, interval_kws=interval_kws),
            rotate_ticks,
        )

    def bar(
        self, metric, x, hue=None, *, unit=None, filters=None, order=None, hue_order=None, center="mean",
        interval="ci", level=0.95, points=False, labels=None, palette=None, fill_alpha=None, xlabel=None, ylabel=None,
        title=None, legend=None, seed=0, ax=None, bar_kws=None, points_kws=None, interval_kws=None, rotate_ticks="auto",
    ):
        """
        Bar chart of `metric` per `x` level: the bar is `center` (default
        mean), the error bar `interval` (default 95% CI). Same arguments as
        `box`.
        """
        return self._raw_categorical(
            "bar", metric, x, hue, unit, filters, order, hue_order, center, interval, level, points, labels, palette,
            fill_alpha, xlabel, ylabel, title, legend, seed, ax,
            dict(bar_kws=bar_kws, points_kws=points_kws, interval_kws=interval_kws),
            rotate_ticks,
        )

    def point(
        self, metric, x, hue=None, *, unit=None, filters=None, order=None, hue_order=None, center="mean",
        interval="ci", level=0.95, points=False, labels=None, palette=None, fill_alpha=None, xlabel=None, ylabel=None,
        title=None, legend=None, seed=0, ax=None, points_kws=None, center_kws=None, interval_kws=None,
        rotate_ticks="auto",
    ):
        """
        `center` as a marker with `interval` as an error bar, per `x` level
        (default mean and 95% CI). Same arguments as `box`.
        """
        return self._raw_categorical(
            "point", metric, x, hue, unit, filters, order, hue_order, center, interval, level, points, labels, palette,
            fill_alpha, xlabel, ylabel, title, legend, seed, ax,
            dict(points_kws=points_kws, center_kws=center_kws, interval_kws=interval_kws),
            rotate_ticks,
        )

    def strip(
        self, metric, x, hue=None, *, unit=None, filters=None, order=None, hue_order=None, center="mean",
        interval="ci", level=0.95, labels=None, palette=None, xlabel=None, ylabel=None, title=None, legend=None,
        seed=0, ax=None, points_kws=None, center_kws=None, interval_kws=None, rotate_ticks="auto",
    ):
        """
        Every unit as a jittered point per `x` level, with `center` and
        `interval` on top (default mean and 95% CI). Same arguments as `box`.
        """
        return self._raw_categorical(
            "strip", metric, x, hue, unit, filters, order, hue_order, center, interval, level, True, labels, palette,
            None, xlabel, ylabel, title, legend, seed, ax,
            dict(points_kws=points_kws, center_kws=center_kws, interval_kws=interval_kws),
            rotate_ticks,
        )

    def line(
        self, metric, x, hue=None, *, unit=None, filters=None, order=None, hue_order=None, center="mean",
        interval="ci", level=0.95, interval_style="band", labels=None, palette=None, fill_alpha=None, xlabel=None,
        ylabel=None, title=None, legend=None, ax=None, line_kws=None, interval_kws=None, smooth=None, raw=None,
        raw_kws=None, highlight=None, spans=None, vlines=None, span_snap=False, span_kws=None, vline_kws=None,
    ):
        """
        `center` of `metric` against coordinate `x`, one line per `hue`
        level, with `interval` as a shaded band (`interval_style="bars"`
        for error bars). Default mean and 95% CI -- e.g. `center="median",
        interval="iqr"` for median and IQR. See the class docstring for the
        common arguments.

        !!! note "Technical details"
            A numeric `x` is drawn at its values (in `order` when given,
            e.g. a descending SNR sweep); any other `x` as evenly spaced
            categories, sorted unless `order` is given.

            `smooth` is for display only: each curve's center -- and its
            interval, bound by bound -- becomes a centered moving average
            over a window `smooth` wide in `x` units, applied after the
            summary. The window shrinks at the edges (no padding), so the
            curve covers the whole range, but an edge point averages one
            side only. A smoothed band is the average of the pointwise
            intervals, not an interval of the smoothed curve. Inference
            should use the unsmoothed values, which `Plot.data` keeps
            (`center`/`low`/`high`) next to the smoothed ones
            (`center_smooth`/`low_smooth`/`high_smooth`).

        Parameters
        ----------
        metric: str
            Which registered metric to plot.

        x: str
            Coordinate column for the x-axis (e.g. "snr", "times").

        hue: str, optional
            One line per level of this column.

        unit: str, optional
            Average to one value per `unit` per `x`/`hue` point first, so
            the interval is across units.

        smooth: float, optional
            Moving-average window width in `x` units (e.g. `0.025` for 25
            ms on "times"); numeric `x` only. See technical details.

        raw: bool, optional
            With `smooth`: also draw the unsmoothed center under each
            smoothed curve, same color, thin and faint (default True, so
            smoothing never hides the data). Style it with `raw_kws`
            (`ax.plot`).

        highlight: level | list, optional
            `hue` level(s) to emphasize: drawn thicker (2.2 vs 1.5 for the
            others) and on top. Only the (smoothed) curve -- the faint raw
            one stays thin. The legend shows the same widths.

        line_kws: dict, optional
            `ax.plot` keywords for every line, or `{hue level: kws}` to
            style lines one by one (when every key is a hue level), applied
            over `highlight`.

        spans: list, optional
            Shaded x-ranges drawn under the curves, as `(start, end)`,
            `(start, end, label)` or `(start, end, label, kws)` -- e.g.
            the samples around a spike peak. Labelled ones get a legend
            entry. Styled by `span_kws` (`ax.axvspan`), then each span's
            own `kws`. Numeric `x` only.

        vlines: list, optional
            Vertical markers, as `x`, `(x, label)` or `(x, label, kws)`;
            labelled ones get a legend entry. Styled by `vline_kws`
            (`ax.axvline`).

        span_snap: bool, optional
            Widen each span to whole samples of `x` -- half-way to the
            neighbouring sample on each side (see
            `mindmap.metric._plotdata.snap_span`), so 5 samples around a
            peak are drawn 5 samples wide without working out the spacing.
            The drawn extents are in `Plot.data.attrs["spans"]`.

        Returns
        -------
        Plot
        """
        check_summary(center, interval, level)
        if center is None:
            raise ValueError("A line is drawn through its center -- pass center='mean' or 'median'.")
        if smooth is None and raw is not None:
            raise ValueError("`raw` draws the unsmoothed curve under a smoothed one -- it needs `smooth`.")
        if smooth is not None and (isinstance(smooth, (bool, np.bool_)) or not np.isscalar(smooth)
                                   or not np.isfinite(smooth) or smooth <= 0):
            raise ValueError(f"`smooth` must be a positive window width in x units, got {smooth!r}.")
        names = _namer(labels)
        df = collapse(self._metric_df(metric, filters), [x, hue], unit)
        numeric = pd.api.types.is_numeric_dtype(df[x])
        x_levels = levels(df[x], order) if order is not None else sorted(pd.unique(df[x].dropna()), key=None if numeric else str)
        hue_levels = levels(df[hue], hue_order) if hue is not None else [None]
        by = ([hue] if hue is not None else []) + [x]
        table = summary_table(df, by, center, interval, level)
        colors = _draw.resolve_colors(hue_levels, palette)
        xs = np.asarray(x_levels, dtype=float) if numeric else np.arange(len(x_levels), dtype=float)
        if smooth is not None and not numeric:
            raise ValueError(f"`smooth` is a window in x units -- it needs a numeric `x`, but {x!r} is categorical.")
        if (spans or vlines) and not numeric:
            raise ValueError(f"`spans`/`vlines` are in x units -- they need a numeric `x`, but {x!r} is categorical.")
        curve_kws, line_kws = _per_level_kws(line_kws, hue_levels, highlight, hue)
        span_specs = [_span_spec(sp) for sp in (spans or [])]
        if span_snap:
            span_specs = [(*snap_span(xs, a, b), label, kws) for a, b, label, kws in span_specs]
        vline_specs = [_vline_spec(v) for v in (vlines or [])]

        curves = {}
        for hv in hue_levels:
            sub = table if hue is None else table[table[hue] == hv]
            sub = sub.set_index(x).reindex(x_levels)
            curves[hv] = (
                xs, sub["center"].to_numpy(dtype=float), sub["low"].to_numpy(dtype=float),
                sub["high"].to_numpy(dtype=float), names(hv) if hue is not None else metric,
            )
        ax = _draw.get_ax(ax)
        span_artists, span_handles = _draw.spans(ax, span_specs, span_kws)
        raw_artists = None
        if smooth is not None:
            if raw is None or raw:
                raw_artists = _draw.raw_lines(ax, curves, colors, raw_kws)
            curves = {
                hv: (cx, *(moving_average(cx, v, smooth) for v in (mid, low, high)), label)
                for hv, (cx, mid, low, high, label) in curves.items()
            }
            smoothed = pd.concat([
                pd.DataFrame({**({hue: hv} if hue is not None else {}), x: x_levels, "center_smooth": cv[1],
                              "low_smooth": cv[2], "high_smooth": cv[3]})
                for hv, cv in curves.items()
            ], ignore_index=True)
            attrs = dict(table.attrs)
            table = table.merge(smoothed, on=by, how="left")
            table.attrs.update(attrs, smooth=smooth)
        artists = _draw.lines(ax, curves, colors, interval_style, fill_alpha, line_kws, interval_kws, curve_kws)
        vline_artists, vline_handles = _draw.vlines(ax, vline_specs, vline_kws)
        if raw_artists is not None:
            artists["raw"] = raw_artists
        if span_specs:
            artists["spans"] = span_artists
            table.attrs["spans"] = [(a, b, label) for a, b, label, _ in span_specs]
        if vline_specs:
            artists["vlines"] = vline_artists
        if not numeric:
            _draw.tick_labels(ax, xs, [names(v) for v in x_levels], rotate=True)
        marks = span_handles + vline_handles
        _draw.finish(
            ax, x if xlabel is None else xlabel, metric if ylabel is None else ylabel, title,
            legend=(hue is not None or bool(marks)) if legend is None else legend,
            handles=artists["lines"] + marks, legend_title=hue,
        )
        return Plot(ax, "line", data=table, points=df, artists=artists)

    def scatter(
        self, metric, x, hue=None, *, unit=None, filters=None, hue_order=None, labels=None, palette=None,
        xlabel=None, ylabel=None, title=None, legend=None, ax=None, points_kws=None,
    ):
        """
        Every row (or `unit`, averaged per `x`/`hue`) of `metric` against
        coordinate `x`, one color per `hue` level. Same arguments as `line`.
        """
        names = _namer(labels)
        df = collapse(self._metric_df(metric, filters), [x, hue], unit)
        hue_levels = levels(df[hue], hue_order) if hue is not None else [None]
        colors = _draw.resolve_colors(hue_levels, palette)
        numeric = pd.api.types.is_numeric_dtype(df[x])
        x_levels = None if numeric else sorted(pd.unique(df[x].dropna()), key=str)
        to_x = (lambda s: s.to_numpy(dtype=float)) if numeric else (lambda s: s.map({v: i for i, v in enumerate(x_levels)}).to_numpy(dtype=float))
        groups = {}
        for hv in hue_levels:
            sub = df if hue is None else df[df[hue] == hv]
            groups[hv] = (to_x(sub[x]), sub["Value"].to_numpy(dtype=float), names(hv) if hue is not None else metric)
        ax = _draw.get_ax(ax)
        artists = {"points": _draw.scatter(ax, groups, colors, points_kws)}
        if not numeric:
            _draw.tick_labels(ax, range(len(x_levels)), [names(v) for v in x_levels], rotate=True)
        _draw.finish(
            ax, x if xlabel is None else xlabel, metric if ylabel is None else ylabel, title,
            legend=(hue is not None) if legend is None else legend, handles=artists["points"], legend_title=hue,
        )
        return Plot(ax, "scatter", data=None, points=df, artists=artists)

    def _value_groups(self, df, hue, hue_order, names, metric):
        hue_levels = levels(df[hue], hue_order) if hue is not None else [None]
        return hue_levels, {
            hv: ((df if hue is None else df[df[hue] == hv])["Value"].to_numpy(dtype=float), names(hv) if hue is not None else metric)
            for hv in hue_levels
        }

    def distribution(
        self, metric, hue=None, *, unit=None, filters=None, hue_order=None, labels=None, palette=None,
        fill_alpha=None, xlabel=None, ylabel=None, title=None, legend=None, ax=None, line_kws=None, fill_kws=None,
    ):
        """
        Kernel-density curve of `metric` (on the x-axis), one filled curve
        per `hue` level, each normalized on its own so shapes compare
        regardless of group size. Without `hue`, one curve for the whole
        population. See the class docstring for the common arguments.
        """
        names = _namer(labels)
        df = collapse(self._metric_df(metric, filters), [hue], unit)
        hue_levels, groups = self._value_groups(df, hue, hue_order, names, metric)
        ax = _draw.get_ax(ax)
        artists = _draw.kde(ax, groups, _draw.resolve_colors(hue_levels, palette), fill_alpha, line_kws, fill_kws)
        _draw.finish(
            ax, metric if xlabel is None else xlabel, "density" if ylabel is None else ylabel, title,
            legend=(hue is not None) if legend is None else legend, handles=artists["lines"], legend_title=hue,
        )
        return Plot(ax, "distribution", data=None, points=df, artists=artists)

    def histogram(
        self, metric, hue=None, *, unit=None, filters=None, hue_order=None, labels=None, palette=None,
        fill_alpha=None, xlabel=None, ylabel=None, title=None, legend=None, ax=None, hist_kws=None,
    ):
        """
        Histogram of `metric`, one translucent layer per `hue` level on
        shared bins (`hist_kws={"bins": 30}` to change them). Same
        arguments as `distribution`.
        """
        names = _namer(labels)
        df = collapse(self._metric_df(metric, filters), [hue], unit)
        hue_levels, groups = self._value_groups(df, hue, hue_order, names, metric)
        ax = _draw.get_ax(ax)
        hists = _draw.histogram(ax, groups, _draw.resolve_colors(hue_levels, palette), fill_alpha, hist_kws)
        artists = {"bars": hists}
        handles = [h[2][0] for h in hists if len(h[2])]
        _draw.finish(
            ax, metric if xlabel is None else xlabel, "count" if ylabel is None else ylabel, title,
            legend=(hue is not None) if legend is None else legend, handles=handles, legend_title=hue,
        )
        return Plot(ax, "histogram", data=None, points=df, artists=artists)

    # -- Stat views: a computed result in, never a metric name ---------------

    def group_comparison(
        self, result, kind=_AUTO, reference=None, alpha=0.05, *, higher_is_better=True, center=_AUTO,
        interval=_AUTO, level=0.95, points=_AUTO, annotate=("stars",), effect_size=None, show_ns=False, order=None,
        labels=None, palette=None, fill_alpha=None, chance=None, chance_value=None, reference_lines=None,
        xlabel=None, ylabel=None, title=None, seed=0, ax=None, box_kws=None, violin_kws=None, bar_kws=None,
        points_kws=None, center_kws=None, interval_kws=None, bracket_kws=None, text_kws=None, rotate_ticks="auto",
    ):
        """
        Draws a `PairwiseResult` (from `suite.stat.wilcoxon`/`.ttest`/
        `.mannwhitneyu`/`.paired_ttest`/`.ttest_ess`) as a categorical plot
        -- one element per group -- with a bracket from `reference` to each
        competitor it significantly differs from.

        !!! note "Technical details"
            The points are the units the test compared (collapsed with its
            `pairing_col`/`unit_col`, filtered with its filters), so the
            plot shows exactly what was tested. `reference` is inferred
            when every pair shares one group. `alpha` only picks which
            pairs get a bracket (0.05/0.01/0.001 star tiers on
            `p_corrected`, or `p_value` on an uncorrected result); it is
            independent of the `alpha`/`correction` the result was
            computed with. Bracket text is `annotate`'s keys joined on one
            line: `"stars"`, `"p"` (`p=0.003`), `"effect"` (the
            `effect_size`, default the result's first), any of the
            result's effect-size names, `"n_eff"` -- effect sizes oriented
            so positive means `reference` better (see `higher_is_better`).

        Parameters
        ----------
        result: PairwiseResult
            An already-computed result.

        kind: str, optional
            "box", "violin", "bar", "point" or "strip". Default `"auto"`:
            "box", or "bar" for McNemar (a 0/1 outcome).

        center, interval: str, optional
            Default `"auto"`: the center that matches the test -- the
            median for Wilcoxon/Mann-Whitney, the mean for the t-tests,
            the proportion for McNemar -- with its confidence interval
            (exact binomial, t, Wilson), the one `suite.estimate` reports.
            None on a box, which already shows median and IQR. Any of the
            common values overrides it.

        reference: str, optional
            The group every bracket starts from.

        alpha: float, optional
            Significance threshold for drawing a bracket (default 0.05).

        higher_is_better: bool, optional
            Orients the effect sizes in the bracket text (default True).

        annotate: tuple[str, ...], optional
            Bracket text, see technical details. Default `("stars",)`.

        show_ns: bool, optional
            Also bracket non-significant pairs (labelled "ns").

        chance, chance_value, reference_lines: optional
            Background guides: a gray `(lo, hi)` band, a dotted line, and
            labelled dashed lines `{label: value | (value, color)}`.

        Returns
        -------
        Plot
            `data` gains the bracket text per competitor.
        """
        if not isinstance(result, PairwiseResult):
            raise TypeError(
                f"suite.plot.group_comparison() expects a PairwiseResult (from suite.stat.wilcoxon(...)/.ttest(...)/"
                f".mannwhitneyu(...)/.paired_ttest(...)/.ttest_ess(...)), got {type(result).__name__}. "
                f"For a ClusterResult, use suite.plot.cluster_permutation() instead."
            )
        annotate = check_annotate(annotate, result, allow_differences=False)
        effect_size = check_effect_size(effect_size, result)
        reference = _require_reference(reference, list(zip(result.table["group_a"], result.table["group_b"])))
        if kind == _AUTO:
            kind = "bar" if result.estimator in BINARY_EFFECT_SIZES else "box"
        center, interval, points = _stat_defaults(kind, result, center, interval, points)

        df = collapse_for_result(self._result_df(result), result.group_by, result.pairing_col, result.unit_col)
        if order is None:
            order = list(pd.unique(df[result.group_by]))
        if reference not in order:
            raise ValueError(f"reference {reference!r} not found in `order`/the result's data.")

        ax = _draw.get_ax(ax)
        _draw.reference_lines(ax, chance, chance_value, reference_lines)
        plot, positions, bottom, top = self._categorical(
            kind, df, result.group_by, None, order, None, center, interval, level, points, labels, palette,
            fill_alpha, seed, ax,
            dict(box_kws=box_kws, violin_kws=violin_kws, bar_kws=bar_kws, points_kws=points_kws,
                 center_kws=center_kws, interval_kws=interval_kws),
        )

        summary = result.vs(reference, higher_is_better=higher_is_better)
        specs, texts = [], {}
        for competitor, row in summary.iterrows():
            if competitor not in order:
                continue
            p = corrected_p(row)
            significant = pd.notna(p) and p < alpha
            text = "  ".join(annotation_lines(annotate, row, effect_size, result.effect_sizes))
            texts[competitor] = text
            if significant or show_ns:
                specs.append((positions[(reference, None)], positions[(competitor, None)], text))
        plot.artists["brackets"] = _draw.brackets(ax, specs, top, bottom, bracket_kws, text_kws)
        plot.data["annotation"] = plot.data[result.group_by].map(texts)
        plot.kind = "group_comparison"
        _draw.finish(ax, result.group_by if xlabel is None else xlabel, result.metric if ylabel is None else ylabel, title)
        _draw.rotate_ticks(ax, rotate_ticks)
        return plot

    def paired_differences(
        self, result, reference=None, higher_is_better=None, *, distribution=_AUTO, kind=_AUTO, center=_AUTO, interval=_AUTO,
        level=0.95, points=_AUTO, annotate=("stars", "effect"), effect_size=None, violin_min_n=30, order=None,
        interval_style="band", show_offset=True, labels=None, palette=None, fill_alpha=None, direction_labels=None,
        xlabel=None, ylabel=None, title=None, legend=None, seed=0, ax=None, violin_kws=None, box_kws=None,
        points_kws=None, center_kws=None, interval_kws=None, line_kws=None, cluster_kws=None, text_kws=None,
        offset_kws=None, rotate_ticks="auto",
    ):
        """
        Draws the per-unit paired differences behind a paired test,
        `reference` minus each competitor -- what the test actually
        compared. The shape follows the result:

        - `PairwiseResult` (`wilcoxon`/`mcnemar`/`paired_ttest`/
          `ttest_ess`/`bootstrap`): one column per competitor, the units
          as points or a violin -- for a bootstrap test, its draws as a
          half-violin (see `distribution`) -- annotated with each test's
          numbers;
        - `ClusterResult`: one difference curve per competitor along the
          coordinate, with its band, and a bar under the curves marking
          each significant cluster.

        !!! note "Technical details"
            The differences are `suite.stat.paired_differences(result,
            reference, higher_is_better)` -- the exact units each test saw
            -- so positive always means `reference` better.

            `center`/`interval` default to `"auto"`, the summary that
            matches the test. Pairwise: the test's own estimate and its
            confidence interval, as `suite.estimate.from_result` reports
            them -- Hodges-Lehmann for Wilcoxon, the mean difference for
            the t-tests (with the effective-sample-size interval for
            `ttest_ess`), the difference in proportions with Newcombe's
            interval for McNemar. Cluster: the mean difference with its t
            interval at each coordinate (the cluster test is t-based).
            Give `center="median"`, `interval="iqr"` etc. to summarize the
            drawn differences instead (`center="auto"` only combines with
            `interval="auto"`, `"iqr"` or None).

            Pairwise annotation keys, stacked top to bottom in the order
            given:

            - `"stars"`: `***`/`**`/`*`/`ns` from `p_corrected` (0.001/
              0.01/0.05; `p_value` on an uncorrected result);
            - `"p"`: `p=0.003`;
            - `"effect"`: the `effect_size` (default: the result's first),
              e.g. `dz=+0.56` or `r=+0.42`;
            - any effect size the result reports, by name -- e.g.
              `"hodges_lehmann"` (`HL=+0.042`) or `"mean_diff"`;
            - `"winrate"`: share of units where `reference` is better,
              minus 50% (e.g. `+26%` = better at 76% of units);
            - `"counts"`: better/tied/worse unit counts, e.g. `61/0/39`;
            - `"n_eff"`: `n_eff=35`, only for `ttest_ess` results.

            `distribution="bootstrap"` (the default for a
            `suite.stat.bootstrap` result with `mean_diff` or
            `hodges_lehmann`) draws, per competitor, the bootstrap
            distribution of the estimate -- `suite.stat.bootstrap_draws`,
            the very draws behind the test's p-value and interval, so the
            interval drawn over it is their percentile interval. It is the
            sampling distribution of the estimate, not the spread of the
            data: with one paired unit per competitor (e.g. one spike
            average per subject), it is the only distribution there is.
            `violin_min_n` then counts draws, and the win rate and counts
            still describe the units.

            Effect sizes, win rate and counts describe the raw differences
            (relative to 0); the offset only changes the test's p-value.
            A test with a non-zero offset gets a dashed line across each
            column at the offset, oriented like the differences (e.g. -2
            for "wCSET is not worse by 2 mm or more" on an error), with a
            legend entry -- so the estimate and its interval can be read
            against it.

            Cluster: the band is a pointwise interval at each coordinate,
            not a simultaneous one -- it describes the curve, while the
            cluster bars carry the inference. A cluster's bar is drawn
            when it is `significant` under the result's correction (its
            own `p_value` < `cluster_alpha` on an uncorrected result).
            Cluster tests don't support claims about where exactly an
            effect starts or ends (Sassenhagen & Draschkow, 2019).

        Parameters
        ----------
        result: PairwiseResult | ClusterResult
            A paired result.

        reference: str, optional
            The group every difference is taken from. Inferred when every
            pair shares one group.

        higher_is_better: bool, optional
            Whether a larger metric value is better; when False,
            differences and effects are negated. Default: True for a
            pairwise result, the test's own setting for a cluster result.

        distribution: str, optional
            Pairwise only: what each column shows -- "units" (the per-unit
            differences), "bootstrap" (a `suite.stat.bootstrap` result's
            draws; `mean_diff`/`hodges_lehmann` only), or "auto"
            (default): "bootstrap" for such a result, "units" otherwise.

        kind: str, optional
            Pairwise only: "violin", "strip", "box", "point", or "auto"
            (default): a violin for each column with at least
            `violin_min_n` units (or draws), points for the others; for
            McNemar (0/1 differences), "point". Bootstrap violins are
            half-violins -- `violin_kws={"side": "both"}` for full ones.

        palette: dict | list | str, optional
            Colors per competitor: the fill, the center and the interval.
            `center_kws`/`interval_kws` override them, e.g.
            `interval_kws={"color": "k"}` for black intervals.

        center, interval: str, optional
            See technical details. Default `"auto"`.

        annotate: tuple[str, ...], optional
            Pairwise only: text lines above each column, see technical
            details.

        interval_style: str, optional
            Cluster only: the interval as a shaded "band" (default) or
            error "bars".

        show_offset: bool, optional
            Pairwise only: draw the test's offset line (default True; no
            line when the offset is 0). Rename its legend entry with
            `labels={"offset": "δ (margin)"}`, restyle it with
            `offset_kws` (`ax.plot`), and place or hide the legend with
            `legend`.

        direction_labels: tuple[str, str], optional
            `(positive, negative)` labels right of the axes at y = 0, e.g.
            `("wCSET better", "other method better")`.

        line_kws, cluster_kws: dict, optional
            Cluster only: the curves (`ax.plot`) and the cluster bars
            (`ax.hlines`).

        offset_kws: dict, optional
            Pairwise only: the offset lines (`ax.plot`).

        Returns
        -------
        Plot
        """
        if isinstance(result, ClusterResult):
            return self._difference_curves(
                result, reference, higher_is_better, center, interval, level, interval_style, labels, palette,
                fill_alpha, direction_labels, xlabel, ylabel, title, legend, ax, line_kws, interval_kws, cluster_kws,
            )
        if not isinstance(result, PairwiseResult):
            raise TypeError(
                f"suite.plot.paired_differences() expects a PairwiseResult (from suite.stat.wilcoxon(...)/"
                f".mcnemar(...)/.paired_ttest(...)/.ttest_ess(...)) or a ClusterResult, got {type(result).__name__}."
            )
        higher_is_better = True if higher_is_better is None else higher_is_better
        annotate = check_annotate(annotate, result, allow_differences=True)
        effect_size = check_effect_size(effect_size, result)
        reference = _require_reference(reference, list(zip(result.table["group_a"], result.table["group_b"])))
        if center == _AUTO and interval not in (_AUTO, "iqr", None):
            raise ValueError(
                f"center='auto' (the test's estimate) doesn't combine with interval={interval!r} -- "
                "pass center='mean' or 'median' as well."
            )

        bootstrapped = result.test == "bootstrap" and result.estimator in _UNIT_ESTIMATORS
        if distribution == _AUTO:
            distribution = "bootstrap" if bootstrapped else "units"
        if distribution not in ("units", "bootstrap"):
            raise ValueError(f"distribution={distribution!r} isn't supported. Choose 'units', 'bootstrap' or 'auto'.")
        if distribution == "bootstrap" and not bootstrapped:
            raise ValueError(
                "distribution='bootstrap' needs a suite.stat.bootstrap(...) result with estimator 'mean_diff' or "
                f"'hodges_lehmann' (got test={result.test!r}, estimator={result.estimator!r})."
            )

        summary = result.vs(reference, higher_is_better=higher_is_better)
        diffs = self._suite.stat.paired_differences(result, reference=reference, higher_is_better=higher_is_better)
        if order is None:
            order = list(summary.index)
        missing = [c for c in order if c not in summary.index]
        if missing:
            raise ValueError(f"`order` entries {missing} were not compared against reference {reference!r} in `result`.")
        if distribution == "bootstrap":
            long = self._suite.stat.bootstrap_draws(result, reference=reference, higher_is_better=higher_is_better)
            long = long[long[result.group_by].isin(order)].reset_index(drop=True)
            violin_kws = {"side": "high", **(violin_kws or {})}
        else:
            long = (
                diffs[order].melt(var_name=result.group_by, value_name="Value", ignore_index=False)
                .dropna(subset=["Value"]).reset_index()
            )
        empty = [c for c in order if not (long[result.group_by] == c).any()]
        if empty:
            raise ValueError(f"No differences to draw for {empty}.")
        cell_kinds = None
        if kind == _AUTO and result.estimator in BINARY_EFFECT_SIZES:
            kind = "point"
        elif kind == _AUTO:
            sizes = long.groupby(result.group_by).size()
            cell_kinds = {c: "violin" if sizes[c] >= violin_min_n else "strip" for c in order}
            kind = "violin" if "violin" in cell_kinds.values() else "strip"
        if kind not in _KIND_POINTS:
            raise ValueError(f"kind={kind!r} isn't supported. Choose from {list(_KIND_POINTS)}.")
        if points == _AUTO:
            # Thousands of draws over a box would only bury it.
            points = _KIND_POINTS[kind] and not (distribution == "bootstrap" and kind == "box")

        overrides, estimator = None, None
        if _AUTO in (center, interval) and result.estimator not in _UNIT_ESTIMATORS:
            # A standardized bootstrapped estimator isn't in the differences' units -- nothing to overlay.
            center, interval = (None if center == _AUTO else center), (None if interval == _AUTO else interval)
        if _AUTO in (center, interval):
            estimator = result.estimator
            if _has_ci(result, estimator, level):
                # The test already holds its estimate's interval at this level -- the same numbers from_result gives.
                est = summary[[estimator, f"{estimator}_ci_low", f"{estimator}_ci_high"]].set_axis(
                    ["estimate", "ci_low", "ci_high"], axis=1)
            else:
                est = self._suite.estimate.from_result(result, alpha=1 - level).vs(reference, higher_is_better)
            overrides = {
                c: {**({"center": est.loc[c, "estimate"]} if center == _AUTO else {}),
                    **({"low": est.loc[c, "ci_low"], "high": est.loc[c, "ci_high"]} if interval == _AUTO else {})}
                for c in order
            }
        ax = _draw.get_ax(ax)
        ax.axhline(0, color=_draw.LINE_COLOR, linewidth=1, zorder=0)
        plot, positions, bottom, top = self._categorical(
            kind, long, result.group_by, None, order, None, None if center == _AUTO else center,
            None if interval == _AUTO else interval, level, points, labels, palette, fill_alpha, seed, ax,
            dict(violin_kws=violin_kws, box_kws=box_kws, points_kws=points_kws, center_kws=center_kws,
                 interval_kws=interval_kws, colored_summary=True),
            cell_kinds=cell_kinds, overrides=overrides,
        )
        plot.data.attrs["distribution"] = distribution
        if center == _AUTO:
            plot.data.attrs["center"] = estimator
        if interval == _AUTO:
            plot.data.attrs.update(interval="ci", level=level)

        lines_by_pos, offsets = {}, []
        offset_style = _draw.merge({"color": _draw.LINE_COLOR, "linestyle": "--", "linewidth": 1.2, "zorder": 0.5}, offset_kws)
        plot.artists["offset"] = []
        for c in order:
            row = summary.loc[c]
            d = diffs[c].dropna().to_numpy(dtype=float)  # the units, whatever the columns show
            lines_by_pos[positions[(c, None)]] = annotation_lines(annotate, row, effect_size, result.effect_sizes, d)
            offset = float(row["offset"])
            if offset and show_offset:
                pos = positions[(c, None)]
                plot.artists["offset"].extend(ax.plot([pos - 0.4, pos + 0.4], [offset, offset], **offset_style))
                offsets.append(offset)
        bottom, top = min([bottom] + offsets), max([top] + offsets)
        plot.artists["text"] = _draw.column_text(ax, lines_by_pos, top, bottom, text_kws)
        plot.data["annotation"] = [" | ".join(lines_by_pos[p]) for p in plot.data["position"]]
        plot.kind = "paired_differences"
        ax.spines[["top", "right"]].set_visible(False)
        if direction_labels is not None:
            _draw.direction_labels(ax, direction_labels)
        default_ylabel = f"{result.metric}: {reference} − other" + ("" if higher_is_better else " (negated)")
        handles = None
        if offsets:
            values = set(offsets)
            default = f"offset ({offsets[0]:+g})" if len(values) == 1 else "offset"
            label = (labels or {}).get("offset", default)
            handles = [Line2D([0], [0], **{k: v for k, v in offset_style.items() if k != "zorder"}, label=label)]
        _draw.finish(
            ax, "" if xlabel is None else xlabel, default_ylabel if ylabel is None else ylabel, title,
            legend=True if legend is None else legend, handles=handles,
        )
        _draw.rotate_ticks(ax, rotate_ticks)
        return plot

    def _difference_curves(
        self, result, reference, higher_is_better, center, interval, level, interval_style, labels, palette,
        fill_alpha, direction_labels, xlabel, ylabel, title, legend, ax, line_kws, interval_kws, cluster_kws,
    ):
        """`paired_differences` for a `ClusterResult` -- see there."""
        center = "mean" if center == _AUTO else center
        interval = "ci" if interval == _AUTO else interval
        check_summary(center, interval, level)
        if center is None:
            raise ValueError("A curve is drawn through its center -- pass center='mean' or 'median'.")
        names = _namer(labels)
        reference = _require_reference(reference, list(result.clusters_by_pair))
        hib = result.higher_is_better if higher_is_better is None else higher_is_better
        long = self._suite.stat.paired_differences(result, reference=reference, higher_is_better=hib)
        coord, group = result.coord, result.group_by
        competitors = list(dict.fromkeys(long[group]))
        coord_order = list(result.coord_order)
        numeric = pd.api.types.is_numeric_dtype(pd.Series(coord_order))
        xs = np.asarray(coord_order, dtype=float) if numeric else np.arange(len(coord_order), dtype=float)

        table = summary_table(long, [group, coord], center, interval, level)
        colors = _draw.resolve_colors(competitors, palette)
        curves = {}
        for c in competitors:
            sub = table[table[group] == c].set_index(coord).reindex(coord_order)
            curves[c] = (xs, sub["center"].to_numpy(dtype=float), sub["low"].to_numpy(dtype=float),
                         sub["high"].to_numpy(dtype=float), names(c))

        ax = _draw.get_ax(ax)
        ax.axhline(0, color=_draw.LINE_COLOR, linewidth=1, zorder=0)
        artists = _draw.lines(ax, curves, colors, interval_style, fill_alpha, line_kws, interval_kws)

        # One bar per significant cluster, a row per competitor under the curves, half a step wide past each end.
        edges = np.concatenate([[xs[0] - (xs[1] - xs[0]) / 2 if len(xs) > 1 else xs[0] - 0.5],
                                (xs[:-1] + xs[1:]) / 2,
                                [xs[-1] + (xs[-1] - xs[-2]) / 2 if len(xs) > 1 else xs[-1] + 0.5]])
        spans = {}
        for (a, b), clusters in result.clusters_by_pair.items():
            if reference not in (a, b) or clusters.empty:
                continue
            significant = clusters["significant"].astype(bool) if result.correction is not None else clusters["p_value"] < result.cluster_alpha
            spans[b if reference == a else a] = [
                (edges[coord_order.index(r[f"{coord}_start"])], edges[coord_order.index(r[f"{coord}_end"]) + 1])
                for _, r in clusters[significant].iterrows()
            ]
        drawn = np.concatenate([np.concatenate([cv[1], cv[2], cv[3]]) for cv in curves.values()] + [[0.0]])
        drawn = drawn[np.isfinite(drawn)]
        low, high = float(drawn.min()), float(drawn.max())
        gap = 0.06 * ((high - low) or 1.0)
        rows = {c: low - gap * (i + 1) for i, c in enumerate(competitors)}
        artists["clusters"] = _draw.cluster_bars(ax, spans, rows, colors, cluster_kws)
        ax.set_ylim(low - gap * (len(competitors) + 1), None)

        if not numeric:
            _draw.tick_labels(ax, xs, [names(v) for v in coord_order], rotate=True)
        ax.spines[["top", "right"]].set_visible(False)
        if direction_labels is not None:
            _draw.direction_labels(ax, direction_labels)
        handles = artists["lines"] + _draw.cluster_legend(cluster_kws)
        default_ylabel = f"{result.metric}: {reference} − other" + ("" if hib else " (negated)")
        _draw.finish(
            ax, coord if xlabel is None else xlabel, default_ylabel if ylabel is None else ylabel, title,
            legend=True if legend is None else legend, handles=handles,
        )
        data = table.rename(columns={group: group})
        data.attrs.update(table.attrs)
        return Plot(ax, "paired_differences", data=data, points=long, artists=artists)

    def cluster_permutation(
        self, result, reference=None, alpha=0.05, *, center=_AUTO, interval=_AUTO, level=0.95,
        interval_style="band", labels=None, palette=None, fill_alpha=None, cluster_colors=None, xlabel=None,
        ylabel=None, title=None, legend=None, ax=None, line_kws=None, center_kws=None, interval_kws=None, span_kws=None,
    ):
        """
        Draws a `ClusterResult` (from `suite.stat.cluster_permutation`) as
        curves along its coordinate.

        !!! note "Technical details"
            The curves are `center`/`interval` (default `"auto"`: the mean
            and its t interval, matching the t-based cluster test) across
            the units the test compared (collapsed per
            `pairing_col`, and `unit_col` first, at each coordinate).
            The style switches on the number of pairs:

            - one pair: both curves with `interval` as a band
              (`interval_style="bars"` for error bars), and each
              significant cluster shaded (green where the first group is
              higher, red otherwise; `cluster_colors={"positive": ...,
              "negative": ...}`), at the result's own `cluster_alpha`;
            - several pairs: every curve on one axes against `reference`,
              each competitor solid with filled markers where `reference`
              significantly beats it (the cluster's `p_corrected` < `alpha`,
              under the result's correction family) and
              dashed with open markers elsewhere, `interval` as thin
              whiskers. `center_kws` styles the markers.

        Parameters
        ----------
        result: ClusterResult
            An already-computed result.

        reference: str, optional
            The baseline of the several-pairs view; inferred when every
            pair shares one group.

        alpha: float, optional
            Significance threshold of the several-pairs view (default 0.05).

        Returns
        -------
        Plot
        """
        if not isinstance(result, ClusterResult):
            raise TypeError(
                f"suite.plot.cluster_permutation() expects a ClusterResult (from suite.stat.cluster_permutation(...)), "
                f"got {type(result).__name__}. For a PairwiseResult, use suite.plot.group_comparison() instead."
            )
        center = "mean" if center == _AUTO else center
        interval = "ci" if interval == _AUTO else interval
        check_summary(center, interval, level)
        if center is None:
            raise ValueError("The curves are drawn through their center -- pass center='mean' or 'median'.")
        names = _namer(labels)
        pairs = list(result.clusters_by_pair.keys())
        involved = list(dict.fromkeys(g for pair in pairs for g in pair))
        df = self._result_df(result, {result.group_by: involved})
        df = _collapse_with_coord(df, result.group_by, result.pairing_col, result.coord, result.unit_col)
        coord_order = list(result.coord_order)
        numeric = pd.api.types.is_numeric_dtype(pd.Series(coord_order))
        xs = np.asarray(coord_order, dtype=float) if numeric else np.arange(len(coord_order), dtype=float)
        to_x = (lambda v: float(v)) if numeric else (lambda v: float(coord_order.index(v)))

        if len(pairs) > 1:
            reference = _require_reference(reference, pairs)
            competitors = {}
            for a, b in pairs:
                if reference not in (a, b):
                    raise ValueError(
                        "The several-pairs plot requires every pair to include `reference` "
                        f"({reference!r}) -- got ({a!r}, {b!r})."
                    )
                competitors[b if a == reference else a] = result.clusters_by_pair[(a, b)]
            groups = [reference] + list(competitors)
        else:
            groups = list(pairs[0])

        table = summary_table(df, [result.group_by, result.coord], center, interval, level)
        curves = {}
        for g in groups:
            sub = table[table[result.group_by] == g].set_index(result.coord).reindex(coord_order)
            curves[g] = (xs, sub["center"].to_numpy(dtype=float), sub["low"].to_numpy(dtype=float), sub["high"].to_numpy(dtype=float))
        colors = _draw.resolve_colors(groups, palette)
        ax = _draw.get_ax(ax)

        if len(pairs) == 1:
            line_curves = {g: (*curves[g], names(g)) for g in groups}
            artists = _draw.lines(ax, line_curves, colors, interval_style, fill_alpha, line_kws, interval_kws)
            clusters = result.clusters_by_pair[pairs[0]].copy()
            for end in ("start", "end"):
                clusters[f"{result.coord}_{end}"] = clusters[f"{result.coord}_{end}"].map(to_x)
            artists["spans"] = _draw.cluster_spans(ax, clusters, result.coord, "p_value", result.cluster_alpha, cluster_colors, span_kws)
            handles = artists["lines"]
        else:
            sig_segments = {}
            n = len(coord_order)
            for name, clusters in competitors.items():
                sig = clusters[(clusters["p_corrected"] < alpha) & (clusters["sign"] > 0)] if not clusters.empty else clusters
                ranges = [
                    tuple(sorted((coord_order.index(r[f"{result.coord}_start"]), coord_order.index(r[f"{result.coord}_end"]))))
                    for _, r in sig.iterrows()
                ]
                sig_segments[name] = [any(lo <= i and i + 1 <= hi for lo, hi in ranges) for i in range(n - 1)]
            artists, handles = _draw.cluster_summary(
                ax, curves, reference, sig_segments, colors, alpha, names, line_kws, center_kws, interval_kws,
            )
        if not numeric:
            _draw.tick_labels(ax, xs, [names(v) for v in coord_order], rotate=True)
        _draw.finish(
            ax, result.coord if xlabel is None else xlabel, result.metric if ylabel is None else ylabel, title,
            legend=True if legend is None else legend, handles=handles,
        )
        return Plot(ax, "cluster_permutation", data=table, points=df, artists=artists)

    def normality_test(
        self, result, kind="distribution", *, labels=None, palette=None, fill_alpha=None, xlabel=None, ylabel=None,
        title=None, legend=None, ax=None, line_kws=None, fill_kws=None, hist_kws=None, text_kws=None,
    ):
        """
        Draws a `NormalityResult` (from `suite.stat.normality_test`) as a
        density (`kind="distribution"`, default) or histogram
        (`kind="histogram"`) per group, annotated with each group's p-value
        and verdict -- red when significantly non-normal. The result's own
        `significant`/`p_corrected` are used; there is no plot-time `alpha`.

        Returns
        -------
        Plot
        """
        if not isinstance(result, NormalityResult):
            raise TypeError(
                f"suite.plot.normality_test() expects a NormalityResult (from suite.stat.normality_test(...)), "
                f"got {type(result).__name__}."
            )
        if kind not in ("distribution", "histogram"):
            raise ValueError(f"Kind '{kind}' not supported. Choose from ['distribution', 'histogram'].")
        names = _namer(labels)
        df = self._result_df(result)
        hue = result.group_by
        hue_levels, groups = self._value_groups(df, hue, None, names, result.metric)
        colors = _draw.resolve_colors(hue_levels, palette)
        ax = _draw.get_ax(ax)
        if kind == "distribution":
            artists = _draw.kde(ax, groups, colors, fill_alpha, line_kws, fill_kws)
            handles = artists["lines"]
        else:
            hists = _draw.histogram(ax, groups, colors, fill_alpha, hist_kws)
            artists, handles = {"bars": hists}, [h[2][0] for h in hists if len(h[2])]

        artists["text"] = []
        for i, (_, row) in enumerate(result.table.iterrows()):
            label = "whole population" if row["group"] is None else names(row["group"])
            p = row["p_corrected"] if pd.notna(row["p_corrected"]) else row["p_value"]
            verdict = "non-normal" if row["significant"] else "normal"
            style = {"transform": ax.transAxes, "ha": "right", "va": "top", "fontsize": _draw.text_size(9),
                     "color": "tab:red" if row["significant"] else "#333333"}
            artists["text"].append(ax.text(0.98, 0.98 - 0.06 * i, f"{label}: p={p:.3g} ({verdict})", **_draw.merge(style, text_kws)))
        _draw.finish(
            ax, result.metric if xlabel is None else xlabel,
            ("density" if kind == "distribution" else "count") if ylabel is None else ylabel, title,
            legend=(hue is not None) if legend is None else legend, handles=handles, legend_title=hue,
        )
        return Plot(ax, "normality_test", data=result.table, points=df, artists=artists)

    def anova(
        self, result, kind=_AUTO, *, center=_AUTO, interval=_AUTO, level=0.95, points=_AUTO, order=None,
        labels=None, palette=None, fill_alpha=None, xlabel=None, ylabel=None, title=None, seed=0, ax=None,
        box_kws=None, violin_kws=None, bar_kws=None, points_kws=None, center_kws=None, interval_kws=None, text_kws=None,
    ):
        """
        Draws an `AnovaResult` (from `suite.stat.anova`) as a categorical
        plot of every included group (`kind`, `center`, `interval` as in
        `group_comparison`: by default a box, or the median for Friedman /
        the mean for the ANOVA with its CI on any other kind), annotated
        with the omnibus verdict (test, statistic, p-value,
        effect size).

        !!! note "Technical details"
            No per-pair brackets -- an omnibus test never compared pairs,
            only whether the groups differ at all; follow up with
            `suite.plot.group_comparison`. Points are collapsed exactly as
            `AnovaResult.compute` collapsed them before testing.

        Returns
        -------
        Plot
        """
        if not isinstance(result, AnovaResult):
            raise TypeError(
                f"suite.plot.anova() expects an AnovaResult (from suite.stat.anova(...)), "
                f"got {type(result).__name__}."
            )
        kind = "box" if kind == _AUTO else kind
        center, interval, points = _stat_defaults(kind, result, center, interval, points)
        df = collapse_for_result(
            self._result_df(result, {result.group_by: list(result.groups)}), result.group_by, result.pairing_col, result.unit_col,
        )
        plot, _, _, _ = self._categorical(
            kind, df, result.group_by, None, order, None, center, interval, level, points, labels, palette,
            fill_alpha, seed, ax,
            dict(box_kws=box_kws, violin_kws=violin_kws, bar_kws=bar_kws, points_kws=points_kws,
                 center_kws=center_kws, interval_kws=interval_kws),
        )
        ax = plot.ax
        test_labels = {"friedman": "Friedman", "anova": "Repeated-measures ANOVA"}
        stat_symbols = {"friedman": "χ²", "anova": "F"}
        verdict = "significant" if result.significant else "not significant"
        text = (
            f"{test_labels.get(result.test, result.test)}: {stat_symbols.get(result.test, 'stat')}={result.statistic:.3g}, "
            f"p={result.p_value:.3g} ({verdict})\n{result.effect_size_type}={result.effect_size:.3g}"
        )
        style = {"transform": ax.transAxes, "ha": "right", "va": "top", "fontsize": _draw.text_size(9),
                 "color": "tab:red" if result.significant else "#333333"}
        plot.artists["text"] = [ax.text(0.98, 0.98, text, **_draw.merge(style, text_kws))]
        plot.kind = "anova"
        _draw.finish(ax, result.group_by if xlabel is None else xlabel, result.metric if ylabel is None else ylabel, title)
        return plot

    def estimate(
        self, result, reference=None, higher_is_better=True, *, order=None, labels=None, palette=None,
        reference_lines=None, xlabel=None, ylabel=None, title=None, ax=None, center_kws=None, interval_kws=None,
    ):
        """
        Draws an `EstimateResult` (from `suite.estimate.<name>`) as points
        with their confidence intervals -- per group, or per comparison for
        a difference estimate (with a line at no difference: 0, or 1 for
        an odds ratio, drawn on a log scale, and 0.5 for `cles`).

        !!! note "Technical details"
            Nothing is recomputed: the points and intervals are the
            result's table. For a difference, give `reference` to orient
            every comparison as `reference` relative to the competitor
            (`result.vs(reference, higher_is_better)`, positive =
            `reference` better); otherwise each pair is drawn as stored,
            labelled `group_a − group_b`. Add guides such as an offset with
            `reference_lines={"offset": 0.05}`.

        Returns
        -------
        Plot
        """
        if not isinstance(result, EstimateResult):
            raise TypeError(
                f"suite.plot.estimate() expects an EstimateResult (from suite.estimate.<name>(...)), "
                f"got {type(result).__name__}."
            )
        names = _namer(labels)
        if not result.is_difference:
            table = result.table.copy()
            keys = [g if g is not None else "all" for g in table["group"]]
        elif reference is not None:
            table = result.vs(reference, higher_is_better=higher_is_better).reset_index()
            keys = list(table[result.group_by])
        else:
            table = result.table.copy()
            keys = [f"{a} − {b}" for a, b in zip(table["group_a"], table["group_b"])]
        table["key"] = keys
        table = table.rename(columns={"estimate": "center", "ci_low": "low", "ci_high": "high"})
        order = levels(table["key"], order)
        table = table.set_index("key").loc[order].reset_index()
        colors = _draw.resolve_colors(order, palette)
        cells = [
            {"pos": i, "width": 0.6, "color_key": k, "values": [], "center": r["center"], "low": r["low"], "high": r["high"], "box": None}
            for i, (k, r) in enumerate(zip(order, table.to_dict("records")))
        ]
        ax = _draw.get_ax(ax)
        if result.is_difference:
            ax.axhline(NULL_VALUES.get(result.estimator, 0.0), color=_draw.LINE_COLOR, linewidth=1, zorder=0)
            if result.estimator in LOG_SCALE_EFFECT_SIZES:
                ax.set_yscale("log")
        _draw.reference_lines(ax, lines=reference_lines)
        artists, _, _ = _draw.categorical(ax, cells, "point", colors, center_kws=center_kws, interval_kws=interval_kws)
        _draw.tick_labels(ax, range(len(order)), [names(k) for k in order])
        ax.set_xlim(-0.5, len(order) - 0.5)
        table["position"] = range(len(order))
        table.attrs.update(center=result.estimator, interval="ci" if result.ci else None, level=1 - result.alpha if result.ci else None)
        default_ylabel = f"{result.metric}: {result.estimator}" + (f" ({reference} vs other)" if reference is not None else "")
        _draw.finish(ax, (result.group_by or "") if xlabel is None else xlabel, default_ylabel if ylabel is None else ylabel, title)
        return Plot(ax, "estimate", data=table, artists=artists)
