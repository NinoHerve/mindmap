"""
Module Name: viz/stats.py
Description: Significance overlays for the `metric_box`/`metric_line`
plots in `mindmap/viz/core.py` -- asterisk/bracket annotations vs a fixed
reference method on box plots (`mindmap.analysis.stats.compare_methods`'s
output), and shaded significant-cluster spans on curve plots
(`mindmap.analysis.stats.cluster_permutation_mne`'s output), plus a
reference-vs-competitors paired-difference panel
(`metric_paired_differences`).

Both plot into a single `Axes` (matching `metric_box`/`metric_line`'s
`ax=None -> (fig, ax)` convention), so there is no per-metric Axes lookup to
do -- the caller is responsible for `df`/`results`/`clusters` already
corresponding to a single metric.

Author: Nino Herve
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mindmap.viz.core import metric_box, metric_line


#: Neutral gray for reference/zero lines, and the lighter gray of shaded chance bands.
LINE_COLOR = "#898781"
CHANCE_COLOR = "#c3c2b7"


def metric_box_significance(
    df: pd.DataFrame,
    results: pd.DataFrame,
    group_name: str = "Method",
    reference: str = None,
    order: list = None,
    p_col: str = "p_corrected",
    alpha: float = 0.05,
    bracket: bool = True,
    palette="viridis", title=None,
    ax=None,
    *,
    chance=None,
    chance_value=None,
    reference_lines=None,
    tick_labels=None,
    reference_suffix=None,
    **kwargs,
):
    """
    Draws `metric_box` and overlays significance markers vs a single fixed
    `reference` method -- not full pairwise brackets across every method
    pair, which would be O(m^2) combinatorial clutter for an m-way
    comparison and isn't needed for the "compare each competitor to one
    reference" design this module targets.

    Parameters
    ----------
    df: pd.DataFrame
        Same long-format table (one metric's worth) passed to `metric_box`/
        `MetricSuite.plot`.

    results: pd.DataFrame
        `mindmap.analysis.stats.compare_methods`'s output: must contain
        `group_name` and `p_col`.

    group_name: str
        Must match the `group_name`/x-axis column used for `df`.

    reference: str
        Required -- the method every bracket is drawn against. Must appear
        in `order` (or `df[group_name]`'s values).

    order: list, optional
        X-axis category order. Defaults to `df[group_name]`'s values in
        order of first appearance (`pd.unique`). Resolved here and fed
        forward into `metric_box` so the drawn boxes and the bracket
        x-positions can never drift apart -- this function is the single
        source of truth for `order` in a given call, not something read
        back after the fact.

    p_col: str
        Column in `results` to threshold against `alpha` (default
        "p_corrected", i.e. already multiple-comparison corrected).

    alpha: float
        Significance threshold; only rows with `p_col < alpha` get annotated.
        Tiers: "*" p<0.05, "**" p<0.01, "***" p<0.001.

    bracket: bool
        If True (default), draws an elbow bracket line from `reference`'s
        box to the competitor's box under the marker. If False, just places
        the marker above the competitor's box.

    palette, title, ax
        Forwarded to `metric_box`.

    chance: tuple[float, float], optional
        `(lo, hi)` range shaded in gray behind the boxes (e.g. a
        permutation chance range).

    chance_value: float, optional
        Dotted horizontal line behind the boxes (e.g. 0 or 0.5).

    reference_lines: dict, optional
        `{label: value}` or `{label: (value, color)}` -- one dashed
        horizontal line per entry, behind the boxes, labelled `label` so
        `ax.legend()` picks it up (e.g. other methods' medians). Default
        color: neutral gray.

    tick_labels: dict, optional
        `{group: display name}` for the x tick labels; groups not listed
        keep their own name.

    reference_suffix: str, optional
        Appended to `reference`'s tick label (e.g. `" (ref.)"`).

    **kwargs
        Forwarded to `metric_box` (e.g. `points=`, `fill_alpha=`, or
        seaborn `boxplot` arguments).

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if reference is None:
        raise ValueError("`reference` is required to place significance annotations.")
    if chance is not None and (not isinstance(chance, (tuple, list)) or len(chance) != 2):
        raise ValueError(f"`chance` must be a (lo, hi) pair, got {chance!r}.")
    if reference_lines is not None and not isinstance(reference_lines, dict):
        raise TypeError(f"`reference_lines` must be a dict {{label: value | (value, color)}}, got {type(reference_lines).__name__}.")
    if tick_labels is not None and not isinstance(tick_labels, dict):
        raise TypeError(f"`tick_labels` must be a dict {{group: label}}, got {type(tick_labels).__name__}.")

    if order is None:
        order = list(pd.unique(df[group_name]))
    if reference not in order:
        raise ValueError(f"reference '{reference}' not found in `order`/`df['{group_name}']`.")
    ref_x = order.index(reference)

    if ax is None:
        _, ax = plt.subplots()
    if chance is not None:
        ax.axhspan(*chance, color=CHANCE_COLOR, alpha=0.35, linewidth=0, zorder=0)
    if chance_value is not None:
        ax.axhline(chance_value, color=LINE_COLOR, linestyle=":", linewidth=1, zorder=0)
    for label, spec in (reference_lines or {}).items():
        value, color = spec if isinstance(spec, (tuple, list)) else (spec, LINE_COLOR)
        ax.axhline(value, color=color, linestyle="--", linewidth=1.5, zorder=0.5, label=label)

    fig, ax = metric_box(df, group_name=group_name, order=order, palette=palette, title=title, ax=ax, **kwargs)

    if tick_labels is not None or reference_suffix is not None:
        labels = [str((tick_labels or {}).get(g, g)) for g in order]
        labels[ref_x] += reference_suffix or ""
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)

    if df.empty:
        return fig, ax

    y_max, y_min = df["Value"].max(), df["Value"].min()
    span = y_max - y_min
    margin = 0.06 * span if span > 0 else 0.06 * (abs(y_max) or 1.0)

    tier = 0
    for _, row in results.iterrows():
        competitor = row[group_name]
        if competitor == reference or competitor not in order:
            continue
        p = row[p_col]
        if pd.isna(p) or p >= alpha:
            continue

        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        comp_x = order.index(competitor)
        y = y_max + margin * (1 + tier)
        tier += 1

        if bracket:
            x0, x1 = sorted((ref_x, comp_x))
            ax.plot([x0, x0, x1, x1], [y - 0.3 * margin, y, y, y - 0.3 * margin], color="black", lw=1)
            ax.text((x0 + x1) / 2, y, stars, ha="center", va="bottom")
        else:
            ax.text(comp_x, y, stars, ha="center", va="bottom")

    if tier:
        ax.set_ylim(top=y_max + margin * (tier + 1.5))

    return fig, ax


def metric_relational_clusters(
    df: pd.DataFrame,
    clusters: pd.DataFrame,
    x: str,
    group_name: str = None,
    p_col: str = "p_value",
    alpha: float = 0.05,
    colors: dict = None,
    palette=None, title=None,
    ax=None,
    **kwargs,
):
    """
    Draws `metric_line` and shades each significant cluster's
    `[f"{x}_start", f"{x}_end"]` span via `ax.axvspan`, colored by
    `clusters["sign"]` -- directly analogous to the `ax.axvspan(...)` cells
    in the superseded `pilot_noise_cluster_permutation.ipynb` /
    `pilot_degradation_cluster_permutation.ipynb` notebooks. Spans are drawn
    in `x`'s own data units (not index positions).

    Parameters
    ----------
    df: pd.DataFrame
        Long-format table, expected pre-filtered to a single metric and the
        two methods being compared (one outcome, one reference-vs-competitor
        pair -- the same granularity as one notebook panel).

    clusters: pd.DataFrame
        `mindmap.analysis.stats.cluster_permutation_mne`'s
        `result["clusters"]`, with columns `f"{x}_start"`, `f"{x}_end"`,
        "sign", and `p_col`.

    x: str
        Coordinate column, same as `coord_col` passed to
        `cluster_permutation_mne`.

    p_col, alpha
        Only clusters with `p_col < alpha` are shaded (default: every
        cluster in `clusters` already reflects `cluster_alpha`, so this
        just re-applies that same threshold defensively).

    colors: dict, optional
        `{"positive": ..., "negative": ...}` matplotlib colors for
        `sign > 0` / `sign < 0` clusters. Defaults to
        `{"positive": "tab:green", "negative": "tab:red"}`.

    group_name, palette, title, ax
        Forwarded to `metric_line`.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    colors = colors or {"positive": "tab:green", "negative": "tab:red"}

    fig, ax = metric_line(df, x=x, group_name=group_name, palette=palette, title=title, ax=ax, **kwargs)

    sig = clusters[clusters[p_col] < alpha] if p_col in clusters.columns else clusters

    for _, row in sig.iterrows():
        color = colors["positive"] if row["sign"] > 0 else colors["negative"]
        ax.axvspan(row[f"{x}_start"], row[f"{x}_end"], color=color, alpha=0.15)

    return fig, ax


def _resolve_method_colors(names, palette, colors):
    """Builds a `{name: color}` dict for `names`, `colors` overriding `palette`'s assignment."""
    base = dict(zip(names, sns.color_palette(palette, n_colors=len(names))))
    if colors:
        base.update(colors)
    return base


def metric_relational_cluster_summary(
    df: pd.DataFrame,
    clusters_by_competitor: dict,
    x: str,
    reference: str,
    coord_order: list,
    group_name: str = None,
    p_col: str = "p_value",
    alpha: float = 0.05,
    colors: dict = None,
    palette = "viridis",
    marker: str = "o",
    markersize: float = 7,
    linewidth: float = 1.6,
    show_spread: bool = True,
    spread_interval: float = 50,
    title: str = None,
    ax=None,
    **kwargs,
):
    """
    Compact, publication-style alternative to drawing one
    `metric_relational_clusters` panel per competitor: every competitor's
    median curve is drawn on one shared `Axes` against a single `reference`
    curve, with each competitor's own line/marker style encoding, *per
    segment* between adjacent `coord_order` points, whether `reference`
    significantly outperforms it there -- solid line + filled marker where
    significant, dashed line + open (white-face) marker where not. Lets a
    reader see, at a glance and without a separate stats table or one
    subplot per pair, which competitors are inferior to `reference` and
    over what range of `x` -- the same question `metric_relational_clusters`
    answers for one pair at a time, condensed onto one axes for every pair
    at once.

    Deliberately encodes significance via marker fill + line style rather
    than opacity (an earlier version of this function faded the line/band
    alpha instead): filled-vs-open and solid-vs-dashed both survive
    grayscale printing and low-vision viewing, where two alpha levels on a
    colored line often don't read as reliably different.

    Spread (`show_spread`, on by default) is drawn as a thin per-point IQR
    whisker rather than a continuous band between points -- an earlier,
    band-based version of this function was dropped entirely, because a
    *continuous, filled* ribbon invites reading "do the ribbons overlap" as
    a proxy for significance, which isn't what the underlying *paired*,
    region-level cluster-permutation test actually computes, and five
    overlapping fills across competitors turned to visual mud. A thin,
    same-style-regardless-of-significance whisker at each point restores the
    uncertainty information good research-figure practice expects, without
    either problem: it doesn't read as a continuous region to compare for
    overlap, and it stays visually subordinate to the solid/dashed +
    filled/open styling that actually carries the significance result.

    Computes each method's median curve directly via pandas (not through
    seaborn/`metric_line`, unlike every other function in this module)
    because per-segment styling needs the actual (x, median) arrays in hand
    to draw one `ax.plot` call per segment, rather than one call for the
    whole curve.

    Parameters
    ----------
    df: pd.DataFrame
        Long-format table (one metric's worth), containing `reference` and
        every key of `clusters_by_competitor` under `group_name`.

    clusters_by_competitor: dict[str, pd.DataFrame]
        `{competitor_name: clusters_df}`, one entry per competitor --
        typically built from `MetricSuite.cluster_permutation(..., comparisons=[...])`'s
        returned `ClusterResult.clusters_by_pair` (see
        `MetricSuite.plot_cluster_permutation`, which does exactly that
        whenever more than one pair is requested). Each `clusters_df` has
        `mindmap.analysis.stats.cluster_permutation_mne`'s columns:
        `f"{x}_start"`, `f"{x}_end"`, "sign", `p_col`.

    x: str
        Coordinate column, same as `coord_col` passed to whatever produced
        `clusters_by_competitor` (e.g. "snr").

    reference: str
        The method every competitor's segments are compared against. Always
        drawn solid, the fixed baseline everything else is read relative to.

    coord_order: list
        The exact, explicit coordinate ordering `clusters_by_competitor` was
        computed with (e.g. `[50, 20, 10, 5, 2, 1]`) -- required (not
        optional, unlike `cluster_permutation`'s own default) because
        segment significance below is resolved by *index position* in this
        list against each cluster's `f"{x}_start"`/`f"{x}_end"`, so it must
        match whatever ordering produced `clusters_by_competitor` exactly,
        not just share the same set of values.

    group_name: str
        Column identifying each method's rows in `df` (`MetricSuite`'s
        `group_name`).

    p_col, alpha: str, float
        A competitor's segment counts as significant where it falls inside
        a cluster with `p_col < alpha` and `sign > 0` (`reference` wins --
        same sign convention as `cluster_permutation_mne`).

    colors: dict, optional
        `{method_name: color}`, overriding/extending `palette`'s assignment
        -- pass a project's own fixed method->color mapping here (e.g. a
        manuscript's shared `COLORS` dict) for a stable, reproducible legend
        across figures.

    palette: str | list
        Seaborn palette used to color any method not covered by `colors`,
        assigned in `[reference] + list(clusters_by_competitor)` order.

    marker: str
        Marker drawn at each `coord_order` point.

    markersize, linewidth: float
        Marker size and line width for every curve (`reference`'s line is
        drawn slightly thicker than this, to read as the fixed baseline).

    show_spread: bool
        Draw a thin IQR whisker at every point (default True). Whiskers are
        drawn the same way regardless of significance -- spread and
        significance are deliberately two separate visual channels, not
        conflated into one.

    spread_interval: float
        Percentile-interval width for the whisker (e.g. 50 -> 25th-75th
        percentile), same convention as `errorbar=("pi", 50)` used elsewhere
        in this codebase's `.plot(kind="line")` calls. Ignored if
        `show_spread=False`.

    title: str, optional
        Axes title.

    ax: matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if not given.

    **kwargs
        Forwarded to every `ax.plot` call (avoid passing `linestyle`, which
        this function sets itself to encode significance).

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if group_name is None:
        raise ValueError("`group_name` is required to identify each method's rows in `df`.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    methods = [reference] + list(clusters_by_competitor.keys())
    resolved_colors = _resolve_method_colors(methods, palette, colors)

    lo_q, hi_q = (100 - spread_interval) / 200, 1 - (100 - spread_interval) / 200

    def curve(name):
        grouped = df[df[group_name] == name].groupby(x)["Value"]
        med = grouped.median().reindex(coord_order).to_numpy()
        if not show_spread:
            return med, None, None
        lo = grouped.quantile(lo_q).reindex(coord_order).to_numpy()
        hi = grouped.quantile(hi_q).reindex(coord_order).to_numpy()
        return med, lo, hi

    def draw_marker(x_val, y_val, color, filled):
        if filled:
            ax.plot(x_val, y_val, marker=marker, color=color, markersize=markersize, linestyle="none", zorder=3)
        else:
            ax.plot(x_val, y_val, marker=marker, markerfacecolor="white", markeredgecolor=color,
                     markeredgewidth=1.6, markersize=markersize, linestyle="none", zorder=3)

    def draw_whisker(x_val, lo_val, hi_val, color):
        ax.plot([x_val, x_val], [lo_val, hi_val], color=color, linewidth=1.1, alpha=0.6, zorder=1)

    n = len(coord_order)

    # 1. Reference: one solid curve, the fixed baseline.
    ref_med, ref_lo, ref_hi = curve(reference)
    ref_color = resolved_colors[reference]
    ax.plot(coord_order, ref_med, color=ref_color, linewidth=linewidth * 1.4, zorder=2, **kwargs)
    for i in range(n):
        if show_spread:
            draw_whisker(coord_order[i], ref_lo[i], ref_hi[i], ref_color)
        draw_marker(coord_order[i], ref_med[i], ref_color, filled=True)

    # 2. Each competitor: per-segment line style/marker fill, resolved from that
    # competitor's own significant (p_col < alpha, sign > 0) cluster index ranges.
    for name, clusters in clusters_by_competitor.items():
        med, lo, hi = curve(name)
        color = resolved_colors[name]
        sig = clusters[(clusters[p_col] < alpha) & (clusters["sign"] > 0)] if not clusters.empty else clusters
        sig_ranges = [
            tuple(sorted((coord_order.index(row[f"{x}_start"]), coord_order.index(row[f"{x}_end"]))))
            for _, row in sig.iterrows()
        ]

        seg_significant = [
            any(lo_idx <= i and i + 1 <= hi_idx for lo_idx, hi_idx in sig_ranges)
            for i in range(n - 1)
        ]

        for i, is_sig in enumerate(seg_significant):
            ax.plot(coord_order[i:i + 2], med[i:i + 2], color=color, linewidth=linewidth,
                     linestyle="-" if is_sig else "--", zorder=2, **kwargs)

        for i in range(n):
            if show_spread:
                draw_whisker(coord_order[i], lo[i], hi[i], color)
            point_sig = any(seg_significant[j] for j in (i - 1, i) if 0 <= j < n - 1)
            draw_marker(coord_order[i], med[i], color, filled=point_sig)

    # 3. Manual legend -- one color swatch per method, plus two generic entries
    # explaining the filled/open + solid/dashed convention (custom, not a
    # standard matplotlib idiom, so it needs to read on its own).
    handles = [Line2D([0], [0], color=resolved_colors[m], linewidth=linewidth) for m in methods]
    handles += [
        Line2D([0], [0], marker=marker, color="gray", linestyle="-", markersize=markersize),
        Line2D([0], [0], marker=marker, color="gray", markerfacecolor="white",
               markeredgewidth=1.6, linestyle="--", markersize=markersize),
    ]
    ax.legend(handles, methods + [f"significant (p < {alpha})", "not significant"], frameon=False)

    ax.set_xlabel(x)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_paired_differences(
    diffs,
    order=None,
    palette=None,
    annotations=None,
    violin_min_n=30,
    fill_alpha=0.45,
    seed=0,
    tick_labels=None,
    direction_labels=None,
    ax=None,
):
    """
    Per-unit paired differences, one column per competitor: violins (with
    an IQR line) for large samples, jittered points for small ones, a thick
    median line each, a horizontal line at 0, and optional text lines
    stacked above each column.

    !!! note "Technical details"
        The switch is per column: a column with at least `violin_min_n`
        values is drawn as a violin (`ax.violinplot`, no extrema) plus a
        thin black Q1-Q3 line; below that, as points jittered uniformly in
        +/-0.12 around the column (reproducible through `seed`). Every
        column gets a thick black median line. Annotation lines are stacked
        so the *first* line of every column sits on top, at the same height
        for every column, 7.5% of the data span apart; the first line is
        drawn larger and darker than the rest. The y-limits leave room for
        the tallest stack.

    Parameters
    ----------
    diffs: pd.DataFrame | dict
        One column (or `{name: 1-D array}` entry) per competitor; `NaN`s
        are dropped per column. Already oriented by the caller (e.g.
        positive = reference better).

    order: list, optional
        Columns to draw, left to right. Defaults to `diffs`' own order.

    palette: dict | str | list, optional
        `{name: color}`, or anything `sns.color_palette` accepts (assigned
        in `order`). Default: seaborn's current palette.

    annotations: dict, optional
        `{name: [line, ...]}` drawn above each column, first line on top.

    violin_min_n: int, optional
        Minimum number of values for a violin instead of points (default 30).

    fill_alpha: float, optional
        Violin fill opacity (default 0.45).

    seed: int, optional
        Seed for the point jitter (default 0).

    tick_labels: dict, optional
        `{name: display name}` for the x tick labels.

    direction_labels: tuple[str, str], optional
        `(positive, negative)` labels, e.g. `("wCSET better", "other method
        better")`, written just right of the axes on either side of y = 0
        with an up/down arrow.

    ax: matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if not given.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if isinstance(diffs, pd.DataFrame):
        values = {c: diffs[c].dropna().to_numpy(dtype=float) for c in diffs.columns}
    elif isinstance(diffs, dict):
        values = {c: np.asarray(v, dtype=float) for c, v in diffs.items()}
        values = {c: v[~np.isnan(v)] for c, v in values.items()}
    else:
        raise TypeError(f"`diffs` must be a pd.DataFrame or a dict of arrays, got {type(diffs).__name__}.")

    if order is None:
        order = list(values)
    missing = [c for c in order if c not in values]
    if missing:
        raise ValueError(f"`order` entries {missing} not found in `diffs` (available: {list(values)}).")
    empty = [c for c in order if len(values[c]) == 0]
    if empty:
        raise ValueError(f"No values to draw for {empty}.")
    if not (isinstance(fill_alpha, (int, float)) and 0 <= fill_alpha <= 1):
        raise ValueError(f"`fill_alpha` must be a number in [0, 1], got {fill_alpha!r}.")
    if direction_labels is not None and (not isinstance(direction_labels, (tuple, list)) or len(direction_labels) != 2):
        raise ValueError(f"`direction_labels` must be a (positive, negative) pair of strings, got {direction_labels!r}.")

    if isinstance(palette, dict):
        unknown = [c for c in order if c not in palette]
        if unknown:
            raise ValueError(f"`palette` has no color for {unknown}.")
        colors = palette
    else:
        colors = dict(zip(order, sns.color_palette(palette, n_colors=len(order))))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.axhline(0, color=LINE_COLOR, linewidth=1, zorder=0)
    rng = np.random.default_rng(seed)
    for i, c in enumerate(order):
        y = values[c]
        if len(y) >= violin_min_n:
            vp = ax.violinplot([y], positions=[i], widths=0.7, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(colors[c])
                body.set_edgecolor(colors[c])
                body.set_alpha(fill_alpha)
                body.set_zorder(1)
            q1, q3 = np.percentile(y, [25, 75])
            ax.plot([i, i], [q1, q3], color="#0b0b0b", linewidth=1.5, zorder=3)
        else:
            x = i + rng.uniform(-0.12, 0.12, size=len(y))
            ax.scatter(x, y, s=18, color=colors[c], alpha=0.8, edgecolors="white", linewidths=0.6, zorder=2)
        med = np.median(y)
        ax.plot([i - 0.22, i + 0.22], [med, med], color="#0b0b0b", linewidth=2, zorder=4)

    ymax = max(np.max(values[c]) for c in order)
    ymin = min(np.min(values[c]) for c in order)
    span = (ymax - ymin) or 1.0
    n_lines = max((len(annotations.get(c, [])) for c in order), default=0) if annotations else 0
    step = 0.075 * span
    for i, c in enumerate(order):
        for k, line in enumerate((annotations or {}).get(c, [])):
            ax.text(
                i, ymax + 0.02 * span + (n_lines - 1 - k) * step, line, ha="center", va="bottom",
                fontsize=8 if k == 0 else 7, color="#222222" if k == 0 else "#555555",
            )
    ax.set_ylim(ymin - 0.05 * span, ymax + (0.06 + n_lines * 0.075) * span)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str((tick_labels or {}).get(c, c)) for c in order])
    ax.spines[["top", "right"]].set_visible(False)

    if direction_labels is not None:
        better, worse = direction_labels
        for text, dy, va in [(f"{better} \u2191", 4, "bottom"), (f"{worse} \u2193", -4, "top")]:
            ax.annotate(
                text, xy=(1.0, 0.0), xycoords=("axes fraction", "data"), xytext=(4, dy),
                textcoords="offset points", fontsize=7, color="#666666", ha="left", va=va,
            )

    return fig, ax
