"""
Module Name: viz/stats.py
Description: Significance overlays for the `metric_box`/`metric_relational`
plots in `mindmap/viz/core.py` -- asterisk/bracket annotations vs a fixed
reference method on box plots (`mindmap.analysis.stats.compare_methods`'s
output), and shaded significant-cluster spans on curve plots
(`mindmap.analysis.stats.cluster_permutation_mne`'s output).

Both plot into a single `Axes` (matching `metric_box`/`metric_relational`'s
`ax=None -> (fig, ax)` convention), so there is no per-metric Axes lookup to
do -- the caller is responsible for `df`/`results`/`clusters` already
corresponding to a single metric.

Author: Nino Herve
"""

import pandas as pd

from mindmap.viz.core import metric_box, metric_relational


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

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if reference is None:
        raise ValueError("`reference` is required to place significance annotations.")

    if order is None:
        order = list(pd.unique(df[group_name]))
    if reference not in order:
        raise ValueError(f"reference '{reference}' not found in `order`/`df['{group_name}']`.")
    ref_x = order.index(reference)

    fig, ax = metric_box(df, group_name=group_name, order=order, palette=palette, title=title, ax=ax, **kwargs)

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
    kind: str = "line",
    palette=None, title=None,
    ax=None,
    **kwargs,
):
    """
    Draws `metric_relational` and shades each significant cluster's
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

    group_name, kind, palette, title, ax
        Forwarded to `metric_relational`.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    colors = colors or {"positive": "tab:green", "negative": "tab:red"}

    fig, ax = metric_relational(df, x=x, group_name=group_name, kind=kind, palette=palette, title=title, ax=ax, **kwargs)

    sig = clusters[clusters[p_col] < alpha] if p_col in clusters.columns else clusters

    for _, row in sig.iterrows():
        color = colors["positive"] if row["sign"] > 0 else colors["negative"]
        ax.axvspan(row[f"{x}_start"], row[f"{x}_end"], color=color, alpha=0.15)

    return fig, ax
