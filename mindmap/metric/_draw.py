"""
Module Name: metric/_draw.py
Description: The drawing half of `MetricSuite.plot` -- plain matplotlib
renderers that turn the numbers `_plotdata.py` computed into artists. No
statistic is computed here (box statistics included: `ax.bxp` draws
precomputed ones), and seaborn isn't used, so a figure can't silently show
a different summary than the one reported.

Every renderer takes the target `Axes`, the colors already resolved, and
one `*_kws` dict per layer, merged over this module's defaults and passed
to exactly one matplotlib call -- an unknown key fails in matplotlib
itself. Global styles (`plt.style.use`, rcParams) apply to anything not
set here; with `palette=None`, group colors come from the active
`axes.prop_cycle`.

Author: Nino Herve
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


#: Dark ink for medians/centers/intervals/brackets, neutral gray for reference lines, lighter gray for chance bands.
INK = "#222222"
LINE_COLOR = "#898781"
CHANCE_COLOR = "#c3c2b7"

#: Default fill opacity per element, when `fill_alpha` isn't given.
FILL_ALPHA = {"box": 1.0, "violin": 0.45, "bar": 0.8, "band": 0.2, "kde": 0.35, "hist": 0.45}

KINDS = ("box", "violin", "bar", "point", "strip")

#: The style every plot method shares -- set it with `mindmap.metric.plot.set_style` (or the
#: `style` context manager). `fontsize` None keeps matplotlib's rcParams for labels, ticks,
#: titles and legends and the built-in annotation sizes; a number sets them all, annotations
#: included, so panels from different methods match.
STYLE = {"fontsize": None}


def text_size(default, step=0):
    """An annotation's font size: `default` without a shared `STYLE["fontsize"]`, else that size minus `step`."""
    base = STYLE["fontsize"]
    return default if base is None else max(base - step, 1)

#: The center marker / interval error bar's defaults per categorical kind. `None` colors mean "the group's color".
_CENTER_STYLE = {
    "box": {"marker": "D", "markersize": 6, "markerfacecolor": "white", "markeredgecolor": INK, "color": INK},
    "violin": {"marker": "_", "markersize": 18, "markeredgewidth": 2.5, "color": INK},
    "strip": {"marker": "_", "markersize": 18, "markeredgewidth": 2.5, "color": INK},
    "point": {"marker": "o", "markersize": 7, "color": None},
    "bar": {"marker": "none", "color": INK},
}
_INTERVAL_STYLE = {
    "box": {"ecolor": INK, "elinewidth": 1.5, "capsize": 3},
    "violin": {"ecolor": INK, "elinewidth": 1.5, "capsize": 0},
    "strip": {"ecolor": INK, "elinewidth": 1.5, "capsize": 0},
    "point": {"ecolor": None, "elinewidth": 1.5, "capsize": 3},
    "bar": {"ecolor": INK, "elinewidth": 1.2, "capsize": 3},
}


def get_ax(ax):
    """`ax`, or a new figure's single `Axes`."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _canonical_names():
    """matplotlib's property aliases (`ec`, `edgecolors` -> `edgecolor`, `lw` -> `linewidth`, ...)."""
    import matplotlib.collections, matplotlib.text
    names = {}
    for cls in (matplotlib.collections.Collection, Line2D, Patch, matplotlib.text.Text):
        for canonical, aliases in getattr(cls, "_alias_map", {}).items():
            for alias in aliases:
                names.setdefault(alias, canonical)
    return names


_CANONICAL = _canonical_names()


def merge(defaults, *overrides):
    """
    `defaults` updated by each `*_kws` dict in turn (`None` skipped). A key
    replaces any earlier key that is a matplotlib alias of it, so e.g. a
    user's `edgecolor` overrides a default `edgecolors` instead of clashing
    with it.
    """
    out = dict(defaults)
    for kws in overrides:
        for key, value in (kws or {}).items():
            canonical = _CANONICAL.get(key, key)
            for old in [k for k in out if k != key and _CANONICAL.get(k, k) == canonical]:
                del out[old]
            out[key] = value
    return out


def with_alpha(color, alpha):
    """`color` as RGBA with its alpha replaced."""
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def resolve_colors(levels, palette=None):
    """
    `{level: color}` for every level, in order.

    `palette` may be a `{level: color}` dict (levels it omits take the next
    colors of the active cycle), a list of colors (cycled), a matplotlib
    colormap name (a qualitative map like "tab10" is cycled, a continuous
    one like "viridis" sampled evenly), a seaborn palette name (when
    seaborn is installed), or `None` for the active `axes.prop_cycle` --
    so a global style sets the default colors.
    """
    levels = list(levels)
    if isinstance(palette, dict):
        rest = resolve_colors([v for v in levels if v not in palette])
        return {v: palette[v] if v in palette else rest[v] for v in levels}
    if palette is None:
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    elif isinstance(palette, str):
        if palette in mpl.colormaps:
            cmap = mpl.colormaps[palette]
            if isinstance(cmap, ListedColormap) and cmap.N <= 20:
                colors = list(cmap.colors)
            else:
                colors = [cmap((i + 0.5) / max(len(levels), 1)) for i in range(len(levels))]
        else:
            try:
                import seaborn as sns
            except ImportError:
                raise ValueError(f"palette={palette!r} isn't a matplotlib colormap.") from None
            colors = sns.color_palette(palette, n_colors=max(len(levels), 1))
    else:
        colors = list(palette)
    if not colors:
        raise ValueError("`palette` has no colors.")
    return {v: colors[i % len(colors)] for i, v in enumerate(levels)}


def tick_labels(ax, positions, names, rotate=False):
    """Category ticks at `positions`, labelled `names`."""
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(n) for n in names])
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _ticks_overlap(ax, pad=2.0):
    """Whether any two neighbouring x tick labels overlap (or come within `pad` pixels), as drawn now."""
    renderer = ax.figure.canvas.get_renderer()
    boxes = sorted(
        (t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()),
        key=lambda b: b.x0,
    )
    return any(left.x1 + pad > right.x0 for left, right in zip(boxes, boxes[1:]))


def rotate_ticks(ax, rotate="auto"):
    """
    Rotates the x tick labels by `rotate` degrees (right-aligned to their
    tick); "auto" rotates by 45 only when they would overlap at the axes'
    current size and font, True means 45, False/0 leaves them.
    """
    if rotate == "auto":
        rotate = 45 if _ticks_overlap(ax) else 0
    elif isinstance(rotate, (bool, np.bool_)):
        rotate = 45 if rotate else 0
    elif not isinstance(rotate, (int, float, np.integer, np.floating)) or not np.isfinite(rotate):
        raise ValueError(f"`rotate_ticks` must be 'auto', a bool or an angle in degrees, got {rotate!r}.")
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right", rotation_mode="anchor")


def finish(ax, xlabel=None, ylabel=None, title=None, legend=False, handles=None, legend_title=None):
    """
    Axis labels, title and legend. `legend` is True/False, or "outside"
    for a legend right of the axes; nothing is drawn without `handles`.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    size = STYLE["fontsize"]
    if size is not None:
        for text in (ax.xaxis.label, ax.yaxis.label, ax.title):
            text.set_fontsize(size)
        ax.tick_params(labelsize=size)
    if legend and handles:
        kws = {"frameon": False, "title": legend_title}
        if size is not None:
            kws.update(fontsize=size, title_fontsize=size)
        if legend == "outside":
            kws.update(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.legend(handles=handles, **kws)


def reference_lines(ax, chance=None, chance_value=None, lines=None):
    """
    Background guides: a gray `chance=(lo, hi)` band, a dotted line at
    `chance_value`, and one labelled dashed line per `lines` entry
    (`{label: value}` or `{label: (value, color)}`), picked up by `ax.legend()`.
    """
    if chance is not None:
        if not isinstance(chance, (tuple, list)) or len(chance) != 2:
            raise ValueError(f"`chance` must be a (lo, hi) pair, got {chance!r}.")
        ax.axhspan(*chance, color=CHANCE_COLOR, alpha=0.35, linewidth=0, zorder=0)
    if chance_value is not None:
        ax.axhline(chance_value, color=LINE_COLOR, linestyle=":", linewidth=1, zorder=0)
    if lines is not None and not isinstance(lines, dict):
        raise TypeError(f"`reference_lines` must be a dict {{label: value | (value, color)}}, got {type(lines).__name__}.")
    for label, spec in (lines or {}).items():
        value, color = spec if isinstance(spec, (tuple, list)) else (spec, LINE_COLOR)
        ax.axhline(value, color=color, linestyle="--", linewidth=1.5, zorder=0.5, label=label)


# -- Categorical: one x position per group (and per hue level) -----------------

def categorical(
    ax, cells, kind, colors, fill_alpha=None, points=False, seed=0, center=True, interval=True,
    box_kws=None, violin_kws=None, bar_kws=None, points_kws=None, center_kws=None, interval_kws=None,
    colored_summary=False,
):
    """
    Draws one element per cell at its x position.

    `cells` is a list of dicts with `pos`, `width`, `color_key`, `values`
    (the per-unit points), `center`, `low`, `high` (NaN when not drawn),
    `box` (matplotlib box statistics, for `kind="box"`), and optionally
    `kind` -- "violin" or "strip", to switch one cell of a violin/strip
    plot. `colored_summary` draws each cell's center and interval in its
    color instead of ink. Returns the artists by layer and the lowest/
    highest y drawn.
    """
    if kind not in KINDS:
        raise ValueError(f"kind={kind!r} isn't supported. Choose from {list(KINDS)}.")
    alpha = FILL_ALPHA.get(kind, 1.0) if fill_alpha is None else fill_alpha
    if not 0 <= alpha <= 1:
        raise ValueError(f"`fill_alpha` must be a number in [0, 1], got {fill_alpha!r}.")
    artists = {"boxes": [], "violins": [], "bars": [], "points": [], "centers": [], "intervals": []}
    rng = np.random.default_rng(seed)
    cells = [c for c in cells if len(c["values"]) or np.isfinite(c["center"])]
    interval_kws = dict(interval_kws or {})
    # A half-violin's flat edge sits at the cell position: move the center and interval just off it.
    side = {"high": -1, "low": 1}.get((violin_kws or {}).get("side"), 0)
    if "color" in interval_kws and "ecolor" not in interval_kws:
        interval_kws["ecolor"] = interval_kws.pop("color")  # errorbar's bar color is `ecolor`

    if kind == "box" and cells:
        box_kws = dict(box_kws or {})
        user_box = box_kws.pop("boxprops", {})
        defaults = {
            "medianprops": {"color": INK, "linewidth": 1.5},
            "whiskerprops": {"color": INK, "linewidth": 1.2},
            "capprops": {"color": INK, "linewidth": 1.2},
            "flierprops": {"marker": "o", "markersize": 3, "markerfacecolor": INK, "markeredgecolor": "none"},
            "showfliers": not points,
        }
        bp = ax.bxp(
            [c["box"] for c in cells], positions=[c["pos"] for c in cells], widths=[c["width"] for c in cells],
            patch_artist=True, manage_ticks=False, boxprops=merge({"edgecolor": INK, "linewidth": 1.2}, user_box),
            **merge(defaults, box_kws),
        )
        if "facecolor" not in user_box:
            for patch, c in zip(bp["boxes"], cells):
                patch.set_facecolor(with_alpha(colors[c["color_key"]], alpha))
        artists["boxes"] = bp["boxes"]
        artists["box_parts"] = bp

    for c in cells:
        color, values, pos, width = colors[c["color_key"]], np.asarray(c["values"], dtype=float), c["pos"], c["width"]
        cell_kind = c.get("kind") or kind
        if cell_kind == "violin" and len(values) and np.ptp(values) > 0:
            vp = ax.violinplot([values], positions=[pos], widths=width, **merge({"showextrema": False}, violin_kws))
            for body in vp["bodies"]:
                body.set_alpha(None)  # violinplot sets 0.3 on the whole body; keep the edge opaque, the fill at `alpha`
                body.set_facecolor(with_alpha(color, alpha))
                body.set_edgecolor(color)
                body.set_zorder(1)
            artists["violins"].append(vp)
        elif kind == "bar":
            artists["bars"].append(ax.bar(
                pos, c["center"], width, **merge({"color": with_alpha(color, alpha), "edgecolor": color, "zorder": 1}, bar_kws),
            ))

        if points or cell_kind == "strip":
            jitter = rng.uniform(-0.25 * width, 0.25 * width, size=len(values))
            style = (
                {"s": 18, "color": color, "alpha": 0.8, "edgecolors": "white", "linewidths": 0.6, "zorder": 2.5}
                if cell_kind in ("strip", "point") else {"s": 10, "color": ".2", "alpha": 0.45, "linewidths": 0, "zorder": 2.5}
            )
            artists["points"].append(ax.scatter(pos + jitter, values, **merge(style, points_kws)))

        summary_pos = pos + side * 0.06 * width if cell_kind == "violin" else pos
        if interval and np.isfinite(c["low"]) and np.isfinite(c["high"]):
            mid = c["center"] if np.isfinite(c["center"]) else (c["low"] + c["high"]) / 2
            style = merge(_INTERVAL_STYLE[cell_kind], {"zorder": 3.5})
            if style["ecolor"] is None or colored_summary:
                style["ecolor"] = color
            artists["intervals"].append(ax.errorbar(
                [summary_pos], [mid], yerr=[[mid - c["low"]], [c["high"] - mid]], fmt="none", **merge(style, interval_kws),
            ))
        if center and np.isfinite(c["center"]) and (kind != "bar" or center_kws):
            style = merge(_CENTER_STYLE[cell_kind], {"linestyle": "none", "zorder": 4})
            if style["color"] is None or colored_summary:
                style["color"] = color
                if "markeredgecolor" in style:
                    style["markeredgecolor"] = color
            artists["centers"].extend(ax.plot([summary_pos], [c["center"]], **merge(style, center_kws)))

    drawn = [v for c in cells for v in c["values"]]
    drawn += [c[k] for c in cells for k in ("center", "low", "high") if np.isfinite(c[k])]
    if kind == "bar":
        drawn.append(0.0)
    finite = np.asarray([v for v in drawn if np.isfinite(v)], dtype=float)
    bottom, top = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    return artists, bottom, top


def legend_handles(kind, levels, colors, names, fill_alpha=None):
    """One legend entry per level, styled like the elements drawn (a filled patch, or a marker for points)."""
    alpha = FILL_ALPHA.get(kind, 1.0) if fill_alpha is None else fill_alpha
    if kind in ("point", "strip"):
        return [Line2D([0], [0], marker="o", linestyle="none", color=colors[v], label=names(v)) for v in levels]
    return [Patch(facecolor=with_alpha(colors[v], alpha), edgecolor=colors[v], label=names(v)) for v in levels]


def brackets(ax, specs, top, bottom, bracket_kws=None, text_kws=None):
    """
    Elbow brackets stacked above `top`, narrowest lowest, one tier each --
    `specs` is a list of `(x0, x1, text)`. Tiers are spaced to clear one
    line of text at the axes' current size, and the y-limit is raised to
    fit them.
    """
    specs = sorted(specs, key=lambda s: abs(s[1] - s[0]))
    base = top - min(bottom, ax.get_ylim()[0])
    base = base if base > 0 else (abs(top) or 1.0)
    margin = 0.06 * base
    if any(text for _, _, text in specs):
        # One tier must clear a text line: its height in pixels, as a share of the final (extended) y-range.
        fontsize = mpl.font_manager.FontProperties(size=(text_kws or {}).get("fontsize", text_size(None))).get_size_in_points()
        share = 1.3 * fontsize * ax.figure.dpi / 72 / max(ax.bbox.height, 1.0)
        free = 1 - share * (len(specs) + 1.5)
        if free > 0.3:
            margin = max(margin, share * base / free)
    artists = []
    for tier, (x0, x1, text) in enumerate(specs):
        y = top + margin * (1 + tier)
        x0, x1 = sorted((x0, x1))
        drop = 0.3 * 0.06 * base
        line, = ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], **merge({"color": "black", "linewidth": 1}, bracket_kws))
        label = ax.text((x0 + x1) / 2, y, text, **merge({"ha": "center", "va": "bottom", "fontsize": text_size(None)}, text_kws))
        artists.append((line, label))
    if specs:
        ax.set_ylim(top=top + margin * (len(specs) + 1.5))
    return artists


def column_text(ax, lines_by_pos, top, bottom, text_kws=None):
    """
    Text lines stacked above each x position, the first line on top and at
    the same height in every column, 7.5% of the data span apart; the first
    line is drawn larger and darker. Leaves y-room for the tallest stack.
    """
    span = (top - bottom) or 1.0
    n_lines = max((len(lines) for lines in lines_by_pos.values()), default=0)
    step = 0.075 * span
    artists = []
    for pos, lines in lines_by_pos.items():
        for k, line in enumerate(lines):
            style = {"ha": "center", "va": "bottom", "fontsize": text_size(8) if k == 0 else text_size(7, 1), "color": INK if k == 0 else "#555555"}
            artists.append(ax.text(pos, top + 0.02 * span + (n_lines - 1 - k) * step, line, **merge(style, text_kws)))
    ax.set_ylim(bottom - 0.05 * span, top + (0.06 + n_lines * 0.075) * span)
    return artists


def direction_labels(ax, labels):
    """`(positive, negative)` labels written right of the axes on either side of y = 0, with an up/down arrow."""
    if not isinstance(labels, (tuple, list)) or len(labels) != 2:
        raise ValueError(f"`direction_labels` must be a (positive, negative) pair of strings, got {labels!r}.")
    better, worse = labels
    for text, dy, va in [(f"{better} ↑", 4, "bottom"), (f"{worse} ↓", -4, "top")]:
        ax.annotate(
            text, xy=(1.0, 0.0), xycoords=("axes fraction", "data"), xytext=(4, dy),
            textcoords="offset points", fontsize=text_size(7, 1), color="#666666", ha="left", va=va,
        )


# -- Relational: a value against a coordinate --------------------------------

def lines(ax, curves, colors, interval_style="band", fill_alpha=None, line_kws=None, interval_kws=None, curve_kws=None):
    """
    One line per curve -- `curves` maps a color key to `(xs, center, low,
    high, label)` arrays -- with its interval as a shaded band or error bars.
    `curve_kws={key: kws}` styles single lines, over `line_kws`.
    """
    if interval_style not in ("band", "bars"):
        raise ValueError(f"interval_style={interval_style!r} isn't supported. Choose 'band' or 'bars'.")
    alpha = FILL_ALPHA["band"] if fill_alpha is None else fill_alpha
    artists = {"lines": [], "intervals": []}
    for key, (xs, center, low, high, label) in curves.items():
        color = colors[key]
        line, = ax.plot(xs, center, **merge({"color": color, "linewidth": 1.8, "label": label}, line_kws, (curve_kws or {}).get(key)))
        artists["lines"].append(line)
        ok = np.isfinite(low) & np.isfinite(high)
        if not ok.any():
            continue
        if interval_style == "band":
            artists["intervals"].append(ax.fill_between(
                xs, low, high, where=ok, **merge({"color": color, "alpha": alpha, "linewidth": 0}, interval_kws),
            ))
        else:
            mid = np.where(np.isfinite(center), center, (low + high) / 2)
            artists["intervals"].append(ax.errorbar(
                np.asarray(xs)[ok], mid[ok], yerr=[(mid - low)[ok], (high - mid)[ok]], fmt="none",
                **merge({"ecolor": color, "elinewidth": 1.2, "capsize": 2}, interval_kws),
            ))
    return artists


def raw_lines(ax, curves, colors, raw_kws=None):
    """The unsmoothed center of each curve -- `curves` as in `lines` -- thin and faint, under the smoothed ones."""
    return [
        ax.plot(xs, center, **merge({"color": colors[key], "linewidth": 0.6, "alpha": 0.25, "label": "_nolegend_"}, raw_kws))[0]
        for key, (xs, center, *_) in curves.items()
    ]


def spans(ax, specs, span_kws=None):
    """
    Shaded x-ranges under the curves -- `specs` is a list of `(start, end,
    label, kws)`; returns the patches and the legend handles of the
    labelled ones.
    """
    patches, handles = [], []
    for start, end, label, kws in specs:
        style = merge({"color": LINE_COLOR, "alpha": 0.18, "linewidth": 0, "zorder": 0}, span_kws, kws)
        patch = ax.axvspan(start, end, **merge(style, {"label": label or "_nolegend_"}))
        patches.append(patch)
        if label:
            handles.append(Patch(facecolor=patch.get_facecolor(), edgecolor="none", label=label))
    return patches, handles


def vlines(ax, specs, vline_kws=None):
    """Vertical markers -- `specs` is a list of `(x, label, kws)`; returns the lines and the labelled ones' handles."""
    out = []
    for x, label, kws in specs:
        out.append(ax.axvline(x, **merge(
            {"color": INK, "linewidth": 0.9, "linestyle": "--", "zorder": 1}, vline_kws, kws, {"label": label or "_nolegend_"},
        )))
    return out, [line for line, (_, label, _) in zip(out, specs) if label]


def scatter(ax, groups, colors, points_kws=None):
    """One scatter per group -- `groups` maps a color key to `(xs, ys, label)`."""
    return [
        ax.scatter(xs, ys, **merge({"s": 18, "color": colors[key], "alpha": 0.8, "linewidths": 0, "label": label}, points_kws))
        for key, (xs, ys, label) in groups.items()
    ]


def kde(ax, groups, colors, fill_alpha=None, line_kws=None, fill_kws=None):
    """
    A Gaussian kernel-density curve per group (scipy's Scott bandwidth),
    filled, each normalized on its own so shapes compare regardless of
    group size. `groups` maps a color key to `(values, label)`; a group
    with fewer than two distinct values has no density and is skipped.
    """
    alpha = FILL_ALPHA["kde"] if fill_alpha is None else fill_alpha
    artists = {"lines": [], "fills": []}
    for key, (values, label) in groups.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2 or np.ptp(v) == 0:
            continue
        density = gaussian_kde(v)
        bw = density.factor * np.std(v, ddof=1)
        grid = np.linspace(v.min() - 3 * bw, v.max() + 3 * bw, 256)
        y = density(grid)
        artists["fills"].append(ax.fill_between(grid, y, **merge({"color": colors[key], "alpha": alpha, "linewidth": 0}, fill_kws)))
        line, = ax.plot(grid, y, **merge({"color": colors[key], "linewidth": 1.5, "label": label}, line_kws))
        artists["lines"].append(line)
    ax.set_ylim(bottom=0)
    return artists


def histogram(ax, groups, colors, fill_alpha=None, hist_kws=None):
    """Overlaid histograms on shared bin edges (`bins` in `hist_kws`, default "auto") -- `groups` maps a color key to `(values, label)`."""
    alpha = FILL_ALPHA["hist"] if fill_alpha is None else fill_alpha
    hist_kws = dict(hist_kws or {})
    pooled = np.concatenate([np.asarray(v, dtype=float) for v, _ in groups.values()]) if groups else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    edges = np.histogram_bin_edges(pooled, bins=hist_kws.pop("bins", "auto"))
    artists = []
    for key, (values, label) in groups.items():
        v = np.asarray(values, dtype=float)
        style = {"bins": edges, "histtype": "stepfilled", "facecolor": with_alpha(colors[key], alpha), "edgecolor": colors[key], "linewidth": 1.2, "label": label}
        artists.append(ax.hist(v[np.isfinite(v)], **merge(style, hist_kws)))
    return artists


# -- Cluster permutation -------------------------------------------------------

def cluster_spans(ax, clusters, x, p_col, alpha, colors=None, span_kws=None):
    """Shades each significant cluster's `[x_start, x_end]` span, green where `sign > 0`, red otherwise (override with `colors`)."""
    colors = merge({"positive": "tab:green", "negative": "tab:red"}, colors)
    sig = clusters[clusters[p_col] < alpha] if p_col in clusters.columns else clusters
    return [
        ax.axvspan(row[f"{x}_start"], row[f"{x}_end"], **merge({"color": colors["positive" if row["sign"] > 0 else "negative"], "alpha": 0.15, "linewidth": 0}, span_kws))
        for _, row in sig.iterrows()
    ]


def cluster_summary(
    ax, curves, reference, sig_segments, colors, alpha, names, line_kws=None, center_kws=None, interval_kws=None,
):
    """
    Every group's curve on one axes against `reference`, each competitor
    styled per segment by whether `reference` significantly beats it
    there: solid line + filled marker where significant, dashed + open
    where not (both survive grayscale printing). The interval is a thin
    whisker per point, drawn the same regardless of significance -- not a
    band, whose overlap would read as a significance test.

    `curves` maps each group to `(xs, center, low, high)`; `sig_segments`
    maps each competitor to one bool per segment between adjacent points.
    """
    line_kws = merge({"linewidth": 1.6}, line_kws)
    marker = merge({"marker": "o", "markersize": 7}, center_kws)
    artists = {"lines": [], "centers": [], "intervals": []}
    n_points = len(next(iter(curves.values()))[0]) if curves else 0

    def whiskers(xs, low, high, color):
        ok = np.isfinite(low) & np.isfinite(high)
        if ok.any():
            artists["intervals"].append(ax.vlines(
                np.asarray(xs)[ok], low[ok], high[ok], **merge({"color": color, "linewidth": 1.1, "alpha": 0.6, "zorder": 1}, interval_kws),
            ))

    def point(x_val, y_val, color, filled):
        style = {"color": color, "linestyle": "none", "zorder": 3}
        if not filled:
            style.update(markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6)
        artists["centers"].extend(ax.plot([x_val], [y_val], **merge(style, marker)))

    for name, (xs, center, low, high) in curves.items():
        color = colors[name]
        whiskers(xs, low, high, color)
        if name == reference:
            artists["lines"].extend(ax.plot(xs, center, **merge(line_kws, {"color": color, "zorder": 2, "linewidth": line_kws["linewidth"] * 1.4})))
            for i in range(n_points):
                point(xs[i], center[i], color, filled=True)
            continue
        segments = sig_segments[name]
        for i, significant in enumerate(segments):
            artists["lines"].extend(ax.plot(
                xs[i:i + 2], center[i:i + 2], **merge(line_kws, {"color": color, "linestyle": "-" if significant else "--", "zorder": 2}),
            ))
        for i in range(n_points):
            filled = any(segments[j] for j in (i - 1, i) if 0 <= j < len(segments))
            point(xs[i], center[i], color, filled)

    handles = [Line2D([0], [0], color=colors[m], linewidth=line_kws["linewidth"], label=names(m)) for m in curves]
    handles += [
        Line2D([0], [0], color="gray", linestyle="-", label=f"significant (p < {alpha})", **marker),
        Line2D([0], [0], color="gray", linestyle="--", markerfacecolor="white", markeredgewidth=1.6, label="not significant", **marker),
    ]
    return artists, handles


def cluster_bars(ax, spans, rows, colors, cluster_kws=None):
    """
    A thick horizontal bar per significant cluster: `spans` maps a color
    key to its `[(x_start, x_end), ...]`, `rows` to the bar's y.
    """
    return [
        ax.hlines([rows[key]] * len(extents), [e[0] for e in extents], [e[1] for e in extents],
                  **merge({"color": colors[key], "linewidth": 4, "capstyle": "butt"}, cluster_kws))
        for key, extents in spans.items() if extents
    ]


def cluster_legend(cluster_kws=None):
    """The legend entry explaining the cluster bars."""
    style = merge({"color": "gray", "linewidth": 4}, {k: v for k, v in (cluster_kws or {}).items() if k in ("color", "linewidth")})
    return [Line2D([0], [0], label="significant cluster", **style)]
