"""
Module Name: viz/core.py
Description: Functions to plot using arrays or dataframe as input.

===============================================================================
MATH-SPACE VISUALIZATION (core.py)
===============================================================================
Targets: np.ndarray, pd.DataFrame
Backends: Matplotlib (2D grid), pyvista (3D point clouds)
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pyvista as pv

def create_line_mesh(pts: np.ndarray) -> pv.PolyData:
    """Create a polydata line"""
    if len(pts) < 2:
        return pv.PolyData(pts)

    cells = np.hstack(([len(pts)], np.arange(len(pts))))

    return pv.PolyData(pts, lines=cells)


def plot_multi_scatterplots(
    data_ref:np.ndarray, 
    other_datasets:dict[str, np.ndarray],
    point_labels: list[str] = None,
    frame_labels: list[str] = None,
    slider_name: str = "Time",
    title: str = "Scatter plot",
    colors: list[str] = None,
) -> go.Figure:
    """
    Core engine to create an interactive scatter plot with a slider.

    Logic:
    - Each 'point' in the cloud is a row in the input matrix (e.g. a channel)
    - The 'slider' moves through the columns (e.g. Timepoints).

    Parameters
    ----------
    data_ref: np.ndarray of shape (M,T)
        Reference array.
    
    other_datasets: dict[str, np.ndarray]
        Dictionary of array, each of shape (M,T). Key will be 
        used as label name.

    title: str
        Plot title.

    colors: list[str]
        List of colors accepted by plotly.go.

    point_labels: list[str]
        Names of points. Will appear when hovering
        on point.

    slider: dict
        Slider 'name' and 'labels'.

    Returns
    -------
    fig: go.Figure
    """
    if colors is None:
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    num_points, num_frames = data_ref.shape

    if point_labels is None:
        point_labels = [f"Sample {j}" for j in range(num_points)]
    if frame_labels is None:
        frame_labels = [f"Frame {j}" for j in range(num_frames)]

    # Determine global scale
    all_vals = [data_ref] + list(other_datasets.values())
    v_min = min(d.min() for d in all_vals)
    v_max = max(d.max() for d in all_vals)
    pad = (v_max - v_min) * 0.05
    global_range = [v_min - pad, v_max + pad]

    # Initialize figure
    fig = go.Figure()

    # add identity line (y=x) as background reference
    fig.add_shape(
        type="line", x0=global_range[0], y0=global_range[0],
        x1=global_range[1], y1=global_range[1],
        line=dict(color="rgba(150,150,150,0.3)", dash="dash"),
        layer="below",
    )

    # add initial state (Frame 0)
    for i, (name, data) in enumerate(other_datasets.items()):
        fig.add_trace(go.Scatter(
            x=data_ref[:, 0],
            y=data[:,0],
            mode="markers",
            name=name,
            text=point_labels,
            marker=dict(color=colors[i % len(colors)], size=7, opacity=0.7),
            hovertemplate="<b>%{text}</b><br>Ref: %{x:.2e}<br>Cmp: %{y:.2e}<extra></extra>"
        ))

    # define animation frame
    fig.frames = [
        go.Frame(
            name=str(t),
            data=[go.Scatter(x=data_ref[:, t], y=d[:, t]) for d in other_datasets.values()],
            layout=go.Layout(title_text=f"{title} | {frame_labels[t]}")
        ) for t in range(num_frames)
    ]

    # UI: slider, animation speed and Layout
    slider_steps = [
        dict(
            method="animate",
            label=str(t),
            args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
        ) for t in range(num_frames)
    ]

    speeds = [("0.5x", 200), ("1x", 100), ("2x", 50), ("5x", 20)]

    speed_buttons = [
        dict(
            label=label,
            method="animate",
            args=[
                None, {"frame": {"duration": dur, "redraw":True},"fromcurrent":True, "transition":{"duration":dur/2}}]
        ) for label, dur in speeds
    ]

    fig.update_layout(
        height=800, width=850,
        template="plotly_white",
        xaxis=dict(title="Reference Dataset", range=global_range, constrain="domain"),
        yaxis=dict(title="Comparison Datasets", range=global_range, scaleanchor="x", scaleratio=1),
        sliders=[dict(active=0, steps=slider_steps, currentvalue={"prefix": f"{slider_name}: "})],
        updatemenus=[
            dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])]),
            dict(direction="down", showactive=True, buttons=speed_buttons, x=-0.05, y=0.9),
        ],
        legend=dict(orientation="h", y=1.05, x=1)
    )

    return fig


def plot_cog_trajectories(
        dict_of_trajs: dict[str:np.ndarray],
        times: np.ndarray = None,
        ref_name: str = None,
        electrode_pos: np.ndarray = None,
        brain_mesh: pv.PolyData = None,
        center_ref: bool = None,
        colors = None,
        units: str = "mm",
):
    """Plot Center of Gravity"""
    plotter = pv.Plotter()
    processed_trajs = dict_of_trajs.copy()

    # colors
    if colors is None:
        colors = ["firebrick", "dodgerblue", "goldenrod", "darkorchid"]

    # center reference
    if center_ref and ref_name in dict_of_trajs:
        ref_data = dict_of_trajs[ref_name]

        # subtract reference
        processed_trajs = {
            name: (traj - ref_data) for name, traj in dict_of_trajs.items()
        }

        # In centered mode, brain and electrodes lose anatomical meaning
        brain_mesh = None
        electrode_pos = None
        title_suffix = " (Relative to Reference)"
    else:
        title_suffix = ""

    # add context (brain + electrodes)
    if brain_mesh is not None:
        plotter.add_mesh(brain_mesh, color="tan", opacity=0.15)
    if electrode_pos is not None:
        plotter.add_points(electrode_pos, color="black", point_size=3, opacity=0.5)

    actors = {}

    # initialize trajectories
    not_ref = 0
    for name, traj in processed_trajs.items():
        # reference appearance
        is_ref = name == ref_name
        color = "black" if is_ref else colors[not_ref]
        width = 5 if is_ref else 3

        # create path actor
        line_mesh = create_line_mesh(traj[:1, :])
        path_actor = plotter.add_mesh(
            line_mesh, color = color, line_width = width,
            label = name, render_lines_as_tubes = True,
        )

        # create head actor
        head_mesh = pv.PolyData(traj[0:1, :])
        head_actor = plotter.add_mesh(
            head_mesh, color = color, point_size = 12, render_points_as_spheres = True,
        )

        actors[name] = {"path": path_actor, "head": head_actor, "data": traj}

        if not is_ref:
            not_ref += 1

    # slider range
    n_samples = list(dict_of_trajs.values())[0].shape[0]
    if times is None:
        times = np.arange(n_samples)
        time_unit = "Index"
    else:
        time_unit = "s"

    t_min, t_max = times[0], times[-1]

    # slider update rule
    def update_all(value):
        idx = np.argmin(np.abs(times - value)) + 1

        if hasattr(plotter, "slider_widgets") and len(plotter.slider_widgets) > 0:
            plotter.slider_widgets[0].GetRepresentation().SetTitleText(f"Time: {value:.3f} {time_unit}")

        for _, info in actors.items():
            # update path
            new_line = create_line_mesh(info["data"][:idx, :])
            info["path"].mapper.dataset.copy_from(new_line)

            # update head
            info["head"].mapper.dataset.points = info["data"][idx-1:idx, :]

        plotter.render()

    # add slider
    plotter.add_slider_widget(
        callback = update_all,
        rng = [t_min, t_max],
        value = t_min,
        title = f"Time [{time_unit}]{title_suffix}",
        interaction_event = "always",
        style = "modern",
    )

    if center_ref and ref_name:
        plotter.add_text("Origin (0,0,0) is Ground Truth", font_size=10, position="upper_right")

    # grid formatting
    label_fmt = "%.1f" if units == "mm" else "%.3f"
    axis_label = f"Distance [{units}]" if not center_ref else f"Error [{units}]"

    plotter.show_grid(
        xtitle = f"X {axis_label}",
        ytitle = f"Y {axis_label}",
        ztitle = f"Z {axis_label}",
        fmt = label_fmt,
        location = "outer",
        color = "gray",
    )

    plotter.add_legend()
    return plotter


def plot_rem(
    rem = np.ndarray,
    extent: list = None,
    title: str = "Relative Energy Map",
    label: str = "Relative Energy",
):
    """Generic Heatmap plotter for REM array."""
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        rem,
        aspect = "auto",
        origin = "lower",
        extent = extent,
        cmap = "magma",
    )

    plt.colorbar(im, label=label)
    ax.set_title(title)

    return fig, ax


def plot_confusion_matrix(tp, tn, fp, fn, title="Confusion matrix", ax=None, **kwargs):
    """
    Plots confusion matrix data.

    Parameters:
    ----------
    tp: int
        Number of True positives.

    tn: int
        Number of True negatives.

    fp: int
        Number of False positives.

    fn: int
        Number of False negatives.

    title: str
        Title

    Returns:
    -------
    fig, axs
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    else:
        fig = ax.get_figure()

    cm = [[tn, fp], [fn, tp]]

    sns_params = {
        "annot": True,
        "fmt": "d",
        "cmap": "Blues",
        "xticklabels": ["Pred 0", "Pred 1"],
        "yticklabels": ["True 0", "True 1"],
    }
    sns_params.update(kwargs)

    sns.heatmap(data=cm, ax=ax, **sns_params)
    ax.set_title(title)

    return fig, ax


def plot_scores(
    scores = dict,
    title = "Evaluation scores",
    ax = None,
    **kwargs
):
    """
    Plots evaluations scores.

    Parameters:
    ----------
    
    Returns:
    -------
    fig, axs
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        fig = ax.get_figure()

    # clean data
    clean = {k: v for k, v in scores.items() if not np.isnan(v)}

    bar_params = {
        "color": "skyblue",
        "edgecolor": "navy",
        "alpha": 0.8,
    }
    bar_params.update(kwargs)

    # create bar plot
    bars = ax.bar(clean.keys(), clean.values(), **bar_params)
    ax.bar_label(bars, fmt="%.2f", padding=3)

    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    return fig, ax


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc:float = None,
    title="Receiver Operating Characteristic",
    ax=None,
    **kwargs
):
    """Plot ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.get_figure()

    line_params = {
        "color": "darkorange",
        "lw": 2,
    }

    # add label
    if auc is not None:
        line_params["label"] = f"ROC (AUC = {auc:.3f})"
    else:
        line_params["label"] = "ROC"

    line_params.update(kwargs)

    # plotting
    ax.plot(fpr, tpr, **line_params)

    # reference: chance level
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")

    # styling & limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    return fig, ax

def plot_pr(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    prevalence: float = None,
    title: str="Precision-Recall Curve",
    ax=None,
    **kwargs
):
    """Plot PR curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.get_figure()

    line_params = {
        "color": "teal",
        "lw": 2,
    }

    # add label
    if ap is not None:
        line_params["label"] = f"AP = {ap:.3f})"
    else:
        line_params["label"] = "PR Curve"

    line_params.update(kwargs)

    # plotting
    ax.plot(precision, recall, **line_params)

    # reference: chance level
    if prevalence is not None:
        ax.axhline(y=prevalence, color="navy", lw=2, linestyle="--", label=f"Chance ({prevalence:.2})")

    # styling & limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Prediction Value)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    return fig, ax


def metric_box(df, group_name="Method", order=None, palette="viridis", points=True, title=None, ax=None, fill_alpha=None, **kwargs):
    """
    Default style: Boxplot with overlaid raw data points.

    Plots exactly the rows in `df` (one metric's worth -- filtering to a
    single metric is the caller's responsibility, e.g.
    `MetricSuite.plot(metric=...)`), onto a single `Axes`.

    Parameters
    ----------
    group_name: str
        Column driving both the x-axis category and the color of each box
        -- a single column plays both roles (no separate hue variable).

    order: list, optional
        X-axis category order. Defaults to seaborn's own ordering.

    palette: str | list | dict
        Color palette for the `group_name` levels -- anything seaborn's
        `palette` argument accepts.

    points: bool
        Whether to overlay a `sns.stripplot` of raw data points (default
        True). Turn off for dense data (hundreds of points per box) where
        the overlay is visual noise rather than signal.

    title: str, optional
        Axes title.

    ax: matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if not given.

    fill_alpha: float, optional
        Opacity of the box fills only, in [0, 1] -- edges, whiskers and
        medians stay opaque. Default `None` leaves seaborn's opaque fills.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if fill_alpha is not None and not (isinstance(fill_alpha, (int, float)) and 0 <= fill_alpha <= 1):
        raise ValueError(f"`fill_alpha` must be a number in [0, 1], got {fill_alpha!r}.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"linewidth": 1.5}
    params = {**defaults, **kwargs}

    n_patches = len(ax.patches)
    sns.boxplot(data=df, x=group_name, y="Value", hue=group_name, order=order, palette=palette, ax=ax, legend=False, **params)
    if fill_alpha is not None:
        # Only the patches this call just drew -- anything already on `ax` (e.g. a shaded band) is left alone.
        for patch in ax.patches[n_patches:]:
            r, g, b, _ = patch.get_facecolor()
            patch.set_facecolor((r, g, b, fill_alpha))
    if points:
        sns.stripplot(data=df, x=group_name, y="Value", order=order, color=".2", alpha=0.4, ax=ax)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_violin(df, group_name="Method", order=None, palette="muted", title=None, ax=None, **kwargs):
    """
    Default style: Split or inner-quartile violin plots.

    See `metric_box` for the shared parameters/return shape.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"inner": "quartile", "cut": 0}
    params = {**defaults, **kwargs}

    sns.violinplot(data=df, x=group_name, y="Value", hue=group_name, order=order, palette=palette, ax=ax, legend=False, **params)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_bar(df, group_name="Method", order=None, palette="magma", title=None, ax=None, **kwargs):
    """
    Default style: Bar chart with error bars (confidence intervals).

    See `metric_box` for the shared parameters/return shape.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"errorbar": "sd", "capsize": .1}
    params = {**defaults, **kwargs}

    sns.barplot(data=df, x=group_name, y="Value", hue=group_name, order=order, palette=palette, ax=ax, legend=False, **params)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_distribution(df, group_name=None, palette=None, title=None, ax=None, **kwargs):
    """
    Kernel-density estimate of "Value"'s distribution -- unlike
    `metric_box`/`metric_violin`/`metric_bar`, `group_name` here is an
    optional color overlay (`sns.kdeplot`'s `hue`), not a required x-axis
    category: "Value" itself is the x-axis, so a single ungrouped curve is
    just as meaningful as several overlaid ones.

    Plots exactly the rows in `df` (one metric's worth -- filtering to a
    single metric is the caller's responsibility), onto a single `Axes`.

    Parameters
    ----------
    df: pd.DataFrame
        The tidy dataframe.

    group_name: str, optional
        Column for color encoding -- when given, draws one filled,
        translucent curve per group, independently normalized
        (`common_norm=False`) so shapes are directly comparable regardless
        of group size, overlaid rather than faceted. Omit for one curve
        over the whole population.

    palette: str | list | dict
        Color palette for the `group_name` levels.

    title: str, optional
        Axes title.

    ax: matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if not given.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"fill": True, "common_norm": False, "alpha": 0.35, "linewidth": 1.5}
    params = {**defaults, **kwargs}

    sns.kdeplot(data=df, x="Value", hue=group_name, palette=palette, ax=ax, **params)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_histogram(df, group_name=None, palette=None, title=None, ax=None, **kwargs):
    """
    Histogram of "Value" -- unlike `metric_box`/`metric_violin`/`metric_bar`,
    `group_name` here is an optional color overlay (`sns.histplot`'s `hue`),
    not a required x-axis category: "Value" itself is the x-axis, so a
    single ungrouped histogram is just as meaningful as several overlaid
    ones. See `metric_distribution` for the shared parameters/return shape.

    !!! note "Technical details"
        `multiple="layer"` (overlapping, not stacked/side-by-side) with a
        translucent `alpha` is the default when `group_name` is given, so
        overlapping bars stay readable instead of hiding each other.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"multiple": "layer", "alpha": 0.4}
    params = {**defaults, **kwargs}

    sns.histplot(data=df, x="Value", hue=group_name, palette=palette, ax=ax, **params)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_heatmap(df, group_name="Method", cmap="RdYlGn", title=None, figsize=(10, 6), ax=None, **kwargs):
    """
    Default style: Clean matrix with centered annotations.

    Unlike `metric_box`/`metric_violin`/`metric_bar`/`metric_line`/`metric_scatter`,
    this plots multiple metrics at once by design -- "Metric" as columns is
    the actual feature (a metric x group summary matrix), not incidental
    faceting, so it's exempt from the "one metric per call" convention the
    other plot kinds follow.

    Parameters
    ----------
    cmap: str | Colormap
        Color map for the matrix -- the heatmap analogue of `metric_box`'s
        `palette` (a continuous color scale here, rather than a discrete
        color per group).

    title: str, optional
        Axes title.

    figsize: tuple
        Only used when `ax` is not given.

    ax: matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if not given.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    defaults = {"annot": True, "fmt": ".2f", "cbar_kws": {"label": "Average Value"}}
    params = {**defaults, **kwargs}

    pivot = df.pivot_table(index=group_name, columns="Metric", values="Value", aggfunc="mean")
    sns.heatmap(pivot, cmap=cmap, ax=ax, **params)
    if title:
        ax.set_title(title)

    return fig, ax


def _metric_relational(df, x, plot_func, group_name=None, palette=None, title=None, ax=None, **kwargs):
    """Shared body for `metric_line`/`metric_scatter` -- only the seaborn plotting function differs between them."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    is_numeric = pd.api.types.is_numeric_dtype(df[x])
    if not is_numeric:
        df = df.sort_values(x)

    plot_func(data=df, x=x, y="Value", hue=group_name, palette=palette, ax=ax, **kwargs)

    if not is_numeric:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel(x)
    if title:
        ax.set_title(title)

    return fig, ax


def metric_line(df, x, group_name=None, palette=None, title=None, ax=None, **kwargs):
    """
    Line plot (`sns.lineplot`) of "Value" against an explicit x-axis
    coordinate.

    Plots exactly the rows in `df` (one metric's worth -- filtering to a
    single metric is the caller's responsibility, e.g.
    `MetricSuite.plot.line(metric=..., x=...)`), onto a single `Axes`.

    Parameters
    ----------
    df: pd.DataFrame
        The tidy dataframe.

    x: str
        Coordinate column to use as the x-axis (e.g. "time", "snr").

    group_name: str
        Column for color encoding (e.g. "Label") -- same name/role as
        `metric_box`'s `group_name`, for a consistent API across plot kinds.

    palette: str | list | dict
        Color palette for the `group_name` levels.

    title: str, optional
        Axes title.

    ax: matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if not given.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    return _metric_relational(df, x, sns.lineplot, group_name=group_name, palette=palette, title=title, ax=ax, **kwargs)


def metric_scatter(df, x, group_name=None, palette=None, title=None, ax=None, **kwargs):
    """
    Scatter plot (`sns.scatterplot`) of "Value" against an explicit x-axis
    coordinate.

    See `metric_line` for the shared parameters/return shape.
    """
    return _metric_relational(df, x, sns.scatterplot, group_name=group_name, palette=palette, title=title, ax=ax, **kwargs)
