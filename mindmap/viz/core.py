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

from mindmap.viz.utils import create_line_mesh

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


def metric_box(df, group_name="Method", **kwargs):
    """Default style: Boxplot with overlaid raw data points."""
    defaults = {"palette": "viridis", "linewidth": 1.5}
    params = {**defaults, **kwargs}

    g = sns.FacetGrid(df, col="Metric", sharey=False, col_wrap=3)
    g.map_dataframe(sns.boxplot, x=group_name, y="Value", hue=group_name, **params)
    g.map_dataframe(sns.stripplot, x=group_name, y="Value", color=".2", alpha=0.4)
    g.add_legend()

    return g

def metric_violin(df, group_name="Method", **kwargs):
    """Default style: Split or inner-quartile violin plots."""
    defaults = {"inner": "quartile", "palette": "muted", "cut":0}
    params = {**defaults, **kwargs}

    g = sns.FacetGrid(df, col="Metric", sharey=False, col_wrap=3)
    g.map_dataframe(sns.violinplot, x=group_name, y="Value", hue=group_name, **params)
    g.add_legend()

    return g

def metric_bar(df, group_name="Method", **kwargs):
    """Default style: Bar chart with error bars (confidence intervals)."""
    defaults = {"errorbar":"sd", "capsize": .1, "palette":"magma"}
    params = {**defaults, **kwargs}

    g = sns.FacetGrid(df, col="Metric", sharey=False, col_wrap=3)
    g.map_dataframe(sns.barplot, x=group_name, y="Value", hue=group_name, **params)
    g.add_legend()

    return g

def metric_heatmap(df, group_name="Method", title="", **kwargs):
    """Default style: Clean matrix with centered annotations."""
    defaults = {"cmap": "RdYlGn", "annot": True, "fmt": ".2f", "cbar_kws": {"label": "Average Value"}}
    params = {**defaults, **kwargs}

    pivot = df.pivot_table(index=group_name, columns="Metric", values="Value", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, **params)
    plt.title(f"{title} Overview")

    return plt.gca()


def metric_relational(df, coords, hue=None, kind="line", **kwargs):
    """
    Relational plot that automatically detect the correct X-axis per metric
    
    Parameters
    ----------
    df : pd.DataFrame
        The tidy dataframe.

    coords: list
        List of all potential coordinate columns (e.g., ['time', 'freq'])

    hue : str
        The column for color encoding (e.g. 'Lable')
    """
    def _sub_plotter(data, **inner_kwargs):
        valid_coords = [c for c in coords if data[c].notna().any()]

        if not valid_coords:
            return

        x_col = valid_coords[0] if valid_coords else None

        # check if coordinate is numerical or categorical (strings)?
        is_numeric = pd.api.types.is_numeric_dtype(data[x_col])

        # select plot type
        if not is_numeric:
            data = data.sort_values(x_col)
            plot_func = sns.lineplot if kind == "line" else sns.scatterplot
        else:
            plot_func = sns.lineplot if kind == "line" else sns.scatterplot

        # plot using the detected x_col
        ax = plot_func(data=data, x=x_col, y="Value", **inner_kwargs)

        if not is_numeric:
            plt.xticks(rotation=45, ha="right")

        plt.gca().set_xlabel(x_col)

    # Initialize the Grid
    g = sns.FacetGrid(
        df,
        col = "Metric",
        hue = hue,
        sharex = False,
        sharey = False,
        col_wrap = 3,
    )

    # map the internal plotter to the grid
    g.map_dataframe(_sub_plotter, **kwargs)
    g.add_legend()

    return g
