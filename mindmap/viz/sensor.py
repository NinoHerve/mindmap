"""
Module Name: viz/sensor.py
Description: Functions to plot sensor properties.

===============================================================================
SENSOR-SPACE VISUALIZATION (sensor.py)
===============================================================================
Targets: mne.Evoked, mne.Epochs
Backends: Matplotlib (2D grid), pyvista (3D point clouds)
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objs as go
import plotly.express as px
import pyvista as pv


import mne

from mindmap.core.metric import get_gfp
from mindmap.core.process import normalize_timeserie_by_baseline
from mindmap.analysis.functional.eeg.io import get_ch_positions
from mindmap.analysis.functional.eeg.process import normalize_evoked
from mindmap.analysis.functional.eeg.metric import (
    compute_evoked_cog, compute_evoked_rem,
)
from mindmap.geometry import surf_to_polydata
from mindmap.viz.utils import BaseBrainPlotter, build_hemi_pair, find_clim, store_in_lists
from mindmap.viz.core import (
    plot_multi_scatterplots, plot_cog_trajectories, plot_rem,
)

def plot_alignment(bem, ave):
    """
    Plot alignment between the head and sensors.
    """
    # outer skin surface
    head = bem["surfs"][0]
    head = surf_to_polydata(head["rr"], head["tris"])

    # sensor positions
    sens = get_ch_positions(ave)
    sens = pv.PolyData(sens)

    # plot
    plotter = pv.Plotter()
    plotter.add_mesh(head, label="head", smooth_shading=True, color="#D2A18C")
    plotter.add_mesh(sens, render_points_as_spheres=True, color="black", label="sensors", point_size=8)

    plotter.add_legend(face="circle")

    return plotter

def plot_ieeg_static(
    src: mne.SourceSpaces,
    ieeg: mne.Evoked,
    signals: np.ndarray | list[np.ndarray],
    titles: str | list[str] = "",
    mesh_params: dict = None,
    pc_params: dict = None,
) -> pv.Plotter:
    """
    Static multi-window plot for iEEG channel metrics (e.g., scores)
    """
    # Handle inputs as lists for multi-window support
    signals, titles = store_in_lists(signals, titles)

    n_plots = len(signals)
    plotter = pv.Plotter(shape=(1, n_plots))

    # Setup default parameters
    mesh_parameters = {
        "smooth_shading": True,
        "color": "skyblue",
        "opacity": 0.1,
    } | (mesh_params or {})

    # Use a fixed clim [0, 1] for scores, or dynamic for raw potential
    clim = find_clim(np.array(signals))

    pc_parameters = {
        "point_size": 12,
        "render_points_as_spheres": True,
        "cmap": "viridis",
        "clim": clim,
    } | (pc_params or {})

    # Build brain geometry
    hemi_l, hemi_r, _, __, ___ = build_hemi_pair(src)
    ch_pos = get_ch_positions(ieeg)

    # Iterate through subplots
    for i in range(n_plots):
        plotter.subplot(0, i)

        # Add Brain Meshes
        plotter.add_mesh(hemi_l, name=f"lh_{i}", **mesh_parameters)
        plotter.add_mesh(hemi_r, name=f"rh_{i}", **mesh_parameters)

        # Add title
        if i < len(titles):
            plotter.add_title(titles[i], font_size=10)

        # Create Point Cloud for this signal
        sig = signals[i]
        pc = pv.PolyData(ch_pos)
        pc["metric"] = sig

        plotter.add_mesh(pc, scalars="metric", **pc_parameters)

    return plotter



def plot_ieeg(
    src : mne.SourceSpaces,
    ieeg : mne.Evoked,
    extent_mask : np.ndarray = None,
    spurious_mask : np.ndarray = None,
    ts_params: dict = None,
    subject : str = "unknown"
):
    """
    Plot iEEG timeseries.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    ieeg: mne.Evoked
        Intracranial EEGs.

    extent_mask: np.ndarray of shape (n_channels,)
        Mask for channels in spike extent area.

    spurious_mask: np.ndarray of shape (n_channels,)
        Mask for channels in spurious area.

    subject:
        Subject id.
    """
    ts_params = ts_params or {}

    # work on copies sine MNE modifies in-place
    timeseries = ieeg.copy()

    # --------------------
    # Process time series
    # --------------------

    if norm := ts_params.get("normalize"):
        timeseries = normalize_evoked(timeseries, "global", norm)

    if ts_params.get("absolute"):
        timeseries.data = np.abs(timeseries.data)

    if np.all(timeseries.data > 0):
        cmap = "kr"
    else:
        cmap = "bkr"

    # ------------------------
    # Prepare plot parameters
    # ------------------------

    mesh_params = {
        "smooth_shading" : True,
        "color" : "skyblue",
        "opacity" : 0.1,
    }

    pc_params = {
        "point_size" : 10,
        "render_points_as_spheres" : True,
        "cmap" : cmap,
    }

    return TimePC(
        source_space = src,
        evoked = timeseries,
        title = f"Intracranial EEG sub-{subject}",
        scalar = "Electric potential [V]",
        extent_mask = extent_mask,
        spurious_mask = spurious_mask,
        mesh_params = mesh_params,
        pc_params = pc_params,
    )


class TimePC(BaseBrainPlotter):
    """
    Interactive visualization of point cloud over time.
    """
    def __init__(
        self,
        source_space : mne.SourceSpaces,
        evoked : mne.Evoked,
        title : Optional[str] = "Channel activity",
        scalar : str = "activity",
        mesh_params : Optional[dict] = None,
        pc_params : Optional[dict] = None,
        extent_mask : Optional[np.ndarray] = None,
        spurious_mask : Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        source_spaces: mne.SourceSpaces
            Source spaces.
        
        timeseries: list[mne.Evoked]
            Timeseries defined at M points (M x T)

        title: str
            Title for plot.

        scalar: str
            Name for the scalar values.

        mesh_params: dict
            Parameters passed to plotter.add_mesh for brain meshes

        pc_params: dict
            Parameters passed to plotter.add_mesh for point cloud

        start_time: float
            Initial time point. Defaults to the midpoint of the times axis.

        extent_time: np.ndarray of shape (M,)
            Mask corresponding to channels belonging to spike extent.

        spurious_mask: np.ndarray of shape (M,)
            Mask corresponding to channels belonging to spurious channels.
        """
        super().__init__(n_states=len(evoked.times))

        self.src = source_space
        self.evk = evoked
        self.title = title
        self.scalar = scalar

        self.times = self.evk.times

        self.extent_mask = extent_mask
        self.spurious_mask = spurious_mask

        self.mesh_params = mesh_params or {"color": "skyblue", "opacity": 0.1}
        self.pc_params = pc_params or {"point_size": 10.0, "render_points_as_spheres": True}

        # build plotter
        self.plotter.add_title(self.title, font_size=10)

        # build brain meshes
        self.hemi_l, self.hemi_r, _, __, ___ = build_hemi_pair(self.src)
        self.plotter.add_mesh(self.hemi_l, **self.mesh_params)
        self.plotter.add_mesh(self.hemi_r, **self.mesh_params)

        # build point cloud
        clim = find_clim(self.evk.data)
        self.pc = pv.PolyData(get_ch_positions(self.evk))
        self.pc[self.scalar] = self.evk.data[:,0]
        self.plotter.add_mesh(self.pc, scalars=self.scalar, clim=clim, **self.pc_params)

        # add widgets and events
        self.setup_keys()
        self.add_slider(label = "Time [s]", rng = [self.times[0], self.times[-1]])
        self.setup_masks()

        # render first time point
        self.update_display(self.current_idx)


    def setup_masks(self):
        """Adds checkbox to visualize masks if exists."""
        if self.extent_mask is not None:
            # create slightly larger "halo" for the extent
            extent = self.pc.points[self.extent_mask]
            extent_poly = pv.PolyData(extent)
            self.extent_actor = self.plotter.add_mesh(
                extent_poly, color="red", point_size=self.pc_params["point_size"]+5,
                render_points_as_spheres = True, name="extent_h", opacity=0.6,
            )
            self.extent_actor.SetVisibility(False)  # Default hidden

            # add checkbox widget
            self.plotter.add_checkbox_button_widget(
                lambda val: self.extent_actor.SetVisibility(val),
                value=False, position=(10, 70), size=25, border_size=2, color_on="red",
            )
            self.plotter.add_text("Show Set H(extent)", position=(40, 75), font_size=8)

        if self.spurious_mask is not None:
            spurious = self.pc.points[self.spurious_mask]
            spurious_poly = pv.PolyData(spurious)
            self.spurious_actor = self.plotter.add_mesh(
                spurious_poly, color="yellow", point_size=self.pc_params["point_size"]+3,
                render_points_as_spheres=True, name="spurious_h", opacity=0.6,
            )
            self.spurious_actor.SetVisibility(False)

            # add checkbox widget
            self.plotter.add_checkbox_button_widget(
                lambda val: self.spurious_actor.SetVisibility(val),
                value=False, position=(10,30), size=25, border_size=2, color_on="yellow",
            )
            self.plotter.add_text("Show spurious mask", position=(40, 35), font_size=8)


    # -------------------
    # Inherited methods
    # -------------------

    def value_to_idx(self, value):
        """Convert value to index"""
        return int(self.evk.time_as_index(value)[0])

    def idx_to_value(self, idx):
        """Convert index to value"""
        return self.times[idx]

    def update_display(self, idx):
        """Update image"""
        self.pc[self.scalar] = self.evk.data[:, idx]
        self.plotter.render()


def plot_eeg(eeg: mne.Evoked, colorscale="Turbo", title="EEG"):
    """
    Plot EEGs using plotly (for dashboard)
    
    Parameters
    ----------
    eeg: mne.Evoked 
        EEG 
    
    colorscale: str
        Plotly colorscale to use.

    Returns
    -------
    plotly lineplot
    """
    # data
    data = eeg.data
    data = {ch:data[i,:] for i,ch in enumerate(eeg.ch_names)}

    times = eeg.times

    n_colors = len(eeg.ch_names)
    colors   = px.colors.sample_colorscale(colorscale, [n/(n_colors-1) for n in range(n_colors)])

    # figure
    fig = go.Figure()
    for ch_name, time_serie in data.items():
        fig.add_trace(go.Scatter(
            x    = times,
            y    = time_serie,
            name = ch_name,
            line = {
                "color": colors.pop(),
                "width": 1,
            }
        ))

    # layout
    fig.update_layout(
        plot_bgcolor = "rgb(34, 34, 34)",
        paper_bgcolor="rgb(34, 34, 34)",
        showlegend   = False,
        title        = {"text":title, "font":{"size": 20, "color":"white"}},
        xaxis        = {
            "linecolor":"green",
            "linewidth": 2,
            "mirror": True,
            "gridcolor": "green",
            "gridwidth": 1,
            "griddash":"dash",
            "tickfont": {"color":"green"}
        },
        yaxis        = {
            "linecolor":"green",
            "linewidth": 2,
            "mirror": True,
            "gridcolor": "green",
            "gridwidth": 1,
            "type": "-",
            "griddash":"dash",
            "tickfont": {"color":"green"},        
        },
    )

    return fig



def plot_channel_grid(
    evoked_ref: mne.Evoked,
    dict_of_evoked_est: dict,
    extent_mask: np.ndarray = None,
    spurious_mask: np.ndarray = None,
    n_visible: int = 4,
    sharey: bool = True,
):
    """ Plot channels """
    print("WARNING: widgets only work in vscode if '%matplotlib widget' is called")

    ch_names = evoked_ref.ch_names
    n_total = len(ch_names)
    times = evoked_ref.times

    # 0. Pre-calculate global Y limits for shared comparison
    if sharey:
        all_data = [evoked_ref.data] + [v.data for v in dict_of_evoked_est.values()]
        y_min = min(d.min() for d in all_data)
        y_max = max(d.max() for d in all_data)

        # add 5% padding
        padding = (y_max - y_min) * 0.05
        global_ylim = (y_min - padding, y_max + padding)

    # 1. Create Figure and main axes
    # We create n_visible subplots
    fig, axes = plt.subplots(n_visible, 1, figsize=(10,8), sharex=True)
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, hspace=0.3)

    # 2. Add the Vertical Slider (Scrollbar)
    # Position: [left, bottom, width, height]
    ax_slider = plt.axes([0.9, 0.1, 0.03, 0.8])
    slider = Slider(
        ax = ax_slider,
        label = "Channels",
        valmin = 0,
        valmax = n_total - n_visible,
        valinit = 0,
        valstep = 1,
        orientation = "vertical",
    )

    def update(_):
        start_idx = int(slider.val)
        # Flip the index logic so slider 'up' feels like scrolling up
        # or keep it simple: 0 is the top of the list.
        current_indices = range(start_idx, start_idx + n_visible)

        for i, ch_idx in enumerate(current_indices):
            ax = axes[i]
            ax.clear()

            if ch_idx < n_total:
                # Apply shared Y limit
                if sharey:
                    ax.set_ylim(global_ylim)

                name = ch_names[ch_idx]

                # ------------------------------
                # Mask highlighting
                # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
                # Set background based on mask
                # ------------------------------
                if extent_mask is not None and extent_mask[ch_idx]:
                    ax.set_facecolor("#ffcccc")     # light red for set H
                elif spurious_mask is not None and spurious_mask[ch_idx]:
                    ax.set_facecolor("#ffffcc")     # light yellow
                else:
                    ax.set_facecolor("white")

                # ---------------
                # Plotting Data
                # ¨¨¨¨¨¨¨¨¨¨¨¨¨
                # --------------
                # plot reference
                ax.plot(times, evoked_ref.data[ch_idx], color="black", label="Ref", lw=1.5)

                # plot estimation
                for est_name, evk_est in dict_of_evoked_est.items():
                    ax.plot(times, evk_est.data[ch_idx], ls="--", label=est_name)

                ax.set_ylabel(name, rotation=0, labelpad=25, fontweight="bold")
                if i == 0:
                    ax.legend(loc="upper right", fontsize="small")
            else:
                ax.axis("off")  # hide axes if we run out of channels

        fig.canvas.draw_idle()

    # Initial call to populate the plot
    slider.on_changed(update)
    update(0)

    plt.show()
    return slider   # keep reference to prevent garbage collection


def plot_gfp_comparison(evoked_ref, dict_of_evoked_est, norm_baseline=None):
    """
    Plot global field power (gfp) of the true iEEG against each 
    reconstruction method.
    """
    fig, ax = plt.subplots(figsize=(10,4))

    if norm_baseline is not None:
        normalize = True
        t_start, t_end = evoked_ref.time_as_index(norm_baseline)
    else:
        normalize = False
        t_start, t_end = None, None

    ref_gfp = get_gfp(evoked_ref.data)
    if normalize:
        ref_gfp = normalize_timeserie_by_baseline(ref_gfp, baseline=(t_start, t_end))

    # plot true iEEG GFP
    ax.plot(evoked_ref.times, ref_gfp, color="black", linewidth=2, label="True iEEG")

    # plot each estimation method
    for name, evoked_est in dict_of_evoked_est.items():
        gfp = get_gfp(evoked_est.data)
        if normalize:
            gfp = normalize_timeserie_by_baseline(gfp, baseline=(t_start, t_end))

        ax.plot(evoked_est.times, gfp,
                label=f"Virtual: {name}", linestyle="--")

    ax.set_title("Global Field Power Comparison")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Spatial Standard Deviation")
    ax.legend()

    return fig


def plot_avg_ref(evoked, title="Average Reference Check", apply_proj = True):
    """
    Plots the spatial mean over time.
    For average-referenced data, this should be a flat line at 0 V.
    """
    # 1. Copy and apply projectors
    ev_check = evoked.copy()
    if apply_proj:
        ev_check = ev_check.apply_proj()

    # 2. Extract data (only EEG to avoid mixing units with EOG/Stim)
    ev_check.pick("eeg")
    data = ev_check.get_data()  # shape (n_channels, n_time)
    times = ev_check.times

    # 2. Calculate the mean across channels for every time point
    spatial_mean = data.mean(axis=0)
    max_dev = np.max(np.abs(spatial_mean))

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, spatial_mean, color="crimson", label="Residual Spatial Mean")
    ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{title}\nMax Deviation after Proj: {max_dev:.2e} V")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average Amplitude (V)")

    limit = max(max_dev * 2, 1e-15)
    ax.set_ylim(-limit, limit)

    plt.close("all")  # Prevent Jupyter from auto-rendering
    return fig


def plot_baseline_topo(evoked, tmin, tmax):
    """
    Visualizes the spatial distribution of the basline period.
    Values should be near zero and spatially unorganized.
    """
    # We use .copy() to avoid cropping the original evoked object
    baseline_evoked = evoked.copy().crop(tmin=tmin, tmax=tmax)

    fig = baseline_evoked.plot_topomap(
        times = (tmin + tmax)/2,
        average = tmax-tmin,
        size = 3,
        colorbar = True,
        units = "V",
        show = "False",
        time_unit = "s",
        proj = True,
        extrapolate="local",
    )
    
    fig.suptitle(f"Baseline Topo (Avg: {tmin}s to {tmax}s)")
    plt.close("all")  # Prevent Jupyter from auto-rendering
    return fig



def plot_interpolation(raw_before, raw_after, bads):
    """Compare PSD of bad channels before and after interpolation."""
    fig, ax = plt.subplots(2, 1, figsize=(10,8))

    raw_before.plot_psd(picks=bads, ax=ax[0], show=False)
    ax[0].set_title("Before Interpolation (Bads)")

    raw_after.plot_psd(picks=bads, ax=ax[1], show=False)
    ax[1].set_title("After Interpolation (Bads)")
    fig.tight_layout()

    return fig


# ------------
# MNE reports
# ------------

def create_pipeline_report(evoked, noise_cov, subject_id, fname=None):
    """MNE report for EEG processing."""
    # Initialize the report
    report = mne.Report(title=f"Processing Report: {subject_id}")

    # Use a copy so we don't modify the original 'evoked' object permanently
    evoked_clean = evoked.copy().apply_proj()

    # ------
    # plots
    # ------

    # 1. Evoked Time series
    report.add_evokeds(
        evokeds = evoked_clean,
        noise_cov = noise_cov,
        titles = "Evoked potentials",
        topomap_kwargs = {"extrapolate":"local"},
        #caption = "Butterly plot of all channels after cleaning and projection.",
        #section = "Preprocessing",
    )

    # 2. Custom average reference plot
    fig_ref = plot_avg_ref(evoked)
    report.add_figure(
        fig = fig_ref,
        title = "Average Reference Validation",
        caption = "The spatial mean across EEG channels after applying SSP projectors.",
        section = "Preprocessing",
    )


    # Save to HTML
    if fname is None:
        fname = f"Report_{subject_id}.html"
    report.save(fname, overwrite=True, open_browser=False)
    plt.close("all")  # Essential if running in a loop to save RAM
    print(f"Report saved for {subject_id}")



def plot_whitened_cov(noise_cov, whitener):
    """
    Plot whitened covariance matrix. The whitened covariance matrix should
    be an identity matrix.
    """
    # compute whitened covariance
    noise_cov = noise_cov.data
    whitened_cov = whitener @ noise_cov @ whitener.T

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate scale for consistent visual comparison
    # For whitened data, we expect values near 1 on diag and 0 off-diag
    v_max = np.max(np.abs(whitened_cov))

    # Plot 1: Original Covariance
    im0 = ax[0].imshow(noise_cov, cmap="RdBu_r", interpolation='nearest')
    ax[0].set_title("Original Covariance ($C$)")
    plt.colorbar(im0, ax=ax[0])

    # Plot 2: Whitened Covariance
    # Using vmin/vmax ensures 0 is always the neutral center color
    im1 = ax[1].imshow(whitened_cov, cmap="RdBu_r", vmin=-v_max, vmax=v_max)
    ax[1].set_title("Whitened Covariance ($W C W^T$)")
    plt.colorbar(im1, ax=ax[1])

    # Calculate metrics
    diag_vals = np.diag(whitened_cov)
    avg_diag = np.mean(diag_vals[diag_vals > 1e-10])

    # Add text to the plot instead of printing
    stats_text = f"Mean of Diagonal: {avg_diag:.4f}\n(Target: 1.0)"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for the text at the bottom
    plt.show()



def plot_gfp(evoked, whitener=None, title="Global Field Power"):
    """
    Plot Global Field Power.
    If whitener is provided, plot before and after whitening.
    """
    fig, ax1 = plt.subplots(figsize=(12,5))

    # GFP - Standard deviation across channels (Volts)
    gfp_raw = np.std(evoked.data, axis=0)

    # Plotting axis 1 (Raw)
    color_raw = "tab:gray"
    ax1.plot(evoked.times, gfp_raw, color=color_raw, alpha=0.5, label="raw GFP")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GFP (V)", color=color_raw)
    ax1.tick_params(axis="y", labelcolor=color_raw)
    ax1.set_ylim(bottom=0)

    # Add whitener
    if whitener is not None:
        # Calculate Whitened & Scaled GFP
        data_white = (whitener @ evoked.data) * np.sqrt(evoked.nave)
        gfp_white = np.std(data_white, axis=0)

        # Plotting axis 2 (Whitened)
        ax2 = ax1.twinx()
        color_white = "tab:blue"
        ax2.plot(evoked.times, gfp_white, color=color_white, linewidth=2, label="Whitened GFP")
        ax2.set_ylabel("Whitened GFP (Scaled by $\sqrt{N}$)", color=color_white)
        ax2.tick_params(axis="y", labelcolor=color_white)

        # Add the Ideal Noise Floor line
        ax2.axhline(1.0, color="crimson", linestyle="--", alpha=0.7, label="Expected Noise Floor")
        ax2.axhline(0.0, color="black", linewidth=0.8)
        ax2.set_ylim(bottom=0)

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Textual confirmation
        baseline_val = np.mean(gfp_white[evoked.times < 0]) if any(evoked.times < 0) else gfp_white[0]
        print(f"Whitened Baseline GFP: {baseline_val:.4f} (Should be ~1.0)")

    plt.title(title)
    fig.tight_layout()

    plt.close("all")

    return fig





def plot_evoked_scatterplots(
    evoked_ref: mne.Evoked,
    other_evokeds: dict[str, mne.Evoked],
    compare_by: str = "space",
) -> go.Figure:
    """
    Plots scatter plots.

    Parameters
    ----------
    evoked_ref: mne.Evoked

    other_evokeds: dict[str,mne.Evoked]
        Dictionary of evokeds to compare to. Key will be
        used as labels for the plot.

    compare_by: str
        - compare_by='space': Dots are channels, Slider moves through time.
        - compare_by='time': Dots are timepoints, Slider moves through channels.

    Returns
    -------
    fig: go.Figure
    """
    # extract raw numpy
    ref_matrix = evoked_ref.data
    cmp_matrices = {name: evk.data for name, evk in other_evokeds.items()}

    if compare_by == "time":
        # transpose to (Time x Channels)
        ref_matrix = ref_matrix.T
        cmp_matrices = {k: v.T for k, v in cmp_matrices.times()}

        point_labels = [f"{t*1000:1f} ms" for t in evoked_ref.times]
        frame_labels = [f"Channel: {ch}" for ch in evoked_ref.ch_names]
        slider_name = "Channel"
    else:
        # keep (Channels x Time)
        point_labels = evoked_ref.ch_names
        frame_labels = [f"Time: {t*1000:.1f} ms" for t in evoked_ref.times]
        slider_name = "Time"

    return plot_multi_scatterplots(
        data_ref=ref_matrix,
        other_datasets=cmp_matrices,
        point_labels=point_labels,
        frame_labels=frame_labels,
        slider_name=slider_name,
    )


def plot_evoked_cog(
    dict_of_evk: dict[str:mne.Evoked],
    ref_name: str = None,
    rng: tuple = (None, None),
    source_space: mne.SourceSpaces = None,
    center_ref: bool = None,
    colors = None,
    units: str = "mm",
):
    """
    Plot center of gravity for evoked objects.
    """
    # colors
    if colors is None:
        colors = ["firebrick", "dodgerblue", "goldenrod", "darkorchid"]

    # units
    scale = 1000.0 if units == "mm" else 1.0

    # center of gravities
    dict_of_traj = {}
    for k, evk in dict_of_evk.items():
        evk_crop = evk.copy().crop(tmin=rng[0], tmax=rng[1])
        dict_of_traj[k] = compute_evoked_cog(evk_crop) * scale

    # electrode positions
    pos = get_ch_positions(evk_crop) * scale

    # times
    times = evk_crop.times

    # brain mesh
    brain_mesh = None
    if source_space is not None:
        hemi_l = surf_to_polydata(source_space[0]["rr"]*scale, source_space[0]["tris"])
        hemi_r = surf_to_polydata(source_space[1]["rr"]*scale, source_space[1]["tris"])

        brain_mesh = hemi_l + hemi_r

    return plot_cog_trajectories(
        dict_of_traj,
        times = times,
        ref_name = ref_name,
        electrode_pos = pos,
        brain_mesh = brain_mesh,
        center_ref = center_ref,
        colors = colors,
        units = units,
    )


def plot_evoked_rem(
    evoked_ref: mne.Evoked,
    evoked_est: mne.Evoked,
    mode: str = "spatial",
    rng: tuple[float, float] = (None, None),
):
    """
    Plots a heatmap of the Relative Energy Map.
    """
    # comptue REM
    rem_map = compute_evoked_rem(evoked_ref, evoked_est, mode=mode, rng=rng)

    # extent
    times = evoked_ref.times
    mask = (times >= (rng[0] or times[0])) & (times <= (rng[1] or times[-1]))
    crop_times = times[mask]

    extent = [crop_times[0], crop_times[-1], 0, rem_map.shape[0]]

    fig, ax = plot_rem(
        rem_map,
        extent = extent,
        title = f"Relative Energy Map - {mode.capitalize()} Normalization",
        label = f"Relative Energy ({mode} normalized)",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Channels (Index)")

    return fig, ax

