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

from mindmap.analysis.functional.eeg.io import get_ch_positions
from mindmap.geometry import surf_to_polydata
from .utils import BaseBrainPlotter, build_hemi_pair

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


def plot_ieeg(src, ieeg, subject="unknown"):
    """
    Plot iEEG timeseries.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    ieeg: mne.Evoked
        Intracranial EEGs.

    mesh_params: dict
        Mesh parameters.

    pc_params:
        Point cloud parameters.

    subject:
        Subject id.
    """

    # -------------------
    # Prepare parameters
    # -------------------

    mesh_params = {
        "smooth_shading" : True,
        "color" : "skyblue",
        "opacity" : 0.1,
    }

    pc_params = {
        "point_size" : 10,
        "render_points_as_spheres" : True,
    }

    return TimePC(
        source_space = src,
        evoked = ieeg,
        title = f"Intracranial EEG sub-{subject}",
        scalar = "Electric potential [V]",
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
        self.pc = pv.PolyData(get_ch_positions(self.evk))
        self.pc[self.scalar] = self.evk.data[:,0]
        self.plotter.add_mesh(self.pc, scalars=self.scalar, **self.pc_params)

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


def plot_gfp_comparison(evoked_ref, dict_of_evoked_est):
    """
    Plot global field power (gfp) of the true iEEG against each 
    reconstruction method.
    """
    fig, ax = plt.subplots(figsize=(10,4))

    # plot true iEEG GFP
    ax.plot(evoked_ref.times, evoked_ref.data.std(axis=0), color="black", linewidth=2, label="True iEEG")

    # plot each estimation method
    for name, evoked_est in dict_of_evoked_est.items():
        ax.plot(evoked_est.times, evoked_est.data.std(axis=0),
                label=f"Virtual: {name}", linestyle="--")

    ax.set_title("Global Field Power Comparison")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Spatial Standard Deviation")
    ax.legend()

    return fig
