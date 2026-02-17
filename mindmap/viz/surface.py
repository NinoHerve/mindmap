"""
Module Name: viz/surface.py
Description: Functions to plot surface data.

===============================================================================
SURFACE-SPACE VISUALIZATION (surface.py)
===============================================================================
Targets: np.ndarray, mne.SourceSpaces, 
Backends: pyvista
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

from typing import Optional

import numpy as np

import pyvista as pv

import mne
import nibabel as nib

from mindmap.anatomy.source_space.io import mne_to_polydata, get_dipole_positions
from mindmap.anatomy.mri.io import nib_to_polydata
from mindmap.analysis.functional.eeg.io import get_ch_positions
from .utils import store_in_lists, find_clim, default_cmap, BaseBrainPlotter, build_hemi_pair



def plot_brain(
    source_space   : mne.SourceSpaces | list[mne.SourceSpaces],
    signal         : np.ndarray | list[np.ndarray] = None,
    overlay        : np.ndarray | nib.Nifti1Image | list[np.ndarray] | list[nib.Nifti1Image]     = None,
    tkras2ras      : np.ndarray | list[np.ndarray] = None,
    title          : str | list[str] = "",
    mesh_params    : dict = None,
    signal_params  : dict = None,
    overlay_params : dict = None,
) -> pv.Plotter:
    """
    Plot Brain
    
    Parameters
    ----------
    source_space: mne.SourceSpaces | list[mne.SourceSpaces] 
        Source space.

    signal: mne.SourceEstimate | np.ndarray (N,)
        Source values.

    overlay: np.ndarray (M,3) | nib.Nifti1Image
        Overlay defined as point cloud.

    tkras2ras: np.ndarray
        Affine transform (freesurfer ras -> T1 ras). Necessary to align MRI images with meshes.

    brain_params: dict
        key : value
        key : value
    
    signal_params: dict
        key : value
        key : value
    
    overlay_params: dict
        key : value
        key : value

    Returns
    -------
    brain: pv.Plotter
        Brain plot.
    """
    # -----------------------
    # Handle inputs as lists
    # -----------------------

    source_space, signal, overlay, tkras2ras, title = store_in_lists(
        source_space, signal, overlay, tkras2ras, title
    )

    # ---------------
    # Process signal
    # ---------------

    if signal_params is None:
        signal_params = {}

    if signal_params.get("std"):
        signal = [(s-np.mean(s))/np.std(s) if s is not None else None for s in signal]

    if signal_params.get("absolute"):
        signal = [np.abs(s) if s is not None else None for s in signal]

    clim  = find_clim(signal)
    cmap  = default_cmap(clim)
    color = "white" if cmap is None else None

    # ------------------------
    # Handle plot parameters
    # ------------------------

    if mesh_params is None:
        mesh_params = {}
    if overlay_params is None:
        overlay_params = {}

    # default plot parameters
    mesh_parameters = {
        "smooth_shading" : True,
        "clim"           : clim,
        "cmap"           : cmap,
        "color"          : color,
        "show_src"       : False,
        "show_max"       : False,
    }
    mesh_parameters = mesh_parameters | mesh_params
    show_src = mesh_parameters.pop("show_src")
    show_max = mesh_parameters.pop("show_max")

    overlay_parameters = {
        "color"      : "green",
        "label"      : None,
        "point_size" : 1,
        "opacity"    : 1,
    }
    overlay_parameters = overlay_parameters | overlay_params

    show_src_parameters = {
        "color"                    : "green",
        "point_size"               : 5,
        "render_points_as_spheres" : True,    
    }

    show_max_parameters = {
        "color"      : "yellow",
        "point_size" : 10,
    }

    # -----------------
    # Plot brain mesh
    # -----------------

    n_plots = len(source_space)
    plotter = pv.Plotter(shape=(1,n_plots))

    # iterate through images
    for i in range(n_plots):
        src = source_space[i]
        sig = signal[i]

        # build polydata meshes
        hemi_l, hemi_r = mne_to_polydata(src, sig)

        # plot
        plotter.subplot(0,i)
        plotter.add_mesh(hemi_l, name="lh", **mesh_parameters)
        plotter.add_mesh(hemi_r, name="rh", **mesh_parameters)
        plotter.add_title(title[i])

        # show src
        if show_src:
            # convert source space to polydata
            pos = pv.PolyData(get_dipole_positions(src))

            # plot source space
            plotter.add_mesh(pos, name="src", **show_src_parameters)

        # show max
        if show_max:
            if sig is not None:
                # select proper hemisphere then find node position
                split   = src[0]["nuse"]
                idx_max = np.argmax(np.abs(sig))
                if idx_max < split:
                    map_max = src[0]["rr"][src[0]["vertno"]][idx_max]
                else:
                    map_max = src[1]["rr"][src[1]["vertno"]][idx_max-split]

                # highlight maximum value
                plotter.add_mesh(map_max, **show_max_parameters)

        # overlay
        if overlay[i] is not None:
            # convert overlay to polydata
            if isinstance(overlay[i], nib.Nifti1Image):
                over = nib_to_polydata(overlay[i], trans=np.linalg.inv(tkras2ras[i]))
            else:
                over = overlay[i]

            # plot overlay
            plotter.add_mesh(over, name="overlay", **overlay_parameters)

    return plotter


def plot_source_estimate(
    source_space   : mne.SourceSpaces | list[mne.SourceSpaces],
    timeseries     : mne.SourceEstimate | list[mne.SourceEstimate] = None,
    overlay        : np.ndarray | nib.Nifti1Image | list[np.ndarray] | list[nib.Nifti1Image] = None,
    tkras2ras      : np.ndarray | list[np.ndarray] = None,
    title          : str | list[str] = "",
    mesh_params    : dict = None,
    ts_params      : dict = None,
    overlay_params : dict = None,
) -> pv.Plotter:
    """
    Plot brain with SourceEstimate

    Parameters
    ----------
    source_space: mne.SourceSpaces | list[mne.SourceSpaces] 
        Source space.

    timeseries: mne.SourceEstimate | list[mne.SourceEstimate]
        Source values over time.

    overlay: np.ndarray (M,3) | nib.Nifti1Image
        Overlay defined as point cloud.

    tkras2ras: np.ndarray
        Affine transform (freesurfer ras -> T1 ras). Necessary to align MRI images with meshes.

    mesh_params: dict
        key : value
        key : value
    
    ts_params: dict
        key : value
        key : value
    
    overlay_params: dict
        key : value
        key : value

    Returns
    -------
    brain: pv.Plotter
        Brain plot.
    """

    # -----------------------
    # Normalize inputs
    # -----------------------

    source_space, timeseries, overlay, tkras2ras, title = store_in_lists(
        source_space, timeseries, overlay, tkras2ras, title,
    )

    mesh_params    = mesh_params or {}
    overlay_params = overlay_params or {}
    ts_params      = ts_params or {}

    # work on copies sine MNE modifies in-place
    timeseries = [ts.copy() for ts in timeseries]

    # --------------------
    # Process time series
    # --------------------

    clim  = find_clim(timeseries)

    if ts_params.get("normalize"):
        for ts in timeseries:
            scale = np.nanpercentile(np.abs(ts.data), 99)  # percentile instead of max to resist to outliers
            if scale > 0:
                ts.data /= scale

        clim = (-1, 1)

    if ts_params.get("absolute"):
        for ts in timeseries:
            ts.data = np.abs(ts.data)

        clim = (0, clim[1] if clim else 1)

    cmap  = default_cmap(clim)

    mesh_params = {
        "smooth_shading" : True,
        "clim" : clim,
        "cmap" : cmap,
    } | mesh_params

    # ------------------
    # Process overlays
    # ------------------

    for i, over in enumerate(overlay):
        if isinstance(over, nib.Nifti1Image):
            overlay[i] = nib_to_polydata(over, trans=tkras2ras[i])

    overlay_params = {
        "color"      : "green",
        "label"      : None,
        "point_size" : 1,
        "opacity"    : 1,
    } | overlay_params


    # ----------------
    # Create TimeMesh
    # ----------------

    return TimeMesh(
        source_space,
        timeseries,
        overlay,
        title,
        mesh_params=mesh_params,
        overlay_params=overlay_params,
    )

class TimeMesh(BaseBrainPlotter):
    """
    Interactive visualization of brain source estimates over time using PyVista
    A slider allows scrolling through time points.
    """

    def __init__(
        self,
        source_spaces : list[mne.SourceSpaces],
        timeseries : list[mne.SourceEstimate],
        overlays : Optional[pv.PolyData] = None,
        titles : Optional[list[str]] = None,
        mesh_params : Optional[dict] = None,
        overlay_params : Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        source_spaces: list[mne.SourceSpaces]
            Source spaces.
        
        timeseries: list[mne.SourceEstimate]
            Source estimates (N x T)

        overlays: list[pv.PolyData]
            Optional overlay meshes

        titles: list[str]
            Titles for each subplot.

        mesh_params: dict
            Parameters passed to plotter.add_mesh for brain meshes

        overlay_params: dict
            Parameters passed to plotter.add_mesh for overlays

        start_time: float
            Initial time point. Defaults to the midpoint of the times axis.
        """
        self.src = list(source_spaces)
        self.ts  = list(timeseries)

        if len(self.src) != len(self.ts):
            raise ValueError("source_spaces and timeseries must have the same length.")

        self.n_plots = len(self.src)
        self.times = self.ts[0].times

        self.overlays = overlays or [None] * self.n_plots
        self.titles   = titles or [""] * self.n_plots

        self.mesh_params    = mesh_params or {}
        self.overlay_params = overlay_params or {}

        # build plotter
        self.plotter = pv.Plotter(shape=(1, self.n_plots))
        super().__init__(self.plotter, n_states=len(self.times))

        # prepare meshes and mapping
        self.hemi_meshes = []
        self.vertex_maps = []
        self.splits      = []

        for i in range(self.n_plots):
            hemi_l, hemi_r, map_l, map_r, nuse_l = build_hemi_pair(self.src[i])

            self.hemi_meshes.append((hemi_l, hemi_r))
            self.vertex_maps.append((map_l, map_r))
            self.splits.append(nuse_l)

            self.plotter.subplot(0,i)
            self.plotter.add_mesh(hemi_l, **self.mesh_params)
            self.plotter.add_mesh(hemi_r, **self.mesh_params)
            self.plotter.add_title(self.titles[i])

            if self.overlays[i] is not None:
                self.plotter.add_mesh(self.overlays[i], **self.overlay_params)

        # add widgets and events
        self.add_slider(label = "Time [s]", rng=[self.times[0], self.times[-1]])
        self.setup_keys()

        # render first time point
        self.update_display(self.current_idx)


    # -------------------
    # Inherited methods
    # -------------------

    def value_to_idx(self, value):
        """Convert value to index"""
        return int(self.ts[0].time_as_index(value)[0])

    def idx_to_value(self, idx):
        """Convert index to value"""
        return self.times[idx]

    def update_display(self, idx):
        """
        Update all meshes for a given time index
        """
        for i in range(self.n_plots):
            # extract data
            stc = self.ts[i]
            hemi_l, hemi_r = self.hemi_meshes[i]
            map_l, map_r   = self.vertex_maps[i]
            split = self.splits[i]

            # extract signal
            signal = stc.data[:, idx]
            sig_l  = signal[:split].squeeze()
            sig_r  = signal[split:].squeeze()

            # append 0 for vertices mapped to -1
            sig_l = np.concatenate([sig_l, [0.0]])
            sig_r = np.concatenate([sig_r, [0.0]])

            # apply new signal
            hemi_l["signal"][:] = sig_l[map_l]
            hemi_r["signal"][:] = sig_r[map_r]

        self.plotter.render()



def plot_leadfield(eeg, forward):
    """Plot to interactively look at leadfields"""
    # get electrode positions
    points = get_ch_positions(eeg)
    cap    = pv.PolyData(points)
    cap["highlight"] = np.zeros(len(points))

    # get leadfield matrix
    leadfield = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, use_cps=True,
    )
    leadfield = leadfield["sol"]["data"]

    mesh_params = {
        "clim"  : find_clim(leadfield),
        "cmap"  : "bwr",
        "color" : None,
    }

    # plot
    plotter = plot_brain(forward["src"], np.zeros(leadfield.shape[1]), mesh_params=mesh_params)
    plotter.add_mesh(cap, point_size=10, render_points_as_spheres=True, name="cap")
    plotter.remove_scalar_bar(title="highlight")

    # callback
    def pick_electrode(picked_point):
        if picked_point is not None:
            # get index of the closest point
            closest_point_id = cap.find_closest_point(picked_point)

            # change highlighted electrode
            cap["highlight"] = np.zeros(len(points))
            cap["highlight"][closest_point_id] = 1

            # change leadfield vector
            tmp = plot_brain(forward["src"], leadfield[closest_point_id], mesh_params=mesh_params)
            plotter.meshes[0]["signal"] = tmp.meshes[0]["signal"]
            plotter.meshes[1]["signal"] = tmp.meshes[1]["signal"]

            # update the plotter
            plotter.render()

    plotter.enable_point_picking(callback=pick_electrode)

    return plotter


def plot_dipole_function(source_space, dipole_function, title="Dipole function"):
    """
    Plots a function that depends on dipole position (e.g. point-spread function, cross-talk function, ...)

    Parameters
    ----------
    source_space: mne.SourceSpaces 
        Dipole space.

    dipole_function: np.ndarray (N,N)
        Dipole function. Row i corresponds to function output of dipole i.

    Returns
    -------
    Pyvista object
    """

    plotter = pv.Plotter()

    # style
    cmap = "bwr"
    smooth_shading = True
    max_abs = np.max(np.abs(dipole_function))
    clim = [-max_abs, max_abs]
    size = 5

    # brain mesh
    hemi_l, hemi_r = mne_to_polydata(source_space, dipole_function[0])

    plotter.add_mesh(hemi_l, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="lh")
    plotter.add_mesh(hemi_r, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="rh")
    plotter.add_title(title)

    # dipoles
    dipoles                 = pv.PolyData(get_dipole_positions(source_space))
    dipoles["selection"]    = np.zeros(dipoles.n_cells)
    dipoles["selection"][0] = 1

    dipoles_actor = plotter.add_mesh(dipoles, render_points_as_spheres=True, point_size=size)
    plotter.remove_scalar_bar(title="selection")

    # dipole visibility checkbox
    def toggle_vis(flag) -> None:
        dipoles_actor.SetVisibility(flag)

    plotter.add_checkbox_button_widget(toggle_vis, value=True)

    # dipole picking
    def pick_dipole(picked_point):
        if picked_point is not None:
            # get index of closest point
            closest_point_id = dipoles.find_closest_point(picked_point)

            # change selected dipole
            dipoles["selection"] = np.zeros(dipoles.n_cells)
            dipoles["selection"][closest_point_id] = 1

            # change corresponding function output
            tmp_hemi_l, tmp_hemi_r = mne_to_polydata(source_space, dipole_function[closest_point_id])
            plotter.meshes[0]["signal"] = tmp_hemi_l["signal"]
            plotter.meshes[1]["signal"] = tmp_hemi_r["signal"]

            plotter.render()

    plotter.enable_point_picking(callback=pick_dipole)

    return plotter
