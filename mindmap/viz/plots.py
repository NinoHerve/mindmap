"""
Module Name: plots.py
Description: Functions to plot data.

This module provides functions to plot data.

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.fft import ifftn
import nibabel as nib
import mne

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import pyvista as pv

from mindmap.head import mne_to_polydata, nib_to_polydata, get_dipole_positions, surf_to_polydata
from mindmap.sensor import get_ch_positions

#####################################
##          Plot functions         ##
##                                 ##
##                                 ##
##                                 ##
#####################################

def show_slices(
        volume        : np.ndarray | nib.Nifti1Image,
        slice_indices : list[int] = None,
        axis          : int = 2,
        cmap          : str = "gray",
) -> None:
    """
    Display slices from a volume.

    Parameters
    ----------
    volume: nib.Nifti1Image
        Volume.

    slice_indices: list[int]
        List of slice indices to display. If None, defaults to the middle slice.

    axis: int
        Axis along which to slice (0,1 or 2).

    cmap: str
        Color map.
    """
    if isinstance(volume, nib.Nifti1Image):
        volume = volume.get_fdata()

    if slice_indices is None:
        slice_indices = [volume.shape[axis]//2] # default

    _, axes = plt.subplots(1, len(slice_indices), figsize=(15,5))

    if len(slice_indices) == 1:
        axes = [axes]

    for i, slice_idx in enumerate(slice_indices):
        if axis == 0:
            slice_vol = volume[slice_idx, :, :]
        elif axis == 1:
            slice_vol = volume[:, slice_idx, :]
        else:
            slice_vol = volume[:, :, slice_idx]

        if len(slice_vol) > 2:  # RGB image
            axes[i].imshow(slice_vol, cmap=cmap, origin="lower")
            axes[i].set_title(f"Slice {slice_idx} (axis {axis})")
            axes[i].axis("off")
        else:
            axes[i].imshow(slice_vol.T, cmap=cmap, origin="lower")
            axes[i].set_title(f"Slice {slice_idx} (axis {axis})")
            axes[i].axis("off")

    plt.show()

def plot_segmentation(
        t1_file  : str | Path,
        seg_file : str | Path,
        lut_file : str | Path = None
        ):
    """
    Opens FreeView to plot structural MRI with segmentation on top.
    """
    cmd = ["freeview", "-v", str(t1_file)]

    if lut_file is None:
        cmd += [str(seg_file)]
    else:
        cmd += [str(f"{seg_file}:lut={lut_file}")]

    subprocess.run(cmd, check=True)


def plot_bem(
    fs_dir  : str | Path,
    subject : str,
):
    """
    Plot boundary element surfaces.

    Parameters
    ----------
    fs_dir: str | Path
        Path to FreeSurfer directory.
    
    subject: str
        Subject id.
    """
    fs_dir = Path(fs_dir)
    sub_id = f"sub-{subject}"

    mri       = fs_dir/sub_id/"mri"/"T1.mgz"
    white_lh  = fs_dir/sub_id/"surf"/"lh.white"
    white_rh  = fs_dir/sub_id/"surf"/"rh.white"
    in_skull  = fs_dir/sub_id/"bem"/"inner_skull.surf"
    out_skull = fs_dir/sub_id/"bem"/"outer_skull.surf"
    out_skin  = fs_dir/sub_id/"bem"/"outer_skin.surf"

    cmd = [
        "freeview",
        "-v", str(mri),
        "-f",
        str(white_lh) + ":edgecolor=white",
        str(white_rh) + ":edgecolor=white",
        str(in_skull) + ":edgecolor=red",
        str(out_skull) + ":edgecolor=green",
        str(out_skin) + ":edgecolor=blue",
    ]

    subprocess.run(cmd, check=True)


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


# def plot_brain(
#     sources         : mne.SourceSpaces | list[mne.SourceSpaces],
#     signal          : np.ndarray | mne.SourceEstimate | list[np.ndarray] | list[mne.SourceSpaces] = None,
#     overlay         : np.ndarray | nib.Nifti1Image | list[np.ndarray] | list[nib.Nifti1Image]     = None,
#     tkras2ras       : np.ndarray | list[np.ndarray] = np.zeros(3),
#     show_max        : bool  = False,
#     show_src        : bool  = False,
#     clim            : list  = None,
#     cmap            : str   = None,
#     absolute        : bool  = False,
#     smooth_shading  : bool  = False,
#     title           : str   = "",
#     std             : str   = False,
#     overlay_color   : str   = "green",
#     overlay_label   : str   = None,
#     overlay_size    : float = 1,
#     overlay_opacity : float = 1,
# ):
#     """
#     Plot the brain !

#     Parameters
#     ----------
#     sources: mne.SourceSpaces 
#         Source space.

#     signal: mne.SourceEstimate | np.ndarray (N,)
#         Source values.

#     overlay: np.ndarray (M,3) | nib.Nifti1Image
#         Overlay defined as point cloud.

#     tkras2ras: np.ndarray
#         Affine transform (freesurfer ras -> T1 ras). Necessary to align MRI images with meshes.

#     show_max: bool
#         Show maximum of signal if True.

#     show_src: bool
#         Show sources.

#     clim: list(float)
#         Color limit.

#     cmap: str
#         Color map.

#     absolute: bool
#         Whether to plot absolute values.

#     smooth_shading: str
#         Smoothen mesh if True.

#     title: str
#         Plot title

#     std: bool
#         Standardize accross signals.
        
#     sname: str
#         Signal name.

#     overlay_color: str
#         Overlay color

#     overlay_label: str
#         Overlay label

#     overlay_size: float
#         Overlay point size

#     overlay_opacity: float
#         Overlay opacity
#     """
#     # find longest list
#     size = 1
#     if not isinstance(sources, mne.SourceSpaces):
#         if len(sources) > size:
#             size = len(sources)
#     for obj in [signal, overlay, tkras2ras]:
#         if isinstance(obj, list):
#             size = max(size, len(obj))

#     # check every entry
#     if isinstance(sources, mne.SourceSpaces):
#         sources = [sources]*size

#     if overlay is not None:
#         if not isinstance(overlay, list):
#             overlay = [overlay]*size

#     if not isinstance(tkras2ras, list):
#         tkras2ras = [tkras2ras]*size

#     if signal is not None:
#         if isinstance(signal, mne.SourceEstimate) or isinstance(signal, np.ndarray):
#             signal = [signal]*size

#     if not isinstance(title, list):
#         title = [title]*size

#     # if Time Series plot TimeMesh
#     signal = [s.copy() for s in signal]
#     if signal is not None and isinstance(signal[0], mne.SourceEstimate):
#         return TimeMesh(
#                 sources, signal, overlay=overlay, titles=title, cmap=cmap, std=std,
#                 absolute=absolute, smooth_shading=smooth_shading, overlay_color=overlay_color,
#                 overlay_label=overlay_label, overlay_size=overlay_size, overlay_opacity=overlay_opacity,
#             )

#     # else plot single time point
#     plotter = pv.Plotter(shape=(1,size))

#     # brain plot
#     if signal is not None:
#         if std:
#             signal = [(s - np.mean(s))/np.std(s) for s in signal]

#         if absolute:
#             signal = [np.abs(s) for s in signal]

#         if clim is None:
#             clim = find_clim(signal)

#         # clim/cmap when positive values
#         if cmap is None:
#             if clim[0] < 0:
#                 cmap = "bwr"
#             else:
#                 cmap = "Reds"

#         for i in range(size):
#             # brain meshes
#             hemi_l, hemi_r = mne_to_polydata(sources[i], signal[i])

#             plotter.subplot(0,i)
#             plotter.add_mesh(hemi_l, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="lh")
#             plotter.add_mesh(hemi_r, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="rh")
#             plotter.add_title(title[i])

#     else:
#         for i in range(size):
#             # brain meshes
#             hemi_l, hemi_r = mne_to_polydata(sources[i], None)

#             plotter.subplot(0,i)
#             plotter.add_mesh(hemi_l, color="white", smooth_shading=smooth_shading, name="lh")
#             plotter.add_mesh(hemi_r, color="white", smooth_shading=smooth_shading, name="rh")
#             plotter.add_title(title[i])

#     # maximum value
#     if show_max:
#         for i in range(size):
#             split   = sources[i][0]["nuse"]
#             idx_max = np.argmax(np.abs(signal[i]))
#             if idx_max < split:
#                 map_max = sources[i][0]["rr"][sources[i][0]["vertno"]][idx_max]
#             else:
#                 map_max = sources[i][1]["rr"][sources[i][1]["vertno"]][idx_max-split]

#             plotter.subplot(0,i)
#             plotter.add_mesh(map_max, color="yellow", point_size=10)

#     # overlay
#     if overlay is not None:
#         for i in range(size):
#             if isinstance(overlay[i], nib.Nifti1Image):
#                 over = nib_to_polydata(overlay[i], trans=np.linalg.inv(tkras2ras[i]))
#             else:
#                 over = overlay[i]

#             plotter.subplot(0,i)
#             plotter.add_mesh(over, color=overlay_color, label=overlay_label, point_size=overlay_size, opacity=overlay_opacity)

#     if show_src:
#         for i in range(size):
#             pos = pv.PolyData(get_dipole_positions(sources[i]))

#             plotter.subplot(0,i)
#             plotter.add_mesh(pos, color="green", point_size=5, render_points_as_spheres=True)


#     return plotter


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


class TimeMesh:
    """
    Interactive visualization of brain source estimates over time using PyVista
    A slider allows scrolling through time points.
    """

    def __init__(
        self,
        source_spaces  : list[mne.SourceSpaces],
        timeseries     : list[mne.SourceEstimate],
        overlays       : Optional[pv.PolyData] = None,
        titles         : Optional[list[str]] = None,
        mesh_params    : Optional[dict] = None,
        overlay_params : Optional[dict] = None,
        start_time     : Optional[float] = None,
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

        self.overlays = overlays or [None] * self.n_plots
        self.titles   = titles or [""] * self.n_plots

        self.mesh_params    = mesh_params or {}
        self.overlay_params = overlay_params or {}

        # build plotter
        self.plotter = pv.Plotter(shape=(1, self.n_plots))

        # prepare meshes and mapping
        self.hemi_meshes = []
        self.vertex_maps = []
        self.splits      = []

        for i in range(self.n_plots):
            hemi_l, hemi_r, map_l, map_r, nuse_l = self._build_hemi_meshes(i)

            self.hemi_meshes.append((hemi_l, hemi_r))
            self.vertex_maps.append((map_l, map_r))
            self.splits.append(nuse_l)

            self.plotter.subplot(0,i)
            self.plotter.add_mesh(hemi_l, **self.mesh_params)
            self.plotter.add_mesh(hemi_r, **self.mesh_params)
            self.plotter.add_title(self.titles[i])

            if self.overlays[i] is not None:
                self.plotter.add_mesh(self.overlays[i], **self.overlay_params)

        # slider
        stc0 = self.ts[0]
        if start_time is None:
            start_idx  = stc0.data.shape[1] // 2
            start_time = stc0.times[start_idx]
        else:
            start_idx = stc0.time_as_index(start_time)

        self.times       = stc0.times
        self.n_times     = len(self.times)
        self.current_idx = start_idx

        self._update_time_index(start_idx)

        slider = self.plotter.add_slider_widget(
            callback = self._on_time_change,
            rng      = [self.times[0], self.times[-1]],
            value    = start_time,
            title    = "Time (s)",
            style    = "modern",
            interaction_event = "always",
        )
        self.slider = slider.GetSliderRepresentation()

        self.plotter.add_key_event("Right", lambda: self._step_time(1))
        self.plotter.add_key_event("Left", lambda: self._step_time(-1))

    # -----------------------
    # Initialization helpers
    # -----------------------

    def _build_hemi_meshes(self, idx):
        """
        Build left/right hemisphere meshes and pinfo-based vertex mappings.
        """
        src = self.src[idx]

        # extract geometry
        hemi_l = surf_to_polydata(src[0]["rr"], src[0]["tris"])
        hemi_r = surf_to_polydata(src[1]["rr"], src[1]["tris"])

        # pinfo-based mapping
        map_l = self._vertex_to_nearest_dipole_map(src[0])
        map_r = self._vertex_to_nearest_dipole_map(src[1])

        nuse_l = src[0]["nuse"]

        # initialize scalars
        hemi_l["signal"] = np.zeros(hemi_l.n_points)
        hemi_r["signal"] = np.zeros(hemi_r.n_points)

        return hemi_l, hemi_r, map_l, map_r, nuse_l

    @staticmethod
    def _vertex_to_nearest_dipole_map(src_hemi):
        """
        Convert src["nearest"] to a vertex -> dipole index map.

        Vertices whose nearest dipole was masked (not in src["vertno"])
        are assigne -1.
        """
        vertno  = np.array(src_hemi["vertno"])
        nearest = src_hemi["nearest"].copy()

        # initialize all as masked
        vertex_map = np.full(nearest.shape[0], -1, dtype=int)

        valid = np.isin(nearest, vertno)

        vertno_to_dip = {v: i for i, v in enumerate(vertno)}

        # map only valid entries
        vertex_map[valid] = np.fromiter(
            (vertno_to_dip[v] for v in nearest[valid]),
            dtype = int,
            count = np.sum(valid),
        )

        return vertex_map


    def _step_time(self, delta):
        """
        Increment time by 1 idx.
        """
        # increment idx
        self.current_idx = max(0, min(self.current_idx+delta, self.n_times-1))

        # change slider value
        self.slider.SetValue(self.times[self.current_idx])

        # update image
        self._update_time_index(self.current_idx)
        self.plotter.render()


    # ---------
    # Callback
    # ---------

    def _update_time_index(self, t_idx):
        """
        Update all meshes for a given time index
        """
        for i in range(self.n_plots):
            stc = self.ts[i]
            hemi_l, hemi_r = self.hemi_meshes[i]
            map_l, map_r   = self.vertex_maps[i]
            split = self.splits[i]

            signal = stc.data[:, t_idx]
            sig_l  = signal[:split].squeeze()
            sig_r  = signal[split:].squeeze()

            # append 0 for vertices mapped to -1
            sig_l = np.concatenate([sig_l, [0.0]])
            sig_r = np.concatenate([sig_r, [0.0]])

            hemi_l["signal"][:] = sig_l[map_l]
            hemi_r["signal"][:] = sig_r[map_r]

    def _on_time_change(self, time_value):
        """
        Slider callback
        """
        # convert time into index
        t_idx = self.ts[0].time_as_index(time_value)

        # store new index
        self.current_idx = int(t_idx)

        # update image
        self._update_time_index(self.current_idx)

    # -------
    # API
    # -------

    def show(self):
        """
        Show plot.
        """
        self.plotter.show()


def plot_eigvec(source_space, eigenvectors, eigenvalues=None):
    """
    Plot eigenvectors.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    eigenvectors: np.ndarray
        Eigenvectors.

    eigenvalues: np.ndarray
        Eigenvalues.
    """

    # -------------------
    # Prepare parameters
    # -------------------

    clim = find_clim(eigenvectors)
    cmap = default_cmap(clim)

    mesh_params = {
        "smooth_shading" : True,
        "clim" : clim,
        "cmap" : cmap,
    }

    return MultiMesh(
        source_space = source_space,
        data         = eigenvectors,
        eigenvalues  = eigenvalues,
        mesh_params  = mesh_params,
    )


class MultiMesh:
    """
    Interactive visualization of brain source estimates over time using PyVista
    A slider allows scrolling through time points.
    """

    def __init__(
        self,
        source_space : mne.SourceSpaces,
        data         : np.ndarray,
        eigenvalues  : Optional[np.ndarray] = None,
        mesh_params  : Optional[dict] = None,
        title        : Optional[str] = "",
        start_image  : Optional[int] = 0,
    ):
        """
        Parameters
        ----------
        source_spaces: list[mne.SourceSpaces]
            Source spaces.
        
        data: np.ndarray
            Brain images (NxM) where N is the number of dipoles
            and M the number of images.

        titles: list[str]
            Titles for each subplot.

        mesh_params: dict
            Parameters passed to plotter.add_mesh for brain meshes
        """
        self.src    = source_space
        self.data   = data
        self.eigval = eigenvalues
        self.title  = title

        self.n_plots = data.shape[1]
        self.current_idx = start_image

        self.mesh_params    = mesh_params or {}

        # build plotter
        self.plotter = pv.Plotter()

        # prepare meshes and mapping
        self.hemi_l, self.hemi_r, self.map_l, self.map_r, self.nuse_l = self._build_hemi_meshes()

        self.plotter.add_mesh(self.hemi_l, **self.mesh_params)
        self.plotter.add_mesh(self.hemi_r, **self.mesh_params)
        self.plotter.add_title(self.title)

        self._update_image_index(start_image)

        # callbacks
        self.plotter.add_key_event("Right", lambda: self._step(+1))
        self.plotter.add_key_event("Left", lambda: self._step(-1))

        self.plotter.add_key_event("d", lambda: self._step(+50))
        self.plotter.add_key_event("a", lambda: self._step(-50))

        self.plotter.add_key_event("c", lambda: self._step(+500))
        self.plotter.add_key_event("y", lambda: self._step(-500))

    # -----------------------
    # Initialization helpers
    # -----------------------

    def _build_hemi_meshes(self):
        """
        Build left/right hemisphere meshes and pinfo-based vertex mappings.
        """
        src = self.src

        # extract geometry
        hemi_l = surf_to_polydata(src[0]["rr"], src[0]["tris"])
        hemi_r = surf_to_polydata(src[1]["rr"], src[1]["tris"])

        # pinfo-based mapping
        map_l = self._vertex_to_nearest_dipole_map(src[0])
        map_r = self._vertex_to_nearest_dipole_map(src[1])

        nuse_l = src[0]["nuse"]

        # initialize scalars
        hemi_l["signal"] = np.zeros(hemi_l.n_points)
        hemi_r["signal"] = np.zeros(hemi_r.n_points)

        return hemi_l, hemi_r, map_l, map_r, nuse_l

    @staticmethod
    def _vertex_to_nearest_dipole_map(src_hemi):
        """
        Convert src["nearest"] to a vertex -> dipole index map.

        Vertices whose nearest dipole was masked (not in src["vertno"])
        are assigne -1.
        """
        vertno  = np.array(src_hemi["vertno"])
        nearest = src_hemi["nearest"].copy()

        # initialize all as masked
        vertex_map = np.full(nearest.shape[0], -1, dtype=int)

        valid = np.isin(nearest, vertno)

        vertno_to_dip = {v: i for i, v in enumerate(vertno)}

        # map only valid entries
        vertex_map[valid] = np.fromiter(
            (vertno_to_dip[v] for v in nearest[valid]),
            dtype = int,
            count = np.sum(valid),
        )

        return vertex_map
    
    def _label_text(self, idx: int) -> str:
        if self.eigval is None:
            return f"Image {idx + 1} / {self.n_plots}"
        else:
            val = self.eigval[idx]
            return f"Eigenvalue: {val:.4g}"

    # ---------
    # Callback
    # ---------

    def _step(self, delta):
        """
        Updates current index.
        """
        self.current_idx = max(0, min(self.current_idx + delta, self.n_plots-1))
        self._update_image_index(self.current_idx)

    def _update_image_index(self, idx):
        """
        Update all meshes for a given image index
        """
        image  = self.data[:,int(idx)]
        sig_l  = image[:self.nuse_l].squeeze()
        sig_r = image[self.nuse_l:].squeeze()

        # append 0 for vertices mapped to -1
        sig_l = np.concatenate([sig_l, [0.0]])
        sig_r = np.concatenate([sig_r, [0.0]])

        self.hemi_l["signal"][:] = sig_l[self.map_l]
        self.hemi_r["signal"][:] = sig_r[self.map_r]

        self.plotter.add_text(
            self._label_text(idx),
            name      = "info_text",
            position  = "upper_left",
            font_size = 12,
            color     = "black"
        )

        self.plotter.render()


    # -------
    # API
    # -------

    def show(self):
        """
        Show plot.
        """
        self.plotter.show()


def plot_eigval(eigval, subject="unknown"):
    """
    Plot eigenvalues.
    """
    ax = sns.lineplot(eigval)
    ax.set_title(f"Eigenvalues sub-{subject}")
    ax.set_xlabel("Harmonic index")
    ax.set_ylabel("Eigenvalue")

    plt.show()


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
        timeseries   = [ieeg],
        title        = f"iEEG measurements sub-{subject}",
        names        = ["iEEG"],
        mesh_params  = mesh_params,
        pc_params    = pc_params,
    )




class TimePC:
    """
    Interactive visualization of point cloud over time.
    """
    def __init__(
        self,
        source_space : mne.SourceSpaces,
        timeseries   : list[mne.Evoked],
        title        : Optional[str] = "Brain Signal Visualization",
        names        : Optional[str] = None,
        mesh_params  : Optional[dict] = None,
        pc_params    : Optional[dict] = None,
        start_time   : Optional[float] = None,
    ):
        """
        Parameters
        ----------
        source_spaces: list[mne.SourceSpaces]
            Source spaces.
        
        timeseries: list[mne.Evoked]
            Timeseries defined at M points (M x T)

        titles: str
            Title for plot.

        mesh_params: dict
            Parameters passed to plotter.add_mesh for brain meshes

        pc_params: dict
            Parameters passed to plotter.add_mesh for point cloud

        start_time: float
            Initial time point. Defaults to the midpoint of the times axis.
        """
        self.src   = source_space
        self.ts    = list(timeseries)
        self.title = title

        self.mesh_params = mesh_params or {"color": "skyblue", "opacity":0.1}
        self.pc_params   = pc_params or {"point_size": 10.0, "render_points_as_spheres": True}

        # build plotter
        self.plotter = pv.Plotter()
        self.plotter.add_title(self.title, font_size=10)

        # build brain meshes
        self.hemi_l, self.hemi_r = self._build_hemi_meshes()
        self.plotter.add_mesh(self.hemi_l, **self.mesh_params)
        self.plotter.add_mesh(self.hemi_r, **self.mesh_params)

        # time logic
        evoked = self.ts[0]
        self.times = evoked.times
        self.n_times = len(self.times)

        if start_time is None:
            self.current_idx = self.n_times // 2
        else:
            self.current_idx = int(evoked.time_as_index(start_time)[0])

        # prepare point clouds
        self.pcs = []
        if names is None:
            names = [f"Data_{i}" for i in range(len(timeseries))]
        self.names = names

        for i, evk in enumerate(self.ts):
            # extract channel positions
            pos      = pv.PolyData(get_ch_positions(evk))

            # create polydata
            point_cloud = pv.PolyData(pos)

            # initialize the data
            init_data = evk.data[:,self.current_idx]
            point_cloud[self.names[i]] = init_data

            # add to plotter and store reference
            self.plotter.add_mesh(point_cloud, scalars=self.names[i], **self.pc_params)
            self.pcs.append(point_cloud)

        slider = self.plotter.add_slider_widget(
            callback = self._on_time_change,
            rng      = [self.times[0], self.times[-1]],
            value    = self.times[self.current_idx],
            title    = "Time (s)",
            style    = "modern",
            interaction_event = "always",
        )
        self.slider = slider.GetSliderRepresentation()

        self.plotter.add_key_event("Right", lambda: self._step_time(1))
        self.plotter.add_key_event("Left", lambda: self._step_time(-1))

    # -----------------------
    # Initialization helpers
    # -----------------------

    def _build_hemi_meshes(self):
        """
        Build left/right hemisphere meshes and pinfo-based vertex mappings.
        """
        src = self.src

        # extract geometry
        hemi_l = surf_to_polydata(src[0]["rr"], src[0]["tris"])
        hemi_r = surf_to_polydata(src[1]["rr"], src[1]["tris"])

        return hemi_l, hemi_r


    def _step_time(self, delta):
        """
        Increment time by 1 idx.
        """
        # increment idx
        self.current_idx = max(0, min(self.current_idx+delta, self.n_times-1))

        # change slider value
        self.slider.SetValue(self.times[self.current_idx])

        # update image
        self._update_time_index(self.current_idx)
        self.plotter.render()


    # ---------
    # Callback
    # ---------

    def _update_time_index(self, t_idx):
        """
        Updates the scalar values on the point cloud meshes.
        """
        for i, pc in enumerate(self.pcs):
            # update the array inside the PolyData object
            pc[self.names[i]] = self.ts[i].data[:, t_idx]
        self.plotter.render()

    def _on_time_change(self, time_value):
        """
        Slider callback
        """
        # convert time into index
        t_idx = self.ts[0].time_as_index(time_value).squeeze()

        # store new index
        self.current_idx = int(t_idx)

        # update image
        self._update_time_index(self.current_idx)

    # -------
    # API
    # -------

    def show(self):
        """
        Show plot.
        """
        self.plotter.show()

    



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

def plot_weight_dist(adj_matrix, ax=None, **kwargs):
    """
    Plot weight distribution.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """

    # get upper triangle
    n       = adj_matrix.shape[0]
    weights = adj_matrix[np.triu_indices(n, k=1)]

    # filter out zero-weight edges
    weights = weights[weights > 0]

    # If no axis is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots()

    sns.histplot(weights, ax=ax, **kwargs)

    # set default labels
    ax.set_title("Edge Weight Distribution")
    ax.set_xlabel("Weigth")
    ax.set_ylabel("Frequency")

    return ax

def plot_degree_dist(adj_matrix, ax=None, **kwargs):
    """
    Plot degree distribution.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """

    # get degrees
    degrees = np.sum(adj_matrix>0, axis=1)

    # If no axis is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots()

    sns.histplot(degrees, ax=ax, **kwargs)

    # set default labels
    ax.set_title("Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    
def plot_con_summary(adj_matrix, subject="unknown"):
    """
    Plot connectome summary.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f"Connectome summary: sub-{subject}")

    # plots
    plot_degree_dist(adj_matrix, axs[0])
    plot_weight_dist(adj_matrix, axs[1])
    axs[1].set_xlabel("Geodesic Distance (mm)")

    plt.show()



def plot_connectome(connectome, name="Connectome", show_adj=True, show_degree=True, show_weights=True, nonzero=False, figsize=(12,5),):
    """
    Plot various measures of the connectome.

    Parameters:
    ----------
    connectome: np.ndarray (N,N)
        Connectome

    name: str (default Connectome)
        Name given to connectome

    nonzero: bool
        Whether to show the zero-weights in the distributions

    show_adj: bool
        Whether to show the adjacency matrix or not

    show_degree: bool
        Whether to show the degree distribution or not

    show_weights: bool
        Whether to show the weight distribution or not
    
    figsize: tuple(width, height)
        Figure size
    """
    n_plots  = np.sum([show_adj, show_degree, show_weights])
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    fig.suptitle(name)

    i = 0
    if show_adj:
        cmap = cm_frrim()

        axs[i].matshow(connectome, cmap=cmap)
        axs[i].set_title("Adjacency Matrix")
        axs[i].set_xlabel("Dipoles")
        axs[i].set_ylabel("Dipoles")

        i += 1

    if show_degree:
        degree = np.sum(connectome, axis=1)

        if nonzero:
            degree = degree[degree>0]

        axs[i].hist(degree, bins=100, color="skyblue")
        axs[i].set_title("Degree Distribution")
        axs[i].set_xlabel("Degree")
        axs[i].set_ylabel("Frequency")

        i += 1

    if show_weights:
        weights = connectome.flatten().astype(float)

        if nonzero:
            weights = weights[weights > 0]

        axs[i].hist(weights, bins=100, color="skyblue")
        axs[i].set_title("Weight Distribution")
        axs[i].set_xlabel("Weight")
        axs[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()



def plot_spectrum(spectrum, times=None):
    """
    Plot the connectome spectrum.

    Parameters
    ----------
        spectrum: nd.array (modes, time)
            Connectome spectrum.

        times: np.ndarray (time,)
            Times.

    Returns
    -------
        fig: plotly.graph_objs
    """
    def new_trace(values):
        trace = go.Bar(
            y       = values,
            marker  = dict(
                color = "darkblue",
                line  = dict(color="darkblue"),
            ),
            visible = False,
        )

        return trace

    border = np.max(np.abs(spectrum))
    border = [-border, border]

    if len(spectrum.shape)>1:
        if times is None: 
            print("\033[93mNo times provided\033[0m")
            times = np.arange(spectrum.shape[1])
        
        fig   = go.Figure()
        steps = []

        for i, t in enumerate(times):
            fig.add_trace(new_trace(spectrum[:,i]))

            step = dict(
                method = "update", 
                args   = [
                    {"visible": [False] * len(times)},
                    {"title": f"Connectum Spectrum: t = {t:.3f}[s]"},
                ],
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)

        fig.data[int(len(times)/2)].visible = True

        sliders = [dict(
            active       = int(len(times)/2),
            currentvalue = {"visible":False},
            pad          = {"t":50},
            steps        = steps,
        )]

        fig.update_layout(sliders = sliders)

    else:
        trace = new_trace(spectrum)

        layout = go.Layout(dict(
            title="Connectome Spectrum",
            showlegend=False, 
        ))

        fig = go.Figure(trace, layout)
        fig.data[0].visible = True


    fig.update_layout(
        plot_bgcolor = "white",
        xaxis        = dict(
            title     = "frequency indices",
            linecolor = "black",
        ),
        yaxis        = dict(
            title     = "amplitude",
            linecolor = "black",
            gridcolor = "black",
            range     = border,
        ),
    )

    return fig





# I DON'T NEED THIS PLOT ANYMORE
# def plot_resection(
#         sources, 
#         resection, 
#         cras=np.zeros(3), 
#         signal=None, 
#         show_max=False, 
#         cmap="reds", 
#         style="vox", 
#         color="green"
# ):
#     """
#     Plot resection area. 

#     Parameters
#     ----------
#     sources: mne.SourceSpaces
#         Source space.

#     resection: nib.Nifti1Image
#         Resection mask.

#     cras: np.ndarray (3,)
#         Translation (FreeSurfer -> MRI)

#     signal: np.ndarray (N,)
#         Brain activity.

#     show_max: bool
#         Show maximum of brain acitivity.

#     cmap: str
#         Color map for brain activity.

#     style: str
#         Style of resection area:
#         - cloud: looks like a cloud
#         - vox: see individual voxels

#     color: str
#         Color of resection area. 
#     """
    
#     if signal is None:
#         signal = np.zeros(sources[0]["nuse"]+sources[1]["nuse"])

#     if style == "cloud":
#         style = dict(
#             emissive = True,
#             style = "points_gaussian",
#             point_size = 10,
#         )
#     elif style == "vox":
#         style = dict(
#             point_size=0.1,
#             opacity=0.3,
#         )
#     else:
#         raise ValueError(f"Unknown style '{style}' for resection plot.")

#     # resection area
#     #voxels = np.argwhere(resection.get_fdata()>0)
#     #points = (nib.affines.apply_affine(resection.affine, voxels)-cras)/1000
#     points = nib_to_polydata(resection, trans=cras)

#      # plotter
#     plotter = plot_brain_signal(sources, signal, cmap=cmap, show_max=show_max)
#     plotter.add_mesh(points, color=color, label="Resected Area", **style)

#     return plotter



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


###############################################
##                Utilities                  ##
##                                           ##
## - store_in_lists()                        ##
## - find_clim()                             ##
## - default_cmap()                          ##
###############################################

def store_in_lists(*args):
    """
    Store elements in lists.
    """
    # determine target size (longest list)
    sizes = [
        len(a) for a in args
        if isinstance(a, list) and not isinstance(a, mne.SourceSpaces)
    ]
    size = max(sizes, default=1)

    # validate list sizes
    if sizes and any(s != size for s in sizes):
        raise ValueError("All list inputs must have the same length.")

    # broadcast non-lists
    new_args = [
        a if isinstance(a, list) and not isinstance(a, mne.SourceSpaces)
        else [a] * size
        for a in args
    ]

    return new_args

def find_clim(data, percentile=99)-> tuple | None:
    """
    Find color limits accross one or multiple datasets.

    Parameters
    ----------
    data: np.ndarray, mne.SourceEstimate or list therefof
        Data arrays or SourceEstimate objects. None entries are ignored.
    
    percentile: float, optional
        Percentile of the data to use for the maximum absolute value.

    Returns
    -------
    clim: tuple or None
        Color limits (vmin, vmax). Returns None if all data is empty or zero.
    """
    if not isinstance(data, list):
        data = [data]

    abs_max = 0
    only_positive = True

    for d in data:
        if d is None:
            continue
        if isinstance(d, mne.SourceEstimate):
            d = d.data

        if d.size == 0:
            continue

        # check if only positive
        d_min         = np.nanmin(d)
        only_positive = only_positive and (d_min >= 0)

        # compute absolute max using percentile
        abs_val = np.nanpercentile(np.abs(d), percentile)
        abs_max = max(abs_max, abs_val)

    if abs_max == 0:
        return None
    elif only_positive:
        return (0, abs_max)
    else:
        return (-abs_max, abs_max)

def default_cmap(clim):
    """
    Select a default colormap based on the color limits.

    Parameters
    ----------
    clim : tuple of float, optional
        Color limits (vmin, vmax). If all values are >= 0, choose a sequential
        colormap ('Reds'). If any values are negative, choose a diverging
        colormap ('bwr'). If None, returns None.

    Returns
    -------
    cmap: str or None
        Name of the colormap to use.
    """
    if clim is None:
        return None

    if not (isinstance(clim, (tuple, list)) and len(clim) == 2):
        raise ValueError(f"clim must be a tuple or list of length 2, go {clim}")

    vmin, _ = clim
    if vmin < 0:
        return "bwr"     # diverging for signed data
    else:
        return "Reds"    # sequential for positive-only data



#################################################
########### Special plot publication ############
#################################################


def brain_gif(
    sources         : mne.SourceSpaces, 
    signals         : np.ndarray,
    fname           : str,
    head            : dict = None, 
    camera_position : str = "yz", 
    azimuth         : int = 0,
    roll            : int = 0,
    zoom            : float = 1,
    fps             : int = 10,
):
    """
    Create gif of brain. 

    sources: mne.SourceSpaces 
        Source space.

    signals: np.ndarray (N,T)
        Brain signals (signals should be max 15 frames).

    fname: str
        File name

    head: dict (mne surface)
        Head mesh

    camera_position: str
        Sets camera position.
    
    azimuth: int
        Sets azimuth angle.
    
    zoom: float
        Camera zoom.
    """
    plotter = pv.Plotter(notebook=False, off_screen=True)

    # brain hemispheres
    hemiL, hemiR = mne_to_polydata(sources, signals[:,0])
    plotter.add_mesh(hemiL, smooth_shading=True, cmap="Reds", clim=[0, np.max(signals)])
    plotter.add_mesh(hemiR, smooth_shading=True, cmap="Reds", clim=[0, np.max(signals)])
    plotter.remove_scalar_bar()

    # head
    if head is not None:
        head = surf_to_polydata(head["rr"], head["tris"])
        plotter.add_mesh(head, show_edges=True, color="white", opacity=0.2, edge_opacity=0.8, edge_color="black")

    # camera position
    plotter.camera_position = camera_position
    plotter.camera.azimuth += azimuth
    plotter.camera.roll += roll
    plotter.camera.zoom(zoom)

    # generate gif
    plotter.open_gif(fname, fps=fps)
    nframe = signals.shape[1]
    for frame in np.arange(nframe):
        new_hemiL, new_hemiR = mne_to_polydata(sources, signals[:,frame])
        hemiL["signal"] = new_hemiL["signal"]
        hemiR["signal"] = new_hemiR["signal"]
        plotter.write_frame()

    plotter.close()


def plot_hops(sim, n_hops=3):

    signals = np.zeros((n_hops+1, sim.act.data.shape[1]))

    # select source
    idx          = np.random.choice(sim.loc_ids)
    signals[0,:] = sim.act.data[idx,:] + sim.bnoise[idx,:]

    # select hop neighbors
    starting_roi = np.zeros(sim.sc.shape[0], dtype=bool)
    starting_roi[sim.loc_ids] = True

    rois = starting_roi
    for i in range(n_hops):
        # compute nex hop roi
        roi = (sim.sc @ rois > 0) & (~rois)

        # check if roi is empty or wrong
        assert np.all((roi==0) | (roi==1))
        if np.all(roi==0):
            print("\033[93mNo more neighbors\033[0m")
            break 
        roi = roi.astype(bool)

        idx          = np.random.choice(np.argwhere(roi).squeeze())
        signals[i+1,:] = sim.act.data[idx,:] + sim.bnoise[idx,:]

        rois = rois | roi 

    _, axes = plt.subplots(nrows=1, ncols=n_hops+1, sharex=True, sharey=True, figsize=(15,3))
    coloors   = ["#EE6C4D", "#FFB627", "#3D5A80", "#98C1D9", "#293241"]

    for i in range(n_hops+1):
        for j in range(i+1):
            axes[i].plot(signals[j,:], color=coloors[j], linewidth=2)
        axes[i].set_xlabel("Time [s]")
    
    axes[0].set_ylabel("Amplitude [V]")

    plt.show()

def plot_fourier_image_example():
    N = 30
    f, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, sharex="col", sharey="row", figsize=(15,3))

    # first image
    xf = np.zeros((N,N))
    xf[0, 1] = 1
    Z = ifftn(xf)
    ax1.imshow(np.real(Z), cmap=cm.gray)

    # second image
    xf = np.zeros((N,N))
    xf[2, 0] = 1
    Z = ifftn(xf)
    ax2.imshow(np.real(Z), cmap=cm.gray)

    # third image
    xf = np.zeros((N,N)) 
    xf[3, 5] = 1
    Z = ifftn(xf)
    ax3.imshow(np.real(Z), cmap=cm.gray)

    # fourth image
    xf = np.zeros((N,N))
    xf[12, 5] = 1
    Z = ifftn(xf)
    ax4.imshow(np.real(Z), cmap=cm.gray)

    # fith image
    xf = np.zeros((N,N))
    xf[14, 14] = 1
    Z = ifftn(xf)
    ax5.imshow(np.real(Z), cmap=cm.gray)

    plt.show()



###############################
########## Color Maps #########
###############################

def cm_frrim():
    """
    Color map used for FRRIM. 
    """
    newcolors          = cm.Blues(np.linspace(0,1,256))
    newcolors[250:, :] = np.array([61/256, 90/256, 128/256, 1])
    newcmp             = ListedColormap(newcolors)

    return newcmp


def plot_harmonics(src, harmonics, frequencies=(0,1,2,3,4,5)):
    """
    Plot brain harmonics.

    Parameters
    ----------
    src: mne.SourceSpace
        Source Space.

    harmonics: np.ndarray 
        Harmonics

    frequencies: list[int]
        Frequencies to plot.

    Returns
    -------
    plot: pv.Plotter
    """
    harmonics = [harmonics[:,freq] for freq in frequencies]
    titles    = [f"harm {i}" for i in frequencies]
    plotter   = plot_brain(src, harmonics, title = titles)
    return plotter



def plot_graph(points, adjacency):
    """
    Parameters
    ----------
    points: np.array (N,3) 
    lines: np.array (N,2)
    """

    P = pv.PolyData(points)
    A1 = adjacency > 0
    A2 = adjacency < 0

    lines1 = []
    lines2 = []
    for i in range(len(A1)):
        for j in range(i, len(A1)):
            if A1[i,j]:
                lines1 += [points[i], points[j]]
            if A2[i,j]:
                lines2 += [points[i], points[j]]
    lines1 = np.array(lines1)
    lines2 = np.array(lines2)

    p = pv.Plotter()
    p.add_mesh(P)
    if len(lines1) > 0:
        p.add_lines(lines1, color="green", width=0.05)
    if len(lines2) > 0:
        p.add_lines(lines2, color="red", width=0.01)

    return p
