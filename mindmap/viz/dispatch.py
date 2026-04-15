"""
Module Name: viz/dispatch.py
Description: User plot API

===============================================================================
API VISUALIZATION (dispatch.py)
===============================================================================
Targets: MNE objects.
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

import numpy as np

import mne
import nibabel as nib

import pyvista as pv

from mindmap.core.process import normalize_data
from mindmap.anatomy.mri.io import nib_to_polydata
from mindmap.analysis.functional.eeg.process import normalize_evoked
from mindmap.viz.utils import (
    store_in_lists, find_clim, default_cmap,
    BrainMesh, SourceMesh, ChannelMesh, OverlayMesh
)
from mindmap.viz.vista import TimeMesh



def plot_brain_activity(
    source_space : mne.SourceSpaces | list[mne.SourceSpaces],
    functionals : np.ndarray | mne.SourceEstimate | list[mne.SourceEstimate | np.ndarray] = None,
    sensors: mne.Evoked | list[mne.Evoked] = None,
    overlay : np.ndarray | nib.Nifti1Image | list[np.ndarray | nib.Nifti1Image] = None,
    tkras2ras : np.ndarray | list[np.ndarray] = None,
    title : str | list[str] = "",
    mesh_params : dict = None,
    func_params : dict = None,
    sensor_params : dict = None,
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
    
    func_params: dict
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

    # ------------------
    # Normalize inputs
    # ------------------

    source_space, functionals, sensors, overlay, tkras2ras, title = store_in_lists(
        source_space, functionals, sensors, overlay, tkras2ras, title,
    )

    mesh_params = mesh_params or {}
    func_params = func_params or {}
    sensor_params = sensor_params or {}
    overlay_params = overlay_params or {}

    # --------------------
    # Process functionals
    # --------------------

    processed_data = []
    processed_times = []

    for func in functionals:
        if func is None:
            processed_data.append(None)
            processed_times.append(None)
            continue

        # extract data matrix and times
        d = func.data if hasattr(func, "data") else func
        t = func.times if hasattr(func, "times") else None

        if norm := func_params.get("normalize"):
            d = normalize_data(d, "global", norm)

        if func_params.get("absolute"):
            d = np.abs(d)

        processed_data.append(d)
        processed_times.append(t)

    # calculate golobal color limits accross all time points and subplots
    clim = find_clim(processed_data, percentile=100)
    if not mesh_params.get("clim"):
        mesh_params["clim"] = clim
    if not mesh_params.get("cmap"):
        mesh_params["cmap"] = default_cmap(clim)

    # -------------------------
    # Instantiate Mesh Objects
    # -------------------------

    plot_groups = []

    for i, src in enumerate(source_space):
        subplot_items = []

        # add brain/source mesh
        if isinstance(functionals[i], mne.SourceEstimate):
            stc = functionals[i].copy()
            stc.data = processed_data[i]
            bm = SourceMesh(
                src = src,
                stc = stc,
                scalar_name = "Source Activity",
                **mesh_params,
            )
        else:
            bm = BrainMesh(
                src = source_space[i],
                functional = processed_data[i],
                times = processed_times[i],
                scalar_name = "Source Activity",
                **mesh_params,
            )
        subplot_items.append(bm)

        # add sensors
        if sensors[i] is not None:
            sm = ChannelMesh(
                evoked = sensors[i],
                scalar_name = "Electrode Potential",
                **sensor_params,
            )
            subplot_items.append(sm)

        # add overlay (static/point cloud mesh)
        if overlay[i] is not None:
            over = overlay[i]
            if isinstance(over, nib.Nifti1Image):
                poly = nib_to_polydata(over, trans=tkras2ras[i])
            else:
                poly = pv.PolyData(over)

            om = OverlayMesh(
                overlay = poly,
                scalar_name = "overlay",
                **overlay_params,
            )

            subplot_items.append(om)

        plot_groups.append(subplot_items)


    # ----------------
    # Create TimeMesh
    # ----------------

    visualizer = TimeMesh(
        items = plot_groups,
        titles = title,
    )

    return visualizer.plotter


def plot_ieeg(
    src: mne.SourceSpaces,
    ieeg: mne.Evoked,
    extent_mask: np.ndarray = None,
    spurious_mask: np.ndarray = None,
    ts_params: dict = None,
    subject: str = "unknown",
):
    """
    Plot iEEG data.
    """
    ts_params = ts_params or {}
    timeseries = ieeg.copy()

    # -----------------
    # Data processing
    # -----------------

    if norm := ts_params.get("normalize"):
        timeseries = normalize_evoked(timeseries, "global", norm)
    if ts_params.get("absolute"):
        timeseries.data = np.abs(timeseries.data)

    # ------------
    # Brain Mesh
    # -----------

    brain = BrainMesh(src, color="skyblue", opacity=0.1, smooth_shading=True)

    # --------------
    # iEEG Channels
    # --------------

    channels = ChannelMesh(
        evoked = ieeg,
        scalar_name = "Electric potential [V]",
        point_size = 10,
        render_points_as_spheres = True,
        cmap = "bkr" if np.any(timeseries.data < 0) else "kr",
    )

    # ---------
    # TimeMesh
    # ---------
    title = f"Intracranial EEG sub-{subject}"
    tm = TimeMesh(items=[[brain, channels]], titles=[title])

    # Add iEEG overlays
    point_positions = channels.polydata.points

    if extent_mask is not None:
        extent_poly = pv.PolyData(point_positions[extent_mask])
        e_actor = tm.plotter.add_mesh(
            extent_poly, color="red", point_size=15,
            render_points_as_spheres=True, name="extent_h", opacity=0.6,
        )
        e_actor.SetVisibility(False)

        tm.plotter.add_checkbox_button_widget(
            e_actor.SetVisibility,
            value = False,
            position = (10, 70),
            size = 25,
            color_on = "red",
        )
        tm.plotter.add_text("Show Set H(extent)", position=(40, 75), font_size=8)

    if spurious_mask is not None:
        spurious_poly = pv.PolyData(point_positions[spurious_mask])
        s_actor = tm.plotter.add_mesh(
            spurious_poly, color="yellow", point_size=13,
            render_points_as_spheres=True, name="spurious_h", opacity=0.6,
        )
        s_actor.SetVisibility(False)

        tm.plotter.add_checkbox_button_widget(
            s_actor.SetVisibility,
            value = False,
            position = (10, 30),
            size = 25,
            color_on = "yellow",
        )
        tm.plotter.add_text("Show spurious mask", position=(40, 35), font_size=8)

    return tm.plotter
