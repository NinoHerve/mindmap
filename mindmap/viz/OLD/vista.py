"""
Module Name: viz/vista.py
Description: Plotting functions using pyvista.

===============================================================================
MATH-SPACE VISUALIZATION (core.py)
===============================================================================
Targets: pv.PolyData
Backends: Pyvista
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

from typing import Optional
import warnings

import numpy as np

import pyvista as pv
import nibabel as nib

from mindmap.core.process import normalize_data
from mindmap.anatomy.mri.io import nib_to_polydata
from mindmap.viz.OLD.utils import (
    BaseBrainPlotter, BaseMesh, BrainMesh, SourceMesh, SurfaceMesh, ChannelMesh, OverlayMesh,
    store_in_lists, find_clim, default_cmap,
)


class TimeMesh(BaseBrainPlotter):
    """
    Interactive visualization of brain source estimates over time using PyVista
    A slider allows scrolling through time points.
    """

    def __init__(
        self,
        items : list[BaseMesh] | list[list[BaseMesh]],
        titles : Optional[list[str]] = None,
        slider_label : str = "Time [s]",
    ):
        """
        Parameters
        ----------
        source_spaces: list[pv.PolyData]
            Brain meshes.

        functionals: list[np.ndarray of shape (N,T)]
            Brain function over time.

        times: np.ndarray of shape (T,)
            Time values.

        overlays: list[pv.PolyData]
            Optional overlay meshes

        titles: list[str]
            Titles for each subplot.

        slider_label: str
            Label shown on the state slider. The slider itself always steps
            through each mesh's `times`/state axis - this only changes its
            caption, so it fits non-temporal use (e.g. "Harmonic index").

        mesh_params: dict
            Parameters passed to plotter.add_mesh for brain meshes

        overlay_params: dict
            Parameters passed to plotter.add_mesh for overlays

        start_time: float
            Initial time point. Defaults to the midpoint of the times axis.
        """
        # Normalize items: ensure everything is a list of lists
        self.plot_groups = [group if isinstance(group, list) else [group] for group in items]

        self.n_plots = len(self.plot_groups)
        # find the global max time states across all groups
        all_times = [
            mesh.times for group in self.plot_groups
            for mesh in group if mesh.times is not None
        ]

        if not all_times:
            self.times = np.array([0])  # fallback for static plots
            self.max_states = 1
        else:
            # validate that all existing time arrays are identical
            ref_time = all_times[0]
            if not all(np.array_equal(t, ref_time) for t in all_times):
                warnings.warn("Inconsistent time vectors detected across meshes. Using the first available.")

            self.times = ref_time
            self.max_states = len(ref_time)

        self.plotter = pv.Plotter(shape=(1, self.n_plots))
        super().__init__(self.plotter, n_states=self.max_states)

        for i, group in enumerate(self.plot_groups):
            self.plotter.subplot(0,i)
            if titles and i < len(titles):
                self.plotter.add_title(titles[i], font_size=10)

            checkbox_row = 0
            for mesh in group:
                # Assign parameters based on mesh type
                params = getattr(mesh, "plot_kwargs", {})
                if isinstance(mesh, (BrainMesh, SourceMesh)):
                    self.plotter.add_mesh(mesh.hemi_l, scalars=mesh.scalar_name, **params)
                    self.plotter.add_mesh(mesh.hemi_r, scalars=mesh.scalar_name, **params)

                elif isinstance(mesh, SurfaceMesh):
                    self.plotter.add_mesh(mesh.polydata, scalars=mesh.scalar_name, **params)

                elif isinstance(mesh, (ChannelMesh, OverlayMesh)):
                    actor = self.plotter.add_mesh(mesh.polydata, scalars=mesh.scalar_name, **params)
                    if getattr(mesh, "toggle", False):
                        # `position` is scoped to the active subplot's viewport, not the
                        # window - verified empirically, so no per-subplot x offset is needed.
                        self.plotter.add_checkbox_button_widget(
                            actor.SetVisibility, value=True,
                            position=(10, 10 + 40 * checkbox_row), size=25,
                            color_on=params.get("color", "grey"),
                        )
                        checkbox_row += 1

        if self.max_states > 1:
            self.add_slider(label=slider_label, rng=[self.times[0], self.times[-1]])

        self.setup_keys()
        self.update_display(0)

        self.plotter.link_views()
    # -------------------
    # Inherited methods
    # -------------------

    def value_to_idx(self, value):
        """Convert value to index"""
        diff = np.abs(self.times - value)
        return np.argmin(diff, axis=0)

    def idx_to_value(self, idx):
        """Convert index to value"""
        return self.times[idx]

    def update_display(self, idx):
        """
        Update all meshes for a given time index
        """
        for group in self.plot_groups:
            for mesh in group:
                mesh.update_signal(idx)
        self.plotter.render()


def plot_surface_activity(
    polydata,
    functional = None,
    times = None,
    vertex_map = None,
    sensors = None,
    overlay = None,
    ras2tkras = None,
    title = "",
    slider_label : Optional[str] = None,
    mesh_params : dict = None,
    func_params : dict = None,
    sensor_params : dict = None,
    overlay_params : dict = None,
) -> pv.Plotter:
    """
    Plot one or more arbitrary surfaces with an optional functional map.

    Low-level counterpart of `mindmap.viz.plot_brain_activity`: it takes bare
    `pv.PolyData` surfaces instead of an `mne.SourceSpaces`, so it can plot a
    single hemisphere, a non-brain surface, or any custom mesh. Supports the
    same slider/overlay/sensor functionality - a state slider appears
    whenever `functional` has more than one column, labeled "Time [s]" if
    `times` is given, otherwise a generic index label.

    Parameters
    ----------
    polydata: pv.PolyData | list[pv.PolyData]
        Surface(s) to plot, one per subplot.

    functional: np.ndarray | list[np.ndarray], optional
        Functional of shape (n, n_timepoints) or (n,) per surface. `n` is the
        number of dipoles if `vertex_map` is given for that surface,
        otherwise it must equal that surface's `n_points` (already dense,
        per-vertex).

    times: np.ndarray | list[np.ndarray], optional
        Time points per surface.

    vertex_map: np.ndarray | list[np.ndarray], optional
        Per-surface dense-vertex-to-dipole map, see `SurfaceMesh`. A None
        entry means `functional` is already dense for that surface.

    sensors: mne.Evoked | list[mne.Evoked], optional
        Sensor positions/values overlaid as a point cloud.

    overlay: np.ndarray (M,3) | nib.Nifti1Image | list thereof, optional
        Overlay defined as a point cloud.

    ras2tkras: np.ndarray | list[np.ndarray], optional
        Affine transform (T1 ras -> freesurfer ras), needed when `overlay`
        is a Nifti1Image.

    title: str | list[str]
        Subplot titles.

    slider_label: str, optional
        Slider caption. Defaults to "Time [s]" if any surface has `times`
        set, otherwise "Map index".

    func_params: dict
        normalize: max | minmax | zscore
        absolute: True | False

    mesh_params, sensor_params, overlay_params: dict
        Parameters passed to the corresponding mesh's plot kwargs.

    Returns
    -------
    pv.Plotter
    """
    # ------------------
    # Normalize inputs
    # ------------------

    polydata, functional, times, vertex_map, sensors, overlay, ras2tkras, title = store_in_lists(
        polydata, functional, times, vertex_map, sensors, overlay, ras2tkras, title,
    )

    mesh_params = mesh_params or {}
    func_params = func_params or {}
    sensor_params = sensor_params or {}
    overlay_params = overlay_params or {}

    # --------------------
    # Process functionals
    # --------------------

    processed_data = []

    for func in functional:
        if func is None:
            processed_data.append(None)
            continue

        d = np.asarray(func)

        if norm := func_params.get("normalize"):
            d = normalize_data(d, "global", norm)

        if func_params.get("absolute"):
            d = np.abs(d)

        processed_data.append(d)

    clim = find_clim(processed_data, percentile=100)
    if not mesh_params.get("clim"):
        mesh_params["clim"] = clim
    if not mesh_params.get("cmap"):
        mesh_params["cmap"] = default_cmap(clim)

    # -------------------------
    # Instantiate Mesh Objects
    # -------------------------

    plot_groups = []
    have_times = False

    for i, poly in enumerate(polydata):
        subplot_items = []

        sm = SurfaceMesh(
            polydata = poly,
            functional = processed_data[i],
            times = times[i],
            vertex_map = vertex_map[i],
            scalar_name = "Surface Activity",
            **mesh_params,
        )
        subplot_items.append(sm)
        have_times = have_times or times[i] is not None

        if sensors[i] is not None:
            ch = ChannelMesh(
                evoked = sensors[i],
                scalar_name = "Electrode Potential",
                **sensor_params,
            )
            subplot_items.append(ch)

        if overlay[i] is not None:
            over = overlay[i]
            if isinstance(over, nib.Nifti1Image):
                poly_over = nib_to_polydata(over, trans=ras2tkras[i])
            else:
                poly_over = pv.PolyData(over)

            om = OverlayMesh(
                polydata = poly_over,
                scalar_name = "overlay",
                **overlay_params,
            )
            subplot_items.append(om)

        plot_groups.append(subplot_items)

    if slider_label is None:
        slider_label = "Time [s]" if have_times else "Map index"

    # ----------------
    # Create TimeMesh
    # ----------------

    visualizer = TimeMesh(
        items = plot_groups,
        titles = title,
        slider_label = slider_label,
    )

    return visualizer.plotter
