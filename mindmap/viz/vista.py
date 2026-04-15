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

from mindmap.viz.utils import (
    BaseBrainPlotter, BaseMesh, BrainMesh, SourceMesh, ChannelMesh, OverlayMesh
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

            for mesh in group:
                # Assign parameters based on mesh type
                params = getattr(mesh, "plot_kwargs", {})
                if isinstance(mesh, (BrainMesh, SourceMesh)):
                    self.plotter.add_mesh(mesh.hemi_l, scalars=mesh.scalar_name, **params)
                    self.plotter.add_mesh(mesh.hemi_r, scalars=mesh.scalar_name, **params)

                elif isinstance(mesh, (ChannelMesh, OverlayMesh)):
                    self.plotter.add_mesh(mesh.polydata, scalars=mesh.scalar_name, **params)

        if self.max_states > 1:
            self.add_slider(label="Time [s]", rng=[self.times[0], self.times[-1]])

        self.setup_keys()
        self.update_display(0)


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
