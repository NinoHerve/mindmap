"""
Module Name: viz/surface.py
Description: `SurfacePlot` - the generic, domain-agnostic plot shortcut:
one surface (or point cloud), one scalar map, clean named parameters.

Combine several with `+`/`|`/`&` (see `viz.panel`) for multi-mesh or
multi-panel plots, instead of passing parallel lists of params.

Author: Nino Herve
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pyvista as pv

from mindmap.viz.mesh import Layer, Provider, find_clim, default_cmap, preprocess, pop_clim_options
from mindmap.viz.panel import Panel


@dataclass
class SurfacePlot(Panel):
    """
    A single surface (or point cloud) with an optional scalar map and/or
    state axis.

    Parameters
    ----------
    mesh: pv.PolyData
        Surface or point-cloud geometry.

    values: np.ndarray | Callable[[int], np.ndarray], optional
        Scalar field of shape (n_points,)/(n_points, n_states), or a
        callable `idx -> (n_points,) array`. See `Layer`.

    states: np.ndarray, optional
        Axis values (time in seconds, harmonic index, ...). Inferred from
        `values`' shape when omitted (arrays only).

    title: str
        Panel title.

    scalar_name: str
        Label for this plot's scalar field - also the selector `Layout.link_scale`
        matches on, so give related plots (e.g. the same quantity across
        several subjects) the same name, and unrelated ones (e.g. EEG vs.
        brain activity in the same panel) different names. Defaults to the
        same literal "value" for every instance - simple and predictable,
        but note that PyVista synchronizes the color range of every actor
        sharing a scalar array name within one Plotter (even across
        subplots) to whichever was added last: combining several
        default-named `SurfacePlot`s via `|`/`&` will silently make them
        all use the last one's range unless you give them distinct names.
        `link_scale` needs explicit, meaningful names to select by anyway,
        so name your plots whenever you're comparing more than one.

    mesh_params: dict
        Passed to this plot's `Layer` (and so to `pv.Plotter.add_mesh`). If
        `clim`/`cmap` aren't set and `values` is a plain array, they default
        to this plot's own data range - each `SurfacePlot` is independent,
        so combining several with different ranges no longer forces a
        shared scale; pass matching `clim`/`cmap` explicitly (or use
        `Layout.link_scale`) to compare panels on the same scale. Two keys
        steer that inferred `clim` instead: `clim_percentile` (default 99,
        the percentile of |values| used as the limit) and `symmetric`
        (True: always `(-m, m)`, even for all-positive values) -- see
        `viz.mesh.find_clim`. Values beyond `clim` saturate.

    func_params: dict
        `normalize`: "max" | "minmax" | "zscore"
        `absolute`: bool
        Applied to array `values` before plotting (callables used as-is).
    """

    mesh: pv.PolyData
    values: Union[np.ndarray, Provider, None] = None
    states: Optional[np.ndarray] = None
    title: str = ""
    scalar_name: str = "value"
    mesh_params: dict = field(default_factory=dict)
    func_params: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[Layer]:
        processed = preprocess(self.values, self.func_params)

        mesh_params = dict(self.mesh_params)
        clim_options = pop_clim_options(mesh_params)
        if processed is not None and not callable(processed):
            mesh_params.setdefault("clim", find_clim([processed], **clim_options))
            mesh_params.setdefault("cmap", default_cmap(mesh_params["clim"]))

        return [Layer(self.mesh, processed, self.states, scalar_name=self.scalar_name, **mesh_params)]
