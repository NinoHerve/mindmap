"""
Module Name: viz/mesh.py
Description: The `Layer` primitive - one renderable object (a surface or
point-cloud mesh, optionally carrying a scalar field across a state axis)
plus the small pure helpers used to style it.

`Layer` deliberately knows nothing about brains, hemispheres, dipoles, or
"time" - it only knows a `pv.PolyData` and, optionally, a function that
returns its scalar values for a given state index. Domain-specific mapping
(e.g. dipole-space -> dense-vertex painting for an mne.SourceSpaces) belongs
in a wrapper that builds that function, not here.

Author: Nino Herve
License: MIT
"""

from typing import Callable, Optional, Union

import numpy as np
import pyvista as pv

from mindmap.core.process import normalize_data

Provider = Callable[[int], np.ndarray]


#: The colormaps `default_cmap` picks: diverging for a `clim` spanning 0, sequential otherwise.
#: Set them module-wide, e.g. `mindmap.viz.mesh.DIVERGING_CMAP = "RdBu_r"`.
DIVERGING_CMAP = "bwr"
SEQUENTIAL_CMAP = "Reds"

#: The `mesh_params` keys that steer the inferred `clim` instead of going to pyvista.
CLIM_OPTIONS = ("clim_percentile", "symmetric")


def find_clim(data, percentile: float = 99, symmetric: bool = False) -> Optional[tuple]:
    """
    Find color limits across one or multiple arrays: `(-m, m)`, or `(0, m)`
    when every value is non-negative, with `m` the `percentile` of |values|
    -- a robust range that a few outliers don't stretch. Values beyond it
    saturate at the colormap's ends when rendered.

    Parameters
    ----------
    data: np.ndarray or list thereof
        Arrays to inspect. None entries are ignored.

    percentile: float
        Percentile of the (absolute) data used as the limit magnitude
        (default 99; 100 is the plain maximum).

    symmetric: bool
        Always `(-m, m)` -- a diverging scale centred on 0 even when the
        values happen to be all positive (e.g. one of several signed maps).

    Returns
    -------
    clim: (vmin, vmax) or None
        None if all data is empty/None/zero.
    """
    if isinstance(percentile, (bool, np.bool_)) or not np.isscalar(percentile) or not 0 < percentile <= 100:
        raise ValueError(f"`clim_percentile` must be in (0, 100], got {percentile!r}.")
    if not isinstance(data, list):
        data = [data]

    abs_max = 0.0
    only_positive = True

    for d in data:
        if d is None:
            continue

        d = np.asarray(d)
        if d.size == 0:
            continue

        only_positive = only_positive and (np.nanmin(d) >= 0)
        abs_max = max(abs_max, np.nanpercentile(np.abs(d), percentile))

    if abs_max == 0:
        return None

    return (0, abs_max) if only_positive and not symmetric else (-abs_max, abs_max)


def pop_clim_options(mesh_params: dict) -> dict:
    """
    Removes `CLIM_OPTIONS` from `mesh_params` (in place, so they never
    reach pyvista) and returns them as `find_clim` keywords:
    `clim_percentile` -> `percentile` (default 99), `symmetric` (default
    False).
    """
    return {
        "percentile": mesh_params.pop("clim_percentile", 99),
        "symmetric": bool(mesh_params.pop("symmetric", False)),
    }


def default_cmap(clim: Optional[tuple]) -> Optional[str]:
    """Pick a sequential (`SEQUENTIAL_CMAP`) or diverging (`DIVERGING_CMAP`) colormap from `clim`."""
    if clim is None:
        return None

    vmin, _ = clim
    return DIVERGING_CMAP if vmin < 0 else SEQUENTIAL_CMAP


def preprocess(values, func_params: dict):
    """
    Apply `func_params` (`normalize`, `absolute`) to plain array `values`
    before plotting. Callables and None pass through unchanged - a
    callable provider is applied per-state at render time, not here; a
    wrapper that wants preprocessing on a callable's output should wrap
    the callable itself.
    """
    if values is None or callable(values):
        return values

    d = np.asarray(values)

    if norm := func_params.get("normalize"):
        d = normalize_data(d, "global", norm)

    if func_params.get("absolute"):
        d = np.abs(d)

    return d


def _make_provider(values, states):
    """
    Normalize `values`/`states` into (provider, resolved_states).

    `values` is either a plain (n,)/(n,T) array or a callable
    `idx -> (n,) array`. `states` is the axis of state labels (time in
    seconds, harmonic index, eigenvalue, ...) - its length is the state
    count, so no separate `n_states` argument is needed.
    """
    if values is None:
        return None, np.array([0])

    if callable(values):
        if states is None:
            raise ValueError(
                "`states` is required when `values` is a callable provider "
                "(its length can't be inferred)."
            )
        return values, np.atleast_1d(states)

    arr = np.asarray(values)
    arr = arr[:, np.newaxis] if arr.ndim == 1 else arr
    n = arr.shape[1]

    resolved = np.atleast_1d(states) if states is not None else np.arange(n)
    if len(resolved) != n:
        raise ValueError(
            f"`states` length ({len(resolved)}) must match values' second axis ({n})."
        )

    return (lambda idx: arr[:, idx]), resolved


def _default_style(style: str, has_scalar: bool) -> dict:
    base = {"style": style} | (
        {"render_points_as_spheres": True, "point_size": 8}
        if style == "points"
        else {"smooth_shading": True}
    )

    if not has_scalar:
        return base | {"color": "white", "opacity": 0.5}

    return base | {"cmap": "Reds"}


class Layer:
    """
    One renderable object: a surface or point-cloud mesh with an optional
    scalar field driven by a state axis.
    """

    def __init__(
        self,
        mesh: pv.PolyData,
        values: Union[np.ndarray, Provider, None] = None,
        states: Optional[np.ndarray] = None,
        scalar_name: str = "value",
        style: Optional[str] = None,
        toggle: bool = False,
        **plot_kwargs,
    ):
        """
        Parameters
        ----------
        mesh: pv.PolyData
            Surface or point-cloud geometry.

        values: np.ndarray of shape (n_points,)/(n_points, n_states), or
            Callable[[int], np.ndarray], optional
            Scalar field. A plain array is used directly; a callable is
            invoked lazily with the state index (used e.g. by a wrapper that
            paints dipole-space data onto a dense mesh on demand, instead of
            precomputing every state up front).

        states: np.ndarray, optional
            Axis values (time in seconds, harmonic index, eigenvalue, ...).
            Its length is the state count. Required when `values` is a
            callable; inferred from `values`' shape otherwise.

        scalar_name: str
            Name of the scalar array painted onto `mesh`.

        style: {"surface", "wireframe", "points"}, optional
            Rendering style, passed straight through to `add_mesh(style=...)`.
            Auto-detected as "surface"/"points" from whether `mesh` has
            faces when left unset.

        toggle: bool
            Whether a visibility checkbox should be shown for this layer
            (handled by `Scene`).

        **plot_kwargs:
            Passed to `pv.Plotter.add_mesh`, overriding the style defaults.
        """
        self.mesh = mesh
        self.toggle = toggle
        self.style = style or ("points" if mesh.n_faces == 0 else "surface")

        self._provider, self.states = _make_provider(values, states)
        self.n_states = len(self.states)
        self.scalar_name = scalar_name if self._provider is not None else None

        self.plot_kwargs = _default_style(self.style, self.scalar_name is not None) | plot_kwargs

        if self.scalar_name is not None:
            self.mesh[self.scalar_name] = np.zeros(self.mesh.n_points)
            self.update(0)

    def update(self, idx: int):
        """Repaint this layer's scalar field for state `idx`."""
        if self._provider is not None:
            self.mesh[self.scalar_name][:] = self._provider(idx % self.n_states)
