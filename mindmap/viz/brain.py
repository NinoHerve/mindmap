"""
Module Name: viz/brain.py
Description: `BrainPlot` - the brain-specific plot shortcut. Takes
surfaces directly (FreeSurfer white/pial/inflated, or any mesh) as its
primary constructor, with `from_source_space` as an alternate constructor
for dipole-space source-reconstruction results.

Author: Nino Herve
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pyvista as pv
import mne

from mindmap.geometry.io import surf_to_polydata
from mindmap.anatomy.source_space.process import build_dense_vertex_map
from mindmap.core.process import paint_on_dense_mesh
from mindmap.viz.mesh import Layer, Provider, find_clim, default_cmap, preprocess
from mindmap.viz.panel import Panel
from mindmap.viz.camera import resolve_hemi_view

HEMI_INDICES = {"lh": [0], "rh": [1], "both": [0, 1]}


def _as_polydata(surface) -> pv.PolyData:
    """Accept a `pv.PolyData` directly, or a `(vertices, faces)` tuple."""
    if isinstance(surface, pv.PolyData):
        return surface

    vertices, faces = surface
    return surf_to_polydata(vertices, faces)


def _as_list(x, n: int) -> list:
    """Broadcast a single value to length `n`, or pass a matching list through."""
    if isinstance(x, list):
        if len(x) != n:
            raise ValueError(f"Expected {n} entries, got {len(x)}.")
        return x

    return [x] * n


def _inpaint_dipole_values(src: mne.SourceSpaces, values, hemis: list) -> list:
    """
    Convert dipole-space `values` (array or callable) into one lazy
    dense-vertex provider per hemisphere in `hemis`, via
    `build_dense_vertex_map` + `paint_on_dense_mesh`.

    Always returns callables, even when `values` is a plain array: painting
    is the expensive step for a long time series (the same concern that
    drove `Layer`'s lazy-provider support in the first place), so this
    avoids precomputing every state at dense resolution up front regardless
    of which form the caller's dipole data took.
    """
    if values is None:
        return [None] * len(hemis)

    vertex_maps = {i: build_dense_vertex_map(src[i]) for i in hemis}

    if hemis == [0, 1]:
        split = src[0]["nuse"]
        slices = {0: slice(0, split), 1: slice(split, None)}
    else:
        slices = {hemis[0]: slice(None)}

    def make_provider(i):
        vmap = vertex_maps[i]
        sl = slices[i]

        def provider(idx):
            col = values(idx) if callable(values) else (
                np.asarray(values)[:, idx] if np.ndim(values) == 2 else np.asarray(values)
            )
            return paint_on_dense_mesh(col[sl], vmap)

        return provider

    return [make_provider(i) for i in hemis]


@dataclass
class BrainPlot(Panel):
    """
    One or two brain surfaces (hemispheres) with an optional scalar map
    and/or state axis, rendered together as one panel.

    Parameters
    ----------
    surfaces: pv.PolyData | (vertices, faces) | list thereof
        One or two surfaces. A `(vertices, faces)` tuple is converted via
        `surf_to_polydata` - the exact shape `dm.load(subject, "white")`
        (or "pial"/"inflated") already returns per hemisphere, and the
        same shape `mne.SourceSpaces`' own `src[i]["rr"]`/`src[i]["tris"]`
        uses (see `from_source_space`).

    values: np.ndarray | Callable[[int], np.ndarray] | list thereof, optional
        Dense, per-vertex scalar field for each surface (broadcast a
        single value to all surfaces, or pass one per surface). This is
        the one shape `BrainPlot`/`Layer` ever understand - dipole-space
        or other non-dense value spaces need an adapter (see
        `from_source_space`) to convert first.

    states: np.ndarray, optional
        Shared axis values (time in seconds, harmonic index, ...) across
        every surface - one brain, one state axis.

    title: str
        Panel title.

    scalar_name: str
        Shared across every surface in this `BrainPlot` (both hemispheres
        are the same quantity) - also the selector `Layout.link_scale`
        matches on. See `SurfacePlot` for the full explanation of the
        "same name = deliberately linked" default trade-off.

    hemi: str | list, optional
        Which hemisphere each surface is ("lh"/"rh"), broadcast like
        `values`. Set automatically by `from_source_space`; only needed
        directly if you want `view` to resolve "lateral"/"medial" against
        surfaces you supplied yourself.

    view: str, optional
        A named camera view (see `viz.camera.get_view`/`resolve_hemi_view`):
        an absolute anatomical direction ("superior", "anterior", ...) or,
        for a single-hemisphere `BrainPlot` with `hemi` set, "lateral"/
        "medial" (resolved relative to that hemisphere). Orthographic,
        auto-framed to this panel's own surfaces - not a fixed number, so
        it works regardless of subject/mesh size. Raises if "lateral"/
        "medial" is requested without a single known hemisphere (it's
        ambiguous across two).

    mesh_params: dict
        Passed to every surface's `Layer`. Unlike `SurfacePlot`, `clim`/
        `cmap` default to one range shared across all surfaces in this
        `BrainPlot` (both hemispheres are one brain, not independent
        plots) - set them explicitly to override.

    func_params: dict
        `normalize`: "max" | "minmax" | "zscore"
        `absolute`: bool
        Applied per-surface, before the shared `clim`/`cmap` is computed.
    """

    surfaces: Union[pv.PolyData, tuple, list]
    values: Union[np.ndarray, Provider, list, None] = None
    states: Optional[np.ndarray] = None
    title: str = ""
    scalar_name: str = "value"
    hemi: Union[str, list, None] = None
    view: Optional[str] = None
    mesh_params: dict = field(default_factory=dict)
    func_params: dict = field(default_factory=dict)

    def __post_init__(self):
        surfaces = self.surfaces if isinstance(self.surfaces, list) else [self.surfaces]
        self.surfaces = [_as_polydata(s) for s in surfaces]
        self.values = _as_list(self.values, len(self.surfaces))
        self.hemi = _as_list(self.hemi, len(self.surfaces))

    @property
    def camera(self) -> Optional[tuple]:
        """`(direction, viewup)` for `view`, or None. See the `view`
        parameter - "lateral"/"medial" need a single known hemisphere to
        resolve; absolute views (e.g. "superior") don't. Framing (focus,
        distance) is computed from the rendered panel itself, not here -
        see `viz.camera.apply_view`."""
        if self.view is None:
            return None

        known_hemis = {h for h in self.hemi if h is not None}
        hemi = next(iter(known_hemis)) if len(known_hemis) == 1 else None

        return resolve_hemi_view(self.view, hemi)

    @property
    def layers(self) -> list[Layer]:
        processed = [preprocess(v, self.func_params) for v in self.values]

        mesh_params = dict(self.mesh_params)
        if not mesh_params.get("clim"):
            mesh_params["clim"] = find_clim([p for p in processed if not callable(p)])
        if not mesh_params.get("cmap"):
            mesh_params["cmap"] = default_cmap(mesh_params["clim"])

        return [
            Layer(surface, values, self.states, scalar_name=self.scalar_name, **mesh_params)
            for surface, values in zip(self.surfaces, processed)
        ]

    @classmethod
    def from_source_space(
        cls,
        src: mne.SourceSpaces,
        values: Union[np.ndarray, Provider, None] = None,
        states: Optional[np.ndarray] = None,
        hemi: str = "both",
        **kwargs,
    ) -> "BrainPlot":
        """
        Build a `BrainPlot` from an `mne.SourceSpaces` and dipole-space
        functional data (e.g. an `mne.SourceEstimate`'s `.data`).

        This is an alternate constructor, not a method called on an
        existing `BrainPlot` - it derives `surfaces`/`values` from `src`
        and calls the same `__init__` a direct `BrainPlot(...)` call
        would, so the object it returns is indistinguishable from one
        built directly.

        Parameters
        ----------
        src: mne.SourceSpaces
            Source space - its dense FreeSurfer mesh (`src[i]["rr"]`/
            `src[i]["tris"]`) is what actually gets plotted.

        values: np.ndarray | Callable[[int], np.ndarray], optional
            Dipole-space functional, ordered [lh dipoles, rh dipoles] for
            `hemi="both"` (split at `src[0]["nuse"]`). Inpainted onto each
            hemisphere's dense mesh lazily (see `_inpaint_dipole_values`)
            regardless of whether this is a plain array or already a
            callable.

        states: np.ndarray, optional
            e.g. an `mne.SourceEstimate`'s `.times`.

        hemi: {"both", "lh", "rh"}
            Which hemisphere(s) to build - "lh"/"rh" skip mapping the
            other hemisphere entirely, unblocking single-hemisphere plots.

        **kwargs:
            Passed to `BrainPlot` (`title`, `scalar_name`, `view`,
            `mesh_params`, `func_params`) - `hemi`/`surfaces`/`values`/
            `states` are already handled by this constructor and can't be
            passed again.

        Returns
        -------
        BrainPlot
        """
        hemis = HEMI_INDICES[hemi]
        # mne.SourceSpaces stores `rr` in meters; every other surface in
        # mindmap (read_bem, dm.load(subject, "white"/"pial"), ...) is in
        # FreeSurfer's native mm - scale here so a from_source_space
        # BrainPlot composes correctly (e.g. `+ HeadModelPlot(bem)`).
        surfaces = [surf_to_polydata(src[i]["rr"] * 1000, src[i]["tris"]) for i in hemis]
        dense_values = _inpaint_dipole_values(src, values, hemis)
        hemi_labels = ["lh" if i == 0 else "rh" for i in hemis]

        return cls(surfaces, dense_values, states, hemi=hemi_labels, **kwargs)
