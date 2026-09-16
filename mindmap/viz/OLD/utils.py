"""
Module Name: viz/utils.py
Description: Contains plot utilities.

Contains functions to read data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from abc import ABC, abstractmethod

import numpy as np

import pyvista as pv

import mne

from mindmap.geometry.io import surf_to_polydata
from mindmap.analysis.functional.eeg.io import get_ch_positions
from mindmap.anatomy.source_space.process import build_dense_vertex_map
from mindmap.core.process import paint_on_dense_mesh

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


class BaseBrainPlotter:
    """Base class for interactive brain visualization."""
    def __init__(self, plotter=None, n_states=1):
        self.plotter = plotter or pv.Plotter()
        self.current_idx = 0
        self.n_states = n_states


    def setup_keys(self, fast_step: int = None, super_step: int = None):
        """
        Builds key event to iterate through samples.

        Parameters
        ----------
        fast_step: int
            A fast step event associated to 'd' and 'a' to iterate 
            by batch (e.g. 50 -> will go 50 by 50)

        super_step: int
            Additional step event associated to 'c' and 'y' to iterate
            at an additional speed.
        """
        self.plotter.add_key_event("Right", lambda: self.step(1))
        self.plotter.add_key_event("Left", lambda: self.step(-1))

        if fast_step:
            self.plotter.add_key_event("d", lambda: self.step(fast_step))
            self.plotter.add_key_event("a", lambda: self.step(-fast_step))

        if super_step:
            self.plotter.add_key_event("c", lambda: self.step(super_step))
            self.plotter.add_key_event("y", lambda: self.step(-super_step))


    def step(self, delta):
        """Step through sample indices"""
        new_idx = max(0, min(self.current_idx + delta, self.n_states - 1))
        if new_idx != self.current_idx:
            self.current_idx = new_idx

            # If a slider exists, sync it
            if hasattr(self, "slider"):
                self.slider.SetValue(self.idx_to_value(new_idx))

            self.update_display(new_idx)


    def add_slider(self, label="Value", rng=None):
        """Standardized slider setup."""
        if rng is None:
            rng = [0, self.n_states -1]

        slider = self.plotter.add_slider_widget(
            callback = self._on_slider_change,
            rng = rng,
            value = rng[0],
            title = label,
            style = "modern",
            interaction_event = "always",
        )

        self.slider = slider.GetSliderRepresentation()


    def _on_slider_change(self, value):
        """Map slider value back to an integer index."""
        idx = self.value_to_idx(value)
        if idx != self.current_idx:
            self.current_idx = idx
            self.update_display(idx)


    def show(self):
        """Show plot"""
        self.plotter.show()


    # ---------------------------------------
    # Methods to be overridden by Subclasses
    # ---------------------------------------


    def value_to_idx(self, value):
        """Convert slider value (float) to data index (int). Defaults to 1:1"""
        return int(round(value))


    def idx_to_value(self, idx):
        """Convert index to slider value. Defaults to 1:1"""
        return float(idx)


    def update_display(self, idx):
        """The specific mesh/PC update logic"""
        raise NotImplementedError



def build_hemi_pair(src):
    """Build left/right hemisphere meshes and vertex to dipole mappings."""
    # extract geometry
    hemi_l = surf_to_polydata(src[0]["rr"], src[0]["tris"])
    hemi_r = surf_to_polydata(src[1]["rr"], src[1]["tris"])

    # dipole-vertex mapping
    map_l = build_dense_vertex_map(src[0])
    map_r = build_dense_vertex_map(src[1])

    # initialize scalar
    hemi_l["signal"] = np.zeros(hemi_l.n_points)
    hemi_r["signal"] = np.zeros(hemi_r.n_points)

    return hemi_l, hemi_r, map_l, map_r, src[0]["nuse"]


def create_line_mesh(pts: np.ndarray) -> pv.PolyData:
    """Create a polydata line"""
    if len(pts) < 2:
        return pv.PolyData(pts)

    cells = np.hstack(([len(pts)], np.arange(len(pts))))

    return pv.PolyData(pts, lines=cells)


class BaseMesh(ABC):
    """
    Interface for all renderable objects in TimeMesh.
    """
    def __init__(self, functional=None, times=None, scalar_name="signal"):

        # standardize functional
        functional = np.asarray(functional) if functional is not None else None

        if (functional is None) or (np.all(functional == 0) and functional.shape[1] == 1):
            self.scalar_name = None
            self._data = None
            self.times = None

        else:
            self.scalar_name = scalar_name

            if functional.ndim == 1:
                self._data = functional[:, np.newaxis]
            else:
                self._data = functional

            if times is not None:
                self.times = np.atleast_1d(times)
            else:
                self.times = np.arange(self._data.shape[1])

    @property
    def n_times(self):
        """Number of time points."""
        if self._data is None:
            return 0

        return self._data.shape[1]

    def get_column(self, idx):
        """Returns 0 if static, otherwise the current index."""
        if self._data is None:
            return None

        return self._data[:, idx % self.n_times].flatten()

    @abstractmethod
    def update_signal(self, idx: int):
        """Must be implemented by subclasses to update PolyData scalars."""
        pass


class BrainMesh(BaseMesh):
    """
    Class to manage brain meshes.
    """
    def __init__(self, src, functional=None, times=None, scalar_name="functional", **kwargs):
        """
        Parameters:
        ----------
        src: mne.SourceSpaces
            Source spaces. Contains FreeSurfer brain mesh.

        functional: np.ndarray
            Brain functional of shape (n_dipoles, n_timepoints) or (n_dipoles,)

        time: np.ndarray
            Time points.
        """
        # Geometry
        self.hemi_l = surf_to_polydata(src[0]["rr"], src[0]["tris"])
        self.hemi_r = surf_to_polydata(src[1]["rr"], src[1]["tris"])

        # Mapping
        self.map_l = build_dense_vertex_map(src[0])
        self.map_r = build_dense_vertex_map(src[1])
        self.split = src[0]["nuse"]

        super().__init__(times=times, functional=functional, scalar_name=scalar_name)

        # default aesthetics for brains
        if self.scalar_name is None:
            self.plot_kwargs = {
                "smooth_shading": True,
                "color": "white",
                "opacity": 0.5,
            } | kwargs

        else:
            self.plot_kwargs = {
                "smooth_shading": True,
                "cmap": "Reds",
            } | kwargs

            # Initialize PolyData
            self.hemi_l[self.scalar_name] = np.zeros(self.hemi_l.n_points)
            self.hemi_r[self.scalar_name] = np.zeros(self.hemi_r.n_points)
            self.update_signal(0)
            
    def update_signal(self, idx):
        """Maps signal to the actual PolyData objects for plotting."""

        if self.scalar_name is not None:
            signal = self.get_column(idx)

            # apply new signal
            self.hemi_l[self.scalar_name][:] = paint_on_dense_mesh(signal[:self.split], self.map_l)
            self.hemi_r[self.scalar_name][:] = paint_on_dense_mesh(signal[self.split:], self.map_r)


class SurfaceMesh(BaseMesh):
    """
    Class to manage a single arbitrary surface mesh with an optional
    functional map. Unlike `BrainMesh`, this is not tied to a two-hemisphere
    `mne.SourceSpaces` - it takes a bare `pv.PolyData`, so it can represent
    one hemisphere, a non-brain surface, or any custom mesh.
    """
    def __init__(self, polydata, functional=None, times=None, vertex_map=None, scalar_name="functional", **kwargs):
        """
        Parameters
        ----------
        polydata: pv.PolyData
            Surface mesh (e.g. one hemisphere, a whole head, or any custom surface).

        functional: np.ndarray
            Functional of shape (n, n_timepoints) or (n,). If `vertex_map` is
            given, `n` is the number of dipoles (sparse, painted onto the
            surface). Otherwise `n` must equal `polydata.n_points` (already
            dense, per-vertex).

        times: np.ndarray
            Time points.

        vertex_map: np.ndarray, optional
            Per-dense-vertex index into `functional` (its nearest dipole), or
            -1 for vertices whose nearest dipole isn't in-use. When given,
            `functional` is broadcast onto `polydata` via `paint_on_dense_mesh`.
            See `mindmap.anatomy.source_space.build_dense_vertex_map`.
        """
        self.polydata = polydata
        self.vertex_map = vertex_map

        super().__init__(times=times, functional=functional, scalar_name=scalar_name)

        # default aesthetics
        if self.scalar_name is None:
            self.plot_kwargs = {
                "smooth_shading": True,
                "color": "white",
                "opacity": 0.5,
            } | kwargs

        else:
            self.plot_kwargs = {
                "smooth_shading": True,
                "cmap": "Reds",
            } | kwargs

            # Initialize PolyData
            self.polydata[self.scalar_name] = np.zeros(self.polydata.n_points)
            self.update_signal(0)

    def update_signal(self, idx):
        """Maps signal to the actual PolyData object for plotting."""

        if self.scalar_name is not None:
            signal = self.get_column(idx)

            if self.vertex_map is not None:
                signal = paint_on_dense_mesh(signal, self.vertex_map)

            self.polydata[self.scalar_name][:] = signal


def build_hemi_meshes(
    src,
    functional=None,
    times=None,
    hemi="both",
    scalar_name="functional",
    **kwargs,
) -> list:
    """
    Build per-hemisphere `SurfaceMesh` objects from an `mne.SourceSpaces`.

    This is the source-space-aware wrapper around `SurfaceMesh`: it does the
    dipole-to-dense-vertex mapping for one or both hemispheres, so a caller
    only needs to hand it a source space (and optionally restrict to a
    single hemisphere) instead of building meshes by hand.

    Parameters
    ----------
    src: mne.SourceSpaces
        Source space (both hemispheres).

    functional: np.ndarray, optional
        Dipole-space functional of shape (n_dipoles, n_timepoints) or
        (n_dipoles,), ordered [left hemisphere dipoles, right hemisphere
        dipoles]. Split at `src[0]["nuse"]` when `hemi="both"`.

    times: np.ndarray, optional
        Time points, shared across hemispheres.

    hemi: {"both", "lh", "rh"}
        Which hemisphere(s) to build. "lh"/"rh" build a single `SurfaceMesh`
        for that hemisphere only (no split needed).

    scalar_name: str
        Name assigned to the painted scalar array.

    Returns
    -------
    list[SurfaceMesh]
        One mesh for "lh"/"rh", two (left, then right) for "both".
    """
    if hemi not in ("both", "lh", "rh"):
        raise ValueError(f"hemi must be 'both', 'lh' or 'rh', got {hemi!r}")

    def _one(hemi_src, func):
        return SurfaceMesh(
            polydata = surf_to_polydata(hemi_src["rr"], hemi_src["tris"]),
            functional = func,
            times = times,
            vertex_map = build_dense_vertex_map(hemi_src),
            scalar_name = scalar_name,
            **kwargs,
        )

    if hemi == "lh":
        return [_one(src[0], functional)]
    if hemi == "rh":
        return [_one(src[1], functional)]

    split = src[0]["nuse"]
    func_l = functional[:split] if functional is not None else None
    func_r = functional[split:] if functional is not None else None
    return [_one(src[0], func_l), _one(src[1], func_r)]


class SourceMesh(BrainMesh):
    """
    Class to manage SourceEstimates.
    """
    def __init__(self, src, stc, scalar_name="source_activity", **kwargs):
        """
        Parameters:
        ----------
        src: mne.SourceSpaces
            Source spaces. Contains FreeSurfer brain mesh.

        stc: mne.SourceEstimate
            Source time currents.

        time: np.ndarray
            Time points.
        """
        super().__init__(
            src = src,
            functional = stc.data,
            times = stc.times,
            scalar_name = scalar_name,
            **kwargs,
        )

        self.subject = stc.subject


class ChannelMesh(BaseMesh):
    """
    Class to manage electrode/channel point clouds.
    """
    def __init__(self, evoked, scalar_name="electrode_potential", **kwargs):
        super().__init__(times=evoked.times, functional=evoked.data, scalar_name=scalar_name)

        # geometry
        self.polydata = pv.PolyData(get_ch_positions(evoked))
        self.polydata[self.scalar_name] = self.get_column(0)

        # Default aesthetics for sensors
        self.plot_kwargs = {
            "point_size": 10,
            "render_points_as_spheres": True,
            "cmap": "bkr",
        } | kwargs


    def update_signal(self, idx):
        """Update the point cloud scalars for the given time index."""
        self.polydata[self.scalar_name] = self.get_column(idx)


class OverlayMesh(BaseMesh):
    """Class to manage 3D overlayers."""
    def __init__(self, polydata, scalar_name="overlay", toggle=False, **kwargs):
        super().__init__(times=[0], functional=np.zeros((polydata.n_points, 1)), scalar_name=scalar_name)
        self.polydata = polydata
        self.toggle = toggle  # show a checkbox to hide/show this overlay (see TimeMesh)

        # Default aesthetics for overlays
        self.plot_kwargs = {
            "color" : "green",
            "label" : None,
            "point_size" : 1,
            "opacity" : 1,
        } | kwargs

    def update_signal(self, idx):
        pass


class HeadMesh(BaseMesh):
    """Class to manage Head meshes."""
    def __init__(self, polydata, scalar_name="head", **kwargs):
        super().__init__(times=[0], functional=np.zeros((polydata.n_points, 1)), scalar_name=scalar_name)
        self.polydata = polydata

        # Default aesthetics for head mesh
        self.plot_kwargs = {
            "color": "black",
            "label": None,
            "line_width": 1,
            "style": "wireframe",
        } | kwargs

    def update_signal(self, idx):
        pass


def get_brain_view(view_name: str) -> list:
    """
    Returns PyVista camera coordinates based on neuroscientific and anatomical viewing angles.

    Parameters
    ----------
    view_name: str
        The requested anatomical view.
        Supported terms:
        - 'Lateral Left' / 'Left'
        - 'Lateral Right' / 'Right'
        - 'Anterior' / 'Front' / 'Rostral'
        - 'Posterior' / 'Back' / 'Caudal'
        - 'Superior' / 'Top' / 'Dorsal'
        - 'Inferior' / 'Down' / 'Ventral'
    
    Returns
    -------
    list
        A PyVista camera position sequence: [Camera_XYZ, Focus_XYZ, ViewUp_XYZ]
    """
    # Normalize input string to ignore casing or spaces
    view_clean = view_name.strip().lower()

    # Define focus point (origin) and the ViewUp vector (Z-axis points up by default)
    focus = (0.0, 0.0, 0.0)
    up_default = (0.0, 0.0, 1.0)
    up_top = (0.0, 1.0, 0.0)

    # Mapping dict containing primary neuroscientific terms and common aliases
    views = {
        # --- Lateral Views (Sides) ---
        "lateral left": [(-2.0, 0.0, 0.0), focus, up_default],
        "left": [(-2.0, 0.0, 0.0), focus, up_default],

        "lateral rigth": [(2.0, 0.0, 0.0), focus, up_default],
        "right": [(2.0, 0.0, 0.0), focus, up_default],

        # --- Anterior / Posterior (Front / Back) ---
        "anterior": [(0.0, 2.0, 0.0), focus, up_default],
        "front": [(0.0, 2.0, 0.0), focus, up_default],
        "rostral": [(0.0, 2.0, 0.0), focus, up_default],

        "posterior": [(0.0, -2.0, 0.0), focus, up_default],
        "back": [(0.0, -2.0, 0.0), focus, up_default],
        "caudal": [(0.0, -2.0, 0.0), focus, up_default],

        # --- Superior / Inferior (Top / Bottom) ---
        "superior": [(0.0, 0.0, 2.0), focus, up_top],
        "top": [(0.0, 0.0, 2.0), focus, up_top],
        "dorsal": [(0.0, 0.0, 2.0), focus, up_top],

        "inferior": [(0.0, 0.0, -2.0), focus, up_top],
        "down": [(0.0, 0.0, -2.0), focus, up_top],
        "ventral": [(0.0, 0.0, -2.0), focus, up_top],
    }

    if view_clean not in views:
        valid_options = ", ".join([{f"'{k}'"} for k in ["Lateral Left", "Lateral Right", "Anterior", "Posterior", "Superior", "Inferior"]])
        raise ValueError(f"Unknown view '{view_name}'. Please use standard neuroscientific orientations such as: {valid_options}.")
    
    return views[view_clean]