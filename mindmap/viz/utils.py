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
    map_l = _vertex_to_dipole_map(src[0])
    map_r = _vertex_to_dipole_map(src[1])

    # initialize scalar
    hemi_l["signal"] = np.zeros(hemi_l.n_points)
    hemi_r["signal"] = np.zeros(hemi_r.n_points)

    return hemi_l, hemi_r, map_l, map_r, src[0]["nuse"]


def _vertex_to_dipole_map(src_hemi):
    """
    Convert src["nearest"] to a vertex -> dipole index map.

    Vertices whose nearest dipole was masked (not in src["vertno"])
    are assigned -1.
    """
    vertno = np.array(src_hemi["vertno"])
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
        self.map_l = self._vertex_to_dipole_map(src[0])
        self.map_r = self._vertex_to_dipole_map(src[1])
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
            
    @staticmethod
    def _vertex_to_dipole_map(src_hemi):
        """
        Convert src["nearest"] to a vertex -> dipole index map.

        Vertices whose nearest dipole was masked (not in src["vertno"])
        are assigned -1.
        """
        vertno = np.array(src_hemi["vertno"])
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


    def update_signal(self, idx):
        """Maps signal to the actual PolyData objects for plotting."""

        if self.scalar_name is not None:
            signal = self.get_column(idx)

            # append 0 for vertices mapped to -1
            sig_l = np.concatenate([signal[:self.split], [0,0]])
            sig_r = np.concatenate([signal[self.split:], [0,0]])

            # apply new signal
            self.hemi_l[self.scalar_name][:] = sig_l[self.map_l]
            self.hemi_r[self.scalar_name][:] = sig_r[self.map_r]


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
    def __init__(self, polydata, scalar_name="overlay", **kwargs):
        super().__init__(times=[0], functional=np.zeros((polydata.n_points, 1)), scalar_name=scalar_name)
        self.polydata = polydata

        # Default aesthetics for overlays
        self.plot_kwargs = {
            "color" : "green",
            "label" : None,
            "point_size" : 1,
            "opacity" : 1,
        } | kwargs

    def update_signal(self, idx):
        pass
