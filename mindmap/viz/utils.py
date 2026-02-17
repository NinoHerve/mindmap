"""
Module Name: viz/utils.py
Description: Contains plot utilities.

Contains functions to read data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np

import pyvista as pv

import mne

from mindmap.geometry.io import surf_to_polydata

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
