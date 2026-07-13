"""
Module Name: analysis/connectivity/metric.py
Description: Metrics to evaluate connectomes.

Thin wrappers around `mindmap.core.metric`'s graph metrics for objects
used elsewhere in this section (mne.SourceSpaces, Parcellation): they
extract dipole coordinates/labels and delegate the actual computation to
`mindmap.core.metric`, which only needs plain numpy arrays and stays
reusable outside of this section. `density`, `small_world_sigma`,
`rich_club_curve`, `modularity`, `rewire_connectome`,
`perturb_connectome_weights` and `permute_connectome_weights` need no such
wrapping -- they already take a bare adjacency matrix -- so they are
re-exported as-is.

See `process.py`'s module docstring for the literature and target ranges
behind these metrics.

Author: Nino Herve
Created: 2026-07-06
Last Modified: 2026-07-06
License: MIT
"""

import numpy as np

import mne

from mindmap.core.metric import (
    density,
    small_world_sigma,
    rich_club_curve,
    modularity,
    distance_decay_fit,
    hemispheric_symmetry,
    block_conservation,
    rewire_connectome,
    perturb_connectome_weights,
    permute_connectome_weights,
)
from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.source_space.io import get_dipole_positions, get_hemi_labels
from mindmap.analysis.connectivity.utils import get_dipole_region_labels


def distance_decay_fit_from_source_space(adj: np.ndarray, source_space: mne.SourceSpaces, n_bins: int = 30) -> dict:
    """
    `mindmap.core.metric.distance_decay_fit`, using dipole positions from
    a source space (native tkras space, [mm]) as node coordinates.
    """
    dipoles = get_dipole_positions(source_space) * 1000   # [m] -> [mm]
    return distance_decay_fit(adj, dipoles, n_bins=n_bins)


def hemispheric_symmetry_from_source_space(adj: np.ndarray, source_space: mne.SourceSpaces) -> dict:
    """
    `mindmap.core.metric.hemispheric_symmetry`, using dipole positions
    and hemisphere labels from a source space (native tkras space, where
    the midline is x=0).
    """
    dipoles = get_dipole_positions(source_space) * 1000   # [m] -> [mm]
    hemi = get_hemi_labels(source_space)
    return hemispheric_symmetry(adj, dipoles, hemi)


def block_conservation_from_atlas(
    fine_adj: np.ndarray,
    source_space: mne.SourceSpaces,
    parcellation: Parcellation,
    ras2tkras: np.ndarray,
    atlas_adj: np.ndarray,
) -> dict:
    """
    `mindmap.core.metric.block_conservation`, labeling dipoles by atlas
    region via the parcellation (see `connectome_from_atlas`, which this
    checks the output of).
    """
    dipole_labels = get_dipole_region_labels(source_space, parcellation, ras2tkras)
    return block_conservation(fine_adj, dipole_labels, atlas_adj)


__all__ = [
    "density",
    "small_world_sigma",
    "rich_club_curve",
    "modularity",
    "distance_decay_fit_from_source_space",
    "hemispheric_symmetry_from_source_space",
    "block_conservation_from_atlas",
    "rewire_connectome",
    "perturb_connectome_weights",
    "permute_connectome_weights",
]
