"""
Module Name: analysis/connectivity/metric.py
Description: Metrics to evaluate connectomes.

Thin wrappers around `mindmap.core.metric`'s graph metrics for objects
used elsewhere in this section (mne.SourceSpaces, Parcellation): they
extract dipole coordinates/labels and delegate the actual computation to
`mindmap.core.metric`, which only needs plain numpy arrays and stays
reusable outside of this section. `density`, `small_world_sigma`,
`rich_club_curve`, `modularity`, `rewire_connectome`, `swap_rewire_connectome`,
`perturb_connectome_weights` and `permute_connectome_weights` need no such
wrapping -- they already take a bare adjacency matrix -- so they are
re-exported as-is. `perturb_connectome_topology` is the one exception in
that list needing a wrapper: `dmri_uncertainty_weights_from_source_space`
turns dipole coordinates into the `deletion_weights`/`addition_weights`
arrays it takes.

See `process.py`'s module docstring for the literature and target ranges
behind these metrics.

Author: Nino Herve
Created: 2026-07-06
Last Modified: 2026-07-06
License: MIT
"""

import numpy as np
from scipy.spatial.distance import cdist

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
    swap_rewire_connectome,
    perturb_connectome_topology,
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


def dmri_uncertainty_weights_from_source_space(adj: np.ndarray, source_space: mne.SourceSpaces) -> tuple[np.ndarray, np.ndarray]:
    """
    `deletion_weights`/`addition_weights` for
    `mindmap.core.metric.perturb_connectome_topology`, calibrated to mimic
    real dMRI/tractography error structure instead of spatially-uniform
    corruption.

    Grounded in the tractography-validation literature (Maier-Hein et al.
    2017, Nat Commun, "The challenge of mapping the human connectome
    based on diffusion tractography", among others): tractography's
    bottleneck is specificity, not sensitivity -- false-positive
    connections dominate, and they're spatially biased short (gyral
    bias: spurious terminations cluster near gyral crowns, i.e. short,
    geometrically-plausible pairs), while false negatives are
    biased long-range (crossing-fiber ambiguity compounds over path
    length, so distant true connections are disproportionately missed).

    `addition_weights` uses this connectome's own empirically-fit
    exponential distance-decay curve (`distance_decay_fit`,
    `P(connected) ~ c * exp(-lambda * d)`) as a short-distance-favoring
    sampling weight for non-edges. `deletion_weights` uses squared
    distance (long-distance-favoring) for existing edges -- squared
    rather than raw distance because a linear weight turned out too weak
    an effect in practice (existing edge lengths don't vary over enough
    orders of magnitude for a linear weight to meaningfully outweigh
    sampling noise at moderate draw counts); this is a tunable choice,
    not a literature-derived constant, and worth revisiting empirically
    against this connectome if the resulting bias looks off.

    Doesn't decide `n_deletions`/`n_additions` -- literature above
    suggests additions should outnumber deletions (e.g. ~2:1), but that
    ratio is a starting point to sanity-check empirically against this
    connectome, not a hard constant, so it's left to the caller.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix used to fit the distance-decay curve (should be
        the same connectome `perturb_connectome_topology` will corrupt).

    source_space: mne.SourceSpaces
        Sources, for dipole coordinates (native tkras space).

    Returns
    -------
    deletion_weights: np.ndarray of shape (n, n)
        Long-distance-favoring (squared-distance) sampling weight for
        existing edges.

    addition_weights: np.ndarray of shape (n, n)
        Short-distance-favoring sampling weight for non-edges.
    """
    dipoles = get_dipole_positions(source_space) * 1000   # [m] -> [mm]
    dist = cdist(dipoles, dipoles)

    lambda_ = distance_decay_fit(adj, dipoles)["lambda_"]

    deletion_weights = dist ** 2
    addition_weights = np.exp(-lambda_ * dist)

    return deletion_weights, addition_weights


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
    "swap_rewire_connectome",
    "perturb_connectome_topology",
    "dmri_uncertainty_weights_from_source_space",
    "perturb_connectome_weights",
    "permute_connectome_weights",
]
