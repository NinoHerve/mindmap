"""
Module Name: analysis/connectivity/utils.py
Description: Useful functions for connectomes.

This module provides functions to handle connectomes such as 
building them from tractograms.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import mne

from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.source_space.io import get_dipole_positions
from mindmap.anatomy.utils import align_freesurfer2subject


def get_transformed_dipoles(source_space: mne.SourceSpaces, ras2tkras: np.ndarray) -> np.ndarray:
    """
    Dipole positions transformed into the parcellation/subject RAS space.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space. Contains dipole positions in freesurfer space
        (tkras) in [m].

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform matrix from FreeSurfer to subject RAS.

    Returns
    -------
    transformed_dipoles: np.ndarray of shape (n_dipoles, 3)
        Dipole positions in the parcellation/subject RAS space, in [mm].
    """
    dipoles = get_dipole_positions(source_space) * 1000   # [m] -> [mm]
    return align_freesurfer2subject(dipoles, ras2tkras)


def get_dipole_region_labels(
    source_space: mne.SourceSpaces,
    parcellation: Parcellation,
    ras2tkras: np.ndarray,
    return_id: bool = True,
) -> np.ndarray:
    """
    Label every dipole in a source space by its parcellation region.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space. Contains dipole positions in freesurfer space
        (tkras) in [m].

    parcellation: Parcellation
        Parcellation in T1-ras space in [mm].

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform matrix from FreeSurfer to subject RAS.

    return_id: bool
        Returns label id rather than label name if True (see
        `Parcellation.label_points`).

    Returns
    -------
    labels: np.ndarray of shape (n_dipoles,)
        Region label (or id) per dipole.
    """
    transformed_dipoles = get_transformed_dipoles(source_space, ras2tkras)

    return parcellation.label_points(transformed_dipoles, return_id=return_id)


def print_connectome_summary(connectome: np.ndarray) -> None:
    """
    Print various information about the connectome matrix.
    """
    # compute number of connections per node
    connections_per_node = (connectome > 0).sum(axis=1)

    n = len(connectome)
    e = np.sum(connections_per_node)/2

    print(f"Number of nodes: {n}")
    print(f"Total number of connections: {e} (sparsity = {2*e/(n*n-n):.3}%)")
    print(f"Minimum (non-zero) weight: {connectome[connectome>0].min()}")
    print(f"Maximum weight: {connectome.max()}")
    print(f"Number of isolated nodes: {np.sum(connections_per_node==0)} (ratio = {np.sum(connections_per_node==0)/n*100:.3})%")


def order_connectome_by_region(
    source_connectome: np.ndarray,
    source_space: mne.SourceSpaces,
    parcellation: Parcellation,
    ras2tkras: np.ndarray,
):
    """
    Reorders the rows and columns of a source-space connectome so that
    dipoles are grouped sequentially by their parcellation region index.
    
    Parameters:
    ----------
    source_connectome: np.ndarray of shape (n_dipoles, n_dipoles)
        The unordered adjacency matrix.
    
    source_space: mne.SourceSpaces
        The MNE source space object.

    parcellation: Parcellation object
        Object with .label_points(points, return_id=True) method.

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform matrix from FreeSurfer to subject RAS.

    Returns:
    -------
    ordered_connectome: np.ndarray of shape (n_dipoles, n_dipoles)
        The reordered connectome matrix grouped by region.

    sorted_indices: np.ndarray of shape (n_dipoles,)
        The index mapping array used for the reordering (useful fro tracking dipole IDs)
    """
    # 1. Get region label IDs for all dipoles
    dipole_labels = get_dipole_region_labels(source_space, parcellation, ras2tkras)

    # 2. Find the permutation vector that sorts the dipoles by region ID
    sorted_indices = np.argsort(dipole_labels)

    # 5. Reorder the matrix symmetrically using advanced indexing
    ordered_connectome = source_connectome[np.ix_(sorted_indices, sorted_indices)]

    return ordered_connectome, sorted_indices