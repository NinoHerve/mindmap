"""
Module Name: connectome.py
Description: Functions to handle connectomes.

This module provides functions to handle connectomes such as 
building them from tractograms.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""
from collections import defaultdict

import numpy as np
import scipy as sp
import nibabel as nib
import mne

from mindmap.head import (
    Parcellation, tract_endpoints, tract_lengths, align_diffusion2subject,
    align_freesurfer2subject, get_ndipole, get_dipole_positions,
)

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

def connectome_from_tracts(
        tracts    : nib.streamlines.TrkFile,
        regions   : Parcellation | mne.SourceSpaces,
        t12diff   : np.ndarray,
        tkras2ras : np.ndarray = None,
        lengths   : bool = False,
        ) -> np.ndarray:
    """
    Compute a connectome from tracts.

    Parameters
    ----------
    tracts: nib.Streamlines.TrkFile
        Tractogram.

    regions: Parcellation | mne.SourceSpaces
        Regions to aggregate tracts onto (brain regions or dipoles).

    t12diff: np.ndarray (4,4)
        Individual space to diffusion MRI space affine transformation.
    
    tkras2ras: np.ndarray (4,4)
        FreeSurfer conform space to individual space affine transform (required when using
        source space only).

    length: bool
        Whether to weigth according to lengths (length is True) or number of fibers (length 
        is False).

    Returns
    -------
    connectome: np.ndarray
        Adjacency matrix indicating neural pathways between brain regions or dipoles.
    """

    # Label tract endpoints to regions
    starts, ends = tract_endpoints(tracts)
    starts = align_diffusion2subject(starts, t12diff)
    ends   = align_diffusion2subject(ends, t12diff)

    if isinstance(regions, Parcellation):
        start_labels, end_labels = _label_by_region(starts, ends, regions)
        n = regions.n_regions
    else:
        assert tkras2ras is not None, "tkras2ras is required when using source spaces in order to map dipole positions to the T1-ras space."
        start_labels, end_labels = _label_by_dipole(starts, ends, regions, tkras2ras)
        n = get_ndipole(regions)

    # build connectivity matrix
    sc = np.zeros((n, n))

    # if lengths is True apply fiber lengths as weights
    if lengths:
        lengths = tract_lengths(tracts)
        sc_dict = defaultdict(list)     # dict to accumulate weights for each pairs

        for start, end, weight in zip(start_labels, end_labels, lengths):
            # skip self-loops
            if start == end:
                continue

            # ensure directed pairs are treated the same
            key = tuple(sorted((start, end)))
            sc_dict[key].append(weight)

        # apply average lengths as weight
        for (start, end), weights in sc_dict.items():
            avg_weight = np.mean(weights)
            sc[start, end] = avg_weight
            sc[end, start] = avg_weight

    else:
        for start, end in zip(start_labels, end_labels):
            # skip self loops
            if start == end:
                continue

            # apply number of tracts as weights
            # ensure undirectedness
            sc[start, end] += 1
            sc[end, start] += 1

    return sc


def _label_by_region(
        starts       : np.ndarray,
        ends         : np.ndarray,
        parcellation : Parcellation,
        ) -> tuple[list, list]:
    """
    Label start and end points by brain regions.

    Parameters
    ----------
    starts: np.ndarray (n,3)
        Starting points of tracts. In T1-ras space in [mm].
    
    ends: np.ndarray (n,3)
        Ending points of tracts. In T1-ras space in [mm].

    parcellation: Parcellation
        Parcellation in T1-ras space in [mm].

    Returns
    -------
    start_labels, end_labels: (np.ndarray (n,3), np.ndarray (n,3))
        Associated brain region id for each endpoint.
    """
    start_labels = parcellation.label_points(starts, return_id=True)
    end_labels   = parcellation.label_points(ends, return_id=True)

    unlabeled = (start_labels==0) | (end_labels==0)
    assert np.all(~unlabeled), "Some tracts can't be labeled to any brain region. Make sure to prune by region properly before, and check if the correct transformation matrices were provided."

    return start_labels, end_labels

def _label_by_dipole(
        starts       : np.ndarray,
        ends         : np.ndarray,
        source_space : mne.SourceSpaces,
        tkras2ras    : np.ndarray,
        ) -> tuple[list, list]:
    """
    Label start and end points by closest dipole.

    Parameters
    ----------
    starts: np.ndarray (n,3)
        Starting points of tracts. In T1-ras space in [mm].
    
    ends: np.ndarray (n,3)
        Ending points of tracts. In T1-ras space in [mm].

    source_space: mne.SourceSpaces
        Source space. Contains dipole positions in freesurfer space (tkras) in [m].

    tkras2ras: np.ndarray (4,4)
        Affine transformation (freesurfer space in [mm] -> T1-ras space in [mm])

    Returns
    -------
    start_labels, end_labels: (np.ndarray (n,3), np.ndarray (n,3))
        Associated dipole id for each endpoint.
    """
    dipoles = get_dipole_positions(source_space) * 1000     # switch from [m] to [m]
    dipoles = align_freesurfer2subject(dipoles, tkras2ras)

    # find closest dipole using nearest neighbor
    kd_tree         = sp.spatial.cKDTree(dipoles)
    _, start_labels = kd_tree.query(starts)
    _, end_labels   = kd_tree.query(ends)

    return start_labels, end_labels
