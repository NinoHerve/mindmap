"""
Module Name: analysis/connectivity/processing.py
Description: Functions to process connectomes.

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

from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.tractogram.processing import tract_endpoints, tract_lengths
from mindmap.anatomy.utils import align_diffusion2subject
from mindmap.anatomy.source_space.io import get_ndipole, get_dipole_positions


def connectome_from_tracts(
        tracts    : nib.streamlines.TrkFile,
        regions   : Parcellation | mne.SourceSpaces,
        t12diff   : np.ndarray,
        ras2tkras : np.ndarray = None,
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
    
    ras2tkras: np.ndarray (4,4)
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
        assert ras2tkras is not None, "tkras2ras is required when using source spaces in order to map dipole positions to the T1-ras space."
        start_labels, end_labels = _label_by_dipole(starts, ends, regions, ras2tkras)
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
        ras2tkras    : np.ndarray,
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
    dipoles = nib.affines.apply_affine(np.linalg.inv(ras2tkras), dipoles)

    # find closest dipole using nearest neighbor
    kd_tree         = sp.spatial.cKDTree(dipoles)
    _, start_labels = kd_tree.query(starts)
    _, end_labels   = kd_tree.query(ends)

    return start_labels, end_labels


def connectome_from_geometry(
        source_space    : mne.SourceSpaces,
        maximum_dist    : float = 10,
        maximum_neigh   : int = None,
        inter_hemi_dist : float = 0,
        weights         : bool = True
        ) -> np.ndarray:
    """
    Build adjacency matrix from brain geometry.

    Parameters
    ----------
    source_space: mne.SourceSpace
        Sources.

    maximum_dist: float
        Maximal distance in [mm] at which to connect neighbors.

    maximum_neigh: int
        Maximum number of neighbors. This will select the N closest neighbors only. But a 
        node can still end up with more connections if it is the closest neighbor of other
        nodes.
        
    inter_hemi_dist: float
        Maximal distance in [mm] at which to connect inter hemispheric nodes.

    weights: bool
        Whether to put weights as geodesic distances (in [mm]) or binarize.
    
    Returns
    -------
    geometry: sp.sparce.csc_matrix
        Brain geometry at source level. 
    """
    hemispheres = []

    for hemi in [0,1]:  # 0 for left hemisphere and 1 for right hemisphere

        # get geodesic distance matrix of used nodes
        vertices = source_space[hemi]["vertno"]
        dist     = source_space[hemi]["dist"][vertices][:,vertices].todense() * 1000     # in [mm]

        # threshold according to dmax
        dist[dist > maximum_dist] = np.inf

        # choose only k neerest connections
        if maximum_neigh is not None:
            # indices of k closest nodes
            knn_indices = np.argpartition(dist, maximum_neigh, axis=1)[:,:maximum_neigh]

            # create symmetric mask of nearest neighbors
            mask = np.zeros_like(dist, dtype=bool)
            rows = np.arange(dist.shape[0])[:,None]
            mask[rows, knn_indices] = True

            sym_mask = np.logical_or(mask, mask.T)

            # apply mask
            dist = np.where(sym_mask, dist, np.inf)

        hemispheres.append(dist)

    # connect hemispheres
    n = source_space[0]["nuse"]

    # add inter hemispheric connections
    pos    = get_dipole_positions(source_space) * 1000  # in [mm]
    lh_src = pos[:n]
    rh_src = pos[n:]

    inter_dist = np.linalg.norm(lh_src[:, np.newaxis, :]-rh_src[np.newaxis, :, :], axis=2)
    inter_dist[inter_dist > inter_hemi_dist] = np.inf

    # geometry
    geometry = np.block([
        [hemispheres[0],  inter_dist    ],
        [inter_dist.T,    hemispheres[1]]
    ])

    # replace infinities with zeros, remove self loops
    geometry[geometry == np.inf] = 0
    np.fill_diagonal(geometry, 0)

    if not weights:
        return (geometry > 0).astype(int)

    return geometry


