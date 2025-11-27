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
import sklearn
import nibabel as nib
import mne
import pygsp

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


def compute_cst(connectome):
    """
    Compute the connectome spectrum transform.

    Parameters
    ----------
    connectome: np.ndarray (n,n)
        Connectome adjacency matrix.

    Returns
    -------
    cst: np.ndarray (n,n)
        Connectome spectrum transform (eigenvectors or normalized Laplacian).
    
    freqs: np.ndarray (n,)
        Spectrum frequencies (eigenvalues of normalized Laplacian).
    """
    
    graph = pygsp.graphs.Graph(connectome, lap_type="normalized")   # pylint: disable=no-member
    graph.compute_fourier_basis()

    return graph.U, graph.e




# Smoothing stuff, see if I used it someday


def smooth_connectome(conn, kernel):
    """
    Smoothen connectome according to Sina Mansour et. al (2020)

    Parameters
    ----------
    conn: np.ndarray (N,N)
        Connectome.

    geodist: sp.sparse.csr (N, N)
        Geodesic distances between nodes.

    dmax: float
        Maximum neigborhood distance.

    fwhm: float
        Full width half maximum. 

    epsilon: float
        Smoothing radius parameter.

    dim:
        Smoothing dimensionality (surface is 2D)
    """
    # define incidence matrices
    indices = np.argwhere(conn>0)
    n, s    = len(conn), len(indices)
    M1, M2  = np.zeros((n,s)), np.zeros((n,s))

    for i, j in enumerate(indices):
        M1[j[0], i] = np.sqrt(conn[j[0],j[1]])
        M2[j[1], i] = np.sqrt(conn[j[1],j[0]])

    # smooth incidence
    M1s = (kernel.T @ M1)
    M2s = (kernel.T @ M2)

    # compute smoothed structural connectome
    sc = (M1s @ M2s.T) + (M2s @ M1s.T)
    sc = (sc + sc.T)/2
    np.fill_diagonal(sc, 0)

    return sc

def gaussian_kernel(geodist, fwhm, epsilon, dim):
    """
    Create Gaussian kernels for connectome smoothing.

    Parameters
    ----------
    geodist: sp.sparse.csr
        Geodesic distances between nodes.

    fwhm: float
        Full width half maximum. 

    epsilon: float
        Smoothing radius parameter.

    dim:
        Smoothing dimensionality (surface is 2D)

    Returns
    ------- 
    gaussian: sp.sparse.csr (N, N)
        Gaussian kernel for each node. The kernel distributes a weight arriving at node i between its neighbors.
    """
    # gaussian variance
    sigma = fwhm / np.sqrt(8*np.log(2))

    # maximum smoothing distance
    #max_smoothing_distance = sigma * (-sp.stats.norm.ppf((1-(1-epsilon)**(1/dim)) /1))
    max_smoothing_distance = sigma * (-2 * np.log(epsilon)) ** (1/dim)
    geodist[geodist>max_smoothing_distance] = 0
    geodist.eliminate_zeros()

    # Smoothing kernel for each node
    gaussian = -(geodist**2/(2*sigma**2))
    np.exp(gaussian.data, out=gaussian.data)
    gaussian += sp.sparse.eye(gaussian.shape[0], dtype=gaussian.dtype, format="csr")
    gaussian /= sp.sparse.linalg.norm(gaussian, ord=1, axis=1)[:, None]
    gaussian = sklearn.preprocessing.normalize(gaussian, norm="l1")     # sums to 1

    return gaussian

def unit_kernel(geodist, max_distance):
    """
    Create unit kernel for connectome smoothing.

    Parameters
    ----------
    geodist: sp.sparse.csr
        Geodesic distances between nodes.

    max_distance:
        Maximum width in [mm] of kernel.

    Returns
    -------
    unit_kernel: sp.sparse.csr (N,N)
        Unit kernel. The kernel of a node of interest is 1 for every neighbor node,
        meaning it is below `max_distance` from the node of interest.
    """
    # apply maximum smoothing distance
    geodist[geodist>max_distance] = 0
    geodist.eliminate_zeros()

    kernel = geodist > 0
    kernel.setdiag(1)   # kernel is 1 if neighborhood else 0

    return kernel


# generate weights
def harmonic_weights(eigvals, relaxation=1):
    """
    Generate harmonic weights from eigenvalues.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics.
    
    relaxation: float [0-1]
        Relaxes the small weights. The weight will give the probability of a 
        harmonic to be chosen during reconstruction. If taking the raw weights,
        this can be pretty tough on middle harmonics. The relaxation allows 
        middle harmonics to have more weigths. When relaxation = 0, all harmonics
        have a weigth of 1.
    """
    arr1 = np.abs(eigvals - np.roll(eigvals, -1))
    arr2 = np.abs(eigvals - np.roll(eigvals, 1))

    weights = (arr1 + arr2)/2

    # correct extremities
    weights[0]  = arr1[0]
    weights[-1] = arr2[-1]

    return weights**relaxation

# harmonic prior
def compute_harmonic_prior(eigvals, relaxation = 1):
    """
    Compute the harmonic prior according to eigenvalues.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics.
    
    relaxation: float [0-1]
        Relaxes the small weights. The weight will give the probability of a 
        harmonic to be chosen during reconstruction. If taking the raw weights,
        this can be pretty tough on middle harmonics. The relaxation allows 
        middle harmonics to have more weigths. When relaxation = 0, all harmonics
        have a weigth of 1.
    """

    # compute eigenvalue gaps
    arr1 = np.abs(eigvals - np.roll(eigvals, -1))
    arr2 = np.abs(eigvals - np.roll(eigvals, 1))

    weights = (arr1 + arr2)/2

    # correct extremities
    weights[0]  = arr1[0]
    weights[-1] = arr2[-1]

    # convert to prior weights and store in prior mixing matrix
    weights = np.diag(1/weights)**relaxation

    return weights