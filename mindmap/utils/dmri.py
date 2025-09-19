
import random
from pathlib import Path
from collections import defaultdict
import sklearn
import mne
import scipy as sp
import h5py
import numpy as np
import pygsp
import mindmap as mm
from ..utils import Parcellation


# ------------------------------------------------------------------------------
#
# Connectome from a template
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#
# ------------------------------------------------------------------------------

def create_cst_from_template(sub, consistency_thr:int = 40, length_thr:float = 25, number_thr:float = 20, distance_thr: float = 0.010):
    """
    Creates the connectome spectrum transform from a connectome atlas.
    Steps:
    1. Prepare the atlas (threshold according to consistency, fiber lengths and number of fibers)
    2. Project the atlas to subject (extrapolate from ~500 regions to ~5000 regions)
    3. Add close connections (connect dipoles with distance below threshold)
    4. Compute the CST

    Parameters
    ----------
    sub: str
        Subject name.

    consistency_thr: float
        Consistency threshold for atlas.

    length_thr: float
        Fiber length [cm] threshold for atlas.

    number_thr: int
        Number of fiber threshold for atlas.

    distance_thr: float
        Maximum dipole distance [mm] to connect.

    Returns
    -------
    cst: ndarray (n_dipoles, n_dipoles)
        Connectome spectrum transform.

    sc: ndarray (n_dipoles, n_dipoles)
        Structural connectome.
    """
    # Retrieve directories
    mm_dir = mm.io.readers._retrieve_env("MM_DIR")

    # Data
    data = mm.io.read_dataset(sub, "src", "parc")
    std  = mm.io.read_standards("conn")

    # label dipoles
    pos    = mm.utils.get_src_position(data["src"])
    pos    = pos*1000                
    labels = data["parc"].label_points(pos, id=True)   # brain region label for each source.

    # some left/right regions are very close and get mixed up
    labels = mm.utils.correct_labels(labels, data["src"], data["parc"])
    

    ###################
    ## Prepare Atlas ##
    ###################

    consistency = std["conn"]["matrices"]["consistency"]
    length      = std["conn"]["matrices"]["length"]
    number      = std["conn"]["matrices"]["numbStlines"]

    sc_atlas = np.bitwise_and(
        consistency[:,:] > consistency_thr,
        length[:,:] > length_thr, 
        number[:,:] > number_thr,
    )

    ##################################
    ## Extrapolate Atlas to Subject ##
    ##################################
 
    N     = pos.shape[0]
    sc_wm = np.zeros((N,N))

    for i, l in enumerate(labels):
        connected_regions = np.argwhere(sc_atlas[l-1]==1)+1
        connections = np.isin(labels, connected_regions)
        sc_wm[i] = connections 

    sc_wm = sp.sparse.csc_array(sc_wm>0)

    ############################
    ## Add Nearby connections ##
    ############################

    sc_gm = []

    for src in data["src"]:
        vertices = src["vertno"]
        dist     = src["dist"][vertices][:,vertices].tolil()

        # binarize distances
        dist[dist>distance_thr] = 0
        dist.setdiag(0)
        dist = dist.tocsc()
        dist.eliminate_zeros()
        dist = dist.astype(bool).astype(int)

        sc_gm.append(dist)

    sc_gm = sp.sparse.block_diag(sc_gm)

    #########################################################
    ## Combine white matter tracks with gray matter tracks ##
    #########################################################

    sc = sc_wm + sc_gm 
    sc = sc.astype(bool).astype(int)

    ###############################################
    ## Compute the Connectome Spectrum Transform ##
    ###############################################

    coords = [s["rr"][s["vertno"]] for s in data["src"]]
    G = pygsp.graphs.Graph(sc.todense(), lap_type="normalized", coords=coords)
    G.compute_fourier_basis()
    cst = G.U.T

    return cst, sc






def infer_wm_tracks(
        sources: mne.SourceSpaces,
        parcel: Parcellation,
        torig: np.ndarray,
        atlas: str | Path,
        consistency_thresh: float,
        length_thresh: float,
) -> sp.sparse.csc_array:
    """
    Build a connectome for a source space using a connectome atlas.

    Parameters
    ----------
    sources: mne.SourceSpaces
        Source space.

    parcel: Parcellation
        Parcellation for source_space. To associate each source to a region of the 
        atlas, the use must provide a parcellation registered to the same space as
        the source space.

    torig: np.ndarray (4x4)
        Transformation matrix to convert voxel indices to coordinates in the MRI surface 
        RAS space.

    atlas: path-like
        Connectome atlas.

    consistency_thresh: float
        Fiber consistency threshold.

    length_thresh: float
        Fiber length threshold.

    Returns
    -------
    sc: sp.sparse.csc_array
        Subject binary structural connectome of white matter tracks.
    """
    # load atlas
    f = h5py.File(atlas)

    # label sources
    points = [s["rr"][s["vertno"]] for s in sources]
    points = np.concatenate(points)*1000                # in surface RAS space in [mm].
    labels = parcel.get_ids(points, torig)              # brain region label for each source.

    # convert atlas to binary adjacency matrix
    consistency = f["matrices"]["consistency"]
    length = f["matrices"]["length"]

    sc_atlas = np.bitwise_and(consistency[:,:]>consistency_thresh, length[:,:]>length_thresh)

    # create connections between sources
    # (if source A belongs to a region that is connected to the region of source B, then connect both sources.)
    N = points.shape[0]
    sc = np.zeros((N,N))

    for i, l in enumerate(labels):
        connected_regions = np.argwhere(sc_atlas[l-1]==1)+1
        connections = np.isin(labels, connected_regions)
        sc[i] = connections 

    sc = sp.sparse.csc_array(sc>0)

    return sc.astype(int)


def interpol_gm_tracks(
        sources: mne.SourceSpaces,
        dmax: float,
) -> sp.sparse.csc_matrix:
    """
    Interpolate gray matter connections. Sources that have a geodesic distance smaller
    than user-given distance are connected.

    Parameters
    ----------
    sources: mne.SourceSpaces
        Source space.

    dmax: float
        Neighborhood distance to connect sources.
    """
    connectomes = []

    for src in sources:
        vertices = src["vertno"]
        sc       = src["dist"][vertices][:,vertices].tolil()

        # binarize distances
        sc[sc>dmax] = 0
        sc.setdiag(0)
        sc = sc.tocsc()
        sc.eliminate_zeros()
        sc = sc.astype(bool).astype(int)

        connectomes.append(sc)

    sc = sp.sparse.block_diag(connectomes)

    return sc
    

def sabotage_connectome(connectome:np.ndarray, deletions:int = 0, additions:int=0, mask=None):
    """
    Sabotage connectome.

    Parameters
    ----------
    connectome: np.ndarray (N,N)
        Connectome.
    
    deletions: int
        Number of connections to delete.
    
    additions: int
        Number of connections to add.
    
    mask: np.ndarray
        Masks edges that should not be changed.

    Returns
    -------
    connectome: np.ndarray
        Sabotaged connectome.
    """
    conn = connectome.astype(bool)

    if mask is None:
        mask = np.zeros_like(conn, dtype=bool)
    else:
        mask = mask.astype(bool)

    # delete edges
    removable_edges = conn * (~mask)                    # adjacency matrix without edges from mask
    removable_edges = np.triu(removable_edges, k=1)     # only upper triangle
    removable_edges = np.argwhere(removable_edges)      # indices

    edges_to_remove = random.sample(removable_edges.tolist(), deletions)
    rows            = [pair[0] for pair in edges_to_remove]
    cols            = [pair[1] for pair in edges_to_remove]

    conn[rows, cols] = 0
    conn[cols, rows] = 0

    # add edges
    addable_edges = ~conn                   # adjacency matrix of no-connections
    np.fill_diagonal(addable_edges, False)              # avoid self-loop    
    addable_edges = np.triu(addable_edges, k=1)
    addable_edges = np.argwhere(addable_edges)
    
    edges_to_add = random.sample(addable_edges.tolist(), additions)
    rows         = [pair[0] for pair in edges_to_add]
    cols         = [pair[1] for pair in edges_to_add]

    conn[rows, cols] = 1
    conn[cols, rows] = 1

    return conn.astype(bool)


def build_geometry(src, dmax=10, k=None, inter_hemi_dmax=0, weights=True):
    """
    Build adjacency matrix from brain geometry.

    Parameters
    ----------
    src: mne.SourceSpace
        Sources.

    dmax: float
        Maximal distance in [mm] at which to connect neighbors.

    k: int
        Maximum number of neighbors.
        
    conn_hemis: bool
        Whether to add connections in order to connect both hemispheres.

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
        vertices = src[hemi]["vertno"]
        dist     = src[hemi]["dist"][vertices][:,vertices].todense() * 1000     # in [mm]

        # threshold according to dmax
        dist[dist > dmax] = np.inf

        # choose only k neerest connections
        if k is not None:
            # indices of k closest nodes
            knn_indices = np.argpartition(dist, k, axis=1)[:,:k]
            
            # create symmetric mask of nearest neighbors
            mask = np.zeros_like(dist, dtype=bool)
            rows = np.arange(dist.shape[0])[:,None]
            mask[rows, knn_indices] = True
            
            sym_mask = np.logical_or(mask, mask.T)

            # apply mask
            dist = np.where(sym_mask, dist, np.inf)

        hemispheres.append(dist)

    # connect hemispheres
    n = src[0]["nuse"]

    # add inter hemispheric connections
    pos    = mm.utils.get_src_position(src) * 1000  # in [mm]
    lh_src = pos[:n]
    rh_src = pos[n:]

    inter_dist = np.linalg.norm(lh_src[:, np.newaxis, :]-rh_src[np.newaxis, :, :], axis=2)
    inter_dist[inter_dist > inter_hemi_dmax] = np.inf

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

def compute_streamline_length(streamline):
    """Compute streamline length"""
    diffs = np.diff(streamline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    return np.sum(segment_lengths)


def build_connectivity_matrix(i, j, num_dipoles, w=None):
    """
    Build dipole-based connectivity matrix.

    Parameters
    ----------
    i: list[int]
        Dipole indices of starting points. Size (num_streams,).
    j: list[int]
        Dipole indices of end points. Size (num_streams,).
    w: list[int]
        Weights to apply for each streamline. Size (num_streams,).
        If `w` is none, apply number of streamlines as weight.
    num_dipoles: int
        Number of dipoles.

    Returns
    -------
    sc: np.ndarray
        Structural connectivity matrix. Size (num_dipoles x num_dipoles)
    """
    # Initialize empty matrix
    sc = np.zeros((num_dipoles, num_dipoles))

    # if w is none, apply number of fiber as weights
    if w is None:
        for start, end in zip(i,j):
            if start == end:
                continue

            # ensure directed pairs are treated the same
            sc[start, end] += 1
            sc[end, start] += 1

        return sc

    # dictionay to accumulate weigths for each unique (i,j) pair
    sc_dict = defaultdict(list)

    for start, end, weight in zip(i,j,w):
        # skip self-loops
        if start == end:
            continue

        # ensure directed pairs are treated the same
        key = tuple(sorted((start, end)))
        sc_dict[key].append(weight)

    for (start, end), weights in sc_dict.items():
        avg_weight = np.mean(weights)
        sc[start, end] = avg_weight
        sc[end, start] = avg_weight

    return sc


# generate weights
def harmonic_weights(eigvals, relaxation=1):
    """
    Generate harmonic weights from eigenvalues.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics.
    
    relaxation: float [0-1]
        Relaxes the small weights. The weight will give the probability of a harmonic to be chosen during reconstruction. If taking the raw weights, this can be pretty
        tough on middle harmonics. The relaxation allows middle harmonics to have more weigths. When relaxation = 0, all harmonics have a weigth of 1.
    """
    arr1 = np.abs(eigvals - np.roll(eigvals, -1))
    arr2 = np.abs(eigvals - np.roll(eigvals, 1))

    weights = (arr1 + arr2)/2

    # correct extremities
    weights[0]  = arr1[0]
    weights[-1] = arr2[-1]

    return weights**relaxation


def smooth_connectome(conn, geodist, fwhm = 12, epsilon = 2, dim=2):
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

    # define incidence matrices
    indices = np.argwhere(conn>0)
    n, s    = len(conn), len(indices)
    M1, M2  = np.zeros((n,s)), np.zeros((n,s))

    for i, j in enumerate(indices):
        M1[j[0], i] = conn[j[0],j[1]]
        M2[j[1], i] = conn[j[1],j[0]]

    # smooth incidence
    M1s = (gaussian.T @ M1)
    M2s = (gaussian.T @ M2)

    # compute smoothed structural connectome
    sc = (M1s @ M2s.T) + (M2s @ M1s.T)
    sc = (sc + sc.T)/2
    np.fill_diagonal(sc, 0)

    return sc