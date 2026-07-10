"""
Module Name: metric.py
Description: Functions to measure metrics.

This module provides functions to measure various metrics. Usefull for evaluation, validation,
or simply testing.

Includes graph/connectome metrics (`density` through `block_conservation` at
the end of this file) operating on plain adjacency matrices. See
`mindmap/analysis/connectivity/process.py`'s module docstring for the
literature and target ranges behind those, and
`mindmap/analysis/connectivity/metric.py` for mne/nibabel-aware wrappers
around them.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, average_precision_score,
)

import bct

def avg_power(signal: np.ndarray) -> np.ndarray:
    """
    Compute average power of a signal.

    Parameters
    ----------
    signal: np.ndarray (t,) or (n,t)

    Returns
    -------
    avg_pow: np.ndarray (1,) or (n,)
        Average power for every sample n.
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(1,-1)   # consider signal as temporal signal (1,T)

    avg_pow = signal - np.mean(signal, axis=1)[:,None]
    avg_pow = np.sum(avg_pow**2, axis=1)/signal.shape[-1]

    return avg_pow


def estimate_snr(
    timeserie: np.ndarray,
    baseline: tuple[int, int],
    method: str="power",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes time-varying SNR relative to a baseline window.

    Parameters
    ----------
    timeserie: np.ndarray of shape (n_samples, n_timepoints)
        Timeseries.

    baseline: tuple[int,int]
        Indices [start, stop) for the noise window.

    method: str ['power', 'variance']
        Whether to use power-based or variance-based 
        estimate.

    Returns
    -------
    snr, snr_db: tuple[np.ndarray, np.ndarray] of shapes (n_times,)
    """
    b_start, b_stop = baseline

    if method == "power":
        noise_power = np.mean(timeserie[:, b_start:b_stop] ** 2)
        signal_power = np.mean(timeserie ** 2, axis=0)
    elif method == "variance":
        noise_power = np.var(timeserie[:, b_start:b_stop])
        signal_power = np.var(timeserie, axis=0)
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'power' or 'variance'.")

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    return snr, snr_db


def point_to_area_distance(
        target_point : np.ndarray,
        target_area  : np.ndarray,
        x            : int = 1,
        ) -> float:
    """
    Measures distance between a point and an area.

    Parameters
    ----------
    target_point: np.ndarray (3,)
        Target point in [m]

    target_area: np.ndarray (3,N) | nibabel.Nifti1Image
        Cloud point in [m] if array, in [mm] if Niti1Image.

    x: int
        Number of points from the cloud point to consider.

    Returns
    -------
    distance: float
        Distance in [m]

    Note: The distance is measured as the length of the average distance vector between
    the target point and the x closest points of the point cloud.:
    - x = 1: Distance to closest point of the area.
    - x = "all": Distance to the center of mass of the area.
    - x != "all" != 1: Distance to the center of mass of the portion consisting of the
      x closest points.
    """

    distances = target_area-target_point

    if x == "all":
        mean_distance = np.mean(distances, axis=0)
        norm          = np.linalg.norm(mean_distance)

    else:
        norms         = np.linalg.norm(distances, ord=2, axis=1)
        nearest       = distances[np.argsort(norms)[:x],:]
        mean_distance = np.mean(nearest, axis=0)
        norm          = np.linalg.norm(mean_distance)

    return norm


def compute_distances(array1, array2, output_type="scalar"):
    """
    Computes distances between points of two arrays. Displacements
    will point from array1 to array2.

    Parameters:
    ----------
    array1: np.ndarray of shape (n_starts, n_dimensions)
        Sarting points.
    
    array2: np.ndarray of shape (n_ends, n_dimensions)
        Ending points

    output_type: str
        'scalar' (Euclidean) or 'vector' (Displacement)

    Returns
    -------
    dist: np.ndarray
        If 'scalar': Array of shape (n_starts, n_ends) containing L2 norms
        If 'vector': Array of shape (n_starts, n_ends, n_dimensions) containing displacement vectors.
    """
    # Ensure inputs are numpy arrays
    a1 = np.atleast_2d(array1)
    a2 = np.atleast_2d(array2)

    # Broadcasting to find difference between every pair
    diff =  a2[np.newaxis,:,:] - a1[:,np.newaxis,:]

    if output_type == "vector":
        return diff
    elif output_type == "scalar":
        return np.linalg.norm(diff, axis=2)
    else:
        raise ValueError("output_type must be 'scalar' or 'vector'")


def extract_voxel_volume(affine: np.ndarray):
    """
    Computes voxel volume in [mm³] from the affine.
    """
    # defines borders of voxel (parallelepiped)
    v1 = affine[:3, 0]
    v2 = affine[:3, 1]
    v3 = affine[:3, 2]

    # volume of a parallelepiped
    volume = np.abs(np.cross(v1, v2) @ v3)

    return volume


def calc_correlation(
    arr1: np.ndarray,
    arr2: np.ndarray,
    method: str = "pearson",
    axis: int = None,
    center_axis: str | int | None = "default",
) -> np.ndarray | float:
    """
    Correlation between two arrays.

    Parameters
    ----------
    arr1: np.ndarray of shape (n_samples, n_timepoints)

    arr2: np.ndarray of shape (n_samples, n_timepoints)

    method: str ['pearson', 'spearman', 'log-envelope']
        Correlation method.

    axis: int or None
        Axis along which to compute correlation
        - None: spatio-temporal correlation (flatten both dims) -> scalar
        - 0: along samples at each timepoint -> (n_timepoints,)
        - 1: along timepoints for each sample -> (n_samples,)

    center_axis: str | int | None
        Axis along which to average reference. Default chooses the same as the 
        'axis' argument.

    eps: float
        Small constant added before log to avoid log(0)

    Returns
    -------
    corr: np.ndarray | float
        Returns 0 where an array has zero variance.
    """
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shape mismatch: arr1 {arr1.shape} and arr2 {arr2.shape} must be identical."
        )
    if method not in ("pearson", "spearman", "log-envelope"):
        raise ValueError(f"Unknown method '{method}'. Expected 'pearson', 'spearman' or 'log-envelope'.")

    # determine how to center
    if center_axis == "default":
        c_axis = axis
    else:
        c_axis = center_axis

    # Upcast to float64 to handle tiny scales safely
    arr1_64 = arr1.astype(np.float64)
    arr2_64 = arr2.astype(np.float64)

    eps = np.finfo(np.float64).eps

    # compute correlation
    if method == "log-envelope":
        # calculate individual data-dependent log floors safely
        std1 = np.std(np.abs(arr1_64)) * 1e-4 or 1e-15
        std2 = np.std(np.abs(arr2_64)) * 1e-4 or 1e-15

        eps_log1 = (std1 * 1e-4) if std1 > eps else 1e-15
        eps_log2 = (std2 * 1e-4) if std2 > eps else 1e-15

        # compute log-envelope
        arr1_64 = np.log(np.abs(arr1_64) + eps_log1)
        arr2_64 = np.log(np.abs(arr2_64) + eps_log2)
        method = "pearson"

    if method == "pearson":
        arr1_c = arr1_64 - arr1_64.mean(axis=c_axis, keepdims=True)
        arr2_c = arr2_64 - arr2_64.mean(axis=c_axis, keepdims=True)

    elif method == "spearman":
        arr1_c = rankdata(arr1_64, axis=axis).astype(float)
        arr2_c = rankdata(arr2_64, axis=axis).astype(float)
        arr1_c -= arr1_c.mean(axis=axis, keepdims=True)
        arr2_c -= arr2_c.mean(axis=axis, keepdims=True)

    norm1 = np.sqrt(np.sum(arr1_c**2, axis=axis, keepdims=True))
    norm2 = np.sqrt(np.sum(arr2_c**2, axis=axis, keepdims=True))

    arr1_normed = np.divide(arr1_c, norm1, out=np.zeros_like(arr1_c), where=norm1 > eps)
    arr2_normed = np.divide(arr2_c, norm2, out=np.zeros_like(arr2_c), where=norm2 > eps)

    corr = np.sum(arr1_normed * arr2_normed, axis=axis)

    return corr.item() if corr.size == 1 else corr

def calc_cosine(
    arr1: np.ndarray,
    arr2: np.ndarray,
    axis: int = None,
    absolute: bool = False,
) -> np.ndarray | float:
    """
    Compute Cosine Similarity between two arrays.

    Parameters
    ----------
    arr1, arr2: np.ndarray
        Input arrays of identical shape.
    
    axis: int or None
        Axis along which to compute similarity.
        None flattens arrays (global similarity).
    
    absolute: bool
        If True, computes similarity on the absolute magnitudes.

    Returns
    -------
    cos: np.ndarray | float
        Cosine similarity range [-1, 1]. Returns 0 for zero-norm vectors.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Shapes must match.")

    # upcast to float64 to handle tiny scales safely
    arr1_64 = arr1.astype(np.float64)
    arr2_64 = arr2.astype(np.float64)

    # apply absolute
    if absolute:
        arr1_64 = np.abs(arr1_64)
        arr2_64 = np.abs(arr2_64)

    # norms
    norm1 = np.linalg.norm(arr1_64, axis=axis)
    norm2 = np.linalg.norm(arr2_64, axis=axis)

    # safely normalize
    eps = np.finfo(np.float64).eps

    arr1_normed = np.divide(arr1_64, norm1, out=np.zeros_like(arr1_64), where=norm1 > eps)
    arr2_normed = np.divide(arr2_64, norm2, out=np.zeros_like(arr2_64), where=norm2 > eps)

    # cosine similarity
    cos = np.sum(arr1_normed * arr2_normed, axis=axis)

    return cos.item() if cos.size == 1 else cos


def calc_correlation_matrix(
    arr1: np.ndarray,
    arr2: np.ndarray,
    method: str = "pearson",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes a correlation matrix where result[i, j] is the correlation
    between arr1[i] and arr2[j].

    Parameters
    ----------
    arr1: np.ndarray of shape (n_samples_1, n_timepoints)
    
    arr2: np.ndarray of shape (n_samples_2, n_timepoints)

    method: str ['pearson', 'spearman', 'log-envelope]
    
    eps: float
        Small constant for log-envelope or to avoid division by zero.

    Returns
    -------
    corr_matrix: np.ndarray of shape (n_samples_1, n_samples_2)
    """
    if arr1.shape[1] != arr2.shape[1]:
        raise ValueError(
            f"Timepoint mismatch: arr1 {arr1.shape[1]} and arr2 {arr2.shape[1]} must match."
        )
    
    # Pre-process for Log-Envelope
    if method == "log-envelope":
        arr1 = np.log(np.abs(arr1) + eps)
        arr2 = np.log(np.abs(arr2) + eps)
        method = "pearson"

    # Transform to Rank for spearman
    if method == "spearman":
        # rankdata returns flattened array if axis=None, so we specify axis=1
        arr1 = rankdata(arr1, axis=1).astype(float)
        arr2 = rankdata(arr2, axis=1).astype(float)
        method = "pearson"

    # Standardize and Compute Pearson via Matrix Multiplication
    if method == "pearson":
        # center across timepoint (axis=1)
        arr1_c = arr1 - arr1.mean(axis=1, keepdims=True)
        arr2_c = arr2 - arr2.mean(axis=1, keepdims=True)

        # compute denominators (L2 norms)
        norm1 = np.sqrt(np.sum(arr1_c**2, axis=1, keepdims=True))
        norm2 = np.sqrt(np.sum(arr2_c**2, axis=1, keepdims=True))

        # normalize the centere arrays
        arr1_n = np.divide(arr1_c, norm1, out=np.zeros_like(arr1_c), where=norm1 > 1e-12)
        arr2_n = np.divide(arr2_c, norm2, out=np.zeros_like(arr2_c), where=norm2 > 1e-12)

        return arr1_n @ arr2_n.T
    
    raise ValueError(f"Unknown method '{method}'")


def calc_pairwise_dist(array1, array2, output_type="scalar"):
    """
    Computes pairwise distances between points of two arrays. Displacements
    will point from array1 to array2.

    Parameters:
    ----------
    array1: np.ndarray
        Array of shape (N,D) or (N,)
    
    array2: np.ndarray
        Array of shape (N,D) or (N,)

    output_type: str
        'scalar' (Euclidean) or 'vector' (Displacement)

    Returns
    -------
    dist: np.ndarray
        If 'scalar': Array of shape (N,) containing L2 norms
        If 'vector': Array of shape (N,D) containing displacement vectors.
    """
    # Ensure inputs are numpy arrays
    a1 = np.atleast_2d(array1)
    a2 = np.atleast_2d(array2)

    # pairwise differences
    diff =  a2 - a1

    if output_type == "vector":
        return diff
    elif output_type == "scalar":
        return np.linalg.norm(diff, axis=1)
    else:
        raise ValueError("output_type must be 'scalar' or 'vector'")
    

def calc_spatial_dispersion_grova(map_est, dist_to_h):
    """
    Grova's Spatial Dispersion (SD): Spread of estimated activity
    relative to the ground-truth clinical focus (Set H).

    Parameters
    ----------
    map_est: np.ndarray of shape (n_observations, n_channels)
        The estimated spatial maps (e.g., virtual iEEG)

    dist_to_h: np.ndarray of shape (n_channels,)
        Minimum distance from each channel i to the closest channel
        j in the reference set H. (dist = 0 if channel i is in H).

    Returns
    -------
    sp_dis: np.ndarray of shape (n_observations,)
        Spatial dispersion (SD). Higher values indicate 'smearing'
        outside the clinical zone or localization error.
    """
    # ensure inputs are at least 2D: (n_observations, n_channels)
    # if input is (M,), it becomes (1,M)
    map_est = np.atleast_2d(map_est)

    # ensure dist_to_h is (1, n_channels) for broadcasting
    dist_to_h = np.atleast_2d(dist_to_h)

    # Energy (V^2)
    energy = map_est**2

    # compute dispersion
    numerator   = np.sum(dist_to_h * energy, axis=1)
    denominator = np.sum(energy, axis=1)

    dispersion = np.sqrt(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12))
    
    return dispersion if dispersion.size > 1 else dispersion[0]


def calc_ratio_spurious_activity(map_est, is_distant_mask):
    """
    Ratio of Spurious Activity (RSA) as per Grova et al.

    Parameters
    ----------
    map_est: np.ndarray of shape (n_observations, n_channels)
        The estimated potentials (V_MEG).
    is_distant_mask: np.ndarray of shape (n_channels,)
        Boolean mask or binary array (1/0) where 1 indicates the contact
        is > 10mm away from the reference set H.

    Returns
    -------
    rsa: np.ndarray of shape (n_observations,)
        The ratio of energy in distant contacts vs total energy.
    """
    # ensure inputs are at least 2D: (n_observations, n_channels)
    # if input is (M,), it becomes (1,M)
    map_est = np.atleast_2d(map_est)

    # energy (V^2)
    energy = map_est**2

    # energy in distant contacts (numerator)
    # we use the mask to zero out energy in contacts close to H.
    spurious_energy = np.sum(energy[:, is_distant_mask], axis=1)

    # Total energy across all contacts (denominator)
    total_energy = np.sum(energy, axis=1)

    # RSA = Sum(V^2_distant) / Sum(V^2_total)
    rsa = np.divide(spurious_energy, total_energy, out=np.zeros_like(spurious_energy), where=total_energy>1e-12)

    return rsa if rsa.size > 1 else rsa[0]


def compute_correlation(data1, data2, axis=0):
    """
    !!!! I have two compute correlation :( Try fusing them !!!!
    Computes Pearson correlation between two arrays along a specific axis.

    Parameters
    ----------
    data1, data2: np.ndarray of shapes (M,N)
        Data.
    
    axis: int
        Axis along which to compute correlation. If None, compute
        a global correlation.

    Returns:
    -------
    corr: np.ndarray
        Pearson correlation.
    """
    # 1. Center the data
    d1_mean = data1 - np.mean(data1, axis=axis, keepdims=True)
    d2_mean = data2 - np.mean(data2, axis=axis, keepdims=True)

    # 2. Normalize
    d1_norm = d1_mean / np.linalg.norm(d1_mean, axis=axis, keepdims=True)
    d2_norm = d2_mean / np.linalg.norm(d2_mean, axis=axis, keepdims=True)

    return np.sum(d1_norm * d2_norm, axis=axis)


def get_gfp(data, axis=0):
    """Global Field Power (GFP) is the spatial standard deviation at each time point"""
    return np.std(data, axis=axis)


def compute_cog_trajectory(data:np.ndarray, pos:np.ndarray) -> np.ndarray:
    """
    Compute the 3D Center of Gravity trajectory over time.

    Parameters
    ----------
    data: np.ndarray of shape (n_samples, n_timepoints)

    pos: np.ndarray of shape (n_samples, 3)
        The 3D coordinates (x,y,z) for each sample.

    Returns
    -------
    cog_trajectory: np.ndarray of shape (n_timepoints, 3)
        The (x,y,z) coordinates of the CoG at each timepoint.
    """
    if data.shape[0] != pos.shape[0]:
        raise ValueError(f"Channel mismatch: data has {data.shape[0]} but pos has {pos.shape[0]}")
    
    # Use absolute values because negative potential/current and positive
    # potential/current both represent acitvity "mass"
    weights = np.abs(data)

    # total "mass"
    total_mass = np.sum(weights, axis=0)

    # avoid division by zero in baseline noise
    total_mass[total_mass == 0] = 1e-15

    # compute weighted average for X, Y, and Z
    cog_trajectory = (pos.T @ weights) / total_mass

    return cog_trajectory.T


def calc_rem(
    data_ref: np.ndarray,
    data_est: np.ndarray,
    axis: int = 0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Compute Relative Energy Map along a specific axis.

    Formula: (data_est^2) / sum(data_ref^2) along the axis
    """
    if data_ref.shape != data_est.shape:
        raise ValueError("Arrays must have the same shape to compute REM.")
    
    # Calculate energy (squared magnitude)
    energy_ref = data_ref**2
    energy_est = data_est**2

    # Sum of reference energy for nomalization
    total_ref_energy = np.sum(energy_ref, axis=axis, keepdims=True)

    # avoid division by zero
    total_ref_energy[total_ref_energy < epsilon] = 1.0

    # return the relative map
    return energy_est / total_ref_energy


def calc_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculates the confusion matrix between a binarized ground truth and 
    a binarized or continuous signal.

    Parameters:
    ----------
    y_true: np.ndarray
        Binary ground truth.

    y_pred: np.ndarray
        Binary prediction.

    Returns:
    -------
    dict
        A dictionary containing Accuracy, Precision, Recall, F1-score, and specificity
    """
    # flatten arrays
    yt = y_true.ravel()
    yp = y_pred.ravel()

    # counts
    cm = confusion_matrix(yt, yp, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    return {"tn":tn, "fp":fp, "fn":fn, "tp":tp}


def calc_activity_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Computes classification metrics for activity data.

    Parameters:
    ----------
    y_true: np.ndarray of shape (samples,) or (samples, time) 
        Ground truth array.

    y_pred: np.ndarray of shape (samples,) or (samples, time)
        Estimated activity array.

    threshold: float [0-1]
        Scalar value to binarize y_pred.

    Returns:
    -------
    dict
        A dictionary containing Accuracy, Precision, Recall, F1-score, and specificity.
    """
    # flatten arrays
    yt = y_true.ravel()
    yp = y_pred.ravel()

    # scores
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=np.nan, labels=[0,1]),
        "recall": recall_score(yt, yp, zero_division=np.nan, labels=[0,1]),
        "f1": f1_score(yt, yp, zero_division=np.nan, labels=[0,1]),
        #"mcc": matthews_corrcoef(yt, yp),
    }

def calc_roc_curve(
    y_true,
    y_pred,
) -> dict:
    """
    Computes data for a ROC curve.

    Parameters:
    ----------
    y_true: np.ndarray
        Binarized ground truth.

    y_pred: np.ndarray
        Continuous estimate.
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()

    fpr, tpr, _ = roc_curve(yt, yp)
    roc_auc = auc(fpr, tpr)

    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc}


def calc_pr_curve(
    y_true,
    y_pred,
) -> dict:
    """
    Computes data for a precision-recall curve.

    Parameters:
    ----------
    y_true: np.ndarray
        Binarized ground truth.

    y_pred: np.ndarray
        Continuous estimate.
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()

    precision, recall, _ = precision_recall_curve(yt, yp, drop_intermediate=True)
    ap = average_precision_score(yt, yp)

    return {"precision": precision, "recall": recall, "ap": ap}


def calc_rdm(arr1, arr2):
    """Calculate the Relative Difference Magnitude (RDM)."""
    # Calculate Frobenius norms
    norm_arr1 = np.linalg.norm(arr1, ord="fro")
    norm_arr2 = np.linalg.norm(arr2, ord="fro")

    # Calculate RDM
    return np.linalg.norm((arr1/norm_arr1) - (arr2/norm_arr2), ord="fro")


def calc_mag(arr1, arr2):
    """Calculate the Magnitude Magnification (MAG)."""
    # Calculate Frobenius norms
    norm_arr1 = np.linalg.norm(arr1, ord="fro")
    norm_arr2 = np.linalg.norm(arr2, ord="fro")

    # Calculate MAG
    return norm_arr1 / norm_arr2


# -----------------------------------------------------------------------
# Graph / connectome metrics
#
# `small_world_sigma`, `rich_club_curve` and `modularity` operate on the
# *binarized* graph. Adjacency weights are often lengths/distances rather
# than connection strengths, and feeding those directly into bctpy's
# weighted functions (which treat larger weight as "more strongly
# connected") would silently invert their meaning. A length-to-strength
# conversion would be needed first; that transform isn't settled, so
# weighted variants are left out rather than guessed.
# -----------------------------------------------------------------------

def binarize(adj: np.ndarray) -> np.ndarray:
    """Binarize a weighted adjacency matrix."""
    return (adj > 0).astype(int)


def _shortest_path_matrix(bin_adj: np.ndarray) -> np.ndarray:
    """
    Binary all-pairs shortest-path distance matrix (unreachable pairs =
    inf), via scipy's BFS rather than bct.distance_bin. bct.distance_bin's
    repeated-matrix-multiplication approach is >80x slower already at
    n=2000 nodes and scales worse than cubically, making it infeasible at
    dipole-level resolution (10^4-10^5 nodes); scipy's BFS is ~O(n^2) and
    stays usable at that scale.
    """
    return shortest_path(sp.csr_matrix(bin_adj), method="D", unweighted=True, directed=False)


def density(adj: np.ndarray) -> float:
    """
    Edge density: fraction of possible undirected edges that are present.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    Returns
    -------
    density: float
        Number of present edges over the number of possible edges,
        n*(n-1)/2.
    """
    n = adj.shape[0]
    if n < 2:
        return 0.0

    n_edges = np.count_nonzero(np.triu(adj, k=1))
    n_possible = n * (n - 1) / 2

    return n_edges / n_possible


def small_world_sigma(
        adj: np.ndarray,
        n_null: int = 20,
        n_rewire: int = 5,
        seed: int | None = None,
) -> dict:
    """
    Small-world coefficient sigma = (C / C_null) / (L / L_null).

    Computed on the binarized graph against a degree-preserving
    randomized null ensemble (bct.randmio_und). sigma > 1 indicates
    small-world organization: clustering above, and path length close
    to, a random graph with the same degree sequence. The absolute value
    of sigma is not comparable across studies/resolutions (reported
    values in the literature span roughly 1 to several hundred) -- only
    sigma > 1 should be treated as meaningful here.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    n_null: int
        Number of randomized null networks to average over.

    n_rewire: int
        Rewiring parameter passed to bct.randmio_und (each edge is
        rewired approximately this many times per null network).

    seed: int, optional
        Seed for the null-network randomization.

    Returns
    -------
    result: dict
        - sigma: small-world coefficient
        - C, C_null: mean clustering coefficient (real, null average)
        - L, L_null: characteristic path length (real, null average)
    """
    rng = np.random.default_rng(seed)
    bin_adj = binarize(adj)

    C = np.mean(bct.clustering_coef_bu(bin_adj))
    D = _shortest_path_matrix(bin_adj)
    L = bct.charpath(D, include_infinite=False)[0]

    C_null, L_null = [], []
    for _ in range(n_null):
        null_seed = int(rng.integers(0, 2**32 - 1))
        rewired, _ = bct.randmio_und(bin_adj, n_rewire, seed=null_seed)
        C_null.append(np.mean(bct.clustering_coef_bu(rewired)))
        D_null = _shortest_path_matrix(rewired)
        L_null.append(bct.charpath(D_null, include_infinite=False)[0])

    C_null, L_null = float(np.mean(C_null)), float(np.mean(L_null))
    sigma = (C / C_null) / (L / L_null)

    return {"sigma": sigma, "C": C, "C_null": C_null, "L": L, "L_null": L_null}


def rich_club_curve(
        adj: np.ndarray,
        n_null: int = 20,
        seed: int | None = None,
) -> dict:
    """
    Normalized rich-club coefficient curve, R_norm(k) = R(k) / R_null(k).

    R_norm(k) > 1 over a range of k indicates that high-degree nodes are
    more densely interconnected than expected by degree alone (rich-club
    organization), computed on the binarized graph against a
    degree-preserving randomized null ensemble.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    n_null: int
        Number of randomized null networks to average over.

    seed: int, optional
        Seed for the null-network randomization.

    Returns
    -------
    result: dict
        - k: degree levels, 1..max(degree)
        - R: rich-club coefficient of the real network at each k
        - R_null: null-average rich-club coefficient at each k
        - R_norm: R / R_null
    """
    bin_adj = binarize(adj)
    R, _, _ = bct.rich_club_bu(bin_adj)
    klevel = len(R)

    rng = np.random.default_rng(seed)
    R_null = np.full((n_null, klevel), np.nan)
    for i in range(n_null):
        null_seed = int(rng.integers(0, 2**32 - 1))
        rewired, _ = bct.randmio_und(bin_adj, 5, seed=null_seed)
        R_null[i], _, _ = bct.rich_club_bu(rewired, klevel=klevel)

    R_null_mean = np.nanmean(R_null, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        R_norm = R / R_null_mean

    return {"k": np.arange(1, klevel + 1), "R": R, "R_null": R_null_mean, "R_norm": R_norm}


def modularity(adj: np.ndarray, gamma: float = 1.0, seed: int | None = None) -> dict:
    """
    Modularity of the binarized graph, via Louvain community detection.

    Uses bct.modularity_louvain_und rather than bct.modularity_und: the
    latter's spectral partitioning builds a ragged (unequal-size-per-
    community) Python list internally and hands it to numpy without an
    explicit dtype, which raises `ValueError: setting an array element
    with a sequence` under numpy>=2.0 for any partition with differently
    sized communities -- i.e. almost always. Louvain does not hit this
    code path and gives an equivalent modularity-maximizing partition.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    gamma: float
        Resolution parameter (higher values favor more, smaller
        communities).

    seed: int, optional
        Seed for the Louvain heuristic, which is stochastic and can
        otherwise vary between runs on the same input.

    Returns
    -------
    result: dict
        - q: modularity value
        - community: community assignment per node
        - n_communities: number of detected communities
    """
    bin_adj = binarize(adj)
    community, q = bct.modularity_louvain_und(bin_adj, gamma=gamma, seed=seed)

    return {"q": q, "community": community, "n_communities": int(np.unique(community).size)}


def distance_decay_fit(adj: np.ndarray, coords: np.ndarray, n_bins: int = 30) -> dict:
    """
    Fit the exponential distance rule (EDR): P(connected) ~ c * exp(-lambda * d).

    Distances are binned, the fraction of connected pairs is computed per
    bin, and log(fraction) is fit linearly against bin distance. This is
    a check on the *binary* topology only (not weights), since adjacency
    weights are often lengths rather than a connection strength/
    probability.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    coords: np.ndarray of shape (n, 3)
        Node positions, in the same units used to judge lambda (e.g. [mm]).

    n_bins: int
        Number of distance bins.

    Returns
    -------
    result: dict
        - lambda_: fitted distance-decay constant
        - c: fitted proportionality constant
        - r2: R^2 of the log-linear fit
        - bin_centers, bin_prob: binned connection probability vs distance
    """
    n = adj.shape[0]
    iu = np.triu_indices(n, k=1)

    d = cdist(coords, coords)[iu]
    connected = (adj[iu] > 0).astype(float)

    bins = np.linspace(0, d.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(d, bins) - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    bin_prob = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = bin_idx == b
        if np.any(in_bin):
            bin_prob[b] = connected[in_bin].mean()

    valid = np.isfinite(bin_prob) & (bin_prob > 0)
    if valid.sum() < 2:
        return {
            "lambda_": np.nan, "c": np.nan, "r2": np.nan,
            "bin_centers": bin_centers, "bin_prob": bin_prob,
        }

    slope, intercept = np.polyfit(bin_centers[valid], np.log(bin_prob[valid]), 1)
    lambda_, c = -slope, np.exp(intercept)

    pred = c * np.exp(-lambda_ * bin_centers[valid])
    ss_res = np.sum((bin_prob[valid] - pred) ** 2)
    ss_tot = np.sum((bin_prob[valid] - bin_prob[valid].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "lambda_": lambda_, "c": c, "r2": r2,
        "bin_centers": bin_centers, "bin_prob": bin_prob,
    }


def hemispheric_symmetry(adj: np.ndarray, coords: np.ndarray, hemi: np.ndarray) -> dict:
    """
    Correlation between the degree of homologous nodes across two halves
    of a bipartitioned graph (e.g. brain hemispheres).

    Homologous pairs are found by mirroring one half across the midline
    (x=0) and nearest-neighbor matching to the other half in 3D. dMRI
    tractography is known to underestimate homotopic interhemispheric
    connections, so this checks structural (degree) symmetry rather than
    assuming strong direct cross-hemisphere edges.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    coords: np.ndarray of shape (n, 3)
        Node positions in a space where the midline is x=0 (e.g.
        FreeSurfer tkras space), in [mm].

    hemi: np.ndarray of shape (n,)
        Side label per node (0=left, 1=right).

    Returns
    -------
    result: dict
        - r: Pearson correlation between matched left/right degree
        - match_dist_mean, match_dist_max: mirror-matching quality [mm]
          (large values mean the two sides are not close to
          mirror-symmetric, so the matching -- and thus `r` -- should be
          interpreted with caution)
    """
    coords = np.asarray(coords)
    hemi = np.asarray(hemi)

    left = np.where(hemi == 0)[0]
    right = np.where(hemi == 1)[0]

    mirrored_left = coords[left].copy()
    mirrored_left[:, 0] *= -1

    tree = cKDTree(coords[right])
    match_dist, nn = tree.query(mirrored_left)

    degree = np.sum(adj > 0, axis=1)
    r = np.corrcoef(degree[left], degree[right][nn])[0, 1]

    return {"r": r, "match_dist_mean": float(match_dist.mean()), "match_dist_max": float(match_dist.max())}


def block_conservation(fine_adj: np.ndarray, block_labels: np.ndarray, coarse_adj: np.ndarray) -> dict:
    """
    Checks whether block-averaging a fine-grained (e.g. dipole-level)
    connectome back down to block (e.g. region) resolution recovers the
    coarser connectome it was built from. This is a correctness check on
    an upscaling procedure, not a realism check: a well-behaved upscaling
    should not introduce a systematic bias relative to its own input.

    For each coarse-connected block pair, the mean of the nonzero
    fine-grained edges in that block is compared to the coarse value.
    Zero-valued (unselected) fine-grained pairs are excluded from the
    block mean, since they reflect a sparsity constraint rather than the
    calibrated edge weight.

    Parameters
    ----------
    fine_adj: np.ndarray of shape (n_fine, n_fine)
        Fine-grained connectome produced from `coarse_adj`.

    block_labels: np.ndarray of shape (n_fine,)
        Block (e.g. region) id per fine node, 1-indexed (0 reserved for
        unlabeled).

    coarse_adj: np.ndarray of shape (n_blocks, n_blocks)
        Coarse connectome `fine_adj` was upscaled from.

    Returns
    -------
    result: dict
        - r: Pearson correlation between coarse and recovered block means
        - mae: mean absolute error (recovered - coarse), in coarse units
        - mean_bias: mean relative error, (recovered - coarse) / coarse
        - n_pairs: number of coarse-connected block pairs used
    """
    n_blocks = coarse_adj.shape[0]
    block_to_fine = {
        b: np.where(block_labels == (b + 1))[0]
        for b in range(n_blocks)
    }

    recovered = np.full((n_blocks, n_blocks), np.nan)
    for i in range(n_blocks):
        idx_i = block_to_fine[i]
        if len(idx_i) == 0:
            continue

        for j in range(i, n_blocks):
            idx_j = block_to_fine[j]
            if len(idx_j) == 0:
                continue

            block = fine_adj[np.ix_(idx_i, idx_j)]
            nonzero = block[block > 0]
            if nonzero.size > 0:
                recovered[i, j] = recovered[j, i] = nonzero.mean()

    triu = np.triu(np.ones_like(coarse_adj, dtype=bool), k=1)
    mask = (coarse_adj > 0) & np.isfinite(recovered) & triu
    coarse_vals = coarse_adj[mask]
    recovered_vals = recovered[mask]

    if coarse_vals.size < 2:
        return {"r": np.nan, "mae": np.nan, "mean_bias": np.nan, "n_pairs": int(coarse_vals.size)}

    r = np.corrcoef(coarse_vals, recovered_vals)[0, 1]
    mae = np.mean(np.abs(recovered_vals - coarse_vals))
    mean_bias = np.mean((recovered_vals - coarse_vals) / coarse_vals)

    return {"r": r, "mae": mae, "mean_bias": mean_bias, "n_pairs": int(coarse_vals.size)}