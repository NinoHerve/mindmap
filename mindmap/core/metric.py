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
from scipy.stats import rankdata, hypergeom
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, average_precision_score,
)
from skimage.metrics import structural_similarity

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
    polarity_invariant: bool = False,
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

    polarity_invariant: bool
        If True, returns |corr| instead of corr -- for when an overall sign
        flip (e.g. an arbitrary polarity/reference convention) shouldn't be
        scored as a worse match than a perfectly in-phase one. Applied to
        the finished coefficient, not the raw data: correlating
        abs(arr1)/abs(arr2) instead would measure magnitude/envelope
        co-modulation, a different and weaker property than "same shape, up
        to one sign flip".

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
    if polarity_invariant:
        corr = np.abs(corr)

    return corr.item() if corr.size == 1 else corr

def calc_cosine(
    arr1: np.ndarray,
    arr2: np.ndarray,
    axis: int = None,
    absolute: bool = False,
    polarity_invariant: bool = False,
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
        If True, computes similarity on the absolute magnitudes -- i.e.
        whether the two arrays' magnitude/envelope co-varies, regardless of
        the sign at any given point. A different (weaker) property than
        `polarity_invariant` below: this changes what's being measured, not
        just the sign of the reported result.

    polarity_invariant: bool
        If True, returns |cos| instead of cos -- for when an overall sign
        flip (e.g. an arbitrary polarity/reference convention) shouldn't be
        scored as a worse match than a perfectly in-phase one. Applied to
        the finished coefficient: still requires the two arrays to actually
        vary together (up to one sign flip), unlike `absolute` above.

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

    # norms (keepdims so they broadcast back against arr1_64/arr2_64 below,
    # not just against axis=None's scalar or axis=0's coincidentally-aligned
    # trailing dimension -- axis=1 needs the kept dim to broadcast at all)
    norm1 = np.linalg.norm(arr1_64, axis=axis, keepdims=True)
    norm2 = np.linalg.norm(arr2_64, axis=axis, keepdims=True)

    # safely normalize
    eps = np.finfo(np.float64).eps

    arr1_normed = np.divide(arr1_64, norm1, out=np.zeros_like(arr1_64), where=norm1 > eps)
    arr2_normed = np.divide(arr2_64, norm2, out=np.zeros_like(arr2_64), where=norm2 > eps)

    # cosine similarity
    cos = np.sum(arr1_normed * arr2_normed, axis=axis)
    if polarity_invariant:
        cos = np.abs(cos)

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


def calc_pattern_geometry_correlation(
    arr_true: np.ndarray,
    arr_pred: np.ndarray,
    method: str = "pearson",
    compare_method: str = "spearman",
) -> float:
    """
    Second-order ("representational similarity") comparison: instead of
    comparing arr_true and arr_pred directly, compares whether the
    *geometry* of how their columns relate to each other matches -- e.g.
    whether the topography at time i is similarly (dis)similar to the
    topography at time j in both arr_true and arr_pred. Robust to a global
    difference in how the two arrays' own internal geometry is scaled,
    which a direct column-wise correlation would still be sensitive to.

    Builds a dissimilarity matrix per array (1 - pairwise correlation
    between columns, via `calc_correlation_matrix`), then correlates the
    two matrices' upper triangles.

    Parameters
    ----------
    arr_true, arr_pred: np.ndarray of shape (n_features, n_conditions)
        e.g. (n_channels, n_timepoints): each column is one condition's
        spatial pattern, compared pairwise against every other column.

    method: str ['pearson', 'spearman', 'log-envelope']
        Method used to build each array's own dissimilarity matrix.

    compare_method: str ['pearson', 'spearman', 'log-envelope']
        Method used for the final comparison between the two dissimilarity
        matrices. Spearman (rank-based) is standard RSA practice, since it
        only requires the two geometries to be monotonically related, not
        linearly.

    Returns
    -------
    float
    """
    if arr_true.shape != arr_pred.shape:
        raise ValueError(
            f"Shape mismatch: arr_true {arr_true.shape} and arr_pred {arr_pred.shape} must be identical."
        )

    dissim_true = 1 - calc_correlation_matrix(arr_true.T, arr_true.T, method=method)
    dissim_pred = 1 - calc_correlation_matrix(arr_pred.T, arr_pred.T, method=method)

    iu = np.triu_indices(dissim_true.shape[0], k=1)

    return calc_correlation(dissim_true[iu], dissim_pred[iu], method=compare_method, axis=None)


def calc_explained_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: int = None,
    center_axis: str | int | None = "default",
    polarity_invariant: bool = False,
) -> np.ndarray | float:
    """
    Explained variance / R^2 = 1 - SS_res/SS_tot.

    Unlike `calc_correlation`/`calc_cosine`, this is sensitive to amplitude
    scale: a prediction that's a perfectly-shaped but rescaled copy of
    y_true (e.g. half amplitude) is penalized here even though it would
    score a perfect correlation. Complements correlation/cosine rather than
    replacing them.

    Parameters
    ----------
    y_true, y_pred: np.ndarray
        Arrays of identical shape.

    axis: int or None
        Axis along which to compute R^2 (mirrors `calc_correlation`):
        - None: spatio-temporal (flatten both dims) -> scalar
        - 0: along samples at each timepoint -> (n_timepoints,)
        - 1: along timepoints for each sample -> (n_samples,)

    center_axis: str | int | None
        Axis along which to compute y_true's mean for SS_tot. Default
        chooses the same as `axis`.

    polarity_invariant: bool
        If True, also computes R^2 against a sign-flipped y_pred and keeps
        whichever is higher, per output element. R^2 isn't a signed
        quantity the way correlation/cosine are -- a polarity flip doesn't
        just flip its sign, it inflates SS_res (a flipped-but-otherwise-
        perfect prediction scores strongly negative, not -1) -- so unlike
        `calc_correlation`/`calc_cosine`'s `polarity_invariant`, `abs()` of
        the score can't undo it; the two candidate reconstructions have to
        actually be compared instead.

    Returns
    -------
    r2: np.ndarray | float
        Returns 0 where y_true has zero variance along `axis`.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must be identical."
        )

    c_axis = axis if center_axis == "default" else center_axis

    y_true_64 = y_true.astype(np.float64)
    y_pred_64 = y_pred.astype(np.float64)

    ss_tot = np.sum((y_true_64 - y_true_64.mean(axis=c_axis, keepdims=True)) ** 2, axis=axis)
    eps = np.finfo(np.float64).eps
    ss_tot_arr = np.atleast_1d(ss_tot).astype(np.float64)

    def _r2_from_residual(diff):
        ss_res = np.sum(diff ** 2, axis=axis)
        ratio = np.divide(ss_res, ss_tot, out=np.ones_like(ss_tot_arr), where=ss_tot > eps)
        return np.where(ss_tot > eps, 1 - ratio, 0.0)

    r2 = _r2_from_residual(y_true_64 - y_pred_64)

    if polarity_invariant:
        r2_flipped = _r2_from_residual(y_true_64 + y_pred_64)
        r2 = np.maximum(r2, r2_flipped)

    r2 = np.asarray(r2)
    return r2.item() if r2.size == 1 else r2


def calc_ssim(
    arr_true: np.ndarray,
    arr_pred: np.ndarray,
    win_size: int = 7,
    data_range: float = None,
) -> float:
    """
    Structural Similarity (SSIM) between two arrays, treating them as a 2D
    signal (e.g. channels x time) rather than flattening or reducing along
    an axis -- captures local joint structure (a patch of channels over a
    short time window) that a single flattened correlation/cosine score can
    average away.

    `win_size` is clamped down to the largest odd size that fits the
    smaller data dimension, since scikit-image requires an odd window no
    larger than the smallest axis -- relevant for iEEG data, which can have
    channel counts well under the default win_size=7. Channel order is used
    as-is; reordering channels by spatial proximity (which would change the
    local-window result) is not attempted here.

    Parameters
    ----------
    arr_true, arr_pred: np.ndarray of shape (n_features, n_conditions)
        e.g. (n_channels, n_timepoints).

    win_size: int
        Side length of the sliding comparison window. Must be odd; even
        values are decremented by 1. Clamped down to fit
        `min(arr_true.shape)` if needed.

    data_range: float, optional
        Dynamic range of the data (max - min), required by SSIM for
        floating-point input. Defaults to the joint range of both arrays.

    Returns
    -------
    float
    """
    if arr_true.shape != arr_pred.shape:
        raise ValueError(
            f"Shape mismatch: arr_true {arr_true.shape} and arr_pred {arr_pred.shape} must be identical."
        )

    if data_range is None:
        joint = np.concatenate([arr_true.ravel(), arr_pred.ravel()])
        data_range = float(joint.max() - joint.min())

    win = min(win_size, min(arr_true.shape))
    if win % 2 == 0:
        win -= 1
    win = max(win, 1)

    score = structural_similarity(
        arr_true, arr_pred,
        win_size=win,
        data_range=data_range,
    )

    return float(score)


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


def calc_hypergeometric_enrichment(
    k: float,
    tp: float,
    population_size: float,
    n_positive: float,
) -> float:
    """
    Probability of matching a ground truth this well, by chance alone.

    Models drawing `k` units without replacement from a finite population of
    `population_size` units, `n_positive` of which are "positive" (e.g. inside
    a resection). Returns the one-sided p-value P(X >= tp) under that null --
    how surprising it is that `tp` of the `k` drawn units are positive.

    Parameters
    ----------
    k: float
        Candidate size (e.g. number of dipoles, or mm^2 of cortex). Rounded
        to the nearest integer -- scipy.stats.hypergeom silently returns NaN
        for non-integer parameters at real-world scale (confirmed: valid for
        all-integer inputs, or all-float at small scale, but not mixed
        large-and-fractional), so non-integer physical units (e.g. mm^2) need
        a fixed, whole-unit resolution to stay numerically valid -- 1 mm^2
        here.

    tp: float
        Number of `k` that are positive (e.g. inside the resection). Same
        unit as `k` and `population_size`/`n_positive`.

    population_size: float
        Total population size (e.g. total dipoles, or total cortex mm^2).

    n_positive: float
        Number of positive units in the population (e.g. resection size).

    Returns
    -------
    float
        P(X >= tp), the hypergeometric p-value. 1.0 if tp <= 0.
    """
    if tp <= 0:
        return 1.0

    k, tp = round(k), round(tp)
    population_size, n_positive = round(population_size), round(n_positive)

    return float(hypergeom.sf(tp - 1, population_size, n_positive, k))


def calc_fold_enrichment(
    k: float,
    tp: float,
    population_size: float,
    n_positive: float,
) -> float:
    """
    Effect-size companion to `calc_hypergeometric_enrichment`.

    How many times denser in positives the candidate is than the population
    base rate -- rewards purity independent of candidate size, unlike the
    p-value, which rewards size independent of purity. Report both together
    (e.g. as a volcano plot) rather than either alone.

    Parameters
    ----------
    k, tp, population_size, n_positive: float
        See `calc_hypergeometric_enrichment`.

    Returns
    -------
    float
        (tp / k) / (n_positive / population_size). 0 if tp <= 0.
    """
    if tp <= 0 or k <= 0:
        return 0.0

    base_rate = n_positive / population_size
    return (tp / k) / base_rate


def label_clusters(
    adjacency: sp.spmatrix,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Groups the active subset of a graph into connected components.

    Parameters
    ----------
    adjacency: scipy.sparse matrix of shape (n_nodes, n_nodes)
        Adjacency over the full node set.

    active_mask: np.ndarray of shape (n_nodes,), bool
        Which nodes to cluster (e.g. vertices above an amplitude threshold).

    Returns
    -------
    labels: np.ndarray of shape (n_active,)
        Cluster label for each active node, in the same order as
        `np.nonzero(active_mask)[0]`.

    sizes: np.ndarray of shape (n_clusters,)
        Number of nodes in each cluster, indexed by label.
    """
    active_idx = np.nonzero(active_mask)[0]
    sub_adjacency = adjacency[active_idx][:, active_idx]

    _, labels = connected_components(sub_adjacency, directed=False)
    sizes = np.bincount(labels)

    return labels, sizes


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


# -----------------------------------------------------------------------
# Graph / connectome perturbations
#
# For stress-testing connectome-prior methods (e.g. wCSET/CSET, see
# mindmap.analysis.functional.inverse.solvers) against uncertainty in the
# structural connectome they take as input. Two independent axes:
# `rewire_connectome` corrupts topology (which edges exist) while carrying
# weights along with their edge; `perturb_connectome_weights` and
# `permute_connectome_weights` corrupt weight magnitude while leaving
# topology untouched. Keeping the two axes independent lets a caller
# attribute a method's performance drop to "wrong structure" versus "wrong
# weight" rather than a conflated single "amount of sabotage" knob.
#
# Like `binarize`/`small_world_sigma` above, these treat edge weights as an
# opaque magnitude and never assume they represent connection strength
# versus length -- multiplicative rewiring/noise/permutation all commute
# with that ambiguity, so nothing here needs to guess at it.
# -----------------------------------------------------------------------

def rewire_connectome(
    adj: np.ndarray,
    n_rewire: int = 5,
    mask: np.ndarray = None,
    connected: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """
    Degree-preserving random rewiring of a connectome (Maslov & Sneppen 2002
    configuration-model null) -- the topology-uncertainty counterpart to
    `perturb_connectome_weights`/`permute_connectome_weights` below.

    `n_rewire=0` leaves `adj` untouched; a handful of rewires already
    saturates towards a fully randomized network with the same degree
    sequence -- the same null network `small_world_sigma` builds internally
    via `bct.randmio_und` for its own averaging, exposed here directly as a
    first-class perturbation instead of an internal implementation detail.
    There's no simple linear map from `n_rewire` to "fraction of edges
    corrupted": mixing saturates quickly (rarely needing more than
    ~10-20 for a fully mixed network), so sweep `n_rewire` itself as the
    severity axis rather than trying to reparametrize it as a percentage.

    Edge weights (whatever they represent -- connection strength, length,
    streamline count) travel with their edge during a swap; per
    `bct.randmio_und`'s own documentation it does not preserve their
    *distribution*, only which edge holds which original value.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected).

    n_rewire: int
        Rewiring parameter: each edge is rewired approximately this many
        times. 0 is a no-op.

    mask: np.ndarray of shape (n, n), bool, optional
        Symmetric mask of entries that must come out of this function
        exactly as they went in -- neither rewired away nor overwritten by
        an incoming swap. `bct.randmio_und` has no native concept of a
        protected edge, so masked entries are zeroed out of the matrix
        handed to it (making them invisible to the rewiring, as if those
        node pairs were never connected) and spliced back in afterwards.
        That guarantees masked entries are unchanged in the output, but as
        a consequence the degree-preservation guarantee formally holds for
        the *unmasked* subgraph, not the full graph -- nodes touching a
        masked edge can end up with their degree off by about the number
        of masked edges touching them. Negligible when `mask` covers a
        small fraction of edges (e.g. protecting a handful of edges of
        interest); not appropriate if `mask` covers most of the graph.
        This is a real improvement over the legacy
        `mindmap/OLD/utils/dmri.py::sabotage_connectome`, which only ever
        protected masked edges from *deletion*, not from being silently
        overwritten by a newly added one.

    connected: bool
        If True, use `bct.randmio_und_connected` instead, which
        additionally guarantees the result stays a single connected
        component (the input must already be connected). Relevant since a
        structural connectome is typically assumed connected.

    seed: int, optional
        Seed for the rewiring RNG.

    Returns
    -------
    np.ndarray of shape (n, n)
        Rewired adjacency matrix.
    """
    if n_rewire <= 0:
        return adj.copy()

    rewire_fn = bct.randmio_und_connected if connected else bct.randmio_und
    rng = np.random.default_rng(seed)
    bct_seed = int(rng.integers(0, 2**32 - 1))

    if mask is None:
        rewired, _ = rewire_fn(adj, n_rewire, seed=bct_seed)
        return rewired

    mask = mask.astype(bool)
    free_adj = adj * (~mask)

    rewired, _ = rewire_fn(free_adj, n_rewire, seed=bct_seed)
    rewired[mask] = adj[mask]

    return rewired


def swap_rewire_connectome(
    adj: np.ndarray,
    n_swaps: int,
    mask: np.ndarray = None,
    seed: int | None = None,
    batch_size: int = 1500,
    max_rounds: int = 200_000,
) -> np.ndarray:
    """
    Degree-preserving double-edge-swap rewiring (Maslov & Sneppen 2002
    configuration-model null), controlled by an *exact swap count* rather
    than `rewire_connectome`'s sweep count.

    Motivation: `rewire_connectome`'s `n_rewire` is passed to
    `bct.randmio_und` as `itr`, which internally becomes `itr *= n_edges`
    -- so `n_rewire=1` already schedules one attempted swap per edge in
    the whole graph. On a large sparse connectome (e.g. ~7500 nodes,
    ~218k edges, <1% density) that means even the smallest nonzero
    `n_rewire` already saturates close to full randomization: there's no
    `n_rewire` value that reliably lands at "20% of edges changed" versus
    "60% changed" versus "100%" -- everything above ~1 looks the same.
    Sweeping severity by `n_rewire` is the wrong knob for a graph this
    size; sweeping by the actual number of swaps performed is not, and
    that's what this function exposes directly.

    Each swap picks two existing edges (a,b) and (c,d) uniformly at
    random and replaces them with (a,d) and (c,b) -- or (a,c) and (d,b),
    decided by a random 50/50 flip so both rewiring directions are
    explored, matching `bct.randmio_und`'s own approach -- provided all
    4 endpoints are distinct and neither new edge already exists.
    Weights travel with their edge, same as `rewire_connectome`.
    Implemented as a vectorized batch process (propose many candidate
    swaps per round, reject invalid ones, then keep only the subset
    whose 4 endpoints don't overlap any other survivor in the same
    round so the whole batch can be applied in one array update) rather
    than `bct.randmio_und`'s scalar Python loop -- on the connectome
    above, full mixing (n_swaps ~ n_edges) takes under a second versus
    minutes.

    There's no exact analogue of "n_rewire=20 is roughly fully mixed"
    here either: the fraction of edges that end up different from the
    original rises with `n_swaps` along a saturating, coupon-collector-
    shaped curve, asymptoting below 1.0 -- around ~98-99% for a graph at
    this density, not 100%. That's because a swap's new edge position is
    drawn from *all* node pairs, not just pairs excluding the original
    edge positions, so a freshly rewired graph and the original one are
    expected to coincide on a small number of edges by chance alone
    (roughly `density * n_edges`). This is a real statistical property
    of a degree-preserving redraw at this density, not a bug -- when
    calibrating `n_swaps` to a target severity, treat the empirically
    measured plateau as the practical ceiling rather than assuming
    `n_swaps = n_edges` reaches 100% "different".

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected, no self-loops).

    n_swaps: int
        Exact number of successful double-edge swaps to perform. 0 is a
        no-op.

    mask: np.ndarray of shape (n, n), bool, optional
        Symmetric mask of entries that must come out of this function
        exactly as they went in. Same mechanism and caveat as
        `rewire_connectome`'s `mask`: masked entries are zeroed out of
        the matrix before rewiring (invisible to the algorithm as
        candidate edges to swap away) and spliced back in afterwards,
        which can silently overwrite a new edge a swap happened to place
        at a masked-absent position -- fine for a `mask` covering a
        small fraction of the graph, not appropriate for one covering
        most of it.

    seed: int, optional
        Seed for the rewiring RNG.

    batch_size: int
        Upper bound on candidate swaps proposed per round before
        filtering -- the actual per-round batch is also capped relative
        to node count (`n // 12`), since the endpoint-disjointness filter
        rejects almost everything once a round's node draws (4 per
        candidate) start approaching `n`, by a birthday-paradox effect.
        Larger batches mean fewer rounds but a lower per-round acceptance
        rate; the default is a reasonable middle ground, not tuned per
        graph.

    max_rounds: int
        Safety cap on the number of batch rounds. Raises `RuntimeError`
        if `n_swaps` isn't reached within this budget, rather than
        silently returning a partially-rewired result.

    Returns
    -------
    np.ndarray of shape (n, n)
        Rewired adjacency matrix.
    """
    if not np.array_equal(adj, adj.T):
        raise ValueError("swap_rewire_connectome: adj must be symmetric (undirected)")

    if n_swaps <= 0:
        return adj.copy()

    if mask is not None:
        mask = mask.astype(bool)
        free_adj = adj * (~mask)
        rewired = swap_rewire_connectome(free_adj, n_swaps, seed=seed, batch_size=batch_size, max_rounds=max_rounds)
        rewired[mask] = adj[mask]
        return rewired

    n = adj.shape[0]
    rng = np.random.default_rng(seed)

    rows, cols = np.triu_indices(n, k=1)
    edge_mask = adj[rows, cols] != 0
    ei, ej = rows[edge_mask].copy(), cols[edge_mask].copy()
    n_edges = len(ei)

    R = adj.copy()
    remaining = n_swaps
    rounds = 0

    while remaining > 0 and rounds < max_rounds:
        rounds += 1
        b = min(batch_size, n_edges // 2, max(4, n // 12))
        pick = rng.choice(n_edges, size=2 * b, replace=False)
        e1, e2 = pick[:b], pick[b:]

        a, bb = ei[e1], ej[e1]
        c, d = ei[e2], ej[e2]

        flip = rng.random(b) < 0.5
        c, d = np.where(flip, d, c), np.where(flip, c, d)

        ok = (a != c) & (a != d) & (bb != c) & (bb != d)
        a, bb, c, d, e1, e2 = a[ok], bb[ok], c[ok], d[ok], e1[ok], e2[ok]
        if len(a) == 0:
            continue

        ok = (R[a, d] == 0) & (R[c, bb] == 0)
        a, bb, c, d, e1, e2 = a[ok], bb[ok], c[ok], d[ok], e1[ok], e2[ok]
        if len(a) == 0:
            continue

        # keep only proposals whose 4 endpoints appear nowhere else in this
        # round's survivors -> fully node-disjoint, safe to apply as one
        # vectorized update with no ordering hazard between proposals.
        nodes = np.stack([a, bb, c, d], axis=1)
        counts = np.bincount(nodes.ravel(), minlength=n)
        ok = np.all(counts[nodes] == 1, axis=1)
        a, bb, c, d, e1, e2 = a[ok], bb[ok], c[ok], d[ok], e1[ok], e2[ok]
        if len(a) == 0:
            continue

        if len(a) > remaining:
            a, bb, c, d, e1, e2 = (x[:remaining] for x in (a, bb, c, d, e1, e2))

        w_ab, w_cd = R[a, bb], R[c, d]

        R[a, bb] = 0
        R[bb, a] = 0
        R[c, d] = 0
        R[d, c] = 0
        R[a, d] = w_ab
        R[d, a] = w_ab
        R[c, bb] = w_cd
        R[bb, c] = w_cd

        ei[e1] = a
        ej[e1] = d
        ei[e2] = c
        ej[e2] = bb

        remaining -= len(a)

    if remaining > 0:
        raise RuntimeError(
            f"swap_rewire_connectome: only {n_swaps - remaining}/{n_swaps} swaps applied "
            f"within max_rounds={max_rounds}; the requested n_swaps may be unreachable at "
            f"this graph's size/density (see the docstring's note on the ~98-99%-not-100% "
            f"asymptote), or max_rounds needs raising."
        )

    return R


def perturb_connectome_topology(
    adj: np.ndarray,
    n_deletions: int,
    n_additions: int,
    deletion_weights: np.ndarray = None,
    addition_weights: np.ndarray = None,
    mask: np.ndarray = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Independent random deletion and addition of edges -- the non-degree-
    preserving counterpart to `swap_rewire_connectome`/`rewire_connectome`.

    Those two only ever *permute* a fixed edge set: same edges in, same
    edges out, just relabeled. That's the right null model for "the
    topology is randomly shuffled but the connectome's coarse statistics
    (degree sequence, edge count) are still correct" -- but it can't
    represent what real dMRI/tractography errors actually look like: a
    pipeline doesn't conserve a node's total connection count while
    getting individual edges wrong, it genuinely misses some true
    connections and hallucinates some spurious ones (see
    `dmri_uncertainty_weights_from_source_space` in
    `mindmap.analysis.connectivity.metric` for weights calibrated to that
    literature). This function is the more general primitive underneath
    both a "brutal", fully unconstrained corruption (uniform sampling,
    the default) and that dMRI-realistic one (spatially-biased sampling
    via `deletion_weights`/`addition_weights`).

    Deletions and additions are drawn independently and simultaneously
    from the *original* `adj` (an edge can never be both deleted and
    immediately re-added, or vice versa, since a deletion candidate is by
    definition an existing edge and an addition candidate is by
    definition not) via weighted sampling without replacement
    (Efraimidis-Spirakis: draw `key = -log(u) / weight` per candidate,
    `u ~ Uniform(0,1)`, keep the smallest `n` keys -- equivalent to
    weighted sampling without replacement, and fully vectorized, unlike
    `numpy.random.Generator.choice(..., p=..., replace=False)`, which
    doesn't scale to the ~10^7 candidate positions a connectome this size
    has). Uniform sampling (`deletion_weights`/`addition_weights=None`,
    the default) skips the weighting machinery entirely and just draws
    uniformly at random.

    New edges' weights are drawn uniformly at random (with replacement)
    from `adj`'s existing nonzero-weight multiset -- the same "treat
    weights as an opaque magnitude, reuse what's already there" approach
    `permute_connectome_weights` uses below, rather than inventing a
    weight model from nothing.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Adjacency matrix (weighted or binary, undirected, no self-loops).

    n_deletions: int
        Number of existing edges to remove.

    n_additions: int
        Number of new edges to add. Can differ from `n_deletions` --
        unlike `swap_rewire_connectome`, total edge count is not
        conserved here; that's the point.

    deletion_weights: np.ndarray of shape (n, n), optional
        Relative sampling weight per existing edge (only entries at
        existing edges are used). None (default) samples deletions
        uniformly at random.

    addition_weights: np.ndarray of shape (n, n), optional
        Relative sampling weight per non-edge position. None (default)
        samples additions uniformly at random.

    mask: np.ndarray of shape (n, n), bool, optional
        Symmetric mask of entries that must come out of this function
        exactly as they went in. Unlike `rewire_connectome`'s mask (which
        zeroes masked entries out before rewiring and splices them back
        in afterwards -- an approximation that degrades as the mask
        covers more of the graph), this function simply excludes masked
        positions from both the deletion and addition candidate pools up
        front, which is an exact guarantee regardless of how much of the
        graph `mask` covers.

    seed: int, optional
        Seed for the sampling RNG.

    Returns
    -------
    np.ndarray of shape (n, n)
        Adjacency matrix with `n_deletions` existing edges removed and
        `n_additions` new edges added.
    """
    n = adj.shape[0]
    rng = np.random.default_rng(seed)

    rows, cols = np.triu_indices(n, k=1)
    values = adj[rows, cols]
    is_edge = values != 0

    if mask is not None:
        not_masked = ~mask.astype(bool)[rows, cols]
    else:
        not_masked = np.ones(len(is_edge), dtype=bool)

    del_candidates = np.flatnonzero(is_edge & not_masked)
    add_candidates = np.flatnonzero(~is_edge & not_masked)

    if n_deletions > len(del_candidates):
        raise ValueError(
            f"perturb_connectome_topology: requested n_deletions={n_deletions} but only "
            f"{len(del_candidates)} unmasked existing edges are available"
        )
    if n_additions > len(add_candidates):
        raise ValueError(
            f"perturb_connectome_topology: requested n_additions={n_additions} but only "
            f"{len(add_candidates)} unmasked non-edge positions are available"
        )

    def weighted_sample(candidates, weights, k):
        if k == 0:
            return candidates[:0]
        if weights is None:
            return rng.choice(candidates, size=k, replace=False)
        w = weights[rows[candidates], cols[candidates]]
        with np.errstate(divide="ignore"):
            # a weight of exactly 0 -> key=+inf -> that candidate is never
            # among the smallest k, i.e. never selected. Intentional.
            keys = -np.log(rng.random(len(candidates))) / w
        top_k = np.argpartition(keys, k - 1)[:k]
        return candidates[top_k]

    del_idx = weighted_sample(del_candidates, deletion_weights, n_deletions)
    add_idx = weighted_sample(add_candidates, addition_weights, n_additions)

    R = adj.copy()

    R[rows[del_idx], cols[del_idx]] = 0
    R[cols[del_idx], rows[del_idx]] = 0

    if n_additions > 0:
        new_weights = rng.choice(values[is_edge], size=n_additions, replace=True)
        R[rows[add_idx], cols[add_idx]] = new_weights
        R[cols[add_idx], rows[add_idx]] = new_weights

    return R


def perturb_connectome_weights(
    adj: np.ndarray,
    sigma: float,
    mask: np.ndarray = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Multiplicative log-normal noise on existing edge weights, topology
    untouched -- the continuous weight-uncertainty axis, independent of
    `rewire_connectome`'s topology corruption.

    Only nonzero (existing) entries of `adj` are perturbed and no new
    edges are created: `sigma=0` is an exact no-op, and larger `sigma`
    increasingly decorrelates a weight from its original value without
    ever changing which edges exist.

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Weighted adjacency matrix, undirected.

    sigma: float
        Log-normal noise scale. Each existing edge weight w becomes
        w * exp(sigma * Z), Z ~ N(0,1) -- one Z per edge, shared between
        (i,j) and (j,i) so the result stays symmetric.

    mask: np.ndarray of shape (n, n), bool, optional
        Symmetric mask of entries excluded from the noise.

    seed: int, optional
        Seed for the noise RNG.

    Returns
    -------
    np.ndarray of shape (n, n)
        Weight-perturbed adjacency matrix.
    """
    adj = adj.astype(float)

    if sigma == 0:
        return adj.copy()

    rng = np.random.default_rng(seed)
    n = adj.shape[0]

    log_noise = np.triu(rng.normal(0, sigma, size=(n, n)), k=1)
    log_noise = log_noise + log_noise.T

    edge_mask = adj != 0
    if mask is not None:
        edge_mask &= ~mask.astype(bool)

    result = adj.copy()
    result[edge_mask] *= np.exp(log_noise[edge_mask])

    return result


def permute_connectome_weights(
    adj: np.ndarray,
    mask: np.ndarray = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Shuffle existing edge weights across existing edge positions --
    topology and the weight *distribution* both untouched. The discrete
    "right graph shape, wrong which edge is strong" endpoint, complementary
    to `perturb_connectome_weights`'s continuous noise (and what
    `perturb_connectome_weights` approaches as sigma grows very large).

    Parameters
    ----------
    adj: np.ndarray of shape (n, n)
        Weighted adjacency matrix, undirected.

    mask: np.ndarray of shape (n, n), bool, optional
        Symmetric mask of edges excluded from the shuffle: their weight
        stays fixed in place, and their value isn't drawn into the pool
        shuffled among the other edges.

    seed: int, optional
        Seed for the shuffle RNG.

    Returns
    -------
    np.ndarray of shape (n, n)
        Adjacency matrix with the same topology and the same multiset of
        edge weights as `adj`, reassigned to different edge positions.
    """
    adj = adj.astype(float)
    rng = np.random.default_rng(seed)

    edge_mask = np.triu(adj != 0, k=1)
    if mask is not None:
        edge_mask &= ~np.triu(mask.astype(bool), k=1)

    rows, cols = np.nonzero(edge_mask)
    shuffled = rng.permutation(adj[rows, cols])

    result = adj.copy()
    result[rows, cols] = shuffled
    result[cols, rows] = shuffled

    return result