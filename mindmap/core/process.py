"""
Module Name: core/process.py
Description: This module provides functions to process simple data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np
from scipy.ndimage import shift, map_coordinates
from scipy.stats import rankdata

from mindmap.core.metric import compute_correlation


def normalize_data(data, strategy="global", method="zscore"):
    """
    Normalizes a dataset (Channels x Time) using various strategies and methods.

    Parameters
    ----------
    data: np.ndarray of shape (n_channels, n_timepoints)
        The input data.

    strategy: str
        'global': Normalize using the max/mean of the entire matrix.
        'spatial': Normalize each timepoint (column) independently.
        'temporal': Normalize each channel (row) independently.

    method: str
        'max': scale by absolute maximum (range [-1, 1]). Preserves zero-baseline.
        'minmax': scale to range [0, 1]. Best for strictly positive data.
        'zscore': (x - mean) / std. Center at 0 with unit variance.
        'rank': percentile rank, scaled to range [0, 1]. Robust to outliers/skew,
        since it only uses the ordering of values, not their magnitude.

    Returns
    -------
    norm_data : np.ndarray
        Normalized data of same shape as input.
    """
    # 1. Determine the reduction axis based on strategy
    # Axis=None for global, Axis=0 for spatial (cols), Axis=1 for temporal (rows)
    strat_map = {"global": None, "spatial":0, "temporal": 1}
    axis = strat_map.get(strategy, None)

    if method == "max":
        # Dividing by max absolute value
        denom = np.max(np.abs(data), axis=axis, keepdims=True)
        return np.divide(data, denom, out=np.zeros_like(data), where=denom > 1e-12)

    elif method == "zscore":
        # Subtract mean and divide by standard deviation
        mu = np.mean(data, axis=axis, keepdims=True)
        sigma = np.std(data, axis=axis, keepdims=True)
        return np.divide(data - mu, sigma, out=np.zeros_like(data - mu), where=sigma > 1e-12)

    elif method == "minmax":
        # Traditional 0 to 1 scaling
        d_min = np.min(data, axis=axis, keepdims=True)
        d_max = np.max(data, axis=axis, keepdims=True)
        denom = d_max - d_min
        return np.divide(data - d_min, denom, out=np.zeros_like(data), where=denom > 1e-12)

    elif method == "robust":
        # Robust scaling (x - median) / (q75 - q25)
        median = np.median(data, axis=axis, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis, keepdims=True)
        iqr = q75 - q25
        return np.divide(data - median, iqr, out=np.zeros_like(data), where=iqr > 1e-12)

    elif method == "rank":
        # percentile rank in [0, 1], ranking over the flattened array when axis is None
        ranks = rankdata(data, axis=axis, method="average")
        n = data.shape[axis] if axis is not None else data.size
        return (ranks - 1) / max(n - 1, 1)

    else:
        raise ValueError(f"Unknown normalization method: {method}")



def align_timeseries(ref, target, max_lag=20, window=None):
    """
    Finds a global shift for 'target' based on a specific window.

    Parameters:
    ----------
    ref: np.ndarray of shape (N_regions, T_time)
        Reference timeseries.

    target: np.ndarray of shape (N_regions, T_time)
        Target timeseries

    max_lag: int
        Search range for the shift.

    window: tuple(int, int)
        (start, end) tuple of indices. If None, uses the whole series.


    Returns
    -------
    aligned_target: np.ndarray of shape (N_regions, T_time)
        The full target array shifted by the best lag.

    best_lag: int
        The integer lag found.
    """
    if window is None:
        start, end = max_lag, ref.shape[1] - max_lag
    else:
        start, end = window

    lags = np.arange(-max_lag, max_lag + 1)
    scores = []

    for lag in lags:
        # Shift target by 'lag' on the time axis (axis 1)
        shifted = shift(target, (0, lag), mode="nearest")

        # Calculate mean spatial correlation within window
        ref_win = ref[:, start:end]
        target_win = shifted[:, start:end]

        r_per_timepoint = compute_correlation(ref_win, target_win, axis=0)
        scores.append(np.mean(r_per_timepoint))

    best_lag = lags[np.argmax(scores)]
    aligned_target = shift(target, (0, best_lag), mode='nearest')

    return aligned_target, best_lag


def normalize_timeserie_by_baseline(
    timeserie: np.ndarray,
    baseline: tuple[int, int],
    epsilon: float = 1e-15,
) -> np.ndarray:
    """
    Normalize timeserie by the standard deviation of the baseline.

    Parameters
    ----------
    timeserie: np.ndarray of shape (n_samples, n_timepoints)
        Timeserie.

    baseline: tuple[int, int]
        (idx_min, idx_max) defining the baseline period.

    epsilon: float
        Small number to check for zeros.

    Returns
    -------
    np.ndarray of shape (n_samples, n_timepoints)
        Baseline normalized timeseries.
    """
    timeserie = np.atleast_2d(timeserie)

    # extract baseline
    idx_min, idx_max = baseline
    baseline_data = timeserie[:, idx_min:idx_max]

    # calculate SD across baseline for each sample
    baseline_std = np.std(baseline_data, axis=1, keepdims=True)
    baseline_mean = np.mean(baseline_data, axis=1, keepdims=True)

    # avoid division by zero flat/bad samples
    baseline_std[baseline_std < epsilon] = 1.0

    # apply z-score
    normalized = (timeserie - baseline_mean) / baseline_std

    return normalized.squeeze()


def sample_volume_at_points(
    volume: np.ndarray,
    points: np.ndarray,
    method: str = "nearest",
    fill_value: float = 0,
) -> np.ndarray:
    """
    Sample a volumetric array at continuous array-index coordinates.

    Parameters
    ----------
    volume: np.ndarray of shape (X, Y, Z)
        Volumetric data.

    points: np.ndarray of shape (n_points, 3)
        Coordinates into `volume`, expressed in array-index (voxel) space --
        i.e. fractional indices, not world/scanner coordinates.

    method: str ['nearest', 'linear']
        Interpolation order.

    fill_value: float
        Value assigned to points falling outside `volume`.

    Returns
    -------
    np.ndarray of shape (n_points,)
        Sampled values.
    """
    order_map = {"nearest": 0, "linear": 1}
    if method not in order_map:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(order_map)}.")

    return map_coordinates(
        volume,
        points.T,
        order=order_map[method],
        mode="constant",
        cval=fill_value,
    )


def _pr_envelope(precision, recall):
    """
    Sorts a PR curve by recall and computes the "best precision achievable at
    any recall >= r" envelope -- shared by `resample_pr_curve` and
    `precision_at_recall`, since both need it before interpolating.
    """
    order = np.argsort(recall)
    recall_sorted = np.asarray(recall)[order]
    precision_sorted = np.asarray(precision)[order]

    envelope = np.maximum.accumulate(precision_sorted[::-1])[::-1]

    return recall_sorted, envelope


def resample_pr_curve(precision, recall, n_points=100):
    """
    Resamples a precision-recall curve onto a fixed, common recall grid.

    A raw curve (e.g. from `mm.core.calc_pr_curve`) carries as many points as
    there are unique prediction values, which gets unwieldy when pooling many
    runs and makes curves from different runs impossible to average directly
    (their recall values don't line up). Interpolating onto a shared grid
    solves both: it bounds the point count and gives every run the same
    x-axis so per-run curves can be averaged into a summary curve with a CI.

    Uses the standard "interpolated precision" (max precision achievable at any
    recall >= r), which also removes the sawtooth wiggle of the raw curve.

    Parameters
    ----------
    precision, recall: array-like
        Raw precision-recall curve, e.g. from `mm.core.calc_pr_curve`.

    n_points: int
        Number of points of the resampled curve.

    Returns
    -------
    precision_grid, recall_grid: np.ndarray of shape (n_points,)
    """
    recall_sorted, envelope = _pr_envelope(precision, recall)

    recall_grid = np.linspace(0, 1, n_points)
    precision_grid = np.interp(recall_grid, recall_sorted, envelope)

    return precision_grid, recall_grid


def precision_at_recall(precision, recall, target_recall):
    """
    Interpolated precision at a fixed recall level.

    Answers a narrower question than AUC-PR: of the vertices needed to reach
    `target_recall`, how many were actually correct? This isolates "is the
    confident top answer reliable" from whole-curve behavior -- AUC-PR can
    rank a smoother, more spread-out method above one with a sharper, more
    trustworthy top answer, since AUC-PR also credits tail behavior this
    doesn't look at at all.

    Uses the same "best precision achievable at any recall >= r" envelope as
    `resample_pr_curve`, to avoid the raw curve's sawtooth noise.

    Parameters
    ----------
    precision, recall: array-like
        Raw precision-recall curve, e.g. from `mm.core.calc_pr_curve`.

    target_recall: float
        Recall level to evaluate precision at.

    Returns
    -------
    float
    """
    recall_sorted, envelope = _pr_envelope(precision, recall)

    return float(np.interp(target_recall, recall_sorted, envelope))


def resample_roc_curve(fpr, tpr, n_points=100):
    """
    Resamples a ROC curve onto a fixed, common false-positive-rate grid.

    Same motivation as `resample_pr_curve` (bound the point count, give every
    run a shared x-axis for averaging), but unlike precision-vs-recall, tpr is
    already a proper non-decreasing function of fpr -- lowering the threshold
    can only add to both the true- and false-positive counts. So there's no
    sawtooth to remove here: this is a plain interpolation, with no envelope
    step. Reusing `resample_pr_curve`'s envelope on (fpr, tpr) would collapse
    the curve to a flat line at max(tpr), since the max of any suffix of an
    already non-decreasing array is just its last element.

    Parameters
    ----------
    fpr, tpr: array-like
        Raw ROC curve, e.g. from `mm.core.calc_roc_curve`.

    n_points: int
        Number of points of the resampled curve.

    Returns
    -------
    tpr_grid, fpr_grid: np.ndarray of shape (n_points,)
    """
    order = np.argsort(fpr)
    fpr_sorted = np.asarray(fpr)[order]
    tpr_sorted = np.asarray(tpr)[order]

    fpr_grid = np.linspace(0, 1, n_points)
    tpr_grid = np.interp(fpr_grid, fpr_sorted, tpr_sorted)

    return tpr_grid, fpr_grid


def paint_on_dense_mesh(
    values: np.ndarray,
    vertex_map: np.ndarray,
    fill_value: float = 0,
) -> np.ndarray:
    """
    Broadcast sparse per-dipole values onto a dense mesh via a vertex map.

    Parameters
    ----------
    values: np.ndarray of shape (n_dipoles, ...)
        Sparse per-dipole values (e.g. source activity, or a binary mask) for
        a single hemisphere. Any trailing dimensions (e.g. time) are carried
        through unchanged.

    vertex_map: np.ndarray of shape (n_dense_vertices,)
        Per-dense-vertex index into `values` (its nearest dipole), or -1 for
        vertices whose nearest dipole isn't in-use. See
        `mm.anatomy.source_space.build_dense_vertex_map`.

    fill_value: float
        Value assigned to dense-mesh vertices mapped to -1.

    Returns
    -------
    np.ndarray of shape (n_dense_vertices, ...)
        `values` broadcast onto the dense mesh.
    """
    pad = np.full((1,) + values.shape[1:], fill_value, dtype=values.dtype)
    padded = np.concatenate([values, pad], axis=0)

    return padded[vertex_map]


def binarize_signal(
    signal: np.ndarray,
    threshold: float = 0.5,
    method: str = "max",
    absolute: bool = False,
    baseline: tuple = (0, -1)
) -> np.ndarray:
    """
    Binarize numpy array.

    Parameters
    ----------
    signal: np.ndarray of shape (samples, time points)
        Signal array.

    threshold: float
        Threshold for binarization.

    method: str [none, ''max', 'std']
        Method for thresholding:
        - none: Consider threshold raw.
        - max: Considers threshold as a percentage [0-1] of maximum value.
        - std: Considers threshold as a multiple of standard deviation (requires noise baseline).

    absolute: bool
        If true, apply thresholding on absolute value of signal.

    Returns:
    -------
    signal_bin: np.ndarray
        Binarized array.
    """

    # absolute
    if absolute:
        signal = np.abs(signal)

    # thresholding
    if method is None:
        return signal > threshold

    if method == "max":
        thresh = np.max(signal) * threshold
        return signal > thresh

    if method == "std":
        # baseline stats
        start, stop = baseline
        baseline_period = signal[:, start:stop]
        b_mean = np.mean(baseline_period)
        b_std = np.std(baseline_period)

        return signal > (b_mean + threshold * b_std)

    raise ValueError(f"Unknown binarization method {method}")
