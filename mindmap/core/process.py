"""
Module Name: core/process.py
Description: This module provides functions to process simple data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np
from scipy.ndimage import shift
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
