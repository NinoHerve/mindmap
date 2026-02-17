"""
Module Name: core/process.py
Description: This module provides functions to process simple data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np


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
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")