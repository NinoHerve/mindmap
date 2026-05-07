"""
Module Name: inverse/metric.py
Description: This module provides functions to evaluate/measure source estimates.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np

import mne

from mindmap.analysis.functional.common.metric import (
    compute_mne_confusion, compute_mne_scores, compute_mne_curves,
    compute_mne_correlation, compute_mne_cosine,
)

def compute_stc_correlation(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "temporal",
    method: str = "pearson",
) -> np.ndarray | float:
    """
    Compute correlation between two evoked objects.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        Reference activity (e.g. True iEEG).

    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG).

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': Correlation across channels at each timepoint -> (n_timepoints,)
        - 'temporal': Correlation across time for each channel -> (n_channels,)
        - 'sptio-temporal': Flattened correlation -> scalar

    method: str ['pearson', 'spearman', 'log-envelope']
        Correlation method.

    Returns
    -------
    corr: float or np.ndarray
        Correlation value(s). Shape depends on axis and average.
    """

    return compute_mne_correlation(
        stc_true,
        stc_pred,
        mode = mode,
        method = method,
    )


def compute_evoked_cosine(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "temporal",
) -> np.ndarray | float:
    """
    Compute cosine between two evoked objects.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        Reference activity (e.g. True iEEG).

    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG).

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': Correlation across channels at each timepoint -> (n_timepoints,)
        - 'temporal': Correlation across time for each channel -> (n_channels,)
        - 'sptio-temporal': Flattened correlation -> scalar

    method: str ['pearson', 'spearman', 'log-envelope']
        Correlation method.

    Returns
    -------
    corr: float or np.ndarray
        Correlation value(s). Shape depends on axis and average.
    """

    return compute_mne_cosine(
        stc_true,
        stc_pred,
        mode = mode,
    )


def compute_stc_confusion(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Wrapper for `compute_mne_confusion`

    Parameters
    ----------
    stc_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    stc_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_confusion(
        obj_true = stc_true,
        obj_pred = stc_pred,
        mode = mode,
    )


def compute_stc_scores(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Wrapper for `compute_mne_scores`

    Parameters
    ----------
    stc_true: mne.SourceEstimate
        Ground truth activity.
    
    stc_pred: mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.
    
    Returns
    -------
    dict
        Dictionary of scores.
    """
    return compute_mne_scores(
        obj_true = stc_true,
        obj_pred = stc_pred,
        mode = mode,
    )


def compute_stc_curves(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
) -> dict:
    """
    Wrapper for `compute_mne_curves`.

    Parameters
    ----------
    stc_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    stc_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_curves(
        obj_true = stc_true,
        obj_pred = stc_pred,
    )
