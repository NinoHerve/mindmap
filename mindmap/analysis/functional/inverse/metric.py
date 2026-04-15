"""
Module Name: inverse/metric.py
Description: This module provides functions to evaluate/measure source estimates.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import mne

from mindmap.analysis.functional.common.metric import (
    compute_mne_confusion, compute_mne_scores, compute_mne_curves,
)

def compute_stc_confusion(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "spatio-temporal",
    rng: tuple = (None, None),
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

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.
    
    threshold:
        Relative maximum threshold for binarization.

    rng: tuple (float, float)
        Range of data to consider for scoring.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_confusion(
        obj_true = stc_true,
        obj_pred = stc_pred,
        mode = mode,
        rng = rng,
    )


def compute_stc_scores(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "spatio-temporal",
    rng: tuple = (None, None),
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

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.
    
    threshold:
        Relative maximum threshold for binarization.

    rng: tuple (float, float)
        Range of data to consider for scoring.
    
    Returns
    -------
    dict
        Dictionary of scores.
    """
    return compute_mne_scores(
        obj_true = stc_true,
        obj_pred = stc_pred,
        mode = mode,
        rng = rng,
    )


def compute_stc_curves(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    rng: tuple = (None, None),
) -> dict:
    """
    Wrapper for `compute_mne_curves`.

    Parameters
    ----------
    stc_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    stc_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.

    rng: tuple (float, float)
        Range of data to consider for scoring.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_curves(
        obj_true = stc_true,
        obj_pred = stc_pred,
        rng = rng,
    )
