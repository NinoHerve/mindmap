"""
Module Name: analysis.functional.common.metric.py
Description: Common metrics for other sub-modules in this module.

This module provides functions to measure data.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import pandas as pd

import mne

from mindmap.core.metric import (
    calc_confusion_matrix, calc_activity_scores,
    calc_performance_curves, calc_correlation,
    calc_cosine,
)
from mindmap.analysis.functional.common.utils import (
    is_binary_mne,
)


def compute_mne_correlation(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
    method: str = "pearson",
) -> float | np.ndarray:
    """
    Compute correlation.
    
    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']

    Returns
    -------
    corr: float | np.ndarray
        Correlation.
    """
    y_true = obj_true.data
    y_pred = obj_pred.data

    # map mode to axis
    axis_map = {"spatio-temporal": None, "spatial": 0, "temporal": 1}
    target_axis = axis_map[mode]
    center_axis = axis_map["temporal"] if target_axis is None else target_axis

    return calc_correlation(y_true, y_pred, method, target_axis, center_axis)


def compute_mne_cosine(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> float | np.ndarray:
    """
    Compute cosine similarity.
    
    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth acitvity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
    
    Returns
    -------
    cos: float | np.ndarray
        Cosine similarity.
    """
    y_true = obj_true.data
    y_pred = obj_pred.data

    # map mode to axis
    axis_map = {"spatio-temporal": None, "spatial": 0, "temporal": 1}
    target_axis = axis_map[mode]

    return calc_cosine(y_true, y_pred, target_axis)


def compute_mne_confusion(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Compute confusion matrix.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binarized ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
        Binarized estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    # check if inputs are binary
    if not is_binary_mne(obj_true) or not is_binary_mne(obj_pred):
        raise TypeError("Input data needs to be binary for computing the confusion matrix.")

    y_true = obj_true.data
    y_pred = obj_pred.data

    # dynamic labels (channels vs dipoles)
    label_name = "channel" if hasattr(obj_true, "ch_names") else "dipole_index"
    labels = obj_true.ch_names if hasattr(obj_true, "ch_names") else np.arange(len(y_true))

    # Count positives & negatives
    if mode == "spatio-temporal":
        return calc_confusion_matrix(y_true, y_pred)

    results = []
    if mode == "spatial":
        for t in range(y_true.shape[1]):
            res = calc_confusion_matrix(y_true[:, t], y_pred[:, t])
            res["time"] = obj_true.times[t]
            results.append(res)
        return pd.DataFrame(results).set_index("time")

    if mode == "temporal":
        for n in range(y_true.shape[0]):
            res = calc_confusion_matrix(y_true[n, :], y_pred[n, :])
            res[label_name] = labels[n]
            results.append(res)
        return pd.DataFrame(results).set_index(label_name)



def compute_mne_scores(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Compute scores.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
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
    # check if inputs are binary
    if not is_binary_mne(obj_true) or not is_binary_mne(obj_pred):
        raise TypeError("Input data needs to be binary for computing performance scores.")

    y_true = obj_true.data
    y_pred = obj_pred.data

    # dynamic labels (channels vs dipoles)
    label_name = "channel" if hasattr(obj_true, "ch_names") else "dipole_index"
    labels = obj_true.ch_names if hasattr(obj_true, "ch_names") else np.arange(len(y_true))

    # calculate scores
    if mode == "spatio-temporal":
        return calc_activity_scores(y_true, y_pred)

    results = []
    if mode == "spatial":
        for t in range(y_true.shape[1]):
            res = calc_activity_scores(y_true[:, t], y_pred[:, t])
            res["time"] = obj_true.times[t]
            results.append(res)
        return pd.DataFrame(results).set_index("time")

    if mode == "temporal":
        for n in range(y_true.shape[0]):
            res = calc_activity_scores(y_true[n, :], y_pred[n, :])
            res[label_name] = labels[n]
            results.append(res)
        return pd.DataFrame(results).set_index(label_name)


def compute_mne_curves(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
) -> dict:
    """
    Computes data for ROC and PR curves.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    # check if obj_true is binary
    if not is_binary_mne(obj_true):
        raise TypeError("Ground truth needs to be binary to compute ROC and PR curves.")

    y_true = obj_true.data
    y_pred = np.abs(obj_pred.data)

    return calc_performance_curves(y_true, y_pred)
