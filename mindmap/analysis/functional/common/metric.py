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
    calc_performance_curves,
)
from mindmap.analysis.functional.common.utils import (
    is_binary_mne,
)

def compute_mne_confusion(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
    rng: tuple = (None, None),
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
    # check if inputs are binary
    if not is_binary_mne(obj_true) or not is_binary_mne(obj_pred):
        raise TypeError("Input data needs to be binary for computing the confusion matrix.")

    # crop data
    o_true = obj_true.copy().crop(*rng)
    o_pred = obj_pred.copy().crop(*rng)

    y_true = o_true.data
    y_pred = o_pred.data

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
            res["time"] = o_true.times[t]
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
    rng: tuple = (None, None),
) -> dict:
    """
    Compute scores.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
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
    # check if inputs are binary
    if not is_binary_mne(obj_true) or not is_binary_mne(obj_pred):
        raise TypeError("Input data needs to be binary for computing performance scores.")

    # crop data
    o_true = obj_true.copy().crop(*rng)
    o_pred = obj_pred.copy().crop(*rng)

    y_true = o_true.data
    y_pred = o_pred.data

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
            res["time"] = o_true.times[t]
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
    rng: tuple = (None, None),
) -> dict:
    """
    Computes data for ROC and PR curves.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
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
    # check if obj_true is binary
    if not is_binary_mne(obj_true):
        raise TypeError("Ground truth needs to be binary to compute ROC and PR curves.")

    # crop data
    o_true = obj_true.copy().crop(*rng)
    o_pred = obj_pred.copy().crop(*rng)

    y_true = o_true.data
    y_pred = np.abs(o_pred.data)

    return calc_performance_curves(y_true, y_pred)
