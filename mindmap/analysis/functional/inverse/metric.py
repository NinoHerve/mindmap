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
    compute_mne_confusion, compute_mne_scores,
    compute_mne_roc, compute_mne_pr,
    compute_mne_correlation, compute_mne_cosine,
    find_peak_localization_error,
)
from mindmap.core.metric import calc_pr_curve

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


def compute_stc_cosine(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
    mode: str = "temporal",
    absolute: bool = False,
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
        absolute = absolute,
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
    stc_true: mne.SourceEstimate
        Binary ground truth activity.
    
    stc_pred: mne.SourceEstimate
        Binary estimated activity.

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
        Binary ground truth activity.
    
    stc_pred: mne.SourceEstimate
        Binary estimated activity.

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


def compute_stc_roc(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
) -> dict:
    """
    Wrapper for `compute_mne_roc`.

    Parameters
    ----------
    stc_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.

    stc_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    dict
        Dictionary with 'fpr', 'tpr' and 'auc'.
    """
    return compute_mne_roc(
        obj_true = stc_true,
        obj_pred = stc_pred,
    )


def compute_stc_pr(
    stc_true: mne.SourceEstimate,
    stc_pred: mne.SourceEstimate,
) -> dict:
    """
    Wrapper for `compute_mne_pr`.

    Parameters
    ----------
    stc_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.

    stc_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    dict
        Dictionary with 'precision', 'recall' and 'ap'.
    """
    return compute_mne_pr(
        obj_true = stc_true,
        obj_pred = stc_pred,
    )


def score_source_reconstruction(
    gold: mne.SourceEstimate,
    solutions: dict,
    hrp_idx: int,
    pp_idx: int,
    curve_idx: np.ndarray,
    hrp_col: int,
    pp_col: int,
    curve_cols: np.ndarray,
    true_source_mask: np.ndarray,
    dipole_positions: np.ndarray,
) -> dict:
    """
    Scores one or more source reconstructions against `gold` on the three
    outcome measures the pilot-sizing studies compare: peak localization
    error at the half-rising-phase timepoint, average precision (AP) at
    half-rising-phase and peak-phase, and a spatial-correlation-to-gold
    curve (raw curve + AUC-over-time summary).

    Indices/columns match `build_phase_data_hrp_pp_curve`'s output --
    `hrp_idx`/`pp_idx`/`curve_idx` index into `gold`'s own time axis,
    `hrp_col`/`pp_col`/`curve_cols` index into each solution's (already
    `phase_data`-restricted) time axis.

    Parameters
    ----------
    gold: mne.SourceEstimate
        Ground-truth source activity for this run.

    solutions: dict[str, mne.SourceEstimate]
        One reconstructed source estimate per method/condition label.

    hrp_idx, pp_idx: int
        Indices into `gold`'s time axis for the half-rising-phase / peak-
        phase timepoints.

    curve_idx: np.ndarray
        Indices into `gold`'s time axis for the spatial-correlation curve.

    hrp_col, pp_col: int
        Positions of `hrp_idx`/`pp_idx` within each solution's time axis.

    curve_cols: np.ndarray
        Positions of `curve_idx` within each solution's time axis.

    true_source_mask: np.ndarray
        Boolean mask over dipoles marking the true simulated source region
        (e.g. `Source.select_targets(...)`).

    dipole_positions: np.ndarray of shape (n_dipoles, 3)
        Dipole positions in the same space/units as `true_source_mask`, for
        `find_peak_localization_error`.

    Returns
    -------
    scores: dict[str, dict]
        One entry per key of `solutions`, each `{"localization_error_hrp",
        "ap_hrp", "ap_pp", "correlation_auc", "correlation_curve"}`.
    """
    gold_hrp = gold.data[:, hrp_idx]
    gold_pp = gold.data[:, pp_idx]
    gold_curve = gold.data[:, curve_idx]

    scores = {}
    for method, sol in solutions.items():
        sol_hrp = sol.data[:, hrp_col]
        sol_pp = sol.data[:, pp_col]
        sol_curve = sol.data[:, curve_cols]

        hit = find_peak_localization_error(sol_hrp, true_source_mask, dipole_positions)

        corr_curve = np.array([
            np.corrcoef(gold_curve[:, t], sol_curve[:, t])[0, 1] for t in range(gold_curve.shape[1])
        ])

        scores[method] = {
            "localization_error_hrp": hit["distance_mm"],
            "ap_hrp": calc_pr_curve(gold_hrp > 0, sol_hrp)["ap"],
            "ap_pp": calc_pr_curve(gold_pp > 0, sol_pp)["ap"],
            "correlation_auc": float(np.nanmean(corr_curve)),
            "correlation_curve": corr_curve,
        }

    return scores
