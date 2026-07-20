"""
Module Name: inverse/process.py
Description: This module provides functions to process source estimates.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np
import mne

from mindmap.core.process import normalize_data
from mindmap.analysis.functional.common.process import (
    binarize_mne_object,
)
from mindmap.analysis.functional.eeg.metric import get_rising_phase

def normalize_source_estimate(
    stc: mne.SourceEstimate,
    strategy: str = "global",
    method: str = "zscore"
) -> mne.Evoked:
    """
    Normalizes Source Estimate data using various strategies and methods.

    Parameters
    ----------
    stc: mne.SourceEstimate
        The input source estimates.

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
    stc_norm : mne.SourceEstimate
        A copy of the input with normalized data.
    """
    # 1. work on a copy to preserve original physical units (Volts/Amps)
    stc_norm = stc.copy()

    # 2. Apply normalization
    stc_norm.data = normalize_data(stc_norm.data, strategy, method)

    # 3. Update the metadata so the user knows this is normalized
    old_comment = stc.comment if hasattr(stc, "comment") else "SourceEstimate"
    stc_norm.comment = f"{old_comment} | {strategy}-{method}"

    return stc_norm


def binarize_stc(
    stc: mne.SourceEstimate,
    threshold: float = 0.5,
    method: str = "max",
    absolute: bool = True,
    baseline: tuple = (None, None),
) -> mne.SourceEstimate:
    """
    Binarize source time currents.

    Parameters:
    stc: mne.SourceEstimate
        Source time currents.
    
    threshold: float
        Threshold for binarization.

    method: str [none, 'max', 'std']
        Method for thresholding:
        - none: Considers threshold raw.
        - max: Considers threshold as a percentage [0-1] of maximum value.
        - std: Considers threshold as a multiple of standard deviation (requires noise baseline).

    absolute: bool
        If True, apply thresholding on absolute value of signal.

    baseline: tuple (float, float)
        Range in [ms] for the noise baseline.

    Returns:
    -------
    stc_bin: mne.SourceEstimate
        Binarized source time currents.
    """
    return binarize_mne_object(
        obj = stc,
        threshold = threshold,
        method = method,
        absolute = absolute,
        baseline = baseline,
    )


def build_phase_data_hrp_pp_curve(
    ave: mne.Evoked,
    gold: mne.SourceEstimate,
    simulator,
    n_curve_points: int = 50,
) -> dict:
    """
    Builds a `phase_data` Evoked restricted to the union of {half-rising-
    phase index, peak-phase index, `n_curve_points` evenly-spaced indices
    across `gold`'s own active-signal duration} -- the timepoint set the
    pilot-sizing outcome measures need (hrp/pp localization error and AP,
    plus a spatial-correlation-to-gold curve), so every solver only needs
    one `.solve()` call per run instead of solving per-timepoint.

    Sampling uniformly across the whole epoch instead of restricting to
    where `gold` is actually active would waste most of the curve points on
    the flat pre/post-spike period where `gold` is exactly zero (undefined
    spatial correlation there).

    Parameters
    ----------
    ave: mne.Evoked
        Sensor data for this run.

    gold: mne.SourceEstimate
        Ground-truth source activity for this run.

    simulator: mindmap.analysis.simulation.Simulator
        The run's simulator (only `simulator.hm.fwd` is used, to project
        `gold` through the forward model and locate its rising/peak phase in
        sensor space). Not type-hinted to avoid a circular import (
        `mindmap.analysis.simulation` already depends on this module).

    n_curve_points: int
        Number of evenly-spaced indices to sample across `gold`'s active
        window for the spatial-correlation curve.

    Returns
    -------
    result: dict
        `phase_data` (mne.EvokedArray on the combined index set), `hrp_idx`/
        `pp_idx`/`curve_idx` (indices into `ave.times`), and `hrp_col`/
        `pp_col`/`curve_cols` (their positions within `phase_data`).
    """
    clean_ave = mne.apply_forward(simulator.hm.fwd, gold, ave.info)
    hrp_idx = get_rising_phase(clean_ave, ratio=0.5, as_idx=True)[0]
    pp_idx = get_rising_phase(clean_ave, ratio=1, as_idx=True)[0]

    gold_envelope = np.linalg.norm(gold.data, axis=0)
    active = np.where(gold_envelope > 1e-3 * gold_envelope.max())[0]
    pad = max(1, int(0.1 * (active[-1] - active[0])))
    i0 = max(0, active[0] - pad)
    i1 = min(len(ave.times) - 1, active[-1] + pad)
    curve_idx = np.unique(np.linspace(i0, i1, n_curve_points).astype(int))

    all_idx = np.unique(np.concatenate([[hrp_idx, pp_idx], curve_idx]))
    hrp_col = int(np.searchsorted(all_idx, hrp_idx))
    pp_col = int(np.searchsorted(all_idx, pp_idx))
    curve_cols = np.searchsorted(all_idx, curve_idx)

    phase_data = mne.EvokedArray(ave.data[:, all_idx], ave.info, tmin=0, nave=ave.nave, kind=ave.kind)

    return {
        "phase_data": phase_data, "hrp_idx": hrp_idx, "pp_idx": pp_idx,
        "hrp_col": hrp_col, "pp_col": pp_col, "curve_cols": curve_cols,
        "curve_idx": curve_idx,
    }
