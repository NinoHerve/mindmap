"""
Module Name: analysis/functional/eeg/processing.py
Description: Functions to process sensor data.

This module provides functions to process sensor data in ways MNE did not implement.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import mne

from mindmap.anatomy.source_space.io import get_dipole_positions, get_dipole_directions
from mindmap.core.metric import compute_distances
from mindmap.core.process import normalize_data, align_timeseries, normalize_timeserie_by_baseline
from mindmap.analysis.functional.common.process import binarize_mne_object
from .io import get_ch_positions


def compute_virtual_ieeg(
    source_space    : mne.SourceSpaces,
    source_estimate : mne.SourceEstimate,
    ieeg            : mne.Evoked,
    conductivity    : float = 0.25,   # S.m⁻¹
    threshold       : float = 0.003,  # m
) -> mne.Evoked:
    """
    Compute the virtual ieeg potentials from source estimates.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space (contains source locations)

    source_estimate: mne.SourceEstimates
        Source estimates.

    ieeg: mne.Evoked
        Intracranial EEGs.

    conductivity: float (default 0.25)
        Electrical conductivity [S.m⁻¹].

    threshold: float (default 0.003)
        Minimum displacement [m] bewteen dipole and sensor. 
        If the displacement is smaller than threshold, take
        threshold as displacement.

    Returns:
    -------
    vieeg: mne.Evoked
        Virtual intracranial EEGs.
    """

    # ------------------
    # Sources (dipoles)
    # ------------------
    # source positions
    n_pos = get_dipole_positions(source_space)  # (N,3)

    # source direction
    n_dir = get_dipole_directions(source_space)  # (N,3)

    # source strengths
    n_norm = source_estimate.data  # (N,T)

    # source vectors
    n_vec = n_dir[:,np.newaxis,:] * n_norm[:,:,np.newaxis]  # (N,T,3)

    # ------------------------
    # Targets (iEEG channels)
    # ------------------------
    # target positions
    u_pos = get_ch_positions(ieeg)  # (M,3)

    # --------------------------
    # Distances (displacements)
    # --------------------------
    # displacements
    r_vec  = compute_distances(n_pos, u_pos, output_type="vector")  # (N,M,3)
    r_norm = np.linalg.norm(r_vec, axis=2)                          # (N,M)
    r_dir  = r_vec / np.maximum(r_norm, 1e-12)[:,:,np.newaxis]      # (N,M,3)

    # handle minimum distance
    r_norm = np.maximum(r_norm, threshold)

    # -----------------------------------------------------
    # Virtual iEEG potentials
    # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
    # Numerator: We want a dot product over the spatial
    # dimension for every N, M, and T.
    #
    # Denominator: 4*pi*sigma*r²
    #
    # Virtual iEEGs: Sum influence of each dipole.
    # -----------------------------------------------------

    denominator = 4 * np.pi * conductivity * (r_norm**2)    # (N,M)
    vieeg_data  = np.einsum("ntd,nmd,nm->mt", n_vec, r_dir, 1.0/denominator)  # (M,T)

    # store in MNE object
    vieeg = ieeg.copy()
    vieeg.data = vieeg_data

    return vieeg


def normalize_evoked(
    ave: mne.Evoked,
    strategy: str = "global",
    method: str = "zscore"
) -> mne.Evoked:
    """
    Normalizes EEG data using various strategies and methods.

    Parameters
    ----------
    ave: mne.Evoked
        The input evoked object.

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
    ave_norm : mne.Evoked
        A copy of the input with normalized data.
    """
    # 1. work on a copy to preserve original physical units (Volts/Amps)
    ave_norm = ave.copy()

    # 2. Apply normalization
    ave_norm.data = normalize_data(ave_norm.data, strategy, method)

    # 3. Update the metadata so the user knows this is normalized
    ave_norm.comment = f"{ave.comment or 'Evoked'} | {strategy}-{method}"

    return ave_norm



def align_eeg_timeseries(
    ref: mne.Evoked,
    target: mne.Evoked,
    max_lag: float = 20,
    window: tuple[float,float] = None,
) -> mne.Evoked:
    """
    Finds a global shift for 'target' that maximizes spatial correlation 
    based on a specific window.

    Parameters:
    ----------
    ref: mne.Evoked
        Reference EEG.
    
    target: mne.Evoked
        Target EEG.

    max_lag: float
        Maximum lag shift in [ms].

    window: (float, float)
        (start, end) tuple of timepoints. If None, uses the whole series.
    """
    # recording parameters
    if not np.allclose(ref.times, target.times):
        raise ValueError("Time acquisitions do not match.")

    # convert timepoints to indices
    max_lag_idx = int((max_lag / 1000.0) * ref.info["sfreq"])

    if window is not None:
        # TODO: truncates (MNE default); use_rounding=True would fix edges landing one sample early, but shifts SNR/lambda and the Study III windows, so rerun first.
        start_idx, end_idx = ref.time_as_index(window)
        window_idx = (start_idx, end_idx)
    else:
        window_idx = None

    # align timeseries
    aligned_data, best_lag = align_timeseries(
        ref.data, target.data, max_lag=max_lag_idx, window=window_idx
    )

    aligned_evoked = target.copy()
    aligned_evoked.data = aligned_data
    best_lag = best_lag/ref.info["sfreq"] * 1000

    return aligned_evoked, best_lag


def normalize_evoked_by_baseline(
    evoked: mne.Evoked,
    baseline: tuple[float, float],
 ) -> mne.Evoked:
    """
    Normalizes evoked data by the standard deviation of the baseline.

    Parameters
    ----------
    evoked: mne.Evoked
        Evoked object.

    baseline: tuple[float, float]
        (tmin, tmax) defining the basline period in seconds.
    """
    # create copy to avoid modifying original object in-place
    norm_evoked = evoked.copy()

    # select the baseline time indices
    # TODO: truncates (MNE default); use_rounding=True would fix edges landing one sample early, but shifts SNR/lambda and the Study III windows, so rerun first.
    idx_min, idx_max = evoked.time_as_index(baseline)

    # apply normalization
    norm_evoked.data = normalize_timeserie_by_baseline(evoked.data, (idx_min, idx_max))

    return norm_evoked

def binarize_evoked(
    evoked: mne.Evoked,
    threshold: float = 0.5,
    method: str = "max",
    absolute: bool = True,
    baseline: tuple = (None, None),
) -> mne.Evoked:
    """
    Binarize Evoked.

    Parameters:
    ----------
    evoked: mne.Evoked
        Evoked.
    
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
    evoked_bin: mne.Evoked
        Binarized evoked.
    """
    return binarize_mne_object(
        obj = evoked,
        threshold = threshold,
        method = method,
        absolute = absolute,
        baseline = baseline,
    )
