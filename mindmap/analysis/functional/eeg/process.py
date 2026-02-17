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
from mindmap.core.process import normalize_data
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
