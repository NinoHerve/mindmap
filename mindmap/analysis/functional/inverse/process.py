"""
Module Name: inverse/process.py
Description: This module provides functions to process source estimates.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import mne

from mindmap.core.process import normalize_data

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
