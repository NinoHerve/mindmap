"""
Module Name: analysis.functional.common.process.py
Description: Common processes for other sub-modules in this module.

This module provides functions to process data.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import mne

from mindmap.core.process import binarize_signal


def binarize_mne_object(
    obj: mne.Evoked | mne.SourceEstimate,
    threshold: float = 0.5,
    method: str = "max",
    absolute: bool = True,
    baseline: tuple = (None, None),
) -> mne.Evoked | mne.SourceEstimate:
    """
    Binarize MNE Evoked or SourceEstimate object.

    Parameters:
    obj: mne.Evoked | mne.SourceEstimate
        MNE object.
    
    threshold: float
        Threshold for binarization.

    method: str [none, 'max', 'std']
        Method for thresholding:
        - none: Considers threshold raw.
        - max: Considers threshold as a percentage [0-1] of maximum value.
        - std: Considers threshold as a multiple of standard deviation (requires noise baseline).

    absolute: bool
        If True, apply thresholding on absolute value of signal.

    Returns:
    -------
    obj_bin: mne.Evoked | mne.SourceEstimate
        Binarized object.
    """
    times = obj.times
    t_min = baseline[0] or times[0]
    t_max = baseline[1] or times[-1]

    # TODO: truncates (MNE default); use_rounding=True would fix edges landing one sample early, but shifts SNR/lambda and the Study III windows, so rerun first.
    start, stop = obj.time_as_index((t_min, t_max))

    # binarize data
    data_bin = binarize_signal(
        signal = obj.data,
        threshold = threshold,
        method = method,
        absolute = absolute,
        baseline = (start, stop)
    )

    # create binarized copy and crop
    obj_bin = obj.copy()
    obj_bin.data = data_bin.astype(int)

    return obj_bin
