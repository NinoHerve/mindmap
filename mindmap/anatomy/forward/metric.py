"""
Module Name: anatomy.forward.metric.py
Description: Functions to measure metrics.

This module provides functions to measure various metrics. Usefull for evaluation, validation,
or simply testing.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import mne

from mindmap.core.metric import calc_mag, calc_rdm

def compute_fwd_rdm(fwd1, fwd2):
    """Computes RDM between two MNE forward solutions."""
    # force cortical/fixed constraint
    fwd1_fixed = mne.convert_forward_solution(fwd1, surf_ori=True, force_fixed=True, use_cps=True, verbose="warning")
    fwd2_fixed = mne.convert_forward_solution(fwd2, surf_ori=True, force_fixed=True, use_cps=True, verbose="warning")

    # extract raw gain matrices
    gain1 = fwd1_fixed["sol"]["data"]
    gain2 = fwd2_fixed["sol"]["data"]

    # calculate RDM
    return calc_rdm(gain1, gain2)


def compute_fwd_mag(fwd1, fwd2):
    """Computes MAG between two MNE forward solutions."""
    # force cortical/fixed constraint
    fwd1_fixed = mne.convert_forward_solution(fwd1, surf_ori=True, force_fixed=True, use_cps=True, verbose="warning")
    fwd2_fixed = mne.convert_forward_solution(fwd2, surf_ori=True, force_fixed=True, use_cps=True, verbose="warning")

    # extract raw gain matrices
    gain1 = fwd1_fixed["sol"]["data"]
    gain2 = fwd2_fixed["sol"]["data"]

    # calculate MAG
    return calc_mag(gain1, gain2)