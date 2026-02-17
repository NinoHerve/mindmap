"""
Module Name: sensor/utils.py
Description: Usefull functions for manipulating sensor objects.

This module provides functions manipulate sensor objects in ways MNE did not think of.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import mne

from .io import get_ch_positions


def scale_montage(montage:mne.channels.DigMontage, scale:float=1e-3) -> mne.channels.DigMontage:
    """
    Scale montage. Usually from [mm] -> [m] (scale=1e-3).
    """
    mm_to_m_trans = np.eye(4)
    mm_to_m_trans[:3, :3] *= scale   # scale the X, Y, Z axes

    mm_to_m_trans = mne.transforms.Transform(fro="head", to="head", trans=mm_to_m_trans)

    # apply scaling
    montage.apply_trans(mm_to_m_trans)

    return montage


def same_evoked(*args: mne.Evoked):
    """
    Checks if Evokeds have the same channels and timepoints.

    Parameters
    ----------
    args: mne.Evoked
        Any number of evokeds.
    """
    # reference (args[0])
    ch_ref   = args[0].ch_names
    pos_ref  = get_ch_positions(args[0])
    time_ref = args[0].times

    # iterate through evokeds
    for a in args:
        if not np.all(ch_ref == a.ch_names):
            return False
        if not np.all(pos_ref == get_ch_positions(a)):
            return False
        if not np.all(time_ref == a.times):
            return False
        
    return True