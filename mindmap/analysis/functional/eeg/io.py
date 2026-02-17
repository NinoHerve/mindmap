"""
Module Name: sensor/io.py
Description: Functions to extract sensor data.

This module provides functions to extract hard to get data from mne.Evoked / mne.Epochs / mne.Raw.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import mne


def get_ch_positions(eeg: mne.io.Raw | mne.Epochs | mne.Evoked) -> np.ndarray:
    """
    Get channel position from mne object.

    Parameters
    ----------
    eeg: mne.io.Raw | mne.Epochs | mne.Evoked
        EEG object.

    Returns
    -------
    pos: np.ndarray
        (n,3) array indicating the (x,y,z) location of n electrodes.
    """

    digs = eeg.info["dig"]
    positions = []

    for d in digs:
        if "eeg" in str(d["kind"]).lower():
            positions.append(d["r"])

    return np.array(positions)


def get_fiducials(obj: mne.io.Raw | mne.Epochs | mne.Evoked) -> np.ndarray:
    """
    Get fiducial position from MNE object. 

    Parameters
    ----------
    obj: mne.Raw | mne.Evoked | mne.Epochs
        MNE object. 

    Returns
    -------
    fid: np.ndarray (3,3)
        Fiducial position in [m]
    """
    montage = obj.info["dig"]

    fiducials = []
    for dig in montage:
        if dig["kind"]._name == "FIFFV_POINT_CARDINAL":
            fiducials.append(dig["r"])

    return fiducials