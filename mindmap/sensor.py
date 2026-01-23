"""
Module Name: sensor.py
Description: Functions to manage sensor data.

This module provides functions to manage mne.Evoked / mne.Epochs / mne.Raw that are not 
implemented in MNE-python.

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


def get_rising_phase(
    eeg    : mne.Evoked,
    ratio  : float = 0.5,
    tmin   : float = None,
    tmax   : float = None,
    as_idx : bool  = False,
) -> float:
    """
    Finds rising phase of spike.

    Parameters
    ----------
    eeg: mne.Evoked 
        EEG. 
    
    ratio: float in [0-1]
        Ratio of the rising phase. A ratio of 1 corresponds to the peak.

    tmin: float
        Minimum point in time to consider for peak (in [s])

    tmax: float
        Maximum point in time to consider for peak (in [s])

    as_idx: bool
        Whether to return the time index.

    Returns
    -------
    out: float
        Time of peak or idx of peak if `as_idx` is True. 
    """
    data = eeg.copy() 

    # get peak
    channel, peak_time, peak_value = data.get_peak(tmin=tmin, tmax=tmax, return_amplitude=True)

    # set threshold
    threshold = ratio * np.abs(peak_value)

    # slice signal up to peak
    data.crop(tmin, peak_time)
    pre_peak = np.abs(data.get_data(channel).squeeze())

    # find last index before peak where it crossed threshold
    above_thresh_indices = np.where(pre_peak >= threshold)[0]
    rising_phase_idx     = above_thresh_indices[0] if above_thresh_indices.size > 0 else None 

    # find time from idx
    out = data.times[rising_phase_idx]

    if as_idx:
        out = eeg.time_as_index(out)

    return out


def build_eeg_template(
        duration  : float = 1,
        frequency : int = 250,
        montage   : str = "spike",
        ) -> mne.Evoked:
    """
    Build an empty EEG template.

    Parameters
    ----------
    duration: float
        Duration of the recording in [s].

    frequency: int
        Sampling frequency in [Hz].

    montage: str
        Dataset name. Copy montage of that dataset.
        - spike: Montage ressembling SPIKE dataset.

    Returns
    -------
    eeg: mne.Evoked
        Empty EEG recording.
    """

    if montage == "spike":
        # load builtin-montage
        montage = mne.channels.make_standard_montage("EGI_256")

        # create info
        info = mne.create_info(ch_names=montage.ch_names, ch_types="eeg", sfreq=frequency)
        info.set_montage(montage)

        info["description"] = "Template EEG"
        info["bads"] = ["E31", "E67", "E73", "E82", "E91", "E92", "E102", "E111", "E120", "E133", "E145", "E165", "E174", "E187", "E199", "E208", "E209", "E216", "E217", "E218", "E219"] + [f"E{i}" for i in range(225,257)]

    else:
        raise ValueError("Unknown montage for building EEG template.")

    # create empty data
    eeg = np.zeros((info["nchan"], int(duration*info["sfreq"])))

    # create eeg
    eeg = mne.EvokedArray(eeg, info, verbose="warning")
    eeg.drop_channels(info["bads"])

    return eeg


def scale_montage(montage, scale=1e-3):
    """
    Scale montage. Usually from [mm] -> [m] (scale=1e-3).
    """
    mm_to_m_trans = np.eye(4)
    mm_to_m_trans[:3, :3] *= scale   # scale the X, Y, Z axes

    mm_to_m_trans = mne.transforms.Transform(fro="head", to="head", trans=mm_to_m_trans)

    # apply scaling
    montage.apply_trans(mm_to_m_trans)

    return montage