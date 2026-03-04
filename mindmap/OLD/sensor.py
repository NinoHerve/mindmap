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

from mindmap.metric import compute_distances
from mindmap.head.source_space import get_dipole_positions, get_dipole_directions

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
    numerator = np.einsum("ntd,nmd->nmt", n_vec, r_dir)     # (N,M,T)

    denominator = 4 * np.pi * conductivity * (r_norm**2)    # (N,M)

    potentials = numerator / denominator[:, :, np.newaxis]  # (N,M,T)

    vieeg_data = np.sum(potentials, axis=0)  # (M,T)

    # store in MNE object
    vieeg = ieeg.copy()
    vieeg.data = vieeg_data

    return vieeg


