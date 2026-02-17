"""
Module Name: sensor/simulation.py
Description: Functions to simulate sensor objects.

This module provides functions to simulate sensor objects.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import mne

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
