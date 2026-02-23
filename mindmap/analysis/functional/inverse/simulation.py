"""
Module Name: analysis/functional/inverse/process.py
Description: Functions to build source estimate objects.

This module functions to build source estimates from numpy array.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""


import numpy as np

import mne


def make_stc(
    source_space: mne.SourceSpaces,
    activity: np.ndarray,
    tmin: float,
    tstep: float,
) -> mne.SourceEstimate:
    """
    Make mne.SourceEstimate object

    Parameter
    ---------
    activity: np.ndarray (N,T)
        Brain activity. N -> number of sources, T -> number of time points

    src: mne.SourceSpaces
        Source space.

    tmin: float
        Starting time in [s]

    tstep: float
        Time step in [s]

    Return
    ------
    stc: mne.SourceEstimate
        Brain activity in MNE format.
    """
    vertices   = [source_space[0]["vertno"], source_space[1]["vertno"]]

    stc = mne.SourceEstimate(
        activity,
        vertices,
        tmin=tmin,
        tstep=tstep,
    )

    return stc