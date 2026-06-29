"""
Module Name: analysis/functional/inverse/process.py
Description: Functions to build source estimate objects.

This module functions to build source estimates from numpy array.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import re
from pathlib import Path

import numpy as np

import mne
from bids.layout import parse_file_entities


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


def stream_source_solutions(stc_dir: str | Path, target_method: list[str]):
    """
    A generator that searches for saved STC files and yields them one by one.
    """
    stc_dir = Path(stc_dir)
    if isinstance(target_method, (str, Path)):
        target_method = [target_method]

    # Grab all left-hemisphere files
    stc_files = list(stc_dir.glob("*_desc-*_stc-lh.stc"))

    filtered_files = []
    for lh_file in stc_files:
        entities = parse_file_entities(lh_file)
        file_method = entities.get("desc", "").lower()

        if file_method in target_method:
            filtered_files.append((lh_file, entities))

    for lh_file, entities in filtered_files:
        # 1. Reconstruct the base filename MNE expects
        base_stc_path = str(lh_file).replace("-lh.stc", "")

        # 2. Load the actual MNE SourceEstimate object into RAM
        stc = mne.read_source_estimate(base_stc_path)

        # 3. Yield the object and its metadata
        yield {
            "meta": entities,
            "stc": stc,
        }


def filter_remaining_files(file_list: list[Path], methods: list[str]) -> list[Path]:
    """Filters out files that have already been completely processed."""
    remaining_files = []

    for fname in file_list:
        all_methods_exist = True

        # check if EVERY requested method output file exists on disk
        for m in methods:
            # Replicate your exact string replacement logic
            foutput = str(fname).replace("ave", "stc")
            foutput = foutput.replace("_stc", f"_desc-{m.lower()}_stc")
            foutput = foutput.replace(".fif", "")

            # MNE saves source estimates with -lh.stc and -rh.stc extensions
            lh_file = Path(foutput + "-lh.stc")
            rh_file = Path(foutput + "-rh.stc")

            if not (lh_file.exists() or rh_file.exists()):
                all_methods_exist = False
                break

        if not all_methods_exist:
            remaining_files.append(fname)
    
    return remaining_files