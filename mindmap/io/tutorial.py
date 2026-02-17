"""
Module Name: io/tutorial.py
Description: This module provides a configured DataManager for the tutorial dataset.

DataManager for tutorial dataset.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""


import glob
from pathlib import Path

import numpy as np

import mne

from mindmap.analysis.functional.eeg.utils import scale_montage
from mindmap.anatomy.parcellation.core import Parcellation
from .dataset import DatasetManager


class TutoManager(DatasetManager):
    """
    Manages the Tutorial dataset.
    """
    def __init__(self, config):
        """
        Configured configuration.
        """
        super().__init__(config)

        # add new roots
        self.roots["spk"] = self.config["DIR_SPK"]

    def get_subject(self, subject_id):
        """
        Rewrite missing data for subject reader.
        """
        # default subject reader
        subreader = super().get_subject(subject_id)

        # configure for vep data
        if "vep" in subject_id:
            pass

        # configure for intra data
        if "intra" in subject_id:
            subreader.add_datatype("eeg", "raw", build_intra_path, load_eeg)
            subreader.add_datatype("ieeg", "raw", build_intra_path, load_eeg)
            subreader.add_datatype("eeg-trans", "mm", lambda *_, : Path("."), load_identity_trans)

        # configure for simf data
        if "simf" in subject_id:
            pass

        # configure for spike data
        if "spike" in subject_id:
            pass

        return subreader

    def get_standards(self):
        """
        Rewrite complex objects for standardized data
        """
        # default standardized data reader
        stdreader = super().get_standards()

        # add exceptions
        stdreader.add_exceptions("THAL", load_thal_atlas)


########################
##   intra builders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨   ##
########################

def build_intra_path(root, subject_id, datatype, **kwargs):
    """
    EEG & iEEG path for intra dataset.
    """
    if datatype in ["eeg", "ieeg"]:
        
        # manage bids entities
        task = kwargs.get("task")
        task = f"_task-{task}_" if task else ""

        run = kwargs.get("run")
        run = f"_run-{run}_" if run else "_run-*_"

        # filename
        if datatype == "eeg":
            eeg_dir = root/f"sub-{subject_id}"/"eeg"
            return eeg_dir/f"sub-{subject_id}{task}{run}eeg.vhdr"
        else:
            ieeg_dir = root/f"sub-{subject_id}"/"ieeg"
            return ieeg_dir/f"sub-{subject_id}_task-spk_*_ieeg.vhdr"

    else:
        raise ValueError(f"Unknown datatype '{datatype}' for build_intra_path.")    

#######################
##   intra loaders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨¨   ##
#######################

def load_eeg(path):
    """
    Load EEGs for intra dataset.
    """
    files = glob.glob(str(path))
    montage = glob.glob(str(path.parent/"*_electrodes.tsv"))[0]

    dat = []

    for f in files:
        spike = mne.io.read_raw_brainvision(f, verbose="warning")
        dat.append(spike["data"][0])
        dat = np.array(dat)

        # info & montage
        info = spike.info
        montage = mne.channels.read_custom_montage(montage, coord_frame="head")
        montage = scale_montage(montage, scale=1e-3)

        # store in epoch object
        epochs = mne.EpochsArray(dat, info, verbose="warning")
        epochs = epochs.set_montage(montage, verbose="warning")

    return epochs

def load_identity_trans(_):
    """
    Load identity transformation matrix.
    """
    return mne.transforms.Transform(fro="head", to="mri")

######################
##   std loaders    ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨   ##
######################

def load_thal_atlas(path):
    """
    Thalamic atals loader.
    """
    clut_path = str(path).replace("-HCP-4DSPAMs.nii.gz", "ColorLUT.txt")
    return Parcellation(path, clut_path)