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
import mne_bids


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
        self.roots["cartool"] = self.config["DIR_CART"]

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
            subreader.add_datatype("eeg", "spk", build_intra_path, load_eeg_intra)
            subreader.add_datatype("ieeg", "spk", build_intra_path, load_eeg_intra)
            subreader.add_datatype("eeg-trans", "mm", lambda *_, : Path("."), load_identity_trans)

        # configure for simf data
        if "simf" in subject_id:
            subreader.add_datatype("eeg", "raw", mne_bids.BIDSPath, lambda f: mne_bids.read_raw_bids(f, verbose="warning").load_data(verbose="warning"))
            subreader.add_datatype("gt", "raw", build_simf_path, mne.read_source_estimate)

        # configure for spike data
        if "spike" in subject_id:
            subreader.add_datatype("eeg", "cartool", build_spike_path, load_eeg_spike)

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

def build_intra_path(root, subject, datatype, **kwargs):
    """
    EEG & iEEG path for intra dataset.
    """
    if datatype in ["eeg", "ieeg"]:
        
        # manage bids entities
        task = kwargs.get("task")
        task = f"_task-{task}" if task else ""

        run = kwargs.get("run")
        run = f"_run-{run}" if run else "_run-*"

        # filename
        if datatype == "eeg":
            eeg_dir = root/f"sub-{subject}"/"eeg"
            return eeg_dir/f"sub-{subject}{task}{run}_eeg.vhdr"
        else:
            ieeg_dir = root/f"sub-{subject}"/"ieeg"
            return ieeg_dir/f"sub-{subject}{task}_*_ieeg.vhdr"

    else:
        raise ValueError(f"Unknown datatype '{datatype}' for build_intra_path.")
    

########################
##   SPIKE builders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨   ##
########################

def build_spike_path(root, subject, datatype):
    """EEG & iEEG path for spike dataset"""
    if datatype == "eeg":
        return root/f"sub-{subject}"/"eeg"/"ses-preop"/f"sub-{subject}_ses-preop_task-rest_desc-spikeerp-204c-interpol-250Hz_eeg.vhdr"

    elif datatype == "resec":
        return NotImplementedError

    else:
        raise ValueError(f"Unknown datatype '{datatype}' for build_pike_path.")


#######################
##   SIMF builders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨¨   ##
#######################


def build_simf_path(root, subject, datatype, **kwargs):
    """
    Ground truth path for simf dataset.
    """
    if datatype in ["gt"]:

        # manage bids entities
        task = kwargs.get("task")
        task = f"_task-{task}" if task else ""

        run = kwargs.get("run")
        run = f"_run-{run}" if run else ""

        desc = kwargs.get("desc")
        desc = f"_desc-{desc}" if desc else ""

        gt_dir = root/f"sub-{subject}"/"stc"
        return gt_dir/f"sub-{subject}{task}{run}{desc}_stc"

    else:
        raise ValueError(f"Unknown datatype '{datatype}' for build_simf_path.")

#######################
##   intra loaders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨¨   ##
#######################

def load_eeg_intra(path):
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
##   SIMF loaders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨   ##
######################


#######################
##   SPIKE loaders   ##
##   ¨¨¨¨¨¨¨¨¨¨¨¨¨   ##
#######################

def load_eeg_spike(path):
    """Load EEGs for spike dataset"""
    eeg = mne.io.read_raw_brainvision(path, scale=1e-6, misc=[], verbose="warning")

    data = eeg.get_data()
    info = eeg.info

    return mne.EvokedArray(data, info)


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