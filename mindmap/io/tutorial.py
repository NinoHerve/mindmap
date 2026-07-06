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
import scipy as sp

import mne
import nibabel as nib

from mindmap.analysis.functional.eeg.utils import scale_montage
from mindmap.anatomy.parcellation.base import Parcellation
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
        self.add_root("spk", self.config["DIR_SPK"])
        self.add_root("cartool", self.config["DIR_CART"])
        self.add_root("cmp", self.config["DIR_CMP"])
        self.add_root("sim", self.config["DIR_SIM"])

        # add new datatype
        self.add_datatype("ave-temp", "mm", self._build_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("eeg", None, self._build_path, self._load_eeg)
        self.add_datatype("ieeg", None, self._build_path, self._load_eeg)
        self.add_datatype("eeg-trans", "mm", self._build_path, load_identity_trans)
        self.add_datatype("gold", "sim", self._build_path, mne.read_source_estimate)
        self.add_datatype("tractogram", "cmp", self._build_path, nib.streamlines.load)
        self.add_datatype("t12diff", "cmp", self._build_path, self._load_t12diff)

    def load_ref(self, ref_key):
        # Make thalamic atlas exception.
        if ref_key.lower() == "ref_thal":
            path = self.refs[ref_key]
            clut_path = path.parent/"Thalamic_Nuclei-ColorLUT.txt"
            return Parcellation(path, clut_path)

        return super().load_ref(ref_key)

    # ----------------------------
    # TutoManager specific logic
    # ----------------------------

    def _build_path(self, root, subject, suffix, **kwargs):
        """Build paths."""
        if suffix == "ave-temp":
            return root/f"sub-{subject}"/"ave"/f"sub-{subject}_desc-template-coreg_ave.fif"

        if suffix == "eeg":
            if "intra" in subject.lower():
                # manage bids entities
                task = kwargs.get("task")
                task = f"_task-{task}" if task else ""

                run = kwargs.get("run")
                run = f"_run-{run}" if run else "_run-*"

                eeg_dir = self.roots["spk"]/f"sub-{subject}"/"eeg"
                return eeg_dir/f"sub-{subject}{task}{run}_eeg.vhdr"

            if "simf" in subject.lower():
                pass

            if "spike" in subject.lower():
                folder = self.roots["cartool"]/f"sub-{subject}"/"eeg"/"ses-preop"
                return folder/f"sub-{subject}_ses-preop_task-rest_desc-spikeerp-204c-interpol-250Hz_eeg.vhdr"

        if suffix == "ieeg":
            if not "intra" in subject:
                raise ValueError(f"Subject sub-{subject} doesn't have any iEEGs.")

            # manage bids entities
            task = kwargs.get("task")
            task = f"_task-{task}" if task else ""

            run = kwargs.get("run")
            run = f"_run-{run}" if run else "_run-*"

            ieeg_dir = self.roots/f"sub-{subject}"/"ieeg"
            return ieeg_dir/f"sub-{subject}{task}{run}_ieeg.vhdr"

        if suffix == "eeg-trans":
            pass

        if suffix == "gold":
            if "simf" in subject:
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
                raise ValueError(f"No 'gold' standard for sub-{subject}.")
            
        if suffix == "eeg-trans":
            return Path(".")
        
        if suffix == "tractogram":
            if "simf" in subject:
                return root/f"sub-{subject}"/"dwi"/f"sub-{subject}_model-CSD_desc-DET_tractogram.trk"
            else:
                raise ValueError(f"No tractogram for sub-{subject}")
            
        if suffix == "t12diff":
            if "simf" in subject:
                return root/f"sub-{subject}"/"xfm"/"final0GenericAffine.mat"
            else:
                raise ValueError(f"No tractogram for sub-{subject}")


    def _load_eeg(self, path):
        """EEG loader."""
        # Figure out which dataset it is.
        dataset = ...

        if dataset == "simf":
            ...

        if dataset == "spike":
            eeg = mne.io.read_raw_brainvision(path, scale=1e-6, misc=[], verbose="warning")

            data = eeg.get_data()
            info = eeg.info

            return mne.EvokedArray(data, info)

        if dataset == "intra":
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


        raise ValueError(f"Unknown dataset {dataset}.")

    def _load_gold(self, path):
        """Gold standard loader."""

    def _load_t12diff(self, path):
        """
        Load t1 to diffusion space transform.

        ANTs .mat affines store a linear part `A`, a `translation`, and a
        `fixed` center of rotation the linear part pivots around -- the
        on-disk transform is `T(x) = A @ (x - center) + center + translation`,
        not `A @ x + translation`. Folding the pivot in here once gives an
        ordinary 4x4 matrix, so every downstream use of this affine via
        `nib.affines.apply_affine` (align_diffusion2subject, etc.) doesn't
        need to know a pivot was ever involved.
        """
        mat = sp.io.loadmat(str(path))
        raw = mat["AffineTransform_double_3_3"].reshape(3, 4, order="F")
        linear, translation = raw[:, :3], raw[:, 3]
        center = mat["fixed"].flatten()

        effective_translation = translation + center - linear @ center

        affine = np.eye(4)
        affine[:3, :3] = linear
        affine[:3, 3] = effective_translation

        return affine


def load_identity_trans(_):
    """
    Load identity transformation matrix.
    """
    return mne.transforms.Transform(fro="head", to="mri")

