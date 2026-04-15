"""
Module Name: io/dataset.py
Description: This module provides a DataManager class allowing you to read data by keyword.

Contains functions to read data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import nibabel as nib
import mne

from mindmap.core.io import read_config, read_eig, get_config
from mindmap.anatomy.utils import read_cortex_label
from mindmap.anatomy.mri.io import read_mri
from mindmap.anatomy.parcellation.core import Parcellation
from mindmap.anatomy.forward.io import read_bem_labels, make_volume_info
from .utils import build_raw_path, build_mm_path, build_fs_path, build_charm_path

class DatasetManager:
    """
    Managers the dataset context and spawns SubjectReaders.
    """
    def __init__(self, config : str | Path | dict):
        # store configuration
        if isinstance(config, (str, Path)):
            config = read_config(config)
        self.config = config

        self.roots = {
            'raw': config["DIR_RAW"],
            'mm' : config["DIR_MM"],
            'fs' : config["DIR_FS"],
            "charm": config["DIR_CHARM"],
        }

        self.stds = {k : v for k, v in config.items() if "STD" in k}

        self.subjects = self.config.get("SUBS", [])

    def get_subject(self, subject_id):
        """
        Returns a reader for a specific subject.
        """
        if subject_id not in self.subjects:
            raise KeyError(f"Subject {subject_id} not in config list.")

        return SubjectReader(subject_id, self.roots)

    def get_standards(self):
        """
        Returns a reader for standardized data (templates, atlases, ...)
        """

        return StandardReader(self.stds)

    def get_bem_labels(self):
        """
        Parses YAML config and resolves inheritance to return
        a dictionary of {layer_name: [list_of_ids]}
        """
        config = self.config.get("BEM_CONFIG")
        if config is None:
            raise KeyError("Missing BEM configuration in configuration file.")

        config = get_config(config)

        return read_bem_labels(config)


class SubjectReader:
    """
    Reader to extract specific data from subject.
    """
    def __init__(self, subject_id, roots):
        self.sub   = str(subject_id).replace("sub-", "")
        self.roots = roots

        self.path_registry = {
            # mindmap
            "dseg": ("mm", build_mm_path),
            "src": ("mm", build_mm_path),
            "bem": ("mm", build_mm_path),
            "fwd": ("mm", build_mm_path),
            "con": ("mm", build_mm_path),
            "eig": ("mm", build_mm_path),
            "ave": ("mm", build_mm_path),
            "i-ave": ("mm", build_mm_path),
            "cov": ("mm", build_mm_path),
            "stc": ("mm", build_mm_path),
            "parc": ("mm", self._build_internal_path),
            "eeg-trans": ("mm", self._build_internal_path),
            "v-ave": ("mm", build_mm_path),

            # freesurfer
            "t1.mgz"     : ("fs", build_fs_path),
            "brain"      : ("fs", build_fs_path),
            "orig"       : ("fs", build_fs_path),
            "white"      : ("fs", build_fs_path),
            "pial"       : ("fs", build_fs_path),
            "inflated"   : ("fs", build_fs_path),
            "vertex-ctx" : ("fs", build_fs_path),
            "vol-info"   : ("fs", self._build_internal_path),

            # charm
            "seg" : ("charm", build_charm_path),
        }

        self.loader_registry = {
            # mindmap
            "dseg"      : read_mri,
            "parc"      : self._load_parc_internal,
            "ctx"       : self._load_ctx_internal,
            "src"       : lambda f: mne.read_source_spaces(f, verbose="warning"),
            "bem"       : lambda f: mne.read_bem_solution(f, verbose="warning"),
            "fwd"       : lambda f: mne.read_forward_solution(f, verbose="warning"),
            "con"       : np.load,
            "eig"       : self._load_eig_internal,
            "ave"       : lambda f: mne.read_evokeds(f, verbose="warning")[0],
            "i-ave"     : lambda f: mne.read_evokeds(f, verbose="warning")[0],
            "cov"       : self._load_cov_internal,
            "stc"       : mne.read_source_estimate,
            "eeg-trans" : lambda f: mne.read_trans(f, verbose="warning"),
            "v-ave"     : lambda f: mne.read_evokeds(f, verbose="warning")[0],

            # freesurfer
            "t1.mgz"     : read_mri,
            "brain"      : read_mri,
            "orig"       : read_mri,
            "white"      : nib.freesurfer.io.read_geometry,
            "pial"       : nib.freesurfer.io.read_geometry,
            "inflated"   : nib.freesurfer.io.read_geometry,
            "vertex-ctx" : read_cortex_label,
            "ras2tkras"  : self._load_ras2tkras_internal,
            "vol-info"   : make_volume_info,

            # charm
            "seg": self._load_seg_internal,
        }

    # -----------------------------------
    # Internal Logic for Complex Objects
    # -----------------------------------

    def _load_parc_internal(self, path):
        """
        Load parcellation.
        """
        clut_path = Path(str(path).replace(".nii.gz", ".txt"))
        return Parcellation(path, clut_path)

    def _load_seg_internal(self, path):
        """
        Load segmentation.
        """
        clut_path = Path(str(path).replace(".nii.gz", "_LUT.txt"))
        return Parcellation(path, clut_path)

    def _load_ctx_internal(self, **kwargs):
        """
        Load cortical mask.
        """
        parc       = self.load("parc", **kwargs)
        clut       = parc.get_clut()
        ctx_labels = clut[clut["Label name"].str.startswith("ctx")]["#No."].values
        return parc.create_mask(ctx_labels)

    def _load_eig_internal(self, path):
        """
        Load eigenvalues and eigenvectors.
        """
        eigvec, eigval = read_eig(path)
        return {"vec":eigvec, "val":eigval}

    def _load_ras2tkras_internal(self, **kwargs):
        """Load ras2tkras."""
        mri = self.load("t1.mgz", **kwargs)

        vox2ras     = mri.header.get_vox2ras()
        vox2ras_tkr = mri.header.get_vox2ras_tkr()

        return vox2ras_tkr @ np.linalg.inv(vox2ras)
    
    def _load_cov_internal(self, path):
        """Load Covariance"""
        cov = mne.read_cov(path, verbose="warning")
        cov["data"] = cov["data"].astype("float")
        return cov
        

    def _build_internal_path(self, root, subject, datatype, **kwargs):
        """
        Build special paths.
        """
        if datatype == "eeg-trans":
            return build_mm_path(root, subject, "ave", suffix="eeg-trans", **kwargs)
        if datatype == "parc":
            return build_mm_path(root, subject, "dseg", **kwargs)
        if datatype == "vol-info":
            return build_fs_path(root, subject, "t1.mgz", **kwargs)
        else:
            raise ValueError(f"Datatype '{datatype}' does not have an internal path implementation.")

    # -----------
    # Public API
    # -----------

    def add_datatype(self, datatype, root_key, builder_func, loader_func):
        """
        Add new non-BIDS or derivative objects for reading.
        """
        self.path_registry[datatype] = (root_key, builder_func)
        self.loader_registry[datatype] = loader_func

    def get_path(self, datatype, **kwargs):
        """
        Dspatches to the correct builder based on datatype.
        """
        if datatype in self.path_registry:
            root_key, builder = self.path_registry[datatype]
            return builder(root=self.roots[root_key], subject=self.sub, datatype=datatype, **kwargs)

        # Default to raw
        return build_raw_path(self.roots["raw"], self.sub, datatype, **kwargs)

    def load(self, datatype, **kwargs):
        """
        The main entry point with logic for expressions.
        """
        if datatype not in self.loader_registry:
            raise KeyError(
                f"No loader reistered for datatype '{datatype}'."
                f"Use reader.add_datatype() to register a custom loader."
            )

        # Exception 1: The cortex mask
        if datatype == "ctx":
            return self.loader_registry["ctx"](**kwargs)
        
        # Exception 2: tkras2ras
        if datatype == "ras2tkras":
            return self.loader_registry["ras2tkras"](**kwargs)

        # Exception 2: Double hemispheres
        if datatype in ["white", "pial", "inflated", "vertex-ctx"] and "hemi" not in kwargs:
            path_lh = self.get_path(datatype, hemi="lh")
            path_rh = self.get_path(datatype, hemi="rh")

            if not path_lh.exists() or not path_rh.exists():
                raise FileNotFoundError(f"Missing one or both hemisphere files for {datatype} at: {path_lh} or {path_rh}")

            lh = self.loader_registry[datatype](path_lh)
            rh = self.loader_registry[datatype](path_rh)

            if datatype == "vertex-ctx":
                lh["hemisphere"] = "L"
                rh["hemisphere"] = "R"
                return pd.concat([lh, rh], ignore_index=True)

            return lh, rh

        # Standard registry loading
        path = self.get_path(datatype, **kwargs)
        # Put file exists in readers!!! Issues with "filenaming_*_.nii.gz"
        #if not path.exists():
        #    raise FileNotFoundError(f"File not found: {path}")

        return self.loader_registry[datatype](path)


class StandardReader:
    """
    Reader to extract standardized data (template, atlas, ...).
    """
    def __init__(self, standards):
        self.stds = standards

        self.loader_registry = {
            ".nii": read_mri,
            ".nii.gz": read_mri,
            ".h5" : h5py.File,
            ".annot": ...,
        }

        self.exceptions_registry = {}

    # ------------
    # Public API
    # -----------

    def get_path(self, datatype):
        """
        Get file path to standardized object.
        """
        # Cleanup inputs (ensure no redundant 'STD-')
        datatype = str(datatype).replace("STD_", "")
        datatype = f"STD_{datatype}"

        return self.stds[datatype]

    def load(self, datatype):
        """
        Load standardized datatype.
        """
        if datatype in self.exceptions_registry:
            return self.exceptions_registry[datatype]

        file_path = self.get_path(datatype)

        if str(file_path).endswith(".nii.gz"):
            return self.loader_registry[".nii.gz"](file_path)

        suffix = file_path.suffix
        return self.loader_registry[suffix](file_path)
    
    def add_exceptions(self, datatype, loader):
        """
        Add exceptional loaders to the exception registry.
        """
        self.exceptions_registry[datatype] = loader

       