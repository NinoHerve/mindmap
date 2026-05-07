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
import importlib.resources

import numpy as np
import pandas as pd
import h5py

from bids import BIDSLayout
import nibabel as nib
import mne

from mindmap.core.io import read_config, read_eig, get_config
from mindmap.anatomy.utils import read_cortex_label
from mindmap.anatomy.mri.io import read_mri
from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.forward.io import read_bem_labels, make_volume_info
from .utils import build_fs_path, build_charm_path


CURRENT_DIR = Path(__file__).parent

class DatasetManager:
    """
    Manages the dataset context and spawns SubjectReaders.
    """
    def __init__(self, config : str | Path | dict):
        # store configuration
        if isinstance(config, (str, Path)):
            config = read_config(config)
        self.config = config

        self.roots = {
            "raw": config["DIR_RAW"],
            "mm" : config["DIR_MM"],
            "fs" : config["DIR_FS"],
            "charm": config["DIR_CHARM"],
        }

        pybids_config = importlib.resources.files("mindmap.resources").joinpath("pybids_config.json")
        self.layout = BIDSLayout(self.roots["raw"])
        self.layout.add_derivatives(self.roots["mm"], config=pybids_config)

        self.stds = {k : v for k, v in config.items() if "STD" in k}

        self.subjects = self.config.get("SUBS", [])

    def get_subject(self, subject_id):
        """
        Returns a reader for a specific subject.
        """
        if subject_id not in self.subjects:
            raise KeyError(f"Subject {subject_id} not in config list.")

        return SubjectReader(subject_id, self.roots, self.layout)

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
    def __init__(self, subject_id, roots, bidslayout):
        self.sub   = str(subject_id).replace("sub-", "")
        self.roots = roots
        self.layout = bidslayout

        self.fs_keys = [
            "t1.mgz", "brain", "orig", "white", "pial", "inflated",
            "inflated", "vertex-ctx", "vol_info",
        ]

        self.charm_keys = ["seg"]

        # -----------------------
        # Build default registry
        # -----------------------

        self.registry = {}

        # add bids registry

        # add mindmap registry
        def build_mm_path(subject, suffix, **kwargs):
            entities = {
                "subject": subject,
                "suffix": suffix,
                **kwargs,
            }

            path = self.layout(entities, scope="mindmap", absolute_path=False, validate=False)
            return self.roots["mm"]/path

        self.add_datatype("", "mm", build_mm_path, ...)
        self.add_datatype("dseg", "mm", build_mm_path, read_mri)
        self.add_datatype("parc", "mm", build_mm_path, self._load_parc)
        self.add_datatype("ctx", "mm", build_mm_path, self._load_ctx)
        self.add_datatype("src", "mm", build_mm_path, lambda f: mne.read_source_spaces(f, verbose="warning"))
        self.add_datatype("bem", "mm", build_mm_path, lambda f: mne.read_bem_solution(f, verbose="warning"))
        self.add_datatype("fwd", "mm", build_mm_path, lambda f: mne.read_forward_solution(f, verbose="warning"))
        self.add_datatype("con", "mm", build_mm_path, np.load)
        self.add_datatype("eig", "mm", build_mm_path, self._load_eig)
        self.add_datatype("ave", "mm", build_mm_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("i-ave", "mm", build_mm_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("v-ave", "mm", build_mm_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("cov", "mm", build_mm_path, ...)
        self.add_datatype("stc", "mm", build_mm_path, mne.read_source_estimate)

        # add FreeSurfer registry
        self.add_datatype("t1.mgz", "fs", build_fs_path, read_mri)
        self.add_datatype("brain", "fs", build_fs_path, read_mri)
        self.add_datatype("orig", "fs", build_fs_path, read_mri)
        self.add_datatype("white", "fs", build_fs_path, nib.freesurfer.io.read_geometry)
        self.add_datatype("pial", "fs", build_fs_path, nib.freesurfer.io.read_geometry)
        self.add_datatype("inflated", "fs", build_fs_path, nib.freesurfer.io.read_geometry)
        self.add_datatype("vertex-ctx", "fs", build_fs_path, read_cortex_label)
        self.add_datatype("vol-info", "fs", lambda root, subject, _, **kwargs: build_fs_path(root, subject, "t1.mgz", **kwargs), make_volume_info)
        self.add_datatype("ras2tkras", "fs", lambda root, subject, _, **kwargs: build_fs_path(root, subject, "t1.mgz", **kwargs), self._load_ras2tkras)

        # add Charm registry
        self.add_datatype("seg", "charm", build_charm_path, self._load_seg)

    # -----------------------------------
    # Internal Logic for Complex Objects
    # -----------------------------------

    def _load_parc(self, path):
        """
        Load parcellation.
        """
        clut_path = Path(str(path).replace(".nii.gz", ".txt"))
        return Parcellation(path, clut_path)

    def _load_seg(self, path):
        """
        Load segmentation.
        """
        clut_path = Path(str(path).replace(".nii.gz", "_LUT.txt"))
        return Parcellation(path, clut_path)

    def _load_ctx(self, **kwargs):
        """
        Load cortical mask.
        """
        parc       = self.load("parc", **kwargs)
        clut       = parc.get_clut()
        ctx_labels = clut[clut["Label name"].str.startswith("ctx")]["#No."].values
        return parc.create_mask(ctx_labels)

    def _load_eig(self, path):
        """
        Load eigenvalues and eigenvectors.
        """
        eigvec, eigval = read_eig(path)
        return {"vec":eigvec, "val":eigval}

    def _load_ras2tkras(self, **kwargs):
        """Load ras2tkras."""
        mri = self.load("t1.mgz", **kwargs)

        vox2ras     = mri.header.get_vox2ras()
        vox2ras_tkr = mri.header.get_vox2ras_tkr()

        return vox2ras_tkr @ np.linalg.inv(vox2ras)

    def _load_cov(self, path):
        """Load Covariance"""
        cov = mne.read_cov(path, verbose="warning")
        cov["data"] = cov["data"].astype("float")
        return cov


    # -----------
    # Public API
    # -----------

    def add_datatype(self, datatype, root_key, builder_func, loader_func):
        """
        Add new non-BIDS or derivative objects for reading.
        """
        self.registry[datatype] = (root_key, builder_func, loader_func)

    def get_path(self, suffix, **kwargs):
        """
        Dispatches to the correct builder based on datatype.
        """
        if suffix not in self.registry:
            raise KeyError(
                f"No registry entry for datatype '{suffix}'"
                f"Use reader.add_datatype() to register a custom datatype."
            )

        root_key, builder, _ = self.registry[suffix]

        return builder(
            self.roots[root_key],
            self.sub,
            suffix,
            **kwargs,
        )

    def load(self, suffix, **kwargs):
        """
        The main entry point with logic for expressions.
        """
        if suffix not in self.registry:
            raise KeyError(
                f"No registry for datatype '{suffix}'."
                f"Use reader.add_datatype() to register a custom loader."
            )

        # Exception 1: Double hemispheres
        if suffix in ["white", "pial", "inflated", "vertex-ctx"] and "hemi" not in kwargs:
            path_lh = self.get_path(suffix, hemi="lh")
            path_rh = self.get_path(suffix, hemi="rh")

            if not path_lh.exists() or not path_rh.exists():
                raise FileNotFoundError(f"Missing one or both hemisphere files for {suffix} at: {path_lh} or {path_rh}")

            lh = self.registry[suffix][2](path_lh)
            rh = self.registry[suffix][2](path_rh)

            if suffix == "vertex-ctx":
                lh["hemisphere"] = "L"
                rh["hemisphere"] = "R"
                return pd.concat([lh, rh], ignore_index=True)

            return lh, rh

        # Standard registry loading
        path = self.get_path(suffix, **kwargs)

        # Put file exists in readers!!! Issues with "filenaming_*_.nii.gz"
        #if not path.exists():
        #    raise FileNotFoundError(f"File not found: {path}")

        return self.registry[suffix][2](path)


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

       