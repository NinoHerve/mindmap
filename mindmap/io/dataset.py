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
from mindmap.anatomy.parcellation.processing import create_mask
from mindmap.anatomy.forward.io import read_bem_labels, make_volume_info, read_bem
from .utils import build_fs_path, build_charm_path, parse_extension


CURRENT_DIR = Path(__file__).parent

class DatasetManager:
    """
    Manages the dataset context and spawns SubjectReaders.
    """
    def __init__(self, config : str | Path | dict):

        # ---------------
        # Store metadata
        # ---------------

        # dataset configuration
        if isinstance(config, (str, Path)):
            config = read_config(config)
        self.config = config

        # add default roots
        self.roots = {}
        self.add_root("raw", config["DIR_RAW"])
        self.add_root("mm", config["DIR_MM"])
        self.add_root("fs", config["DIR_FS"])
        self.add_root("charm", config["DIR_CHARM"])

        # bids layout (path builder for raw and mindmap)
        pybids_config = importlib.resources.files("mindmap.resources").joinpath("pybids_config.json")
        self.layout = BIDSLayout(self.roots["raw"])
        self.layout.add_derivatives(self.roots["mm"], config=pybids_config)

        # list of dataset subjects
        self.subjects = self.layout.get(return_type="id", target="subject")

        # references
        self.refs = {k : v for k, v in config.items() if "REF" in k}

        # -------------------------------------------------------------------------
        # Build registry
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        # Format: {suffix: (root_key, path_builder, loader)}
        #
        # The registry saves for every suffix, the folder to look in (root_key),
        # a function to build the path to the desired file (path_builder) and
        # a function to load the desired file (loader).
        # ------------------------------------------------------------------------

        self.registry = {}

        # add bids registry
        def build_raw_path(_, subject, suffix, **kwargs):
            entities = {
                "subject": subject,
                "suffix": suffix,
                **kwargs,
            }
            path = self.layout.build_path(entities)
            return path
        
        self.add_datatype("T1w", "raw", build_raw_path, read_mri)

        # add mindmap registry
        def build_mm_path(_, subject, suffix, **kwargs):
            entities = {
                "subject": subject,
                "suffix": suffix,
                **kwargs,
            }

            path = self.layout.build_path(entities, scope="mindmap", absolute_paths=False, validate=False)
            return self.roots["mm"]/path

        self.add_datatype("dseg", "mm", build_mm_path, self._load_parc)
        self.add_datatype("ctx", "mm", lambda root, subject, _, **kwargs: build_mm_path(root, subject, "dseg", **kwargs), self._load_ctx)
        self.add_datatype("src", "mm", build_mm_path, lambda f: mne.read_source_spaces(f, verbose="warning"))
        self.add_datatype("fwd", "mm", build_mm_path, lambda f: mne.read_forward_solution(f, verbose="warning"))
        self.add_datatype("con", "mm", build_mm_path, self.load_by_extension)
        self.add_datatype("eig", "mm", build_mm_path, self._load_eig)
        self.add_datatype("ave", "mm", build_mm_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("i-ave", "mm", build_mm_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("v-ave", "mm", build_mm_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("cov", "mm", build_mm_path, self._load_cov)
        self.add_datatype("stc", "mm", build_mm_path, mne.read_source_estimate)

        # add FreeSurfer registry
        self.add_datatype("t1.mgz", "fs", build_fs_path, self.load_by_extension)
        self.add_datatype("brain.mgz", "fs", build_fs_path, self.load_by_extension)
        self.add_datatype("orig", "fs", build_fs_path, self.load_by_extension)
        self.add_datatype("white", "fs", self._build_fs_hemis, self._load_surface_hemis)
        self.add_datatype("pial", "fs", self._build_fs_hemis, self._load_surface_hemis)
        self.add_datatype("inflated", "fs", self._build_fs_hemis, self._load_surface_hemis)
        self.add_datatype("vertex-ctx", "fs", self._build_fs_hemis, self._read_cortex_labels)
        self.add_datatype("vol-info", "fs", lambda root, subject, _, **kwargs: build_fs_path(root, subject, "t1.mgz", **kwargs), make_volume_info)
        self.add_datatype("ras2tkras", "fs", lambda root, subject, _, **kwargs: build_fs_path(root, subject, "t1.mgz", **kwargs), self._load_ras2tkras)
        self.add_datatype("bem", "fs", build_fs_path, read_bem)

        # add Charm registry
        self.add_datatype("seg", "charm", build_charm_path, self._load_seg)
        self.add_datatype("t1-charm", "charm", build_charm_path, read_mri)

        # ------------------------------------------------
        # Build extension registry
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        # The extension registry handles loaders for the
        # reference data. Loader is chosen based on the
        # extension of the file name.
        # ------------------------------------------------

        self.ext_registry = {}
        self.add_extension(".nii", read_mri)
        self.add_extension(".nii.gz", read_mri)
        self.add_extension(".mgz", read_mri)
        self.add_extension(".h5", h5py.File)
        self.add_extension(".annot", ...)
        self.add_extension(".npy", np.load)

    # -----------
    # Public API
    # -----------

    def add_root(self, root_key, path):
        """Add new root folder."""
        self.roots[root_key] = path

    def add_reference(self, ref_key, path):
        """Add new reference file."""
        self.refs[ref_key] = path

    def add_datatype(self, suffix, root_key, builder_func, loader_func):
        """
        Add new object to the registry for automatic path building and loading.

        Parameters
        ----------
        suffix: str
            Key to access registry.

        root_key: str
            Root key ('raw', 'mm', 'fs', 'charm' or custom) to indicate
            folder in which to find the data.

        builder_func: func
            Function to build path to object. Every path builder needs to have the
            synthax `func(root, subject, suffix, **kwargs)` where **kwargs are 
            additional options to filter the type of data (like bids entities).

        loader_func: func
            Function to load the data given it's path. Every loader needs to have the
            synthax `func(path)`.
        """
        self.registry[suffix] = (root_key, builder_func, loader_func)

    def add_extension(self, extension, loader):
        """
        Add an extension specific loader.
        
        Parameters
        ----------
        ext_key: str
            Extension.

        loader: func
            Loader to use with this specific extension. The loader must
            have the synthax `func(path)`.
        """
        self.ext_registry[extension] = loader

    def get_path(self, subject, suffix, **kwargs):
        """
        Dispatches to the correct path builder based on suffix.
        """
        if suffix not in self.registry:
            raise KeyError(
                f"No registry entry for datatype '{suffix}'"
                f"Use reader.add_datatype() to register a custom datatype."
            )

        root_key, builder, _ = self.registry[suffix]

        return builder(
            self.roots[root_key],
            subject,
            suffix,
            **kwargs,
        )

    def load(self, subject, suffix, **kwargs):
        """
        Dispatches to the correct path builder & loader based on suffix.
        """
        if subject not in self.subjects:
            raise ValueError(f"Subject sub-{subject} does not exist in dataset.")
        
        if suffix not in self.registry:
            raise KeyError(
                f"No registry for datatype '{suffix}'."
                "Use reader.add_datatype() to register a custom datatype."
            )

        path = self.get_path(subject, suffix, **kwargs)

        return self.registry[suffix][2](path)

    def load_ref(self, ref_key):
        """Dispatches the correct loader for the reference data."""
        if ref_key not in self.refs.keys():
            raise KeyError(
                f"No registry for reference '{ref_key}'."
                "Add reference in the mindmap configuration file."
            )

        path = self.refs[ref_key]

        # check extension
        extension = parse_extension(path)

        if extension not in self.ext_registry:
            raise ValueError(
                f"No registry of extension '{extension}'."
                "Use `add_extension()` to configure a loader."
            )

        return self.ext_registry[extension](path)

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
    
    def get_subjects(self):
        """Return subject list."""
        return self.subjects


    # -----------------------------------
    # Internal logic for complex objects
    # -----------------------------------

    def load_by_extension(self, path):
        """Load file using the associated loader from the extension registry."""
        extension = parse_extension(path)
        return self.ext_registry[extension](path)

    def _build_fs_hemis(self, root, subject, suffix):
        """Duplicate path building for each hemisphere."""
        path_lh = build_fs_path(root, subject, suffix, hemi="lh")
        path_rh = build_fs_path(root, subject, suffix, hemi="rh")

        if not path_lh.exists() or not path_rh.exists():
            raise FileNotFoundError(f"Missing one or both hemisphere files for {suffix} at: {path_lh} or {path_rh}")

        return (path_lh, path_rh)

    def _load_surface_hemis(self, path):
        """Duplicate loading for each hemisphere."""
        lh = nib.freesurfer.io.read_geometry(path[0])
        rh = nib.freesurfer.io.read_geometry(path[1])

        return lh, rh

    def _read_cortex_labels(self, path):
        """Duplicate loading for each hemisphere."""
        lh = read_cortex_label(path[0])
        rh = read_cortex_label(path[1])

        lh["hemisphere"] = "L"
        rh["hemisphere"] = "R"

        return pd.concat([lh, rh], ignore_index=True)

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

    def _load_ctx(self, path_to_dseg):
        """
        Load cortical mask.
        """
        parc = self.registry["dseg"][2](path_to_dseg)
        clut = parc.get_clut()
        ctx_labels = clut[clut["Label name"].str.startswith("ctx")]["#No."].values
        return create_mask(parc, ctx_labels)

    def _load_eig(self, path):
        """
        Load eigenvalues and eigenvectors.
        """
        eigvec, eigval = read_eig(path)
        return {"vec":eigvec, "val":eigval}

    def _load_ras2tkras(self, path_to_t1):
        """Load ras2tkras."""
        mri = self.registry["t1.mgz"][2](path_to_t1)

        vox2ras     = mri.header.get_vox2ras()
        vox2ras_tkr = mri.header.get_vox2ras_tkr()

        return vox2ras_tkr @ np.linalg.inv(vox2ras)

    def _load_cov(self, path):
        """Load Covariance"""
        cov = mne.read_cov(path, verbose="warning")
        cov["data"] = cov["data"].astype("float")
        return cov

       