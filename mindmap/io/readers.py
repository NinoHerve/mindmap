"""
Module Name: readers.py
Description: This module provides functions to read various data types and formats.

Contains functions to read data. Also contains the abstract DataSetReader class
for building specific dataset readers.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path
from typing import Any
import os
import re
import glob
from abc import ABC, abstractmethod

import yaml
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
import ants
import mne
import mne_bids


# -------------------
# BIDS file naming
# -------------------

def get_bidspath(
        root        : str | Path,
        subject     : str,
        datatype    : str,
        session     : str = None,
        task        : str = None,
        run         : str | int = None,
        acquisition : str = None,
        space       : str = None,
        description : str = None,
        atlas       : str = None,
        resolution  : str = None,
        suffix      : str = None,
        extension   : str = None,
        mne         : bool= True,
        ) -> Path:
    """
    Retrieves file name following BIDS standards.

    Parameters
    ----------
    mne: bool
        Avoid mne_bids if False.
    """
    mne_bids_datatypes = ["meg", "eeg", "ieeg", "nirs", "anat", "beh"]
    other_datatypes    = ["bem", "fwd", "src", "cov", "con", "dseg", "ave", "stc", "eig"]

    assert datatype in mne_bids_datatypes + other_datatypes, f"datatype ({datatype}) is not valid. Should be one of {mne_bids_datatypes+other_datatypes}"

    # use mne_bids library if datatype available
    if datatype in mne_bids_datatypes and mne:
        fname = mne_bids.BIDSPath(
            root        = root,
            subject     = subject,
            datatype    = datatype,
            session     = session,
            task        = task,
            run         = run,
            acquisition = acquisition,
            space       = space,
            description = description,
            suffix      = suffix,
        )

        return fname.fpath

    # file path
    fpath = Path(root)/f"sub-{subject}"

    if session is not None:
        fpath = fpath/f"ses-{session}"

    fpath = fpath/f"{datatype}"

    # file name
    keys   = ["ses", "task", "run", "acq", "space", "desc", "atlas", "res"]
    values = [session, task, run, acquisition, space, description, atlas, resolution]

    fname = f"sub-{subject}"
    for key, val in zip(keys, values):
        if val is not None:
            fname += f"_{key}-{val}"

    # suffix and extension
    if suffix is None:
        datatype_suffix = {
            "bem"  : "_bem",
            "fwd"  : "_fwd",
            "src"  : "_src",
            "cov"  : "_cov",
            "con"  : "_con",
            "dseg" : "_dseg",
            "ave"  : "_ave",
            "stc"  : "_stc",
            "eig"  : "_eigvec",
            "anat" : "_T1w"
        }

        fname += datatype_suffix[datatype]
    else:
        fname += f"_{suffix}"

    # extension
    if extension is None:
        datatype_extension = {
            "bem"  : ".fif",
            "fwd"  : ".fif",
            "src"  : ".fif",
            "cov"  : ".fif",
            "con"  : ".npy",
            "dseg" : ".nii.gz",
            "ave"  : ".fif",
            "stc"  : "",
            "eig"  : ".npy",
            "anat" : ".nii.gz"
        }

        fname += datatype_extension[datatype]

    else:
        fname += f".{extension}"

    return fpath/fname

def get_charmpath(
        root     : str | Path,
        subject  : str,
        datatype : str,
) -> Path:
    """
    Retrieve file name for the CHARM folders.

    Parameters
    ----------
    root: str | Path
        Root folder where CHARM was run (folder of the T1).

    subject: str | Path
        Subject name.
        
    datatype: str
        Type of data to retrieve.

    Returns
    -------
    fname: Path
        File name.
    """
    types = ["t1", "seg", "lut"]
    assert datatype.lower() in types, f"Unknown datatype '{datatype}'. Expected {types}"

    root = root/f"m2m_sub-{subject}"/"segmentation"

    if datatype.lower() == "t1":
        fname = root/"T1_bias_corrected.nii.gz"

    elif datatype.lower() == "seg":
        fname = root/"labeling.nii.gz"

    else:
        fname = root/"labeling_LUT.txt"

    return fname


def check_file(
        fname     : str | Path,
        extension : list[str] | str = None,
        instance  : str = None,
        ) -> str | Path:
    """
    Check file:
    1. Check file existance. Raises FileNotFound if false.
    2. Check file extension if one provided. Raises ValueError if false.

    Parameters
    ----------
    fname: str | Path
        File name.

    extension: str
        Required extension.

    instance: str
        "str" or "path". Instance in which to return the file name.
        If None, return the file name as it was given.

    Return:
    ------
    fname: str | Path
        File name returned as string or Path.
    """

    # check if file exists
    f = Path(fname)
    if not f.exists():
        raise FileNotFoundError(f"Following file was not found: {fname}")

    # check extension
    if extension is not None:
        if not isinstance(extension, list):
            extension = [extension]

        extension = [ext if ext.startswith(".") else "." + ext for ext in extension]

        if not f.suffix in extension:
            raise ValueError(f"Invalid file extension for {f}. Expected {extension}")

    # return requested instance
    if instance is not None:
        if instance == "str":
            return str(f)
        return f

    return f




# --------------------
# Independent readers
# --------------------


def read_config(file: str | Path, env: bool = True) -> dict[str, Any]:
    """
    Read configuration file.

    Parameters
    ----------
    file: path-like
        Path to configuration file.

    env: bool
        If True, then the variables in the configuration file will be saved in the os 
        environment.

    Returns
    -------
    variables: dict[str, Any]
        If env is False, then the variables are returned in a dict object.
    """

    with open(file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    for key, value in config.items():
        if env:
            os.environ[key] = str(value)
        if isinstance(value, str) and "/" in value:
            config[key] = Path(value)

    return config



def read_clut(fname: str | Path) -> pd.DataFrame:
    """
    Read color lookup table.

    Parameters
    ----------
    fname: path-like
        File name. Can either be a .txt or .annot file.

    Returns
    -------
    clut: pd.DataFrame
        - #No.: str = label indices
        - Label name: str = label name
        - R: int, G: int, B: int, A: float = RGBA color code
    """

    clut = {"#No.":[], "Label name":[], "R":[], "G":[], "B":[], "A":[]}

    if str(fname).endswith(".annot"):
        annot = list(nib.freesurfer.io.read_annot(fname))
        clut["#No."] = np.arange(len(annot[2]))
        clut["Label name"] = [byte.decode("utf-8").strip() for byte in annot[2]]    # convert byte string and remove white space at the end
        clut["R"] = annot[1][:,0]
        clut["G"] = annot[1][:,1]
        clut["B"] = annot[1][:,2]
        clut["A"] = annot[1][:,3]

    else:
        with open(fname, "r", encoding="utf-8") as f:
            raw_clut = f.readlines()

        # read and process line by line
        pattern = re.compile(r"^\s*\d{1,5}\s+[a-zA-Z-_0-9*.]+\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}")
        for line in raw_clut:
            if pattern.match(line):
                s = line.rstrip().split(" ")
                s = [item.strip('\t') for item in s]
                s = list(filter(None, s))     

                clut["#No."].append(int(s[0]))
                clut["Label name"].append(s[1])
                clut["R"].append(int(s[2]))
                clut["G"].append(int(s[3]))
                clut["B"].append(int(s[4]))
                clut["A"].append(float(s[5]))

    return pd.DataFrame(clut)


def read_eeg(
    folder: str | Path,
    dataset: str,
) -> mne.Epochs | mne.Evoked:
    """
    Read EEG recordings.

    Implemented datasets:
    - "spike" for spike dataset (returns mne.Evoked)
    - "intra" for intra-cranial (returns mne.Epochs)

    Parameters
    ----------
    fname: path-like
        EEG folder.

    dataset: str
        Dataset name. Reading is different depending on the dataset.

    Returns
    -------
    eeg: mne.Epochs | mne.Evoked depending on dataset.
    """
    folder = Path(folder)

    if dataset == "spike":
        file = glob.glob(str(folder/"*204*.vhdr"))[0]
        eeg  = mne.io.read_raw_brainvision(file, scale=1e-6, misc=[], verbose="error")    # reads it in [V], hence the scale. misc=[] is a hack I found so that it reads the electrodes according to metadata. Else all electrodes are misc.
        data = eeg.get_data()
        info = eeg.info

        return mne.EvokedArray(data, info)

    elif dataset == "intra":
        # files
        folder = Path(folder)
        eegs   = glob.glob(str(folder/"*_desc-cleaned_eeg.vhdr")) 
        pos    = glob.glob(str(folder/"*electrodes.tsv"))[0]

        # eeg data
        data = []
        for f in eegs:
            spike = mne.io.read_raw_brainvision(f, verbose="error")
            data.append(spike["data"][0])
        data = np.array(data)

        # info
        info = spike.info

        # electrode positions
        pos     = pd.read_csv(pos, sep="\t", header=0)
        pos     = pos[pos["name"].isin(info["ch_names"])]
        pos     = dict(zip(pos["name"].values, pos[["x","y","z"]].values/1000))
        montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head")


        epochs = mne.EpochsArray(data, info)
        epochs = epochs.set_montage(montage)

        return epochs

    else:
        raise ValueError("Unknown dataset ('{}') for eeg reading.")


def read_talairach(fname: str | Path) -> np.ndarray:
    """
    Reads the Talairach transform from a talairach.lta file.
    """
    trans = np.loadtxt(fname, str, skiprows=17, max_rows=4)[:,2:5].astype(float).T
    trans = np.vstack([trans, [0,0,0,1]])

    return trans


def read_mri(file, mri_dtype="nib"):
    """
    Read nifti or mgz file format.
    """
    if mri_dtype.lower() == "nib":
        return nib.load(file)
    elif mri_dtype.lower() == "ants":
        return ants.image_read(str(file))

    raise ValueError("Unknown mri file type")


def read_eigendecomposition(fname: str | Path, return_eigval=False):
    """
    Read eigenvectors and eigenvalues (optional).

    Parameters
    ----------
    fname: path-like
        File name of eigenvectors.

    return_eigval: bool
        Whether to return eigval if eigval exists.

    Returns
    -------
    eigvec: np.ndarray (n,n)
        Eigenvectors.

    eigval: np.ndarray (n,) (optional)
        Eigenvalues.
    """
    eigvec = np.load(fname, allow_pickle=True)
    
    if return_eigval:
        new_name = fname.name.replace("eigvec", "eigval")
        eigval = np.load(fname.with_name(new_name), allow_pickle=True)
        return eigvec, eigval
    
    return eigvec


def read_cortex_label(fname):
    """
    Read cortex.label file from FreeSurfer.
    """
    cortex_label = pd.read_csv(
        fname,
        sep = "\s+",        # skips one or more spaces
        skiprows = 2,
        header = None,
        names = ["vertex id", "x", "y", "z", "statistics"],
        engine = "python",  # required for using the expression "\s+"
    )

    return cortex_label



##########################################################
# DatasetReader class
# --------------------
#
# Contains default readers for the FreeSurfer and MindMap
# derivatives. Implement child class to read a specific
# dataset.
##########################################################

class DatasetReader(ABC):
    """
    Abstract class to implement dataset specific readers.
    """
    def __init__(self, config=None):
        if isinstance(config, str):
            config = read_config(config)

        self.config = config

    @abstractmethod
    def __call__(self,):
        """Return requeseted data of a specific dataset."""


    def read_freesurfer(self, subject, *args, mri_dtype="nib"):
        """
        Read data inside FreeSurfer directory.

        Parameters:
        ----------
        subject: str
            Subject name.

        *args: list[str]
            Requested data. Data are requested via an abreviated code. See Returns to check 
            available abreviations and associated datatype.

        mri_dtype: str
            Type for the mri dataobject (nibabel (nib) or ants)

        Returns:
        -------
        Dictionnary containing (only) requested data:
            - t1-fs: nib.Nifti1Image | ants.ANTsImage
                T1-weigthed MRI in freesurfer space (tkras).
            - t1-brain: nib.Nifti1Image | ants.ANTsImage
                Brain mask.
            - white: list[np.ndarray (n,3), np.ndarray (m,3)]
                White surface (boundary between white a gray matter). The surface correponds
                to a mesh defined by its vertex locations (n,3) and triangular faces (m,3).
            - charm: nib.Nifti1Image | ants.ANTsImage
                CHARM head segmentation.
            - tkras2ras: np.ndarray (4,4)
                Affine transformation (freesurfer space (tkras) -> T1 ras space)
            - vertex-ctx
                List of vertices of brain meshes belonging to cortex.
        """
        fs_dir = self.config["FS_DIR"]

        data = {}

        for key in args:

            # -------------
            # Read Anatomy
            # -------------

            # T1-weighted MRI (freesurfer space)
            if key.lower() == "t1":
                file      = fs_dir/f"sub-{subject}"/"mri"/"T1.mgz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            # PLAN TO DELETE THIS ONE
            if key.lower() == "t1-orig":
                # T1 original image
                file =  fs_dir/f"sub-{subject}"/"mri"/"orig"/"001.mgz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            # Brain mask (freesurfer space)
            if key.lower() == "t1-brain":
                file = fs_dir/f"sub-{subject}"/"mri"/"brain.mgz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            # White surface (freesurfer space)
            if key.lower() == "white":
                file_l = fs_dir/f"sub-{subject}"/"surf"/"lh.white"
                file_r = fs_dir/f"sub-{subject}"/"surf"/"rh.white"

                white_l = nib.freesurfer.io.read_geometry(file_l)
                white_r = nib.freesurfer.io.read_geometry(file_r)
                
                data[key] = [white_l, white_r]

            if key.lower() == "vertex-ctx":
                file_l = fs_dir/f"sub-{subject}"/"label"/"lh.cortex.label"
                file_r = fs_dir/f"sub-{subject}"/"label"/"rh.cortex.label"

                lh_vertex = read_cortex_label(file_l)
                rh_vertex = read_cortex_label(file_r)

                # add hemisphere
                lh_vertex["hemisphere"] = "L"
                rh_vertex["hemisphere"] = "R"

                data[key] = pd.concat([lh_vertex, rh_vertex], ignore_index=True)


            # CHARM head segmentation (freesurfer or T1 space ???)
            if key.lower() == "seg":
                # Head segmentation (charm)
                from mindmap.head import Parcellation
                fseg = fs_dir/f"sub-{subject}"/"mri"/f"m2m_sub-{subject}/segmentation/labeling.nii.gz"
                flut = fs_dir/f"sub-{subject}"/"mri"/f"m2m_sub-{subject}/segmentation/labeling_LUT.txt"
                data[key] = Parcellation(fseg, flut)

            # ---------------------
            # Read transformations
            # ---------------------

            # Affine transform (freesurfer space -> T1 space)
            if key.lower() == "tkras2ras":
                file = fs_dir/f"sub-{subject}"/"mri"/"T1.mgz"
                mri  = nib.load(file)

                vox2ras     = mri.header.get_vox2ras()
                vox2ras_tkr = mri.header.get_vox2ras_tkr()

                data[key] = vox2ras @ np.linalg.inv(vox2ras_tkr)

            # PLAN TO DELETE THIS ONE
            if key.lower() == "cras":
                # Translation part of the Talairach transform
                file = fs_dir/f"sub-{subject}"/"mri"/"transforms"/"talairach.lta"
                data[key] = read_talairach(file)[:3,3]
                
        return data


    def read_mindmap(
            self,
            subject : str,
            *args   : list,
            session : str = None,
            task    : str = None,
            run     : str = None
            ) -> dict:
        """
        Read data inside Mindmap directory.

        Parameters
        ----------
        subject: str
            Subjecte name.

        *args: list[str]
            Requested data. Data are requested via an abreviated code. See Returns to check 
            available abreviations and associated datatype.

        session: str
            Session name.

        task: str
            Task name.

        run: str | int
            Run name or number.

        Returns
        -------
        data: dict
            Dictionnary containing (only) requested data:
            - parc: Parcellation
                Parcellation (T1-ras space)
            - ctx: nib.Nifti1Image
                Cortical mask.
            - src: mne.SourceSpaces
                Source space.
            - src-raw: mne.SourceSpaces
                Source space before forward solution.
            - bem: mne.BEM
                Boundary element model.
            - bem-sim: mne.BEM
                Boundary element model for simulations.
            - fwd: mne.Forward
                Forward solution
            - sc-nfiber: np.ndarray
                Structural connectome with number of fibers as weights.
            - sc-length: np.ndarray
                Structural connectome with fiber lengths as weights.
            - sc-rb-nfiber: np.ndarray
                Region-based structural connectome with number of fibers as weights.
            - sc-rb-nfiber: np.ndarray
                Region-based structural connectome with fiber lengths as weights.
            - eeg: mne.Evoked
                Averaged and preprocessed EEG.
            - cset: mne.SourceEstimate
                CSET inverse solution.
            - sloreta: mne.SourceEstimate
                sLORETA inverse solution.
            - wcset: mne.SourceEstimate
                weighted CSET inverse solution.
            - trans: mne.Transformation
                Affine transformation (sensor space -> freesurfer space)
        """
        mm_dir = self.config["MM_DIR"]

        data = {}

        for key in args:

            # -------------
            # Read Anatomy
            # -------------

            # parcellation (T1-ras space)
            if key.lower() == "parc":
                from mindmap.head import Parcellation
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    space       = "T1w",
                    atlas       = "MultiConn2022",
                    description = "scale4",
                    datatype    = "dseg",
                )
                clut = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    space       = "T1w",
                    atlas       = "MultiConn2022",
                    description = "scale4",
                    datatype    = "dseg",
                    extension   = "txt",
                )
                data[key] = Parcellation(file, clut)

            # cortical mask (T1-ras space)
            if key.lower() == "ctx":
                d         = self.read_mindmap(subject, "parc")
                clut      = d["parc"].get_clut()
                ctx       = clut[clut["Label name"].str.startswith("ctx")]
                data[key] = d["parc"].create_mask(ctx["#No."].values)

            # -----------------------------------
            # Read Boundary Element Models (BEM)
            # -----------------------------------

            # Boundary element model (freesurfer space in [mm])
            if key.lower() == "bem":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "inverse",
                    datatype    = "bem",
                )
                data[key] = mne.read_bem_solution(file, verbose="warning")

            # Boundary element model for simulations (freesurfer space in [mm])
            if key.lower() == "bem-sim":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "simulation",
                    datatype    = "bem",
                )
                data[key] = mne.read_bem_solution(file, verbose="warning")


            # ------------------
            # Read source space
            # ------------------

            if key.lower() == "src":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "inverse",
                    datatype    = "fwd",
                )
                fwd       = mne.read_forward_solution(file, verbose="warning")
                data[key] = fwd["src"]

            if key.lower() == "src-raw":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    datatype    = "src",
                )
                data[key] = mne.read_source_spaces(file, verbose="warning")


            # -----------------------
            # Read forward solutions
            # -----------------------

            if key.lower() == "fwd":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "inverse",
                    datatype    = "fwd",
                )
                data[key] = mne.read_forward_solution(file, verbose="warning")

            if key.lower() == "fwd-sim":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "simulation",
                    datatype    = "fwd",
                )
                data[key] = mne.read_forward_solution(file, verbose="warning")

            # ------------------
            # Read Connectomics
            # ------------------

            # distance-based structural connectome
            if key.lower() == "sc":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "distance",
                    atlas       = "sourcespace",
                    datatype    = "con",
                )
                data[key] = np.load(file)

            # region-based structural connectome with fiber lengths
            if key.lower() == "sc-rb-length":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "fiberlength",
                    atlas       = "MultiConn2022",
                    resolution  = "scale4",
                    datatype    = "bem",
                )
                data[key] = np.load(file)

            # region-based structural connectome with number of fibers
            if key.lower() == "sc-rb-nfiber":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    description = "numfibers",
                    atlas       = "MultiConn2022",
                    resolution  = "scale4",
                    datatype    = "bem",
                )
                data[key] = np.load(file)

            if key.lower() == "eig":
                file = get_bidspath(
                    root = mm_dir,
                    subject = subject,
                    datatype = "eig",
                )
                eigvec, eigval = read_eigendecomposition(file, return_eigval=True)
                data[key] = {"vec":eigvec, "val":eigval}

            # -------------
            # Read EEG
            # -------------

            # evoked signal
            if key.lower() == "ave":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    datatype    = "ave",
                    session     = session,
                    task        = task,
                    run         = run,
                )

                assert file.exists(), "Averaged EEG file not found. Check if the proper 'task' and/or 'run' are provided."
                
                data[key] = mne.Evoked(file, verbose="warning")

            # noise covariance
            if key.lower() == "ncov":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    datatype    = "cov",
                    session     = session,
                    task        = task,
                    run         = run,
                )

                assert file.exists(), "Noise covariance not found. Check if the proper 'task' and/or 'run' are provided."

                data[key] = mne.read_cov(file, verbose="warning")

            # -------------------------------
            # Read source time currents (stc)
            # -------------------------------

            # connectome spectrum electromagnetic tomography
            if key.lower() == "cset":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    datatype    = "stc",
                    task        = task,
                    run         = run,
                    description = "cset",
                )

                assert file.exists(), "CSET solution not found. Check if the proper 'task' and/or 'run' are provided."

                data[key] = mne.read_source_estimate(file)

            # standardized low resolution electromagnetic tomography
            if key.lower() == "sloreta":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    datatype    = "stc",
                    task        = task,
                    run         = run,
                    description = "sloreta",
                )

                assert file.exists(), "sLORETA solution not found. Check if the proper 'task' and/or 'run' are provided."

                data[key] = mne.read_source_estimate(file)

            # weighted connectome spectrum electromagnetic tomography
            if key.lower() == "wcset":
                file = get_bidspath(
                    root        = mm_dir,
                    subject     = subject,
                    datatype    = "stc",
                    task        = task,
                    run         = run,
                    description = "wcset",
                )

                assert file.exists(), "wCSET solution not found. Check if the proper 'task' and/or 'run' are provided."

                data[key] = mne.read_source_estimate(file)

            # -----------------------
            # Read transformations
            # -----------------------



            # Coregistration (sensor space -> freesurfer space)
            if key.lower() == "trans":
                file = mm_dir/f"sub-{subject}"/"eeg"/f"sub-{subject}_eeg-trans.fif"
                data[key] = mne.read_trans(file)

        return data


    def read_standards(self, *args, mri_dtype="nib"):
        """
        Read standardized data (atlases, templates, etc ...) files.

        Returns
        -------
        data: dict
            Dictionnary containing (only) requested data:
            - std-conn: h5py
                Connectome atlas (MNI space)
            - std-ctx: nib.Nifti1Image | ants.ANTsImage
                Cortex atlas.
            - std-thal: nib.Nifti1Image | ants.ANTsImage
                Thalamic Nuclei atlas.
            - std-thalclut: str
                File path to Thalamic nuclei color lookup table.
            - std-mni: nib.Nifti1Image | ants.ANTsImage
                MNI template.
        """
        data = {}

        for name in args:
            # connectome atlas
            if name.lower() == "std-conn":
                # Connectome atlas (h5 format)
                file = self.config["CONN"]
                data[name] = h5py.File(file)

            if name.lower() == "std-ctx":
                # Cortical atlas (lausanne 2018 file paths)
                data[name] = {"LAUSL":str(self.config["LAUSL"]), "LAUSR":str(self.config["LAUSR"])}

            if name.lower() == "std-thal":
                # Thalamus Nuclei atlas (ants)
                file = self.config["THAL"]
                data[name] = read_mri(file, mri_dtype)

            if name.lower() == "std-thalclut":
                # Thalamus Nuclei color look up table (file path)
                data[name] = self.config["THAL_CLUT"]

            if name.lower() == "std-mni":
                # MNI template (MNI152 NLIN 2009)
                file = self.config["MNI"]
                data[name] = read_mri(file, mri_dtype)

        return data


    def log(self, data, args):
        """
        Print state of retrieved data 
        """
        missing   = list(set(args) - set(data.keys()))
        retrieved = list(set(data.keys())-set(["name", "sub", "task", "run"]))

        print("\033[95mDATA SUMMARY\033[0m")
        print(f"Dataset: {data['name']}")
        print(f"Subject: {data['sub']}")
        print(f"Task: {data['task']}")
        print(f"Run: {data['run']}")

        print("Retrieved objects:")
        for r in retrieved:
            print(f"- \033[92m{r}\033[0m")

        print("Missing objects:")
        for m in missing:
            print(f"- \033[91m{m}\033[0m")



# ########### OLD CODE ################
# ####### DELETE THIS PART SOON #######

# def read_dataset(subject: str, *args: str, sample: str = None, mri_dtype: str = "nib", verbose: int = 1) -> dict:
#     """
#     Read data according to configuration. To use this function, first read a configuration file with env=True.

#     Parameters
#     ----------
#     subject: str
#         Subject name.

#     *args: str
#         Names of data objects to load.

#     sample: str
#         Specify sample if multiple samples.

#     mri_file_type: str
#         MRI file type -> ants or nibabel (nib)

#     Returns
#     -------
#     data: dict(dataName: dataObject)
#         Data.
#     """
#     dataset = _retrieve_env("NAME")

#     if dataset.lower() == "spike":
#         data = _read_dataset_spike(subject, *args, mri_dtype=mri_dtype)

#     elif dataset.lower() == "intra":
#         data = _read_dataset_intra(subject, *args)

#     elif dataset.lower() == "vep":
#         data = _read_dataset_vep(subject, *args)
    
#     elif dataset.lower() == "vepcon":
#         data = _read_dataset_vepcon(subject, *args, sample=sample)

#     elif dataset.lower() == "simf":
#         data = _read_dataset_simf(subject, *args, sample=sample, mri_dtype=mri_dtype)

#     else: 
#         raise ValueError(f"Unknown dataset '{dataset}' for read_dataset()")

#     if verbose == 1:
#         missing = list(set(args) - set(data.keys()))
#         _log(data, missing)

#     return data




# def _log(data, args):
#     """
#     Print state of retrieving data 
#     """
#     retrieved = list(set(data.keys())-set(["name", "sub"]))
    
#     print("\033[95mDATA SUMMARY\033[0m")
#     print(f"Dataset: {data['name']}")
#     print(f"Subject: {data['sub']}")

#     print(f"Retrieved objects:")
#     for r in retrieved:
#         print(f"- \033[92m{r}\033[0m")

#     print(f"Missing objects:")
#     for a in args:
#         print(f"- \033[91m{a}\033[0m")




# def read_standards(*args: str) -> dict:
#     """
#     Read standardized data (atlas, templates ...) according to configuration. To use this function, first read a configuration file with env=True.

#     Parameters
#     ----------
#     *args: str
#         Names of data objects to load.

#     Returns
#     -------
#     data: dict(dataName: dataObject)
#         Data.
#     """
    
#     dataset = _retrieve_env("NAME")
    
#     data = dict()

#     if dataset.lower() == "spike":
#         data = _read_standards_spike(*args)

#     elif dataset.lower() == "intra": 
#         pass 

#     elif dataset.lower() == "vep":
#         pass

#     elif dataset.lower() == "vepcon":
#         data = _read_standards_vepcon(*args)

#     elif dataset.lower() == "simf":
#         data = _read_standards_spike(*args)

#     else: 
#         raise ValueError(f"Unknown dataset '{dataset}' for read_standards()")

#     if len(data.keys()) < len(args):
#         print("\033[93mSome arguments did not correpond to the list of available objects to load. Check the list of arguments again.")

#     return data



# # -------------------------------
# # ******* DATASET READER ********
# # -------------------------------



# def _read_dataset_spike(subject, *args, mri_dtype="nib"):
#     """
#     Read data for dataset SPIKE.

#     Parameters
#     ----------
#     subject: str
#         Subject name

#     *args: list(str)
#         Data ids:
#         - eeg: EEG (looks for eeg_raw if no EEG in mindmap folder).
#         - eeg_raw: EEG from the original dataset.
#         - resec: Resection area
#         - mri_post: Postsurgery T1w MRI.
#         For the rest, see read_freesurfer and read_mindmap

#     mri_dtype: str
#         MRI dtype (Nibabel or ANTs) for MRI images.

#     Returns:
#     data: dict
#         Dictionnary of data using data ids as keys.
#     """
    
#     env  = _retrieve_env("NAME", "DS_DIR", "MM_DIR")
#     data = dict(name=env["NAME"], sub=subject)

#     # Others
#     for name in args:
            
#         if name.lower() == "eeg":
#             file = env["MM_DIR"]/subject/"eeg"/f"{subject}-spike_ave.fif"
#             if file.exists():
#                 continue    # will read with read_mindmap() 
#             else:
#                 print(f"\033[93mPreprocessed EEG not found, reading raw instead.\033[93m")
#                 d = _read_dataset_spike(subject, "eeg-raw")
#                 data[name] = d["eeg-raw"]

#         if name.lower() == "eeg-raw":
#             folder = env["DS_DIR"]/"derivatives"/"cartool-v3.80"/subject/"ses-preop"/"eeg"
#             data[name] = read_eeg(folder, dataset=env["NAME"])
        
#         if name.lower() == "resec":
#             # mask of resected area
#             manual = env["MM_DIR"]/subject/"resec"/f"{subject}_man-recon-seg.nii.gz"
#             auto   = env["MM_DIR"]/subject/"resec"/f"{subject}_auto-seg.nii.gz"

#             if manual.exists() == True:
#                 data["resec"] = nib.load(manual)
#             elif auto.exists() == True:
#                 data["resec"] = nib.load(auto)
#             else:
#                 continue

#         if name.lower() == "mri_post":
#             file = env["DS_DIR"]/"derivatives"/"cmp-v3.0.0-beta-RC1"/subject/"ses-postop"/"anat"/f"{subject}_ses-postop_space-individual_desc-brain_T1w.nii.gz"
#             data[name] = read_mri(file, mri_dtype=mri_dtype)

#     args = list(set(args) - set(data.keys()))    # update missing arguments

#     # FreeSurfer
#     d    = read_freesurfer(subject, *args, mri_dtype=mri_dtype) 
#     data.update(d)                                                 
#     args = list(set(args) - set(data.keys()))                       

#     # Mindmap
#     d   = read_mindmap(subject, *args, sample="spike")
#     data.update(d)
#     args = list(set(args) - set(data.keys()))

#     return data


# def _read_dataset_intra(subject, *args):
#     raise NotImplementedError("_read_dataset_intra(subject *args)")


# def _read_dataset_vep(subject, *args):
        
#     env = _retrieve_env("FS_DIR", "TVB_DIR", "MM_DIR", "DS_DIR")

#     data = dict()

#     for name in args:
#         if name.lower() == "t1":
#             # T1 MRI template (Colin27)
#             file =  env["FS_DIR"]/"Colin27"/"mri"/"orig"/"001.mgz"
#             data[name] = nib.load(file)

#         if name.lower() == "t1-ants":
#             # T1 MRI loaded with ants
#             file = env["FS_DIR"]/"Colin27"/"mri"/"orig"/"001.mgz"
#             data[name] = ants.image_read(str(file))

#         if name.lower() == "parc":
#             # Brain parcellation of template (VEP atlas)
#             file = env["FS_DIR"]/"Colin27"/"mri"/"aparc+aseg.vep.mgz"
#             clut = env["FS_DIR"]/"VepFreeSurferColorLut.txt"
#             data[name] = mm.utils.Parcellation(file, clut)

#         if name.lower() == "ctx_mask":
#             # Cortical mask of template (without cerebellum)
#             file = env["FS_DIR"]/"Colin27"/"mri"/"aparc+aseg.vep.mgz"
#             clut = env["FS_DIR"]/"VepFreeSurferColorLut.txt"
#             parc = mm.utils.Parcellation(file, clut)

#             df  = parc.get_clut()
#             ctx = df[df["#No."] > 71000]
#             ctx = ctx[~ctx["Label name"].str.contains("Cerebellar")]

#             data[name] = parc.retain(ctx["#No."].values).create_mask()

#         if name.lower() in ["eeg_dum", "eeg_fwd"]:
#             # Dummy eeg (zero signal)
#             montage  = mne.channels.make_standard_montage("biosemi64")
#             ch_names = montage.ch_names
#             ch_types = ["eeg"] * len(ch_names)
#             sfreq    = 200 

#             info = mne.create_info(ch_names, sfreq, ch_types)
#             info.set_montage(montage)

#             recordings = np.zeros((len(ch_names), sfreq*10))
#             data[name] = mne.io.RawArray(recordings, info, verbose="warning")

#         if name.lower() == "eeg_tmp":
#             # Template EEG (coregistered to template head)
#             file = env["MM_DIR"]/"Colin27"/"eeg"/"Colin27_coreg_null-raw.fif"
#             data[name] = mne.io.read_raw(file, verbose="warning")

#         if name.lower() in ["eeg_sim", "eeg_proc"]:
#             # Simulated eeg (coregistered to template head)
#             file = env["MM_DIR"]/subject/f"{subject}_sim-raw.fif"
#             data[name] = mne.io.read_raw(file, verbose="warning")

#         if name.lower() == "fwd":
#             # Forward solution of template head
#             file = env["MM_DIR"]/"Colin27"/"fwd"/"Colin27_bem-fwd.fif"
#             data[name] = mne.read_forward_solution(file, verbose="warning")

#         if name.lower() == "src":
#             # Sources of template head
#             file = env["MM_DIR"]/"Colin27"/"fwd"/"Colin27_bem-fwd.fif"
#             fwd  = mne.read_forward_solution(file, verbose="warning")
#             data[name] = fwd["src"]

#         if name.lower() == "bem":
#             # Boundary Element Model of template head
#             file = env["MM_DIR"]/"Colin27"/"fwd"/"Colin27_hm-bem.fif"
#             data[name] = mne.read_bem_surfaces(file, verbose="warning")

#         if name.lower() == "cras":
#             # Translation part of the Talairach transform from template head
#             file = env["FS_DIR"]/"Colin27"/"mri"/"transforms"/"talairach.lta"
#             data[name] = read_talairach(file)[:3,3]

#         if name.lower() == "sim_tvb":
#             # Simulated brain signals (tvb)
#             file = env["TVB_DIR"]/subject/"ses-03"/"VEPhypothesis"/f"{subject}_simulated_source_timeseries_run-01.npz"
#             data[name] = np.load(file)

#         if name.lower() == "atlas":
#             # Region names
#             file  = env["DS_DIR"]/"vep_atlas.tsv"
#             atlas = pd.read_csv(file, sep="\t")
#             data[name] = atlas

#         if name.lower() == "sim":
#             # Simulated brain signals (per dipole)
#             try:
#                 tmp = mm.io.read_dataset(subject, "sim_tvb", "src", "cras", "parc", "atlas")
#             except FileNotFoundError:
#                 raise FileNotFoundError("To load simulated data, you first need a source model. Run 2_forward_problem.ipynb to generate one.")

#             # label dipoles
#             pos    = mm.utils.get_src_position(tmp["src"])
#             pos    = pos * 1000 + tmp["cras"]
#             labels = tmp["parc"].label_points(pos)

#             # map label name to idx in source signal data
#             name_to_idx = dict(zip(tmp["atlas"]["region_name"], tmp["atlas"]["label"]))
#             idx         = [name_to_idx[l]-1 for l in labels]            # -1 because 0 is unknown and does not exist. so signal[0] is label 1.
#             sim         = tmp["sim_tvb"]["source_signal"][:,idx]

#             times = tmp["sim_tvb"]["time_steps"]
#             sim = mne.SourceEstimate(
#                 data     = sim.T,
#                 vertices = [tmp["src"][0]["vertno"], tmp["src"][1]["vertno"]],
#                 tmin     = times[0],
#                 tstep    = times[1]-times[0]
#             )

#             sim.labels = labels
#             data[name] = sim

#         if name.lower() == "ez":
#             file = env["TVB_DIR"]/subject/"ses-03"/"VEPhypothesis"/"parameters"/f"{subject}_epileptor_parameters_run-01.tsv"
#             df   = pd.read_csv(file, sep="\t")
#             data[name] = ast.literal_eval(df["EZ"].values[0])

#     return data


# def _read_dataset_vepcon(subject, *args):
#     env = _retrieve_env("NAME", "DS_DIR", "FS_DIR", "MM_DIR")

#     data = dict()
    

#     for name in args:
#         if name.lower() == "fwd":
#             # forward solution
#             file =  env["DS_DIR"]/"derivatives"/"cscs"/subject/f"{subject}_fwd.fif"
#             data[name] = mne.read_forward_solution(file, verbose="warning")

#         if name.lower() in ["eeg_raw", "eeg_fwd"]:
#             # T1 MRI loaded with ants
#             file = env["DS_DIR"]/"derivatives"/"eeglab-v14.1.1"/subject/"eeg"/f"{subject}_task-faces_desc-preproc_eeg.set"
#             data[name] = mne.read_epochs_eeglab(file, verbose="warning")
            
#         if name.lower() == "cst":
#             # Connectome spectrum transform
#             file = env["DS_DIR"]/"derivatives"/"cscs"/subject/f"{subject}-harmonics.npy"
#             data[name] = np.load(file, allow_pickle=True)

#         if name.lower() == "bem":
#             # Boundary Element Surfaces
#             file = env["FS_DIR"]/subject/"bem"/f"{subject}-head.fif"
#             data[name] = mne.read_bem_surfaces(file, verbose="warning")

#         if name.lower() == "cras":
#             # Translation part of the Talairach transform
#             file = env["FS_DIR"]/subject/"mri"/"transforms"/"talairach.lta"
#             data[name] = read_talairach(file)[:3,3]

#         if name.lower() == "t1":
#             # T1 MRI
#             file =  env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
#             data[name] = nib.load(file)

#         if name.lower() == "t1-ants":
#             # T1 MRI loaded with ants
#             file = env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
#             data[name] = ants.image_read(str(file))

#         if name.lower() == "ctx_mask":
#             # Cortical mask
#             file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
#             clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
#             parc = mm.utils.Parcellation(file, clut)

#             df  = parc.get_clut()
#             ctx = df[df["Label name"].str.startswith("ctx")]
#             data[name] = parc.create_mask(ctx["#No."].values)

#     return data


# def _read_dataset_simf(subject, *args, sample=None, mri_dtype="nib"):
#     """
#     Read data for dataset SIMF.

#     Parameters
#     ----------
#     subject: str
#         Subject name

#     *args: list(str)
#         Data ids:
#         - eeg: EEG (looks for eeg_raw if sample is None)
#         - eeg_raw: Coregistered EEG with zeros if exists, else not coregistered EEG with zeros
#         - trans: Identity transformation to coregister EEG
#         - tracto: Tractography
#         - t12diff: Transformation from T1 to diffusion space
#         - gt: Ground truth
#         For the rest, see read_freesurfer and read_mindmap

#     sample: str
#         Sample name for EEG.

#     mri_dtype: str
#         MRI dtype (Nibabel or ANTs) for MRI images.

#     Returns:
#     data: dict
#         Dictionnary of data using data ids as keys.
#     """
#     env  = _retrieve_env("NAME", "MM_DIR", "CMP_DIR", "SIM_DIR")

#     data = dict(name=env["NAME"], sub=subject)

#     if isinstance(sample, int):
#         sample = f"sample{sample:03}"

#     # dataset specifics (use this section to also overwrite freesurfer and mindmap)
#     for name in args:
#         if name.lower() == "eeg":
#             if sample is None:
#                 # look for raw EEG
#                 d = _read_dataset_simf(subject, "eeg-raw")
#                 data[name] = d["eeg-raw"]
#             else:
#                 file = env["SIM_DIR"]/"eeg"/f"{subject}-{sample}_ave.fif"
#                 data[name] = mne.Evoked(file, verbose="warning")

#         if name.lower() == "eeg-raw":
#             file = env["MM_DIR"]/subject/"eeg"/f"{subject}_coreg-ave.fif"

#             if file.exists():
#                 print("\033[93mCoregistered EEG found\033[0m")
#                 data[name] = mne.Evoked(file, verbose="warning")

#             else:
#                 print("\033[93mNo coregistered EEG found, making EEG from scratch\033[0m")
#                 # null EEG
#                 # load builtin montage
#                 montage = mne.channels.make_standard_montage("EGI_256")
        
#                 # create info
#                 info    = mne.create_info(ch_names=montage.ch_names, ch_types="eeg", sfreq=250)
#                 info.set_montage(montage)

#                 info["description"] = "Simulation EEG"
#                 info["bads"] = ["E31", "E67", "E73", "E82", "E91", "E92", "E102", "E111", "E120", "E133", "E145", "E165", "E174", "E187", "E199", "E208", "E209", "E216", "E217", "E218", "E219"] + [f"E{i}" for i in range(225,257)]

#                 # create null data
#                 eeg = np.zeros((info["nchan"], int(info["sfreq"])))

#                 # create eeg
#                 eeg = mne.EvokedArray(eeg, info, verbose="warning")
#                 eeg.drop_channels(info["bads"])

#                 data[name] = eeg

#         if name.lower() == "trans":
#             data[name] = mne.transforms.Transform("head", "mri")    # identity transformation

#         if name.lower() == "tracto":
#             # Tractogram
#             if subject == "vepcon_sub-10":
#                 file = env["CMP_DIR"]/subject/"dwi"/f"sub-10_model-CSD_desc-DET_tractogram.trk"
#             else:
#                 file = env["CMP_DIR"]/subject/"fibtracking"/"sd_stream"/"dwi_denoised_eddy_distc_corrected_1mm_shore_sdstream.trk"
#             data[name] = nib.streamlines.load(file)

#         if name.lower() == "t12diff":
#             # Transforms from T1 to diffusion
#             if subject == "vepcon_sub-10":
#                 file = env["CMP_DIR"]/"vepcon_sub-10"/"xfm"/"final0GenericAffine.mat"
#             else:
#                 file = env["CMP_DIR"]/subject/"transforms"/"all_t12diff0GenericAffine.mat"
#             affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
#             data[name] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))


#         if name.lower() == "diff-affine":
#             # Affine (diffusion -> T1)
#             file   = env["CMP_DIR"]/subject/"xfm"/"final0GenericAffine.mat"
#             affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
#             data[name] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))

#         if name.lower() == "gt":
#             # ground truth 
#             if sample is None: raise ValueError("Require sample name to read data 'gt'.")
#             file = env["SIM_DIR"]/"stc"/f"{subject}-{sample}_stc.fif"
#             data[name] = mne.read_source_estimate(file) 

#         if name.lower() == "lam":
#             # eigenvalues
#             file = env["MM_DIR"]/subject/"sc"/f"{subject}_lams-template.npy"
#             data[name] = np.load(file)


#     args = list(set(args) - set(data.keys()))   # update list of missing arguments

#     # FreeSurfer
#     d    = read_freesurfer(subject, *args, mri_dtype=mri_dtype)     
#     data.update(d)                                                 
#     args = list(set(args) - set(data.keys()))                   

#     # mindmap
#     d    = read_mindmap(subject, *args, sample=sample)
#     data.update(d)
#     args = list(set(args) - set(data.keys()))

#     return data


# # -----------------------------------
# # Derivatives reader
# # 
# # FreeSurfer reader & MindMap reader
# # -----------------------------------



# def read_freesurfer(subject, *args, mri_dtype="nib"):
#     """
#     Read data inside FreeSurfer directory.

#     Parameters:
#     ----------
#     subject: str
#         Subject name.

#     *args:
#         Data ids:
#         - T1: T1.mgz
#         - T1-orig: original image
#         - T1-brain: brain mask
#         - white: white surface (intersection white matter and gray matter)
#         - cras: Translation to go from original space to FreeSurfer space.

#     mri_dtype: str
#         Type for the mri dataobject (nibabel (nib) or ants)

#     Returns:
#     -------
#     data: dict
#         Dictionnary of data using data ids as keys.
#     """
#     fs_dir = _retrieve_env("FS_DIR")

#     data = dict()

#     for name in args:
#         if name.lower() == "t1":
#             # T1 FreeSurfer conformed image
#             file = fs_dir/subject/"mri"/"T1.mgz"
#             data[name] = read_mri(file, mri_dtype=mri_dtype)

#         if name.lower() == "t1-orig":
#             # T1 original image
#             file =  fs_dir/subject/"mri"/"orig"/"001.mgz"
#             data[name] = read_mri(file, mri_dtype=mri_dtype)

#         if name.lower() == "t1-brain":
#             # skull stripped T1 MRI
#             file = fs_dir/subject/"mri"/"brain.mgz"
#             data[name] = read_mri(file, mri_dtype=mri_dtype)

#         if name.lower() == "white":
#             # White surfaced (boundary between white and gray matter)
#             fileL = fs_dir/subject/"surf"/"lh.white"
#             fileR = fs_dir/subject/"surf"/"rh.white"

#             whiteL = nib.freesurfer.io.read_geometry(fileL)
#             whiteR = nib.freesurfer.io.read_geometry(fileR)
#             data[name] = [whiteL, whiteR]

#         if name.lower() == "cras":
#             # Translation part of the Talairach transform
#             file = fs_dir/subject/"mri"/"transforms"/"talairach.lta"
#             data[name] = read_talairach(file)[:3,3]

#     return data

# def read_mindmap(subject, *args, sample=None):
#     """
#     Read data inside Mindmap directory.

#     Parameters
#     ----------
#     subject: str
#         Subjecte name.

#     *args:
#         Data ids:
#         - parc    : Parcellation
#         - ctx     : Cortical mask
#         - sc      : Structural Connectome
#         - cst     : Connectome Spectrum Transform
#         - eeg     : Preprocessed EEG
#         - trans   : Transformation for coregistering EEG
#         - src     : Source space (after forward solution)
#         - src_raw : Source space (before forward solution)
#         - bem     : Head conductor model
#         - bem-sim : Head conductor model to generate simulations
#         - fwd     : Forward operator
#         - cset*   : CSET solutions
#         - sloreta*: sLORETA solutions
#         - wCSET*  : Weighted CSET solutions
#         - eNET*   : Elastic Net solutions
#         The * indicates the lambda parameter applied.

#     Returns
#     -------
#     data: dict
#         Dictionnary pf data using data ids as keys.
#     """
#     mm_dir = _retrieve_env("MM_DIR")

#     data = dict()
    
#     for name in args:
#         if name.lower() == "parc":
#             file = mm_dir/subject/"parc"/f"{subject}_multiConn4-dseg.nii.gz"
#             clut = mm_dir/subject/"parc"/f"{subject}_multiConn4-dseg.txt"
#             data[name] = mm.utils.Parcellation(file, clut)

#         if name.lower() == "ctx":
#             d    = read_mindmap(subject, "parc")
#             clut = d["parc"].get_clut()
#             ctx  = clut[clut["Label name"].str.startswith("ctx")]
#             data[name] = d["parc"].create_mask(ctx["#No."].values)

#         if name.lower() == "sc":
#             file1 = mm_dir/subject/"sc"/f"{subject}_sc-tracto.npy"
#             file2 = mm_dir/subject/"sc"/f"{subject}_sc-template.npy"

#             if file1.exists():
#                 data[name] = np.load(file1)
#             else:
#                 data[name] = np.load(file2)

        


#         if name.lower() == "cst":
#             file = mm_dir/subject/"sc"/f"{subject}_cst.npy"
#             data[name] = np.load(file, allow_pickle=True)

#         if name.lower() == "eeg":
#             if sample is None: raise ValueError("Require sample name to read EEG.")
#             file = mm_dir/subject/"eeg"/f"{subject}-{sample}_ave.fif"
#             data[name] = mne.Evoked(file, verbose="warning")

#         if name.lower() == "trans":
#             file = mm_dir/subject/"eeg"/f"{subject}_eeg-trans.fif"
#             data[name] = mne.read_trans(file)

#         if name.lower() == "src":
#             file = mm_dir/subject/"fwd"/f"{subject}_fwd.fif"
#             if file.exists():
#                 fwd = mne.read_forward_solution(file, verbose="warning")
#                 data[name] = fwd["src"]
#             else:
#                 print("\033[93mCould not find forward solution, reading src_raw instead.\033[0m")  # src can change when generating forward solution 
#                 d = read_mindmap(subject, "src-raw")
#                 data[name] = d["src-raw"]

#         if name.lower() == "src-raw":
#             file = mm_dir/subject/"src"/f"{subject}_src.fif"
#             data[name] = mne.read_source_spaces(file, verbose="warning")

#         if name.lower() == "hmodel":
#             file = mm_dir/subject/"fwd"/f"{subject}_bem.fif"

#             if file.exists():
#                 d = read_mindmap(subject, "bem")
#                 data[name] = d["bem"]
#             else:
#                 d = read_mindmap(subject, "fem")
#                 data[name] = d["fem"]

#         if name.lower() == "hmodel-sim":
#             file = mm_dir/subject/"fwd"/f"{subject}_sim_bem.fif"

#             if file.exists():
#                 d = read_mindmap(subject, "bem-sim")
#                 data[name] = d["bem-sim"]
#             else:
#                 d = read_mindmap(subject, "fem-sim")
#                 data[name] = d["fem-sim"]

#         if name.lower() == "bem":
#             file = mm_dir/subject/"fwd"/f"{subject}_bem.fif"
#             data[name] = mne.read_bem_solution(file, verbose="warning")

#         if name.lower() == "bem-sim":
#             file = mm_dir/subject/"fwd"/f"{subject}_sim_bem.fif"
#             data[name] = mne.read_bem_solution(file, verbose="warning")

#         if name.lower() == "fem":
#             raise NotImplementedError("FEM models not available yet")

#         if name.lower() == "fwd":
#             file = mm_dir/subject/"fwd"/f"{subject}_fwd.fif"
#             data[name] = mne.read_forward_solution(file, verbose="warning")

#         if name.lower() == "fwd-sim":
#             file = mm_dir/subject/"fwd"/f"{subject}_sim_fwd.fif"
#             data[name] = mne.read_forward_solution(file, verbose="warning")
         
#         if name.lower().startswith("cset"):
#             file = mm_dir/subject/"inv"/f"{subject}-{sample}_{name}"
#             data[name] = mne.read_source_estimate(file)
            
#         if name.lower().startswith("sloreta"):
#             file = mm_dir/subject/"inv"/f"{subject}-{sample}_{name}"
#             data[name] = mne.read_source_estimate(file)

#         if name.lower().startswith("wcset"):
#             pass 
#         if name.lower().startswith("eNET"):
#             pass 
        
#     return data


# ##########################################################
# ####################                  ####################
# #################### DATASET READERS  ####################
# ####################                  ####################
# ##########################################################


# ##########################################################
# ####################                  ####################
# ####################   ATLAS READERS  ####################
# ####################                  ####################
# ##########################################################





# def _read_standards_spike(*args):

#     data = dict()

#     for name in args:
#         if name.lower() == "conn":
#             # Connectome atlas (h5 format)
#             file = _retrieve_env("CONN")
#             data[name] = h5py.File(file)

#         if name.lower() == "ctx":
#             # Cortical atlas (lausanne 2018 file paths)
#             data[name] = _retrieve_env("LAUSL", "LAUSR")

#         if name.lower() == "thal":
#             # Thalamus Nuclei atlas (ants)
#             file = _retrieve_env("THAL")
#             data[name] = ants.image_read(str(file))

#         if name.lower() == "thal_clut":
#             # Thalamus Nuclei color look up table (file path)
#             data[name] = _retrieve_env("THAL_CLUT")

#         if name.lower() == "mni":
#             # MNI template (MNI152 NLIN 2009) (ants)
#             file = _retrieve_env("MNI")
#             data[name] = ants.image_read(str(file)) 

#     return data


# def _read_standards_vepcon(*args):

#     data = dict() 

#     for name in args:
#         if name.lower() == "ctx":
#             # Cortical atlas (lausanne 2018 file paths)
#             data[name] = _retrieve_env("LAUSL", "LAUSR")

#         if name.lower() == "thal":
#             # Thalamus Nuclei atlas (ants)
#             file = _retrieve_env("THAL")
#             data[name] = ants.image_read(str(file))

#         if name.lower() == "thal_clut":
#             # Thalamus Nuclei color look up table (file path)
#             data[name] = _retrieve_env("THAL_CLUT")

#         if name.lower() == "mni":
#             # MNI template (MNI152 NLIN 2009) (ants)
#             file = _retrieve_env("MNI")
#             data[name] = ants.image_read(str(file)) 

#     return data
