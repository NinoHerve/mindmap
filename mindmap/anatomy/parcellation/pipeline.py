"""
Module Name: anatomy/parcellation/pipeline.py
Description: Pipelines to run parcellations.

This module provides pipeline to run parcellations.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import os
import subprocess
from pathlib import Path

import pandas as pd

import nibabel as nib
import ants

from mindmap.core.io import read_config
from mindmap.io.utils import build_fs_path
from mindmap.anatomy.mri.io import ants_to_nibabel, read_mri
from .io import read_clut
from .base import Parcellation
from .processing import prob2dseg


def parcellate_lausanne2018(sub:str, fconfig:str, margin:float = 0, verbose:int = 1):
    """
    Register the Lausanne 2018 atlas to subject. Output in FreeSurfer space.

    Parameters:
    ----------
    sub: str
        Subject.

    fconfig: str
        Configuration file.

    margin: float
        Margin of gray matter area onto white matter area.

    verbose: int
        Verbosity level (0 or 1).

    Returns:
    -------
    laus2018: mm.Parcellation
        Lausanne2018 Parcellation in FreeSurfer space.
    """
    if verbose == 1:
        print(f"\033[96mSegmenting subject {sub} according to the Lausanne 2018 atlas ...\033[0m")

    # retrieve data
    if verbose == 1:
        print("\033[96mLoading data ...\033[0m")

    # mindmap configuration
    config = read_config(fconfig)
    if "DIR_FS" not in config.keys():
        raise KeyError("Missing FreeSurfer directory ('DIR_FS') in mindmap configuration file.")
    if "FS_HOME" not in config.keys():
        raise KeyError("Missing FreeSurfer home directory ('FS_HOME') in mindmap configuration file.")
    if "REF_LAUSL" not in config.keys():
        raise KeyError("Missing Lausanne 2018 left annotation ('REF_LAUSL') in mindmap configuration file.")
    if "REF_LAUSR" not in config.keys():
        raise KeyError("Missing Lausanne 2018 right annotation ('REF_LAUSR') in mindmap configuration file.")

    # set freesurfer configuration
    os.environ["SUBJECTS_DIR"]    = str(config["DIR_FS"])
    os.environ["FREESURFER_HOME"] = str(config["FS_HOME"])

    sub_id = str(sub).replace("sub-", "")
    sub_id = f"sub-{sub_id}"

    # temporary files
    annot_l  = config["DIR_FS"]/sub_id/"label"/"lh.my-atlas.annot"
    annot_r  = config["DIR_FS"]/sub_id/"label"/"rh.my-atlas.annot"
    dparc_fs = config["DIR_FS"]/sub_id/"mri"/"my-atlas_space-fs_dparc.nii.gz"

    # Registration of surface-based segmentation
    if verbose == 1:
        print("\033[96mRegistering surface-based segmentation ...\033[0m")

    try:
        subprocess.run(
            [
                "mri_surf2surf",
                "--srcsubject", "fsaverage",
                "--trgsubject", sub_id,
                "--hemi", "lh",
                "--sval-annot", str(config["REF_LAUSL"]),
                "--tval", str(annot_l),
            ],
            stdout             = subprocess.PIPE,
            universal_newlines = True,
            check              = True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    try:
        subprocess.run(
            [
                "mri_surf2surf",
                "--srcsubject", "fsaverage",
                "--trgsubject", sub_id,
                "--hemi", "rh",
                "--sval-annot", str(config["REF_LAUSR"]),
                "--tval", str(annot_r),
            ],
            stdout             = subprocess.PIPE,
            universal_newlines = True,
            check              = True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    # Expanding surface-based segmentation to volumetric segmentation
    if verbose == 1:
        print("\033[96mExpanding surface-based segmentation to volumetric segmentation ...\033[0m")

    try:
        subprocess.run(
            [
                "mri_aparc2aseg",
                "--s", sub_id,
                "--annot", "my-atlas",
                "--wmparc-dmax", str(margin),
                "--labelwm",
                "--hypo-as-wm",
                "--new-ribbon",
                "--o", str(dparc_fs),
            ],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check = True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    # load data
    volume  = nib.load(dparc_fs)
    fs_clut = read_clut(config["FS_HOME"]/"FreeSurferColorLUT.txt")     # FreeSurfer color LUT
    lh_clut = read_clut(annot_l)                                        # Left annotation color LUT
    rh_clut = read_clut(annot_r)                                        # Right annotation color LUT

    # correct clut
    if verbose == 1:
        print("\033[96mFixing FreeSurfer color LUT ...\033[0m")
    laus2018_clut = fs_clut[(fs_clut["#No."]<1000) | (fs_clut["#No."]>4999)]     # Keep non-cortical regions

    gm_lh_clut, wm_lh_clut = lh_clut.copy(), lh_clut.copy()
    gm_rh_clut, wm_rh_clut = rh_clut.copy(), rh_clut.copy()

    gm_lh_clut["#No."] += 1000          # 1000-1999
    gm_rh_clut["#No."] += 2000          # 2000-2999
    wm_lh_clut["#No."] += 3000          # 3000-3999
    wm_rh_clut["#No."] += 4000          # 4000-4999

    laus2018_clut = pd.concat([laus2018_clut, gm_lh_clut, gm_rh_clut, wm_lh_clut, wm_lh_clut])
    laus2018_clut = laus2018_clut.sort_values(by=["#No."])

    laus2018 = Parcellation(volume, laus2018_clut)
    laus2018.change_ids([(3001, 3999), (4001, 4999)], lambda x: x-2000)

    # delete temporary files
    if verbose == 1:
        print("\033[96mDeleting Temporary files ...\033[0m")
    annot_l.unlink()
    annot_r.unlink()
    dparc_fs.unlink()

    if verbose == 1:
        print(f"\033[92mLausanne Parcellation of subject {sub} finished\033[0m")

    return laus2018


def parcellate_thal(sub, fconfig, thresh=0.05, verbose=1):
    """
    Register the Thalamic atlas to the subject.

    Parameters:
    ----------
    sub: str
        Subject id.

    fconfig: str
        Mindmap configuration file.

    thresh: float
        Threshold to convert probabilistic atlas to deterministic atlas.
    
    verbose: int
        Verbosity level.

    Returns:
    -------
    thal: mm.Parcellation
        Thalamic parcellation in FreeSurfer space.
    """
    config = read_config(fconfig)

    if not "DIR_FS" in config.keys():
        raise KeyError("Missing FreeSurfer directory ('DIR_FS') in mindmap configuration file.")
    if not "REF_MNI" in config.keys():
        raise KeyError("Missing MNI template ('REF_MNI') in mindmap configuration file.")
    if not "REF_THAL" in config.keys():
        raise KeyError("Missing Thalamic atlas ('REF_THAL') in mindmap configuration file.")

    if verbose == 1:
        print(f"\033[96mSegmenting {sub} according to the thalamic atlas ...\033[0m")

    # retrieve data
    if verbose == 1:
        print("\033[96mLoading data ...\033[0m")

    t1_file = build_fs_path(config["DIR_FS"], sub, "t1.mgz")
    t1 = read_mri(t1_file, mri_dtype="ants")
    mni = read_mri(config["REF_MNI"], mri_dtype="ants")
    thal = read_mri(config["REF_THAL"], mri_dtype="ants")

    clut_file = config["REF_THAL"].parent/"Thalamic_Nuclei-ColorLUT.txt"
    thal_clut = read_clut(clut_file)

    if verbose == 1:
        print("\033[96mANTS registration of MNI space ...\033[0m")

    # registration transform
    mytx = ants.registration(
        fixed = mni,
        moving = t1,
        type_of_transform = "SyN",
    )

    # registration 
    volume = ants.apply_transforms(
        fixed = t1,
        moving = thal,  ### Needs to be ants
        transformlist = mytx["invtransforms"],
        imagetype = 3,
        interpolator = "linear",
    )

    volume = ants_to_nibabel(volume)
    thal   = Parcellation(volume, thal_clut)
    thal   = prob2dseg(thal, thresh=thresh)

    if verbose == 1:
        print("\033[92mThalamic Parcellation finished\033[0m")

    return thal

def parcellate_multiconn(
    sub : str,
    fconfig: str | Path,
    margin:float = 2,
    thresh:float = 0.05,
    verbose: int = 1
) -> Parcellation:
    """
    Generate the multiconn parcellation for subject.

    Parameters:
    ----------
    sub: str
        Subject ID

    fconfig: str | Path
        Mindmap configuration file name.

    t1_file: str | Path
        T1 file name.

    margin: float
        Margin for Lausanne 2018 atlas

    thresh: float
        Probability threshold for Thalamic atlas.

    Returns:
    -------
    multi_conn: mm.Parcellation
        Multi conn parcellation in FreeSurfer space.
    """
    # ***** Generate Lausanne2018 and Thalamic parcellations *****

    # Lausanne Parcellation
    laus2018 = parcellate_lausanne2018(sub, fconfig, margin=margin, verbose=verbose)

    # Thal Parcellation
    thal = parcellate_thal(sub, fconfig, thresh=thresh, verbose=verbose)

    # ***** Fuse Lausanne2018 and Thalamic parcellations *****

    # Extract and rename parts of interest from the Lausanne 2018 parcellation.
    ctx_lh = laus2018.retain([(1001, 1999), (3001, 3999)])
    ctx_lh.change_labels(prefix="ctx-lh-")

    ctx_rh = laus2018.retain([(2001, 2999), (4001, 4999)])
    ctx_rh.change_labels(prefix="ctx-rh-")

    subc_lh = laus2018.retain([11, 12, 13, 26, 18, 17])
    subc_lh.change_labels(replace={"Left-":"", "Accumbens-area":"accumbens_area"}, prefix="subc-lh-", lower=True)

    subc_rh = laus2018.retain([50, 51, 52, 58, 54, 53])
    subc_rh.change_labels(replace={"Right-":"", "Accumbens-area":"accumbens_area"}, prefix="subc-rh-", lower=True)

    # Extract and rename parts of interest from the thalamic parcellation.
    thal_lh = thal.retain([(8, 14)])

    thal_rh = thal.retain([(1,7)])

    brain_stem = laus2018.retain([16])
    brain_stem.change_labels(replace={"-":"_"})

    # join parts of interest
    multi_conn = Parcellation()
    multi_conn = multi_conn.join(ctx_rh)
    multi_conn = multi_conn.join(thal_rh)
    multi_conn = multi_conn.join(subc_rh)
    multi_conn = multi_conn.join(ctx_lh)
    multi_conn = multi_conn.join(thal_lh)
    multi_conn = multi_conn.join(subc_lh)
    multi_conn = multi_conn.join(brain_stem)

    if verbose == 1:
        print(f"\033[92mMultiConn parcellation of subject {sub} finished\033[0m")

    return multi_conn
