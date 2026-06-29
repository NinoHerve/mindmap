"""
Module Name: mri/io.py
Description: Functions to extract or convert mri data.

This module provides functions to manage mri volumes. Most of the time, mri volumes are
stored in nibabel.NiftiImage or ants.ANTsImage. This module complements functions already 
available via nibabel and ants.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path
import subprocess

import numpy as np

import nibabel as nib
import ants
import pyvista as pv

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


def nibabel_to_ants(nib_img:nib.Nifti1Image) -> ants.core.ants_image.ANTsImage:
    """Convert nibabel object to ants object"""
    tmp = Path("./tmp.nii.gz")

    nib.save(nib_img, tmp)
    ants_img = ants.image_read(str(tmp))

    tmp.unlink()

    return ants_img

def ants_to_nibabel(ants_img:ants.core.ants_image.ANTsImage) -> nib.Nifti1Image:
    """Convert ants object to nibabel object"""
    tmp = Path("./tmp.nii.gz")

    ants.image_write(ants_img, str(tmp))
    nib_img = nib.load(tmp)
    nib_img = nib.Nifti1Image(nib_img.get_fdata(), nib_img.affine)          # so that data is on the RAM. Else it accesses data only when using .get_fdata() but file will be deleted by then

    tmp.unlink()

    return nib_img


def get_affine(image: ants.ANTsImage):
    """
    Return the image affine from an ANTsImage object. 
    """
    nib_img = ants_to_nibabel(image)

    return nib_img.affine


def nib_to_polydata(image:nib.Nifti1Image, trans:np.ndarray = np.eye(4)):
    """
    Convert nibabel to pyvista point cloud.

    Parameters
    ----------
    image: nib.Nifti1Image
        Nibabel volume image.
    
    trans: np.ndarray (3,) or (4,4)
        Transformation subject -> FreeSurfer ((3,) is a translation)

    Returns
    -------
    cloud_point: pv.PolyData
    """
    # voxel -> subject space
    voxels = np.argwhere(image.get_fdata()>0)
    points = nib.affines.apply_affine(image.affine, voxels)

    # subject -> FreeSurfer space
    if trans.shape == (3,): # translation
        points = (points - trans)/1000
    else:
        points = nib.affines.apply_affine(trans, points)/1000

    return pv.PolyData(points)

def load_fs_in_native_space(fs_subject_dir, volume_type="brain", cache_dir=None):
    """
    Transforms a FreeSurfer conformed volume back to native T1 space
    and returns it as a nibabel object.

    Parameters
    ----------
    fs_subject_dir: str
        Path to the specific subject's FreeSurfer folder.
    
    volume_type: str
        The name of the file in the mri/ folder without extension.

    cache_dir: str, optional
        If provided, saves the converted .nii.gz file here to skip conversion next time.
        If None, uses the mri/ folder as cache_dir.
    """
    mri_dir = Path(fs_subject_dir)/"mri"
    moving_volume = mri_dir/f"{volume_type}.mgz"
    target_volume = mri_dir/"rawavg.mgz"

    # Sanity checks
    if not moving_volume.exists():
        raise FileNotFoundError(f"Requested FreeSurfer volume not found: {moving_volume}")
    if not target_volume.exists():
        raise FileNotFoundError(f"Native target space 'rawavg.mgz' not found in {mri_dir}")

    # Setup output paths
    output_filename = f"{volume_type}_native.nii.gz"

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        output_path = cache_dir/output_filename
    else:
        output_path = mri_dir/output_filename

    # If cached file doesn't exist, compute the transformation
    if not output_path.exists():
        print(f"Transforming {volume_type}.mgz to native space...")

        # Construct the FreeSurfer command
        cmd = [
            "mri_vol2vol",
            "--mov", str(moving_volume),
            "--targ", str(target_volume),
            "--regheader",
            "--o", str(output_path),
        ]

        # Execute command silently
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Load into nibabel
    img = nib.load(output_path)

    return img

