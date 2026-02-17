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
    spacing = image.spacing
    origin  = image.origin
    direction = image.direction

    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing) @ direction
    affine[:3, 3] = origin

    affine[:2, :] *= -1    # flip y and z axis to be in nibabel convention

    return affine


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

