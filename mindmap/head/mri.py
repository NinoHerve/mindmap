"""
Module Name: mri.py
Description: Functions to manage mri volumes.

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

def extract_voxel_volume(affine):
    """
    Computes voxel volume in [mm³] from the affine.
    """
    # defines borders of voxel (parallelepiped)
    v1 = affine[:3, 0]
    v2 = affine[:3, 1]
    v3 = affine[:3, 2]

    # volume of a parallelepiped
    volume = np.abs(np.cross(v1, v2) @ v3)

    return volume