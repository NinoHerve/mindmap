"""
Module Name: anatomy/mri/processing.py
Description: Functions to process MRI data.

This module provides functions to process MRI data.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import nibabel as nib
import ants

from .io import get_affine
from ...geometry.io import vol_to_surf

def mri_to_surf(mri: nib.Nifti1Image | ants.ANTsImage):
    """
    Generate mesh from an MRI mask.

    Parameters
    ----------
    volume: nib.Nifti1Image, ants.ANTsImage
        Volume mask.

    Returns
    -------
    mesh: pv.PolyData
        Surface mesh.
    """

    if isinstance(mri, ants.ANTsImage):
        affine = get_affine(mri)
        volume = mri.numpy()
    else:
        affine = mri.affine
        volume = mri.get_fdata()

    mesh = vol_to_surf(volume, remesh=True, compute_normals=False)

    mesh.points = nib.affines.apply_affine(affine, mesh.points)
    mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh
