"""
Module Name: anatomy/utils.py
Description: Usefull functions for the anatomy modules.

This module provides functions to manage anatomical data.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import pandas as pd

import nibabel as nib


def align_freesurfer2subject(fs_points: np.ndarray, ras2tkras: np.ndarray):
    """
    Map points from freesurfer space (tkras) to T1 ras space.

    Parameters
    ----------
    fs_points: np.ndarray (n,3)
        Points in the freesurfer space in [m]. Carefull ! Points from MNE objects
        are often in [m]. Make the scaling (*1000) before using this alignment.
    ras2tkras: np.ndarray (4,4)
        Affine transform (T1 ras [mm] -> tkras [mm])

    Returns
    -------
    new_points: np.ndarray (n,3)
        Points mapped to the tkras space (freesurfer space) in [mm]
    """

    return nib.affines.apply_affine(np.linalg.inv(ras2tkras), fs_points)


def align_diffusion2subject(dwi_points, t12diff):
    """
    Map points from diffusion space to T1 ras space.

    Parameters
    ----------
    dwi_points: np.ndarray (n,3)
        Points in the diffusion space in [mm].
    
    t12diff: np.ndarray (4,4)
        Affine transform (diffusion ras [mm] -> T1 ras [mm])

    Returns
    -------
    new_points: np.ndarray (n,3)
        Points mapped to the T1 ras space in [mm]
    """
    return nib.affines.apply_affine(t12diff, dwi_points)


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