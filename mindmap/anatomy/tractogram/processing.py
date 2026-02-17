"""
Module Name: sensor.py
Description: Functions to process tractograms.

This module provides functions to process tractograms.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import nibabel as nib

from ...anatomy.utils import align_diffusion2subject

def tract_endpoints(tracts: nib.streamlines.TrkFile) -> list[list, list]:
    """Get endpoints of every tract as (x,y,z) coordinates."""
    voxel_size = tracts.header["voxel_sizes"].reshape(1,-1)

    endpoints1 = np.array([stream[0] for stream in tracts.streamlines]) / voxel_size
    endpoints2 = np.array([stream[-1] for stream in tracts.streamlines]) / voxel_size

    return endpoints1, endpoints2


def tract_lengths(tracts: nib.streamlines.TrkFile) -> list:
    """Get length for every tract"""
    lengths = np.array([streamline_length(stream) for stream in tracts.streamlines])
    return lengths


def streamline_length(streamline: np.ndarray) -> float:
    """
    Compute the length of a streamline.  

    Parameters
    ----------
    streamline: np.ndarray (n,3)
        A streamline consists of a list of n voxel coordinates  
        ([[x1, y1, z1], [x2, y2, z2], ...]) that define its path in voxel space.  
        The length is computed by summing the distances between consecutive voxels.

    Returns
    -------
    length: float
        Length in [mm]
    """
    diffs   = np.diff(streamline, axis=0)    # computes differences between consecutive elements
    lengths = np.linalg.norm(diffs, axis=1)

    return np.sum(lengths)


def prune_tracts(
        tractogram: nib.streamlines.Tractogram | nib.streamlines.TrkFile | nib.streamlines.TckFile, 
        mask_cortex: nib.Nifti1Image,
        t12diff: np.ndarray,
        return_pruned: bool,
        ) -> nib.streamlines.Tractogram:
    """
    Prune tracts.

    Parameters
    ----------
    tractogram: nib.streamlines.TrkFile | nib.streamlines.TckFile
        Tractogram.

    mask_cortex: nib.Nifti1Image
        Cortical mask. If the endpoints of a streamline lie within voxels labeled as 1, we keep t
        he tract; otherwise, we prune it.

    t12diff: np.ndarray
        Affine transform (diffusion ras -> T1 ras)

    return_pruned: bool
        Flag to return pruned tracts. For comparison.

    Returns
    -------
    pruned_tractogram: nib.streamlines.Tractogram
        Pruned tractogram.

    Note: add other important pruning (max length, min length ...)
    """
    starts, ends = tract_endpoints(tractogram)

    # map endpoints to T1-ras space
    starts = align_diffusion2subject(starts, t12diff)
    ends   = align_diffusion2subject(ends, t12diff)

    # map endpoints to T1-voxel space
    affine = np.linalg.inv(mask_cortex.affine)
    starts = np.round(nib.affines.apply_affine(affine, starts)).astype(int)
    ends   = np.round(nib.affines.apply_affine(affine, ends)).astype(int)

    # check endpoints inside cortex
    mask = mask_cortex.get_fdata()
    starts_in_cortex = mask[starts[:,0], starts[:,1], starts[:,2]].astype(bool)
    ends_in_cortex   = mask[ends[:,0], ends[:,1], ends[:,2]].astype(bool)

    # prune by keeping only streamline with endpoints inside cortex
    pruned_tractogram = nib.streamlines.Tractogram(
        streamlines     = tractogram.streamlines[starts_in_cortex*ends_in_cortex],
        affine_to_rasmm = tractogram.affine,
    )
    pruned_tractogram = nib.streamlines.TrkFile(
        tractogram = pruned_tractogram,
        header     = tractogram.header,
    )

    # return a tractogram containing only deleted tracts (sanity check)
    if return_pruned:
        deleted_tractogram = nib.streamlines.Tractogram(
            streamlines     = tractogram.streamlines[~(starts_in_cortex*ends_in_cortex)],
            affine_to_rasmm = tractogram.affine,
        )
        deleted_tractogram = nib.streamlines.TrkFile(
            tractogram = deleted_tractogram,
            header     = tractogram.header,
        )
        return pruned_tractogram, deleted_tractogram

    return pruned_tractogram



