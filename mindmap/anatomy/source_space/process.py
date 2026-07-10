"""
Module Name: source_space/processing.py
Description: Functions to process source spaces.

This module provides functions to process source spaces in ways MNE did not foresee.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import pandas as pd

import mne
import nibabel as nib

from mindmap.core.process import sample_volume_at_points


def sample_nifti_at_source_space(
    source_space: mne.SourceSpaces,
    mask: nib.Nifti1Image,
    ras2tkras: np.ndarray = None,
) -> np.ndarray:
    """
    Sample a volumetric image at every in-use source-space vertex.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space (space-freesurfer). Only 'inuse' dipoles are sampled,
        matching the vertex order of any mne.SourceEstimate built from it.

    mask: nib.Nifti1Image
        Volume to sample (e.g. a binary mask).

    ras2tkras: np.ndarray (4,4)
        Affine transform from subject/T1 scanner-RAS [mm] to FreeSurfer
        surface-RAS (tkRAS) [mm]. Required whenever `source_space` and `mask`
        don't already share the same world coordinate frame.

    Returns
    -------
    np.ndarray of shape (n_inuse_vertices,)
        Values of `mask` at each vertex, concatenated hemisphere by
        hemisphere in the same order as `source_space`.
    """
    values = []

    for src in source_space:
        inuse = src["inuse"].astype(bool)
        pos = src["rr"][inuse]*1000                     # dipole positions in talairach space [mm]

        if ras2tkras is not None:
            pos = nib.affines.apply_affine(np.linalg.inv(ras2tkras), pos)

        # place sources in voxel space
        pos_vox = nib.affines.apply_affine(np.linalg.inv(mask.affine), pos)

        values.append(sample_volume_at_points(mask.get_fdata(), pos_vox))

    return np.concatenate(values)


def filter_dipoles_by_nifti_mask(
    source_space: mne.SourceSpaces,
    mask: nib.Nifti1Image,
    ras2tkras: np.ndarray = None,
) -> mne.SourceSpaces:
    """
    Mask sources not in mask (usually gray matter mask).

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space (space-freesurfer).

    mask: nib.NIfTi1Image
        Binary mask. If a dipole is on a voxel with value 0 it will be masked ('inuse'=False).

    ras2tkras: np.ndarray (4,4)
        Affine transform to convert volume from talairach conform space to individual space.

    Returns
    -------
    source_space: mne.SourceSpaces
        Source space with updated 'inuse', 'use' and 'vertno' attributes.
    """
    source_space = source_space.copy()
    values = sample_nifti_at_source_space(source_space, mask, ras2tkras)

    offset = 0
    for src in source_space:
        inuse = src["inuse"].astype(bool)
        n = np.sum(inuse)
        keep = values[offset:offset + n].astype(bool)
        offset += n

        new_inuse = np.zeros_like(inuse, dtype=bool)
        new_inuse[inuse] = keep

        # save new variables in source space
        nuse = np.sum(new_inuse)                              # number of used sources
        vertno = np.nonzero(new_inuse)[0]                       # vertex number of used sources

        src["inuse"] = new_inuse
        src["nuse"] = nuse
        src["vertno"] = vertno
        src["pinfo"] = [patch for i, patch in enumerate(src["pinfo"]) if keep[i]]

    return source_space


def filter_dipoles_by_id_list(
    source_space : mne.SourceSpaces,
    id_list      : pd.DataFrame,
) -> mne.SourceSpaces:
    """
    Mask Sources not in id_list.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    id_list: pd.DataFrame
        DataFrame with columns vertex id and hemisphere.

    Returns
    -------
    source_space: mne.SourceSpaces
        Source space with updated 'inuse', 'use' and 'vertno' attributes.
    """

    if not all(col in id_list for col in ["vertex id", "hemisphere"]):
        raise ValueError("'vertex id' or 'hemisphere' is missing from id_list.")

    vertex_ids = [
        id_list[id_list["hemisphere"]=="L"]["vertex id"].values,
        id_list[id_list["hemisphere"]=="R"]["vertex id"].values,
    ]

    for src, vertices in zip(source_space, vertex_ids):
        # new attributes
        vertno = list(set(src["vertno"]) & set(vertices))        # vertices that were sampled and are cortical
        inuse = np.zeros_like(src["inuse"], dtype=bool)
        inuse[vertno] = True
        keep = inuse[src["inuse"].astype(bool)]     # dipole list where 1 is keep dipole and 0 is removed dipole

        # change object attributes
        src["inuse"] = inuse
        src["nuse"] = np.sum(inuse)
        src["vertno"] = vertno
        src["pinfo"] = [patch for i, patch in enumerate(src["pinfo"]) if keep[i]]

    return source_space