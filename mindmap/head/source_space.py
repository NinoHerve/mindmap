"""
Module Name: source_space.py
Description: Functions to manage source spaces.

This module provides functions to manage source spaces that are not implemented in MNE. For 
example, use `.get_src_position(mne.SourceSpaces)` to retrieve the (x, y, z) coordinates of 
dipoles.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import random

import numpy as np
import pandas as pd
import scipy as sp
import sklearn
import nibabel as nib
import mne

if TYPE_CHECKING:
    from mindmap.head.parcellation import Parcellation

# --------
# Getters
# --------

def get_dipole_positions(source_space:mne.SourceSpaces) -> np.ndarray:
    "Get source positions."
    positions = []

    for src in source_space:
        positions.append(src["rr"][src["vertno"]])

    return np.concatenate(positions, axis=0)

def get_dipole_directions(source_space):
    "Get source directions."
    directions = []

    for src in source_space:
        directions.append(src["nn"][src["vertno"]])

    return np.concatenate(directions, axis=0)

def get_geodist(source_space:mne.SourceSpaces) -> sp.sparse.csr_array:
    """
    Get matrix showing geodesic distances between sources. Left and right hemispheres are seperated 
    surfaces, hence distances between a left and right source is infinite.
    """

    geodist = []
    for src in source_space:
        vertno = src["vertno"]
        geodist.append(src["dist"][vertno][:, vertno])

    return sp.sparse.block_diag(geodist, format="csr")

def get_ndipole(source_space: mne.SourceSpaces) -> int:
    """Get number of dipoles in source space."""
    return source_space[0]["nuse"] + source_space[1]["nuse"]

# --------
# Makers
# --------

def make_stc(
    source_space: mne.SourceSpaces,
    activity: np.ndarray,
    tmin: float,
    tstep: float,
) -> mne.SourceEstimate:
    """
    Make mne.SourceEstimate object

    Parameter
    ---------
    activity: np.ndarray (N,T)
        Brain activity. N -> number of sources, T -> number of time points

    src: mne.SourceSpaces
        Source space.

    tmin: float
        Starting time in [s]

    tstep: float
        Time step in [s]

    Return
    ------
    stc: mne.SourceEstimate
        Brain activity in MNE format.
    """
    vertices   = [source_space[0]["vertno"], source_space[1]["vertno"]]

    stc = mne.SourceEstimate(
        activity,
        vertices,
        tmin=tmin,
        tstep=tstep,
    )

    return stc



# ---------
# Actions
# ---------


def filter_dipoles_by_nifti_mask(
    source_space   : mne.SourceSpaces,
    mask           : nib.Nifti1Image,
    tkras2ras      : np.ndarray,
) -> mne.SourceSpaces:
    """
    Mask sources not in mask (usually gray matter mask).

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    mask: nib.NIfTi1Image
        Binary mask. If a dipole is on a voxel with value 0 it will be masked ('inuse'=False).

    tkras2ras: np.ndarray (4,4)
        Affine transform to convert volume from talairach conform space to individual space.

    Returns
    -------
    source_space: mne.SourceSpaces
        Source space with updated 'inuse', 'use' and 'vertno' attributes.
    """
    source_space = source_space.copy()

    for src in source_space:
        # select used dipoles
        inuse = src["inuse"].astype(bool)
        pos   = src["rr"][inuse]*1000                     # dipole positions in talairach space [mm]
        pos   = align_freesurfer2subject(pos, tkras2ras)

        # place sources in voxel space
        inv_affine = np.linalg.inv(mask.affine)
        pos_vox    = mne.transforms.apply_trans(inv_affine, pos)
        pos_vox    = np.round(pos_vox).astype(int)               # source positions in voxel space

        # keep only sources inside GM mask
        volume = mask.get_fdata()
        keep   = volume[pos_vox[:,0], pos_vox[:,1], pos_vox[:,2]].astype(bool)

        new_inuse = np.zeros_like(inuse, dtype=bool)
        new_inuse[inuse] = keep

        # save new variables in source space
        nuse   = np.sum(new_inuse)                              # number of used sources
        vertno = np.nonzero(new_inuse)[0]                       # vertex number of used sources

        src["inuse"]  = new_inuse
        src["nuse"]   = nuse
        src["vertno"] = vertno
        src["pinfo"]  = [patch for i, patch in enumerate(src["pinfo"]) if keep[i]]

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
        vertno        = list(set(src["vertno"]) & set(vertices))        # vertices that were sampled and are cortical
        inuse         = np.zeros_like(src["inuse"], dtype=bool)
        inuse[vertno] = True
        keep          = inuse[src["inuse"].astype(bool)]     # dipole list where 1 is keep dipole and 0 is removed dipole

        # change object attributes
        src["inuse"]  = inuse
        src["nuse"]   = np.sum(inuse)
        src["vertno"] = vertno
        src["pinfo"]  = [patch for i, patch in enumerate(src["pinfo"]) if keep[i]]

    return source_space


def add_subcortical_dipoles(
    source_space : mne.SourceSpaces,
    parcellation : Parcellation,
    cras         : np.ndarray = np.zeros(3)
) -> mne.SourceSpaces:
    """
    Add subcortical dipoles to source space.

    Parameters
    ----------
    source: mne.SourceSpaces
        Source space (FreeSurfer space).

    parcellation: mm.Parcellation 
        Parcellation (RAS space).

    cras: 
        Translation FreeSurfer -> RAS.

    Returns
    -------
        source: mne.SourceSpaces
            Source space with subcortical dipoles. 
    """
    if np.all(cras == np.zeros(3)):
        print("\033[93mNo cras applied.\033[0m")

    # transpose dipole position to RAS space
    pos = get_dipole_positions(source_space) * 1000 + cras

    # label dipoles by brain region
    labels = parcellation.label_points(pos, id=False)

    # compute dipole density per brain region
    regions, dip_per_region = np.unique(labels, return_counts=True)
    parcellation.compute_volumes()
    vol_per_region          = parcellation.get_volumes(regions)
    density_per_region      = dip_per_region / vol_per_region
    density                 = density_per_region.mean()

    # labels of none-cortical regions
    mask   = ~parcellation.info["Region name"].str.contains("ctx", case=False, regex=True)
    subctx = parcellation.info.loc[mask, "Region ID"].values

    # add dipoles in every missing region. Place them using k-means
    volume         = parcellation.volume
    vol_per_region = parcellation.get_volumes(subctx)
    points = []
    for i, region in enumerate(subctx):
        voxels       = np.argwhere(volume == region)
        voxel_coords = (nib.affines.apply_affine(parcellation.affine, voxels) - cras)/1000
        k            = max(int(density*vol_per_region[i]), 1)
        kmeans       = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(voxel_coords)
        points.append(kmeans.cluster_centers_)

    pos = {
        "rr": np.concat(points),
        "nn": np.ones((len(points), 3)),
    }

    subctx_src = mne.setup_volume_source_space(pos=pos, verbose="warning")

    return source_space + subctx_src


def random_region(
        source_space : mne.SourceSpaces,
        parc         : Parcellation,
        tkras2ras    : np.ndarray,
        n            : int = 1
        ) -> str:
    """
    Select a random region amongst the ones that have dipoles in it.
    """

    dipoles = get_dipole_positions(source_space)*1000
    dipoles = align_freesurfer2subject(dipoles, tkras2ras)

    labels = parc.label_points(dipoles)
    labels = np.unique(labels)

    random.shuffle(labels)

    return labels[:n]


# ------------------
# Dipole alignments
# ------------------

def align_freesurfer2subject(fs_points, tkras2ras):
    """
    Map points from freesurfer space (tkras) to T1 ras space.

    Parameters
    ----------
    fs_points: np.ndarray (n,3)
        Points in the freesurfer space in [m]. Carefull ! Points from MNE objects
        are often in [m]. Make the scaling (*1000) before using this alignment.
    tkras2ras: np.ndarray (4,4)
        Affine transform (tkras [mm] -> T1 ras [mm])

    Returns
    -------
    new_points: np.ndarray (n,3)
        Points mapped to the T1 ras space in [mm]
    """

    return nib.affines.apply_affine(tkras2ras, fs_points)
