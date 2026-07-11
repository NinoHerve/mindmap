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
from mindmap.geometry.processing import compute_vertex_areas
from .metric import average_dipole_spacing


def sample_nifti_at_points(
    points: np.ndarray,
    mask: nib.Nifti1Image,
    ras2tkras: np.ndarray = None,
) -> np.ndarray:
    """
    Sample a volumetric image at arbitrary world-space points.

    Factored out of `sample_nifti_at_source_space` so any point in tkRAS
    space -- not just source-space vertices -- can be tested against a
    mask (e.g. a single dipole position, to check whether it falls inside
    a resection).

    Parameters
    ----------
    points: np.ndarray of shape (n_points, 3)
        Points in FreeSurfer surface-RAS (tkRAS) [mm], e.g. source-space
        dipole positions (`rr`) scaled by 1000.

    mask: nib.Nifti1Image
        Volume to sample (e.g. a binary mask).

    ras2tkras: np.ndarray (4,4)
        Affine transform from subject/T1 scanner-RAS [mm] to FreeSurfer
        surface-RAS (tkRAS) [mm]. Required whenever `points` and `mask`
        don't already share the same world coordinate frame.

    Returns
    -------
    np.ndarray of shape (n_points,)
        Values of `mask` at each point.
    """
    pos = points

    if ras2tkras is not None:
        pos = nib.affines.apply_affine(np.linalg.inv(ras2tkras), pos)

    # place points in voxel space
    pos_vox = nib.affines.apply_affine(np.linalg.inv(mask.affine), pos)

    return sample_volume_at_points(mask.get_fdata(), pos_vox)


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

        values.append(sample_nifti_at_points(pos, mask, ras2tkras))

    return np.concatenate(values)


def nifti_mask_to_points(
    mask: nib.Nifti1Image,
    ras2tkras: np.ndarray = None,
    threshold: float = 0,
) -> np.ndarray:
    """
    Extract the world-space coordinates of every voxel in a thresholded
    mask.

    Inverse of the coordinate chain used by `sample_nifti_at_points`/
    `sample_nifti_at_source_space`: instead of sampling a mask at given
    points, this returns the points corresponding to every voxel above
    `threshold` -- e.g. to measure Euclidean distance from a dipole to the
    resection's own voxel geometry, rather than to a coarser vertex-sampled
    approximation of it.

    Parameters
    ----------
    mask: nib.Nifti1Image
        Volume to threshold (e.g. a resection mask).

    ras2tkras: np.ndarray (4,4)
        Affine transform from subject/T1 scanner-RAS [mm] to FreeSurfer
        surface-RAS (tkRAS) [mm]. Required whenever `mask` and the target
        source space don't already share the same world coordinate frame.

    threshold: float
        Voxels with value > threshold are included.

    Returns
    -------
    np.ndarray of shape (n_voxels, 3)
        Voxel coordinates in tkRAS [mm], the same frame as source-space
        dipole positions (`rr`) scaled by 1000.
    """
    voxel_ijk = np.argwhere(mask.get_fdata() > threshold)
    pos = nib.affines.apply_affine(mask.affine, voxel_ijk)

    if ras2tkras is not None:
        pos = nib.affines.apply_affine(ras2tkras, pos)

    return pos


def build_dipole_patch_area(
    source_space: mne.SourceSpaces,
) -> np.ndarray:
    """
    Physical surface area (mm^2) of each in-use dipole's patch.

    A decimated source space's dipoles don't cover equal cortical area --
    patches vary severalfold in physical size depending on local curvature
    and gray-matter density (confirmed empirically: the same top-30-vertex
    selection produced dense-mesh patches ranging from 18x to 43x their
    decimated vertex count). Raw dipole/vertex counts are therefore a biased
    proxy for extent, and not comparable across source spaces built at
    different resolutions.

    This reuses `build_dense_vertex_map`'s nearest-dipole assignment to sum
    each dipole's real per-vertex area -- computed directly from the source
    space's own full mesh (`rr`/`tris`) via `compute_vertex_areas`, rather
    than merely counting dense-mesh vertices -- giving true,
    resolution-independent physical area per dipole. Computing from the
    source space's own mesh (instead of e.g. a separately loaded `?h.area`
    file) guarantees the area matches whatever surface the source space was
    actually built on -- verified to agree with FreeSurfer's own `?h.area`
    to within floating-point rounding when the source space uses the white
    surface, but self-consistent even if that ever changes (e.g. to pial).

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space (space-freesurfer).

    Returns
    -------
    np.ndarray of shape (n_dipoles,)
        Physical patch area (mm^2) per in-use dipole, concatenated
        hemisphere by hemisphere in mne.SourceEstimate vertex order.
    """
    patch_area = []

    for src_hemi in source_space:
        area = compute_vertex_areas(src_hemi["rr"], src_hemi["tris"])
        vertex_map = build_dense_vertex_map(src_hemi)
        valid = vertex_map >= 0

        patch_area.append(np.bincount(
            vertex_map[valid],
            weights = area[valid],
            minlength = src_hemi["nuse"],
        ))

    return np.concatenate(patch_area)


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


def build_dense_vertex_map(src_hemi) -> np.ndarray:
    """
    Map every vertex of the full (non-decimated) FreeSurfer surface to its
    nearest in-use dipole.

    Uses MNE's own `src["nearest"]` (the geodesic-nearest decimated candidate
    for every dense-mesh vertex, computed when the source space is built with
    `add_dist=True`), restricted to candidates that survived into `vertno`
    (e.g. after gray-matter masking). Dense-mesh vertices whose nearest
    candidate was masked out are assigned -1.

    Parameters
    ----------
    src_hemi: dict
        One hemisphere's entry of an mne.SourceSpaces (e.g. `source_space[0]`).

    Returns
    -------
    np.ndarray of shape (n_dense_vertices,)
        For each dense-mesh vertex, the index (0..nuse-1, matching
        mne.SourceEstimate row order for this hemisphere) of its nearest
        in-use dipole, or -1 if its nearest candidate was masked out.
    """
    vertno = np.array(src_hemi["vertno"])
    nearest = src_hemi["nearest"].copy()

    # initialize all as masked
    vertex_map = np.full(nearest.shape[0], -1, dtype=int)

    valid = np.isin(nearest, vertno)
    vertno_to_dip = {v: i for i, v in enumerate(vertno)}

    # map only valid entries
    vertex_map[valid] = np.fromiter(
        (vertno_to_dip[v] for v in nearest[valid]),
        dtype = int,
        count = np.sum(valid),
    )

    return vertex_map


def build_dipole_adjacency(
    source_space: mne.SourceSpaces,
    dist: float = None,
):
    """
    Build a holes-free adjacency graph over in-use dipoles.

    MNE's default triangulation-based `mne.spatial_src_adjacency` connects
    dipoles via the decimated candidate mesh's triangles -- when gray-matter
    masking (or any other restriction) removes some candidates, triangles
    touching a removed vertex disappear, leaving "holes" that can incorrectly
    split one true cluster into several (confirmed empirically: MNE warns
    "tri-based adjacency will have holes" on every source space in this
    pipeline, since ~7% of candidates get masked out).

    Passing `dist` switches MNE to geodesic-distance-based adjacency instead,
    built from the pairwise distances between the dipoles that actually
    remain (available since source spaces here are built with
    `add_dist=True`) -- unaffected by which candidates got masked out.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space (space-freesurfer).

    dist: float, optional
        Maximum geodesic distance (in m) for two dipoles to be considered
        connected. Defaults to 1.5x the average nearest-neighbor geodesic
        dipole spacing for this source space.

    Returns
    -------
    scipy.sparse matrix of shape (n_dipoles, n_dipoles)
        Adjacency over in-use dipoles, in mne.SourceEstimate vertex order.
    """
    if dist is None:
        dist = 1.5 * average_dipole_spacing(source_space, distance_type="geodesic")

    return mne.spatial_src_adjacency(source_space, dist=dist, verbose="warning").tocsr()


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