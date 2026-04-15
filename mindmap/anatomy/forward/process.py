"""
Module Name: forward/process.py
Description: Functions to manage boundary element models.

This module provides additional functions that MNE-python does not implement to handle
BEM models.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path

import numpy as np
import pyvista as pv

import nibabel as nib
import ants
from mne.bem import _get_ico_surface

from mindmap.geometry.io import surf_to_polydata
from mindmap.geometry.processing import surf_correction, fit_surface_to_volume
from mindmap.anatomy.mri.io import get_affine, nibabel_to_ants
from mindmap.anatomy.parcellation.core import Parcellation
from mindmap.anatomy.parcellation.processing import create_mask

def correct_skull(
        subject     : str,
        offset      : float,
        fs_dir      : str | Path,
        save_arrows : bool = False,
) -> None:
    """
    Correct skull by displacing wrong vertices of inner skull back into the outer skull
    with an additional offset. 

    subject: str
        FreeSurfer subject name.

    offset: float
        Minimal distance between inner and outer skull.

    fs_dir: path-like
        FreeSurfer directory.

    save_arrows: bool
        If True then the vector field showing the displacement done.
    """

    # files
    fs_dir = Path(fs_dir)

    inner_fname = fs_dir/subject/"bem"/"inner_skull.surf"
    outer_fname = fs_dir/subject/"bem"/"outer_skull.surf"

    # load
    vertices, faces, metadata = nib.freesurfer.io.read_geometry(inner_fname, read_metadata=True)
    in_skull = surf_to_polydata(vertices, faces)

    vertices, faces = nib.freesurfer.io.read_geometry(outer_fname)
    out_skull = surf_to_polydata(vertices, faces)

    # apply correction
    outputs = surf_correction(
        in_skull,
        out_skull,
        offset = offset,
        return_arrows = save_arrows,
    )

    # rename old in_skull, save new in_skull as inner_skull.surf
    inner_fname.rename(fs_dir/subject/"bem"/"inner_skull_raw.surf")

    if save_arrows: 
        in_skull_fixed, arrows = outputs 
        arrows_fname = fs_dir/subject/"bem"/"inner_skull_changes.vtk"
        arrows.save(arrows_fname)

    else:
        in_skull_fixed = outputs 

    fname    = fs_dir/subject/"bem"/"inner_skull.surf"
    vertices = in_skull_fixed.points
    faces    = in_skull_fixed.faces

    # rewrite faces into right format
    idx   = np.arange(0, len(faces), step=4)
    faces = np.delete(faces, idx).reshape(-1, 3)

    nib.freesurfer.io.write_geometry(fname, vertices, faces, volume_info=metadata)


def fit_surface_to_mri_mask(
    mask: nib.Nifti1Image | ants.ANTsImage,
    **kwargs
):
    """
    Fit 5th order icosahedron sphere to MRI mask.

    Parameters
    ----------
    mask: nib.NifTi1Image | ants.ANTSImage
        MRI mask.

    tkras2ras: np.ndarray of shape (4,4)
        Voxel space -> Surface RAS transformation. Needed to convert
        meshes in FreeSurfer space if masks where generated from the 
        scanner space.

    kwargs: dict
        See arguments of fit_surface_to_volume.

    Returns
    -------
    sphere: pv.PolyData
        Sphere fitted to MRI mask.
    """
    if isinstance(mask, nib.Nifti1Image):
        volume = mask.get_fdata()
        affine = mask.affine
    elif isinstance(mask, ants.ANTsImage):
        volume = mask.numpy()
        affine = get_affine(mask)
    else:
        raise TypeError("MRI mask type needs to be nib.Nifti1Image or ants.ANTsImage")

    # use MNE's icosahedron
    template_ico = _get_ico_surface(grade=5)
    template_ico = surf_to_polydata(template_ico["rr"], template_ico["tris"])

    # fit surface
    sphere, interp = fit_surface_to_volume(volume, sphere=template_ico, **kwargs)

    # move to [mm] space
    sphere.transform(affine, inplace=True)

    # fix the winding order and normals
    if np.linalg.det(affine) < 0:
        sphere.flip_faces(inplace=True)

    sphere.compute_normals(inplace=True, consistent_normals=False, auto_orient_normals=False)

    assert np.array_equal(template_ico.faces, sphere.faces), "Topology was corrupted during fitting!"

    return sphere, interp



def build_bem_layers(
    seg: Parcellation,
    bem_labels: dict,
    **kwargs,
) -> dict[str, pv.PolyData]:
    """
    Build BEM layers from segmentation.
    """
    # extract labels from segmentation
    skin_mask = create_mask(seg, regions=bem_labels["skin"])
    oskull_mask = create_mask(seg, regions=bem_labels["outer_skull"])
    iskull_mask = create_mask(seg, regions=bem_labels["inner_skull"])

    # skin
    skin_mask = nibabel_to_ants(skin_mask)
    skin_mask = skin_mask.morphology("close", radius=10)
    skin_mesh, skin_interp = fit_surface_to_mri_mask(
        skin_mask, z_floor = 20, **kwargs,
    )

    # outer skull
    oskull_mask = nibabel_to_ants(oskull_mask)
    oskull_mask = oskull_mask.morphology("close", radius=10)
    oskull_mesh, oskull_interp = fit_surface_to_mri_mask(
        oskull_mask, outer_interpolator=skin_interp, z_floor = 35, **kwargs,
    )

    # inner skull
    iskull_mask = nibabel_to_ants(iskull_mask)
    iskull_mask = iskull_mask.morphology("close", radius=10)
    iskull_mesh, _ = fit_surface_to_mri_mask(
        iskull_mask, outer_interpolator=oskull_interp, z_floor=45, **kwargs,
    )

    return {
        "skin": skin_mesh,
        "outer skull": oskull_mesh,
        "inner skull": iskull_mesh,
    }


def bem_transform(bem, trans):
    """
    Apply transform on BEM layer vertices.
    
    Parameters
    ----------
    bem: dict[str, pv.PolyData]
        BEM layers.

    trans: np.ndarray of shape (4,4)
        Transformation.

    Returns
    -------
    bem: dict[str, pv.PolyData]
        Transformed BEM
    """
    for k in bem.keys():
        bem[k].transform(trans, inplace=True)

        # fix the winding order and normals
        if np.linalg.det(trans) < 0:
            bem[k].flip_faces(inplace=True)

        bem[k].compute_normals(inplace=True, consistent_normals=False, auto_orient_normals=False)

    return bem