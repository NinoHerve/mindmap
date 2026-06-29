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
import mne
from mne.bem import _get_ico_surface

from mindmap.geometry.io import surf_to_polydata
from mindmap.geometry.processing import surf_correction, fit_surface_to_volume, polydata_transform
from mindmap.anatomy.mri.io import get_affine, nibabel_to_ants
from mindmap.anatomy.parcellation.base import Parcellation
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

    sphere.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=False)

    assert np.array_equal(template_ico.faces, sphere.faces), "Topology was corrupted during fitting!"

    return sphere, interp


def build_bem_layersOLD(
    seg: Parcellation,
    bem_labels: dict,
    brain_interp: np.ndarray = None,
    **kwargs,
) -> dict[str, pv.PolyData]:
    """
    Build BEM layers from segmentation.

    Parameters:
    ----------
    seg: mm.Parcellation
        Segmentation image.
    
    bem_labels: dict
        Tissue labels belonging inside each surface.

    brain_interp: np.ndarray
        Interpolator of brain tissue.
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
        iskull_mask, outer_interpolator=oskull_interp, inner_interpolator=brain_interp, max_buffer=0, z_floor=45, **kwargs,
    )

    return {
        "skin": skin_mesh,
        "outer skull": oskull_mesh,
        "inner skull": iskull_mesh,
    }


def build_bem_layers(
    seg: Parcellation,
    bem_labels: dict,
    brain_interp: np.ndarray = None,
    **kwargs,
) -> dict[str, pv.PolyData]:
    """
    Build BEM layers from segmentation.

    Parameters:
    ----------
    seg: mm.Parcellation
        Segmentation image.
    
    bem_labels: dict
        Tissue labels belonging inside each surface.

    brain_interp: np.ndarray
        Interpolator of brain tissue.
    """
    # extract labels from segmentation
    iskull_mask = create_mask(seg, regions=bem_labels["inner_skull"])
    oskull_mask = create_mask(seg, regions=bem_labels["outer_skull"])
    skin_mask = create_mask(seg, regions=bem_labels["skin"])

    # inner skull
    iskull_mask = nibabel_to_ants(iskull_mask)
    iskull_mask = iskull_mask.morphology("close", radius=10)
    iskull_mesh, iskull_interp = fit_surface_to_mri_mask(
        iskull_mask, inner_interpolator=brain_interp, z_floor=45, **kwargs,
    )

    # outer skull
    oskull_mask = nibabel_to_ants(oskull_mask)
    oskull_mask = oskull_mask.morphology("close", radius=10)
    oskull_mesh, oskull_interp = fit_surface_to_mri_mask(
        oskull_mask, inner_interpolator=iskull_interp, z_floor = 35, **kwargs,
    )

    # skin
    skin_mask = nibabel_to_ants(skin_mask)
    skin_mask = skin_mask.morphology("close", radius=10)
    skin_mesh, _ = fit_surface_to_mri_mask(
        skin_mask, inner_interpolator=oskull_interp, z_floor = 20, **kwargs,
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
        bem[k] = polydata_transform(bem[k], trans)

    return bem


def template_coregistration(
    eeg_temp: mne.Evoked | mne.Epochs | mne.io.Raw,
    subject: str,
    subjects_dir: Path,
    iterations: int = 10
) -> mne.Evoked | mne.Epochs | mne.io.Raw:
    """
    Coregisters by projecting electrodes radially from the head center.
    Prevents clustering by maintaining angular spacing.
    """
    # 1. SETUP HEAD MESH
    head_surf = mne.read_surface(Path(subjects_dir)/subject/"bem"/"outer_skin.surf")
    head_mesh = surf_to_polydata(head_surf[0]/1000.0, head_surf[1])
    
    # Calculate geometric center of the head (the 'origin' for rays)
    center = np.mean(head_mesh.points, axis=0)

    # 2. INITIAL FIDUCIAL ALIGNMENT
    coreg = mne.coreg.Coregistration(eeg_temp.info, subject, subjects_dir)
    coreg.fit_fiducials(verbose=False)
    
    # We apply the rigid transform to get the montage into head space
    montage = eeg_temp.get_montage()
    montage.apply_trans(coreg.trans)
    
    ch_pos = montage.get_positions()['ch_pos']
    ch_names = list(ch_pos.keys())
    current_points = np.array([ch_pos[name] for name in ch_names])

    # 3. RADIAL RAY-CASTING (The 'Expansion' Phase)
    # Shoot rays from the center through each electrode to see
    # where they hit the scalp.
    final_points = []
    
    for pt in current_points:
        # Define a ray starting at center and going far past the electrode
        direction = pt - center
        direction /= np.linalg.norm(direction) # unit vector
        
        # Shoot ray 20cm out from center
        ray_start = center
        ray_end = center + (direction * 0.2) 
        
        points, _ = head_mesh.ray_trace(ray_start, ray_end)
        
        if len(points) > 0:
            # Take the intersection furthest from center (the scalp)
            final_points.append(points[-1])
        else:
            # Fallback: if ray misses, use closest point
            idx = head_mesh.find_closest_point(pt)
            final_points.append(head_mesh.points[idx])

    # 4. ITERATIVE SURFACE SMOOTHING (Optional but keeps it 'clean')
    # If the sensors look jittery, relax them while snapping them back to surface
    final_points = np.array(final_points)
    for _ in range(iterations):
        # Create a mesh of the electrodes
        cloud = pv.PolyData(final_points)
        surf_cap = cloud.delaunay_2d(alpha=0.05)
        # Relax (Smoothing)
        final_points = surf_cap.smooth(n_iter=1, relaxation_factor=0.1).points
        # Re-snap to surface (so they don't 'sink' into the head during smoothing)
        snapped = []
        for p in final_points:
            idx = head_mesh.find_closest_point(p)
            snapped.append(head_mesh.points[idx])
        final_points = np.array(snapped)

    # 5. UPDATE MNE
    new_ch_pos = dict(zip(ch_names, final_points))
    new_montage = mne.channels.make_dig_montage(ch_pos=new_ch_pos)
    
    eeg_coreg = eeg_temp.copy()
    eeg_coreg.set_montage(new_montage)
    
    return eeg_coreg