"""
Module Name: geometry.py
Description: Functions to load/save/convert meshes and extract values of interest.

A number of surfaces are generated for the head model (BEM surfaces, brain surfaces, etc.). 
This module contains functions for handling surface meshes.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path

import numpy as np
import skimage

import pyvista as pv

def vol_to_surf(
        volume          : np.ndarray,
        remesh          : bool = True,
        compute_normals : bool = True,
        ) -> pv.PolyData:
    """
    Generate mesh from volumetric mask.

    Parameters
    ----------
    volume: np.ndarray
        Volume mask.

    remesh: bool
        Whether to remesh into an icosahedral surface.

    subdivision: int = 4
        Icosahedral subdivision for template sphere. 

    compute_normals: bool = True
        Compute normals.

    Returns
    -------
    mesh: pv.Polydata
        Surface mesh.
    """

    # 1. -- Extract surface with marching cubes --
    verts, faces, _, _ = skimage.measure.marching_cubes(volume, level=0.5)
    mesh               = surf_to_polydata(verts, faces)
    mesh               = mesh.extract_largest()

    if remesh:
        # 2. -- Create an icosphere mesh --
        center          = mesh.points.mean(axis=0)
        centered_mesh = mesh.points - center 
        radius          = np.linalg.norm(centered_mesh, axis=1).max()

        icosphere = pv.Icosphere(radius, center, nsub=4)

        # 3. -- Project icosphere vertices onto the original surface --
        # build direction vector from the center
        directions = icosphere.points - center 
        norms      = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= norms

        # perform ray tracing
        points, rays, _ = mesh.multi_ray_trace(icosphere.points, -directions, first_point=True)
        idx             = rays.argsort()    # order may be lost with GPU interaction
        new_points      = points[idx]

        # 4. -- Remesh surface --
        mesh = pv.PolyData(new_points, icosphere.faces)

    if compute_normals:
        mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh


def surf_to_polydata(
        vertices : np.ndarray,
        faces    : np.ndarray,
) -> pv.PolyData:
    """
    Converts a list of vertices and faces to a PolyData structure.
    """

    # The faces for the pv.PolyData object needs to be a list of [..., n,v1,v2,v3, ...]
    # where n indicates the number of vertices of the face and v1-vn indicate the
    # vertices belonging to that face.
    pv_faces = np.empty_like(faces, shape=(faces.shape[0], faces.shape[1]+1))    # pylint: disable=unexpected-keyword-arg
    pv_faces[:,1:] = faces
    pv_faces[:,0]  = faces.shape[1]
    pv_faces       = np.hstack(pv_faces)

    return pv.PolyData(vertices, pv_faces)


def write_surface(
    fname : str | Path,
    mesh  : pv.PolyData | np.ndarray, 
    faces : np.ndarray = None, 
    meta = None,
):
    """
    Save a surface to FreeSurfer geometry format (.surf, .pial, .white, etc.).
    """
    import nibabel as nib

    if isinstance(mesh, pv.PolyData):
        coords = mesh.points
        # PyVista faces are [3, v1, v2, v3, ...]. 
        # We need to reshape back to (N, 3) and skip the '3' prefix.
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
    else:
        coords = mesh
        if faces is None:
            raise ValueError("Faces must be provided if mesh is a numpy array.")

    nib.freesurfer.io.write_geometry(str(fname), coords, faces, volume_info=meta)