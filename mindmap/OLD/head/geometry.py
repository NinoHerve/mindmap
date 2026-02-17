"""
Module Name: geometry.py
Description: Functions to handle geometries.

A number of surfaces are generated for the head model (BEM surfaces, brain surfaces, etc.). 
This module contains functions for handling surface meshes.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import scipy as sp
import nibabel as nib
import ants
import skimage
import mne
import pyvista as pv

# ----------------
# Mesh operations
# -----------------

def project_points_to_surface(
        points         : np.ndarray, 
        target_surface : pv.PolyData 
) -> np.ndarray:
    """
    Project a point cloud onto a surface by finding the closest points on the surface.

    Parameters
    ----------
    points: np.ndarray (N,3)
        Point cloud to project.
    
    target_surface: pv.PolyData
        Target surface to project onto.

    Return
    ------
    projected_points: np.ndarray (N,3)
        Points projected onto the surface.
    """
    # Get the target surface points
    target_points = np.asarray(target_surface.points)

    # Build a KDTree for fast nearest-neighbor search
    target_tree = sp.spatial.KDTree(target_points)

    # Find the closest points on the surface for each point in the source point cloud
    _, indices = target_tree.query(points)

    # Get the corresponding closest points from the surface
    projected_points = target_points[indices]

    return projected_points


def find_neighbor_regions(surface, parcellation, cras):
    """
    Defines neighboring regions. 

    Parameters:
    ----------
    surface: [np.ndarray, np.ndarray]
        Cortical surface (or white surface). First element indicate vertices.
        Second element indicate triangles.

    parcellation: Parcellation
        Volumetric parcellation with N regions.

    cras: np.ndarray
        Shift of the Talaraich transform.

    Returns:
    -------
    neighbors: (N,N)
        Neighborhood matrix. Like an adjacency matrix where a connection 
        means that two regions are neighbors.
    """
    labels    = parcellation.label_points(surface[0]+cras, id=True)
    regions   = parcellation.get_clut()["#No."].values
    neighbors = np.zeros((len(regions), len(regions)))

    # find triangles having vertices in different regions
    tri_regions = labels[surface[1]]                             # triangles where vertex id is replaced by region id
    tri_regions = tri_regions[np.sum(tri_regions==0, axis=1)==0] # delete triangles having vertex with id 0 (unknown)
    info1 = np.abs(tri_regions[:,0] - tri_regions[:,1]) > 0      # True if vertex 1 is not in the same region as vertex 2
    info2 = np.abs(tri_regions[:,0] - tri_regions[:,2]) > 0      # True if vertex 1 is not in the same region as vertex 3

    # mark neighbors based on info1
    idx1 = np.searchsorted(regions, tri_regions[info1][:,0])
    idx2 = np.searchsorted(regions, tri_regions[info1][:,1])
    neighbors[idx1, idx2]= 1

    # mark neighbors based on info2
    idx1 = np.searchsorted(regions, tri_regions[info2][:,0])
    idx2 = np.searchsorted(regions, tri_regions[info2][:,2])
    neighbors[idx1, idx2]= 1

    return neighbors



def vol_to_surf(
        volume          : np.ndarray | ants.ANTsImage | nib.Nifti1Image,
        remesh          : bool = True,
        compute_normals : bool = True,
        apply_affine    : bool = True
        ) -> pv.PolyData:
    """
    Generate mesh from volumetric mask.

    Parameters
    ----------
    volume: np.ndarray, ants, nibabel
        Volume mask.

    remesh: bool
        Whether to remesh into an icosahedral surface.

    subdivision: int = 4
        Icosahedral subdivision for template sphere. 

    compute_normals: bool = True
        Compute normals.

    apply_affine: bool = True
        If volume is an ants or nibabel object, apply affine to be in [mm] space.

    Returns
    -------
    mesh: pv.Polydata
        Surface mesh.
    """
    def get_affine(image: ants.ANTsImage):
        """
        Return the image affine from an ANTsImage object. 
        """
        spacing = image.spacing
        origin  = image.origin
        direction = image.direction

        affine = np.eye(4)
        affine[:3, :3] = np.diag(spacing) @ direction
        affine[:3, 3] = origin

        affine[:2, :] *= -1    # flip y and z axis to be in nibabel convention

        return affine

    affine = None
    if isinstance(volume, ants.ANTsImage):
        affine = get_affine(volume)
        volume = volume.numpy()
    elif isinstance(volume, nib.Nifti1Image):
        affine = volume.affine
        volume = volume.get_fdata()

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

    if apply_affine:
        if affine is not None:
            mesh.points = nib.affines.apply_affine(affine, mesh.points)

    if compute_normals:
        mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh

def surf_correction(
        in_skull      : pv.PolyData, 
        out_skull     : pv.PolyData,
        offset        : float,
        return_arrows : bool = False,
) -> tuple[pv.PolyData]:
    """
    Corrects surfaces that should be concentric to each other but instead have 
    some overlaps.
    """

    # Step 1: Select points of in_skull outside of out_skull
    error_skull = in_skull.boolean_difference(out_skull)
    error_mask  = np.zeros(in_skull.n_points, dtype=bool)

    for point in error_skull.points:
        distance = np.linalg.norm(in_skull.points - point, axis=1) 
        idx = np.where(distance < 10e-8)[0]

        # omit new points created by boolean_difference
        if len(idx) == 0:
            pass 

        # save index of points present in in_skull 
        elif len(idx) == 1:
            error_mask[idx[0]] = 1

        # multiple points in in_skull that coincide should not happen
        else:
            raise ValueError("Multiple points for in_skull.")
        
    # Step 2: Select points of in_skull having distance smaller than offset to out_skull
    
    # compute directions
    in_skull.compute_normals()
    signs = np.where(error_mask, -1, 1)
    signs = np.expand_dims(signs, axis=1)
    directions = in_skull.point_normals*signs 

    # compute ray tracing 
    points, rays, _ = out_skull.multi_ray_trace(in_skull.points, directions)

    # compute distances to out_skull
    distance2surface = 1000 * np.ones(in_skull.n_points)
    for i in range(rays.max()+1):
        idx = np.where(rays==i)
        p = points[idx]

        distance2surface[i] = np.linalg.norm(p-in_skull.points[i], axis=1).min()

    closeness_mask = (distance2surface < offset)

    # Step 3: Correct the inner skull

    final_mask = np.ma.mask_or(error_mask, closeness_mask, shrink=False)
    in_skull_fixed = in_skull.copy(deep=True)

    # arrange arrays for computation
    final_mask = np.expand_dims(final_mask, axis=1)
    distance2surface = np.expand_dims(distance2surface, axis=1)
    
    # change points
    changes = -1 * final_mask * in_skull.point_normals * (offset - signs*distance2surface)
    in_skull_fixed.points = in_skull.points + changes

    
    if return_arrows:
        changed_mesh = pv.PolyData(in_skull.points)
        changed_mesh["vectors"] = np.array(changes)
        arrows = changed_mesh.glyph(orient="vectors", scale=True)

        return in_skull_fixed, arrows
    
    return in_skull_fixed



# ----------------
# PolyData makers
# ----------------

def nib_to_polydata(image:nib.Nifti1Image, trans:np.ndarray = np.eye(4)):
    """
    Convert nibabel to pyvista point cloud.

    Parameters
    ----------
    image: nib.Nifti1Image
        Nibabel volume image.
    
    trans: np.ndarray (3,) or (4,4)
        Transformation subject -> FreeSurfer ((3,) is a translation)

    Returns
    -------
    cloud_point: pv.PolyData
    """
    # voxel -> subject space
    voxels = np.argwhere(image.get_fdata()>0)
    points = nib.affines.apply_affine(image.affine, voxels)

    # subject -> FreeSurfer space
    if trans.shape == (3,): # translation
        points = (points - trans)/1000
    else:
        points = nib.affines.apply_affine(trans, points)/1000

    return pv.PolyData(points)


def mne_to_polydata(
        source_space:mne.SourceSpaces,
        signal:np.ndarray = None,
        n_points:int = 2
        ) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Convert MNE object to pyvista polydata.

    Parameters
    ----------
    src: mne.SourceSpaces
        Source space.

    scalars: np.ndarray 
        Dipole values.

    n_points: int
        Number of points to consider for dipole->vertex interpolation.

    Returns
    -------
    polydata: list(pv.PolyData)
        Left and right hedef resample_mesh(volume, remesh=True, subdivision=4):
    """
    # dipoles
    dipoles_l = pv.PolyData(source_space[0]["rr"][source_space[0]["vertno"]])
    dipoles_r = pv.PolyData(source_space[1]["rr"][source_space[1]["vertno"]])

    # hemispheres
    vertices, faces = source_space[0]["rr"], source_space[0]["tris"]
    hemi_l = surf_to_polydata(vertices, faces)

    vertices, faces = source_space[1]["rr"], source_space[1]["tris"]
    hemi_r = surf_to_polydata(vertices, faces)

    if signal is not None:
        split = source_space[0]["nuse"]
        dipoles_l["signal"] = signal[:split]
        dipoles_r["signal"] = signal[split:]

        # interpolate dipoles -> hemisphere vertices
        hemi_l = hemi_l.interpolate(dipoles_l, strategy="closest_point", n_points=n_points)
        hemi_r = hemi_r.interpolate(dipoles_r, strategy="closest_point", n_points=n_points)

    return hemi_l, hemi_r

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
