"""
Module Name: geometry/processing.py
Description: Functions to operate on meshes and point clouds.

A number of surfaces are generated for the head model (BEM surfaces, brain surfaces, etc.). 
This module contains functions for handling surface meshes.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt, center_of_mass, gaussian_filter, binary_fill_holes
from scipy.interpolate import RegularGridInterpolator

import pyvista as pv



def project_points_to_surface(
        points         : np.ndarray,
        target_surface : pv.PolyData,
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
    target_tree = KDTree(target_points)

    # Find the closest points on the surface for each point in the source point cloud
    _, indices = target_tree.query(points)

    # Get the corresponding closest points from the surface
    projected_points = target_points[indices]

    return projected_points


def find_neighbor_regions(surface, labels, regions):
    """
    Defines neighboring regions. 

    Parameters:
    ----------
    surface: [np.ndarray, np.ndarray]
        Cortical surface (or white surface). First element indicate vertices.
        Second element indicate triangles.

    labels: list[int]
        Labels for every dipole.

    regions: list[int]
        Region names.

    cras: np.ndarray
        Shift of the Talaraich transform.

    Returns:
    -------
    neighbors: (N,N)
        Neighborhood matrix. Like an adjacency matrix where a connection 
        means that two regions are neighbors.
    """
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



def fill_mask_all_axes(mask: np.ndarray) -> np.ndarray:
    """
    Applies binary_fill_holes across all three primary axes to seal
    complex 3D tunnels.
    """
    filled = mask.astype(bool)
    for axis in range(3):
        for i in range(filled.shape[axis]):
            slc = [slice(None)] * 3     # handles any axis
            slc[axis] = i
            filled[tuple(slc)] = binary_fill_holes(filled[tuple(slc)])

    return filled


def fit_surface_to_volume(
    volume: np.ndarray,
    iterations: int = 100,
    step_size: float = 0.6,
    threshold: float = 0.5,
    n_smooth: int = 1,
    z_floor: int = None,
    outer_interpolator: RegularGridInterpolator = None,
    min_buffer: float = 3.0,
    sphere: pv.PolyData = None, # In order to use MNE's icosahedron (for tri ordering)
):
    """
    Fit a 5th order icosahedron with Z-capping and containment constraints.

    Parameters
    ----------
    volume: np.ndarray
        Volume.
    
    iterations: int
        Number of iterations.
    
    step_size: float
        Step size per iteration.
    """
    # pre-process mask: fill holes and apply Z-floor
    solid_mask = fill_mask_all_axes(volume > threshold)
    if z_floor is not None:
        solid_mask[:,:,:z_floor] = 0    # soft floor

    # create a signed distance map ("Force Field")
    # positive values inside the mask, negative outside
    # This guides the vertices much more smoothly than raw intensities
    dist_map = distance_transform_edt(solid_mask) - distance_transform_edt(~solid_mask)
    dist_map = gaussian_filter(dist_map.astype(float), sigma=1.2)

    # setup coordinate grid for interpolation
    # This allows to sample at non-integer vertex positions
    coords = [np.arange(s) for s in dist_map.shape]
    interpolator = RegularGridInterpolator(coords, dist_map, bounds_error=False, fill_value=None)

    # initialize 5th order icosahedron (~5k vertices)
    # start with a large sphere outside the mask
    com = center_of_mass(solid_mask)

    if sphere is None:
        sphere = pv.Icosphere(radius=120, center=com, nsub=5)
    else:
        sphere.points = (sphere.points * 120) + com

    for i in range(iterations):
        # calculate surface normals
        sphere.compute_normals(inplace=True, consistent_normals=False)
        vertices = sphere.points
        normals = sphere.point_data["Normals"]

        # current move based on target mask
        d_target = interpolator(vertices)

        # Pull back if we are exiting the "outer" container
        if outer_interpolator is not None:
            d_outer = outer_interpolator(vertices)
            # fuce the fields: the vertex is contrained by the 'outer' envelope
            # this keeps the inner layer strictly inside the outer layer
            combined_d = np.minimum(d_target, d_outer - min_buffer)
        else:
            combined_d = d_target

        # define movement
        move = np.clip(combined_d, -2, 2)

        # damping to prevent jitter
        dampening = np.clip(np.abs(combined_d) / 5.0, 0.2, 1.0)
        sphere.points += normals * (move[:, np.newaxis] * step_size * dampening[:, np.newaxis])

        # smooth at each iteration to preserve manifold
        smoothed = sphere.smooth(n_iter=n_smooth, relaxation_factor=0.05)
        sphere.points = smoothed.points        # ensures thate sphere.faces remains untouched

        if i % 10 == 0:
            # prevent triangle stretching in complex area like the skull base
            taubin = sphere.smooth_taubin(n_iter=5, pass_band=0.1)
            sphere.points = taubin.points

    return sphere, interpolator