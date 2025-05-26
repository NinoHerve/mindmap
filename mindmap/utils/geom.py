import pyvista as pv
import numpy as np
import nibabel as nib
import mindmap as mm
import mne
import skimage
from scipy.spatial import cKDTree
import ants
import nibabel as nib 


###########################################
#########        Operations       #########
###########################################

def project_to_mesh(mesh: pv.PolyData, points: np.ndarray):
    """
    Project points to mesh.
    """

    new_points = points.copy()

    for i, p in enumerate(points):
        # find three closest points
        tri   = mesh.find_closest_point(p, n=3)
        q,r,s = mesh.points[tri]

        # define surface
        normal = np.cross(r-q, s-q)
        normal = normal / np.linalg.norm(normal)

        # project point onto surface
        d             = np.dot(p-q, normal)      # distance to surface
        new_points[i] = p - d*normal

    return new_points 


def measure_distance(target_point, target_area, x=1, cras=np.zeros(3)):
    """
    Measures distance between a point and an area.

    Parameters
    ----------
    target_point: nd.array (3,)
        Target point

    target_area: nd.array (3,N) | nibabel.Nifti1
        Cloud point

    x: int
        Number of points from the cloud point to consider.

    Returns
    -------
    distance: float
        Distance in [m]

    Note: The distance is measured as the length of the average distance vector between the target point and the x closest points of the point cloud. 
    - x = 1: Distance to closest point of the area.
    - x = "all": Distance to the center of mass of the area.
    - x != "all" != 1: Distance to the center of mass of the portion consisting of the x closest points.
    """

    if isinstance(target_area, nib.Nifti1Image):
        affine      = target_area.affine
        target_area = np.argwhere(target_area.get_fdata()>0)
        target_area = (nib.affines.apply_affine(affine, target_area)-cras)/1000


    distances = target_area-target_point

    if x == "all":
        mean_distance = np.mean(distances, axis=0)
        norm          = np.linalg.norm(mean_distance)

    else:
        norms         = np.linalg.norm(distances, ord=2, axis=1)
        nearest       = distances[np.argsort(norms)[:x],:]
        mean_distance = np.mean(nearest, axis=0)
        norm          = np.linalg.norm(mean_distance)

    return norm

# Delete measure or compute distance !!!!!!!!!!!

def compute_distance(pos, area, m=1):
    """
    Compute distance between points and area.
    The algorithm computes the average vector distance to the m closest voxels of the area. If m=1, it's the distance to
    the closest point in the area, if m='all' it's the distance to the center of mass.

    Parameters
    ----------
    pos: np.ndarray (N,3)
        Position of points.

    area: nib.Nifti1Image
        Area as a nibabel image.

    m: int or 'all' 
        Number of voxels to take into account.

    Returns
    -------
    distances: np.ndarray (N,)
        Distances to area as norm of average distance vector.
    """
    if len(pos.shape) < 2:
        pos = pos.reshape(1,-1)

    voxels = np.argwhere(area.get_fdata()>0)
    voxels = mne.transforms.apply_trans(area.affine, voxels)
    
    if m == "all":
        m = voxels.shape[0]

    # compute vector distances
    vectors = voxels[:,None,:] - pos
    norms   = np.linalg.norm(vectors, ord=2, axis=-1)
    norms_i = np.argsort(norms, axis=0)
    ord_vec = np.take_along_axis(vectors, norms_i[:,:,None], axis=0)        # ordered from smallest to highest

    # compute norm of average of m closest distances
    avg  = np.mean(ord_vec[:m,:,:], axis=0)
    dist = np.linalg.norm(avg, axis=1)

    return dist


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



def vol_to_surf(volume, remesh=True, subdivision=4, compute_normals=True, apply_affine=True):
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
    mesh               = mm.utils.surf_to_polydata(verts, faces)
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
        points, rays, cells = mesh.multi_ray_trace(icosphere.points, -directions, first_point=True)
        idx                 = rays.argsort()    # order may be lost with GPU interaction
        new_points          = points[idx]

        # 4. -- Remesh surface --
        mesh = pv.PolyData(new_points, icosphere.faces)
    
    if apply_affine:
        if affine is not None:
            mesh.points = nib.affines.apply_affine(affine, mesh.points)

    if compute_normals:
        mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh


###########################################
#########   PolyData converters   #########
###########################################

def nib_to_polydata(image, trans=np.eye(4)):
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
    points = (nib.affines.apply_affine(image.affine, voxels))

    # subject -> FreeSurfer space
    if trans.shape == (3,): # translation
        points = (points - trans)/1000 
    else:
        points = nib.affines.apply_affine(trans, points)/1000

    return pv.PolyData(points)



def mne_to_polydata(src: mne.SourceSpaces, signal:np.ndarray=None, n_points=2):
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
    dipolesL = pv.PolyData(src[0]["rr"][src[0]["vertno"]])
    dipolesR = pv.PolyData(src[1]["rr"][src[1]["vertno"]])

    # hemispheres
    vertices, faces = src[0]["rr"], src[0]["tris"]
    hemiL = mm.utils.surf_to_polydata(vertices, faces)

    vertices, faces = src[1]["rr"], src[1]["tris"]
    hemiR = mm.utils.surf_to_polydata(vertices, faces)

    if signal is not None:
        split = src[0]["nuse"]
        dipolesL["signal"] = signal[:split]
        dipolesR["signal"] = signal[split:]

        # interpolate dipoles -> hemisphere vertices
        hemiL = hemiL.interpolate(dipolesL, strategy="closest_point", n_points=n_points)
        hemiR = hemiR.interpolate(dipolesR, strategy="closest_point", n_points=n_points)

    return hemiL, hemiR



def surf_to_polydata(
        vertices: np.ndarray, 
        faces: np.ndarray,
) -> pv.PolyData:
    """
    Converts a list of vertices and faces to a PolyData structure.
    """
    
    # The faces for the pv.PolyData object needs to be a list of [..., n,v1,v2,v3, ...] 
    # where n indicates the number of vertices of the face and v1-vn indicate the 
    # vertices belonging to that face.
    pv_faces = np.empty_like(faces, shape=(faces.shape[0], faces.shape[1]+1))
    pv_faces[:,1:] = faces 
    pv_faces[:,0] = faces.shape[1]
    pv_faces = np.hstack(pv_faces)

    return pv.PolyData(vertices, pv_faces)



