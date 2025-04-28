
import mne
import numpy as np
import pyvista as pv
import scipy as sp

# ----------------------- helpers to handle MNE objects ------------------------
def get_ch_positions(eeg: mne.io.Raw | mne.Epochs | mne.Evoked) -> np.ndarray:
    """
    Get channel position from mne object.

    Parameters
    ----------
    eeg: mne.io.Raw | mne.Epochs | mne.Evoked
        EEG object.

    Returns
    -------
    pos: np.ndarray
        (n,3) array indicating the (x,y,z) location of n electrodes.
    """

    digs = eeg.info["dig"]
    positions = []

    for d in digs: 
        if "eeg" in str(d["kind"]).lower():
            positions.append(d["r"])

    return np.array(positions)


def eeg_rising_phase(
    eeg    : mne.Evoked,
    ratio  : float = 0.5,
    tmin   : float = None,
    tmax   : float = None,
    as_idx : bool  = False,
): 
    """
    Finds rising phase of spike.

    Parameters
    ----------
    eeg: mne.Evoked 
        EEG. 
    
    ratio: float in [0-1]
        Ratio of the rising phase. A ratio of 1 corresponds to the peak.

    tmin: float
        Minimum point in time to consider for peak (in [s])

    tmax: float
        Maximum point in time to consider for peak (in [s])

    as_idx: bool
        Whether to return the time index.

    Returns
    -------
    out: float
        Time of peak or idx of peak if `as_idx` is True. 
    """
    data = eeg.copy() 

    # get peak
    channel, peak_time, peak_value = data.get_peak(tmin=tmin, tmax=tmax, return_amplitude=True)

    # set threshold
    threshold = ratio * np.abs(peak_value)

    # slice signal up to peak
    data.crop(tmin, peak_time)
    pre_peak = np.abs(data.get_data(channel).squeeze())

    # find last index before peak where it crossed threshold
    above_thresh_indices = np.where(pre_peak >= threshold)[0]
    rising_phase_idx     = above_thresh_indices[0] if above_thresh_indices.size > 0 else None 

    # find time from idx
    out = data.times[rising_phase_idx]

    if as_idx:
        out = eeg.time_as_index(out)

    return out



# -----------------  Electrode coregistration --------------------------------
def apply_transformation(points, transformation, scale):
    """ Apply the 4x4 transformation matrix and scaling to the points. """
    points_homogeneous = np.hstack((points * scale, np.ones((points.shape[0], 1))))
    transformed_points = points_homogeneous @ transformation.T
    return transformed_points[:, :3]

def point_to_plane_icp_with_scaling(
        source_points: np.ndarray, 
        target_surface: pv.PolyData, 
        max_iterations: int = 1,
        tolerance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Iterative Closest Point (ICP) with point-to-plane association and scaling.
    
    Parameters
    ----------
    source_points: np.ndarray (N,3)
        Source points to displace.
    
    target_surface: pv.PolyData
        Reference surface.

    max_iterations: int (default 1)
        Maximum number of iterations.

    tolerance: float (default 10e-6)
        Convergence tolerance based on the error change.
    
    Returns
    -------
    source_points: np.ndarray (N,3)
        Transformed source points.

    transform: np.ndarray (4,4)
        Affine transformation matrix.

    scaling: float
        Scaling factor.
    """
    # Get the normals from the target surface
    target_surface = target_surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    target_normals = target_surface.point_normals
    target_points  = np.asarray(target_surface.points)

    # Create a KDTree for nearest neighbor search
    target_tree = sp.spatial.KDTree(target_points)

    # Initialize transformation matrix (Identity matrix) and scaling factor
    transformation_matrix = np.eye(4)
    scaling_factor = 1.0  # Initial scaling

    previous_error = np.inf

    for iteration in range(max_iterations):
        # Step 1: Find closest points (point-to-plane)
        distances, indices = target_tree.query(source_points)
        matched_points     = target_points[indices]
        matched_normals    = target_normals[indices]

        # Step 2: Compute point-to-plane distances and form the system of equations
        A = []
        b = []

        for i in range(source_points.shape[0]):
            p = source_points[i]
            q = matched_points[i]
            n = matched_normals[i]
            A.append(np.concatenate((np.cross(p, n), n, [n @ p])))  # 3 rotational, 3 translation and 1 scaling term
            b.append(n @ (q - p))

        A = np.array(A)
        b = np.array(b)

        # Step 3: Solve the linear system to find the optimal rotation, translation, and scaling
        x, _, _, _ = sp.linalg.lstsq(A, b)

        rotation_vector    = x[:3]
        translation_vector = x[3:6]
        scale_delta        = x[6] 

        # Step 4: Convert rotation vector to rotation matrix using Rodrigues' formula
        theta = np.linalg.norm(rotation_vector)
        if theta > 1e-12:
            r = rotation_vector / theta
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        else:
            R = np.eye(3)

        # Step 5: Construct the homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = translation_vector

        # Step 6: Update the scaling factor
        scaling_factor *= (1 + scale_delta)

        # Step 7: Apply the transformation and scaling to the source points
        source_points = apply_transformation(source_points, T, scaling_factor)

        # Update the global transformation
        transformation_matrix = T @ transformation_matrix

        # Step 8: Check for convergence (change in error)
        current_error = np.mean(distances)
        if np.abs(previous_error - current_error) < tolerance:
            print(f"Converged at iteration {iteration + 1}")
            break

        previous_error = current_error

    return source_points, transformation_matrix, scaling_factor


def optimize_scaling_translation(source_points, target_surface, max_iterations=50, tolerance=1e-6):
    """
    Optimize scaling and translation without rotation to minimize point-to-plane distances.
    
    :param source_points: np.ndarray of shape (N, 3) representing source point cloud.
    :param target_surface: pyvista.PolyData representing the reference surface.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :return: Optimized scaling factor and translation vector.
    """
    # Get the normals and points from the target surface
    target_surface = target_surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    target_normals = target_surface.point_normals
    target_points  = np.asarray(target_surface.points)
    
    # Create a KDTree for nearest neighbor search
    target_tree = sp.spatial.KDTree(target_points)
    
    # Initialize scaling factor and translation vector
    s = 1.0  # Start with no scaling
    t = np.zeros(3)  # Start with no translation

    previous_error = np.inf

    for iteration in range(max_iterations):
        # Step 1: Find closest points (point-to-plane)
        distances, indices = target_tree.query(source_points)
        matched_points = target_points[indices]
        matched_normals = target_normals[indices]
        
        # Step 2: Build the linear system A * x = b to solve for scaling and translation
        A = []
        b = []

        for i in range(source_points.shape[0]):
            p = source_points[i]
            q = matched_points[i]
            n = matched_normals[i]
            
            # Row for scaling (n_i . p_i) and translation (n_i)
            A.append([p @ n, *n])
            # Right-hand side (q_i . n_i)
            b.append(n @ q)

        A = np.array(A)  # Matrix A has shape (N, 4) where N is the number of points
        b = np.array(b)  # Vector b has shape (N,)

        # Step 3: Solve the linear system to find scaling and translation
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        s = x[0]         # Scaling factor
        t = x[1:4]       # Translation vector

        # Step 4: Apply the scaling and translation to the source points
        source_points = s * source_points + t

        # Step 5: Check for convergence (change in error)
        current_error = np.mean(distances)
        if np.abs(previous_error - current_error) < tolerance:
            print(f"Converged at iteration {iteration + 1}")
            break
        
        previous_error = current_error

    return s, t, source_points



def project_points_to_surface(
        points: np.ndarray, 
        target_surface: pv.PolyData 
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