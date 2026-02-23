"""
Module Name: metric.py
Description: Functions to measure metrics.

This module provides functions to measure various metrics. Usefull for evaluation, validation,
or simply testing.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

def avg_power(signal: np.ndarray) -> np.ndarray:
    """
    Compute average power of a signal.

    Parameters
    ----------
    signal: np.ndarray (t,) or (n,t)

    Returns
    -------
    avg_pow: np.ndarray (1,) or (n,)
        Average power for every sample n.
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(1,-1)   # consider signal as temporal signal (1,T)

    avg_pow = signal - np.mean(signal, axis=1)[:,None]
    avg_pow = np.sum(avg_pow**2, axis=1)/signal.shape[-1]

    return avg_pow


def point_to_area_distance(
        target_point : np.ndarray,
        target_area  : np.ndarray,
        x            : int = 1,
        ) -> float:
    """
    Measures distance between a point and an area.

    Parameters
    ----------
    target_point: np.ndarray (3,)
        Target point in [m]

    target_area: np.ndarray (3,N) | nibabel.Nifti1Image
        Cloud point in [m] if array, in [mm] if Niti1Image.

    x: int
        Number of points from the cloud point to consider.

    Returns
    -------
    distance: float
        Distance in [m]

    Note: The distance is measured as the length of the average distance vector between
    the target point and the x closest points of the point cloud.:
    - x = 1: Distance to closest point of the area.
    - x = "all": Distance to the center of mass of the area.
    - x != "all" != 1: Distance to the center of mass of the portion consisting of the
      x closest points.
    """

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


def compute_distances(array1, array2, output_type="scalar"):
    """
    Computes distances between points of two arrays. Displacements
    will point from array1 to array2.

    Parameters:
    ----------
    array1: np.ndarray of shape (n_starts, n_dimensions)
        Sarting points.
    
    array2: np.ndarray of shape (n_ends, n_dimensions)
        Ending points

    output_type: str
        'scalar' (Euclidean) or 'vector' (Displacement)

    Returns
    -------
    dist: np.ndarray
        If 'scalar': Array of shape (n_starts, n_ends) containing L2 norms
        If 'vector': Array of shape (n_starts, n_ends, n_dimensions) containing displacement vectors.
    """
    # Ensure inputs are numpy arrays
    a1 = np.atleast_2d(array1)
    a2 = np.atleast_2d(array2)

    # Broadcasting to find difference between every pair
    diff =  a2[np.newaxis,:,:] - a1[:,np.newaxis,:]

    if output_type == "vector":
        return diff
    elif output_type == "scalar":
        return np.linalg.norm(diff, axis=2)
    else:
        raise ValueError("output_type must be 'scalar' or 'vector'")
    


def extract_voxel_volume(affine: np.ndarray):
    """
    Computes voxel volume in [mm³] from the affine.
    """
    # defines borders of voxel (parallelepiped)
    v1 = affine[:3, 0]
    v2 = affine[:3, 1]
    v3 = affine[:3, 2]

    # volume of a parallelepiped
    volume = np.abs(np.cross(v1, v2) @ v3)

    return volume


def calc_spatial_correlation(map_ref, map_est):
    """
    Pearson correlation between reference and estimated spatial maps.

    Ths function performs a vectorized calculation of the spatial correlation
    for each observation (e.g. timepoint or trial) independently.

    Parameters
    ----------
    map_ref: np.ndarray of shape (n_observations, n_channels)
        The ground truth of reference spatial maps (e.g., actual iEEG).

    map_est: np.ndarray of shape (n_observations, n_channels)
        The estimated spatial maps (e.g., virtual iEEG from EEG)

    Returns
    -------
    sp_cor: np.ndarray of shape (n_observations,)
        The spatial correlation coefficient for every observation.
        Returns 0 where a map has zero variance.
    """
    # ensure inputs are at least 2D: (n_observations, n_channels)
    # if input is (M,), it becomes (1,M)
    map_ref = np.atleast_2d(map_ref)
    map_est = np.atleast_2d(map_est)

    if map_ref.shape != map_est.shape:
        raise ValueError(f"Shape mismatch: map_ref {map_ref.shape} and map_est {map_est.shape} must be identical.")

    # center data
    ref_c = map_ref - map_ref.mean(axis=1, keepdims=True)
    est_c = map_est - map_est.mean(axis=1, keepdims=True)

    # cacluate correlation
    upper = np.sum(ref_c * est_c, axis=1)
    lower = np.sqrt(np.sum(ref_c**2, axis=1) * np.sum(est_c**2, axis=1))

    sp_cor = np.divide(upper, lower, out=np.zeros_like(upper), where=lower > 1e-12)
    
    return sp_cor if sp_cor.size > 1 else sp_cor[0]


def calc_pairwise_dist(array1, array2, output_type="scalar"):
    """
    Computes pairwise distances between points of two arrays. Displacements
    will point from array1 to array2.

    Parameters:
    ----------
    array1: np.ndarray
        Array of shape (N,D) or (N,)
    
    array2: np.ndarray
        Array of shape (N,D) or (N,)

    output_type: str
        'scalar' (Euclidean) or 'vector' (Displacement)

    Returns
    -------
    dist: np.ndarray
        If 'scalar': Array of shape (N,) containing L2 norms
        If 'vector': Array of shape (N,D) containing displacement vectors.
    """
    # Ensure inputs are numpy arrays
    a1 = np.atleast_2d(array1)
    a2 = np.atleast_2d(array2)

    # pairwise differences
    diff =  a2 - a1

    if output_type == "vector":
        return diff
    elif output_type == "scalar":
        return np.linalg.norm(diff, axis=1)
    else:
        raise ValueError("output_type must be 'scalar' or 'vector'")
    

def calc_spatial_dispersion_grova(map_est, dist_to_h):
    """
    Grova's Spatial Dispersion (SD): Spread of estimated activity
    relative to the ground-truth clinical focus (Set H).

    Parameters
    ----------
    map_est: np.ndarray of shape (n_observations, n_channels)
        The estimated spatial maps (e.g., virtual iEEG)

    dist_to_h: np.ndarray of shape (n_channels,)
        Minimum distance from each channel i to the closest channel
        j in the reference set H. (dist = 0 if channel i is in H).

    Returns
    -------
    sp_dis: np.ndarray of shape (n_observations,)
        Spatial dispersion (SD). Higher values indicate 'smearing'
        outside the clinical zone or localization error.
    """
    # ensure inputs are at least 2D: (n_observations, n_channels)
    # if input is (M,), it becomes (1,M)
    map_est = np.atleast_2d(map_est)

    # ensure dist_to_h is (1, n_channels) for broadcasting
    dist_to_h = np.atleast_2d(dist_to_h)

    # Energy (V^2)
    energy = map_est**2

    # compute dispersion
    numerator   = np.sum(dist_to_h * energy, axis=1)
    denominator = np.sum(energy, axis=1)

    dispersion = np.sqrt(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12))
    
    return dispersion if dispersion.size > 1 else dispersion[0]


def calc_ratio_spurious_activity(map_est, is_distant_mask):
    """
    Ratio of Spurious Activity (RSA) as per Grova et al.

    Parameters
    ----------
    map_est: np.ndarray of shape (n_observations, n_channels)
        The estimated potentials (V_MEG).
    is_distant_mask: np.ndarray of shape (n_channels,)
        Boolean mask or binary array (1/0) where 1 indicates the contact
        is > 10mm away from the reference set H.

    Returns
    -------
    rsa: np.ndarray of shape (n_observations,)
        The ratio of energy in distant contacts vs total energy.
    """
    # ensure inputs are at least 2D: (n_observations, n_channels)
    # if input is (M,), it becomes (1,M)
    map_est = np.atleast_2d(map_est)

    # energy (V^2)
    energy = map_est**2

    # energy in distant contacts (numerator)
    # we use the mask to zero out energy in contacts close to H.
    spurious_energy = np.sum(energy[:, is_distant_mask], axis=1)

    # Total energy across all contacts (denominator)
    total_energy = np.sum(energy, axis=1)

    # RSA = Sum(V^2_distant) / Sum(V^2_total)
    rsa = np.divide(spurious_energy, total_energy, out=np.zeros_like(spurious_energy), where=total_energy>1e-12)

    return rsa if rsa.size > 1 else rsa[0]
