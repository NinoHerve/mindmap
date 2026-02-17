"""
Module Name: analysis/connectivity/smooth.py
Description: Functions to apply graph smoothing on connectomes.

This module provides functions to apply graph smoothing on connectomes.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import scipy as sp
import sklearn


def smooth_connectome(conn : np.ndarray, kernel) -> np.ndarray:
    """
    Smoothen connectome according to Sina Mansour et. al (2020)

    Parameters
    ----------
    conn: np.ndarray (N,N)
        Connectome.

    kernel: np.ndarray
        Smoothing kernel

    Returns
    -------
    sc: np.ndarray (N,N)
        Smoothed connectome
    """
    # define incidence matrices
    indices = np.argwhere(conn>0)
    n, s    = len(conn), len(indices)
    M1, M2  = np.zeros((n,s)), np.zeros((n,s))

    for i, j in enumerate(indices):
        M1[j[0], i] = np.sqrt(conn[j[0],j[1]])
        M2[j[1], i] = np.sqrt(conn[j[1],j[0]])

    # smooth incidence
    M1s = (kernel.T @ M1)
    M2s = (kernel.T @ M2)

    # compute smoothed structural connectome
    sc = (M1s @ M2s.T) + (M2s @ M1s.T)
    sc = (sc + sc.T)/2
    np.fill_diagonal(sc, 0)

    return sc


def gaussian_kernel(
    geodist : sp.sparse.csr,
    fwhm    : float,
    epsilon : float,
    dim     : int
) -> np.ndarray:
    """
    Create Gaussian kernels for connectome smoothing.

    Parameters
    ----------
    geodist: sp.sparse.csr
        Geodesic distances between nodes.

    fwhm: float
        Full width half maximum. 

    epsilon: float
        Smoothing radius parameter.

    dim:
        Smoothing dimensionality (surface is 2D)

    Returns
    ------- 
    gaussian: sp.sparse.csr (N, N)
        Gaussian kernel for each node. The kernel distributes a weight arriving at node i between its neighbors.
    """
    # gaussian variance
    sigma = fwhm / np.sqrt(8*np.log(2))

    # maximum smoothing distance
    #max_smoothing_distance = sigma * (-sp.stats.norm.ppf((1-(1-epsilon)**(1/dim)) /1))
    max_smoothing_distance = sigma * (-2 * np.log(epsilon)) ** (1/dim)
    geodist[geodist>max_smoothing_distance] = 0
    geodist.eliminate_zeros()

    # Smoothing kernel for each node
    gaussian = -(geodist**2/(2*sigma**2))
    np.exp(gaussian.data, out=gaussian.data)
    gaussian += sp.sparse.eye(gaussian.shape[0], dtype=gaussian.dtype, format="csr")
    gaussian /= sp.sparse.linalg.norm(gaussian, ord=1, axis=1)[:, None]
    gaussian = sklearn.preprocessing.normalize(gaussian, norm="l1")     # sums to 1

    return gaussian

def unit_kernel(geodist, max_distance):
    """
    Create unit kernel for connectome smoothing.

    Parameters
    ----------
    geodist: sp.sparse.csr
        Geodesic distances between nodes.

    max_distance:
        Maximum width in [mm] of kernel.

    Returns
    -------
    unit_kernel: sp.sparse.csr (N,N)
        Unit kernel. The kernel of a node of interest is 1 for every neighbor node,
        meaning it is below `max_distance` from the node of interest.
    """
    # apply maximum smoothing distance
    geodist[geodist>max_distance] = 0
    geodist.eliminate_zeros()

    kernel = geodist > 0
    kernel.setdiag(1)   # kernel is 1 if neighborhood else 0

    return kernel

