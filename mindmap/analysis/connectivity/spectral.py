"""
Module Name: analysis/connectivity/spectral.py
Description: Functions to apply spectral analysis on connectomes.

This module provides functions to apply spectral analysis on connectomes.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

import pygsp

def compute_cst(connectome : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the connectome spectrum transform.

    Parameters
    ----------
    connectome: np.ndarray (n,n)
        Connectome adjacency matrix.

    Returns
    -------
    cst: np.ndarray (n,n)
        Connectome spectrum transform (eigenvectors or normalized Laplacian).
    
    freqs: np.ndarray (n,)
        Spectrum frequencies (eigenvalues of normalized Laplacian).
    """
    
    graph = pygsp.graphs.Graph(connectome, lap_type="normalized")   # pylint: disable=no-member
    graph.compute_fourier_basis()

    return graph.U, graph.e

