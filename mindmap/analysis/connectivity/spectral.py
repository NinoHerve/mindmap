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
    Compute the connectome spectrum transform: the eigendecomposition of the
    normalized graph Laplacian.

    Purely spectral -- takes whatever adjacency matrix it's handed. Any
    semantic transform of the weights (e.g. inverting lengths into
    strengths, `1/connectome`) is the caller's decision and must happen
    before calling this, not in here.

    Uses an explicit dense `np.linalg.eigh` on the Laplacian rather than
    pygsp's own `compute_fourier_basis` for directly-verifiable numerics,
    and forces the first eigenvalue to exactly 0 -- for a connected graph
    it's theoretically exactly 0, but `eigh` can return a tiny nonzero
    floating-point residual there instead.

    Parameters
    ----------
    connectome: np.ndarray (n,n)
        Connectome adjacency matrix.

    Returns
    -------
    eigvec: np.ndarray (n,n)
        Eigenvectors of the normalized graph Laplacian (columns), the
        connectome's spectral basis.

    eigval: np.ndarray (n,)
        Corresponding eigenvalues (spatial frequencies), ascending order.
    """
    graph = pygsp.graphs.Graph(connectome, lap_type="normalized")   # pylint: disable=no-member
    eigval, eigvec = np.linalg.eigh(graph.L.toarray())
    eigval[0] = 0

    return eigvec, eigval

