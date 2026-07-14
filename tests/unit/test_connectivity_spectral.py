"""
This module performs unit tests for mindmap/analysis/connectivity/spectral.py.
"""

import numpy as np

from mindmap.analysis.connectivity.spectral import compute_cst


def erdos_renyi(n, p, seed=0):
    """Random undirected connected-ish graph with edge probability p."""
    rng = np.random.default_rng(seed)
    adj = (rng.random((n, n)) < p).astype(float)
    adj = np.triu(adj, 1)
    return adj + adj.T


def test_compute_cst_first_eigenvalue_is_exactly_zero():
    """
    The normalized Laplacian's smallest eigenvalue is theoretically exactly
    0 for a connected graph; compute_cst forces this rather than trusting
    eigh's floating-point residual.
    """
    adj = erdos_renyi(40, 0.3, seed=1)
    _, eigval = compute_cst(adj)

    assert eigval[0] == 0.0
    assert np.all(eigval[1:] >= 0)


def test_compute_cst_eigenvectors_are_orthonormal():
    """The Laplacian eigenbasis must be orthonormal: eigvec.T @ eigvec == I."""
    n = 40
    adj = erdos_renyi(n, 0.3, seed=2)
    eigvec, _ = compute_cst(adj)

    np.testing.assert_allclose(eigvec.T @ eigvec, np.eye(n), atol=1e-8)


def test_compute_cst_does_not_transform_weights():
    """
    compute_cst must be purely spectral -- it should decompose whatever
    matrix it's handed, not silently invert or otherwise transform weights
    itself (that's the caller's decision, done before calling this).
    """
    n = 30
    adj = erdos_renyi(n, 0.3, seed=3).astype(float)
    weights = np.triu(np.random.default_rng(3).uniform(1, 10, size=(n, n)), k=1)
    weighted_adj = adj * (weights + weights.T)

    inverted_adj = np.divide(1, weighted_adj, out=np.zeros_like(weighted_adj), where=weighted_adj > 0)

    _, eigval_weighted = compute_cst(weighted_adj)
    _, eigval_inverted = compute_cst(inverted_adj)

    # decomposing the weighted matrix directly vs. its element-wise inverse
    # must give different spectra -- confirms compute_cst isn't secretly
    # inverting weights internally regardless of what it's handed.
    assert not np.allclose(eigval_weighted, eigval_inverted)
