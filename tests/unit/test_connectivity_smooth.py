"""
This module performs unit tests for the analysis/connectivity/smooth.py module.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from mindmap.analysis.connectivity.smooth import gaussian_kernel, smooth_connectome


@pytest.fixture
def chain_geodist():
    """5 nodes on a line, 2mm apart (0,2,4,6,8mm): geodesic distance = |i-j|*2."""
    pos = np.arange(5) * 2.0
    dist = np.abs(pos[:, None] - pos[None, :])
    return sp.csr_matrix(dist)


def test_gaussian_kernel_rows_sum_to_one(chain_geodist):
    kernel = gaussian_kernel(chain_geodist, fwhm=4.0, epsilon=0.01, dim=2)
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0)


def test_gaussian_kernel_weight_decreases_monotonically_with_distance(chain_geodist):
    """
    Regression test: `geodist**2` on a scipy sparse matrix is matrix power
    (matrix multiplication), not an elementwise square, which used to make
    the kernel's weight non-monotonic in distance (e.g. a node 4mm away
    could get more weight than one 2mm away). `.power(2)` fixes this --
    weight must strictly decrease as distance from the source node grows.
    """
    kernel = gaussian_kernel(chain_geodist, fwhm=4.0, epsilon=0.01, dim=2).toarray()

    row0 = kernel[0]
    distances = np.abs(np.arange(5) * 2.0 - 0.0)
    order = np.argsort(distances)

    assert np.all(np.diff(row0[order]) <= 0)   # monotonically non-increasing with distance
    assert row0[0] > row0[1] > row0[2]          # and strictly so over the non-truncated range


def test_gaussian_kernel_tiny_fwhm_is_identity(chain_geodist):
    """A vanishingly small fwhm truncates all off-diagonal smoothing, leaving pure self-weight."""
    kernel = gaussian_kernel(chain_geodist, fwhm=1e-9, epsilon=0.01, dim=2)
    assert np.allclose(kernel.toarray(), np.eye(5))


def test_gaussian_kernel_does_not_mutate_input(chain_geodist):
    """`gaussian_kernel` truncates distances beyond its cutoff in-place internally; the
    caller's matrix must not be affected."""
    original = chain_geodist.toarray().copy()
    gaussian_kernel(chain_geodist, fwhm=1.0, epsilon=0.01, dim=2)
    assert np.array_equal(chain_geodist.toarray(), original)


def _reference_smooth_connectome(conn, kernel):
    """Dense, unvectorized reference implementation of the same formula (not the
    original dense-loop code, but the underlying Mansour et al. definition), to
    check the sparse/vectorized rewrite in smooth.py against."""
    indices = np.argwhere(conn > 0)
    n, s = len(conn), len(indices)
    M1, M2 = np.zeros((n, s)), np.zeros((n, s))

    for i, (j0, j1) in enumerate(indices):
        M1[j0, i] = np.sqrt(conn[j0, j1])
        M2[j1, i] = np.sqrt(conn[j1, j0])

    kernel_dense = kernel.toarray()
    M1s = kernel_dense.T @ M1
    M2s = kernel_dense.T @ M2

    sc = (M1s @ M2s.T) + (M2s @ M1s.T)
    sc = (sc + sc.T) / 2
    np.fill_diagonal(sc, 0)

    return sc


def test_smooth_connectome_matches_reference_formula(chain_geodist):
    kernel = gaussian_kernel(chain_geodist, fwhm=4.0, epsilon=0.01, dim=2)

    # conn must be symmetric, like every other connectome in this module
    # (see smooth_connectome's docstring)
    conn = np.zeros((5, 5))
    conn[0, 3] = conn[3, 0] = 2
    conn[1, 4] = conn[4, 1] = 1

    expected = _reference_smooth_connectome(conn, kernel)
    actual = smooth_connectome(conn, kernel)

    assert np.allclose(actual, expected)


def test_smooth_connectome_conserves_total_mass_without_self_overlap():
    """
    With a kernel that has no cross-node spread (here: a vanishingly small
    fwhm, i.e. an identity kernel), no mass can leak onto the diagonal, so
    summing the (symmetric) output should equal exactly 2x the input total
    -- each undirected edge counted from both sides.

    This invariant only holds exactly when there's no self-overlap: with a
    wider kernel, spread mass from two different edges can land on the same
    node's diagonal entry, which `np.fill_diagonal(sc, 0)` then discards --
    so total mass is preserved up to that (typically small) leak, not
    exactly, once real smoothing is involved.
    """
    kernel = sp.eye(5, format="csr")

    conn = np.zeros((5, 5))
    conn[0, 3] = conn[3, 0] = 2
    conn[1, 4] = conn[4, 1] = 1

    sc = smooth_connectome(conn, kernel)

    assert sc.sum() == pytest.approx(2 * conn.sum())


def test_smooth_connectome_empty_connectome():
    kernel = sp.eye(4, format="csr")
    sc = smooth_connectome(np.zeros((4, 4)), kernel)
    assert np.all(sc == 0)
    assert sc.shape == (4, 4)
