"""
This module performs unit tests for the graph/connectome metrics in core/metric.py.
"""

import numpy as np
import pytest

from mindmap.core.metric import (
    density,
    small_world_sigma,
    modularity,
    distance_decay_fit,
    hemispheric_symmetry,
    block_conservation,
)


def ring_lattice(n, k):
    """Regular ring lattice: each node connected to its k nearest neighbors on a ring."""
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            adj[i, (i + j) % n] = 1
            adj[i, (i - j) % n] = 1
    return adj


def erdos_renyi(n, p, seed=0):
    """Random undirected graph with edge probability p."""
    rng = np.random.default_rng(seed)
    adj = (rng.random((n, n)) < p).astype(int)
    adj = np.triu(adj, 1)
    return adj + adj.T


def two_block_graph(sizes, p_in, p_out, seed=0):
    """Two blocks of the given (possibly unequal) sizes, denser within than across."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    block = np.repeat([0, 1], sizes)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            same_block = block[i] == block[j]
            p = p_in if same_block else p_out
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1
    return adj


def test_density():
    """A fully connected graph has density 1, an empty graph has density 0."""
    n = 10
    full = np.ones((n, n)) - np.eye(n)
    empty = np.zeros((n, n))

    assert density(full) == pytest.approx(1.0)
    assert density(empty) == pytest.approx(0.0)


def test_small_world_sigma_lattice_vs_random():
    """
    A ring lattice should be strongly small-world (sigma >> 1: high
    clustering relative to a degree-matched random null), while an
    Erdos-Renyi graph should have sigma close to 1.
    """
    n = 200
    lattice = ring_lattice(n, 10)
    er = erdos_renyi(n, p=lattice.sum() / (n * (n - 1)), seed=1)

    lattice_result = small_world_sigma(lattice, n_null=10, seed=0)
    er_result = small_world_sigma(er, n_null=10, seed=0)

    assert lattice_result["sigma"] > 2.0
    assert er_result["sigma"] == pytest.approx(1.0, abs=0.3)


def test_modularity_detects_two_blocks():
    """
    A clear two-block structure with unequal block sizes (130/70) should
    be detected with high modularity. Unequal sizes matter here: bct's
    spectral modularity_und builds a ragged per-community list internally
    and crashes under numpy>=2.0 whenever community sizes differ (fixed
    by switching to modularity_louvain_und) -- equal-size blocks would
    not have caught that.
    """
    adj = two_block_graph(sizes=(130, 70), p_in=0.25, p_out=0.005, seed=0)

    result = modularity(adj, seed=0)

    assert result["n_communities"] == 2
    assert result["q"] > 0.3


def test_distance_decay_fit_recovers_known_lambda():
    """
    Build a graph whose connection probability decays exactly as
    exp(-lambda*d) and check the fit recovers lambda.
    """
    rng = np.random.default_rng(0)
    n = 400
    coords = rng.uniform(0, 100, size=(n, 3))

    true_lambda = 0.05
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    prob = np.exp(-true_lambda * d)

    rand = rng.random((n, n))
    connected = np.triu(rand < prob, k=1)
    adj = (connected + connected.T).astype(int)

    result = distance_decay_fit(adj, coords, n_bins=20)

    assert result["lambda_"] == pytest.approx(true_lambda, rel=0.3)
    assert result["r2"] > 0.8


def test_hemispheric_symmetry_perfect_mirror():
    """
    If the right hemisphere is an exact mirror of the left (same degree
    per matched pair), symmetry correlation should be 1 and matching
    distance should be ~0.
    """
    rng = np.random.default_rng(0)
    n_side = 50

    left_coords = rng.uniform(1, 50, size=(n_side, 3))
    left_coords[:, 0] *= -1  # left hemisphere: negative x

    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1  # exact mirror

    coords = np.concatenate([left_coords, right_coords], axis=0)
    hemi = np.concatenate([np.zeros(n_side), np.ones(n_side)])

    # symmetric adjacency: dipole i (left) and its mirror i+n_side (right)
    # share the same degree by construction
    n = 2 * n_side
    adj = erdos_renyi(n_side, p=0.1, seed=2)
    full_adj = np.zeros((n, n), dtype=int)
    full_adj[:n_side, :n_side] = adj
    full_adj[n_side:, n_side:] = adj

    result = hemispheric_symmetry(full_adj, coords, hemi)

    assert result["r"] == pytest.approx(1.0, abs=1e-6)
    assert result["match_dist_mean"] == pytest.approx(0.0, abs=1e-6)


def test_block_conservation_exact_recovery():
    """
    If the fine-grained connectome is built so that every nonzero pair in
    a region-pair block equals the atlas value exactly, block averaging
    should recover the atlas exactly (r=1, mae=0, mean_bias=0).
    """
    n_regions = 3
    dipoles_per_region = 4
    n_dipoles = n_regions * dipoles_per_region

    dipole_labels = np.repeat(np.arange(1, n_regions + 1), dipoles_per_region)

    atlas_adj = np.array([
        [0, 10, 0],
        [10, 0, 20],
        [0, 20, 0],
    ], dtype=float)

    fine_adj = np.zeros((n_dipoles, n_dipoles))
    for i in range(n_regions):
        for j in range(n_regions):
            if atlas_adj[i, j] > 0:
                idx_i = np.where(dipole_labels == i + 1)[0]
                idx_j = np.where(dipole_labels == j + 1)[0]
                fine_adj[np.ix_(idx_i, idx_j)] = atlas_adj[i, j]

    result = block_conservation(fine_adj, dipole_labels, atlas_adj)

    assert result["r"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["mean_bias"] == pytest.approx(0.0)
    assert result["n_pairs"] == 2  # upper triangle: (0,1) and (1,2)
