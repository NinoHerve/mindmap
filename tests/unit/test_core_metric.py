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
    rewire_connectome,
    swap_rewire_connectome,
    perturb_connectome_topology,
    perturb_connectome_weights,
    permute_connectome_weights,
    calc_correlation,
    calc_cosine,
    calc_pattern_geometry_correlation,
    calc_explained_variance,
    calc_ssim,
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


def weighted_erdos_renyi(n, p, seed=0):
    """Random undirected weighted graph: erdos_renyi topology, uniform(1,10) weights."""
    rng = np.random.default_rng(seed)
    adj = erdos_renyi(n, p, seed=seed).astype(float)
    weights = np.triu(rng.uniform(1, 10, size=adj.shape), k=1)
    weights = weights + weights.T
    return adj * weights


def degree_sequence(adj):
    """Sorted per-node degree (nonzero-entry count), for order-independent comparison."""
    return np.sort(np.count_nonzero(adj, axis=1))


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


# -----------------------------------------------------------------------
# Graph / connectome perturbations
# -----------------------------------------------------------------------

def test_rewire_connectome_noop_at_zero():
    """n_rewire=0 must return the connectome unchanged."""
    adj = erdos_renyi(30, 0.2, seed=0)
    rewired = rewire_connectome(adj, n_rewire=0)
    assert np.array_equal(rewired, adj)


def test_rewire_connectome_preserves_degree_sequence():
    """Rewiring relabels which edges exist but must preserve each node's degree."""
    adj = erdos_renyi(50, 0.15, seed=1)
    rewired = rewire_connectome(adj, n_rewire=10, seed=2)

    assert np.array_equal(degree_sequence(adj), degree_sequence(rewired))
    assert not np.array_equal(adj, rewired)


def test_rewire_connectome_preserves_weight_multiset():
    """Weights travel with their edge during a swap: same multiset of nonzero values."""
    adj = weighted_erdos_renyi(40, 0.2, seed=3)
    rewired = rewire_connectome(adj, n_rewire=10, seed=4)

    orig = np.sort(adj[np.triu_indices_from(adj, k=1)])
    new = np.sort(rewired[np.triu_indices_from(rewired, k=1)])

    np.testing.assert_allclose(orig[orig != 0], new[new != 0])


def test_rewire_connectome_respects_mask_both_directions():
    """
    A masked present edge must never be deleted, and a masked absent pair
    must never receive a new edge -- unlike the legacy sabotage_connectome,
    which only protected masked edges from deletion.
    """
    n = 30
    adj = erdos_renyi(n, 0.3, seed=5).astype(float)
    adj[0, 1] = adj[1, 0] = 1.0  # force a protected edge to exist
    adj[2, 3] = adj[3, 2] = 0.0  # force a protected non-edge to stay absent

    mask = np.zeros_like(adj, dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[2, 3] = mask[3, 2] = True

    rewired = rewire_connectome(adj, n_rewire=15, mask=mask, seed=6)

    assert rewired[0, 1] == 1.0 and rewired[1, 0] == 1.0
    assert rewired[2, 3] == 0.0 and rewired[3, 2] == 0.0


def test_rewire_connectome_reproducible_with_seed():
    """Same seed must give bit-identical output across repeated calls."""
    adj = erdos_renyi(40, 0.2, seed=7)
    r1 = rewire_connectome(adj, n_rewire=10, seed=8)
    r2 = rewire_connectome(adj, n_rewire=10, seed=8)
    assert np.array_equal(r1, r2)


def test_swap_rewire_connectome_noop_at_zero():
    """n_swaps=0 must return the connectome unchanged."""
    adj = erdos_renyi(30, 0.2, seed=0)
    rewired = swap_rewire_connectome(adj, n_swaps=0)
    assert np.array_equal(rewired, adj)


def test_swap_rewire_connectome_preserves_degree_sequence():
    """Rewiring relabels which edges exist but must preserve each node's degree."""
    adj = erdos_renyi(50, 0.15, seed=1)
    rewired = swap_rewire_connectome(adj, n_swaps=30, seed=2)

    assert np.array_equal(degree_sequence(adj), degree_sequence(rewired))
    assert not np.array_equal(adj, rewired)


def test_swap_rewire_connectome_preserves_weight_multiset():
    """Weights travel with their edge during a swap: same multiset of nonzero values."""
    adj = weighted_erdos_renyi(40, 0.2, seed=3)
    rewired = swap_rewire_connectome(adj, n_swaps=30, seed=4)

    orig = np.sort(adj[np.triu_indices_from(adj, k=1)])
    new = np.sort(rewired[np.triu_indices_from(rewired, k=1)])

    np.testing.assert_allclose(orig[orig != 0], new[new != 0])


def test_swap_rewire_connectome_conserves_edge_count():
    """Each swap trades 2 edges for 2 different edges: total edge count is exact."""
    adj = erdos_renyi(50, 0.15, seed=1)
    rewired = swap_rewire_connectome(adj, n_swaps=30, seed=2)

    assert np.count_nonzero(rewired) == np.count_nonzero(adj)


def test_swap_rewire_connectome_respects_mask_both_directions():
    """
    A masked present edge must never be deleted, and a masked absent pair
    must never receive a new edge.
    """
    n = 30
    adj = erdos_renyi(n, 0.3, seed=5).astype(float)
    adj[0, 1] = adj[1, 0] = 1.0  # force a protected edge to exist
    adj[2, 3] = adj[3, 2] = 0.0  # force a protected non-edge to stay absent

    mask = np.zeros_like(adj, dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[2, 3] = mask[3, 2] = True

    rewired = swap_rewire_connectome(adj, n_swaps=15, mask=mask, seed=6)

    assert rewired[0, 1] == 1.0 and rewired[1, 0] == 1.0
    assert rewired[2, 3] == 0.0 and rewired[3, 2] == 0.0


def test_swap_rewire_connectome_reproducible_with_seed():
    """Same seed must give bit-identical output across repeated calls."""
    adj = erdos_renyi(40, 0.2, seed=7)
    r1 = swap_rewire_connectome(adj, n_swaps=30, seed=8)
    r2 = swap_rewire_connectome(adj, n_swaps=30, seed=8)
    assert np.array_equal(r1, r2)


def test_perturb_connectome_topology_net_edge_count():
    """Total edge count must change by exactly n_additions - n_deletions."""
    adj = erdos_renyi(50, 0.15, seed=1)
    n_edges_before = np.count_nonzero(adj) // 2

    rewired = perturb_connectome_topology(adj, n_deletions=10, n_additions=15, seed=2)
    n_edges_after = np.count_nonzero(rewired) // 2

    assert n_edges_after == n_edges_before - 10 + 15


def test_perturb_connectome_topology_changes_degree_sequence():
    """Unlike swap_rewire_connectome, degree sequence is not preserved in general."""
    adj = erdos_renyi(50, 0.2, seed=1)
    rewired = perturb_connectome_topology(adj, n_deletions=10, n_additions=5, seed=2)
    assert not np.array_equal(degree_sequence(adj), degree_sequence(rewired))


def test_perturb_connectome_topology_weighted_deletion_is_deterministic():
    """A deletion_weights array with all mass on one edge must always select it."""
    adj = erdos_renyi(20, 0.3, seed=1).astype(float)
    edges = np.argwhere(np.triu(adj, k=1))
    i, j = edges[0]

    weights = np.zeros_like(adj)
    weights[i, j] = weights[j, i] = 1.0

    rewired = perturb_connectome_topology(adj, n_deletions=1, n_additions=0, deletion_weights=weights, seed=3)
    assert rewired[i, j] == 0 and rewired[j, i] == 0


def test_perturb_connectome_topology_weighted_addition_is_deterministic():
    """An addition_weights array with all mass on one non-edge must always select it."""
    n = 20
    adj = erdos_renyi(n, 0.3, seed=1).astype(float)
    non_edges = np.argwhere(np.triu(adj == 0, k=1))
    i, j = non_edges[0]

    weights = np.zeros_like(adj)
    weights[i, j] = weights[j, i] = 1.0

    rewired = perturb_connectome_topology(adj, n_deletions=0, n_additions=1, addition_weights=weights, seed=4)
    assert rewired[i, j] != 0 and rewired[j, i] != 0


def test_perturb_connectome_topology_respects_mask_both_directions():
    """A masked present edge must never be deleted, and a masked absent pair must never receive a new edge."""
    n = 30
    adj = erdos_renyi(n, 0.3, seed=5).astype(float)
    adj[0, 1] = adj[1, 0] = 1.0  # force a protected edge to exist
    adj[2, 3] = adj[3, 2] = 0.0  # force a protected non-edge to stay absent

    mask = np.zeros_like(adj, dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[2, 3] = mask[3, 2] = True

    rewired = perturb_connectome_topology(adj, n_deletions=5, n_additions=5, mask=mask, seed=6)

    assert rewired[0, 1] == 1.0 and rewired[1, 0] == 1.0
    assert rewired[2, 3] == 0.0 and rewired[3, 2] == 0.0


def test_perturb_connectome_topology_reproducible_with_seed():
    """Same seed must give bit-identical output across repeated calls."""
    adj = erdos_renyi(40, 0.2, seed=7)
    r1 = perturb_connectome_topology(adj, n_deletions=10, n_additions=10, seed=8)
    r2 = perturb_connectome_topology(adj, n_deletions=10, n_additions=10, seed=8)
    assert np.array_equal(r1, r2)


def test_perturb_connectome_topology_distance_biased_sampling():
    """
    Distance-shaped weights (short-favoring additions, long-favoring deletions --
    the same shape mindmap.analysis.connectivity.metric.dmri_uncertainty_weights_from_source_space
    builds from real dipole coordinates) must bias added edges shorter, and deleted
    edges longer, than the population mean distance among all non-edges/edges
    respectively (comparing against the fixed population mean rather than a second
    random uniform draw avoids two independent samples' variance stacking).
    """
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.uniform(0, 100, size=(n, 3))
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    triu = np.triu_indices(n, k=1)

    adj = erdos_renyi(n, 0.2, seed=1).astype(float)
    is_edge = adj[triu] != 0
    population_edge_dist = dist[triu][is_edge].mean()
    population_nonedge_dist = dist[triu][~is_edge].mean()

    addition_weights = np.exp(-0.05 * dist)
    deletion_weights = dist ** 2
    n_del = n_add = 60

    biased = perturb_connectome_topology(
        adj, n_deletions=n_del, n_additions=n_add,
        deletion_weights=deletion_weights, addition_weights=addition_weights, seed=2,
    )

    added = (biased != 0) & (adj == 0)
    deleted = (biased == 0) & (adj != 0)

    assert dist[triu][added[triu]].mean() < population_nonedge_dist
    assert dist[triu][deleted[triu]].mean() > population_edge_dist


def test_perturb_connectome_weights_noop_at_sigma_zero():
    """sigma=0 must be an exact no-op."""
    adj = weighted_erdos_renyi(30, 0.2, seed=10)
    perturbed = perturb_connectome_weights(adj, sigma=0)
    np.testing.assert_allclose(perturbed, adj)


def test_perturb_connectome_weights_preserves_topology():
    """Weight noise must never create or remove an edge, only rescale existing ones."""
    adj = weighted_erdos_renyi(30, 0.2, seed=11)
    perturbed = perturb_connectome_weights(adj, sigma=0.5, seed=12)

    assert np.array_equal(adj != 0, perturbed != 0)
    assert not np.allclose(adj, perturbed)


def test_perturb_connectome_weights_respects_mask():
    """A masked edge's weight must be left exactly as-is."""
    n = 20
    adj = weighted_erdos_renyi(n, 0.3, seed=13)
    adj[0, 1] = adj[1, 0] = 5.0

    mask = np.zeros_like(adj, dtype=bool)
    mask[0, 1] = mask[1, 0] = True

    perturbed = perturb_connectome_weights(adj, sigma=1.0, mask=mask, seed=14)

    assert perturbed[0, 1] == 5.0 and perturbed[1, 0] == 5.0


def test_perturb_connectome_weights_reproducible_with_seed():
    """Same seed must give bit-identical output across repeated calls."""
    adj = weighted_erdos_renyi(30, 0.2, seed=15)
    p1 = perturb_connectome_weights(adj, sigma=0.3, seed=16)
    p2 = perturb_connectome_weights(adj, sigma=0.3, seed=16)
    np.testing.assert_allclose(p1, p2)


def test_permute_connectome_weights_preserves_multiset_and_topology():
    """Shuffling weights must leave topology and the weight multiset untouched."""
    adj = weighted_erdos_renyi(30, 0.2, seed=20)
    permuted = permute_connectome_weights(adj, seed=21)

    assert np.array_equal(adj != 0, permuted != 0)

    orig = np.sort(adj[np.triu_indices_from(adj, k=1)])
    new = np.sort(permuted[np.triu_indices_from(permuted, k=1)])

    np.testing.assert_allclose(orig[orig != 0], new[new != 0])


def test_permute_connectome_weights_respects_mask():
    """A masked edge's weight must stay in place and out of the shuffled pool."""
    n = 20
    adj = weighted_erdos_renyi(n, 0.3, seed=22)
    adj[0, 1] = adj[1, 0] = 999.0  # distinctive protected weight

    mask = np.zeros_like(adj, dtype=bool)
    mask[0, 1] = mask[1, 0] = True

    permuted = permute_connectome_weights(adj, mask=mask, seed=23)

    assert permuted[0, 1] == 999.0 and permuted[1, 0] == 999.0

    others = permuted.copy()
    others[0, 1] = others[1, 0] = 0
    assert not np.any(others == 999.0)


# -----------------------------------------------------------------------
# Fidelity metrics (correlation/cosine axis broadcast, pattern geometry,
# explained variance, SSIM)
# -----------------------------------------------------------------------

def test_calc_cosine_temporal_axis_does_not_crash():
    """
    Regression test: calc_cosine used to compute norms without keepdims,
    which silently relied on axis=0's norm shape happening to align with
    the array's trailing dimension -- axis=1 (temporal mode: per-channel
    norm across time) had no such coincidence and raised a broadcast
    ValueError. Identical arrays should give cosine similarity 1 per channel.
    """
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(6, 40))

    cos = calc_cosine(arr, arr, axis=1)

    assert cos.shape == (6,)
    np.testing.assert_allclose(cos, 1.0)


def test_calc_pattern_geometry_correlation_identical():
    """Identical input must score ~1: the pattern geometry is trivially preserved."""
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(8, 50))

    score = calc_pattern_geometry_correlation(arr, arr)

    assert score == pytest.approx(1.0, abs=1e-6)


def test_calc_explained_variance_identical_and_scale_sensitive():
    """
    Identical input scores a perfect 1.0. A perfectly-shaped but rescaled
    copy (half amplitude) must be penalized -- unlike correlation, which
    stays 1.0 under any positive rescaling.
    """
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(8, 50))

    assert calc_explained_variance(arr, arr) == pytest.approx(1.0)

    r2_scaled = calc_explained_variance(arr, arr * 0.5)
    corr_scaled = np.corrcoef(arr.ravel(), (arr * 0.5).ravel())[0, 1]

    assert r2_scaled < 1.0
    assert corr_scaled == pytest.approx(1.0)


def test_calc_ssim_identical_and_small_channel_count():
    """Identical input scores ~1, and a small channel count (5) must not raise a win_size error."""
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(8, 60))

    assert calc_ssim(arr, arr) == pytest.approx(1.0)

    small_arr = rng.normal(size=(5, 60))
    assert calc_ssim(small_arr, small_arr) == pytest.approx(1.0)


def test_polarity_invariant_correlation_and_cosine():
    """
    A perfectly anti-correlated pair (an overall sign flip) scores -1
    without polarity_invariant, and +1 with it -- an unrelated pair stays
    near 0 either way, since abs() of noise is still noise.
    """
    rng = np.random.default_rng(4)
    arr = rng.normal(size=(6, 40))
    flipped = -arr

    assert calc_correlation(arr, flipped, axis=None) == pytest.approx(-1.0)
    assert calc_correlation(arr, flipped, axis=None, polarity_invariant=True) == pytest.approx(1.0)

    cos = calc_cosine(arr, flipped, axis=1)
    cos_pi = calc_cosine(arr, flipped, axis=1, polarity_invariant=True)
    np.testing.assert_allclose(cos, -1.0)
    np.testing.assert_allclose(cos_pi, 1.0)

    unrelated = rng.normal(size=(6, 40))
    corr_unrelated = calc_correlation(arr, unrelated, axis=None)
    corr_unrelated_pi = calc_correlation(arr, unrelated, axis=None, polarity_invariant=True)
    assert corr_unrelated_pi == pytest.approx(abs(corr_unrelated))


def test_polarity_invariant_explained_variance():
    """
    A perfectly anti-correlated pair scores strongly negative R^2 (SS_res is
    inflated by the flip, not just sign-negated) without polarity_invariant,
    and a perfect 1.0 with it, since the sign-flipped candidate has zero
    residual. An unrelated pair isn't rescued either way, since flipping a
    decorrelated signal doesn't make it a better predictor.
    """
    rng = np.random.default_rng(5)
    arr = rng.normal(size=(6, 40))
    flipped = -arr

    r2 = calc_explained_variance(arr, flipped, axis=None)
    r2_pi = calc_explained_variance(arr, flipped, axis=None, polarity_invariant=True)

    assert r2 < -1.0
    assert r2_pi == pytest.approx(1.0)

    unrelated = rng.normal(size=(6, 40))
    r2_unrelated = calc_explained_variance(arr, unrelated, axis=None)
    r2_unrelated_pi = calc_explained_variance(arr, unrelated, axis=None, polarity_invariant=True)
    assert r2_unrelated_pi == pytest.approx(max(
        r2_unrelated,
        calc_explained_variance(arr, -unrelated, axis=None),
    ))
