"""
This module performs unit tests for the analysis/connectivity/process.py module.
"""

import numpy as np
import pytest

from mindmap.analysis.connectivity.process import connectome_from_atlas


class FakeParcellation:
    """
    Minimal stand-in for `Parcellation`: labels points by which side of
    x=0 they fall on (region 1: x<0, region 2: x>=0), matching the
    1-indexed region-id convention `connectome_from_atlas` expects.
    """
    def label_points(self, points, return_id=True):
        assert return_id
        return np.where(points[:, 0] < 0, 1, 2)


@pytest.fixture
def two_region_source_space():
    """
    5 dipoles along the x-axis: 2 in region A (x<0), 3 in region B (x>=0).
    All in a single fake hemisphere; positions in [m] (as real
    mne.SourceSpaces store them), to be scaled to [mm] internally.
    """
    positions_mm = np.array([
        [-3, 0, 0],
        [-2, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
    ], dtype=float)
    positions_m = positions_mm / 1000.0

    return [
        {"rr": positions_m, "vertno": np.arange(5), "nuse": 5},
        {"rr": np.zeros((0, 3)), "vertno": np.array([], dtype=int), "nuse": 0},
    ]


def test_connectome_from_atlas_full_sparsity_matches_expected_weights(two_region_source_space):
    """
    With sparsity=1.0 every cross-region dipole pair should be kept, with
    weight = pairwise_distance - min_pairwise_distance + atlas_weight.
    """
    atlas_matrix = np.array([[0, 10], [10, 0]], dtype=float)
    ras2tkras = np.eye(4)

    con = connectome_from_atlas(
        two_region_source_space, FakeParcellation(), atlas_matrix, ras2tkras,
        sparsity=1.0, deterministic=True,
    )

    assert con.shape == (5, 5)
    assert np.allclose(con, con.T)

    # region A = dipoles 0,1 (x=-3,-2); region B = dipoles 2,3,4 (x=1,2,3)
    # min pairwise cross-region distance is |(-2)-1| = 3
    expected = {
        (0, 2): 4 - 3 + 10, (0, 3): 5 - 3 + 10, (0, 4): 6 - 3 + 10,
        (1, 2): 3 - 3 + 10, (1, 3): 4 - 3 + 10, (1, 4): 5 - 3 + 10,
    }
    for (i, j), value in expected.items():
        assert con[i, j] == pytest.approx(value)
        assert con[j, i] == pytest.approx(value)


def test_connectome_from_atlas_no_intra_region_edges(two_region_source_space):
    """Only cross-region pairs are ever considered: intra-region blocks stay zero."""
    atlas_matrix = np.array([[0, 10], [10, 0]], dtype=float)
    ras2tkras = np.eye(4)

    con = connectome_from_atlas(
        two_region_source_space, FakeParcellation(), atlas_matrix, ras2tkras,
        sparsity=1.0, deterministic=True,
    )

    assert con[0, 1] == 0   # both region A
    assert con[2, 3] == con[2, 4] == con[3, 4] == 0   # all region B


def test_connectome_from_atlas_sparsity_keeps_closest_pair_only(two_region_source_space):
    """
    With sparsity low enough that k_keep=1, deterministic mode should
    keep only the single closest cross-region dipole pair.
    """
    atlas_matrix = np.array([[0, 10], [10, 0]], dtype=float)
    ras2tkras = np.eye(4)

    con = connectome_from_atlas(
        two_region_source_space, FakeParcellation(), atlas_matrix, ras2tkras,
        sparsity=1 / 6, deterministic=True,
    )

    nonzero = np.argwhere(np.triu(con, k=1) > 0)
    assert len(nonzero) == 1
    i, j = nonzero[0]
    assert (i, j) == (1, 2)   # dipoles at x=-2 and x=1: closest cross-region pair
    assert con[1, 2] == pytest.approx(10.0)   # dist(3) - min(3) + W(10)


def test_connectome_from_atlas_skips_unconnected_regions(two_region_source_space):
    """A zero atlas entry between two regions should produce no dipole-level edges."""
    atlas_matrix = np.array([[0, 0], [0, 0]], dtype=float)
    ras2tkras = np.eye(4)

    con = connectome_from_atlas(
        two_region_source_space, FakeParcellation(), atlas_matrix, ras2tkras,
        sparsity=1.0, deterministic=True,
    )

    assert np.all(con == 0)


def test_connectome_from_atlas_requires_sparsity_range_with_strength_matrix(two_region_source_space):
    """strength_matrix without sparsity_range is ambiguous and should raise."""
    atlas_matrix = np.array([[0, 10], [10, 0]], dtype=float)
    strength_matrix = np.array([[0, 50], [50, 0]], dtype=float)
    ras2tkras = np.eye(4)

    with pytest.raises(ValueError):
        connectome_from_atlas(
            two_region_source_space, FakeParcellation(), atlas_matrix, ras2tkras,
            strength_matrix=strength_matrix,
        )


class FakeParcellation3:
    """Labels points into 3 regions by x thresholds: A (x<-5), B (-5<=x<10), C (x>=10)."""
    def label_points(self, points, return_id=True):
        assert return_id
        x = points[:, 0]
        return np.where(x < -5, 1, np.where(x < 10, 2, 3))


@pytest.fixture
def three_region_source_space():
    """
    6 dipoles: 2 in region A (x=-10,-9), 2 in region B (x=-1,0), 2 in
    region C (x=20,21). A-B and A-C are atlas-connected (with different
    strengths below), B-C is not.
    """
    positions_mm = np.array([
        [-10, 0, 0], [-9, 0, 0],
        [-1, 0, 0], [0, 0, 0],
        [20, 0, 0], [21, 0, 0],
    ], dtype=float)
    positions_m = positions_mm / 1000.0

    return [
        {"rr": positions_m, "vertno": np.arange(6), "nuse": 6},
        {"rr": np.zeros((0, 3)), "vertno": np.array([], dtype=int), "nuse": 0},
    ]


def test_connectome_from_atlas_strength_modulated_sparsity(three_region_source_space):
    """
    A-B (weak, strength=30) and A-C (strong, strength=90) both have 2x2=4
    possible dipole pairs. With sparsity_range=(0.1, 1.0) and only these
    two connected pairs, percentile rank puts A-B at sparsity=0.1
    (k_keep=1) and A-C at sparsity=1.0 (k_keep=4, i.e. all kept) -- so
    the stronger atlas connection should end up denser at dipole level,
    despite both region pairs being the same size.
    """
    atlas_matrix = np.array([
        [0, 5, 5],
        [5, 0, 0],
        [5, 0, 0],
    ], dtype=float)
    strength_matrix = np.array([
        [0, 30, 90],
        [30, 0, 0],
        [90, 0, 0],
    ], dtype=float)
    ras2tkras = np.eye(4)

    con = connectome_from_atlas(
        three_region_source_space, FakeParcellation3(), atlas_matrix, ras2tkras,
        strength_matrix=strength_matrix, sparsity_range=(0.1, 1.0), deterministic=True,
    )

    ab_block = con[0:2, 2:4]
    ac_block = con[0:2, 4:6]
    bc_block = con[2:4, 4:6]

    assert np.count_nonzero(ab_block) == 1   # weak -> low sparsity -> k_keep=1
    assert np.count_nonzero(ac_block) == 4   # strong -> high sparsity -> all kept
    assert np.all(bc_block == 0)             # atlas-unconnected pair untouched
