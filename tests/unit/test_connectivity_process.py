"""
This module performs unit tests for the analysis/connectivity/process.py module.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from mindmap.analysis.connectivity.process import connectome_from_atlas, connectome_from_tracts


class FakeTracts:
    """Minimal stand-in for a loaded tractogram: just needs `.streamlines`."""
    def __init__(self, streamlines):
        self.streamlines = streamlines


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


@pytest.fixture
def four_dipole_source_space():
    """
    4 dipoles along the x-axis at 0, 2, 10, 12mm, all in one fake hemisphere.
    `dist` (cortical-surface geodesic distance, needed by the `lengths=False`
    Gaussian-smoothing path) is just the along-axis distance here, in
    [m] -- matching real mne source spaces, since `_geodesic_gaussian_kernel`
    converts to [mm] via `* 1000` (same convention as `connectome_from_geometry`).
    """
    positions_mm = np.array([[0, 0, 0], [2, 0, 0], [10, 0, 0], [12, 0, 0]], dtype=float)
    positions_m = positions_mm / 1000.0
    x_m = positions_m[:, 0]
    dist = sp.csr_matrix(np.abs(x_m[:, None] - x_m[None, :]))

    return [
        {"rr": positions_m, "vertno": np.arange(4), "nuse": 4, "dist": dist},
        {"rr": np.zeros((0, 3)), "vertno": np.array([], dtype=int), "nuse": 0,
         "dist": sp.csr_matrix((0, 0))},
    ]


def test_connectome_from_tracts_requires_radius_for_source_space(four_dipole_source_space):
    """A source space without a radius is ambiguous and should raise."""
    tracts = FakeTracts([np.array([[0.5, 0, 0], [10.5, 0, 0]])])
    with pytest.raises(AssertionError):
        connectome_from_tracts(tracts, four_dipole_source_space, np.eye(4), ras2tkras=np.eye(4))


def test_connectome_from_tracts_connects_by_nearest_neighbor_and_smooths(four_dipole_source_space):
    """
    `lengths=False` snaps each endpoint to its single nearest dipole (0 and
    2, both within radius=3) rather than fanning out to every dipole in
    reach (that fan-out is only for `lengths=True`, see
    `test_connectome_from_tracts_length_is_plain_average`), then spreads
    that raw nearest-neighbor connection to geometric neighbors via a
    Gaussian smoothing kernel (Mansour et al. 2021).
    """
    tracts = FakeTracts([np.array([[0.5, 0, 0], [10.5, 0, 0]])])

    con = connectome_from_tracts(
        tracts, four_dipole_source_space, np.eye(4), ras2tkras=np.eye(4), radius=3.0,
    )

    assert con.shape == (4, 4)
    assert np.allclose(con, con.T)
    assert np.all(np.diagonal(con) == 0)

    # smoothing only ever moves mass along the (0,2) connection this
    # streamline actually contributed -- it never bridges within a side
    assert con[0, 1] == 0
    assert con[2, 3] == 0

    # the directly-snapped pair keeps the most weight; the two "one hop
    # away" pairs match exactly, by this fixture's symmetric geometry
    assert con[0, 2] > con[0, 3] > 0
    assert con[0, 2] > con[1, 3] > 0
    assert con[0, 3] == pytest.approx(con[1, 2])

    # total streamline mass is conserved by the kernel (rows sum to 1). The
    # raw (pre-smoothing) connectome stores this one streamline's edge
    # symmetrically (con[0,2]=con[2,0]=1, i.e. 2 units of raw mass), and
    # summing the final symmetric result again counts each undirected edge
    # from both sides, for a total of 2*2=4
    assert con.sum() == pytest.approx(4.0)


def test_connectome_from_tracts_drops_orphaned_streamlines(four_dipole_source_space):
    """A streamline with no dipole within radius of an endpoint contributes nothing."""
    tracts = FakeTracts([np.array([[0.5, 0, 0], [10.5, 0, 0]])])

    con = connectome_from_tracts(
        tracts, four_dipole_source_space, np.eye(4), ras2tkras=np.eye(4), radius=0.3,
    )

    assert np.all(con == 0)


def test_connectome_from_tracts_length_is_plain_average():
    """
    Two streamlines share the (A,D) edge. Even though they have different
    fan-out sizes at their start (streamline 1 reaches 2 dipoles, streamline
    2 reaches 3), the reported length at (A,D) is the plain mean of their
    two lengths -- a streamline's length is a property of that streamline,
    not a mass, so it isn't down-weighted just because its own endpoint had
    more candidate dipoles in reach. (C,D), only supported by streamline 2,
    keeps exactly streamline 2's own length.
    """
    # A, B, C close together (start-side candidates), D far away (end-side, always alone)
    positions_mm = np.array([
        [0, 0, 0],      # A
        [1, 0, 0],      # B
        [2, 0, 0],      # C
        [100, 0, 0],    # D
    ], dtype=float)
    source_space = [
        {"rr": positions_mm / 1000.0, "vertno": np.arange(4), "nuse": 4},
        {"rr": np.zeros((0, 3)), "vertno": np.array([], dtype=int), "nuse": 0},
    ]
    radius = 1.5

    # streamline 1: start at x=0.4 -> reaches A (0.4mm), B (0.6mm), not C (1.6mm > radius)
    stream1 = np.array([[0.4, 0, 0], [99.0, 0, 0]])
    # streamline 2: start at x=1.0 -> reaches A (1.0mm), B (0mm), C (1.0mm), all <= radius
    stream2 = np.array([[1.0, 0, 0], [100.5, 0, 0]])

    tracts = FakeTracts([stream1, stream2])
    len1 = np.linalg.norm(stream1[1] - stream1[0])
    len2 = np.linalg.norm(stream2[1] - stream2[0])

    con = connectome_from_tracts(
        tracts, source_space, np.eye(4), ras2tkras=np.eye(4), radius=radius, lengths=True,
    )

    A, C, D = 0, 2, 3

    expected_AD = np.mean([len1, len2])   # plain mean, not weighted by fan-out size
    expected_CD = len2                    # only stream2 supports C

    assert con[A, D] == pytest.approx(expected_AD)
    assert con[C, D] == pytest.approx(expected_CD)

    w1 = 1.0 / (2 * 1)
    w2 = 1.0 / (3 * 1)
    fanout_weighted = (w1 * len1 + w2 * len2) / (w1 + w2)
    assert expected_AD != pytest.approx(fanout_weighted)   # not the old fan-out-weighted average
