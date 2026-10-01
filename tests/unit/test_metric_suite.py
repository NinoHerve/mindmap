"""
This module performs unit tests for MetricSuite.stat (mindmap/metric/stat.py)
-- the wilcoxon/ttest/mannwhitneyu/paired_ttest/ttest_ess/cluster_permutation/
normality_test/anova cached, typed-result stat entry points -- and
MetricSuite.plot (mindmap/metric/plot.py) -- the raw (box/violin/bar/line/
scatter/distribution/histogram) and annotated (group_comparison/
cluster_permutation/normality_test/anova) plotting accessor built on top of
them.
"""

import json

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mindmap.metric import MetricSuite
from mindmap.metric.results import PairwiseResult, ClusterResult, NormalityResult, AnovaResult
import mindmap.metric.results as results_module


def _make_compare_suite(n_regions=20, seed=0):
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_regions)]
    reference_vals = rng.normal(0, 1, size=n_regions)
    shifted_vals = reference_vals - 2.0 + rng.normal(0, 0.1, size=n_regions)
    noise_vals = reference_vals + rng.normal(0, 1, size=n_regions)

    suite.register("outcome", reference_vals, Method="reference", region=regions)
    suite.register("outcome", shifted_vals, Method="shifted", region=regions)
    suite.register("outcome", noise_vals, Method="noise", region=regions)
    return suite


def test_metricsuite_wilcoxon_returns_expected_columns():
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]),
        pairing_col="region",
    )

    expected_cols = {
        "group_a", "group_b", "n_pairs", "statistic", "p_value", "p_corrected",
        "effect_size", "effect_size_type", "significant",
    }
    assert expected_cols.issubset(set(result.table.columns))
    assert set(result.table["group_b"]) == {"shifted", "noise"}
    assert (result.table["group_a"] == "reference").all()


def test_metricsuite_wilcoxon_requires_comparisons():
    """`comparisons` has no default -- omitting it entirely is a TypeError, not a silent fallback."""
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.stat.wilcoxon("outcome", group_by="Method", pairing_col="region")


def test_metricsuite_wilcoxon_requires_pairing_col():
    """`pairing_col` has no default either -- wilcoxon is inherently a paired test."""
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "shifted")])


def test_metricsuite_wilcoxon_requires_group_by():
    """`group_by` has no default -- omitting it entirely is a TypeError, not a silent fallback."""
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.stat.wilcoxon("outcome", comparisons=[("reference", "shifted")], pairing_col="region")


def test_metricsuite_wilcoxon_comparisons_supports_many_to_many():
    """Without a fixed reference, `comparisons` can name arbitrary pairs, not just one-vs-many."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "shifted"), ("shifted", "noise")],
        pairing_col="region",
    )

    assert set(zip(result.table["group_a"], result.table["group_b"])) == {
        ("reference", "shifted"), ("shifted", "noise"),
    }


def test_metricsuite_wilcoxon_has_no_plot_kwargs():
    """wilcoxon is pure computation -- plotting lives entirely in suite.plot.group_comparison."""
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.stat.wilcoxon(
            "outcome", group_by="Method",
            comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]),
            pairing_col="region", plot=True,
        )


def test_metricsuite_stat_methods_moved_off_suite_directly():
    """Every stat entry point lives under suite.stat now, not on the suite itself."""
    suite = _make_compare_suite()
    names = [
        "wilcoxon", "ttest", "mannwhitneyu", "paired_ttest", "ttest_ess",
        "cluster_permutation", "normality_test", "anova",
        "build_comparisons", "list_cached_stats", "clear_stats_cache",
    ]
    for name in names:
        assert not hasattr(suite, name), f"MetricSuite.{name} should have moved to suite.stat.{name}"


# --- StatAccessor.build_comparisons -----------------------------------------

def test_stat_accessor_build_comparisons_one_reference_many_competitors():
    pairs = MetricSuite().stat.build_comparisons("cset", ["wcset", "mne"])
    assert pairs == [("cset", "wcset"), ("cset", "mne")]


def test_stat_accessor_build_comparisons_accepts_bare_strings():
    pairs = MetricSuite().stat.build_comparisons("cset", "wcset")
    assert pairs == [("cset", "wcset")]


def test_stat_accessor_build_comparisons_skips_self_and_duplicate_pairs():
    """Overlapping references/competitors lists must not produce a self-comparison or the same pair twice (in either order)."""
    pairs = MetricSuite().stat.build_comparisons(["cset", "wcset"], ["wcset", "mne"])
    assert pairs == [("cset", "wcset"), ("cset", "mne"), ("wcset", "mne")]


def _make_compare_suite_with_reps(n_regions=6, n_reps=4, seed=0):
    """
    Unlike `_make_compare_suite` (one row per region already), each region here
    carries `n_reps` repeated rows -- the shape needed to actually exercise
    the pairwise tests' pre-test averaging, since with only one row per region
    there's nothing to collapse and a stale "plot the raw df" bug wouldn't show.
    """
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_regions)]
    for label, shift in [("reference", 0.0), ("shifted", -2.0)]:
        for region in regions:
            vals = rng.normal(0, 1, size=n_reps) + shift
            suite.register("outcome", vals, Method=label, region=region, rep=np.arange(n_reps))
    return suite


def test_metricsuite_wilcoxon_unit_col_collapses_before_testing():
    """unit_col must collapse repeated rows per (group_by, pairing_col, unit_col) before the paired test runs -- n_pairs is the region count, not region x reps."""
    n_regions, n_reps = 6, 4
    suite = _make_compare_suite_with_reps(n_regions=n_regions, n_reps=n_reps)
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "shifted")],
        pairing_col="region", unit_col="rep",
    )
    assert (result.table["n_pairs"] == n_regions).all()


def _make_ar1_diff_suite(n=150, rho=0.8, offset=0.15, seed=0):
    """
    One reference/competitor pair whose paired difference along "unit" is a
    strongly autocorrelated AR(1) series -- units are named so alphabetical
    pivot order does NOT recover the true generation order (mirrors real
    channel-name mis-ordering), so this also exercises subject_order.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=n)
    diff = np.zeros(n)
    for i in range(1, n):
        diff[i] = rho * diff[i - 1] + noise[i]
    diff += offset

    reference_vals = rng.normal(0, 1, size=n)
    competitor_vals = reference_vals - diff

    perm = rng.permutation(n)
    true_order = [f"u{perm[i]:03d}" for i in range(n)]

    suite = MetricSuite()
    for unit, ref_val, comp_val in zip(true_order, reference_vals, competitor_vals):
        suite.register("outcome", ref_val, Method="reference", unit=unit)
        suite.register("outcome", comp_val, Method="competitor", unit=unit)
    return suite, true_order


def test_metricsuite_ttest_ess_and_subject_order_pass_through():
    """
    MetricSuite.stat.ttest_ess must forward subject_order down to
    PairwiseResult.compute/paired_test unchanged: without the true
    subject_order, n_eff should come back close to the nominal n
    (autocorrelation not detected in the scrambled default pivot order);
    with it, n_eff should drop well below n.
    """
    suite, true_order = _make_ar1_diff_suite()

    default_result = suite.stat.ttest_ess(
        "outcome", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="unit",
    )
    explicit_result = suite.stat.ttest_ess(
        "outcome", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="unit",
        subject_order=true_order,
    )

    n = default_result.table.loc[0, "n_pairs"]
    assert default_result.table.loc[0, "n_eff"] == pytest.approx(n, abs=2)
    assert explicit_result.table.loc[0, "n_eff"] < n * 0.6


def test_metricsuite_paired_ttest_returns_expected_columns():
    suite = _make_compare_suite()
    result = suite.stat.paired_ttest(
        "outcome", group_by="Method", comparisons=[("reference", "shifted")], pairing_col="region",
    )
    assert result.table.loc[0, "significant"]
    assert result.table.loc[0, "effect_size_type"] == "cohens_dz"


# --- Independent-samples tests: ttest / mannwhitneyu ------------------------

def _make_independent_suite(n=30, seed=0):
    """Two independent-samples groups, no shared pairing axis."""
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for i in range(n):
        suite.register("outcome", float(rng.normal(0, 1)), Method="a", i=i)
        suite.register("outcome", float(rng.normal(3, 1)), Method="b", i=i)
    return suite


def test_metricsuite_ttest_independent_detects_shifted_groups():
    suite = _make_independent_suite()
    result = suite.stat.ttest("outcome", group_by="Method", comparisons=[("a", "b")])

    assert result.table.loc[0, "significant"]
    assert result.table.loc[0, "n_a"] == 30
    assert result.table.loc[0, "n_b"] == 30
    assert pd.isna(result.table.loc[0, "n_pairs"])
    assert result.table.loc[0, "effect_size_type"] == "cohens_d"


def test_metricsuite_ttest_has_no_pairing_col_parameter():
    """Independent-samples tests have nothing to align across groups -- ttest takes no pairing_col at all."""
    suite = _make_independent_suite()
    with pytest.raises(TypeError):
        suite.stat.ttest("outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="i")


def test_metricsuite_mannwhitneyu_detects_shifted_groups():
    suite = _make_independent_suite()
    result = suite.stat.mannwhitneyu("outcome", group_by="Method", comparisons=[("a", "b")])

    assert result.table.loc[0, "significant"]
    assert result.table.loc[0, "effect_size_type"] == "rank_biserial"


def test_metricsuite_mannwhitneyu_unit_col_collapses_before_testing():
    """unit_col, when given, collapses to one value per (group_by, unit_col) before the independent test runs."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for i in range(10):
        for rep in range(3):
            suite.register("outcome", float(rng.normal(0, 0.2)), Method="a", i=i, rep=rep)
            suite.register("outcome", float(rng.normal(3, 0.2)), Method="b", i=i, rep=rep)

    result = suite.stat.mannwhitneyu("outcome", group_by="Method", comparisons=[("a", "b")], unit_col="i")
    assert result.table.loc[0, "n_a"] == 10
    assert result.table.loc[0, "n_b"] == 10


def test_pairwise_result_compute_rejects_pairing_col_for_independent_test():
    suite = _make_compare_suite()
    df = suite.to_df(metrics="outcome")
    with pytest.raises(ValueError):
        PairwiseResult.compute(
            df, metric="outcome", group_by="Method", comparisons=[("reference", "shifted")],
            test="ttest", pairing_col="region",
        )


def test_pairwise_result_compute_requires_pairing_col_for_paired_test():
    suite = _make_compare_suite()
    df = suite.to_df(metrics="outcome")
    with pytest.raises(ValueError):
        PairwiseResult.compute(
            df, metric="outcome", group_by="Method", comparisons=[("reference", "shifted")], test="wilcoxon",
        )


# --- The motivating scenario: one suite, several metrics, several axes -----

def test_metricsuite_same_suite_groups_different_metrics_by_different_columns():
    """
    One suite holding two metrics, each naturally compared along a different
    column -- "accuracy" across methods, "reaction_time" across subjects.
    `group_by` being a per-call choice (not a suite-level constant) is
    exactly what makes this possible.
    """
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for method, mean in [("A", 0.8), ("B", 0.5)]:
        for region in ["r1", "r2", "r3"]:
            suite.register("accuracy", rng.normal(mean, 0.02), Method=method, region=region)
    for subject, mean in [("sub-01", 0.3), ("sub-02", 0.5), ("sub-03", 0.5)]:
        for region in ["r1", "r2", "r3"]:
            suite.register("reaction_time", rng.normal(mean, 0.01), Subject=subject, region=region)

    by_method = suite.stat.wilcoxon("accuracy", group_by="Method", comparisons=[("A", "B")], pairing_col="region")
    by_subject = suite.stat.wilcoxon(
        "reaction_time", group_by="Subject", comparisons=[("sub-01", "sub-02"), ("sub-01", "sub-03")],
        pairing_col="region",
    )

    assert set(by_method.table["group_b"]) == {"B"}
    assert set(by_subject.table["group_b"]) == {"sub-02", "sub-03"}


# --- Caching ---------------------------------------------------------------

def test_metricsuite_register_clears_stats_cache():
    suite = _make_compare_suite()
    suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    assert not suite.stat.list_cached_stats().empty

    suite.register("outcome", np.zeros(20), Method="extra", region=[f"r{i}" for i in range(20)])
    assert suite.stat.list_cached_stats().empty


def test_metricsuite_clear_stats_cache():
    suite = _make_compare_suite()
    suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    assert not suite.stat.list_cached_stats().empty

    suite.stat.clear_stats_cache()
    assert suite.stat.list_cached_stats().empty


def test_metricsuite_wilcoxon_reuses_cached_raw_across_corrections(monkeypatch):
    """Changing correction/alpha on an already-computed comparison must never rerun paired_test."""
    suite = _make_compare_suite()

    call_count = {"n": 0}
    original = results_module.paired_test

    def counting_paired_test(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "paired_test", counting_paired_test)

    comparisons = suite.stat.build_comparisons("reference", ["shifted", "noise"])
    suite.stat.wilcoxon("outcome", group_by="Method", comparisons=comparisons, pairing_col="region", correction="fdr_bh")
    n_after_first = call_count["n"]
    assert n_after_first > 0

    suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=comparisons, pairing_col="region",
        correction="bonferroni", alpha=0.01,
    )
    assert call_count["n"] == n_after_first


def test_metricsuite_wilcoxon_only_computes_new_pair_in_superset_request(monkeypatch):
    """Requesting a superset of an already-cached pair set only computes the new pair(s)."""
    suite = _make_compare_suite()

    call_count = {"n": 0}
    original = results_module.paired_test

    def counting_paired_test(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "paired_test", counting_paired_test)

    suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "shifted")], pairing_col="region")
    assert call_count["n"] == 1

    suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "shifted"), ("reference", "noise")],
        pairing_col="region",
    )
    assert call_count["n"] == 2  # only "noise" is new -- "shifted" was already cached


def test_metricsuite_wilcoxon_different_group_by_is_part_of_cache_identity(monkeypatch):
    """The same pair name compared under a different `group_by` column must not reuse the other's cached result."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for i in range(10):
        suite.register("outcome", rng.normal(), Method="reference", Subject="reference", region=f"r{i}")
        suite.register("outcome", rng.normal() - 2.0, Method="shifted", Subject="shifted", region=f"r{i}")

    call_count = {"n": 0}
    original = results_module.paired_test

    def counting_paired_test(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "paired_test", counting_paired_test)

    suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "shifted")], pairing_col="region")
    assert call_count["n"] == 1

    suite.stat.wilcoxon("outcome", group_by="Subject", comparisons=[("reference", "shifted")], pairing_col="region")
    assert call_count["n"] == 2  # different group_by -- must recompute


def test_pairwise_result_merge_and_correct():
    """Two independently obtained single-pair results can be merged into a custom family and corrected together."""
    suite = _make_compare_suite()

    r1 = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "shifted")], pairing_col="region")
    r2 = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "noise")], pairing_col="region")

    merged = PairwiseResult.merge([r1, r2])
    corrected = merged.correct("fdr_bh", 0.05)

    assert set(zip(corrected.table["group_a"], corrected.table["group_b"])) == {
        ("reference", "shifted"), ("reference", "noise"),
    }
    assert corrected.table["p_corrected"].notna().all()


def _make_cluster_suite(n_regions=16, n_coord=8, shift_slice=slice(2, 5), seed=0):
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        comp_curve = ref_curve + rng.normal(0, 0.3, size=n_coord)
        comp_curve[shift_slice] -= 3.0
        suite.register("outcome", ref_curve, Method="reference", region=region, level=coord_values)
        suite.register("outcome", comp_curve, Method="competitor", region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_returns_expected_pair():
    suite = _make_cluster_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="region",
        coord_order=list(range(8)), n_permutations=500, seed=0,
    )

    pair = ("reference", "competitor")
    assert isinstance(result, ClusterResult)
    assert set(result.clusters_by_pair.keys()) == {pair}
    assert isinstance(result.clusters_by_pair[pair], pd.DataFrame)


def test_metricsuite_cluster_permutation_requires_group_by():
    suite = _make_cluster_suite()
    with pytest.raises(TypeError):
        suite.stat.cluster_permutation(
            "outcome", coord="level", comparisons=[("reference", "competitor")], pairing_col="region",
        )


def test_metricsuite_cluster_permutation_requires_pairing_col():
    """`pairing_col` has no default -- cluster_permutation_mne is inherently a paired test."""
    suite = _make_cluster_suite()
    with pytest.raises(TypeError):
        suite.stat.cluster_permutation(
            "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")],
        )


def test_metricsuite_cluster_permutation_has_no_plot_kwargs():
    """cluster_permutation is pure computation -- plotting lives entirely in suite.plot.cluster_permutation."""
    suite = _make_cluster_suite()
    with pytest.raises(TypeError):
        suite.stat.cluster_permutation(
            "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")],
            pairing_col="region", plot=True,
        )


def _make_cluster_suite_with_peak(n_regions=20, n_coord=8, peak_idx=4, peak_shift=5.0, base_shift=0.5, seed=0):
    """
    Mild-but-consistent shift at every coordinate (whole curve forms one
    cluster) except one coordinate with a much larger shift -- see
    tests/unit/test_analysis_stats.py's `_make_cluster_df_with_peak` (same
    fixture design, one level up through MetricSuite).
    """
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        comp_curve = ref_curve.copy()
        for c in range(n_coord):
            shift = peak_shift if c == peak_idx else base_shift
            comp_curve[c] -= shift + rng.normal(0, 0.2)
        suite.register("outcome", ref_curve, Method="reference", region=region, level=coord_values)
        suite.register("outcome", comp_curve, Method="competitor", region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_effect_size_summary_pass_through():
    """
    MetricSuite.cluster_permutation must forward effect_size_summary down to
    cluster_permutation_mne unchanged: "peak" must report a bigger
    effect_size than "cluster_mean" on a cluster diluted by many mild
    coordinates plus one much stronger one.
    """
    suite = _make_cluster_suite_with_peak()
    pair = ("reference", "competitor")

    result_mean = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="region",
        coord_order=list(range(8)), n_permutations=300, seed=0, effect_size_summary="cluster_mean",
    )
    result_peak = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="region",
        coord_order=list(range(8)), n_permutations=300, seed=0, effect_size_summary="peak",
    )

    assert "effect_size" in result_mean.clusters_by_pair[pair].columns
    assert result_peak.clusters_by_pair[pair].loc[0, "effect_size"] > result_mean.clusters_by_pair[pair].loc[0, "effect_size"]


def _make_significance_suite(n_regions=20, n_coord=6, seed=0):
    """
    Three methods sharing one `reference`: `worse` is shifted far enough below
    `reference` at every coordinate to form one significant cluster spanning
    the whole range, `same` is indistinguishable from it (no cluster at all)
    -- the two extremes cluster_permutation's multi-pair summary plot's
    per-segment line/marker styling needs to tell apart correctly.
    """
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        worse_curve = ref_curve - 5.0 + rng.normal(0, 0.1, size=n_coord)
        same_curve = ref_curve + rng.normal(0, 0.05, size=n_coord)
        suite.register("outcome", ref_curve, Method="reference", region=region, level=coord_values)
        suite.register("outcome", worse_curve, Method="worse", region=region, level=coord_values)
        suite.register("outcome", same_curve, Method="same", region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_reuses_cached_pairs_across_requests(monkeypatch):
    """A single-competitor call followed by the full multi-competitor request only computes the new pair."""
    suite = _make_significance_suite()

    call_count = {"n": 0}
    original = results_module.cluster_permutation_mne

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "cluster_permutation_mne", counting)

    suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=300, seed=0,
    )
    assert call_count["n"] == 1

    suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=300, seed=0,
    )
    assert call_count["n"] == 2  # only "same" is new


def test_metricsuite_cluster_permutation_multi_pair_effect_size_survives_fdr_pooling():
    """
    Pooling across pairs (multiple competitors requested in one
    cluster_permutation call) concatenates every pair's clusters DataFrame
    and reassigns p_corrected/significant -- effect_size/effect_size_type
    must ride along unchanged, not get dropped.
    """
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=300, seed=0,
    )
    fig, ax = suite.plot.cluster_permutation(result)
    plt.close(fig)

    for clusters in result.clusters_by_pair.values():
        if clusters.empty:
            continue
        assert "effect_size" in clusters.columns
        assert "effect_size_type" in clusters.columns


def test_metricsuite_cluster_permutation_multi_pair_corrects_across_competitors():
    """
    Regression guard for the actual point of this design: cluster p-values
    from different competitors must be pooled into one family and corrected
    together (matching the pairwise tests' own correction="fdr_bh" default
    across their competitors) -- not each competitor left at its own raw,
    uncorrected per-cluster p-value, which would silently under-correct for
    testing multiple competitors against the same reference.
    """
    suite = _make_multi_competitor_cluster_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse_strong"), ("reference", "worse_weak")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=1000, seed=0,
    )

    non_empty = {pair: cl for pair, cl in result.clusters_by_pair.items() if not cl.empty}
    assert len(non_empty) == 2
    pooled_raw_p = pd.concat([cl["p_value"] for cl in non_empty.values()], ignore_index=True)
    assert pooled_raw_p.nunique() >= 2  # otherwise this fixture can't tell "corrected" from "not"

    from mindmap.core.stats import correct_pvalues
    expected_corrected, expected_significant = correct_pvalues(pooled_raw_p, method="fdr_bh", alpha=0.05)

    got_corrected = pd.concat([cl["p_corrected"] for cl in non_empty.values()], ignore_index=True)
    got_significant = pd.concat([cl["significant"] for cl in non_empty.values()], ignore_index=True)

    np.testing.assert_allclose(got_corrected.to_numpy(), expected_corrected)
    assert (got_significant.to_numpy() == expected_significant).all()
    # and the correction must have actually changed at least one value, not been a no-op
    assert not np.allclose(got_corrected.to_numpy(), pooled_raw_p.to_numpy())


def _make_multi_competitor_cluster_suite(n_regions=20, n_coord=6, seed=1):
    """
    Reference + two competitors that each find at least one cluster, at three
    distinct raw p-values (`worse_strong` very significant, `worse_weak` two
    more marginal clusters) -- the shape needed to tell "pooled and corrected
    across competitors" apart from "left at each competitor's own raw p-value",
    since with fewer than two distinct p-values in the family a correction bug
    could go unnoticed (e.g. two identical, already-minimal p-values are
    unchanged by BH correction whether or not it actually ran).
    """
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        strong_curve = ref_curve - 5.0 + rng.normal(0, 0.1, size=n_coord)
        weak_curve = ref_curve - 0.3 + rng.normal(0, 1.2, size=n_coord)
        suite.register("outcome", ref_curve, Method="reference", region=region, level=coord_values)
        suite.register("outcome", strong_curve, Method="worse_strong", region=region, level=coord_values)
        suite.register("outcome", weak_curve, Method="worse_weak", region=region, level=coord_values)

    return suite


def test_metricsuite_plot_cluster_permutation_multi_pair_style_reflects_significance():
    """
    Regression guard for the actual point of the multi-pair summary plot: a
    clearly inferior competitor's drawn segments/markers must all be
    solid/filled, and a statistically indistinguishable one's must all be
    dashed/open -- not just "a plot was returned". Colors are pinned
    explicitly so each competitor's own Line2D artists can be isolated by
    color and checked directly.
    """
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=500, seed=0,
    )
    fig, ax = suite.plot.cluster_permutation(
        result, colors={"reference": "black", "worse": "red", "same": "blue"},
    )

    assert result.clusters_by_pair[("reference", "worse")]["significant"].any()
    assert not result.clusters_by_pair[("reference", "same")]["significant"].any()

    def is_marker(l):
        return len(l.get_xdata()) == 1

    def is_segment(l):
        # a two-point line connecting two *different* x values -- excludes the
        # vertical (same x twice) spread whiskers, which are always solid and
        # aren't part of the significance encoding.
        xdata = l.get_xdata()
        return len(xdata) == 2 and xdata[0] != xdata[1]

    def lines_for(color):
        return [l for l in ax.get_lines() if l.get_color() == color]

    # "worse" (significant everywhere): every segment solid, every marker filled.
    worse_lines = lines_for("red")
    assert worse_lines
    assert all(l.get_linestyle() == "-" for l in worse_lines if is_segment(l))
    assert all(l.get_markerfacecolor() == "red" for l in worse_lines if is_marker(l))

    # "same" (never significant): every segment dashed, every marker open (white face).
    same_lines = lines_for("blue")
    assert same_lines
    assert all(l.get_linestyle() == "--" for l in same_lines if is_segment(l))
    assert all(l.get_markerfacecolor() == "white" for l in same_lines if is_marker(l))

    plt.close(fig)


def test_metricsuite_plot_cluster_permutation_multi_pair_spread_whiskers_toggle():
    """
    show_spread=True (the default -- good practice is to show uncertainty,
    not just a significance verdict) must draw one extra thin whisker Line2D
    per (method, coordinate) pair versus show_spread=False; the whiskers
    aren't part of the significance encoding itself, so this only checks
    they're drawn/omitted correctly, not their style.
    """
    suite = _make_significance_suite(n_coord=6)
    n_methods = 3  # reference + worse + same
    n_coord = 6

    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=500, seed=0,
    )
    fig_with, ax_with = suite.plot.cluster_permutation(result)
    fig_without, ax_without = suite.plot.cluster_permutation(result, show_spread=False)

    assert len(ax_with.get_lines()) - len(ax_without.get_lines()) == n_methods * n_coord
    plt.close(fig_with)
    plt.close(fig_without)


# --- Normality test ----------------------------------------------------------

def _make_normality_suite(n=30, seed=0):
    """One group drawn from a normal distribution, one clearly skewed (exponential) -- shapiro should tell them apart."""
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for region in range(n):
        suite.register("outcome", float(rng.normal()), Method="normal", region=f"r{region}")
        suite.register("outcome", float(rng.exponential()), Method="skewed", region=f"r{region}")
    return suite


def test_metricsuite_normality_test_without_group_by_returns_one_row():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome")

    assert isinstance(result, NormalityResult)
    assert len(result.table) == 1
    assert result.table.loc[0, "group"] is None
    assert result.table.loc[0, "n"] == 60  # both groups pooled


def test_metricsuite_normality_test_with_group_by_returns_one_row_per_group():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")

    assert set(result.table["group"]) == {"normal", "skewed"}
    assert (result.table["n"] == 30).all()


def test_metricsuite_normality_test_detects_non_normal_group():
    """The null hypothesis is normality -- a small p-value/`significant=True` means significantly non-normal, not the reverse."""
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")

    normal_row = result.table[result.table["group"] == "normal"].iloc[0]
    skewed_row = result.table[result.table["group"] == "skewed"].iloc[0]
    assert not normal_row["significant"]
    assert skewed_row["significant"]
    assert skewed_row["p_value"] < normal_row["p_value"]


def test_metricsuite_normality_test_group_by_is_optional():
    """Unlike the pairwise tests/cluster_permutation, group_by is optional here -- omitting it tests the whole population, not an error."""
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome")
    assert isinstance(result, NormalityResult)


def test_metricsuite_normality_test_reuses_cached_groups_across_repeated_calls(monkeypatch):
    """Calling normality_test again with the same group_by/filters reuses every group's cached result -- no new test_normality calls."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for region in range(30):
        suite.register("outcome", float(rng.normal()), Method="a", region=f"r{region}")
        suite.register("outcome", float(rng.normal()), Method="b", region=f"r{region}")

    call_count = {"n": 0}
    original = results_module.test_normality

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "test_normality", counting)

    suite.stat.normality_test("outcome", group_by="Method")
    assert call_count["n"] == 2  # "a" and "b"

    suite.stat.normality_test("outcome", group_by="Method")
    assert call_count["n"] == 2  # both already cached -- no new calls


def test_metricsuite_normality_test_reuses_cached_raw_across_corrections(monkeypatch):
    """Changing correction/alpha on an already-computed test must never rerun test_normality."""
    suite = _make_normality_suite()

    call_count = {"n": 0}
    original = results_module.test_normality

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "test_normality", counting)

    suite.stat.normality_test("outcome", group_by="Method", correction="fdr_bh")
    n_after_first = call_count["n"]
    assert n_after_first > 0

    suite.stat.normality_test("outcome", group_by="Method", correction="bonferroni", alpha=0.01)
    assert call_count["n"] == n_after_first


def test_metricsuite_normality_test_group_by_is_part_of_cache_identity(monkeypatch):
    """The same group name tested under a different group_by column must not reuse the other's cached result."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for i in range(30):
        suite.register("outcome", float(rng.normal()), Method="a", Subject="a", region=f"r{i}")

    call_count = {"n": 0}
    original = results_module.test_normality

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "test_normality", counting)

    suite.stat.normality_test("outcome", group_by="Method")
    assert call_count["n"] == 1

    suite.stat.normality_test("outcome", group_by="Subject")
    assert call_count["n"] == 2  # different group_by -- must recompute


def test_normality_result_merge_and_correct():
    suite = _make_normality_suite()

    r1 = suite.stat.normality_test("outcome", group_by="Method", filters={"Method": "normal"})
    r2 = suite.stat.normality_test("outcome", group_by="Method", filters={"Method": "skewed"})

    merged = NormalityResult.merge([r1, r2])
    corrected = merged.correct("fdr_bh", 0.05)

    assert set(corrected.table["group"]) == {"normal", "skewed"}
    assert corrected.table["p_corrected"].notna().all()


def test_metricsuite_save_load_roundtrip_preserves_normality_stats_cache(tmp_path):
    suite = _make_normality_suite()
    suite.stat.normality_test("outcome", group_by="Method")

    folder = tmp_path/"suite_with_normality_stats"
    suite.save(folder)
    loaded = MetricSuite.load(folder)

    assert len(loaded.stat._stats_cache) == len(suite.stat._stats_cache)
    assert set(loaded.stat.list_cached_stats()["pair"]) == set(suite.stat.list_cached_stats()["pair"])


# --- ANOVA (omnibus) test ----------------------------------------------------

def _make_anova_suite(n_regions=15, seed=0):
    """Four groups sharing a per-region subject effect, one (B) clearly shifted -- both friedman and anova should detect a real omnibus difference."""
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for region in range(n_regions):
        subject_effect = rng.normal()
        suite.register("outcome", subject_effect + rng.normal(0, 0.3), Method="A", region=f"r{region}")
        suite.register("outcome", subject_effect + rng.normal(1.5, 0.3), Method="B", region=f"r{region}")
        suite.register("outcome", subject_effect + rng.normal(0, 0.3), Method="C", region=f"r{region}")
        suite.register("outcome", subject_effect + rng.normal(0, 0.3), Method="D", region=f"r{region}")
    return suite


def test_metricsuite_anova_returns_expected_fields():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")

    assert isinstance(result, AnovaResult)
    assert result.groups == ("A", "B", "C", "D")
    assert result.k == 4
    assert result.n == 15
    assert result.effect_size_type == "kendalls_w"


def test_metricsuite_anova_requires_pairing_col():
    """`pairing_col` has no default -- an omnibus RM test is inherently a paired/matched-groups test."""
    suite = _make_anova_suite()
    with pytest.raises(TypeError):
        suite.stat.anova("outcome", group_by="Method")


def test_metricsuite_anova_detects_real_group_difference():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")
    assert result.significant
    assert result.p_value < 0.05


def test_metricsuite_anova_requires_at_least_three_groups():
    suite = _make_anova_suite()
    with pytest.raises(ValueError):
        suite.stat.anova("outcome", group_by="Method", pairing_col="region", groups=["A", "B"])


def test_metricsuite_anova_test_parameter_selects_parametric_variant():
    suite = _make_anova_suite()
    result_friedman = suite.stat.anova("outcome", group_by="Method", pairing_col="region", test="friedman")
    result_anova = suite.stat.anova("outcome", group_by="Method", pairing_col="region", test="anova")

    assert result_friedman.effect_size_type == "kendalls_w"
    assert result_anova.effect_size_type == "eta_sq_p"
    assert result_anova.significant


def test_metricsuite_anova_groups_parameter_restricts_included_groups():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region", groups=["A", "C", "D"])
    assert result.groups == ("A", "C", "D")
    assert result.k == 3
    assert not result.significant  # A/C/D are all drawn from the same distribution, B (the shifted one) excluded


def test_metricsuite_anova_has_no_correction_parameter():
    """Unlike the pairwise tests/cluster_permutation/normality_test, there's no family to multiple-comparison-correct -- only alpha, no correction method."""
    suite = _make_anova_suite()
    with pytest.raises(TypeError):
        suite.stat.anova("outcome", group_by="Method", pairing_col="region", correction="bonferroni")


def test_metricsuite_anova_reuses_cached_raw_across_alpha_changes(monkeypatch):
    """Changing alpha on an already-computed anova must never rerun repeated_measures_anova."""
    suite = _make_anova_suite()

    call_count = {"n": 0}
    original = results_module.repeated_measures_anova

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "repeated_measures_anova", counting)

    suite.stat.anova("outcome", group_by="Method", pairing_col="region", alpha=0.05)
    assert call_count["n"] == 1

    suite.stat.anova("outcome", group_by="Method", pairing_col="region", alpha=0.01)
    assert call_count["n"] == 1  # alpha-only change -- no recompute


def test_metricsuite_anova_test_choice_is_part_of_cache_identity(monkeypatch):
    """Requesting a different test= must not reuse the other test's cached result."""
    suite = _make_anova_suite()

    call_count = {"n": 0}
    original = results_module.repeated_measures_anova

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "repeated_measures_anova", counting)

    suite.stat.anova("outcome", group_by="Method", pairing_col="region", test="friedman")
    assert call_count["n"] == 1

    suite.stat.anova("outcome", group_by="Method", pairing_col="region", test="anova")
    assert call_count["n"] == 2  # different test -- must recompute


def test_anova_result_correct_returns_new_object():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region", alpha=0.05)
    corrected = result.correct(alpha=1.0)

    assert corrected is not result
    assert corrected.significant
    assert corrected.p_value == result.p_value


def test_metricsuite_save_load_roundtrip_preserves_anova_stats_cache(tmp_path):
    suite = _make_anova_suite()
    suite.stat.anova("outcome", group_by="Method", pairing_col="region")

    folder = tmp_path/"suite_with_anova_stats"
    suite.save(folder)
    loaded = MetricSuite.load(folder)

    assert len(loaded.stat._stats_cache) == len(suite.stat._stats_cache)
    assert set(loaded.stat.list_cached_stats()["pair"]) == set(suite.stat.list_cached_stats()["pair"])


# --- Standalone descriptive utilities --------------------------------------

def test_metricsuite_summary_requires_metric():
    """`metric` has no default -- omitting it entirely is a TypeError, not a silent fallback."""
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.summary()


def test_metricsuite_summary_without_group_by_describes_whole_population():
    suite = _make_compare_suite()
    summary = suite.summary("outcome")
    assert summary.loc["outcome", "count"] == 60  # 3 methods x 20 regions


def test_metricsuite_summary_with_group_by_describes_each_group():
    suite = _make_compare_suite()
    summary = suite.summary("outcome", group_by="Method")
    assert set(summary["Method"]) == {"reference", "shifted", "noise"}
    assert {"mean", "std", "count"}.issubset(summary.columns)
    assert (summary["count"] == 20).all()


def _make_describe_suite():
    """Two metrics with different coord sets -- categorical (Method) and numerical (snr/times)."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for method in ["a", "b", "c"]:
        for snr in [1, 5, 10, 50]:
            suite.register("correlation", rng.normal(), Method=method, snr=snr, region=f"r{snr}")
    for times in np.linspace(0, 1, 5):
        suite.register("latency", rng.normal(), times=times)
    return suite


def test_metricsuite_describe_reports_one_row_per_metric_coord():
    suite = _make_describe_suite()
    described = suite.describe()

    assert set(described["Metric"]) == {"correlation", "latency"}
    # "correlation" carries Method/snr/region; "latency" carries only times -- no cross-contamination
    correlation_coords = set(described.loc[described["Metric"] == "correlation", "Coord"])
    latency_coords = set(described.loc[described["Metric"] == "latency", "Coord"])
    assert correlation_coords == {"Method", "snr", "region"}
    assert latency_coords == {"times"}


def test_metricsuite_describe_classifies_categorical_vs_numerical():
    suite = _make_describe_suite()
    described = suite.describe(metric="correlation")

    method_row = described[described["Coord"] == "Method"].iloc[0]
    snr_row = described[described["Coord"] == "snr"].iloc[0]
    assert method_row["Kind"] == "categorical"
    assert method_row["N unique"] == 3
    assert "a" in method_row["Detail"] and "b" in method_row["Detail"] and "c" in method_row["Detail"]
    assert snr_row["Kind"] == "numerical"
    assert snr_row["Detail"] == "[1, 50]"


def test_metricsuite_describe_metric_filter_scopes_to_one_metric():
    suite = _make_describe_suite()
    described = suite.describe(metric="latency")
    assert set(described["Metric"]) == {"latency"}


def test_metricsuite_describe_empty_suite_returns_empty_frame():
    suite = MetricSuite()
    described = suite.describe()
    assert described.empty
    assert list(described.columns) == ["Metric", "Coord", "Kind", "N unique", "Detail"]


def test_metricsuite_describe_truncates_long_categorical_lists():
    suite = MetricSuite()
    for i in range(12):
        suite.register("outcome", float(i), Method=f"m{i}", region="r0")
    described = suite.describe()
    method_row = described[described["Coord"] == "Method"].iloc[0]
    assert method_row["N unique"] == 12
    assert "more" in method_row["Detail"]


# --- suite.plot: raw views (metric name in, no test, one visual style each) --

def test_metricsuite_plot_box_returns_fig_ax():
    suite = _make_compare_suite()
    fig, ax = suite.plot.box("outcome", group_by="Method")

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_box_requires_group_by():
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.plot.box("outcome")


def test_metricsuite_plot_box_points_toggle():
    """
    points=False must drop the raw-point stripplot overlay entirely (dense
    data, e.g. hundreds of points per box, makes it visual noise) while
    leaving the box plot itself intact; points=True (default) keeps it.
    """
    suite = _make_compare_suite()

    fig_with, ax_with = suite.plot.box("outcome", group_by="Method", points=True)
    fig_without, ax_without = suite.plot.box("outcome", group_by="Method", points=False)

    assert len(ax_with.collections) > 0
    assert len(ax_without.collections) == 0
    plt.close(fig_with)
    plt.close(fig_without)


def test_metricsuite_plot_violin_returns_fig_ax():
    suite = _make_compare_suite()
    fig, ax = suite.plot.violin("outcome", group_by="Method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_bar_returns_fig_ax():
    suite = _make_compare_suite()
    fig, ax = suite.plot.bar("outcome", group_by="Method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_line_returns_fig_ax():
    suite = _make_compare_suite_with_reps()
    fig, ax = suite.plot.line("outcome", group_by="Method", x="rep")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_scatter_returns_fig_ax():
    suite = _make_compare_suite_with_reps()
    fig, ax = suite.plot.scatter("outcome", group_by="Method", x="rep")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_distribution_without_group_by_draws_one_curve():
    """Unlike box/violin/bar/line/scatter, group_by is optional here -- the metric's value is the x-axis, so an ungrouped curve is already meaningful."""
    suite = _make_compare_suite()
    fig, ax = suite.plot.distribution("outcome")
    assert isinstance(fig, plt.Figure)
    assert len(ax.collections) > 0 or len(ax.lines) > 0
    plt.close(fig)


def _artist_alpha(artist):
    """
    An artist's effective alpha -- seaborn's kdeplot/histplot fill bake
    alpha into the facecolor's RGBA (4th channel) rather than calling
    `Artist.set_alpha()`, so `get_alpha()` alone reports `None` even though
    the fill really is translucent.
    """
    alpha = artist.get_alpha()
    if alpha is not None:
        return alpha
    facecolor = artist.get_facecolor()
    rgba = facecolor[0] if getattr(facecolor, "ndim", 1) == 2 else facecolor
    return rgba[3]


def test_metricsuite_plot_distribution_with_group_by_overlays_translucent_curves():
    """group_by overlays one filled, translucent curve per group onto the same axes -- not separate figures."""
    suite = _make_compare_suite()
    fig, ax = suite.plot.distribution("outcome", group_by="Method")
    assert isinstance(fig, plt.Figure)
    assert len(ax.collections) == 3  # reference/shifted/noise, all on one Axes
    assert all(_artist_alpha(c) < 1.0 for c in ax.collections)
    plt.close(fig)


def test_metricsuite_plot_histogram_without_group_by_draws_one_set_of_bars():
    suite = _make_compare_suite()
    fig, ax = suite.plot.histogram("outcome")
    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) > 0
    plt.close(fig)


def test_metricsuite_plot_histogram_with_group_by_overlays_translucent_bars():
    """group_by overlays translucent bars (multiple="layer") onto the same axes -- not stacked or side-by-side."""
    suite = _make_compare_suite()
    fig, ax = suite.plot.histogram("outcome", group_by="Method")
    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) > 0
    assert all(_artist_alpha(p) < 1.0 for p in ax.patches)
    plt.close(fig)


def test_metricsuite_plot_ax_passthrough():
    suite = _make_compare_suite()
    fig_in, ax_in = plt.subplots()

    fig_out, ax_out = suite.plot.box("outcome", group_by="Method", ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


# --- suite.plot: annotated views (a computed result in, never a metric name) -

def test_metricsuite_plot_group_comparison_returns_fig_ax():
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    fig, ax = suite.plot.group_comparison(result)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_group_comparison_infers_reference():
    """Reference is inferred from the result when every pair shares one common group -- no need to repeat it."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    fig, ax = suite.plot.group_comparison(result)  # no reference= given
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_group_comparison_requires_reference_when_ambiguous():
    """An arbitrary many-to-many result with no common group across its pairs can't infer a baseline -- must be told explicitly."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for label in ["a", "b", "c", "d"]:
        for region in ["r1", "r2", "r3"]:
            suite.register("outcome", rng.normal(), Method=label, region=region)

    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b"), ("c", "d")], pairing_col="region",
    )
    with pytest.raises(ValueError):
        suite.plot.group_comparison(result)

    fig, ax = suite.plot.group_comparison(result, reference="a")
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_group_comparison_dots_match_tested_pairs():
    """
    Regression guard: the box plot's raw-point overlay must show the same
    (Method, region)-averaged pairs the Wilcoxon test itself ran on (one dot
    per region per method, matching result.table's own "n_pairs") -- not the
    raw, uncollapsed rows (which would be n_reps x too many and visually
    imply far more spread than the test ever saw).
    """
    n_regions, n_reps = 6, 4
    suite = _make_compare_suite_with_reps(n_regions=n_regions, n_reps=n_reps)
    assert len(suite.to_df()) == 2 * n_regions * n_reps  # sanity: raw data really is repeated

    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "shifted")], pairing_col="region",
    )
    fig, ax = suite.plot.group_comparison(result)
    assert (result.table["n_pairs"] == n_regions).all()

    total_points = sum(len(c.get_offsets()) for c in ax.collections)
    assert total_points == 2 * n_regions  # 2 methods x n_regions, not x n_reps
    plt.close(fig)


def test_metricsuite_plot_group_comparison_uses_given_ax():
    suite = _make_compare_suite()
    fig_in, ax_in = plt.subplots()

    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    fig_out, ax_out = suite.plot.group_comparison(result, ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def test_metricsuite_plot_group_comparison_rejects_cluster_result():
    """A ClusterResult passed to suite.plot.group_comparison() is a clear TypeError, not a confusing crash downstream."""
    suite = _make_cluster_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="region",
        coord_order=list(range(8)), n_permutations=200, seed=0,
    )
    with pytest.raises(TypeError):
        suite.plot.group_comparison(result)


def test_metricsuite_plot_cluster_permutation_single_pair_returns_fig_ax():
    suite = _make_cluster_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="region",
        coord_order=list(range(8)), n_permutations=500, seed=0,
    )
    fig, ax = suite.plot.cluster_permutation(result)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_cluster_permutation_uses_given_ax():
    suite = _make_cluster_suite()
    fig_in, ax_in = plt.subplots()

    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="region",
        coord_order=list(range(8)), n_permutations=500, seed=0,
    )
    fig_out, ax_out = suite.plot.cluster_permutation(result, ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def test_metricsuite_plot_cluster_permutation_multi_pair_returns_fig_ax():
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=500, seed=0,
    )
    fig, ax = suite.plot.cluster_permutation(result)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_cluster_permutation_multi_pair_infers_reference():
    """Multi-pair reference is inferred when every pair shares one common group, same as group_comparison."""
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "worse"), ("reference", "same")],
        pairing_col="region", coord_order=list(range(6)), n_permutations=300, seed=0,
    )
    fig, ax = suite.plot.cluster_permutation(result)  # no reference= given
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_cluster_permutation_multi_pair_requires_reference_when_ambiguous():
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    coord_values = np.arange(6)
    for label in ["a", "b", "c", "d"]:
        for region in [f"r{i}" for i in range(10)]:
            suite.register("outcome", rng.normal(size=6), Method=label, region=region, level=coord_values)

    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("a", "b"), ("c", "d")],
        pairing_col="region", coord_order=list(range(6)), n_permutations=200, seed=0,
    )
    with pytest.raises(ValueError):
        suite.plot.cluster_permutation(result)


def test_metricsuite_plot_cluster_permutation_rejects_pairwise_result():
    """A PairwiseResult passed to suite.plot.cluster_permutation() is a clear TypeError, not a confusing crash downstream."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    with pytest.raises(TypeError):
        suite.plot.cluster_permutation(result)


def test_metricsuite_plot_normality_test_distribution_annotates_each_group():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")
    fig, ax = suite.plot.normality_test(result)

    assert isinstance(fig, plt.Figure)
    assert len(ax.texts) == 2
    plt.close(fig)


def test_metricsuite_plot_normality_test_histogram_kind():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")
    fig, ax = suite.plot.normality_test(result, kind="histogram")

    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) > 0
    assert len(ax.texts) == 2
    plt.close(fig)


def test_metricsuite_plot_normality_test_whole_population_one_annotation():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome")
    fig, ax = suite.plot.normality_test(result)

    assert len(ax.texts) == 1
    assert "whole population" in ax.texts[0].get_text()
    plt.close(fig)


def test_metricsuite_plot_normality_test_colors_significant_group_red():
    """The non-normal ("skewed") group's annotation must be visually distinct (red) from the normal one."""
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")
    fig, ax = suite.plot.normality_test(result)

    by_text = {t.get_text(): t.get_color() for t in ax.texts}
    normal_text = next(t for t in by_text if t.startswith("normal:"))
    skewed_text = next(t for t in by_text if t.startswith("skewed:"))
    assert by_text[skewed_text] == "tab:red"
    assert by_text[normal_text] != "tab:red"
    plt.close(fig)


def test_metricsuite_plot_normality_test_rejects_bad_kind():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")
    with pytest.raises(ValueError):
        suite.plot.normality_test(result, kind="bogus")


def test_metricsuite_plot_normality_test_rejects_non_normality_result():
    """A PairwiseResult passed to suite.plot.normality_test() is a clear TypeError, not a confusing crash downstream."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    with pytest.raises(TypeError):
        suite.plot.normality_test(result)


def test_metricsuite_plot_normality_test_ax_passthrough():
    suite = _make_normality_suite()
    result = suite.stat.normality_test("outcome", group_by="Method")
    fig_in, ax_in = plt.subplots()

    fig_out, ax_out = suite.plot.normality_test(result, ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def test_metricsuite_plot_anova_box_annotates_omnibus_verdict():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")
    fig, ax = suite.plot.anova(result)

    assert isinstance(fig, plt.Figure)
    assert len(ax.texts) == 1
    assert "Friedman" in ax.texts[0].get_text()
    assert "kendalls_w" in ax.texts[0].get_text()
    assert ax.texts[0].get_color() == "tab:red"  # significant in this fixture


def test_metricsuite_plot_anova_dots_match_tested_pairs():
    """
    Regression guard: the box plot's raw-point overlay must show the same
    (Method, region)-averaged pairs the omnibus test itself ran on (one
    dot per region per method), not the raw, uncollapsed rows (which would
    be n_reps x too many and visually imply far more spread than the test
    ever saw) -- matches plot.group_comparison's own such guard.
    """
    n_regions, n_reps = 6, 4
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for method in ["A", "B", "C"]:
        for region in range(n_regions):
            vals = rng.normal(size=n_reps)
            suite.register("outcome", vals, Method=method, region=f"r{region}", rep=np.arange(n_reps))
    assert len(suite.to_df()) == 3 * n_regions * n_reps  # sanity: raw data really is repeated

    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")
    fig, ax = suite.plot.anova(result)

    total_points = sum(len(c.get_offsets()) for c in ax.collections)
    assert total_points == 3 * n_regions  # 3 methods x n_regions, not x n_reps
    plt.close(fig)


def test_metricsuite_plot_anova_not_significant_uses_neutral_color():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region", groups=["A", "C", "D"])  # excludes the shifted "B"
    fig, ax = suite.plot.anova(result)

    assert not result.significant
    assert ax.texts[0].get_color() != "tab:red"
    plt.close(fig)


def test_metricsuite_plot_anova_draws_no_pairwise_brackets():
    """Unlike plot.group_comparison, plot.anova never draws brackets -- an omnibus test never compared individual pairs."""
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")
    fig, ax = suite.plot.anova(result)

    assert not any(len(line.get_xdata()) == 4 for line in ax.get_lines())  # metric_box_significance's bracket shape
    plt.close(fig)


def test_metricsuite_plot_anova_supports_violin_and_bar_kinds():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")

    fig, ax = suite.plot.anova(result, kind="violin")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig, ax = suite.plot.anova(result, kind="bar")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_metricsuite_plot_anova_rejects_bad_kind():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")
    with pytest.raises(ValueError):
        suite.plot.anova(result, kind="bogus")


def test_metricsuite_plot_anova_rejects_non_anova_result():
    """A PairwiseResult passed to suite.plot.anova() is a clear TypeError, not a confusing crash downstream."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    with pytest.raises(TypeError):
        suite.plot.anova(result)


def test_metricsuite_plot_anova_respects_groups_subset():
    """Only the groups included in the AnovaResult are plotted, even if the suite has more."""
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region", groups=["A", "C", "D"])
    fig, ax = suite.plot.anova(result)

    labels = {t.get_text() for t in ax.get_xticklabels()}
    assert labels == {"A", "C", "D"}
    plt.close(fig)


def test_metricsuite_plot_anova_ax_passthrough():
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region")
    fig_in, ax_in = plt.subplots()

    fig_out, ax_out = suite.plot.anova(result, ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


# --- Save/load --------------------------------------------------------------

def test_metricsuite_save_load_roundtrip_preserves_pairwise_stats_cache(tmp_path):
    suite = _make_compare_suite()
    suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )

    folder = tmp_path/"suite_with_pairwise_stats"
    suite.save(folder)
    loaded = MetricSuite.load(folder)

    assert len(loaded.stat._stats_cache) == len(suite.stat._stats_cache)
    assert set(loaded.stat.list_cached_stats()["pair"]) == set(suite.stat.list_cached_stats()["pair"])


def test_metricsuite_save_load_roundtrip_preserves_cluster_stats_cache(tmp_path):
    suite = _make_significance_suite()
    suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=300, seed=0,
    )

    folder = tmp_path/"suite_with_cluster_stats"
    suite.save(folder)
    loaded = MetricSuite.load(folder)

    assert len(loaded.stat._stats_cache) == len(suite.stat._stats_cache)
    assert set(loaded.stat.list_cached_stats()["pair"]) == set(suite.stat.list_cached_stats()["pair"])


def test_metricsuite_save_removes_stale_stats_after_cache_cleared(tmp_path):
    """save() must not leave a stale stats/ folder behind after the cache is cleared and re-saved."""
    suite = _make_compare_suite()
    suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )

    folder = tmp_path/"suite_stale_stats"
    suite.save(folder)
    assert (folder/"stats").exists()

    suite.stat.clear_stats_cache()
    suite.save(folder, overwrite=True)
    assert not (folder/"stats").exists()


def test_metricsuite_save_load_roundtrip_no_legacy_fields(tmp_path):
    """`save()`/`load()` no longer write/read suite-level "subject"/"group_name" meta.json fields at all."""
    suite = _make_compare_suite()
    folder = tmp_path/"suite_no_legacy_fields"
    suite.save(folder)

    meta = json.loads((folder/"meta.json").read_text())
    assert "subject" not in meta
    assert "group_name" not in meta
    assert "label" not in meta["entries"][0]

    loaded = MetricSuite.load(folder)
    assert not hasattr(loaded, "subject")
    assert not hasattr(loaded, "group_name")


def test_metricsuite_load_reconstructs_legacy_group_name_and_label_as_coord(tmp_path):
    """
    A hand-built legacy-format save (suite-level "group_name" + per-entry
    dedicated "label" field, as `save()` used to write before `group_by`
    became a per-call choice) must still load correctly -- `label` folded
    into an ordinary coord under the old `group_name` column.
    """
    folder = tmp_path/"legacy_suite"
    folder.mkdir()

    values = np.array([1.0, 2.0, 3.0])
    np.savez(folder/"data.npz", entry0__value=values, entry0__coord__region=np.array(["r1", "r2", "r3"]))
    meta = {
        "name": "legacy", "group_name": "Method", "subject": None, "metadata": {},
        "saved_at": "2020-01-01T00:00:00",
        "entries": [{
            "label": "wcset", "metric": "outcome",
            "scalar_coords": {}, "array_coord_names": ["region"],
        }],
    }
    (folder/"meta.json").write_text(json.dumps(meta))

    loaded = MetricSuite.load(folder)
    df = loaded.to_df()
    assert set(df["Method"]) == {"wcset"}
    np.testing.assert_array_equal(df.sort_values("region")["Value"].to_numpy(), values)


def test_metricsuite_save_load_roundtrip_many_entries(tmp_path):
    """
    Regression guard for MetricSuite.load()'s array extraction, which now
    goes through mindmap.core.io.read_npz_fast instead of np.load/NpzFile
    (NpzFile.__getitem__'s O(n) per-lookup list scan made a loop of n
    lookups O(n^2) -- see read_npz_fast's docstring). A few hundred entries
    (3 methods x 50 regions, each with an array-valued "level" coord, so
    each entry contributes 2 npz members) is enough to exercise the loop
    meaningfully without the real-world ~88,000-entry case's runtime.
    """
    suite = MetricSuite(name="roundtrip")
    rng = np.random.default_rng(0)
    n_regions = 50
    coord_values = np.arange(5)

    for method in ["a", "b", "c"]:
        for j in range(n_regions):
            curve = rng.normal(size=5)
            suite.register("outcome", curve, Method=method, region=f"r{j}", level=coord_values)

    folder = tmp_path/"suite"
    suite.save(folder)
    loaded = MetricSuite.load(folder)

    original_df = suite.to_df().sort_values(["Method", "region", "level"]).reset_index(drop=True)
    loaded_df = loaded.to_df().sort_values(["Method", "region", "level"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(original_df, loaded_df)


# --- register(): rep-axis coordinates, generic filtering, no privileged tags ---

def test_metricsuite_register_rep_axis_coordinate():
    """
    A coordinate matching the *first* axis of a 2D (n_reps, n_points) array
    tags each repetition itself (repeated across every point within that
    rep), riding alongside -- not replacing -- the automatic "rep" index.
    """
    suite = MetricSuite()
    values_2d = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])  # n_reps=2, n_points=3
    suite.register("outcome", values_2d, Method="a", times=[0.1, 0.2, 0.3], seed=[42, 99])

    df = suite.to_df().sort_values("Value").reset_index(drop=True)
    np.testing.assert_array_equal(df["rep"].to_numpy(), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(df["seed"].to_numpy(), [42, 42, 42, 99, 99, 99])
    np.testing.assert_array_equal(df["times"].to_numpy(), [0.1, 0.2, 0.3, 0.1, 0.2, 0.3])


def test_metricsuite_register_coordinate_matching_neither_axis_raises():
    suite = MetricSuite()
    values_2d = np.zeros((2, 3))  # n_reps=2, n_points=3
    with pytest.raises(ValueError):
        suite.register("outcome", values_2d, Method="a", bad=[1, 2, 3, 4, 5])


def test_metricsuite_register_has_no_privileged_tags():
    """
    Neither `MetricSuite(...)` nor `register(...)` has any dedicated
    grouping/subject parameter left -- `label`/`group_name`/`subject` are
    all just ordinary `**coords` tags now, named however the caller likes.
    """
    with pytest.raises(TypeError):
        MetricSuite(subject="sub-01")
    with pytest.raises(TypeError):
        MetricSuite(group_name="Method")
    with pytest.raises(TypeError):
        MetricSuite(context={"src": None})

    suite = MetricSuite()
    suite.register("outcome", 1.0, subject="sub-01", Method="a")
    assert suite.to_df().loc[0, "subject"] == "sub-01"
    assert suite.to_df().loc[0, "Method"] == "a"


def _make_condition_suite(n_regions=10, seed=0):
    """Two methods x two conditions x n_regions -- for exercising to_df's generic `filters`."""
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for condition in ["rest", "task"]:
        for i in range(n_regions):
            ref_val = rng.normal()
            suite.register("outcome", ref_val, Method="a", region=f"r{i}", condition=condition)
            suite.register("outcome", ref_val - 2.0, Method="b", region=f"r{i}", condition=condition)
    return suite


def test_metricsuite_to_df_generic_filter():
    suite = _make_condition_suite()
    df = suite.to_df(metrics="outcome", filters={"condition": "rest"})
    assert set(df["condition"].unique()) == {"rest"}


def test_metricsuite_to_df_numeric_scalar_filter():
    """A bare numeric scalar (not wrapped in a list) must filter correctly, not just a bare string -- both are scalars, not lists."""
    suite = MetricSuite()
    for snr in [1, 5, 10]:
        suite.register("outcome", float(snr), Method="a", snr=snr)
    df = suite.to_df(filters={"snr": 5})
    assert len(df) == 1
    assert df.iloc[0]["snr"] == 5


def test_metricsuite_to_df_unregistered_column_raises():
    suite = _make_condition_suite()
    with pytest.raises(ValueError):
        suite.to_df(metrics="outcome", filters={"nonexistent": "x"})


def test_metricsuite_wilcoxon_filters_scope_the_comparison():
    suite = _make_condition_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region",
        filters={"condition": "rest"},
    )
    assert (result.table["n_pairs"] == 10).all()
    assert result.table.loc[0, "significant"]


def test_metricsuite_wilcoxon_numeric_scalar_filter():
    """A bare numeric scalar filter value (e.g. filters={"snr": 5}) must work through the stat cache-key freezing, not just strings."""
    suite = MetricSuite()
    rng = np.random.default_rng(0)
    for snr in [1, 5, 10]:
        for region in range(15):
            suite.register("outcome", rng.normal(), Method="a", region=f"r{region}", snr=snr)
            suite.register("outcome", rng.normal() - 2.0, Method="b", region=f"r{region}", snr=snr)
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region", filters={"snr": 5},
    )
    assert (result.table["n_pairs"] == 15).all()


def test_metricsuite_wilcoxon_filters_are_part_of_cache_identity(monkeypatch):
    """Two calls differing only in `filters` must not silently share a cached raw result."""
    suite = _make_condition_suite()

    call_count = {"n": 0}
    original = results_module.paired_test

    def counting_paired_test(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results_module, "paired_test", counting_paired_test)

    suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region",
        filters={"condition": "rest"},
    )
    assert call_count["n"] == 1

    suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region",
        filters={"condition": "task"},
    )
    assert call_count["n"] == 2  # different filters -- must recompute, not reuse "rest"'s cached entry

    suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region",
        filters={"condition": "rest"},
    )
    assert call_count["n"] == 2  # back to an already-seen filters value -- reused


def test_metricsuite_plot_group_comparison_respects_result_filters():
    """suite.plot.group_comparison() rebuilds its plotting df from result.filters, not the whole suite."""
    suite = _make_condition_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region",
        filters={"condition": "rest"},
    )
    fig, ax = suite.plot.group_comparison(result)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


# --- reference vs. competitors: PairwiseResult.vs / paired differences -------

def _make_vs_suite(n_regions=12, missing_noise=(), seed=0):
    """`reference` is clearly better than `worse` (+2) and ~equal to `noise`; `noise` can miss some regions."""
    rng = np.random.default_rng(seed)
    regions = [f"r{i:02d}" for i in range(n_regions)]
    reference_vals = rng.normal(0, 1, size=n_regions)
    suite = MetricSuite()
    suite.register("outcome", reference_vals, Method="reference", region=regions)
    suite.register("outcome", reference_vals - 2.0 + rng.normal(0, 0.1, size=n_regions), Method="worse", region=regions)
    keep = [i for i in range(n_regions) if regions[i] not in missing_noise]
    suite.register(
        "outcome", reference_vals[keep] + rng.normal(0, 1, size=len(keep)), Method="noise",
        region=[regions[i] for i in keep],
    )
    return suite


def test_pairwise_result_vs_orients_effect_toward_reference_as_group_a():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")], pairing_col="region",
    )
    s = result.vs("reference")

    assert list(s.index) == ["worse", "noise"]
    assert s.index.name == "Method"
    assert list(s.columns) == ["p_value", "p_corrected", "significant", "effect", "effect_size_type", "n_pairs"]
    assert s.loc["worse", "effect"] == pytest.approx(1.0)  # reference higher in every region
    assert s.loc["worse", "effect"] == result.table.loc[0, "effect_size"]
    assert s.loc["worse", "p_corrected"] == result.table.loc[0, "p_corrected"]


def test_pairwise_result_vs_flips_effect_when_reference_is_group_b():
    suite = _make_vs_suite()
    result = suite.stat.paired_ttest(
        "outcome", group_by="Method", comparisons=[("worse", "reference"), ("noise", "reference")], pairing_col="region",
    )
    s = result.vs("reference")

    assert result.table.loc[0, "effect_size"] < 0  # stored as worse - reference
    assert s.loc["worse", "effect"] == pytest.approx(-result.table.loc[0, "effect_size"])
    assert s.loc["worse", "effect"] > 0
    assert (s["effect_size_type"] == "cohens_dz").all()

    # Seen from the competitor's side, the same row reads the other way round.
    assert result.vs("worse").loc["reference", "effect"] == pytest.approx(result.table.loc[0, "effect_size"])


def test_pairwise_result_vs_higher_is_better_false_negates_effect():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("worse", "reference")], pairing_col="region",
    )
    hib = result.vs("reference")
    lib = result.vs("reference", higher_is_better=False)
    assert lib.loc["worse", "effect"] == pytest.approx(-hib.loc["worse", "effect"])
    assert lib.loc["worse", "effect"] < 0


def test_pairwise_result_vs_includes_n_eff_only_when_present():
    suite = _make_vs_suite(n_regions=30)
    comparisons = [("reference", "worse"), ("reference", "noise")]
    ess = suite.stat.ttest_ess("outcome", group_by="Method", comparisons=comparisons, pairing_col="region")
    wil = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=comparisons, pairing_col="region")
    assert "n_eff" in ess.vs("reference").columns
    assert "n_eff" not in wil.vs("reference").columns


def test_pairwise_result_vs_unknown_reference_raises():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    with pytest.raises(ValueError, match="appears in no pair"):
        result.vs("noise")


def test_stat_paired_differences_rows_match_n_pairs_and_sign():
    """One non-missing value per tested unit (= n_pairs, per competitor), reference - competitor, NaN where a competitor has no data."""
    suite = _make_vs_suite(missing_noise=("r03", "r07"))
    result = suite.stat.paired_ttest(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("noise", "reference")], pairing_col="region",
    )
    d = suite.stat.paired_differences(result)

    assert list(d.columns) == ["worse", "noise"]
    assert d.index.name == "region"
    s = result.vs("reference")
    for c in d.columns:
        assert d[c].notna().sum() == s.loc[c, "n_pairs"]
    assert d.loc[["r03", "r07"], "noise"].isna().all()

    raw = suite.to_df(metrics="outcome").pivot(index="region", columns="Method", values="Value")
    np.testing.assert_allclose(d["worse"], (raw["reference"] - raw["worse"]).loc[d.index])
    assert (d["worse"] > 0).all()

    # Same sign as the oriented effect size, and negated for lower-is-better metrics.
    lower = suite.stat.paired_differences(result, higher_is_better=False)
    pd.testing.assert_frame_equal(lower, -d)


def test_stat_paired_differences_collapses_units_like_the_test():
    n_regions, n_reps = 6, 4
    suite = _make_compare_suite_with_reps(n_regions=n_regions, n_reps=n_reps)
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "shifted")], pairing_col="region", unit_col="rep",
    )
    d = suite.stat.paired_differences(result)
    assert len(d) == n_regions == result.table.loc[0, "n_pairs"]

    means = suite.to_df(metrics="outcome").groupby(["region", "Method"])["Value"].mean().unstack()
    np.testing.assert_allclose(d["shifted"], (means["reference"] - means["shifted"]).loc[d.index])


def test_stat_paired_differences_follows_subject_order_and_filters():
    suite, true_order = _make_ar1_diff_suite(n=40)
    result = suite.stat.ttest_ess(
        "outcome", group_by="Method", comparisons=[("reference", "competitor")], pairing_col="unit",
        subject_order=true_order,
    )
    d = suite.stat.paired_differences(result)
    assert list(d.index) == true_order

    suite = _make_condition_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="region", filters={"condition": "rest"},
    )
    d = suite.stat.paired_differences(result)
    assert len(d) == result.table.loc[0, "n_pairs"]


def test_stat_paired_differences_rejects_independent_result():
    suite = _make_independent_suite()
    result = suite.stat.ttest("outcome", group_by="Method", comparisons=[("a", "b")])
    with pytest.raises(ValueError, match="paired result"):
        suite.stat.paired_differences(result)


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_metricsuite_plot_paired_differences_annotation_keys():
    suite = _make_vs_suite(n_regions=30, missing_noise=("r03",))
    result = suite.stat.ttest_ess(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")], pairing_col="region",
    )
    s = result.vs("reference")
    d = suite.stat.paired_differences(result)

    fig, ax = suite.plot.paired_differences(
        result, annotate=("stars", "effect", "winrate", "counts", "n_eff"),
    )
    texts = _texts(ax)
    assert texts[:5] == [
        "***", f"dz={s.loc['worse', 'effect']:+.2f}", "+50%", "30/0/0", f"n_eff={s.loc['worse', 'n_eff']:.0f}",
    ]
    noise = d["noise"].dropna()
    assert texts[7] == f"{(noise > 0).mean() - 0.5:+.0%}"
    assert texts[8] == f"{(noise > 0).sum()}/{(noise == 0).sum()}/{(noise < 0).sum()}"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["worse", "noise"]

    # First line on top and larger, same height for every column.
    assert ax.texts[0].get_position()[1] > ax.texts[1].get_position()[1]
    assert ax.texts[0].get_fontsize() > ax.texts[1].get_fontsize()
    assert ax.texts[0].get_position()[1] == ax.texts[5].get_position()[1]
    plt.close(fig)


def test_metricsuite_plot_paired_differences_rank_biserial_and_orientation():
    """Wilcoxon results print r=..., and lower-is-better flips both the plotted values and the effect sign."""
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("worse", "reference")], pairing_col="region")

    fig, ax = suite.plot.paired_differences(result, reference="reference", annotate=("effect",))
    assert _texts(ax) == ["r=+1.00"]
    plt.close(fig)

    fig, ax = suite.plot.paired_differences(result, reference="reference", higher_is_better=False, annotate=("effect", "counts"))
    assert _texts(ax) == ["r=-1.00", "0/0/12"]
    plt.close(fig)


def test_metricsuite_plot_paired_differences_n_eff_skipped_when_absent():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    fig, ax = suite.plot.paired_differences(result, annotate=("stars", "n_eff"))
    assert _texts(ax) == ["***"]
    plt.close(fig)


def test_metricsuite_plot_paired_differences_rejects_unknown_annotation():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    with pytest.raises(ValueError, match="Unknown annotation"):
        suite.plot.paired_differences(result, annotate=("stars", "cohen"))


def test_metricsuite_plot_paired_differences_violin_vs_points():
    """A column with at least violin_min_n units is a violin; below that, jittered points."""
    from matplotlib.collections import PathCollection, PolyCollection

    suite = _make_vs_suite(n_regions=12, missing_noise=("r00", "r01", "r02"))
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")], pairing_col="region",
    )
    fig, ax = suite.plot.paired_differences(result, violin_min_n=10, palette={"worse": "red", "noise": "blue"})
    violins = [c for c in ax.collections if isinstance(c, PolyCollection)]
    points = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(violins) == 1  # worse: 12 units
    assert len(points) == 1 and len(points[0].get_offsets()) == 9  # noise: 9 units
    assert violins[0].get_alpha() == pytest.approx(0.45)
    plt.close(fig)

    fig, ax = suite.plot.paired_differences(result)  # default violin_min_n=30: points only
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
    plt.close(fig)


def test_metricsuite_plot_paired_differences_order_labels_and_direction():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")], pairing_col="region",
    )
    fig_in, ax_in = plt.subplots()
    fig, ax = suite.plot.paired_differences(
        result, order=["noise", "worse"], tick_labels={"noise": "Noise"}, annotate=(),
        direction_labels=("ref better", "other better"), ax=ax_in,
    )
    assert ax is ax_in and fig is fig_in
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Noise", "worse"]
    assert _texts(ax) == ["ref better ↑", "other better ↓"]
    plt.close(fig)

    with pytest.raises(ValueError):
        suite.plot.paired_differences(result, order=["unknown"])


def test_metricsuite_plot_group_comparison_defaults_unchanged():
    """No new keyword given: opaque fills, no extra lines or bands, tick labels are the group names."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    fig, ax = suite.plot.group_comparison(result, order=["reference", "shifted", "noise"])
    assert all(p.get_facecolor()[3] == 1.0 for p in ax.patches)
    assert len(ax.patches) == 3
    assert not [l for l in ax.lines if l.get_linestyle() in ("--", ":")]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["reference", "shifted", "noise"]
    plt.close(fig)


def test_metricsuite_plot_group_comparison_decorations():
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    fig, ax = suite.plot.group_comparison(
        result, order=["reference", "shifted", "noise"], chance=(-0.5, 0.5), chance_value=0.0,
        reference_lines={"dSPM": 1.0, "sLORETA": (2.0, "green")}, fill_alpha=0.45,
        tick_labels={"reference": "Ref", "noise": "Noisy"}, reference_suffix=" (ref.)",
    )
    band, *boxes = ax.patches
    assert band.get_zorder() == 0
    assert len(boxes) == 3 and all(p.get_facecolor()[3] == pytest.approx(0.45) for p in boxes)
    assert all(p.get_edgecolor()[3] == 1.0 for p in boxes)
    dashed = {l.get_label(): l for l in ax.lines if l.get_linestyle() == "--"}
    assert set(dashed) == {"dSPM", "sLORETA"}
    assert dashed["sLORETA"].get_ydata()[0] == 2.0 and dashed["sLORETA"].get_color() == "green"
    assert len([l for l in ax.lines if l.get_linestyle() == ":"]) == 1
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Ref (ref.)", "shifted", "Noisy"]
    plt.close(fig)

    with pytest.raises(ValueError):
        suite.plot.group_comparison(result, fill_alpha=2)
