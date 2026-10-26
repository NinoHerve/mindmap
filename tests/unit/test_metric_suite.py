"""
This module performs unit tests for MetricSuite.stat (mindmap/metric/stat.py)
-- the wilcoxon/ttest/mannwhitneyu/paired_ttest/ttest_ess/cluster_permutation/
normality_test/anova cached, typed-result stat entry points -- and
MetricSuite.plot (mindmap/metric/plot.py) -- the raw (box/violin/bar/point/
strip/line/scatter/distribution/histogram) and result (group_comparison/
paired_differences/cluster_permutation/normality_test/anova/estimate)
plotting accessor built on top of them.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mindmap.metric import MetricSuite
from mindmap.metric.results import PairwiseResult, ClusterResult, NormalityResult, AnovaResult, EstimateResult
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
        "group_a", "group_b", "n_pairs", "statistic", "p_value", "p_corrected", "significant", "rank_biserial",
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
        "build_comparisons",
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
    assert result.effect_sizes == ("cohens_dz",)


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
    assert result.effect_sizes == ("cohens_d",)


def test_metricsuite_ttest_has_no_pairing_col_parameter():
    """Independent-samples tests have nothing to align across groups -- ttest takes no pairing_col at all."""
    suite = _make_independent_suite()
    with pytest.raises(TypeError):
        suite.stat.ttest("outcome", group_by="Method", comparisons=[("a", "b")], pairing_col="i")


def test_metricsuite_mannwhitneyu_detects_shifted_groups():
    suite = _make_independent_suite()
    result = suite.stat.mannwhitneyu("outcome", group_by="Method", comparisons=[("a", "b")])

    assert result.table.loc[0, "significant"]
    assert result.effect_sizes == ("rank_biserial",)


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





def test_pairwise_result_correct_recorrects_without_rerunning(monkeypatch):
    """result.correct() changes the correction/alpha on an existing result -- no test reruns."""
    suite = _make_compare_suite()
    comparisons = suite.stat.build_comparisons("reference", ["shifted", "noise"])
    result = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=comparisons, pairing_col="region")

    def fail(*args, **kwargs):
        raise AssertionError("paired_test should not rerun")
    monkeypatch.setattr(results_module, "paired_test", fail)

    recorrected = result.correct("bonferroni", alpha=0.01)
    assert (recorrected.correction, recorrected.alpha) == ("bonferroni", 0.01)
    assert (result.correction, result.alpha) == ("fdr_bh", 0.05)  # never mutated
    assert (recorrected.table["p_corrected"] >= result.table["p_corrected"]).all()




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
        result, palette={"reference": "black", "worse": "red", "same": "blue"},
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
    The interval (default IQR) is drawn as one whisker per (method,
    coordinate); interval=None drops them. The whiskers aren't part of the
    significance encoding, so only their presence is checked.
    """
    suite = _make_significance_suite(n_coord=6)
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method",
        comparisons=[("reference", "worse"), ("reference", "same")], pairing_col="region",
        coord_order=list(range(6)), n_permutations=500, seed=0,
    )
    with_whiskers = suite.plot.cluster_permutation(result)
    without = suite.plot.cluster_permutation(result, interval=None)

    assert sum(len(c.get_segments()) for c in with_whiskers.artists["intervals"]) == 3 * 6
    assert not without.artists["intervals"]
    plt.close(with_whiskers.fig)
    plt.close(without.fig)


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


def test_anova_result_correct_rethresholds_without_rerunning(monkeypatch):
    suite = _make_anova_suite()
    result = suite.stat.anova("outcome", group_by="Method", pairing_col="region", alpha=0.05)

    def fail(*args, **kwargs):
        raise AssertionError("repeated_measures_anova should not rerun")
    monkeypatch.setattr(results_module, "repeated_measures_anova", fail)
    assert result.correct(0.01).alpha == 0.01


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
    fig, ax = suite.plot.box("outcome", x="Method")

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_box_requires_x():
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.plot.box("outcome")


def test_metricsuite_plot_box_points_toggle():
    """
    points=False must drop the per-unit point overlay entirely (dense
    data, e.g. hundreds of points per box, makes it visual noise) while
    leaving the box plot itself intact; points=True (default) keeps it.
    """
    suite = _make_compare_suite()

    fig_with, ax_with = suite.plot.box("outcome", x="Method", points=True)
    fig_without, ax_without = suite.plot.box("outcome", x="Method", points=False)

    assert len(ax_with.collections) > 0
    assert len(ax_without.collections) == 0
    plt.close(fig_with)
    plt.close(fig_without)


def test_metricsuite_plot_violin_returns_fig_ax():
    suite = _make_compare_suite()
    fig, ax = suite.plot.violin("outcome", x="Method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_bar_returns_fig_ax():
    suite = _make_compare_suite()
    fig, ax = suite.plot.bar("outcome", x="Method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_line_returns_fig_ax():
    suite = _make_compare_suite_with_reps()
    fig, ax = suite.plot.line("outcome", x="rep", hue="Method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_scatter_returns_fig_ax():
    suite = _make_compare_suite_with_reps()
    fig, ax = suite.plot.scatter("outcome", x="rep", hue="Method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_distribution_without_group_by_draws_one_curve():
    """Unlike box/violin/bar/line/scatter, there is no x here -- the metric's value is the x-axis, so an ungrouped curve is already meaningful."""
    suite = _make_compare_suite()
    fig, ax = suite.plot.distribution("outcome")
    assert isinstance(fig, plt.Figure)
    assert len(ax.collections) > 0 or len(ax.lines) > 0
    plt.close(fig)


def _artist_alpha(artist):
    """
    An artist's effective alpha -- either set with `Artist.set_alpha()` or
    baked into the facecolor's RGBA (4th channel), in which case
    `get_alpha()` alone reports `None` even though the fill is translucent.
    """
    alpha = artist.get_alpha()
    if alpha is not None:
        return alpha
    facecolor = artist.get_facecolor()
    rgba = facecolor[0] if getattr(facecolor, "ndim", 1) == 2 else facecolor
    return rgba[3]


def test_metricsuite_plot_distribution_with_group_by_overlays_translucent_curves():
    """hue overlays one filled, translucent curve per group onto the same axes -- not separate figures."""
    suite = _make_compare_suite()
    fig, ax = suite.plot.distribution("outcome", hue="Method")
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
    """hue overlays translucent bars onto the same axes -- not stacked or side-by-side."""
    suite = _make_compare_suite()
    fig, ax = suite.plot.histogram("outcome", hue="Method")
    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) > 0
    assert all(_artist_alpha(p) < 1.0 for p in ax.patches)
    plt.close(fig)


def test_metricsuite_plot_ax_passthrough():
    suite = _make_compare_suite()
    fig_in, ax_in = plt.subplots()

    fig_out, ax_out = suite.plot.box("outcome", x="Method", ax=ax_in)

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

    assert not any(len(line.get_xdata()) == 4 for line in ax.get_lines())  # a significance bracket's shape
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

def test_metricsuite_save_keeps_only_data_and_warns_about_old_stats_folders(tmp_path):
    suite = _make_compare_suite()
    suite.stat.wilcoxon(
        "outcome", group_by="Method",
        comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]), pairing_col="region",
    )
    suite.save(tmp_path/"s")
    assert not (tmp_path/"s"/"stats").exists()

    # An older save's stats/ folder is left alone, and loading points at it instead of silently ignoring it.
    old = tmp_path/"s"/"stats"
    old.mkdir()
    (old/"index.json").write_text("{}")
    with pytest.warns(UserWarning, match="stats/"):
        MetricSuite.load(tmp_path/"s")
    suite.save(tmp_path/"s", overwrite=True)
    assert (old/"index.json").exists()






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
    assert list(s.columns) == [
        "p_value", "p_corrected", "significant", "rank_biserial", "rank_biserial_ci_low", "rank_biserial_ci_high",
        "offset", "alternative", "n_pairs", "n_zero",
    ]
    assert s.loc["worse", "rank_biserial"] == pytest.approx(1.0)  # reference higher in every region
    assert s.loc["worse", "rank_biserial"] == result.table.loc[0, "rank_biserial"]
    assert s.loc["worse", "p_corrected"] == result.table.loc[0, "p_corrected"]


def test_pairwise_result_vs_flips_effect_when_reference_is_group_b():
    suite = _make_vs_suite()
    result = suite.stat.paired_ttest(
        "outcome", group_by="Method", comparisons=[("worse", "reference"), ("noise", "reference")], pairing_col="region",
    )
    s = result.vs("reference")

    assert result.table.loc[0, "cohens_dz"] < 0  # stored as worse - reference
    assert s.loc["worse", "cohens_dz"] == pytest.approx(-result.table.loc[0, "cohens_dz"])
    assert s.loc["worse", "cohens_dz"] > 0

    # Seen from the competitor's side, the same row reads the other way round.
    assert result.vs("worse").loc["reference", "cohens_dz"] == pytest.approx(result.table.loc[0, "cohens_dz"])


def test_pairwise_result_vs_higher_is_better_false_negates_effect():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("worse", "reference")], pairing_col="region",
    )
    hib = result.vs("reference")
    lib = result.vs("reference", higher_is_better=False)
    assert lib.loc["worse", "rank_biserial"] == pytest.approx(-hib.loc["worse", "rank_biserial"])
    assert lib.loc["worse", "rank_biserial"] < 0


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
        "***", f"dz={s.loc['worse', 'cohens_dz']:+.2f}", "+50%", "30/0/0", f"n_eff={s.loc['worse', 'n_eff']:.0f}",
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
    assert violins[0].get_facecolor()[0][3] == pytest.approx(0.45)
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
        result, order=["noise", "worse"], labels={"noise": "Noise"}, annotate=(),
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
        labels={"reference": "Ref (ref.)", "noise": "Noisy"},
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


# --- offset / alternative ------------------------------------------------------


def test_pairwise_offset_tests_margin_and_is_recorded():
    """reference beats worse by ~2: significant beyond 1.5, not beyond 2.5; offset/alternative stored on the result."""
    suite = _make_vs_suite()
    within = suite.stat.ttest_ess("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region", offset=1.5)
    beyond = suite.stat.ttest_ess("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region", offset=2.5)

    assert (within.offset, within.alternative) == (1.5, "greater")
    assert within.table.loc[0, "significant"]
    assert not beyond.table.loc[0, "significant"]
    assert "offset=1.5" in repr(within)




def test_pairwise_vs_reexpresses_offset_and_alternative_toward_reference():
    suite = _make_vs_suite()
    result = suite.stat.paired_ttest("outcome", group_by="Method", comparisons=[("worse", "reference")], pairing_col="region", offset=-1.5, alternative="less")
    s = result.vs("reference")
    assert s.loc["worse", "offset"] == pytest.approx(1.5)
    assert s.loc["worse", "alternative"] == "greater"
    assert s.loc["worse", "cohens_dz"] > 0

    lower = result.vs("reference", higher_is_better=False)
    assert lower.loc["worse", "offset"] == pytest.approx(-1.5)
    assert lower.loc["worse", "alternative"] == "less"


def test_pairwise_effect_sizes_are_chosen_per_call_and_limited_to_the_test():
    suite = _make_vs_suite()
    kwargs = dict(group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    assert PairwiseResult.EFFECT_SIZES["paired_ttest"] == ("cohens_dz", "hedges_gz", "mean_diff")
    assert PairwiseResult.EFFECT_SIZES["wilcoxon"] == ("rank_biserial", "hodges_lehmann")

    result = suite.stat.paired_ttest("outcome", **kwargs, effect_sizes=("hedges_gz", "cohens_dz", "mean_diff"))
    assert result.effect_sizes == ("hedges_gz", "cohens_dz", "mean_diff")
    s = result.vs("reference")
    assert 0 < s.loc["worse", "hedges_gz"] < s.loc["worse", "cohens_dz"]
    est = suite.estimate.mean_diff("outcome", **kwargs)
    assert s.loc["worse", "mean_diff"] == pytest.approx(est.vs("reference").loc["worse", "estimate"])

    wilcoxon = suite.stat.wilcoxon("outcome", **kwargs, effect_sizes=("rank_biserial", "hodges_lehmann"))
    hl = suite.estimate.from_result(wilcoxon).vs("reference")
    assert wilcoxon.vs("reference").loc["worse", "hodges_lehmann"] == pytest.approx(hl.loc["worse", "estimate"])

    with pytest.raises(ValueError, match="aren't available"):
        suite.stat.paired_ttest("outcome", **kwargs, effect_sizes="rank_biserial")


def test_pairwise_vs_cles_flips_on_probability_scale():
    suite = _make_independent_suite()
    result = suite.stat.mannwhitneyu("outcome", group_by="Method", comparisons=[("a", "b")], effect_sizes=("rank_biserial", "cles"))
    cles = result.table.loc[0, "cles"]
    assert cles < 0.5  # b is higher
    assert result.table.loc[0, "rank_biserial"] < 0  # positive when group_a is higher
    assert result.vs("b").loc["a", "cles"] == pytest.approx(1 - cles)
    est = suite.estimate.cles("outcome", group_by="Method", comparisons=[("a", "b")], ci=False)
    assert est.vs("b").loc["a", "estimate"] == pytest.approx(1 - cles)


def test_pairwise_correct_records_settings():
    suite = _make_vs_suite()
    r1 = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region", correction="bonferroni", alpha=0.01)
    assert (r1.correction, r1.alpha) == ("bonferroni", 0.01)
    assert r1.correct("holm").correction == "holm"


def test_pairwise_save_load_roundtrip_keeps_new_fields(tmp_path):
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region", offset=1.0,
        effect_sizes=("hodges_lehmann", "rank_biserial"), correction="holm",
    )
    result.save(tmp_path/"r")
    loaded = PairwiseResult.load(tmp_path/"r")
    assert (loaded.offset, loaded.alternative, loaded.correction) == (1.0, "greater", "holm")
    assert loaded.effect_sizes == ("hodges_lehmann", "rank_biserial")
    pd.testing.assert_frame_equal(loaded.table, result.table, check_dtype=False)
    assert suite.verify(loaded)


def test_pairwise_load_of_old_format_defaults(tmp_path):
    """A result saved before offsets/data_key/effect_sizes existed: defaults, effect sizes read from its columns."""
    suite = _make_vs_suite()
    result = suite.stat.paired_ttest(
        "outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region",
        effect_sizes=("cohens_dz", "hedges_gz"),
    )
    result.save(tmp_path/"r")
    meta = json.loads((tmp_path/"r"/"identity.json").read_text())
    for key in ("offset", "alternative", "correction", "alpha", "data_key", "effect_sizes", "effect_size_version",
                "ci", "ci_level", "ci_methods", "ci_seed"):
        meta.pop(key)
    (tmp_path/"r"/"identity.json").write_text(json.dumps(meta))
    table = json.loads((tmp_path/"r"/"table.json").read_text())
    for row in table:  # version 2 tables held every effect size column, NaN where a test had none, and no intervals
        row.update(effect_size=row["cohens_dz"], effect_size_type="cohens_dz", rank_biserial=None, cles=None)
        for key in [k for k in row if k.endswith(("_ci_low", "_ci_high"))]:
            del row[key]
    (tmp_path/"r"/"table.json").write_text(json.dumps(table))

    loaded = PairwiseResult.load(tmp_path/"r")
    assert (loaded.offset, loaded.alternative, loaded.data_key) == (0.0, "two-sided", None)
    assert loaded.effect_sizes == ("cohens_dz", "hedges_gz")
    assert list(loaded.table.columns)[-2:] == ["cohens_dz", "hedges_gz"]
    assert loaded.ci is False and loaded.ci_methods is None
    with pytest.raises(ValueError, match="no data_key"):
        suite.verify(loaded)


# --- data_key / verify -----------------------------------------------------------


def test_verify_tracks_the_data_a_result_came_from():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    assert suite.verify(result)

    other = _make_vs_suite(seed=1)
    assert not other.verify(result)
    with pytest.raises(ValueError, match="different data"):
        other.plot.paired_differences(result)

    # Another metric's data doesn't matter.
    suite.register("other_metric", [1.0, 2.0], Method="x", region=["r00", "r01"], extra_coord=[3, 4])
    assert suite.verify(result)

    # New data for this metric does: the result is stale.
    suite.register("outcome", [0.0], Method="late", region=["r00"])
    assert not suite.verify(result)


def test_concat_keeps_results_on_untouched_metrics_and_invalidates_fused_ones():
    """Batch 1 + batch 2: a metric only one batch has keeps its results; a metric both have gets new data."""
    batch1 = _make_vs_suite()
    batch2 = _make_vs_suite(seed=1)
    batch1.register("only_in_batch1", [1.0, 2.0], Method="x", region=["r00", "r01"])
    on_shared = batch1.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    on_own = batch1.estimate.mean("only_in_batch1", group_by="Method")
    combined = MetricSuite.concat([batch1, batch2])
    assert combined.verify(on_own)
    assert not combined.verify(on_shared)


def test_data_key_ignores_registration_order():
    a, b = MetricSuite(), MetricSuite()
    a.register("m", [1.0, 2.0], Method="x", region=["r0", "r1"])
    a.register("m", [3.0, 4.0], Method="y", region=["r0", "r1"])
    b.register("m", [3.0, 4.0], Method="y", region=["r0", "r1"])
    b.register("m", [1.0, 2.0], Method="x", region=["r0", "r1"])
    result = a.stat.paired_ttest("m", group_by="Method", comparisons=[("x", "y")], pairing_col="region")
    assert b.verify(result)




# --- estimates ---------------------------------------------------------------------


def test_estimate_per_group_and_whole_population():
    suite = _make_vs_suite()
    per_group = suite.estimate.median("outcome", group_by="Method")
    assert not per_group.is_difference
    assert set(per_group.table["group"]) == {"reference", "worse", "noise"}
    assert (per_group.table["ci_low"] <= per_group.table["estimate"]).all()
    assert suite.verify(per_group)

    whole = suite.estimate.mean("outcome")
    assert whole.table.loc[0, "group"] is None
    assert whole.table.loc[0, "estimate"] == pytest.approx(suite.to_df("outcome")["Value"].mean())


def test_estimate_hodges_lehmann_paired_differences():
    suite = _make_vs_suite()
    result = suite.estimate.hodges_lehmann(
        "outcome", group_by="Method", comparisons=[("worse", "reference")], pairing_col="region",
    )
    row = result.table.loc[0]
    assert row["estimate"] == pytest.approx(-2.0, abs=0.1)
    assert row["ci_low"] < row["estimate"] < row["ci_high"]

    s = result.vs("reference")
    assert s.loc["worse", "estimate"] == pytest.approx(-row["estimate"])
    assert (s.loc["worse", "ci_low"], s.loc["worse", "ci_high"]) == pytest.approx((-row["ci_high"], -row["ci_low"]))
    with pytest.raises(ValueError, match="difference result"):
        suite.estimate.median("outcome", group_by="Method").vs("reference")


def test_estimate_from_result_reuses_the_tests_units():
    suite = _make_vs_suite(missing_noise=("r03", "r07"))
    test = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")], pairing_col="region",
    )
    est = suite.estimate.from_result(test)
    assert est.pairing_col == "region" and est.metric == "outcome"
    assert list(est.table["n_a"]) == [12, 10]

    with pytest.raises(ValueError, match="different data"):
        _make_vs_suite(seed=3).estimate.from_result(test)


def test_estimate_group_and_difference_estimators_are_separate():
    suite = _make_independent_suite()
    with pytest.raises(TypeError):
        suite.estimate.median("outcome", group_by="Method", comparisons=[("a", "b")])
    with pytest.raises(ValueError, match="describes a difference"):
        EstimateResult.compute(suite.to_df(metrics="outcome"), "outcome", "hodges_lehmann", group_by="Method")
    shift = suite.estimate.hodges_lehmann("outcome", group_by="Method", comparisons=[("b", "a")])
    assert shift.table.loc[0, "estimate"] == pytest.approx(3.0, abs=0.6)
    with pytest.raises(NotImplementedError, match="ci=False"):
        suite.estimate.cohens_d("outcome", group_by="Method", comparisons=[("b", "a")])
    d = suite.estimate.cohens_d("outcome", group_by="Method", comparisons=[("b", "a")], ci=False)
    ttest = suite.stat.ttest("outcome", group_by="Method", comparisons=[("b", "a")])
    assert d.table.loc[0, "estimate"] == pytest.approx(ttest.table.loc[0, "cohens_d"])
    assert np.isnan(d.table.loc[0, "ci_low"]) and not d.ci


def test_metricsuite_plot_paired_differences_offset_and_effect_size_keys():
    suite = _make_vs_suite()
    result = suite.stat.paired_ttest(
        "outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region", offset=1.5,
        effect_sizes=("cohens_dz", "mean_diff"),
    )
    s = result.vs("reference")
    fig, ax = suite.plot.paired_differences(result, annotate=("effect", "counts", "mean_diff"))
    # Offset-free effect sizes and counts; the offset is only the dashed line.
    assert _texts(ax) == [
        f"dz={s.loc['worse', 'cohens_dz']:+.2f}", "12/0/0", f"Δmean={s.loc['worse', 'mean_diff']:+.3g}",
    ]
    dashed = [l for l in ax.lines if l.get_linestyle() == "--"]
    assert len(dashed) == 1 and dashed[0].get_ydata()[0] == pytest.approx(1.5)
    plt.close(fig)

    with pytest.raises(ValueError, match="hodges_lehmann"):
        suite.plot.paired_differences(result, annotate=("hodges_lehmann",))
    with pytest.raises(ValueError, match="isn't in this result"):
        suite.plot.paired_differences(result, effect_size="hedges_gz")


def test_pairwise_holm_correction_is_recorded():
    suite = _make_vs_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")],
        pairing_col="region", correction="holm",
    )
    assert result.correction == "holm"
    assert (result.table["p_corrected"] >= result.table["p_value"]).all()


def test_estimate_survives_save_load_without_the_suite(tmp_path):
    suite = _make_vs_suite()
    est = suite.estimate.hodges_lehmann("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    est.save(tmp_path/"est")
    alone = EstimateResult.load(tmp_path/"est")
    pd.testing.assert_frame_equal(alone.table, est.table)
    assert (alone.estimator, alone.ci, alone.ess, alone.data_key) == (est.estimator, est.ci, est.ess, est.data_key)
    assert alone.vs("reference").loc["worse", "estimate"] == pytest.approx(est.vs("reference").loc["worse", "estimate"])
    assert suite.verify(alone)


def test_estimate_from_ttest_ess_result_uses_the_effective_sample_size():
    suite = _make_vs_suite(n_regions=30)
    result = suite.stat.ttest_ess("outcome", group_by="Method", comparisons=[("reference", "worse")], pairing_col="region")
    est = suite.estimate.from_result(result)
    assert est.estimator == "mean_diff" and est.ess
    assert est.table.loc[0, "ci_method"] == "t_ess"
    assert est.vs("reference").loc["worse", "n_eff"] == pytest.approx(result.table.loc[0, "n_eff"])
    with pytest.raises(NotImplementedError, match="effective-sample-size"):
        suite.estimate.from_result(result, estimator="hodges_lehmann")


def test_load_warns_on_results_with_outdated_effect_sizes(tmp_path):
    """Version 1 wilcoxon effect sizes, and version 2 ones with a non-zero offset, are outdated; others aren't."""
    suite = _make_vs_suite()
    kwargs = dict(group_by="Method", comparisons=[("reference", "noise")], pairing_col="region")
    cases = {
        "wilcoxon_v1": (suite.stat.wilcoxon("outcome", **kwargs), 1, True),
        "ttest_v1": (suite.stat.paired_ttest("outcome", **kwargs), 1, False),
        "offset_v2": (suite.stat.paired_ttest("outcome", **kwargs, offset=0.5), 2, True),
        "current": (suite.stat.wilcoxon("outcome", **kwargs, offset=0.5), None, False),
    }
    for name, (result, version, outdated) in cases.items():
        result.save(tmp_path/name)
        if version is not None:
            meta = json.loads((tmp_path/name/"identity.json").read_text())
            meta["effect_size_version"] = version
            (tmp_path/name/"identity.json").write_text(json.dumps(meta))
        if outdated:
            with pytest.warns(UserWarning, match="outdated"):
                assert PairwiseResult.load(tmp_path/name).has_outdated_effect_sizes
        else:
            assert not PairwiseResult.load(tmp_path/name).has_outdated_effect_sizes

# --- suite.plot: computed estimates, shared arguments, the Plot handle ----------

def test_plot_returns_handle_that_unpacks_as_fig_ax():
    suite = _make_compare_suite()
    plot = suite.plot.point("outcome", x="Method")
    fig, ax = plot
    assert fig is plot.fig and ax is plot.ax
    assert list(plot.data["Method"]) == ["reference", "shifted", "noise"]
    assert plot.data.attrs == {"center": "mean", "interval": "ci", "level": 0.95}
    assert plot.artists["centers"] and plot.artists["intervals"]
    plt.close(fig)


@pytest.mark.parametrize("center", ["mean", "median"])
def test_plot_ci_is_the_estimate_ci(center):
    """interval='ci' draws exactly suite.estimate's interval -- the same collapse (unit) and the same method."""
    suite = _make_compare_suite_with_reps()
    plot = suite.plot.point("outcome", x="Method", unit="region", center=center, interval="ci", level=0.9)
    est = getattr(suite.estimate, center)("outcome", group_by="Method", pairing_col="region", alpha=0.1)
    drawn = plot.data.set_index("Method")[["center", "low", "high"]]
    expected = est.table.set_index("group")[["estimate", "ci_low", "ci_high"]].loc[drawn.index]
    np.testing.assert_allclose(drawn.to_numpy(), expected.to_numpy(), equal_nan=True)
    plt.close(plot.fig)


def test_plot_unit_collapses_to_one_point_per_unit():
    n_regions, n_reps = 6, 4
    suite = _make_compare_suite_with_reps(n_regions=n_regions, n_reps=n_reps)
    raw = suite.plot.strip("outcome", x="Method")
    per_unit = suite.plot.strip("outcome", x="Method", unit="region")
    assert sum(len(c.get_offsets()) for c in raw.artists["points"]) == 2 * n_regions * n_reps
    assert sum(len(c.get_offsets()) for c in per_unit.artists["points"]) == 2 * n_regions
    assert (per_unit.data["n"] == n_regions).all()
    plt.close(raw.fig)
    plt.close(per_unit.fig)


def test_plot_interval_kinds():
    suite = _make_compare_suite()
    values = suite.to_df().query("Method == 'noise'")["Value"].to_numpy()
    for interval, low, high in [
        ("iqr", *np.percentile(values, [25, 75])),
        ("sd", values.mean() - values.std(ddof=1), values.mean() + values.std(ddof=1)),
        ("sem", values.mean() - values.std(ddof=1) / np.sqrt(len(values)), values.mean() + values.std(ddof=1) / np.sqrt(len(values))),
    ]:
        plot = suite.plot.point("outcome", x="Method", center="mean", interval=interval)
        row = plot.data.set_index("Method").loc["noise"]
        assert (row["low"], row["high"]) == pytest.approx((low, high))
        plt.close(plot.fig)


def test_plot_rejects_bad_center_interval():
    suite = _make_compare_suite()
    with pytest.raises(ValueError, match="center"):
        suite.plot.point("outcome", x="Method", center="mode")
    with pytest.raises(ValueError, match="interval"):
        suite.plot.point("outcome", x="Method", interval="pi")
    with pytest.raises(ValueError, match="center"):
        suite.plot.point("outcome", x="Method", center=None, interval="ci")
    with pytest.raises(ValueError, match="bar"):
        suite.plot.bar("outcome", x="Method", center=None, interval=None)
    with pytest.raises(ValueError, match="order"):
        suite.plot.box("outcome", x="Method", order=["missing"])


def test_plot_box_overlays_center_marker_only_when_asked():
    suite = _make_compare_suite()
    plain = suite.plot.box("outcome", x="Method")
    overlay = suite.plot.box("outcome", x="Method", center="mean", interval="sd")
    assert not plain.artists["centers"] and not plain.artists["intervals"]
    assert len(overlay.artists["centers"]) == 3 and len(overlay.artists["intervals"]) == 3
    assert {"median", "q1", "q3", "whislo", "whishi"} <= set(plain.data.columns)
    plt.close(plain.fig)
    plt.close(overlay.fig)


def test_plot_hue_dodges_and_adds_legend():
    suite = _make_compare_suite_with_reps(n_reps=2)
    plot = suite.plot.bar("outcome", x="Method", hue="rep", labels={0: "first"})
    positions = plot.data.set_index(["Method", "rep"])["position"]
    assert positions[("reference", 0)] < 0 < positions[("reference", 1)]
    assert positions[("shifted", 0)] < 1 < positions[("shifted", 1)]
    legend = plot.ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["first", "1"]
    assert legend.get_title().get_text() == "rep"
    no_legend = suite.plot.bar("outcome", x="Method", hue="rep", legend=False)
    assert no_legend.ax.get_legend() is None
    plt.close(plot.fig)
    plt.close(no_legend.fig)


def test_plot_colors_follow_palette_cycle_and_fill_alpha():
    from matplotlib.colors import to_rgba

    suite = _make_compare_suite()
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=["#111111", "#222222", "#333333"])}):
        themed = suite.plot.box("outcome", x="Method")
    assert [p.get_facecolor() for p in themed.artists["boxes"]] == [to_rgba(c) for c in ["#111111", "#222222", "#333333"]]

    partial = suite.plot.box("outcome", x="Method", palette={"noise": "red"}, fill_alpha=0.5)
    faces = {g: p.get_facecolor() for g, p in zip(partial.data["Method"], partial.artists["boxes"])}
    assert faces["noise"] == to_rgba("red", 0.5)
    assert all(face[3] == 0.5 for face in faces.values())
    plt.close(themed.fig)
    plt.close(partial.fig)


def test_plot_layer_kws_reach_matplotlib():
    suite = _make_compare_suite()
    plot = suite.plot.strip("outcome", x="Method", points_kws={"s": 50, "marker": "s"}, center_kws={"color": "red"})
    assert all((c.get_sizes() == 50).all() for c in plot.artists["points"])
    assert all(line.get_color() == "red" for line in plot.artists["centers"])
    with pytest.raises(AttributeError):
        suite.plot.strip("outcome", x="Method", points_kws={"not_a_property": 1})
    plt.close("all")


def test_plot_labels_axis_text():
    suite = _make_compare_suite()
    default = suite.plot.box("outcome", x="Method")
    assert (default.ax.get_xlabel(), default.ax.get_ylabel()) == ("Method", "outcome")
    custom = suite.plot.box("outcome", x="Method", xlabel="", ylabel="Score", title="T", labels={"noise": "Noisy"})
    assert (custom.ax.get_xlabel(), custom.ax.get_ylabel(), custom.ax.get_title()) == ("", "Score", "T")
    assert [t.get_text() for t in custom.ax.get_xticklabels()] == ["reference", "shifted", "Noisy"]
    plt.close("all")


def test_plot_line_median_iqr_per_coordinate():
    suite = _make_compare_suite_with_reps()
    plot = suite.plot.line("outcome", x="rep", hue="Method", center="median", interval="iqr")
    df = suite.to_df().query("Method == 'shifted' and rep == 2")["Value"]
    row = plot.data.set_index(["Method", "rep"]).loc[("shifted", 2)]
    assert row["center"] == pytest.approx(df.median())
    assert (row["low"], row["high"]) == pytest.approx(tuple(np.percentile(df, [25, 75])))
    assert len(plot.artists["lines"]) == 2 and len(plot.artists["intervals"]) == 2
    bars = suite.plot.line("outcome", x="rep", hue="Method", interval_style="bars")
    assert len(bars.artists["intervals"]) == 2
    plt.close("all")


def test_plot_group_comparison_kinds_and_bracket_text():
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon(
        "outcome", group_by="Method", comparisons=suite.stat.build_comparisons("reference", ["shifted", "noise"]),
        pairing_col="region", effect_sizes=("rank_biserial", "hodges_lehmann"),
    )
    plot = suite.plot.group_comparison(result, kind="point", annotate=("stars", "effect", "hodges_lehmann"), show_ns=True)
    assert plot.data.attrs["center"] == "median" and plot.data.attrs["interval"] == "ci"  # matches the Wilcoxon test
    assert len(plot.artists["brackets"]) == 2
    shifted = plot.data.set_index("Method").loc["shifted", "annotation"]
    assert shifted.startswith("***") and "r=+1.00" in shifted and "HL=+2" in shifted
    only_significant = suite.plot.group_comparison(result, kind="violin")
    assert len(only_significant.artists["brackets"]) == 1
    with pytest.raises(ValueError, match="paired_differences"):
        suite.plot.group_comparison(result, annotate=("winrate",))
    plt.close("all")


def test_plot_estimate_groups_and_oriented_differences():
    suite = _make_vs_suite()
    groups = suite.estimate.median("outcome", group_by="Method", pairing_col="region")
    plot = suite.plot.estimate(groups)
    np.testing.assert_allclose(plot.data["center"], groups.table["estimate"])
    assert len(plot.artists["centers"]) == 3

    hl = suite.estimate.hodges_lehmann(
        "outcome", group_by="Method", comparisons=[("worse", "reference"), ("reference", "noise")], pairing_col="region",
    )
    stored = suite.plot.estimate(hl)
    assert [t.get_text() for t in stored.ax.get_xticklabels()] == ["worse − reference", "reference − noise"]
    oriented = suite.plot.estimate(hl, reference="reference", reference_lines={"offset": 1.0})
    worse = oriented.data.set_index("Method").loc["worse"]
    assert worse["center"] > 0 and worse["low"] <= worse["center"] <= worse["high"]
    with pytest.raises(TypeError):
        suite.plot.estimate(suite.stat.wilcoxon("outcome", group_by="Method", comparisons=[("reference", "noise")], pairing_col="region"))
    plt.close("all")


# --- cluster permutation: correction family ---------------------------------------

def _cluster_kwargs(comparisons):
    return dict(coord="level", group_by="Method", comparisons=comparisons, pairing_col="region",
                coord_order=list(range(6)), n_permutations=300, seed=0)


def test_cluster_pair_family_corrects_each_pairs_smallest_p():
    """family='pair': one p per pair (smallest cluster p, 1 without clusters), Holm across the pairs."""
    from mindmap.core.stats import correct_pvalues

    suite = _make_multi_competitor_cluster_suite()
    raw = suite.stat.cluster_permutation(
        "outcome", **_cluster_kwargs([("reference", "worse_strong"), ("reference", "worse_weak")]), correction="none",
    )
    result = raw.correct("holm", alpha=0.05, family="pair")
    p_min = [cl["p_value"].min() if not cl.empty else 1.0 for cl in raw.clusters_by_pair.values()]
    expected, _ = correct_pvalues(p_min, method="holm")

    table = result.pair_table
    np.testing.assert_allclose(table["p_min"], p_min)
    np.testing.assert_allclose(table["p_corrected"], expected)
    assert (result.family, result.correction) == ("pair", "holm")
    for (pair, clusters), p_pair in zip(result.clusters_by_pair.items(), expected):
        np.testing.assert_allclose(clusters["p_corrected"], np.maximum(clusters["p_value"], p_pair))
        assert (clusters["significant"] == (clusters["p_corrected"] <= 0.05)).all()


def test_cluster_pair_family_counts_pairs_without_clusters():
    """A pair with no cluster still counts in the family (p = 1), so Holm's multiplier is the number of pairs."""
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", **_cluster_kwargs([("reference", "worse"), ("reference", "same")]), correction="holm", family="pair",
    )
    table = result.pair_table.set_index("group_b")
    if table.loc["same", "n_clusters"] == 0:
        assert table.loc["same", "p_corrected"] == 1.0
    assert table.loc["worse", "p_corrected"] == pytest.approx(min(1.0, 2 * table.loc["worse", "p_min"]))
    assert bool(table.loc["worse", "significant"]) and not bool(table.loc["same", "significant"])


def test_cluster_family_default_pools_clusters_and_rejects_unknown():
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation("outcome", **_cluster_kwargs([("reference", "worse"), ("reference", "same")]))
    assert result.family == "cluster"
    assert "cluster family" in repr(result)
    with pytest.raises(ValueError, match="family"):
        result.correct("holm", family="pairs")
    uncorrected = suite.stat.cluster_permutation(
        "outcome", **_cluster_kwargs([("reference", "worse")]), correction="none",
    )
    assert uncorrected.correct("holm", family="pair").pair_table["p_corrected"].notna().all()


def test_cluster_family_survives_save_load(tmp_path):
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", **_cluster_kwargs([("reference", "worse"), ("reference", "same")]), correction="holm", family="pair",
    )
    result.save(tmp_path/"pair")
    loaded = ClusterResult.load(tmp_path/"pair")
    assert loaded.family == "pair"
    pd.testing.assert_frame_equal(loaded.pair_table, result.pair_table)

    meta = json.loads((tmp_path/"pair"/"identity.json").read_text())
    del meta["family"]  # a file saved before families existed was always pooled
    (tmp_path/"pair"/"identity.json").write_text(json.dumps(meta))
    assert ClusterResult.load(tmp_path/"pair").family == "cluster"


def test_cluster_pair_family_drives_plot_styling():
    """The several-pairs plot reads each cluster's p_corrected, so a pair failing the pair-family correction is drawn dashed."""
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", **_cluster_kwargs([("reference", "worse"), ("reference", "same")]), correction="holm", family="pair",
    )
    plot = suite.plot.cluster_permutation(result, palette={"reference": "black", "worse": "red", "same": "blue"})
    segments = [l for l in plot.artists["lines"] if len(l.get_xdata()) == 2]
    assert all(l.get_linestyle() == "-" for l in segments if l.get_color() == "red")
    assert all(l.get_linestyle() == "--" for l in segments if l.get_color() == "blue")
    plt.close(plot.fig)


# --- Wilcoxon zero_method --------------------------------------------------------

def _make_ties_suite(n_regions=26, n_ties=10, seed=0):
    """`reference` has a lower distance than `other` in most regions, and an identical one (a tie) in `n_ties`."""
    rng = np.random.default_rng(seed)
    regions = [f"p{i:02d}" for i in range(n_regions)]
    other = rng.gamma(2.0, 5.0, size=n_regions)
    reference = other - rng.gamma(2.0, 2.0, size=n_regions) * rng.choice([1, 1, 1, -1], size=n_regions)
    reference[:n_ties] = other[:n_ties] = 0.0
    suite = MetricSuite()
    suite.register("distance", np.round(reference, 3), Method="reference", patient=regions)
    suite.register("distance", np.round(other, 3), Method="other", patient=regions)
    return suite


@pytest.mark.parametrize("zero_method", ["wilcox", "pratt", "zsplit"])
def test_wilcoxon_zero_method_matches_scipy_and_counts_ties(zero_method):
    from scipy import stats

    suite = _make_ties_suite()
    result = suite.stat.wilcoxon(
        "distance", group_by="Method", comparisons=[("reference", "other")], pairing_col="patient",
        zero_method=zero_method, correction="none",
    )
    wide = suite.to_df().pivot(index="patient", columns="Method", values="Value")
    expected = stats.wilcoxon(wide["reference"] - wide["other"], zero_method=zero_method)
    row = result.table.iloc[0]
    assert row["p_value"] == pytest.approx(expected.pvalue)
    assert row["n_zero"] == 10
    assert row["n_pairs"] == (16 if zero_method == "wilcox" else 26)
    assert result.zero_method == zero_method
    assert (zero_method != "wilcox") == (f"zero_method={zero_method!r}" in repr(result))


def test_wilcoxon_zero_method_rank_biserial_and_estimate_agree():
    suite = _make_ties_suite()
    kwargs = dict(group_by="Method", comparisons=[("reference", "other")], pairing_col="patient")
    r = {zm: suite.stat.wilcoxon("distance", **kwargs, zero_method=zm).table.loc[0, "rank_biserial"]
         for zm in ("wilcox", "pratt", "zsplit")}
    assert r["pratt"] == pytest.approx(r["zsplit"])
    assert abs(r["pratt"]) < abs(r["wilcox"])  # ties pull r toward 0

    pratt = suite.stat.wilcoxon("distance", **kwargs, zero_method="pratt")
    from_result = suite.estimate.from_result(pratt, estimator="rank_biserial", ci=False)
    direct = suite.estimate.rank_biserial("distance", **kwargs, zero_method="pratt", ci=False)
    assert from_result.zero_method == direct.zero_method == "pratt"
    assert from_result.table.loc[0, "estimate"] == pytest.approx(r["pratt"])
    assert direct.table.loc[0, "estimate"] == pytest.approx(r["pratt"])
    assert suite.estimate.from_result(pratt).zero_method is None  # hodges_lehmann: zeros are just Walsh averages


def test_wilcoxon_zero_method_saved_and_validated(tmp_path):
    suite = _make_ties_suite()
    kwargs = dict(group_by="Method", comparisons=[("reference", "other")], pairing_col="patient")
    result = suite.stat.wilcoxon("distance", **kwargs, zero_method="pratt")
    result.save(tmp_path/"pratt")
    loaded = PairwiseResult.load(tmp_path/"pratt")
    assert loaded.zero_method == "pratt"
    pd.testing.assert_frame_equal(loaded.table, result.table, check_dtype=False)

    meta = json.loads((tmp_path/"pratt"/"identity.json").read_text())
    del meta["zero_method"]  # saved before the option existed: always "wilcox"
    (tmp_path/"pratt"/"identity.json").write_text(json.dumps(meta))
    assert PairwiseResult.load(tmp_path/"pratt").zero_method == "wilcox"
    assert suite.stat.paired_ttest("distance", **kwargs).zero_method is None

    with pytest.raises(ValueError, match="zero_method"):
        suite.stat.wilcoxon("distance", **kwargs, zero_method="drop")
    with pytest.raises(ValueError, match="zero_method"):
        suite.estimate.rank_biserial("distance", group_by="Method", comparisons=[("reference", "other")], zero_method="pratt", ci=False)


# --- McNemar / binary outcomes -----------------------------------------------------

def _make_binary_suite(seed=0):
    """26 patients, peak inside the resection (1) or not (0): `wcset` vs `sloreta` (6 vs 1 discordant) and `dspm` (2 vs 2)."""
    patients = [f"p{i:02d}" for i in range(26)]
    wcset = np.array([1] * 14 + [1] * 6 + [0] * 1 + [0] * 5)
    sloreta = np.array([1] * 14 + [0] * 6 + [1] * 1 + [0] * 5)
    dspm = wcset.copy()
    dspm[[14, 15]] = 0
    dspm[[20, 21]] = 1
    suite = MetricSuite()
    for name, values in [("wcset", wcset), ("sloreta", sloreta), ("dspm", dspm)]:
        suite.register("inside", values.astype(float), method=name, patient=patients)
    return suite


def test_mcnemar_table_counts_effects_and_holm():
    suite = _make_binary_suite()
    result = suite.stat.mcnemar(
        "inside", group_by="method", comparisons=[("wcset", "sloreta"), ("dspm", "wcset")], pairing_col="patient",
        correction="holm", effect_sizes=("prop_diff", "odds_ratio"),
    )
    row = result.table.set_index("group_b").loc["sloreta"]
    assert (row["n_pairs"], row["n_zero"], row["n_a_only"], row["n_b_only"]) == (26, 19, 6, 1)
    assert row["p_value"] == pytest.approx(0.125)
    assert (row["prop_diff"], row["odds_ratio"]) == pytest.approx((5 / 26, 6.0))
    assert result.correction == "holm" and result.offset == 0 and result.zero_method is None

    vs = result.vs("wcset")
    assert vs.loc["sloreta", "n_reference_only"] == 6 and vs.loc["sloreta", "n_competitor_only"] == 1
    flipped = vs.loc["dspm"]  # stored as dspm - wcset: re-expressed as wcset relative to dspm
    stored = result.table.set_index("group_a").loc["dspm"]
    assert flipped["prop_diff"] == pytest.approx(-stored["prop_diff"])
    assert flipped["odds_ratio"] == pytest.approx(1 / stored["odds_ratio"])
    assert (flipped["n_reference_only"], flipped["n_competitor_only"]) == (stored["n_b_only"], stored["n_a_only"])


def test_mcnemar_estimates_and_save_load(tmp_path):
    suite = _make_binary_suite()
    kwargs = dict(group_by="method", comparisons=[("wcset", "sloreta")], pairing_col="patient")
    result = suite.stat.mcnemar("inside", **kwargs)
    est = suite.estimate.from_result(result)
    assert est.estimator == "prop_diff" and est.table.loc[0, "ci_method"] == "newcombe"
    assert est.table.loc[0, "estimate"] == pytest.approx(5 / 26)
    assert (est.table.loc[0, "ci_low"], est.table.loc[0, "ci_high"]) == pytest.approx((-0.0103, 0.3747), abs=1e-4)

    odds = suite.estimate.odds_ratio("inside", **kwargs)
    mirrored = odds.vs("sloreta").loc["wcset"]
    row = odds.table.loc[0]
    assert mirrored["estimate"] == pytest.approx(1 / row["estimate"])
    assert (mirrored["ci_low"], mirrored["ci_high"]) == pytest.approx((1 / row["ci_high"], 1 / row["ci_low"]))

    result.save(tmp_path/"mcnemar")
    loaded = PairwiseResult.load(tmp_path/"mcnemar")
    pd.testing.assert_frame_equal(loaded.table, result.table, check_dtype=False)
    assert {"n_a_only", "n_b_only"} <= set(loaded.table.columns)


def test_mcnemar_requires_binary_values():
    suite = _make_compare_suite()
    with pytest.raises(ValueError, match="binary"):
        suite.stat.mcnemar("outcome", group_by="Method", comparisons=[("reference", "noise")], pairing_col="region")

    patients = [f"p{i}" for i in range(4)]
    repeated = MetricSuite()
    for run, values in enumerate([[1, 0, 1, 0], [1, 1, 0, 0]]):
        repeated.register("inside", np.array(values, dtype=float), method="a", patient=patients, run=run)
        repeated.register("inside", np.zeros(4), method="b", patient=patients, run=run)
    with pytest.raises(ValueError, match="binary"):  # averaging two runs gives 0.5
        repeated.stat.mcnemar("inside", group_by="method", comparisons=[("a", "b")], pairing_col="patient", unit_col="run")


def test_mcnemar_plots_label_effects_and_log_odds_axis():
    suite = _make_binary_suite()
    result = suite.stat.mcnemar(
        "inside", group_by="method", comparisons=[("wcset", "sloreta"), ("wcset", "dspm")], pairing_col="patient",
        effect_sizes=("prop_diff", "odds_ratio"),
    )
    plot = suite.plot.paired_differences(result, annotate=("p", "effect", "odds_ratio", "counts"))
    text = plot.data.set_index("method").loc["sloreta", "annotation"]
    assert "Δp=+0.192" in text and "OR=6" in text and "6/19/1" in text

    odds = suite.estimate.odds_ratio("inside", group_by="method", comparisons=[("wcset", "sloreta")], pairing_col="patient")
    est_plot = suite.plot.estimate(odds, reference="wcset")
    assert est_plot.ax.get_yscale() == "log"
    assert any(np.allclose(line.get_ydata(), 1.0) for line in est_plot.ax.get_lines())
    plt.close("all")


# --- Stat views: test-matched summaries, difference curves --------------------------

def test_paired_differences_auto_draws_the_tests_own_estimate():
    suite = _make_vs_suite(n_regions=30)
    kwargs = dict(group_by="Method", comparisons=[("reference", "worse"), ("reference", "noise")], pairing_col="region")
    for test, estimator in [("wilcoxon", "hodges_lehmann"), ("paired_ttest", "mean_diff"), ("ttest_ess", "mean_diff")]:
        result = getattr(suite.stat, test)("outcome", **kwargs)
        plot = suite.plot.paired_differences(result)
        est = suite.estimate.from_result(result).vs("reference")
        drawn = plot.data.set_index("Method")[["center", "low", "high"]]
        np.testing.assert_allclose(drawn.to_numpy(), est.loc[drawn.index, ["estimate", "ci_low", "ci_high"]].to_numpy())
        assert plot.data.attrs["center"] == estimator and plot.data.attrs["interval"] == "ci"
        plt.close(plot.fig)

    result = suite.stat.wilcoxon("outcome", **kwargs)
    explicit = suite.plot.paired_differences(result, center="median", interval="iqr")
    d = suite.stat.paired_differences(result)["worse"].dropna()
    row = explicit.data.set_index("Method").loc["worse"]
    assert (row["center"], row["low"], row["high"]) == pytest.approx((d.median(), *np.percentile(d, [25, 75])))
    with pytest.raises(ValueError, match="auto"):
        suite.plot.paired_differences(result, interval="sd")
    plt.close("all")


def test_mcnemar_stat_views_use_proportions():
    suite = _make_binary_suite()
    kwargs = dict(group_by="method", comparisons=[("wcset", "sloreta")], pairing_col="patient")
    result = suite.stat.mcnemar("inside", **kwargs)

    groups = suite.plot.group_comparison(result)
    assert groups.kind == "group_comparison" and groups.artists["bars"]
    expected = suite.estimate.proportion("inside", group_by="method", pairing_col="patient").table.set_index("group")
    drawn = groups.data.set_index("method")
    for g in ["wcset", "sloreta"]:
        assert (drawn.loc[g, "center"], drawn.loc[g, "low"], drawn.loc[g, "high"]) == pytest.approx(
            tuple(expected.loc[g, ["estimate", "ci_low", "ci_high"]]))
    assert expected.loc["wcset", "ci_method"] == "wilson"

    diffs = suite.plot.paired_differences(result)
    row = diffs.data.set_index("method").loc["sloreta"]
    assert (row["center"], row["low"], row["high"]) == pytest.approx((5 / 26, -0.0103, 0.3747), abs=1e-4)
    assert diffs.data.attrs["center"] == "prop_diff" and not diffs.artists["violins"]
    plt.close("all")


def test_cluster_paired_differences_long_table_and_orientation():
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "worse"), ("reference", "same")],
        pairing_col="region", coord_order=list(range(6)), n_permutations=300, seed=0, correction="holm", family="pair",
    )
    long = suite.stat.paired_differences(result)
    assert list(long.columns) == ["region", "level", "Method", "Value"]
    assert set(long["Method"]) == {"worse", "same"}
    stored = result.diff_by_pair[("reference", "worse")]
    worse = long[long["Method"] == "worse"].pivot(index="region", columns="level", values="Value")
    np.testing.assert_allclose(worse.loc[stored.index, stored.columns].to_numpy(), stored.to_numpy())
    flipped = suite.stat.paired_differences(result, higher_is_better=False)
    np.testing.assert_allclose(flipped["Value"], -long["Value"])
    with pytest.raises(ValueError, match="reference"):
        suite.stat.paired_differences(result, reference="nobody")


def test_cluster_difference_curves_mark_significant_clusters():
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "worse"), ("reference", "same")],
        pairing_col="region", coord_order=list(range(6)), n_permutations=300, seed=0, correction="holm", family="pair",
    )
    plot = suite.plot.paired_differences(result, palette={"worse": "red", "same": "blue"})
    assert plot.kind == "paired_differences" and len(plot.artists["lines"]) == 2
    long = suite.stat.paired_differences(result)
    mean_worse = long[long["Method"] == "worse"].groupby("level")["Value"].mean()
    drawn = plot.data[plot.data["Method"] == "worse"].set_index("level")["center"]
    np.testing.assert_allclose(drawn.loc[mean_worse.index], mean_worse)
    assert plot.data.attrs["center"] == "mean" and plot.data.attrs["interval"] == "ci"

    from matplotlib.colors import to_rgba
    bar_colors = [tuple(c.get_colors()[0]) for c in plot.artists["clusters"]]
    assert bar_colors == [to_rgba("red")]  # only "worse" has a significant cluster
    segment = plot.artists["clusters"][0].get_segments()[0]
    assert (segment[0][0], segment[1][0]) == (-0.5, 5.5)  # the whole range, half a step past each end
    assert "significant cluster" in [t.get_text() for t in plot.ax.get_legend().get_texts()]

    median = suite.plot.paired_differences(result, center="median", interval="iqr", interval_style="bars")
    assert median.data.attrs["center"] == "median"
    plt.close("all")


def test_cluster_permutation_group_curves_default_to_mean_ci():
    suite = _make_significance_suite()
    result = suite.stat.cluster_permutation(
        "outcome", coord="level", group_by="Method", comparisons=[("reference", "worse")],
        pairing_col="region", coord_order=list(range(6)), n_permutations=200, seed=0,
    )
    plot = suite.plot.cluster_permutation(result)
    assert plot.data.attrs["center"] == "mean" and plot.data.attrs["interval"] == "ci"
    plt.close(plot.fig)


# -- Bootstrap ------------------------------------------------------------

def test_stat_bootstrap_detects_the_shift_and_matches_the_estimate_interval():
    suite = _make_compare_suite()
    comps = [("reference", "shifted"), ("reference", "noise")]
    result = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=2000, seed=5, correction="holm")
    assert result.test == "bootstrap" and result.effect_sizes == ("mean_diff",)
    assert result.bootstrap == {"n_boot": 2000, "seed": 5, "replicates": None, "replicate_col": None, "estimator": "mean_diff",
                                "interval": "percentile", "acceleration": None, "jackknife": None, "jackknife_col": None}
    assert result.table.loc[0, "p_value"] == pytest.approx(2 / 2001)
    assert (result.table["n_boot"] == 2000).all()

    from_result = suite.estimate.from_result(result)
    direct = suite.estimate.mean_diff("outcome", "Method", comps, pairing_col="region", ci_method="bootstrap", n_boot=2000, seed=5)
    assert from_result.ci_method == "bootstrap" and from_result.bootstrap["seed"] == 5
    pd.testing.assert_frame_equal(from_result.table, direct.table)
    for (_, est), p in zip(from_result.table.iterrows(), result.table["p_value"]):
        assert (p < 0.05) == (est["ci_low"] > 0 or est["ci_high"] < 0)
    np.testing.assert_allclose(result.table["mean_diff"], suite.estimate.mean_diff("outcome", "Method", comps, pairing_col="region").table["estimate"])


def test_stat_bootstrap_records_a_random_seed_that_reproduces_it():
    suite = _make_compare_suite()
    comps = [("reference", "noise")]
    first = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=500)
    assert isinstance(first.bootstrap["seed"], int)
    again = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=500, seed=first.bootstrap["seed"])
    pd.testing.assert_frame_equal(first.table, again.table)


def test_stat_bootstrap_independent_resamples_collapsed_units():
    suite = _make_compare_suite_with_reps()
    result = suite.stat.bootstrap(
        "outcome", "Method", [("reference", "shifted")], unit_col="region", estimator="cles", n_boot=500, seed=0,
    )
    assert result.pairing_col is None
    assert result.table.loc[0, "n_a"] == 6 and result.table.loc[0, "n_b"] == 6
    with pytest.raises(ValueError, match="its own null"):
        suite.stat.bootstrap("outcome", "Method", [("reference", "shifted")], estimator="cles", offset=0.1)
    with pytest.raises(ValueError, match="isn't defined for paired"):
        suite.stat.bootstrap("outcome", "Method", [("reference", "shifted")], pairing_col="region", estimator="cles")


def test_stat_bootstrap_offset_is_a_non_inferiority_margin():
    suite = _make_compare_suite()
    comps = [("reference", "noise")]
    two_sided = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=2000, seed=1)
    margin = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=2000, seed=1, offset=-2.0)
    assert margin.alternative == "greater"
    assert margin.table.loc[0, "p_value"] < 0.01 < two_sided.table.loc[0, "p_value"]


def test_bootstrap_results_round_trip_through_save(tmp_path):
    suite = _make_compare_suite()
    comps = [("reference", "shifted")]
    result = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=300, seed=2)
    result.save(tmp_path/"test")
    loaded = PairwiseResult.load(tmp_path/"test")
    assert loaded.bootstrap == result.bootstrap and loaded.estimator == "mean_diff"
    pd.testing.assert_frame_equal(loaded.table, result.table, check_dtype=False)
    assert suite.verify(loaded)

    est = suite.estimate.from_result(result)
    est.save(tmp_path/"est")
    loaded_est = EstimateResult.load(tmp_path/"est")
    assert loaded_est.bootstrap == est.bootstrap and loaded_est.ci_method == "bootstrap"
    pd.testing.assert_frame_equal(loaded_est.table, est.table, check_dtype=False)


def test_estimate_bootstrap_argument_checks():
    suite = _make_compare_suite()
    comps = [("reference", "shifted")]
    with pytest.raises(ValueError, match="only apply to ci_method='bootstrap'"):
        suite.estimate.mean("outcome", "Method", n_boot=100)
    with pytest.raises(ValueError, match="go together"):
        suite.estimate.mean("outcome", "Method", ci_method="bootstrap", replicates="outcome_boot")
    with pytest.raises(ValueError, match="no interval to bootstrap"):
        suite.estimate.mean("outcome", "Method", ci=False, ci_method="bootstrap")
    with pytest.raises(TypeError, match="seed"):
        suite.estimate.mean("outcome", "Method", ci_method="bootstrap", seed=np.random.default_rng(0))
    est = suite.estimate.rank_biserial("outcome", "Method", comps, pairing_col="region", ci_method="bootstrap", n_boot=300, seed=0)
    assert est.table.loc[0, "ci_low"] <= est.table.loc[0, "estimate"] <= est.table.loc[0, "ci_high"]


def _make_replicate_suite(n_boot=400, seed=0):
    """Per subject, an observed correlation per method plus `n_boot` paired draws (shared resample noise)."""
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for subject, effect in [("s1", 0.1), ("s2", 0.0)]:
        shared = rng.normal(0, 0.05, size=n_boot)
        for method, mu in [("w", 0.5 + effect), ("d", 0.5)]:
            suite.register("rho", mu, Method=method, subject=subject)
            draws = mu + shared + rng.normal(0, 0.01, size=n_boot)
            suite.register("rho_boot", draws, Method=method, subject=subject, boot=np.arange(n_boot))
    return suite


def test_bootstrap_reads_registered_replicates_per_subject():
    suite = _make_replicate_suite()
    kwargs = dict(pairing_col="subject", replicates="rho_boot", replicate_col="boot")
    s1 = suite.stat.bootstrap("rho", "Method", [("w", "d")], filters={"subject": "s1"}, **kwargs)
    s2 = suite.stat.bootstrap("rho", "Method", [("w", "d")], filters={"subject": "s2"}, **kwargs)
    assert s1.table.loc[0, "mean_diff"] == pytest.approx(0.1) and s1.table.loc[0, "n_boot"] == 400
    assert s1.table.loc[0, "p_value"] < 0.01 < s2.table.loc[0, "p_value"]
    assert s1.bootstrap["n_boot"] is None and s1.bootstrap["seed"] is None

    est = suite.estimate.from_result(s1)
    assert est.bootstrap["replicates"] == "rho_boot"
    assert est.table.loc[0, "ci_low"] < 0.1 < est.table.loc[0, "ci_high"]
    assert est.table.loc[0, "ci_low"] > 0

    assert suite.verify(s1)
    suite.register("rho_boot", 0.9, Method="w", subject="s1", boot=400)
    assert not suite.verify(s1)  # a new draw changes the data behind the result


def test_bootstrap_replicates_argument_checks():
    suite = _make_replicate_suite()
    comps = [("w", "d")]
    with pytest.raises(ValueError, match="separate metric"):
        suite.stat.bootstrap("rho", "Method", comps, pairing_col="subject", replicates="rho", replicate_col="boot")
    with pytest.raises(ValueError, match="don't apply"):
        suite.stat.bootstrap("rho", "Method", comps, pairing_col="subject", replicates="rho_boot", replicate_col="boot", n_boot=10)
    with pytest.raises(ValueError, match="No rows registered"):
        suite.stat.bootstrap("rho", "Method", comps, pairing_col="subject", replicates="nope", replicate_col="boot")
    suite.register("rho_boot", 0.5, Method="d", subject="s1", boot=999)
    with pytest.raises(ValueError, match="boot=999"):
        suite.stat.bootstrap("rho", "Method", comps, pairing_col="subject", replicates="rho_boot", replicate_col="boot")


def test_plots_accept_a_bootstrap_result():
    suite = _make_compare_suite()
    comps = [("reference", "shifted"), ("reference", "noise")]
    result = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=500, seed=0)
    plot = suite.plot.paired_differences(result)
    est = suite.estimate.from_result(result).vs("reference")
    np.testing.assert_allclose(plot.data.set_index("Method").loc[est.index, "center"], est["estimate"])
    assert plot.data.attrs["center"] == "mean_diff"
    assert suite.plot.group_comparison(result, kind="point").data.attrs["center"] == "mean"

    dz = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", estimator="cohens_dz", n_boot=300, seed=0)
    plot = suite.plot.paired_differences(dz, annotate=("stars", "effect"))
    assert plot.data["center"].isna().all()  # dz isn't in the differences' units: no overlay
    plt.close("all")


def test_plot_kws_override_matplotlib_aliases_of_the_defaults():
    """A user's `edgecolor` replaces the default `edgecolors` instead of raising matplotlib's alias clash."""
    suite = _make_compare_suite()
    plot = suite.plot.strip("outcome", x="Method", points_kws={"edgecolor": "black", "lw": 2})
    assert plot.artists["points"]
    plt.close("all")


def test_verify_is_false_when_the_result_filters_on_a_column_the_suite_lacks():
    """A result filtered on e.g. `subject` doesn't come from a suite that has no `subject` column -- False, not an error."""
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon("outcome", "Method", [("reference", "shifted")], pairing_col="region",
                                 filters={"region": ["r0", "r1", "r2", "r3", "r4", "r5"]})
    other = MetricSuite()
    other.register("outcome", 1.0, Method="reference", subject="s1")
    assert suite.verify(result)
    assert other.verify(result) is False


def test_paired_differences_labels_the_offset_line_and_can_hide_it():
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon("outcome", "Method", [("reference", "shifted")], pairing_col="region", offset=1.0)
    plot = suite.plot.paired_differences(result)
    assert len(plot.artists["offset"]) == 1 and plot.artists["offset"][0].get_ydata()[0] == 1.0
    assert [t.get_text() for t in plot.ax.get_legend().get_texts()] == ["offset (+1)"]

    plot = suite.plot.paired_differences(result, labels={"offset": "δ"}, offset_kws={"color": "red", "lw": 2})
    assert [t.get_text() for t in plot.ax.get_legend().get_texts()] == ["δ"]
    assert plot.artists["offset"][0].get_color() == "red"

    assert suite.plot.paired_differences(result, legend=False).ax.get_legend() is None
    hidden = suite.plot.paired_differences(result, show_offset=False)
    assert hidden.artists["offset"] == [] and hidden.ax.get_legend() is None

    no_offset = suite.stat.wilcoxon("outcome", "Method", [("reference", "shifted")], pairing_col="region")
    assert suite.plot.paired_differences(no_offset).ax.get_legend() is None
    plt.close("all")


# -- Confidence intervals inside test results ---------------------------------------------------


def test_tests_report_a_ci_for_every_effect_size_matching_suite_estimate():
    suite = _make_compare_suite()
    comps = [("reference", "shifted"), ("reference", "noise")]
    result = suite.stat.wilcoxon("outcome", "Method", comps, pairing_col="region",
                                 effect_sizes=("rank_biserial", "hodges_lehmann"), ci_seed=3)
    assert result.ci and result.ci_level == 0.95 and result.ci_seed == 3
    assert result.ci_methods == {"rank_biserial": "bootstrap", "hodges_lehmann": "analytic"}

    hl = suite.estimate.from_result(result).table
    np.testing.assert_allclose(result.table["hodges_lehmann_ci_low"], hl["ci_low"])
    np.testing.assert_allclose(result.table["hodges_lehmann_ci_high"], hl["ci_high"])
    r = suite.estimate.rank_biserial("outcome", "Method", comps, pairing_col="region", ci_method="bootstrap", seed=3).table
    np.testing.assert_allclose(result.table["rank_biserial_ci_low"], r["ci_low"])
    np.testing.assert_allclose(result.table["rank_biserial_ci_high"], r["ci_high"])


def test_vs_orients_the_ci_with_its_effect_size():
    suite = _make_compare_suite()
    result = suite.stat.paired_ttest("outcome", "Method", [("shifted", "reference")], pairing_col="region",
                                     effect_sizes=("mean_diff",))
    row, oriented = result.table.iloc[0], result.vs("reference").iloc[0]
    assert oriented["mean_diff"] == pytest.approx(-row["mean_diff"])
    assert oriented["mean_diff_ci_low"] == pytest.approx(-row["mean_diff_ci_high"])
    assert oriented["mean_diff_ci_high"] == pytest.approx(-row["mean_diff_ci_low"])
    assert oriented["mean_diff_ci_low"] < oriented["mean_diff"] < oriented["mean_diff_ci_high"]


def test_ci_options_level_off_and_ttest_ess():
    suite = _make_compare_suite()
    comps = [("reference", "noise")]
    wide = suite.stat.paired_ttest("outcome", "Method", comps, pairing_col="region", effect_sizes=("mean_diff",)).table.iloc[0]
    narrow = suite.stat.paired_ttest("outcome", "Method", comps, pairing_col="region", effect_sizes=("mean_diff",),
                                     ci_level=0.9).table.iloc[0]
    assert wide["mean_diff_ci_low"] < narrow["mean_diff_ci_low"] < narrow["mean_diff_ci_high"] < wide["mean_diff_ci_high"]

    off = suite.stat.wilcoxon("outcome", "Method", comps, pairing_col="region", ci=False)
    assert not off.ci and off.ci_methods is None and "rank_biserial_ci_low" not in off.table.columns

    regions = [f"r{i}" for i in range(20)]
    ess = suite.stat.ttest_ess("outcome", "Method", comps, pairing_col="region", subject_order=regions,
                               effect_sizes=("cohens_dz", "mean_diff"))
    assert ess.ci_methods == {"cohens_dz": None, "mean_diff": "analytic"} and ess.ci_seed is None
    assert ess.table["cohens_dz_ci_low"].isna().all() and ess.table["mean_diff_ci_low"].notna().all()
    assert "[no CI]" in repr(ess)
    with pytest.raises(ValueError, match="ci_level"):
        suite.stat.wilcoxon("outcome", "Method", comps, pairing_col="region", ci_level=95)


def test_ci_survives_save_load_and_feeds_the_plot(tmp_path):
    suite = _make_compare_suite()
    result = suite.stat.wilcoxon("outcome", "Method", [("reference", "shifted"), ("reference", "noise")],
                                 pairing_col="region", effect_sizes=("rank_biserial", "hodges_lehmann"))
    result.save(tmp_path/"r")
    loaded = PairwiseResult.load(tmp_path/"r")
    assert (loaded.ci, loaded.ci_level, loaded.ci_methods, loaded.ci_seed) == (
        result.ci, result.ci_level, result.ci_methods, result.ci_seed)
    pd.testing.assert_frame_equal(loaded.table, result.table, check_dtype=False)

    plot = suite.plot.paired_differences(result)
    vs = result.vs("reference")
    data = plot.data.set_index("Method").loc[vs.index]
    np.testing.assert_allclose(data["low"], vs["hodges_lehmann_ci_low"])
    np.testing.assert_allclose(data["high"], vs["hodges_lehmann_ci_high"])
    plt.close("all")


def test_independent_tests_bootstrap_their_rank_effect_sizes():
    suite = _make_independent_suite()
    result = suite.stat.mannwhitneyu("outcome", "Method", [("a", "b")], effect_sizes=("rank_biserial", "cles", "hodges_lehmann"))
    assert result.ci_methods == {"rank_biserial": "bootstrap", "cles": "bootstrap", "hodges_lehmann": "analytic"}
    row = result.table.iloc[0]
    for name in ("rank_biserial", "cles", "hodges_lehmann"):
        assert row[f"{name}_ci_low"] <= row[name] <= row[f"{name}_ci_high"]
    assert result.vs("b").iloc[0]["cles_ci_low"] == pytest.approx(1 - row["cles_ci_high"])


# ---------------------------
# paired_differences: bootstrap distribution, palette; line: smoothing
# ---------------------------


def _make_spike_bootstrap_suite(n_subjects=6, n_boot=300, seed=0):
    """One observed value per (Method, subject) plus `n_boot` registered draws each -- Study III's shape."""
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for s in range(n_subjects):
        base = rng.normal()
        for method, shift in [("w", 0.5), ("a", 0.0), ("b", 0.4)]:
            suite.register("rho", base + shift, Method=method, subject=f"s{s}")
            suite.register("rho_boot", base + shift + rng.gamma(2, 0.1, size=n_boot), Method=method,
                           subject=f"s{s}", boot=np.arange(n_boot))
    return suite


def test_paired_differences_bootstrap_draws_registered_replicates():
    """Drawn values = estimator(reference draws) - estimator(competitor draws), per draw; the CI is their percentile interval."""
    from matplotlib.collections import PolyCollection

    suite = _make_spike_bootstrap_suite()
    result = suite.stat.bootstrap("rho", "Method", [("w", "a"), ("w", "b")], pairing_col="subject",
                                  replicates="rho_boot", replicate_col="boot")
    plot = suite.plot.paired_differences(result)
    assert plot.data.attrs["distribution"] == "bootstrap"
    assert len([c for c in plot.ax.collections if isinstance(c, PolyCollection)]) == 2

    boot = suite.to_df(metrics="rho_boot").pivot_table(index=["boot", "subject"], columns="Method", values="Value")
    table = result.vs("w").set_index(result.vs("w").index)
    for c in ["a", "b"]:
        expected = (boot["w"] - boot[c]).groupby(level="boot").mean().to_numpy()
        drawn = plot.points.loc[plot.points["Method"] == c].sort_values("draw")["Value"].to_numpy()
        np.testing.assert_allclose(drawn, expected)
        assert np.median(drawn) == pytest.approx(table.loc[c, "mean_diff"], abs=0.05)
        row = plot.data.set_index("Method").loc[c]
        assert (row["low"], row["high"]) == pytest.approx((table.loc[c, "mean_diff_ci_low"], table.loc[c, "mean_diff_ci_high"]))
        lo, hi = np.quantile(drawn, 0.025, method="lower"), np.quantile(drawn, 0.975, method="higher")
        assert (lo, hi) == pytest.approx((row["low"], row["high"]))
    plt.close("all")


def test_paired_differences_bootstrap_sign_flips_for_lower_is_better():
    suite = _make_spike_bootstrap_suite()
    result = suite.stat.bootstrap("rho", "Method", [("w", "a")], pairing_col="subject",
                                  replicates="rho_boot", replicate_col="boot")
    up = suite.plot.paired_differences(result).points["Value"].to_numpy()
    down = suite.plot.paired_differences(result, higher_is_better=False).points["Value"].to_numpy()
    np.testing.assert_allclose(down, -up)
    assert suite.plot.paired_differences(result, higher_is_better=False).data["center"].iloc[0] < 0
    plt.close("all")


def test_paired_differences_bootstrap_resampled_matches_from_result():
    """Resampled draws are redrawn from the result's seed -- the same ones its test and from_result used."""
    suite = _make_compare_suite()
    comps = [("reference", "shifted"), ("noise", "reference")]
    result = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=800, seed=4)
    plot = suite.plot.paired_differences(result)
    est = suite.estimate.from_result(result).vs("reference")
    for c in ["shifted", "noise"]:
        drawn = plot.points.loc[plot.points["Method"] == c, "Value"].to_numpy()
        lo, hi = np.quantile(drawn, 0.025, method="lower"), np.quantile(drawn, 0.975, method="higher")
        assert (lo, hi) == pytest.approx((est.loc[c, "ci_low"], est.loc[c, "ci_high"]))
    plt.close("all")


def test_paired_differences_bootstrap_survives_save_and_load(tmp_path):
    suite = _make_spike_bootstrap_suite()
    result = suite.stat.bootstrap("rho", "Method", [("w", "a")], pairing_col="subject",
                                  replicates="rho_boot", replicate_col="boot")
    result.save(tmp_path / "boot")
    loaded = PairwiseResult.load(tmp_path / "boot")
    np.testing.assert_allclose(
        suite.plot.paired_differences(loaded).points["Value"], suite.plot.paired_differences(result).points["Value"],
    )
    plt.close("all")


def test_paired_differences_distribution_argument():
    suite = _make_compare_suite()
    comps = [("reference", "shifted")]
    boot = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", n_boot=300, seed=0)
    units = suite.plot.paired_differences(boot, distribution="units")
    assert units.data.attrs["distribution"] == "units" and len(units.points) == 20
    with pytest.raises(ValueError, match="distribution='bootstrap' needs"):
        suite.plot.paired_differences(suite.stat.wilcoxon("outcome", "Method", comps, pairing_col="region"),
                                      distribution="bootstrap")
    dz = suite.stat.bootstrap("outcome", "Method", comps, pairing_col="region", estimator="cohens_dz", n_boot=300, seed=0)
    assert suite.plot.paired_differences(dz).data.attrs["distribution"] == "units"
    with pytest.raises(ValueError, match="distribution='bootstrap' needs"):
        suite.plot.paired_differences(dz, distribution="bootstrap")
    with pytest.raises(ValueError, match="isn't supported"):
        suite.plot.paired_differences(boot, distribution="draws")
    plt.close("all")


def test_paired_differences_palette_colors_center_and_interval():
    from matplotlib.colors import to_rgba

    suite = _make_vs_suite()
    result = suite.stat.wilcoxon("outcome", "Method", [("reference", "worse"), ("reference", "noise")], pairing_col="region")
    palette = {"worse": "red", "noise": "blue"}
    plot = suite.plot.paired_differences(result, palette=palette)
    bar_colors = [to_rgba(container.lines[2][0].get_color()[0]) for container in plot.artists["intervals"]]
    assert bar_colors == [to_rgba("red"), to_rgba("blue")]
    assert [to_rgba(line.get_color()) for line in plot.artists["centers"]] == [to_rgba("red"), to_rgba("blue")]

    plot = suite.plot.paired_differences(result, palette=palette, interval_kws={"color": "k"})
    assert all(to_rgba(c.lines[2][0].get_color()[0]) == to_rgba("k") for c in plot.artists["intervals"])
    plt.close("all")


def _line_suite(values_by_time, times):
    suite = MetricSuite()
    for subj, values in enumerate(values_by_time):
        suite.register("v", values, Method="m", subject=subj, times=times)
    return suite


def test_line_smooth_keeps_a_constant_signal_constant_up_to_the_edges():
    times = np.round(np.arange(0, 0.1, 0.001), 4)
    plot = _line_suite([np.full(times.size, 3.0)] * 3, times).plot.line("v", x="times", smooth=0.02)
    np.testing.assert_allclose(plot.data["center_smooth"], 3.0)
    np.testing.assert_allclose(plot.data["center"], 3.0)
    plt.close("all")


def test_line_smooth_reduces_white_noise_variance_by_the_window_length():
    rng = np.random.default_rng(0)
    times = np.round(np.arange(0, 2.0, 0.001), 4)
    plot = _line_suite([rng.normal(size=times.size)], times).plot.line("v", x="times", interval=None, smooth=0.0205)
    interior = plot.data.iloc[50:-50]  # a 21-sample window, full away from the edges
    ratio = interior["center"].var() / interior["center_smooth"].var()
    assert ratio == pytest.approx(21, rel=0.2)
    assert len(plot.artists["raw"]) == 1
    assert "raw" not in _line_suite([rng.normal(size=times.size)], times).plot.line(
        "v", x="times", smooth=0.02, raw=False).artists
    plt.close("all")


def test_line_smooth_follows_x_not_the_drawing_order():
    times = np.arange(10, dtype=float)
    suite = _line_suite([times ** 2] * 2, times)
    plain = suite.plot.line("v", x="times", smooth=3).data.set_index("times")["center_smooth"]
    reversed_ = suite.plot.line("v", x="times", smooth=3, order=list(times[::-1])).data.set_index("times")["center_smooth"]
    pd.testing.assert_series_equal(plain.sort_index(), reversed_.sort_index())
    assert plain.loc[0.0] == pytest.approx((0 + 1) / 2)  # the window shrinks at the edge
    plt.close("all")


def test_line_smooth_argument_checks():
    suite = _make_compare_suite()
    with pytest.raises(ValueError, match="numeric `x`"):
        suite.plot.line("outcome", x="Method", smooth=1)
    times = np.arange(5, dtype=float)
    line_suite = _line_suite([times], times)
    with pytest.raises(ValueError, match="positive window"):
        line_suite.plot.line("v", x="times", smooth=0)
    with pytest.raises(ValueError, match="needs `smooth`"):
        line_suite.plot.line("v", x="times", raw=True)
    plt.close("all")


# -- Bootstrap: basic / BCa intervals and bias diagnostics -------------------

def _make_attenuated_suite(n_subjects=8, n_boot=1000, n_spikes=12, attenuation=0.95, seed=0):
    """
    Study III's shape: one observed value per (Method, subject), and draws pulled towards 0
    (`attenuation` x the observed value, plus noise) -- so the draws are biased towards 0 --
    plus a registered leave-one-spike-out jackknife.
    """
    suite = MetricSuite()
    rng = np.random.default_rng(seed)
    for s in range(n_subjects):
        base = rng.normal(0, 0.2)
        for method, shift in [("w", 0.6), ("a", 0.0)]:
            obs = base + shift
            suite.register("rho", obs, Method=method, subject=f"s{s}")
            suite.register("rho_boot", attenuation * obs + rng.normal(0, 0.1, size=n_boot), Method=method,
                           subject=f"s{s}", boot=np.arange(n_boot))
            suite.register("rho_jack", obs + rng.normal(0, 0.01, size=n_spikes), Method=method, subject=f"s{s}",
                           left_out=np.arange(n_spikes))
    return suite


_ATTENUATED = dict(pairing_col="subject", replicates="rho_boot", replicate_col="boot")


def test_bootstrap_bias_warning_fires_for_biased_draws_and_names_the_remedy():
    suite = _make_attenuated_suite()
    with pytest.warns(UserWarning, match="consider ci_method='basic' or 'bca'"):
        result = suite.stat.bootstrap("rho", "Method", [("w", "a")], **_ATTENUATED)
    row = result.table.iloc[0]
    assert row["boot_bias"] == pytest.approx(-0.05 * row["mean_diff"], abs=0.005)
    assert row["bias_se_ratio"] < -0.25 and row["z0"] > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="basic", **_ATTENUATED)
        unbiased = _make_compare_suite(n_regions=60)
        suite_result = unbiased.stat.bootstrap("outcome", "Method", [("reference", "noise")], pairing_col="region",
                                               n_boot=2000, seed=0)
    assert abs(suite_result.table.loc[0, "bias_se_ratio"]) < 0.25


def test_bootstrap_basic_interval_corrects_the_attenuation_the_percentile_one_inherits():
    suite = _make_attenuated_suite()
    with pytest.warns(UserWarning):
        pct = suite.stat.bootstrap("rho", "Method", [("w", "a")], **_ATTENUATED)
    basic = suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="basic", **_ATTENUATED)
    est = basic.table.loc[0, "mean_diff"]
    pct_mid = pct.table.loc[0, ["mean_diff_ci_low", "mean_diff_ci_high"]].mean()
    basic_mid = basic.table.loc[0, ["mean_diff_ci_low", "mean_diff_ci_high"]].mean()
    bias = pct.table.loc[0, "boot_bias"]
    assert bias < 0
    assert pct_mid == pytest.approx(est + bias, abs=0.005)  # pulled by the bias, towards 0
    assert basic_mid == pytest.approx(est - bias, abs=0.005)  # reflected: corrected away from 0
    assert basic.ci_methods == {"mean_diff": "bootstrap_basic"} and pct.ci_methods == {"mean_diff": "bootstrap"}
    assert basic.bootstrap["interval"] == "basic"


def test_bootstrap_basic_pvalue_agrees_with_its_interval():
    rng = np.random.default_rng(7)
    for seed in range(15):
        suite = MetricSuite()
        regions = [f"r{i}" for i in range(12)]
        ref = rng.normal(size=12)
        suite.register("outcome", ref, Method="a", region=regions)
        suite.register("outcome", ref - rng.normal(0.6, 1.0, size=12), Method="b", region=regions)
        r = suite.stat.bootstrap("outcome", "Method", [("a", "b")], pairing_col="region", n_boot=999, seed=seed,
                                 ci_method="basic").table.iloc[0]
        excludes = r["mean_diff_ci_low"] > 0 or r["mean_diff_ci_high"] < 0
        if abs(r["p_value"] - 0.05) > 2 / 1000:  # away from the one-draw rounding at the interval end
            assert (r["p_value"] < 0.05) == excludes


def test_bootstrap_draws_of_a_basic_result_are_the_reflected_draws():
    suite = _make_attenuated_suite()
    basic = suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="basic", **_ATTENUATED)
    with pytest.warns(UserWarning):
        pct = suite.stat.bootstrap("rho", "Method", [("w", "a")], **_ATTENUATED)
    est = basic.table.loc[0, "mean_diff"]
    reflected = suite.stat.bootstrap_draws(basic, reference="w")["Value"].to_numpy()
    plain = suite.stat.bootstrap_draws(pct, reference="w")["Value"].to_numpy()
    np.testing.assert_allclose(reflected, 2 * est - plain)
    from mindmap.core.stats import percentile_interval
    vs = basic.vs("w")
    assert percentile_interval(reflected) == pytest.approx((vs.loc["a", "mean_diff_ci_low"], vs.loc["a", "mean_diff_ci_high"]))
    plt.close("all")


def test_bootstrap_bca_matches_scipy_on_a_resampled_case():
    from scipy import stats as scipy_stats

    suite = _make_compare_suite(n_regions=25)
    result = suite.stat.bootstrap("outcome", "Method", [("noise", "reference")], pairing_col="region",
                                  n_boot=40_000, seed=1, ci_method="bca")
    df = suite.to_df(metrics="outcome").pivot_table(index="region", columns="Method", values="Value")
    d = (df["noise"] - df["reference"]).to_numpy()
    ref = scipy_stats.bootstrap((d,), np.mean, method="BCa", n_resamples=40_000, random_state=2)
    row = result.table.iloc[0]
    tol = 0.05 * ref.standard_error
    assert row["mean_diff_ci_low"] == pytest.approx(ref.confidence_interval.low, abs=tol)
    assert row["mean_diff_ci_high"] == pytest.approx(ref.confidence_interval.high, abs=tol)
    jack = np.array([np.delete(d, i).mean() for i in range(d.size)])
    u = jack.mean() - jack
    assert row["acceleration"] == pytest.approx(np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5))
    assert result.ci_methods == {"mean_diff": "bootstrap_bca"}


def test_bootstrap_bca_with_registered_replicates_needs_a_jackknife_or_an_acceleration():
    suite = _make_attenuated_suite()
    with pytest.raises(ValueError, match="jackknife.*acceleration=0"):
        suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="bca", **_ATTENUATED)
    bc = suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="bca", acceleration=0, **_ATTENUATED)
    assert bc.table.loc[0, "acceleration"] == 0.0
    jk = suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="bca", jackknife="rho_jack",
                              jackknife_col="left_out", **_ATTENUATED)
    jack = suite.to_df(metrics="rho_jack").pivot_table(index=["left_out", "subject"], columns="Method", values="Value")
    values = (jack["w"] - jack["a"]).groupby(level="left_out").mean().to_numpy()
    u = values.mean() - values
    assert jk.table.loc[0, "acceleration"] == pytest.approx(np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5))
    assert suite.verify(jk)
    with pytest.raises(ValueError, match="not both"):
        suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="bca", acceleration=0, jackknife="rho_jack",
                             jackknife_col="left_out", **_ATTENUATED)
    with pytest.raises(ValueError, match="only apply to the 'bca'"):
        suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="basic", acceleration=0, **_ATTENUATED)


def test_bootstrap_basic_refuses_non_additive_estimators():
    suite = _make_compare_suite()
    with pytest.raises(ValueError, match="additive"):
        suite.stat.bootstrap("outcome", "Method", [("reference", "noise")], pairing_col="region",
                             estimator="rank_biserial", n_boot=200, seed=0, ci_method="basic")


def test_bootstrap_interval_carries_over_to_from_result_and_save_load(tmp_path):
    suite = _make_attenuated_suite()
    jk = suite.stat.bootstrap("rho", "Method", [("w", "a")], ci_method="bca", jackknife="rho_jack",
                              jackknife_col="left_out", **_ATTENUATED)
    est = suite.estimate.from_result(jk)
    assert est.table.loc[0, "ci_method"] == "bootstrap_bca"
    assert est.table.loc[0, ["ci_low", "ci_high"]].tolist() == pytest.approx(
        jk.table.loc[0, ["mean_diff_ci_low", "mean_diff_ci_high"]].tolist())
    jk.save(tmp_path / "r")
    loaded = PairwiseResult.load(tmp_path / "r")
    assert loaded.bootstrap == jk.bootstrap and loaded.ci_methods == jk.ci_methods
    pd.testing.assert_frame_equal(loaded.table, jk.table, check_dtype=False)
    assert suite.verify(loaded)


def test_bootstrap_bias_columns_follow_the_vs_orientation():
    suite = _make_attenuated_suite()
    result = suite.stat.bootstrap("rho", "Method", [("a", "w")], ci_method="basic", **_ATTENUATED)
    row, vs = result.table.iloc[0], result.vs("w")
    assert vs.loc["a", "boot_bias"] == pytest.approx(-row["boot_bias"])
    assert vs.loc["a", "z0"] == pytest.approx(-row["z0"])
    assert vs.loc["a", "boot_se"] == pytest.approx(row["boot_se"])


# -- line: highlight, per-level styles, spans and vlines ----------------------

def _three_method_line_suite(step=0.005):
    suite = MetricSuite()
    times = np.round(np.arange(1.9, 2.1 + step / 2, step), 4)
    rng = np.random.default_rng(0)
    for m in ["wCSET", "sLORETA", "dSPM"]:
        for subj in range(3):
            suite.register("rho", np.sin(times * 20) + rng.normal(0, 0.2, times.size), Method=m, subject=subj, times=times)
    return suite


def test_line_highlight_thickens_one_level_on_top_and_leaves_raw_lines_thin():
    plot = _three_method_line_suite().plot.line("rho", "times", hue="Method", unit="subject", smooth=0.02,
                                                hue_order=["wCSET", "sLORETA", "dSPM"], highlight="wCSET")
    widths = [line.get_linewidth() for line in plot.artists["lines"]]
    assert widths == [2.2, 1.5, 1.5]
    assert plot.artists["lines"][0].get_zorder() > plot.artists["lines"][1].get_zorder()
    assert {line.get_linewidth() for line in plot.artists["raw"]} == {0.6}
    assert [h.get_linewidth() for h in plot.ax.get_legend().get_lines()] == [2.2, 1.5, 1.5]
    with pytest.raises(ValueError, match="aren't `hue` levels"):
        _three_method_line_suite().plot.line("rho", "times", hue="Method", highlight="MNE")
    plt.close("all")


def test_line_kws_per_level_styles_single_lines_over_the_highlight():
    plot = _three_method_line_suite().plot.line(
        "rho", "times", hue="Method", hue_order=["wCSET", "sLORETA", "dSPM"], highlight="wCSET",
        line_kws={"sLORETA": {"linestyle": "--"}, "wCSET": {"linewidth": 3.0}},
    )
    wcset, sloreta, dspm = plot.artists["lines"]
    assert wcset.get_linewidth() == 3.0 and sloreta.get_linestyle() == "--" and dspm.get_linestyle() == "-"
    plt.close("all")


def test_line_spans_snap_to_whole_samples_and_join_the_legend():
    plot = _three_method_line_suite(step=0.005).plot.line(
        "rho", "times", hue="Method", spans=[(1.99, 2.01, "spike peak")], span_snap=True, vlines=[(2.0, "peak"), 1.95],
    )
    (start, end, label), = plot.data.attrs["spans"]
    assert (start, end) == pytest.approx((1.9875, 2.0125))  # 5 samples at 200 Hz -> 25 ms
    assert label == "spike peak"
    assert plot.artists["spans"][0].get_zorder() < min(line.get_zorder() for line in plot.artists["lines"])
    legend = [t.get_text() for t in plot.ax.get_legend().get_texts()]
    assert legend[-2:] == ["spike peak", "peak"] and len(plot.artists["vlines"]) == 2
    unsnapped = _three_method_line_suite().plot.line("rho", "times", spans=[(1.99, 2.01)])
    assert unsnapped.data.attrs["spans"] == [(1.99, 2.01, None)] and unsnapped.ax.get_legend() is None
    plt.close("all")


def test_line_spans_need_a_numeric_x():
    suite = MetricSuite()
    for c in ["a", "b"]:
        suite.register("v", [1.0, 2.0], Method="m", cond=c, subject=[0, 1])
    with pytest.raises(ValueError, match="numeric `x`"):
        suite.plot.line("v", "cond", spans=[(0, 1)])
    plt.close("all")


# -- Shared font size and tick rotation ---------------------------------------

def test_style_fontsize_reaches_every_method_and_is_restored():
    from mindmap.metric import plot as plot_module

    suite = _make_spike_bootstrap_suite()
    result = suite.stat.bootstrap("rho", "Method", [("w", "a"), ("w", "b")], pairing_col="subject",
                                  replicates="rho_boot", replicate_col="boot")
    with plot_module.style(fontsize=8):
        diffs = suite.plot.paired_differences(result, direction_labels=("w better", "w worse"))
        line = _three_method_line_suite().plot.line("rho", "times", hue="Method")
    for p in (diffs, line):
        ax = p.ax
        assert ax.xaxis.label.get_fontsize() == 8 and ax.yaxis.label.get_fontsize() == 8
        assert {t.get_fontsize() for t in ax.get_xticklabels() + ax.get_yticklabels()} == {8}
    annotation_sizes = {t.get_fontsize() for t in diffs.ax.texts}
    assert annotation_sizes <= {8, 7} and 8 in annotation_sizes
    assert line.ax.get_legend().get_texts()[0].get_fontsize() == 8
    assert plot_module._draw.STYLE["fontsize"] is None
    after = _three_method_line_suite().plot.line("rho", "times")
    assert after.ax.xaxis.label.get_fontsize() == matplotlib.rcParams["font.size"]
    with pytest.raises(ValueError, match="positive"):
        plot_module.set_style(fontsize=0)
    plt.close("all")


def test_rotate_ticks_auto_rotates_only_overlapping_labels():
    def labels_suite(names):
        suite = MetricSuite()
        for n in names:
            suite.register("v", np.arange(5.0), Method=n, subject=np.arange(5))
        return suite

    long_names = [f"a rather long method name {i}" for i in range(6)]
    crowded = labels_suite(long_names).plot.box("v", "Method")
    assert {t.get_rotation() for t in crowded.ax.get_xticklabels()} == {45}
    roomy = labels_suite(["a", "b"]).plot.box("v", "Method")
    assert {t.get_rotation() for t in roomy.ax.get_xticklabels()} == {0}
    fixed = labels_suite(["a", "b"]).plot.point("v", "Method", rotate_ticks=30)
    assert {t.get_rotation() for t in fixed.ax.get_xticklabels()} == {30}
    off = labels_suite(long_names).plot.violin("v", "Method", rotate_ticks=False)
    assert {t.get_rotation() for t in off.ax.get_xticklabels()} == {0}
    with pytest.raises(ValueError, match="rotate_ticks"):
        labels_suite(["a", "b"]).plot.bar("v", "Method", rotate_ticks="yes")
    plt.close("all")


def test_paired_differences_rotate_ticks():
    suite = _make_spike_bootstrap_suite()
    result = suite.stat.bootstrap("rho", "Method", [("w", "a"), ("w", "b")], pairing_col="subject",
                                  replicates="rho_boot", replicate_col="boot")
    plot = suite.plot.paired_differences(result, rotate_ticks=45)
    assert {t.get_rotation() for t in plot.ax.get_xticklabels()} == {45}
    plt.close("all")
