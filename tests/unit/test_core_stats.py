"""
This module performs unit tests for the paired-comparison primitives in
core/stats.py.
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from mindmap.core.stats import (
    paired_test,
    correct_pvalues,
    compare_to_reference,
    cluster_threshold,
    effective_sample_size,
)


def _make_ar1(n, rho, seed, offset=0.0):
    """AR(1) series: x[i] = rho*x[i-1] + noise[i], plus a constant offset."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + noise[i]
    return x + offset


def test_paired_test_wilcoxon_detects_known_shift():
    """
    A consistent positive shift (x always well above y, plus small noise)
    should be detected with a small p-value and a positive effect size
    close to 1 (nearly every pair has x > y).
    """
    rng = np.random.default_rng(0)
    n = 30
    y = rng.normal(0, 1, size=n)
    x = y + 2.0 + rng.normal(0, 0.1, size=n)

    result = paired_test(x, y, test="wilcoxon")

    assert result["p_value"] < 0.001
    assert result["effect_size"] == pytest.approx(1.0, abs=0.05)
    assert result["effect_size_type"] == "rank_biserial"
    assert result["n"] == n


def test_paired_test_wilcoxon_null_case_effect_size_near_zero():
    """
    Symmetric-around-zero paired noise (no true difference) should give an
    effect size near 0. A fixed p-value threshold would be flaky across
    seeds (5% of true-null cases are expected to reject by chance), so this
    checks the more stable summary instead: the effect size shouldn't
    reliably indicate a large, consistent shift.
    """
    rng = np.random.default_rng(1)
    n = 200
    y = rng.normal(0, 1, size=n)
    x = y + rng.normal(0, 1, size=n)  # no systematic shift

    result = paired_test(x, y, test="wilcoxon")

    assert abs(result["effect_size"]) < 0.2


def test_paired_test_ttest_matches_scipy_ttest_rel():
    """Cross-check statistic/p_value against scipy.stats.ttest_rel directly."""
    rng = np.random.default_rng(2)
    x = rng.normal(1, 1, size=25)
    y = rng.normal(0, 1, size=25)

    result = paired_test(x, y, test="ttest")
    expected = scipy_stats.ttest_rel(x, y)

    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["effect_size_type"] == "cohens_dz"


def test_paired_test_rejects_unknown_test():
    with pytest.raises(ValueError):
        paired_test([1, 2], [1, 2], test="not_a_test")


def test_paired_test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_test([1, 2, 3], [1, 2])


def test_correct_pvalues_fdr_bh_matches_scipy():
    """Cross-check against scipy.stats.false_discovery_control directly."""
    p_values = np.array([0.001, 0.02, 0.03, 0.5, 0.8])
    p_corrected, significant = correct_pvalues(p_values, method="fdr_bh", alpha=0.05)

    expected = scipy_stats.false_discovery_control(p_values, method="bh")
    np.testing.assert_allclose(p_corrected, expected)
    np.testing.assert_array_equal(significant, p_corrected <= 0.05)


def test_correct_pvalues_bonferroni_exact():
    """Bonferroni is an exact p*n arithmetic check, clipped to 1."""
    p_values = np.array([0.01, 0.5, 0.9])
    p_corrected, significant = correct_pvalues(p_values, method="bonferroni", alpha=0.05)

    np.testing.assert_allclose(p_corrected, np.clip(p_values * 3, 0, 1))
    np.testing.assert_array_equal(significant, p_corrected <= 0.05)


def test_correct_pvalues_none_passthrough():
    p_values = np.array([0.2, 0.001, 0.9])
    p_corrected, _ = correct_pvalues(p_values, method="none")
    np.testing.assert_allclose(p_corrected, p_values)


def test_compare_to_reference_shapes_and_names():
    """
    m=3 competitors -> m results, names propagate in order, and FDR
    correction never makes a p-value smaller than the raw one
    (Benjamini-Hochberg is never more liberal than the uncorrected test).
    """
    rng = np.random.default_rng(3)
    n = 20
    reference = rng.normal(0, 1, size=n)
    competitors = np.stack([
        reference + 2.0 + rng.normal(0, 0.1, size=n),  # clearly different
        reference + rng.normal(0, 1, size=n),          # noise, no shift
        reference + 1.5 + rng.normal(0, 0.1, size=n),  # clearly different
    ], axis=1)
    names = ["strong", "null", "also_strong"]

    results = compare_to_reference(reference, competitors, competitor_names=names)

    assert [r["name"] for r in results] == names
    assert len(results) == 3
    for r in results:
        assert r["p_corrected"] >= r["p_value"] - 1e-12
    assert results[0]["significant"]
    assert results[2]["significant"]


def test_compare_to_reference_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_to_reference(np.zeros(5), np.zeros((4, 2)))


def test_effective_sample_size_white_noise_close_to_n():
    """No serial correlation -> n_eff should sit right at n (nothing to correct)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    n_eff = effective_sample_size(x)
    assert n_eff == pytest.approx(300, abs=1)


def test_effective_sample_size_shrinks_with_autocorrelation():
    """
    Strong AR(1) serial correlation must give a substantially smaller n_eff
    than the nominal n -- and n_eff should shrink further as rho increases,
    the core, falsifiable behavior this whole correction depends on.
    """
    n = 300
    x_low_rho = _make_ar1(n, rho=0.2, seed=5)
    x_high_rho = _make_ar1(n, rho=0.85, seed=5)

    n_eff_low = effective_sample_size(x_low_rho)
    n_eff_high = effective_sample_size(x_high_rho)

    assert n_eff_low < n
    assert n_eff_high < n_eff_low  # more autocorrelation -> smaller effective n


def test_effective_sample_size_clipped_to_n():
    """Never reports more "independent" points than were actually observed."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=50)
    assert effective_sample_size(x) <= 50


def test_paired_test_ttest_ess_matches_plain_ttest_when_iid():
    """
    With no serial correlation in the paired-difference series, ttest_ess's
    n_eff should recover ~n, and its p-value should closely match plain
    ttest's on the same data -- the correction should be a no-op here.
    """
    rng = np.random.default_rng(7)
    n = 200
    y = rng.normal(0, 1, size=n)
    x = y - 0.3 + rng.normal(0, 1, size=n)

    result_ttest = paired_test(x, y, test="ttest")
    result_ess = paired_test(x, y, test="ttest_ess")

    assert result_ess["n_eff"] == pytest.approx(n, abs=5)
    assert result_ess["p_value"] == pytest.approx(result_ttest["p_value"], abs=0.02)
    assert result_ess["effect_size_type"] == "cohens_dz"


def test_paired_test_ttest_ess_corrects_spurious_significance():
    """
    Regression guard for the actual point of ttest_ess: a small true shift
    riding on strong AR(1) autocorrelation can look "significant" under a
    plain ttest purely because the nominal n overstates the independent
    evidence -- ttest_ess must come out measurably less significant (larger
    p-value, smaller n_eff) on the identical data, not just "a different
    number".
    """
    diff = _make_ar1(n=200, rho=0.8, seed=8, offset=0.15)
    x, y = diff, np.zeros_like(diff)

    result_ttest = paired_test(x, y, test="ttest")
    result_ess = paired_test(x, y, test="ttest_ess")

    assert result_ess["n_eff"] < 200 * 0.5
    assert result_ess["p_value"] > result_ttest["p_value"]


def test_cluster_threshold_matches_t_ppf_formula():
    """Direct cross-check of the two-tailed and one-tailed threshold formulas."""
    n_subjects = 15

    two_tailed = cluster_threshold(n_subjects, alpha=0.05, tail=0)
    expected_two = scipy_stats.t.ppf(1 - 0.025, df=n_subjects - 1)
    assert two_tailed == pytest.approx(expected_two)

    upper = cluster_threshold(n_subjects, alpha=0.05, tail=1)
    expected_upper = scipy_stats.t.ppf(1 - 0.05, df=n_subjects - 1)
    assert upper == pytest.approx(expected_upper)

    lower = cluster_threshold(n_subjects, alpha=0.05, tail=-1)
    assert lower == pytest.approx(-expected_upper)
