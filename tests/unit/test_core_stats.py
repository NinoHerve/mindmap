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
)


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
