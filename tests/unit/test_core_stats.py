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


# --- offset / alternative ----------------------------------------------------

from mindmap.core.stats import independent_test, estimate, effect_size, resolve_alternative


def test_resolve_alternative_defaults_to_greater_with_offset():
    assert resolve_alternative(0.0, None) == "two-sided"
    assert resolve_alternative(0.1, None) == "greater"
    assert resolve_alternative(-0.1, None) == "greater"
    assert resolve_alternative(0.1, "less") == "less"
    with pytest.raises(ValueError, match="alternative"):
        resolve_alternative(0.0, "bigger")


def test_paired_test_offset_matches_scipy_on_shifted_differences():
    rng = np.random.default_rng(0)
    x, y = rng.normal(1, 1, 25), rng.normal(0, 1, 25)
    for test, reference in [
        ("wilcoxon", lambda: scipy_stats.wilcoxon(x - y - 0.3, alternative="greater").pvalue),
        ("ttest", lambda: scipy_stats.ttest_rel(x - 0.3, y, alternative="greater").pvalue),
    ]:
        result = paired_test(x, y, test=test, offset=0.3)
        assert result["alternative"] == "greater"
        assert result["offset"] == 0.3
        assert result["p_value"] == pytest.approx(reference())


def test_paired_test_offset_zero_is_unchanged():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=15), rng.normal(size=15)
    assert paired_test(x, y)["p_value"] == pytest.approx(scipy_stats.wilcoxon(x, y).pvalue)
    assert paired_test(x, y, test="ttest")["p_value"] == pytest.approx(scipy_stats.ttest_rel(x, y).pvalue)


def test_paired_test_effect_sizes_ignore_the_offset():
    """The offset only changes the p-value; every effect size describes the raw x - y."""
    rng = np.random.default_rng(2)
    x, y = rng.normal(1, 1, 30), np.zeros(30)
    names = ("cohens_dz", "hedges_gz", "mean_diff")
    shifted = paired_test(x, y, test="ttest", offset=0.4, effect_sizes=names)
    plain = paired_test(x, y, test="ttest", effect_sizes=names)
    d = x - y
    assert shifted["effect_sizes"] == pytest.approx(plain["effect_sizes"])
    assert shifted["effect_sizes"]["cohens_dz"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert shifted["effect_sizes"]["mean_diff"] == pytest.approx(d.mean())
    assert abs(shifted["effect_sizes"]["hedges_gz"]) < abs(shifted["effect_sizes"]["cohens_dz"])
    assert shifted["p_value"] != pytest.approx(plain["p_value"])
    w = paired_test(x, y, offset=0.4, effect_sizes=("rank_biserial", "hodges_lehmann"))
    assert w["effect_sizes"] == pytest.approx(paired_test(x, y, effect_sizes=("rank_biserial", "hodges_lehmann"))["effect_sizes"])


def test_paired_test_reports_requested_effect_sizes_in_order():
    rng = np.random.default_rng(12)
    x, y = rng.normal(1, 1, 15), rng.normal(0, 1, 15)
    assert list(paired_test(x, y)["effect_sizes"]) == ["rank_biserial"]
    assert list(paired_test(x, y, test="ttest")["effect_sizes"]) == ["cohens_dz"]
    result = paired_test(x, y, effect_sizes=("hodges_lehmann", "rank_biserial"))
    assert list(result["effect_sizes"]) == ["hodges_lehmann", "rank_biserial"]
    assert result["effect_size_type"] == "hodges_lehmann"
    assert result["effect_size"] == pytest.approx(estimate(x, y)["estimate"])
    with pytest.raises(ValueError, match="aren't available"):
        paired_test(x, y, effect_sizes="cohens_dz")
    with pytest.raises(ValueError, match="aren't available"):
        independent_test(x, y, effect_sizes="cohens_dz")


def test_paired_test_less_alternative_mirrors_greater():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=20), rng.normal(size=20)
    greater = paired_test(x, y, test="ttest", offset=0.2, alternative="greater")["p_value"]
    less = paired_test(y, x, test="ttest", offset=-0.2, alternative="less")["p_value"]
    assert greater == pytest.approx(less)


def test_independent_test_rank_biserial_positive_when_a_higher():
    result = independent_test([5.0, 6, 7], [1.0, 2, 3], effect_sizes=("rank_biserial", "cles"))
    assert result["effect_size"] == pytest.approx(1.0)
    assert result["effect_sizes"]["cles"] == pytest.approx(1.0)


def test_independent_test_offset_shifts_a():
    rng = np.random.default_rng(4)
    a, b = rng.normal(2, 1, 20), rng.normal(0, 1, 25)
    result = independent_test(a, b, test="ttest_ind", offset=1.0)
    assert result["alternative"] == "greater"
    assert result["p_value"] == pytest.approx(scipy_stats.ttest_ind(a - 1.0, b, alternative="greater").pvalue)
    assert set(result["effect_sizes"]) == {"cohens_d"}
    raw = independent_test(a, b, test="ttest_ind", effect_sizes=("cohens_d", "hedges_g", "mean_diff"))
    shifted = independent_test(a, b, test="ttest_ind", offset=1.0, effect_sizes=("cohens_d", "hedges_g", "mean_diff"))
    assert shifted["effect_sizes"] == pytest.approx(raw["effect_sizes"])
    assert raw["effect_sizes"]["mean_diff"] == pytest.approx(a.mean() - b.mean())


# --- estimate ------------------------------------------------------------------


def test_estimate_hodges_lehmann_is_median_of_walsh_averages():
    rng = np.random.default_rng(5)
    d = rng.normal(size=31)
    walsh = [(d[i] + d[j]) / 2 for i in range(31) for j in range(i, 31)]
    result = estimate(d, np.zeros(31))
    assert result["estimate"] == pytest.approx(np.median(walsh))
    assert result["ci_low"] < result["estimate"] < result["ci_high"]
    assert result["ci_method"] == "signed_rank_exact"


def test_estimate_hodges_lehmann_interval_matches_one_sided_wilcoxon():
    """The 1-alpha HL interval's lower bound is exactly where the alpha/2 one-sided Wilcoxon test flips."""
    rng = np.random.default_rng(6)
    x = rng.normal(0.4, 1, 20)
    low = estimate(x, np.zeros(20), alpha=0.05)["ci_low"]
    below = paired_test(x, np.zeros(20), offset=low - 1e-9, alternative="greater")["p_value"]
    above = paired_test(x, np.zeros(20), offset=low + 1e-9, alternative="greater")["p_value"]
    assert below <= 0.025 < above


def test_estimate_mean_matches_scipy_t_interval():
    rng = np.random.default_rng(7)
    x = rng.normal(1, 2, 12)
    result = estimate(x, estimator="mean", alpha=0.1)
    low, high = scipy_stats.t.interval(0.9, df=11, loc=x.mean(), scale=scipy_stats.sem(x))
    assert (result["ci_low"], result["ci_high"]) == pytest.approx((low, high))


def test_estimate_median_binomial_interval_and_small_n():
    x = np.arange(1.0, 11.0)
    result = estimate(x, estimator="median")
    assert result["estimate"] == 5.5
    assert (result["ci_low"], result["ci_high"]) == (2.0, 9.0)  # k=2 for n=10 at 95%
    tiny = estimate([1.0, 2.0], estimator="median")
    assert np.isnan(tiny["ci_low"]) and np.isnan(tiny["ci_high"])


def test_estimate_independent_shift_and_median_rejection():
    rng = np.random.default_rng(8)
    a, b = rng.normal(1, 1, 40), rng.normal(0, 1, 35)
    result = estimate(a, b, paired=False)
    assert result["estimate"] == pytest.approx(np.median(np.subtract.outer(a, b)))
    assert (result["n_a"], result["n_b"]) == (40, 35)
    with pytest.raises(ValueError, match="describes one sample"):
        estimate(a, b, estimator="median", paired=False)
    with pytest.raises(ValueError, match="describes a difference"):
        estimate(a, estimator="hodges_lehmann")
    with pytest.raises(ValueError, match="isn't defined for paired"):
        estimate(a[:30], b[:30], estimator="cles")


def test_estimate_hodges_lehmann_refuses_huge_inputs(monkeypatch):
    import mindmap.core.stats as core_stats
    monkeypatch.setattr(core_stats, "_MAX_PAIRWISE_TERMS", 10)
    with pytest.raises(ValueError, match="cap"):
        estimate(np.arange(10.0), np.zeros(10))


def test_paired_test_wilcoxon_rank_biserial_is_matched_pairs_textbook():
    """Kerby (2014): (R+ - R-) / (R+ + R-), average ranks for ties, zero differences dropped."""
    d = np.array([3.0, -1.0, 2.0, 2.0, 0.0, 5.0, -4.0])
    # nonzero |d| = 3, 1, 2, 2, 5, 4 -> ranks 4, 1, 2.5, 2.5, 6, 5 ; R- = 1 + 5 = 6, total 21
    result = paired_test(d, np.zeros_like(d))
    assert result["effect_size"] == pytest.approx((21 - 2 * 6) / 21)
    assert result["n"] == 6


def test_correct_pvalues_holm_matches_reference_and_keeps_nans():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.2, np.nan])
    adjusted, significant = correct_pvalues(p, method="holm", alpha=0.05)
    # m = 5 valid: sorted 0.005, 0.01, 0.03, 0.04, 0.2 -> x5, x4, x3, x2, x1, running max
    expected = np.array([0.04, 0.09, 0.09, 0.025, 0.2, np.nan])
    np.testing.assert_allclose(adjusted, expected)
    assert list(significant) == [True, False, False, True, False, False]
    bonferroni, _ = correct_pvalues(p[:5], method="bonferroni")
    assert (adjusted[:5] <= bonferroni).all()


def test_estimate_point_values_match_the_tests_effect_sizes():
    """One definition: estimate(..., ci=False) gives exactly what a test reports."""
    rng = np.random.default_rng(13)
    x, y = rng.normal(1, 1, 25), rng.normal(0, 1, 25)
    paired = paired_test(x, y, test="ttest", effect_sizes=("cohens_dz", "hedges_gz", "mean_diff"))["effect_sizes"]
    paired.update(paired_test(x, y, effect_sizes=("rank_biserial", "hodges_lehmann"))["effect_sizes"])
    for name, value in paired.items():
        assert estimate(x, y, estimator=name, ci=False)["estimate"] == pytest.approx(value)
        assert effect_size(x, y, name) == pytest.approx(value)
    independent = independent_test(x, y, test="ttest_ind", effect_sizes=("cohens_d", "hedges_g", "mean_diff"))["effect_sizes"]
    independent.update(independent_test(x, y, effect_sizes=("rank_biserial", "cles", "hodges_lehmann"))["effect_sizes"])
    for name, value in independent.items():
        assert estimate(x, y, estimator=name, paired=False, ci=False)["estimate"] == pytest.approx(value)


def test_estimate_without_an_interval_raises_unless_ci_false():
    rng = np.random.default_rng(14)
    x, y = rng.normal(1, 1, 20), rng.normal(0, 1, 20)
    with pytest.raises(NotImplementedError, match="ci=False"):
        estimate(x, y, estimator="cohens_dz")
    result = estimate(x, y, estimator="cohens_dz", ci=False)
    assert np.isnan(result["ci_low"]) and result["ci_method"] is None
    assert np.isnan(estimate(x, y, estimator="mean_diff", ci=False)["ci_high"])


def test_estimate_mean_diff_interval_with_n_eff_matches_ttest_ess():
    """The n_eff interval's lower bound is exactly where the alpha/2 one-sided ttest_ess flips."""
    x = _make_ar1(80, 0.7, seed=15, offset=0.5)
    y = np.zeros(80)
    n_eff = effective_sample_size(x - y)
    result = estimate(x, y, estimator="mean_diff", n_eff=n_eff)
    plain = estimate(x, y, estimator="mean_diff")
    assert result["ci_method"] == "t_ess" and result["n_eff"] == pytest.approx(n_eff)
    assert result["ci_high"] - result["ci_low"] > plain["ci_high"] - plain["ci_low"]
    p = paired_test(x, y, test="ttest_ess", offset=result["ci_low"], alternative="greater")["p_value"]
    assert p == pytest.approx(0.025)
    with pytest.raises(NotImplementedError, match="effective-sample-size"):
        estimate(x, y, estimator="hodges_lehmann", n_eff=n_eff)


def test_rank_biserial_zero_methods_worked_example():
    """d = [0, 0, 0, -1, 2, 5, 7]: wilcox ranks the 4 nonzero (r = 8/10); pratt/zsplit rank all 7 (r = 14/28)."""
    from mindmap.core.stats import effect_size

    d = np.array([0, 0, 0, -1, 2, 5, 7], dtype=float)
    zeros = np.zeros_like(d)
    assert effect_size(d, zeros, "rank_biserial") == pytest.approx(0.8)
    assert effect_size(d, zeros, "rank_biserial", zero_method="pratt") == pytest.approx(0.5)
    assert effect_size(d, zeros, "rank_biserial", zero_method="zsplit") == pytest.approx(0.5)
    assert effect_size(zeros, zeros, "rank_biserial", zero_method="pratt") == 0.0


def test_paired_test_zero_method_counts_and_restrictions():
    from mindmap.core.stats import paired_test

    d = np.array([0, 0, 0, -1, 2, 5, 7, 3, 4, -2], dtype=float)
    zeros = np.zeros_like(d)
    wilcox = paired_test(d, zeros)
    pratt = paired_test(d, zeros, zero_method="pratt")
    assert (wilcox["n"], wilcox["n_zero"], wilcox["zero_method"]) == (7, 3, "wilcox")
    assert (pratt["n"], pratt["n_zero"], pratt["zero_method"]) == (10, 3, "pratt")
    assert pratt["p_value"] == pytest.approx(scipy_stats.wilcoxon(d, zero_method="pratt").pvalue)
    shifted = paired_test(d, zeros, offset=2.0, zero_method="pratt")  # zeros are counted after the shift
    assert shifted["n_zero"] == 1
    assert np.isnan(paired_test(zeros, zeros, zero_method="pratt")["p_value"])
    with pytest.raises(ValueError, match="zero_method"):
        paired_test(d, zeros, test="ttest", zero_method="pratt")
    with pytest.raises(ValueError, match="zero_method"):
        paired_test(d, zeros, zero_method="drop")


def _paired_binary(a, b, c, d):
    """Matched 0/1 outcomes from the paired 2x2 counts: a both 1, b only x, c only y, d both 0."""
    cells = [a, b, c, d]
    return np.repeat([1, 1, 0, 0], cells).astype(float), np.repeat([1, 0, 1, 0], cells).astype(float)


def test_newcombe_matches_published_example():
    """Fagerland, Lydersen & Laake (2014), airway hyper-responsiveness: a=1, b=1, c=7, d=12 -> -0.286 (-0.507 to -0.026)."""
    from mindmap.core.stats import estimate

    x, y = _paired_binary(1, 1, 7, 12)
    out = estimate(x, y, estimator="prop_diff")
    assert out["estimate"] == pytest.approx(-6 / 21)
    assert (out["ci_low"], out["ci_high"]) == pytest.approx((-0.507, -0.026), abs=5e-4)
    assert out["ci_method"] == "newcombe"


def test_newcombe_stays_in_bounds_and_contains_estimate():
    from mindmap.core.stats import estimate

    for cells in [(0, 10, 0, 0), (10, 0, 0, 0), (0, 0, 0, 10), (5, 0, 0, 5), (3, 4, 4, 3), (14, 6, 1, 5)]:
        out = estimate(*_paired_binary(*cells), estimator="prop_diff")
        assert -1 <= out["ci_low"] <= out["estimate"] <= out["ci_high"] <= 1


def test_mcnemar_exact_test_and_counts():
    from mindmap.core.stats import paired_test

    x, y = _paired_binary(14, 6, 1, 5)
    out = paired_test(x, y, test="mcnemar", effect_sizes=("prop_diff", "odds_ratio"))
    assert out["p_value"] == pytest.approx(scipy_stats.binomtest(6, 7, 0.5).pvalue) == pytest.approx(0.125)
    assert (out["n"], out["n_zero"], out["n_x_only"], out["n_y_only"], out["statistic"]) == (26, 19, 6, 1, 6)
    assert out["effect_sizes"] == pytest.approx({"prop_diff": 5 / 26, "odds_ratio": 6.0})
    greater = paired_test(x, y, test="mcnemar", alternative="greater")
    assert greater["p_value"] == pytest.approx(scipy_stats.binomtest(6, 7, 0.5, alternative="greater").pvalue)
    assert paired_test(*_paired_binary(5, 0, 0, 5), test="mcnemar")["p_value"] == 1.0  # no discordant pair


def test_mcnemar_rejects_offsets_non_binary_and_zero_method():
    from mindmap.core.stats import paired_test, effect_size, estimate

    x, y = _paired_binary(14, 6, 1, 5)
    with pytest.raises(ValueError, match="offset"):
        paired_test(x, y, test="mcnemar", offset=0.1)
    with pytest.raises(ValueError, match="binary"):
        paired_test(x * 0.5, y, test="mcnemar")
    with pytest.raises(ValueError, match="binary"):
        effect_size(x + 2, y, "prop_diff")
    with pytest.raises(ValueError, match="zero_method"):
        paired_test(x, y, test="mcnemar", zero_method="pratt")
    with pytest.raises(ValueError, match="n_eff"):
        estimate(x, y, estimator="prop_diff", n_eff=10)
    with pytest.raises(ValueError, match="independent"):
        estimate(x, y, estimator="prop_diff", paired=False)


def test_odds_ratio_exact_interval():
    from mindmap.core.stats import estimate

    out = estimate(*_paired_binary(14, 6, 1, 5), estimator="odds_ratio")
    p_low, p_high = scipy_stats.binomtest(6, 7).proportion_ci(method="exact")
    assert out["estimate"] == 6.0
    assert (out["ci_low"], out["ci_high"]) == pytest.approx((p_low / (1 - p_low), p_high / (1 - p_high)))
    one_sided = estimate(*_paired_binary(14, 6, 0, 5), estimator="odds_ratio")
    assert one_sided["estimate"] == np.inf and one_sided["ci_high"] == np.inf and one_sided["ci_low"] > 0
    assert np.isnan(estimate(*_paired_binary(5, 0, 0, 5), estimator="odds_ratio", ci=False)["estimate"])


# -- Bootstrap ------------------------------------------------------------

def test_bootstrap_pvalue_counts_draws_beyond_the_null_with_plus_one():
    from mindmap.core.stats import bootstrap_pvalue
    draws = np.arange(1.0, 100.0)  # 99 draws, all above 0
    assert bootstrap_pvalue(draws, 0.0, "greater") == pytest.approx(1 / 100)
    assert bootstrap_pvalue(draws, 0.0, "two-sided") == pytest.approx(2 / 100)
    assert bootstrap_pvalue(draws, 0.0, "less") == pytest.approx(1.0)
    symmetric = np.linspace(-1, 1, 101)
    assert bootstrap_pvalue(symmetric, 0.0) == pytest.approx(1.0)
    assert np.isnan(bootstrap_pvalue([np.nan, np.nan]))


def test_bootstrap_pvalue_agrees_with_the_percentile_interval():
    """p < alpha exactly when the 1 - alpha percentile interval excludes the null."""
    from mindmap.core.stats import bootstrap_pvalue, percentile_interval
    rng = np.random.default_rng(0)
    for shift in np.linspace(0, 0.5, 11):
        draws = rng.normal(shift, 0.1, size=4000)
        low, high = percentile_interval(draws, 0.05)
        excluded = low > 0 or high < 0
        assert (bootstrap_pvalue(draws, 0.0) < 0.05) == excluded


def test_percentile_interval_handles_infinite_draws():
    from mindmap.core.stats import percentile_interval
    low, high = percentile_interval(np.r_[np.linspace(1, 5, 90), np.full(10, np.inf)], 0.05)
    assert np.isfinite(low) and high == np.inf


def test_bootstrap_draws_paired_keeps_pairs_together():
    """With x = y + 1 exactly, every paired resample has a mean difference of exactly 1."""
    from mindmap.core.stats import bootstrap_draws
    y = np.random.default_rng(0).normal(size=25)
    draws = bootstrap_draws(y + 1.0, y, "mean_diff", paired=True, n_boot=200, seed=0)
    np.testing.assert_allclose(draws, 1.0)


def test_estimate_bootstrap_is_seeded_and_close_to_the_t_interval():
    from mindmap.core.stats import estimate
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1, size=60)
    y = rng.normal(0, 1, size=60)
    boot = estimate(x, y, "mean_diff", ci_method="bootstrap", n_boot=4000, seed=7)
    again = estimate(x, y, "mean_diff", ci_method="bootstrap", n_boot=4000, seed=7)
    t = estimate(x, y, "mean_diff")
    assert boot["ci_method"] == "bootstrap_percentile" and boot["n_boot"] == 4000
    assert (boot["ci_low"], boot["ci_high"]) == (again["ci_low"], again["ci_high"])
    assert boot["estimate"] == t["estimate"]
    width = t["ci_high"] - t["ci_low"]
    assert abs(boot["ci_low"] - t["ci_low"]) < 0.15 * width
    assert abs(boot["ci_high"] - t["ci_high"]) < 0.15 * width


def test_estimate_bootstrap_covers_the_true_mean_at_about_the_nominal_rate():
    from mindmap.core.stats import estimate
    rng = np.random.default_rng(2)
    hits = 0
    for i in range(200):
        out = estimate(rng.normal(1.0, 1, size=40), estimator="mean", ci_method="bootstrap", n_boot=1000, seed=i)
        hits += out["ci_low"] <= 1.0 <= out["ci_high"]
    assert 0.88 <= hits / 200 <= 0.99


def test_estimate_bootstrap_gives_standardized_effect_sizes_an_interval():
    from mindmap.core.stats import estimate
    rng = np.random.default_rng(3)
    x, y = rng.normal(1, 1, size=30), rng.normal(0, 1, size=30)
    with pytest.raises(NotImplementedError, match="ci_method='bootstrap'"):
        estimate(x, y, "cohens_dz")
    out = estimate(x, y, "cohens_dz", ci_method="bootstrap", n_boot=1000, seed=0)
    assert out["ci_low"] < out["estimate"] < out["ci_high"]
    out = estimate(x, y, "cles", paired=False, ci_method="bootstrap", n_boot=1000, seed=0)
    assert 0 <= out["ci_low"] < out["estimate"] < out["ci_high"] <= 1


def test_estimate_bootstrap_reads_precomputed_draws():
    from mindmap.core.stats import estimate
    out = estimate([1.0, 2.0], [0.0, 0.0], "mean_diff", ci_method="bootstrap", draws=np.linspace(0, 3, 1001))
    assert out["estimate"] == 1.5
    assert out["ci_low"] == pytest.approx(0.075) and out["ci_high"] == pytest.approx(2.925)
    assert out["n_boot"] == 1001


def test_estimate_bootstrap_rejects_n_eff_and_unknown_methods():
    from mindmap.core.stats import estimate
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="serial correlation"):
        estimate(x, x[::-1], "mean_diff", ci_method="bootstrap", n_eff=5.0)
    with pytest.raises(ValueError, match="ci_method="):
        estimate(x, estimator="mean", ci_method="bca")
    with pytest.raises(ValueError, match="draws"):
        estimate(x, estimator="mean", draws=x)


# -- Bootstrap intervals: percentile / basic / BCa ---------------------------

from mindmap.core.stats import (
    bootstrap_interval, bootstrap_pvalue, bootstrap_bias, bias_correction, jackknife_values, jackknife_acceleration,
    reflect_draws, estimate, bootstrap_draws,
)


def _symmetric_draws(n=999, seed=0):
    """Draws symmetric around 0 (each value with its mirror image), plus a 0."""
    e = np.random.default_rng(seed).normal(size=n // 2)
    return np.concatenate([e, -e, [0.0]])


def test_basic_interval_equals_percentile_for_draws_symmetric_around_the_estimate():
    est = 1.3
    draws = est + _symmetric_draws()
    assert bootstrap_interval(draws, est, interval="basic") == pytest.approx(bootstrap_interval(draws, est))
    assert bootstrap_pvalue(draws, 0.5, interval="basic", estimate=est) == pytest.approx(bootstrap_pvalue(draws, 0.5))


def test_basic_and_percentile_intervals_move_opposite_ways_when_the_draws_are_shifted():
    est, b = 1.3, 0.4
    centered = bootstrap_interval(est + _symmetric_draws(), est)
    shifted = est + _symmetric_draws() - b
    assert bootstrap_interval(shifted, est) == pytest.approx((centered[0] - b, centered[1] - b))
    assert bootstrap_interval(shifted, est, interval="basic") == pytest.approx((centered[0] + b, centered[1] + b))


def test_basic_pvalue_agrees_with_the_basic_interval():
    """p < alpha exactly when the interval excludes the null -- up to the one draw at each interval end."""
    rng = np.random.default_rng(3)
    draws, est = rng.gamma(2.0, 1.0, size=1999), 1.0
    low, high = bootstrap_interval(draws, est, 0.05, "basic")
    reflected = np.sort(reflect_draws(draws, est))
    nulls = (reflected[1:] + reflected[:-1]) / 2
    disagree = [n for n in nulls if (bootstrap_pvalue(draws, n, interval="basic", estimate=est) < 0.05) != (n < low or n > high)]
    assert len(disagree) <= 2
    assert all(min(abs(n - low), abs(n - high)) < 2 * np.max(np.diff(reflected)) for n in disagree)


def test_bca_pvalue_reduces_to_percentile_without_bias_or_acceleration():
    draws = 0.7 + _symmetric_draws()
    for null in (0.0, 0.3, 1.5):
        assert bootstrap_pvalue(draws, null, interval="bca", estimate=0.7) == pytest.approx(bootstrap_pvalue(draws, null))
    assert bias_correction(draws, 0.7) == pytest.approx(0.0)


def test_bca_pvalue_agrees_with_the_bca_interval():
    rng = np.random.default_rng(4)
    x = rng.lognormal(size=30)
    draws = bootstrap_draws(x, estimator="mean", n_boot=4000, seed=0)
    est, a = x.mean(), jackknife_acceleration(*jackknife_values(x, estimator="mean"))
    low, high = bootstrap_interval(draws, est, 0.05, "bca", a)
    for null in np.linspace(low - 0.3, high + 0.3, 41):
        p = bootstrap_pvalue(draws, null, interval="bca", estimate=est, acceleration=a)
        if min(abs(null - low), abs(null - high)) > 0.01:
            assert (p < 0.05) == (null < low or null > high)


def test_jackknife_acceleration_matches_the_textbook_formula():
    rng = np.random.default_rng(1)
    x = rng.lognormal(size=25)
    jack = np.array([np.delete(x, i).mean() for i in range(x.size)])
    u = jack.mean() - jack
    expected = np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5)
    assert jackknife_acceleration(*jackknife_values(x, estimator="mean")) == pytest.approx(expected)


def test_bca_matches_scipy_on_resampled_data():
    rng = np.random.default_rng(1)
    x, y = rng.lognormal(size=25), rng.lognormal(size=25) * 0.8
    ours = estimate(x, y, "mean_diff", paired=True, ci_method="bootstrap", interval="bca", n_boot=50_000, seed=2)
    ref = scipy_stats.bootstrap((x - y,), np.mean, method="BCa", n_resamples=50_000, random_state=3)
    tol = 0.05 * ref.standard_error
    assert ours["ci_low"] == pytest.approx(ref.confidence_interval.low, abs=tol)
    assert ours["ci_high"] == pytest.approx(ref.confidence_interval.high, abs=tol)
    assert ours["ci_method"] == "bootstrap_bca"


def test_bca_warns_when_the_adjusted_levels_reach_the_extreme_draws():
    draws = np.concatenate([np.full(195, -1.0), np.linspace(0.1, 1, 5)])  # 97.5% of the draws below the estimate
    with pytest.warns(RuntimeWarning, match="extreme draw"):
        bootstrap_interval(draws, 0.0, interval="bca")


def test_bootstrap_bias_reports_bias_se_and_z0():
    draws = 2.0 + _symmetric_draws() - 0.5
    out = bootstrap_bias(draws, 2.0)
    assert out["boot_bias"] == pytest.approx(-0.5)
    assert out["boot_se"] == pytest.approx(np.std(draws, ddof=1))
    assert out["bias_se_ratio"] == pytest.approx(-0.5 / np.std(draws, ddof=1))
    assert out["z0"] > 0  # most draws below the estimate


def test_basic_and_bca_refuse_non_additive_estimators():
    x, y = np.r_[1, 0, 1, 1, 0, 1], np.r_[0, 0, 1, 0, 0, 1]
    with pytest.raises(ValueError, match="additive"):
        estimate(x, y, "rank_biserial", ci_method="bootstrap", interval="basic", n_boot=100, seed=0)
    with pytest.raises(ValueError, match="only apply to ci_method='bootstrap'"):
        estimate(x, y, "mean_diff", interval="basic")
