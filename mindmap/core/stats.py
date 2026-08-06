"""
Module Name: core/stats.py
Description: Pure numpy/scipy primitives for paired (repeated-measures)
statistical comparisons -- one paired test between two matched arrays,
multiple-comparison correction across a family of p-values, and the
cluster-forming threshold used by cluster-permutation tests.

Independent of pandas/mne -- so it can be called directly on plain arrays
without pulling in the DataFrame/MNE glue in `mindmap.analysis.stats`.

Author: Nino Herve
"""

import numpy as np
from scipy import stats


def effective_sample_size(x, max_lag=None):
    """
    Effective sample size of an autocorrelated series (Quenouille, 1952;
    Thiebaux & Zwiers, 1984; the standard fix used for e.g. testing the mean
    of a serially-correlated climate/time-series record):

        n_eff = n / (1 + 2 * sum_{k=1}^{K} rho_k)

    where rho_k is x's own lag-k autocorrelation. Answers "how many
    independent observations is x's n actual points actually worth" -- reduces
    to ~n when x is white noise (rho_k ~ 0 for all k), shrinks well below n
    when x is strongly serially correlated (e.g. adjacent timepoints in an
    evoked response, or nearby recording channels).

    Parameters
    ----------
    x: array-like of shape (n,)
        The series to estimate n_eff for -- typically a paired-difference
        series `a - b`, in its *actual* temporal/spatial order (this function
        has no way to check that; a shuffled `x` gives a meaningless result).

    max_lag: int, optional
        Largest lag to include in the sum. Defaults to the largest k before
        the sample ACF first drops below the white-noise 95% confidence band
        `1.96/sqrt(n)` (i.e. stops summing once the autocorrelation is no
        longer distinguishable from zero) -- capped at `n // 4` regardless,
        since ACF estimates at lags approaching n are themselves unreliable.

    Returns
    -------
    float, clipped to [1, n] -- this is a correction that can only shrink the
    effective n, never grow it past the actual number of points.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2:
        return float(max(n, 1))

    centered = x - x.mean()
    denom = np.sum(centered ** 2)
    if denom == 0:
        return float(n)  # constant series -- no autocorrelation structure to speak of

    cap = max(1, n // 4)
    if max_lag is None:
        max_lag = cap
    ci = 1.96 / np.sqrt(n)

    acf_sum = 0.0
    for k in range(1, min(max_lag, cap) + 1):
        rho_k = np.sum(centered[:-k] * centered[k:]) / denom
        if abs(rho_k) < ci:
            break
        acf_sum += rho_k

    n_eff = n / (1 + 2 * acf_sum)
    return float(np.clip(n_eff, 1.0, n))


def paired_test(x, y, test="wilcoxon"):
    """
    One paired comparison between two arrays of matched observations (e.g.
    a reference method's per-region values and one competitor's).

    Parameters
    ----------
    x, y: array-like of shape (n,)
        Matched pairs -- x[i] and y[i] are the same subject/region.

    test: str
        "wilcoxon" (default -- `scipy.stats.wilcoxon(x, y,
        zero_method="wilcox", alternative="two-sided")`; makes no
        normality assumption, appropriate for the typically small,
        non-normal per-region samples this library compares) | "ttest"
        (`scipy.stats.ttest_rel(x, y)`) | "ttest_ess" (a paired t-test using
        `effective_sample_size(x - y)` in place of the nominal n for the
        standard error and degrees of freedom -- for the case where x/y are
        indexed by something with real serial correlation, e.g. channels or
        timepoints from a single recording, where the nominal n overstates
        how much independent evidence is actually there; requires x/y to
        already be in their true temporal/spatial order, see
        `effective_sample_size`'s docstring).

    Returns
    -------
    dict with keys "statistic", "p_value", "effect_size", "effect_size_type"
    ("rank_biserial" for wilcoxon -- (n_pos - n_neg) / n over the nonzero
    differences, since scipy doesn't expose wilcoxon's internal rank sums
    separately; "cohens_dz" for ttest/ttest_ess -- mean(diff) / std(diff,
    ddof=1)), and "n" (pairs actually used -- wilcoxon's zero_method="wilcox"
    drops exact-zero differences, so this can be < len(x)). "ttest_ess" adds
    one more key, "n_eff" (the corrected sample size actually used for the
    test's standard error/degrees of freedom).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape (got {x.shape} vs {y.shape}).")

    diff = x - y

    if test == "wilcoxon":
        nonzero = diff[diff != 0]
        n = nonzero.shape[0]
        if n == 0:
            return {
                "statistic": np.nan, "p_value": np.nan, "effect_size": 0.0,
                "effect_size_type": "rank_biserial", "n": 0,
            }
        result = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        n_pos = int(np.sum(nonzero > 0))
        n_neg = int(np.sum(nonzero < 0))
        effect_size = (n_pos - n_neg) / n
        return {
            "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "effect_size": float(effect_size), "effect_size_type": "rank_biserial", "n": n,
        }

    if test == "ttest":
        n = diff.shape[0]
        sd = diff.std(ddof=1)
        se = sd / np.sqrt(n) if sd > 0 else 0.0
        t_stat = diff.mean() / se if se > 0 else 0.0
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        effect_size = diff.mean() / sd if sd > 0 else 0.0
        return {
            "statistic": float(t_stat), "p_value": p_value,
            "effect_size": float(effect_size), "effect_size_type": "cohens_dz", "n": n,
        }

    if test == "ttest_ess":
        n = diff.shape[0]
        n_eff = effective_sample_size(diff)
        sd = diff.std(ddof=1)
        se = sd / np.sqrt(n_eff) if sd > 0 else 0.0
        t_stat = diff.mean() / se if se > 0 else 0.0
        df = max(n_eff - 1, 1.0)
        p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
        effect_size = diff.mean() / sd if sd > 0 else 0.0
        return {
            "statistic": float(t_stat), "p_value": p_value,
            "effect_size": float(effect_size), "effect_size_type": "cohens_dz",
            "n": n, "n_eff": float(n_eff),
        }

    raise ValueError(f"Unknown test '{test}'. Choose 'wilcoxon', 'ttest', or 'ttest_ess'.")


def test_normality(x, test="shapiro"):
    """
    Tests whether a sample looks drawn from a normal distribution -- the
    null hypothesis is normality, so a small p-value is evidence *against*
    it.

    Parameters
    ----------
    x: array-like of shape (n,)
        NaNs are dropped before testing.

    test: str
        "shapiro" (default -- `scipy.stats.shapiro`; the most powerful
        general-purpose normality test for the small-to-moderate samples
        typical of one registered group) | "normaltest"
        (`scipy.stats.normaltest`, D'Agostino-Pearson K^2, based on
        skewness and kurtosis -- needs a larger n to be reliable, but is
        less sensitive than Shapiro-Wilk to the tiny, practically
        meaningless deviations that make "significant" too easy to reach
        at large n).

    Returns
    -------
    dict with keys "statistic", "p_value", "n" (non-NaN observations
    actually used). Below each test's minimum n (3 for shapiro, ~8 for
    normaltest), scipy returns "statistic"/"p_value" as NaN rather than
    raising -- passed through unchanged here.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.shape[0]

    if test == "shapiro":
        result = stats.shapiro(x)
    elif test == "normaltest":
        result = stats.normaltest(x)
    else:
        raise ValueError(f"Unknown test '{test}'. Choose 'shapiro' or 'normaltest'.")

    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n}


def repeated_measures_anova(wide, test="friedman"):
    """
    One omnibus test across three or more matched groups (e.g. every
    method's per-region values at once) -- answers "do these groups differ
    at all", complementary to `paired_test`'s pairwise "does A differ from
    B" (the standard next step after a significant result here is pairwise
    testing + `correct_pvalues` as post-hoc, exactly how
    `MetricSuite.stat.compare_groups` already works).

    Parameters
    ----------
    wide: array-like of shape (n_subjects, n_groups)
        Matched, complete-case observations -- `wide[i, j]` is subject i's
        value for group j. No missing values -- both tests need a fully
        balanced n x k matrix (drop/average incomplete rows before
        calling, same convention `PairwiseResult.compute` uses for pairs).

    test: str
        "friedman" (default -- `scipy.stats.friedmanchisquare`; makes no
        normality assumption, appropriate for the typically small,
        non-normal per-region samples this library compares) | "anova" (a
        one-way repeated-measures ANOVA, via the standard within-subjects
        sum-of-squares partition -- computed directly here rather than via
        `statsmodels.stats.anova.AnovaRM`, since a single within-factor
        design needs nothing AnovaRM provides beyond this; cross-checked
        against it numerically while developing this function).

    Returns
    -------
    dict with keys "statistic", "p_value", "effect_size", "effect_size_type"
    ("kendalls_w" -- Kendall's coefficient of concordance,
    `statistic / (n * (k - 1))` -- for "friedman"; "eta_sq_p" -- partial
    eta-squared, `ss_conditions / (ss_conditions + ss_error)` -- for
    "anova"), "n" (subjects), "k" (groups).
    """
    wide = np.asarray(wide, dtype=float)
    if wide.ndim != 2:
        raise ValueError(f"`wide` must be 2D (n_subjects, n_groups), got shape {wide.shape}.")
    n, k = wide.shape
    if k < 3:
        raise ValueError(f"An omnibus test needs at least 3 groups (got {k}) -- use paired_test for 2.")

    if test == "friedman":
        result = stats.friedmanchisquare(*[wide[:, j] for j in range(k)])
        denom = n * (k - 1)
        kendalls_w = float(result.statistic) / denom if denom > 0 else np.nan
        return {
            "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "effect_size": kendalls_w, "effect_size_type": "kendalls_w", "n": n, "k": k,
        }

    if test == "anova":
        grand_mean = wide.mean()
        condition_means = wide.mean(axis=0)
        subject_means = wide.mean(axis=1)

        ss_conditions = n * np.sum((condition_means - grand_mean) ** 2)
        ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
        ss_total = np.sum((wide - grand_mean) ** 2)
        ss_error = ss_total - ss_conditions - ss_subjects

        df_conditions = k - 1
        df_error = (k - 1) * (n - 1)
        ms_conditions = ss_conditions / df_conditions
        ms_error = ss_error / df_error if df_error > 0 else np.nan

        f_stat = ms_conditions / ms_error if ms_error and ms_error > 0 else np.nan
        p_value = float(stats.f.sf(f_stat, df_conditions, df_error)) if np.isfinite(f_stat) else np.nan
        denom = ss_conditions + ss_error
        eta_sq_p = ss_conditions / denom if denom > 0 else np.nan

        return {
            "statistic": float(f_stat) if np.isfinite(f_stat) else np.nan, "p_value": p_value,
            "effect_size": float(eta_sq_p), "effect_size_type": "eta_sq_p", "n": n, "k": k,
        }

    raise ValueError(f"Unknown test '{test}'. Choose 'friedman' or 'anova'.")


def independent_test(a, b, test="mannwhitneyu"):
    """
    One independent-samples comparison between two unrelated arrays -- e.g.
    a reference method's per-region values and a competitor's, when no
    subject-level correspondence between them is assumed (unlike
    `paired_test`, `a`/`b` don't need matching, and don't even need to be
    the same length).

    Parameters
    ----------
    a, b: array-like
        Independent samples -- lengths may differ.

    test: str
        "mannwhitneyu" (default -- `scipy.stats.mannwhitneyu`; makes no
        normality assumption, appropriate for the typically small,
        non-normal per-region samples this library compares) | "ttest_ind"
        (`scipy.stats.ttest_ind`).

    Returns
    -------
    dict with keys "statistic", "p_value", "effect_size", "effect_size_type"
    ("rank_biserial" for mannwhitneyu -- `1 - 2*U/(n_a*n_b)`; "cohens_d" for
    ttest_ind -- mean difference over the pooled standard deviation), "n_a",
    "n_b" (can differ, unlike `paired_test`'s single shared "n").
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.shape[0], b.shape[0]

    if test == "mannwhitneyu":
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        denom = n_a * n_b
        effect_size = 1 - (2 * result.statistic) / denom if denom > 0 else np.nan
        return {
            "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "effect_size": float(effect_size), "effect_size_type": "rank_biserial",
            "n_a": n_a, "n_b": n_b,
        }

    if test == "ttest_ind":
        result = stats.ttest_ind(a, b)
        df = n_a + n_b - 2
        if df > 0 and n_a > 1 and n_b > 1:
            pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / df)
        else:
            pooled_std = np.nan
        effect_size = (a.mean() - b.mean()) / pooled_std if pooled_std and pooled_std > 0 else np.nan
        return {
            "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "effect_size": float(effect_size), "effect_size_type": "cohens_d",
            "n_a": n_a, "n_b": n_b,
        }

    raise ValueError(f"Unknown test '{test}'. Choose 'mannwhitneyu' or 'ttest_ind'.")


def correct_pvalues(p_values, method="fdr_bh", alpha=0.05):
    """
    Multiple-comparison correction across one family of p-values (e.g. one
    per competitor method, all compared against the same reference).

    Parameters
    ----------
    p_values: array-like of shape (n,)

    method: str
        "fdr_bh" (default -- Benjamini-Hochberg, via
        `scipy.stats.false_discovery_control(method="bh")`) | "fdr_by"
        (Benjamini-Yekutieli, `method="by"`) | "bonferroni" (`p * n`,
        clipped to 1) | "none" (`p_values` returned unchanged).

    alpha: float

    Returns
    -------
    p_corrected: np.ndarray of shape (n,)
    significant: np.ndarray[bool] of shape (n,), `p_corrected <= alpha`
    """
    p_values = np.asarray(p_values, dtype=float)

    if method == "none":
        p_corrected = p_values.copy()
    elif method in ("fdr_bh", "fdr_by"):
        bh_method = "bh" if method == "fdr_bh" else "by"
        p_corrected = stats.false_discovery_control(p_values, method=bh_method)
    elif method == "bonferroni":
        p_corrected = np.clip(p_values * p_values.shape[0], 0, 1)
    else:
        raise ValueError(f"Unknown correction method '{method}'.")

    significant = p_corrected <= alpha
    return p_corrected, significant


def compare_to_reference(
    reference, competitors, competitor_names=None,
    test="wilcoxon", correction="fdr_bh", alpha=0.05,
):
    """
    Paired comparison of `reference` against every column of `competitors`,
    with the resulting p-values corrected together as one family.

    Parameters
    ----------
    reference: array-like of shape (n,)
        e.g. n regions.

    competitors: array-like of shape (n, m)
        Same n/ordering as `reference`; one column per competing method.

    competitor_names: list[str] of length m, optional
        Defaults to "competitor_0", "competitor_1", ...

    test, correction, alpha
        Forwarded to `paired_test` / `correct_pvalues`.

    Returns
    -------
    list[dict], length m: each `paired_test(...)`'s dict plus "name",
    "p_corrected", "significant".
    """
    reference = np.asarray(reference, dtype=float)
    competitors = np.asarray(competitors, dtype=float)
    if competitors.ndim != 2 or competitors.shape[0] != reference.shape[0]:
        raise ValueError(
            "`competitors` must have shape (n, m) matching `reference`'s length n "
            f"(got reference={reference.shape}, competitors={competitors.shape})."
        )

    m = competitors.shape[1]
    if competitor_names is None:
        competitor_names = [f"competitor_{j}" for j in range(m)]
    elif len(competitor_names) != m:
        raise ValueError(f"len(competitor_names)={len(competitor_names)} does not match m={m}.")

    results = [paired_test(reference, competitors[:, j], test=test) for j in range(m)]
    p_values = np.array([r["p_value"] for r in results])
    p_corrected, significant = correct_pvalues(p_values, method=correction, alpha=alpha)

    for r, name, pc, sig in zip(results, competitor_names, p_corrected, significant):
        r["name"] = name
        r["p_corrected"] = float(pc)
        r["significant"] = bool(sig)

    return results


def cluster_threshold(n_subjects, alpha=0.05, tail=0):
    """
    Cluster-forming t-threshold for a one-sample (paired-difference,
    sign-flip) cluster-permutation test with `n_subjects` subjects.

    This is exactly the threshold `mne.stats.permutation_cluster_1samp_test`
    computes internally when `threshold=None` and `alpha=0.05` (its
    auto-threshold hardcodes p=0.05), so it's only needed by
    `mindmap.analysis.stats.cluster_permutation_mne` when `cluster_alpha !=
    0.05`, to build an explicit threshold MNE's own automatic one can't
    produce.

    Parameters
    ----------
    n_subjects: int

    alpha: float
        Cluster-forming (not final cluster-level) significance threshold.

    tail: int
        0 (two-tailed), 1 (upper), or -1 (lower) -- same convention as
        `mne.stats.permutation_cluster_1samp_test`'s `tail` parameter.

    Returns
    -------
    float
    """
    p_thresh = alpha / 2 if tail == 0 else alpha
    t = float(stats.t.ppf(1 - p_thresh, df=n_subjects - 1))
    return t if tail >= 0 else -t
