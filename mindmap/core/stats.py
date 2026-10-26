"""
Module Name: core/stats.py
Description: Pure numpy/scipy primitives for paired (repeated-measures)
statistical comparisons -- one paired test between two matched arrays
(optionally against a non-zero offset), the effect sizes both tests and
estimates share (`effect_size`), point estimates with optional confidence
intervals (`estimate`), multiple-comparison correction
across a family of p-values, and the cluster-forming threshold used by
cluster-permutation tests.

Independent of pandas/mne -- so it can be called directly on plain arrays
without pulling in the DataFrame/MNE glue in `mindmap.analysis.stats`.

Author: Nino Herve
"""

import warnings

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


_ALTERNATIVES = ("two-sided", "greater", "less")


def resolve_alternative(offset=0.0, alternative=None):
    """
    The `alternative` a test actually runs with: an explicit value is
    validated and returned as-is; `None` means `"two-sided"` when `offset`
    is 0 and `"greater"` otherwise -- a non-zero offset only makes sense
    as a one-sided "is the difference beyond the offset" question.
    """
    if alternative is None:
        return "two-sided" if offset == 0 else "greater"
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"Unknown alternative {alternative!r}. Choose from {list(_ALTERNATIVES)}.")
    return alternative


def _t_pvalue(t_stat, df, alternative):
    if alternative == "greater":
        return float(stats.t.sf(t_stat, df=df))
    if alternative == "less":
        return float(stats.t.cdf(t_stat, df=df))
    return float(2 * stats.t.sf(abs(t_stat), df=df))


def _hedges_correction(df):
    """Small-sample bias correction J turning Cohen's d into Hedges' g (Hedges, 1981)."""
    return 1 - 3 / (4 * df - 1) if df > 0.25 else np.nan


#: Effect sizes a paired comparison (on the differences `x - y`) can report, and those an
#: independent one (between samples `a` and `b`) can. "mean_diff"/"hodges_lehmann" are
#: unstandardized (in the metric's units), the rest standardized.
PAIRED_EFFECT_SIZES = ("mean_diff", "hodges_lehmann", "rank_biserial", "cohens_dz", "hedges_gz", "prop_diff", "odds_ratio")
INDEPENDENT_EFFECT_SIZES = ("mean_diff", "hodges_lehmann", "rank_biserial", "cles", "cohens_d", "hedges_g")
#: Paired effect sizes of a binary (0/1) outcome -- McNemar's: "prop_diff" (unstandardized,
#: in proportion units) and "odds_ratio" (the conditional, discordant-pair odds ratio).
BINARY_EFFECT_SIZES = ("prop_diff", "odds_ratio")

#: Which effect sizes each test may report -- the first is its default.
TEST_EFFECT_SIZES = {
    "wilcoxon": ("rank_biserial", "hodges_lehmann"),
    "mcnemar": ("prop_diff", "odds_ratio"),
    "ttest": ("cohens_dz", "hedges_gz", "mean_diff"),
    "ttest_ess": ("cohens_dz", "hedges_gz", "mean_diff"),
    "mannwhitneyu": ("rank_biserial", "cles", "hodges_lehmann"),
    "ttest_ind": ("cohens_d", "hedges_g", "mean_diff"),
}


def resolve_effect_sizes(test, effect_sizes=None):
    """
    The effect sizes a test reports: `None` means its default (the first of
    `TEST_EFFECT_SIZES[test]`), a single name is wrapped in a tuple, and
    every name is checked against what the test allows.
    """
    allowed = TEST_EFFECT_SIZES[test]
    if effect_sizes is None:
        return (allowed[0],)
    if isinstance(effect_sizes, str):
        effect_sizes = (effect_sizes,)
    effect_sizes = tuple(dict.fromkeys(effect_sizes))
    if not effect_sizes:
        raise ValueError("`effect_sizes` must name at least one effect size.")
    unknown = [e for e in effect_sizes if e not in allowed]
    if unknown:
        raise ValueError(f"Effect size(s) {unknown} aren't available for test={test!r}. Choose from {list(allowed)}.")
    return effect_sizes


def _walsh_averages(d):
    """Every Walsh average `(d_i + d_j) / 2`, i <= j, sorted -- the one-sample Hodges-Lehmann estimate is their median."""
    n = d.shape[0]
    if n == 0:
        return np.nan
    total = n * (n + 1) // 2
    if total > _MAX_PAIRWISE_TERMS:
        raise ValueError(
            f"Hodges-Lehmann on n={n} needs {total:,} Walsh averages (cap {_MAX_PAIRWISE_TERMS:,}) -- "
            "subsample or collapse the data first (e.g. via `unit_col`)."
        )
    return np.sort(np.concatenate([(d[i] + d[i:]) / 2 for i in range(n)]))


def _pairwise_differences(a, b):
    """Every `a_i - b_j`, sorted -- what the two-sample Hodges-Lehmann estimate is the median of."""
    total = a.shape[0] * b.shape[0]
    if total > _MAX_PAIRWISE_TERMS:
        raise ValueError(
            f"Two-sample Hodges-Lehmann needs {total:,} pairwise differences (cap {_MAX_PAIRWISE_TERMS:,}) -- "
            "subsample or collapse the data first (e.g. via `unit_col`)."
        )
    return np.sort(np.subtract.outer(a, b).ravel())


#: How the Wilcoxon signed-rank test (and the matched-pairs rank-biserial) treats zero differences.
ZERO_METHODS = ("wilcox", "pratt", "zsplit")


def _check_zero_method(zero_method):
    if zero_method not in ZERO_METHODS:
        raise ValueError(f"zero_method={zero_method!r} isn't supported. Choose from {list(ZERO_METHODS)}.")


def _rank_biserial(d, zero_method="wilcox"):
    """
    Matched-pairs rank-biserial (Kerby 2014), `(R+ - R-) / (sum of ranks)`,
    ranked the way `zero_method` ranks for the Wilcoxon test: "wilcox"
    drops the zeros before ranking; "pratt" and "zsplit" rank them with
    the rest, and their ranks stay in the denominator (Pratt leaves them
    out of R+ and R-, zsplit splits them evenly -- the same value). So with
    "pratt"/"zsplit", tied pairs pull r toward 0.
    """
    if zero_method == "wilcox":
        d = d[d != 0]
    if d.shape[0] == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def _check_binary(*arrays):
    """Raises unless every non-NaN value is 0 or 1 -- what McNemar and its effect sizes need."""
    for v in arrays:
        v = np.asarray(v, dtype=float)
        bad = np.unique(v[~np.isnan(v) & (v != 0) & (v != 1)])
        if bad.size:
            raise ValueError(
                f"McNemar and its effect sizes need a binary outcome coded 0/1 (e.g. register `inside = distance == 0` "
                f"as its own metric) -- got values such as {bad[:3].tolist()}. Averaging repeated 0/1 rows "
                "(`unit_col`) gives fractions, which aren't binary either."
            )


def _paired_effect_size(d, name, n_eff=None, zero_method="wilcox"):
    n = d.shape[0]
    if name in ("mean_diff", "prop_diff"):
        return float(d.mean()) if n else np.nan
    if name == "odds_ratio":
        b, c = int(np.sum(d > 0)), int(np.sum(d < 0))
        return b / c if c else (np.inf if b else np.nan)
    if name == "hodges_lehmann":
        return float(np.median(_walsh_averages(d))) if n else np.nan
    if name == "rank_biserial":
        return _rank_biserial(d, zero_method)
    if name in ("cohens_dz", "hedges_gz"):
        if n < 2:
            return np.nan
        sd = d.std(ddof=1)
        dz = float(d.mean() / sd) if sd > 0 else 0.0
        if name == "cohens_dz":
            return dz
        df = max(n_eff - 1, 1.0) if n_eff is not None else n - 1
        return float(dz * _hedges_correction(df))
    raise ValueError(f"Unknown paired effect size {name!r}. Choose from {list(PAIRED_EFFECT_SIZES)}.")


def _independent_effect_size(a, b, name):
    n_a, n_b = a.shape[0], b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.nan
    if name == "mean_diff":
        return float(a.mean() - b.mean())
    if name == "hodges_lehmann":
        return float(np.median(_pairwise_differences(a, b)))
    if name in ("rank_biserial", "cles"):
        # Mann-Whitney's U for `a` from the joint ranks (ties averaged) -- what scipy's mannwhitneyu reports,
        # without its p-value, which the bootstrap would recompute thousands of times.
        u = stats.rankdata(np.concatenate([a, b]))[:n_a].sum() - n_a * (n_a + 1) / 2
        cles = float(u / (n_a * n_b))
        return cles if name == "cles" else 2 * cles - 1
    if name in ("cohens_d", "hedges_g"):
        df = n_a + n_b - 2
        if n_a < 2 or n_b < 2:
            return np.nan
        pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / df)
        if pooled_std == 0:
            return np.nan
        d = float((a.mean() - b.mean()) / pooled_std)
        return d if name == "cohens_d" else float(d * _hedges_correction(df))
    raise ValueError(f"Unknown independent effect size {name!r}. Choose from {list(INDEPENDENT_EFFECT_SIZES)}.")


def effect_size(x, y, name, paired=True, n_eff=None, zero_method="wilcox"):
    """
    One effect size of `x` relative to `y`, positive when `x` is higher --
    the single definition both the tests and `estimate` use.

    Parameters
    ----------
    x, y: array-like
        Matched (same length) when `paired=True`; any lengths otherwise.

    name: str
        Paired (on `d = x - y`): "mean_diff" (mean of d), "hodges_lehmann"
        (median of d's Walsh averages), "rank_biserial" (matched-pairs
        rank-biserial correlation, Kerby 2014: `(R+ - R-) / (sum of
        ranks)` over the ranks of |d|, ties averaged, zeros handled as
        `zero_method` says), "prop_diff" and "odds_ratio" (binary 0/1
        outcome only: the difference in the proportion of 1s, `(b - c) /
        n`, and the conditional odds ratio `b / c`, where b counts the
        pairs with x = 1, y = 0 and c those with x = 0, y = 1 -- `inf`
        when c = 0, NaN without discordant pairs), "cohens_dz"
        (`mean(d) / std(d, ddof=1)`), "hedges_gz" (dz times the small-sample
        correction J, Hedges 1981). Independent: "mean_diff" (difference of
        means), "hodges_lehmann" (median of every `x_i - y_j`),
        "rank_biserial" (`2 * cles - 1`), "cles" (`P(x > y) + P(x = y) / 2`,
        i.e. `U / (n_x * n_y)`), "cohens_d" (mean difference over the pooled
        standard deviation), "hedges_g" (d times J).

    paired: bool

    n_eff: float, optional
        Paired only: an effective sample size (see `effective_sample_size`)
        used for hedges_gz's degrees of freedom instead of n - 1.

    zero_method: str, optional
        Paired "rank_biserial" only: "wilcox" (default -- zero differences
        dropped), "pratt" or "zsplit" (zeros ranked, their ranks kept in the
        denominator, so ties pull r toward 0) -- match the Wilcoxon test's.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _check_zero_method(zero_method)
    if paired:
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape when paired (got {x.shape} vs {y.shape}).")
        if name in BINARY_EFFECT_SIZES:
            _check_binary(x, y)
        return _paired_effect_size(x - y, name, n_eff, zero_method)
    return _independent_effect_size(x, y, name)


def paired_test(x, y, test="wilcoxon", offset=0.0, alternative=None, effect_sizes=None, zero_method=None):
    """
    One paired comparison between two arrays of matched observations (e.g.
    a reference method's per-region values and one competitor's).

    Parameters
    ----------
    x, y: array-like of shape (n,)
        Matched pairs -- x[i] and y[i] are the same subject/region.

    test: str
        "wilcoxon" (default -- `scipy.stats.wilcoxon` on the shifted
        differences, zeros handled per `zero_method`; makes no normality assumption,
        appropriate for the typically small, non-normal per-region samples
        this library compares) | "ttest" (paired t-test) | "ttest_ess" (a
        paired t-test using `effective_sample_size(x - y)` in place of the
        nominal n for the standard error and degrees of freedom -- for the
        case where x/y are indexed by something with real serial
        correlation, e.g. channels or timepoints from a single recording,
        where the nominal n overstates how much independent evidence is
        actually there; requires x/y to already be in their true
        temporal/spatial order, see `effective_sample_size`'s docstring)
        | "mcnemar" (exact McNemar test for a binary 0/1 outcome, e.g.
        "peak inside the resection": only the discordant pairs carry
        information -- b with x = 1, y = 0 and c with x = 0, y = 1 -- and
        under the null each is equally likely, so b ~ Binomial(b + c, 1/2);
        the p-value is that exact binomial test, 1 when there is no
        discordant pair. No offset.)

    offset: float
        The difference tested against (default 0): the null hypothesis is
        `x - y = offset` (location of the differences), so e.g.
        `offset=0.1, alternative="greater"` asks "is x better than y by
        more than 0.1?". Only the test runs on the shifted differences
        `x - y - offset`; effect sizes always describe the raw `x - y`.

    alternative: str, optional
        "two-sided" | "greater" (`x - y > offset`) | "less"
        (`x - y < offset`). Defaults to "two-sided" when `offset` is 0 and
        "greater" otherwise -- see `resolve_alternative`.

    effect_sizes: str | tuple[str, ...], optional
        Which effect sizes to report -- any of `TEST_EFFECT_SIZES[test]`
        (see `effect_size` for their definitions). Defaults to the test's
        first: "rank_biserial" for wilcoxon, "cohens_dz" for
        ttest/ttest_ess.

    zero_method: str, optional
        Wilcoxon only -- what to do with pairs whose (shifted) difference
        is exactly 0, a tie between the two methods: "wilcox" (the default
        -- drop them, so the test asks "among the pairs that differ, which
        side wins?"), "pratt" (rank them with the others, lowest, then
        leave their ranks out of both sums -- they stay in the sample and
        in the null distribution, so ties weaken the evidence), or
        "zsplit" (rank them, and split their ranks evenly between both
        sums). The rank-biserial uses the same convention (see
        `effect_size`). Must be None for the t-tests.

    Returns
    -------
    dict with keys "statistic", "p_value", "effect_sizes" (`{name: value}`,
    in the order requested), "effect_size"/"effect_size_type" (the first
    one), "offset", "alternative" (resolved), "n" (pairs actually used --
    with zero_method="wilcox", differences exactly at the offset are
    dropped, so this can be < len(x)), "n_zero" (pairs whose shifted
    difference is exactly 0), and, for wilcoxon, "zero_method".
    "ttest_ess" adds "n_eff" (the corrected sample size used for the
    test's standard error/degrees of freedom, and for hedges_gz).
    "mcnemar" adds "n_x_only" (b) and "n_y_only" (c), the discordant
    counts; its "statistic" is b, and "n_zero" counts the concordant pairs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape (got {x.shape} vs {y.shape}).")
    if test not in ("wilcoxon", "mcnemar", "ttest", "ttest_ess"):
        raise ValueError(f"Unknown test '{test}'. Choose 'wilcoxon', 'mcnemar', 'ttest', or 'ttest_ess'.")
    if test == "mcnemar":
        _check_binary(x, y)
        if float(offset) != 0:
            raise ValueError("McNemar tests equal proportions -- it takes no `offset`.")
    if test == "wilcoxon":
        zero_method = "wilcox" if zero_method is None else zero_method
        _check_zero_method(zero_method)
    elif zero_method is not None:
        raise ValueError(f"`zero_method` only applies to test='wilcoxon', not {test!r}.")
    offset = float(offset)
    alternative = resolve_alternative(offset, alternative)
    names = resolve_effect_sizes(test, effect_sizes)

    diff = x - y
    shifted = diff - offset
    n_zero = int(np.sum(shifted == 0))
    out = {"offset": offset, "alternative": alternative, "n_zero": n_zero}
    n_eff = None

    if test == "wilcoxon":
        out["zero_method"] = zero_method
        n = diff.shape[0] - n_zero if zero_method == "wilcox" else diff.shape[0]
        if n_zero == diff.shape[0]:
            out.update(statistic=np.nan, p_value=np.nan, n=n)
        else:
            result = stats.wilcoxon(shifted, zero_method=zero_method, alternative=alternative)
            out.update(statistic=float(result.statistic), p_value=float(result.pvalue), n=n)
    elif test == "mcnemar":
        b, c = int(np.sum(diff > 0)), int(np.sum(diff < 0))
        p = float(stats.binomtest(b, b + c, 0.5, alternative=alternative).pvalue) if b + c else 1.0
        out.update(statistic=float(b), p_value=p, n=diff.shape[0], n_x_only=b, n_y_only=c)
    else:
        n = diff.shape[0]
        n_used = effective_sample_size(shifted) if test == "ttest_ess" else n
        sd = shifted.std(ddof=1)
        se = sd / np.sqrt(n_used) if sd > 0 else 0.0
        t_stat = shifted.mean() / se if se > 0 else 0.0
        df = max(n_used - 1, 1.0) if test == "ttest_ess" else n - 1
        out.update(statistic=float(t_stat), p_value=_t_pvalue(t_stat, df, alternative), n=n)
        if test == "ttest_ess":
            n_eff = float(n_used)
            out["n_eff"] = n_eff

    values = {name: _paired_effect_size(diff, name, n_eff, zero_method or "wilcox") for name in names}
    out.update(effect_sizes=values, effect_size=values[names[0]], effect_size_type=names[0])
    return out


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


def independent_test(a, b, test="mannwhitneyu", offset=0.0, alternative=None, effect_sizes=None):
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

    offset: float
        The location shift tested against (default 0): the test compares
        `a - offset` with `b`, i.e. the null is "a is shifted from b by
        exactly `offset`". Effect sizes always describe the raw `a` vs `b`.

    alternative: str, optional
        "two-sided" | "greater" (a exceeds b by more than `offset`) |
        "less". Defaults as in `paired_test` (see `resolve_alternative`).

    effect_sizes: str | tuple[str, ...], optional
        Which effect sizes to report -- any of `TEST_EFFECT_SIZES[test]`
        (see `effect_size`). Defaults to "rank_biserial" for mannwhitneyu,
        "cohens_d" for ttest_ind.

    Returns
    -------
    dict with keys "statistic", "p_value", "effect_sizes" (`{name: value}`,
    in the order requested), "effect_size"/"effect_size_type" (the first
    one), "offset", "alternative", "n_a", "n_b" (can differ, unlike
    `paired_test`'s single shared "n").
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if test not in ("mannwhitneyu", "ttest_ind"):
        raise ValueError(f"Unknown test '{test}'. Choose 'mannwhitneyu' or 'ttest_ind'.")
    offset = float(offset)
    alternative = resolve_alternative(offset, alternative)
    names = resolve_effect_sizes(test, effect_sizes)

    if test == "mannwhitneyu":
        result = stats.mannwhitneyu(a - offset, b, alternative=alternative)
    else:
        result = stats.ttest_ind(a - offset, b, alternative=alternative)

    values = {name: _independent_effect_size(a, b, name) for name in names}
    return {
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "effect_sizes": values, "effect_size": values[names[0]], "effect_size_type": names[0],
        "offset": offset, "alternative": alternative, "n_a": a.shape[0], "n_b": b.shape[0],
    }


#: Largest number of pairwise averages/differences `estimate("hodges_lehmann")` will
#: materialize (~400 MB as float64) -- above it, raise rather than exhaust memory.
_MAX_PAIRWISE_TERMS = 50_000_000

#: Largest n for which the signed-rank null distribution is computed exactly (O(n^3)
#: but trivially cheap at this size); above it, the normal approximation is used.
_MAX_EXACT_SIGNED_RANK = 200


def _signed_rank_cdf(n):
    """Exact null CDF of the Wilcoxon signed-rank statistic T+ for n untied, nonzero values, indexed by T+ = 0..n(n+1)/2."""
    total = n * (n + 1) // 2
    pmf = np.zeros(total + 1)
    pmf[0] = 1.0
    for i in range(1, n + 1):
        shifted = np.zeros_like(pmf)
        shifted[i:] = pmf[:-i]
        pmf = 0.5 * (pmf + shifted)
    return np.cumsum(pmf)


def _normal_rank_depth(mean, var, alpha):
    """Normal-approximation counterpart of the exact depth: largest k with P(T <= k - 1) <= alpha/2 (continuity-corrected)."""
    z = stats.norm.ppf(1 - alpha / 2)
    return int(np.floor(mean - z * np.sqrt(var) + 0.5))


def _order_stat_ci(sorted_values, k):
    """`(sorted_values[k-1], sorted_values[N-k])` -- the k-th smallest/largest -- or NaNs when the sample is too small for the requested level (k < 1)."""
    if k < 1:
        return np.nan, np.nan
    return float(sorted_values[k - 1]), float(sorted_values[sorted_values.shape[0] - k])


def _one_sample_estimate(d, estimator, alpha, n_eff=None):
    n = d.shape[0]
    out = {"n": n}
    if n == 0:
        return {**out, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_method": None}

    if estimator == "mean":
        m = float(d.mean())
        if n < 2:
            return {**out, "estimate": m, "ci_low": np.nan, "ci_high": np.nan, "ci_method": "t"}
        n_used = n if n_eff is None else n_eff
        df = n - 1 if n_eff is None else max(n_eff - 1, 1.0)
        half = stats.t.ppf(1 - alpha / 2, df=df) * d.std(ddof=1) / np.sqrt(n_used)
        ci_method = "t" if n_eff is None else "t_ess"
        return {**out, "estimate": m, "ci_low": float(m - half), "ci_high": float(m + half), "ci_method": ci_method}

    if estimator == "median":
        k = int(stats.binom.ppf(alpha / 2, n, 0.5))
        low, high = _order_stat_ci(np.sort(d), k)
        return {**out, "estimate": float(np.median(d)), "ci_low": low, "ci_high": high, "ci_method": "binomial_exact"}

    if estimator == "hodges_lehmann":
        walsh = _walsh_averages(d)
        total = walsh.shape[0]
        if n <= _MAX_EXACT_SIGNED_RANK:
            k = int(np.searchsorted(_signed_rank_cdf(n), alpha / 2))
            ci_method = "signed_rank_exact"
        else:
            k = _normal_rank_depth(total / 2, n * (n + 1) * (2 * n + 1) / 24, alpha)
            ci_method = "signed_rank_normal"
        low, high = _order_stat_ci(walsh, k)
        return {**out, "estimate": float(np.median(walsh)), "ci_low": low, "ci_high": high, "ci_method": ci_method}

    raise ValueError(f"Unknown estimator '{estimator}'.")


def _wilson_interval(k, n, alpha):
    """Wilson score interval for a proportion k / n (Wilson, 1927) -- stays in [0, 1] and keeps close to nominal coverage at small n."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def _binary_paired_estimate(x, y, estimator, alpha):
    """
    "prop_diff" with Newcombe's interval, or "odds_ratio" with its exact
    interval, for matched 0/1 outcomes (complete pairs only).

    Newcombe (1998), method 10: a Wilson interval for each of the two
    proportions, combined with a correction for their correlation, `phi`
    (with Newcombe's adjustment: when `ad > bc`, the numerator is
    `max(ad - bc - n/2, 0)`; 0 when a margin is empty). The conditional
    odds ratio `b / c` gets the exact (Clopper-Pearson) interval of the
    share of discordant pairs that favour x, `b / (b + c)`, mapped
    through `p / (1 - p)` (Breslow & Day).
    """
    n = x.shape[0]
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = n - a - b - c
    out = {"n": n}
    if estimator == "prop_diff":
        if n == 0:
            return {**out, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_method": "newcombe"}
        p1, p2 = (a + b) / n, (a + c) / n
        l1, u1 = _wilson_interval(a + b, n, alpha)
        l2, u2 = _wilson_interval(a + c, n, alpha)
        margins = (a + b) * (c + d) * (a + c) * (b + d)
        if margins == 0:
            phi = 0.0
        else:
            num = a * d - b * c
            phi = (max(num - n / 2, 0.0) if num > 0 else num) / np.sqrt(margins)
        diff = p1 - p2
        low = diff - np.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
        high = diff + np.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
        return {**out, "estimate": float(diff), "ci_low": float(max(low, -1.0)), "ci_high": float(min(high, 1.0)), "ci_method": "newcombe"}

    if estimator == "odds_ratio":
        point = b / c if c else (np.inf if b else np.nan)
        if b + c == 0:
            return {**out, "estimate": point, "ci_low": np.nan, "ci_high": np.nan, "ci_method": "binomial_exact"}
        p_low, p_high = stats.binomtest(b, b + c).proportion_ci(confidence_level=1 - alpha, method="exact")
        to_odds = lambda p: p / (1 - p) if p < 1 else np.inf
        return {**out, "estimate": point, "ci_low": float(to_odds(p_low)), "ci_high": float(to_odds(p_high)), "ci_method": "binomial_exact"}

    raise ValueError(f"Unknown estimator '{estimator}'.")


def _two_sample_estimate(a, b, estimator, alpha):
    n_a, n_b = a.shape[0], b.shape[0]
    out = {"n_a": n_a, "n_b": n_b}
    if n_a == 0 or n_b == 0:
        return {**out, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_method": None}

    if estimator == "mean":
        m = float(a.mean() - b.mean())
        if n_a < 2 or n_b < 2:
            return {**out, "estimate": m, "ci_low": np.nan, "ci_high": np.nan, "ci_method": "welch"}
        va, vb = a.var(ddof=1) / n_a, b.var(ddof=1) / n_b
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1)) if se > 0 else n_a + n_b - 2
        half = stats.t.ppf(1 - alpha / 2, df=df) * se
        return {**out, "estimate": m, "ci_low": float(m - half), "ci_high": float(m + half), "ci_method": "welch"}

    if estimator == "hodges_lehmann":
        diffs = _pairwise_differences(a, b)
        total = diffs.shape[0]
        k = _normal_rank_depth(total / 2, n_a * n_b * (n_a + n_b + 1) / 12, alpha)
        low, high = _order_stat_ci(diffs, k)
        return {**out, "estimate": float(np.median(diffs)), "ci_low": low, "ci_high": high, "ci_method": "mann_whitney_normal"}

    raise ValueError(f"Unknown estimator '{estimator}'.")


#: Estimators describing one sample on its own, vs. a difference between two.
GROUP_ESTIMATORS = ("mean", "median", "proportion")
DIFFERENCE_ESTIMATORS = (
    "mean_diff", "hodges_lehmann", "rank_biserial", "cles", "cohens_dz", "hedges_gz", "cohens_d", "hedges_g",
    "prop_diff", "odds_ratio",
)
#: The estimators `estimate(..., ci=True)` has an analytic interval for; the rest raise
#: NotImplementedError unless `ci_method="bootstrap"`.
ESTIMATORS_WITH_CI = ("mean", "median", "proportion", "mean_diff", "hodges_lehmann", "prop_diff", "odds_ratio")
#: Where each difference estimator means "no difference" (0 for the rest).
NULL_EFFECT_VALUES = {"cles": 0.5, "odds_ratio": 1.0}

#: How `estimate` builds its interval: the estimator's own analytic interval, or a percentile bootstrap.
CI_METHODS = ("analytic", "bootstrap")
DEFAULT_N_BOOT = 10_000


def _sample_value(v, estimator, one_sample, zero_method="wilcox"):
    """`estimator` on one clean (NaN-free) sample `v` -- the values themselves (`one_sample`) or paired differences."""
    if one_sample:
        if not v.shape[0]:
            return np.nan
        return float(np.median(v) if estimator == "median" else np.mean(v))
    return _paired_effect_size(v, estimator, None, zero_method)


def bootstrap_draws(x, y=None, estimator="mean_diff", paired=True, n_boot=DEFAULT_N_BOOT, seed=None, zero_method="wilcox"):
    """
    The bootstrap distribution of `estimator`: its value on `n_boot`
    resamples of the data, units drawn with replacement -- one sample
    (`y=None`): the values; paired: whole pairs, so each `x_i` stays with
    its `y_i`; independent: each sample on its own, keeping its size.
    NaNs are dropped first (pairwise when paired), as in `estimate`.

    Parameters
    ----------
    x, y: array-like
        As in `estimate`.

    estimator: str
        Any estimator `estimate` takes.

    paired: bool
        Only used when `y` is given.

    n_boot: int
        Number of resamples (default 10,000).

    seed: int | np.random.Generator, optional
        Seeds the resampling; a Generator is used as-is (so successive
        calls continue its stream).

    zero_method: str, optional
        Paired "rank_biserial" only -- see `effect_size`.

    Returns
    -------
    np.ndarray of shape (n_boot,)
        NaN where a resample has no value (e.g. an odds ratio without
        discordant pairs).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n_boot = int(n_boot)
    draws = np.full(n_boot, np.nan)
    if y is None or paired:
        v = x if y is None else x - np.asarray(y, dtype=float)
        v = v[~np.isnan(v)]
        n = v.shape[0]
        if not n:
            return draws
        if estimator in _VECTORIZED_ONE_SAMPLE if y is None else estimator in _VECTORIZED_PAIRED:
            for start in range(0, n_boot, _BOOT_CHUNK):
                k = min(_BOOT_CHUNK, n_boot - start)
                draws[start:start + k] = _sample_values(v[rng.integers(0, n, (k, n))], estimator, y is None, zero_method)
        else:
            for i in range(n_boot):
                draws[i] = _sample_value(v[rng.integers(0, n, n)], estimator, y is None, zero_method)
        return draws
    y = np.asarray(y, dtype=float)
    a, b = x[~np.isnan(x)], y[~np.isnan(y)]
    n_a, n_b = a.shape[0], b.shape[0]
    if not (n_a and n_b):
        return draws
    if estimator in _VECTORIZED_INDEPENDENT:
        for start in range(0, n_boot, _BOOT_CHUNK):
            k = min(_BOOT_CHUNK, n_boot - start)
            draws[start:start + k] = _independent_values(
                a[rng.integers(0, n_a, (k, n_a))], b[rng.integers(0, n_b, (k, n_b))], estimator,
            )
    else:
        for i in range(n_boot):
            draws[i] = _independent_effect_size(a[rng.integers(0, n_a, n_a)], b[rng.integers(0, n_b, n_b)], estimator)
    return draws


#: Resamples evaluated together, one row each -- bounds the memory of the vectorized bootstrap.
_BOOT_CHUNK = 1000
#: Estimators `_sample_values`/`_independent_values` compute on a whole block of resamples at once;
#: the rest (Hodges-Lehmann, the odds ratio, a paired median) loop over the resamples.
_VECTORIZED_ONE_SAMPLE = ("mean", "median", "proportion")
_VECTORIZED_PAIRED = ("mean_diff", "prop_diff", "cohens_dz", "hedges_gz", "rank_biserial")
_VECTORIZED_INDEPENDENT = ("mean_diff", "rank_biserial", "cles", "cohens_d", "hedges_g")


def _sample_values(V, estimator, one_sample, zero_method="wilcox"):
    """`_sample_value` on every row of `V` (one resample per row) at once -- same definitions, see `effect_size`."""
    if one_sample:
        return np.median(V, axis=1) if estimator == "median" else V.mean(axis=1)
    if estimator in ("mean_diff", "prop_diff"):
        return V.mean(axis=1)
    if estimator in ("cohens_dz", "hedges_gz"):
        n = V.shape[1]
        if n < 2:
            return np.full(V.shape[0], np.nan)
        sd = V.std(axis=1, ddof=1)
        dz = np.divide(V.mean(axis=1), sd, out=np.zeros(V.shape[0]), where=sd > 0)
        return dz if estimator == "cohens_dz" else dz * _hedges_correction(n - 1)
    # rank_biserial: ranks of |d| per row. Zeros are the smallest |d|, so dropping them ("wilcox") lowers
    # every other rank by the number of zeros.
    ranks = stats.rankdata(np.abs(V), axis=1)
    if zero_method == "wilcox":
        ranks = np.where(V != 0, ranks - (V == 0).sum(axis=1, keepdims=True), 0.0)
    signed = np.where(V > 0, ranks, 0.0).sum(axis=1) - np.where(V < 0, ranks, 0.0).sum(axis=1)
    total = ranks.sum(axis=1)
    return np.divide(signed, total, out=np.zeros(V.shape[0]), where=total > 0)


def _independent_values(A, B, estimator):
    """`_independent_effect_size` on every row pair of `A`/`B` (one resample per row) at once."""
    n_a, n_b = A.shape[1], B.shape[1]
    if estimator == "mean_diff":
        return A.mean(axis=1) - B.mean(axis=1)
    if estimator in ("rank_biserial", "cles"):
        u = stats.rankdata(np.concatenate([A, B], axis=1), axis=1)[:, :n_a].sum(axis=1) - n_a * (n_a + 1) / 2
        cles = u / (n_a * n_b)
        return cles if estimator == "cles" else 2 * cles - 1
    if n_a < 2 or n_b < 2:
        return np.full(A.shape[0], np.nan)
    df = n_a + n_b - 2
    pooled = np.sqrt(((n_a - 1) * A.var(axis=1, ddof=1) + (n_b - 1) * B.var(axis=1, ddof=1)) / df)
    d = np.divide(A.mean(axis=1) - B.mean(axis=1), pooled, out=np.full(A.shape[0], np.nan), where=pooled > 0)
    return d if estimator == "cohens_d" else d * _hedges_correction(df)


def percentile_interval(draws, alpha=0.05):
    """
    The `1 - alpha` percentile bootstrap interval: the `alpha/2` and
    `1 - alpha/2` quantiles of `draws` (NaN draws ignored), taken as order
    statistics (the draw just below / just above) so infinite draws -- an
    odds ratio with no pair favouring y -- stay well-defined.
    """
    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    if not draws.shape[0]:
        return np.nan, np.nan
    return float(np.quantile(draws, alpha / 2, method="lower")), float(np.quantile(draws, 1 - alpha / 2, method="higher"))


def bootstrap_pvalue(draws, null=0.0, alternative="two-sided", interval="percentile", estimate=None, acceleration=0.0):
    """
    The p-value matching the percentile interval: the share of bootstrap
    draws on the far side of `null`, with the +1 correction so it is never
    0 -- `"greater"` (H1: the quantity is above `null`): `(#{draw <= null}
    + 1) / (B + 1)`; `"less"`: the mirror image; two-sided: twice the
    smaller of the two, capped at 1. Draws exactly at `null` count against
    H1 (conservative for discrete estimators). NaN draws are ignored.

    !!! note "Technical details"
        This inverts the percentile interval: the two-sided p is below
        `alpha` exactly when the `1 - alpha` interval excludes `null` (up
        to the +1 correction and quantile rounding), so the test and the
        interval never disagree. It resamples the observed data rather
        than data shifted to the null, the usual choice for
        percentile-interval inversion (Efron & Tibshirani 1993, ch. 16).
        The smallest attainable p is `2 / (B + 1)` two-sided -- with Holm
        across m comparisons, choose B well above `2m / alpha`.

    Parameters
    ----------
    draws: array-like
        The bootstrap distribution (see `bootstrap_draws`).

    null: float
        The value tested against (default 0).

    alternative: str
        "two-sided" | "greater" | "less".

    interval: str
        The interval the p-value inverts (see `bootstrap_interval`):
        "percentile" (default, as above), "basic" (the same count on the
        reflected draws `2 * estimate - draw`) or "bca" (the share of
        draws beyond `null`, mapped back through the BCa adjustment) -- so
        the p-value is below `alpha` exactly when that interval excludes
        `null`, whichever is used.

    estimate: float, optional
        "basic"/"bca" only: the point estimate the draws are read against.

    acceleration: float, optional
        "bca" only (default 0, the bias-corrected "BC" interval).

    Returns
    -------
    float
        NaN without any valid draw.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative={alternative!r} isn't supported. Choose from {list(_ALTERNATIVES)}.")
    check_bootstrap_interval(interval)
    if interval != "percentile" and estimate is None:
        raise ValueError(f"interval={interval!r} reads the draws against the point estimate -- pass `estimate`.")
    if interval == "basic":
        return bootstrap_pvalue(reflect_draws(draws, estimate), null, alternative)
    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    n = draws.shape[0]
    if not n:
        return np.nan
    if interval == "bca":
        z0 = bias_correction(draws, estimate)
        if not np.isfinite(z0):
            return np.nan
        below = (np.sum(draws <= null) + 1) / (n + 1)
        above = (np.sum(draws >= null) + 1) / (n + 1)
        # A one-sided p is the level whose BCa interval end sits at `null`; the upper end is the
        # lower end of the mirrored draws (z0 -> -z0, a -> -a).
        greater = max(_bca_tail(below, z0, acceleration), 1 / (n + 1))
        less = max(_bca_tail(above, -z0, -acceleration), 1 / (n + 1))
        if alternative == "greater":
            return float(greater)
        if alternative == "less":
            return float(less)
        return float(min(1.0, 2 * min(greater, less)))
    below = (np.sum(draws <= null) + 1) / (n + 1)
    above = (np.sum(draws >= null) + 1) / (n + 1)
    if alternative == "greater":
        return float(below)
    if alternative == "less":
        return float(above)
    return float(min(1.0, 2 * min(below, above)))


#: How a bootstrap interval is read from its draws -- see `bootstrap_interval`.
BOOTSTRAP_INTERVALS = ("percentile", "basic", "bca")
#: The estimators "basic"/"bca" accept: unbounded, on an additive scale, so a draw reflected
#: around the estimate is still a possible value. Not proportions, rank-based or ratio
#: estimators (a reflected odds ratio can be negative, a reflected `cles` above 1).
ADDITIVE_ESTIMATORS = ("mean", "median", "mean_diff", "hodges_lehmann", "cohens_dz", "hedges_gz", "cohens_d", "hedges_g")


def check_bootstrap_interval(interval, estimator=None):
    """Raises unless `interval` is a known bootstrap interval that suits `estimator` (when given)."""
    if interval not in BOOTSTRAP_INTERVALS:
        raise ValueError(f"Bootstrap interval {interval!r} isn't supported. Choose from {list(BOOTSTRAP_INTERVALS)}.")
    if interval != "percentile" and estimator is not None and estimator not in ADDITIVE_ESTIMATORS:
        raise ValueError(
            f"The {interval!r} interval reflects the draws around the estimate, which only suits an unbounded, "
            f"additive estimator ({list(ADDITIVE_ESTIMATORS)}) -- not {estimator!r}. Use the 'percentile' interval."
        )


def reflect_draws(draws, estimate):
    """`2 * estimate - draws`: the draws mirrored around the estimate -- the distribution the basic interval reads."""
    return 2 * float(estimate) - np.asarray(draws, dtype=float)


def bias_correction(draws, estimate):
    """
    The BCa bias correction `z0 = Phi^-1(share of draws below the
    estimate)`, draws equal to it counted half (so a discrete estimator
    isn't biased by its ties): 0 when the draws are median-unbiased,
    +-inf when every draw lies on one side. NaN draws are ignored.
    """
    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    if not draws.shape[0] or not np.isfinite(estimate):
        return np.nan
    share = (np.sum(draws < estimate) + 0.5 * np.sum(draws == estimate)) / draws.shape[0]
    return float(stats.norm.ppf(share))


def bootstrap_bias(draws, estimate):
    """
    How far the bootstrap draws sit from the estimate: `boot_bias` (mean
    draw minus the estimate), `boot_se` (the draws' standard deviation),
    `bias_se_ratio` (their ratio) and `z0` (see `bias_correction`).
    Infinite draws are left out of the mean and the SD.

    !!! note "Technical details"
        A bias above about a quarter of the SE matters for the interval
        (Efron & Tibshirani 1993, ch. 10): the percentile interval then
        lands about twice the bias off, towards where the draws lean. The
        basic and BCa intervals correct for it (see `bootstrap_interval`).

    Returns
    -------
    dict
        NaN entries without at least two finite draws.
    """
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    out = {"boot_bias": np.nan, "boot_se": np.nan, "bias_se_ratio": np.nan, "z0": bias_correction(draws, estimate)}
    if finite.shape[0] >= 2 and np.isfinite(estimate):
        bias, se = float(finite.mean() - estimate), float(finite.std(ddof=1))
        out.update(boot_bias=bias, boot_se=se, bias_se_ratio=bias / se if se > 0 else np.nan)
    return out


def jackknife_values(x, y=None, estimator="mean_diff", paired=True, zero_method="wilcox"):
    """
    The leave-one-out values of `estimator`, one array per resampled
    sample: one (the values, or the pairs when paired), or two for
    independent samples (each unit of `x` left out in turn, then each of
    `y`). NaNs are dropped first, as in `bootstrap_draws`.
    """
    x = np.asarray(x, dtype=float)
    if y is None or paired:
        v = x if y is None else x - np.asarray(y, dtype=float)
        v = v[~np.isnan(v)]
        return [np.array([_sample_value(np.delete(v, i), estimator, y is None, zero_method) for i in range(v.shape[0])])]
    y = np.asarray(y, dtype=float)
    a, b = x[~np.isnan(x)], y[~np.isnan(y)]
    return [
        np.array([_independent_effect_size(np.delete(a, i), b, estimator) for i in range(a.shape[0])]),
        np.array([_independent_effect_size(a, np.delete(b, j), estimator) for j in range(b.shape[0])]),
    ]


def jackknife_acceleration(*jackknife):
    """
    The BCa acceleration from leave-one-out values (see
    `jackknife_values`), one array per resampled sample -- Efron's
    skewness of the jackknife influence values, in the multi-sample form
    `scipy.stats.bootstrap` uses. 0 without spread. NaN values are ignored.
    """
    num = den = 0.0
    for values in jackknife:
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        n = values.shape[0]
        if n < 2:
            continue
        u = (n - 1) * (values.mean() - values)
        num += np.sum(u ** 3) / n ** 3
        den += np.sum(u ** 2) / n ** 2
    return float(num / (6 * den ** 1.5)) if den > 0 else 0.0


def _bca_levels(levels, z0, acceleration):
    """The quantile levels the BCa interval reads instead of the nominal `levels`."""
    z = z0 + stats.norm.ppf(levels)
    return stats.norm.cdf(z0 + z / (1 - acceleration * z))


def _bca_tail(share, z0, acceleration):
    """
    Inverts `_bca_levels`: the nominal level whose adjusted level is
    `share` -- 0 (or 1) past the point the adjustment can reach.
    """
    w = stats.norm.ppf(share) - z0
    if np.isinf(w):  # every draw beyond the null
        return 0.0 if w < 0 else 1.0
    d = 1 + acceleration * w
    if d <= 0:
        return 0.0 if w < 0 else 1.0
    return float(stats.norm.cdf(w / d - z0))


def bootstrap_interval(draws, estimate, alpha=0.05, interval="percentile", acceleration=0.0):
    """
    The `1 - alpha` bootstrap interval of a quantity estimated at
    `estimate`, read from its `draws` one of three ways:

    - "percentile": the `alpha/2` and `1 - alpha/2` quantiles of the draws
      (see `percentile_interval`);
    - "basic": `[2 * estimate - q_high, 2 * estimate - q_low]` -- the
      percentile interval of the draws reflected around the estimate
      (`reflect_draws`);
    - "bca": the quantiles at the levels `Phi(z0 + (z0 + z) / (1 - a (z0
      + z)))` for `z` the nominal normal quantiles, with the bias
      correction `z0` (`bias_correction`) and the `acceleration` a (0:
      the bias-corrected "BC" interval).

    !!! note "Technical details"
        The percentile interval is valid only when the draws are not
        median-biased: if they lean away from the estimate (e.g. resampled
        spike averages, noisier than the full average, attenuating a
        correlation), it lands about twice the bias off, on the side the
        draws lean. The basic interval assumes instead that the error from
        the truth to the estimate equals the error from the estimate to a
        draw, and so corrects for that bias (Davison & Hinkley 1997, ch.
        5); BCa corrects for the bias and for a skew that changes with the
        parameter (Efron & Tibshirani 1993, ch. 14). Beware the names:
        Hall (1992) calls the basic interval the "percentile method" (and
        this module's percentile interval "the other percentile method").

        Quantiles are order statistics (the draw just below / above), as in
        `percentile_interval`. BCa warns when an adjusted level falls
        outside `[1/n, 1 - 1/n]` for `n` draws: that end is then just the
        extreme draw -- more draws, or a less extreme `z0`, are needed.

    Parameters
    ----------
    draws: array-like
        The bootstrap distribution (NaN draws ignored).

    estimate: float
        The point estimate ("basic"/"bca" only).

    alpha: float
        The interval is `1 - alpha` (default 95%).

    interval: str
        "percentile" (default), "basic" or "bca".

    acceleration: float
        "bca" only -- see `jackknife_acceleration`.

    Returns
    -------
    (float, float)
        NaN without any valid draw (or, for BCa, when every draw lies on
        one side of the estimate).
    """
    check_bootstrap_interval(interval)
    if interval == "percentile":
        return percentile_interval(draws, alpha)
    if interval == "basic":
        return percentile_interval(reflect_draws(draws, estimate), alpha)
    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    n = draws.shape[0]
    z0 = bias_correction(draws, estimate)
    if not n or not np.isfinite(z0):
        if n:
            warnings.warn("Every bootstrap draw lies on one side of the estimate -- the BCa interval is undefined.", RuntimeWarning, stacklevel=2)
        return np.nan, np.nan
    low, high = _bca_levels(np.array([alpha / 2, 1 - alpha / 2]), z0, acceleration)
    if min(low, high) < 1 / n or max(low, high) > 1 - 1 / n:
        warnings.warn(
            f"The BCa-adjusted quantile levels ({low:.4g}, {high:.4g}) fall outside [1/n, 1 - 1/n] for n={n} draws: "
            "that interval end is just the extreme draw -- use more draws.", RuntimeWarning, stacklevel=2,
        )
    low, high = np.clip([low, high], 0.0, 1.0)
    return float(np.quantile(draws, low, method="lower")), float(np.quantile(draws, high, method="higher"))


def estimate(
    x, y=None, estimator="hodges_lehmann", paired=True, alpha=0.05, ci=True, n_eff=None, zero_method="wilcox",
    ci_method="analytic", n_boot=DEFAULT_N_BOOT, seed=None, draws=None, interval="percentile", acceleration=None,
):
    """
    A point estimate, optionally with its `1 - alpha` confidence interval
    -- of one sample (`y=None`: "mean", "median"), or of the difference
    between `x` and `y` (every effect size `effect_size` defines: matched
    when `paired=True`, independent samples otherwise). Unlike
    `paired_test`, there is no hypothesis and no offset here: it reports
    *how large* a quantity is, to be read against whatever offset matters.

    Parameters
    ----------
    x: array-like of shape (n,)

    y: array-like, optional
        Matched to `x` (same length) when `paired=True`; any length when
        `paired=False`. NaNs are dropped (pairwise when paired).

    estimator: str
        One sample: "mean" (t interval), "median" (exact binomial
        order-statistic interval), "proportion" (the share of 1s of a 0/1
        outcome, with the Wilson score interval). Difference: "mean_diff" (paired: t
        interval on the differences, or with `n_eff`, the same interval
        with the effective sample size `ttest_ess` uses; independent:
        Welch interval), "hodges_lehmann" (default -- paired: the
        pseudo-median of the differences, with the distribution-free
        signed-rank interval, exact for n <= 200 and normal approximation
        above; independent: the median of every `x_i - y_j`, with the
        normal-approximation Mann-Whitney interval; neither corrects for
        ties or serial correlation; memory is O(n^2), see
        `_MAX_PAIRWISE_TERMS`), "prop_diff" (paired 0/1 outcome: the
        difference in proportions, with Newcombe's 1998 hybrid score
        interval for paired data), "odds_ratio" (paired 0/1 outcome: the
        conditional odds ratio b / c, with its exact interval), and the
        standardized "rank_biserial",
        "cohens_dz", "hedges_gz" (paired), "cles", "cohens_d", "hedges_g"
        (independent) -- bootstrap interval only (`ci_method`).

    paired: bool
        Only used when `y` is given.

    alpha: float
        The interval is `1 - alpha` (default 95%). `ci_low`/`ci_high` are
        NaN when the sample is too small to reach that level at all.

    ci: bool
        Whether to compute the interval (default True). With
        `ci_method="analytic"`, raises `NotImplementedError` for an
        estimator with no analytic interval (see `ESTIMATORS_WITH_CI`) --
        pass `ci_method="bootstrap"`, or `ci=False` for its point value
        alone.

    n_eff: float, optional
        Paired only: an effective sample size (see `effective_sample_size`)
        for an autocorrelated series -- used by "mean_diff"'s interval and
        "hedges_gz"'s correction. Not supported with "hodges_lehmann"'s
        interval (no autocorrelation-corrected version exists here).

    zero_method: str, optional
        Paired "rank_biserial" only -- how zero differences are ranked
        (default "wilcox"; see `effect_size`).

    ci_method: str, optional
        "analytic" (default -- the estimator's own interval, as above) or
        "bootstrap": the percentile interval of `bootstrap_draws` (see
        `percentile_interval`), available for every estimator, the
        standardized ones included. Not with `n_eff`: the bootstrap
        resamples units independently, so it ignores serial correlation.

    n_boot: int, optional
        "bootstrap" only: number of resamples (default 10,000).

    seed: int | np.random.Generator, optional
        "bootstrap" only: seeds the resampling.

    draws: array-like, optional
        "bootstrap" only: a precomputed bootstrap distribution to read the
        interval from instead of resampling `x`/`y` -- e.g. one value per
        resample of a quantity this function can't resample itself
        (spikes averaged upstream). `n_boot`/`seed` are then unused.

    interval: str, optional
        "bootstrap" only: how the interval is read from the draws --
        "percentile" (default), "basic" or "bca" (see
        `bootstrap_interval`). "basic"/"bca" only for the additive
        estimators (`ADDITIVE_ESTIMATORS`).

    acceleration: float, optional
        "bca" only. Defaults to the jackknife acceleration of `x`/`y`
        (`jackknife_values`) -- pass it with `draws` that don't resample
        `x`/`y` themselves (0 for the bias-corrected "BC" interval).

    Returns
    -------
    dict with keys "estimate", "ci_low", "ci_high", "ci_method" (NaN/NaN/
    `None` when `ci=False`; "bootstrap_<interval>" with
    `ci_method="bootstrap"`), "estimator", and "n" (one sample/paired) or
    "n_a"/"n_b" (independent). A bootstrap interval adds "n_boot", the
    number of valid (non-NaN) draws, and "acceleration" for BCa.
    """
    x = np.asarray(x, dtype=float)
    _check_zero_method(zero_method)
    if y is None:
        if estimator not in GROUP_ESTIMATORS:
            raise ValueError(
                f"estimator={estimator!r} describes a difference -- pass `y`. "
                f"One sample takes {list(GROUP_ESTIMATORS)}."
            )
        sample = x[~np.isnan(x)]
        if estimator == "proportion":
            _check_binary(sample)
    else:
        if estimator not in DIFFERENCE_ESTIMATORS:
            raise ValueError(
                f"estimator={estimator!r} describes one sample, not a difference -- "
                f"use one of {list(DIFFERENCE_ESTIMATORS)} (e.g. 'mean_diff' or 'hodges_lehmann')."
            )
        allowed = PAIRED_EFFECT_SIZES if paired else INDEPENDENT_EFFECT_SIZES
        if estimator not in allowed:
            kind = "paired" if paired else "independent"
            raise ValueError(f"estimator={estimator!r} isn't defined for {kind} samples. Choose from {list(allowed)}.")
        y = np.asarray(y, dtype=float)
        if paired:
            if x.shape != y.shape:
                raise ValueError(f"x and y must have the same shape when paired (got {x.shape} vs {y.shape}).")
            d = x - y
            sample = d[~np.isnan(d)]
            if estimator in BINARY_EFFECT_SIZES:
                _check_binary(x, y)
                keep = ~np.isnan(x) & ~np.isnan(y)
                pair_x, pair_y = x[keep], y[keep]
        else:
            a, b = x[~np.isnan(x)], y[~np.isnan(y)]
    if n_eff is not None and (y is None or not paired):
        raise ValueError("`n_eff` only applies to paired differences.")
    if n_eff is not None and estimator in BINARY_EFFECT_SIZES:
        raise ValueError(f"`n_eff` doesn't apply to {estimator!r}.")

    if ci_method not in CI_METHODS:
        raise ValueError(f"ci_method={ci_method!r} isn't supported. Choose from {list(CI_METHODS)}.")
    if ci_method == "bootstrap" and n_eff is not None:
        raise ValueError(
            "The bootstrap resamples units independently, so it can't honour `n_eff` (serial correlation) -- "
            "use ci_method='analytic' with 'mean_diff'."
        )
    if draws is not None and ci_method != "bootstrap":
        raise ValueError("`draws` only applies to ci_method='bootstrap'.")
    if (interval != "percentile" or acceleration is not None) and ci_method != "bootstrap":
        raise ValueError("`interval`/`acceleration` only apply to ci_method='bootstrap'.")
    if acceleration is not None and interval != "bca":
        raise ValueError("`acceleration` only applies to interval='bca'.")
    check_bootstrap_interval(interval, estimator)
    if ci and ci_method == "analytic" and estimator not in ESTIMATORS_WITH_CI:
        raise NotImplementedError(
            f"No analytic confidence interval is implemented for {estimator!r} -- pass ci_method='bootstrap' "
            "for a bootstrap interval, or ci=False for its point value."
        )
    if ci and n_eff is not None and estimator == "hodges_lehmann":
        raise NotImplementedError(
            "The Hodges-Lehmann interval has no effective-sample-size correction -- pass ci=False, "
            "or use 'mean_diff', whose interval does."
        )

    core = {"mean_diff": "mean"}.get(estimator, estimator)
    if ci and ci_method == "analytic":
        if estimator == "proportion":
            low, high = _wilson_interval(int(sample.sum()), sample.shape[0], alpha)
            point = float(sample.mean()) if sample.shape[0] else np.nan
            out = {"n": sample.shape[0], "estimate": point, "ci_low": float(low), "ci_high": float(high), "ci_method": "wilson"}
        elif estimator in BINARY_EFFECT_SIZES:
            out = _binary_paired_estimate(pair_x, pair_y, estimator, alpha)
        elif y is None or paired:
            out = _one_sample_estimate(sample, core, alpha, n_eff)
        else:
            out = _two_sample_estimate(a, b, core, alpha)
    else:
        if y is None:
            point = float(np.median(sample) if estimator == "median" else np.mean(sample)) if sample.shape[0] else np.nan
            out = {"n": sample.shape[0]}
        elif paired:
            point = _paired_effect_size(sample, estimator, n_eff, zero_method)
            out = {"n": sample.shape[0]}
        else:
            point = _independent_effect_size(a, b, estimator)
            out = {"n_a": a.shape[0], "n_b": b.shape[0]}
        out.update(estimate=point, ci_low=np.nan, ci_high=np.nan, ci_method=None)
        if ci:
            if draws is None:
                draws = bootstrap_draws(x, y, estimator, paired, n_boot, seed, zero_method)
            draws = np.asarray(draws, dtype=float)
            if interval == "bca" and acceleration is None:
                acceleration = jackknife_acceleration(*jackknife_values(x, y, estimator, paired, zero_method))
            low, high = bootstrap_interval(draws, point, alpha, interval, acceleration or 0.0)
            out.update(ci_low=low, ci_high=high, ci_method=f"bootstrap_{interval}", n_boot=int(np.sum(~np.isnan(draws))))
            if interval == "bca":
                out["acceleration"] = float(acceleration)
    out["estimator"] = estimator
    if n_eff is not None:
        out["n_eff"] = float(n_eff)
    return out


def _holm(p_values):
    """
    Holm (1979) step-down adjusted p-values: the i-th smallest of m p-values
    is multiplied by `m - i + 1`, then made monotone (running maximum) and
    clipped to 1. NaNs stay NaN and don't count towards m.
    """
    adjusted = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    p = p_values[valid]
    m = p.shape[0]
    if m == 0:
        return adjusted
    order = np.argsort(p)
    stepped = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = stepped
    adjusted[valid] = out
    return adjusted


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
        (Benjamini-Yekutieli, `method="by"`) | "holm" (Holm-Bonferroni
        step-down: controls the family-wise error rate like "bonferroni",
        but is uniformly more powerful) | "bonferroni" (`p * n`, clipped
        to 1) | "none" (`p_values` returned unchanged).

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
    elif method == "holm":
        p_corrected = _holm(p_values)
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
