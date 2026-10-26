"""
Module Name: metric/stat.py
Description: `MetricSuite.stat`'s accessor -- one method per statistical
test, bound to a `MetricSuite` instance, each pure computation returning a
typed, self-describing result (`PairwiseResult`/`ClusterResult`/
`NormalityResult`/`AnovaResult`) -- computed fresh on every call; save a
result (`result.save`) to keep it. Separate from `MetricSuite.plot` -- `suite.stat.X` computes,
`suite.plot.X` (when it exists for `X`) visualizes an already-computed
result.

Author: Nino Herve
"""

import warnings

import numpy as np
import pandas as pd

from mindmap.metric.results import (
    PairwiseResult, ClusterResult, NormalityResult, AnovaResult, _freeze_filters, _paired_wide, _pair_samples,
    _split_replicates, comparison_draws, resolve_bootstrap, bootstrap_metrics, BIAS_SE_WARNING,
)
from mindmap.core.stats import reflect_draws
from mindmap.metric.plot import _filters_to_dict
from mindmap.metric._plotdata import infer_reference as _infer_reference


class StatAccessor:
    """
    `MetricSuite.stat`'s namespace -- bound to one suite, accessed as
    `suite.stat.wilcoxon(...)`, `suite.stat.cluster_permutation(...)`,
    `suite.stat.normality_test(...)`, `suite.stat.anova(...)`. Every method
    here is pure computation (no plotting), run fresh on every call -- see
    each method's own docstring, and `suite.plot.group_comparison`/
    `.cluster_permutation`/`.normality_test`/`.anova` for visualizing what
    comes back.

    !!! note "Technical details"
        Nothing is cached: a result carries every setting that produced it
        and a `data_key` fingerprint of its rows, so it can be saved on
        its own (`result.save`) and checked against a suite later
        (`MetricSuite.verify`).

        Seven methods cover pairwise comparisons -- `wilcoxon`, `mcnemar`
        (binary outcome), `paired_ttest`, `ttest_ess` (paired/matched,
        need `pairing_col`), `ttest`, `mannwhitneyu` (independent-samples,
        no `pairing_col`), and `bootstrap` (either, on any estimator) --
        all returning `PairwiseResult`, all but `mcnemar` able to test
        against a non-zero `offset` ("is A better than B by more than
        0.05?"), and all reporting effect sizes (`effect_sizes`, see
        `PairwiseResult.EFFECT_SIZES`; `bootstrap` reports its
        `estimator`) -- each with a confidence interval by default (`ci`,
        `ci_level`): analytic when the estimator has one, a seeded
        percentile bootstrap otherwise.
        `build_comparisons` is a plain helper for assembling the
        `comparisons` list any of them take.
    """

    def __init__(self, suite):
        self._suite = suite

    # ------------------------------------------------------------------
    # Pairwise comparisons: wilcoxon / mcnemar / ttest / mannwhitneyu / paired_ttest / ttest_ess.
    # ------------------------------------------------------------------

    @staticmethod
    def build_comparisons(references, competitors):
        """
        Builds every `(reference, competitor)` pair from two lists, for
        the common "each of several baselines vs each of several
        candidates" shape -- skips self-comparisons and pairs already seen
        in either order, so overlapping `references`/`competitors` lists
        don't produce the same pair twice.

        Parameters
        ----------
        references: str | list[str]

        competitors: str | list[str]

        Returns
        -------
        list[tuple[str, str]]

        Examples
        --------
        >>> StatAccessor.build_comparisons(["cset"], ["wcset", "mne"])
        [('cset', 'wcset'), ('cset', 'mne')]
        >>> StatAccessor.build_comparisons(["cset", "wcset"], ["wcset", "mne"])
        [('cset', 'wcset'), ('cset', 'mne'), ('wcset', 'mne')]
        """
        if isinstance(references, str):
            references = [references]
        if isinstance(competitors, str):
            competitors = [competitors]

        seen = set()
        pairs = []
        for r in references:
            for c in competitors:
                if r == c:
                    continue
                key = frozenset((r, c))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((r, c))
        return pairs

    def _pairwise(
        self, metric, group_by, comparisons, test, correction, alpha, unit_col, pairing_col, subject_order, filters, offset,
        alternative, effect_sizes, zero_method=None, ci=True, ci_level=0.95, ci_seed=None,
    ):
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        result = PairwiseResult.compute(
            df, metric=metric, group_by=group_by, comparisons=comparisons, test=test, unit_col=unit_col,
            pairing_col=pairing_col, subject_order=subject_order, filters=_freeze_filters(filters),
            offset=offset, alternative=alternative, effect_sizes=effect_sizes, zero_method=zero_method,
            ci=ci, ci_level=ci_level, ci_seed=ci_seed,
        )
        return result.correct(correction, alpha)

    def wilcoxon(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None, offset=0.0, alternative=None,
        effect_sizes=None, zero_method="wilcox", ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Wilcoxon signed-rank test -- the non-parametric paired comparison,
        matched by `pairing_col` (e.g. "is wCSET better than CSET,
        region-by-region?"). Pure computation -- for a plot, see
        `suite.plot.group_comparison`.

        !!! note "Technical details"
            Each pair is tested on its own complete-case subjects,
            independent of any other pair requested in the same call;
            `correction` then treats all pairs as one family. `unit_col`,
            when given, is a dedup axis: rows are first collapsed to one
            value per `(group_by, pairing_col, unit_col)`, then to one per
            `(group_by, pairing_col)`, before testing -- so e.g. several
            repeated runs from one region don't get treated as several
            independent pieces of evidence. To change only the
            correction/alpha afterwards, `result.correct(...)` re-corrects
            without rerunning the tests.

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        group_by: str
            Which registered column holds the groups being compared (e.g.
            "Method").

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons` for an easy way to build this.

        pairing_col: str
            The repeated-measures/matched column (e.g. "region") --
            required, since this is a paired test.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        correction: str, optional
            Multiple-comparison correction across every pair in the
            result (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        offset: float, optional
            Test `group_a - group_b` against this value instead of 0
            (default 0) -- e.g. `offset=0.05` asks "is group_a higher by
            more than 0.05?". Only the p-value changes: effect sizes always
            describe the raw difference (compare e.g. `hodges_lehmann`
            with the offset).

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"` (`group_a - group_b`
            vs. `offset`). Defaults to `"two-sided"` when `offset` is 0
            and `"greater"` otherwise. For a lower-is-better metric, "is
            group_a lower by more than d" is `offset=-d,
            alternative="less"`.

        effect_sizes: str | tuple[str, ...], optional
            Which effect sizes to report, one column each -- any of
            `"rank_biserial"`, `"hodges_lehmann"` (default `"rank_biserial"`). The first is the one `vs()` and
            the plots show by default.

        zero_method: str, optional
            What to do with pairs whose difference (minus `offset`) is
            exactly 0 -- a tie, e.g. both methods inside the target region:

            - `"wilcox"` (default): drop them. The test then asks "among
              the pairs that differ, which side wins?", and `n_pairs`
              shrinks.
            - `"pratt"`: rank them with the others, then leave their
              ranks out of both sums. They stay in the sample, so many
              ties weaken the evidence; `n_pairs` keeps every pair.
            - `"zsplit"`: rank them, and split their ranks evenly between
              both sums.

            `table["n_zero"]` reports how many ties there were. The
            rank-biserial follows the same convention: with `"pratt"`/
            `"zsplit"`, the zeros' ranks stay in its denominator, so ties
            pull r toward 0 (both give the same r).

        ci: bool, optional
            Add a confidence interval to every effect size (default True),
            as `<name>_ci_low`/`<name>_ci_high` columns (oriented by
            `vs()`): the analytic interval `suite.estimate` reports when
            the estimator has one, else a percentile bootstrap.
            `result.ci_methods` says which. `ci=False` skips them (faster
            when only the p-values matter).

        ci_level: float, optional
            The intervals' coverage (default 0.95), independent of `alpha`
            -- a one-sided test at 0.05 matches a 0.90 interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals; drawn at random when omitted and
            stored on the result (`ci_seed`).

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "wilcoxon", correction, alpha, unit_col, pairing_col, None, filters, offset, alternative, effect_sizes, zero_method, ci=ci, ci_level=ci_level, ci_seed=ci_seed)

    def mcnemar(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None,
        alternative="two-sided", effect_sizes=None, ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Exact McNemar test -- the paired comparison of a binary 0/1 outcome
        (e.g. "peak inside the resection"), matched by `pairing_col`: do
        the two groups reach the outcome equally often? Pure computation --
        estimates with intervals are `suite.estimate.prop_diff` /
        `.odds_ratio`.

        !!! note "Technical details"
            Only the discordant pairs carry information: `n_a_only`
            (group_a's outcome 1, group_b's 0) and `n_b_only` (the
            reverse). Under the null each discordant pair is equally likely
            to go either way, so the p-value is the exact binomial test of
            `n_a_only` out of `n_a_only + n_b_only` at 1/2 (1 when there is
            no discordant pair) -- not the chi-square approximation, which
            is unreliable with few discordant pairs. `n_zero` counts the
            concordant pairs, which don't affect the p-value but dilute
            `prop_diff`. There is no `offset`: McNemar tests equal
            proportions.

            The metric must hold 0/1 values -- register the outcome as its
            own metric (e.g. `inside = distance == 0`). With `unit_col`,
            averaging repeated rows must still give 0/1 per pair.

        Parameters
        ----------
        metric: str
            Which registered metric (coded 0/1) to compare.

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons`.

        pairing_col: str
            The repeated-measures/matched column (e.g. "patient") --
            required.

        unit_col: str, optional
            Dedup axis.

        correction: str, optional
            Multiple-comparison correction (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        alternative: str, optional
            `"two-sided"` (default) | `"greater"` (group_a's outcome is 1
            more often) | `"less"`.

        effect_sizes: str | tuple[str, ...], optional
            Which effect sizes to report, one column each -- any of
            `"prop_diff"` (the difference in proportions, `(n_a_only -
            n_b_only) / n_pairs`, in proportion units) and `"odds_ratio"`
            (the conditional odds ratio `n_a_only / n_b_only`; `inf` when
            `n_b_only` is 0) (default `"prop_diff"`).

        ci: bool, optional
            Add a confidence interval to every effect size (default True),
            as `<name>_ci_low`/`<name>_ci_high` columns (oriented by
            `vs()`): the analytic interval `suite.estimate` reports when
            the estimator has one, else a percentile bootstrap.
            `result.ci_methods` says which. `ci=False` skips them (faster
            when only the p-values matter).

        ci_level: float, optional
            The intervals' coverage (default 0.95), independent of `alpha`
            -- a one-sided test at 0.05 matches a 0.90 interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals; drawn at random when omitted and
            stored on the result (`ci_seed`).

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "mcnemar", correction, alpha, unit_col, pairing_col, None, filters, 0.0, alternative, effect_sizes, ci=ci, ci_level=ci_level, ci_seed=ci_seed)

    def paired_ttest(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None, offset=0.0, alternative=None,
        effect_sizes=None, ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Paired (related-samples) t-test, matched by `pairing_col` --
        the parametric analog of `wilcoxon`. Assumes the paired
        differences are roughly normal (see `normality_test`). Pure
        computation -- for a plot, see `suite.plot.group_comparison`.

        !!! note "Technical details"
            Same collapse behavior as `wilcoxon` -- see its technical
            details.

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons`.

        pairing_col: str
            The repeated-measures/matched column -- required.

        unit_col: str, optional
            Dedup axis.

        correction: str, optional
            Multiple-comparison correction (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        offset: float, optional
            Test `group_a - group_b` against this value instead of 0
            (default 0) -- e.g. `offset=0.05` asks "is group_a higher by
            more than 0.05?". Only the p-value changes: effect sizes always
            describe the raw difference (compare e.g. `hodges_lehmann`
            with the offset).

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"` (`group_a - group_b`
            vs. `offset`). Defaults to `"two-sided"` when `offset` is 0
            and `"greater"` otherwise. For a lower-is-better metric, "is
            group_a lower by more than d" is `offset=-d,
            alternative="less"`.

        effect_sizes: str | tuple[str, ...], optional
            Which effect sizes to report, one column each -- any of
            `"cohens_dz"`, `"hedges_gz"`, `"mean_diff"` (default `"cohens_dz"`). The first is the one `vs()` and
            the plots show by default.

        ci: bool, optional
            Add a confidence interval to every effect size (default True),
            as `<name>_ci_low`/`<name>_ci_high` columns (oriented by
            `vs()`): the analytic interval `suite.estimate` reports when
            the estimator has one, else a percentile bootstrap.
            `result.ci_methods` says which. `ci=False` skips them (faster
            when only the p-values matter).

        ci_level: float, optional
            The intervals' coverage (default 0.95), independent of `alpha`
            -- a one-sided test at 0.05 matches a 0.90 interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals; drawn at random when omitted and
            stored on the result (`ci_seed`).

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "paired_ttest", correction, alpha, unit_col, pairing_col, None, filters, offset, alternative, effect_sizes, ci=ci, ci_level=ci_level, ci_seed=ci_seed)

    def ttest_ess(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, subject_order=None, correction="fdr_bh", alpha=0.05, filters=None, offset=0.0, alternative=None,
        effect_sizes=None, ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Paired t-test with an autocorrelation-corrected effective sample
        size -- for when `pairing_col` has real serial/spatial correlation
        (e.g. adjacent timepoints or channels), which would otherwise
        overstate independent evidence if treated at face-value `n`.
        Pure computation -- for a plot, see `suite.plot.group_comparison`.

        !!! note "Technical details"
            Same collapse behavior as `wilcoxon` -- see its technical
            details. `n_eff` is computed via
            `mindmap.core.stats.effective_sample_size` from the lag-k
            autocorrelation of the paired-difference series, so
            `subject_order` (the true order of `pairing_col`'s values,
            often not the default pivot order) matters here specifically
            -- for the other four tests, order is irrelevant.

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons`.

        pairing_col: str
            The repeated-measures/matched column -- required.

        unit_col: str, optional
            Dedup axis.

        subject_order: list, optional
            Explicit true order of `pairing_col`'s values, needed for
            correct autocorrelation estimation.

        correction: str, optional
            Multiple-comparison correction (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        offset: float, optional
            Test `group_a - group_b` against this value instead of 0
            (default 0) -- e.g. `offset=0.05` asks "is group_a higher by
            more than 0.05?". Only the p-value changes: effect sizes always
            describe the raw difference (compare e.g. `hodges_lehmann`
            with the offset).

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"` (`group_a - group_b`
            vs. `offset`). Defaults to `"two-sided"` when `offset` is 0
            and `"greater"` otherwise. For a lower-is-better metric, "is
            group_a lower by more than d" is `offset=-d,
            alternative="less"`.

        effect_sizes: str | tuple[str, ...], optional
            Which effect sizes to report, one column each -- any of
            `"cohens_dz"`, `"hedges_gz"`, `"mean_diff"` (default `"cohens_dz"`). The first is the one `vs()` and
            the plots show by default.

        ci: bool, optional
            Add a confidence interval to every effect size (default True),
            as `<name>_ci_low`/`<name>_ci_high` columns (oriented by
            `vs()`): the analytic interval `suite.estimate` reports when
            the estimator has one, else a percentile bootstrap.
            `result.ci_methods` says which. `ci=False` skips them (faster
            when only the p-values matter).

        ci_level: float, optional
            The intervals' coverage (default 0.95), independent of `alpha`
            -- a one-sided test at 0.05 matches a 0.90 interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals; drawn at random when omitted and
            stored on the result (`ci_seed`).

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "ttest_ess", correction, alpha, unit_col, pairing_col, subject_order, filters, offset, alternative, effect_sizes, ci=ci, ci_level=ci_level, ci_seed=ci_seed)

    def ttest(
        self, metric, group_by, comparisons, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None, offset=0.0, alternative=None,
        effect_sizes=None, ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Independent-samples t-test -- no correspondence between groups is
        assumed (unlike `paired_ttest`). Pure computation -- for a plot,
        see `suite.plot.group_comparison`.

        !!! note "Technical details"
            No `pairing_col` -- there's nothing to align across groups for
            an independent-samples test. `unit_col`, when given, is a
            dedup axis: rows are first collapsed to one value per
            `(group_by, unit_col)` before testing.

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons`.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        correction: str, optional
            Multiple-comparison correction (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        offset: float, optional
            Test `group_a - group_b` against this value instead of 0
            (default 0) -- e.g. `offset=0.05` asks "is group_a higher by
            more than 0.05?". Only the p-value changes: effect sizes always
            describe the raw difference (compare e.g. `hodges_lehmann`
            with the offset).

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"` (`group_a - group_b`
            vs. `offset`). Defaults to `"two-sided"` when `offset` is 0
            and `"greater"` otherwise. For a lower-is-better metric, "is
            group_a lower by more than d" is `offset=-d,
            alternative="less"`.

        effect_sizes: str | tuple[str, ...], optional
            Which effect sizes to report, one column each -- any of
            `"cohens_d"`, `"hedges_g"`, `"mean_diff"` (default `"cohens_d"`). The first is the one `vs()` and
            the plots show by default.

        ci: bool, optional
            Add a confidence interval to every effect size (default True),
            as `<name>_ci_low`/`<name>_ci_high` columns (oriented by
            `vs()`): the analytic interval `suite.estimate` reports when
            the estimator has one, else a percentile bootstrap.
            `result.ci_methods` says which. `ci=False` skips them (faster
            when only the p-values matter).

        ci_level: float, optional
            The intervals' coverage (default 0.95), independent of `alpha`
            -- a one-sided test at 0.05 matches a 0.90 interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals; drawn at random when omitted and
            stored on the result (`ci_seed`).

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "ttest", correction, alpha, unit_col, None, None, filters, offset, alternative, effect_sizes, ci=ci, ci_level=ci_level, ci_seed=ci_seed)

    def mannwhitneyu(
        self, metric, group_by, comparisons, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None, offset=0.0, alternative=None,
        effect_sizes=None, ci=True, ci_level=0.95, ci_seed=None,
    ):
        """
        Mann-Whitney U test -- the non-parametric independent-samples
        comparison, the analog of `wilcoxon` for unmatched groups. Pure
        computation -- for a plot, see `suite.plot.group_comparison`.

        !!! note "Technical details"
            Same as `ttest` -- no `pairing_col`, `unit_col` is an optional
            dedup axis.

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons`.

        unit_col: str, optional
            Dedup axis.

        correction: str, optional
            Multiple-comparison correction (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        offset: float, optional
            Test `group_a - group_b` against this value instead of 0
            (default 0) -- e.g. `offset=0.05` asks "is group_a higher by
            more than 0.05?". Only the p-value changes: effect sizes always
            describe the raw difference (compare e.g. `hodges_lehmann`
            with the offset).

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"` (`group_a - group_b`
            vs. `offset`). Defaults to `"two-sided"` when `offset` is 0
            and `"greater"` otherwise. For a lower-is-better metric, "is
            group_a lower by more than d" is `offset=-d,
            alternative="less"`.

        effect_sizes: str | tuple[str, ...], optional
            Which effect sizes to report, one column each -- any of
            `"rank_biserial"`, `"cles"`, `"hodges_lehmann"` (default `"rank_biserial"`). The first is the one `vs()` and
            the plots show by default.

        ci: bool, optional
            Add a confidence interval to every effect size (default True),
            as `<name>_ci_low`/`<name>_ci_high` columns (oriented by
            `vs()`): the analytic interval `suite.estimate` reports when
            the estimator has one, else a percentile bootstrap.
            `result.ci_methods` says which. `ci=False` skips them (faster
            when only the p-values matter).

        ci_level: float, optional
            The intervals' coverage (default 0.95), independent of `alpha`
            -- a one-sided test at 0.05 matches a 0.90 interval.

        ci_seed: int, optional
            Seeds the bootstrap intervals; drawn at random when omitted and
            stored on the result (`ci_seed`).

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "mannwhitneyu", correction, alpha, unit_col, None, None, filters, offset, alternative, effect_sizes, ci=ci, ci_level=ci_level, ci_seed=ci_seed)

    def bootstrap(
        self, metric, group_by, comparisons, pairing_col=None, unit_col=None, estimator="mean_diff", correction="fdr_bh",
        alpha=0.05, filters=None, offset=0.0, alternative=None, n_boot=None, seed=None, replicates=None,
        replicate_col=None, zero_method="wilcox", ci=True, ci_level=0.95, ci_method="percentile", acceleration=None,
        jackknife=None, jackknife_col=None,
    ):
        """
        Bootstrap test of a difference estimator -- paired (resampling
        whole `pairing_col` units) or independent (resampling each group
        on its own): is `estimator` of `group_a` vs `group_b` different
        from no difference (or beyond `offset`)? The p-value comes from
        the same draws as `suite.estimate.<estimator>(...,
        ci_method="bootstrap")`'s interval, so the two always agree. Pure
        computation -- the interval is `suite.estimate.from_result`.

        !!! note "Technical details"
            The p-value is the share of bootstrap draws on the far side of
            the null, with a +1 correction (see
            `mindmap.core.stats.bootstrap_pvalue`), read through the same
            `ci_method` as the interval: it is below `alpha` exactly when
            the `1 - alpha` interval excludes the null. The smallest p it
            can reach is `2 / (n_boot + 1)` two-sided, so keep `n_boot`
            well above `2 * m / alpha` when correcting m comparisons.

            Which interval (see `mindmap.core.stats.bootstrap_interval`):

            - "percentile" (default): the quantiles of the draws. Valid
              only when the draws are not median-biased -- e.g. resampled
              spike averages are noisier than the full average, which
              attenuates a correlation and pulls every draw towards 0; the
              percentile interval then lands about twice that bias towards
              0.
            - "basic": the quantiles reflected around the estimate,
              `[2 est - q_high, 2 est - q_low]`, i.e. the percentile
              interval of `2 est - draw`. It assumes the error from the
              truth to the estimate equals the error from the estimate to
              a draw, and so corrects for the bias. Hall (1992) calls this
              one the "percentile method" -- beware the names.
            - "bca": bias-corrected and accelerated (Efron & Tibshirani
              1993, ch. 14). The bias correction `z0` comes from the
              draws; the acceleration from leaving out each unit in turn
              when resampled here, or, with registered replicates, from a
              registered `jackknife` (one value per left-out resampling
              unit, e.g. per spike, numbered by `jackknife_col` and paired
              like the replicates) -- or pass `acceleration` (0: the
              bias-corrected "BC" interval). Warns when an adjusted
              quantile level falls outside `[1/n_boot, 1 - 1/n_boot]`.

            "basic"/"bca" only for an unbounded, additive estimator
            (`mindmap.core.stats.ADDITIVE_ESTIMATORS`: `mean_diff`,
            `hodges_lehmann` and the standardized d's).

            The table always reports, per pair, how biased the draws are:
            `boot_bias` (mean draw minus the estimate), `boot_se`,
            `bias_se_ratio` and `z0`, plus BCa's `acceleration`. A
            percentile result warns when some `|bias_se_ratio|` exceeds
            0.25 (Efron & Tibshirani 1993, ch. 10).

            The draws come from one of two places:

            - resampled here: `n_boot` resamples (default 10,000) of the
              units the test reads after collapsing (the `pairing_col`
              values when paired, else the `unit_col` values, else the
              rows) -- one generator seeded by `seed`, used across pairs in
              table order;
            - registered (`replicates`/`replicate_col`): for resamples
              made upstream, e.g. spikes re-averaged and re-inverted per
              draw. Register the observed values under `metric` and one
              value per draw under the `replicates` metric, numbered by
              `replicate_col`; draw r of every group must come from the
              same resample (a paired bootstrap), and each draw is
              collapsed and paired exactly like the observed data. The
              point estimate comes from `metric`, the distribution from
              the draws.

            Use the internal bootstrap only on independent units: it
            ignores serial correlation (for timepoints, see
            `ttest_ess`).

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(group_a, group_b)` pairs to test -- see
            `build_comparisons`.

        pairing_col: str, optional
            Match pairs on this column (e.g. "region"): a paired
            bootstrap. Omit for independent samples.

        unit_col: str, optional
            Dedup axis, collapsed exactly like the other tests.

        estimator: str, optional
            The difference estimator bootstrapped, `group_a` relative to
            `group_b` (default `"mean_diff"`) -- any of `suite.estimate`'s
            per-pair estimators valid for the design: `"mean_diff"`,
            `"hodges_lehmann"`, `"rank_biserial"`, and paired
            `"cohens_dz"`, `"hedges_gz"`, `"prop_diff"`, `"odds_ratio"`,
            or independent `"cles"`, `"cohens_d"`, `"hedges_g"`. It is the
            result's `statistic` and its only effect size.

        correction: str, optional
            Multiple-comparison correction across every pair in the
            result (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs
            (applied to the replicates too).

        offset: float, optional
            Test the estimator against this value instead of no difference
            (default 0), in the estimator's units -- e.g. `offset=-2,
            alternative="greater"` with `"mean_diff"` in mm asks "is
            group_a worse by less than 2 mm?" (non-inferiority). Not for
            `"cles"`/`"odds_ratio"`, which are tested against 0.5 / 1.

        alternative: str, optional
            `"two-sided"` | `"greater"` | `"less"` (the estimator vs.
            `offset`). Defaults to `"two-sided"` when `offset` is 0 and
            `"greater"` otherwise.

        n_boot: int, optional
            Number of resamples (default 10,000). Not with `replicates`.

        seed: int, optional
            Seeds the resampling. Drawn at random when omitted; stored on
            the result either way (`result.bootstrap["seed"]`). Not with
            `replicates`.

        replicates: str, optional
            The registered metric holding one value per bootstrap draw --
            see technical details.

        replicate_col: str, optional
            With `replicates`: the column numbering the draws.

        zero_method: str, optional
            Paired `"rank_biserial"` only -- see `suite.stat.wilcoxon`.

        ci: bool, optional
            Add the estimator's percentile interval from the same draws
            (default True), as `<estimator>_ci_low`/`_ci_high` columns --
            the interval `suite.estimate.from_result` gives.

        ci_level: float, optional
            The interval's coverage (default 0.95).

        ci_method: str, optional
            `"percentile"` (default), `"basic"` or `"bca"` -- how the
            interval and the p-value are read from the draws (see
            technical details). Recorded in `result.ci_methods` and
            `result.bootstrap["interval"]`.

        acceleration: float, optional
            `"bca"` only: the acceleration to use instead of a jackknife
            (0 for the bias-corrected "BC" interval).

        jackknife: str, optional
            `"bca"` with `replicates` only: the registered metric holding
            the leave-one-out values -- see technical details.

        jackknife_col: str, optional
            With `jackknife`: the column numbering the left-out units.

        Returns
        -------
        PairwiseResult
            `test == "bootstrap"`; `result.bootstrap` holds the settings
            and `table["n_boot"]` the number of valid draws per pair.
        """
        bootstrap = resolve_bootstrap(
            "bootstrap", n_boot, seed, replicates, replicate_col, ci_method, acceleration, jackknife, jackknife_col,
        )
        bootstrap["estimator"] = estimator
        df = self._suite.to_df(metrics=bootstrap_metrics(metric, bootstrap), filters=filters)
        if df.empty or not (df["Metric"] == metric).any():
            raise ValueError(f"No data registered for metric '{metric}'.")
        paired_rank_biserial = estimator == "rank_biserial" and pairing_col is not None
        result = PairwiseResult.compute(
            df, metric=metric, group_by=group_by, comparisons=[tuple(pair) for pair in comparisons], test="bootstrap",
            unit_col=unit_col, pairing_col=pairing_col, filters=_freeze_filters(filters), offset=offset,
            alternative=alternative, zero_method=zero_method if paired_rank_biserial else None, bootstrap=bootstrap,
            ci=ci, ci_level=ci_level,
        )
        if ci_method == "percentile":
            biased = result.table.loc[result.table["bias_se_ratio"].abs() > BIAS_SE_WARNING]
            if not biased.empty:
                pairs = ", ".join(
                    f"{a} vs {b} ({r:+.2f} SE)" for a, b, r in zip(biased["group_a"], biased["group_b"], biased["bias_se_ratio"])
                )
                warnings.warn(
                    f"The bootstrap draws are biased relative to the estimate by more than {BIAS_SE_WARNING} SE for "
                    f"{pairs}: the percentile interval and p-value are pulled the way the draws lean -- consider "
                    "ci_method='basic' or 'bca'.", UserWarning, stacklevel=2,
                )
        return result.correct(correction, alpha)

    def paired_differences(self, result, reference=None, higher_is_better=None):
        """
        The per-unit paired differences `reference - competitor` behind a
        paired test -- e.g. to plot or summarize what each test actually
        compared. For the plot, see `suite.plot.paired_differences`. The
        shape follows the result:

        - `PairwiseResult` (`wilcoxon`/`mcnemar`/`paired_ttest`/
          `ttest_ess`): one column per competitor, one row per
          `pairing_col` value;
        - `ClusterResult`: a long table, one row per unit, coordinate and
          competitor -- columns `pairing_col`, `coord`, `group_by` (the
          competitor) and "Value".

        !!! note "Technical details"
            Pairwise: rebuilt from the suite's data with exactly the
            test's own filters, `unit_col` collapsing, `subject_order`
            reindexing and per-pair complete-case dropping (shared code
            with `PairwiseResult.compute`), so each column has exactly that
            pair's `n_pairs` non-missing values. Rows are the union of
            every competitor's complete cases, in the test's
            `pairing_col` order (`subject_order` when the test had one);
            a unit missing for one competitor only is `NaN` in that
            column. Only paired results have per-unit differences. The
            differences are raw -- not shifted by `result.offset` -- so
            they read in the metric's own units;
            `result.vs(reference)["offset"]` gives the offset in the same
            orientation.

            Cluster: the very unit x coordinate matrix each permutation
            test ran on (`result.diff_by_pair`), re-oriented as
            `reference - competitor`.

            Read-only: the test is neither recomputed nor modified.
            `reference` is inferred like `suite.plot.group_comparison`
            does when every pair shares one common group. Raises if
            `result` doesn't come from this suite's current data (see
            `MetricSuite.verify`).

        Parameters
        ----------
        result: PairwiseResult | ClusterResult
            A paired result, e.g. from `suite.stat.wilcoxon(...)` or
            `suite.stat.cluster_permutation(...)`.

        reference: str, optional
            The group subtracted from. Inferred from `result` when every
            pair shares one common group; required otherwise.

        higher_is_better: bool, optional
            Whether a larger metric value is better. When False,
            differences are negated (`competitor - reference`), so
            positive always means `reference` better, matching
            `PairwiseResult.vs`. Default: True for a pairwise result, the
            test's own setting for a cluster result.

        Returns
        -------
        pd.DataFrame
            Pairwise: index = `pairing_col` values, columns = competitors
            (in `result.vs(reference)` order). Cluster: the long table
            above, competitors in the result's pair order.
        """
        if isinstance(result, ClusterResult):
            return self._cluster_differences(result, reference, higher_is_better)
        if not isinstance(result, PairwiseResult):
            raise TypeError(
                f"suite.stat.paired_differences() expects a PairwiseResult (from suite.stat.wilcoxon(...)/"
                f".mcnemar(...)/.paired_ttest(...)/.ttest_ess(...)) or a ClusterResult, got {type(result).__name__}."
            )
        higher_is_better = True if higher_is_better is None else higher_is_better
        if result.pairing_col is None:
            raise ValueError(
                f"suite.stat.paired_differences() needs a paired result, but test={result.test!r} is an "
                f"independent-samples test (no `pairing_col`)."
            )
        if not isinstance(higher_is_better, bool):
            raise TypeError(f"`higher_is_better` must be a bool, got {type(higher_is_better).__name__}.")

        if reference is None:
            reference = _infer_reference(list(zip(result.table["group_a"], result.table["group_b"])))
        if reference is None:
            raise ValueError(
                "`reference` is required -- this result's pairs don't share one common group "
                "for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
            )
        competitors = list(result.vs(reference).index)

        self._suite._require_origin(result)
        df = self._suite.to_df(metrics=result.metric, filters=_filters_to_dict(result.filters))
        if df.empty:
            raise ValueError(f"No data registered for metric '{result.metric}'.")
        wide = _paired_wide(df, result.group_by, result.pairing_col, result.unit_col, result.subject_order)

        sign = 1.0 if higher_is_better else -1.0
        columns = {}
        for c in competitors:
            complete = wide[[reference, c]].dropna()
            columns[c] = sign * (complete[reference] - complete[c])
        diffs = pd.DataFrame(columns, index=wide.index, columns=competitors)
        diffs = diffs.dropna(how="all")
        diffs.columns.name = result.group_by
        return diffs

    def _cluster_differences(self, result, reference, higher_is_better):
        """`paired_differences` for a `ClusterResult` -- see there."""
        if higher_is_better is None:
            higher_is_better = result.higher_is_better
        if not isinstance(higher_is_better, bool):
            raise TypeError(f"`higher_is_better` must be a bool, got {type(higher_is_better).__name__}.")
        pairs = list(result.clusters_by_pair)
        if reference is None:
            reference = _infer_reference(pairs)
        if reference is None:
            raise ValueError(
                "`reference` is required -- this result's pairs don't share one common group "
                "for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
            )
        self._suite._require_origin(result)

        frames = []
        for a, b in pairs:
            if reference not in (a, b):
                continue
            # diff_by_pair holds sign * (a - b), sign = +1 when the test's higher_is_better is True.
            factor = (1.0 if reference == a else -1.0) * (1.0 if higher_is_better == result.higher_is_better else -1.0)
            wide = factor * result.diff_by_pair[(a, b)]
            long = wide.rename_axis(index=result.pairing_col, columns=result.coord).stack().rename("Value").reset_index()
            long.insert(2, result.group_by, b if reference == a else a)
            frames.append(long)
        if not frames:
            groups = sorted({g for pair in pairs for g in pair}, key=str)
            raise ValueError(f"reference {reference!r} appears in no pair of this result (groups compared: {groups}).")
        return pd.concat(frames, ignore_index=True)

    def bootstrap_draws(self, result, reference=None, higher_is_better=True):
        """
        The bootstrap distribution behind a `suite.stat.bootstrap` result:
        the estimator's value on every draw, `reference` minus each
        competitor -- e.g. to plot the sampling distribution of the
        estimate (see `suite.plot.paired_differences`).

        !!! note "Technical details"
            The very draws the test's p-value and interval came from:
            registered replicates are read back from the suite and paired
            per draw like the observed data, resampled ones are redrawn
            from the result's own `seed` (shared code with
            `PairwiseResult.compute`), oriented like `result.vs(reference,
            higher_is_better)`. For a `ci_method="basic"` result they are
            the reflected draws `2 * estimate - draw` the basic interval
            reads, so for the percentile and basic intervals the
            interval is exactly the 2.5-97.5% range of what is returned. A
            BCa interval reads other quantiles of the plain draws, which
            are returned as they are.

            Only for the unstandardized estimators, `mean_diff` and
            `hodges_lehmann`, whose orientation is a sign flip. Read-only;
            raises if `result` doesn't come from this suite's current data
            (see `MetricSuite.verify`).

        Parameters
        ----------
        result: PairwiseResult
            From `suite.stat.bootstrap(...)`.

        reference: str, optional
            The group subtracted from. Inferred when every pair shares one
            common group; required otherwise.

        higher_is_better: bool, optional
            When False, draws are negated, so positive always means
            `reference` better.

        Returns
        -------
        pd.DataFrame
            Long table, one row per draw and competitor -- columns
            `group_by` (the competitor, in `result.vs(reference)` order),
            "draw" (0..n-1) and "Value". NaN draws are dropped.
        """
        if not isinstance(result, PairwiseResult) or result.test != "bootstrap":
            raise TypeError("suite.stat.bootstrap_draws() expects a PairwiseResult from suite.stat.bootstrap(...).")
        if result.estimator not in ("mean_diff", "hodges_lehmann"):
            raise NotImplementedError(
                f"Bootstrap draws are only available for 'mean_diff' and 'hodges_lehmann', not {result.estimator!r}."
            )
        if not isinstance(higher_is_better, bool):
            raise TypeError(f"`higher_is_better` must be a bool, got {type(higher_is_better).__name__}.")
        pairs = list(zip(result.table["group_a"], result.table["group_b"]))
        if reference is None:
            reference = _infer_reference(pairs)
        if reference is None:
            raise ValueError(
                "`reference` is required -- this result's pairs don't share one common group "
                "for it to be inferred from (an arbitrary many-to-many `comparisons` result)."
            )
        competitors = list(result.vs(reference).index)

        self._suite._require_origin(result)
        boot = result.bootstrap
        df = self._suite.to_df(metrics=bootstrap_metrics(result.metric, boot), filters=_filters_to_dict(result.filters))
        observed, draws_df = _split_replicates(df, result.metric, boot)
        samples = _pair_samples(
            observed, result.group_by, result.pairing_col, result.unit_col, pairs, result.subject_order,
        )
        draws = comparison_draws(
            samples, draws_df, result.group_by, result.pairing_col, result.unit_col, pairs, result.subject_order,
            boot, result.estimator, result.zero_method or "wilcox",
        )

        if boot.get("interval") == "basic":
            draws = [reflect_draws(d, est) for d, est in zip(draws, result.table["statistic"])]
        sign = 1.0 if higher_is_better else -1.0
        by_competitor = {}
        for (a, b), d in zip(pairs, draws):
            if reference in (a, b):
                by_competitor[b if reference == a else a] = sign * (d if reference == a else -d)
        frames = []
        for c in competitors:
            d = by_competitor[c]
            keep = ~np.isnan(d)
            frames.append(pd.DataFrame({result.group_by: c, "draw": np.flatnonzero(keep), "Value": d[keep]}))
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Cluster permutation.
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coord_order(df, group_by, coord, groups):
        """Sorted intersection of `coord`'s values across every group involved -- resolved once per request, not per pair, so every pair shares one consistent coord_order."""
        common = None
        for g in groups:
            values = set(df.loc[df[group_by] == g, coord].unique())
            common = values if common is None else (common & values)
        return sorted(common or set())

    def _compare_cluster(
        self, metric, coord, group_by, comparisons, pairing_col, unit_col, higher_is_better,
        coord_order, cluster_alpha, correction, alpha, family, n_permutations, seed, effect_size_summary, filters,
    ):
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")

        if coord_order is None:
            groups_involved = {g for pair in comparisons for g in pair}
            coord_order = self._resolve_coord_order(df, group_by, coord, groups_involved)

        result = ClusterResult.compute(
            df, metric=metric, group_by=group_by, coord=coord, comparisons=comparisons,
            pairing_col=pairing_col, unit_col=unit_col, higher_is_better=higher_is_better, coord_order=coord_order,
            cluster_alpha=cluster_alpha, n_permutations=n_permutations, seed=seed,
            effect_size_summary=effect_size_summary, filters=_freeze_filters(filters),
        )
        return result.correct(correction, alpha, family=family)

    def cluster_permutation(
        self, metric, coord, group_by, comparisons, pairing_col, unit_col=None,
        higher_is_better=True, coord_order=None, cluster_alpha=0.05, correction="fdr_bh", alpha=0.05,
        n_permutations=3000, seed=0, effect_size_summary="cluster_mean", filters=None, family="cluster",
    ):
        """
        Tests *where* along a coordinate (e.g. SNR, time) groups differ
        significantly, matched by `pairing_col` -- e.g. "at which noise
        levels does wCSET beat CSET?" Pure computation -- for a plot, see
        `suite.plot.cluster_permutation`. This is the expensive test: save
        the result (`result.save`) rather than rerunning it.

        !!! note "Technical details"
            Wraps `mindmap.analysis.stats.cluster_permutation_mne`
            (unchanged) with a cluster-permutation test between groups
            along `coord`.

            Each pair gets its own, fully independent permutation test
            (`n_permutations` sign-flips) -- this is the expensive step.
            `unit_col`, when given, pre-collapses `df` to one value
            per `(group_by, pairing_col, coord)` before that call -- see
            `ClusterResult.compute`.

            `cluster_alpha` is a *test* parameter (changing it reruns the
            permutation test) -- it's the cluster-forming and per-pair
            significance threshold, distinct from `correction`/`alpha`
            below, which control the cheap cross-pair correction applied
            on top of the raw permutation results (`result.correct(...)`
            re-applies it without rerunning the permutations).

        Parameters
        ----------
        metric: str
            Which registered metric to compare.

        coord: str
            Coordinate column to sweep along (e.g. "snr").

        group_by: str
            Which registered column holds the groups being compared.

        comparisons: list[tuple[str, str]]
            Which `(reference, competitor)` pairs to test -- see
            `build_comparisons`.

        pairing_col: str
            The repeated-measures/matched column (e.g. "region") --
            required, since `cluster_permutation_mne` is inherently a
            paired test.

        unit_col: str, optional
            Dedup axis -- see technical details above.

        higher_is_better: bool, optional
            Sign convention for which direction counts as "reference
            wins" (default True). Forwarded to `cluster_permutation_mne`.

        coord_order: list, optional
            Explicit coordinate ordering. Defaults to the sorted
            intersection of `coord`'s values across every group involved,
            resolved once per call so every pair shares one consistent
            ordering.

        cluster_alpha: float, optional
            The permutation test's own significance threshold (default
            0.05).

        correction: str, optional
            Multiple-comparison correction across the pairs (default
            `"fdr_bh"`; e.g. `"holm"`), applied to the family `family` sets.

        alpha: float, optional
            Significance threshold for the correction above (default
            0.05).

        n_permutations, seed, effect_size_summary
            Forwarded to `cluster_permutation_mne` -- see its docstring.

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        family: str, optional
            What `correction` treats as one family: `"cluster"` (default,
            every cluster of every pair pooled) or `"pair"` (one p-value
            per pair, its smallest cluster p -- e.g. Holm across the
            pairs). See `ClusterResult.correct`; `result.pair_table` gives
            the per-pair verdict.

        Returns
        -------
        ClusterResult
        """
        return self._compare_cluster(
            metric, coord, group_by, comparisons, pairing_col, unit_col, higher_is_better, coord_order,
            cluster_alpha, correction, alpha, family, n_permutations, seed, effect_size_summary, filters,
        )

    # ------------------------------------------------------------------
    # Normality.
    # ------------------------------------------------------------------

    def normality_test(self, metric, group_by=None, test="shapiro", correction="fdr_bh", alpha=0.05, filters=None):
        """
        Tests whether `metric`'s distribution is close to normal/Gaussian --
        the whole population by default, or one test per group if
        `group_by` is given.

        !!! note "Technical details"
            The null hypothesis is normality, so a *small* p-value is
            evidence *against* it (unlike the pairwise tests, where a
            small p-value is evidence *for* a real difference) --
            `significant` here means "significantly non-normal".

            Each group is tested independently (exactly like each pair in
            `wilcoxon`/etc.); `correction` then treats all groups as one
            family.

        Parameters
        ----------
        metric: str
            Which registered metric to test.

        group_by: str, optional
            Which registered column to test one group at a time. Omit to
            test the whole population as a single group.

        test: str, optional
            Which normality test to run: `"shapiro"` (default) or
            `"normaltest"` (see `mindmap.core.stats.test_normality`).

        correction: str, optional
            Multiple-comparison correction across every group in the
            result (default `"fdr_bh"`) -- irrelevant (but harmless) for
            the single-row whole-population case.

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only test rows matching these `{column: value(s)}` pairs.

        Returns
        -------
        NormalityResult
        """
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        result = NormalityResult.compute(df, metric=metric, group_by=group_by, test=test, filters=_freeze_filters(filters))
        return result.correct(correction, alpha)

    # ------------------------------------------------------------------
    # ANOVA (omnibus).
    # ------------------------------------------------------------------

    def anova(self, metric, group_by, pairing_col, unit_col=None, test="friedman", groups=None, alpha=0.05, filters=None):
        """
        Tests whether three or more groups differ at all on one metric,
        matched by `pairing_col` -- e.g. "do any of these five methods
        differ?" The omnibus counterpart to the pairwise tests: run this
        first, and only chase pairwise comparisons (`wilcoxon`/
        `paired_ttest`/`ttest_ess`, FDR-corrected across them) if this
        comes back significant -- the standard "omnibus, then post-hoc"
        two-step, which avoids inflating the false-positive rate that
        running every pairwise test directly (with no omnibus gate) would
        risk.

        !!! note "Technical details"
            Needs at least 3 groups -- for exactly 2, this is exactly the
            same question `wilcoxon`/`paired_ttest`/`ttest_ess` already
            answer, use one of those instead. Matched on `pairing_col`,
            complete-case across *every* included group at once (a subject
            missing from any one of them is dropped from the whole test)
            -- unlike the pairwise tests, which drop a subject only from
            the one pair it's actually missing from, an omnibus test needs
            one consistent subject set shared by every group simultaneously
            to be a meaningful single test. `unit_col`, when given, is a
            dedup axis -- see `AnovaResult.compute`.

            Unlike the pairwise tests/`cluster_permutation`/
            `normality_test`, there's no multiple-comparison correction
            here -- this is already one single p-value, not a family of
            pairs/groups. `alpha` only (re-)sets `significant = p_value <
            alpha`.

        Parameters
        ----------
        metric: str
            Which registered metric to test.

        group_by: str
            Which registered column holds the groups being compared.

        pairing_col: str
            The repeated-measures/matched column (e.g. "region") --
            required, since an omnibus RM test is inherently a
            paired/matched-groups test.

        unit_col: str, optional
            Dedup axis.

        test: str, optional
            Which omnibus test to run: `"friedman"` (default,
            non-parametric) or `"anova"` (parametric, repeated-measures)
            -- see `mindmap.core.stats.repeated_measures_anova`.

        groups: list, optional
            Which groups to include. Defaults to every group present for
            `metric`.

        alpha: float, optional
            Significance threshold (default 0.05).

        filters: dict, optional
            Only test rows matching these `{column: value(s)}` pairs.

        Returns
        -------
        AnovaResult
        """
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        result = AnovaResult.compute(
            df, metric=metric, group_by=group_by, pairing_col=pairing_col, unit_col=unit_col,
            test=test, groups=groups, filters=_freeze_filters(filters),
        )
        return result.correct(alpha)
