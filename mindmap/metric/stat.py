"""
Module Name: metric/stat.py
Description: `MetricSuite.stat`'s accessor -- one method per statistical
test, bound to a `MetricSuite` instance, each pure computation returning a
cached, typed result (`PairwiseResult`/`ClusterResult`/`NormalityResult`/
`AnovaResult`). Separate from `MetricSuite.plot` -- `suite.stat.X` computes,
`suite.plot.X` (when it exists for `X`) visualizes an already-computed
result.

Author: Nino Herve
"""

import pandas as pd

from mindmap.metric.results import (
    PairwiseResult, ClusterResult, NormalityResult, AnovaResult, _freeze_filter, _freeze_filters, _paired_wide,
)
from mindmap.metric.plot import _infer_reference, _filters_to_dict


class StatAccessor:
    """
    `MetricSuite.stat`'s namespace -- bound to one suite, accessed as
    `suite.stat.wilcoxon(...)`, `suite.stat.cluster_permutation(...)`,
    `suite.stat.normality_test(...)`, `suite.stat.anova(...)`. Every method
    here is pure computation (no plotting) and caches its results -- see
    each method's own docstring, and `suite.plot.group_comparison`/
    `.cluster_permutation`/`.normality_test`/`.anova` for visualizing what
    comes back.

    !!! note "Technical details"
        Owns the suite's `_stats_cache` (not `MetricSuite` itself) --
        `register()` clears it via `clear_stats_cache()` whenever new data
        is added, and `save()`/`load()` persist/restore it directly.

        Five methods cover pairwise comparisons -- `wilcoxon`,
        `paired_ttest`, `ttest_ess` (paired/matched, need `pairing_col`),
        `ttest`, `mannwhitneyu` (independent-samples, no `pairing_col`) --
        all returning `PairwiseResult`. `build_comparisons` is a plain
        helper for assembling the `comparisons` list any of them take.
    """

    def __init__(self, suite):
        self._suite = suite
        self._stats_cache = {}

    # ------------------------------------------------------------------
    # Pairwise comparisons: wilcoxon / ttest / mannwhitneyu / paired_ttest / ttest_ess.
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

    @staticmethod
    def _pairwise_key(metric, group_by, test, unit_col, pairing_col, subject_order, filters, pair):
        return ("pairwise", metric, group_by, test, unit_col, pairing_col, subject_order, filters, pair)

    def _get_or_compute_pairwise(self, df, metric, group_by, test, unit_col, pairing_col, comparisons, subject_order, filters):
        frozen_order = _freeze_filter(subject_order)
        frozen_filters = _freeze_filters(filters)

        results = []
        for pair in comparisons:
            key = self._pairwise_key(metric, group_by, test, unit_col, pairing_col, frozen_order, frozen_filters, pair)
            cached = self._stats_cache.get(key)
            if cached is not None:
                results.append(cached)
                continue
            single = PairwiseResult.compute(
                df, metric=metric, group_by=group_by, comparisons=[pair], test=test,
                unit_col=unit_col, pairing_col=pairing_col, subject_order=subject_order, filters=frozen_filters,
            )
            self._stats_cache[key] = single
            results.append(single)

        return PairwiseResult.merge(results)

    def _pairwise(self, metric, group_by, comparisons, test, correction, alpha, unit_col, pairing_col, subject_order, filters):
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        merged = self._get_or_compute_pairwise(df, metric, group_by, test, unit_col, pairing_col, comparisons, subject_order, filters)
        return merged.correct(correction, alpha)

    def wilcoxon(self, metric, group_by, comparisons, pairing_col, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None):
        """
        Wilcoxon signed-rank test -- the non-parametric paired comparison,
        matched by `pairing_col` (e.g. "is wCSET better than CSET,
        region-by-region?"). Results are cached, so asking the same
        question again (or just adjusting the correction/alpha) is
        instant. Pure computation -- for a plot, see
        `suite.plot.group_comparison`.

        !!! note "Technical details"
            Each pair is tested on its own complete-case subjects,
            independent of any other pair requested in the same call --
            this is what makes individual pairs cacheable and mergeable
            (`PairwiseResult.merge`) across separate calls. `unit_col`,
            when given, is a dedup axis: rows are first collapsed to one
            value per `(group_by, pairing_col, unit_col)`, then to one per
            `(group_by, pairing_col)`, before testing -- so e.g. several
            repeated runs from one region don't get treated as several
            independent pieces of evidence. `group_by`/`filters` are part
            of this call's cache identity -- a different `group_by` or
            `filters` never reuses a cached result computed under a
            different one, but `correction`/`alpha` are always re-applied
            fresh (cheap) on top of whatever's cached.

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

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "wilcoxon", correction, alpha, unit_col, pairing_col, None, filters)

    def paired_ttest(self, metric, group_by, comparisons, pairing_col, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None):
        """
        Paired (related-samples) t-test, matched by `pairing_col` --
        the parametric analog of `wilcoxon`. Assumes the paired
        differences are roughly normal (see `normality_test`). Results are
        cached, so asking the same question again (or just adjusting the
        correction/alpha) is instant. Pure computation -- for a plot, see
        `suite.plot.group_comparison`.

        !!! note "Technical details"
            Same caching/collapse behavior as `wilcoxon` -- see its
            technical details.

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

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "paired_ttest", correction, alpha, unit_col, pairing_col, None, filters)

    def ttest_ess(self, metric, group_by, comparisons, pairing_col, unit_col=None, subject_order=None, correction="fdr_bh", alpha=0.05, filters=None):
        """
        Paired t-test with an autocorrelation-corrected effective sample
        size -- for when `pairing_col` has real serial/spatial correlation
        (e.g. adjacent timepoints or channels), which would otherwise
        overstate independent evidence if treated at face-value `n`.
        Results are cached, so asking the same question again (or just
        adjusting the correction/alpha) is instant. Pure computation --
        for a plot, see `suite.plot.group_comparison`.

        !!! note "Technical details"
            Same caching/collapse behavior as `wilcoxon` -- see its
            technical details. `n_eff` is computed via
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

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "ttest_ess", correction, alpha, unit_col, pairing_col, subject_order, filters)

    def ttest(self, metric, group_by, comparisons, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None):
        """
        Independent-samples t-test -- no correspondence between groups is
        assumed (unlike `paired_ttest`). Results are cached, so asking the
        same question again (or just adjusting the correction/alpha) is
        instant. Pure computation -- for a plot, see
        `suite.plot.group_comparison`.

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

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "ttest", correction, alpha, unit_col, None, None, filters)

    def mannwhitneyu(self, metric, group_by, comparisons, unit_col=None, correction="fdr_bh", alpha=0.05, filters=None):
        """
        Mann-Whitney U test -- the non-parametric independent-samples
        comparison, the analog of `wilcoxon` for unmatched groups. Results
        are cached, so asking the same question again (or just adjusting
        the correction/alpha) is instant. Pure computation -- for a plot,
        see `suite.plot.group_comparison`.

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

        Returns
        -------
        PairwiseResult
        """
        return self._pairwise(metric, group_by, comparisons, "mannwhitneyu", correction, alpha, unit_col, None, None, filters)

    def paired_differences(self, result, reference=None, higher_is_better=True):
        """
        The per-unit paired differences `reference - competitor` behind a
        paired `PairwiseResult` -- one column per competitor, one row per
        `pairing_col` value -- e.g. to plot or summarize what each test
        actually compared. For the plot, see
        `suite.plot.paired_differences`.

        !!! note "Technical details"
            Rebuilt from the suite's data with exactly the test's own
            filters, `unit_col` collapsing, `subject_order` reindexing and
            per-pair complete-case dropping (shared code with
            `PairwiseResult.compute`), so each column has exactly that
            pair's `n_pairs` non-missing values. Rows are the union of
            every competitor's complete cases, in the test's
            `pairing_col` order (`subject_order` when the test had one);
            a unit missing for one competitor only is `NaN` in that
            column. Read-only: the cached test is neither recomputed nor
            modified. `reference` is inferred like
            `suite.plot.group_comparison` does when every pair shares one
            common group. Only paired results (`wilcoxon`/`paired_ttest`/
            `ttest_ess`) have per-unit differences.

        Parameters
        ----------
        result: PairwiseResult
            A paired result, e.g. from `suite.stat.wilcoxon(...)`.

        reference: str, optional
            The group subtracted from. Inferred from `result` when every
            pair shares one common group; required otherwise.

        higher_is_better: bool, optional
            Whether a larger metric value is better (default True). When
            False, differences are negated (`competitor - reference`), so
            positive always means `reference` better, matching
            `PairwiseResult.vs`.

        Returns
        -------
        pd.DataFrame
            Index = `pairing_col` values, columns = competitors (in
            `result.vs(reference)` order).
        """
        if not isinstance(result, PairwiseResult):
            raise TypeError(
                f"suite.stat.paired_differences() expects a PairwiseResult (from suite.stat.wilcoxon(...)/"
                f".paired_ttest(...)/.ttest_ess(...)), got {type(result).__name__}."
            )
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

    # ------------------------------------------------------------------
    # Cluster permutation.
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_key(metric, coord, pairing_col, unit_col, group_by, higher_is_better, coord_order, cluster_alpha, n_permutations, seed, effect_size_summary, filters, pair):
        return (
            "cluster", metric, coord, pairing_col, unit_col, group_by, higher_is_better, coord_order,
            cluster_alpha, n_permutations, seed, effect_size_summary, filters, pair,
        )

    @staticmethod
    def _resolve_coord_order(df, group_by, coord, groups):
        """Sorted intersection of `coord`'s values across every group involved -- resolved once per request, not per pair, so every pair cached from the same call shares one consistent coord_order."""
        common = None
        for g in groups:
            values = set(df.loc[df[group_by] == g, coord].unique())
            common = values if common is None else (common & values)
        return sorted(common or set())

    def _get_or_compute_cluster(self, df, metric, coord, group_by, pairing_col, unit_col, comparisons, higher_is_better, coord_order, cluster_alpha, n_permutations, seed, effect_size_summary, filters):
        frozen_coord_order = tuple(coord_order)
        frozen_filters = _freeze_filters(filters)

        results = []
        for pair in comparisons:
            key = self._cluster_key(
                metric, coord, pairing_col, unit_col, group_by, higher_is_better, frozen_coord_order,
                cluster_alpha, n_permutations, seed, effect_size_summary, frozen_filters, pair,
            )
            cached = self._stats_cache.get(key)
            if cached is not None:
                results.append(cached)
                continue
            single = ClusterResult.compute(
                df, metric=metric, group_by=group_by, coord=coord, comparisons=[pair],
                pairing_col=pairing_col, unit_col=unit_col, higher_is_better=higher_is_better, coord_order=coord_order,
                cluster_alpha=cluster_alpha, n_permutations=n_permutations, seed=seed,
                effect_size_summary=effect_size_summary, filters=frozen_filters,
            )
            self._stats_cache[key] = single
            results.append(single)

        return ClusterResult.merge(results)

    def _compare_cluster(
        self, metric, coord, group_by, comparisons, pairing_col, unit_col, higher_is_better,
        coord_order, cluster_alpha, correction, alpha, n_permutations, seed, effect_size_summary, filters,
    ):
        df = self._suite.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")

        if coord_order is None:
            groups_involved = {g for pair in comparisons for g in pair}
            coord_order = self._resolve_coord_order(df, group_by, coord, groups_involved)

        merged = self._get_or_compute_cluster(
            df, metric, coord, group_by, pairing_col, unit_col, comparisons, higher_is_better, coord_order,
            cluster_alpha, n_permutations, seed, effect_size_summary, filters,
        )
        return merged.correct(correction, alpha)

    def cluster_permutation(
        self, metric, coord, group_by, comparisons, pairing_col, unit_col=None,
        higher_is_better=True, coord_order=None, cluster_alpha=0.05, correction="fdr_bh", alpha=0.05,
        n_permutations=3000, seed=0, effect_size_summary="cluster_mean", filters=None,
    ):
        """
        Tests *where* along a coordinate (e.g. SNR, time) groups differ
        significantly, matched by `pairing_col` -- e.g. "at which noise
        levels does wCSET beat CSET?" Results are cached per pair, so this
        and the pairwise tests never repeat expensive work once it's been
        done. Pure computation -- for a plot, see
        `suite.plot.cluster_permutation`.

        !!! note "Technical details"
            Wraps `mindmap.analysis.stats.cluster_permutation_mne`
            (unchanged) with a cluster-permutation test between groups
            along `coord`. Pure compute, always cached per pair (see
            `list_cached_stats`).

            Each pair gets its own, fully independent permutation test
            (`n_permutations` sign-flips) -- this is the expensive step,
            cached per `(pair, params)` so a pair already computed
            (standalone, or as part of a different request) is never
            rerun. `unit_col`, when given, pre-collapses `df` to one value
            per `(group_by, pairing_col, coord)` before that call -- see
            `ClusterResult.compute`.

            `cluster_alpha` is a *test* parameter (changing it reruns the
            permutation test) -- it's the cluster-forming and per-pair
            significance threshold, distinct from `correction`/`alpha`
            below, which control the cheap, always-fresh cross-pair
            correction applied on top of the cached raw permutation
            results. `group_by`/`filters` are part of this call's cache
            identity.

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
            resolved once per call so every pair cached from one request
            shares one consistent ordering.

        cluster_alpha: float, optional
            The permutation test's own significance threshold (default
            0.05).

        correction: str, optional
            Multiple-comparison correction pooling every cluster's
            p-value across every pair in the result (default `"fdr_bh"`).

        alpha: float, optional
            Significance threshold for the correction above (default
            0.05).

        n_permutations, seed, effect_size_summary
            Forwarded to `cluster_permutation_mne` -- see its docstring.

        filters: dict, optional
            Only compare rows matching these `{column: value(s)}` pairs.

        Returns
        -------
        ClusterResult
        """
        return self._compare_cluster(
            metric, coord, group_by, comparisons, pairing_col, unit_col, higher_is_better, coord_order,
            cluster_alpha, correction, alpha, n_permutations, seed, effect_size_summary, filters,
        )

    # ------------------------------------------------------------------
    # Normality.
    # ------------------------------------------------------------------

    @staticmethod
    def _normality_key(metric, group_by, test, filters, group):
        return ("normality", metric, group_by, test, filters, group)

    def _get_or_compute_normality(self, df, metric, group_by, test, filters, groups):
        frozen_filters = _freeze_filters(filters)

        results = []
        for group in groups:
            key = self._normality_key(metric, group_by, test, frozen_filters, group)
            cached = self._stats_cache.get(key)
            if cached is not None:
                results.append(cached)
                continue
            single = NormalityResult.compute(
                df, metric=metric, group_by=group_by, test=test,
                groups=None if group_by is None else [group], filters=frozen_filters,
            )
            self._stats_cache[key] = single
            results.append(single)

        return NormalityResult.merge(results)

    def normality_test(self, metric, group_by=None, test="shapiro", correction="fdr_bh", alpha=0.05, filters=None):
        """
        Tests whether `metric`'s distribution is close to normal/Gaussian --
        the whole population by default, or one test per group if
        `group_by` is given. Results are cached, so asking the same
        question again (or just adjusting the correction/alpha) is
        instant.

        !!! note "Technical details"
            The null hypothesis is normality, so a *small* p-value is
            evidence *against* it (unlike the pairwise tests, where a
            small p-value is evidence *for* a real difference) --
            `significant` here means "significantly non-normal".

            Each group is tested independently and cached independently
            (exactly like each pair in `wilcoxon`/etc.), so requesting a
            superset of an already-tested set of groups only computes the
            new ones. `group_by`/`filters` are part of this call's cache
            identity -- a different `group_by` or `filters` never reuses a
            cached result computed under a different one, but
            `correction`/`alpha` are always re-applied fresh (cheap) on top
            of whatever's cached.

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
        groups = [None] if group_by is None else sorted(df[group_by].unique())
        merged = self._get_or_compute_normality(df, metric, group_by, test, filters, groups)
        return merged.correct(correction, alpha)

    # ------------------------------------------------------------------
    # ANOVA (omnibus).
    # ------------------------------------------------------------------

    @staticmethod
    def _anova_key(metric, group_by, pairing_col, unit_col, test, filters, groups):
        return ("anova", metric, group_by, pairing_col, unit_col, test, filters, groups)

    def _get_or_compute_anova(self, df, metric, group_by, pairing_col, unit_col, test, groups, filters):
        frozen_filters = _freeze_filters(filters)
        # Resolved up front (not left as `None`) so the cache key always matches
        # the resolved `result.groups` a reload would reconstruct it from -- see
        # AnovaResult.compute, which resolves the same way when groups is None.
        resolved_groups = tuple(sorted(groups)) if groups is not None else tuple(sorted(df[group_by].unique()))

        key = self._anova_key(metric, group_by, pairing_col, unit_col, test, frozen_filters, resolved_groups)
        cached = self._stats_cache.get(key)
        if cached is not None:
            return cached
        result = AnovaResult.compute(
            df, metric=metric, group_by=group_by, pairing_col=pairing_col, unit_col=unit_col,
            test=test, groups=groups, filters=frozen_filters,
        )
        self._stats_cache[key] = result
        return result

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
        risk. Results are cached, so asking the same question again (or
        just adjusting `alpha`) is instant.

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
            alpha`, cheaply, without rerunning the test. `group_by`/
            `pairing_col`/`unit_col`/`test`/`groups`/`filters` are part of
            this call's cache identity.

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
        result = self._get_or_compute_anova(df, metric, group_by, pairing_col, unit_col, test, groups, filters)
        return result.correct(alpha)

    # ------------------------------------------------------------------
    # Cache introspection.
    # ------------------------------------------------------------------

    def list_cached_stats(self):
        """
        Shows what's currently cached -- one row per cached result, most
        recent first, so it's clear exactly which metric/group/pair/method
        produced each one.

        Returns
        -------
        pd.DataFrame
            Columns: `"kind"` (`"pairwise"` | `"cluster"` | `"normality"` | `"anova"`), `"metric"`,
            `"group_by"`, `"pair"`, `"computed_at"`.
        """
        if not self._stats_cache:
            return pd.DataFrame(columns=["kind", "metric", "group_by", "pair", "computed_at"])
        rows = [
            {
                "kind": key[0], "metric": result.metric, "group_by": result.group_by,
                "pair": key[-1], "computed_at": result.computed_at,
            }
            for key, result in self._stats_cache.items()
        ]
        return pd.DataFrame(rows).sort_values("computed_at", ascending=False).reset_index(drop=True)

    def clear_stats_cache(self):
        """Manually drops every cached stats result. `register()` already does this automatically whenever new data is added."""
        self._stats_cache = {}
