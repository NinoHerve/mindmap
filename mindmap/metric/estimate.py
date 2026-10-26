"""
Module Name: metric/estimate.py
Description: `MetricSuite.estimate`'s accessor -- point estimates, with
optional confidence intervals, each returned as an `EstimateResult`: the
mean or median of each group, or any difference / effect size between two
groups (`mean_diff`, `hodges_lehmann`, and every standardized effect size
the tests report). The counterpart of `MetricSuite.stat`: a test answers
"is there a difference (beyond an offset)?", an estimate answers "how
large is it?".

Author: Nino Herve
"""

from mindmap.metric.results import PairwiseResult, EstimateResult, _freeze_filters, resolve_bootstrap, bootstrap_metrics
from mindmap.metric.plot import _filters_to_dict


class EstimateAccessor:
    """
    `MetricSuite.estimate`'s namespace -- bound to one suite, accessed as
    `suite.estimate.median(...)`, `suite.estimate.hodges_lehmann(...)`,
    etc. Every method returns an `EstimateResult`.

    !!! note "Technical details"
        Two families:

        - per group -- `mean`, `median`, `proportion` (0/1 outcome): one estimate per group of
          `group_by` (or of the whole population if `group_by` is
          omitted);
        - per pair -- `mean_diff`, `hodges_lehmann` (unstandardized, in the
          metric's units) and `rank_biserial`, `cohens_dz`, `hedges_gz`,
          `cles`, `cohens_d`, `hedges_g` (standardized), and for a binary
          0/1 outcome `prop_diff` and `odds_ratio` (paired): one estimate of
          `group_a` relative to `group_b` per pair of `comparisons` --
          paired when `pairing_col` is given (matched exactly like
          `suite.stat.wilcoxon` matches), independent otherwise.

        Every effect size a test can report is here, computed by the same
        function, so the two always agree. `ci=True` (default) adds a
        `1 - alpha` interval, built as `ci_method` says:

        - `"analytic"` (default): the estimator's own interval (t,
          binomial, Wilson, signed-rank, Newcombe, ...); raises
          `NotImplementedError` for the standardized effect sizes, which
          have none here;
        - `"bootstrap"`: the percentile bootstrap interval, for every
          estimator -- resampling the units the estimate reads (`n_boot`
          resamples, seeded by `seed`), or reading draws registered
          upstream (`replicates`/`replicate_col`, e.g. one value per
          spike resample).

        `ci=False` gives the point value alone.

        The tests already report their effect sizes with intervals (see
        `suite.stat.wilcoxon`'s `ci`), computed by these same functions --
        use this accessor for what a test doesn't give: per-group
        summaries, another estimator or interval, or an `EstimateResult`
        to draw with `suite.plot.estimate`. `from_result` builds one for
        the differences behind an existing test, with that test's own
        pairing, filters and units.
    """

    def __init__(self, suite):
        self._suite = suite

    def _compute(
        self, metric, estimator, group_by=None, comparisons=None, pairing_col=None, unit_col=None, subject_order=None,
        alpha=0.05, ci=True, ess=False, filters=None, zero_method=None, ci_method="analytic", n_boot=None, seed=None,
        replicates=None, replicate_col=None, interval="percentile", acceleration=None, jackknife=None, jackknife_col=None,
    ):
        bootstrap = resolve_bootstrap(
            ci_method, n_boot, seed, replicates, replicate_col, interval, acceleration, jackknife, jackknife_col,
        )
        df = self._suite.to_df(metrics=bootstrap_metrics(metric, bootstrap), filters=filters)
        if df.empty or not (df["Metric"] == metric).any():
            raise ValueError(f"No data registered for metric '{metric}'.")
        return EstimateResult.compute(
            df, metric=metric, estimator=estimator, group_by=group_by,
            comparisons=[tuple(pair) for pair in comparisons] if comparisons is not None else None,
            pairing_col=pairing_col, unit_col=unit_col, subject_order=subject_order, alpha=alpha, ci=ci, ess=ess,
            filters=_freeze_filters(filters), zero_method=zero_method, bootstrap=bootstrap,
        )

    # -- Per group --------------------------------------------------------

    def mean(
        self, metric, group_by=None, pairing_col=None, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Mean of each group, with a t confidence interval.

        Parameters
        ----------
        metric: str
            Which registered metric to estimate.

        group_by: str, optional
            Which registered column holds the groups. Omit for the whole
            population.

        pairing_col: str, optional
            Collapse to one value per `(group_by, pairing_col)` first,
            exactly like a paired test does (e.g. one value per subject).

        unit_col: str, optional
            Dedup axis, collapsed exactly like the tests do.

        alpha: float, optional
            The interval is `1 - alpha` (default 0.05, i.e. 95%).

        ci: bool, optional
            Whether to compute the interval (default True).

        filters: dict, optional
            Only use rows matching these `{column: value(s)}` pairs.

        ci_method: str, optional
            `"analytic"` (default -- the interval above) or `"bootstrap"`
            (the percentile bootstrap interval, see
            `mindmap.core.stats.percentile_interval`).

        n_boot: int, optional
            Bootstrap only: number of resamples (default 10,000).

        seed: int, optional
            Bootstrap only: seeds the resampling. Drawn at random when
            omitted; either way it is stored on the result, so the same
            draws can be reproduced (e.g. by `suite.stat.bootstrap`).

        replicates: str, optional
            Bootstrap only: read the draws from this registered metric
            instead of resampling -- for resamples made upstream (e.g.
            spikes re-averaged per draw): one value per draw, with the same
            `group_by`/`pairing_col` columns as `metric`. Draws are matched
            across groups by `replicate_col`, so draw 3 of every group must
            come from the same resample. `metric` itself holds the observed
            (full-sample) values, which give the point estimate.

        replicate_col: str, optional
            With `replicates`: the column numbering the draws (e.g. "boot").

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "mean", group_by, pairing_col=pairing_col, unit_col=unit_col, alpha=alpha, ci=ci, filters=filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def median(
        self, metric, group_by=None, pairing_col=None, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Median of each group, with the exact, distribution-free binomial
        interval.

        Parameters
        ----------
        metric, group_by, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "median", group_by, pairing_col=pairing_col, unit_col=unit_col, alpha=alpha, ci=ci, filters=filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def proportion(
        self, metric, group_by=None, pairing_col=None, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Share of 1s of a binary 0/1 outcome in each group (e.g. the share
        of patients with the peak inside the resection), with the Wilson
        score interval -- which stays within [0, 1] and keeps close to
        nominal coverage at small n, unlike the t interval of the mean.

        Parameters
        ----------
        metric, group_by, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "proportion", group_by, pairing_col=pairing_col, unit_col=unit_col, alpha=alpha, ci=ci, filters=filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    # -- Per pair: unstandardized -----------------------------------------

    def mean_diff(
        self, metric, group_by, comparisons, pairing_col=None, unit_col=None, alpha=0.05, ci=True, filters=None,
        ess=False, subject_order=None, ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Mean difference `group_a - group_b` per pair -- the unstandardized
        effect size that goes with the t-tests: the mean of the paired
        differences with a t interval (`pairing_col` given), or the
        difference of means with a Welch interval.

        Parameters
        ----------
        metric: str
            Which registered metric to estimate.

        group_by: str
            Which registered column holds the groups.

        comparisons: list[tuple[str, str]]
            The `(group_a, group_b)` pairs -- see
            `suite.stat.build_comparisons`.

        pairing_col: str, optional
            Match pairs on this column (e.g. "region"); omit for
            independent samples.

        unit_col: str, optional
            Dedup axis, collapsed exactly like the tests do.

        alpha: float, optional
            The interval is `1 - alpha` (default 0.05, i.e. 95%).

        ci: bool, optional
            Whether to compute the interval (default True).

        filters: dict, optional
            Only use rows matching these `{column: value(s)}` pairs.

        ess: bool, optional
            Paired only: use each pair's effective sample size in the
            interval, as `suite.stat.ttest_ess` does -- for serially
            correlated units (default False).

        subject_order: list, optional
            The true order of `pairing_col`'s values, needed by `ess`.

        ci_method, n_boot, seed, replicates, replicate_col
            As in `mean` -- a paired bootstrap resamples whole pairs, an
            independent one each group on its own. Not with `ess`: the
            bootstrap resamples units independently.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "mean_diff", group_by, comparisons, pairing_col, unit_col, subject_order, alpha, ci, ess, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def hodges_lehmann(
        self, metric, group_by, comparisons, pairing_col=None, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Hodges-Lehmann estimate of `group_a - group_b` per pair, with its
        distribution-free interval -- the unstandardized effect size that
        goes with the Wilcoxon signed-rank (paired: median of the Walsh
        averages of the differences) and Mann-Whitney (independent: median
        of every pairwise difference) tests.

        !!! note "Technical details"
            Robust to outliers like the median but more efficient
            (close to the mean's efficiency for normal data). The
            interval assumes independent units: for serially correlated
            units (the `ttest_ess` use case) it is too narrow -- use
            `mean_diff(..., ess=True)` there. Memory is O(n^2) -- see
            `mindmap.core.stats.estimate`.

        Parameters
        ----------
        metric, group_by, comparisons, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "hodges_lehmann", group_by, comparisons, pairing_col, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    # -- Per pair: binary 0/1 outcome (McNemar) -----------------------------

    def prop_diff(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Difference in the proportion of 1s, `group_a - group_b`, per pair
        of a binary 0/1 outcome (paired only), with Newcombe's interval --
        the unstandardized effect size that goes with `suite.stat.mcnemar`
        (e.g. "+0.19 = 19 percentage points more patients with the peak
        inside the resection").

        !!! note "Technical details"
            Equals `(n_a_only - n_b_only) / n`: concordant pairs count in
            n, so they shrink the difference. The interval is Newcombe's
            (1998, method 10) hybrid score interval for paired
            proportions: a Wilson interval per proportion, combined with
            their correlation -- it stays within [-1, 1] and keeps close
            to nominal coverage at small n, unlike the simple Wald
            interval. The metric must hold 0/1 values.

        Parameters
        ----------
        metric, group_by, comparisons, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff` (`pairing_col` required).

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "prop_diff", group_by, comparisons, pairing_col, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def odds_ratio(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Conditional (McNemar) odds ratio per pair of a binary 0/1 outcome
        (paired only), `n_a_only / n_b_only`, with its exact interval --
        "among the pairs where the two disagree, group_a reaches the
        outcome this many times as often". 1 means no difference.

        !!! note "Technical details"
            Only discordant pairs count. The interval is the exact
            (Clopper-Pearson) interval of `n_a_only / (n_a_only +
            n_b_only)`, mapped through `p / (1 - p)`. `inf` when
            `n_b_only` is 0, NaN without discordant pairs. Read it on a
            log scale: an odds ratio of 2 and one of 0.5 are equally
            large effects in opposite directions.

        Parameters
        ----------
        metric, group_by, comparisons, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff` (`pairing_col` required).

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "odds_ratio", group_by, comparisons, pairing_col, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    # -- Per pair: standardized (bootstrap interval only) -----------------

    def rank_biserial(
        self, metric, group_by, comparisons, pairing_col=None, unit_col=None, alpha=0.05, ci=True, filters=None,
        zero_method="wilcox", ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Rank-biserial correlation per pair -- matched-pairs (Kerby, 2014)
        with `pairing_col`, Mann-Whitney's `2U / (n_a * n_b) - 1` without.
        No analytic interval: pass `ci_method="bootstrap"` or `ci=False`.

        Parameters
        ----------
        metric, group_by, comparisons, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff`.

        zero_method: str, optional
            Matched pairs only: `"wilcox"` (default -- zero differences
            dropped), `"pratt"` or `"zsplit"` (zeros ranked, their ranks
            kept in the denominator, so ties pull r toward 0) -- use the
            Wilcoxon test's, see `suite.stat.wilcoxon`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "rank_biserial", group_by, comparisons, pairing_col, unit_col, None, alpha, ci, False, filters,
            zero_method, ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def cohens_dz(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Cohen's dz per pair (paired only): mean of the differences over
        their standard deviation. No analytic interval: pass
        `ci_method="bootstrap"` or `ci=False`.

        Parameters
        ----------
        metric, group_by, comparisons, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff` (`pairing_col` required).

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "cohens_dz", group_by, comparisons, pairing_col, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def hedges_gz(
        self, metric, group_by, comparisons, pairing_col, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Hedges' gz per pair (paired only): Cohen's dz with the small-sample
        bias correction. No analytic interval: pass `ci_method="bootstrap"` or `ci=False`.

        Parameters
        ----------
        metric, group_by, comparisons, pairing_col, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff` (`pairing_col` required).

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "hedges_gz", group_by, comparisons, pairing_col, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def cohens_d(
        self, metric, group_by, comparisons, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Cohen's d per pair (independent samples): difference of means over
        the pooled standard deviation. No analytic interval: pass
        `ci_method="bootstrap"` or `ci=False`.

        Parameters
        ----------
        metric, group_by, comparisons, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "cohens_d", group_by, comparisons, None, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def hedges_g(
        self, metric, group_by, comparisons, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Hedges' g per pair (independent samples): Cohen's d with the
        small-sample bias correction. No analytic interval: pass
        `ci_method="bootstrap"` or `ci=False`.

        Parameters
        ----------
        metric, group_by, comparisons, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "hedges_g", group_by, comparisons, None, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    def cles(
        self, metric, group_by, comparisons, unit_col=None, alpha=0.05, ci=True, filters=None,
        ci_method="analytic", n_boot=None, seed=None, replicates=None, replicate_col=None,
    ):
        """
        Common-language effect size per pair (independent samples):
        `P(a > b) + P(a = b) / 2`, 0.5 meaning no effect. No analytic
        interval: pass `ci_method="bootstrap"` or `ci=False`.

        Parameters
        ----------
        metric, group_by, comparisons, unit_col, alpha, ci, filters, ci_method, n_boot, seed, replicates, replicate_col
            As in `mean_diff`.

        Returns
        -------
        EstimateResult
        """
        return self._compute(
            metric, "cles", group_by, comparisons, None, unit_col, None, alpha, ci, False, filters,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col,
        )

    # -- From a test ------------------------------------------------------

    def from_result(self, result, estimator=None, alpha=0.05, ci=True, ci_method=None, n_boot=None, seed=None):
        """
        An `EstimateResult` for the differences behind an already-computed
        test -- e.g. to draw them with `suite.plot.estimate`, or to estimate
        an effect the test didn't report. (The test's own effect sizes
        already carry their intervals in its table.) Covers every
        `(group_a, group_b)` pair of `result`, with exactly its metric,
        filters, `pairing_col`, `unit_col` and `subject_order` -- so the
        estimate describes the very units the test saw.

        !!! note "Technical details"
            Raises if `result` doesn't come from this suite's current data
            (see `MetricSuite.verify`). The estimate is on the raw
            difference, never shifted by `result.offset` -- compare its
            interval with the offset directly. A `ttest_ess` result's
            effective sample size carries over (`ess=True`), which only
            `mean_diff`'s analytic interval supports. A paired
            rank-biserial uses the Wilcoxon result's `zero_method`. With
            `zero_method="wilcox"`, Wilcoxon drops differences exactly at
            the offset from its `n_pairs`; the estimate keeps them, so its
            `n_a` can be larger.

            A `suite.stat.bootstrap` result's interval is a bootstrap one
            by default, with the test's own settings: the same seed (so
            the very draws behind its p-value) or the same registered
            replicates, read the same way (`ci_method="percentile"`,
            `"basic"` or `"bca"`, with its acceleration or jackknife).

        Parameters
        ----------
        result: PairwiseResult
            From `suite.stat.wilcoxon(...)`, `.ttest_ess(...)`,
            `.bootstrap(...)`, etc.

        estimator: str, optional
            Any per-pair estimator (see the class docstring). Defaults to
            `result.estimator`: the test's unstandardized effect size
            (`"hodges_lehmann"` for wilcoxon/mannwhitneyu, `"mean_diff"`
            for the t-tests, `"prop_diff"` for mcnemar), or the estimator
            a bootstrap test bootstrapped.

        alpha: float, optional
            The interval is `1 - alpha` (default 0.05).

        ci: bool, optional
            Whether to compute the interval (default True).

        ci_method: str, optional
            `"analytic"` or `"bootstrap"`. Defaults to `"bootstrap"` for a
            bootstrap test and `"analytic"` otherwise.

        n_boot, seed: int, optional
            Bootstrap only, as in `mean`. On a bootstrap test they default
            to the test's own.

        Returns
        -------
        EstimateResult
        """
        if not isinstance(result, PairwiseResult):
            raise TypeError(f"suite.estimate.from_result() expects a PairwiseResult, got {type(result).__name__}.")
        self._suite._require_origin(result)
        if estimator is None:
            estimator = result.estimator
        boot = result.bootstrap
        if ci_method is None:
            ci_method = "bootstrap" if boot is not None and ci else "analytic"
        replicates = replicate_col = None
        settings = {}
        if ci_method == "bootstrap" and boot is not None and n_boot is None and seed is None:
            n_boot, seed, replicates, replicate_col = boot["n_boot"], boot["seed"], boot["replicates"], boot["replicate_col"]
            settings = {k: boot.get(k) for k in ("acceleration", "jackknife", "jackknife_col")}
            settings["interval"] = boot.get("interval", "percentile")
        comparisons = list(zip(result.table["group_a"], result.table["group_b"]))
        return self._compute(
            result.metric, estimator, result.group_by, comparisons, result.pairing_col, result.unit_col,
            result.subject_order, alpha, ci, result.test == "ttest_ess",
            _filters_to_dict(result.filters),
            result.zero_method if estimator == "rank_biserial" and result.pairing_col is not None else None,
            ci_method=ci_method, n_boot=n_boot, seed=seed, replicates=replicates, replicate_col=replicate_col, **settings,
        )
