"""
This module performs unit tests for MetricSuite.compare_methods and
MetricSuite.cluster_permutation (mindmap/metric/base.py), the MetricSuite-
level entry points for mindmap.analysis.stats.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mindmap.metric import MetricSuite


def _make_compare_suite(n_regions=20, seed=0):
    suite = MetricSuite(group_name="Method")
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_regions)]
    reference_vals = rng.normal(0, 1, size=n_regions)
    shifted_vals = reference_vals - 2.0 + rng.normal(0, 0.1, size=n_regions)
    noise_vals = reference_vals + rng.normal(0, 1, size=n_regions)

    suite.register("reference", "outcome", reference_vals, region=regions)
    suite.register("shifted", "outcome", shifted_vals, region=regions)
    suite.register("noise", "outcome", noise_vals, region=regions)
    return suite


def test_metricsuite_compare_methods_returns_expected_columns():
    suite = _make_compare_suite()
    results = suite.compare_methods("outcome", reference="reference")

    expected_cols = {
        "Method", "n_pairs", "statistic", "p_value", "p_corrected",
        "effect_size", "effect_size_type", "significant",
    }
    assert expected_cols.issubset(set(results.columns))
    assert set(results["Method"]) == {"shifted", "noise"}


def test_metricsuite_compare_methods_requires_reference():
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.compare_methods("outcome")  # reference is a required positional arg


def test_metricsuite_compare_methods_plot_returns_tuple():
    suite = _make_compare_suite()
    results, fig, ax = suite.compare_methods("outcome", reference="reference", plot=True)

    assert isinstance(results, pd.DataFrame)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def _make_compare_suite_with_reps(n_regions=6, n_reps=4, seed=0):
    """
    Unlike `_make_compare_suite` (one row per region already), each region here
    carries `n_reps` repeated rows -- the shape needed to actually exercise
    compare_methods' pre-test averaging, since with only one row per region
    there's nothing to collapse and a stale "plot the raw df" bug wouldn't show.
    """
    suite = MetricSuite(group_name="Method")
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_regions)]
    for label, shift in [("reference", 0.0), ("shifted", -2.0)]:
        for region in regions:
            vals = rng.normal(0, 1, size=n_reps) + shift
            suite.register(label, "outcome", vals, region=region, rep=np.arange(n_reps))
    return suite


def test_metricsuite_compare_methods_plot_dots_match_tested_pairs():
    """
    Regression guard: the box plot's raw-point overlay must show the same
    (Method, region)-averaged pairs the Wilcoxon test itself ran on (one dot
    per region per method, matching results' own "n_pairs") -- not the raw,
    uncollapsed rows (which would be n_reps x too many and visually imply far
    more spread than the test ever saw).
    """
    n_regions, n_reps = 6, 4
    suite = _make_compare_suite_with_reps(n_regions=n_regions, n_reps=n_reps)
    assert len(suite.dataframe) == 2 * n_regions * n_reps  # sanity: raw data really is repeated

    results, fig, ax = suite.compare_methods("outcome", reference="reference", subject_col="region", plot=True)
    assert (results["n_pairs"] == n_regions).all()

    total_points = sum(len(c.get_offsets()) for c in ax.collections)
    assert total_points == 2 * n_regions  # 2 methods x n_regions, not x n_reps
    plt.close(fig)


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

    suite = MetricSuite(group_name="Method")
    for unit, ref_val, comp_val in zip(true_order, reference_vals, competitor_vals):
        suite.register("reference", "outcome", ref_val, unit=unit)
        suite.register("competitor", "outcome", comp_val, unit=unit)
    return suite, true_order


def test_metricsuite_compare_methods_ttest_ess_and_subject_order_pass_through():
    """
    MetricSuite.compare_methods must forward test="ttest_ess" and
    subject_order down to mindmap.analysis.stats.compare_methods unchanged:
    without the true subject_order, n_eff should come back close to the
    nominal n (autocorrelation not detected in the scrambled default pivot
    order); with it, n_eff should drop well below n.
    """
    suite, true_order = _make_ar1_diff_suite()

    default_results = suite.compare_methods(
        "outcome", reference="reference", subject_col="unit", test="ttest_ess",
    )
    explicit_results = suite.compare_methods(
        "outcome", reference="reference", subject_col="unit", test="ttest_ess",
        subject_order=true_order,
    )

    n = default_results.loc[0, "n_pairs"]
    assert default_results.loc[0, "n_eff"] == pytest.approx(n, abs=2)
    assert explicit_results.loc[0, "n_eff"] < n * 0.6


def test_metricsuite_compare_methods_plot_uses_given_ax():
    suite = _make_compare_suite()
    fig_in, ax_in = plt.subplots()

    results, fig_out, ax_out = suite.compare_methods("outcome", reference="reference", plot=True, ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def _make_cluster_suite(n_regions=16, n_coord=8, shift_slice=slice(2, 5), seed=0):
    suite = MetricSuite(group_name="Method")
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        comp_curve = ref_curve + rng.normal(0, 0.3, size=n_coord)
        comp_curve[shift_slice] -= 3.0
        suite.register("reference", "outcome", ref_curve, region=region, level=coord_values)
        suite.register("competitor", "outcome", comp_curve, region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_returns_expected_keys():
    suite = _make_cluster_suite()
    result = suite.cluster_permutation(
        "outcome", coord="level", reference="reference", competitor="competitor",
        coord_order=list(range(8)), n_permutations=500, seed=0,
    )

    assert set(result.keys()) == {"clusters", "t_obs", "coord_values", "diff"}
    assert isinstance(result["clusters"], pd.DataFrame)


def test_metricsuite_cluster_permutation_plot_returns_tuple():
    suite = _make_cluster_suite()
    result, fig, ax = suite.cluster_permutation(
        "outcome", coord="level", reference="reference", competitor="competitor",
        coord_order=list(range(8)), n_permutations=500, seed=0, plot=True,
    )

    assert isinstance(result, dict)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_cluster_permutation_plot_uses_given_ax():
    suite = _make_cluster_suite()
    fig_in, ax_in = plt.subplots()

    result, fig_out, ax_out = suite.cluster_permutation(
        "outcome", coord="level", reference="reference", competitor="competitor",
        coord_order=list(range(8)), n_permutations=500, seed=0, plot=True, ax=ax_in,
    )

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def _make_cluster_suite_with_peak(n_regions=20, n_coord=8, peak_idx=4, peak_shift=5.0, base_shift=0.5, seed=0):
    """
    Mild-but-consistent shift at every coordinate (whole curve forms one
    cluster) except one coordinate with a much larger shift -- see
    tests/unit/test_analysis_stats.py's `_make_cluster_df_with_peak` (same
    fixture design, one level up through MetricSuite).
    """
    suite = MetricSuite(group_name="Method")
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        comp_curve = ref_curve.copy()
        for c in range(n_coord):
            shift = peak_shift if c == peak_idx else base_shift
            comp_curve[c] -= shift + rng.normal(0, 0.2)
        suite.register("reference", "outcome", ref_curve, region=region, level=coord_values)
        suite.register("competitor", "outcome", comp_curve, region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_effect_size_summary_pass_through():
    """
    MetricSuite.cluster_permutation must forward effect_size_summary down to
    cluster_permutation_mne unchanged: "peak" must report a bigger
    effect_size than "cluster_mean" on a cluster diluted by many mild
    coordinates plus one much stronger one.
    """
    suite = _make_cluster_suite_with_peak()

    result_mean = suite.cluster_permutation(
        "outcome", coord="level", reference="reference", competitor="competitor",
        coord_order=list(range(8)), n_permutations=300, seed=0, effect_size_summary="cluster_mean",
    )
    result_peak = suite.cluster_permutation(
        "outcome", coord="level", reference="reference", competitor="competitor",
        coord_order=list(range(8)), n_permutations=300, seed=0, effect_size_summary="peak",
    )

    assert "effect_size" in result_mean["clusters"].columns
    assert result_peak["clusters"].loc[0, "effect_size"] > result_mean["clusters"].loc[0, "effect_size"]


def test_metricsuite_cluster_permutation_summary_effect_size_survives_fdr_pooling():
    """
    cluster_permutation_summary's FDR-pooling step concatenates every
    competitor's clusters DataFrame and reassigns p_corrected/significant --
    effect_size/effect_size_type must ride along unchanged, not get dropped.
    """
    suite = _make_significance_suite()
    clusters_by_competitor, fig, ax = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=300, seed=0,
    )
    plt.close(fig)

    for clusters in clusters_by_competitor.values():
        assert "effect_size" in clusters.columns
        assert "effect_size_type" in clusters.columns


def _make_significance_suite(n_regions=20, n_coord=6, seed=0):
    """
    Three methods sharing one `reference`: `worse` is shifted far enough below
    `reference` at every coordinate to form one significant cluster spanning
    the whole range, `same` is indistinguishable from it (no cluster at all)
    -- the two extremes `cluster_permutation_summary`'s per-segment line/marker styling
    needs to tell apart correctly.
    """
    suite = MetricSuite(group_name="Method")
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        worse_curve = ref_curve - 5.0 + rng.normal(0, 0.1, size=n_coord)
        same_curve = ref_curve + rng.normal(0, 0.05, size=n_coord)
        suite.register("reference", "outcome", ref_curve, region=region, level=coord_values)
        suite.register("worse", "outcome", worse_curve, region=region, level=coord_values)
        suite.register("same", "outcome", same_curve, region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_summary_returns_expected_shapes():
    suite = _make_significance_suite()
    clusters_by_competitor, fig, ax = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=500, seed=0,
    )

    assert set(clusters_by_competitor.keys()) == {"worse", "same"}
    for clusters in clusters_by_competitor.values():
        assert isinstance(clusters, pd.DataFrame)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


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
    suite = MetricSuite(group_name="Method")
    rng = np.random.default_rng(seed)
    coord_values = np.arange(n_coord)
    regions = [f"r{i}" for i in range(n_regions)]

    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        strong_curve = ref_curve - 5.0 + rng.normal(0, 0.1, size=n_coord)
        weak_curve = ref_curve - 0.3 + rng.normal(0, 1.2, size=n_coord)
        suite.register("reference", "outcome", ref_curve, region=region, level=coord_values)
        suite.register("worse_strong", "outcome", strong_curve, region=region, level=coord_values)
        suite.register("worse_weak", "outcome", weak_curve, region=region, level=coord_values)

    return suite


def test_metricsuite_cluster_permutation_summary_corrects_across_competitors():
    """
    Regression guard for the actual point of this fix: cluster p-values from
    different competitors must be pooled into one family and corrected
    together (matching compare_methods' own correction="fdr_bh" default
    across its competitors) -- not each competitor left at its own raw,
    uncorrected per-cluster p-value, which would silently under-correct for
    testing multiple competitors against the same reference.
    """
    suite = _make_multi_competitor_cluster_suite()
    clusters_by_competitor, fig, ax = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=1000, seed=0,
    )

    non_empty = {name: cl for name, cl in clusters_by_competitor.items() if not cl.empty}
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
    plt.close(fig)


def test_metricsuite_cluster_permutation_summary_style_reflects_significance():
    """
    Regression guard for the actual point of cluster_permutation_summary: a clearly
    inferior competitor's drawn segments/markers must all be solid/filled,
    and a statistically indistinguishable one's must all be dashed/open --
    not just "a plot was returned". Colors are pinned explicitly so each
    competitor's own Line2D artists can be isolated by color and checked
    directly.
    """
    suite = _make_significance_suite()
    clusters_by_competitor, fig, ax = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=500, seed=0,
        colors={"reference": "black", "worse": "red", "same": "blue"},
    )

    assert clusters_by_competitor["worse"]["significant"].any()
    assert not clusters_by_competitor["same"]["significant"].any()

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


def test_metricsuite_cluster_permutation_summary_spread_whiskers_toggle():
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

    _, fig_with, ax_with = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=500, seed=0,
    )
    _, fig_without, ax_without = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=500, seed=0, show_spread=False,
    )

    assert len(ax_with.get_lines()) - len(ax_without.get_lines()) == n_methods * n_coord
    plt.close(fig_with)
    plt.close(fig_without)


def test_metricsuite_cluster_permutation_summary_plot_uses_given_ax():
    suite = _make_significance_suite()
    fig_in, ax_in = plt.subplots()

    _, fig_out, ax_out = suite.cluster_permutation_summary(
        "outcome", coord="level", reference="reference",
        coord_order=list(range(6)), n_permutations=500, seed=0, ax=ax_in,
    )

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def test_metricsuite_plot_returns_fig_ax():
    suite = _make_compare_suite()
    fig, ax = suite.plot(metric="outcome", kind="box")

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_metricsuite_plot_ax_passthrough():
    suite = _make_compare_suite()
    fig_in, ax_in = plt.subplots()

    fig_out, ax_out = suite.plot(metric="outcome", kind="box", ax=ax_in)

    assert ax_out is ax_in
    assert fig_out is ax_in.get_figure()
    plt.close(fig_in)


def test_metricsuite_plot_rejects_multiple_metrics():
    suite = _make_compare_suite()
    with pytest.raises(TypeError):
        suite.plot(metric=["outcome", "other"])


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
    suite = MetricSuite(name="roundtrip", group_name="Method")
    rng = np.random.default_rng(0)
    n_regions = 50
    coord_values = np.arange(5)

    for method in ["a", "b", "c"]:
        for j in range(n_regions):
            curve = rng.normal(size=5)
            suite.register(method, "outcome", curve, region=f"r{j}", level=coord_values)

    folder = tmp_path / "suite"
    suite.save(folder)
    loaded = MetricSuite.load(folder)

    original_df = suite.dataframe.sort_values(["Method", "region", "level"]).reset_index(drop=True)
    loaded_df = loaded.dataframe.sort_values(["Method", "region", "level"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(original_df, loaded_df)


def test_metricsuite_plot_box_points_toggle():
    """
    points=False must drop the raw-point stripplot overlay entirely (dense
    data, e.g. hundreds of points per box, makes it visual noise) while
    leaving the box plot itself intact; points=True (default) keeps it.
    """
    suite = _make_compare_suite()

    fig_with, ax_with = suite.plot(metric="outcome", kind="box", points=True)
    fig_without, ax_without = suite.plot(metric="outcome", kind="box", points=False)

    assert len(ax_with.collections) > 0
    assert len(ax_without.collections) == 0
    plt.close(fig_with)
    plt.close(fig_without)
