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
