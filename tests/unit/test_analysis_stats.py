"""
This module performs unit tests for the DataFrame/MNE-level statistical
comparisons in analysis/stats.py.
"""

import numpy as np
import pandas as pd
import pytest

from mindmap.analysis.stats import compare_methods, cluster_permutation_mne


def _make_compare_df(n_regions=20, seed=0):
    """
    Long-format table: a reference method vs a clearly-shifted competitor
    and a pure-noise competitor, matched by region.
    """
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_regions)]
    reference_vals = rng.normal(0, 1, size=n_regions)

    rows = []
    for region, ref_val in zip(regions, reference_vals):
        rows.append({"Method": "reference", "region": region, "Value": ref_val})
        rows.append({"Method": "shifted", "region": region, "Value": ref_val - 2.0 + rng.normal(0, 0.1)})
        rows.append({"Method": "noise", "region": region, "Value": ref_val + rng.normal(0, 1)})

    return pd.DataFrame(rows)


def test_compare_methods_recovers_known_difference():
    """The shifted competitor should come back significant, the noise one should not."""
    df = _make_compare_df(seed=1)

    results = compare_methods(df, group_col="Method", reference="reference", subject_col="region")

    shifted = results[results["Method"] == "shifted"].iloc[0]
    noise = results[results["Method"] == "noise"].iloc[0]

    assert shifted["significant"]
    assert not noise["significant"]


def test_compare_methods_averages_reps_first():
    """
    Two rows per (region, method) with different 'rep' values but the same
    mean must give an identical result to a single pre-averaged row --
    regression guard on the groupby-mean-first behavior, mirroring the
    superseded notebooks' convention of averaging repeated observations
    before building the region-level comparison.
    """
    base = _make_compare_df(n_regions=10, seed=1)

    expanded_rows = []
    for _, row in base.iterrows():
        for rep, delta in enumerate([-0.3, 0.3]):
            expanded_rows.append({**row.to_dict(), "Value": row["Value"] + delta, "rep": rep})
    expanded = pd.DataFrame(expanded_rows)

    result_base = compare_methods(base, group_col="Method", reference="reference", subject_col="region")
    result_expanded = compare_methods(expanded, group_col="Method", reference="reference", subject_col="region")

    pd.testing.assert_frame_equal(
        result_base.reset_index(drop=True), result_expanded.reset_index(drop=True),
    )


def test_compare_methods_separates_families_per_metric():
    """
    Two Metrics in df must be corrected independently: pulling one metric's
    rows out into its own call must reproduce the combined call's results
    for that metric exactly (the other metric's p-values don't leak in).
    """
    df1 = _make_compare_df(n_regions=15, seed=2)
    df1["Metric"] = "outcome_a"
    df2 = _make_compare_df(n_regions=15, seed=3)
    df2["Metric"] = "outcome_b"
    combined = pd.concat([df1, df2], ignore_index=True)

    result_a_alone = compare_methods(df1, group_col="Method", reference="reference", subject_col="region")
    result_combined = compare_methods(combined, group_col="Method", reference="reference", subject_col="region")

    a_from_combined = result_combined[result_combined["Metric"] == "outcome_a"].reset_index(drop=True)
    a_alone = result_a_alone.reset_index(drop=True)

    pd.testing.assert_frame_equal(a_from_combined, a_alone)


def test_compare_methods_requires_known_reference():
    df = _make_compare_df(n_regions=5, seed=4)
    with pytest.raises(ValueError):
        compare_methods(df, group_col="Method", reference="not_a_method", subject_col="region")


def _make_cluster_df(n_regions=16, n_coord=8, shift_slice=slice(2, 5), seed=0):
    """
    (n_regions, n_coord) reference/competitor pair with a clear consistent
    shift injected only in `shift_slice`, pure noise elsewhere.
    """
    rng = np.random.default_rng(seed)
    coord_values = list(range(n_coord))
    regions = [f"r{i}" for i in range(n_regions)]

    rows = []
    for region in regions:
        ref_curve = rng.normal(0, 1, size=n_coord)
        comp_curve = ref_curve + rng.normal(0, 0.3, size=n_coord)
        comp_curve[shift_slice] -= 3.0  # reference clearly better in this span
        for c, ref_v, comp_v in zip(coord_values, ref_curve, comp_curve):
            rows.append({"Method": "reference", "region": region, "level": c, "Value": ref_v})
            rows.append({"Method": "competitor", "region": region, "level": c, "Value": comp_v})

    return pd.DataFrame(rows), coord_values


def test_cluster_permutation_mne_recovers_known_cluster():
    df, coord_values = _make_cluster_df()

    result = cluster_permutation_mne(
        df, group_col="Method", coord_col="level", reference="reference", competitor="competitor",
        subject_col="region", coord_order=coord_values, n_permutations=500, seed=0,
    )

    clusters = result["clusters"]
    sig = clusters[clusters["significant"]]
    assert len(sig) >= 1

    row = sig.iloc[0]
    assert row["level_start"] <= 3 <= row["level_end"]
    assert row["sign"] == 1  # positive = reference wins, matches the injected shift


def test_cluster_permutation_mne_sign_convention_flips():
    df, coord_values = _make_cluster_df(seed=1)

    result_higher = cluster_permutation_mne(
        df, group_col="Method", coord_col="level", reference="reference", competitor="competitor",
        subject_col="region", coord_order=coord_values, higher_is_better=True, n_permutations=500, seed=0,
    )
    result_lower = cluster_permutation_mne(
        df, group_col="Method", coord_col="level", reference="reference", competitor="competitor",
        subject_col="region", coord_order=coord_values, higher_is_better=False, n_permutations=500, seed=0,
    )

    np.testing.assert_allclose(result_higher["t_obs"], -result_lower["t_obs"])


def test_cluster_permutation_mne_raises_on_unbalanced_design():
    df, coord_values = _make_cluster_df(n_regions=6, seed=2)
    df = df[~((df["Method"] == "competitor") & (df["region"] == "r0"))]

    with pytest.raises(ValueError):
        cluster_permutation_mne(
            df, group_col="Method", coord_col="level", reference="reference", competitor="competitor",
            subject_col="region", coord_order=coord_values,
        )
