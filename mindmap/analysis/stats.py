"""
Module Name: analysis/stats.py
Description: DataFrame/MNE-level statistical comparisons for `MetricSuite`
data -- paired method-vs-reference comparisons (feeding box-plot
significance annotations) and cluster-permutation tests between curves
(wrapping `mne.stats.permutation_cluster_1samp_test`).

Region (or whatever `subject_col` names) is treated as the repeated-measures
"subject" throughout, matching the convention already used by
`mindmap.analysis.power.variance.fit_variance_components` and the hand-rolled
cluster-permutation notebooks under `tutorials/analysis/simulations/` that
this module's `cluster_permutation_mne` supersedes.

Author: Nino Herve
"""

import numpy as np
import pandas as pd

import mne

from mindmap.core.stats import compare_to_reference, cluster_threshold


def compare_methods(
    df: pd.DataFrame,
    group_col: str,
    reference: str,
    subject_col: str = "region",
    competitors: list = None,
    test: str = "wilcoxon",
    correction: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Paired comparison of `reference` against every other (or listed
    `competitors`) level of `group_col`, matched by `subject_col`.

    Repeated observations per `(subject_col, group_col[, "Metric"])` cell
    (e.g. a "rep" column, one row per simulation repetition) are averaged
    first -- the same convention the superseded notebooks used when
    building their region x level matrices. If a "Metric" column is present
    and multi-valued, each metric is treated as its own comparison family
    (p-values corrected independently per metric), matching how the
    notebooks looped `for outcome in OUTCOMES`.

    Parameters
    ----------
    df: pd.DataFrame
        Long-format table (`MetricSuite.dataframe`-shaped), with at least
        `[group_col, subject_col, "Value"]`, optionally "Metric" and "rep".

    group_col: str
        Column identifying the method/label being compared (`MetricSuite`'s
        `group_name`, e.g. "Method").

    reference: str
        Required -- the method every competitor is compared against.
        Deliberately has no default: which method is "baseline" determines
        what "significant" means, so it should be an explicit choice.

    subject_col: str
        Repeated-measures grouping column, default "region".

    competitors: list[str], optional
        Defaults to every other `group_col` level present for a given metric.

    test, correction, alpha
        Forwarded to `mindmap.core.stats.compare_to_reference`.

    Returns
    -------
    pd.DataFrame with columns (["Metric"] if present in `df`), `group_col`,
    "n_pairs", "statistic", "p_value", "p_corrected", "effect_size",
    "effect_size_type", "significant" -- one row per (metric, competitor).
    """
    if df.empty:
        raise ValueError("df is empty.")
    for col in (group_col, subject_col, "Value"):
        if col not in df.columns:
            raise ValueError(f"df is missing required column '{col}'.")

    has_metric = "Metric" in df.columns
    group_keys = [group_col, subject_col] + (["Metric"] if has_metric else [])
    collapsed = df.groupby(group_keys, as_index=False)["Value"].mean()

    metrics = collapsed["Metric"].unique().tolist() if has_metric else [None]

    rows = []
    for metric in metrics:
        sub = collapsed[collapsed["Metric"] == metric] if has_metric else collapsed

        if reference not in sub[group_col].unique():
            raise ValueError(
                f"Reference '{reference}' not found in '{group_col}'"
                + (f" for Metric '{metric}'." if has_metric else ".")
            )

        wide = sub.pivot(index=subject_col, columns=group_col, values="Value")

        comp_names = competitors if competitors is not None else [c for c in wide.columns if c != reference]
        missing = set(comp_names) - set(wide.columns)
        if missing:
            raise ValueError(
                f"Competitors {sorted(missing)} not found in '{group_col}'"
                + (f" for Metric '{metric}'." if has_metric else ".")
            )

        complete = wide.dropna(subset=[reference] + list(comp_names))
        if complete.empty:
            raise ValueError(
                "No subjects with complete data across reference and competitors"
                + (f" for Metric '{metric}'." if has_metric else ".")
            )

        results = compare_to_reference(
            complete[reference].to_numpy(), complete[list(comp_names)].to_numpy(),
            competitor_names=list(comp_names), test=test, correction=correction, alpha=alpha,
        )

        for r in results:
            row = {"Metric": metric} if has_metric else {}
            row.update({
                group_col: r["name"], "n_pairs": r["n"], "statistic": r["statistic"],
                "p_value": r["p_value"], "p_corrected": r["p_corrected"],
                "effect_size": r["effect_size"], "effect_size_type": r["effect_size_type"],
                "significant": r["significant"],
            })
            rows.append(row)

    columns = (["Metric"] if has_metric else []) + [
        group_col, "n_pairs", "statistic", "p_value", "p_corrected",
        "effect_size", "effect_size_type", "significant",
    ]
    return pd.DataFrame(rows, columns=columns)


def cluster_permutation_mne(
    df: pd.DataFrame,
    group_col: str,
    coord_col: str,
    reference: str,
    competitor: str,
    subject_col: str = "region",
    higher_is_better: bool = True,
    coord_order: list = None,
    cluster_alpha: float = 0.05,
    n_permutations: int = 3000,
    tail: int = 0,
    seed: int = 0,
) -> dict:
    """
    Region-as-subject cluster-permutation test of `reference` vs
    `competitor` along `coord_col` (e.g. "snr", "percentage"), wrapping
    `mne.stats.permutation_cluster_1samp_test` -- replacing the hand-rolled
    sign-flip permutation in `tutorials/analysis/simulations/
    pilot_noise_cluster_permutation.ipynb` /
    `pilot_degradation_cluster_permutation.ipynb`, which implements exactly
    the same one-sample/paired sign-flip null that MNE's function already
    provides.

    Parameters
    ----------
    df: pd.DataFrame
        Long-format table, pre-filtered to a single "Metric" (this function
        compares one outcome at a time), with at least
        `[group_col, coord_col, subject_col, "Value"]`. Repeated
        observations per `(subject_col, coord_col)` cell (e.g. a "rep"
        column) are averaged first, same convention as `compare_methods`
        and the superseded notebooks.

    group_col, coord_col, subject_col: str
        Column names for method, x-axis coordinate, and repeated-measures
        subject, respectively.

    reference, competitor: str
        The two `group_col` levels being compared.

    higher_is_better: bool
        Sign convention: the paired difference is `sign * (reference -
        competitor)` with `sign = +1` if True else `-1`, so a positive
        result always means "reference wins" regardless of the outcome's
        natural direction -- matches the superseded notebooks' convention.

    coord_order: list, optional
        Explicit ordering of `coord_col` values along the contiguity axis.
        Defaults to the sorted ascending values common to both methods --
        pass this explicitly when the meaningful contiguity direction is
        descending (e.g. decreasing SNR = increasing noise, as in the
        noise-sweep notebook).

    cluster_alpha: float
        Cluster-forming threshold's alpha, and also the alpha applied to
        each cluster's final p-value. When 0.05 (the default), `threshold`
        is left as `None` for `mne.stats.permutation_cluster_1samp_test` to
        auto-compute -- confirmed equivalent to the superseded notebooks'
        manual `scipy.stats.t.ppf(1 - cluster_alpha / 2, df=n_subjects - 1)`
        formula. For any other `cluster_alpha`, an explicit threshold is
        built via `mindmap.core.stats.cluster_threshold`, since MNE's own
        auto-threshold can't be parameterized by alpha.

    n_permutations, tail, seed
        Forwarded to `mne.stats.permutation_cluster_1samp_test`.

    Returns
    -------
    dict:
      "clusters": pd.DataFrame, columns `[f"{coord_col}_start",
          f"{coord_col}_end", "mass", "sign", "p_value", "significant"]`
          (one row per cluster MNE found, in original `coord_col` units,
          `significant` = `p_value < cluster_alpha`), possibly empty.
      "t_obs": np.ndarray of shape (n_coord,) -- per-coordinate observed
          t-statistic.
      "coord_values": list -- the `coord_order` actually used (x-axis for
          `t_obs`).
      "diff": pd.DataFrame (`subject_col` index x `coord_col` columns) --
          the paired-difference matrix itself, for a mean+SE curve overlay.

    Raises
    ------
    ValueError
        If reference/competitor rows are missing, or the two methods don't
        share the exact same `(subject_col, coord_col)` cells (unbalanced
        design) -- the superseded notebooks implicitly assumed a fully
        balanced design too.
    """
    for col in (group_col, coord_col, subject_col, "Value"):
        if col not in df.columns:
            raise ValueError(f"df is missing required column '{col}'.")

    def _wide(label):
        sub = df[df[group_col] == label]
        if sub.empty:
            raise ValueError(f"No rows found for '{group_col}' == '{label}'.")
        return sub.groupby([subject_col, coord_col])["Value"].mean().unstack(coord_col)

    wide_ref = _wide(reference)
    wide_comp = _wide(competitor)

    if coord_order is None:
        coord_order = sorted(set(wide_ref.columns) & set(wide_comp.columns))

    missing_ref = set(coord_order) - set(wide_ref.columns)
    missing_comp = set(coord_order) - set(wide_comp.columns)
    if missing_ref or missing_comp:
        raise ValueError(
            f"coord_order values missing from data: reference missing {sorted(missing_ref)}, "
            f"competitor missing {sorted(missing_comp)}."
        )
    wide_ref = wide_ref[coord_order]
    wide_comp = wide_comp[coord_order]

    only_ref = set(wide_ref.index) - set(wide_comp.index)
    only_comp = set(wide_comp.index) - set(wide_ref.index)
    if only_ref or only_comp:
        raise ValueError(
            f"Unbalanced design: reference and competitor do not share the same "
            f"'{subject_col}' values (only in reference: {sorted(only_ref)}, "
            f"only in competitor: {sorted(only_comp)})."
        )

    subjects = sorted(wide_ref.index)
    wide_ref = wide_ref.loc[subjects]
    wide_comp = wide_comp.loc[subjects]

    if wide_ref.isna().any().any() or wide_comp.isna().any().any():
        raise ValueError(
            f"Unbalanced design: some ('{subject_col}', '{coord_col}') cells are "
            "missing for reference or competitor."
        )

    sign = 1.0 if higher_is_better else -1.0
    diff = sign * (wide_ref - wide_comp)

    n_subjects = diff.shape[0]
    threshold = None if cluster_alpha == 0.05 else cluster_threshold(n_subjects, alpha=cluster_alpha, tail=tail)

    t_obs, cluster_masks, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(
        diff.to_numpy(), threshold=threshold, n_permutations=n_permutations,
        tail=tail, seed=seed, out_type="mask", verbose=False,
    )

    rows = []
    for cluster, pv in zip(cluster_masks, cluster_pv):
        # `out_type="mask"` still returns a tuple of slices/index-arrays rather
        # than a boolean array for this 2D, no-explicit-adjacency case (MNE's
        # memory-efficient representation for contiguous 1D runs) -- building
        # the mask via fancy-index assignment handles both representations,
        # where `np.where(cluster)` would silently misinterpret a slice tuple.
        mask = np.zeros(t_obs.shape, dtype=bool)
        mask[cluster] = True
        idx = np.where(mask)[0]
        mass = float(t_obs[idx].sum())
        rows.append({
            f"{coord_col}_start": coord_order[idx[0]],
            f"{coord_col}_end": coord_order[idx[-1]],
            "mass": mass,
            "sign": 1 if mass > 0 else -1,
            "p_value": float(pv),
            "significant": bool(pv < cluster_alpha),
        })

    clusters_df = pd.DataFrame(
        rows,
        columns=[f"{coord_col}_start", f"{coord_col}_end", "mass", "sign", "p_value", "significant"],
    )

    return {
        "clusters": clusters_df,
        "t_obs": t_obs,
        "coord_values": list(coord_order),
        "diff": diff,
    }
