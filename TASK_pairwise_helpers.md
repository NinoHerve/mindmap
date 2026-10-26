# Task: reference-vs-competitors helpers for `MetricSuite` statistics and plots

## Context

mindmap is used by the wCSET manuscript (`/home/nino/Manuscripts/wCSET`), whose figures compare one
reference method (usually `"wcset"`) against several competitors with paired tests
(`suite.stat.wilcoxon` / `.ttest_ess`). The manuscript currently re-implements, in its own
`scripts/figures.py`, three things that belong in mindmap because any "reference vs. competitors"
analysis needs them:

1. reading one competitor's row from a `PairwiseResult` with the effect size oriented toward the reference;
2. a paired-difference panel (per-unit differences reference − competitor, as violins or points,
   annotated with significance, effect size and win rate);
3. small additions to `suite.plot.group_comparison` (chance band, reference lines, fill opacity,
   tick labels) so callers stop patching the axes afterwards.

Goal: implement these in mindmap so the manuscript can later replace its local helpers with calls
such as:

```python
s = result.vs("wcset")                       # one row per competitor, effect > 0 = wCSET better
s.loc["sloreta", "p_corrected"]

suite.plot.paired_differences(result, reference="wcset", palette=COLORS,
                              annotate=("stars", "effect", "winrate"), ax=ax)
```

Do **not** edit the manuscript repository; it is only reference material (a later session will
switch the manuscript to the new API).

## Files to read first

mindmap (editable install, package root `/home/nino/Code/mindmap/mindmap`):

- `/home/nino/Code/mindmap/mindmap/metric/results.py`: `PairwiseResult` (class starts around line 144:
  `table`, `metric`, `group_by`, `pairing_col`, `unit_col`, `subject_order`, `filters`, `correct`,
  `merge`, `save`/`load`).
- `/home/nino/Code/mindmap/mindmap/metric/plot.py`: `PlotAccessor`, especially `group_comparison`
  (around line 226) and the private helpers `_reference_view`, `_infer_reference`,
  `_filters_to_dict`, `_collapse_for_plot` (lines 27-100). Follow the same pattern: everything about
  *what* was tested comes from the result object, never re-passed by the caller.
- `/home/nino/Code/mindmap/mindmap/metric/stat.py`: `StatAccessor`, `ttest_ess` (around line 231),
  `_pairwise`, and the caching of results (cache key includes `subject_order`).
- `/home/nino/Code/mindmap/mindmap/core/stats.py`: `paired_test` (around line 80). Sign convention:
  `diff = x - y` with `x = group_a`, `y = group_b`, so `effect_size > 0` means `group_a` is higher.
  `effect_size_type` is `"rank_biserial"` (Wilcoxon) or `"cohens_dz"` (t-tests).
- `/home/nino/Code/mindmap/mindmap/viz/stats.py`: `metric_box_significance` (drawn by `group_comparison`).
- `/home/nino/Code/mindmap/mindmap/viz/core.py`: `metric_box` (around line 494).
- `/home/nino/Code/mindmap/tests/unit/test_metric_suite.py`: existing tests for `MetricSuite`,
  `PairwiseResult` and `group_comparison`; add the new tests here.
- `/home/nino/Code/mindmap/tutorials/analysis/metrics/metricsuite-intra-vieeg.ipynb`: tutorial using
  `ttest_ess` on the same kind of data (optional: add a short example of the new API).

Manuscript code to reproduce (reference only, do not modify):

- `/home/nino/Manuscripts/wCSET/scripts/figures.py`: `competitor_row`, `ref_effect`, `stars_for`,
  `paired_diff_panel`, `vs_annotations`, `method_boxes`, `soften_boxes`, `ref_labels`, `mark_direction`.
  These are the behaviours to port.
- `/home/nino/Manuscripts/wCSET/index.qmd`: cells `results_intra_vieeg_significance.py` and
  `results_intra_vieeg_significance_plot.py` (paired t-test with effective sample size, timepoint-paired,
  annotations stars / dz / win rate), `simf_localization_correlation_plot.py` and
  `results_resection_localization_plot.py` (Wilcoxon, region- or subject-paired, annotations
  stars / rank-biserial r / better-tied-worse counts, and lower-is-better metrics), and
  `simf_surrogate_plot.py` (reference = `"original"`, dashed reference lines on the boxplots).
- Test data with cached statistics: `/home/nino/Manuscripts/wCSET/results/intra/spatial_metrics`
  (metrics `correlation_spatial_signed`, `auc_spatial`; `group_by="Method"`, `pairing_col="times"`,
  filter `{"Subject": "48"}`) and `/home/nino/Manuscripts/wCSET/results/simf/surrogates`
  (`group_by="Surrogate"`, `pairing_col="region"`).

Python environment: `/home/nino/miniconda3/envs/mindmap/bin/python` (mindmap is installed in editable mode there).

## What to implement

### 1. `PairwiseResult.vs(reference, higher_is_better=True)`

Return a `pd.DataFrame` indexed by competitor (every group paired with `reference` in `self.table`,
in table order), with columns `p_value`, `p_corrected`, `significant`, `effect` (the effect size
oriented so that **positive = reference better**: flip the sign when `reference` is `group_b`, flip
again when `higher_is_better=False`), `effect_size_type`, `n_pairs`, and `n_eff` when present.
Raise a clear `ValueError` if `reference` appears in no pair. This replaces the manuscript's
`competitor_row` + `ref_effect`.

### 2. Paired differences: computation and plot

- `suite.stat.paired_differences(result, reference=None, higher_is_better=True)`: return a wide
  `pd.DataFrame` (index = pairing unit, columns = competitors) of `reference - competitor`
  (negated when `higher_is_better=False`), built from the suite's data with exactly the same filters,
  pairing, unit collapsing and complete-case / `subject_order` handling as the test that produced
  `result`, so that the number of rows equals the test's `n_pairs`. Infer `reference` like
  `group_comparison` does when omitted.
- `suite.plot.paired_differences(result, reference=None, higher_is_better=True, order=None,
  palette=None, annotate=("stars", "effect"), violin_min_n=30, fill_alpha=0.45, seed=0, ax=None)`:
  draw one column per competitor, as violins (with IQR line) when the column has at least
  `violin_min_n` values, otherwise jittered points; thick median line; horizontal line at 0;
  annotation lines stacked above each column, the first line (stars) larger. Supported annotation
  keys:
  - `"stars"`: `***`/`**`/`*`/`ns` from `p_corrected`;
  - `"effect"`: `"dz=+0.56"` or `"r=+0.42"` depending on `effect_size_type`, oriented as in `vs()`;
  - `"winrate"`: share of units where the reference is better, minus 50%, e.g. `"+26%"`;
  - `"counts"`: `"better/tied/worse"` counts, e.g. `"61/0/39"`;
  - `"n_eff"`: `"n_eff=35"` when available.
  
  Return `(fig, ax)` like `group_comparison`. Port the look from `paired_diff_panel` in the
  manuscript's `scripts/figures.py` so existing figures stay visually identical.
  Optional keyword `direction_labels=("wCSET better", "other method better")` reproducing
  `mark_direction` (labels to the right of the axes, at y = 0).

### 3. `group_comparison` additions (backward compatible, all optional)

- `chance=(lo, hi)`: gray band behind the boxes; `chance_value=float`: dotted horizontal line.
- `reference_lines={label: value}` or `{label: (value, color)}`: dashed horizontal lines (used for
  dSPM/sLORETA reference medians in the surrogate figure).
- `fill_alpha=float`: box fill opacity with opaque edges (replaces the manuscript's `soften_boxes`).
- `tick_labels={group: label}` and `reference_suffix=" (ref.)"`: display names for the x ticks, with
  the suffix appended to the reference (replaces `ref_labels`).

Check whether these belong in `group_comparison` itself or in `metric_box` / `metric_box_significance`
(`viz/core.py`, `viz/stats.py`), and keep the existing default output unchanged when they are not given.

## Constraints

- The working tree has unrelated uncommitted changes in `mindmap/viz/__init__.py`, `viz/brain.py`,
  `viz/camera.py`, `viz/mesh.py` and `environment.yml` (a visualization refactor in progress).
  Do not modify, revert or commit those files.
- Keep the existing public API and default behaviour unchanged; new arguments are keyword-only with
  defaults reproducing today's output.
- Match the surrounding code style: numpy-style docstrings with a `!!! note "Technical details"`
  block where the existing methods have one, and type checks with explicit error messages as in
  `group_comparison`.
- Statistics stay where they are computed today: the new methods only read results and data, they
  never recompute or alter a cached test.

## Validation

1. Unit tests in `/home/nino/Code/mindmap/tests/unit/test_metric_suite.py` covering: `vs()`
   orientation for both `group_a`/`group_b` placements and `higher_is_better=False`; the error for an
   unknown reference; `paired_differences` row count equal to `n_pairs` and sign convention; every
   annotation key; violin vs. points switch; `group_comparison` defaults unchanged.
   Run `/home/nino/miniconda3/envs/mindmap/bin/python -m pytest /home/nino/Code/mindmap/tests/unit -q`.
2. A check on real data (a scratch script, not committed): load
   `/home/nino/Manuscripts/wCSET/results/intra/spatial_metrics`, run the same `ttest_ess` call as the
   manuscript cell `results_intra_vieeg_significance.py` for subject 48 and metric `auc_spatial`,
   and confirm that `result.vs("wcset")` gives the same p-values and dz as the manuscript's
   `competitor_row` / `ref_effect`, that `suite.stat.paired_differences` gives the same arrays as the
   manuscript's `vieeg_diffs`, and that `suite.plot.paired_differences(..., annotate=("stars",
   "effect", "winrate"))` looks like panel a3 of `fig-intracranial-diff` (draw both side by side and
   inspect).
3. Report what was added, the test results, and the exact signatures, so the manuscript can be
   switched over in a follow-up session. Do not commit; leave the changes for review.
