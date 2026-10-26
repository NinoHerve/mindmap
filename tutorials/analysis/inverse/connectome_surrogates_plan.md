# Plan: connectome surrogates vs. connectivity-informed source reconstruction

Goal: test how wCSET/CSET (connectivity-informed source reconstruction methods that
use a connectome's graph-Laplacian eigenspectrum as a spatial prior) perform when
given different connectome *construction methods* instead of the true/empirical
connectome, evaluated against sLORETA as a non-connectivity-informed reference.

Companion to `tutorials/analysis/connectomic/sabotage_connectome.ipynb` (structurally
similar harness -- wCSET/CSET at hrp/pp, `MetricSuite`-based -- but that study varies
*corruption severity* of the empirical connectome; this one varies *construction
method* instead, at no corruption).

## Step 0 — Verify data integrity before trusting anything

Known trap (see memory `stale-connectome-bug`): `sub-simf10`'s `con`/`eig` and the
`Simulator`'s own saved head-model state have silently diverged before (different
regeneration schedules), which once made a corrupted connectome spuriously outperform
the "clean" one. Last verified consistent 2026-07-14. Re-check before this experiment:
mtimes of `con.npy`, `eig.npz`, and `simulator_dir/conn_lengths.npz` for `sub-simf10`;
if suspicious, diff directly and confirm `eig` is bit-identical to a fresh
`compute_cst` on the current `con`.

## Step 1 — Region subset (reuse existing `sub-simf10` runs, no new simulation)

`sub-simf10` already has 473 runs (one per parcellation region, fixed `snr_target=10`)
— no regeneration needed.

- `mm.anatomy.parcellation.select_uniform_regions(fwd_sim["src"], parc, ras2tkras, n_regions=5, cortex_only=True, seed=0)` (same FPS approach as `simulation_spike.ipynb`) to pick 5 representative regions for the pilot run.
- Filter `simulation_index.tsv`'s `source_index_list` column (string-parsed) to match those region names -> subset of existing `run` ids.
- **N=5 for this initial pipeline implementation/test pass.** Scale up once the pipeline is validated end-to-end.

## Step 2 — Build the 2 surrogate connectomes (+ true)

All surrogates are length/distance matrices built at the *simulation*-resolution
source space `fwd_sim["src"]` (7465 dipoles, matching `con`'s shape) -- not the
inversion forward.

1. **Fused** (`connectome-atlas.ipynb` recipe, unchanged): `connectome_from_atlas(..., sparsity=0.14, deterministic=True)` fused with `connectome_from_geometry(..., maximum_dist=22, maximum_neigh=35, weights=True)` filling gaps.
2. **Geometric-only, atlas params**: `connectome_from_geometry(src_sim, maximum_dist=22, maximum_neigh=35, weights=True, inter_hemi_mirror=True)` alone.

**Dropped: atlas-only (no geometric fill).** Removed -- known from prior experience to
leave isolated nodes, which is exactly why the atlas construction is always merged
with geometry in practice. Without the geometric component there's no good way to
connect those nodes, so it's not a meaningful standalone surrogate.

**Dropped: geometric-only, empirical params** (`maximum_dist=15, maximum_neigh=3`).
Verified empirically (see below) that even with the interhemispheric mirror fix, this
parameter set leaves ~2 residual disconnected components from fragmentation *within*
each hemisphere -- these tight params were only ever validated in
`connectome-empirical.ipynb` as a thin supplement layered on already-dense
tractography, not as a standalone graph. Same underlying problem as atlas-only
(construction method that only works as a supplement, not standalone), so dropped for
the same reason rather than patched further.

The atlas-params geometric-only surrogate does *not* have this problem -- confirmed
(see below) it reaches a single connected component with the mirror fix, since its
much looser parameters (`maximum_dist=22, maximum_neigh=35`) don't fragment within a
hemisphere the way the tight empirical params do.

Plus each of the 2 remaining surrogates' **binary version** (`(con > 0).astype(float)`
-- no existing reusable binarizer, just inline thresholding, consistent with how it's
done elsewhere in the codebase) -> 4 surrogates total, plus the true connectome used
as-is (already has a verified `eig`) = **5 connectomes total**.

### Interhemispheric connectivity fix (`connectome_from_geometry` update)

`connectome_from_geometry` builds each hemisphere as an independent KNN graph with
*zero* interhemispheric edges by default (`inter_hemi_dist=0`) -- so both geometric-only
surrogates above are, by construction, exactly two disconnected components unless
fixed. The existing `inter_hemi_dist` (Euclidean-proximity threshold across the whole
brain) is anatomically crude -- it connects whatever happens to be geometrically close
across the midline regardless of which regions are actually being linked.

**Implemented fix**: new `inter_hemi_mirror: bool = False` parameter. When `True`,
additionally connects each node to its homotopic (mirror-symmetric) partner in the
other hemisphere -- reflect one hemisphere's coordinates across the midline (x=0),
nearest-neighbor match to the other hemisphere (same technique already used by
`hemispheric_symmetry` in `mindmap/core/metric.py`, reused here for construction
instead of just diagnostics), weighted by the true (non-mirrored) 3D distance between
the matched pair. Real interhemispheric white matter (the corpus callosum) is
predominantly homotopic, so this is a principled, always-connecting alternative to the
proximity threshold -- and it's additive/independent of `inter_hemi_dist`, so both can
be combined if wanted. Default `False` preserves existing behavior for all current
callers (`connectome-atlas.ipynb`, `connectome-empirical.ipynb` do not need this fix --
their fused connectomes get real interhemispheric edges from atlas/tractography data,
not from the geometric gap-filler).

Use `inter_hemi_mirror=True` for the geometric-only/atlas-params surrogate above. Do
not change the fused recipe's own geometric gap-fill call -- that stays exactly as
originally written in `connectome-atlas.ipynb`.

**Empirical verification** (on real `sub-simf10` data, `fwd_sim["src"]`, 7465 dipoles):
- Default behavior unchanged: `connectome_from_geometry(..., maximum_dist=15, maximum_neigh=3)` with no interhemispheric option -> 7 components (confirms each hemisphere independently fragments under tight params, before interhemispheric connectivity even enters the picture) -- this is the empirical-params case that got dropped above.
- `connectome_from_geometry(..., maximum_dist=22, maximum_neigh=35, inter_hemi_mirror=True)` -> **1 component** (fully connected) -- this is the surrogate actually kept.
- Mirror edges: 3748 non-zero interhemispheric edges, matrix symmetric, edge lengths 4.5-122.4mm (plausible callosal-fiber range).

## Step 3 — Compute eigenspectra once, upfront, not per-task

For each of the 4 surrogates: invert length -> strength the same way the shipped `eig`
was built (`strength = np.divide(1, con, out=np.zeros_like(con), where=con>0)` -- a
no-op for binary matrices, since 1/1=1), then `compute_cst(strength)`. Reuse the
existing `dm.load("simf10", "eig")` for the true connectome rather than recomputing.

Per the sabotage study, eigendecomposition dominates wall time (~18s/call on this
7465-node graph) -- 4 fresh `eigh` calls is the real cost (~1-1.5 min total), and it
must happen once per surrogate, shared across every region, not recomputed per
reconstruction task.

## Step 4 — Reconstruction: 5x wCSET + 5x CSET + 1x sLORETA, at hrp/pp, across the region subset

Reuse `InverseTask`/`run_parallel_pipeline` (`mindmap/analysis/functional/inverse/pipeline.py`)
and the same `preprocess_hrp_pp` already built for `inverse-noisesweep.ipynb`, unchanged
(gold-projection-based hrp/pp, immune to noise). `reconstruct_fn` becomes a factory that
loops over the 5 precomputed `(label, eig)` pairs, building one `wCSET`/`CSET` per pair
(same lambda2 convention as before: auto-estimated per run for wCSET/sLORETA, fixed
`0.1` for CSET) -- 11 solutions per run. `postprocess_fn` scores each against that run's
`gold` with the same 5 metrics as `inverse-noisesweep.ipynb` (correlation_hrp/pp,
localization_error_hrp, ap_hrp/pp), tagged with `label` (e.g. `"wcset_fused"`,
`"cset_geom_atlas_bin"`, `"sloreta"`) instead of `snr`, since the swept axis here is
surrogate identity, not noise level.

Label count: true (2: wcset_true, cset_true) + fused (4: wcset/cset x weighted/binary)
+ geom_atlas (4) + sloreta (1) = **11 total**.

## Step 5 — Visualize as boxplots

`mm.metric.MetricSuite.plot(kind="box")` already supports exactly this (it's the
statistical-plot default) -- `metrics.plot(metrics=["correlation_hrp"], kind="box")`
groups by the registered `Label`, no `coord=` needed since there's no continuous sweep
axis here. One boxplot per metric (5 total), each showing all 11 labels' distributions
across the region subset.

## Status

**Done.** All 5 steps implemented in `tutorials/analysis/inverse/connectome_surrogates.ipynb`
and executed successfully end to end (no errors), first as an N=5 pilot to validate the
pipeline, then scaled to **N=15** regions. `connectome_from_geometry`'s
`inter_hemi_mirror` update implemented and empirically verified (see Step 2).

Beyond the original plan: the boxplots (Step 5) were extended with explicit method-grouped
ordering (all wCSET variants, then all CSET variants, then sLORETA -- via `order=` passed
through `MetricSuite.plot`) and a numeric median/mean/std summary table, since both made
the actual finding easier to read/verify than the plan's plain per-metric boxplot alone.

**Headline result** (N=15): wCSET clearly beats CSET beats sLORETA on correlation/AP
(large, consistent gap). More importantly for this experiment's actual question:
connectome *construction method* barely matters for wCSET -- all 5 wCSET variants
(true, fused, fused_bin, geom_atlas, geom_atlas_bin) land within a narrow band, and the
true empirical connectome is not clearly the best of the 5. Localization error inverts
the usual read: wCSET's median is best but its mean is inflated by heavy-tailed
outliers (occasional bad misses), while CSET is more consistently mediocre. See the
analysis delivered in conversation for full numbers and caveats (N=15 is still a fairly
small sample for the finer between-surrogate distinctions).
