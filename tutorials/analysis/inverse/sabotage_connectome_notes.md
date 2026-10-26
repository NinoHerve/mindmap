# Sabotage Connectome — Status Notes (2026-07-14)

Study setup: `wCSET(harmonic_method="lowpass", harm=5)` vs `CSET(lambda2=0.1)`,
stress-tested against 3 connectome corruption methods (`swap`, `brutal`, `dmri`)
at varying severity, on `sub-simf10`. Reconstructs only at hrp/pp
timepoints (both solvers process timepoints independently — confirmed by
reading `solvers.py` — so this is a valid speedup, not an approximation).
Metrics: localization error (hrp), spatial correlation (hrp/pp), AP (hrp/pp),
all plotted vs. corruption percentage. As of 2026-07-16 the notebook evaluates
across **multiple simulated runs** (not just `run=1`) via a two-stage parallel
pipeline — see "Multi-run parallel pipeline" below; the "Latest completed run"
section right below predates that and describes the older single-run version.

## Harmonic method decision (settled)

`lowpass` with `beta=5` was chosen over `gap`/`gap-spectrum` after a 10-run
sweep in `../inverse/harmonic-prior-tuning.ipynb`: best balance of
correlation/AP/localization-error, beats sLORETA on every metric, and is the
easiest of the three to justify from the structure-function-coupling
literature (Abdelnour et al. 2018's exponential eigenvalue mapping) — one
citable equation, one tuned constant. Not up for re-litigation absent new
evidence.

## Latest completed run

`percentages=[0, 20, 40, 60, 80, 100]`, `n_repeats=1` — **334.3s total wall
time**, no errors, calibration (swap ceiling/`calibrated_n_swaps`,
brutal/dmri counts) all printed cleanly.

- Eigendecomposition dominates: 15 real evals (3 methods × 5 non-zero
  percentages), 264.8s total, **17.7s avg**.
- wCSET solves: 18, 1.27s avg. CSET solves: 18, 2.58s avg.
- Projected time for the full `n_repeats=5` study at this same percentage
  grid: **~28-30 min** (75 real eigendecompositions dominate). This is
  sensitive to system load — observed range across the session was
  18s-150s+ per eigendecomposition depending on load, so check `uptime`
  before launching and re-estimate if it looks busy.

## Open finding — not yet resolved

The `dmri` corruption method's reconstruction performance appears to
**increase again at higher severity levels** (non-monotonic vs. expectation).

Checked whether this is a recurrence of the earlier stale-connectome bug
(where `con` was regenerated independently of the `Simulator`'s saved
`HeadModel`, causing a corrupted connectome to spuriously outperform the
clean one). **It is not** — verified on 2026-07-14:

- `con` (used for corruption + to build `eig`) is **bit-identical** to the
  Simulator's own `conn_lengths.npz`.
- `eig` (the percentage=0 baseline) is **bit-identical** to a fresh
  `compute_cst(con_strength)` from that same `con`.
- `fwd["src"]` (desc="inv") and `simulator.hm.src` have identical dipole
  positions — localization error is measured in a consistent frame.
- `fwd` desc="inv" vs desc="sim" gain matrices differ by ~33% relative
  Frobenius norm, but this is confirmed **intentional** (avoid-inverse-crime
  design: simulator's internal forward matches desc="sim" exactly), not a
  stale file.
- All of run=1's `ave`/`cov`/`gold`/`source` and the `Simulator` itself were
  regenerated together on 2026-07-14, internally consistent with the
  July-8-dated `con`/`eig`/`fwd` checked above.

So all ground-truth-generating components are mutually consistent — the
dmri uptick looks like a genuine effect of that corruption method, not a
data-integrity artifact.

## Multi-run parallel pipeline (2026-07-16): wired, small-scale proof done, not yet scaled up

`sabotage_connectome.ipynb` was rewritten around a two-stage parallel design
(full design discussion in chat history) so the study can evaluate across
many simulated runs, not just `run="1"`, without exhausting memory on the
sweep's ~450MB-per-connectome dense arrays:

- **Stage 1** (`mm.analysis.connectivity.cache_cst_batch`, new — in
  `mindmap/analysis/connectivity/spectral.py`): corrupts + eigendecomposes
  every `(method, percentage, repeat)` connectome *in parallel*, each inside
  its own worker process, writing straight to
  `{mm_root}/sub-simf10/cst_cache/{label}.npz`. Never returns the big arrays
  to the main process, so peak memory is `n_jobs` connectomes at once,
  regardless of total sweep size (15-75+ for realistic configs). Thread-capped
  per worker via `threadpoolctl` (`n_threads_per_job` param) — necessary, not
  optional, since dense `eigh` multithreads internally by default and would
  otherwise oversubscribe cores together with `n_jobs`'s own process-level
  parallelism.
- **Stage 2** (`mm.analysis.functional.inverse.run_parallel_pipeline`,
  *unmodified* — same function `connectome_surrogates.ipynb` already uses):
  sequential loop over the now-cached connectomes; for each, a freshly-built
  `reconstruct_fn` closure (bakes in that connectome's `eigvec`/`eigval`) is
  handed to `run_parallel_pipeline`, which parallelizes reconstruction+scoring
  *across runs*. Both `wCSET` and `CSET` solved per run, same 5 metrics as
  before (correlation hrp/pp, AP hrp/pp, localization error hrp).

**Small-scale proof run** (2026-07-16, `methods`=all 3, `percentages=[0,100]`,
`n_repeats=1`, `runs=["1","2","3"]`, done deliberately small since another
parallel job was already running on this machine at the time):
correctness independently verified beforehand (bit-identical Stage 1 output
vs. sequential reference; exact match on a full Stage 1+2 pipeline result vs.
an independent sequential reference). The actual notebook run: Stage 1 =
543.3s for 3 connectomes (`n_jobs=2, n_threads_per_job=2`, deliberately
conservative), Stage 2 = 166.1s for 36 scored rows, no errors, resulting plot
sensible (correlation/AP decrease and localization error increases with
severity for `swap`/`brutal`; `dmri` shows less degradation, consistent with —
not new evidence for — the still-open anomaly below).

**Known gap, not yet addressed**: Stage 2 has no thread-capping (unlike Stage
1) — during the proof run this caused a bigger load spike (peak 41 on a
32-core machine, vs. Stage 1's ~24-27) even at tiny scale (`n_jobs=3`, one per
run). The other job's 20 workers stayed healthy throughout and `mpstat` showed
29% CPU still idle at the peak, so it was fine *this time*, but that headroom
won't necessarily be there at the real scale (`n_jobs=10` for `runs=1-10`).

**Before scaling up to a real sweep**:
1. Add thread-capping to Stage 2 (either inside `run_parallel_pipeline`
   itself, or by wrapping `reconstruct_fn`'s solver calls in
   `threadpool_limits`, matching Stage 1's approach) — do this before
   increasing `n_jobs`, not after hitting a problem.
2. Re-check `uptime`/whether the other job on this machine has finished —
   don't assume the headroom that existed for the tiny proof run still holds.
3. Scale the config up: full percentage grid (`[0,20,40,60,80,100]`), decide
   `n_repeats` (this subsumes the old "Next steps" item 2 below — with
   multiple runs now available, "how many repeats x how many runs" replaces
   the old single-run-only `n_repeats=5` question), and pick how many `runs`
   (`harmonic-prior-tuning.ipynb`'s convention was runs 1-10 for sub-simf10;
   worth reusing the same set for cross-notebook comparability).

## Next steps

1. **Investigate the dmri anomaly** before trusting/reporting that curve.
   Leading hypothesis: `dmri_uncertainty_weights_from_source_space` biases
   additions toward *short*-range edges and deletions toward *long*-range
   edges; at high severity this could systematically inflate low-eigenvalue
   spectral mass in a way that happens to align with `lowpass`'s own
   low-eigenvalue preference — i.e. the corruption method might be
   accidentally *reinforcing* the harmonic prior's assumptions rather than
   genuinely stressing them. Check: mean/median eigenvalue (or gap
   statistics) across `dmri` severity levels vs. `swap`/`brutal`, to confirm
   or rule this out.
2. ~~Once (1) is understood, decide whether to launch the full `n_repeats=5`
   sweep on the same percentage grid (~28-30 min projected).~~ Superseded by
   "Multi-run parallel pipeline" above — the question is now framed as
   repeats x runs, not repeats alone, and needs the Stage 2 thread-capping
   gap closed first.
3. **Test the Beltrami operator on the cortical mesh as a graph surrogate** —
   i.e. build the harmonic basis from mesh geometry (Laplace-Beltrami
   eigenmodes) instead of from the structural connectome, and compare
   robustness to `swap`/`brutal`/`dmri` corruption against the current
   connectome-eigenmode approach. Design rationale, decisions, and
   implementation status now live in their own note:
   [[beltrami-harmonics-notes]] (`../connectomic/beltrami_harmonics_notes.md`)
   — core function is implemented and verified as of 2026-07-16, beta tuned
   and settled at `beta=30`; not yet wired into this notebook's `wCSET`
   runs.
4. **Test Euclidean distance as a graph surrogate** — build a graph/weighting
   directly from dipole Euclidean distance rather than the structural
   connectome. Also relevant to (1): `dmri_uncertainty_weights_from_source_space`
   already uses Euclidean (not geodesic/tract-length) distance for its
   addition/deletion weights, so this surrogate would double as a sanity
   check on how much that approximation matters.
