"""
Degradation pilot 2a: sizes region x repetition for study 2 (2 methods x 6
connectome-degradation levels), given *one* representative corrupted
connectome per severity level -- the connectome-draw-count question itself
is handled separately by pilot_degradation_draws_02_solve.py (see that
script's docstring for why these are two separate pilots rather than one
joint model: an independence assumption between region-variability and
draw-variability, traded for reusing the existing 2-way power machinery
unchanged instead of building a new 3-way one).

No new simulation needed -- same reasoning as pilot_surrogate_02_solve.py:
degradation severity only changes which connectome wCSET/CSET use as their
harmonic-prior source at reconstruction time, not what ground truth gets
simulated. Reuses `sub-simf10-pilot`'s existing 208 runs directly.

Corruption method: `brutal` (`perturb_connectome_topology`, uniform random
edge deletions/additions, exact closed-form severity) -- not `dmri` (has an
unresolved non-monotonic performance anomaly at high severity, see memory
`sabotage-connectome-study`) or `swap` (needs empirical calibration, more
setup overhead for no benefit here). Same recipe as
`sabotage_connectome.ipynb`'s `corrupt("brutal", ...)`.

Severity levels: `[0, 20, 40, 60, 80, 100]` (percent of edges deleted +
re-added elsewhere), matching `sabotage_connectome.ipynb`'s own grid exactly
-- `percentage=0` is a no-op, reuses the clean `eig` directly, no caching
needed. One draw (repeat=0, fixed seed) per non-zero severity level, cached
via `cache_cst_batch` (same Stage-1 infrastructure `sabotage_connectome.ipynb`
already uses).

2 methods x 6 severities = 12 conditions per run, tagged as separate `method`/
`percentage` columns (not one combined `condition` string, unlike
pilot_surrogate_02_solve.py) -- the 2a analysis needs to slice by `percentage`
and fit `method` as its own factor per severity level, mirroring the noise
pilot's per-SNR loop, not a single joint 12-level fit. Reuses
`build_phase_data_hrp_pp_curve`/`score_source_reconstruction` unchanged.

Checkpointed per run into `pilot_degradation_metrics/` -- same crash-recovery
behavior as the other pilots' solve scripts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.sparse import load_npz

import mindmap as mm
from mindmap.analysis.functional.inverse import (
    InverseSimTask, build_phase_data_hrp_pp_curve, score_source_reconstruction,
)
from mindmap.analysis.connectivity import ConnectomeCacheTask, cache_cst_batch
from mindmap.tasks import run_parallel_pipeline

N_CURVE_POINTS = 50
N_JOBS = 8
PERCENTAGES = [0, 20, 40, 60, 80, 100]

dm = mm.io.TutoManager("../../config.yml")
sub = "simf10"
sub_pilot = "simf10-pilot"

fwd_inv = dm.load(sub, "fwd", desc="inv")
con = dm.load(sub, "con")
eig_true = dm.load(sub, "eig")

dipole_positions = mm.anatomy.source_space.get_dipole_positions(fwd_inv["src"]) * 1000
simulator_dir = dm.roots["sim"] / f"sub-{sub_pilot}" / "simulator"

# ---------------------------------------------------------------------------
# Step 0: verify data integrity (see memory `stale-connectome-bug`)
# ---------------------------------------------------------------------------

conn_lengths = load_npz(simulator_dir / "conn_lengths.npz").toarray()
assert np.allclose(conn_lengths, con), "con does not match sub-simf10-pilot's own saved connectome!"

con_strength = np.divide(1, con, out=np.zeros_like(con), where=con > 0)
fresh_vec, fresh_val = mm.analysis.connectivity.compute_cst(con_strength)
assert np.allclose(fresh_val, eig_true["val"]), "eig val is not bit-identical to a fresh compute_cst!"
assert np.allclose(np.abs(fresh_vec), np.abs(eig_true["vec"])), "eig vec is not bit-identical to a fresh compute_cst!"
print("con/eig/Simulator are mutually consistent.")

# ---------------------------------------------------------------------------
# Step 1: brutal-corrupt at each non-zero severity (one draw each), cache
# ---------------------------------------------------------------------------

triu_idx = np.triu_indices_from(con_strength, k=1)
total = int((con_strength[triu_idx] != 0).sum())
brutal_counts = {p: round(p / 100 * total) for p in PERCENTAGES}


def corrupt(percentage, seed):
    k = brutal_counts[percentage]
    return mm.analysis.connectivity.perturb_connectome_topology(
        con_strength, n_deletions=k, n_additions=k, seed=seed,
    )


cache_dir = dm.roots["mm"] / f"sub-{sub}" / "cst_cache"
cache_tasks = [
    ConnectomeCacheTask(label=f"brutal_pct{p}_rep0", args=(p, 0))
    for p in PERCENTAGES[1:]
]
print(f"{len(cache_tasks)} connectomes to cache: {[t.label for t in cache_tasks]}")
cache_paths = cache_cst_batch(cache_tasks, corrupt, cache_dir, n_jobs=4, n_threads_per_job=2)

connectomes = {0: eig_true}
for p in PERCENTAGES[1:]:
    cached = np.load(cache_paths[f"brutal_pct{p}_rep0"])
    connectomes[p] = {"vec": cached["vec"], "val": cached["val"]}
print("Connectomes ready for severities:", list(connectomes.keys()))

# ---------------------------------------------------------------------------
# Step 2: solve -- wCSET + CSET per severity level, per run
# ---------------------------------------------------------------------------

metrics_dir = Path("pilot_degradation_metrics")
metrics_dir.mkdir(exist_ok=True)

df_index = pd.read_csv(dm.roots["sim"] / f"sub-{sub_pilot}" / "simulation_index.tsv", sep="\t")


def preprocess_curve(ctx):
    return build_phase_data_hrp_pp_curve(
        ctx["ave"], ctx["gold"], ctx["simulator"], n_curve_points=N_CURVE_POINTS,
    )


def reconstruct(ctx):
    data, ncov = ctx["phase_data"], ctx["ncov"]

    snr, _ = mm.analysis.eeg.estimate_evoked_snr(
        evoked=ctx["ave"], baseline=ctx["simulator"].epoch_config.baseline, noise_cov=ncov)
    lambda2_auto = max(0.01, 1.0 / (snr.max() ** 2))

    solutions = {}
    for percentage, eig in connectomes.items():
        inv_wcset = mm.analysis.inverse.wCSET(
            forward=fwd_inv, noise_cov=ncov, depth=0.5,
            eigenvectors=eig["vec"], eigenvalues=eig["val"], harm=5, harmonic_method="lowpass",
        )
        solutions[f"wcset_pct{percentage}"] = inv_wcset.solve(data, lambda2_auto)

        inv_cset = mm.analysis.inverse.CSET(forward=fwd_inv, noise_cov=ncov, depth=0.8, eigenvectors=eig["vec"])
        solutions[f"cset_pct{percentage}"] = inv_cset.solve(data, lambda2=0.1)

    return {"solutions": solutions}


def postprocess_and_checkpoint(ctx):
    gold, task, src = ctx["gold"], ctx["task"], ctx["src"]

    true_source_mask = src.select_targets(
        ctx["simulator"].hm.src, ctx["simulator"].hm.parc, ctx["simulator"].hm.ras2tkras,
    )

    scores = score_source_reconstruction(
        gold=gold,
        solutions=ctx["solutions"],
        hrp_idx=ctx["hrp_idx"], pp_idx=ctx["pp_idx"], curve_idx=ctx["curve_idx"],
        hrp_col=ctx["hrp_col"], pp_col=ctx["pp_col"], curve_cols=ctx["curve_cols"],
        true_source_mask=true_source_mask, dipole_positions=dipole_positions,
    )

    rows = []
    curves = {}
    for condition, s in scores.items():
        curves[condition] = s.pop("correlation_curve")
        method, _, pct = condition.partition("_pct")
        rows.append({"run": task.run, "method": method, "percentage": int(pct), **s})

    with open(metrics_dir / f"run-{task.run:03d}.json", "w") as f:
        json.dump(rows, f)
    np.savez(metrics_dir / f"run-{task.run:03d}_curves.npz", **curves)

    return rows


tasks = []
for _, row in df_index.iterrows():
    run = int(row["run"])
    if (metrics_dir / f"run-{run:03d}.json").exists():
        continue     # checkpoint hit -- already solved
    tasks.append(InverseSimTask(
        run=run,
        simulator_dir=simulator_dir,
        fave=dm.get_path(sub_pilot, "ave", run=run),
        fcov=dm.get_path(sub_pilot, "cov", run=run),
        fgold=dm.get_path(sub_pilot, "gold", run=run),
        fsource=dm.get_path(sub_pilot, "source", run=run),
        target_snr=row["target_snr"],
    ))

print(f"{len(tasks)} runs left to solve (of {len(df_index)} total; "
      f"{len(df_index) - len(tasks)} already checkpointed).")

if tasks:
    run_parallel_pipeline(tasks, preprocess_curve, reconstruct, postprocess_and_checkpoint, n_jobs=N_JOBS, n_threads_per_job=1)

print("Done.")
