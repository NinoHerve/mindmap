"""
Degradation pilot 2b: sizes connectome-draw count for study 2 (how many
independently-corrupted connectomes are needed at a given degradation
severity) -- the region x repetition question is handled separately by
pilot_degradation_02_solve.py. See that script's docstring for why these are
two independent pilots (an independence assumption between region- and
draw-variability) rather than one joint 3-way model.

No new simulation needed, same reasoning as the other pilots -- but restricted
to a SUBSET of the existing sub-simf10-pilot runs (all 26 regions x the first
3 of 8 repetitions = 78 runs, not the full 208) to control cost: this pilot's
per-run cost is `n_percentages x n_draws x 2 methods` solves instead of just
`n_percentages x 2` (pilot 2a's cost), since every draw needs its own
reconstruction, not just every severity level.

Corruption method: `brutal`, same recipe/reasoning as pilot_degradation_02_solve.py.

Severity levels: `[20, 40, 60, 80, 100]` -- `percentage=0` is deliberately
EXCLUDED here (unlike pilot 2a): at zero corruption every "draw" is identical
(no-op), so draw-to-draw variance is trivially zero there and degenerate for
a variance-component fit. Pilot 2a already covers `percentage=0` on its own.

Draws per severity: 5 (`n_repeats=5`, matching sabotage_connectome.ipynb's own
existing default -- no reason to deviate). 5 severities x 5 draws = 25
connectomes to cache.

2 methods x 5 severities x 5 draws = 50 conditions per run, tagged as
separate `method`/`percentage`/`draw` columns (not one combined string) --
the 2b analysis needs to slice by `percentage`, treat `draw` as the grouping
factor (playing the role `region` plays in every other pilot), and pool
`region`/`repetition` as replicate observations within each `(draw, method)`
cell. Reuses `build_phase_data_hrp_pp_curve`/`score_source_reconstruction`
unchanged.

Checkpointed per run into `pilot_degradation_draws_metrics/` -- same
crash-recovery behavior as the other pilots' solve scripts.
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
PERCENTAGES = [20, 40, 60, 80, 100]
N_DRAWS = 5
MAX_REPETITION = 3   # subset of sub-simf10-pilot's 8 repetitions, to control cost

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
# Step 1: brutal-corrupt at each severity x draw, cache
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
    ConnectomeCacheTask(label=f"brutal_pct{p}_draw{d}", args=(p, d))
    for p in PERCENTAGES
    for d in range(N_DRAWS)
]
print(f"{len(cache_tasks)} connectomes to cache.")
cache_paths = cache_cst_batch(cache_tasks, corrupt, cache_dir, n_jobs=4, n_threads_per_job=2)

connectomes = {}   # {(percentage, draw): {"vec":..., "val":...}}
for p in PERCENTAGES:
    for d in range(N_DRAWS):
        cached = np.load(cache_paths[f"brutal_pct{p}_draw{d}"])
        connectomes[(p, d)] = {"vec": cached["vec"], "val": cached["val"]}
print(f"Connectomes ready: {len(connectomes)} (percentage, draw) pairs.")

# ---------------------------------------------------------------------------
# Step 2: solve -- wCSET + CSET per (severity, draw), per run, on the subset
# ---------------------------------------------------------------------------

metrics_dir = Path("pilot_degradation_draws_metrics")
metrics_dir.mkdir(exist_ok=True)

df_index = pd.read_csv(dm.roots["sim"] / f"sub-{sub_pilot}" / "simulation_index.tsv", sep="\t")
df_index = df_index[df_index["repetition"] <= MAX_REPETITION].reset_index(drop=True)
print(f"{len(df_index)} runs selected (all regions x repetitions 1-{MAX_REPETITION}).")


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
    for (percentage, draw), eig in connectomes.items():
        inv_wcset = mm.analysis.inverse.wCSET(
            forward=fwd_inv, noise_cov=ncov, depth=0.5,
            eigenvectors=eig["vec"], eigenvalues=eig["val"], harm=5, harmonic_method="lowpass",
        )
        solutions[f"wcset_pct{percentage}_draw{draw}"] = inv_wcset.solve(data, lambda2_auto)

        inv_cset = mm.analysis.inverse.CSET(forward=fwd_inv, noise_cov=ncov, depth=0.8, eigenvectors=eig["vec"])
        solutions[f"cset_pct{percentage}_draw{draw}"] = inv_cset.solve(data, lambda2=0.1)

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
        method, rest = condition.split("_pct")
        pct_str, draw_str = rest.split("_draw")
        rows.append({
            "run": task.run, "method": method,
            "percentage": int(pct_str), "draw": int(draw_str), **s,
        })

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
