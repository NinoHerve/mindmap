"""
Surrogate-comparison pilot: sizes study 3 (2 methods x 6 connectome-construction
surrogates). No simulation step -- surrogates only change which connectome
wCSET/CSET use as their harmonic-prior source at reconstruction time, not
what ground truth gets simulated, so this reuses the existing 208-run
sub-simf10-pilot dataset (26 regions x 8 repetitions x SNR=10, from the
original 6-method pilot) directly.

Surrogates (6, matching connectome_surrogates.ipynb's recipe plus one new
one -- see connectome_surrogates_plan.md next to that notebook for the
empirical verification behind fused/geom_atlas's parameters):
  - true          -- the empirical connectome (already-computed `eig`)
  - fused         -- connectome-atlas.ipynb recipe (atlas upscaling +
                      geometric gap-fill)
  - fused_bin     -- binarized fused
  - geom_atlas    -- connectome_from_geometry alone (geodesic-distance KNN,
                      atlas params, homotopic interhemispheric mirroring)
  - geom_atlas_bin -- binarized geom_atlas
  - euclid        -- connectome_from_euclidean_distance (new -- same KNN
                      sparsification as geom_atlas but built from raw 3D
                      Euclidean distance instead of mesh-geodesic distance,
                      so no hemisphere-mirroring needed). No binary variant
                      (would make 7 surrogates, not 6).

2 methods x 6 surrogates = 12 conditions per run, tagged `condition` (e.g.
"wcset_true", "cset_euclid") in each checkpoint row -- reuses
`build_phase_data_hrp_pp_curve`/`score_source_reconstruction` unchanged from
the original pilot, just with a 12-way `solutions` dict per run instead of 6.

Checkpointed per run into `pilot_surrogate_metrics/` -- same crash-recovery
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
from mindmap.tasks import run_parallel_pipeline

N_CURVE_POINTS = 50
N_JOBS = 8

dm = mm.io.TutoManager("../../config.yml")
sub = "simf10"
sub_pilot = "simf10-pilot"

fwd_sim = dm.load(sub, "fwd", desc="sim")
fwd_inv = dm.load(sub, "fwd", desc="inv")
con = dm.load(sub, "con")
eig_true = dm.load(sub, "eig")
parc = dm.load(sub, "dseg", space="fs", segmentation="multiconn2022", scale="4")
ras2tkras = dm.load(sub, "ras2tkras")
probcon = dm.load_ref("REF_CON")

dipole_positions = mm.anatomy.source_space.get_dipole_positions(fwd_inv["src"]) * 1000
simulator_dir = dm.roots["sim"] / f"sub-{sub_pilot}" / "simulator"

# ---------------------------------------------------------------------------
# Step 0: verify data integrity (see memory `stale-connectome-bug` -- con/eig
# and a Simulator's own saved head-model state have silently diverged before)
# ---------------------------------------------------------------------------

conn_lengths = load_npz(simulator_dir / "conn_lengths.npz").toarray()
assert np.allclose(conn_lengths, con), "con does not match sub-simf10-pilot's own saved connectome!"

strength_true = np.divide(1, con, out=np.zeros_like(con), where=con > 0)
fresh_vec, fresh_val = mm.analysis.connectivity.compute_cst(strength_true)
assert np.allclose(fresh_val, eig_true["val"]), "eig val is not bit-identical to a fresh compute_cst!"
assert np.allclose(np.abs(fresh_vec), np.abs(eig_true["vec"])), "eig vec is not bit-identical to a fresh compute_cst!"
print("con/eig/Simulator are mutually consistent.")

# ---------------------------------------------------------------------------
# Step 1: build the 5 surrogate connectomes (+ true), at fwd_sim's resolution
# ---------------------------------------------------------------------------

consistency = np.nan_to_num(probcon["matrices"]["consistency"][:, :])
length = np.nan_to_num(probcon["matrices"]["length"][:, :])
number = np.nan_to_num(probcon["matrices"]["numbStlines"][:, :])

atlas_mask = np.bitwise_and(consistency > 65, length > 0)
atlas_mask = np.bitwise_and(atlas_mask, number > 0)
atlas = length * atlas_mask

con_atlas = mm.analysis.connectivity.connectome_from_atlas(
    fwd_sim["src"], parc, atlas, ras2tkras, sparsity=0.14, deterministic=True,
)
con_geom_fill = mm.analysis.connectivity.connectome_from_geometry(
    fwd_sim["src"], maximum_dist=22, maximum_neigh=35, weights=True,
)
con_fused = con_atlas.copy()
fill_idx = np.argwhere((con_geom_fill > 0) & (con_atlas == 0))
con_fused[fill_idx[:, 0], fill_idx[:, 1]] = con_geom_fill[fill_idx[:, 0], fill_idx[:, 1]]

con_geom_atlas = mm.analysis.connectivity.connectome_from_geometry(
    fwd_sim["src"], maximum_dist=22, maximum_neigh=35, weights=True, inter_hemi_mirror=True,
)
con_euclid = mm.analysis.connectivity.connectome_from_euclidean_distance(
    fwd_sim["src"], maximum_dist=22, maximum_neigh=35, weights=True,
)

surrogates = {
    "fused": con_fused,
    "fused_bin": (con_fused > 0).astype(float),
    "geom_atlas": con_geom_atlas,
    "geom_atlas_bin": (con_geom_atlas > 0).astype(float),
    "euclid": con_euclid,
}

# ---------------------------------------------------------------------------
# Step 2: eigenspectra (once, upfront -- not per reconstruction task)
# ---------------------------------------------------------------------------

connectomes = {"true": eig_true}
for label, con_surrogate in surrogates.items():
    strength = np.divide(1, con_surrogate, out=np.zeros_like(con_surrogate), where=con_surrogate > 0)
    eigvec, eigval = mm.analysis.connectivity.compute_cst(strength)
    connectomes[label] = {"vec": eigvec, "val": eigval}

print("Connectomes ready:", list(connectomes.keys()))

# ---------------------------------------------------------------------------
# Step 3: solve -- wCSET + CSET per connectome, per run
# ---------------------------------------------------------------------------

metrics_dir = Path("pilot_surrogate_metrics")
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
    for label, eig in connectomes.items():
        inv_wcset = mm.analysis.inverse.wCSET(
            forward=fwd_inv, noise_cov=ncov, depth=0.5,
            eigenvectors=eig["vec"], eigenvalues=eig["val"], harm=5, harmonic_method="lowpass",
        )
        solutions[f"wcset_{label}"] = inv_wcset.solve(data, lambda2_auto)

        inv_cset = mm.analysis.inverse.CSET(forward=fwd_inv, noise_cov=ncov, depth=0.8, eigenvectors=eig["vec"])
        solutions[f"cset_{label}"] = inv_cset.solve(data, lambda2=0.1)

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
        rows.append({"run": task.run, "condition": condition, **s})

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
