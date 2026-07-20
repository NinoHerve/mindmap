"""
Pilot study step 4a: generate the simulation dataset for the power-analysis
pilot. Reuses sub-simf10's cached forward/connectome/eeg-template artifacts,
same head model / epoch / propagation config as simulation_spike.ipynb.

Sources: the 26 regions selected in pilot_region_selection.tsv (stratified by
depth x orientation x lobe, see pilot_region_selection_rationale.md).
Repetitions: 8 (fresh stochastic noise draw each time), snr_target=10 (same
convention as simulation_spike.ipynb's per-region "Bonus" dataset).

Run once -- generate_simulation_dataset skips resimulating if simulator_dir
already exists, and each run's ave/cov/gold/source files are written to disk
immediately inside `_run_one` (see mindmap/analysis/simulation/pipeline.py),
so a crash partway through only loses the *remaining* runs, not completed
ones; simulation_index.tsv itself is only written after every run finishes,
so re-running this script is the recovery path if it does crash (finished
runs' output files already exist, only the index needs regenerating -- but
`generate_simulation_dataset` doesn't currently skip already-done runs, so a
genuine resume would require filtering `tasks` by existing output files first;
not needed for a single ~3 minute pilot run).
"""

import pandas as pd

import mindmap as mm

dm = mm.io.TutoManager("../../config.yml")
sub = "simf10"

fwd = dm.load(sub, "fwd", desc="sim")
con = dm.load(sub, "con")
eeg = dm.load(sub, "ave-temp")
parc = dm.load(sub, "dseg", space="fs", segmentation="multiconn2022", scale="4")
ras2tkras = dm.load(sub, "ras2tkras")

spike = mm.analysis.simulation.SpikeSignal(width=0.1, onset=0.5, repeat=1, scale=1e-8, event_id=1)

head_model = mm.analysis.simulation.HeadModel(
    sensors=eeg, forward=fwd, connectivity_weights=con > 0,
    connectivity_lengths=con, parcellation=parc, ras2tkras=ras2tkras,
)
epoch_config = mm.analysis.simulation.EpochConfig(
    tmin=-0.45, tmax=0.45, baseline=(-0.45, -0.2), signal_window=(-0.1, 0.45),
)
prop_config = mm.analysis.simulation.PropagationConfig(
    conduction_speed=30, nodal_delay=10, spatial_decay=0.5, jitter=1,
)
sim = mm.analysis.simulation.Simulator(head_model, epoch_config)

pilot_regions = pd.read_csv("pilot_region_selection.tsv", sep="\t")["region"].tolist()
sources = [mm.analysis.simulation.Source(r, spike) for r in pilot_regions]

mm.analysis.simulation.generate_simulation_dataset(
    dm=dm,
    subject="simf10-pilot",
    simulator=sim,
    sources=sources,
    prop_config=prop_config,
    repetitions=8,
    snr_targets=[10],
    n_jobs=8,
)
