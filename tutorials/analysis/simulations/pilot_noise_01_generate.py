"""
Noise-sweep pilot step 1: generate the simulation dataset for sizing study 1
(6 methods x 5 noise levels). Reuses sub-simf10's cached forward/connectome/
eeg-template artifacts, same head model / epoch / propagation config as
simulation_spike.ipynb -- and the same 26 regions as the original method-
comparison pilot (pilot_region_selection.tsv, stratified by depth x
orientation x lobe, see pilot_region_selection_rationale.md): region
stratification is about spatial coverage, not noise level, so there's no
reason to reselect it here.

SNR levels: [20, 10, 5, 2, 1] -- the existing noise-sweep convention from
simulation_spike.ipynb's "Noise/SNR sweep dataset" (`[50, 20, 10, 5, 2, 1,
None]`), minus `50` (the easiest level -- the original 6-method pilot found
the method main effect already saturated near 1.0 across the board, so the
highest-SNR condition is the least likely to be informative for detecting
method differences) and minus the noise-only `None` condition (inverse-
noisesweep.ipynb already excludes it for the same reason: AP/localization-
error are degenerate at pure noise).

Repetitions: 4 (down from the original pilot's 8) -- crossing 26 regions x 5
SNR levels already multiplies run count 5x over the original single-SNR
pilot, so repetitions are halved to keep the solve step's cost in the same
ballpark (~520 runs here vs. 208 there) while still giving the repetition
variance component a comfortable 26*5*(4-1)=390 pooled degrees of freedom
(more than the original pilot's 182).

Run once -- generate_simulation_dataset skips resimulating if simulator_dir
already exists, and each run's ave/cov/gold/source files are written to disk
immediately (see pilot_01_generate.py's docstring for the crash-recovery
details, identical here).
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
    subject="simf10-pilot-noise",
    simulator=sim,
    sources=sources,
    prop_config=prop_config,
    repetitions=4,
    snr_targets=[20, 10, 5, 2, 1],
    n_jobs=8,
)
