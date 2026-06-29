"""
DOC string
"""

from pathlib import Path

import numpy as np
import pandas as pd

import mne

from mindmap.io import DatasetManager
from mindmap.analysis.simulation.components import Source
from mindmap.analysis.simulation.engine import Simulator, PropagationConfig


def generate_simulation_dataset(
    dm: DatasetManager,
    subject: str,
    simulator: Simulator,
    sources: list[Source],
    prop_config: PropagationConfig,
    repetitions: int,
    entities: dict = None,
    snr_target: float = 10.0,
    base_seed: int = 42,
):
    """
    Run simulations for given sources and save the data in a folder including
    metadata to reproduce the simulation.

    Parameters
    ----------
    dm: DatasetManager
        Mindmap's DatasetManager to handle folder and file naming.

    subject: str
        Name for the subject.

    simulator: Simulator
        Mindmap's Simulator object.

    sources: list[Source]
        Source of each simulation.

    prop_config: PropagationConfig
        Propagation parameters.

    repetitions: int
        Number of repetitions per source.

    snr_target:
        Target Signal to noise ratio for single trial.

    base_seed: 
        Seed randomizer.
    
    entities: dict
        Additional BIDS entities to put in file names.
    """
    entities = entities or {}

    # 1. Establish the base directory
    subject_root = dm.roots["sim"]/f"sub-{subject}"
    subject_root.mkdir(parents=True, exist_ok=True)

    # 2. Save the Master Structural Simulator
    simulator_dir = subject_root/"simulator"
    if not simulator_dir.exists():
        simulator.save(simulator_dir, prop_config=prop_config)

    # Prepare Structural holders for the tracking index table
    index_rows = []
    sources_dir = subject_root/"sources"
    sources_dir.mkdir(exist_ok=True)

    # Cache configurations for speed
    tmax_noise = simulator.epoch_config.baseline[1]
    epoch_config = simulator.epoch_config

    # Initialize reproducible random seed generator sequence
    rng = np.random.default_rng(base_seed)

    count = 0
    for src in sources:
        for rep in range(repetitions):
            count += 1

            # Draw a unique deterministic seed for the specific run cycle
            run_seed = int(rng.integers(0, 2**32 - 1 ))
            np.random.seed(run_seed)

            # ---------------
            # Run simulation
            # ---------------
            gold, raw = simulator.run(src, prop_config, snr_target=snr_target)

            # -----------------------------------------
            # Process EEG (Referencing, Epoching, Cov)
            # -----------------------------------------
            raw = raw.set_eeg_reference(
                ref_channels = "average",
                projection = True,
                verbose = "warning",
            )

            events = mne.events_from_annotations(raw, verbose="warning")[0]
            epochs = mne.Epochs(
                raw,
                events,
                tmin = epoch_config.tmin,
                tmax = epoch_config.tmax,
                baseline = epoch_config.baseline,
                verbose = "warning",
            )

            noise_cov = mne.compute_covariance(
                epochs,
                tmax = tmax_noise,
                method = "empirical",
                verbose = "warning",
            )
            ave = epochs.average()

            # -------------
            # Save results
            # -------------
            fave = dm.get_path(subject, "ave", run=count, **entities)
            fcov = dm.get_path(subject, "cov", run=count, **entities)
            fgold = dm.get_path(subject, "gold", run=count, **entities)

            # Ensure subdirectories exist
            for fpath in [fave, fcov, fgold]:
                fpath.parent.mkdir(parents=True, exist_ok=True)

            ave.save(fave, verbose="warning")
            noise_cov.save(fcov, verbose="warning")
            gold.save(fgold, verbose="warning")

            # ------------------------------------------
            # Save Source State & Populate Metadata Log
            # ------------------------------------------
            # Save the flat JSON file describing the active source variant for this run
            run_src_name = f"run-{count:03d}_source.json"
            src.save(sources_dir/run_src_name)

            # Record metrics, parameters, and targets inside the tracking registry
            index_rows.append({
                "run": count,
                "repetition": rep + 1,
                "source_index_list": [str(item) for item in np.atleast_1d(src.loc)], # Tracks active brain region
                "signal_type": src.signal.name,
                "target_snr": snr_target,
                "conduction_speed": prop_config.conduction_speed,
                "spatial_decay": prop_config.spatial_decay,
                "run_seed": run_seed,
                "ave_file": Path(fave).name,
                "cov_file": Path(fcov).name,
                "gold_file": Path(fgold).name,
                "source_file": run_src_name,
            })
    
    # 3. Export the central Simulation Index Table
    df_index = pd.DataFrame(index_rows)
    df_index.to_csv(subject_root/"simulation_index.tsv", sep="\t", index=False, encoding="utf-8")

    print(f"Successfully generated dataset for sub-{subject}. Total runs compiled: {count}")
