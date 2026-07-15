"""
DOC string
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import mne

from mindmap.io import DatasetManager
from mindmap.analysis.simulation.components import Source
from mindmap.analysis.simulation.engine import Simulator, PropagationConfig


def _zero_scale_source(reference: Source) -> Source:
    """
    Builds a zero-amplitude copy of a reference source's signal, for noise-only runs.
    Location is carried over only because `Source`/`Simulator.run` require one; it has
    no effect on the output since the propagated activity is exactly zero.
    """
    signal = copy.deepcopy(reference.signal)
    signal.scale = 0
    return Source(reference.loc, signal)


def _run_one(
    simulator_dir: Path,
    prop_config: PropagationConfig,
    src: Source,
    snr_target: float | None,
    count: int,
    rep: int,
    run_seed: int,
    fave: Path,
    fcov: Path,
    fgold: Path,
    fsource: Path,
) -> dict:
    """
    Runs and saves a single simulation run.

    Loads its own `Simulator` from `simulator_dir` rather than receiving the live
    object, and receives plain output file paths (precomputed in the main process,
    where building them is cheap) rather than the `DatasetManager` itself --
    `DatasetManager` is backed by a pybids `BIDSLayout` that isn't picklable across
    worker processes (`cannot pickle 'weakref.ReferenceType' object`), so it can't be
    passed through `joblib.Parallel` directly. `np.random.seed(run_seed)` mutates
    numpy's global RNG state, which is only safe here because each call to this
    function runs in its own process when parallelized -- there's no shared global
    state to race on across runs.
    """
    simulator, _, _ = Simulator.load(simulator_dir)
    epoch_config = simulator.epoch_config
    tmax_noise = epoch_config.baseline[1]

    # Set the run's unique deterministic seed
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
    src.save(fsource)

    # Region is meaningless for noise-only runs (zero-amplitude signal), so
    # record that explicitly instead of the carried-over placeholder location.
    source_label = "noise-only" if snr_target is None else \
        [str(item) for item in np.atleast_1d(src.loc)]

    # Record metrics, parameters, and targets for the tracking registry
    return {
        "run": count,
        "repetition": rep + 1,
        "source_index_list": source_label, # Tracks active brain region
        "signal_type": src.signal.name,
        "target_snr": snr_target,
        "conduction_speed": prop_config.conduction_speed,
        "spatial_decay": prop_config.spatial_decay,
        "run_seed": run_seed,
        "ave_file": Path(fave).name,
        "cov_file": Path(fcov).name,
        "gold_file": Path(fgold).name,
        "source_file": Path(fsource).name,
    }


def generate_simulation_dataset(
    dm: DatasetManager,
    subject: str,
    simulator: Simulator,
    sources: list[Source],
    prop_config: PropagationConfig,
    repetitions: int,
    snr_targets: list[float | None] = (10.0,),
    entities: dict = None,
    base_seed: int = 42,
    n_jobs: int = 1,
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
        Number of repetitions per source (and per noise-only run, if requested).

    snr_targets: list[float | None]
        Target Signal to noise ratio(s) for single trial. Each value is run against
        every source in `sources`. Include `None` to additionally generate a
        noise-only condition (zero-amplitude signal, no SNR calibration) run
        `repetitions` times total, independent of `sources` since source location
        has no effect when the signal amplitude is zero.

    base_seed:
        Seed randomizer.

    entities: dict
        Additional BIDS entities to put in file names.

    n_jobs: int
        Number of worker processes to run simulations in parallel (via
        `joblib.Parallel`). Each `(source, snr, repetition)` run's seed is drawn
        up front in submission order, so results are reproducible for a given
        `base_seed` regardless of `n_jobs`. Default 1 (fully sequential).
    """
    entities = entities or {}

    # 1. Establish the base directory
    subject_root = dm.roots["sim"]/f"sub-{subject}"
    subject_root.mkdir(parents=True, exist_ok=True)

    # 2. Save the Master Structural Simulator
    simulator_dir = subject_root/"simulator"
    if not simulator_dir.exists():
        simulator.save(simulator_dir, prop_config=prop_config)

    sources_dir = subject_root/"sources"
    sources_dir.mkdir(exist_ok=True)

    # Initialize reproducible random seed generator sequence
    rng = np.random.default_rng(base_seed)

    # 3. Build the full task list up front (cheap, deterministic) -- run_seed and
    # output paths are computed here (using `dm`, which can't itself cross into a
    # worker process -- see _run_one), so parallelizing execution below doesn't
    # change what each (source, snr, repetition) run computes, only when it runs.
    finite_snrs = [snr for snr in snr_targets if snr is not None]
    noise_only = None in snr_targets

    def _make_task(src, snr, count, rep, run_seed):
        fave = dm.get_path(subject, "ave", run=count, **entities)
        fcov = dm.get_path(subject, "cov", run=count, **entities)
        fgold = dm.get_path(subject, "gold", run=count, **entities)
        fsource = sources_dir/f"run-{count:03d}_source.json"
        return (src, snr, count, rep, run_seed, fave, fcov, fgold, fsource)

    tasks = []
    count = 0
    for src in sources:
        for snr in finite_snrs:
            for rep in range(repetitions):
                count += 1
                run_seed = int(rng.integers(0, 2**32 - 1))
                tasks.append(_make_task(src, snr, count, rep, run_seed))

    if noise_only:
        noise_src = _zero_scale_source(sources[0])
        for rep in range(repetitions):
            count += 1
            run_seed = int(rng.integers(0, 2**32 - 1))
            tasks.append(_make_task(noise_src, None, count, rep, run_seed))

    # 4. Run every task -- in parallel worker processes when n_jobs != 1. Each
    # worker loads its own Simulator from simulator_dir (see _run_one) rather than
    # receiving the live object, and sets np.random.seed in its own process, so
    # there's no cross-task race on numpy's global RNG state.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(simulator_dir, prop_config, *task)
        for task in tqdm(tasks, desc=f"sub-{subject}")
    )

    # 5. Export the central Simulation Index Table
    df_index = pd.DataFrame(results)
    df_index.to_csv(subject_root/"simulation_index.tsv", sep="\t", index=False, encoding="utf-8")

    print(f"Successfully generated dataset for sub-{subject}. Total runs compiled: {count}")
