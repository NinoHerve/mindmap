"""
Module Name: analysis/functional/inverse/pipeline.py
Description: Helpers to run pipelines.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import mne

from mindmap.analysis.functional.inverse.solvers import (
    wCSET, CSET, MCE,
)
from mindmap.analysis.simulation.engine import Simulator
from mindmap.analysis.simulation.components import Source


@dataclass
class InverseTask:
    """
    Per-run inputs for `run_parallel_pipeline`. Plain paths and scalars only
    (no live `DatasetManager`/`Simulator` objects) so a list of these is safe
    to hand to `joblib.Parallel` worker processes.

    foutputs: only needed by a save-oriented `postprocess_fn` -- a mapping of
    method name to the output path it should be written to (precomputed by
    the caller, e.g. via `dm.get_path(...)`, since `dm` itself can't cross
    into a worker process).
    """
    run: int
    simulator_dir: Path
    fave: Path
    fcov: Path
    fgold: Path
    fsource: Path
    target_snr: float | None = None
    foutputs: dict[str, Path] | None = None


def run_parallel_pipeline(
    tasks: list[InverseTask],
    preprocess_fn,
    reconstruct_fn,
    postprocess_fn,
    n_jobs: int = 1,
    verbose: str = "warning",
) -> list:
    """
    Runs preprocess -> reconstruct -> postprocess for every task, optionally
    across parallel worker processes (`joblib.Parallel`, loky backend).

    Loading is hardcoded and always the same regardless of what the three
    stages do: each task's `simulator`/`ave`/`ncov`/`gold`/`src` are loaded
    from the paths on `task` (never via a live `DatasetManager`, which isn't
    picklable across worker processes -- `cannot pickle 'weakref.ReferenceType'
    object`). Everything else flows through one shared context dict, seeded
    with those raw objects plus the task itself:

        ctx = {"task": task, "simulator": ..., "ave": ..., "ncov": ...,
               "gold": ..., "src": ...}
        ctx.update(preprocess_fn(ctx) or {})
        ctx.update(reconstruct_fn(ctx) or {})
        return postprocess_fn(ctx)

    `preprocess_fn`, `reconstruct_fn`, and `postprocess_fn` are entirely
    caller-defined -- they read whatever keys they need from `ctx` and add
    whatever keys they produce (e.g. `preprocess_fn` might add "phase_data"/
    "hrp_idx"/"pp_idx", or just crop "ave"; `reconstruct_fn` adds "solutions";
    `postprocess_fn` either saves solutions to `task.foutputs` or scores them
    against `ctx["gold"]` and returns a metrics dict). Nothing beyond the
    initial seeded keys is required or enforced, so different stage
    combinations (e.g. a static crop vs. hrp/pp, or save vs. score) can be
    mixed freely without changing this function.

    Parameters
    ----------
    tasks: list[InverseTask]
        One entry per run.

    preprocess_fn, reconstruct_fn, postprocess_fn: Callable[[dict], dict | Any]
        See above.

    n_jobs: int
        Number of worker processes (1 = fully sequential).

    verbose: str
        MNE log level applied inside each worker (each is a separate process, so
        setting this in the main process/notebook wouldn't reach them). Applies
        to every MNE call made by `preprocess_fn`/`reconstruct_fn`/`postprocess_fn`
        too, as long as they don't pass their own `verbose=` override. Default
        "warning" -- pass "info" or "debug" for the usual noisier MNE output.

    Returns
    -------
    results: list
        One `postprocess_fn(ctx)` return value per task, in submission order.
    """
    def _run_one(task):
        mne.set_log_level(verbose)
        simulator, _, _ = Simulator.load(task.simulator_dir)
        ctx = {
            "task": task,
            "simulator": simulator,
            "ave": mne.read_evokeds(task.fave, verbose="warning")[0],
            "ncov": mne.read_cov(task.fcov, verbose="warning"),
            "gold": mne.read_source_estimate(task.fgold),
            "src": Source.load(task.fsource),
        }
        ctx.update(preprocess_fn(ctx) or {})
        ctx.update(reconstruct_fn(ctx) or {})
        return postprocess_fn(ctx)

    return Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(task) for task in tqdm(tasks, desc="pipeline")
    )

def make_process_sample(
    forward: mne.Forward,
    eig: np.ndarray = None,
    methods: list[str] = ["sloreta", "dspm"],
    lambda2: list[float] = [0.01, 0.01],
    crop: list[float] = [None, None],
):
    """
    Make a process_sample function for parallel processing.

    Parameters
    ----------
    methods: list[str]
        Source reconstruction methods.

    lambda2: list[float]
        List of regularization parameters.

    forward: mne.Forward
        Forward solution.

    ncov: mne.Covariance
        Noise covariance.

    eig: np.ndarray
        Eigenspectrum.

    crop: list[float]
        Cropping borders.
    """

    if ("wcset" in methods or "cset" in methods) and eig is None:
        raise ValueError("Can't run wCSET or CSET without eigenspectrum.")
    
    # standardize to lower case
    methods = [m.lower() for m in methods]

    # adjust casing
    mne_methods = {
        "sloreta": "sLORETA",
        "dspm": "dSPM",
        "mne": "MNE",
    }

    def process_sample(fave: str | Path, fcov: str | Path = None):
        """Process one sample."""
        # load
        data = mne.read_evokeds(fave, verbose="warning")[0]
        
        if fcov is None:
            ncov = mne.cov.make_ad_hoc_cov(data.info)
        else:
            ncov = mne.read_cov(fcov, verbose="warning")
        
        # crop
        data = data.copy().crop(tmin=crop[0], tmax=crop[1])

        # ----------------------
        # Source reconstruction
        # ----------------------

        if any(m in methods for m in ["sloreta", "dspm", "mne"]):
            inv_mne = mne.minimum_norm.make_inverse_operator(
                info = data.info,
                forward = forward,
                noise_cov = ncov,
                fixed = True,
                depth = None,
                verbose = "warning",
            )

        for i, m in enumerate(methods):
            sol = None

            if m == "cset":
                inv = CSET(
                    forward = forward,
                    noise_cov = ncov,
                    depth = 0.8,
                    eigenvectors = eig["vec"],
                )
                sol = inv.solve(data, lambda2 = lambda2[i])

            elif m == "wcset":
                inv = wCSET(
                    forward = forward,
                    noise_cov = ncov,
                    depth = 0.8,
                    eigenvectors = eig["vec"],
                    eigenvalues = eig["val"],
                    harm = 0.8,
                )
                sol = inv.solve(data, lambda2[i])

            elif m == "mce":
                inv = MCE(
                    forward = forward,
                    noise_cov = ncov,
                    depth = 0.8,
                )
                sol = inv.solve(data, lambda2 = lambda2[i])

            elif m in ["sloreta", "dspm", "mne"]: 
                sol = mne.minimum_norm.apply_inverse(
                    data,
                    inv_mne,
                    lambda2[i],
                    method = mne_methods[m],
                    verbose = "warning",
                )

            # save solution
            if sol is not None:
                foutput = str(fave)
                foutput = foutput.replace("ave", "stc")
                foutput = foutput.replace("_stc", f"_desc-{m}_stc")
                foutput = foutput.replace(".fif", "")

                sol.save(foutput, overwrite=True, verbose="warning")

        return f"Finished processing: {Path(fave).name}"

    return process_sample

