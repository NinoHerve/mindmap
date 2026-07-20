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

import mne

from mindmap.analysis.simulation.components import Source
from mindmap.tasks import Task, run_parallel_pipeline, read_ave, read_cov, read_simulator


@dataclass
class InverseSimTask(Task):
    """
    Solve-and-validate task for simulated data: loads `ave`/`ncov`/`gold`/
    `src`/`simulator` and hands them to a `preprocess_fn`/`process_fn`/
    `postprocess_fn` chain that reconstructs source estimates and can score
    them against the known `gold` standard. Use this when the data came from
    `SimulationTask`/`generate_simulation_dataset` and you want to benchmark
    a solver against the ground truth it was generated from -- for solving
    real recordings with no ground truth available, use `InverseTask`.

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

    def load(self) -> dict:
        return {
            "simulator": read_simulator(self.simulator_dir),
            "ave": read_ave(self.fave),
            "ncov": read_cov(self.fcov),
            "gold": mne.read_source_estimate(self.fgold),
            "src": Source.load(self.fsource),
        }


@dataclass
class MeasureTask(Task):
    """
    Read-and-score task: loads previously saved source estimates (e.g. from
    an `InverseSimTask` run's save-oriented `postprocess_fn`) plus `gold`,
    for pipelines that only need to compute metrics -- no `preprocess_fn`/
    `process_fn` needs to run a solver.

    fave/fcov/simulator_dir are optional and only needed if `preprocess_fn`
    has to recompute something from the original sensor data (e.g. hrp/pp
    phase indices, or a crop window) rather than working off `gold` alone.
    """
    run: int
    fgold: Path
    fsolutions: dict[str, Path]
    fave: Path | None = None
    fcov: Path | None = None
    simulator_dir: Path | None = None

    def load(self) -> dict:
        ctx = {
            "gold": mne.read_source_estimate(self.fgold),
            "solutions": {
                m: mne.read_source_estimate(p) for m, p in self.fsolutions.items()
            },
        }
        if self.fave is not None:
            ctx["ave"] = read_ave(self.fave)
        if self.fcov is not None:
            ctx["ncov"] = read_cov(self.fcov)
        if self.simulator_dir is not None:
            ctx["simulator"] = read_simulator(self.simulator_dir)
        return ctx


@dataclass
class InverseTask(Task):
    """
    General-purpose solve task: "I have `ave`/`ncov`, solve and save" --
    no ground truth or simulator involved, works on any recording, real or
    simulated. For simulated data where you also want to validate a solver
    against the known source it was generated from, use `InverseSimTask`
    instead.

    fcov: optional -- if omitted, falls back to an ad-hoc noise covariance
    built from `ave.info`.

    foutputs: only needed by a save-oriented `postprocess_fn` -- a mapping of
    method name to the output path it should be written to (e.g. built via
    `resolve_solution_paths(fave, methods)`).
    """
    fave: Path
    fcov: Path | None = None
    foutputs: dict[str, Path] | None = None

    def load(self) -> dict:
        ave = read_ave(self.fave)
        ncov = read_cov(self.fcov) if self.fcov is not None else mne.cov.make_ad_hoc_cov(ave.info)
        return {"ave": ave, "ncov": ncov}

