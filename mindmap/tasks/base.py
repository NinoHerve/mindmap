"""
Module Name: tasks/base.py
Description: Generic Task/run_parallel_pipeline framework shared across
analysis domains (inverse, simulation, ...).

Author: Nino Herve
Created: 2026-07-17
Last Modified: 2026-07-17
License: MIT
"""

from abc import ABC, abstractmethod
from pathlib import Path

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import mne


def read_ave(path: Path) -> mne.Evoked:
    return mne.read_evokeds(path, verbose="warning")[0]


def read_cov(path: Path) -> mne.Covariance:
    return mne.read_cov(path, verbose="warning")


def read_simulator(path: Path):
    # Local import: `mindmap.analysis.functional.inverse.pipeline` imports this
    # module before `mindmap.analysis.simulation` has ever been loaded (see
    # `mindmap/analysis/__init__.py`'s import order), and `simulation/pipeline.py`
    # itself imports from `mindmap.tasks` -- a top-of-file import here would
    # deadlock that cycle ("cannot import name 'Task' from partially initialized
    # module"). Deferring the import to call time (inside a worker process, long
    # after every module has finished loading) sidesteps it entirely.
    from mindmap.analysis.simulation.engine import Simulator
    simulator, _, _ = Simulator.load(path)
    return simulator


class Task(ABC):
    """
    Base contract for one run's worth of input to `run_parallel_pipeline`.

    Subclasses must be plain paths/scalars (no live `DatasetManager` or other
    unpicklable objects) so a list of them is safe to hand to
    `joblib.Parallel` worker processes -- `load()` runs inside the worker,
    after `self` has already been unpickled there.

    This is the extension point for task shapes not covered by the built-ins
    (`InverseTask`, `InverseSimTask`, `MeasureTask`, `SimulationTask`):
    subclass `Task`, implement `load()`, and pass a list of instances to
    `run_parallel_pipeline` like any other task.
    """

    @abstractmethod
    def load(self) -> dict:
        """Return the ctx-seeding dict for one run."""


def run_parallel_pipeline(
    tasks: list[Task],
    preprocess_fn,
    process_fn,
    postprocess_fn,
    n_jobs: int = 1,
    n_threads_per_job: int = None,
    verbose: str = "warning",
) -> list:
    """
    Runs load -> preprocess -> process -> postprocess for every task,
    optionally across parallel worker processes (`joblib.Parallel`, loky
    backend).

    Loading is owned by the task itself (`task.load()`, never a live
    `DatasetManager`, which isn't picklable across worker processes --
    `cannot pickle 'weakref.ReferenceType' object`). `InverseTask` loads just
    `ave`/`ncov` to solve with no ground truth involved; `InverseSimTask`
    additionally loads `simulator`/`gold`/`src` to validate a solver against
    the known source simulated data was generated from; `MeasureTask` loads
    previously saved solutions plus `gold` for scoring; `SimulationTask`
    loads the `simulator` a forward simulation runs against; a custom `Task`
    subclass can load whatever else a new use case needs. Everything else
    flows through one shared context dict, seeded with `task.load()`'s
    return value plus the task itself:

        ctx = {"task": task, **task.load()}
        ctx.update(preprocess_fn(ctx) or {})
        ctx.update(process_fn(ctx) or {})
        return postprocess_fn(ctx)

    `preprocess_fn`, `process_fn`, and `postprocess_fn` are entirely
    caller-defined -- they read whatever keys they need from `ctx` and add
    whatever keys they produce (e.g. `preprocess_fn` might add "phase_data"/
    "hrp_idx"/"pp_idx", or just crop "ave"; `process_fn` might solve an
    inverse problem and add "solutions", run a forward simulation and add
    "gold"/"raw", or do nothing if `task.load()` already loaded everything
    needed; `postprocess_fn` saves results and/or scores them against
    `ctx["gold"]` and returns a metrics/metadata dict). Nothing beyond the
    keys `task.load()` seeds is required or enforced, so different task types
    and stage combinations can be mixed freely without changing this
    function.

    Parameters
    ----------
    tasks: list[Task]
        One entry per run (`InverseTask`, `InverseSimTask`, `MeasureTask`,
        `SimulationTask`, or a custom `Task` subclass).

    preprocess_fn, process_fn, postprocess_fn: Callable[[dict], dict | Any]
        See above.

    n_jobs: int
        Number of worker processes (1 = fully sequential).

    n_threads_per_job: int, optional
        Caps each worker's BLAS/LAPACK thread count via `threadpoolctl`.
        Default `None` (no cap, current library defaults) -- **not safe at
        `n_jobs > 1`**: the linear-algebra calls inside `preprocess_fn`/
        `process_fn`/`postprocess_fn` (`wCSET`/`CSET`/MNE solves, mostly)
        multithread internally by default, so `n_jobs` worker processes each
        spawning their own full-width BLAS thread pool oversubscribes the
        machine's cores rather than truly parallelizing -- confirmed on the
        surrogate-comparison pilot (`n_jobs=8` on a 32-core machine): observed
        throughput was ~41s/run, vs. ~11s/run implied by dividing the
        measured single-threaded per-run cost by 8. Set this so
        `n_jobs * n_threads_per_job` stays comfortably under the available
        core count -- same pattern `cache_cst_batch` already applies to its
        own worker processes, just not previously wired into this function.

    verbose: str
        MNE log level applied inside each worker (each is a separate process, so
        setting this in the main process/notebook wouldn't reach them). Applies
        to every MNE call made by `preprocess_fn`/`process_fn`/`postprocess_fn`
        too, as long as they don't pass their own `verbose=` override. Default
        "warning" -- pass "info" or "debug" for the usual noisier MNE output.

    Returns
    -------
    results: list
        One `postprocess_fn(ctx)` return value per task, in submission order.
    """
    def _run_one(task: Task):
        mne.set_log_level(verbose)
        with threadpool_limits(limits=n_threads_per_job):
            ctx = {"task": task, **(task.load() or {})}
            ctx.update(preprocess_fn(ctx) or {})
            ctx.update(process_fn(ctx) or {})
            return postprocess_fn(ctx)

    return Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(task) for task in tqdm(tasks, desc="pipeline")
    )
