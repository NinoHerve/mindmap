"""
Module Name: analysis/connectivity/spectral.py
Description: Functions to apply spectral analysis on connectomes.

This module provides functions to apply spectral analysis on connectomes.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import pygsp
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm import tqdm


@dataclass
class ConnectomeCacheTask:
    """
    One entry for `cache_cst_batch`.

    label: str
        Unique identifier -- becomes the cache filename (`{label}.npz`).

    args: tuple
        Positional arguments passed to `cache_cst_batch`'s `corrupt_fn`, e.g.
        `(method, percentage, seed)` for a corruption sweep.
    """
    label: str
    args: tuple


def cache_cst_batch(
    tasks: list[ConnectomeCacheTask],
    corrupt_fn: Callable[..., np.ndarray],
    cache_dir: Path,
    n_jobs: int = 1,
    n_threads_per_job: int = 1,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Parallel, memory-bounded counterpart to calling `corrupt_fn` + `compute_cst`
    for many connectomes in a loop.

    Each task's connectome is built (`corrupt_fn(*task.args)`) and
    eigendecomposed (`compute_cst`) *inside its own worker process*, then
    written straight to `cache_dir/{label}.npz` (keys "vec"/"val") -- the raw
    connectome and its eigenvectors never travel back to the main process, so
    at most `n_jobs` connectome+eigenvector pairs (each ~450MB for a
    7,465-dipole connectome) are ever alive in memory at once, regardless of
    how many total tasks there are. This is what makes a large sweep (dozens+
    of corrupted connectomes) safe to run in parallel without exhausting
    memory the way holding every connectome/eigenbasis at once would.

    Already-cached labels are skipped unless `overwrite=True` -- makes this
    resumable after a partial run or crash, and cheap to re-invoke
    incrementally as a sweep's parameters are extended.

    `n_threads_per_job` caps each worker's BLAS/LAPACK thread count via
    `threadpoolctl`. `compute_cst`'s dense `eigh` multithreads internally by
    default; without this, `n_jobs` worker processes each spawning their own
    full-width BLAS thread pool would oversubscribe the machine's cores
    rather than truly parallelizing -- not just a performance nicety, but
    important for staying a good citizen on a shared/already-loaded machine.
    Total threads in flight is roughly `n_jobs * n_threads_per_job`; keep
    that comfortably under the available core count (check `uptime`/current
    load first, not just total core count, if anything else is running).

    Parameters
    ----------
    tasks: list[ConnectomeCacheTask]
        One entry per connectome to build and cache.

    corrupt_fn: Callable[..., np.ndarray]
        Builds one (already weight-transformed, e.g. length -> strength)
        connectome from `task.args`. Kept fully caller-defined -- typically a
        closure over the clean connectome plus whatever calibration data a
        specific corruption method needs (e.g. `dmri`'s distance-biased
        weights) -- so this function knows nothing about any particular
        corruption method's specifics.

    cache_dir: Path
        Directory `{label}.npz` files are written to (created if missing).

    n_jobs: int
        Number of worker processes (1 = fully sequential).

    n_threads_per_job: int
        BLAS/LAPACK threads allowed per worker -- see above.

    overwrite: bool
        Recompute and overwrite already-cached labels instead of skipping them.

    Returns
    -------
    dict[str, Path]
        `{label: cache_path}`, one entry per task in `tasks` (whether just
        computed this call or already cached from an earlier one).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(label):
        return cache_dir / f"{label}.npz"

    pending = [task for task in tasks if overwrite or not _cache_path(task.label).exists()]

    def _run_one(task):
        with threadpool_limits(limits=n_threads_per_job):
            connectome = corrupt_fn(*task.args)
            eigvec, eigval = compute_cst(connectome)
        np.savez(_cache_path(task.label), vec=eigvec, val=eigval)
        return task.label

    if pending:
        Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(task) for task in tqdm(pending, desc="caching connectome spectra")
        )

    return {task.label: _cache_path(task.label) for task in tasks}


def compute_cst(connectome : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the connectome spectrum transform: the eigendecomposition of the
    normalized graph Laplacian.

    Purely spectral -- takes whatever adjacency matrix it's handed. Any
    semantic transform of the weights (e.g. inverting lengths into
    strengths, `1/connectome`) is the caller's decision and must happen
    before calling this, not in here.

    Uses an explicit dense `np.linalg.eigh` on the Laplacian rather than
    pygsp's own `compute_fourier_basis` for directly-verifiable numerics,
    and forces the first eigenvalue to exactly 0 -- for a connected graph
    it's theoretically exactly 0, but `eigh` can return a tiny nonzero
    floating-point residual there instead.

    Parameters
    ----------
    connectome: np.ndarray (n,n)
        Connectome adjacency matrix.

    Returns
    -------
    eigvec: np.ndarray (n,n)
        Eigenvectors of the normalized graph Laplacian (columns), the
        connectome's spectral basis.

    eigval: np.ndarray (n,)
        Corresponding eigenvalues (spatial frequencies), ascending order.
    """
    graph = pygsp.graphs.Graph(connectome, lap_type="normalized")   # pylint: disable=no-member
    eigval, eigvec = np.linalg.eigh(graph.L.toarray())
    eigval[0] = 0

    return eigvec, eigval

