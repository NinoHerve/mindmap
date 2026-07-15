"""
Modules for computing inverse solutions.
"""

from .solvers import (
    MNE,
    wCSET,
    MCE,
    CSET,
)

from .process import (
    normalize_source_estimate,
    binarize_stc,
)

from .metric import (
    compute_stc_confusion,
    compute_stc_scores,
    compute_stc_roc,
    compute_stc_pr,
    compute_stc_correlation,
    compute_stc_cosine,
)

from .io import (
    make_stc,
    filter_remaining_files,
    stream_source_solutions,
)

from .pipeline import (
    make_process_sample,
    InverseTask,
    run_parallel_pipeline,
)

__all__ = [
    "MNE",
    "wCSET",
    "MCE",
    "CSET",
    "normalize_source_estimate",
    "binarize_stc",
    "compute_stc_confusion",
    "compute_stc_scores",
    "compute_stc_roc",
    "compute_stc_pr",
    "make_stc",
    "compute_stc_correlation",
    "compute_stc_cosine",
    "make_process_sample",
    "filter_remaining_files",
    "stream_source_solutions",
    "InverseTask",
    "run_parallel_pipeline",
]
