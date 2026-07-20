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
    build_phase_data_hrp_pp_curve,
)

from .metric import (
    compute_stc_confusion,
    compute_stc_scores,
    compute_stc_roc,
    compute_stc_pr,
    compute_stc_correlation,
    compute_stc_cosine,
    score_source_reconstruction,
)

from .io import (
    make_stc,
    filter_remaining_files,
    stream_source_solutions,
    stc_path_for,
    resolve_solution_paths,
)

from .pipeline import (
    Task,
    InverseTask,
    InverseSimTask,
    MeasureTask,
    run_parallel_pipeline,
)

__all__ = [
    "MNE",
    "wCSET",
    "MCE",
    "CSET",
    "normalize_source_estimate",
    "binarize_stc",
    "build_phase_data_hrp_pp_curve",
    "compute_stc_confusion",
    "compute_stc_scores",
    "compute_stc_roc",
    "compute_stc_pr",
    "make_stc",
    "compute_stc_correlation",
    "compute_stc_cosine",
    "score_source_reconstruction",
    "filter_remaining_files",
    "stream_source_solutions",
    "stc_path_for",
    "resolve_solution_paths",
    "Task",
    "InverseTask",
    "InverseSimTask",
    "MeasureTask",
    "run_parallel_pipeline",
]
