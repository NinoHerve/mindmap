"""
Modules for computing inverse solutions.
"""

from .solvers import (
    MNE,
    wCSET,
)

from .process import (
    normalize_source_estimate,
    binarize_stc,
)

from .metric import (
    compute_stc_confusion,
    compute_stc_scores,
    compute_stc_curves,
)

from .io import (
    make_stc,
)

__all__ = [
    "MNE",
    "wCSET",
    "normalize_source_estimate",
    "binarize_stc",
    "compute_stc_confusion",
    "compute_stc_scores",
    "compute_stc_curves",
    "make_stc",
]
