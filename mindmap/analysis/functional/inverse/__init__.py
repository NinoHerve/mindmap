"""
Modules for computing inverse solutions.
"""

from .solvers import (
    MNE,
    wCSET,
)

from .process import (
    normalize_source_estimate,
)

__all__ = [
    "MNE",
    "wCSET",
    "normalize_source_estimate",
]
