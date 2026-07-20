"""
Utilities for sizing simulation-based pilot studies: stratified region
selection, checkpoint aggregation, and mixed-effects variance-component
fitting used to drive `mindmap.core.power`'s Monte Carlo power/precision
grid.
"""

from .region_selection import compute_region_geometry, stratified_region_selection
from .aggregate import load_pilot_checkpoints
from .variance import fit_variance_components

__all__ = [
    "compute_region_geometry",
    "stratified_region_selection",
    "load_pilot_checkpoints",
    "fit_variance_components",
]
