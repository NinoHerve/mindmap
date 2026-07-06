"""
Modules for managing source_space objects
"""

from .io import (
    get_dipole_directions,
    get_dipole_positions,
    get_ndipole,
    get_geodist,
    get_hemi_labels,
)

from .process import (
    filter_dipoles_by_id_list,
    filter_dipoles_by_nifti_mask,
)

from .utils import correct_labels

__all__ = [
    "correct_labels",
    "get_dipole_directions",
    "get_dipole_positions",
    "get_ndipole",
    "get_geodist",
    "get_hemi_labels",
    "filter_dipoles_by_id_list",
    "filter_dipoles_by_nifti_mask",
]
