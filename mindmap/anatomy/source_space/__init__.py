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
    sample_nifti_at_points,
    sample_nifti_at_source_space,
    nifti_mask_to_points,
    build_dense_vertex_map,
    build_dipole_adjacency,
    build_dipole_patch_area,
)

from .metric import (
    average_dipole_spacing,
)

from .spectral import (
    compute_beltrami_harmonics,
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
    "sample_nifti_at_points",
    "sample_nifti_at_source_space",
    "nifti_mask_to_points",
    "build_dense_vertex_map",
    "build_dipole_adjacency",
    "build_dipole_patch_area",
    "average_dipole_spacing",
    "compute_beltrami_harmonics",
]
