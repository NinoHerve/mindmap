"""
Modules for managing connectivity.
"""

from .process import (
    connectome_from_tracts,
    connectome_from_geometry,
    connectome_from_euclidean_distance,
    connectome_from_atlas,
)

from .spectral import (
    compute_cst,
    cache_cst_batch,
    ConnectomeCacheTask,
)

from .utils import (
    order_connectome_by_region,
    get_dipole_region_labels,
    get_transformed_dipoles,
)

from .metric import (
    density,
    small_world_sigma,
    rich_club_curve,
    modularity,
    distance_decay_fit_from_source_space,
    hemispheric_symmetry_from_source_space,
    block_conservation_from_atlas,
    rewire_connectome,
    swap_rewire_connectome,
    perturb_connectome_topology,
    dmri_uncertainty_weights_from_source_space,
    perturb_connectome_weights,
    permute_connectome_weights,
)

__all__ = [
    "connectome_from_tracts",
    "connectome_from_geometry",
    "connectome_from_euclidean_distance",
    "connectome_from_atlas",
    "compute_cst",
    "cache_cst_batch",
    "ConnectomeCacheTask",
    "order_connectome_by_region",
    "get_dipole_region_labels",
    "get_transformed_dipoles",
    "density",
    "small_world_sigma",
    "rich_club_curve",
    "modularity",
    "distance_decay_fit_from_source_space",
    "hemispheric_symmetry_from_source_space",
    "block_conservation_from_atlas",
    "rewire_connectome",
    "swap_rewire_connectome",
    "perturb_connectome_topology",
    "dmri_uncertainty_weights_from_source_space",
    "perturb_connectome_weights",
    "permute_connectome_weights",
]