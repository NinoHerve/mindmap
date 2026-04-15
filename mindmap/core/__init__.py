"""
Core module containing simple functions independent of heavy packages.
"""

from .metric import (
    avg_power,
    compute_distances,
    extract_voxel_volume,
    point_to_area_distance,
)

from .utils import (
    create_config,
    add_to_config,
    create_tmp_config,
    get_spike_extent_mask,
    get_spurious_mask,
)

from .io import (
    write_eig,
    write_csv_row,
    get_config,
)

from .process import normalize_data


__all__ = [
    "avg_power",
    "compute_distances",
    "extract_voxel_volume",
    "point_to_area_distance",
    "normalize_data",
    "create_config",
    "add_to_config",
    "create_tmp_config",
    "get_spike_extent_mask",
    "get_spurious_mask",
    "write_eig",
    "write_csv_row",
    "get_config",
]
