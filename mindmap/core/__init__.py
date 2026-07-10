"""
Core module containing simple functions independent of heavy packages.
"""

from .metric import (
    avg_power,
    compute_distances,
    extract_voxel_volume,
    point_to_area_distance,
    calc_roc_curve,
    calc_pr_curve,
)

from .utils import (
    create_config,
    add_to_config,
    create_tmp_config,
    get_spike_extent_mask,
    get_spurious_mask,
    create_dataset_description,
)

from .io import (
    write_eig,
    write_csv_row,
    get_config,
)

from .process import (
    normalize_data,
    resample_pr_curve,
    resample_roc_curve,
)


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
    "create_dataset_description",
    "calc_roc_curve",
    "calc_pr_curve",
    "resample_pr_curve",
    "resample_roc_curve",
]
