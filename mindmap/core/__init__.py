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
    calc_hypergeometric_enrichment,
    calc_fold_enrichment,
    label_clusters,
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
    read_npz_fast,
)

from .process import (
    normalize_data,
    paint_on_dense_mesh,
    resample_pr_curve,
    resample_roc_curve,
    precision_at_recall,
)

from .power import (
    simulate_splitplot_power,
    precision_of_mean,
)

from .stats import (
    paired_test,
    correct_pvalues,
    compare_to_reference,
    cluster_threshold,
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
    "read_npz_fast",
    "create_dataset_description",
    "calc_roc_curve",
    "calc_pr_curve",
    "calc_hypergeometric_enrichment",
    "calc_fold_enrichment",
    "label_clusters",
    "paint_on_dense_mesh",
    "resample_pr_curve",
    "resample_roc_curve",
    "precision_at_recall",
    "simulate_splitplot_power",
    "precision_of_mean",
    "paired_test",
    "correct_pvalues",
    "compare_to_reference",
    "cluster_threshold",
]
