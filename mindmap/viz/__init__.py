"""
Visualization tools for the MindMap package.

This sub-package contains plots for visualization outputs.
"""

from .volume import (
    show_slices,
    plot_segmentation,
    plot_bem,
)

from .surface import (
    plot_brain,
    plot_source_estimate,
)

from .sensor import (
    plot_ieeg,
    plot_channel_grid,
    plot_gfp_comparison,
    plot_alignment,
    plot_avg_ref,
    plot_baseline_topo,
    create_pipeline_report,
    plot_whitened_cov,
    plot_gfp,
)

from .graph import (
    plot_eigvec,
    plot_eigval,
    plot_con_summary,
    plot_spectrum,
)

from .forward import (
    plot_fwd_avg_ref,
    plot_fwd_sensitivity,
    plot_fwd_correlation,
    plot_fwd_singular_values,
    plot_leadfield,
)

from .inverse import(
    plot_predicted_data_covariance,
)

__all__ = [
    "show_slices",
    "plot_segmentation",
    "plot_bem",
    "plot_alignment",
    "plot_brain",
    "plot_source_estimate",
    "plot_eigvec",
    "plot_ieeg",
    "plot_channel_grid",
    "plot_gfp_comparison",
    "plot_avg_ref",
    "plot_baseline_topo",
    "create_pipeline_report",
    "plot_eigval",
    "plot_leadfield",
    "plot_con_summary",
    "plot_fwd_avg_ref",
    "plot_fwd_sensitivity",
    "plot_fwd_correlation",
    "plot_fwd_singular_values",
    "plot_whitened_cov",
    "plot_gfp",
    "plot_predicted_data_covariance",
    "plot_spectrum",
]
