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
    plot_leadfield,
)

from .sensor import (
    plot_ieeg,
    plot_channel_grid,
    plot_gfp_comparison,
    plot_alignment,
    plot_avg_ref,
    plot_baseline_topo,
    create_pipeline_report,
)

from .graph import (
    plot_eigvec,
    plot_eigval,
    plot_con_summary,
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
]