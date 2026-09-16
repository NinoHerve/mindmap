"""
Visualization tools for the MindMap package.

The 3D/mesh plotting side (brain surfaces, source spaces, sensors,
forward/inverse models, connectomes) is being rewritten from scratch -
the previous implementation is kept for reference in `viz/OLD/` but is no
longer imported here. 2D array/dataframe plotting (`core`, `stats`,
`power`) is unaffected and still live.

`mesh`/`scene`/`panel`/`surface`/`brain` are the new 3D framework:
- `Layer` (one mesh + an optional scalar field) and `Scene` (a multi-panel
  interactive plotter) are the low-level primitives.
- `Panel` typed shortcuts (`SurfacePlot`, `BrainPlot`) hold clean, named
  parameters for one plot; combine them with `+` (merge into one panel),
  `|` (side by side) or `&` (stacked, data model ready - rendering not
  implemented yet) into a `Layout`.
"""

from .core import (
    plot_multi_scatterplots,
)

from .mesh import (
    Layer,
)

from .scene import (
    Scene,
)

from .panel import (
    Panel,
    CombinedPanel,
    Layout,
)

from .surface import (
    SurfacePlot,
)

from .brain import (
    BrainPlot,
)

from .power import (
    plot_power_precision_curves,
)

from .stats import (
    metric_box_significance,
    metric_relational_clusters,
    metric_relational_cluster_summary,
)

__all__ = [
    "plot_multi_scatterplots",
    "Layer",
    "Scene",
    "Panel",
    "CombinedPanel",
    "Layout",
    "SurfacePlot",
    "BrainPlot",
    "plot_power_precision_curves",
    "metric_box_significance",
    "metric_relational_clusters",
    "metric_relational_cluster_summary",
]
