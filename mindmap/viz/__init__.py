"""
Visualization tools for the MindMap package.

The 3D/mesh plotting side (brain surfaces, source spaces, sensors,
forward/inverse models, connectomes) is being rewritten from scratch -
the previous implementation is kept for reference in `viz/OLD/` but is no
longer imported here. 2D array/dataframe plotting (`core`, `power`) is
unaffected and still live; metric plots live in `MetricSuite.plot`
(`mindmap.metric.plot`).

`mesh`/`scene`/`panel`/`surface`/`brain`/`sensor`/`head`/`camera` are the new
3D framework:
- `Layer` (one mesh + an optional scalar field) and `Scene` (a multi-panel
  interactive plotter, grids included) are the low-level primitives.
- `Panel` typed shortcuts (`SurfacePlot`, `BrainPlot`, `SensorPlot`,
  `HeadModelPlot`) hold clean, named parameters for one plot; combine them
  with `+` (merge into one panel), `|` (side by side) or `&` (stack as a
  new row) into a `Layout`.
- `BrainPlot`/`HeadModelPlot` support a named `view` (see `viz.camera`) for
  a fixed, orthographic, auto-framed camera - `Layout.screenshot()` exports
  a fixed image with no slider/key widgets, for publication figures.

`eeg`/`report` cover EEG QC reporting: `eeg` holds small matplotlib
diagnostics (avg-ref check, GFP, whitened covariance, ...), and
`report.add_panel` bridges any `Panel`/`Layout` from this framework into
an `mne.Report` - `report.build_preprocessing_report` assembles the
standard per-subject report from both plus `mne.Report`'s own native
sections (`add_evokeds`, `add_covariance`, ...).
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

from .sensor import (
    SensorPlot,
)

from .head import (
    HeadModelPlot,
)

from .eeg import (
    plot_avg_ref,
    plot_baseline_topo,
    plot_interpolation,
    plot_whitened_cov,
    plot_gfp,
    plot_gfp_comparison,
)

from .report import (
    add_panel,
    build_preprocessing_report,
)

from .power import (
    plot_power_precision_curves,
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
    "SensorPlot",
    "HeadModelPlot",
    "plot_avg_ref",
    "plot_baseline_topo",
    "plot_interpolation",
    "plot_whitened_cov",
    "plot_gfp",
    "plot_gfp_comparison",
    "add_panel",
    "build_preprocessing_report",
    "plot_power_precision_curves",
]
