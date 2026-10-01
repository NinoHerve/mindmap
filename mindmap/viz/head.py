"""
Module Name: viz/head.py
Description: `HeadModelPlot` - the BEM shell plot shortcut (inner_skull,
outer_skull, outer_skin - exactly what `read_bem` returns).

Deliberately single-purpose: no brain surface field. Getting the brain in
the picture is `BrainPlot(...) + HeadModelPlot(bem)` - the same
"independent shortcuts, combine them" pattern the rest of this framework
follows, rather than duplicating surface-handling logic here.

Author: Nino Herve
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional

from mindmap.viz.mesh import Layer
from mindmap.viz.panel import Panel
from mindmap.viz.camera import get_view


@dataclass
class HeadModelPlot(Panel):
    """
    The 3 BEM shells (inner_skull, outer_skull, outer_skin) as one panel,
    wireframe by default so they never occlude whatever's composed with
    them (electrodes via `+`, a `BrainPlot` underneath, ...) - the same
    reasoning the old `plot_bem`'s freeview edge-coloring used.

    Parameters
    ----------
    bem: dict
        `{"inner_skull": pv.PolyData, "outer_skull": pv.PolyData, "skin": pv.PolyData}`
        - exactly what `mindmap.anatomy.forward.io.read_bem` returns.

    title: str
        Panel title.

    view: str, optional
        A named camera view (see `viz.camera.get_view`) - an absolute
        anatomical direction ("anterior", "superior", "left", "right", ...).
        No hemisphere concept applies to whole-head shells, so
        "lateral"/"medial" aren't valid here (unlike `BrainPlot`) - use
        "left"/"right" directly; passing them raises the same clear error
        `get_view` gives for any unknown view name.

    inner_skull_params, outer_skull_params, skin_params: dict
        Passed to each shell's `Layer`. Default to thin, fully-opaque
        wireframe in distinct colors (softened from the old freeview
        convention's pure red/green/blue).
    """

    bem: dict
    title: str = ""
    view: Optional[str] = None
    inner_skull_params: dict = field(
        default_factory=lambda: {"color": "crimson", "style": "wireframe", "opacity": 1.0, "line_width": 1}
    )
    outer_skull_params: dict = field(
        default_factory=lambda: {"color": "seagreen", "style": "wireframe", "opacity": 1.0, "line_width": 1}
    )
    skin_params: dict = field(
        default_factory=lambda: {"color": "steelblue", "style": "wireframe", "opacity": 1.0, "line_width": 1}
    )

    @property
    def camera(self) -> Optional[tuple]:
        return None if self.view is None else get_view(self.view)

    @property
    def layers(self) -> list[Layer]:
        return [
            Layer(self.bem["inner_skull"], **self.inner_skull_params),
            Layer(self.bem["outer_skull"], **self.outer_skull_params),
            Layer(self.bem["skin"], **self.skin_params),
        ]
