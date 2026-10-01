"""
This module performs unit tests for viz/head.py: `HeadModelPlot`, the BEM
shell (inner_skull, outer_skull, outer_skin) plot shortcut. Runs headless
(off-screen PyVista).
"""

import pyvista as pv
import pytest

from mindmap.viz.head import HeadModelPlot
from mindmap.viz.brain import BrainPlot

pv.OFF_SCREEN = True


@pytest.fixture
def bem():
    return {
        "inner_skull": pv.Sphere(radius=0.8, theta_resolution=8, phi_resolution=8),
        "outer_skull": pv.Sphere(radius=0.9, theta_resolution=8, phi_resolution=8),
        "skin": pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8),
    }


def test_three_layers_one_per_shell(bem):
    plot = HeadModelPlot(bem)
    layers = plot.layers

    assert len(layers) == 3
    assert [l.mesh.n_points for l in layers] == [
        bem["inner_skull"].n_points,
        bem["outer_skull"].n_points,
        bem["skin"].n_points,
    ]


def test_default_style_is_wireframe_and_reaches_plot_kwargs(bem):
    """The actual bug this shortcut surfaced: `style="wireframe"` must
    reach add_mesh, not just be recorded on the Layer."""
    plot = HeadModelPlot(bem)

    for layer in plot.layers:
        assert layer.style == "wireframe"
        assert layer.plot_kwargs["style"] == "wireframe"


def test_default_colors_are_distinct(bem):
    plot = HeadModelPlot(bem)
    colors = [layer.plot_kwargs["color"] for layer in plot.layers]

    assert len(set(colors)) == 3


def test_default_shells_are_fully_opaque(bem):
    """Wireframe lines, not translucent fills - opacity=1.0 so the thin
    lines themselves stay crisp instead of fading out."""
    plot = HeadModelPlot(bem)

    for layer in plot.layers:
        assert layer.plot_kwargs["opacity"] == 1.0


def test_shell_params_are_overridable(bem):
    plot = HeadModelPlot(
        bem,
        inner_skull_params={"color": "black", "style": "surface", "opacity": 0.3},
    )
    inner, outer, skin = plot.layers

    assert inner.plot_kwargs["color"] == "black"
    assert inner.plot_kwargs["style"] == "surface"
    assert outer.plot_kwargs["style"] == "wireframe"  # untouched default
    assert skin.plot_kwargs["style"] == "wireframe"


def test_camera_none_without_view(bem):
    plot = HeadModelPlot(bem)
    assert plot.camera is None


def test_camera_absolute_view(bem):
    plot = HeadModelPlot(bem, view="anterior")
    direction, _ = plot.camera
    assert direction == (0.0, 1.0, 0.0)


def test_camera_lateral_medial_not_valid_for_whole_head(bem):
    """No hemisphere concept for whole-head shells - "lateral"/"medial"
    aren't in the named-view table `get_view` resolves against."""
    plot = HeadModelPlot(bem, view="lateral")

    with pytest.raises(ValueError):
        _ = plot.camera


def test_composes_with_brainplot_via_add(bem):
    brain = BrainPlot(pv.Sphere(radius=0.5))
    combined = brain + HeadModelPlot(bem)

    assert len(combined.layers) == 1 + 3
