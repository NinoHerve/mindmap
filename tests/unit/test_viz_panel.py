"""
This module performs unit tests for viz/panel.py: the `Panel`/`CombinedPanel`/
`Layout` classes and the `+`/`|`/`&` operators that combine them.
Runs headless (off-screen PyVista).
"""

import warnings

import numpy as np
import pyvista as pv
import pytest

from mindmap.viz.panel import CombinedPanel, Layout
from mindmap.viz.surface import SurfacePlot

pv.OFF_SCREEN = True


@pytest.fixture
def sphere():
    return pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)


def _plot(mesh, **kwargs):
    return SurfacePlot(mesh, np.random.randn(mesh.n_points), **kwargs)


def test_or_combines_two_panels_side_by_side(sphere):
    a, b = _plot(sphere, title="A"), _plot(sphere, title="B")
    layout = a | b

    assert isinstance(layout, Layout)
    assert layout.rows == [[a, b]]

    scene = layout.build_scene()
    assert len(scene.panels) == 2
    scene.plotter.close()


def test_or_chains_three_panels(sphere):
    a, b, c = _plot(sphere), _plot(sphere), _plot(sphere)
    layout = a | b | c

    assert layout.rows == [[a, b, c]]


def test_plus_merges_two_panels_into_one(sphere):
    a, b = _plot(sphere), _plot(sphere)
    combined = a + b

    assert isinstance(combined, CombinedPanel)
    assert len(combined.layers) == 2  # both layers in ONE panel


def test_plus_chains_flatten_instead_of_nesting(sphere):
    a, b, c = _plot(sphere), _plot(sphere), _plot(sphere)
    combined = a + b + c

    assert isinstance(combined, CombinedPanel)
    assert combined.parts == [a, b, c]   # flat, not CombinedPanel([CombinedPanel([a,b]), c])
    assert len(combined.layers) == 3


def test_plus_title_defaults_to_first_nonempty_part():
    sphere = pv.Sphere()
    a = _plot(sphere, title="")
    b = _plot(sphere, title="brain")

    assert (a + b).title == "brain"
    assert (b + a).title == "brain"


def test_plus_title_explicit_override(sphere):
    combined = CombinedPanel([_plot(sphere, title="A"), _plot(sphere, title="B")], title="custom")
    assert combined.title == "custom"


def test_precedence_plus_binds_tighter_than_or(sphere):
    """`a + b | c` must parse as `(a + b) | c`, matching Python's native
    operator precedence (+  >  &  >  |), with no parentheses needed."""
    a, b, c = _plot(sphere), _plot(sphere), _plot(sphere)

    result = a + b | c
    expected = (a + b) | c

    assert isinstance(result, Layout)
    assert len(result.rows[0]) == 2                 # (a+b) is one panel, c is another
    assert isinstance(result.rows[0][0], CombinedPanel)
    assert result.rows[0][1] is c
    assert result.rows == expected.rows


def test_layout_plus_panel_broadcasts_to_every_panel(sphere):
    """The N-vs-1 case: adding a single panel to a multi-panel Layout merges
    it into EVERY panel, keeping the same number of panels."""
    a, b = _plot(sphere, title="A"), _plot(sphere, title="B")
    extra = _plot(sphere, title="extra")

    layout = (a | b) + extra

    assert isinstance(layout, Layout)
    assert len(layout.rows[0]) == 2                       # still 2 panels, not 3
    assert all(isinstance(p, CombinedPanel) for p in layout.rows[0])
    assert all(len(p.layers) == 2 for p in layout.rows[0])  # each got `extra` merged in


def test_panel_plus_layout_broadcasts_too(sphere):
    """Broadcast works in either operand order (`extra + layout`)."""
    a, b = _plot(sphere), _plot(sphere)
    extra = _plot(sphere)

    layout = extra + (a | b)

    assert len(layout.rows[0]) == 2
    assert all(len(p.layers) == 2 for p in layout.rows[0])


def test_layout_plus_layout_raises(sphere):
    a, b = _plot(sphere), _plot(sphere)
    layout1 = a | b
    layout2 = _plot(sphere) | _plot(sphere)

    with pytest.raises(TypeError):
        layout1 + layout2


def test_and_stacks_as_a_new_row(sphere):
    a, b, c = _plot(sphere), _plot(sphere), _plot(sphere)
    layout = (a | b) & c

    assert layout.rows == [[a, b], [c]]


def test_multirow_layout_builds_a_grid_scene(sphere):
    a, b = _plot(sphere), _plot(sphere)
    layout = a & b

    scene = layout.build_scene()
    assert scene.shape == (2, 1)
    assert len(scene.panels) == 2
    scene.plotter.close()


def test_2x2_grid_places_panels_row_major(sphere):
    a, b, c, d = (_plot(sphere, title=t) for t in "abcd")
    layout = (a | b) & (c | d)
    assert layout.rows == [[a, b], [c, d]]

    scene = layout.build_scene()
    assert scene.shape == (2, 2)
    assert len(scene.plotter.renderers) == 4
    # row-major: index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    assert scene.plotter.renderers[0].actors  # sanity: something was added
    scene.plotter.close()


def test_grid_slider_updates_every_panel(sphere):
    """The shared state slider must still step all panels uniformly once
    they're arranged in a multi-row grid, not just a single row."""
    times = np.linspace(0, 1, 5)
    a = SurfacePlot(sphere.copy(), np.random.randn(sphere.n_points, 5), states=times, title="a")
    b = SurfacePlot(sphere.copy(), np.random.randn(sphere.n_points, 5), states=times, title="b")
    layout = a & b

    scene = layout.build_scene(slider_label="Time [s]")
    assert scene.n_states == 5

    scene._step(2)
    assert scene.current_idx == 2
    for panel_layers in scene.panels:
        assert np.allclose(panel_layers[0].mesh[panel_layers[0].scalar_name], panel_layers[0]._provider(2))

    scene.plotter.close()


def test_ragged_rows_raise_on_build_scene(sphere):
    a, b, c = _plot(sphere), _plot(sphere), _plot(sphere)
    layout = Layout([[a, b], [c]])  # built by hand: mismatched row lengths

    with pytest.raises(ValueError, match="same number of panels"):
        layout.build_scene()


def test_panel_show_renders_single_panel(sphere):
    plot = _plot(sphere)
    plot.show()  # off-screen, just must not raise


def test_show_forwards_jupyter_backend_to_plotter_show(sphere, monkeypatch):
    """`jupyter_backend` (and any other pv.Plotter.show() kwarg) must reach
    the real plotter.show() call - construction-time kwargs (slider_label)
    must NOT leak into it."""
    captured = {}
    original_show = pv.Plotter.show

    def fake_show(self, **kwargs):
        captured.update(kwargs)
        return original_show(self)  # off-screen no-op, just exercises the real call

    monkeypatch.setattr(pv.Plotter, "show", fake_show)

    plot = _plot(sphere)
    plot.show(jupyter_backend="none", slider_label="Time [s]")

    assert captured == {"jupyter_backend": "none"}


def test_show_without_jupyter_backend_passes_no_extra_kwargs(sphere, monkeypatch):
    captured = {}
    original_show = pv.Plotter.show

    def fake_show(self, **kwargs):
        captured.update(kwargs)
        return original_show(self)

    monkeypatch.setattr(pv.Plotter, "show", fake_show)

    plot = _plot(sphere)
    plot.show()

    assert captured == {}


# ---------------------------
# Layout.link_scale
# ---------------------------


def test_link_scale_no_match_raises(sphere):
    a = _plot(sphere, scalar_name="Source Activity")
    layout = a | _plot(sphere, scalar_name="Source Activity")

    with pytest.raises(ValueError, match="no layer named"):
        layout.link_scale("Nonexistent")


def test_link_scale_single_match_warns_and_is_noop(sphere):
    a = _plot(sphere, scalar_name="Source Activity", mesh_params={"clim": (0, 1)})
    b = _plot(sphere, scalar_name="EEG Potential")
    layout = a | b

    with pytest.warns(UserWarning, match="only one layer named"):
        layout.link_scale("Source Activity")

    assert a.mesh_params["clim"] == (0, 1)  # untouched


def test_link_scale_single_match_with_explicit_override_is_silent(sphere):
    a = _plot(sphere, scalar_name="Source Activity")
    layout = a | _plot(sphere, scalar_name="EEG Potential")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        layout.link_scale("Source Activity", clim=(0, 5), cmap="Reds")

    assert a.mesh_params["clim"] == (0, 5)
    assert a.mesh_params["cmap"] == "Reds"


def test_link_scale_two_matches_auto_infers_shared_range(sphere):
    a = SurfacePlot(sphere, np.array([-1.0, 1.0] * (sphere.n_points // 2)), scalar_name="Source Activity")
    b = SurfacePlot(sphere, np.array([-5.0, 5.0] * (sphere.n_points // 2)), scalar_name="Source Activity")
    layout = a | b

    layout.link_scale("Source Activity")

    assert a.mesh_params["clim"] == (-5.0, 5.0)
    assert b.mesh_params["clim"] == (-5.0, 5.0)
    assert a.mesh_params["cmap"] == b.mesh_params["cmap"] == "bwr"


def test_link_scale_only_touches_matching_name(sphere):
    """The brain+EEG vs brain+iEEG scenario: link only the brain layers."""
    brain_a = _plot(sphere, scalar_name="Source Activity", title="A")
    eeg_a = _plot(sphere, scalar_name="EEG Potential", title="A")
    brain_b = _plot(sphere, scalar_name="Source Activity", title="B")
    ieeg_b = _plot(sphere, scalar_name="iEEG Potential", title="B")

    eeg_a.mesh_params["clim"] = (0, 1)
    ieeg_b.mesh_params["clim"] = (0, 2)

    layout = (brain_a + eeg_a) | (brain_b + ieeg_b)
    layout.link_scale("Source Activity")

    assert brain_a.mesh_params.get("clim") == brain_b.mesh_params.get("clim")
    assert eeg_a.mesh_params["clim"] == (0, 1)     # untouched
    assert ieeg_b.mesh_params["clim"] == (0, 2)    # untouched


def test_link_scale_explicit_override_applies_to_all_matches(sphere):
    a = _plot(sphere, scalar_name="Source Activity")
    b = _plot(sphere, scalar_name="Source Activity")
    layout = a | b

    layout.link_scale("Source Activity", clim=(0, 10), cmap="viridis")

    assert a.mesh_params["clim"] == b.mesh_params["clim"] == (0, 10)
    assert a.mesh_params["cmap"] == b.mesh_params["cmap"] == "viridis"


def test_link_scale_all_callable_matches_without_override_raises(sphere):
    provider = lambda idx: np.zeros(sphere.n_points)  # noqa: E731
    a = SurfacePlot(sphere, provider, states=np.arange(3), scalar_name="Source Activity")
    b = SurfacePlot(sphere, provider, states=np.arange(3), scalar_name="Source Activity")
    layout = a | b

    with pytest.raises(ValueError, match="callable provider"):
        layout.link_scale("Source Activity")


def test_link_scale_mixed_callable_and_array_infers_from_arrays_only(sphere):
    provider = lambda idx: np.zeros(sphere.n_points)  # noqa: E731
    a = SurfacePlot(sphere, provider, states=np.arange(3), scalar_name="Source Activity")
    b = SurfacePlot(sphere, np.array([-3.0, 3.0] * (sphere.n_points // 2)), scalar_name="Source Activity")
    layout = a | b

    layout.link_scale("Source Activity")

    assert a.mesh_params["clim"] == (-3.0, 3.0)   # applied even though inferred from b only
    assert b.mesh_params["clim"] == (-3.0, 3.0)


def test_link_scale_is_chainable_and_mutates_in_place(sphere):
    a = _plot(sphere, scalar_name="Source Activity")
    b = _plot(sphere, scalar_name="Source Activity")
    c = _plot(sphere, scalar_name="EEG Potential")
    d = _plot(sphere, scalar_name="EEG Potential")
    layout = (a + c) | (b + d)

    result = layout.link_scale("Source Activity").link_scale("EEG Potential", clim=(0, 1))

    assert result is layout
    assert a.mesh_params.get("clim") == b.mesh_params.get("clim")
    assert c.mesh_params["clim"] == d.mesh_params["clim"] == (0, 1)


# ---------------------------
# screenshot
# ---------------------------


def test_panel_screenshot_returns_image(sphere):
    img = _plot(sphere).screenshot()
    assert img.ndim == 3


def test_layout_screenshot_writes_a_file(sphere, tmp_path):
    path = tmp_path / "out.png"
    a, b = _plot(sphere), _plot(sphere)

    (a | b).screenshot(path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_screenshot_with_state_renders_the_requested_timepoint(sphere):
    """Different `state=` values must actually change what gets rendered -
    verified through the public API by comparing the resulting images."""
    times = np.linspace(0, 1, 5)
    values = np.random.randn(sphere.n_points, 5) * 100  # large so the visual difference is unambiguous
    plot = SurfacePlot(sphere, values, states=times, scalar_name="v", mesh_params={"clim": (-300, 300)})

    img_first = plot.screenshot(state=0)
    img_last = plot.screenshot(state=4)

    assert not np.array_equal(img_first, img_last)


def test_screenshot_with_brainplot_camera(sphere):
    from mindmap.viz.brain import BrainPlot

    plot = BrainPlot(sphere, hemi="lh", view="lateral")
    img = plot.screenshot()
    assert img.ndim == 3
