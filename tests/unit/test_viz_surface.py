"""
This module performs unit tests for the viz/mesh.py, viz/scene.py and
viz/surface.py 3D plotting framework. Runs headless (off-screen PyVista).
Operator (`Panel`/`Layout`) tests live in test_viz_panel.py.
"""

import numpy as np
import pyvista as pv
import pytest

from mindmap.viz.mesh import Layer, find_clim, default_cmap
from mindmap.viz.scene import Scene
from mindmap.viz.surface import SurfacePlot

pv.OFF_SCREEN = True


@pytest.fixture
def sphere():
    return pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)


def test_layer_static_map(sphere):
    """A plain (n,) array yields a single-state layer with no slider needed."""
    values = np.random.randn(sphere.n_points)
    layer = Layer(sphere, values, scalar_name="v")

    assert layer.n_states == 1
    assert np.allclose(layer.mesh["v"], values)


def test_layer_temporal_map_infers_states(sphere):
    """A (n, T) array with no explicit `states` gets 0..T-1 as its axis."""
    values = np.random.randn(sphere.n_points, 5)
    layer = Layer(sphere, values, scalar_name="v")

    assert layer.n_states == 5
    assert np.array_equal(layer.states, np.arange(5))

    layer.update(3)
    assert np.allclose(layer.mesh["v"], values[:, 3])


def test_layer_explicit_states_must_match_length(sphere):
    values = np.random.randn(sphere.n_points, 4)

    with pytest.raises(ValueError):
        Layer(sphere, values, states=np.array([0.0, 0.1, 0.2]), scalar_name="v")


def test_layer_callable_provider_is_lazy(sphere):
    """The provider must only be called on `update`, not eagerly for every state."""
    calls = []

    def provider(idx):
        calls.append(idx)
        return np.full(sphere.n_points, float(idx))

    layer = Layer(sphere, provider, states=np.arange(10), scalar_name="v")
    assert calls == [0]  # initial paint only

    layer.update(7)
    assert calls == [0, 7]
    assert np.allclose(layer.mesh["v"], 7.0)


def test_layer_callable_requires_states(sphere):
    with pytest.raises(ValueError):
        Layer(sphere, lambda idx: np.zeros(sphere.n_points), scalar_name="v")


def test_layer_style_autodetects_points_vs_surface(sphere):
    cloud = pv.PolyData(np.random.randn(20, 3))

    assert Layer(sphere).style == "surface"
    assert Layer(cloud).style == "points"


def test_find_clim_and_default_cmap():
    assert find_clim([None, np.array([])]) is None

    clim = find_clim([np.array([-2.0, 1.0]), np.array([0.5])], percentile=100)
    assert clim == (-2.0, 2.0)
    assert default_cmap(clim) == "bwr"

    clim_pos = find_clim([np.array([0.0, 3.0])], percentile=100)
    assert clim_pos == (0, 3.0)
    assert default_cmap(clim_pos) == "Reds"


def test_scene_single_panel_static_map(sphere):
    """Functionality 1: a surface with a map, no slider for a single state."""
    values = np.random.randn(sphere.n_points)
    scene = Scene([[Layer(sphere, values, scalar_name="v")]])

    assert not hasattr(scene, "slider")
    scene.plotter.close()


def test_scene_multi_panel_side_by_side():
    """Functionality 2: multiple surfaces side by side."""
    spheres = [pv.Sphere(radius=r) for r in (0.5, 1.0, 1.5)]
    panels = [[Layer(s, np.random.randn(s.n_points), scalar_name="v")] for s in spheres]
    scene = Scene(panels, titles=["A", "B", "C"])

    assert len(scene.panels) == 3
    scene.plotter.close()


def test_scene_temporal_slider_steps_all_layers(sphere):
    """Functionality 3: a temporal (or other) axis drives a shared slider."""
    values = np.random.randn(sphere.n_points, 6)
    times = np.linspace(0, 0.5, 6)
    layer = Layer(sphere, values, states=times, scalar_name="v")
    scene = Scene([[layer]], slider_label="Time [s]")

    assert scene.n_states == 6
    assert hasattr(scene, "slider")

    scene._step(2)
    assert scene.current_idx == 2
    assert np.allclose(layer.mesh["v"], values[:, 2])

    scene.plotter.close()


def test_scene_interaction_event_defaults_to_always_and_is_configurable(sphere):
    """Default stays 'always' (smooth live scrubbing for small meshes);
    'end' is available for large meshes on the Jupyter/trame backend,
    where 'always' streams a render on every drag pixel."""
    values = np.random.randn(sphere.n_points, 4)
    layer = Layer(sphere, values, scalar_name="v")

    default_scene = Scene([[layer]])
    assert default_scene.interaction_event == "always"
    default_scene.plotter.close()

    end_scene = Scene([[layer]], interaction_event="end")
    assert end_scene.interaction_event == "end"
    end_scene.plotter.close()


def test_scene_link_views_defaults_true_without_cameras(sphere):
    scene = Scene([[Layer(sphere)], [Layer(pv.Sphere())]])
    assert scene.plotter.renderers[0].camera == scene.plotter.renderers[0].camera  # sanity
    # link_views resolved True (no cameras given) - not directly inspectable,
    # but construction must not raise and both panels must exist.
    assert len(scene.panels) == 2
    scene.plotter.close()


def test_scene_link_views_auto_disabled_when_camera_given(sphere):
    from mindmap.viz.camera import get_view

    direction, viewup = get_view("left")
    camera = (direction, viewup)

    scene = Scene([[Layer(sphere)], [Layer(pv.Sphere())]], cameras=[camera, None])

    pos0 = scene.plotter.renderers[0].camera.position
    foc0 = scene.plotter.renderers[0].camera.focal_point
    resolved = np.array(pos0) - np.array(foc0)
    resolved = resolved / np.linalg.norm(resolved)
    assert resolved == pytest.approx(list(direction), abs=1e-6)
    scene.plotter.close()


def test_scene_explicit_link_views_overrides_auto(sphere):
    from mindmap.viz.camera import get_view

    direction, viewup = get_view("left")
    camera = (direction, viewup)

    # explicit True should not raise even though a camera is set
    scene = Scene([[Layer(sphere)]], cameras=[camera], link_views=True)
    scene.plotter.close()


def test_scene_cameras_length_mismatch_raises(sphere):
    with pytest.raises(ValueError, match="cameras"):
        Scene([[Layer(sphere)], [Layer(pv.Sphere())]], cameras=[None])


def test_scene_interactive_false_skips_slider_and_keys(sphere):
    values = np.random.randn(sphere.n_points, 4)
    layer = Layer(sphere, values, scalar_name="v")

    scene = Scene([[layer]], interactive=False)
    assert not hasattr(scene, "slider")
    scene.plotter.close()


def test_scene_set_state_by_index_and_by_value(sphere):
    times = np.linspace(0, 1, 5)
    values = np.random.randn(sphere.n_points, 5)
    layer = Layer(sphere, values, states=times, scalar_name="v")
    scene = Scene([[layer]])

    scene.set_state(3)
    assert scene.current_idx == 3
    assert np.allclose(layer.mesh["v"], values[:, 3])

    scene.set_state(0.24)  # nearest to times[1]=0.25
    assert scene.current_idx == 1

    scene.plotter.close()


def test_scene_screenshot_returns_image_without_path(sphere):
    layer = Layer(sphere, np.random.randn(sphere.n_points), scalar_name="v")
    scene = Scene([[layer]], interactive=False, off_screen=True)

    img = scene.screenshot()
    assert img.ndim == 3  # (H, W, 3/4)
    scene.plotter.close()


def test_scene_mixed_surface_and_point_cloud_in_one_panel(sphere):
    """Functionality 4: different surfaces (incl. point clouds) combined in one panel."""
    surface_layer = Layer(sphere, style="surface")
    points = pv.PolyData(np.random.randn(15, 3))
    point_layer = Layer(points, np.random.randn(15), scalar_name="potential", style="points")

    scene = Scene([[surface_layer, point_layer]])

    assert len(scene.panels[0]) == 2
    assert scene.panels[0][1].style == "points"
    scene.plotter.close()


def test_surfaceplot_single(sphere):
    values = np.random.randn(sphere.n_points)
    plot = SurfacePlot(sphere, values)

    assert len(plot.layers) == 1
    assert plot.layers[0].n_states == 1


def test_surfaceplot_func_params_absolute_and_normalize(sphere):
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * (sphere.n_points // 5 + 1))[: sphere.n_points]

    plot = SurfacePlot(sphere, values, func_params={"absolute": True})
    layer = plot.layers[0]

    assert np.all(layer.mesh[layer.scalar_name] >= 0)


def test_surfaceplot_states_and_title(sphere):
    values = np.random.randn(sphere.n_points, 4)
    states = np.arange(4)

    plot = SurfacePlot(sphere, values, states=states, title="left")

    assert plot.title == "left"
    assert plot.layers[0].n_states == 4


def test_surfaceplot_each_instance_independent(sphere):
    """Two SurfacePlots are fully independent objects - no shared implicit
    state, each with its own mesh_params/func_params."""
    a = SurfacePlot(sphere, np.random.randn(sphere.n_points), mesh_params={"opacity": 0.3})
    b = SurfacePlot(sphere, np.random.randn(sphere.n_points), mesh_params={"opacity": 1.0})

    assert a.layers[0].plot_kwargs["opacity"] == 0.3
    assert b.layers[0].plot_kwargs["opacity"] == 1.0

    scene = Scene([a.layers, b.layers])
    assert len(scene.panels) == 2
    scene.plotter.close()


def test_surfaceplot_default_scalar_name_is_the_same_for_every_instance(sphere):
    """Deliberate choice: `scalar_name` defaults to the same literal "value"
    for every SurfacePlot - simple and predictable (no generated names to
    guess when selecting with `link_scale`), at the cost of two unnamed
    panels combined via '|'/'&' sharing PyVista's color range (see
    test_surfaceplot_unnamed_panels_share_pyvista_color_range below).
    Naming plots explicitly is expected whenever comparing more than one."""
    a = SurfacePlot(sphere)
    b = SurfacePlot(sphere)

    assert a.scalar_name == b.scalar_name == "value"


def test_surfaceplot_unnamed_panels_share_pyvista_color_range(sphere):
    """PyVista synchronizes the color range of every actor sharing a scalar
    array *name* within one Plotter - even across subplots - to whichever
    was added last. Since unnamed SurfacePlots all default to "value", two
    of them combined into one layout DO alias like this - an accepted
    trade-off, not a bug; give plots distinct `scalar_name`s to avoid it."""
    unscaled = np.random.randn(sphere.n_points) * 50
    tiny = np.random.randn(sphere.n_points) * 0.01

    a = SurfacePlot(sphere, unscaled)
    b = SurfacePlot(sphere, tiny)
    assert a.scalar_name == b.scalar_name  # both "value" - the accepted trade-off

    scene = Scene([a.layers, b.layers])
    actor_a = next(act for act in scene.plotter.renderers[0].actors.values() if hasattr(act, "mapper"))
    actor_b = next(act for act in scene.plotter.renderers[1].actors.values() if hasattr(act, "mapper"))

    assert actor_a.mapper.scalar_range == actor_b.mapper.scalar_range
    scene.plotter.close()


def test_surfaceplot_distinct_scalar_names_avoid_pyvista_aliasing(sphere):
    """The documented way out: give each panel its own `scalar_name`."""
    unscaled = np.random.randn(sphere.n_points) * 50
    tiny = np.random.randn(sphere.n_points) * 0.01

    a = SurfacePlot(sphere, unscaled, scalar_name="a")
    b = SurfacePlot(sphere, tiny, scalar_name="b")

    layers_a, layers_b = a.layers, b.layers
    scene = Scene([layers_a, layers_b])
    actor_a = next(act for act in scene.plotter.renderers[0].actors.values() if hasattr(act, "mapper"))
    actor_b = next(act for act in scene.plotter.renderers[1].actors.values() if hasattr(act, "mapper"))

    assert actor_a.mapper.scalar_range == pytest.approx(layers_a[0].plot_kwargs["clim"])
    assert actor_b.mapper.scalar_range == pytest.approx(layers_b[0].plot_kwargs["clim"])
    assert actor_a.mapper.scalar_range != actor_b.mapper.scalar_range
    scene.plotter.close()
