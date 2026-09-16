"""
This module performs unit tests for viz/brain.py: `BrainPlot`, its primary
(direct-surface) constructor and its `from_source_space` alternate
constructor. Runs headless (off-screen PyVista).
"""

from pathlib import Path

import numpy as np
import pyvista as pv
import mne
import pytest

from mindmap.viz.brain import BrainPlot
from mindmap.viz.panel import CombinedPanel, Layout
from mindmap.geometry.io import surf_to_polydata
from mindmap.anatomy.source_space.process import build_dense_vertex_map
from mindmap.core.process import paint_on_dense_mesh

pv.OFF_SCREEN = True

SRC_PATH = Path("data/derivatives/mindmap/sub-intra12/src/sub-intra12_src.fif")


@pytest.fixture
def spheres():
    return pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10), \
           pv.Sphere(radius=1.0, center=(3, 0, 0), theta_resolution=10, phi_resolution=10)


@pytest.fixture(scope="module")
def src():
    if not SRC_PATH.exists():
        pytest.skip(f"real source space fixture not available: {SRC_PATH}")
    return mne.read_source_spaces(str(SRC_PATH), verbose="error")


# ---------------------------
# Primary constructor
# ---------------------------


def test_single_surface_no_values(spheres):
    sphere_l, _ = spheres
    plot = BrainPlot(sphere_l)

    assert len(plot.layers) == 1
    assert plot.layers[0].scalar_name is None


def test_single_surface_is_normalized_to_a_list(spheres):
    sphere_l, _ = spheres
    plot = BrainPlot(sphere_l, np.random.randn(sphere_l.n_points))

    assert isinstance(plot.surfaces, list)
    assert len(plot.surfaces) == 1


def test_vertices_faces_tuple_is_converted(spheres):
    sphere_l, _ = spheres
    plot = BrainPlot((sphere_l.points, sphere_l.regular_faces))

    assert isinstance(plot.surfaces[0], pv.PolyData)
    assert plot.surfaces[0].n_points == sphere_l.n_points


def test_two_surfaces_share_one_color_scale(spheres):
    """Unlike SurfacePlot, both hemispheres of one BrainPlot are the same
    quantity - they must share a clim/cmap by default, not each get their
    own independent range."""
    sphere_l, sphere_r = spheres
    plot = BrainPlot(
        [sphere_l, sphere_r],
        [np.random.randn(sphere_l.n_points), np.random.randn(sphere_r.n_points) * 5],
    )
    layer_l, layer_r = plot.layers

    assert layer_l.plot_kwargs["clim"] == layer_r.plot_kwargs["clim"]
    assert layer_l.plot_kwargs["cmap"] == layer_r.plot_kwargs["cmap"]


def test_scalar_name_shared_across_surfaces(spheres):
    sphere_l, sphere_r = spheres
    plot = BrainPlot(
        [sphere_l, sphere_r],
        [np.random.randn(sphere_l.n_points), np.random.randn(sphere_r.n_points)],
        scalar_name="Source Activity",
    )
    layer_l, layer_r = plot.layers

    assert layer_l.scalar_name == layer_r.scalar_name == "Source Activity"


def test_values_list_length_must_match_surfaces():
    sphere_l = pv.Sphere()
    sphere_r = pv.Sphere(center=(3, 0, 0))

    with pytest.raises(ValueError):
        BrainPlot([sphere_l, sphere_r], values=[np.random.randn(sphere_l.n_points)])


def test_func_params_applied_before_shared_clim(spheres):
    sphere_l, sphere_r = spheres
    signed = np.array([-2.0, 2.0] * (sphere_l.n_points // 2 + 1))[: sphere_l.n_points]
    plot = BrainPlot(
        [sphere_l, sphere_r],
        [signed, signed],
        func_params={"absolute": True},
    )
    layer_l, layer_r = plot.layers

    assert np.all(layer_l.mesh[layer_l.scalar_name] >= 0)
    assert layer_l.plot_kwargs["clim"][0] == 0  # absolute -> one-sided clim


def test_brainplot_composes_with_operators(spheres):
    sphere_l, sphere_r = spheres
    brain_a = BrainPlot(sphere_l, np.random.randn(sphere_l.n_points), title="A")
    brain_b = BrainPlot(sphere_r, np.random.randn(sphere_r.n_points), title="B")

    layout = brain_a | brain_b
    assert isinstance(layout, Layout)

    combined = brain_a + brain_b
    assert isinstance(combined, CombinedPanel)
    assert len(combined.layers) == 2

    scene = layout.build_scene()
    assert len(scene.panels) == 2
    scene.plotter.close()


def test_brainplot_link_scale(spheres):
    sphere_l, sphere_r = spheres
    a = BrainPlot(sphere_l, np.random.randn(sphere_l.n_points), scalar_name="Source Activity")
    b = BrainPlot(sphere_r, np.random.randn(sphere_r.n_points) * 5, scalar_name="Source Activity")

    layout = a | b
    layout.link_scale("Source Activity")

    assert a.mesh_params.get("clim") == b.mesh_params.get("clim")


# ---------------------------
# from_source_space
# ---------------------------


def test_from_source_space_is_an_alternate_constructor_not_a_method(src):
    """Never called on an existing instance - it's a classmethod that
    derives surfaces/values from `src` and calls the same __init__."""
    plot = BrainPlot.from_source_space(src, hemi="lh")
    assert isinstance(plot, BrainPlot)
    assert len(plot.surfaces) == 1


def test_from_source_space_matches_manual_inpainting(src):
    """Regression test against the exact dipole->dense mapping the
    tutorial builds by hand: build_dense_vertex_map + paint_on_dense_mesh."""
    np.random.seed(0)
    n_dipoles = src[0]["nuse"] + src[1]["nuse"]
    dipole_data = np.random.randn(n_dipoles, 5)
    times = np.linspace(0, 0.4, 5)
    split = src[0]["nuse"]

    map_l = build_dense_vertex_map(src[0])
    map_r = build_dense_vertex_map(src[1])

    def manual_l(idx):
        return paint_on_dense_mesh(dipole_data[:split, idx], map_l)

    def manual_r(idx):
        return paint_on_dense_mesh(dipole_data[split:, idx], map_r)

    plot = BrainPlot.from_source_space(src, dipole_data, states=times, hemi="both")
    layer_l, layer_r = plot.layers

    for idx in range(5):
        layer_l.update(idx)
        layer_r.update(idx)
        assert np.allclose(layer_l.mesh[layer_l.scalar_name], manual_l(idx))
        assert np.allclose(layer_r.mesh[layer_r.scalar_name], manual_r(idx))


def test_from_source_space_single_hemisphere(src):
    """The original motivating bug fix: hemi='lh' builds exactly one
    surface/layer, using only that hemisphere's dipole data."""
    np.random.seed(1)
    n_dipoles = src[0]["nuse"] + src[1]["nuse"]
    dipole_data = np.random.randn(n_dipoles, 3)
    times = np.arange(3)
    split = src[0]["nuse"]
    map_l = build_dense_vertex_map(src[0])

    plot = BrainPlot.from_source_space(src, dipole_data, states=times, hemi="lh")
    assert len(plot.surfaces) == 1
    assert len(plot.layers) == 1

    layer = plot.layers[0]
    layer.update(1)
    assert np.allclose(layer.mesh[layer.scalar_name], paint_on_dense_mesh(dipole_data[:split, 1], map_l))


def test_from_source_space_always_lazy_even_for_plain_array(src):
    """The inpainting itself is expensive for a long time series - values
    must be wrapped as a lazy per-state provider even when the caller
    passed a plain array, not just when they already passed a callable."""
    calls = []
    n_dipoles = src[0]["nuse"]

    class TrackedArray(np.ndarray):
        pass

    dipole_data = np.random.randn(n_dipoles, 4)
    times = np.arange(4)

    plot = BrainPlot.from_source_space(src, dipole_data, states=times, hemi="lh")
    layer = plot.layers[0]

    # provider must exist (values weren't precomputed into a dense array
    # for all 4 states up front) - Layer stores a callable, not an array
    assert callable(layer._provider)


def test_from_source_space_no_values_is_anatomy_only(src):
    plot = BrainPlot.from_source_space(src, hemi="both")
    layer_l, layer_r = plot.layers

    assert layer_l.scalar_name is None
    assert layer_r.scalar_name is None


def test_from_source_space_invalid_hemi_raises(src):
    with pytest.raises(KeyError):
        BrainPlot.from_source_space(src, hemi="invalid")


# ---------------------------
# camera / view
# ---------------------------


def test_camera_none_without_view(spheres):
    sphere_l, _ = spheres
    plot = BrainPlot(sphere_l, hemi="lh")
    assert plot.camera is None


def test_camera_lateral_medial_resolves_via_hemi(spheres):
    sphere_l, sphere_r = spheres

    lh_lateral = BrainPlot(sphere_l, hemi="lh", view="lateral")
    lh_medial = BrainPlot(sphere_l, hemi="lh", view="medial")
    rh_lateral = BrainPlot(sphere_r, hemi="rh", view="lateral")

    direction_ll, _ = lh_lateral.camera
    direction_lm, _ = lh_medial.camera
    direction_rl, _ = rh_lateral.camera

    assert direction_ll == (-1.0, 0.0, 0.0)
    assert direction_lm == (1.0, 0.0, 0.0)
    assert direction_rl == (1.0, 0.0, 0.0)


def test_camera_absolute_view_needs_no_hemi(spheres):
    sphere_l, _ = spheres
    plot = BrainPlot(sphere_l, view="superior")
    direction, viewup = plot.camera
    assert direction == (0.0, 0.0, 1.0)


def test_camera_lateral_on_two_hemisphere_panel_raises(spheres):
    sphere_l, sphere_r = spheres
    plot = BrainPlot([sphere_l, sphere_r], hemi=["lh", "rh"], view="lateral")

    with pytest.raises(ValueError, match="ambiguous"):
        _ = plot.camera


def test_from_source_space_populates_hemi_for_view(src):
    plot = BrainPlot.from_source_space(src, hemi="lh", view="medial")
    direction, _ = plot.camera
    assert direction == (1.0, 0.0, 0.0)  # medial of lh = viewed from the right


def test_combined_panel_inherits_first_camera(spheres):
    """`BrainPlot(view=...) + SurfacePlot(...)` should carry the BrainPlot's
    camera onto the merged panel."""
    sphere_l, _ = spheres
    from mindmap.viz.surface import SurfacePlot

    brain = BrainPlot(sphere_l, hemi="lh", view="lateral")
    extra = SurfacePlot(pv.Sphere())
    combined = brain + extra

    assert combined.camera == brain.camera
