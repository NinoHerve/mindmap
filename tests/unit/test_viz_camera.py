"""
This module performs unit tests for viz/camera.py: named anatomical views,
hemisphere-aware view resolution, and applying a fixed orthographic camera
to a pv.Plotter subplot. Runs headless (off-screen PyVista).
"""

import numpy as np
import pyvista as pv
import pytest

from mindmap.viz.camera import get_view, resolve_hemi_view, apply_view

pv.OFF_SCREEN = True


def test_get_view_known_directions():
    direction, viewup = get_view("left")
    assert direction == (-1.0, 0.0, 0.0)

    direction, viewup = get_view("right")
    assert direction == (1.0, 0.0, 0.0)

    direction, viewup = get_view("superior")
    assert direction == (0.0, 0.0, 1.0)
    assert viewup == (0.0, 1.0, 0.0)  # top-down view needs a different up vector


def test_get_view_aliases_and_case_insensitive():
    assert get_view("Dorsal") == get_view("superior") == get_view("Top")
    assert get_view("ventral") == get_view("inferior") == get_view("bottom")
    assert get_view("rostral") == get_view("anterior") == get_view("front")


def test_get_view_unknown_raises():
    with pytest.raises(ValueError, match="Unknown view"):
        get_view("sideways")


def test_resolve_hemi_view_lateral_medial_flip_by_hemisphere():
    """The core subtlety: lateral/medial are opposite absolute directions
    depending on which hemisphere you're looking at."""
    assert resolve_hemi_view("lateral", "lh") == get_view("left")
    assert resolve_hemi_view("medial", "lh") == get_view("right")
    assert resolve_hemi_view("lateral", "rh") == get_view("right")
    assert resolve_hemi_view("medial", "rh") == get_view("left")


def test_resolve_hemi_view_absolute_views_ignore_hemisphere():
    assert resolve_hemi_view("superior", "lh") == resolve_hemi_view("superior", "rh") == get_view("superior")
    assert resolve_hemi_view("superior", None) == get_view("superior")


def test_resolve_hemi_view_lateral_without_hemisphere_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_hemi_view("lateral", None)

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_hemi_view("medial", None)


def test_apply_view_preserves_direction_and_frames_the_mesh():
    """Regression test for the exact fitting mechanism: direction/viewup
    must be preserved exactly, focus set to the rendered content's own
    center (read from the renderer, not a fixed world origin)."""
    mesh = pv.Sphere(radius=0.7, center=(-1.5, 0.3, 0.0))
    mesh["value"] = mesh.points[:, 0]

    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars="value")

    direction, viewup = get_view("left")
    apply_view(pl, direction, viewup)

    assert pl.camera.GetParallelProjection()  # orthographic

    pos = np.array(pl.camera.position)
    foc = np.array(pl.camera.focal_point)
    resolved_direction = (pos - foc) / np.linalg.norm(pos - foc)
    assert resolved_direction == pytest.approx(list(direction), abs=1e-6)
    assert foc == pytest.approx(list(mesh.center), abs=1e-3)

    pl.close()


def test_apply_view_adapts_to_mesh_size_not_a_fixed_distance():
    """Two meshes of very different sizes must each get a parallel_scale
    that actually frames them - not the same fixed number."""
    small = pv.Sphere(radius=0.1)
    large = pv.Sphere(radius=10.0)
    direction, viewup = get_view("superior")

    scales = []
    for mesh in (small, large):
        pl = pv.Plotter()
        pl.add_mesh(mesh)
        apply_view(pl, direction, viewup)
        scales.append(pl.camera.parallel_scale)
        pl.close()

    assert scales[1] > scales[0] * 10  # large mesh needs a much bigger parallel_scale


def test_apply_view_tight_fit_survives_multi_panel_layout():
    """Regression test for a real bug: fitting must use the FINAL window
    size, not whatever the window defaulted to before being resized. Two
    meshes side by side in a 2-panel plotter, each with its own camera,
    must each fill their panel about as tightly as a single mesh alone in
    an equivalently-shaped panel - not shrink because the fit was computed
    against a since-changed window size."""
    mesh = pv.Sphere(radius=1.0)
    direction, viewup = get_view("left")

    # reference: single mesh, panel shape matching one cell of the 2-panel case
    solo = pv.Plotter(off_screen=True, window_size=(500, 500))
    solo.add_mesh(mesh)
    apply_view(solo, direction, viewup)
    solo_scale = solo.camera.parallel_scale
    solo.close()

    # window_size set at construction, matching what Scene now does
    grid = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1000, 500))
    grid.subplot(0, 0)
    grid.add_mesh(mesh)
    apply_view(grid, direction, viewup)
    grid_scale = grid.camera.parallel_scale
    grid.close()

    assert grid_scale == pytest.approx(solo_scale, rel=0.05)
