"""
This module performs unit tests for viz/sensor.py: `SensorPlot`, the
EEG/iEEG sensor-position plot shortcut. Runs headless (off-screen PyVista).
Uses the session-scoped `ave`/`raw`/`epoch` fixtures from tests/conftest.py
(a 16-channel biosemi montage, so real digitized positions are available).
"""

import numpy as np
import pyvista as pv
import pytest

from mindmap.viz.sensor import SensorPlot, _default_sensor_cmap
from mindmap.viz.brain import BrainPlot
from mindmap.analysis.functional.eeg.io import get_ch_positions

pv.OFF_SCREEN = True


def test_positions_match_get_ch_positions(ave):
    plot = SensorPlot(ave, show_values=False)
    layer = plot.layers[0]

    assert np.allclose(layer.mesh.points, get_ch_positions(ave))


def test_no_values_defaults_to_opaque_black_markers(ave):
    plot = SensorPlot(ave, show_values=False)
    layer = plot.layers[0]

    assert layer.scalar_name is None
    assert layer.plot_kwargs["color"] == "black"
    assert layer.plot_kwargs["opacity"] == 1.0


def test_show_values_true_colors_by_data(ave):
    plot = SensorPlot(ave, show_values=True)
    layer = plot.layers[0]

    assert layer.scalar_name == "Potential"
    assert layer.n_states == len(ave.times)
    assert "clim" in layer.plot_kwargs
    assert "cmap" in layer.plot_kwargs


def test_cmap_signed_vs_nonnegative():
    assert _default_sensor_cmap(np.array([-1.0, 0.0, 1.0])) == "bkr"
    assert _default_sensor_cmap(np.array([0.0, 1.0, 2.0])) == "kr"


def test_scalar_name_default_is_potential(ave):
    plot = SensorPlot(ave)
    assert plot.scalar_name == "Potential"


def test_mesh_params_override_defaults(ave):
    plot = SensorPlot(ave, show_values=False, mesh_params={"color": "orange"})
    layer = plot.layers[0]

    assert layer.plot_kwargs["color"] == "orange"


def test_func_params_applied_before_clim(ave):
    plot = SensorPlot(ave, func_params={"absolute": True})
    layer = plot.layers[0]
    layer.update(0)

    assert np.all(layer.mesh[layer.scalar_name] >= 0)
    assert plot.layers[0].plot_kwargs["clim"][0] == 0


def test_rendered_as_spheres_by_default(ave):
    """Point clouds default to sphere glyphs (Layer's own "points" style
    default) - no explicit style override needed here."""
    plot = SensorPlot(ave, show_values=False)
    layer = plot.layers[0]

    assert layer.style == "points"
    assert layer.plot_kwargs["render_points_as_spheres"] is True


def test_composes_with_brainplot_via_add(ave):
    brain = BrainPlot(pv.Sphere(radius=0.5))
    combined = brain + SensorPlot(ave, show_values=False)

    assert len(combined.layers) == 2
