"""
Module Name: viz/sensor.py
Description: `SensorPlot` - the EEG/iEEG sensor-position plot shortcut.
Positions come from an MNE Raw/Epochs/Evoked object (or anything
`get_ch_positions` accepts); values, if shown, come from that same
object's `.data`/`.times`.

Author: Nino Herve
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pyvista as pv
import mne

from mindmap.analysis.functional.eeg.io import get_ch_positions
from mindmap.viz.mesh import Layer, find_clim, preprocess, pop_clim_options
from mindmap.viz.panel import Panel


def _default_sensor_cmap(values) -> str:
    """"bkr" for signed data, "kr" for non-negative - mirrors a refinement
    already present in the old plot_ieeg, and keeps sensor-space visually
    distinct from the "Reds"/"bwr" family BrainPlot defaults to (a glance
    tells you which space a panel is showing)."""
    arr = np.asarray(values)
    return "bkr" if np.nanmin(arr) < 0 else "kr"


@dataclass
class SensorPlot(Panel):
    """
    EEG/iEEG sensor positions, optionally colored by their recorded values
    over time.

    Parameters
    ----------
    sensors: mne.io.Raw | mne.Epochs | mne.Evoked
        Anything `get_ch_positions` accepts - positions come from
        `sensors.info["dig"]`.

    show_values: bool
        `True` (default) colors each sensor by `sensors.data`, with the
        shared time axis `sensors.times`. `False` renders plain position
        markers - the primary use case (checking electrode placement),
        no scalar data needed.

    title: str
        Panel title.

    scalar_name: str
        See `SurfacePlot` - also the selector `Layout.link_scale` matches
        on.

    mesh_params: dict
        Passed to this plot's `Layer` (rendered as sphere glyphs - a
        point cloud's default style). Without values (`show_values=False`,
        or the sensor object has no `.data`), defaults to solid, fully
        opaque black markers - deliberately *not* `Layer`'s generic
        translucent-white no-scalar default, which would be nearly
        invisible for small point markers against a typical light
        background. With values, `clim`/`cmap` default to this plot's own
        data range/sign (see `_default_sensor_cmap`).

    func_params: dict
        `normalize`: "max" | "minmax" | "zscore"
        `absolute`: bool
        Applied to `sensors.data` before plotting (only relevant when
        `show_values` is True).
    """

    sensors: Union[mne.io.Raw, mne.Epochs, mne.Evoked]
    show_values: bool = True
    title: str = ""
    scalar_name: str = "Potential"
    mesh_params: dict = field(default_factory=dict)
    func_params: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[Layer]:
        mesh = pv.PolyData(get_ch_positions(self.sensors))

        has_data = self.show_values and getattr(self.sensors, "data", None) is not None
        values = self.sensors.data if has_data else None
        states = self.sensors.times if has_data else None
        processed = preprocess(values, self.func_params)

        mesh_params = dict(self.mesh_params)
        clim_options = pop_clim_options(mesh_params)
        if processed is None:
            mesh_params.setdefault("color", "black")
            mesh_params.setdefault("opacity", 1.0)
        else:
            mesh_params.setdefault("clim", find_clim([processed], **clim_options))
            mesh_params.setdefault("cmap", _default_sensor_cmap(processed) if not clim_options["symmetric"] else "bkr")

        return [Layer(mesh, processed, states, scalar_name=self.scalar_name, **mesh_params)]
