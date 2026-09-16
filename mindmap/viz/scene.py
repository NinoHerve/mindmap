"""
Module Name: viz/scene.py
Description: `Scene` - the single interactive multi-panel PyVista plotter.

One or more panels (subplots) each hold a list of `Layer`s. A shared slider
steps every layer through its state axis whenever more than one state is
present. This replaces the old package's four near-duplicate plotter
classes (`BaseBrainPlotter`, two `TimeMesh`s, `MultiMesh`, `MultiConditionMesh`).

Author: Nino Herve
License: MIT
"""

from typing import Optional
import warnings

import numpy as np
import pyvista as pv

from mindmap.viz.mesh import Layer
from mindmap.viz.camera import apply_view


def _resolve_shared_states(panels: list[list[Layer]]) -> np.ndarray:
    """Pick a reference state axis, warning if layers disagree."""
    all_states = [layer.states for group in panels for layer in group if layer.n_states > 1]

    if not all_states:
        return np.array([0])

    ref = all_states[0]
    if not all(np.array_equal(s, ref) for s in all_states):
        warnings.warn("Inconsistent state axes detected across layers. Using the first available.")

    return ref


class Scene:
    """
    Interactive multi-panel visualization of `Layer`s using PyVista.
    A slider steps through the shared state axis when any layer has more
    than one state.
    """

    def __init__(
        self,
        panels: list[list[Layer]],
        titles: Optional[list[str]] = None,
        shape: Optional[tuple] = None,
        cameras: Optional[list] = None,
        window_size: Optional[tuple] = None,
        slider_label: str = "State",
        link_views: Optional[bool] = None,
        interaction_event: str = "always",
        interactive: bool = True,
        off_screen: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        panels: list[list[Layer]]
            One inner list per subplot; layers within a subplot are drawn
            together (e.g. a brain surface + an electrode point cloud).
            Laid out row-major according to `shape`.

        titles: list[str], optional
            Title per subplot.

        shape: (n_rows, n_cols), optional
            Grid shape for laying out `panels` in row-major order. Must
            satisfy `n_rows * n_cols == len(panels)`. Defaults to a single
            row, `(1, len(panels))` - unchanged behavior when omitted.

        cameras: list, optional
            One entry per subplot: `(direction, viewup)` (see `viz.camera`)
            to fix that subplot's camera (orthographic, auto-framed), or
            `None` to leave it at PyVista's interactive default. Matches
            `panels` in length when given.

        window_size: (width, height), optional
            Set at `pv.Plotter` construction, before any camera is fit -
            required (not just recommended) whenever `cameras` is used:
            the tight fit depends on each subplot's actual pixel aspect
            ratio, computed once from the window size at the time it's
            applied. Setting the size *after* fitting (e.g. via
            `Plotter.screenshot(window_size=...)`) leaves the fit
            mismatched against the final image - `Layout.screenshot` sets
            this correctly for you.

        slider_label: str
            Caption for the state slider (e.g. "Time [s]", "Harmonic").

        link_views: bool, optional
            Link camera across subplots. Left unset, defaults to `True`
            unless any `cameras` entry is set - linking would otherwise
            immediately fight per-subplot fixed views (e.g. a lateral and
            a medial view can't both be "linked" to the same camera).
            Explicit `True`/`False` always wins over that default.

        interaction_event: {"always", "end"}
            "always" (default) re-renders on every drag movement - smooth
            live scrubbing for small meshes, but on the Jupyter/trame
            backend every render streams the updated scene to the browser,
            which gets slow for large meshes (e.g. a full-resolution brain
            surface). "end" only re-renders when the slider is released -
            far fewer renders during interaction, at the cost of live
            feedback while dragging. Use "end" for large meshes over trame.

        interactive: bool
            `False` skips the slider/key widgets entirely - for a fixed
            image (see `screenshot`), a slider would just show up as
            clutter in the exported picture. `set_state`/`screenshot` still
            work either way.

        off_screen: bool, optional
            Passed to `pv.Plotter` only if explicitly given - left unset
            (the default), pyvista decides on its own (honoring the global
            `pv.OFF_SCREEN` flag some callers, e.g. tests, rely on).
            `screenshot` sets this itself; only useful directly if you
            want an off-screen interactive Scene for some other reason.
        """
        self.panels = panels
        self.states = _resolve_shared_states(panels)
        self.n_states = len(self.states)
        self.current_idx = 0
        self.interaction_event = interaction_event
        self.interactive = interactive

        self.shape = shape or (1, len(panels))
        n_rows, n_cols = self.shape
        if n_rows * n_cols != len(panels):
            raise ValueError(f"shape {self.shape} doesn't match {len(panels)} panels.")

        cameras = cameras or [None] * len(panels)
        if len(cameras) != len(panels):
            raise ValueError(f"cameras ({len(cameras)}) must match panels ({len(panels)}).")

        if link_views is None:
            link_views = not any(cameras)

        plotter_kwargs = {} if off_screen is None else {"off_screen": off_screen}
        if window_size is not None:
            plotter_kwargs["window_size"] = window_size
        self.plotter = pv.Plotter(shape=self.shape, **plotter_kwargs)

        for i, layers in enumerate(panels):
            row, col = divmod(i, n_cols)
            self.plotter.subplot(row, col)
            if titles and i < len(titles) and titles[i]:
                self.plotter.add_title(titles[i], font_size=10)

            checkbox_row = 0
            for layer in layers:
                actor = self.plotter.add_mesh(
                    layer.mesh, scalars=layer.scalar_name, **layer.plot_kwargs
                )
                if layer.toggle:
                    self.plotter.add_checkbox_button_widget(
                        actor.SetVisibility,
                        value=True,
                        position=(10, 10 + 40 * checkbox_row),
                        size=25,
                        color_on=layer.plot_kwargs.get("color", "grey"),
                    )
                    checkbox_row += 1

            if cameras[i] is not None:
                direction, viewup = cameras[i]
                apply_view(self.plotter, direction, viewup)

        if link_views:
            self.plotter.link_views()

        if self.interactive:
            if self.n_states > 1:
                self._add_slider(slider_label)
            self._setup_keys()

        self._update(0)

    # ------------------
    # State navigation
    # ------------------

    def _add_slider(self, label: str):
        slider = self.plotter.add_slider_widget(
            callback=self._on_slider_change,
            rng=[0, self.n_states - 1],
            value=0,
            title=label,
            style="modern",
            interaction_event=self.interaction_event,
        )
        self.slider = slider.GetSliderRepresentation()

    def _on_slider_change(self, value):
        idx = int(round(value))
        if idx != self.current_idx:
            self.current_idx = idx
            self._update(idx)

    def _setup_keys(self, fast_step: int = 10):
        self.plotter.add_key_event("Right", lambda: self._step(1))
        self.plotter.add_key_event("Left", lambda: self._step(-1))
        self.plotter.add_key_event("d", lambda: self._step(fast_step))
        self.plotter.add_key_event("a", lambda: self._step(-fast_step))

    def _step(self, delta):
        new_idx = max(0, min(self.current_idx + delta, self.n_states - 1))
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            if hasattr(self, "slider"):
                self.slider.SetValue(new_idx)
            self._update(new_idx)

    def _update(self, idx: int):
        for layers in self.panels:
            for layer in layers:
                layer.update(idx)
        self.plotter.render()

    def set_state(self, state):
        """
        Paint every panel at a given state, by index (`int`) or by value
        (nearest match in `self.states`, e.g. a time in seconds) - useful
        before `screenshot()` to pick a specific timepoint.
        """
        idx = state if isinstance(state, (int, np.integer)) else int(np.argmin(np.abs(self.states - state)))
        self.current_idx = idx
        if hasattr(self, "slider"):
            self.slider.SetValue(idx)
        self._update(idx)

    # ------
    # Show
    # ------

    def show(self, **kwargs):
        """
        Show the scene.

        **kwargs: forwarded to `pv.Plotter.show()` - e.g. `jupyter_backend`
        to pick how it's displayed inside a Jupyter/IPython kernel
        ("trame" for the default inline interactive widget, "none" for a
        native OS render window, "static" for a single non-interactive
        image, ...). Ignored (and irrelevant) outside a notebook, where
        `.show()` already opens a native window on its own.
        """
        self.plotter.show(**kwargs)

    def screenshot(self, path=None, **kwargs):
        """
        Export the current state as an image, without opening any window.

        Parameters
        ----------
        path: str | Path, optional
            Where to save. If omitted, returns the image array instead
            (`return_img=True` behavior) rather than writing a file.

        **kwargs: forwarded to `pv.Plotter.screenshot()` - e.g.
        `window_size`, `transparent_background`.
        """
        if path is None:
            return self.plotter.screenshot(return_img=True, **kwargs)
        return self.plotter.screenshot(path, **kwargs)
