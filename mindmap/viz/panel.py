"""
Module Name: viz/panel.py
Description: `Panel` - the base class for typed plot shortcuts (e.g.
`SurfacePlot`), and the `+`/`|`/`&` operators used to combine them into a
`Layout`.

- `+`  merges panels into the SAME subplot (or, given a `Layout` on one
       side, broadcasts a single panel onto every one of its panels - the
       "add this to every plot" case).
- `|`  lays panels out side by side.
- `&`  stacks panels as a new row, rendered as a grid (every row must have
       the same number of panels - `|`/`&` always produce that naturally).

Python's own operator precedence (`+` binds tighter than `&`, which binds
tighter than `|`) means `a + b | c` always parses as `(a + b) | c` with no
parentheses needed: meshes must be merged into one panel before panels are
laid out, and that's enforced by Python's grammar, not just convention.
Once layouts grow past two panels, parenthesize explicitly for readability.

`Layout.link_scale(*scalar_names)` synchronizes the color scale of panels
that share a `scalar_name`, across a layout, while leaving everything else
independent - e.g. link brain activity across several comparison panels
without forcing each panel's own EEG/iEEG scale to match too.

Author: Nino Herve
License: MIT
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from mindmap.viz.mesh import Layer, find_clim, default_cmap
from mindmap.viz.scene import Scene


class Panel(ABC):
    """
    Base class for a typed plot shortcut: anything that knows how to
    produce the `Layer`(s) for one subplot from clean, named arguments.
    """

    title: str = ""
    camera: Optional[tuple] = None
    """`(direction, viewup)` for a fixed orthographic camera on this panel
    (see `viz.camera.apply_view`), or None for PyVista's default
    interactive view. Set by subclasses that support it (e.g. `BrainPlot`'s
    `view=`)."""

    @property
    @abstractmethod
    def layers(self) -> list[Layer]:
        """The Layer(s) that make up this one subplot."""

    def __add__(self, other: Union["Panel", "Layout"]):
        if isinstance(other, Layout):
            return NotImplemented  # let Layout.__radd__ broadcast instead
        if isinstance(other, Panel):
            return CombinedPanel([self, other])
        return NotImplemented

    def __or__(self, other: Union["Panel", "Layout"]) -> "Layout":
        return Layout([[self]]) | other

    def __and__(self, other: Union["Panel", "Layout"]) -> "Layout":
        return Layout([[self]]) & other

    def show(self, **kwargs):
        """Render this panel alone. See `Layout.show` (e.g. `jupyter_backend="none"`
        to open a native window instead of the inline Jupyter widget)."""
        Layout([[self]]).show(**kwargs)

    def screenshot(self, **kwargs):
        """Export this panel alone as a fixed image. See `Layout.screenshot`."""
        return Layout([[self]]).screenshot(**kwargs)


@dataclass
class CombinedPanel(Panel):
    """
    Several panels merged into a single subplot - the result of
    `Panel + Panel`. Chained `+` flattens: `a + b + c` stays one flat
    `CombinedPanel`, never nested.
    """

    parts: list[Panel]
    title: Optional[str] = None

    def __post_init__(self):
        if self.title is None:
            self.title = next((p.title for p in self.parts if p.title), "")

    @property
    def layers(self) -> list[Layer]:
        return [layer for part in self.parts for layer in part.layers]

    @property
    def camera(self) -> Optional[tuple]:
        """First part with a camera set, if any - matches the `title`
        first-non-empty-wins convention above."""
        return next((p.camera for p in self.parts if p.camera is not None), None)

    def __add__(self, other):
        if isinstance(other, Panel) and not isinstance(other, Layout):
            return CombinedPanel(self.parts + [other], title=self.title)
        return super().__add__(other)


class Layout:
    """
    A grid of panels: `rows` is one list per row, each row a list of
    panels laid out side by side. Built by combining `Panel`/`Layout`
    objects with `|` (extend the current row(s)) or `&` (add a new row).
    """

    def __init__(self, rows: list[list[Panel]]):
        self.rows = rows

    def __or__(self, other: Union[Panel, "Layout"]) -> "Layout":
        other_rows = other.rows if isinstance(other, Layout) else [[other]]
        if len(self.rows) != len(other_rows):
            raise ValueError(
                f"Row count mismatch for '|': {len(self.rows)} vs {len(other_rows)}."
            )
        return Layout([r1 + r2 for r1, r2 in zip(self.rows, other_rows)])

    def __and__(self, other: Union[Panel, "Layout"]) -> "Layout":
        other_rows = other.rows if isinstance(other, Layout) else [[other]]
        return Layout(self.rows + other_rows)

    def __add__(self, other: Union[Panel, "Layout"]) -> "Layout":
        if isinstance(other, Layout):
            raise TypeError(
                "Can't merge two multi-panel Layouts with '+' - merge into a "
                "single panel with '+' before laying out with '|'/'&'."
            )
        if isinstance(other, Panel):
            return Layout([[p + other for p in row] for row in self.rows])
        return NotImplemented

    __radd__ = __add__

    def _leaves(self) -> list[Panel]:
        """Every leaf panel in this layout - unwraps `CombinedPanel` one
        level (chained `+` already flattens, so one level is enough)."""
        leaves = []
        for row in self.rows:
            for panel in row:
                if isinstance(panel, CombinedPanel):
                    leaves.extend(panel.parts)
                else:
                    leaves.append(panel)
        return leaves

    def link_scale(self, *scalar_names: str, clim: Optional[tuple] = None, cmap: Optional[str] = None) -> "Layout":
        """
        Synchronize the color scale of every leaf panel whose `scalar_name`
        matches one of `scalar_names`, across the whole layout - e.g. link
        brain activity across several comparison panels while leaving each
        panel's own EEG/iEEG scale independent.

        Panels not matching any given name are left completely untouched.
        Matching is per name: with no explicit `clim`/`cmap`, each name's
        matches get their own independently-inferred shared range (skipping
        any matches whose `values` is a callable provider, which can't be
        inspected); an explicit `clim`/`cmap` is instead applied directly to
        every match for that name.

        Parameters
        ----------
        *scalar_names: str
            One or more `scalar_name`s to link. Each name is resolved and
            applied independently - call `link_scale` again (it's
            chainable) for a second, separately-scaled group.

        clim: tuple, optional
        cmap: str, optional
            Explicit override, applied to every match for every given name
            instead of auto-inferring from their values.

        Returns
        -------
        Layout
            `self`, mutated in place, for chaining.

        Raises
        ------
        ValueError
            If a given name matches nothing in this layout, or (only when
            auto-inferring) if every match for a name uses a callable
            provider, so no range can be inferred.
        """
        leaves = self._leaves()

        for name in scalar_names:
            matches = [p for p in leaves if getattr(p, "scalar_name", None) == name]

            if not matches:
                raise ValueError(f"link_scale: no layer named {name!r} found in this layout.")

            if clim is None and cmap is None:
                if len(matches) == 1:
                    warnings.warn(
                        f"link_scale: only one layer named {name!r} found - "
                        "nothing to link yet."
                    )
                    continue

                values = [
                    v for p in matches
                    if (v := getattr(p, "values", None)) is not None and not callable(v)
                ]
                resolved_clim = find_clim(values)
                if resolved_clim is None:
                    raise ValueError(
                        f"link_scale: can't auto-infer a color scale for {name!r} "
                        "(every matching layer uses a callable provider) - "
                        "pass clim/cmap explicitly."
                    )
                resolved_cmap = default_cmap(resolved_clim)
            else:
                resolved_clim, resolved_cmap = clim, cmap

            for p in matches:
                if resolved_clim is not None:
                    p.mesh_params["clim"] = resolved_clim
                if resolved_cmap is not None:
                    p.mesh_params["cmap"] = resolved_cmap

        return self

    def build_scene(self, slider_label: str = "State", **kwargs) -> Scene:
        """
        Build the `Scene` for this layout, laid out as a grid matching
        `rows` (row-major). Every row must have the same number of panels -
        `&`/`|` always produce that naturally; only worth knowing if you
        build `rows` by hand.
        """
        n_rows = len(self.rows)
        n_cols = len(self.rows[0])
        if any(len(row) != n_cols for row in self.rows):
            raise ValueError(
                "Every row must have the same number of panels to render as a "
                f"grid (got row lengths {[len(row) for row in self.rows]})."
            )

        panels = [p for row in self.rows for p in row]
        return Scene(
            [p.layers for p in panels],
            titles=[p.title for p in panels],
            shape=(n_rows, n_cols),
            cameras=[p.camera for p in panels],
            slider_label=slider_label,
            **kwargs,
        )

    def show(self, jupyter_backend: Optional[str] = None, **kwargs):
        """
        Render this layout.

        Parameters
        ----------
        jupyter_backend: str, optional
            How to display this inside a Jupyter/IPython kernel. Left
            unset, pyvista picks its own default (typically "trame" - an
            inline interactive widget streamed to the browser on every
            render, which gets slow for large meshes since every slider
            drag re-renders). Pass "none" to instead open a native OS
            render window - much faster interaction, at the cost of it
            no longer being embedded in the notebook (needs a real
            display reachable from the kernel; on a genuinely headless/
            remote kernel this silently does nothing - see
            `pv.set_jupyter_backend` for the full list of options,
            including "static" for a single non-interactive image).
            Ignored outside a notebook, where `.show()` already opens a
            native window on its own.

        **kwargs:
            Forwarded to `build_scene` (`slider_label`, `link_views`,
            `interaction_event`).
        """
        scene = self.build_scene(**kwargs)
        show_kwargs = {} if jupyter_backend is None else {"jupyter_backend": jupyter_backend}
        scene.show(**show_kwargs)

    def screenshot(self, path=None, state=None, window_size=None, **kwargs):
        """
        Export this layout as a fixed image - no slider/key widgets, no
        window (off-screen). Each panel with a `camera` (e.g. `BrainPlot`'s
        `view=`) renders at its own fixed orthographic angle; panels
        without one keep PyVista's own default framing.

        Parameters
        ----------
        path: str | Path, optional
            Where to save. Omit to get the image array back without
            writing a file.

        state: int | float, optional
            Which state to render - an index, or the nearest match by
            value (e.g. a time in seconds). Defaults to each layer's
            first state.

        window_size: (width, height), optional
            Set *before* any per-panel camera is fit (see `Scene`'s
            `window_size`) - unlike passing `window_size` straight to
            `pv.Plotter.screenshot()`, this keeps orthographic panels
            correctly framed for the actual output size, not whatever
            size the window happened to default to.

        **kwargs:
            Forwarded to `pv.Plotter.screenshot()` (`transparent_background`, ...).

        Returns
        -------
        np.ndarray
            The rendered image (also written to `path` if given).
        """
        scene = self.build_scene(interactive=False, off_screen=True, window_size=window_size)
        if state is not None:
            scene.set_state(state)
        image = scene.screenshot(path, **kwargs)
        scene.plotter.close()
        return image
