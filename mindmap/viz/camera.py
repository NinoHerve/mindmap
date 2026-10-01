"""
Module Name: viz/camera.py
Description: Named anatomical camera views, and applying one to a single
`pv.Plotter` subplot with automatic per-mesh framing.

Two separate concerns, deliberately kept apart:
- **Direction** is semantic and fixed (`get_view`) - it only depends on
  FreeSurfer's tkRAS convention (+x=Right, +y=Anterior, +z=Superior), which
  holds across subjects, so a named view means the same thing regardless
  of which subject/mesh it's applied to.
- **Framing** (focus point, distance/zoom) is computed from the actual
  mesh being rendered (`apply_view`), never hardcoded - a fixed distance
  that looks right for one subject's brain would clip or float in empty
  space for another's.

`resolve_hemi_view` adds one more layer specific to two-hemisphere anatomy:
"lateral"/"medial" are relative to which hemisphere you're looking at (the
medial surface of the right hemisphere faces -x, the same direction as the
left hemisphere's lateral surface) - "dorsal"/"ventral"/"anterior"/
"posterior" don't depend on hemisphere and pass through unchanged.

Author: Nino Herve
License: MIT
"""

from typing import Optional

import numpy as np
import pyvista as pv

_UP_DEFAULT = (0.0, 0.0, 1.0)
_UP_TOP = (0.0, 1.0, 0.0)

# Absolute anatomical directions - assumes tkRAS: +x=Right, +y=Anterior, +z=Superior.
_VIEWS = {
    "left": ((-1.0, 0.0, 0.0), _UP_DEFAULT),
    "right": ((1.0, 0.0, 0.0), _UP_DEFAULT),
    "anterior": ((0.0, 1.0, 0.0), _UP_DEFAULT),
    "posterior": ((0.0, -1.0, 0.0), _UP_DEFAULT),
    "superior": ((0.0, 0.0, 1.0), _UP_TOP),
    "inferior": ((0.0, 0.0, -1.0), _UP_TOP),
}

_ALIASES = {
    "rostral": "anterior", "front": "anterior",
    "caudal": "posterior", "back": "posterior",
    "dorsal": "superior", "top": "superior",
    "ventral": "inferior", "down": "inferior", "bottom": "inferior",
}

# (hemisphere, hemisphere-relative view) -> absolute view name.
_HEMI_VIEWS = {
    ("lh", "lateral"): "left",
    ("lh", "medial"): "right",
    ("rh", "lateral"): "right",
    ("rh", "medial"): "left",
}


def get_view(name: str) -> tuple:
    """
    Named absolute anatomical direction -> `(direction, viewup)`, both unit
    vectors. `direction` points from the focus point towards the camera.

    Parameters
    ----------
    name: str
        One of "left"/"right"/"anterior"/"posterior"/"superior"/"inferior",
        or a common alias ("dorsal", "ventral", "rostral", "caudal", "top",
        "bottom", "front", "back"). Case-insensitive.
    """
    key = name.strip().lower()
    key = _ALIASES.get(key, key)

    if key not in _VIEWS:
        raise ValueError(
            f"Unknown view {name!r}. Valid views: {sorted(_VIEWS)} "
            f"(aliases: {sorted(_ALIASES)})."
        )

    return _VIEWS[key]


def resolve_hemi_view(view: str, hemi: Optional[str]) -> tuple:
    """
    Resolve a possibly hemisphere-relative view into `(direction, viewup)`.

    "lateral"/"medial" require `hemi` ("lh"/"rh") to resolve - anything else
    (an absolute view, or an alias) passes straight to `get_view` and
    ignores `hemi` entirely, since it doesn't depend on it.

    Raises
    ------
    ValueError
        If a hemisphere-relative view is requested without a known `hemi`
        (e.g. a two-hemisphere panel, where "lateral" is ambiguous - it
        means something different for each side).
    """
    key = view.strip().lower()

    if key in ("lateral", "medial"):
        if hemi not in ("lh", "rh"):
            raise ValueError(
                f"View {view!r} is relative to a single hemisphere and needs "
                f"hemi='lh' or 'rh' to resolve (got {hemi!r}) - it's ambiguous "
                "for a panel spanning both hemispheres. Use an absolute view "
                "(e.g. 'superior') there instead."
            )
        return get_view(_HEMI_VIEWS[(hemi, key)])

    return get_view(view)


def apply_view(plotter: pv.Plotter, direction, viewup, padding: float = 0.05):
    """
    Point the plotter's *current* subplot camera along `direction`,
    orthographic (parallel projection - no perspective distortion), tightly
    framed to whatever's already in that subplot (focus, distance and zoom
    all computed from the renderer's own bounds - never a fixed number, so
    it works regardless of subject/mesh size).

    Same fitting math as `pv.Camera.tight()`, parameterized by our own
    `direction`/`viewup` vectors instead of its axis-pair string interface
    (`view="xy"`-style), since our views come from a lookup table, not a
    fixed set of axis pairs.

    Call this *after* adding this subplot's mesh(es), and only once the
    plotter's final `window_size` is set (at `pv.Plotter(...)`
    construction, not changed afterward) - the tight fit depends on the
    subplot's actual pixel aspect ratio, computed from the current window
    size; resizing later without recomputing leaves it mismatched.

    Parameters
    ----------
    plotter: pv.Plotter
        Call `plotter.subplot(row, col)` first if there's more than one.

    direction: (3,) array-like
        Unit vector from the focus towards the camera (from `get_view`/
        `resolve_hemi_view`).

    viewup: (3,) array-like

    padding: float
        Extra space around the content, as a fraction of the tight fit
        (0.05 = 5% margin).
    """
    renderer = plotter.renderer
    x0, x1, y0, y1, z0, z1 = renderer.bounds

    plotter.camera.enable_parallel_projection()
    renderer.ComputeAspect()
    aspect = renderer.GetAspect()

    lower = np.array([x0, y0, z0])
    upper = np.array([x1, y1, z1])
    size = upper - lower
    focus = lower + size / 2

    direction = np.asarray(direction, dtype=float)
    viewup = np.asarray(viewup, dtype=float)
    horizontal = np.cross(direction, viewup)

    vertical_extent = abs(size @ viewup)
    horizontal_extent = abs(size @ horizontal)

    # The exact distance is arbitrary for the *projection* under parallel
    # projection, but it isn't arbitrary for `ResetCameraClippingRange`
    # below: too close (e.g. distance=1) puts the camera inside the mesh,
    # so the near clip plane slices through the surface facing it. Scaling
    # by the bounding-box diagonal keeps the camera safely outside the
    # mesh regardless of subject/mesh size.
    distance = np.linalg.norm(size)
    plotter.camera.SetViewUp(*viewup)
    plotter.camera.SetPosition(*(focus + direction * distance))
    plotter.camera.SetFocalPoint(*focus)

    parallel_scale = max(horizontal_extent / aspect[0], vertical_extent) / 2
    plotter.camera.parallel_scale = parallel_scale * (1 + padding)
    renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)
