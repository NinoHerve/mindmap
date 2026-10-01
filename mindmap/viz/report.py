"""
Module Name: viz/report.py
Description: Bridges the `Panel`/`Layout` 3D framework into `mne.Report`
(`add_panel`), and a slim per-subject preprocessing QC report built from
that bridge + `mne.Report`'s own native sections + `viz.eeg`'s 2D
diagnostics (`build_preprocessing_report`).

`mne.Report` already covers the standard 2D EEG diagnostics well
(`add_evokeds`, `add_covariance`, `add_raw`, ...) - the only real gap is
that it has no notion of this package's own 3D framework, hence `add_panel`
being the one new primitive here rather than a whole new report engine.

Author: Nino Herve
License: MIT
"""

from html import escape as _html_escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import numpy as np
import mne
from mne.cov import compute_whitener

from mindmap.viz import eeg as eeg_plots
from mindmap.viz.head import HeadModelPlot
from mindmap.viz.panel import Panel, Layout
from mindmap.viz.sensor import SensorPlot


def add_panel(
    report: mne.Report,
    panel: Union[Panel, Layout],
    title: str,
    *,
    interactive: bool = False,
    state=None,
    window_size: Optional[tuple] = None,
    iframe_height: int = 600,
    section: Optional[str] = None,
    tags=("3d",),
    caption: Optional[str] = None,
) -> None:
    """
    Render a `Panel`/`Layout` (e.g. `BrainPlot(...) + HeadModelPlot(...)`)
    and add it to an `mne.Report`, in place.

    Parameters
    ----------
    report: mne.Report
        Mutated in place - not returned.

    panel: Panel | Layout
        Anything from the viz framework: a single shortcut, a `+`-merged
        `CombinedPanel`, or a full `|`/`&` `Layout`.

    title: str
        Section title in the report.

    interactive: bool
        `False` (default) embeds a static screenshot (`add_image`) - a few
        hundred KB, works everywhere, no caveats. `True` embeds a live,
        explorable 3D scene instead (`Plotter.export_html` wrapped in an
        `<iframe srcdoc=...>`, `add_html`) so the report stays one
        self-contained file - but each interactive panel adds roughly 1MB
        of VTK.js runtime on top of its mesh data, regardless of mesh
        complexity, since the runtime isn't shared/deduped across iframes.
        Prefer it selectively (e.g. the one section worth exploring live)
        rather than for every 3D panel in a report.

    state: int | float, optional
        Which state to render - an index, or the nearest match by value.
        Defaults to each layer's first state. See `Layout.screenshot`.

    window_size: (width, height), optional
        Only meaningful when `interactive=False` - see `Layout.screenshot`.

    iframe_height: int
        Pixel height of the embedded iframe. Only used when
        `interactive=True`.

    section, tags, caption:
        Forwarded to the underlying `mne.Report.add_image`/`add_html`.
    """
    layout = panel if isinstance(panel, Layout) else Layout([[panel]])

    if not interactive:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.png"
            layout.screenshot(path, state=state, window_size=window_size)
            report.add_image(str(path), title=title, caption=caption, section=section, tags=tags)
        return

    scene = layout.build_scene(interactive=False, off_screen=True, window_size=window_size)
    if state is not None:
        scene.set_state(state)
    html = scene.plotter.export_html(None).getvalue()
    scene.plotter.close()

    iframe = (
        f'<iframe srcdoc="{_html_escape(html, quote=True)}" '
        f'style="width:100%; height:{iframe_height}px; border:none;"></iframe>'
    )
    report.add_html(iframe, title=title, tags=tags, section=section)


def _densify_cov(noise_cov: mne.Covariance) -> mne.Covariance:
    """A diagonal/ad-hoc `Covariance` (e.g. `mne.make_ad_hoc_cov`) stores
    only the diagonal as a 1-D array - `mne.viz.plot_cov` (used internally
    by `add_covariance`) assumes a full (n, n) matrix and raises on it.
    Off-diagonal terms are genuinely zero for a diagonal covariance, so
    `np.diag()` is an exact dense reconstruction, not an approximation."""
    if noise_cov.data.ndim != 1:
        return noise_cov
    return mne.Covariance(
        np.diag(noise_cov.data), noise_cov["names"], noise_cov["bads"],
        noise_cov["projs"], noise_cov["nfree"],
    )


def build_preprocessing_report(
    evoked: mne.Evoked,
    noise_cov: Optional[mne.Covariance] = None,
    *,
    subject_id: str = "unknown",
    bem: Optional[dict] = None,
    baseline: Optional[tuple] = None,
    interpolated: Optional[tuple] = None,
    fname: Optional[Union[str, Path]] = None,
) -> mne.Report:
    """
    Standard per-subject preprocessing QC report - one section per
    pipeline step that has something worth checking. Every section is
    optional except the evoked itself; skip what you don't have.

    Parameters
    ----------
    evoked: mne.Evoked
        The averaged, projected EEG - the evoked/avg-ref/GFP sections are
        all built from this (a projected copy is used internally, the
        original is left untouched).

    noise_cov: mne.Covariance, optional
        Adds a whitened-GFP overlay, a whitened-covariance check, and
        mne's own `add_covariance` section. Skipped entirely if not given.

    subject_id: str
        Report title, and the default filename stem.

    bem: dict, optional
        `{"inner_skull", "outer_skull", "skin"}` - see
        `mindmap.anatomy.forward.io.read_bem`. Adds a static 3D sensor
        alignment section (`HeadModelPlot(bem) + SensorPlot(evoked,
        show_values=False)`) via `add_panel` - the one section the old
        report never had despite the plotting code already existing.

    baseline: (tmin, tmax), optional
        Adds a baseline-topography section if given.

    interpolated: (raw_before, raw_after, bads), optional
        Adds a before/after interpolation PSD section if given.

    fname: str | Path, optional
        Where to save. Defaults to `"Report_{subject_id}.html"` in the
        current directory.

    Returns
    -------
    mne.Report
        Also saved to `fname` as a side effect.
    """
    report = mne.Report(title=f"Preprocessing Report: {subject_id}")
    evoked_clean = evoked.copy().apply_proj()

    report.add_evokeds(
        evokeds=evoked_clean,
        noise_cov=noise_cov,
        titles="Evoked potentials",
        topomap_kwargs={"extrapolate": "local"},
    )

    report.add_figure(
        eeg_plots.plot_avg_ref(evoked),
        title="Average Reference Check",
        section="Preprocessing",
    )

    if baseline is not None:
        report.add_figure(
            eeg_plots.plot_baseline_topo(evoked, *baseline),
            title="Baseline Topography",
            section="Preprocessing",
        )

    if interpolated is not None:
        report.add_figure(
            eeg_plots.plot_interpolation(*interpolated),
            title="Bad Channel Interpolation",
            section="Preprocessing",
        )

    if noise_cov is not None:
        whitener, _ = compute_whitener(noise_cov, info=evoked_clean.info)

        report.add_figure(
            eeg_plots.plot_gfp(evoked_clean, whitener=whitener),
            title="Global Field Power",
            section="Preprocessing",
        )
        report.add_figure(
            eeg_plots.plot_whitened_cov(noise_cov, whitener),
            title="Whitened Covariance",
            section="Preprocessing",
        )
        report.add_covariance(_densify_cov(noise_cov), info=evoked_clean.info, title="Noise Covariance")

    if bem is not None:
        add_panel(
            report,
            HeadModelPlot(bem) + SensorPlot(evoked_clean, show_values=False),
            title="Sensor Alignment",
            section="Anatomy",
        )

    fname = fname or f"Report_{subject_id}.html"
    report.save(str(fname), overwrite=True, open_browser=False)
    return report
