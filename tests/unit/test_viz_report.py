"""
This module performs unit tests for viz/report.py: `add_panel` (the
Panel/Layout <-> mne.Report bridge) and `build_preprocessing_report`.
Runs headless (off-screen PyVista). Uses the session-scoped `ave` fixture
from tests/conftest.py (16-channel biosemi montage).
"""

import pyvista as pv
import mne
import pytest

from mindmap.viz.report import add_panel, build_preprocessing_report
from mindmap.viz.surface import SurfacePlot
from mindmap.viz.panel import Layout

pv.OFF_SCREEN = True


@pytest.fixture
def bem():
    return {
        "inner_skull": pv.Sphere(radius=0.06, theta_resolution=6, phi_resolution=6),
        "outer_skull": pv.Sphere(radius=0.08, theta_resolution=6, phi_resolution=6),
        "skin": pv.Sphere(radius=0.1, theta_resolution=6, phi_resolution=6),
    }


@pytest.fixture
def noise_cov(epoch):
    return mne.compute_covariance(epoch, tmax=None, method="empirical", verbose="error")


# ---------------------------
# add_panel
# ---------------------------


def test_add_panel_static_adds_image_content():
    report = mne.Report()
    add_panel(report, SurfacePlot(pv.Sphere()), title="Sphere")

    assert len(report._content) == 1
    block = report._content[0]
    assert block.name == "Sphere"
    assert "3d" in block.tags
    assert "<img" in block.html


def test_add_panel_accepts_a_layout():
    report = mne.Report()
    layout = SurfacePlot(pv.Sphere(radius=0.5)) | SurfacePlot(pv.Sphere(radius=1.0))
    add_panel(report, layout, title="Two spheres")

    assert len(report._content) == 1


def test_add_panel_interactive_embeds_single_iframe():
    report = mne.Report()
    add_panel(report, SurfacePlot(pv.Sphere()), title="Sphere", interactive=True)

    assert len(report._content) == 1
    html = report._content[0].html
    assert html.count("<iframe") == 1
    assert "srcdoc=" in html


def test_add_panel_section_and_caption_forwarded():
    report = mne.Report()
    add_panel(report, SurfacePlot(pv.Sphere()), title="Sphere", section="Anatomy", caption="a caption")

    block = report._content[0]
    assert block.section == "Anatomy"


# ---------------------------
# build_preprocessing_report
# ---------------------------


def test_minimal_report_has_evoked_and_avg_ref(ave, tmp_path):
    report = build_preprocessing_report(ave, subject_id="test", fname=tmp_path / "r.html")

    tags = {t for block in report._content for t in block.tags}
    assert "evoked" in tags
    assert "custom-figure" in tags
    assert (tmp_path / "r.html").exists()


def test_noise_cov_adds_whitening_sections(ave, noise_cov, tmp_path):
    report = build_preprocessing_report(ave, noise_cov, subject_id="test", fname=tmp_path / "r.html")

    names = [block.name for block in report._content]
    assert "Global Field Power" in names
    assert "Whitened Covariance" in names


def test_ad_hoc_diagonal_noise_cov_does_not_crash(ave, tmp_path):
    """Regression: processing_spike.qmd builds noise_cov via
    mne.make_ad_hoc_cov for every subject - a diagonal Covariance whose
    .data is 1-D, which used to crash both plot_whitened_cov and mne's
    own add_covariance (via mne.viz.plot_cov)."""
    noise_cov = mne.make_ad_hoc_cov(ave.info)
    report = build_preprocessing_report(ave, noise_cov, subject_id="test", fname=tmp_path / "r.html")

    names = [block.name for block in report._content]
    assert "Whitened Covariance" in names
    assert "Covariance matrix" in names  # mne's own add_covariance section
    tags = {t for block in report._content for t in block.tags}
    assert "covariance" in tags


def test_no_noise_cov_skips_whitening_sections(ave, tmp_path):
    report = build_preprocessing_report(ave, subject_id="test", fname=tmp_path / "r.html")

    names = [block.name for block in report._content]
    assert "Global Field Power" not in names
    assert "Whitened Covariance" not in names


def test_bem_adds_3d_sensor_alignment_section(ave, bem, tmp_path):
    report = build_preprocessing_report(ave, subject_id="test", bem=bem, fname=tmp_path / "r.html")

    names = [block.name for block in report._content]
    assert "Sensor Alignment" in names


def test_no_bem_skips_3d_section(ave, tmp_path):
    report = build_preprocessing_report(ave, subject_id="test", fname=tmp_path / "r.html")

    names = [block.name for block in report._content]
    assert "Sensor Alignment" not in names


def test_baseline_adds_topo_section(ave, tmp_path):
    report = build_preprocessing_report(ave, subject_id="test", baseline=(0.0, 0.2), fname=tmp_path / "r.html")

    names = [block.name for block in report._content]
    assert "Baseline Topography" in names


def test_interpolated_adds_psd_section(ave, raw, tmp_path):
    report = build_preprocessing_report(
        ave, subject_id="test",
        interpolated=(raw, raw, raw.ch_names[:2]),
        fname=tmp_path / "r.html",
    )

    names = [block.name for block in report._content]
    assert "Bad Channel Interpolation" in names


def test_default_fname_is_subject_named(ave, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_preprocessing_report(ave, subject_id="abc123")

    assert (tmp_path / "Report_abc123.html").exists()
