"""
This module performs unit tests for viz/eeg.py: the 2D matplotlib QC
diagnostics for EEG/iEEG preprocessing. Uses the session-scoped `ave`/`raw`/
`epoch` fixtures from tests/conftest.py (16-channel biosemi montage).
"""

import matplotlib
import matplotlib.pyplot as plt
import mne
import pytest

from mindmap.viz.eeg import (
    plot_avg_ref,
    plot_baseline_topo,
    plot_interpolation,
    plot_whitened_cov,
    plot_gfp,
    plot_gfp_comparison,
)

matplotlib.use("Agg")


@pytest.fixture
def noise_cov(epoch):
    return mne.compute_covariance(epoch, tmax=None, method="empirical", verbose="error")


@pytest.fixture
def whitener(noise_cov, info):
    from mne.cov import compute_whitener
    w, _ = compute_whitener(noise_cov, info=info)
    return w


def test_plot_avg_ref_returns_closed_figure(ave):
    fig = plot_avg_ref(ave)
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_ylabel() == "Average Amplitude (V)"


def test_plot_baseline_topo_returns_figure(ave):
    fig = plot_baseline_topo(ave, 0.0, 0.2)
    assert isinstance(fig, plt.Figure)


def test_plot_interpolation_returns_figure(raw):
    fig = plot_interpolation(raw, raw, raw.ch_names[:2])
    assert isinstance(fig, plt.Figure)
    titles = [ax.get_title() for ax in fig.axes]
    assert "Before Interpolation (Bads)" in titles
    assert "After Interpolation (Bads)" in titles


def test_plot_whitened_cov_returns_figure(noise_cov, whitener):
    fig = plot_whitened_cov(noise_cov, whitener)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # 2 imshow axes + 2 colorbars


def test_plot_whitened_cov_handles_diagonal_covariance(info):
    """Regression: mne.make_ad_hoc_cov (used unconditionally by some
    pipelines, e.g. processing_spike.qmd) stores .data as 1-D (diagonal
    only) - used to crash with 'Invalid shape (n,) for image data'."""
    from mne.cov import compute_whitener

    noise_cov = mne.make_ad_hoc_cov(info)
    assert noise_cov.data.ndim == 1  # sanity: this is the diagonal case

    whitener, _ = compute_whitener(noise_cov, info=info)
    fig = plot_whitened_cov(noise_cov, whitener)
    assert isinstance(fig, plt.Figure)


def test_plot_gfp_without_whitener(ave):
    fig = plot_gfp(ave)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1


def test_plot_gfp_with_whitener_adds_second_axis(ave, whitener):
    fig = plot_gfp(ave, whitener=whitener)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_plot_gfp_comparison_returns_figure(ave):
    fig = plot_gfp_comparison(ave, {"est": ave})
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].lines) == 2  # reference + one estimate
