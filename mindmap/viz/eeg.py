"""
Module Name: viz/eeg.py
Description: 2D matplotlib QC diagnostics for EEG/iEEG preprocessing -
one small, pure function per check, each returning a `Figure` so it can be
embedded directly in an `mne.Report` (see `viz.report`) or shown standalone.

Author: Nino Herve
License: MIT
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import mne

from mindmap.core.metric import get_gfp
from mindmap.core.process import normalize_timeserie_by_baseline


def plot_avg_ref(evoked: mne.Evoked, title: str = "Average Reference Check", apply_proj: bool = True):
    """
    Spatial mean across EEG channels over time - should sit flat at 0V for
    properly average-referenced data.
    """
    ev_check = evoked.copy()
    if apply_proj:
        ev_check.apply_proj()
    ev_check.pick("eeg")

    data = ev_check.get_data()
    times = ev_check.times
    spatial_mean = data.mean(axis=0)
    max_dev = np.max(np.abs(spatial_mean))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, spatial_mean, color="crimson", label="Residual Spatial Mean")
    ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{title}\nMax Deviation after Proj: {max_dev:.2e} V")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average Amplitude (V)")

    limit = max(max_dev * 2, 1e-15)
    ax.set_ylim(-limit, limit)

    plt.close(fig)
    return fig


def plot_baseline_topo(evoked: mne.Evoked, tmin: float, tmax: float):
    """
    Spatial topography averaged over the baseline window - should look
    small and spatially unstructured.
    """
    baseline_evoked = evoked.copy().crop(tmin=tmin, tmax=tmax)

    fig = baseline_evoked.plot_topomap(
        times=(tmin + tmax) / 2,
        average=tmax - tmin,
        size=3,
        colorbar=True,
        units="V",
        show=False,
        time_unit="s",
        proj=True,
        extrapolate="local",
    )
    fig.suptitle(f"Baseline Topo (Avg: {tmin}s to {tmax}s)")

    plt.close(fig)
    return fig


def plot_interpolation(raw_before: mne.io.Raw, raw_after: mne.io.Raw, bads: list):
    """PSD of the interpolated channels before vs. after, to check that
    interpolation didn't introduce/remove structure it shouldn't have."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    raw_before.plot_psd(picks=bads, ax=ax[0], show=False)
    ax[0].set_title("Before Interpolation (Bads)")

    raw_after.plot_psd(picks=bads, ax=ax[1], show=False)
    ax[1].set_title("After Interpolation (Bads)")
    fig.tight_layout()

    plt.close(fig)
    return fig


def plot_whitened_cov(noise_cov: mne.Covariance, whitener: np.ndarray):
    """
    Raw vs. whitened noise covariance, side by side. The whitened matrix
    should look like the identity (flat ~1 diagonal, ~0 off-diagonal).
    """
    cov = noise_cov.data
    if cov.ndim == 1:
        # ad-hoc/diagonal Covariance (e.g. mne.make_ad_hoc_cov) stores only
        # the diagonal as a 1-D array - off-diagonal terms are genuinely
        # zero, so np.diag() is the exact dense reconstruction, not an
        # approximation.
        cov = np.diag(cov)
    whitened_cov = whitener @ cov @ whitener.T
    v_max = np.max(np.abs(whitened_cov))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    im0 = ax[0].imshow(cov, cmap="RdBu_r", interpolation="nearest")
    ax[0].set_title("Original Covariance ($C$)")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(whitened_cov, cmap="RdBu_r", vmin=-v_max, vmax=v_max)
    ax[1].set_title("Whitened Covariance ($W C W^T$)")
    plt.colorbar(im1, ax=ax[1])

    diag_vals = np.diag(whitened_cov)
    avg_diag = np.mean(diag_vals[diag_vals > 1e-10])
    stats_text = f"Mean of Diagonal: {avg_diag:.4f}\n(Target: 1.0)"
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=12,
              bbox=dict(facecolor="white", alpha=0.5, edgecolor="gray"))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.close(fig)
    return fig


def plot_gfp(evoked: mne.Evoked, whitener: Optional[np.ndarray] = None, title: str = "Global Field Power"):
    """
    Global Field Power over time. With `whitener`, overlays the whitened
    GFP (scaled by sqrt(nave)) against an expected noise floor of 1.0 -
    the standard "is my noise covariance well-calibrated" check.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    gfp_raw = get_gfp(evoked.data)
    color_raw = "tab:gray"
    ax1.plot(evoked.times, gfp_raw, color=color_raw, alpha=0.5, label="raw GFP")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GFP (V)", color=color_raw)
    ax1.tick_params(axis="y", labelcolor=color_raw)
    ax1.set_ylim(bottom=0)

    if whitener is not None:
        data_white = (whitener @ evoked.data) * np.sqrt(evoked.nave)
        gfp_white = get_gfp(data_white)

        ax2 = ax1.twinx()
        color_white = "tab:blue"
        ax2.plot(evoked.times, gfp_white, color=color_white, linewidth=2, label="Whitened GFP")
        ax2.set_ylabel(r"Whitened GFP (Scaled by $\sqrt{N}$)", color=color_white)
        ax2.tick_params(axis="y", labelcolor=color_white)

        ax2.axhline(1.0, color="crimson", linestyle="--", alpha=0.7, label="Expected Noise Floor")
        ax2.axhline(0.0, color="black", linewidth=0.8)
        ax2.set_ylim(bottom=0)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(title)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_gfp_comparison(
    evoked_ref: mne.Evoked,
    dict_of_evoked_est: dict[str, mne.Evoked],
    norm_baseline: Optional[tuple[float, float]] = None,
):
    """GFP of a reference evoked against one or more estimates (e.g.
    inverse-reconstructed virtual sensors), optionally baseline-normalized."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if norm_baseline is not None:
        t_start, t_end = evoked_ref.time_as_index(norm_baseline)
    else:
        t_start = t_end = None

    ref_gfp = get_gfp(evoked_ref.data)
    if norm_baseline is not None:
        ref_gfp = normalize_timeserie_by_baseline(ref_gfp, baseline=(t_start, t_end))
    ax.plot(evoked_ref.times, ref_gfp, color="black", linewidth=2, label="Reference")

    for name, evoked_est in dict_of_evoked_est.items():
        gfp = get_gfp(evoked_est.data)
        if norm_baseline is not None:
            gfp = normalize_timeserie_by_baseline(gfp, baseline=(t_start, t_end))
        ax.plot(evoked_est.times, gfp, label=name, linestyle="--")

    ax.set_title("Global Field Power Comparison")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Spatial Standard Deviation")
    ax.legend()

    plt.close(fig)
    return fig
