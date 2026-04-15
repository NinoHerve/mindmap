"""
Modules for managing sensor objects.
"""

from .io import (
    get_ch_positions,
    get_fiducials,
)

from .metric import (
    get_rising_phase,
    peak_baseline_snr,
    estimate_snr,
    compute_evoked_correlation,
    compute_spatial_dispersion,
    compute_minimum_distance,
    compute_ratio_spurious,
    compute_grova_metrics,
    estimate_evoked_snr,
    compute_evoked_corr_matrix,
    calc_rem,
    find_spike_window,
    compute_evoked_confusion,
    compute_evoked_scores,
    compute_evoked_curves,
)

from .process import (
    compute_virtual_ieeg,
    normalize_evoked,
    align_eeg_timeseries,
    normalize_evoked_by_baseline,
    binarize_evoked,
)

from .simulation import build_eeg_template

from .utils import scale_montage


__all__ = [
    "get_ch_positions",
    "get_fiducials",
    "get_rising_phase",
    "peak_baseline_snr",
    "estimate_snr",
    "compute_virtual_ieeg",
    "normalize_evoked",
    "binarize_evoked",
    "build_eeg_template",
    "compute_spatial_dispersion",
    "compute_minimum_distance",
    "compute_ratio_spurious",
    "compute_grova_metrics",
    "scale_montage",
    "align_eeg_timeseries",
    "normalize_evoked_by_baseline",
    "compute_evoked_correlation",
    "estimate_evoked_snr",
    "compute_evoked_corr_matrix",
    "calc_rem",
    "find_spike_window",
    "compute_evoked_confusion",
    "compute_evoked_scores",
    "compute_evoked_curves",
]
