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
    compute_spatial_correlation,
    compute_spatial_dispersion,
    compute_minimum_distance,
    compute_ratio_spurious,
    compute_grova_metrics,
)

from .process import (
    compute_virtual_ieeg,
    normalize_evoked,
    align_eeg_timeseries,
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
    "build_eeg_template",
    "compute_spatial_correlation",
    "compute_spatial_dispersion",
    "compute_minimum_distance",
    "compute_ratio_spurious",
    "compute_grova_metrics",
    "scale_montage",
    "align_eeg_timeseries",
]
