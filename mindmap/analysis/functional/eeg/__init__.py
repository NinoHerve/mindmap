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
)

from .process import (
    compute_virtual_ieeg,
    normalize_evoked,
)

from .simulation import build_eeg_template


__all__ = [
    "get_ch_positions",
    "get_fiducials",
    "get_rising_phase",
    "peak_baseline_snr",
    "estimate_snr",
    "compute_virtual_ieeg",
    "normalize_evoked",
    "build_eeg_template",
]
