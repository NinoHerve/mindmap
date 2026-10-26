"""
This module performs unit tests for analysis/functional/eeg/metric.py.
"""

import mne
import numpy as np
import pytest

from mindmap.analysis.functional.eeg.metric import get_rising_phase


def _spike(peak_idx, sfreq=1000.0, tmin=-0.2, crop=(-0.02, 0.15)):
    """A triangular spike peaking at sample `peak_idx` of the cropped evoked."""
    info = mne.create_info(2, sfreq, "eeg")
    n = int(0.6 * sfreq)
    evoked = mne.EvokedArray(np.zeros((2, n)), info, tmin=tmin, verbose=False).crop(*crop)
    t = np.arange(evoked.times.size)
    evoked.data[0] = np.clip(1 - np.abs(t - peak_idx) / 6, 0, None)
    evoked.data[1] = 0.5 * evoked.data[0]
    return evoked


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1])
def test_rising_phase_index_matches_its_time_after_a_crop(ratio):
    # At 1000 Hz, about one cropped sample in nine sits just below an integer
    # after (t - times[0]) * sfreq, which truncation put one sample early.
    for peak_idx in range(10, 160):
        evoked = _spike(peak_idx)
        idx = get_rising_phase(evoked, ratio, as_idx=True)[0]
        assert evoked.times[idx] == get_rising_phase(evoked, ratio)
