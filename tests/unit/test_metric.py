"""
This module performs unit tests of the metric.py module.
"""
import numpy as np
import pytest

from mindmap.metric import *


@pytest.mark.parametrize("input_signal, expected", [
    (np.arange(5), np.array([2])),
    (np.tile(np.arange(5), (3,1)), np.array([2,2,2])),
])
def test_avg_power(input_signal: np.ndarray, expected: np.ndarray):
    """
    Test average power function.
    """
    # act
    output_value = avg_power(input_signal)

    # assertion
    assert output_value == pytest.approx(expected)


EXPECTED_PEAK_BASELINE_SNR = np.array([20*np.log10(2e-5/(5e-6/np.sqrt(2)))]*16)
@pytest.mark.parametrize(
    "dummy_eeg, baseline_window, epoch_window, expected",
    [
        ("raw",   (-0.1, -0.06), (-0.2, 0.2), EXPECTED_PEAK_BASELINE_SNR),
        ("epoch", (0.1, 0.14),   None,        EXPECTED_PEAK_BASELINE_SNR),
        ("ave",   (0.1, 0.14),   None,        EXPECTED_PEAK_BASELINE_SNR),
    ],
)
def test_peak_baseline_snr(request, dummy_eeg, baseline_window, epoch_window, expected):
    """
    Test peak_baseline_snr.
    """
    # arrange
    dummy_eeg = request.getfixturevalue(dummy_eeg)

    # act
    snr = peak_baseline_snr(dummy_eeg, baseline_window, epoch_window)

    # assert
    assert snr == pytest.approx(expected, abs=1e-1)


def test_point_to_area_distance():
    """
    Test point_to_area_distance.
    """
    pass