"""
conftest file
"""

import pytest
import numpy as np
import mne

# pylint: disable=redefined-outer-name

@pytest.fixture(scope="session")
def temporal_sequence():
    """
    Single channel temporal sequence fixture.
    The signal is a sinusoidal with noise amplitude until half the
    sequence. Then the amplitude is increased by a factor.
    """
    samples   = 100
    cycles    = 10
    fr        = cycles/samples   # frequency (in samples⁻¹)
    noise_amp = 5e-6             # noise amplitude
    factor    = 4                # factor between noise and signal

    sequence       = noise_amp * np.sin(2*np.pi*fr*np.arange(samples))
    sequence[50:] *= factor

    return sequence

@pytest.fixture(scope="session")
def montage():
    """
    Montage fixture
    """
    output = mne.channels.make_standard_montage("biosemi16")

    return output

@pytest.fixture(scope="session")
def info(montage):
    """
    Info fixture.
    """
    sfreq   = 250
    output  = mne.create_info(montage.ch_names, sfreq, "eeg").set_montage(montage)

    return output

@pytest.fixture(scope="session")
def raw(temporal_sequence, info):
    # data
    recording    = np.hstack([np.vstack([temporal_sequence]*16)]*5)
    onsets       = [0.2, 0.6, 1.0, 1.4, 1.8]
    durations    = np.ones_like(onsets)*0.2
    descriptions = np.ones_like(onsets)

    # mne object
    output      = mne.io.RawArray(recording, info, verbose="warning")
    annotations = mne.Annotations(onsets, durations, descriptions)
    output      = output.set_annotations(annotations, verbose="warning")

    return output

@pytest.fixture(scope="session")
def epoch(temporal_sequence, info):
    """
    epoch EEG fixture
    """
    # data
    recording = [np.vstack([temporal_sequence]*16)]*5

    # mne object
    output = mne.EpochsArray(recording, info, verbose="warning")

    return output

@pytest.fixture(scope="session")
def ave(temporal_sequence, info):
    """
    ave EEG fixture
    """
    # data
    recording = np.vstack([temporal_sequence]*16)
    nave      = 5

    # mne object
    output  = mne.EvokedArray(recording, info, nave=nave, verbose="warning")

    return output

@pytest.fixture(scope="session", params=["nib", "ants"])
def mri():
    """
    MRI volume fixture.
    """
    pass


@pytest.fixture(scope="session")
def area():
    """
    MRI binary mask representing area.
    """
    pass