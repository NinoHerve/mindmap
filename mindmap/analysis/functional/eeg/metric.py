"""
Module Name: analysis/functional/eeg/metric.py
Description: Functions to measure EEG properties.

This module provides functions to measure EEG properties not implemented by MNE.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""


import numpy as np

import mne

from mindmap.core.metric import (
    avg_power, calc_spatial_correlation, calc_pairwise_dist,
    calc_spatial_dispersion_grova, calc_ratio_spurious_activity,
    compute_distances,
)
from mindmap.core.utils import get_spike_extent_mask, get_spurious_mask
from .io import get_ch_positions
from .utils import same_evoked

def get_rising_phase(
    eeg    : mne.Evoked,
    ratio  : float = 0.5,
    tmin   : float = None,
    tmax   : float = None,
    as_idx : bool  = False,
) -> float:
    """
    Finds rising phase of spike.

    Parameters
    ----------
    eeg: mne.Evoked 
        EEG. 
    
    ratio: float in [0-1]
        Ratio of the rising phase. A ratio of 1 corresponds to the peak.

    tmin: float
        Minimum point in time to consider for peak (in [s])

    tmax: float
        Maximum point in time to consider for peak (in [s])

    as_idx: bool
        Whether to return the time index.

    Returns
    -------
    out: float
        Time of peak or idx of peak if `as_idx` is True. 
    """
    data = eeg.copy() 

    # get peak
    channel, peak_time, peak_value = data.get_peak(tmin=tmin, tmax=tmax, return_amplitude=True)

    # set threshold
    threshold = ratio * np.abs(peak_value)

    # slice signal up to peak
    data.crop(tmin, peak_time)
    pre_peak = np.abs(data.get_data(channel).squeeze())

    # find last index before peak where it crossed threshold
    above_thresh_indices = np.where(pre_peak >= threshold)[0]
    rising_phase_idx     = above_thresh_indices[0] if above_thresh_indices.size > 0 else None

    # find time from idx
    out = data.times[rising_phase_idx]

    if as_idx:
        out = eeg.time_as_index(out)

    return out

def estimate_snr(
        eeg      : mne.Evoked,
        baseline : list[float] = None,
        signal   : list[float] = None
        ) -> np.ndarray:
    """
    Estimate SNR from EEG. 

    Parameters
    ----------
    eeg: mne.Evoked
        EEG

    baseline: list[float]
        tmin and tmax of baseline. If `[]` then first 30% of time serie will be considered as 
        baseline.

    signal: list[float]
        tmin and tmax of signal. If `[]` then only the max of recordings will be taken.

    Returns
    -------
        Estimated SNR at each electrode. 
    """
    if baseline is None:
        tmax     = int(0.3*len(eeg.times))
        baseline = [eeg.times[0], eeg.times[tmax]]

    eeg_      = eeg.copy()
    baseline  = eeg_.crop(tmin=baseline[0], tmax=baseline[1])
    pow_noise = avg_power(baseline.data)

    if signal is None:
        pow_sig = eeg.data.max(axis=1)**2

    else:
        eeg_    = eeg.copy()
        signal  = eeg_.crop(tmin=signal[0], tmax=signal[1])
        pow_sig = avg_power(signal.data)

    return 10*np.log10(pow_sig/pow_noise)


def peak_baseline_snr(
        sensor          : mne.io.Raw | mne.Epochs | mne.Evoked,
        baseline_window : tuple = (-0.44, -0.2),
        epoch_window    : tuple = (-0.45, 0.45),
        ) -> np.ndarray:
    """
    Estimate amplitude signal to noise ratio.

    Parameters
    ----------
    sensor: mne.io.Raw | mne.Epochs | mne.Evoked
        Sensor data.

    baseline_window: tuple
        Baseline window (same window will be applied to each epoch).

    epoch_window: tuple
        Epoch windows if sensor is Raw.

    Returns
    -------
    snr: np.ndarray
        SNR for each channel (averaged accross epochs).
    """

    if isinstance(sensor, (mne.io.Raw, mne.io.RawArray)):
        # split into epochs
        events = mne.events_from_annotations(sensor, verbose="warning")[0]
        sensor = mne.Epochs(sensor, events, tmin=epoch_window[0], tmax=epoch_window[1], baseline=baseline_window, verbose="warning")

    if isinstance(sensor, (mne.Epochs, mne.EpochsArray)):
        epochs_ = sensor.copy()
        epochs_.load_data()

        # compute snr for each epoch
        amps = np.array([np.max(np.abs(ep), axis=1) for ep in epochs_])                                           # n_epochs * n_channels
        stds = np.array([np.std(ep, axis=1) for ep in epochs_.crop(baseline_window[0], baseline_window[1])])      # n_epochs * n_channels
        snr  = 20*np.log10(amps/stds)

        # average accross epochs
        snr = np.mean(snr, axis=0)

    else:
        baseline = sensor.copy().crop(baseline_window[0], baseline_window[1])
        amp      = np.max(np.abs(sensor.data), axis=1)
        std      = np.std(baseline.data, axis=1)
        snr      = 20*np.log10(amp/std)

    return snr


def compute_spatial_correlation(
    evoked_ref : mne.Evoked,
    evoked_est : mne.Evoked,
    average    : bool = False,
) -> np.ndarray | float:
    """
    Computes spatial correlation between evoked objects.
    Assumes objects have the same channels and timepoints.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        Reference activity (e.g. true iEEG)

    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG)

    average: bool defaults to True
        Whether to compute average spatial correlation over time points.

    Returns
    -------
    sp_cor: float or np.ndarray of size (n_timepoints,)
        Average spatial correlation (if average is True) or spatial 
        correlation per timepoint.
    """
    if not same_evoked(evoked_ref, evoked_est):
        raise ValueError("Evokeds must have same channels and timepoints.")

    # extract data
    data_ref = evoked_ref.data
    data_est = evoked_est.data

    # spatial correlation
    sp_cor = calc_spatial_correlation(data_ref, data_est)

    return np.mean(sp_cor) if average else sp_cor


def compute_minimum_distance(
    evoked_ref : mne.Evoked,
    evoked_est : mne.Evoked,
    average    : bool = False,
) -> np.ndarray | float:
    """
    Computes the minimum euclidean distance between the maximum channel
    value of the reference and the maximum channel value of the 
    estimated.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        Reference activity (e.g. true iEEG)

    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG)

    average: bool defaults to True
        Whether to compute average spatial correlation over time points.

    Returns
    -------
    dmin: float or np.ndarray of size (n_timepoints,)
        Minimum euclidean distance between channels with maximum
        activity.
    """
    if not same_evoked(evoked_ref, evoked_est):
        raise ValueError("Evokeds must have same channels and timepoints.")

    # extract data
    ch_pos   = get_ch_positions(evoked_ref)     # (n_channels, 3)
    data_ref = evoked_ref.data                                  # (n_channels, n_timepoints)
    data_est = evoked_est.data

    # maximum channels
    ch_ref = ch_pos[np.argmax(data_ref, axis=0)]
    ch_est = ch_pos[np.argmax(data_est, axis=0)]

    # distance
    dist = calc_pairwise_dist(ch_ref, ch_est)

    return np.mean(dist) if average else dist


def compute_spatial_dispersion(
    evoked_est          : mne.Evoked,
    true_spatial_extent : list[str] | float = 0.7,
    evoked_ref          : mne.Evoked | None = None,
    average             : bool = False,
) -> np.ndarray:
    """
    Computes Spatial Dispersion (SD) as defined by Grova et al.
    
    Parameters
    ----------
    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG)

    true_spatial_extent: list[str]
        Name of channels belonging to spatial extent or percentile to
        threshold channels belonging to extent.

    evoked_ref: mne.Evoked
        Reference activity (e.g. true iEEG)

    average: bool defaults to True
        Whether to compute average spatial dispersion over time points.

    Returns
    -------
    sp_ds: float or np.ndarray of size (n_timepoints,)
        Spatial dispersion or averaged spatial dispersion.
        
    """

    # 1. Coordinate and Estimation data extraction
    data_est = evoked_est.data.T
    ch_pos   = get_ch_positions(evoked_est)

    # 2. Determine Set H Indices
    if isinstance(true_spatial_extent, (float, int)):
        # automatic mode
        if evoked_ref is None:
            raise ValueError(
                "evoked_ref must be provided when true_spatial_extent is a float "
                "to calculate the automatic spike extent."
            )

        # check consistency
        if not same_evoked(evoked_ref, evoked_est):
            raise ValueError("evoked_ref and evoked_est must have same channels/timepoints.")

        extent_mask = get_spike_extent_mask(evoked_ref.data.T, perc_max_amp=true_spatial_extent)

    else:
        # manual mode
        extent_mask = np.isin(evoked_est.ch_names, true_spatial_extent)

    if extent_mask.sum() == 0:
        raise ValueError("Set H (true_spatial_extent) is empty. Check channel names or threshold.")

    ch_ref_coords = ch_pos[extent_mask]

    # 3. Compute distance to set H
    dist_matrix = compute_distances(ch_pos, ch_ref_coords)    # (n_channels, size_extent)
    dist_to_h   = np.min(dist_matrix, axis=1)                                # (n_channels,)

    # compute spatial dispersion
    sp_dis = calc_spatial_dispersion_grova(data_est, dist_to_h)

    return np.mean(sp_dis) if average else sp_dis


def compute_ratio_spurious(
        evoked_est : mne.Evoked,
        true_spatial_extent : list[str] | float = 0.7,
        spurious_dist: float = 0.01,
        evoked_ref : mne.Evoked = None,
        average : bool = False,
):
    """
    Computes ratio of spurious activity as defined by Grova et al.
    
    Parameters
    ----------
    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG)

    true_spatial_extent: list[str]
        Name of channels belonging to spatial extent or percentile to
        threshold channels belonging to extent.

    evoked_ref: mne.Evoked
        Reference activity (e.g. true iEEG)

    average: bool defaults to True
        Whether to compute average spatial dispersion over time points.

    Returns
    -------
    sp_ds: float or np.ndarray of size (n_timepoints,)
        Spatial dispersion or averaged spatial dispersion.
        
    """
    # 1. Coordinate and Estimation data extraction
    data_est = evoked_est.data.T
    ch_pos   = get_ch_positions(evoked_est)

    # 2. Determine Set H Indices
    if isinstance(true_spatial_extent, (float, int)):
        # automatic mode
        if evoked_ref is None:
            raise ValueError(
                "evoked_ref must be provided when true_spatial_extent is a float "
                "to calculate the automatic spike extent."
            )

        # check consistency
        if not same_evoked(evoked_ref, evoked_est):
            raise ValueError("evoked_ref and evoked_est must have same channels/timepoints.")

        extent_mask = get_spike_extent_mask(evoked_ref.data.T, perc_max_amp=true_spatial_extent)

    else:
        # manual mode
        extent_mask = np.isin(evoked_est.ch_names, true_spatial_extent)

    # 3. Determine spurious mask
    is_distant_mask = get_spurious_mask(extent_mask, ch_pos, spurious_dist = spurious_dist)

    # 4. Compute ratio of spurious activity
    rsa = calc_ratio_spurious_activity(data_est, is_distant_mask)

    return np.mean(rsa) if average else rsa


def compute_grova_metrics(
    evoked_ref : mne.Evoked,
    evoked_est : mne.Evoked,
    true_extent : list[str] | float = 0.7,
    spurious_dist : float = 0.01,
) -> dict:
    """
    Computes quantitative metrics to evaluate source reconstruction results
    projected onto virtual intracranial sensors and the true values of the
    intracranial values.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        True iEEG data.

    evoked_est: mne.Evoked
        Source reconstruction projected onto virtual iEEG.

    Returns
    -------
    metrics: dict
        corr: Average spatial correlation 5 timepoints around
        peak.
        dmin: Minimum distance between maximum values.
        sd: Average spatial dispersion 5 timepoints around
        peak.
        rsa: Average ratio of spurious activity 5 timepoints around peak.
    """
    # check consistency
    if not same_evoked(evoked_ref, evoked_est):
        raise ValueError("evoked_ref and evoked_est must have same channels/timepoints.")

    # 1. Get peak time point index
    peak_idx = get_rising_phase(evoked_ref, ratio=1, as_idx=True)

    # 2. Extract 5-point window
    start, stop = max(0, peak_idx - 2)[0], min(len(evoked_ref.times), peak_idx + 3)[0]

    # Data transposed to (n_timepoints, n_channels)
    win_ref = evoked_ref.data[:, start:stop].T
    win_est = evoked_est.data[:, start:stop].T
    coords = get_ch_positions(evoked_ref)

    # 3. Determine extent mask and spurious mask.
    if isinstance(true_extent, (float, int)):
        # automatic mode
        extent_mask = get_spike_extent_mask(win_ref, perc_max_amp=true_extent)
    else:
        # manual mode
        extent_mask = np.isin(evoked_est.ch_names, true_extent)

    is_distant_mask = get_spurious_mask(extent_mask, coords, spurious_dist)

    # 4. pre-calculate distance components
    dist_matrix = compute_distances(coords, coords[extent_mask])
    dist_to_h = np.min(dist_matrix, axis=1)

    # 5. Spatial peaks (at the peak_idx moment)
    rel_peak = peak_idx - start
    ch_ref_peak = coords[np.argmax(np.abs(win_ref[rel_peak, :]))]
    ch_est_peak = coords[np.argmax(np.abs(win_est[rel_peak, :]))]

    return {
        "spatial_correlation": np.mean(calc_spatial_correlation(win_ref, win_est)),
        "min_distance": calc_pairwise_dist(ch_ref_peak, ch_est_peak)[0],
        "spatial_dispersion": np.mean(calc_spatial_dispersion_grova(win_est, dist_to_h)),
        "ratio_spurious": np.mean(calc_ratio_spurious_activity(win_est, is_distant_mask)),
    }
