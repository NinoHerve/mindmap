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
from scipy import ndimage

import mne

from mindmap.core.metric import (
    calc_correlation, calc_pairwise_dist,
    calc_spatial_dispersion_grova, calc_ratio_spurious_activity,
    compute_distances, estimate_snr, calc_correlation_matrix,
    get_gfp, compute_cog_trajectory, calc_rem,
)

from mindmap.core.utils import get_spike_extent_mask, get_spurious_mask

from mindmap.analysis.functional.common.metric import (
    compute_mne_confusion, compute_mne_scores, compute_mne_curves,
)

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

def find_spike_window(
    evoked: mne.Evoked,
    baseline: tuple = (None, 0),
    threshold_sd: float = 3.0,
    gap_fill_ms: float = 100.0,
    min_duration_ms: float = 20.0,
):
    """
    Finds the continuous time window of a spike based on Global Filed Power (GFP).

    Parameters
    ----------
    evoked: mne.Evoked
        Evoked response.

    baseline: tuple[float, float]
        Basline range in [ms].
    
    threshold_sd: float
        Threshold in standard deviation above which to consider
        signal as spike.

    gap_fill_ms: float
        Maximum gap between events before considering different
        events.

    min_duration_ms: float
        Minimum duration to consider event as signal.

    Returns
    -------
    tmin, tmax: float
        Starting/Ending time of spike.
    """
    times = evoked.times
    sfreq = evoked.info["sfreq"]

    # compute GFP
    gfp = get_gfp(evoked.data, axis=0)

    # baseline stats
    b_mask = (times >= (baseline[0] or times[0])) & (times <= (baseline[1] or 0))
    b_mean = np.mean(gfp[b_mask])
    b_std = np.std(gfp[b_mask])

    # create initial binary mask
    spike_mask = gfp > (b_mean + threshold_sd * b_std)

    # fill crevasses (morphological closing)
    gap_samples = int((gap_fill_ms / 1000.0) * sfreq)
    spike_mask = ndimage.binary_closing(spike_mask, structure=np.ones(gap_samples))

    # remove short bursts (morphological opening)
    min_samples = int((min_duration_ms / 1000.0) * sfreq)
    spike_mask = ndimage.binary_opening(spike_mask, structure=np.ones(min_samples))

    # extract start and end of largest continuous segment
    label_mask, num_labels = ndimage.label(spike_mask)
    if num_labels == 0:
        print("Warning: No spike detected above threshold.")
        return times[0], times[-1]

    sizes = ndimage.sum(spike_mask, label_mask, range(1, num_labels + 1))
    main_label = np.argmax(sizes) + 1

    main_spike_indices = np.where(label_mask == main_label)[0]
    tmin, tmax = times[main_spike_indices[0]], times[main_spike_indices[-1]]

    return tmin, tmax


def estimate_evoked_snr(
    evoked: mne.Evoked,
    baseline: tuple[float, float],
    noise_cov: mne.Covariance=None,
    method: str="power",
)-> tuple[np.ndarray, np.ndarray]:
    """
    Compute sensor-space SNR time course from an evoked response.

    Parameters
    ----------
    evoked: mne.Evoked
        Evoked data.

    baseline: tuple[float, float]
        Time window (s) for noise estimate.

    noise_cov: mne.Covariance, optional
        If provided, data is whitened before SNR estimation, making 
        the result consistent with MNE's regularization framework.
    
    method: str ['power', 'variance']
        Whether to use power-based or variance-based 
        estimate.

    Returns
    -------
    snr, snr_db: tuple[np.ndarray, np.ndarray] of shapes (n_times,)
    """
    data = evoked.data
    if noise_cov is not None:
        whitener, _ = mne.cov.compute_whitener(noise_cov, evoked.info, pca=False)
        data = whitener @ data

    b_start, b_stop = evoked.time_as_index(baseline)
    return estimate_snr(data, (b_start, b_stop), method=method)



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


def compute_evoked_correlation(
    evoked_ref: mne.Evoked,
    evoked_est: mne.Evoked,
    mode: str = "temporal",
    method: str = "pearson",
    threshold: float | None = None,
    average: bool = False,
) -> np.ndarray | float:
    """
    Compute correlation between two evoked objects.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        Reference activity (e.g. True iEEG).

    evoked_est: mne.Evoked
        Estimated activity (e.g. virtual iEEG).

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': Correlation across channels at each timepoint -> (n_timepoints,)
        - 'temporal': Correlation across time for each channel -> (n_channels,)
        - 'sptio-temporal': Flattened correlation -> scalar

    method: str ['pearson', 'spearman', 'log-envelope']
        Correlation method.

    threshold: float | None
        If provided, only timepoints where Ground Truth GFP is above this 
        value are included in the calculation.
    
    average: bool
        If True, return the mean over the result vector.

    Returns
    -------
    corr: float or np.ndarray
        Correlation value(s). Shape depends on axis and average.
    """
    if not same_evoked(evoked_ref, evoked_est):
        raise ValueError("Evokeds must have same channels and timepoints.")

    # mapping mode to axis
    mode_map = {"spatial":0, "temporal":1, "spatio-temporal":None}
    if mode not in mode_map:
        raise ValueError(f"Unknown mode '{mode}'. Use 'spatial', 'temporal', or 'spatio-temporal'.")

    axis = mode_map[mode]

    # handle thresholding (Masking)
    if threshold is not None:
        # calculate GFP of the reference to find "active" timepoints
        gfp_ref = get_gfp(evoked_ref.data, axis=0)
        mask = gfp_ref > threshold

        # slice the data to only include active timepoints
        # Note: this only makes sense for 'spatial' or 'spatio-temporal' modes
        ref_data = evoked_ref.data[:, mask]
        est_data = evoked_est.data[:, mask]

        if ref_data.shape[1] == 0:
            print("Warning: No timepoints exceed the threshold. Returning 0.")
            return 0.0
    else:
        ref_data = evoked_ref.data
        est_data = evoked_est.data

    corr = calc_correlation(
        ref_data, est_data, method=method, axis=axis,
    )
    corr = np.mean(corr) if average else corr

    if threshold is not None:
        return corr, evoked_ref.times[mask]

    return corr


def compute_evoked_corr_matrix(
    evoked1: mne.Evoked, 
    evoked2: mne.Evoked,
    window: tuple[float, float]=(None, None),
    method: str = "pearson",
) -> np.ndarray:
    """
    Compute a correlation matrix where result[i, j] is the correlation
    between channel i and channel j.

    Parameters
    ----------
    evoked1: mne.Evoked

    evoked2: mne.Evoked

    method: str ['pearson', 'spearman', 'log-envelope']

    Returns
    -------
    corr_matrix: np.ndarry of shape (n_channels_1, n_channels_2)
    """
    # time window defaults
    tmin = window[0] if window[0] is not None else evoked1.tmin
    tmax = window[1] if window[1] is not None else evoked1.tmax

    # crop copies to the desired window
    evk1 = evoked1.copy().crop(tmin=tmin, tmax=tmax)
    evk2 = evoked2.copy().crop(tmin=tmin, tmax=tmax)

    data1 = evk1.data
    data2 = evk2.data

    # compute correlation matrix
    return calc_correlation_matrix(data1, data2, method=method, eps=1e-8)



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
        "spatial_correlation": np.mean(calc_correlation(win_ref, win_est, axis=0, method="pearson")),
        "min_distance": calc_pairwise_dist(ch_ref_peak, ch_est_peak)[0],
        "spatial_dispersion": np.mean(calc_spatial_dispersion_grova(win_est, dist_to_h)),
        "ratio_spurious": np.mean(calc_ratio_spurious_activity(win_est, is_distant_mask)),
    }


def compute_evoked_cog(evoked: mne.Evoked) -> np.ndarray:
    """
    Extracts channel positions and computes CoG trajectory.
    """
    pos = get_ch_positions(evoked)

    if np.allclose(pos, 0):
        raise ValueError("Channel positions are missing. Please set a montage.")

    return compute_cog_trajectory(evoked.data, pos)

def compute_evoked_rem(
    evoked_ref: mne.Evoked,
    evoked_est: mne.Evoked,
    mode: str = "spatial",
    rng: tuple[float, float] = (None, None),
) -> np.ndarray:
    """
    Compute REM for Evoked data.

    Parameters:
    ----------
    evoked_ref: mne.Evoked
        Reference evoked.

    evoked_est: mne.Evoked
        Estimated evoked.

    mode: str ['spatial', 'temporal']
        'spatial' -> Energy of each channel relative to total spatial energy
                     at that timepoint. Result: (n_channels, n_timepoints)
        'temporal' -> Energy at each timepoint relative to total temporal energy
                      for that channel. Result: (n_channels, n_timepoints)
    """
    # crop
    evk_ref = evoked_ref.copy().crop(tmin=rng[0], tmax=rng[1])
    evk_est = evoked_est.copy().crop(tmin=rng[0], tmax=rng[1])

    # map mode to the axis we want to normalize
    mode_map = {"spatial":0, "temporal":1}

    if mode not in mode_map:
        raise ValueError("Mode must be 'spatial' or 'temporal'")

    return calc_rem(
        evk_ref.data,
        evk_est.data,
        axis = mode_map[mode],
    )


def compute_evoked_confusion(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "spatio-temporal",
    rng: tuple = (None, None),
) -> dict:
    """
    Wrapper for `compute_mne_confusion`

    Parameters
    ----------
    evoked_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    evoked_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.
    
    threshold:
        Relative maximum threshold for binarization.

    rng: tuple (float, float)
        Range of data to consider for scoring.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_confusion(
        obj_true = evoked_true,
        obj_pred = evoked_pred,
        mode = mode,
        rng = rng,
    )


def compute_evoked_scores(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "spatio-temporal",
    rng: tuple = (None, None),
) -> dict:
    """
    Wrapper for `compute_mne_scores`

    Parameters
    ----------
    evoked_true: mne.SourceEstimate
        Ground truth activity.
    
    evoked_pred: mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.
    
    threshold:
        Relative maximum threshold for binarization.

    rng: tuple (float, float)
        Range of data to consider for scoring.
    
    Returns
    -------
    dict
        Dictionary of scores.
    """
    return compute_mne_scores(
        obj_true = evoked_true,
        obj_pred = evoked_pred,
        mode = mode,
        rng = rng,
    )


def compute_evoked_curves(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    rng: tuple = (None, None),
) -> dict:
    """
    Wrapper for `compute_mne_curves`.

    Parameters
    ----------
    evoked_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.
    
    evoked_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    true_threshold: dict
        Keyword arguments for `binarize_mne_object`.

    rng: tuple (float, float)
        Range of data to consider for scoring.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_curves(
        obj_true = evoked_true,
        obj_pred = evoked_pred,
        rng = rng,
    )