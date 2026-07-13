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
    compute_mne_confusion, compute_mne_scores,
    compute_mne_roc, compute_mne_pr,
    compute_mne_correlation, compute_mne_cosine,
    compute_mne_pattern_geometry_correlation,
    compute_mne_explained_variance, compute_mne_ssim,
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


def estimate_raw_snr(
    raw: mne.io.Raw,
    events,
    event_id: int = None,
    tmin: float = -0.05,
    tmax: float = 0.05,
    baseline: tuple[float, float] = (-0.05, -0.005),
    signal_window: tuple[float, float] = None,
    picks: str = "eeg",
):
    """
    Computes a robust single-trial SNR metric designed to anchor
    generative simulations based on median single-trial statistics.
    
    Parameters
    ----------
    raw: mne.io.Raw
        The raw continuous data.

    events: array, shape (n_events, 3)
        The MNE events array.

    event_id: int
        The id of the event to epoch around.

    tmin, tmax: float
        Total time window around the event. Must encompass both baseline
        and signal winodws completely.

    baseline: tuple of float (start, end)
        The time window to extract background noise from.

    signal_window: tuple of float (start, end)
        The time window where the spike peak is expected to occur. If None,
        defaults to (0, tmax).

    picks: str | list
        Channels to include.

    Returns
    -------
    snrs:
        SNR per channel.
    """
    # Default signal window
    if signal_window is None:
        signal_window = (0.0, tmax)

    # Load epochs without MNE's baseline correction (keep noise intact)
    epochs = mne.Epochs(
        raw,
        events,
        event_id = event_id,
        tmin = tmin,
        tmax = tmax,
        baseline = None,
        picks = picks,
        preload = True,
        verbose = False,
    )

    # data shape: (n_epochs, n_channels, n_times)
    data = epochs.get_data(copy=False)
    times = epochs.times

    # 1. Isolate the Signal Window and find peak per trial, per channel
    s_start = np.argmin(np.abs(times - signal_window[0]))
    s_end = np.argmin(np.abs(times - signal_window[1]))
    spike_data = data[:, :, s_start:s_end]
    single_trial_peaks = np.max(np.abs(spike_data), axis=2)

    # 2. Isolate the Baseline Window and find noise standard deviation per trial
    # per channel
    b_start = np.argmin(np.abs(times - baseline[0]))
    b_end = np.argmin(np.abs(times - baseline[1]))
    baseline_data = data[:, :, b_start:b_end]
    single_trial_noise = np.std(baseline_data, axis=2)

    # 3. Take the median across all trials (axis 0) for each channel seperately
    median_peaks = np.median(single_trial_peaks, axis=0)
    median_noise = np.median(single_trial_noise, axis=0)

    # 4. Comptue SNR per channel, protecting against any dead/flat channels.
    snrs = np.divide(
        median_peaks,
        median_noise,
        out = np.zeros_like(median_peaks),
        where = median_noise != 0,
    )

    return snrs



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
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "temporal",
    method: str = "pearson",
    polarity_invariant: bool = False,
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

    polarity_invariant: bool
        If True, returns |corr| instead of corr, so an overall sign flip
        (e.g. an arbitrary polarity/reference convention) doesn't score as a
        worse match than a perfectly in-phase one. See
        `mindmap.core.metric.calc_correlation`.

    Returns
    -------
    corr: float or np.ndarray
        Correlation value(s). Shape depends on axis and average.
    """
    if not same_evoked(evoked_true, evoked_pred):
        raise ValueError("Evokeds must have same channels and timepoints.")

    return compute_mne_correlation(
        evoked_true,
        evoked_pred,
        mode = mode,
        method = method,
        polarity_invariant = polarity_invariant,
    )


def compute_evoked_cosine(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "temporal",
    polarity_invariant: bool = False,
) -> np.ndarray | float:
    """
    Compute cosine between two evoked objects.

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

    polarity_invariant: bool
        If True, returns |cos| instead of cos -- see
        `mindmap.core.metric.calc_cosine`.

    Returns
    -------
    corr: float or np.ndarray
        Correlation value(s). Shape depends on axis and average.
    """
    if not same_evoked(evoked_true, evoked_pred):
        raise ValueError("Evokeds must have same channels and timepoints.")

    return compute_mne_cosine(
        evoked_true,
        evoked_pred,
        mode = mode,
        polarity_invariant = polarity_invariant,
    )


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


def compute_evoked_pattern_geometry_correlation(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    method: str = "pearson",
    compare_method: str = "spearman",
) -> float:
    """
    Second-order ("representational similarity") comparison between two
    evoked objects: whether the geometry of how the spatial pattern
    evolves across time matches, rather than the raw topographies
    themselves. See `mindmap.core.metric.calc_pattern_geometry_correlation`
    for the full explanation.

    Parameters
    ----------
    evoked_true: mne.Evoked
        Reference activity (e.g. True iEEG).

    evoked_pred: mne.Evoked
        Estimated activity (e.g. virtual iEEG).

    method: str ['pearson', 'spearman', 'log-envelope']
        Method used to build each evoked's own dissimilarity matrix.

    compare_method: str ['pearson', 'spearman', 'log-envelope']
        Method used for the final comparison between the two dissimilarity
        matrices.

    Returns
    -------
    float
    """
    if not same_evoked(evoked_true, evoked_pred):
        raise ValueError("Evokeds must have same channels and timepoints.")

    return compute_mne_pattern_geometry_correlation(
        evoked_true, evoked_pred, method=method, compare_method=compare_method,
    )


def compute_evoked_explained_variance(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "spatio-temporal",
    polarity_invariant: bool = False,
) -> np.ndarray | float:
    """
    Compute explained variance (R^2) between two evoked objects. Unlike
    correlation/cosine, this is sensitive to amplitude scale -- see
    `mindmap.core.metric.calc_explained_variance`.

    Parameters
    ----------
    evoked_true: mne.Evoked
        Reference activity (e.g. True iEEG).

    evoked_pred: mne.Evoked
        Estimated activity (e.g. virtual iEEG).

    mode: str ['spatial', 'temporal', 'spatio-temporal']

    polarity_invariant: bool
        If True, also scores against a sign-flipped evoked_pred and keeps
        whichever is higher -- see
        `mindmap.core.metric.calc_explained_variance`.

    Returns
    -------
    r2: float | np.ndarray
    """
    if not same_evoked(evoked_true, evoked_pred):
        raise ValueError("Evokeds must have same channels and timepoints.")

    return compute_mne_explained_variance(evoked_true, evoked_pred, mode=mode, polarity_invariant=polarity_invariant)


def compute_evoked_ssim(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    win_size: int = 7,
    data_range: float = None,
) -> float:
    """
    Compute Structural Similarity (SSIM) between two evoked objects,
    treating their data as a 2D (channels x time) signal -- see
    `mindmap.core.metric.calc_ssim`.

    Parameters
    ----------
    evoked_true: mne.Evoked
        Reference activity (e.g. True iEEG).

    evoked_pred: mne.Evoked
        Estimated activity (e.g. virtual iEEG).

    win_size: int
        Sliding comparison window size, clamped to fit the data if needed.

    data_range: float, optional
        Dynamic range of the data. Defaults to the joint range of both
        evokeds.

    Returns
    -------
    float
    """
    if not same_evoked(evoked_true, evoked_pred):
        raise ValueError("Evokeds must have same channels and timepoints.")

    return compute_mne_ssim(evoked_true, evoked_pred, win_size=win_size, data_range=data_range)


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


def compute_evoked_rem_correlation(
    evoked_ref: mne.Evoked,
    evoked_est: mne.Evoked,
    mode: str = "temporal",
    rng: tuple[float, float] = (None, None),
) -> np.ndarray | float:
    """
    Reduces `compute_evoked_rem`'s full relative-energy map to a score per
    element of `mode`'s axis: how well the estimate's energy distribution
    tracks the reference's own normalized energy distribution.

    `compute_evoked_rem(evoked_ref, evoked_est, mode)` returns
    est_energy / sum(ref_energy along mode's axis) -- not yet comparable to
    a single ideal target. Calling it a second time with the reference
    against itself, `compute_evoked_rem(evoked_ref, evoked_ref, mode)`,
    gives exactly that target (the reference's own normalized energy
    profile, which by construction sums to 1 along the reduced axis).
    Correlating the two maps *along the axis mode already reduced over*
    answers "does the estimate's energy distribution match the reference's"
    per channel (mode="temporal") or per timepoint (mode="spatial") --
    mirroring `compute_evoked_correlation`/`compute_evoked_cosine`'s
    per-mode granularity, rather than collapsing straight to one number.

    Parameters
    ----------
    evoked_ref: mne.Evoked
        Reference evoked.

    evoked_est: mne.Evoked
        Estimated evoked.

    mode: str ['spatial', 'temporal']
        Passed through to `compute_evoked_rem`.
        - 'spatial': one score per timepoint -> (n_timepoints,)
        - 'temporal': one score per channel -> (n_channels,)

    rng: tuple[float, float]
        Time window passed through to `compute_evoked_rem`.

    Returns
    -------
    float | np.ndarray
    """
    rem_ideal = compute_evoked_rem(evoked_ref, evoked_ref, mode=mode, rng=rng)
    rem_actual = compute_evoked_rem(evoked_ref, evoked_est, mode=mode, rng=rng)

    mode_map = {"spatial": 0, "temporal": 1}
    return calc_correlation(rem_ideal, rem_actual, method="pearson", axis=mode_map[mode])


def compute_evoked_confusion(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Wrapper for `compute_mne_confusion`

    Parameters
    ----------
    evoked_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.
    
    evoked_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.
    
    threshold:
        Relative maximum threshold for binarization.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    return compute_mne_confusion(
        obj_true = evoked_true,
        obj_pred = evoked_pred,
        mode = mode,
    )


def compute_evoked_scores(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
    mode: str = "spatio-temporal",
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
    )


def compute_evoked_roc(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
) -> dict:
    """
    Wrapper for `compute_mne_roc`.

    Parameters
    ----------
    evoked_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    evoked_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    dict
        Dictionary with 'fpr', 'tpr' and 'auc'.
    """
    return compute_mne_roc(
        obj_true = evoked_true,
        obj_pred = evoked_pred,
    )


def compute_evoked_pr(
    evoked_true: mne.Evoked,
    evoked_pred: mne.Evoked,
) -> dict:
    """
    Wrapper for `compute_mne_pr`.

    Parameters
    ----------
    evoked_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    evoked_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    dict
        Dictionary with 'precision', 'recall' and 'ap'.
    """
    return compute_mne_pr(
        obj_true = evoked_true,
        obj_pred = evoked_pred,
    )


def get_channel_distances(evoked: mne.Evoked, target_ch: str) -> dict[str, float]:
    """
    Calculates the spatial Euclidean distance from all valid channels to an anchor channel.

    Parameters
    ----------
    evoked: mne.Evoked
        The MNE Evoked object.

    target_ch: str
        Name of the anchor channel.

    Returns
    -------
    dict
        A dictionary mapping channel names to their distance in centimeters (cm)
    """
    # 1. Extract 3D coordinate
    channel_positions = {}
    for ch_info in evoked.info["chs"]:
        name = ch_info["ch_name"]
        loc = ch_info["loc"][:3]

        # check that the channel has genuine coordinates
        if not np.isnan(loc).any() and not (loc==0).all():
            channel_positions[name] = loc

    # 2. Guard against structural coordinate absence
    if target_ch not in channel_positions:
        raise ValueError(
            f"Target anchor channel '{target_ch}' lacks spatial 3D coordinates in this object's montage."
        )

    anchor_xyz = channel_positions[target_ch]
    calculated_distances = {}

    # 3. Vectorized Euclidean distance conversion (Meters -> Centimeters)
    for name, xyz in channel_positions.items():
        # L2 Norm calculation translates to standard straigth-line 3D Euclidean distance
        distance_cm = np.linalg.norm(xyz - anchor_xyz) * 100
        calculated_distances[name] = distance_cm

    return calculated_distances
