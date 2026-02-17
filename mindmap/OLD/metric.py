"""
Module Name: metric.py
Description: Functions to measure metrics.

This module provides functions to measure various metrics. Usefull for evaluation, validation,
or simply testing.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import nibabel as nib
import mne

def avg_power(signal: np.ndarray) -> np.ndarray:
    """
    Compute average power of a signal.

    Parameters
    ----------
    signal: np.ndarray (t,) or (n,t)

    Returns
    -------
    avg_pow: np.ndarray (1,) or (n,)
        Average power for every sample n.
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(1,-1)   # consider signal as temporal signal (1,T)

    avg_pow = signal - np.mean(signal, axis=1)[:,None]
    avg_pow = np.sum(avg_pow**2, axis=1)/signal.shape[-1]

    return avg_pow

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



def point_to_area_distance(
        target_point : np.ndarray,
        target_area  : np.ndarray | nib.Nifti1Image,
        cras         : np.ndarray,
        x            : int = 1,
        ) -> float:
    """
    Measures distance between a point and an area.

    Parameters
    ----------
    target_point: np.ndarray (3,)
        Target point in [m]

    target_area: np.ndarray (3,N) | nibabel.Nifti1Image
        Cloud point in [m] if array, in [mm] if Niti1Image.

    x: int
        Number of points from the cloud point to consider.

    Returns
    -------
    distance: float
        Distance in [m]

    Note: The distance is measured as the length of the average distance vector between
    the target point and the x closest points of the point cloud.:
    - x = 1: Distance to closest point of the area.
    - x = "all": Distance to the center of mass of the area.
    - x != "all" != 1: Distance to the center of mass of the portion consisting of the
      x closest points.
    """

    if isinstance(target_area, nib.Nifti1Image):
        affine      = target_area.affine
        target_area = np.argwhere(target_area.get_fdata()>0)
        target_area = (nib.affines.apply_affine(affine, target_area)-cras)/1000


    distances = target_area-target_point

    if x == "all":
        mean_distance = np.mean(distances, axis=0)
        norm          = np.linalg.norm(mean_distance)

    else:
        norms         = np.linalg.norm(distances, ord=2, axis=1)
        nearest       = distances[np.argsort(norms)[:x],:]
        mean_distance = np.mean(nearest, axis=0)
        norm          = np.linalg.norm(mean_distance)

    return norm


def compute_distances(array1, array2, output_type="scalar"):
    """
    Computes distances between points of two arrays. Displacements
    will point from array1 to array2.

    Parameters:
    ----------
    array1: np.ndarray
        Array of shape (N,D) or (N,)
    
    array2: np.ndarray
        Array of shape (M,D) or (M,)

    output_type: str
        'scalar' (Euclidean) or 'vector' (Displacement)

    Returns
    -------
    dist: np.ndarray
        If 'scalar': Array of shape (N,M) containing L2 norms
        If 'vector': Array of shape (N,M,D) containing displacement vectors.
    """
    # Ensure inputs are numpy arrays
    a1 = np.atleast_2d(array1)
    a2 = np.atleast_2d(array2)

    # Broadcasting to find difference between every pair
    diff =  a2[np.newaxis,:,:] - a1[:,np.newaxis,:]

    if output_type == "vector":
        return diff
    elif output_type == "scalar":
        return np.linalg.norm(diff, axis=2)
    else:
        raise ValueError("output_type must be 'scalar' or 'vector'")