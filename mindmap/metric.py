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
        Target point

    target_area: np.ndarray (3,N) | nibabel.Nifti1
        Cloud point

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
