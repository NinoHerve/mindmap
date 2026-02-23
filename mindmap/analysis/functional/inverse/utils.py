"""
Module Name: analysis/functional/inverse/utils.py
Description: Helpers specific to the inverse module.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np


# harmonic prior
def compute_harmonic_prior(eigvals : np.ndarray, relaxation : float = 1) -> np.ndarray:
    """
    Compute the harmonic prior according to eigenvalues.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics.
    
    relaxation: float [0-1]
        Relaxes the small weights. The weight will give the probability of a 
        harmonic to be chosen during reconstruction. If taking the raw weights,
        this can be pretty tough on middle harmonics. The relaxation allows 
        middle harmonics to have more weigths. When relaxation = 0, all harmonics
        have a weigth of 1.
    """

    # compute eigenvalue gaps
    arr1 = np.abs(eigvals - np.roll(eigvals, -1))
    arr2 = np.abs(eigvals - np.roll(eigvals, 1))

    weights = (arr1 + arr2)/2

    # correct extremities
    weights[0]  = arr1[0]
    weights[-1] = arr2[-1]

    # convert to prior weights and store in prior mixing matrix
    weights = np.diag(1/weights)**relaxation

    return weights
