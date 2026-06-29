"""
Module Name: analysis/functional/inverse/pipeline.py
Description: Helpers to run pipelines.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path

import numpy as np

import mne

from mindmap.analysis.functional.inverse.solvers import (
    wCSET, CSET, MCE,
)

def make_process_sample(
    forward: mne.Forward,
    eig: np.ndarray = None,
    methods: list[str] = ["sloreta", "dspm"],
    lambda2: list[float] = [0.01, 0.01],
    crop: list[float] = [None, None],
):
    """
    Make a process_sample function for parallel processing.

    Parameters
    ----------
    methods: list[str]
        Source reconstruction methods.

    lambda2: list[float]
        List of regularization parameters.

    forward: mne.Forward
        Forward solution.

    ncov: mne.Covariance
        Noise covariance.

    eig: np.ndarray
        Eigenspectrum.

    crop: list[float]
        Cropping borders.
    """

    if ("wcset" in methods or "cset" in methods) and eig is None:
        raise ValueError("Can't run wCSET or CSET without eigenspectrum.")
    
    # standardize to lower case
    methods = [m.lower() for m in methods]

    # adjust casing
    mne_methods = {
        "sloreta": "sLORETA",
        "dspm": "dSPM",
        "mne": "MNE",
    }

    def process_sample(fave: str | Path, fcov: str | Path = None):
        """Process one sample."""
        # load
        data = mne.read_evokeds(fave, verbose="warning")[0]
        
        if fcov is None:
            ncov = mne.cov.make_ad_hoc_cov(data.info)
        else:
            ncov = mne.read_cov(fcov, verbose="warning")
        
        # crop
        data = data.copy().crop(tmin=crop[0], tmax=crop[1])

        # ----------------------
        # Source reconstruction
        # ----------------------

        if any(m in methods for m in ["sloreta", "dspm", "mne"]):
            inv_mne = mne.minimum_norm.make_inverse_operator(
                info = data.info,
                forward = forward,
                noise_cov = ncov,
                fixed = True,
                depth = None,
                verbose = "warning",
            )

        for i, m in enumerate(methods):
            sol = None

            if m == "cset":
                inv = CSET(
                    forward = forward,
                    noise_cov = ncov,
                    depth = 0.8,
                    eigenvectors = eig["vec"],
                )
                sol = inv.solve(data, lambda2 = lambda2[i])

            elif m == "wcset":
                inv = wCSET(
                    forward = forward,
                    noise_cov = ncov,
                    depth = 0.8,
                    eigenvectors = eig["vec"],
                    eigenvalues = eig["val"],
                    harm = 0.8,
                )
                sol = inv.solve(data, lambda2[i])

            elif m == "mce":
                inv = MCE(
                    forward = forward,
                    noise_cov = ncov,
                    depth = 0.8,
                )
                sol = inv.solve(data, lambda2 = lambda2[i])

            elif m in ["sloreta", "dspm", "mne"]: 
                sol = mne.minimum_norm.apply_inverse(
                    data,
                    inv_mne,
                    lambda2[i],
                    method = mne_methods[m],
                    verbose = "warning",
                )

            # save solution
            if sol is not None:
                foutput = str(fave)
                foutput = foutput.replace("ave", "stc")
                foutput = foutput.replace("_stc", f"_desc-{m}_stc")
                foutput = foutput.replace(".fif", "")

                sol.save(foutput, overwrite=True, verbose="warning")

        return f"Finished processing: {Path(fave).name}"

    return process_sample

