"""
Module Name: analysis/functional/inverse/utils.py
Description: Helpers specific to the inverse module.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


# Starting beta for compute_harmonic_prior's "lowpass" method (weight =
# exp(-beta*eigvals)), absent a value tuned against this specific
# connectome's own reconstruction accuracy (see
# tutorials/analysis/inverse/harmonic-prior-tuning.ipynb, which exists to do
# that tuning). Derived from the structure-function coupling literature,
# which fits this exact model -- same exponential form, same normalized-
# Laplacian convention as compute_cst (eigenvalues in [0, 2]):
#   - Abdelnour et al. 2018 (NeuroImage): fits alpha=4.08 pooling all
#     subjects' eigenvalue pairs into one group-level regression (88-node
#     AAL atlas, N=26).
#   - Cummings et al. 2022 (Frontiers in Neuroscience): reuses that same
#     model but fits it per subject (86-node Desikan-Killiany, N=83):
#     alpha ranges 0.18-2.64, mean 0.99.
# We fit/use a single subject's connectome rather than a pooled group, so
# the per-subject statistic is the closer analog -- 1.0 is that mean,
# rounded. Neither paper's parcellation (~70-90 nodes) is remotely close to
# this connectome's resolution (~7500 nodes), so this is an extrapolation
# across a real resolution gap, not a validated value for this pipeline --
# a starting point to tune from, not a settled answer.
DEFAULT_LOWPASS_BETA = 1.0


def _spline_gap_spectrum(eigvals: np.ndarray) -> np.ndarray:
    """
    Smooth version of "gap"'s local-gap idea, following the exact recipe in
    Sipes, Nagarajan & Raj 2024 (Commun Biol, "Integrative, segregative, and
    degenerate harmonics of the structural connectome"): fit a smoothing
    spline (few knots -- capturing only the broad shape of the ascending
    eigenvalue curve, not point-to-point noise) to eigenvalue vs. harmonic
    index, then take its analytical derivative as the "gap-spectrum".

    The derivative is high wherever the eigenvalue curve is climbing steeply
    (isolated, sparse eigenvalues -- integrative at the low end, segregative
    at the high end) and low wherever it's flat (densely clustered
    eigenvalues near the bulk of the spectrum -- degenerate, in the middle).
    Unlike "gap"'s raw finite differences, this doesn't need the
    `weights[5000:]` cutoff hack to behave at the high end -- it's smooth
    there by construction, the whole reason the paper smooths in the first
    place.

    The paper fits an order-10 spline with 3 knots in MATLAB; scipy's spline
    routines cap degree at 5 (FITPACK has no higher-order support), so this
    uses k=5 (the closest available) with the same 3-interior-knot count --
    an adaptation, not an exact replication.
    """
    n = len(eigvals)
    x = np.arange(n)

    interior_knots = np.linspace(0, n - 1, 5)[1:-1]  # 3 interior knots
    spline = LSQUnivariateSpline(x, eigvals, t=interior_knots, k=5)
    gap_spectrum = spline.derivative()(x)

    # a derivative of a monotonically non-decreasing curve should be >= 0;
    # guard against tiny negative spline-boundary overshoot breaking
    # fractional-power relaxation downstream (negative base, fractional
    # exponent -> nan).
    return np.clip(gap_spectrum, a_min=0, a_max=None)


# harmonic prior
def compute_harmonic_prior(eigvals : np.ndarray, relaxation : float = None, method : str = "gap") -> np.ndarray:
    """
    Compute the harmonic prior according to eigenvalues.

    Three weighting schemes, selected via `method`:

    - "gap" (default -- preserves existing behavior for all current callers):
      weights each eigenvalue by its LOCAL GAP to its neighbors in the sorted
      spectrum, i.e. by how much of an outlier it is relative to nearby
      eigenvalues, not by its own magnitude. Motivated by Sipes, Nagarajan &
      Raj 2024 (Commun Biol, "Integrative, segregative, and degenerate
      harmonics of the structural connectome" -- see "gap-spectrum" below),
      but computed as a raw finite difference rather than that paper's
      smoothed version, which makes it noisy and sensitive to the local
      clustering density of a specific graph's spectrum in ways unrelated to
      genuine structure-function relevance (see the sabotage-connectome
      investigation that motivated adding "lowpass" below). The
      `weights[5000:]` flattening is a pre-existing patch over that same
      instability at the high end of the spectrum, kept as-is here rather
      than silently changed.

    - "lowpass": `weights = exp(-relaxation * eigvals)`, a standard graph
      low-pass / heat-kernel filter -- a smooth, monotonically decreasing
      function of each eigenvalue's own magnitude, not of local spectral
      density. Grounded in the structure-function coupling literature
      (Abdelnour et al. 2018, NeuroImage, "Functional brain connectivity is
      predictable from anatomic network's Laplacian eigen-structure": FC's
      eigenvalues relate to the structural Laplacian's via an approximately
      exponential mapping, i.e. a small number of low-frequency structural
      modes dominate). Unlike "gap", this only depends on where an eigenvalue
      sits on the [0, 2] normalized-Laplacian spectrum, not on how densely
      its neighbors happen to be packed for a particular connectome. Being
      monotonic, it can only ever favor low eigenvalues and suppress
      everything above them -- it cannot reproduce "gap"'s U-shaped
      preference for both the low (integrative) and high (segregative) ends
      of the spectrum.

    - "gap-spectrum": `_spline_gap_spectrum`'s output (see that function's
      docstring), i.e. "gap" done the way Sipes et al. 2024 actually do it --
      a smoothing spline fit to eigenvalue vs. harmonic index, weighted by
      the analytical derivative of that spline, instead of "gap"'s raw
      finite differences. Smooth and U-shaped (integrative and segregative
      both favored, degenerate suppressed) without needing "gap"'s
      `weights[5000:]` patch.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics.

    relaxation: float, optional
        "gap" / "gap-spectrum": relaxes small weights, intended range [0-1].
        The weight will give the probability of a harmonic to be chosen
        during reconstruction. If taking the raw weights, this can be pretty
        tough on middle harmonics. The relaxation allows middle harmonics to
        have more weights. When relaxation = 0, all harmonics have a weight
        of 1. Defaults to 1 (this function's original default) if not given.

        "lowpass": the decay rate beta in exp(-beta*eigvals). 0 gives uniform
        weights (same no-op meaning as "gap"'s relaxation=0); larger values
        increasingly favor low eigenvalues. Needs retuning against "gap"'s
        values -- the two parametrizations aren't numerically comparable.
        Defaults to `DEFAULT_LOWPASS_BETA` if not given -- see that
        constant's comment for where it comes from.

    method: str
        "gap" (default), "lowpass", or "gap-spectrum" -- see above.
    """
    if relaxation is None:
        relaxation = DEFAULT_LOWPASS_BETA if method == "lowpass" else 1

    if method == "lowpass":
        return np.exp(-relaxation * eigvals)

    if method == "gap-spectrum":
        return _spline_gap_spectrum(eigvals) ** relaxation

    if method != "gap":
        raise ValueError(f"compute_harmonic_prior: unknown method {method!r}, expected 'gap', 'lowpass', or 'gap-spectrum'")

    # compute eigenvalue gaps
    arr1 = np.abs(eigvals - np.roll(eigvals, -1))
    arr2 = np.abs(eigvals - np.roll(eigvals, 1))

    weights = (arr1 + arr2)/2

    # correct extremities
    weights[0]  = arr1[0]
    weights[-1] = arr2[-1]

    # convert to prior weights and store in prior mixing matrix
    weights[5000:] = weights[5000]  # high harmonics seems to be problematic ...
    weights = weights**relaxation

    return weights


def compute_depth_prior(gain, depth):
    """Computes weight for each source based on lead field column norms."""
    norms = np.linalg.norm(gain, axis=0)
    depth_prior = np.power(norms, -depth)

    return depth_prior
