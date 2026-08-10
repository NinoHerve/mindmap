"""
Module Name: analysis/functional/inverse/utils.py
Description: Helpers specific to the inverse module.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import warnings
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


# Placeholder beta for compute_harmonic_prior's "logpass" method. Unlike
# DEFAULT_LOWPASS_BETA, this value has NO effect on the reconstructed source
# estimate: wCSET.prepare_kernel renormalizes w_source by
# sqrt(rank/trace(gain_weighted @ gain_weighted.T)) before the SVD, which
# exactly cancels any global multiplicative constant applied to w_source --
# and beta enters "logpass"'s weight formula as exactly such a constant
# (weight_k = sqrt(-log(lambda_tilde_k) / beta), i.e. beta factors out as a
# uniform 1/sqrt(beta) across every k, unlike "lowpass"'s exp(-beta*eigval)
# where beta sits inside a per-k exponent and genuinely reshapes relative
# weighting). Kept in the signature purely for parity with "lowpass" /
# paper-fidelity (Wang et al. 2026, Nat Biomed Eng, Extended Data Fig. 1g's
# "near beta=0.1"), not because it does anything here.
DEFAULT_LOGPASS_BETA = 1.0

# Clip floor for the normalized eigenvalue ratio in "logpass", guarding the
# lambda->0 singularity in -log(lambda) at the DC/constant mode
# (eigvals[0] == 0 by construction in both compute_cst and
# compute_beltrami_harmonics -- two such modes for the mesh basis, one per
# disconnected hemisphere). Pure floating-point necessity, not a design
# choice -- same floor already used in wCSET.estimate_lambda2's
# `np.maximum(signal_power, 1e-8)`. Internal only, not exposed as a
# parameter: unlike `cutoff` this isn't "the paper's S", it's just what
# standing in for an undefined log(0) requires.
_LOGPASS_EPS = 1e-8


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


def _apply_cutoff(weights: np.ndarray, cutoff: int) -> np.ndarray:
    """
    Zero out weights at and beyond `cutoff`. Shared post-processing for the
    "gap"/"lowpass"/"gap-spectrum" methods of compute_harmonic_prior, where
    cutoff is a pure optional truncation (unlike "logpass", where cutoff
    also sets the normalization reference and is handled inline there
    instead of through this helper).
    """
    if cutoff is None:
        return weights
    if not (0 < cutoff <= len(weights)):
        raise ValueError(
            f"compute_harmonic_prior: cutoff={cutoff} out of range for "
            f"eigvals of length {len(weights)}"
        )
    weights = weights.copy()
    weights[cutoff:] = 0
    return weights


# harmonic prior
def compute_harmonic_prior(eigvals : np.ndarray, relaxation : float = None, method : str = "gap", cutoff : int = None) -> np.ndarray:
    """
    Compute the harmonic prior according to eigenvalues.

    Four weighting schemes, selected via `method`:

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

    - "logpass": `weights = sqrt(-log(eigvals_tilde) / relaxation)` over the
      first `cutoff` eigenvalues (all later ones get weight exactly 0), where
      `eigvals_tilde = clip(eigvals[:cutoff] / eigvals[cutoff-1], eps, 1)` --
      the logarithmic spectral prior from Wang et al. 2026 (Nat Biomed Eng,
      "A geometry aware framework enhances noninvasive mapping of whole human
      brain dynamics", the GBF paper), `Sigma^-1 = diag(-beta/log(lambda))`
      (their Eq. 5) applied to their first `S=300` Laplace-Beltrami modes
      (their Eq. 2). This function returns weights in the same
      eigenvector-rescaling convention as "lowpass"/"gap" (i.e.
      `weight_k = sqrt(Sigma_kk)`, the standard weighted-minimum-norm
      equivalent of a Gaussian prior with covariance `Sigma`), not `Sigma^-1`
      itself.

      Qualitatively different shape from "lowpass": nearly flat across most
      of the retained spectrum, then a steep wall right at the cutoff
      boundary (`eigvals_tilde -> 1`), rather than "lowpass"'s uniform
      geometric decay across the whole range.

      IMPORTANT: unlike every other method here, `relaxation` (the paper's
      beta) has NO effect on the weights' shape -- it only rescales every
      entry by the same constant (`weight_k(beta) = weight_k(1)/sqrt(beta)`),
      and that constant is exactly cancelled by `wCSET.prepare_kernel`'s
      trace-renormalization before it ever reaches the solve. See
      `DEFAULT_LOGPASS_BETA`'s comment. What actually shapes this prior is
      `cutoff` (which modes are kept, and the normalization reference) --
      required for this method, no default (raises `ValueError` if omitted;
      see `cutoff`'s own entry below). A `RuntimeWarning` is raised if
      `relaxation` is given a value other than `DEFAULT_LOGPASS_BETA`, since
      that's otherwise a silent no-op easy to mistake for a real knob.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics, ascending order.

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

        "logpass": the paper's beta. Has NO effect on the resulting weights'
        shape or on any `wCSET` reconstruction using them -- see "logpass"
        above and `DEFAULT_LOGPASS_BETA`'s comment. Kept only for signature/
        paper-fidelity parity with "lowpass". Defaults to
        `DEFAULT_LOGPASS_BETA` if not given.

    method: str
        "gap" (default), "lowpass", "gap-spectrum", or "logpass" -- see above.

    cutoff: int, optional
        Number of leading (lowest-eigenvalue) modes to retain; all modes at
        or beyond `cutoff` get weight exactly 0.

        "logpass": also serves as the normalization reference
        (`eigvals[cutoff-1]`) -- see that method's description above.
        REQUIRED for this method -- raises `ValueError` if not given (e.g.
        `cutoff=300`, the paper's `S`). No default: unlike the other three
        methods, there's no sensible "no truncation" fallback here, since
        `cutoff` also fixes what "1.0" means on the normalized eigenvalue
        axis.

        "gap" / "lowpass" / "gap-spectrum": a pure optional truncation on
        top of that method's existing weight shape -- no normalization
        interaction, since these methods don't have one. Defaults to None
        (no truncation, full spectrum retained) if not given.
    """
    if relaxation is None:
        relaxation = DEFAULT_LOWPASS_BETA if method == "lowpass" else (
            DEFAULT_LOGPASS_BETA if method == "logpass" else 1
        )

    if method == "lowpass":
        return _apply_cutoff(np.exp(-relaxation * eigvals), cutoff)

    if method == "gap-spectrum":
        return _apply_cutoff(_spline_gap_spectrum(eigvals) ** relaxation, cutoff)

    if method == "logpass":
        if relaxation != DEFAULT_LOGPASS_BETA:
            warnings.warn(
                f"compute_harmonic_prior: relaxation={relaxation} given for "
                "method='logpass', but beta has no effect on the resulting "
                "weights or on any wCSET reconstruction using them -- it's a "
                "uniform rescale exactly cancelled by wCSET's own "
                "trace-renormalization before the ridge solve (see "
                "DEFAULT_LOGPASS_BETA's comment). If you intended to change "
                "the log-form prior's strength or shape, use `cutoff` instead.",
                RuntimeWarning,
            )

        if cutoff is None:
            raise ValueError(
                "compute_harmonic_prior: 'cutoff' must be given for "
                "method='logpass' -- e.g. cutoff=300, the paper's S (Eq. 2). "
                "Unlike 'gap'/'lowpass'/'gap-spectrum', this method has no "
                "sensible no-truncation default: cutoff also sets the "
                "normalization reference, and diag(-beta/log(lambda)) isn't "
                "defined without one."
            )
        if not (0 < cutoff <= len(eigvals)):
            raise ValueError(
                f"compute_harmonic_prior: cutoff={cutoff} out of range for "
                f"eigvals of length {len(eigvals)}"
            )

        lam_cutoff = eigvals[cutoff - 1]
        if lam_cutoff <= 0:
            raise ValueError(
                "compute_harmonic_prior: eigvals[cutoff-1] must be > 0 to "
                "normalize the 'logpass' spectrum -- cutoff is too small"
            )

        # normalize the retained eigenvalues into (0, 1] against the cutoff
        # boundary itself (not eigvals.max()) -- Eq. 2 restricts k=1..S from
        # the outset, so "lambda_max" in the paper's own formula can only
        # mean the max within that retained set, not some larger untruncated
        # spectrum. Clip away from exactly 0 (the DC mode) rather than
        # let log(0) produce -inf; never evaluate log() on anything above 1
        # (indices beyond cutoff are just zeroed, not computed through the
        # formula, since eigvals[cutoff:] > lam_cutoff by construction and
        # would otherwise make -log(...) negative -> sqrt(...) -> nan).
        normalized = np.clip(eigvals[:cutoff] / lam_cutoff, _LOGPASS_EPS, 1.0)

        weights = np.zeros_like(eigvals, dtype=float)
        weights[:cutoff] = np.sqrt(-np.log(normalized) / relaxation)
        return weights

    if method != "gap":
        raise ValueError(f"compute_harmonic_prior: unknown method {method!r}, expected 'gap', 'lowpass', 'gap-spectrum', or 'logpass'")

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

    return _apply_cutoff(weights, cutoff)


def compute_depth_prior(gain, depth):
    """Computes weight for each source based on lead field column norms."""
    norms = np.linalg.norm(gain, axis=0)
    depth_prior = np.power(norms, -depth)

    return depth_prior
