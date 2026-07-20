"""
Module Name: core/power.py
Description: Monte Carlo power/precision simulation for split-plot
repeated-measures pilot-sizing studies (a "group" random factor -- e.g.
brain region, playing the role of "subject" -- crossed with a fixed
condition factor and r sub-replicate repetitions per cell). Given
variance components (e.g. from `mindmap.analysis.power.fit_variance_components`),
estimates statistical power for a grid of candidate (k groups, r
repetitions) designs.

Pure numpy/scipy -- independent of pandas/statsmodels/mne -- so it can be
called from a quick sizing script without pulling in the heavier fitting
machinery in `mindmap.analysis.power`.

Author: Nino Herve
"""

import numpy as np
from scipy import stats


def simulate_splitplot_power(
    condition_means,
    sigma2_group: float,
    sigma2_interaction: float,
    sigma2_rep: float,
    sigma2_resid: float,
    k: int,
    r: int,
    m: int,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple:
    """
    Classical two-way (group x condition) split-plot ANOVA Monte Carlo, with
    `k` groups random (e.g. brain region -- the "subject"), an `m`-level
    condition fixed (e.g. method), group:condition a random interaction, and
    `r` sub-replicates (repetitions) per (group,condition) cell -- repetition
    is itself a random group:repetition effect, shared across all `m`
    conditions within a (group,repetition) cell (e.g. "this trial's noise
    draw was unusually easy, for every method reconstructing it"), not just
    extra per-observation noise:

      SS_group       = r*m * sum_i (group_mean_i - grand_mean)^2           df = k-1
      SS_condition   = k*r * sum_l (condition_mean_l - grand_mean)^2       df = m-1
      SS_interaction = r   * sum_{i,l} (cell_mean_il - group_mean_i
                                        - condition_mean_l + grand_mean)^2  df = (k-1)(m-1)
      SS_rep         = m   * sum_{i,j} (rep_mean_ij - group_mean_i)^2      df = k*(r-1)
      SS_residual    =       sum_{i,j,l} (y_ijl - cell_mean_il
                                        - rep_mean_ij + group_mean_i)^2    df = k*(r-1)*(m-1)

    E[MS_residual] = sigma2_resid
    E[MS_interaction] = sigma2_resid + r*sigma2_interaction (+ noncentrality if testing condition)
    E[MS_condition] = sigma2_resid + r*sigma2_interaction + k*r*(true condition-effect variance)

    So F(condition) = MS_condition/MS_interaction (the group:condition
    interaction is the correct error term for the fixed condition effect --
    standard repeated-measures-ANOVA logic with group as "subject"), and
    F(interaction) = MS_interaction/MS_residual. Simulating cell means
    analytically (instead of raw rows + a full mixed-model refit per
    iteration) is exact for this balanced design and orders of magnitude
    faster, which is what makes a dense (k,r) grid tractable.

    `SS_residual` must explicitly remove `rep_mean`'s own structure (not just
    `cell_mean`'s), or `MS_residual` is contaminated by `sigma2_rep` bleeding
    through -- confirmed as a real bug (not just a theoretical concern) via a
    real pilot fit where `sigma2_rep` was ~6x `sigma2_resid`: the naive
    `sum((y - cell_mean)**2) / (k*m*(r-1))` residual estimate came out ~16x
    too large in a 3000-iteration Monte Carlo check against known ground
    truth, silently killing interaction-test power (it was landing at exactly
    0.000, reproducibly across seeds -- a giveaway, since a true-null F-test
    should still reject at roughly the alpha rate by chance, not exactly
    0%). The version below (subtracting `rep_mean` too, with the correspondingly
    smaller `k*(r-1)*(m-1)` df) was verified unbiased in the same check.

    Parameters
    ----------
    condition_means: array-like of shape (m,)
        True condition-effect contrast (mean-zeroed), e.g.
        `fit_variance_components(...)["condition_means"]`.

    sigma2_group, sigma2_interaction, sigma2_rep, sigma2_resid: float
        Variance components, e.g. from `fit_variance_components(...)`'s
        `sigma2_{group_col}`, `sigma2_{interaction vc name}`, `sigma2_rep`,
        `sigma2_resid`.

    k: int
        Number of groups (e.g. brain regions).

    r: int
        Number of repetitions per (group, condition) cell.

    m: int
        Number of condition levels. Must match `len(condition_means)`.

    n_sim: int
        Number of Monte Carlo iterations.

    alpha: float
        Significance threshold.

    seed: int
        Random seed.

    Returns
    -------
    power_condition: float
        Fraction of iterations where the condition main-effect F-test
        rejects at `alpha`.

    power_interaction: float
        Fraction of iterations where the group:condition interaction F-test
        rejects at `alpha`. `nan` if `r == 1` (no residual df to test it).
    """
    condition_means = np.asarray(condition_means)
    if condition_means.shape[0] != m:
        raise ValueError(
            f"len(condition_means)={condition_means.shape[0]} does not match m={m}."
        )

    rng = np.random.default_rng(seed)

    df1_cond, df2_cond = m - 1, (k - 1) * (m - 1)
    df1_inter, df2_inter = (k - 1) * (m - 1), k * (r - 1) * (m - 1)

    pow_cond = 0
    pow_interaction = 0
    for _ in range(n_sim):
        group_re = rng.normal(0, np.sqrt(sigma2_group), size=k)
        interaction_re = rng.normal(0, np.sqrt(sigma2_interaction), size=(k, m))
        rep_re = rng.normal(0, np.sqrt(sigma2_rep), size=(k, r))
        resid = rng.normal(0, np.sqrt(sigma2_resid), size=(k, r, m))

        y = (condition_means[None, None, :] + group_re[:, None, None]
             + interaction_re[:, None, :] + rep_re[:, :, None] + resid)

        cell_mean = y.mean(axis=1)               # (k, m): group x condition cell means (avg over reps)
        rep_mean = y.mean(axis=2)                # (k, r): group x repetition means (avg over conditions)
        grand_mean = cell_mean.mean()
        condition_mean = cell_mean.mean(axis=0)   # (m,)
        group_mean = cell_mean.mean(axis=1)       # (k,)

        ss_cond = k * r * np.sum((condition_mean - grand_mean) ** 2)
        ss_inter = r * np.sum((cell_mean - condition_mean[None, :] - group_mean[:, None] + grand_mean) ** 2)
        # subtract rep_mean's own structure too, not just cell_mean's -- otherwise sigma2_rep
        # bleeds into what should be a pure residual estimate (see docstring)
        ss_resid = np.sum((y - cell_mean[:, None, :] - rep_mean[:, :, None] + group_mean[:, None, None]) ** 2)

        ms_cond = ss_cond / df1_cond
        ms_inter = ss_inter / df1_inter
        ms_resid = ss_resid / df2_inter if df2_inter > 0 else np.nan

        f_cond = ms_cond / ms_inter if ms_inter > 0 else np.inf
        f_inter = ms_inter / ms_resid if (ms_resid and ms_resid > 0) else np.inf

        crit_cond = stats.f.ppf(1 - alpha, df1_cond, df2_cond)
        crit_inter = stats.f.ppf(1 - alpha, df1_inter, df2_inter) if df2_inter > 0 else np.nan

        pow_cond += f_cond > crit_cond
        pow_interaction += (f_inter > crit_inter) if df2_inter > 0 else np.nan

    power_condition = pow_cond / n_sim
    power_interaction = pow_interaction / n_sim if df2_inter > 0 else np.nan
    return power_condition, power_interaction


def precision_of_mean(sigma2_rep: float, sigma2_resid: float, r: int, m: int) -> float:
    """
    SE of a single group's mean (averaged over `r` repetitions and all `m`
    conditions), independent of `k`.
    """
    return np.sqrt(sigma2_rep / r + sigma2_resid / (r * m))
