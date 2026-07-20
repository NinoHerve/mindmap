"""
Module Name: viz/power.py
Description: Plots for pilot-sizing power/precision sweeps
(`mindmap.core.power`, `mindmap.analysis.power`).

Author: Nino Herve
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_power_precision_curves(
    power_results: dict,
    precision_results: dict,
    k_grid: list,
    r_grid: list,
    outcomes: list,
):
    """
    One row per outcome: condition main-effect power, group:condition
    interaction power (both vs. total simulation count `k*r`, one line per
    `r`), and per-group-mean precision (SE) vs. `r`.

    Parameters
    ----------
    power_results: dict
        `{outcome: {"condition": np.ndarray of shape (len(k_grid),
        len(r_grid)), "interaction": same shape}}`, e.g. built by looping
        `mindmap.core.power.simulate_splitplot_power` over `k_grid` x
        `r_grid`.

    precision_results: dict
        `{outcome: np.ndarray of shape (len(r_grid),)}`, e.g. from
        `mindmap.core.power.precision_of_mean` over `r_grid`.

    k_grid, r_grid: list
        Candidate group counts / repetition counts swept.

    outcomes: list[str]
        Outcome names to plot, in row order.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(len(outcomes), 3, figsize=(15, 4 * len(outcomes)), squeeze=False)
    for row, outcome in enumerate(outcomes):
        ax = axes[row, 0]
        for j, r in enumerate(r_grid):
            ax.plot(np.array(k_grid) * r, power_results[outcome]["condition"][:, j], marker="o", label=f"r={r}")
        ax.axhline(0.8, ls="--", c="gray")
        ax.set_xlabel("total sims (k x r)"); ax.set_ylabel("power")
        ax.set_title(f"{outcome}: condition main effect")
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        for j, r in enumerate(r_grid):
            ax.plot(np.array(k_grid) * r, power_results[outcome]["interaction"][:, j], marker="o", label=f"r={r}")
        ax.axhline(0.8, ls="--", c="gray")
        ax.set_xlabel("total sims (k x r)"); ax.set_ylabel("power")
        ax.set_title(f"{outcome}: group x condition interaction")
        ax.legend(fontsize=7)

        ax = axes[row, 2]
        ax.plot(r_grid, precision_results[outcome], marker="o")
        ax.set_xlabel("r (repetitions per group)"); ax.set_ylabel("SE of per-group mean")
        ax.set_title(f"{outcome}: per-group mean precision (indep. of k)")

    fig.tight_layout()
    return fig
