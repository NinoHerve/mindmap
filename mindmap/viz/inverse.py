
"""
Module Name: viz/inverse.py
Description: Functions to plot inverse properties.

===============================================================================
INVERSE VISUALIZATION (inverse.py)
===============================================================================
Targets: mm.analysis.inverse
Backends: Matplotlib (2D), pyvista (3D point clouds)
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

import numpy as np

import matplotlib.pyplot as plt


def plot_predicted_data_covariance(inverse, evoked):
    """
    Plot expected variance.

    A whitened system with trace adjustment should have a variance
    Var(y) = G_scaled @ R_scaled @ G.T_scaled + I, where the 'signal' part of
    the model is forced to have a total variance equal to the number of degrees of
    freedom (rank).
    """
    proj, _, whitener = inverse._setup_preprocessor(evoked.info)
    kern = inverse.prepare_kernel(proj, whitener)

    # GRGT = (G*W_s) @ (G*W_s).T
    # This is equivalent to U @ S^2 @ U.T if using SVD on weighted gain
    data_cov = (kern["u"]* (kern["s"]**2)) @ kern["u"].T

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Calculate scale for consistent visual comparison
    # For whitened data, we expect values near 1 on diag and 0 off-diag
    v_limit = np.percentile(np.abs(data_cov), 99)
    im = ax.imshow(data_cov, cmap="RdBu_r", vmin=-v_limit, vmax=v_limit)

    ax.set_title("Predicted Data Covariance ($G R G^T$)")
    plt.colorbar(im, ax=ax)

    # Verification metric
    avg_diag = np.mean(np.diag(data_cov))
    stats_text = f"Mean of Diagonal: {avg_diag:.4f}\n(Target: 1.0)"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.close("all")

    return fig