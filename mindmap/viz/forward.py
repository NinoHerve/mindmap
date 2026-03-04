"""
Module Name: viz/forward.py
Description: Functions to plot forward properties.

===============================================================================
FORWARD VISUALIZATION (forward.py)
===============================================================================
Targets: mne.Forward
Backends: Matplotlib (2D grid), pyvista (3D point clouds)
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

import numpy as np

import matplotlib.pyplot as plt
import pyvista as pv

import mne

from mindmap.analysis.functional.eeg.io import get_ch_positions
from .surface import plot_brain
from .utils import find_clim



def plot_leadfield(eeg, forward):
    """Plot to interactively look at leadfields"""
    # get electrode positions
    points = get_ch_positions(eeg)
    cap    = pv.PolyData(points)
    cap["highlight"] = np.zeros(len(points))

    # get leadfield matrix
    leadfield = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, use_cps=True,
    )
    leadfield = leadfield["sol"]["data"]

    mesh_params = {
        "clim"  : find_clim(leadfield),
        "cmap"  : "bwr",
        "color" : None,
    }

    # plot
    plotter = plot_brain(forward["src"], np.zeros(leadfield.shape[1]), mesh_params=mesh_params)
    plotter.add_mesh(cap, point_size=10, render_points_as_spheres=True, name="cap")
    plotter.remove_scalar_bar(title="highlight")

    # callback
    def pick_electrode(picked_point):
        if picked_point is not None:
            # get index of the closest point
            closest_point_id = cap.find_closest_point(picked_point)

            # change highlighted electrode
            cap["highlight"] = np.zeros(len(points))
            cap["highlight"][closest_point_id] = 1

            # change leadfield vector
            tmp = plot_brain(forward["src"], leadfield[closest_point_id], mesh_params=mesh_params)
            plotter.meshes[0]["signal"] = tmp.meshes[0]["signal"]
            plotter.meshes[1]["signal"] = tmp.meshes[1]["signal"]

            # update the plotter
            plotter.render()

    plotter.enable_point_picking(callback=pick_electrode)

    return plotter


def plot_fwd_singular_values(fwd_or_gain, title="Forward Solution Singular Values", fixed=True):
    """
    Plot singular values.
    
    Parameters
    ----------
    fwd_or_gain : mne.Forward | np.ndarray
        The forward solution or the gain matrix.
    title : str
        The title of the plot.
    fixed : bool
        If True and input is a Forward object, force fixed orientations.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # 1. Extract gain matrix
    if isinstance(fwd_or_gain, mne.Forward):
        if fixed:
            fwd_or_gain = mne.convert_forward_solution(fwd_or_gain, force_fixed=True, verbose=0)
        gain = fwd_or_gain["sol"]["data"]
    else:
        gain = fwd_or_gain

    # 2. Compute SVD (singular values only)
    s = np.linalg.svd(gain, compute_uv=False)
    
    # 3. Compute Condition Number
    # Filter out values below numerical epsilon to get the true rank-based condition
    rank_thresh = s.max() * 1e-15
    s_significant = s[s > rank_thresh]
    cond_num = s_significant[0] / s_significant[-1]

    # 4. Build Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(s, linewidth=3, color='#2c7fb8', label='Singular Values')
    
    # Visual markers for max/min significant values
    ax.axhline(s_significant[0], color='green', linestyle=':', alpha=0.5, label='Max $\sigma$')
    ax.axhline(s_significant[-1], color='red', linestyle=':', alpha=0.5, label='Min $\sigma$')

    ax.set_title(f"{title}\nCondition Number ($\kappa$): {cond_num:.2e}")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Singular Value (Log Scale)")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.close(fig) # Prevents double-display in Jupyter when returned

    return fig


def plot_fwd_sensitivity(fwd, fixed=True, title="Forward Sensitivity"):
    """
    Calculate the norm of each column (source)
    This represents the total 'power' a source sends to the sensors.
    """
    # Extract gain matrix
    if fixed:
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    gain = fwd["sol"]["data"]

    sensitivity = np.linalg.norm(gain, axis=0)

    return plot_brain(fwd["src"], sensitivity, title=title)


def plot_fwd_correlation(fwd, fixed=True):
    """
    Plot correlation between columns (sources)
    Warning: for very large source spaces, this matrix is massive.
    """

    # Extract gain matrix
    if fixed:
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    gain = fwd["sol"]["data"]

    if gain.shape[1] > 2000:
        indices = np.linspace(0, gain.shape[1] - 1, 1000, dtype=int)
        gain_sub = gain[:, indices]
    else:
        gain_sub = gain

    corr = np.corrcoef(gain_sub.T)

    plt.figure(figsize=(6,5))
    plt.imshow(np.abs(corr), cmap="viridis", origin="lower")
    plt.colorbar(label="Absolute Correlation")
    plt.title("Source Correlation Matrix (Sub-sampled)")
    plt.xlabel("Source Index")
    plt.ylabel("Source Index")
    plt.show()


def plot_fwd_avg_ref(fwd, title="Forward Average Referencing", fixed=True):
    """
    Plots the spatial mean over columns (sources)
    For average-referenced data, this should be a flat line at 0.
    """
    # Extract gain
    if fixed:
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    gain = fwd["sol"]["data"]

    spatial_mean = gain.mean(axis=0)
    max_dev = np.max(np.abs(spatial_mean))

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spatial_mean, color="crimson", label="Residual Spatial Mean")
    ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax.set_title(f"{title}\nMax Deviation: {max_dev:.2e} V/A")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Average Amplitude (V/A)")

    limit = max(max_dev * 2, 1e-15)
    ax.set_ylim(-limit, limit)

    plt.close("all")  # Prevent Jupyter from auto-rendering
    return fig
