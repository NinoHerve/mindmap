"""
Module Name: viz/graph.py
Description: Functions to plot graph properties.

===============================================================================
GRAPH-SPACE VISUALIZATION (graph.py)
===============================================================================
Targets: np.ndarray, mne.SourceSpaces
Backends: pyvista, matplotlib
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import mne

from .utils import find_clim, default_cmap, BaseBrainPlotter, build_hemi_pair

def plot_eigvec(source_space, eigenvectors, eigenvalues=None):
    """
    Plot eigenvectors.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    eigenvectors: np.ndarray
        Eigenvectors.

    eigenvalues: np.ndarray
        Eigenvalues.
    """

    # -------------------
    # Prepare parameters
    # -------------------

    clim = find_clim(eigenvectors)
    cmap = default_cmap(clim)

    mesh_params = {
        "smooth_shading" : True,
        "clim" : clim,
        "cmap" : cmap,
    }

    return MultiMesh(
        source_space = source_space,
        data         = eigenvectors,
        eigenvalues  = eigenvalues,
        mesh_params  = mesh_params,
    )


class MultiMesh(BaseBrainPlotter):
    """
    Interactive visualization of brain source estimates over time using PyVista
    A slider allows scrolling through time points.
    """

    def __init__(
        self,
        source_space : mne.SourceSpaces,
        data         : np.ndarray,
        eigenvalues  : Optional[np.ndarray] = None,
        mesh_params  : Optional[dict] = None,
        title        : Optional[str] = "",
    ):
        """
        Parameters
        ----------
        source_spaces: list[mne.SourceSpaces]
            Source spaces.
        
        data: np.ndarray of shape (n_dipoles, n_samples)
            Brain images.

        titles: list[str]
            Title for plot.

        mesh_params: dict
            Parameters passed to plotter.add_mesh for brain meshes
        """
        super().__init__(n_states=data.shape[1])

        self.src    = source_space
        self.data   = data
        self.eigval = eigenvalues
        self.title  = title

        self.n_plots = data.shape[1]

        self.mesh_params    = mesh_params or {}

        # prepare meshes and mapping
        self.hemi_l, self.hemi_r, self.map_l, self.map_r, self.nuse_l = build_hemi_pair(self.src)

        self.plotter.add_mesh(self.hemi_l, **self.mesh_params)
        self.plotter.add_mesh(self.hemi_r, **self.mesh_params)
        self.plotter.add_title(self.title)

        # events
        self.setup_keys(fast_step=50, super_step=500)
        self.add_slider(label="Harmonic")

        # render first image
        self.update_display(self.current_idx)


    def _label_text(self, idx: int) -> str:
        if self.eigval is None:
            return f"Image {idx + 1} / {self.n_plots}"
        else:
            val = self.eigval[idx]
            return f"Eigenvalue: {val:.4g}"

    # -------------------
    # Inherited methods
    # -------------------

    def update_display(self, idx):
        """
        Update brain image.
        """
        image = self.data[:,int(idx)]
        sig_l = image[:self.nuse_l].squeeze()
        sig_r = image[self.nuse_l:].squeeze()

        # append 0 for vertices mapped to -1
        sig_l = np.concatenate([sig_l, [0.0]])
        sig_r = np.concatenate([sig_r, [0.0]])

        self.hemi_l["signal"][:] = sig_l[self.map_l]
        self.hemi_r["signal"][:] = sig_r[self.map_r]

        self.plotter.add_text(
            self._label_text(idx),
            name = "info_text",
            position = "upper_left",
            font_size = 12,
            color = "black"
        )

        self.plotter.render()


def plot_eigval(eigval, subject="unknown"):
    """
    Plot eigenvalues.
    """
    ax = sns.lineplot(eigval)
    ax.set_title(f"Eigenvalues sub-{subject}")
    ax.set_xlabel("Harmonic index")
    ax.set_ylabel("Eigenvalue")

    plt.show()


def plot_weight_dist(adj_matrix, ax=None, **kwargs):
    """
    Plot weight distribution.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """

    # get upper triangle
    n       = adj_matrix.shape[0]
    weights = adj_matrix[np.triu_indices(n, k=1)]

    # filter out zero-weight edges
    weights = weights[weights > 0]

    # If no axis is provided, create a new figure
    if ax is None:
        _, ax = plt.subplots()

    sns.histplot(weights, ax=ax, **kwargs)

    # set default labels
    ax.set_title("Edge Weight Distribution")
    ax.set_xlabel("Weigth")
    ax.set_ylabel("Frequency")

    return ax


def plot_degree_dist(adj_matrix, ax=None, **kwargs):
    """
    Plot degree distribution.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """

    # get degrees
    degrees = np.sum(adj_matrix>0, axis=1)

    # If no axis is provided, create a new figure
    if ax is None:
        _, ax = plt.subplots()

    sns.histplot(degrees, ax=ax, **kwargs)

    # set default labels
    ax.set_title("Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")


def plot_con_summary(adj_matrix, subject="unknown"):
    """
    Plot connectome summary.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f"Connectome summary: sub-{subject}")

    # plots
    plot_degree_dist(adj_matrix, axs[0])
    plot_weight_dist(adj_matrix, axs[1])
    axs[1].set_xlabel("Geodesic Distance (mm)")

    plt.show()


def plot_spectrum(spectrum, times=None):
    """
    Plot the connectome spectrum.

    Parameters
    ----------
        spectrum: nd.array (modes, time)
            Connectome spectrum.

        times: np.ndarray (time,)
            Times.

    Returns
    -------
        fig: plotly.graph_objs
    """
    def new_trace(values):
        trace = go.Bar(
            y       = values,
            marker  = dict(
                color = "darkblue",
                line  = dict(color="darkblue"),
            ),
            visible = False,
        )

        return trace

    border = np.max(np.abs(spectrum))
    border = [-border, border]

    if len(spectrum.shape)>1:
        if times is None:
            print("\033[93mNo times provided\033[0m")
            times = np.arange(spectrum.shape[1])

        fig   = go.Figure()
        steps = []

        for i, t in enumerate(times):
            fig.add_trace(new_trace(spectrum[:,i]))

            step = dict(
                method = "update",
                args   = [
                    {"visible": [False] * len(times)},
                    {"title": f"Connectum Spectrum: t = {t:.3f}[s]"},
                ],
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)

        fig.data[int(len(times)/2)].visible = True

        sliders = [dict(
            active       = int(len(times)/2),
            currentvalue = {"visible":False},
            pad          = {"t":50},
            steps        = steps,
        )]

        fig.update_layout(sliders = sliders)

    else:
        trace = new_trace(spectrum)

        layout = go.Layout(dict(
            title="Connectome Spectrum",
            showlegend=False,
        ))

        fig = go.Figure(trace, layout)
        fig.data[0].visible = True


    fig.update_layout(
        plot_bgcolor = "white",
        xaxis        = dict(
            title     = "frequency indices",
            linecolor = "black",
        ),
        yaxis        = dict(
            title     = "amplitude",
            linecolor = "black",
            gridcolor = "black",
            range     = border,
        ),
    )

    return fig
