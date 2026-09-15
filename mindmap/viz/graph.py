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
import pyvista as pv

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


def plot_eigvec_multi(source_space, eigs, eigenvalues=None):
    """
    Plot several conditions' eigenvectors side by side (e.g. a true
    connectome and its surrogates), one panel per condition, scrolled
    through harmonics together with a single shared slider/keyboard.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space, shared across every condition/panel.

    eigs: dict[str, np.ndarray]
        Mapping from condition label (used as the panel title) to its
        eigenvectors, shape (n_dipoles, n_modes). Panels are laid out
        left-to-right in dict order. Conditions may have different numbers
        of modes - scrolling is capped at the smallest one.

    eigenvalues: dict[str, np.ndarray], optional
        Mapping from condition label to its eigenvalues, same keys as
        `eigs`. A missing/None entry falls back to a plain harmonic index
        for that panel's label.
    """
    mesh_params = {}
    for name, eigvec in eigs.items():
        clim = find_clim(eigvec)
        mesh_params[name] = {
            "smooth_shading" : True,
            "clim" : clim,
            "cmap" : default_cmap(clim),
        }

    return MultiConditionMesh(
        source_space = source_space,
        eigs         = eigs,
        eigenvalues  = eigenvalues,
        mesh_params  = mesh_params,
    )


class MultiConditionMesh(BaseBrainPlotter):
    """
    Interactive side-by-side visualization of several conditions' harmonics,
    each in its own panel (linked camera), scrolled through together with
    one shared "Harmonic" slider/keyboard - for comparing e.g. a true
    connectome's harmonics against its surrogates'.
    """

    def __init__(
        self,
        source_space : mne.SourceSpaces,
        eigs         : dict,
        eigenvalues  : Optional[dict] = None,
        mesh_params  : Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        source_space: mne.SourceSpaces
            Source space, shared across every condition/panel.

        eigs: dict[str, np.ndarray of shape (n_dipoles, n_modes)]
            Eigenvectors per condition; panels laid out in this order.

        eigenvalues: dict[str, np.ndarray of shape (n_modes,)], optional
            Eigenvalues per condition, for panel labels.

        mesh_params: dict[str, dict]
            Per-condition parameters passed to plotter.add_mesh.
        """
        self.names  = list(eigs.keys())
        self.data   = eigs
        self.eigval = eigenvalues or {}

        mesh_params = mesh_params or {}
        n_states    = min(vec.shape[1] for vec in eigs.values())
        plotter     = pv.Plotter(shape=(1, len(self.names)))

        super().__init__(plotter=plotter, n_states=n_states)

        self.hemi_l, self.hemi_r = {}, {}
        self.map_l, self.map_r   = {}, {}
        self.nuse_l = {}

        for i, name in enumerate(self.names):
            hemi_l, hemi_r, map_l, map_r, nuse_l = build_hemi_pair(source_space)
            self.hemi_l[name], self.hemi_r[name] = hemi_l, hemi_r
            self.map_l[name], self.map_r[name] = map_l, map_r
            self.nuse_l[name] = nuse_l

            self.plotter.subplot(0, i)
            self.plotter.add_mesh(hemi_l, **mesh_params.get(name, {}))
            self.plotter.add_mesh(hemi_r, **mesh_params.get(name, {}))
            self.plotter.add_text(name, position="upper_edge", font_size=10, color="black")

        self.plotter.link_views()

        # events (slider lives in the first panel, keyboard is window-wide)
        self.setup_keys(fast_step=50, super_step=500)
        self.plotter.subplot(0, 0)
        self.add_slider(label="Harmonic")

        # render first image
        self.update_display(self.current_idx)


    def _label_text(self, name: str, idx: int) -> str:
        eigval = self.eigval.get(name)
        if eigval is None:
            return f"Harmonic {idx + 1}"
        return f"Eigenvalue: {eigval[idx]:.4g}"


    # -------------------
    # Inherited methods
    # -------------------

    def update_display(self, idx):
        """
        Update every panel's brain image to harmonic `idx`.
        """
        for i, name in enumerate(self.names):
            image = self.data[name][:, int(idx)]
            nuse_l = self.nuse_l[name]

            sig_l = image[:nuse_l].squeeze()
            sig_r = image[nuse_l:].squeeze()

            # append 0 for vertices mapped to -1
            sig_l = np.concatenate([sig_l, [0.0]])
            sig_r = np.concatenate([sig_r, [0.0]])

            self.hemi_l[name]["signal"][:] = sig_l[self.map_l[name]]
            self.hemi_r[name]["signal"][:] = sig_r[self.map_r[name]]

            self.plotter.subplot(0, i)
            self.plotter.add_text(
                self._label_text(name, idx),
                name = "info_text",
                position = "lower_edge",
                font_size = 10,
                color = "black",
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


def plot_adj_matrix(
    connectome,
    title: str = "Adjacency matrix", 
    clabel: str = "weights",
    xylabel: str = "Dipole Index",
):
    """Plot adjacency matrix."""
    # Create a masked array to hide zero-valued elements
    masked_matrix = np.ma.masked_equal(connectome, 0)

    plt.figure(figsize=(10, 8))

    # Using 'nearest' interpolation and 'rasterized=True' handles big matrices 
    im = plt.imshow(
        masked_matrix,
        cmap = "viridis_r",
        interpolation = "nearest",
        aspect = "equal",
        rasterized = True,
    )

    # Add a colorbar and label it appropratiely since the weights are distances
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(clabel, rotation=270, labelpad=15)

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xylabel, fontsize=12)
    plt.ylabel(xylabel, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_con_summary(adj_matrix, title="Connectome summary", xlabel="weights"):
    """
    Plot connectome summary.

    Parameters
    ----------
    adj_matrix: np.ndarray
        Adjacency matrix.
    """
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    # plots
    plot_degree_dist(adj_matrix, axs[0])
    plot_weight_dist(adj_matrix, axs[1])
    axs[1].set_xlabel(xlabel)

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
