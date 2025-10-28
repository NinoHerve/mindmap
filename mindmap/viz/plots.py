"""
Module Name: plots.py
Description: Functions to plot data.

This module provides functions to plot data.

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""
import numpy as np
from scipy.fft import ifftn
import nibabel as nib
import mne

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import plotly.graph_objs as go
import plotly.express as px
import pyvista as pv

from mindmap.head import mne_to_polydata, nib_to_polydata, get_dipole_positions
from mindmap.sensor import get_ch_positions


def show_slices(
        volume        : np.ndarray | nib.Nifti1Image,
        slice_indices : list[int] = None,
        axis          : int = 2,
        cmap          : str = "gray",
) -> None:
    """
    Display slices from a volume.

    Parameters
    ----------
    volume: nib.Nifti1Image
        Volume.

    slice_indices: list[int]
        List of slice indices to display. If None, defaults to the middle slice.

    axis: int
        Axis along which to slice (0,1 or 2).

    cmap: str
        Color map.
    """
    if isinstance(volume, nib.Nifti1Image):
        volume = volume.get_fdata()

    if slice_indices is None:
        slice_indices = [volume.shape[axis]//2] # default

    _, axes = plt.subplots(1, len(slice_indices), figsize=(15,5))

    if len(slice_indices) == 1:
        axes = [axes]

    for i, slice_idx in enumerate(slice_indices):
        if axis == 0:
            slice_vol = volume[slice_idx, :, :]
        elif axis == 1:
            slice_vol = volume[:, slice_idx, :]
        else:
            slice_vol = volume[:, :, slice_idx]

        if len(slice_vol) > 2:  # RGB image
            axes[i].imshow(slice_vol, cmap=cmap, origin="lower")
            axes[i].set_title(f"Slice {slice_idx} (axis {axis})")
            axes[i].axis("off")
        else:
            axes[i].imshow(slice_vol.T, cmap=cmap, origin="lower")
            axes[i].set_title(f"Slice {slice_idx} (axis {axis})")
            axes[i].axis("off")

    plt.show()


# I DON'T NEED THIS FUNCTION ANYMORE
#def plot_sources(
#        sources: mne.SourceSpaces,
#        mask: nib.Nifti1Image = None,
#        cras: np.ndarray = np.array([0,0,0]),
#        src_size: float = 5,
#        src_color: str = "red",
#        src_clut: np.ndarray = None,
#        brain_color: str = "lightblue",
#        mask_opacity: float = 0.2,
#) -> pv.Plotter:
#    """
#    Plot source space.
#
#    Parameters
#    ----------
#    sources: mne.SourceSpaces
#        Source space.

#    mask: nib.Nifti1Image
#        Gray matter.

#    ....
#    """
#    plotter = pv.Plotter()

#    # sources
#    points = mm.utils.get_src_position(sources)

#    if mask is None:
        # surfaces
#        vertices, faces = sources[0]["rr"], sources[0]["tris"]
#        surflh = mm.utils.surf_to_polydata(vertices, faces)

#        vertices, faces = sources[1]["rr"], sources[1]["tris"]
#        surfrh = mm.utils.surf_to_polydata(vertices, faces)

        # plot
#        plotter.add_mesh(surflh, show_edges=True, color=brain_color)
#        plotter.add_mesh(surfrh, show_edges=True, color=brain_color)
        
#    else: 
#        if np.all(cras == [0,0,0]):
#            print("\033[93mNo cras translation applied")
        
        # sources
#        points     = points*1000 + cras
#        inv_affine = np.linalg.inv(mask.affine)
#        points     = mne.transforms.apply_trans(inv_affine, points)
#        points     = pv.PolyData(points)

        # mask
#        volume = pv.ImageData(dimensions=mask.shape, spacing=(1,1,1), origin=(0,0,0))
#        volume.point_data["scalars"] = mask.get_fdata().flatten(order="F")

        # plot
#        plotter.add_mesh(volume.contour(3), color="white", opacity=mask_opacity)

#    if src_clut is None:
#        plotter.add_mesh(points, src_color, point_size=src_size, render_points_as_spheres=True)
#    else:
#        plotter.add_mesh(points, scalars=src_clut, point_size=src_size, render_points_as_spheres=True)


#    return plotter

# I DON'T NEED THIS FUNCTION ANYMORE
#def plot_brain_signal(sources, signal=None, n_points=2, name="signal", cmap="bwr", clim=None, show_max=False):

#    hemiL, hemiR = mm.utils.mne_to_polydata(sources, signal)

#    plotter = pv.Plotter()
#    plotter.add_mesh(hemiL, cmap=cmap, smooth_shading=True, clim=clim, name="lh")
#    plotter.add_mesh(hemiR, cmap=cmap, smooth_shading=True, clim=clim, name="rh")

#    if show_max == True:
#        split = sources[0]["nuse"]
#        idx_max = np.argmax(np.abs(signal))
#        if idx_max < split:
#            map_max = sources[0]["rr"][sources[0]["vertno"]][idx_max]
#        else:
#            map_max = sources[1]["rr"][sources[1]["vertno"]][idx_max-split]
        
#        plotter.add_mesh(map_max, render_points_as_spheres=True, color="violet", point_size=10)


#    return plotter



def plot_brain(
    sources         : mne.SourceSpaces | list[mne.SourceSpaces],
    signal          : np.ndarray | mne.SourceEstimate | list[np.ndarray] | list[mne.SourceSpaces] = None,
    overlay         : np.ndarray | nib.Nifti1Image | list[np.ndarray] | list[nib.Nifti1Image]     = None,
    tkras2ras       : np.ndarray | list[np.ndarray] = np.zeros(3),
    show_max        : bool  = False,
    show_src        : bool  = False,
    clim            : list  = None,
    cmap            : str   = None,
    absolute        : bool  = False,
    smooth_shading  : bool  = False,
    title           : str   = "",
    signal_label    : str   = None,
    overlay_color   : str   = "green",
    overlay_label   : str   = None,
    overlay_size    : float = 1,
    overlay_opacity : float = 1, 
):
    """
    Plot the brain !

    Parameters
    ----------
    sources: mne.SourceSpaces 
        Source space.

    signal: mne.SourceEstimate | np.ndarray (N,)
        Source values.

    overlay: np.ndarray (M,3) | nib.Nifti1Image
        Overlay defined as point cloud.

    tkras2ras: np.ndarray
        Affine transform (freesurfer ras -> T1 ras). Necessary to align MRI images with meshes.

    show_max: bool
        Show maximum of signal if True.

    show_src: bool
        Show sources.

    clim: list(float)
        Color limit.

    cmap: str
        Color map.

    absolute: bool
        Whether to plot absolute values.

    smooth_shading: str
        Smoothen mesh if True.

    title: str
        Plot title

    sname: str
        Signal name.

    overlay_color: str
        Overlay color

    overlay_label: str
        Overlay label

    overlay_size: float
        Overlay point size

    overlay_opacity: float
        Overlay opacity
    """
    # find longest list
    size = 1
    if not isinstance(sources, mne.SourceSpaces):
        if len(sources) > size:
            size = len(sources)
    for obj in [signal, overlay, tkras2ras]:
        if isinstance(obj, list):
            size = max(size, len(obj))

    # check every entry
    if isinstance(sources, mne.SourceSpaces):
        sources = [sources]*size

    if overlay is not None:
        if not isinstance(overlay, list):
            overlay = [overlay]*size

    if not isinstance(tkras2ras, list):
        tkras2ras = [tkras2ras]*size

    if signal is not None:
        if isinstance(signal, mne.SourceEstimate) or isinstance(signal, np.ndarray):
            signal = [signal]*size

    if not isinstance(title, list):
        title = [title]*size

    # if Time Series plot TimeMesh
    if signal is not None and isinstance(signal[0], mne.SourceEstimate):
        return TimeMesh(
                sources, signal, overlay=overlay, titles=title, cmap=cmap, 
                absolute=absolute, smooth_shading=smooth_shading, overlay_color=overlay_color,
                overlay_label=overlay_label, overlay_size=overlay_size, overlay_opacity=overlay_opacity,
            )

    # else plot single time point
    plotter = pv.Plotter(shape=(1,size))

    # brain plot
    if signal is not None:
        if absolute:
            signal = [np.abs(s) for s in signal]

        if clim is None:
            clim = find_clim(signal)

        # clim/cmap when positive values
        if cmap is None:
            if clim[0] < 0:
                cmap = "bwr"
            else:
                cmap = "Reds"

        for i in range(size):
            # brain meshes
            hemi_l, hemi_r = mne_to_polydata(sources[i], signal[i])

            plotter.subplot(0,i)
            plotter.add_mesh(hemi_l, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="lh")
            plotter.add_mesh(hemi_r, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="rh")
            plotter.add_title(title[i])

    else:
        for i in range(size):
            # brain meshes
            hemi_l, hemi_r = mne_to_polydata(sources[i], None)

            plotter.subplot(0,i)
            plotter.add_mesh(hemi_l, color="white", smooth_shading=smooth_shading, name="lh")
            plotter.add_mesh(hemi_r, color="white", smooth_shading=smooth_shading, name="rh")
            plotter.add_title(title[i])

    # maximum value
    if show_max:
        for i in range(size):
            split   = sources[i][0]["nuse"]
            idx_max = np.argmax(np.abs(signal[i]))
            if idx_max < split:
                map_max = sources[i][0]["rr"][sources[i][0]["vertno"]][idx_max]
            else:
                map_max = sources[i][1]["rr"][sources[i][1]["vertno"]][idx_max-split]

            plotter.subplot(0,i)
            plotter.add_mesh(map_max, color="yellow", point_size=10)

    # overlay
    if overlay is not None:
        for i in range(size):
            if isinstance(overlay[i], nib.Nifti1Image):
                over = nib_to_polydata(overlay[i], trans=np.linalg.inv(tkras2ras[i]))
            else:
                over = overlay[i]

            plotter.subplot(0,i)
            plotter.add_mesh(over, color=overlay_color, label=overlay_label, point_size=overlay_size, opacity=overlay_opacity)

    if show_src:
        for i in range(size):
            pos = pv.PolyData(get_dipole_positions(sources[i]))

            plotter.subplot(0,i)
            plotter.add_mesh(pos, color="green", point_size=5, render_points_as_spheres=True)


    return plotter


class TimeMesh:
    """
    Class to manage temporal brain plots.
    """
    def __init__(
        self, 
        sources         : mne.SourceSpaces | list[mne.SourceSpaces], 
        stc             : mne.SourceEstimate | list[mne.SourceEstimate],
        overlay         : np.ndarray | nib.Nifti1Image | list[np.ndarray] | list[nib.Nifti1Image] = None,
        tkras2ras       : np.ndarray | list[np.ndarray] = np.zeros(3),
        titles          : str | list[str] = "",
        cmap            : str = None,
        absolute        : bool = False,
        smooth_shading  : bool = False,
        overlay_color   : str   = "green",
        overlay_label   : str   = None,
        overlay_size    : float = 1,
        overlay_opacity : float = 1,
    ):
        """
        Init

        Parameters
        ----------
        sources: mne.SourceSpace 
            Source space.

        stc: mne.SourceEstimate 
            Source values over time (data with size N,T)

        cmap: Color map 

        absolute: bool
            Whether to take absolute values.
        """
        # instanciate objects
        self.to_self(sources, stc, overlay, tkras2ras)
        self.select_cmap(cmap, absolute)

        size = len(self.sources)

        # instanciate first plot
        start      = int(self.stc[0].data.shape[1]/2)
        signals    = [s.data[:, start] for s in self.stc]
        self.hemis = [mne_to_polydata(self.sources[i], signals[i]) for i in range(size)]

        self.plotter = pv.Plotter(shape=(1,size))

        for i in range(size):
            self.plotter.subplot(0,i)
            self.plotter.add_mesh(self.hemis[i][0], cmap=self.cmap, clim=self.clim, smooth_shading=smooth_shading)
            self.plotter.add_mesh(self.hemis[i][1], cmap=self.cmap, clim=self.clim, smooth_shading=smooth_shading)

            if isinstance(titles, list):
                self.plotter.add_title(titles[i])
            else:
                self.plotter.add_title(titles)

        # overlay
        if overlay is not None:
            for i in range(size):
                if isinstance(self.overlay[i], nib.Nifti1Image):
                    over = nib_to_polydata(self.overlay[i], trans=self.cras[i])
                else:
                    over = self.overlay[i]

                self.plotter.subplot(0,i)
                self.plotter.add_mesh(over, color=overlay_color, label=overlay_label, point_size=overlay_size, opacity=overlay_opacity)

        # add time slider
        self.plotter.add_slider_widget(
            callback = self.__call__,
            rng      = [self.stc[0].tmin, self.stc[0].times[-1]],
            value    = self.stc[0].times[start],
            title    = "Time [s]",
            style    = "modern",
        )

    def show(self):
        """show plot"""
        self.plotter.show()

    def __call__(self, time_point):
        """
        Change point of time. 
        """
        idx    = self.stc[0].time_as_index(time_point)
        signal = [s.data[:,idx] for s in self.stc]
        size   = len(signal)

        # update values (plot update automatically like magic)
        for i in range(size):
            hemi_l, hemi_r = mne_to_polydata(self.sources[i], signal[i])
            self.hemis[i][0]["signal"] = hemi_l["signal"]
            self.hemis[i][1]["signal"] = hemi_r["signal"]

    def to_self(self, sources, stc, overlay, cras):
        """basically __init__?"""
        # Find longest list
        longest_list = 1
        if not isinstance(sources, mne.SourceSpaces):
            longest_list = max(longest_list, len(sources))
        for obj in [stc, overlay, cras]:
            if isinstance(obj, list):
                longest_list = max(longest_list, len(obj))

        # Check data and convert to list
        if isinstance(sources, mne.SourceSpaces):    # sources are list so investigate if element inside is SourceSpace or list
            sources = [sources]*longest_list

        if isinstance(stc, list):
            times = stc[0].times
            for s in stc:
                assert len(times) == len(s.times), "Source estimates must have the same time span in TimeMesh"
        else:
            stc = [stc]*longest_list

        if overlay is not None:
            if not isinstance(overlay, list):
                overlay = [overlay]*longest_list

        if not isinstance(cras, list):
            cras = [cras]*longest_list

        assert len(sources) == len(stc), "List length of sources and stc must match in TimeMesh"

        self.sources = sources
        self.stc     = stc
        self.overlay = overlay
        self.cras    = cras


    def select_cmap(self, cmap=None, absolute=False):
        """Select proper colormap"""
        maxi = -np.inf
        mini = np.inf

        # find max and min amongst all data
        for stc in self.stc:

            if absolute:
                stc.data = np.abs(stc.data)

            stc_max = np.max(stc.data)
            stc_min = np.min(stc.data)

            maxi = max(maxi, stc_max)
            mini = min(mini, stc_min)

        # clim/cmap when positive values
        if mini > 0:
            clim = [0, maxi]

            if cmap is None:
                cmap = "Reds"

        # clim/cmap when pos/neg values
        else:
            extremum = np.max([np.abs(mini), np.abs(maxi)])
            clim     = [-extremum, extremum]

            if cmap is None:
                cmap = "bwr"

        self.clim = clim
        self.cmap = cmap


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

    clim = find_clim(leadfield)

    # plot
    plotter = plot_brain(forward["src"], np.zeros(leadfield.shape[1]), clim=clim)
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
            tmp = plot_brain(forward["src"], leadfield[closest_point_id], clim=clim)
            plotter.meshes[0]["signal"] = tmp.meshes[0]["signal"]
            plotter.meshes[1]["signal"] = tmp.meshes[1]["signal"]

            # update the plotter
            plotter.render()

    plotter.enable_point_picking(callback=pick_electrode)

    return plotter

def plot_dipole_function(source_space, dipole_function, title="Dipole function"):
    """
    Plots a function that depends on dipole position (e.g. point-spread function, cross-talk function, ...)

    Parameters
    ----------
    source_space: mne.SourceSpaces 
        Dipole space.

    dipole_function: np.ndarray (N,N)
        Dipole function. Row i corresponds to function output of dipole i.

    Returns
    -------
    Pyvista object
    """

    plotter = pv.Plotter()

    # style
    cmap = "bwr"
    smooth_shading = True
    max_abs = np.max(np.abs(dipole_function))
    clim = [-max_abs, max_abs]
    size = 5

    # brain mesh
    hemi_l, hemi_r = mne_to_polydata(source_space, dipole_function[0])

    plotter.add_mesh(hemi_l, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="lh")
    plotter.add_mesh(hemi_r, cmap=cmap, smooth_shading=smooth_shading, clim=clim, name="rh")
    plotter.add_title(title)

    # dipoles
    dipoles                 = pv.PolyData(get_dipole_positions(source_space))
    dipoles["selection"]    = np.zeros(dipoles.n_cells)
    dipoles["selection"][0] = 1

    dipoles_actor = plotter.add_mesh(dipoles, render_points_as_spheres=True, point_size=size)
    plotter.remove_scalar_bar(title="selection")

    # dipole visibility checkbox
    def toggle_vis(flag) -> None:
        dipoles_actor.SetVisibility(flag)

    plotter.add_checkbox_button_widget(toggle_vis, value=True)

    # dipole picking
    def pick_dipole(picked_point):
        if picked_point is not None:
            # get index of closest point
            closest_point_id = dipoles.find_closest_point(picked_point)

            # change selected dipole
            dipoles["selection"] = np.zeros(dipoles.n_cells)
            dipoles["selection"][closest_point_id] = 1

            # change corresponding function output
            tmp_hemi_l, tmp_hemi_r = mne_to_polydata(source_space, dipole_function[closest_point_id])
            plotter.meshes[0]["signal"] = tmp_hemi_l["signal"]
            plotter.meshes[1]["signal"] = tmp_hemi_r["signal"]

            plotter.render()

    plotter.enable_point_picking(callback=pick_dipole)

    return plotter



def plot_connectome(connectome, name="Connectome", show_adj=True, show_degree=True, show_weights=True, nonzero=False, figsize=(12,5),):
    """
    Plot various measures of the connectome.

    Parameters:
    ----------
    connectome: np.ndarray (N,N)
        Connectome

    name: str (default Connectome)
        Name given to connectome

    nonzero: bool
        Whether to show the zero-weights in the distributions

    show_adj: bool
        Whether to show the adjacency matrix or not

    show_degree: bool
        Whether to show the degree distribution or not

    show_weights: bool
        Whether to show the weight distribution or not
    
    figsize: tuple(width, height)
        Figure size
    """
    n_plots  = np.sum([show_adj, show_degree, show_weights])
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    fig.suptitle(name)

    i = 0
    if show_adj:
        cmap = cm_frrim()

        axs[i].matshow(connectome, cmap=cmap)
        axs[i].set_title("Adjacency Matrix")
        axs[i].set_xlabel("Dipoles")
        axs[i].set_ylabel("Dipoles")

        i += 1

    if show_degree:
        degree = np.sum(connectome, axis=1)

        if nonzero:
            degree = degree[degree>0]

        axs[i].hist(degree, bins=100, color="skyblue")
        axs[i].set_title("Degree Distribution")
        axs[i].set_xlabel("Degree")
        axs[i].set_ylabel("Frequency")

        i += 1

    if show_weights:
        weights = connectome.flatten().astype(float)

        if nonzero:
            weights = weights[weights > 0]

        axs[i].hist(weights, bins=100, color="skyblue")
        axs[i].set_title("Weight Distribution")
        axs[i].set_xlabel("Weight")
        axs[i].set_ylabel("Frequency")

    plt.tight_layout()
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


def find_clim(data):
    """
    Find color limits to account accross all data.

    Parameters
    ----------
    data: list[np.ndarray]
        List of datas.

    Returns
    -------
    clim: tuple
        Value range defined accross all datas.
    """
    if not isinstance(data, list):
        data = [data]

    abs_max       = 0
    only_positive = True

    for d in data:
        abs_max = max(abs_max, np.max(np.abs(d)))
        if np.any(d < 0):
            only_positive = False

    if only_positive:
        return [0, abs_max]
    
    return [-abs_max, abs_max]


def plot_resection(
        sources, 
        resection, 
        cras=np.zeros(3), 
        signal=None, 
        show_max=False, 
        cmap="reds", 
        style="vox", 
        color="green"
):
    """
    Plot resection area. 

    Parameters
    ----------
    sources: mne.SourceSpaces
        Source space.

    resection: nib.Nifti1Image
        Resection mask.

    cras: np.ndarray (3,)
        Translation (FreeSurfer -> MRI)

    signal: np.ndarray (N,)
        Brain activity.

    show_max: bool
        Show maximum of brain acitivity.

    cmap: str
        Color map for brain activity.

    style: str
        Style of resection area:
        - cloud: looks like a cloud
        - vox: see individual voxels

    color: str
        Color of resection area. 
    """
    
    if signal is None:
        signal = np.zeros(sources[0]["nuse"]+sources[1]["nuse"])

    if style == "cloud":
        style = dict(
            emissive = True,
            style = "points_gaussian",
            point_size = 10,
        )
    elif style == "vox":
        style = dict(
            point_size=0.1,
            opacity=0.3,
        )
    else:
        raise ValueError(f"Unknown style '{style}' for resection plot.")

    # resection area
    #voxels = np.argwhere(resection.get_fdata()>0)
    #points = (nib.affines.apply_affine(resection.affine, voxels)-cras)/1000
    points = nib_to_polydata(resection, trans=cras)

     # plotter
    plotter = plot_brain_signal(sources, signal, cmap=cmap, show_max=show_max)
    plotter.add_mesh(points, color=color, label="Resected Area", **style)

    return plotter



def plot_eeg(eeg: mne.Evoked, colorscale="Turbo", title="EEG"):
    """
    Plot EEGs using plotly (for dashboard)
    
    Parameters
    ----------
    eeg: mne.Evoked 
        EEG 
    
    colorscale: str
        Plotly colorscale to use.

    Returns
    -------
    plotly lineplot
    """
    # data
    data = eeg.data
    data = {ch:data[i,:] for i,ch in enumerate(eeg.ch_names)}

    times = eeg.times

    n_colors = len(eeg.ch_names)
    colors   = px.colors.sample_colorscale(colorscale, [n/(n_colors-1) for n in range(n_colors)])

    # figure
    fig = go.Figure()
    for ch_name, time_serie in data.items():
        fig.add_trace(go.Scatter(
            x    = times,
            y    = time_serie,
            name = ch_name,
            line = {
                "color": colors.pop(),
                "width": 1,
            } 
        ))

    # layout
    fig.update_layout(
        plot_bgcolor = "rgb(34, 34, 34)",
        paper_bgcolor="rgb(34, 34, 34)",
        showlegend   = False,
        title        = {"text":title, "font":{"size": 20, "color":"white"}},
        xaxis        = {
            "linecolor":"green",
            "linewidth": 2,
            "mirror": True,
            "gridcolor": "green",
            "gridwidth": 1,
            "griddash":"dash",
            "tickfont": {"color":"green"}
        },
        yaxis        = {
            "linecolor":"green",
            "linewidth": 2,
            "mirror": True,
            "gridcolor": "green",
            "gridwidth": 1,
            "type": "-",
            "griddash":"dash",
            "tickfont": {"color":"green"},        
        },
    )

    return fig


#################################################
########### Special plot publication ############
#################################################


def brain_gif(
    sources         : mne.SourceSpaces, 
    signals         : np.ndarray,
    fname           : str,
    head            : dict = None, 
    camera_position : str = "yz", 
    azimuth         : int = 0,
    roll            : int = 0,
    zoom            : float = 1,
    fps             : int = 10,
):
    """
    Create gif of brain. 

    sources: mne.SourceSpaces 
        Source space.

    signals: np.ndarray (N,T)
        Brain signals (signals should be max 15 frames).

    fname: str
        File name

    head: dict (mne surface)
        Head mesh

    camera_position: str
        Sets camera position.
    
    azimuth: int
        Sets azimuth angle.
    
    zoom: float
        Camera zoom.
    """
    plotter = pv.Plotter(notebook=False, off_screen=True)

    # brain hemispheres
    hemiL, hemiR = mne_to_polydata(sources, signals[:,0])
    plotter.add_mesh(hemiL, smooth_shading=True, cmap="Reds", clim=[0, np.max(signals)])
    plotter.add_mesh(hemiR, smooth_shading=True, cmap="Reds", clim=[0, np.max(signals)])
    plotter.remove_scalar_bar()

    # head
    if head is not None:
        head = surf_to_polydata(head["rr"], head["tris"])
        plotter.add_mesh(head, show_edges=True, color="white", opacity=0.2, edge_opacity=0.8, edge_color="black")

    # camera position
    plotter.camera_position = camera_position
    plotter.camera.azimuth += azimuth
    plotter.camera.roll += roll
    plotter.camera.zoom(zoom)

    # generate gif
    plotter.open_gif(fname, fps=fps)
    nframe = signals.shape[1]
    for frame in np.arange(nframe):
        new_hemiL, new_hemiR = mne_to_polydata(sources, signals[:,frame])
        hemiL["signal"] = new_hemiL["signal"]
        hemiR["signal"] = new_hemiR["signal"]
        plotter.write_frame()

    plotter.close()


def plot_hops(sim, n_hops=3):

    signals = np.zeros((n_hops+1, sim.act.data.shape[1]))

    # select source
    idx          = np.random.choice(sim.loc_ids)
    signals[0,:] = sim.act.data[idx,:] + sim.bnoise[idx,:]

    # select hop neighbors
    starting_roi = np.zeros(sim.sc.shape[0], dtype=bool)
    starting_roi[sim.loc_ids] = True

    rois = starting_roi
    for i in range(n_hops):
        # compute nex hop roi
        roi = (sim.sc @ rois > 0) & (~rois)

        # check if roi is empty or wrong
        assert np.all((roi==0) | (roi==1))
        if np.all(roi==0):
            print("\033[93mNo more neighbors\033[0m")
            break 
        roi = roi.astype(bool)

        idx          = np.random.choice(np.argwhere(roi).squeeze())
        signals[i+1,:] = sim.act.data[idx,:] + sim.bnoise[idx,:]

        rois = rois | roi 

    _, axes = plt.subplots(nrows=1, ncols=n_hops+1, sharex=True, sharey=True, figsize=(15,3))
    coloors   = ["#EE6C4D", "#FFB627", "#3D5A80", "#98C1D9", "#293241"]

    for i in range(n_hops+1):
        for j in range(i+1):
            axes[i].plot(signals[j,:], color=coloors[j], linewidth=2)
        axes[i].set_xlabel("Time [s]")
    
    axes[0].set_ylabel("Amplitude [V]")

    plt.show()

def plot_fourier_image_example():
    N = 30
    f, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, sharex="col", sharey="row", figsize=(15,3))

    # first image
    xf = np.zeros((N,N))
    xf[0, 1] = 1
    Z = ifftn(xf)
    ax1.imshow(np.real(Z), cmap=cm.gray)

    # second image
    xf = np.zeros((N,N))
    xf[2, 0] = 1
    Z = ifftn(xf)
    ax2.imshow(np.real(Z), cmap=cm.gray)

    # third image
    xf = np.zeros((N,N)) 
    xf[3, 5] = 1
    Z = ifftn(xf)
    ax3.imshow(np.real(Z), cmap=cm.gray)

    # fourth image
    xf = np.zeros((N,N))
    xf[12, 5] = 1
    Z = ifftn(xf)
    ax4.imshow(np.real(Z), cmap=cm.gray)

    # fith image
    xf = np.zeros((N,N))
    xf[14, 14] = 1
    Z = ifftn(xf)
    ax5.imshow(np.real(Z), cmap=cm.gray)

    plt.show()



###############################
########## Color Maps #########
###############################

def cm_frrim():
    """
    Color map used for FRRIM. 
    """
    newcolors          = cm.Blues(np.linspace(0,1,256))
    newcolors[250:, :] = np.array([61/256, 90/256, 128/256, 1])
    newcmp             = ListedColormap(newcolors)

    return newcmp


def plot_harmonics(src, harmonics, frequencies=[0,1,2,3,4,5]):
    """
    Plot brain harmonics.

    Parameters
    ----------
    src: mne.SourceSpace
        Source Space.

    harmonics: np.ndarray 
        Harmonics

    frequencies: list[int]
        Frequencies to plot.

    Returns
    -------
    plot: pv.Plotter
    """
    harmonics = [harmonics[freq,:] for freq in frequencies]
    titles    = [f"harm {i}" for i in frequencies]
    plotter   = mm.viz.plot_brain(src, harmonics, title = titles)
    return plotter



def plot_graph(points, adjacency):
    """
    Parameters
    ----------
    points: np.array (N,3) 
    lines: np.array (N,2)
    """

    P = pv.PolyData(points)
    A1 = adjacency > 0
    A2 = adjacency < 0

    lines1 = []
    lines2 = []
    for i in range(len(A1)):
        for j in range(i, len(A1)):
            if A1[i,j]:
                lines1 += [points[i], points[j]]
            if A2[i,j]:
                lines2 += [points[i], points[j]]
    lines1 = np.array(lines1)
    lines2 = np.array(lines2)

    p = pv.Plotter()
    p.add_mesh(P)
    if len(lines1) > 0:
        p.add_lines(lines1, color="green", width=0.05)
    if len(lines2) > 0:
        p.add_lines(lines2, color="red", width=0.01)

    return p
