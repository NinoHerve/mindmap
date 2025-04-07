import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import mindmap as mm
import mne
import plotly.graph_objs as go


def show_slices(
        volume: np.ndarray | nib.Nifti1Image, 
        slice_indices: list[int] = None, 
        axis: int = 2,
        cmap: str = "gray",
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

    if slice_indices == None:
        slice_indices = [volume.shape[axis]//2] # default

    fig, axes = plt.subplots(1, len(slice_indices), figsize=(15,5))

    if len(slice_indices) == 1:
        axes = [axes]

    for i, slice_idx in enumerate(slice_indices):
        if axis == 0:
            slice_vol = volume[slice_idx, :, :]
        elif axis == 1:
            slice_vol = volume[:, slice_idx, :]
        elif axis == 2:
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


def plot_sources(
        sources: mne.SourceSpaces,
        mask: nib.Nifti1Image = None,
        cras: np.ndarray = np.array([0,0,0]),
        src_size: float = 5,
        src_color: str = "red",
        src_clut: np.ndarray = None, 
        brain_color: str = "lightblue",
        mask_opacity: float = 0.2, 
) -> pv.Plotter:
    
    plotter = pv.Plotter()

    # sources
    points = mm.utils.get_src_position(sources)

    if mask is None:
        # surfaces
        vertices, faces = sources[0]["rr"], sources[0]["tris"]
        surflh = mm.utils.surf_to_polydata(vertices, faces)

        vertices, faces = sources[1]["rr"], sources[1]["tris"]
        surfrh = mm.utils.surf_to_polydata(vertices, faces)

        # plot
        plotter.add_mesh(surflh, show_edges=True, color=brain_color)
        plotter.add_mesh(surfrh, show_edges=True, color=brain_color)
        
    else: 
        if np.all(cras == [0,0,0]):
            print("\033[93mNo cras translation applied")
        
        # sources
        points     = points*1000 + cras
        inv_affine = np.linalg.inv(mask.affine)
        points     = mne.transforms.apply_trans(inv_affine, points)
        points     = pv.PolyData(points)

        # mask
        volume = pv.ImageData(dimensions=mask.shape, spacing=(1,1,1), origin=(0,0,0))
        volume.point_data["scalars"] = mask.get_fdata().flatten(order="F")

        # plot
        plotter.add_mesh(volume.contour(3), color="white", opacity=mask_opacity)

    if src_clut is None:
        plotter.add_mesh(points, src_color, point_size=src_size, render_points_as_spheres=True)
    else:
        plotter.add_mesh(points, scalars=src_clut, point_size=src_size, render_points_as_spheres=True)


    return plotter


def plot_brain_signal(sources, signal=None, n_points=2, name="signal", cmap="bwr", clim=None, show_max=False):
    # sources
    src_lh = sources[0]["rr"][sources[0]["vertno"]]
    src_rh = sources[1]["rr"][sources[1]["vertno"]]

    pointslh = pv.PolyData(src_lh)
    pointsrh = pv.PolyData(src_rh)

    # signal
    split = sources[0]["nuse"]
    pointslh[name] = signal[:split]
    pointsrh[name] = signal[split:]

    # surfaces
    vertices, faces = sources[0]["rr"], sources[0]["tris"]
    surflh = mm.utils.surf_to_polydata(vertices, faces)

    vertices, faces = sources[1]["rr"], sources[1]["tris"]
    surfrh = mm.utils.surf_to_polydata(vertices, faces)

    # interpolate signal on surface
    surflh = surflh.interpolate(pointslh, strategy="closest_point", n_points=n_points)
    surfrh = surfrh.interpolate(pointsrh, strategy="closest_point", n_points=n_points)

    plotter = pv.Plotter()
    plotter.add_mesh(surflh, cmap=cmap, smooth_shading=True, clim=clim, name="lh")
    plotter.add_mesh(surfrh, cmap=cmap, smooth_shading=True, clim=clim, name="rh")

    if show_max == True:
        idx_max = np.argmax(np.abs(signal))
        if idx_max < split:
            map_max = src_lh[idx_max]
        else:
            map_max = src_rh[idx_max-split]
        
        plotter.add_mesh(map_max, render_points_as_spheres=True, color="violet", point_size=10)


    return plotter
    


class TimeMesh:
    def __init__(self, sources, t_start, time_series, sfreq, cmap="RdBu"):
        self.sources = sources
        self.time_series = time_series
        self.sfreq = sfreq
        self.t_start = t_start

        if np.all(time_series>-1e-20):
            clim = [0, time_series.max()]
        else:
            max_abs = max(abs(time_series.min()), abs(time_series.max()))
            clim    = [-max_abs, max_abs]

        half = int(time_series.shape[1]/2)
        self.plotter = mm.viz.plot_brain_signal(
            sources, 
            time_series[:,half], 
            clim = clim, 
            cmap = cmap
        )


    def __call__(self, time_point):
        # convert time_point to indices
        i = int((time_point-self.t_start)*self.sfreq)

        # generate new images
        new_plotter = mm.viz.plot_brain_signal(
            self.sources,
            self.time_series[:, i],
        )

        # instanciate new images
        self.plotter.meshes[0].copy_from(new_plotter.meshes[0])
        self.plotter.meshes[1].copy_from(new_plotter.meshes[1])


def plot_brain_time_series(sources, time_series, t_start=0.0, sfreq=200, cmap="bwr"):
    time_mesh = TimeMesh(sources, t_start, time_series, sfreq, cmap=cmap)
 
    time_mesh.plotter.add_slider_widget(
        callback = lambda value: time_mesh(value),
        rng      = [t_start, t_start + time_series.shape[1]/sfreq],
        value    =  t_start + time_series.shape[1]/sfreq/2,
        title    = "Time [s]",
        style    = "modern",
    )

    time_mesh.plotter.show()


def plot_leadfield(eeg, forward):
    # get electrode positions
    points = mm.utils.get_ch_positions(eeg)
    cap    = pv.PolyData(points)
    cap["highlight"] = np.zeros(len(points))

    # get leadfield matrix
    leadfield = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, use_cps=True,
    )
    leadfield = leadfield["sol"]["data"]

    max_abs = max([abs(leadfield.max()), abs(leadfield.min())])
    clim    = [-max_abs, max_abs]

    # plot
    plotter = mm.viz.plot_brain_signal(forward["src"], np.zeros(leadfield.shape[1]), clim=clim)
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
            tmp = mm.viz.plot_brain_signal(forward["src"], leadfield[closest_point_id], clim=clim)
            plotter.meshes[0]["signal"] = tmp.meshes[0]["signal"]
            plotter.meshes[1]["signal"] = tmp.meshes[1]["signal"]

            # update the plotter
            plotter.render()

    plotter.enable_point_picking(callback=pick_electrode)
    
    return plotter



def plot_connectome(connectome, name="Connectome", nonzero=False, figsize=(12,5)):
    degree = np.sum(connectome, axis=1)

    fig, axs = plt.subplots(1,2, figsize=figsize)

    axs[0].matshow(connectome, cmap="Blues")
    axs[0].set_title(name)
    axs[0].set_xlabel("Dipoles")
    axs[0].set_ylabel("Dipoles")

    if nonzero == True:
        degree = degree[degree>0]

    axs[1].hist(degree, bins=100, color="skyblue")
    axs[1].set_title("Degree Distribution")
    axs[1].set_xlabel("Degree")
    axs[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()



def plot_spectrum(spectrum):
    """
    Plot the connectome spectrum.

    Parameters
    ----------
        spectrum: nd.array (nb_nodes,)
            Connectome spectrum.

    Returns
    -------
        fig: plotly.graph_objs
    """
    trace = go.Bar(
        y=spectrum,
        marker=dict(
            color="darkblue",
            line=dict(color="darkblue"),
        ),
    )

    layout = go.Layout(dict(
        title="Fourier graph coefficients",
        showlegend=False, 
        plot_bgcolor="white",
        xaxis=dict(
            title="frequency indices",
            linecolor="black",
        ),
        yaxis=dict(
            title="amplitude",
            linecolor="black",
            gridcolor="black",
        ),
    ))

    return go.Figure(trace, layout=layout)


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
    voxels = np.argwhere(resection.get_fdata()>0)
    points = (nib.affines.apply_affine(resection.affine, voxels)-cras)/1000

     # plotter
    plotter = mm.viz.plot_brain_signal(sources, signal, cmap=cmap, show_max=show_max)
    plotter.add_mesh(pv.PolyData(points), color=color, label="Resected Area", **style)

    return plotter