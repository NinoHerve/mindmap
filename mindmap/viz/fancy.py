"""
Fancy plots for presentations.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyvista as pv

import mne

from mindmap.viz.utils import BaseMesh, BrainMesh, ChannelMesh

def create_dynamic_eeg_plot(
    evoked: mne.Evoked,
    picks: list[str],
    total_duration = 5.0,
    figsize = (10, 6),
    save_path = None,
):
    """
    Creates a dynamic, synchronized animation of EEG/iEEG channels drawing over time.

    Parameters
    ----------
    evoked: mne.Evoked
        The MNE Evoked object containing the data.

    picks: list or str
        The specific channel names to plot.

    total_duration: float
        The exact time (in seconds) the entire animation should take to draw.
    
    figsize: tuple
        The width and height of the figure in inches.

    save_path: str, optional
        If provided, saves the animation to this path (e.g. 'plot.gif' or 'plot.mp4').
    """
    # 1. Extract data based on picks
    evoked_subset = evoked.copy().pick(picks)
    data = evoked_subset.data
    times = evoked_subset.times
    ch_names = evoked_subset.ch_names

    total_frames = len(times)
    # Calculate millisecond interval between frames for live display
    frame_interval = (total_duration*1000) / total_frames

    # 2. Setup Figure and Axes
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate vertical spacing so channels don't overlap
    offset = np.max(np.abs(data)) * 1.5
    offsets = np.arange(len(ch_names)) * offset

    # Initialize empty lines
    lines = []
    for _, ch_name in enumerate(ch_names):
        (line,) = ax.plot([], [], label=ch_name, lw=1.5)
        lines.append(line)

    # Styling
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-offset, offsets[-1] + offset)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Channels", fontsize=11)
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # 3. Animation Update Logic
    def update(frame):
        current_times = times[:frame]
        for i, line in enumerate(lines):
            current_data = data[i, :frame] + offsets[i]
            line.set_data(current_times, current_data)
        return lines

    # 4. Generate Animation
    anim = FuncAnimation(
        fig,
        update,
        frames = total_frames,
        interval = frame_interval,
        blit = True,
        repeat = False,
    )

    # 5. Save or Show
    if save_path:
        print(f"Saving animation to {save_path}... This might take a moment.")
        # Calculate precise FPS to match the target duration
        target_fps = int(total_frames/total_duration)

        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=target_fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=target_fps, codec="libx264")
        else:
            anim.save(save_path, fps=target_fps)
        print("Saving complete!")
    else:
        plt.show()

    return fig, anim


def create_mesh_animation(
    mesh_obj: BrainMesh,
    addons: list[BaseMesh] = None,
    total_duration: float = 5.0,
    save_path: str = "brain_activity.mp4",
    camera_position: str | list = "iso",
    **plotter_kwargs
):
    """
    Creates a dynamic, frame-locked video animation.

    Parameters
    ----------
    mesh_obj: BrainMesh or SourceMesh
        An instance of your class containing the geometries and functional time series.

    addons: list[BaseMesh]
        A list containing instances of HeadMesh, ChannelMesh, or other BaseMesh structures to render and update alongside the primary cortical source.

    total_duration: float
        The exact length of the final video/GIF in seconds.

    save_path: str
        Path to save the video. Must end in '.mp4' or '.gif'

    **plotter_kwargs: dict
        Optional formatting adjustments passed directly to PyVista Plotter initializers.
    """
    # Initialize add-ons afely if none are provided
    if addons is None:
        addons = []

    # 1. Access properties natively managed by BaseMesh parent class
    original_frames = mesh_obj.n_times
    times = mesh_obj.times
    scalar_name = mesh_obj.scalar_name

    if original_frames == 0 or scalar_name is None:
        raise ValueError("The provided mesh object contains no dynamic data to animate.")

    # 2. Timing & Frame Downsampling
    TARGET_FPS = 40
    max_allowed_frames = int(total_duration * TARGET_FPS)

    if original_frames > max_allowed_frames:
        step = int(np.ceil(original_frames / max_allowed_frames))
        frame_indices = np.arange(0, original_frames, step)
        if frame_indices[-1] != original_frames - 1:
            frame_indices = np.append(frame_indices, original_frames - 1)
    else:
        frame_indices = np.arange(original_frames)

    actual_fps = len(frame_indices) / total_duration
    print(f"Syncing: Rendering {len(frame_indices)} frames at {actual_fps:.2f} FPS (Target: {total_duration}s).")

    # 3. Setup the PyVista Plotter Environment
    # Merging default aesthetics with any manual presentation adjustments
    plotter_defaults = {"off_screen": True}
    plotter = pv.Plotter(**(plotter_defaults | plotter_kwargs))

    # Calculate global range from your internal data matrix to keep color scaling constant over time.
    global_max = np.max(np.abs(mesh_obj._data))
    clim = [0, global_max]  # Using [0, max] matches default "Reds" map smoothly

    # Update to initial frame using your method
    mesh_obj.update_signal(frame_indices[0])

    # Inject your default dictionaries directly into PyVista's actor configuration
    actor_l = plotter.add_mesh(mesh_obj.hemi_l, scalars=scalar_name, clim=clim, show_scalar_bar=True, **mesh_obj.plot_kwargs)
    actor_r = plotter.add_mesh(mesh_obj.hemi_r, scalars=scalar_name, clim=clim, show_scalar_bar=False, **mesh_obj.plot_kwargs)

    # Initialize and plot all secondary structural add-ons
    for addon in addons:
        addon.update_signal(frame_indices[0])

        # Calculate dynamic color limits specifically for electrode voltage recordings if applicable
        if isinstance(addon, ChannelMesh):
            ch_max = np.max(np.abs(addon._data))
            # Use a symmetric range [-max, max] since 'bkr' is a divergin colormap
            addon_clim = [-ch_max, ch_max]
            plotter.add_mesh(addon.polydata, scalars=addon.scalar_name, clim=addon_clim, **addon.plot_kwargs)
        else:
            # Fallback for structural boundaries like HeadMesh that might nor require dynamic colormaps
            plotter.add_mesh(addon.polydata, scalars=addon.scalar_name, **addon.plot_kwargs)

    # Frame the camera view beautifully
    plotter.camera_position = camera_position
    plotter.reset_camera()

    # 4. Bind File Writers
    if save_path.endswith(".gif"):
        plotter.open_gif(save_path, fps=int(actual_fps))
    elif save_path.endswith(".mp4"):
        plotter.open_movie(save_path, framerate=int(actual_fps))
    else:
        raise ValueError("File path extension must be either '.mp4' or '.gif'")
    
    # 5. Core Recording Loop
    print("Writing frames...")
    for idx in frame_indices:
        # 1. Update internal mapping
        mesh_obj.update_signal(idx)

        # Loop over every supplementary mesh interface and compute its update
        for addon in addons:
            addon.update_signal(idx)

        # 2. PyVista tracks the active memory references automatically; force write the current pixel array
        plotter.write_frame()

    # Clean up file handlers and finish pipeline
    plotter.close()
    print(f"Video safely compiled to: {save_path}")