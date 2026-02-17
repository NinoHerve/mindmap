"""
Module Name: viz/volume.py
Description: Functions to plot volume data.

===============================================================================
VOLUME-SPACE VISUALIZATION (volume.py)
===============================================================================
Targets: np.ndarray, nib.Nifti1Image, mm.anatomy.Parcellation
Backends: matplotlib (heatmaps), freeview
===============================================================================

Author: Nino Herve
Created: 2025-10-28
Last Modified: 2025-10-28
License: MIT
"""

from pathlib import Path
import subprocess

import numpy as np

import matplotlib.pyplot as plt

import nibabel as nib


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


def plot_segmentation(
        t1_file  : str | Path,
        seg_file : str | Path,
        lut_file : str | Path = None
        ):
    """
    Opens FreeView to plot structural MRI with segmentation on top.
    """
    cmd = ["freeview", "-v", str(t1_file)]

    if lut_file is None:
        cmd += [str(seg_file)]
    else:
        cmd += [str(f"{seg_file}:lut={lut_file}")]

    subprocess.run(cmd, check=True)


def plot_bem(
    fs_dir  : str | Path,
    subject : str,
):
    """
    Plot boundary element surfaces.

    Parameters
    ----------
    fs_dir: str | Path
        Path to FreeSurfer directory.
    
    subject: str
        Subject id.
    """
    fs_dir = Path(fs_dir)
    sub_id = f"sub-{subject}"

    mri       = fs_dir/sub_id/"mri"/"T1.mgz"
    white_lh  = fs_dir/sub_id/"surf"/"lh.white"
    white_rh  = fs_dir/sub_id/"surf"/"rh.white"
    in_skull  = fs_dir/sub_id/"bem"/"inner_skull.surf"
    out_skull = fs_dir/sub_id/"bem"/"outer_skull.surf"
    out_skin  = fs_dir/sub_id/"bem"/"outer_skin.surf"

    cmd = [
        "freeview",
        "-v", str(mri),
        "-f",
        str(white_lh) + ":edgecolor=white",
        str(white_rh) + ":edgecolor=white",
        str(in_skull) + ":edgecolor=red",
        str(out_skull) + ":edgecolor=green",
        str(out_skin) + ":edgecolor=blue",
    ]

    subprocess.run(cmd, check=True)
