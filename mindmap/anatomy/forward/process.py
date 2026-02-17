"""
Module Name: forward/process.py
Description: Functions to manage boundary element models.

This module provides additional functions that MNE-python does not implement to handle
BEM models.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path

import numpy as np

import nibabel as nib

from ...geometry import surf_to_polydata, surf_correction

def correct_skull(
        subject     : str,
        offset      : float,
        fs_dir      : str | Path,
        save_arrows : bool = False,
) -> None:
    """
    Correct skull by displacing wrong vertices of inner skull back into the outer skull
    with an additional offset. 

    subject: str
        FreeSurfer subject name.

    offset: float
        Minimal distance between inner and outer skull.

    fs_dir: path-like
        FreeSurfer directory.

    save_arrows: bool
        If True then the vector field showing the displacement done.
    """

    # files
    fs_dir = Path(fs_dir)

    inner_fname = fs_dir/subject/"bem"/"inner_skull.surf"
    outer_fname = fs_dir/subject/"bem"/"outer_skull.surf"

    # load
    vertices, faces, metadata = nib.freesurfer.io.read_geometry(inner_fname, read_metadata=True)
    in_skull = surf_to_polydata(vertices, faces)

    vertices, faces = nib.freesurfer.io.read_geometry(outer_fname)
    out_skull = surf_to_polydata(vertices, faces)

    # apply correction
    outputs = surf_correction(
        in_skull,
        out_skull,
        offset = offset,
        return_arrows = save_arrows,
    )

    # rename old in_skull, save new in_skull as inner_skull.surf
    inner_fname.rename(fs_dir/subject/"bem"/"inner_skull_raw.surf")

    if save_arrows: 
        in_skull_fixed, arrows = outputs 
        arrows_fname = fs_dir/subject/"bem"/"inner_skull_changes.vtk"
        arrows.save(arrows_fname)

    else:
        in_skull_fixed = outputs 

    fname    = fs_dir/subject/"bem"/"inner_skull.surf"
    vertices = in_skull_fixed.points
    faces    = in_skull_fixed.faces

    # rewrite faces into right format
    idx   = np.arange(0, len(faces), step=4)
    faces = np.delete(faces, idx).reshape(-1, 3)

    nib.freesurfer.io.write_geometry(fname, vertices, faces, volume_info=metadata)