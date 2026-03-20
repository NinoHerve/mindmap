"""
Module Name: anatomy/forward/io.py
Description: Functions to read forward related objects.

This module provides functions to manage forward related objects not implemented in MNE-Python.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path

import numpy as np
import pyvista as pv

import nibabel as nib
import mne

from mindmap.core.io import read_config

def read_bem_labels(config: str | Path) -> dict[str, list[int]]:
    """
    Parses YAML config and resolves inheritance to return
    a dictionary of {layer_name: [list_of_ids]}
    """
    config = read_config(config)

    label_map = config.get("label_map", {})
    layers_def = config.get("layers", {})

    result = {}

    def resolve_item(item: str) -> list[int]:
        # Check if the item is a category in label_map
        if item in label_map:
            return label_map[item]

        # Check if the item refers to another layer (Inheritance)
        if item in layers_def:
            indices = []
            for sub_item in layers_def[item]:
                indices.extend(resolve_item(sub_item))
            return indices

        # Handle typos or missing keys
        print(f"Warning: '{item}' not found in label_map or layers.")
        return []

    for layer_name in layers_def:
        # Resolve all items and ensure unique indices (sorted for cleanliness)
        raw_indices = resolve_item(layer_name)
        result[layer_name] = sorted(list(set(raw_indices)))

    return result


def make_volume_info(t1_mgz_path):
    """
    Extracts volume metadata from a FreeSurfer T1.mgz file
    for use in mne.write_surface.
    """
    # Load the conformed volume (usually 256x256x256, 1mm iso)
    img = nib.load(t1_mgz_path)

    # image properties
    volume = np.array(img.shape)
    voxelsize = np.array(img.header.get_zooms())
    affine = img.affine

    # Decompose affine into xras, yras, zras and cras
    xras = affine[:3, 0]
    yras = affine[:3, 1]
    zras = affine[:3, 2]

    # "cras" is the translation component of the affine
    # FreeSurfer centers the brain around [0,0,0] but it does it himself.
    # Hence, the centering should be omitted in the vol_info object
    centering = np.stack([xras*voxelsize[0], yras*voxelsize[1], zras*voxelsize[2]], axis=1) @ (volume/2.0)
    cras = affine[:3, 3] + centering

    vol_info = {
        "head": np.array([2, 0, 20], dtype=np.int32),   # Generic header tag
        "valid": "1  # volume info valid",              # indicates the header is valid
        "filename": str(t1_mgz_path),
        "volume": volume,              # e.g., (256, 256, 256)
        "voxelsize": voxelsize,  # e.g., (1.0, 1.0, 1.0)
        "xras": xras,
        "yras": yras,
        "zras": zras,
        "cras": cras,
    }

    return vol_info


def save_pv_as_freesurfer(
    path: str | Path,
    mesh: pv.PolyData,
    volume_info: dict,
    overwrite=False,
):
    """Save pyvista surface as FreeSurfer surface."""
    # get coordinates (N,3)
    coords = mesh.points

    # reshape faces: PyVista faces are [3, v1, v2, v3, 3, v4, v5, v6, ...]
    # this assumes meshes are fully triangulated
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    # write using MNE
    mne.write_surface(path, coords, faces, volume_info=volume_info, overwrite=overwrite)


def save_bem_as_freesurfer(
    folder: str | Path,
    bem: dict[str, pv.PolyData],
    volume_info: dict,
    overwrite: str = False,
):
    """
    Save BEM layers as FreeSurfer surfaces.

    Parameters
    ----------
    folder: str | Path
        Folder where to save bem layers.

    bem: dict[str, pv.PolyData]
        PolyData meshes for "skin", "inner skull" and "outer skull".

    volume_info: dict
        Volume info. Necessary for correct implementation of FreeSurfer
        and MNE tools.

    Saves
    -----
    outer_skin.surf: Outer skin surface.
    outer_skull.surf: Outer skull surface.
    inner_skull.surf: Inner skull surface.
    """
    keys = ["skin", "outer skull", "inner skull"]
    files = ["outer_skin.surf", "outer_skull.surf", "inner_skull.surf"]
    folder = Path(folder)

    for k, f in zip(keys, files):
        save_pv_as_freesurfer(folder/f, bem[k], volume_info, overwrite)
