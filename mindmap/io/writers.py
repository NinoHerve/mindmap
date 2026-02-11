"""
Module Name: writers.py
Description: This module provides functions to write various data types and formats.

Contains functions to write data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path

import yaml
import numpy as np
import nibabel as nib
import pyvista as pv



def mkdir(path: Path):
    """Make parent folder if it does not exist."""
    if isinstance(path, str):
        path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir()


def write_eigendecomposition(fname, eigenvectors, eigenvalues=None):
    """Write eigendecomposition."""
    np.save(fname, eigenvectors)

    if eigenvalues is not None:
        new_name = fname.name.replace("eigvec", "eigval")
        np.save(fname.with_name(new_name), eigenvalues)


def write_surface(fname: Path|str, surf: pv.PolyData|tuple[np.ndarray], meta=None):
    """
    Write surface into FreeSurfer .surf format.

    Parameters
    ----------
    fname: path-like
        File name.

    surf: pv.PolyData|tuple[np.ndarray]
        Surface. In Pyvista format or a tuple containing vertices and faces.

    meta: ???
        Volume info.
    """
    if isinstance(surf, tuple|list):
        vertices, faces = surf

    elif isinstance(surf, pv.PolyData):
        vertices = surf.points
        faces = surf.faces

        # rewrite faces into right format
        idx = np.arange(0, len(faces), step=4)
        faces = np.delete(faces, idx).reshape(-1,3)

    else:
        raise ValueError("Unknown surface format to write.")

    nib.freesurfer.io.write_geometry(fname, vertices, faces, volume_info=meta)


def write_config(fname, config):
    """Write configuration file."""
    for key in config.keys():
        if isinstance(config[key], Path):
            config[key] = str(config[key])

    with open(fname, 'w', encoding="utf-8") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        

def write_eig(fname, eigvec, eigval):
    """
    Write eigendecomposition.
    """
    np.savez(fname, eigvec=eigvec, eigval=eigval)
