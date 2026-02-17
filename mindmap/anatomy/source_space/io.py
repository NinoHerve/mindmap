"""
Module Name: source_space/io.py
Description: Functions to extract source space data.

This module provides functions to extract hard to get data from mne.SourceSpaces.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import scipy as sp

import mne

import pyvista as pv

from ...geometry.io import surf_to_polydata


def get_dipole_positions(source_space:mne.SourceSpaces) -> np.ndarray:
    "Get source positions."
    positions = []

    for src in source_space:
        positions.append(src["rr"][src["vertno"]])

    return np.concatenate(positions, axis=0)


def get_dipole_directions(source_space):
    "Get source directions."
    directions = []

    for src in source_space:
        directions.append(src["nn"][src["vertno"]])

    return np.concatenate(directions, axis=0)


def get_geodist(source_space:mne.SourceSpaces) -> sp.sparse.csr_array:
    """
    Get matrix showing geodesic distances between sources. Left and right hemispheres are seperated 
    surfaces, hence distances between a left and right source is infinite.
    """

    geodist = []
    for src in source_space:
        vertno = src["vertno"]
        geodist.append(src["dist"][vertno][:, vertno])

    return sp.sparse.block_diag(geodist, format="csr")



def get_ndipole(source_space: mne.SourceSpaces) -> int:
    """Get number of dipoles in source space."""
    return sum(src["nuse"] for src in source_space)




def mne_to_polydata(
        source_space : mne.SourceSpaces,
        signal       : np.ndarray = None,
        n_points     : int = 2
        ) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Convert MNE object to pyvista polydata.

    Parameters
    ----------
    src: mne.SourceSpaces
        Source space.

    scalars: np.ndarray 
        Dipole values.

    n_points: int
        Number of points to consider for dipole->vertex interpolation.

    Returns
    -------
    polydata: list(pv.PolyData)
        Left and right hedef resample_mesh(volume, remesh=True, subdivision=4):
    """
    # dipoles
    dipoles_l = pv.PolyData(source_space[0]["rr"][source_space[0]["vertno"]])
    dipoles_r = pv.PolyData(source_space[1]["rr"][source_space[1]["vertno"]])

    # hemispheres
    vertices, faces = source_space[0]["rr"], source_space[0]["tris"]
    hemi_l = surf_to_polydata(vertices, faces)

    vertices, faces = source_space[1]["rr"], source_space[1]["tris"]
    hemi_r = surf_to_polydata(vertices, faces)

    if signal is not None:
        split = source_space[0]["nuse"]
        dipoles_l["signal"] = signal[:split]
        dipoles_r["signal"] = signal[split:]

        # interpolate dipoles -> hemisphere vertices
        hemi_l = hemi_l.interpolate(dipoles_l, strategy="closest_point", n_points=n_points)
        hemi_r = hemi_r.interpolate(dipoles_r, strategy="closest_point", n_points=n_points)

    return hemi_l, hemi_r

