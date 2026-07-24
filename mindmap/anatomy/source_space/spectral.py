"""
Module Name: source_space/spectral.py
Description: Functions to apply spectral analysis on source-space geometry.

Mesh-geometry counterpart to `mindmap.analysis.connectivity.spectral`'s
`compute_cst` (connectome-Laplacian spectrum): computes a source basis from
cortical *shape* alone (Laplace-Beltrami eigenmodes of the source space's
own triangulated mesh), with no connectome/connectivity input at all --
see `../../../tutorials/analysis/connectomic/sabotage_connectome_notes.md`
("Beltrami-operator (mesh-LBO) implementation notes") for the full design
rationale (why the coarse source-space mesh is used directly rather than
the dense FreeSurfer surface, why the basis must be re-orthonormalized,
why hemispheres are kept independent, and the medial-wall open-boundary
handling below).

Author: Nino Herve
Created: 2026-07-16
Last Modified: 2026-07-16
License: MIT
"""

import warnings

import numpy as np

import mne

from mindmap.geometry.processing import compute_laplace_beltrami_basis


def _cortex_triangulation(src_hemi: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a clean, cortex-only triangulation from one hemisphere of an
    mne.SourceSpaces object.

    `src_hemi["use_tris"]` is the triangulation of a *pre-cortex-mask*,
    closed decimated surface -- it still references non-cortex vertices
    (e.g. medial wall) that were excluded from `vertno` by whatever masking
    built this source space (e.g. `mm.utils.mask_sources`). Keeping only
    faces whose 3 vertices are all in `vertno` gives an open
    manifold-with-boundary (verified empirically on sub-simf10: single
    connected component, a single ~89-edge boundary loop around the medial
    wall, zero non-manifold edges) -- the correct mesh for a per-hemisphere
    Laplace-Beltrami computation with natural (Neumann) boundary conditions.

    One consequence of this strict "all 3 vertices must be cortex" rule,
    also verified empirically: a small number of `vertno` dipoles right at
    the boundary can end up with zero incident kept faces (1 of 3,748 for
    sub-simf10's left hemisphere, 0 for the right) -- these are not silently
    dropped: a `RuntimeWarning` naming the affected `vertno` indices is
    raised directly from here (not returned -- see `compute_beltrami_harmonics`
    for how they end up handled, as all-zero rows in the eigenvector basis).

    Parameters
    ----------
    src_hemi: dict
        One hemisphere's entry of an mne.SourceSpaces (e.g. `source_space[0]`).

    Returns
    -------
    verts: np.ndarray of shape (n_referenced, 3)
        Vertex positions (meters, mne.SourceSpaces convention) of the
        cortex-only mesh.

    faces: np.ndarray of shape (n_faces, 3)
        Triangle indices into `verts` (compact, 0-based).

    referenced: np.ndarray of shape (n_referenced,)
        Indices into `src_hemi["rr"]`/`src_hemi["vertno"]` of the vertices
        kept in `verts`/`faces`, ascending order (always a subset of
        `vertno` -- any `vertno` entries missing from this are the ones
        the `RuntimeWarning` above reports, not returned separately).
    """
    rr = src_hemi["rr"]
    use_tris = src_hemi["use_tris"]
    vertno = src_hemi["vertno"]

    is_used = np.zeros(len(rr), dtype=bool)
    is_used[vertno] = True

    keep = is_used[use_tris].all(axis=1)
    faces_cortex = use_tris[keep]

    referenced = np.unique(faces_cortex)

    missing = np.setdiff1d(vertno, referenced)
    if len(missing):
        warnings.warn(
            f"_cortex_triangulation: {len(missing)} of {len(vertno)} dipoles have no "
            f"incident all-cortex face and will get zero representation in the "
            f"Beltrami harmonic basis (vertno indices: {missing.tolist()})",
            RuntimeWarning,
        )

    remap = -np.ones(len(rr), dtype=int)
    remap[referenced] = np.arange(len(referenced))

    faces = remap[faces_cortex]
    verts = rr[referenced]

    return verts, faces, referenced


def compute_beltrami_harmonics(source_space: mne.SourceSpaces) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the source space's mesh-geometry spectral basis: Laplace-Beltrami
    eigenmodes of the source space's own (cortex-only, per-hemisphere)
    triangulated mesh.

    Drop-in-compatible with `mindmap.analysis.connectivity.spectral.compute_cst`'s
    `(eigvec, eigval)` return convention (same dipole ordering: hemisphere 0
    first, then hemisphere 1, matching `get_dipole_positions`/connectome
    ordering elsewhere in this package), for use as
    `wCSET(eigenvectors=eigvec, eigenvalues=eigval, ...)` -- but built purely
    from cortical shape, with no connectome/connectivity input at all.

    Each hemisphere is solved as its own, independent Laplace-Beltrami
    problem (see `_cortex_triangulation`) via
    `mindmap.geometry.processing.compute_laplace_beltrami_basis`, using the
    *full* per-hemisphere basis (not a truncated handful of low modes) --
    matching `compute_cst`'s convention of always returning the complete
    spectrum, so that both bases can be discounted by the same tunable
    `exp(-β·λ)`-style weighting (`compute_harmonic_prior`) rather than
    needing a hard, basis-size-mismatched truncation on one side. See the
    notes file referenced in this module's docstring for why hemispheres
    are NOT connected to each other (no cross-hemisphere pairing needed for
    this use case, and connecting them would reintroduce a
    connectivity-like assumption into what's meant to be a pure-geometry
    basis).

    Each hemisphere's own smallest eigenvalue is clamped to exactly 0 (the
    Neumann/natural-boundary constant mode -- theoretically exact, but
    `eigh` can return a tiny nonzero floating-point residual instead, same
    reasoning as `compute_cst`'s `eigval[0] = 0`). Because the two
    hemispheres are disconnected components, there are two such modes in
    the combined spectrum, not one.

    A `vertno` dipole with no incident all-cortex face (see
    `_cortex_triangulation`'s `missing`) is left as an exact all-zero row
    (that dipole gets no representation in the mesh-harmonic basis at all)
    rather than patched via e.g. nearest-neighbor duplication. Checked
    empirically on sub-simf10 (1 such dipole, of 7,465): duplicating a
    neighbor's row is NOT a negligible approximation -- it's a copy of an
    already-included direction, so it breaks the basis's orthonormality by
    a measurable, exactly rank-1 amount (verified: up to 13% on some
    diagonal entries of `UᵀU`, entirely explained by and removed along with
    that one row). An all-zero row costs nothing in exchange: it keeps
    `UᵀU = I` exact (a zero row contributes nothing to the Gram matrix) at
    the cost of that one dipole always reading out as 0 from this basis
    specifically -- and also makes that hemisphere's basis one dimension
    short of a perfectly complete (n_used x n_used) basis, same as the
    duplication approach would have.

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space (space-freesurfer), built with cortex masking (e.g.
        `mm.utils.mask_sources`) -- i.e. the same source space used
        everywhere else in this pipeline (dipole positions, connectome).

    Returns
    -------
    eigvec: np.ndarray of shape (n, n_total_modes)
        Euclidean-orthonormal eigenvectors (columns) of the per-hemisphere
        Laplace-Beltrami operators, block-diagonal across hemispheres,
        globally sorted by ascending eigenvalue (n_total_modes <= n,
        short by exactly the number of `missing` dipoles patched above,
        summed across hemispheres).

    eigval: np.ndarray of shape (n_total_modes,)
        Corresponding eigenvalues, ascending order.
    """
    eigvec_blocks = []
    eigval_blocks = []
    offset = 0

    for src_hemi in source_space:
        vertno = src_hemi["vertno"]
        n_hemi = len(vertno)

        verts, faces, referenced = _cortex_triangulation(src_hemi)
        eigvec_ref, eigval = compute_laplace_beltrami_basis(verts * 1000, faces)   # [m] -> [mm]
        eigval[0] = 0

        # dipoles not covered by any all-cortex face (see docstring) are
        # left as all-zero rows -- eigvec_full is zero-initialized, so no
        # action needed beyond scattering the referenced rows in.
        idx_in_vertno = np.searchsorted(vertno, referenced)
        eigvec_full = np.zeros((n_hemi, eigvec_ref.shape[1]))
        eigvec_full[idx_in_vertno] = eigvec_ref

        eigvec_blocks.append((offset, n_hemi, eigvec_full))
        eigval_blocks.append(eigval)
        offset += n_hemi

    n_total = offset
    eigval = np.concatenate(eigval_blocks)
    eigvec = np.zeros((n_total, len(eigval)))

    col_offset = 0
    for row_offset, n_hemi, eigvec_full in eigvec_blocks:
        n_modes = eigvec_full.shape[1]
        eigvec[row_offset:row_offset+n_hemi, col_offset:col_offset+n_modes] = eigvec_full
        col_offset += n_modes

    order = np.argsort(eigval)
    return eigvec[:, order], eigval[order]
