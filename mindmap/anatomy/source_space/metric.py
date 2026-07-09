import numpy as np
from scipy.spatial import cKDTree

import mne
 
def dipole_spacing_distribution(
    src_list: mne.SourceSpaces,
    distance_type: str = 'euclidean'
) -> list[np.ndarray]:
    """Compute the distribution of nearest-neighbor dipole spacings.
 
    Parameters
    ----------
    src_list : list of mne.SourceSpaces
        A list of SourceSpaces objects (each itself a list of dicts, one
        per hemisphere/structure). A single SourceSpaces object is also
        accepted and will be wrapped in a list automatically.
    distance_type : 'euclidean' | 'geodesic'
        'euclidean' uses straight-line 3D nearest-neighbor distance between
        active dipole locations (fast, always available).
        'geodesic' uses the along-surface distance matrix computed by MNE
        (requires the source space to have been created with
        add_dist=True or add_dist='patch', or to have had
        mne.add_source_space_distances run on it).
 
    Returns
    -------
    distributions : list of np.ndarray
        One 1-D array per hemisphere/structure encountered (in the order
        they appear across src_list), each containing that structure's
        nearest-neighbor spacing for every active dipole, in meters.
    """
    if distance_type not in ('euclidean', 'geodesic'):
        raise ValueError(
            f"distance_type must be 'euclidean' or 'geodesic', got {distance_type!r}"
        )
 
    # Allow passing a single SourceSpaces object instead of a list of them
    if hasattr(src_list, 'kind'):  # crude check: a single SourceSpaces (list-of-dicts) object
        src_list = [src_list]
 
    distributions = []
 
    for src in src_list:
        for hemi in src:
            vertno = hemi['vertno']
            if len(vertno) < 2:
                continue  # can't compute spacing with fewer than 2 active dipoles
 
            if distance_type == 'euclidean':
                rr = hemi['rr'][vertno]  # (n_dipoles, 3), in meters
                tree = cKDTree(rr)
                dists, _ = tree.query(rr, k=2)  # k=2: self + nearest neighbor
                nn_dist = dists[:, 1]
 
            else:  # geodesic
                dist = hemi.get('dist', None)
                if dist is None:
                    raise ValueError(
                        "No geodesic distance matrix found on this source "
                        "space. Recreate it with add_dist=True (or "
                        "add_dist='patch'), or run "
                        "mne.add_source_space_distances(src) first."
                    )
                # dist is a sparse (n_vertices, n_vertices) CSR matrix, indexed
                # by full vertex number, containing distances only for nearby
                # vertices (within dist_limit). For each active dipole, take
                # the minimum nonzero distance to its neighbors.
                nn_dist = np.empty(len(vertno))
                for i, v in enumerate(vertno):
                    row = dist[v]
                    data = row.data if hasattr(row, 'data') else np.asarray(row).ravel()
                    data = data[data > 0]
                    if len(data) == 0:
                        nn_dist[i] = np.nan
                    else:
                        nn_dist[i] = data.min()
                nn_dist = nn_dist[~np.isnan(nn_dist)]
                if len(nn_dist) == 0:
                    raise ValueError(
                        "Geodesic distance matrix exists but contains no "
                        "usable entries for active vertices. Check that "
                        "dist_limit was large enough when distances were "
                        "computed."
                    )
 
            distributions.append(nn_dist)
 
    if not distributions:
        raise ValueError("No source spaces with >= 2 active dipoles were found.")
 
    return distributions
 
 
def average_dipole_spacing(
    src_list: mne.SourceSpaces,
    distance_type: str = 'euclidean',
    per_space: bool = False
) -> float | list[float]:
    """Compute the average nearest-neighbor dipole spacing.
 
    This is a thin wrapper around `dipole_spacing_distribution` that
    reduces its output to a mean.
 
    Parameters
    ----------
    src_list : list of mne.SourceSpaces
        See `dipole_spacing_distribution`.
    distance_type : 'euclidean' | 'geodesic'
        See `dipole_spacing_distribution`.
    per_space : bool
        If True, also return the mean spacing for each individual
        hemisphere/structure encountered.
 
    Returns
    -------
    mean_spacing : float
        Overall average spacing pooled across all hemispheres/structures,
        in meters (multiply by 1000 for mm).
    per_space_means : list of float, optional
        Only returned if per_space=True.
    """
    distributions = dipole_spacing_distribution(src_list, distance_type=distance_type)
 
    overall_mean = float(np.concatenate(distributions).mean())
 
    if per_space:
        per_space_means = [float(d.mean()) for d in distributions]
        return overall_mean, per_space_means
    return overall_mean

