"""
Module Name: analysis/connectivity/process.py
Description: Functions to process connectomes.

This module provides functions to handle connectomes such as
building them from tractograms or from a coarse structural-connectivity
atlas (e.g. `connectome_from_atlas`, upscaling a region-level atlas to
dipole/vertex resolution as a surrogate structural prior for subjects
without their own diffusion MRI).

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT

Background: upscaling a region-level atlas to dipole/vertex resolution
------------------------------------------------------------------------
No published, validated method upscales an already-aggregated empirical
region x region atlas (e.g. the ProbConn / multi-scale structural
connectivity atlas of Aleman-Gomez et al. 2020, Sci Data 7:296) to
dipole/vertex resolution (10^3-10^5 nodes). The field's default is to
build dense connectomes directly from tractography instead. Closest
precedents and useful building blocks:

- Rue-Queralt, Hagmann, Aleman-Gomez et al. 2024, "Connectome Spectrum
  Electromagnetic Tomography (CSET)", Hum Brain Mapp 45(5):e26638.
  Builds ~8000-node vertex-level connectomes from individual DWI; for
  subjects without dMRI, substitutes a group-consensus vertex-level
  connectome as a fixed graph-Laplacian prior for M/EEG source
  reconstruction. Closest precedent to this module's use case (same
  atlas lineage), though it keeps a separately-built dense connectome
  rather than upscaling a coarse one.
- Ercsey-Ravasz et al. 2013, Neuron 80(1):184-197. Exponential distance
  rule (EDR): connection probability decays ~exp(-lambda*d) with
  distance. Basis for the `sigma`-controlled stochastic edge sampling
  in `connectome_from_atlas`.
- Vertes et al. 2012, PNAS 109(15):5868; Betzel et al. 2016, NeuroImage
  124:1054. Generative models combining a wiring-cost/distance term
  with a topological-homophily term to reproduce realistic small-world
  /modular topology -- a more principled analogue of a distance +
  atlas-weight rule.
- Faskowitz et al. 2018, Sci Rep 8:12997. Weighted stochastic block
  model of the human connectome -- the natural framework for "sample
  fine edges from region-level statistics", though not applied there
  to expand an empirical atlas.
- Cammoun et al. 2012, J Neurosci Methods 203(2):386; Hagmann et al.
  2008, PLoS Biol 6(7):e159. Origin of the multi-scale Lausanne
  parcellation scheme (source of the "multiconn2022" scales used
  here); tops out at ~1000 regions/hemisphere, i.e. the field's
  practical resolution ceiling before vertex-level extension.

Known tractography biases inherited by any dMRI-derived atlas (relevant
when treating the atlas as ground truth to replicate at high
resolution rather than as an imperfect estimate):
- Gyral bias: streamlines over-terminate at gyral crowns vs sulcal
  banks (Schilling et al. 2018, PMC5807146; mitigation in Schilling et
  al. 2019, NeuroImage).
- Under-detection of long-range/interhemispheric connections, due to
  superficial U-fiber systems and crossing-fiber ambiguity (Reveley et
  al. 2015, PNAS 112:E2820).

Target statistics for validating an upscaled/surrogate connectome (see
`mindmap.viz.graph` for degree/weight-distribution plots, and
`mindmap.core.metric` / `analysis/connectivity/metric.py` for the
small-world, rich-club, modularity, EDR-fit, hemispheric-symmetry and
block-conservation implementations):
- Density: ~10-20% is the typical benchmark at Lausanne scale (~1000
  regions; Hagmann et al. 2008); expected to decrease at finer
  resolution, but no established vertex-level target exists.
- Weight/degree distribution: lognormal weights with exponential
  distance-decay fit better than a power law/scale-free model (Shastin
  et al. 2021, eNeuro 8(1); arXiv:2407.17220).
- Small-worldness: sigma = (C/Cr)/(L/Lr) > 1 against a degree-preserving
  null; absolute sigma is not comparable across studies/scales.
- Rich-club: known hubs (precuneus, superior frontal/parietal cortex,
  hippocampus, putamen, thalamus at Lausanne scale) should be more
  interconnected than a degree-matched null (van den Heuvel & Sporns
  2011, J Neurosci 31:15775).
- Modularity: ~4 modules (bilateral anterior/posterior) is the typical
  qualitative signature at this scale.
- Hemispheric symmetry: left/right connectivity-profile correlation;
  dMRI is known to underestimate homotopic interhemispheric
  connections, so a surrogate should not be tuned to reproduce that
  bias faithfully.
- Block conservation: block-averaging the upscaled connectome back down
  to region resolution should approximately reproduce the original
  atlas matrix.
"""

from collections import defaultdict

import numpy as np
import scipy as sp

import nibabel as nib
import mne

from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.tractogram.processing import tract_endpoints, tract_lengths
from mindmap.anatomy.utils import align_diffusion2subject
from mindmap.anatomy.source_space.io import get_dipole_positions
from mindmap.analysis.connectivity.utils import get_transformed_dipoles, get_dipole_region_labels
from mindmap.analysis.connectivity.smooth import gaussian_kernel, smooth_connectome
from mindmap.core.process import normalize_data


def connectome_from_tracts(
        tracts    : nib.streamlines.TrkFile,
        regions   : Parcellation | mne.SourceSpaces,
        t12diff   : np.ndarray,
        ras2tkras : np.ndarray = None,
        lengths   : bool = False,
        radius    : float = None,
        fwhm      : float = None,
        epsilon   : float = 0.01,
        ) -> np.ndarray:
    """
    Compute a connectome from tracts.

    Parameters
    ----------
    tracts: nib.Streamlines.TrkFile
        Tractogram.

    regions: Parcellation | mne.SourceSpaces
        Regions to aggregate tracts onto (brain regions or dipoles).

    t12diff: np.ndarray (4,4)
        Individual space to diffusion MRI space affine transformation.

    ras2tkras: np.ndarray (4,4)
        FreeSurfer conform space to individual space affine transform (required when using
        source space only).

    length: bool
        Whether to weigth according to lengths (length is True) or number of fibers (length
        is False). At source-space resolution, these two use different
        constructions (see `_connect_by_radius` and `_connect_by_smoothing`):
        length is not a mass to conserve like a streamline count is, so it
        is not treated like one.

    radius: float
        Required when `regions` is a source space. Maximum distance in [mm]
        at which a streamline endpoint connects to a dipole. When
        `lengths=True`, every dipole within this radius of an endpoint
        connects (soft assignment, matching the positional uncertainty of
        a streamline endpoint); when `lengths=False`, only the single
        nearest dipole within this radius does (streamlines with no dipole
        in reach are dropped either way). Not used when `regions` is a
        Parcellation.

    fwhm: float
        Only used when `lengths=False` and `regions` is a source space:
        full width at half maximum, in [mm], of the Gaussian kernel used to
        smooth the raw nearest-dipole streamline-count connectome, following
        Mansour et al. (2021, NeuroImage, Supplementary S.2.1). Defaults to
        `radius` if not given.

    epsilon: float
        Only used when `lengths=False` and `regions` is a source space:
        truncation parameter for the Gaussian smoothing kernel, see
        `mindmap.analysis.connectivity.smooth.gaussian_kernel`.

    Returns
    -------
    connectome: np.ndarray
        Adjacency matrix indicating neural pathways between brain regions or dipoles.
    """

    # Label tract endpoints to regions
    starts, ends = tract_endpoints(tracts)
    starts = align_diffusion2subject(starts, t12diff)
    ends   = align_diffusion2subject(ends, t12diff)

    if isinstance(regions, Parcellation):
        start_labels, end_labels = _label_by_region(starts, ends, regions)
        n = regions.n_regions

        # build connectivity matrix
        sc = np.zeros((n, n))

        # if lengths is True apply fiber lengths as weights
        if lengths:
            lengths_arr = tract_lengths(tracts)
            sc_dict = defaultdict(list)     # dict to accumulate weights for each pairs

            for start, end, weight in zip(start_labels, end_labels, lengths_arr):
                # skip self-loops
                if start == end:
                    continue

                # ensure directed pairs are treated the same
                key = tuple(sorted((start, end)))
                sc_dict[key].append(weight)

            # apply average lengths as weight
            for (start, end), weights in sc_dict.items():
                avg_weight = np.mean(weights)
                sc[start, end] = avg_weight
                sc[end, start] = avg_weight

        else:
            for start, end in zip(start_labels, end_labels):
                # skip self loops
                if start == end:
                    continue

                # apply number of tracts as weights
                # ensure undirectedness
                sc[start, end] += 1
                sc[end, start] += 1

        return sc

    # source-space (dipole-level)
    assert ras2tkras is not None, "ras2tkras is required when using source spaces in order to map dipole positions to the T1-ras space."
    assert radius is not None, "radius is required when using source spaces, to define how far a streamline endpoint may reach to connect to dipoles."

    if lengths:
        lengths_arr = tract_lengths(tracts)
        return _connect_by_radius(starts, ends, regions, ras2tkras, radius, lengths_arr)

    return _connect_by_smoothing(starts, ends, regions, ras2tkras, radius, fwhm or radius, epsilon)


def _label_by_region(
        starts       : np.ndarray,
        ends         : np.ndarray,
        parcellation : Parcellation,
        ) -> tuple[list, list]:
    """
    Label start and end points by brain regions.

    Parameters
    ----------
    starts: np.ndarray (n,3)
        Starting points of tracts. In T1-ras space in [mm].

    ends: np.ndarray (n,3)
        Ending points of tracts. In T1-ras space in [mm].

    parcellation: Parcellation
        Parcellation in T1-ras space in [mm].

    Returns
    -------
    start_labels, end_labels: (np.ndarray (n,3), np.ndarray (n,3))
        Associated brain region id for each endpoint.
    """
    start_labels = parcellation.label_points(starts, return_id=True)
    end_labels   = parcellation.label_points(ends, return_id=True)

    unlabeled = (start_labels==0) | (end_labels==0)
    assert np.all(~unlabeled), "Some tracts can't be labeled to any brain region. Make sure to prune by region properly before, and check if the correct transformation matrices were provided."

    return start_labels, end_labels


def _connect_by_radius(
        starts       : np.ndarray,
        ends         : np.ndarray,
        source_space : mne.SourceSpaces,
        ras2tkras    : np.ndarray,
        radius       : float,
        lengths      : np.ndarray,
        ) -> np.ndarray:
    """
    Build a dipole-level length connectome by connecting each streamline
    endpoint to every dipole within `radius`, instead of snapping to a
    single nearest dipole -- matching the true positional uncertainty of a
    streamline endpoint rather than an arbitrary nearest-neighbor snap.

    Each edge's weight is the plain (unweighted) mean of the raw lengths of
    the streamlines that connect it. This is deliberately not a mass-
    weighted average: a streamline's length is a property of that
    streamline, not a mass to conserve, so it should not shrink just
    because its endpoint happened to have more candidate dipoles within
    reach than another streamline's. (Streamline count, in
    `_connect_by_smoothing`, is a mass, and is treated accordingly.)

    Parameters
    ----------
    starts, ends: np.ndarray of shape (n, 3)
        Streamline endpoints, in T1-ras space in [mm].

    source_space: mne.SourceSpaces
        Source space. Contains dipole positions in freesurfer space (tkras) in [m].

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform matrix from FreeSurfer to subject RAS.

    radius: float
        Maximum distance in [mm] at which a streamline endpoint connects to a dipole.

    lengths: np.ndarray of shape (n,)
        Per-streamline lengths.

    Returns
    -------
    sc: np.ndarray of shape (n_dipoles, n_dipoles)
        Dipole-level length connectome. Streamlines with no dipole within
        `radius` of one of their endpoints are dropped (contribute nothing).
    """
    dipoles = get_transformed_dipoles(source_space, ras2tkras)
    n_dipoles = dipoles.shape[0]

    tree = sp.spatial.cKDTree(dipoles)
    start_neighbors = tree.query_ball_point(starts, radius)
    end_neighbors = tree.query_ball_point(ends, radius)

    rows, cols, length_vals = [], [], []

    for c, (si, ei) in enumerate(zip(start_neighbors, end_neighbors)):
        if len(si) == 0 or len(ei) == 0:
            continue    # streamline has no dipole in reach of one endpoint; drop it

        si = np.asarray(si)
        ei = np.asarray(ei)

        ii, jj = np.meshgrid(si, ei, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()

        # this streamline's full (unweighted) length, assigned to every
        # candidate edge it could plausibly support
        lv = np.full(ii.shape, lengths[c])

        # symmetric accumulation (undirected graph)
        rows.append(ii); cols.append(jj); length_vals.append(lv)
        rows.append(jj); cols.append(ii); length_vals.append(lv)

    if not rows:
        return np.zeros((n_dipoles, n_dipoles))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    length_vals = np.concatenate(length_vals)

    # coo_matrix sums duplicate (row, col) entries on conversion, which is
    # exactly the accumulation this needs across streamlines/endpoint pairs;
    # dividing the summed lengths by the summed count of contributors gives
    # a plain mean
    total_length = sp.sparse.coo_matrix((length_vals, (rows, cols)), shape=(n_dipoles, n_dipoles)).toarray()
    n_contrib = sp.sparse.coo_matrix((np.ones_like(length_vals), (rows, cols)), shape=(n_dipoles, n_dipoles)).toarray()

    sc = np.divide(total_length, n_contrib, out=np.zeros_like(total_length), where=n_contrib > 0)
    np.fill_diagonal(sc, 0)

    return sc


def _connect_by_smoothing(
        starts       : np.ndarray,
        ends         : np.ndarray,
        source_space : mne.SourceSpaces,
        ras2tkras    : np.ndarray,
        radius       : float,
        fwhm         : float,
        epsilon      : float,
        ) -> np.ndarray:
    """
    Build a dipole-level streamline-count connectome following Mansour et
    al. (2021, NeuroImage, Supplementary S.2.1): each streamline endpoint is
    first snapped to its single nearest dipole (the streamline is dropped
    if either endpoint has none within `radius`), giving a raw,
    high-resolution streamline-count connectome; that raw connectome is
    then smoothed with a Gaussian kernel over cortical-surface geodesic
    distance (`fwhm`), spreading each edge's count to neighboring dipoles
    so that sparse dipole-level sampling doesn't understate connectivity.
    Total streamline mass is conserved by construction (`smooth_connectome`
    spreads existing mass across neighbors; it does not create any).

    This is deliberately a different construction from `_connect_by_radius`
    (used when `lengths=True`): a streamline count is a mass, so spreading
    it via a kernel and reconstituting the connectome from smoothed
    incidence matrices is the right model; a length is not a mass, and is
    instead handled with a plain mean over contributing streamlines.

    Parameters
    ----------
    starts, ends: np.ndarray of shape (n, 3)
        Streamline endpoints, in T1-ras space in [mm].

    source_space: mne.SourceSpaces
        Source space. Contains dipole positions in freesurfer space (tkras)
        in [m], and per-hemisphere cortical-surface geodesic distances used
        to build the smoothing kernel.

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform matrix from FreeSurfer to subject RAS.

    radius: float
        Maximum distance in [mm] between a streamline endpoint and its
        nearest dipole, for the streamline to be kept.

    fwhm: float
        Full width at half maximum, in [mm], of the Gaussian smoothing kernel.

    epsilon: float
        Smoothing-kernel truncation parameter, see
        `mindmap.analysis.connectivity.smooth.gaussian_kernel`.

    Returns
    -------
    sc: np.ndarray of shape (n_dipoles, n_dipoles)
        Smoothed dipole-level streamline-count connectome.
    """
    dipoles = get_transformed_dipoles(source_space, ras2tkras)
    n_dipoles = dipoles.shape[0]

    tree = sp.spatial.cKDTree(dipoles)
    start_dist, start_nn = tree.query(starts)
    end_dist, end_nn = tree.query(ends)

    # keep streamlines with a dipole in reach of both endpoints, dropping
    # self-loops (both endpoints snapping to the same dipole) up front so
    # they don't contaminate the smoothing step below
    keep = (start_dist <= radius) & (end_dist <= radius) & (start_nn != end_nn)
    start_nn, end_nn = start_nn[keep], end_nn[keep]

    if start_nn.size == 0:
        return np.zeros((n_dipoles, n_dipoles))

    # canonicalize edge direction (store each pair once, as lo->hi) before
    # counting, since a streamline's start/end is an arbitrary tracking
    # artifact, not a meaningful direction: two streamlines between the same
    # two dipoles traced in opposite directions must add up to a count of 2,
    # not populate raw[i,j] and raw[j,i] separately as if they were
    # different edges. Symmetrize afterwards -- `smooth_connectome` expects
    # a symmetric connectome, like every other connectome in this module.
    lo = np.minimum(start_nn, end_nn)
    hi = np.maximum(start_nn, end_nn)
    raw = sp.sparse.coo_matrix((np.ones(lo.size), (lo, hi)), shape=(n_dipoles, n_dipoles)).toarray()
    raw = raw + raw.T

    kernel = _geodesic_gaussian_kernel(source_space, fwhm, epsilon)

    return smooth_connectome(raw, kernel)


def _geodesic_gaussian_kernel(
        source_space : mne.SourceSpaces,
        fwhm         : float,
        epsilon      : float,
        ) -> sp.sparse.csr_matrix:
    """
    Block-diagonal Gaussian smoothing kernel across both hemispheres, built
    from cortical-surface geodesic distance (`source_space[hemi]["dist"]`).

    Cross-hemisphere smoothing is not attempted: surface geodesic distance
    isn't defined between two separate hemisphere meshes -- the same
    limitation `connectome_from_geometry` works around by handling
    interhemispheric connections separately (there, via a direct Euclidean
    distance cutoff instead of a kernel).
    """
    blocks = []

    for hemi in [0, 1]:
        vertices = source_space[hemi]["vertno"]
        geodist = source_space[hemi]["dist"][vertices][:, vertices].tocsr() * 1000    # [mm]
        blocks.append(gaussian_kernel(geodist, fwhm=fwhm, epsilon=epsilon, dim=2))

    return sp.sparse.block_diag(blocks, format="csr")


def connectome_from_geometry(
        source_space    : mne.SourceSpaces,
        maximum_dist    : float = 10,
        maximum_neigh   : int = None,
        inter_hemi_dist : float = 0,
        weights         : bool = True
        ) -> np.ndarray:
    """
    Build adjacency matrix from brain geometry.

    Parameters
    ----------
    source_space: mne.SourceSpace
        Sources.

    maximum_dist: float
        Maximal distance in [mm] at which to connect neighbors.

    maximum_neigh: int
        Maximum number of neighbors. This will select the N closest neighbors only. But a 
        node can still end up with more connections if it is the closest neighbor of other
        nodes.
        
    inter_hemi_dist: float
        Maximal distance in [mm] at which to connect inter hemispheric nodes.

    weights: bool
        Whether to put weights as geodesic distances (in [mm]) or binarize.
    
    Returns
    -------
    geometry: sp.sparce.csc_matrix
        Brain geometry at source level. 
    """
    hemispheres = []

    for hemi in [0,1]:  # 0 for left hemisphere and 1 for right hemisphere

        # get geodesic distance matrix of used nodes
        vertices = source_space[hemi]["vertno"]
        dist     = source_space[hemi]["dist"][vertices][:,vertices].todense() * 1000     # in [mm]

        # threshold according to dmax
        dist[dist > maximum_dist] = np.inf

        # choose only k neerest connections
        if maximum_neigh is not None:
            # indices of k closest nodes
            knn_indices = np.argpartition(dist, maximum_neigh, axis=1)[:,:maximum_neigh]

            # create symmetric mask of nearest neighbors
            mask = np.zeros_like(dist, dtype=bool)
            rows = np.arange(dist.shape[0])[:,None]
            mask[rows, knn_indices] = True

            sym_mask = np.logical_or(mask, mask.T)

            # apply mask
            dist = np.where(sym_mask, dist, np.inf)

        hemispheres.append(dist)

    # connect hemispheres
    n = source_space[0]["nuse"]

    # add inter hemispheric connections
    pos    = get_dipole_positions(source_space) * 1000  # in [mm]
    lh_src = pos[:n]
    rh_src = pos[n:]

    inter_dist = np.linalg.norm(lh_src[:, np.newaxis, :]-rh_src[np.newaxis, :, :], axis=2)
    inter_dist[inter_dist > inter_hemi_dist] = np.inf

    # geometry
    geometry = np.block([
        [hemispheres[0],  inter_dist    ],
        [inter_dist.T,    hemispheres[1]]
    ])

    # replace infinities with zeros, remove self loops
    geometry[geometry == np.inf] = 0
    np.fill_diagonal(geometry, 0)

    if not weights:
        return (geometry > 0).astype(int)

    return geometry


def connectome_from_atlas(
        source_space    : mne.SourceSpaces,
        parcellation    : Parcellation,
        atlas_matrix    : np.ndarray,
        ras2tkras       : np.ndarray,
        sparsity        : float = 0.2,
        strength_matrix : np.ndarray = None,
        sparsity_range  : tuple[float, float] = None,
        deterministic   : bool = True,
        sigma           : float = 50.0,
        ) -> np.ndarray:
    """
    Builds a source-space distance connectome from a coarse region-level
    atlas, using boundary proximity mapping to upscale it to dipole
    resolution. See this module's docstring for the literature and
    known limitations behind this approach (in particular:
    `deterministic=True` concentrates edges at the mutually closest
    boundary dipoles).

    Parameters
    ----------
    source_space: mne.SourceSpaces
        Source space.

    parcellation: Parcellation
        Parcellation related to atlas.

    atlas_matrix: np.ndarray of shape (m, m)
        Macro connetcome matrix where entries are average fiber lengths.

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform matrix from FreeSurfer to subject RAS.

    sparsity: float
        Value between 0.0 and 1.0 dictating the portion of connections to
        keep per region pair. Used as-is when `strength_matrix` is None;
        otherwise only used for atlas-connected region pairs that end up
        with 0 valid entries in `strength_matrix` (e.g. NaNs).

    strength_matrix: np.ndarray of shape (m, m), optional
        Per-region-pair connection strength/confidence (e.g. the atlas's
        consistency matrix -- percentage of subjects with this tract).
        When given (together with `sparsity_range`), the sparsity used
        for each atlas-connected region pair is modulated by the
        percentile rank of its strength among all atlas-connected pairs,
        instead of a single value uniform across pairs -- so a
        near-universal, high-confidence tract gets a denser dipole-level
        fan-out than one that only just cleared the atlas's own
        thresholding. Percentile rank (not raw min-max scaling) is used
        so that a couple of outlier pairs don't compress the rest of the
        distribution.

    sparsity_range: tuple[float, float], optional
        (sparsity_min, sparsity_max) used when `strength_matrix` is
        given: the weakest atlas-connected pair gets sparsity_min, the
        strongest gets sparsity_max. Required if `strength_matrix` is set.

    deterministic: bool, otpional (default=True)
        If True, connects the absolute closest dipoles.
        If False, uses a distance-weigthed random lottery to spread out connections.

    sigma: float, optional (defautl=50.0)
        The spatial decay constant (in mm) used only when deterministic=False.
        Smaller values force tighter clustering; larger values allow more widespread randomness.

    Returns
    -------
    source_connectome: np.ndarray of shape (n_dipoles, n_dipoles)
        High-resolution distance matrix for the source space.
    """
    # 1. Label dipoles according to atlas brain regions
    transformed_dipoles = get_transformed_dipoles(source_space, ras2tkras)
    dipole_labels = get_dipole_region_labels(source_space, parcellation, ras2tkras)
    n_dipoles = transformed_dipoles.shape[0]

    # 2. Initialize the high-resolution connectome matrix
    source_connectome = np.zeros((n_dipoles, n_dipoles))

    # pre-group dipole indices by their region ID to avoid searching inside the loop
    num_regions = atlas_matrix.shape[0]
    global_indices = np.arange(n_dipoles)
    region_to_dipoles = {
        r: global_indices[dipole_labels == (r + 1)]
        for r in range(num_regions)
    }

    # modulate per-pair sparsity by atlas connection strength, if given
    if strength_matrix is not None:
        if sparsity_range is None:
            raise ValueError("sparsity_range=(low, high) must be given when strength_matrix is used.")

        low, high = sparsity_range
        sparsity_matrix = np.full(atlas_matrix.shape, sparsity, dtype=float)

        # rank over unique region pairs only (upper triangle): atlas_matrix is
        # symmetric, so `atlas_matrix > 0` would otherwise count each pair
        # twice and skew the percentile rank.
        triu_i, triu_j = np.triu_indices(num_regions, k=1)
        connected = atlas_matrix[triu_i, triu_j] > 0

        if np.any(connected):
            rows, cols = triu_i[connected], triu_j[connected]
            rank = normalize_data(strength_matrix[rows, cols], strategy="global", method="rank")
            pair_sparsity = low + (high - low) * rank
            sparsity_matrix[rows, cols] = pair_sparsity
            sparsity_matrix[cols, rows] = pair_sparsity
    else:
        sparsity_matrix = None

    # Iterate through unique macro-region pairs
    for i in range(num_regions):
        idx_i = region_to_dipoles[i]
        if len(idx_i) == 0:
            continue

        for j in range(i+1, num_regions):
            idx_j = region_to_dipoles[j]
            if len(idx_j) == 0:
                continue

            # check if an actual white matter tract exists in the atlas
            W_ij = atlas_matrix[i, j]
            if W_ij == 0:
                continue

            # extract 3D coordinates for dipoles in both regions
            coords_i = transformed_dipoles[idx_i]
            coords_j = transformed_dipoles[idx_j]

            # 3. Compute pairwise 3D euclidean distances
            dist_3d_matrix = sp.spatial.distance.cdist(coords_i, coords_j, metric="euclidean")

            # find closest boundary distance between these two subsets
            min_3d_dist = np.min(dist_3d_matrix)

            # calibrate distances based on the Atlas average tract length
            tract_dist_matrix = dist_3d_matrix - min_3d_dist + W_ij

            # 4. Apply sparsity constraint (strength-modulated if sparsity_matrix is set)
            pair_sparsity = sparsity_matrix[i, j] if sparsity_matrix is not None else sparsity
            total_possible_edges = len(idx_i) * len(idx_j)
            k_keep = max(1, int(np.round(total_possible_edges * pair_sparsity)))

            flat_tract_distances = tract_dist_matrix.ravel()
            keep_mask_flat = np.zeros(total_possible_edges, dtype=bool)

            # --- METHOD A: Purely Deterministic (Closest Neighbor) ---
            if deterministic:
                if k_keep < total_possible_edges:
                    threshold_val = np.partition(flat_tract_distances, k_keep - 1)[k_keep - 1]
                    keep_mask_flat = flat_tract_distances <= threshold_val

                    # exact tie-breaker to guarantee exactly k_keep indices
                    if np.sum(keep_mask_flat) > k_keep:
                        true_indices = np.where(keep_mask_flat)[0]
                        keep_mask_flat[true_indices[k_keep:]] = False
                else:
                    keep_mask_flat = np.ones(total_possible_edges, dtype=bool)

            # --- METHOD B: Stochastic (Distance-Weighted Lottery) ---
            else:
                if k_keep < total_possible_edges:
                    # Map distances to weights using exponential decay
                    # subtracting min ensures numerical stability (prevents underflow).
                    weights = np.exp(-(flat_tract_distances - np.min(flat_tract_distances)) / sigma)

                    # Normalize weights into a true probability distribution summing to 1
                    probabilities = weights / np.sum(weights)

                    # sample k_keep unique edge indices without replacement
                    chosen_indices = np.random.choice(
                        total_possible_edges,
                        size = k_keep,
                        replace = False,
                        p = probabilities,
                    )
                    keep_mask_flat[chosen_indices] = True
                else:
                    keep_mask_flat = np.ones(total_possible_edges, dtype=bool)

            # Reshape the flat mask to the 2D sub-matrix shape
            keep_mask = keep_mask_flat.reshape(tract_dist_matrix.shape)

            # 5. Map back to global source connectome coordinates
            # Get the local row/column positions where mask is True
            row_indices, col_indices = np.where(keep_mask)

            # Translate local sub-matrix indices back to global source-space indices
            global_row = idx_i[row_indices]
            global_col = idx_j[col_indices]

            # Get the distances for those chosen pairs
            final_distances = tract_dist_matrix[keep_mask]

            # Symmetrically assign to the master matrix
            source_connectome[global_row, global_col] = final_distances
            source_connectome[global_col, global_row] = final_distances

    return source_connectome


