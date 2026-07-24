"""
Module Name: analysis.functional.common.metric.py
Description: Common metrics for other sub-modules in this module.

This module provides functions to measure data.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import pandas as pd

import mne
import nibabel as nib

from mindmap.core.metric import (
    calc_confusion_matrix, calc_activity_scores,
    calc_roc_curve, calc_pr_curve, calc_correlation,
    calc_cosine, calc_hypergeometric_enrichment,
    calc_fold_enrichment, label_clusters, point_to_area_distance,
    calc_pattern_geometry_correlation, calc_explained_variance, calc_ssim,
    calc_global_scale, calc_scaled_rmse,
)
from mindmap.anatomy.source_space import (
    get_dipole_positions, sample_nifti_at_points, nifti_mask_to_points,
)
from mindmap.analysis.functional.common.utils import (
    is_binary_mne,
)


def compute_mne_correlation(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
    method: str = "pearson",
    polarity_invariant: bool = False,
) -> float | np.ndarray:
    """
    Compute correlation.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']

    polarity_invariant: bool
        If True, returns |corr| -- see `mindmap.core.metric.calc_correlation`.

    Returns
    -------
    corr: float | np.ndarray
        Correlation.
    """
    y_true = obj_true.data
    y_pred = obj_pred.data

    # map mode to axis
    axis_map = {"spatio-temporal": None, "spatial": 0, "temporal": 1}
    target_axis = axis_map[mode]
    center_axis = axis_map["temporal"] if target_axis is None else target_axis

    return calc_correlation(y_true, y_pred, method, target_axis, center_axis, polarity_invariant)


def compute_mne_cosine(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
    absolute: bool = False,
    polarity_invariant: bool = False,
) -> float | np.ndarray:
    """
    Compute cosine similarity.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth acitvity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']

    polarity_invariant: bool
        If True, returns |cos| -- see `mindmap.core.metric.calc_cosine`.

    Returns
    -------
    cos: float | np.ndarray
        Cosine similarity.
    """
    y_true = obj_true.data
    y_pred = obj_pred.data

    # map mode to axis
    axis_map = {"spatio-temporal": None, "spatial": 0, "temporal": 1}
    target_axis = axis_map[mode]

    return calc_cosine(y_true, y_pred, target_axis, absolute, polarity_invariant)


def compute_mne_pattern_geometry_correlation(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    method: str = "pearson",
    compare_method: str = "spearman",
) -> float:
    """
    Second-order ("representational similarity") comparison of how the
    spatial pattern's geometry evolves across time. See
    `calc_pattern_geometry_correlation` for the full explanation.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    method: str ['pearson', 'spearman', 'log-envelope']
        Method used to build each object's own dissimilarity matrix.

    compare_method: str ['pearson', 'spearman', 'log-envelope']
        Method used for the final comparison between the two dissimilarity
        matrices.

    Returns
    -------
    float
    """
    return calc_pattern_geometry_correlation(
        obj_true.data, obj_pred.data, method=method, compare_method=compare_method,
    )


def compute_mne_explained_variance(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
    polarity_invariant: bool = False,
) -> float | np.ndarray:
    """
    Compute explained variance (R^2). See `calc_explained_variance` --
    unlike correlation/cosine, this is sensitive to amplitude scale.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']

    polarity_invariant: bool
        If True, also scores against a sign-flipped obj_pred and keeps
        whichever is higher -- see `mindmap.core.metric.calc_explained_variance`.

    Returns
    -------
    r2: float | np.ndarray
    """
    y_true = obj_true.data
    y_pred = obj_pred.data

    axis_map = {"spatio-temporal": None, "spatial": 0, "temporal": 1}
    target_axis = axis_map[mode]
    center_axis = axis_map["temporal"] if target_axis is None else target_axis

    return calc_explained_variance(y_true, y_pred, target_axis, center_axis, polarity_invariant)


def compute_mne_global_scale(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
) -> float:
    """
    Compute the single global least-squares gain. See
    `mindmap.core.metric.calc_global_scale`.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    float
    """
    return calc_global_scale(obj_true.data, obj_pred.data)


def compute_mne_scaled_rmse(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> float | np.ndarray:
    """
    Compute RMSE after a single globally-fitted rescale. See
    `mindmap.core.metric.calc_scaled_rmse`.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']

    Returns
    -------
    rmse: float | np.ndarray
    """
    y_true = obj_true.data
    y_pred = obj_pred.data

    axis_map = {"spatio-temporal": None, "spatial": 0, "temporal": 1}
    target_axis = axis_map[mode]

    return calc_scaled_rmse(y_true, y_pred, target_axis)


def compute_mne_ssim(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    win_size: int = 7,
    data_range: float = None,
) -> float:
    """
    Compute Structural Similarity (SSIM), treating the object's data as a
    2D (channels/vertices x time) signal. See `calc_ssim`.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    win_size: int
        Sliding comparison window size, clamped to fit the data if needed.

    data_range: float, optional
        Dynamic range of the data. Defaults to the joint range of both
        objects.

    Returns
    -------
    float
    """
    return calc_ssim(obj_true.data, obj_pred.data, win_size=win_size, data_range=data_range)


def compute_mne_confusion(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Compute confusion matrix.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binarized ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
        Binarized estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.
    
    Returns
    -------
    dict
        Dictionary of True/False Positive/Negative.
    """
    # check if inputs are binary
    if not is_binary_mne(obj_true) or not is_binary_mne(obj_pred):
        raise TypeError("Input data needs to be binary for computing the confusion matrix.")

    y_true = obj_true.data
    y_pred = obj_pred.data

    # dynamic labels (channels vs dipoles)
    label_name = "channel" if hasattr(obj_true, "ch_names") else "dipole_index"
    labels = obj_true.ch_names if hasattr(obj_true, "ch_names") else np.arange(len(y_true))

    # Count positives & negatives
    if mode == "spatio-temporal":
        return calc_confusion_matrix(y_true, y_pred)

    results = []
    if mode == "spatial":
        for t in range(y_true.shape[1]):
            res = calc_confusion_matrix(y_true[:, t], y_pred[:, t])
            res["time"] = obj_true.times[t]
            results.append(res)
        return pd.DataFrame(results).set_index("time")

    if mode == "temporal":
        for n in range(y_true.shape[0]):
            res = calc_confusion_matrix(y_true[n, :], y_pred[n, :])
            res[label_name] = labels[n]
            results.append(res)
        return pd.DataFrame(results).set_index(label_name)



def compute_mne_scores(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
    mode: str = "spatio-temporal",
) -> dict:
    """
    Compute scores.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.
    
    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    mode: str ['spatial', 'temporal', 'spatio-temporal']
        - 'spatial': compute scores for each time point.
        - 'temporal': compute scores for each channel.
        - 'spatio-temporal': compute scores overall.
    
    Returns
    -------
    dict
        Dictionary of scores.
    """
    # check if inputs are binary
    if not is_binary_mne(obj_true) or not is_binary_mne(obj_pred):
        raise TypeError("Input data needs to be binary for computing performance scores.")

    y_true = obj_true.data
    y_pred = obj_pred.data

    # dynamic labels (channels vs dipoles)
    label_name = "channel" if hasattr(obj_true, "ch_names") else "dipole_index"
    labels = obj_true.ch_names if hasattr(obj_true, "ch_names") else np.arange(len(y_true))

    # calculate scores
    if mode == "spatio-temporal":
        return calc_activity_scores(y_true, y_pred)

    results = []
    if mode == "spatial":
        for t in range(y_true.shape[1]):
            res = calc_activity_scores(y_true[:, t], y_pred[:, t])
            res["time"] = obj_true.times[t]
            results.append(res)
        return pd.DataFrame(results).set_index("time")

    if mode == "temporal":
        for n in range(y_true.shape[0]):
            res = calc_activity_scores(y_true[n, :], y_pred[n, :])
            res[label_name] = labels[n]
            results.append(res)
        return pd.DataFrame(results).set_index(label_name)


def compute_mne_roc(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
) -> dict:
    """
    Computes data for a ROC curve.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    dict
        Dictionary with 'fpr', 'tpr' and 'auc'.
    """
    if not is_binary_mne(obj_true):
        raise TypeError("Ground truth needs to be binary to compute a ROC curve.")

    y_true = obj_true.data
    y_pred = np.abs(obj_pred.data)

    return calc_roc_curve(y_true, y_pred)


def compute_mne_pr(
    obj_true: mne.Evoked | mne.SourceEstimate,
    obj_pred: mne.Evoked | mne.SourceEstimate,
) -> dict:
    """
    Computes data for a precision-recall curve.

    Parameters
    ----------
    obj_true: mne.Evoked | mne.SourceEstimate
        Binary ground truth activity.

    obj_pred: mne.Evoked | mne.SourceEstimate
        Estimated activity.

    Returns
    -------
    dict
        Dictionary with 'precision', 'recall' and 'ap'.
    """
    if not is_binary_mne(obj_true):
        raise TypeError("Ground truth needs to be binary to compute a precision-recall curve.")

    y_true = obj_true.data
    y_pred = np.abs(obj_pred.data)

    return calc_pr_curve(y_true, y_pred)


def evaluate_resection_pr(
    resection_vertex_mask: np.ndarray,
    vertex_activity: np.ndarray,
) -> dict:
    """
    Computes data for a precision-recall curve against a resection-mask ground truth.

    Parameters
    ----------
    resection_vertex_mask: np.ndarray of shape (n_vertices,)
        Binary ground truth, e.g. from
        `mm.anatomy.source_space.sample_nifti_at_source_space`.

    vertex_activity: np.ndarray of shape (n_vertices,)
        Per-vertex estimated activity (e.g. peak amplitude of a
        mne.SourceEstimate).

    Returns
    -------
    dict
        Dictionary with 'precision', 'recall' and 'ap'.
    """
    y_true = resection_vertex_mask.astype(int)
    y_pred = np.abs(vertex_activity)

    return calc_pr_curve(y_true, y_pred)


def find_peak_resection_distance(
    vertex_activity: np.ndarray,
    source_space: mne.SourceSpaces,
    mask: nib.Nifti1Image,
    ras2tkras: np.ndarray = None,
) -> dict:
    """
    Euclidean distance from the single global-activity-maximum dipole to
    the resection.

    The classic ESI baseline this evaluation set was built to move past --
    a multifocal map's real hit can be outvoted by a taller unrelated peak
    elsewhere -- kept here as a comparison point against
    `find_earliest_resection_cluster`/`evaluate_resection_pr`, not as the
    primary metric.

    Measures against the resection's own voxel geometry (via
    `mm.anatomy.source_space.nifti_mask_to_points`), not a vertex-sampled
    approximation of it: vertex-sampling the resection is the right ground
    truth when comparing vertex-to-vertex against activity (PR/cluster
    metrics need that correspondence), but it would bias a point-to-area
    distance, since the true resected tissue can sit closer to the peak
    than any vertex that happened to fall inside it, by up to the local
    dipole spacing. The one exception is the inside/outside test at the
    peak vertex's own position, which is exact regardless -- it's a direct
    volume sample at that one point, not a search over neighboring
    vertices.

    Parameters
    ----------
    vertex_activity: np.ndarray of shape (n_vertices,)
        Per-vertex estimated activity (e.g. peak amplitude of a
        mne.SourceEstimate).

    source_space: mne.SourceSpaces
        Source space (space-freesurfer) `vertex_activity` was computed on,
        same vertex order as any mne.SourceEstimate built from it.

    mask: nib.Nifti1Image
        Resection volume (e.g. a binary mask).

    ras2tkras: np.ndarray (4,4), optional
        Affine transform from subject/T1 scanner-RAS [mm] to FreeSurfer
        surface-RAS (tkRAS) [mm]. Required whenever `source_space` and
        `mask` don't already share the same world coordinate frame -- same
        argument as `sample_nifti_at_source_space`.

    Returns
    -------
    dict
        'peak_vertex': int -- index of the global |activity| maximum.
        'inside_resection': bool -- whether the peak vertex itself falls
            inside the resection.
        'distance_mm': float -- 0 if inside; else the Euclidean distance
            from the peak vertex to the nearest resection voxel, in mm.
    """
    activity = np.abs(np.ravel(vertex_activity))
    peak_vertex = int(np.argmax(activity))
    peak_pos = get_dipole_positions(source_space)[peak_vertex] * 1000

    inside = bool(sample_nifti_at_points(peak_pos[None, :], mask, ras2tkras)[0] > 0)
    if inside:
        return {"peak_vertex": peak_vertex, "inside_resection": True, "distance_mm": 0.0}

    resection_points = nifti_mask_to_points(mask, ras2tkras)
    distance = point_to_area_distance(peak_pos, resection_points, x=1)

    return {"peak_vertex": peak_vertex, "inside_resection": False, "distance_mm": float(distance)}


def find_peak_localization_error(
    vertex_activity: np.ndarray,
    true_source_mask: np.ndarray,
    dipole_positions: np.ndarray,
) -> dict:
    """
    Euclidean distance from the single global-activity-maximum vertex to
    the true simulated source region.

    The simulation-ground-truth analogue of `find_peak_resection_distance`:
    same classic localization-error baseline (single loudest vertex vs. a
    known true region, 0 if it lands inside), just against a source region
    already known in vertex space (e.g. from a simulation's
    `Source.select_targets(source_space, parc, ras2tkras)`) rather than one
    that needs resolving from a volumetric nifti mask -- so unlike the
    resection case, no affine/voxel-sampling step is needed here: the
    "inside" test and the distance's target cloud both come directly from
    `true_source_mask`, since the ground truth was defined on this exact
    vertex grid to begin with (nothing lossy about working vertex-to-vertex
    here, unlike sampling a real anatomical resection onto a decimated
    source space). Blind to multifocal maps by the same construction as
    `find_peak_resection_distance` -- read alongside a cluster/PR-style
    metric, not as the sole score.

    Parameters
    ----------
    vertex_activity: np.ndarray of shape (n_vertices,)
        Per-vertex estimated activity (e.g. a reconstructed
        mne.SourceEstimate's data at one timepoint).

    true_source_mask: np.ndarray of shape (n_vertices,), bool
        Ground-truth vertex mask for the true simulated source, e.g. from
        `Source.select_targets(source_space, parc, ras2tkras)`.

    dipole_positions: np.ndarray of shape (n_vertices, 3)
        Vertex coordinates in tkRAS [mm], e.g.
        `mm.anatomy.source_space.get_dipole_positions(src) * 1000`, same
        vertex order as `vertex_activity`/`true_source_mask`.

    Returns
    -------
    dict
        'peak_vertex': int -- index of the global |activity| maximum.
        'inside_source': bool -- whether the peak vertex itself is part of
            the true source region.
        'distance_mm': float -- 0 if inside; else the Euclidean distance
            from the peak vertex to the nearest true-source vertex, in mm.
    """
    true_source_mask = np.ravel(true_source_mask).astype(bool)
    activity = np.abs(np.ravel(vertex_activity))
    peak_vertex = int(np.argmax(activity))

    if true_source_mask[peak_vertex]:
        return {"peak_vertex": peak_vertex, "inside_source": True, "distance_mm": 0.0}

    distance = point_to_area_distance(dipole_positions[peak_vertex], dipole_positions[true_source_mask], x=1)

    return {"peak_vertex": peak_vertex, "inside_source": False, "distance_mm": float(distance)}


def find_best_enrichment(
    vertex_activity: np.ndarray,
    resection_vertex_mask: np.ndarray,
    adjacency,
    patch_area: np.ndarray,
    thresholds: list,
    min_cluster_area: float = 0,
) -> pd.DataFrame:
    """
    Sweeps amplitude thresholds and reports the most enriched candidate cluster at each.

    At each threshold, the top-N vertices by |activity| are clustered via
    connected components on `adjacency`; every resulting cluster is scored
    with `calc_hypergeometric_enrichment`/`calc_fold_enrichment` using
    physical area (mm^2, from `patch_area`) rather than raw vertex counts --
    keeping results comparable across source spaces built at different
    resolutions. Only the most significant (lowest p-value) cluster per
    threshold is kept.

    Parameters
    ----------
    vertex_activity: np.ndarray of shape (n_vertices,)
        Per-vertex estimated activity (e.g. peak amplitude of a
        mne.SourceEstimate).

    resection_vertex_mask: np.ndarray of shape (n_vertices,)
        Binary ground truth, e.g. from
        `mm.anatomy.source_space.sample_nifti_at_source_space`.

    adjacency: scipy.sparse matrix of shape (n_vertices, n_vertices)
        e.g. from `mm.anatomy.source_space.build_dipole_adjacency`. Depends
        only on the source space, not on activity or threshold -- build once
        per subject and reuse across every solver.

    patch_area: np.ndarray of shape (n_vertices,)
        Physical patch area (mm^2) per vertex, e.g. from
        `mm.anatomy.source_space.build_dipole_patch_area`. Same
        once-per-subject reuse as `adjacency`.

    thresholds: list of int
        Candidate cutoffs, each the number of top-|activity| vertices to
        include before clustering.

    min_cluster_area: float
        Minimum physical area (mm^2) for a cluster to be scored; smaller
        clusters are skipped. Defaults to 0 (no filtering).

    Returns
    -------
    pd.DataFrame indexed by threshold
        Columns: 'n_clusters', 'k_mm2', 'tp_mm2', 'precision', 'p_value',
        'surprise' (-log10 p_value), 'fold_enrichment' -- for the best
        (lowest p-value) cluster at that threshold. Rows where no cluster
        meets `min_cluster_area` have NaN scores.
    """
    resection_vertex_mask = np.ravel(resection_vertex_mask).astype(bool)
    n_positive = patch_area[resection_vertex_mask].sum()
    population_size = patch_area.sum()

    activity = np.abs(np.ravel(vertex_activity))
    order = np.argsort(-activity)

    results = []
    for threshold in thresholds:
        active_mask = np.zeros_like(activity, dtype=bool)
        active_mask[order[:threshold]] = True

        active_idx = np.nonzero(active_mask)[0]
        labels, sizes = label_clusters(adjacency, active_mask)

        best = {"n_clusters": len(sizes), "k_mm2": np.nan, "tp_mm2": np.nan,
                "precision": np.nan, "p_value": np.nan, "surprise": np.nan,
                "fold_enrichment": np.nan}
        best_p = np.inf

        for cluster_id in range(len(sizes)):
            members = active_idx[labels == cluster_id]

            k_mm2 = patch_area[members].sum()
            if k_mm2 < min_cluster_area:
                continue

            tp_mm2 = patch_area[members[resection_vertex_mask[members]]].sum()
            p = calc_hypergeometric_enrichment(k_mm2, tp_mm2, population_size, n_positive)

            if p < best_p:
                best_p = p
                best = {
                    "n_clusters": len(sizes),
                    "k_mm2": k_mm2,
                    "tp_mm2": tp_mm2,
                    "precision": tp_mm2 / k_mm2 if k_mm2 > 0 else np.nan,
                    "p_value": p,
                    "surprise": -np.log10(p) if p > 0 else np.inf,
                    "fold_enrichment": calc_fold_enrichment(k_mm2, tp_mm2, population_size, n_positive),
                }

        results.append({"threshold": threshold, **best})

    return pd.DataFrame(results).set_index("threshold")


def _find_qualifying_cluster(order, k, adjacency, patch_area, resection_vertex_mask, min_cluster_area):
    """
    Activates the top-k ranked vertices, clusters them, and returns the
    member indices of the first cluster that is >= min_cluster_area and
    overlaps the resection -- or None if no such cluster exists at this k.

    Shared by `find_earliest_resection_cluster` (which only needs to know
    whether one exists) and `reconstruct_resection_cluster` (which needs the
    actual members to rebuild the map), so the "what counts as a qualifying
    cluster" logic lives in exactly one place.
    """
    n = len(order)
    active_mask = np.zeros(n, dtype=bool)
    active_mask[order[:k]] = True

    active_idx = np.nonzero(active_mask)[0]
    labels, sizes = label_clusters(adjacency, active_mask)

    for cluster_id in range(len(sizes)):
        members = active_idx[labels == cluster_id]
        if patch_area[members].sum() >= min_cluster_area and resection_vertex_mask[members].any():
            return members

    return None


def find_earliest_resection_cluster(
    vertex_activity: np.ndarray,
    resection_vertex_mask: np.ndarray,
    adjacency,
    patch_area: np.ndarray,
    min_cluster_area: float = 600.0,
) -> dict:
    """
    How much of the cortex has to be marked active before a real cluster
    first touches the resection.

    Ranks vertices by |activity| and grows the active set from the top down.
    A cluster only ever grows or merges as more vertices are added -- it
    never shrinks -- so "does a cluster >= min_cluster_area already overlap
    the resection" is monotonic in the threshold: once true, it stays true
    (confirmed empirically on real data, zero counterexamples over a 200-point
    scan across three solvers). That makes this a binary search for the
    earliest crossing, not a full sweep.

    Unlike `find_best_enrichment`, this reports no significance test --
    purely "how early", so there's no dipole-count-vs-area independence
    question to get wrong. `min_cluster_area` (mm^2) exists only to keep a
    single lucky vertex from trivially counting as a "cluster"; the default
    600 mm^2 is grounded in the literature-established minimum synchronously
    active cortex needed to produce a visible interictal discharge, cross-
    checked against a chance-level simulation on this exact source space
    (see the notebook design-decision note).

    Parameters
    ----------
    vertex_activity: np.ndarray of shape (n_vertices,)
        Per-vertex estimated activity (e.g. peak amplitude of a
        mne.SourceEstimate).

    resection_vertex_mask: np.ndarray of shape (n_vertices,)
        Binary ground truth, e.g. from
        `mm.anatomy.source_space.sample_nifti_at_source_space`.

    adjacency: scipy.sparse matrix of shape (n_vertices, n_vertices)
        e.g. from `mm.anatomy.source_space.build_dipole_adjacency`.

    patch_area: np.ndarray of shape (n_vertices,)
        Physical patch area (mm^2) per vertex, e.g. from
        `mm.anatomy.source_space.build_dipole_patch_area`.

    min_cluster_area: float
        Minimum physical area (mm^2) for a cluster to count.

    Returns
    -------
    dict
        'found': bool -- whether any qualifying cluster ever touches the
            resection, even including every vertex.
        'threshold': int -- rank at first crossing (vertices needed).
        'fraction_of_cortex': float -- physical area of the *whole active set*
            at that threshold (every cluster, not just the winning one), as a
            fraction of total cortex area. How early a hit was found -- the
            number to compare across solvers/subjects, lower is better.
        'area_mm2': float -- same, in mm^2.
        'cluster_area_mm2': float -- physical area of just the winning
            cluster itself (a different, usually smaller, number than
            `area_mm2`, which includes any other unrelated clusters active
            at the same threshold).
        'cluster_precision': float -- fraction of the winning cluster's own
            area that's inside the resection. Answers a different question
            than `fraction_of_cortex`: not how early the cluster was found,
            but whether it's actually centered on the resection (high) or
            merely grazing its edge while mostly sitting outside it (low) --
            two clusters found at the same threshold can have very different
            precision.
    """
    resection_vertex_mask = np.ravel(resection_vertex_mask).astype(bool)
    activity = np.abs(np.ravel(vertex_activity))
    order = np.argsort(-activity)
    n = len(order)
    total_area = patch_area.sum()

    def has_qualifying_cluster(k):
        return _find_qualifying_cluster(order, k, adjacency, patch_area, resection_vertex_mask, min_cluster_area) is not None

    if not has_qualifying_cluster(n):
        return {
            "found": False, "threshold": n, "fraction_of_cortex": 1.0, "area_mm2": total_area,
            "cluster_area_mm2": np.nan, "cluster_precision": np.nan,
        }

    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if has_qualifying_cluster(mid):
            hi = mid
        else:
            lo = mid + 1

    area_included = patch_area[order[:lo]].sum()

    members = _find_qualifying_cluster(order, lo, adjacency, patch_area, resection_vertex_mask, min_cluster_area)
    cluster_area = patch_area[members].sum()
    cluster_tp_area = patch_area[members[resection_vertex_mask[members]]].sum()

    return {
        "found": True,
        "threshold": lo,
        "fraction_of_cortex": area_included / total_area,
        "area_mm2": area_included,
        "cluster_area_mm2": cluster_area,
        "cluster_precision": cluster_tp_area / cluster_area,
    }


def reconstruct_resection_cluster(
    vertex_activity: np.ndarray,
    resection_vertex_mask: np.ndarray,
    adjacency,
    patch_area: np.ndarray,
    threshold: int,
    min_cluster_area: float = 600.0,
) -> dict:
    """
    Rebuilds the binary activity map at a specific threshold from
    `find_earliest_resection_cluster`, isolating the cluster that touched
    the resection.

    Pass `threshold` from that function's return value (`hit["threshold"]`),
    not a re-typed percentage: `fraction_of_cortex` is only shown rounded to
    one decimal for display, and several of the real crossing points here are
    a single vertex wide (see the binary-search verification), so
    reconstructing a threshold from a rounded percentage risks landing one
    vertex off the true boundary. The exact integer threshold has no such
    risk.

    Parameters
    ----------
    vertex_activity, resection_vertex_mask, adjacency, patch_area, min_cluster_area:
        Same as `find_earliest_resection_cluster` -- pass the same values
        used to obtain `threshold`.

    threshold: int
        Exact rank to rebuild the map at, e.g. `hit["threshold"]`.

    Returns
    -------
    dict
        'active_mask': np.ndarray of shape (n_vertices,), bool -- every
            vertex active at this threshold (every cluster, not just the
            one that touched the resection).
        'cluster_mask': np.ndarray of shape (n_vertices,), bool, or None --
            only the vertices of the specific cluster that overlaps the
            resection *and* meets `min_cluster_area`. None if no cluster at
            this threshold qualifies (e.g. a mismatched threshold/
            min_cluster_area was passed).
        'signed_map': np.ndarray of shape (n_vertices,), float -- every
            cluster active at this threshold, regardless of size (no
            `min_cluster_area` filtering here, unlike `cluster_mask`) --
            +1 for vertices in a cluster that touches the resection at all,
            -1 for vertices in a cluster that doesn't, 0 for inactive
            vertices. Meant for visualizing how noisy the map actually is at
            this threshold (how many other clusters exist besides the
            qualifying one), not just the isolated winning cluster.
    """
    resection_vertex_mask = np.ravel(resection_vertex_mask).astype(bool)
    activity = np.abs(np.ravel(vertex_activity))
    order = np.argsort(-activity)
    n = len(order)

    active_mask = np.zeros(n, dtype=bool)
    active_mask[order[:threshold]] = True

    members = _find_qualifying_cluster(order, threshold, adjacency, patch_area, resection_vertex_mask, min_cluster_area)

    cluster_mask = None
    if members is not None:
        cluster_mask = np.zeros(n, dtype=bool)
        cluster_mask[members] = True

    active_idx = np.nonzero(active_mask)[0]
    labels, sizes = label_clusters(adjacency, active_mask)

    signed_map = np.zeros(n)
    for cluster_id in range(len(sizes)):
        cluster_members = active_idx[labels == cluster_id]
        signed_map[cluster_members] = 1.0 if resection_vertex_mask[cluster_members].any() else -1.0

    return {"active_mask": active_mask, "cluster_mask": cluster_mask, "signed_map": signed_map}
