"""
Module Name: analysis/power/region_selection.py
Description: Region selection utilities for sizing pilot studies -- per-region
depth/orientation/lobe metadata, and a stratified farthest-point-sampling
subset selection over that metadata.

Author: Nino Herve
"""

import re

import numpy as np
import pandas as pd

import mne

from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.source_space.io import get_dipole_positions, get_dipole_directions
from mindmap.analysis.connectivity.utils import get_dipole_region_labels
from mindmap.geometry import farthest_point_sampling


_DK_LOBE = {
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
                "parsopercularis", "parstriangularis", "parsorbitalis",
                "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral", "frontalpole", "rostralanteriorcingulate",
                "caudalanteriorcingulate"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal",
                 "postcentral", "precuneus", "posteriorcingulate", "isthmuscingulate"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal",
                 "bankssts", "fusiform", "transversetemporal", "entorhinal",
                 "temporalpole", "parahippocampal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "insula": ["insula"],
}
_NAME_TO_LOBE = {name: lobe for lobe, names in _DK_LOBE.items() for name in names}


def _region_to_lobe(region_name: str) -> str:
    m = re.match(r"ctx-(lh|rh)-([a-z]+)_\d+", region_name)
    return _NAME_TO_LOBE.get(m.group(2), "other") if m else "other"


def _region_to_hemi(region_name: str) -> str:
    m = re.match(r"ctx-(lh|rh)-", region_name)
    return m.group(1) if m else "other"


def compute_region_geometry(
    forward: mne.Forward,
    parcellation: Parcellation,
    ras2tkras: np.ndarray,
) -> pd.DataFrame:
    """
    Per-region depth/orientation/lobe/hemi metadata, for stratified pilot
    region selection (see `stratified_region_selection`).

    depth: mean fixed-orientation leadfield column norm per region (higher =
    more superficial/EEG-visible).

    orientation (radiality): mean |cos(dipole normal, local radial direction
    from the brain centroid)| per region -- 1 = gyral/radial, 0 =
    sulcal/tangential.

    lobe: parsed from the Desikan-Killiany region name
    (`ctx-{hemi}-{DKname}_{n}`).

    Only cortical regions with dipoles in `forward["src"]` are returned --
    e.g. a surface-only source space has no subcortical/thalamic/brainstem
    dipoles, so those regions can't be simulation sources and are dropped
    here rather than left in with NaN geometry.

    Parameters
    ----------
    forward: mne.Forward
        Forward solution defining candidate dipole positions (`forward["src"]`).

    parcellation: Parcellation
        Parcellation used to label dipoles into brain regions.

    ras2tkras: np.ndarray of shape (4,4)
        Affine transform used to align source-space dipole positions with the
        parcellation's subject space (as loaded via `dm.load(sub, "ras2tkras")`).

    Returns
    -------
    region_meta: pd.DataFrame
        One row per cortical region, columns `region`, `depth`,
        `orientation`, `n_dipoles`, `lobe`, `hemi`.
    """
    src = forward["src"]
    dipole_labels = get_dipole_region_labels(src, parcellation, ras2tkras, return_id=False)

    positions = get_dipole_positions(src)
    normals = get_dipole_directions(src)

    fwd_fixed = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, use_cps=True, verbose="warning",
    )
    leadfield_norm = np.linalg.norm(fwd_fixed["sol"]["data"], axis=0)

    brain_center = positions.mean(axis=0)
    radial_unit = (positions - brain_center) / np.linalg.norm(positions - brain_center, axis=1, keepdims=True)
    radiality = np.abs(np.sum(normals * radial_unit, axis=1))

    df_dipole = pd.DataFrame({
        "region": dipole_labels, "leadfield_norm": leadfield_norm, "radiality": radiality,
    })

    region_meta = df_dipole.groupby("region").agg(
        depth=("leadfield_norm", "mean"), orientation=("radiality", "mean"), n_dipoles=("leadfield_norm", "size"),
    ).reset_index()
    region_meta["lobe"] = region_meta["region"].apply(_region_to_lobe)
    region_meta["hemi"] = region_meta["region"].apply(_region_to_hemi)
    region_meta = region_meta[region_meta["lobe"] != "other"].reset_index(drop=True)

    return region_meta


def stratified_region_selection(
    region_meta: pd.DataFrame,
    n_target: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Stratified pilot region selection: quantile-bins `depth` x `orientation`
    into tertiles (9 cells), allocates `n_target` regions proportionally
    across cells, picks within each cell via farthest-point sampling over
    standardized `(depth, orientation)` + a small-weight lobe code (so picks
    spread across lobes too), then tops up any candidate lobe left with zero
    picks with its region closest to that lobe's own median.

    Parameters
    ----------
    region_meta: pd.DataFrame
        Output of `compute_region_geometry` (needs `depth`, `orientation`,
        `lobe` columns; any others are carried through untouched).

    n_target: int
        Number of regions to select.

    seed: int
        Random seed for farthest-point sampling.

    Returns
    -------
    selection: pd.DataFrame
        Subset of `region_meta`'s rows (plus `depth_bin`/`orientation_bin`
        columns), one row per selected region.
    """
    region_meta = region_meta.copy()
    region_meta["depth_bin"] = pd.qcut(region_meta["depth"], 3, labels=["shallow", "mid", "deep"])
    region_meta["orientation_bin"] = pd.qcut(region_meta["orientation"], 3, labels=["tangential", "oblique", "radial"])

    cells_grp = region_meta.groupby(["depth_bin", "orientation_bin"], observed=True)
    sizes = cells_grp.size()
    alloc = np.maximum(1, np.round(sizes / sizes.sum() * n_target).astype(int))
    while alloc.sum() > n_target:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n_target:
        alloc[alloc.idxmin()] += 1

    lobes = sorted(region_meta["lobe"].unique())
    lobe_code = {l: i for i, l in enumerate(lobes)}

    selected_rows = []
    for (dbin, obin), grp in cells_grp:
        n_take = int(alloc[(dbin, obin)])
        grp = grp.reset_index(drop=True)
        if n_take >= len(grp):
            chosen = grp
        else:
            feats = np.column_stack([
                (grp["depth"] - grp["depth"].mean()) / (grp["depth"].std() + 1e-9),
                (grp["orientation"] - grp["orientation"].mean()) / (grp["orientation"].std() + 1e-9),
                0.5 * grp["lobe"].map(lobe_code).to_numpy(),
            ])
            idx = farthest_point_sampling(feats, n_take, seed=seed)
            chosen = grp.iloc[idx]
        selected_rows.append(chosen)

    selection = pd.concat(selected_rows, ignore_index=True)

    # top-up: guarantee every candidate lobe has >=1 representative
    missing_lobes = set(region_meta["lobe"].unique()) - set(selection["lobe"].unique())
    for lobe in missing_lobes:
        lobe_grp = region_meta[region_meta["lobe"] == lobe].copy()
        med_depth, med_orient = lobe_grp["depth"].median(), lobe_grp["orientation"].median()
        lobe_grp["dist_to_median"] = np.hypot(
            (lobe_grp["depth"] - med_depth) / (region_meta["depth"].std() + 1e-9),
            (lobe_grp["orientation"] - med_orient) / (region_meta["orientation"].std() + 1e-9),
        )
        selection = pd.concat([selection, lobe_grp.sort_values("dist_to_median").iloc[[0]]], ignore_index=True)

    return selection
