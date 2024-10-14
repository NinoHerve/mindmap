

from ..utils import Parcellation, make_dparc, make_pparc 
from pathlib import Path
import mne 
import scipy as sp
import h5py
import numpy as np


# ------------------------------------------------------------------------------
#
# Connectome from a template
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#
# ------------------------------------------------------------------------------

def make_MultiConn(
    subject: str,
    atlasL: str | Path,
    atlasR: str | Path,
    parc_atlas: str | Path,
    ref_atlas: str | Path,
    clut: str | Path,
    wmparc_dmax: float,
    thresh: float,
    fs_dir: str | Path = None,
    fs_home: str | Path = None,
) -> Parcellation:
    """
    Generate a MultiConn's parcellation for given subject. The MultiConn parcellation is 
    necessary for registering the MultiConn atlas to a subject.

    Parameters
    ----------
    subject: path-like
        Subject's name.

    atlasL: path-like
        Path to right hemisphere atlas annotation.

    atlasR: path-like
        Path to right hemisphere atlas annotation.

    parc_atlas: path-like
        Parcellation probability volume atlas.

    ref_atlas: path-like
        Structural T1 atlas of reference of the probability volume atlas.

    clut: path-like
        Color look up table for the parcellation atlas.

    wmparc_dmax: float
        Maximum distance in [mm] from cortex to be labeled as gyral white matter.

    thresh: float
        Minimum probability to consider a label in the probability atlas.

    fs_dir: path-like
        FreeSurfer subjects directory.

    fs_home: path-like
        FreeSurfer home directory.


    Returns
    -------
    parc: Parcellation
        Parcellated volume.

    Note: If the file name when calling ants.image_read() is not an nii, the kernel
    crashes without giving you any cause. Make sure that the file names of 
    mri and atlas have .nii or .nii.gz extensions.
    """
        
    # make cortical parcellation (3 min)
    laus2018 = make_dparc(
        subject     = subject,
        atlasL      = atlasL,
        atlasR      = atlasR,
        wmparc_dmax = wmparc_dmax,
        fs_dir      = fs_dir,
        fs_home     = fs_home,
    )
    laus2018.change_ids([(3001, 3999), (4001, 4999)], lambda x: x-2000)      # change white matter ids to grey matter ids

    # make sub-cortical parcellation (1 min)
    thal = make_pparc(
        subject    = subject,
        parc_atlas = parc_atlas,
        ref_atlas  = ref_atlas,
        clut       = clut,
        fs_dir     = fs_dir,
    )
    thal = thal.prob2dseg(thresh=thresh)         # convert probability volume to deterministic volume

    # Extract and rename parts of interest from the Lausanne 2018 parcellation.
    ctx_lh = laus2018.retain([(1001, 1999), (3001, 3999)])
    ctx_lh.change_labels(prefix="ctx-lh-")

    ctx_rh = laus2018.retain([(2001, 2999), (4001, 4999)])
    ctx_rh.change_labels(prefix="ctx-rh-")

    subc_lh = laus2018.retain([11, 12, 13, 26, 18, 17])
    subc_lh.change_labels(replace={"Left-":"", "Accumbens-area":"accumbens_area"}, prefix="subc-lh-", lower=True)

    subc_rh = laus2018.retain([50, 51, 52, 58, 54, 53])
    subc_rh.change_labels(replace={"Right-":"", "Accumbens-area":"accumbens_area"}, prefix="subc-rh-", lower=True)

    # Extract and rename parts of interest from the thalamic parcellation.
    thal_lh = thal.retain([(8, 14)])

    thal_rh = thal.retain([(1,7)])

    brain_stem = laus2018.retain([16])
    brain_stem.change_labels(replace={"-":"_"})

    # join parts of interest
    multi_conn = Parcellation()
    multi_conn = multi_conn.join(ctx_rh)
    multi_conn = multi_conn.join(thal_rh)
    multi_conn = multi_conn.join(subc_rh)
    multi_conn = multi_conn.join(ctx_lh)
    multi_conn = multi_conn.join(thal_lh)
    multi_conn = multi_conn.join(subc_lh)
    multi_conn = multi_conn.join(brain_stem)

    return multi_conn


def infer_wm_tracks(
        sources: mne.SourceSpaces,
        parcel: Parcellation,
        torig: np.ndarray,
        atlas: str | Path,
        consistency_thresh: float,
        length_thresh: float,
) -> sp.sparse.csc_array:
    """
    Build a connectome for a source space using a connectome atlas.

    Parameters
    ----------
    sources: mne.SourceSpaces
        Source space.

    parcel: Parcellation
        Parcellation for source_space. To associate each source to a region of the 
        atlas, the use must provide a parcellation registered to the same space as
        the source space.

    torig: np.ndarray (4x4)
        Transformation matrix to convert voxel indices to coordinates in the MRI surface 
        RAS space.

    atlas: path-like
        Connectome atlas.

    consistency_thresh: float
        Fiber consistency threshold.

    length_thresh: float
        Fiber length threshold.

    Returns
    -------
    sc: sp.sparse.csc_array
        Subject binary structural connectome of white matter tracks.
    """
    # load atlas
    f = h5py.File(atlas)

    # label sources
    points = [s["rr"][s["vertno"]] for s in sources]
    points = np.concatenate(points)*1000                # in surface RAS space in [mm].
    labels = parcel.get_ids(points, torig)              # brain region label for each source.

    # convert atlas to binary adjacency matrix
    consistency = f["matrices"]["consistency"]
    length = f["matrices"]["length"]

    sc_atlas = np.bitwise_and(consistency[:,:]>consistency_thresh, length[:,:]>length_thresh)

    # create connections between sources
    # (if source A belongs to a region that is connected to the region of source B, then connect both sources.)
    N = points.shape[0]
    sc = np.zeros((N,N))

    for i, l in enumerate(labels):
        connected_regions = np.argwhere(sc_atlas[l-1]==1)+1
        connections = np.isin(labels, connected_regions)
        sc[i] = connections 

    sc = sp.sparse.csc_array(sc>0)

    return sc.astype(int)


def interpol_gm_tracks(
        sources: mne.SourceSpaces,
        dmax: float,
) -> sp.sparse.csc_matrix:
    """
    Interpolate gray matter connections. Sources that have a geodesic distance smaller
    than user-given distance are connected.

    Parameters
    ----------
    sources: mne.SourceSpaces
        Source space.

    dmax: float
        Neighborhood distance to connect sources.
    """
    connectomes = []

    for src in sources:
        vertices = src["vertno"]
        sc       = src["dist"][vertices][:,vertices].tolil()

        # binarize distances
        sc[sc>dmax] = 0
        sc.setdiag(0)
        sc = sc.tocsc()
        sc.eliminate_zeros()
        sc = sc.astype(bool).astype(int)

        connectomes.append(sc)

    sc = sp.sparse.block_diag(connectomes)

    return sc
    

