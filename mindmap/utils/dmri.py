

from ..utils import Parcellation
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
    

