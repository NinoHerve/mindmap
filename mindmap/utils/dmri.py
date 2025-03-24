

from ..utils import Parcellation
from pathlib import Path
import mne 
import scipy as sp
import h5py
import numpy as np
import mindmap as mm
import pygsp


# ------------------------------------------------------------------------------
#
# Connectome from a template
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#
# ------------------------------------------------------------------------------

def create_cst_from_template(sub, consistency_thr:int = 40, length_thr:float = 25, number_thr:float = 20, distance_thr: float = 0.010):
    """
    Creates the connectome spectrum transform from a connectome atlas.
    Steps:
    1. Prepare the atlas (threshold according to consistency, fiber lengths and number of fibers)
    2. Project the atlas to subject (extrapolate from ~500 regions to ~5000 regions)
    3. Add close connections (connect dipoles with distance below threshold)
    4. Compute the CST

    Parameters
    ----------
    sub: str
        Subject name.

    consistency_thr: float
        Consistency threshold for atlas.

    length_thr: float
        Fiber length [cm] threshold for atlas.

    number_thr: int
        Number of fiber threshold for atlas.

    distance_thr: float
        Maximum dipole distance [mm] to connect.

    Returns
    -------
    cst: ndarray (n_dipoles, n_dipoles)
        Connectome spectrum transform.

    sc: ndarray (n_dipoles, n_dipoles)
        Structural connectome.
    """
    # Retrieve directories
    mm_dir = mm.io.readers._retrieve_env("MM_DIR")

    # Data
    data = mm.io.read_dataset(sub, "src", "parc")
    std  = mm.io.read_standards("conn")

    # label dipoles
    pos    = mm.utils.get_src_position(data["src"])
    pos    = pos*1000                
    labels = data["parc"].label_points(pos, id=True)   # brain region label for each source.

    # some left/right regions are very close and get mixed up
    labels = mm.utils.correct_labels(labels, data["src"], data["parc"])
    

    ###################
    ## Prepare Atlas ##
    ###################

    consistency = std["conn"]["matrices"]["consistency"]
    length      = std["conn"]["matrices"]["length"]
    number      = std["conn"]["matrices"]["numbStlines"]

    sc_atlas = np.bitwise_and(
        consistency[:,:] > consistency_thr,
        length[:,:] > length_thr, 
        number[:,:] > number_thr,
    )

    ##################################
    ## Extrapolate Atlas to Subject ##
    ##################################
 
    N     = pos.shape[0]
    sc_wm = np.zeros((N,N))

    for i, l in enumerate(labels):
        connected_regions = np.argwhere(sc_atlas[l-1]==1)+1
        connections = np.isin(labels, connected_regions)
        sc_wm[i] = connections 

    sc_wm = sp.sparse.csc_array(sc_wm>0)

    ############################
    ## Add Nearby connections ##
    ############################

    sc_gm = []

    for src in data["src"]:
        vertices = src["vertno"]
        dist     = src["dist"][vertices][:,vertices].tolil()

        # binarize distances
        dist[dist>distance_thr] = 0
        dist.setdiag(0)
        dist = dist.tocsc()
        dist.eliminate_zeros()
        dist = dist.astype(bool).astype(int)

        sc_gm.append(dist)

    sc_gm = sp.sparse.block_diag(sc_gm)

    #########################################################
    ## Combine white matter tracks with gray matter tracks ##
    #########################################################

    sc = sc_wm + sc_gm 
    sc = sc.astype(bool).astype(int)

    ###############################################
    ## Compute the Connectome Spectrum Transform ##
    ###############################################

    coords = [s["rr"][s["vertno"]] for s in data["src"]]
    G = pygsp.graphs.Graph(sc.todense(), lap_type="normalized", coords=coords)
    G.compute_fourier_basis()
    cst = G.U.T

    return cst, sc






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
    

