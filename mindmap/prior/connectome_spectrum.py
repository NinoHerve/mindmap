import mne 
from ..utils import Parcellation, infer_wm_tracks, interpol_gm_tracks
from pathlib import Path
import numpy as np
import pygsp

# ------------------------------------------------------------------------------
#
# Connectome Spectrum from tractography
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# Connectome Spectrum from a template
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#
# ------------------------------------------------------------------------------


def compute_cst_from_template(
        sources: mne.SourceSpaces,
        parcel: Parcellation,
        atlas: str | Path,
        consistency_thresh: float,
        length_thresh: float,
        dmax: float,
) -> np.ndarray:
    """
    Computes the Connectome Spectrum Transform using a template Connectome.

    Parameters
    ----------
    sources: mne.SourceSpaces
        Source space.
        
    parcel: Parcellation
        Subject specific parcellation containing same brain regions as altas.
        
    atlas: path-like
        Connectome probability atlas.

    consistency_threshold: float
        Value to threshold connectome according to fiber consistency.
        
    length_thresh: float
        Value to threshold connectome according to fiber length.

    Returns
    -------
    cst: np.ndarray
        Connectome spectrum transform. NxN matrix where N corresponds to the number of sources.
    """

    # infer white matter tracks
    wmt = infer_wm_tracks(
        sources,
        parcel,
        atlas,
        consistency_thresh,
        length_thresh,
    )

    # interpolate gray matter tracks
    gmt = interpol_gm_tracks(sources, dmax)

    # combine tracks
    sc = wmt + gmt 
    sc = sc.astype(bool).astype(int)

    # compute graph fourier transform (30 s)
    coords = [s["rr"][s["vertno"]] for s in sources]
    G = pygsp.graphs.Graph(sc.todense(), lap_type="normalized", coords=coords)
    G.compute_fourier_basis()
    cst = G.U.T

    return cst