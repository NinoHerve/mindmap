import numpy as np 
import nibabel as nib 
import pyvista as pv
from pathlib import Path
import pybv
import mindmap as mm
import mne 
import pandas as pd
 


def write_surface(fname: Path|str, surf: pv.PolyData|tuple[np.ndarray], meta=None):
    """
    Write surface into FreeSurfer .surf format.

    Parameters
    ----------
    fname: path-like
        File name.

    surf: pv.PolyData|tuple[np.ndarray]
        Surface. In Pyvista format or a tuple containing vertices and faces.

    meta: ???
        Volume info.
    """
    if isinstance(surf, tuple|list):
        vertices, faces = surf 

    elif isinstance(surf, pv.PolyData):
        vertices = surf.points
        faces = surf.faces 

        # rewrite faces into right format
        idx = np.arange(0, len(faces), step=4)
        faces = np.delete(faces, idx).reshape(-1,3)

    else:
        raise ValueError("Unknown surface format to write.")

    nib.freesurfer.io.write_geometry(fname, vertices, faces, volume_info=meta)   


def write_stc(
        fname: str, 
        folder: str | Path,
        stc: mne.SourceEstimate, 
        src: mne.SourceSpaces, 
        trans: mne.Transform, 
        overwrite: bool = False
):
    """
    Write source time series in brain vision format.

    Parameters:
    fname: str
        Base file name.
    
    folder: path-like 
        Path to folder.

    stc: mne.SourceEstimate
        Source time series.

    src: mne.SourceSpaces
        Sources.

    trans: mne.Transform
        Source transform.

    overwrite: bool
        Whether to overwrite existing files or not.
    """
    folder = Path(folder)
    if folder.exists() == False:
        folder.mkdir()

    # save source time serie in brain vision format
    names = range(1, stc.data.shape[0]+1)
    pybv.write_brainvision(
        data       = stc.data, 
        sfreq      = stc.sfreq,
        ch_names   = range(1, stc.data.shape[0]+1),
        fname_base = fname,
        folder_out = folder,
        overwrite  = overwrite,
    )

    # convert source position and orientation to the right space
    points = mm.utils.get_src_position(src)
    points = mne.transforms.apply_trans(trans, points)*1000

    orient = mm.utils.get_src_directions(src)
    orient = orient @ trans["trans"][:3,:3]

    # write source position and orientation
    data = np.hstack([np.expand_dims(names, axis=1), points, orient])
    data = pd.DataFrame(data, columns=["id", "x", "y", "z", "nx", "ny", "nz"])
    data["id"] = data["id"].astype(int)
    data.to_csv(folder/f"{fname}.csv")