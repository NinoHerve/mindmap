from pathlib import Path
import numpy as np 
import nibabel as nib 
import pyvista as pv
import yaml
 


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


def write_config(fname, config):
    """
    Write configuration file.
    """
    for key in config.keys():
        if isinstance(config[key], Path):
            config[key] = str(config[key])

    with open(fname, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)