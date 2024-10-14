import mne 
import numpy as np
import nibabel as nib



def mask_sources(sources: mne.SourceSpaces, mask: nib.Nifti1Image, torig: np.ndarray):
    """
    Mask sources not in mask (usually gray matter mask).

    Parameters
    ----------
    sources: MNE.source_space.SourceSpace
        Sources.

    mask: nib.NIfTi1Image
        Binary mask. If a source is on a voxel with value 0 it will be masked ('inuse'=False).

    torig: np.ndarray (4x4)
        Transformation matrix to convert voxel indices to coordinates in the MRI surface 
        RAS space.

    Returns
    -------
    sources: MNE.source_space.SourceSpace
        Sources with updated 'inuse', 'use' and 'vertno' attributes.
    """
    sources = sources.copy()

    for src in sources:
        # select used sources
        inuse = src["inuse"].astype(bool)
        pos   = src["rr"][inuse]*1000                           # source positions in surface RAS space [mm]

        # place sources in voxel space
        inv_torig = np.linalg.inv(torig)
        pos_vox   = mne.transforms.apply_trans(inv_torig, pos)
        pos_vox   = np.round(pos_vox).astype(int)               # source positions in voxel space

        # keep only sources inside GM mask
        volume = mask.get_fdata()
        keep   = volume[pos_vox[:,0], pos_vox[:,1], pos_vox[:,2]].astype(bool)

        new_inuse = np.zeros_like(inuse, dtype=bool)
        new_inuse[inuse] = keep 

        # save new variables in source space    
        nuse   = np.sum(new_inuse)                              # number of used sources
        vertno = np.nonzero(new_inuse)[0]                       # vertex number of used sources

        src["inuse"]  = new_inuse
        src["nuse"]   = nuse
        src["vertno"] = vertno

    return sources



def get_src_position(sources):
    "Get source positions."
    positions = []

    for src in sources:
        positions.append(src["rr"][src["vertno"]])

    return np.concatenate(positions, axis=0)


def get_src_directions(sources):
    "Get source directions."
    positions = []

    for src in sources:
        positions.append(src["nn"][src["vertno"]])

    return np.concatenate(positions, axis=0)