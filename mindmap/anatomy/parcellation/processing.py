
import numpy as np
import pandas as pd

import nibabel as nib
import mne

from .core import Parcellation
from ...core import extract_voxel_volume

def prob2dseg(
    parc   : Parcellation,
    thresh : float,
    mask   : nib.Nifti1Image = None
) -> Parcellation:
    """
    Convert a probabilistic parcellation to a deterministic parcellation.

    Parameters
    ----------
    thresh: float
        Threshold for probabilities to be considered.

    mask: nib.Nifti1Image
        Restricted region to define the segmentation.
    """
    # check inputs
    volume = parc.volume.copy()
    assert len(volume.shape) == 4, "This parcelation is not a probabilistic parcellation (requires 4D volume). Cancel prob2dseg."

    volume[volume<thresh] = 0
    volume[volume>1] = 1            # Is it ever bigger than 1?

    #  probabilistic to deterministic
    idx = np.where(np.sum(volume, axis=3)==0)       # voxels with 0 probability of being thalamus nuclei
    new_volume = volume.argmax(axis=3) + 1          # assign id according to max probability
    new_volume[idx] = 0                             # replace non thalamic voxels with 0

    # apply mask
    if mask is not None:
        mask = mask.get_fdata().astype(bool)
        new_volume[~mask] = 0

    image = nib.Nifti1Image(new_volume.astype(float), parc.affine)
    parcellation = Parcellation(image, parc.get_clut())

    return parcellation


def compute_volumes(parc: Parcellation):
    """
    Compute volume in [mm³] of each brain region.
    """
    # compute volume per region
    voxels_per_region = np.bincount(parc.volume.ravel())
    voxel_volume      = extract_voxel_volume(parc.affine)
    volume_per_region = voxel_volume * voxels_per_region

    # keep only existing regions
    parc.info["Volume"] = volume_per_region[parc.info["Region ID"]]


def correct_labels(labels:list, source_space:mne.SourceSpaces, parcel:Parcellation) -> list:
    """
    Correct labels that misplace left and right hemisphere dipoles because the parcellation is not
    perfect.
    """
    # map to switch from label id to label name
    clut = parcel.get_clut()
    map_ = pd.Series(clut["Label name"].values, index=clut["#No."]).to_dict()

    label_names = [map_[id] for id in labels]

    # left/right hemisphere seperation
    split = len(source_space[0]["vertno"])

    # switch 'lh' labels in right hemisphere to 'rh' homologue and vice virsa
    label_names[:split] = [l.replace("-rh-", "-lh-") for l in label_names[:split]]
    label_names[split:] = [l.replace("-lh-", "-rh-") for l in label_names[split:]]

    # convert back to lable ids
    new_labels = [list(map_.keys())[list(map_.values()).index(n)] for n in label_names]

    return new_labels


def get_position(parcel : Parcellation, rois : int | list[int] = None):
    """
    Get position of regions of interest. The position is defined as the center of mass of 
    all voxels belonging to the region. 
    """
    if rois is None:
        rois = parcel.clut["#No."]
    if (not isinstance(rois, list)) and (not isinstance(rois, np.ndarray)):
        rois = [rois]
    if isinstance(rois[0], str):
        rois = parcel.name2id(rois)
    
    positions = []

    for r in rois:
        voxels = np.argwhere(parcel.volume == r)
        pos    = np.mean(voxels, axis=0)
        pos    = nib.affines.apply_affine(parcel.affine, pos)

        positions.append(pos)

    return np.vstack(positions)
