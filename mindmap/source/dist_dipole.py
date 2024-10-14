import os
import nibabel as nib 
import mne 
from typing import Dict
from ..utils.sources import mask_sources


def make_ddm(subject: str, constraint: Dict[str, nib.Nifti1Image], 
             spacing: float, n_jobs: int = 1, fs_dir: str = None):
    """
    Generate a Distributed Dipole Model (DDM) source space. Dipoles (= sources) are 
    distributed uniformly inside a constrained area.

    Parameters
    ----------
    subject: str
        The FreeSurfer subject name.

    fs_dir: str
        The FreeSurfer directory path. If None than the function will look for the FS_DIR
        os environment variable.

    constraint: dict['type':str, 'mask':nib.Nifti1Image]
        Spatial constraint. The 'type' should be one of ['surf', 'vol'].
        - If 'surf': The sources will be constrained to the inner cortical surface of 
        the subject.
        - If 'vol': The sources will be distributed uniformly inside a cube roughly the
        size of the head.
        The 'mask' should either be an binary brain volume (nib.Nifti1Image) showing the
        regions in the brain where one can find sources (usually the gray matter) or None.

    spacing: int
        Spacing between the sources, in [mm].

    n_jobs: int
        Number of parrallel jobs to boost computational speed.
    """
    if fs_dir is None:
        try:
            fs_dir = os.environ["FS_DIR"]
        except KeyError:
            raise KeyError("FS_DIR (FreeSurfer subjects directory) env variable not found. Please provide 'make_DDM' with subjects_dir or read conf file before calling 'make_DDM'.")

    if constraint["type"] == "surf":
        sources = mne.setup_source_space(
            subject=subject,
            spacing=spacing,
            surface="white",
            add_dist=True,      # geodesic distances between sources
            subjects_dir=fs_dir,
            n_jobs=n_jobs
        )
    elif constraint["type"] == "vol":
        sources = mne.setup_volume_source_space(
            subject=subject,
            pos=spacing,
            subjects_dir=fs_dir
        )
    else:
        raise ValueError(f"Unknown constraint type for DDM source space: '{constraint['type']}'")
        
    if constraint["mask"] is not None:
        sources = mask_sources(sources, constraint["mask"])

    return sources
        

    