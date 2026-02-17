"""
Module Name: anatomy/segmentation/pipeline.py
Description: Functions to run segmentation pipelines.

This module provides functions to run brain tissue segmentations.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from pathlib import Path
import shutil
import subprocess

from mindmap.io import check_file

def run_charm(
        subject   : str,
        t1_file   : str | Path,
        t2_file   : str | Path = None,
        overwrite : bool = False
        ):
    """
    Run the charm command line. The charm comand line generates a head
    segmentation from a T1 image (optional T2 for better reconstruction)

    Parameters
    ----------
    folder: Path | str
        Folder in which to generate the output.

    subject: str
        Subject ID.

    t1_file: Path | str
        Path to T1 image.

    t2_file: Path | str
        Path to T2 image.

    overwrite: bool
        Overwrite existing files if True.

    Files
    ------
    m2m_subID: folder
        Folder containing charm segmentation.
        - m2m_subId/segmentation/labeling.nii.gz: Segemented image.
        - m2m_subId/segmentation/labeling_LUT.txt: CLUT of image.
        - ...
    """
    # check files
    t1 = check_file(t1_file, extension = [".nii",".gz"], instance = "path")

    if t2_file is not None:
        t2 = check_file(t2_file, extension = [".nii", ".gz"], instance = "path")
    else:
        t2 = None

    output = t1.parent/f"m2m_sub-{subject}"/"segmentation"/"labeling.nii.gz"
    if output.exists():
        if overwrite:
            shutil.rmtree(t1.parent/f"m2m_sub-{subject}")
        else:
            raise FileExistsError(f"The file '{output}' already exists.")

    # command line for charm using T1 and T2
    if t2 is not None:
        assert t1.parent == t2.parent, "T1 and T2 must be in the same folder to run charm."

        cml = [
            "charm",
            f"sub-{subject}",
            t1.name,
            t2.name,
            "--segment",
            "--initatlas",
        ]

    # command line for charm using T1
    else:
        cml = [
            "charm",
            f"sub-{subject}",
            t1.name,
            "--segment",
            "--initatlas",
        ]

    # run subprocess
    result = subprocess.run(cml, cwd=str(t1.parent), check=True, capture_output=True)

    return result
