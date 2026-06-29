"""
Module Name: anatomy/segmentation/pipeline.py
Description: Functions to run segmentation pipelines.

This module provides functions to run brain tissue segmentations.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import os
import shutil
import subprocess
from pathlib import Path

def run_charm(
        subject: str,
        t1_file: str | Path,
        t2_file: str | Path = None,
        charm_dir: str | Path = None,
        fs_dir: str | Path = None,
        overwrite: bool = False
        ):
    """
    Run the charm command line. The charm comand line generates a head
    segmentation from a T1 image (optional T2 for better reconstruction)

    Parameters
    ----------
    subject: str
        Subject ID.

    t1_file: Path | str
        Path to T1 image.

    t2_file: Path | str
        Path to T2 image.

    charm_dir: Path | str
        CHARM output directory.

    overwrite: bool
        Overwrite existing files if True.

    Files
    ------
    m2m_subID: folder
        Folder containing charm segmentation.
        - {output_dir}/segmentation/labeling.nii.gz: Segemented image.
        - {output_dir}/segmentation/labeling_LUT.txt: CLUT of image.
        - ...
    """
    t1 = Path(t1_file).resolve()
    t2 = Path(t2_file).resolve() if t2_file else None

    # Define where output should go (default to T1 folder)
    work_dir = Path(charm_dir).resolve()/f"sub-{subject}" if charm_dir else t1.parent
    m2m_folder = work_dir/f"m2m_sub-{subject}"

    # Handle overwrite
    if m2m_folder.exists():
        if overwrite:
            shutil.rmtree(m2m_folder)
        else:
            raise FileExistsError(f"SimNIBS output {m2m_folder} already exists.")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Create Symlinks inside the workspace
    # SimNIBS needs the files in the CWD to find them
    t1_sym = work_dir/t1.name
    if t1_sym.exists():
        t1_sym.unlink()
    os.symlink(t1, work_dir/t1.name)

    if t2:
        t2_sym = work_dir/t2.name
        if t2_sym.exists():
            t2_sym.unlink()
        os.symlink(t2, work_dir/t2.name)

    # build the command
    cml = ["charm", f"sub-{subject}", t1.name]
    if t2:
        cml.append(t2.name)
    if fs_dir:
        cml.append(f"--fs-dir {fs_dir/subject}")
    cml.extend(["--segment", "--initatlas"])

    # run subprocess inside the work_dir
    try:
        result = subprocess.run(
            cml,
            cwd = str(work_dir),
            check = True,
            capture_output = True,
            text = True,
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"CHARM failed with error:\n{e.stderr}")
        raise
