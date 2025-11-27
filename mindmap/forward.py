"""
Module Name: forward.py
Description: Functions to solve forward problems.

This module provides functions to make boundary element models (BEM).

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



# def forward_problem(
#         subject: str,
#         spacing: float = 7,
#         conductivities: list = [0.3, 0.006, 0.3],
#         min_skull_thick: float = 2,
#         icp: bool = True,
#         n_jobs: int = 1,
#         verbose: int = 0,
# ):
#     """
#     Solves the forward problem for every subject in the dataset.

#     Parameters:
#     ----------
#         subject: str
#             Name of dataset subject.

#         spacing: float
#             Source space dipole spacing.

#         conductivities: [float]*3
#             Electrical conductivities between BEM layers.

#         min_skull_thick: float
#             Minimum skull thickness in [mm] when correcting BEM surfaces.

#         icp: bool
#             If True, apply the ICP algorithm to align electrodes and scalp.

#         n_jobs: int
#             Number of parallel jobs when generating source spaces.

#         verbose: int [0,1]
#             Level of verbosity.

#     Returns:
#     -------
#         FreeSurferDirectory/bem/inner_skull.surf
#             Inner skull surface.

#         FreeSurferDirectory/bem/outer_skull.surf
#             Outer skull surface.

#         FreeSurferDirectory/bem/outer_skin.surf
#             Outer skin surface.

#         eeg: mne.Evoked | mne.Epoched | mne.Raw
#             Coregistered EEG.

#         bem: list[dict]
#             Boundary Element Method (BEM) model.

#         forward: mne.Forward
#             Forward solution.
#     """
    
#     # Configuration
#     try:
#         fs_dir = os.environ["FS_DIR"]
#     except Exception as e:
#         raise Exception(f"Configuration error -> {e}") from e

#     # Load data
#     try: 
#         data = mm.io.read_dataset(subject, "eeg_fwd", "ctx_mask", "cras")
#     except Exception as e:
#         raise Exception(f"Load data error -> {e}") from e
    

#     ##########################
#     ###### Source Space ######
#     ##########################

#     try:
#         if verbose == 1:
#             print("\033[96mGenerating Source Space (1/4)\033[0m")

#         # surface-based
#         sources = mne.setup_source_space(
#             subject=subject,
#             spacing=spacing,
#             surface="white",
#             add_dist=True,    # geodesic distances between sources
#             subjects_dir=fs_dir,
#             n_jobs=n_jobs,
#             verbose="warning",
#         )
    
#         # gray matter constraint 
#         sources = mm.utils.mask_sources(sources, data["ctx_mask"], data["cras"])

#         if verbose == 1:
#             print("\033[92mSource space summary\033[0m")
#             for src in sources:
#                 print(f"Source space id: {src['id']}")
#                 print(f"Type: {src['type']}")
#                 print(f"Number of sources: {src['nuse']}\n")

#     except Exception as e:
#         raise Exception(f"Source space error -> {e}") from e 


#     ##########################
#     ####### Head Model #######
#     ##########################

#     try: 
#         if verbose == 1:
#             print("\033[96mGenerating BEM surfaces (2/4)\033[0m")

#         # watershed
#         mne.bem.make_watershed_bem(
#             subject = subject,
#             subjects_dir = fs_dir,
#             volume = "T1",
#             overwrite = True,
#             verbose = "warning",
#         )

#         # correct
#         correct_skull(
#             subject,
#             offset = min_skull_thick,
#             fs_dir = fs_dir,
#             save_arrows = True,
#         )

#         # assemble
#         head_model = mne.make_bem_model(
#             subject=subject,
#             subjects_dir=fs_dir,
#             conductivity=conductivities,
#             #ico=None,
#             verbose="warning",
#         )
        
#     except Exception as e:
#         raise Exception(f"Head model error -> {e}") from e 


#     ##########################
#     ### Electrode Alignment ##
#     ##########################

#     try:
#         if verbose == 1:
#             print("\033[96mAlign electrodes to scalp (3/4)\033[0m")

#         points = mm.utils.get_ch_positions(data["eeg_fwd"])
#         head   = mm.utils.surf_to_polydata(head_model[0]["rr"], head_model[0]["tris"])
        
#         # icp
#         if icp == True:
#             points, _, _ = mm.utils.point_to_plane_icp_with_scaling(points, head, max_iterations=1)

#         # nearest point projection
#         points = mm.utils.project_points_to_surface(points, head)

#         # update
#         channels = dict(zip(data["eeg_fwd"].info["ch_names"], points))
#         montage  = mne.channels.make_dig_montage(ch_pos=channels, coord_frame="head")
#         eeg      = data["eeg_fwd"].set_montage(montage)


#     except Exception as e:
#         raise Exception(f"Coregistration error -> {e}") from e 
    

#     ##########################
#     ######## Solution ########
#     ##########################

#     try: 
#         if verbose == 1:
#             print("\033[96mSolve volume conduction problem (4/4)\033[0m")

#         bem   = mne.make_bem_solution(head_model)
#         trans = mne.transforms.Transform(fro="head", to="mri")      # identity matrix

#         forward = mne.make_forward_solution(
#             info=eeg.info,
#             trans=trans,
#             src=sources,
#             bem=bem,
#             meg=False,
#             eeg=True,
#             verbose="warning",
#         )

#     except Exception as e:
#         raise Exception(f"Volume conduction error -> {e}") from e
    
#     return eeg, bem, forward



# def solve_volume_conduction(sub, method="bem", min_skull_thick=2, conductivities=[0.3, 0.006, 0.3]):
#     """
#     Solve the volume conduction problem.

#     Parameters
#     ----------
#     sub: subject
#         Subject name.

#     method: str
#         Type of conduction model:
#         - bem: Boundary Element Model
#         - fem: Finite Element Model

#     min_skull_thick: str
#         Minimum skull thickness in [cm]. Used to correct BEM.

#     Returns
#     -------
#     cm: mne.ConductorModel | ??FEM??
#     """

#     if method.lower() == "bem":
#         fs_dir = mm.io.readers._retrieve_env("FS_DIR")

#         mne.bem.make_watershed_bem(
#             subject = sub,
#             subjects_dir = fs_dir,
#             volume = "T1",
#             overwrite = True,
#             verbose = "warning",
#         )

#         correct_skull(
#             subject     = sub,
#             offset      = min_skull_thick,
#             fs_dir      = fs_dir,
#             save_arrows = True,
#         )

#         model = mne.make_bem_model(
#             subject=sub,
#             subjects_dir=fs_dir,
#             conductivity=conductivities,
#             #ico=None,
#             verbose="warning",
#         )

#         bem  = mne.make_bem_solution(model, verbose="warning")

#     elif method.lower() == "fem":
#         raise NotImplementedError("FEM not implemented yet.")
    
#     else: 
#         raise ValueError(f"Unknown method '{method}' for volume conduction problem.")

#     return bem