from pathlib import Path
from typing import Optional, Union, Any
import yaml
import os
import pandas as pd
import nibabel as nib
import numpy as np
import re
import mne
from ..utils.eeg import get_ch_positions
import glob
import h5py 
import ants
import ast
import scipy as sp

import mindmap as mm
from mindmap.utils.misc import _retrieve_env


def read_config(file: str | Path) -> Optional[dict[str, Any]]:
    """
    Read configuration file.

    Parameters
    ----------
    file: path-like
        Path to configuration file.

    env: bool
        If True, then the variables in the configuration file will be saved in the os 
        environment.

    Returns
    -------
    variables: dict[str, Any]
        If env is False, then the variables are returned in a dict object.
    """

    with open(file, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        os.environ[key] = str(value)

    return config
    


def read_clut(fname: str | Path) -> pd.DataFrame:
    """
    Read color lookup table.

    Parameters
    ----------
    fname: path-like
        File name. Can either be a .txt or .annot file.

    Returns
    -------
    clut: pd.DataFrame
        - #No.: str = label indices
        - Label name: str = label name
        - R: int, G: int, B: int, A: float = RGBA color code
    """

    clut = {"#No.":[], "Label name":[], "R":[], "G":[], "B":[], "A":[]}

    if str(fname).endswith(".annot"):
        annot = list(nib.freesurfer.io.read_annot(fname))
        clut["#No."] = np.arange(len(annot[2]))
        clut["Label name"] = [byte.decode("utf-8").strip() for byte in annot[2]]    # convert byte string and remove white space at the end
        clut["R"] = annot[1][:,0]
        clut["G"] = annot[1][:,1]
        clut["B"] = annot[1][:,2]
        clut["A"] = annot[1][:,3]

    else:
        with open(fname, "r") as f:
            raw_clut = f.readlines()

        # read and process line by line
        pattern = re.compile(r"^\s*\d{1,5}\s+[a-zA-Z-_0-9*.]+\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}")
        for line in raw_clut:
            if pattern.match(line):
                s = line.rstrip().split(" ")
                s = [item.strip('\t') for item in s]
                s = list(filter(None, s))     

                clut["#No."].append(int(s[0]))
                clut["Label name"].append(s[1])
                clut["R"].append(int(s[2]))
                clut["G"].append(int(s[3]))
                clut["B"].append(int(s[4]))
                clut["A"].append(float(s[5]))

    return pd.DataFrame(clut)
    

def read_eeg(
    folder: str | Path,
    dataset: str,
) -> mne.Epochs | mne.Evoked:
    """
    Read EEG recordings.

    Implemented datasets:
    - "spike" for spike dataset (returns mne.Evoked)
    - "intra" for intra-cranial (returns mne.Epochs)

    Parameters
    ----------
    fname: path-like
        EEG folder.

    dataset: str
        Dataset name. Reading is different depending on the dataset.

    Returns
    -------
    eeg: mne.Epochs | mne.Evoked depending on dataset.
    """ 
    folder = Path(folder)

    if dataset == "spike":
        file = glob.glob(str(folder/"*204*.vhdr"))[0]
        eeg  = mne.io.read_raw_brainvision(file, scale=1e-6, misc=[], verbose="error")    # reads it in [V], hence the scale. misc=[] is a hack I found so that it reads the electrodes according to metadata. Else all electrodes are misc.
        data = eeg.get_data()
        info = eeg.info

        return mne.EvokedArray(data, info)

    elif dataset == "intra":
        # files
        folder = Path(folder)
        eegs   = glob.glob(str(folder/"*_desc-cleaned_eeg.vhdr")) 
        pos    = glob.glob(str(folder/"*electrodes.tsv"))[0]

        # eeg data
        data = []
        for f in eegs:
            spike = mne.io.read_raw_brainvision(f, verbose="error")
            data.append(spike["data"][0])
        data = np.array(data)

        # info
        info = spike.info

        # electrode positions
        pos     = pd.read_csv(pos, sep="\t", header=0)
        pos     = pos[pos["name"].isin(info["ch_names"])]
        pos     = dict(zip(pos["name"].values, pos[["x","y","z"]].values/1000))
        montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head")


        epochs = mne.EpochsArray(data, info)
        epochs = epochs.set_montage(montage)

        return epochs 
    
    else:
        raise ValueError("Unknown dataset ('{}') for eeg reading.")


def read_talairach(fname: str | Path) -> np.ndarray:
    """
    Reads the Talairach transform from a talairach.lta file.
    """
    trans = np.loadtxt(fname, str, skiprows=17, max_rows=4)[:,2:5].astype(float).T 
    trans = np.vstack([trans, [0,0,0,1]])

    return trans



def read_dataset(subject: str, *args: str, sample: str = None) -> dict:
    """
    Read data according to configuration. To use this function, first read a configuration file with env=True.

    Parameters
    ----------
    subject: str
        Subject name.

    *args: str
        Names of data objects to load.

    Returns
    -------
    data: dict(dataName: dataObject)
        Data.
    """
    dataset = _retrieve_env("NAME")

    if dataset.lower() == "spike":
        data = _read_dataset_spike(subject, *args)

    elif dataset.lower() == "intra":
        data = _read_dataset_intra(subject, *args)

    elif dataset.lower() == "vep":
        data = _read_dataset_vep(subject, *args)
    
    elif dataset.lower() == "vepcon":
        data = _read_dataset_vepcon(subject, *args, sample=sample)

    elif dataset.lower() == "simf":
        data = _read_dataset_simf(subject, *args, sample=sample)

    else: 
        raise ValueError(f"Unknown dataset '{dataset}' for read_dataset()")
                

    if len(data.keys()) < len(args):
        print("\033[93mSome arguments did not correpond to the list of available objects to load. Check the list of arguments again.")

    return data



def read_standards(*args: str) -> dict:
    """
    Read standardized data (atlas, templates ...) according to configuration. To use this function, first read a configuration file with env=True.

    Parameters
    ----------
    *args: str
        Names of data objects to load.

    Returns
    -------
    data: dict(dataName: dataObject)
        Data.
    """
    
    dataset = _retrieve_env("NAME")
    
    data = dict()

    if dataset.lower() == "spike":
        data = _read_standards_spike(*args)

    elif dataset.lower() == "intra": 
        pass 

    elif dataset.lower() == "vep":
        pass

    elif dataset.lower() == "vepcon":
        data = _read_standards_vepcon(*args)

    elif dataset.lower() == "simf":
        data = _read_standards_spike(*args)

    else: 
        raise ValueError(f"Unknown dataset '{dataset}' for read_standards()")

    if len(data.keys()) < len(args):
        print("\033[93mSome arguments did not correpond to the list of available objects to load. Check the list of arguments again.")

    return data



# -------------------------------
# ******* DATASET READER ********
# -------------------------------

def _read_dataset_spike(subject, *args):
    
    env = _retrieve_env("NAME", "DS_DIR", "FS_DIR", "MM_DIR")

    data = dict()

    for name in args:
        if name.lower() == "t1":
            # T1 MRI
            file =  env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
            data[name] = nib.load(file)

        if name.lower() == "t1-ants":
            # T1 MRI loaded with ants
            file = env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
            data[name] = ants.image_read(str(file))
            
        if name.lower() == "eeg_raw":
            # Initial EEG
            folder = env["DS_DIR"]/"derivatives"/"cartool-v3.80"/subject/"ses-preop"/"eeg"
            data[name] = read_eeg(folder, dataset=env["NAME"])

        if name.lower() in ["eeg_clean", "eeg_proc"]:
            # EEG after cleaning
            file = env["MM_DIR"]/subject/"eeg"/f"{subject}_clean-ave.fif"
            data[name] = mne.read_evokeds(file, verbose="warning")[0]

        if name.lower() in ["eeg_coreg", "eeg_fwd", "eeg_inv"]:
            # EEG after cleaning + coregistered to head
            file = env["MM_DIR"]/subject/"eeg"/f"{subject}_coreg_ave.fif"
            data[name] = mne.read_evokeds(file, verbose="warning")[0]

        if name.lower() == "cst":
            # Connectome Spectrum Transform
            file = env["MM_DIR"]/subject/"cst"/f"{subject}_cst-template.npy"
            data[name] = np.load(file)

        if name.lower() == "bem":
            # Boundary Element Model (contains surfaces)
            file = env["MM_DIR"]/subject/"fwd"/f"{subject}_hm-bem.fif"
            data[name] = mne.read_bem_surfaces(file, verbose="warning")

        if name.lower() == "fwd":
            # Forward solution (a.k.a lead field matrix)
            file = env["MM_DIR"]/subject/"fwd"/f"{subject}_bem_surf-7mm_fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")

        if name.lower() == "parc":
            # Brain parcellation (MultiConn Atlas)
            file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
            clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
            data[name] = mm.utils.Parcellation(file, clut)

        if name.lower() == "gm":
            # Gray matter mask
            file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
            clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
            parc = mm.utils.Parcellation(file, clut)
            data[name] = parc.create_mask()
            
        if name.lower() == "ctx_mask":
            # Cortical mask
            file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
            clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
            parc = mm.utils.Parcellation(file, clut)

            df  = parc.get_clut()
            ctx = df[df["Label name"].str.startswith("ctx")]
            data[name] = parc.create_mask(ctx["#No."].values)

        if name.lower() == "src":
            # Brain sources (distributed dipoles)
            file = env["MM_DIR"]/subject/"fwd"/f"{subject}_bem_surf-7mm_fwd.fif"
            fwd  = mne.read_forward_solution(file, verbose="warning")
            data[name] = fwd["src"]

        if name.lower() == "white":
            # White surfaced (boundary between white and gray matter)
            fileL = env["FS_DIR"]/subject/"surf"/"lh.white"
            fileR = env["FS_DIR"]/subject/"surf"/"rh.white"

            whiteL = nib.freesurfer.io.read_geometry(fileL)
            whiteR = nib.freesurfer.io.read_geometry(fileR)
            data[name] = [whiteL, whiteR]

        if name.lower() == "cras":
            # Translation part of the Talairach transform
            file = env["FS_DIR"]/subject/"mri"/"transforms"/"talairach.lta"
            data[name] = read_talairach(file)[:3,3]

    return data


def _read_dataset_intra(subject, *args):
    raise NotImplementedError("_read_dataset_intra(subject *args)")


def _read_dataset_vep(subject, *args):
        
    env = _retrieve_env("FS_DIR", "TVB_DIR", "MM_DIR", "DS_DIR")

    data = dict()

    for name in args:
        if name.lower() == "t1":
            # T1 MRI template (Colin27)
            file =  env["FS_DIR"]/"Colin27"/"mri"/"orig"/"001.mgz"
            data[name] = nib.load(file)

        if name.lower() == "t1-ants":
            # T1 MRI loaded with ants
            file = env["FS_DIR"]/"Colin27"/"mri"/"orig"/"001.mgz"
            data[name] = ants.image_read(str(file))

        if name.lower() == "parc":
            # Brain parcellation of template (VEP atlas)
            file = env["FS_DIR"]/"Colin27"/"mri"/"aparc+aseg.vep.mgz"
            clut = env["FS_DIR"]/"VepFreeSurferColorLut.txt"
            data[name] = mm.utils.Parcellation(file, clut)

        if name.lower() == "ctx_mask":
            # Cortical mask of template (without cerebellum)
            file = env["FS_DIR"]/"Colin27"/"mri"/"aparc+aseg.vep.mgz"
            clut = env["FS_DIR"]/"VepFreeSurferColorLut.txt"
            parc = mm.utils.Parcellation(file, clut)

            df  = parc.get_clut()
            ctx = df[df["#No."] > 71000]
            ctx = ctx[~ctx["Label name"].str.contains("Cerebellar")]

            data[name] = parc.retain(ctx["#No."].values).create_mask()

        if name.lower() in ["eeg_dum", "eeg_fwd"]:
            # Dummy eeg (zero signal)
            montage  = mne.channels.make_standard_montage("biosemi64")
            ch_names = montage.ch_names
            ch_types = ["eeg"] * len(ch_names)
            sfreq    = 200 

            info = mne.create_info(ch_names, sfreq, ch_types)
            info.set_montage(montage)

            recordings = np.zeros((len(ch_names), sfreq*10))
            data[name] = mne.io.RawArray(recordings, info, verbose="warning")

        if name.lower() == "eeg_tmp":
            # Template EEG (coregistered to template head)
            file = env["MM_DIR"]/"Colin27"/"eeg"/"Colin27_coreg_null-raw.fif"
            data[name] = mne.io.read_raw(file, verbose="warning")

        if name.lower() in ["eeg_sim", "eeg_proc"]:
            # Simulated eeg (coregistered to template head)
            file = env["MM_DIR"]/subject/f"{subject}_sim-raw.fif"
            data[name] = mne.io.read_raw(file, verbose="warning")

        if name.lower() == "fwd":
            # Forward solution of template head
            file = env["MM_DIR"]/"Colin27"/"fwd"/"Colin27_bem-fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")

        if name.lower() == "src":
            # Sources of template head
            file = env["MM_DIR"]/"Colin27"/"fwd"/"Colin27_bem-fwd.fif"
            fwd  = mne.read_forward_solution(file, verbose="warning")
            data[name] = fwd["src"]

        if name.lower() == "bem":
            # Boundary Element Model of template head
            file = env["MM_DIR"]/"Colin27"/"fwd"/"Colin27_hm-bem.fif"
            data[name] = mne.read_bem_surfaces(file, verbose="warning")

        if name.lower() == "cras":
            # Translation part of the Talairach transform from template head
            file = env["FS_DIR"]/"Colin27"/"mri"/"transforms"/"talairach.lta"
            data[name] = read_talairach(file)[:3,3]

        if name.lower() == "sim_tvb":
            # Simulated brain signals (tvb)
            file = env["TVB_DIR"]/subject/"ses-03"/"VEPhypothesis"/f"{subject}_simulated_source_timeseries_run-01.npz"
            data[name] = np.load(file)

        if name.lower() == "atlas":
            # Region names
            file  = env["DS_DIR"]/"vep_atlas.tsv"
            atlas = pd.read_csv(file, sep="\t")
            data[name] = atlas

        if name.lower() == "sim":
            # Simulated brain signals (per dipole)
            try:
                tmp = mm.io.read_dataset(subject, "sim_tvb", "src", "cras", "parc", "atlas")
            except FileNotFoundError:
                raise FileNotFoundError("To load simulated data, you first need a source model. Run 2_forward_problem.ipynb to generate one.")

            # label dipoles
            pos    = mm.utils.get_src_position(tmp["src"])
            pos    = pos * 1000 + tmp["cras"]
            labels = tmp["parc"].label_points(pos)

            # map label name to idx in source signal data
            name_to_idx = dict(zip(tmp["atlas"]["region_name"], tmp["atlas"]["label"]))
            idx         = [name_to_idx[l]-1 for l in labels]            # -1 because 0 is unknown and does not exist. so signal[0] is label 1.
            sim         = tmp["sim_tvb"]["source_signal"][:,idx]

            times = tmp["sim_tvb"]["time_steps"]
            sim = mne.SourceEstimate(
                data     = sim.T,
                vertices = [tmp["src"][0]["vertno"], tmp["src"][1]["vertno"]],
                tmin     = times[0],
                tstep    = times[1]-times[0]
            )

            sim.labels = labels
            data[name] = sim

        if name.lower() == "ez":
            file = env["TVB_DIR"]/subject/"ses-03"/"VEPhypothesis"/"parameters"/f"{subject}_epileptor_parameters_run-01.tsv"
            df   = pd.read_csv(file, sep="\t")
            data[name] = ast.literal_eval(df["EZ"].values[0])

    return data


def _read_dataset_vepcon(subject, *args):
    env = _retrieve_env("NAME", "DS_DIR", "FS_DIR", "MM_DIR")

    data = dict()

    for name in args:
        if name.lower() == "fwd":
            # forward solution
            file =  env["DS_DIR"]/"derivatives"/"cscs"/subject/f"{subject}_fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")

        if name.lower() in ["eeg_raw", "eeg_fwd"]:
            # T1 MRI loaded with ants
            file = env["DS_DIR"]/"derivatives"/"eeglab-v14.1.1"/subject/"eeg"/f"{subject}_task-faces_desc-preproc_eeg.set"
            data[name] = mne.read_epochs_eeglab(file, verbose="warning")
            
        if name.lower() == "cst":
            # Connectome spectrum transform
            file = env["DS_DIR"]/"derivatives"/"cscs"/subject/f"{subject}-harmonics.npy"
            data[name] = np.load(file, allow_pickle=True)

        if name.lower() == "bem":
            # Boundary Element Surfaces
            file = env["FS_DIR"]/subject/"bem"/f"{subject}-head.fif"
            data[name] = mne.read_bem_surfaces(file, verbose="warning")

        if name.lower() == "cras":
            # Translation part of the Talairach transform
            file = env["FS_DIR"]/subject/"mri"/"transforms"/"talairach.lta"
            data[name] = read_talairach(file)[:3,3]

        if name.lower() == "t1":
            # T1 MRI
            file =  env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
            data[name] = nib.load(file)

        if name.lower() == "t1-ants":
            # T1 MRI loaded with ants
            file = env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
            data[name] = ants.image_read(str(file))

        if name.lower() == "ctx_mask":
            # Cortical mask
            file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
            clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
            parc = mm.utils.Parcellation(file, clut)

            df  = parc.get_clut()
            ctx = df[df["Label name"].str.startswith("ctx")]
            data[name] = parc.create_mask(ctx["#No."].values)

    return data


def _read_dataset_simf(subject, *args, sample=None):
    env = _retrieve_env("FS_DIR", "MM_DIR", "CMP_DIR", "HR", "SIM_DIR", "DATA", "MONTAGE")

    data = dict()

    for name in args:
        if name.lower() == "fwd":
            # forward solution for source reconstruction
            file = env["MM_DIR"]/subject/"fwd"/f"{subject}_bem-sim-fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")

        if name.lower() == "fwd_sim":
            # forward solution for simulations
            file = env["MM_DIR"]/subject/"fwd"/f"{subject}_bem-sim-fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")

        if name.lower() == "src":
            # source space 
            file = env["MM_DIR"]/subject/"fwd"/f"{subject}_bem-sim-fwd.fif"
            fwd  = mne.read_forward_solution(file, verbose="warning")
            data[name] = fwd["src"]

        if name.lower() == "sc":
            # Structural connectome
            file = env["MM_DIR"]/subject/"sc"/f"{subject}_sc-tracto.npy"
            data[name] = np.load(file) 

        if name.lower() == "cst":
            file = env["MM_DIR"]/subject/"sc"/f"{subject}_cst-tracto.npy"
            data[name] = np.load(file, allow_pickle=True)

        if name.lower() in ["eeg_dum", "eeg_fwd"]:
            # Initial EEG
            folder   = env["MONTAGE"]
            eeg      = mm.io.read_eeg(folder, dataset="spike")
            eeg.data = np.zeros((eeg.info["nchan"], int(eeg.info["sfreq"]*1)))

            data[name] = eeg

        #if name.lower() in ["eeg_dum", "eeg_fwd"]:
        #    # Dummy eeg (zero signal)
        #    montage  = mne.channels.make_standard_montage("biosemi256")
        #    ch_names = montage.ch_names
        #    ch_types = ["eeg"] * len(ch_names)
        #    sfreq    = 200 

        #    info = mne.create_info(ch_names, sfreq, ch_types)
        #    info.set_montage(montage)
        #    info["bads"] = ...

        #    recordings = np.zeros((len(ch_names), sfreq))
        #    data[name] = mne.io.RawArray(recordings, info, verbose="warning")

        if name.lower() == "eeg_coreg":
            # Coregistered dummy eeg (zero signal)
            file = env["MM_DIR"]/subject/"eeg"/f"{subject}_coreg_null-ave.fif"
            data[name] = mne.read_evokeds(file, verbose="warning")[0]

        if name.lower() == "bem":
            # Boundary Element Surfaces
            file = env["FS_DIR"]/subject/"bem"/f"{subject}-head.fif"
            data[name] = mne.read_bem_surfaces(file, verbose="warning")

        if name.lower() == "cras":
            # Translation part of the Talairach transform
            file = env["FS_DIR"]/subject/"mri"/"transforms"/"talairach.lta"
            data[name] = read_talairach(file)[:3,3]

        if name.lower() == "t1":
            # T1 MRI
            file =  env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
            data[name] = nib.load(file)

        if name.lower() == "t1-ants":
            # T1 MRI loaded with ants
            file = env["FS_DIR"]/subject/"mri"/"orig"/"001.mgz"
            data[name] = ants.image_read(str(file))

        if name.lower() == "ctx_mask":
            # Cortical mask
            file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
            clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
            parc = mm.utils.Parcellation(file, clut)

            df  = parc.get_clut()
            ctx = df[df["Label name"].str.startswith("ctx")]
            data[name] = parc.create_mask(ctx["#No."].values)

        if name.lower() == "parc":
            # parcellation
            file = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
            clut = env["MM_DIR"]/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
            data[name] = mm.utils.Parcellation(file, clut)

        if name.lower() == "white":
            # White surfaced (boundary between white and gray matter)
            fileL = env["FS_DIR"]/subject/"surf"/"lh.white"
            fileR = env["FS_DIR"]/subject/"surf"/"rh.white"

            whiteL = nib.freesurfer.io.read_geometry(fileL)
            whiteR = nib.freesurfer.io.read_geometry(fileR)
            data[name] = [whiteL, whiteR]

        if name.lower() == "tracto":
            # Tractogram
            if subject == "vepcon_sub-10":
                file = env["CMP_DIR"]/subject/"dwi"/f"sub-10_model-CSD_desc-DET_tractogram.trk"
            else:
                file = env["CMP_DIR"]/subject/"fibtracking"/"sd_stream"/"dwi_denoised_eddy_distc_corrected_1mm_shore_sdstream.trk"
            data[name] = nib.streamlines.load(file)

        if name.lower() == "t12diff":
            # Transforms from T1 to diffusion
            if subject == "vepcon_sub-10":
                file = env["CMP_DIR"]/"vepcon_sub-10"/"xfm"/"final0GenericAffine.mat"
            else:
                file = env["CMP_DIR"]/subject/"transforms"/"all_t12diff0GenericAffine.mat"
            affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
            data[name] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))


        if name.lower() == "diff_affine":
            # Affine (diffusion -> T1)
            file   = env["CMP_DIR"]/subject/"xfm"/"final0GenericAffine.mat"
            affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
            data[name] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))

        if name.lower() in ["eeg", "eeg_inv"]:
            # simulation data
            if sample is None: raise ValueError("Require sample name to read data 'eeg'.")
            file = env["SIM_DIR"]/env["DATA"]/sample/f"{subject}_{sample}_ave.fif"
            data[name] = mne.read_evokeds(file, verbose="warning")[0]

        if name.lower() == "gt":
            # ground truth 
            if sample is None: raise ValueError("Require sample name to read data 'gt'.")
            file = env["SIM_DIR"]/env["DATA"]/sample/f"{subject}_{sample}_stc.fif"
            data[name] = mne.read_source_estimate(file)

        if name.lower() == "cset":
            # reconstructed source time course using CSET
            if sample is None: raise ValueError("Require sample name to read reconstructed CSET source time course")
            file = env["MM_DIR"]/subject/"stc"/env["DATA"]/f"{subject}_{sample}_cset-0.1-lh.stc"
            data[name] = mne.read_source_estimate(file)

        if name.lower() == "sloreta":
            # reconstructed source time course using sloreta
            if sample is None: raise ValueError("Require sample name to read reconstructed sLORETA source time course")
            file = env["MM_DIR"]/subject/"stc"/env["DATA"]/f"{subject}_{sample}_sloreta-0.1-lh.stc"
            data[name] = mne.read_source_estimate(file)



    return data



# -------------------------------
# ****** STANDARDS READER *******
# -------------------------------

def _read_standards_spike(*args):

    data = dict()

    for name in args:
        if name.lower() == "conn":
            # Connectome atlas (h5 format)
            file = _retrieve_env("CONN")
            data[name] = h5py.File(file)

        if name.lower() == "ctx":
            # Cortical atlas (lausanne 2018 file paths)
            data[name] = _retrieve_env("LAUSL", "LAUSR")

        if name.lower() == "thal":
            # Thalamus Nuclei atlas (ants)
            file = _retrieve_env("THAL")
            data[name] = ants.image_read(str(file))

        if name.lower() == "thal_clut":
            # Thalamus Nuclei color look up table (file path)
            data[name] = _retrieve_env("THAL_CLUT")

        if name.lower() == "mni":
            # MNI template (MNI152 NLIN 2009) (ants)
            file = _retrieve_env("MNI")
            data[name] = ants.image_read(str(file)) 

    return data


def _read_standards_vepcon(*args):

    data = dict() 

    for name in args:
        if name.lower() == "ctx":
            # Cortical atlas (lausanne 2018 file paths)
            data[name] = _retrieve_env("LAUSL", "LAUSR")

        if name.lower() == "thal":
            # Thalamus Nuclei atlas (ants)
            file = _retrieve_env("THAL")
            data[name] = ants.image_read(str(file))

        if name.lower() == "thal_clut":
            # Thalamus Nuclei color look up table (file path)
            data[name] = _retrieve_env("THAL_CLUT")

        if name.lower() == "mni":
            # MNI template (MNI152 NLIN 2009) (ants)
            file = _retrieve_env("MNI")
            data[name] = ants.image_read(str(file)) 

    return data
