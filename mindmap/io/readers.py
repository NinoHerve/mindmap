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
from ..utils.misc import _retrieve_env



##########################################################
#################                     ####################
################# INDEPENDENT READERS ####################
#################                     ####################
##########################################################


def read_config(file: str | Path, env: bool = True) -> dict[str, Any]:
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
        if env:
            os.environ[key] = str(value)
        if isinstance(value, str) and "/" in value:
            config[key] = Path(value)

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


def read_mri(file, mri_dtype="nib"):
    if mri_dtype.lower() == "nib":
        return nib.load(file)
    elif mri_dtype.lower() == "ants":
        return ants.image_read(str(file))
    else:
        raise ValueError(f"Unknown mri file type")


##########################################################
####################                  ####################
#################### DATASET READER   ####################
####################                  ####################
##########################################################

class DatasetReader():
    def __init__(self, config=None):
        if isinstance(config, str):
            config = read_config(config)

        self.config = config

    def __call__(self,):
        raise NotImplementedError("Implement __call__ in child.")


    def read_freesurfer(self, subject, *args, mri_dtype="nib"):
        """
        Read data inside FreeSurfer directory.

        Parameters:
        ----------
        subject: str
            Subject name.

        *args:
            Data ids:
            - T1: T1.mgz
            - T1-orig: original image
            - T1-brain: brain mask
            - white: white surface (intersection white matter and gray matter)
            - cras: Translation to go from original space to FreeSurfer space.

        mri_dtype: str
            Type for the mri dataobject (nibabel (nib) or ants)

        Returns:
        -------
        data: dict
            Dictionnary of data using data ids as keys.
        """
        fs_dir = self.config["FS_DIR"]

        data = dict()

        for key in args:
            if key.lower() == "t1":
                # T1 FreeSurfer conformed image
                file = fs_dir/subject/"mri"/"T1.mgz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            if key.lower() == "t1-orig":
                # T1 original image
                file =  fs_dir/subject/"mri"/"orig"/"001.mgz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            if key.lower() == "t1-brain":
                # skull stripped T1 MRI
                file = fs_dir/subject/"mri"/"brain.mgz"
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            if key.lower() == "white":
                # White surfaced (boundary between white and gray matter)
                fileL = fs_dir/subject/"surf"/"lh.white"
                fileR = fs_dir/subject/"surf"/"rh.white"

                whiteL = nib.freesurfer.io.read_geometry(fileL)
                whiteR = nib.freesurfer.io.read_geometry(fileR)
                data[key] = [whiteL, whiteR]

            if key.lower() == "cras":
                # Translation part of the Talairach transform
                file = fs_dir/subject/"mri"/"transforms"/"talairach.lta"
                data[key] = read_talairach(file)[:3,3]

            if key.lower() == "seg":
                # Head segmentation (charm)
                fseg = fs_dir/subject/"mri"/f"m2m_{subject}/segmentation/labeling.nii.gz"
                flut = fs_dir/subject/"mri"/f"m2m_{subject}/segmentation/labeling_LUT.txt"
                data[key] = mm.utils.Parcellation(fseg, flut)
                
        return data


    def read_mindmap(self, subject, *args, sample=None):
        """
        Read data inside Mindmap directory.

        Parameters
        ----------
        subject: str
            Subjecte name.

        *args:
            Data ids:
            - parc    : Parcellation
            - ctx     : Cortical mask
            - sc      : Structural Connectome
            - cst     : Connectome Spectrum Transform
            - eeg     : Preprocessed EEG
            - trans   : Transformation for coregistering EEG
            - src     : Source space (after forward solution)
            - src_raw : Source space (before forward solution)
            - bem     : Head conductor model
            - bem-sim : Head conductor model to generate simulations
            - fwd     : Forward operator
            - cset*   : CSET solutions
            - sloreta*: sLORETA solutions
            - wCSET*  : Weighted CSET solutions
            - eNET*   : Elastic Net solutions
            The * indicates the lambda parameter applied.

        Returns
        -------
        data: dict
            Dictionnary pf data using data ids as keys.
        """
        mm_dir = self.config["MM_DIR"]

        data = dict()
        
        for key in args:
            if key.lower() == "parc":
                file = mm_dir/subject/"parc"/f"{subject}_multiConn4-dseg.nii.gz"
                clut = mm_dir/subject/"parc"/f"{subject}_multiConn4-dseg.txt"
                data[key] = mm.utils.Parcellation(file, clut)

            if key.lower() == "ctx":
                d    = read_mindmap(subject, "parc")
                clut = d["parc"].get_clut()
                ctx  = clut[clut["Label name"].str.startswith("ctx")]
                data[key] = d["parc"].create_mask(ctx["#No."].values)

            if key.lower() == "sc":
                file1 = mm_dir/subject/"sc"/f"{subject}_sc-tracto.npy"
                file2 = mm_dir/subject/"sc"/f"{subject}_sc-template.npy"

                if file1.exists():
                    data[key] = np.load(file1)
                else:
                    data[key] = np.load(file2)

            if key.lower() == "cst":
                file = mm_dir/subject/"sc"/f"{subject}_cst.npy"
                data[key] = np.load(file, allow_pickle=True)

            if key.lower() == "eeg":
                if sample is None: raise ValueError("Require sample name to read EEG.")
                file = mm_dir/subject/"eeg"/f"{subject}-{sample}_ave.fif"
                data[key] = mne.Evoked(file, verbose="warning")

            if key.lower() == "trans":
                file = mm_dir/subject/"eeg"/f"{subject}_eeg-trans.fif"
                data[key] = mne.read_trans(file)

            if key.lower() == "src":
                file = mm_dir/subject/"fwd"/f"{subject}_fwd.fif"
                if file.exists():
                    fwd = mne.read_forward_solution(file, verbose="warning")
                    data[key] = fwd["src"]
                else:
                    print("\033[93mCould not find forward solution, reading src_raw instead.\033[0m")  # src can change when generating forward solution 
                    d = read_mindmap(subject, "src-raw")
                    data[key] = d["src-raw"]

            if key.lower() == "src-raw":
                file = mm_dir/subject/"src"/f"{subject}_src.fif"
                data[key] = mne.read_source_spaces(file, verbose="warning")

            if key.lower() == "hmodel":
                file = mm_dir/subject/"fwd"/f"{subject}_bem.fif"

                if file.exists():
                    d = read_mindmap(subject, "bem")
                    data[key] = d["bem"]
                else:
                    d = read_mindmap(subject, "fem")
                    data[key] = d["fem"]

            if key.lower() == "hmodel-sim":
                file = mm_dir/subject/"fwd"/f"{subject}_sim_bem.fif"

                if file.exists():
                    d = read_mindmap(subject, "bem-sim")
                    data[key] = d["bem-sim"]
                else:
                    d = read_mindmap(subject, "fem-sim")
                    data[key] = d["fem-sim"]

            if key.lower() == "bem":
                file = mm_dir/subject/"fwd"/f"{subject}_bem.fif"
                data[key] = mne.read_bem_solution(file, verbose="warning")

            if key.lower() == "bem-sim":
                file = mm_dir/subject/"fwd"/f"{subject}_sim_bem.fif"
                data[key] = mne.read_bem_solution(file, verbose="warning")

            if key.lower() == "fem":
                raise NotImplementedError("FEM models not available yet")

            if key.lower() == "fwd":
                file = mm_dir/subject/"fwd"/f"{subject}_fwd.fif"
                data[key] = mne.read_forward_solution(file, verbose="warning")

            if key.lower() == "fwd-sim":
                file = mm_dir/subject/"fwd"/f"{subject}_sim_fwd.fif"
                data[key] = mne.read_forward_solution(file, verbose="warning")
            
            if key.lower().startswith("cset"):
                file = mm_dir/subject/"inv"/f"{subject}-{sample}_{key}"
                data[key] = mne.read_source_estimate(file)
                
            if key.lower().startswith("sloreta"):
                file = mm_dir/subject/"inv"/f"{subject}-{sample}_{key}"
                data[key] = mne.read_source_estimate(file)

            if key.lower().startswith("wcset"):
                pass 
            if key.lower().startswith("eNET"):
                pass 
            
        return data


    def read_charm(self, *args):
        """
        Read data inside Charm directory. 

        Parameters
        ----------
        subject: str
            Subjecte name.

        *args:
            Data ids:
            - seg: Head segmentation

        Returns
        -------
        data: dict
            Dictionnary pf data using data ids as keys.
        """
        cmp_dir = self.config["CMP_DIR"]
        pass



    def log(self, data, args):
        """
        Print state of retrieved data 
        """
        missing   = list(set(args) - set(data.keys()))
        retrieved = list(set(data.keys())-set(["name", "sub"]))
        
        print("\033[95mDATA SUMMARY\033[0m")
        print(f"Dataset: {data['name']}")
        print(f"Subject: {data['sub']}")

        print(f"Retrieved objects:")
        for r in retrieved:
            print(f"- \033[92m{r}\033[0m")

        print(f"Missing objects:")
        for a in args:
            print(f"- \033[91m{a}\033[0m")



    

def read_dataset(subject: str, *args: str, sample: str = None, mri_dtype: str = "nib", verbose: int = 1) -> dict:
    """
    Read data according to configuration. To use this function, first read a configuration file with env=True.

    Parameters
    ----------
    subject: str
        Subject name.

    *args: str
        Names of data objects to load.

    sample: str
        Specify sample if multiple samples.

    mri_file_type: str
        MRI file type -> ants or nibabel (nib)

    Returns
    -------
    data: dict(dataName: dataObject)
        Data.
    """
    dataset = _retrieve_env("NAME")

    if dataset.lower() == "spike":
        data = _read_dataset_spike(subject, *args, mri_dtype=mri_dtype)

    elif dataset.lower() == "intra":
        data = _read_dataset_intra(subject, *args)

    elif dataset.lower() == "vep":
        data = _read_dataset_vep(subject, *args)
    
    elif dataset.lower() == "vepcon":
        data = _read_dataset_vepcon(subject, *args, sample=sample)

    elif dataset.lower() == "simf":
        data = _read_dataset_simf(subject, *args, sample=sample, mri_dtype=mri_dtype)

    else: 
        raise ValueError(f"Unknown dataset '{dataset}' for read_dataset()")

    if verbose == 1:
        missing = list(set(args) - set(data.keys()))
        _log(data, missing)

    return data




def _log(data, args):
    """
    Print state of retrieving data 
    """
    retrieved = list(set(data.keys())-set(["name", "sub"]))
    
    print("\033[95mDATA SUMMARY\033[0m")
    print(f"Dataset: {data['name']}")
    print(f"Subject: {data['sub']}")

    print(f"Retrieved objects:")
    for r in retrieved:
        print(f"- \033[92m{r}\033[0m")

    print(f"Missing objects:")
    for a in args:
        print(f"- \033[91m{a}\033[0m")




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



def _read_dataset_spike(subject, *args, mri_dtype="nib"):
    """
    Read data for dataset SPIKE.

    Parameters
    ----------
    subject: str
        Subject name

    *args: list(str)
        Data ids:
        - eeg: EEG (looks for eeg_raw if no EEG in mindmap folder).
        - eeg_raw: EEG from the original dataset.
        - resec: Resection area
        - mri_post: Postsurgery T1w MRI.
        For the rest, see read_freesurfer and read_mindmap

    mri_dtype: str
        MRI dtype (Nibabel or ANTs) for MRI images.

    Returns:
    data: dict
        Dictionnary of data using data ids as keys.
    """
    
    env  = _retrieve_env("NAME", "DS_DIR", "MM_DIR")
    data = dict(name=env["NAME"], sub=subject)

    # Others
    for name in args:
            
        if name.lower() == "eeg":
            file = env["MM_DIR"]/subject/"eeg"/f"{subject}-spike_ave.fif"
            if file.exists():
                continue    # will read with read_mindmap() 
            else:
                print(f"\033[93mPreprocessed EEG not found, reading raw instead.\033[93m")
                d = _read_dataset_spike(subject, "eeg-raw")
                data[name] = d["eeg-raw"]

        if name.lower() == "eeg-raw":
            folder = env["DS_DIR"]/"derivatives"/"cartool-v3.80"/subject/"ses-preop"/"eeg"
            data[name] = read_eeg(folder, dataset=env["NAME"])
        
        if name.lower() == "resec":
            # mask of resected area
            manual = env["MM_DIR"]/subject/"resec"/f"{subject}_man-recon-seg.nii.gz"
            auto   = env["MM_DIR"]/subject/"resec"/f"{subject}_auto-seg.nii.gz"

            if manual.exists() == True:
                data["resec"] = nib.load(manual)
            elif auto.exists() == True:
                data["resec"] = nib.load(auto)
            else:
                continue

        if name.lower() == "mri_post":
            file = env["DS_DIR"]/"derivatives"/"cmp-v3.0.0-beta-RC1"/subject/"ses-postop"/"anat"/f"{subject}_ses-postop_space-individual_desc-brain_T1w.nii.gz"
            data[name] = read_mri(file, mri_dtype=mri_dtype)

    args = list(set(args) - set(data.keys()))    # update missing arguments

    # FreeSurfer
    d    = read_freesurfer(subject, *args, mri_dtype=mri_dtype) 
    data.update(d)                                                 
    args = list(set(args) - set(data.keys()))                       

    # Mindmap
    d   = read_mindmap(subject, *args, sample="spike")
    data.update(d)
    args = list(set(args) - set(data.keys()))

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


def _read_dataset_simf(subject, *args, sample=None, mri_dtype="nib"):
    """
    Read data for dataset SIMF.

    Parameters
    ----------
    subject: str
        Subject name

    *args: list(str)
        Data ids:
        - eeg: EEG (looks for eeg_raw if sample is None)
        - eeg_raw: Coregistered EEG with zeros if exists, else not coregistered EEG with zeros
        - trans: Identity transformation to coregister EEG
        - tracto: Tractography
        - t12diff: Transformation from T1 to diffusion space
        - gt: Ground truth
        For the rest, see read_freesurfer and read_mindmap

    sample: str
        Sample name for EEG.

    mri_dtype: str
        MRI dtype (Nibabel or ANTs) for MRI images.

    Returns:
    data: dict
        Dictionnary of data using data ids as keys.
    """
    env  = _retrieve_env("NAME", "MM_DIR", "CMP_DIR", "SIM_DIR")

    data = dict(name=env["NAME"], sub=subject)

    if isinstance(sample, int):
        sample = f"sample{sample:03}"

    # dataset specifics (use this section to also overwrite freesurfer and mindmap)
    for name in args:
        if name.lower() == "eeg":
            if sample is None:
                # look for raw EEG
                d = _read_dataset_simf(subject, "eeg-raw")
                data[name] = d["eeg-raw"]
            else:
                file = env["SIM_DIR"]/"eeg"/f"{subject}-{sample}_ave.fif"
                data[name] = mne.Evoked(file, verbose="warning")

        if name.lower() == "eeg-raw":
            file = env["MM_DIR"]/subject/"eeg"/f"{subject}_coreg-ave.fif"

            if file.exists():
                print("\033[93mCoregistered EEG found\033[0m")
                data[name] = mne.Evoked(file, verbose="warning")

            else:
                print("\033[93mNo coregistered EEG found, making EEG from scratch\033[0m")
                # null EEG
                # load builtin montage
                montage = mne.channels.make_standard_montage("EGI_256")
        
                # create info
                info    = mne.create_info(ch_names=montage.ch_names, ch_types="eeg", sfreq=250)
                info.set_montage(montage)

                info["description"] = "Simulation EEG"
                info["bads"] = ["E31", "E67", "E73", "E82", "E91", "E92", "E102", "E111", "E120", "E133", "E145", "E165", "E174", "E187", "E199", "E208", "E209", "E216", "E217", "E218", "E219"] + [f"E{i}" for i in range(225,257)]

                # create null data
                eeg = np.zeros((info["nchan"], int(info["sfreq"])))

                # create eeg
                eeg = mne.EvokedArray(eeg, info, verbose="warning")
                eeg.drop_channels(info["bads"])

                data[name] = eeg

        if name.lower() == "trans":
            data[name] = mne.transforms.Transform("head", "mri")    # identity transformation

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


        if name.lower() == "diff-affine":
            # Affine (diffusion -> T1)
            file   = env["CMP_DIR"]/subject/"xfm"/"final0GenericAffine.mat"
            affine = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
            data[name] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))

        if name.lower() == "gt":
            # ground truth 
            if sample is None: raise ValueError("Require sample name to read data 'gt'.")
            file = env["SIM_DIR"]/"stc"/f"{subject}-{sample}_stc.fif"
            data[name] = mne.read_source_estimate(file) 

        if name.lower() == "lam":
            # eigenvalues
            file = env["MM_DIR"]/subject/"sc"/f"{subject}_lams-template.npy"
            data[name] = np.load(file)


    args = list(set(args) - set(data.keys()))   # update list of missing arguments

    # FreeSurfer
    d    = read_freesurfer(subject, *args, mri_dtype=mri_dtype)     
    data.update(d)                                                 
    args = list(set(args) - set(data.keys()))                   

    # mindmap
    d    = read_mindmap(subject, *args, sample=sample)
    data.update(d)
    args = list(set(args) - set(data.keys()))

    return data


##########################################################
####################                  ####################
#################### SOFTWARE READERS ####################
####################                  ####################
##########################################################



def read_freesurfer(subject, *args, mri_dtype="nib"):
    """
    Read data inside FreeSurfer directory.

    Parameters:
    ----------
    subject: str
        Subject name.

    *args:
        Data ids:
        - T1: T1.mgz
        - T1-orig: original image
        - T1-brain: brain mask
        - white: white surface (intersection white matter and gray matter)
        - cras: Translation to go from original space to FreeSurfer space.

    mri_dtype: str
        Type for the mri dataobject (nibabel (nib) or ants)

    Returns:
    -------
    data: dict
        Dictionnary of data using data ids as keys.
    """
    fs_dir = _retrieve_env("FS_DIR")

    data = dict()

    for name in args:
        if name.lower() == "t1":
            # T1 FreeSurfer conformed image
            file = fs_dir/subject/"mri"/"T1.mgz"
            data[name] = read_mri(file, mri_dtype=mri_dtype)

        if name.lower() == "t1-orig":
            # T1 original image
            file =  fs_dir/subject/"mri"/"orig"/"001.mgz"
            data[name] = read_mri(file, mri_dtype=mri_dtype)

        if name.lower() == "t1-brain":
            # skull stripped T1 MRI
            file = fs_dir/subject/"mri"/"brain.mgz"
            data[name] = read_mri(file, mri_dtype=mri_dtype)

        if name.lower() == "white":
            # White surfaced (boundary between white and gray matter)
            fileL = fs_dir/subject/"surf"/"lh.white"
            fileR = fs_dir/subject/"surf"/"rh.white"

            whiteL = nib.freesurfer.io.read_geometry(fileL)
            whiteR = nib.freesurfer.io.read_geometry(fileR)
            data[name] = [whiteL, whiteR]

        if name.lower() == "cras":
            # Translation part of the Talairach transform
            file = fs_dir/subject/"mri"/"transforms"/"talairach.lta"
            data[name] = read_talairach(file)[:3,3]

    return data

def read_mindmap(subject, *args, sample=None):
    """
    Read data inside Mindmap directory.

    Parameters
    ----------
    subject: str
        Subjecte name.

    *args:
        Data ids:
        - parc    : Parcellation
        - ctx     : Cortical mask
        - sc      : Structural Connectome
        - cst     : Connectome Spectrum Transform
        - eeg     : Preprocessed EEG
        - trans   : Transformation for coregistering EEG
        - src     : Source space (after forward solution)
        - src_raw : Source space (before forward solution)
        - bem     : Head conductor model
        - bem-sim : Head conductor model to generate simulations
        - fwd     : Forward operator
        - cset*   : CSET solutions
        - sloreta*: sLORETA solutions
        - wCSET*  : Weighted CSET solutions
        - eNET*   : Elastic Net solutions
        The * indicates the lambda parameter applied.

    Returns
    -------
    data: dict
        Dictionnary pf data using data ids as keys.
    """
    mm_dir = _retrieve_env("MM_DIR")

    data = dict()
    
    for name in args:
        if name.lower() == "parc":
            file = mm_dir/subject/"parc"/f"{subject}_multiConn4-dseg.nii.gz"
            clut = mm_dir/subject/"parc"/f"{subject}_multiConn4-dseg.txt"
            data[name] = mm.utils.Parcellation(file, clut)

        if name.lower() == "ctx":
            d    = read_mindmap(subject, "parc")
            clut = d["parc"].get_clut()
            ctx  = clut[clut["Label name"].str.startswith("ctx")]
            data[name] = d["parc"].create_mask(ctx["#No."].values)

        if name.lower() == "sc":
            file1 = mm_dir/subject/"sc"/f"{subject}_sc-tracto.npy"
            file2 = mm_dir/subject/"sc"/f"{subject}_sc-template.npy"

            if file1.exists():
                data[name] = np.load(file1)
            else:
                data[name] = np.load(file2)

        if name.lower() == "cst":
            file = mm_dir/subject/"sc"/f"{subject}_cst.npy"
            data[name] = np.load(file, allow_pickle=True)

        if name.lower() == "eeg":
            if sample is None: raise ValueError("Require sample name to read EEG.")
            file = mm_dir/subject/"eeg"/f"{subject}-{sample}_ave.fif"
            data[name] = mne.Evoked(file, verbose="warning")

        if name.lower() == "trans":
            file = mm_dir/subject/"eeg"/f"{subject}_eeg-trans.fif"
            data[name] = mne.read_trans(file)

        if name.lower() == "src":
            file = mm_dir/subject/"fwd"/f"{subject}_fwd.fif"
            if file.exists():
                fwd = mne.read_forward_solution(file, verbose="warning")
                data[name] = fwd["src"]
            else:
                print("\033[93mCould not find forward solution, reading src_raw instead.\033[0m")  # src can change when generating forward solution 
                d = read_mindmap(subject, "src-raw")
                data[name] = d["src-raw"]

        if name.lower() == "src-raw":
            file = mm_dir/subject/"src"/f"{subject}_src.fif"
            data[name] = mne.read_source_spaces(file, verbose="warning")

        if name.lower() == "hmodel":
            file = mm_dir/subject/"fwd"/f"{subject}_bem.fif"

            if file.exists():
                d = read_mindmap(subject, "bem")
                data[name] = d["bem"]
            else:
                d = read_mindmap(subject, "fem")
                data[name] = d["fem"]

        if name.lower() == "hmodel-sim":
            file = mm_dir/subject/"fwd"/f"{subject}_sim_bem.fif"

            if file.exists():
                d = read_mindmap(subject, "bem-sim")
                data[name] = d["bem-sim"]
            else:
                d = read_mindmap(subject, "fem-sim")
                data[name] = d["fem-sim"]

        if name.lower() == "bem":
            file = mm_dir/subject/"fwd"/f"{subject}_bem.fif"
            data[name] = mne.read_bem_solution(file, verbose="warning")

        if name.lower() == "bem-sim":
            file = mm_dir/subject/"fwd"/f"{subject}_sim_bem.fif"
            data[name] = mne.read_bem_solution(file, verbose="warning")

        if name.lower() == "fem":
            raise NotImplementedError("FEM models not available yet")

        if name.lower() == "fwd":
            file = mm_dir/subject/"fwd"/f"{subject}_fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")

        if name.lower() == "fwd-sim":
            file = mm_dir/subject/"fwd"/f"{subject}_sim_fwd.fif"
            data[name] = mne.read_forward_solution(file, verbose="warning")
         
        if name.lower().startswith("cset"):
            file = mm_dir/subject/"inv"/f"{subject}-{sample}_{name}"
            data[name] = mne.read_source_estimate(file)
            
        if name.lower().startswith("sloreta"):
            file = mm_dir/subject/"inv"/f"{subject}-{sample}_{name}"
            data[name] = mne.read_source_estimate(file)

        if name.lower().startswith("wcset"):
            pass 
        if name.lower().startswith("eNET"):
            pass 
        
    return data


##########################################################
####################                  ####################
#################### DATASET READERS  ####################
####################                  ####################
##########################################################


##########################################################
####################                  ####################
####################   ATLAS READERS  ####################
####################                  ####################
##########################################################





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
