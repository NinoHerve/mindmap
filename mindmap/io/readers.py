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

import mindmap as mm
from mindmap.utils.misc import _retrieve_path


def read_config(file: str | Path, env: bool = False) -> Optional[dict[str, Any]]:
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

    if env == True:
        for key, value in config.items():
            os.environ[key] = value 

    else:
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


# -------------------------------
# ******* DATASET READER ********
# -------------------------------

def read_dataset(subject: str, *args: str) -> dict:
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
    dataset = _retrieve_path("NAME")
    assert dataset.lower() in ["spike", "intra", "vep"], f"Unknown dataset '{dataset}'"
    
    data = dict()

    if dataset.lower() == "spike":
        dirs   = _retrieve_path("DS_DIR", "FS_DIR", "MM_DIR")
        ds_dir = Path(dirs["DS_DIR"])
        fs_dir = Path(dirs["FS_DIR"])
        mm_dir = Path(dirs["MM_DIR"])

        for name in args:
            if name.lower() == "t1":
                # T1 MRI
                file =  fs_dir/subject/"mri"/"orig"/"001.mgz"
                data[name] = nib.load(file)

            if name.lower() == "t1-ants":
                # T1 MRI loaded with ants
                file = fs_dir/subject/"mri"/"orig"/"001.mgz"
                data[name] = ants.image_read(str(file))
            
            if name.lower() == "eeg_raw":
                # Initial EEG
                folder = ds_dir/"derivatives"/"cartool-v3.80"/subject/"ses-preop"/"eeg"
                data[name] = read_eeg(folder, dataset=dataset)

            if name.lower() == "eeg_clean":
                # EEG after cleaning
                file = mm_dir/subject/"eeg"/f"{subject}_clean-ave.fif"
                data[name] = mne.read_evokeds(file, verbose="warning")[0]

            if name.lower() == "eeg_coreg":
                # EEG after cleaning + coregistered to head
                file = mm_dir/subject/"eeg"/f"{subject}_coreg_ave.fif"
                data[name] = mne.read_evokeds(file, verbose="warning")[0]

            if name.lower() == "cst":
                # Connectome Spectrum Transform
                file = mm_dir/subject/"cst"/f"{subject}_cst-template.npy"
                data[name] = np.load(file)

            if name.lower() == "bem":
                # Boundary Element Model (contains surfaces)
                file = mm_dir/subject/"fwd"/f"{subject}_hm-bem.fif"
                data[name] = mne.read_bem_surfaces(file, verbose="warning")

            if name.lower() == "fwd":
                # Forward solution (a.k.a lead field matrix)
                file = mm_dir/subject/"fwd"/f"{subject}_bem_surf-7mm_fwd.fif"
                data[name] = mne.read_forward_solution(file, verbose="warning")

            if name.lower() == "parc":
                # Brain parcellation (MultiConn Atlas)
                file = mm_dir/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
                clut = mm_dir/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
                data[name] = mm.utils.Parcellation(file, clut)

            if name.lower() == "gm":
                # Gray matter mask
                file = mm_dir/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
                clut = mm_dir/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
                parc = mm.utils.Parcellation(file, clut)
                data[name] = parc.create_mask()
            
            if name.lower() == "ctx":
                # Cortical mask
                file = mm_dir/subject/"parc"/f"{subject}_multiConn4-2-0.05_dparc.nii.gz"
                clut = mm_dir/subject/"parc"/f"{subject}_multiConn4-2-0.05_clut.txt"
                parc = mm.utils.Parcellation(file, clut)

                df  = parc.get_clut()
                ctx = df[df["Label name"].str.startswith("ctx")]
                data[name] = parc.create_mask(ctx["#No."].values)

            if name.lower() == "src":
                # Brain sources (distributed dipoles)
                file = mm_dir/subject/"src"/f"{subject}_ddm-surf-7-src.fif"
                data[name] = mne.read_source_spaces(file, verbose="warning")

            if name.lower() == "white":
                # White surfaced (boundary between white and gray matter)
                fileL = fs_dir/subject/"surf"/"lh.white"
                fileR = fs_dir/subject/"surf"/"rh.white"

                whiteL = nib.freesurfer.io.read_geometry(fileL)
                whiteR = nib.freesurfer.io.read_geometry(fileR)
                data[name] = [whiteL, whiteR]


    elif dataset.lower() == "intra":
        pass 

    elif dataset.lower() == "vep":
        dirs   = _retrieve_path("DS_DIR", "FS_DIR", "MM_DIR")
        #ds_dir = Path(dirs["DS_DIR"])
        fs_dir = Path(dirs["FS_DIR"])
        #mm_dir = Path(dirs["MM_DIR"])

        for name in args:
            if name.lower() == "t1":
                # T1 MRI template (Colin27)
                file =  fs_dir/"Colin27"/"mri"/"orig"/"001.mgz"
                data[name] = nib.load(file)

            if name.lower() == "t1-ants":
                # T1 MRI loaded with ants
                file = fs_dir/"Colin27"/"mri"/"orig"/"001.mgz"
                data[name] = ants.image_read(str(file))

            if name.lower() == "parc":
                # Brain parcellation (VEP atlas)
                file = fs_dir/"Colin27"/"mri"/"aparc+aseg.vep.mgz"
                clut = fs_dir/"Colin27"/"label"/"VepFreeSurferColorLut.txt"
                data[name] = mm.utils.Parcellation(file, clut)

            if name.lower() in ["eeg_dum", "eeg_clean"]:        # to fit standardization of mindmap
                # Dummy eeg (zero signal)
                montage  = mne.channels.make_standard_montage("biosemi64")
                ch_names = montage.ch_names
                ch_types = ["eeg"] * len(ch_names)
                sfreq    = 200 

                info = mne.create_info(ch_names, sfreq, ch_types)
                info.set_montage(montage)

                recordings = np.zeros((len(ch_names), sfreq*10))
                data[name] = mne.io.RawArray(recordings, info, verbose="warning")
                


    if len(data.keys()) < len(args):
        print("\033[93mSome arguments did not correpond to the list of available objects to load. Check the list of arguments again.")

    return data


# -------------------------------
# ****** STANDARDS READER *******
# -------------------------------

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
    
    dataset = _retrieve_path("NAME")
    assert dataset.lower() in ["spike", "intra", "vep"], f"Unknown dataset '{dataset}'"
    
    data = dict()

    if dataset.lower() == "spike":
        for name in args:
            if name.lower() == "conn":
                # Connectome atlas (h5 format)
                file = _retrieve_path("CONN")
                data[name] = h5py.File(file)

            if name.lower() == "ctx":
                # Cortical atlas (lausanne 2018 file paths)
                data[name] = _retrieve_path("LAUSL", "LAUSR")

            if name.lower() == "thal":
                # Thalamus Nuclei atlas (ants)
                file = _retrieve_path("THAL")
                data[name] = ants.image_read(file)

            if name.lower() == "thal_clut":
                # Thalamus Nuclei color look up table (file path)
                data[name] = _retrieve_path("THAL_CLUT")

            if name.lower() == "mni":
                # MNI template (MNI152 NLIN 2009) (ants)
                file = _retrieve_path("MNI")
                data[name] = ants.image_read(file)


    elif dataset.lower() == "intra": 
        pass 

    elif dataset.lower() == "vep":
        pass

    if len(data.keys()) < len(args):
        print("\033[93mSome arguments did not correpond to the list of available objects to load. Check the list of arguments again.")

    return data