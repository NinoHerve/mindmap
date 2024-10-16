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
    fname: str | Path,
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
        EEG file name or folder name.

    dataset: str
        Dataset name. Reading is different depending on the dataset.

    Returns
    -------
    eeg: mne.Epochs | mne.Evoked depending on dataset.
    """ 

    if dataset == "spike":
        eeg  = mne.io.read_raw_brainvision(fname, scale=1e-6, misc=[], verbose="error")    # reads it in [V], hence the scale. misc=[] is a hack I found so that it reads the electrodes according to metadata. Else all electrodes are misc.
        data = eeg.get_data()
        info = eeg.info

        return mne.EvokedArray(data, info)

    elif dataset == "intra":
        # files
        folder = Path(fname)
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