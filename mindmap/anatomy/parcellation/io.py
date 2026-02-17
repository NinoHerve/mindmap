
from pathlib import Path
import re

import numpy as np
import pandas as pd

import nibabel as nib

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
        with open(fname, "r", encoding="utf-8") as f:
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