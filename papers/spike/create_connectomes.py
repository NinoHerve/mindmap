'''
This script generate atlas-based connectomes and Connectome Spectrum Transforms (CST) for every subject of the SPIKE dataset.

Prerequisits
------------
- Parcellations for every subject (run `create_parcellations.py`)
- Source spaces for every subject (run `create_source_space.py`)

Parameters
----------
- consistency_thr: Threshold for the consistency parameter of the atlas.
- length_thr: Threshold for the length parameter of the atlas.
- number_thr: Threshold for the number of fiber parameter of the atlas.
- distance_thr: Maximum distance to connect neighbor brain sources.
- overwrite: Whether to overwrite existing connectomes.

Outputs
-------
- {mm_dir}/{sub}/sc/{sub}_sc.npy: Connectome
- {mm_dir}/{sub}/sc/{sub}_cst.npy

Notes
-----
- Subjects that won't work.
- For more detailed information or to run step by step, see `mindmap/tutorials/connectomic/CST_from_template.ipynb`.
'''


import mindmap as mm
from pathlib import Path
import numpy as np


###################### Parameters #######################
# atlas
consistency_thr = 40
length_thr = 25
number_thr = 20

# gm tracts
distance_thr = 0.010

# other options
overwrite = True
########################################################## 

def main(consistency_thr, length_thr, number_thr, distance_thr, overwrite):
    # configuration
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]
    mm_dir   = reader.config["MM_DIR"]

    # file naming
    fsc  = lambda sub: mm_dir/sub/"sc"/f"{sub}_sc.npy"
    fcst = lambda sub: mm_dir/sub/"sc"/f"{sub}_cst.npy"


    for sub in subjects:
        # check file existence 
        sc_fname  = fsc(sub)
        cst_fname = fcst(sub)

        if cst_fname.exists() and not overwrite:
            print(f"\033[93mConnectome spectrum transform already exists for subject {sub}\033[0m")
            continue
        elif not cst_fname.parent.exists():
            cst_fname.parent.mkdir(parents=True)
    
        try:
            # Generate CST
            cst, sc = mm.utils.create_cst_from_template(sub, consistency_thr, length_thr, number_thr, distance_thr)
        except Exception as e:
            print(f"\033[91mFailed to generate CST for subject {sub} due to: {e}\033[0m")
            continue

        # save
        np.save(sc_fname, sc.todense())
        np.save(cst_fname, cst)

        print(f"\033[92mSuccessfully computed CST for subject '{sub}'\033[0m")


if __name__ == "__main__":
    main(consistency_thr, length_thr, number_thr, distance_thr, overwrite)