'''
This script solves the inverse problem using the sLORETA method.

Prerequisites
-------------
- Solve the forward problem.
- Process EEG.

Parameters
----------
- lam: Regularizer parameter.
- overwrite: Whether to overwrite.

Outputs
-------
- {mm_dir}/{sub}/inv/{sub}-{samp}_sloreta-{lam}.stc: Source Estimate using CSET

Notes
-----
- Subject 06, 20 cannot be processed because they do cannot be processed through FreeSurfer.
- For more detailed information, or step by step process, check `mindmap/tutorials/forward/fwd.ipynb`
'''


import mindmap as mm
from pathlib import Path


################# Parameters #################
# CSET
lam = 0.1

# other options
overwrite = True
##############################################

def main(lam, overwrite=False):
    # config
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]
    mm_dir   = reader.config["MM_DIR"]

    for sub in subjects:
        # check file existence
        fstc = mm_dir/sub/"inv"/f"{sub}-{samp}_cset-{lam}"

        if fname.exists() and not overwrite:
            print(f"\033[93mCSET solution already exists for subject {sub} with lam={lam}\033[0m")
        if not fname.parent.exists():
            fname.mkdir(parents=True)

    try: 
        data = reader(sub, "eeg", "fwd", sample=samp)
        stc  = mm.sloreta(data["eeg"], data["fwd"], lam)

        # save
        stc.save(fstc, overwrite=overwrite, verbose="warning")

    except Exception as e:
        print(f"\033[91mFailed to compute sLORETA solution for subject '{sub}', due to {e}\033[0m")
    else:
        print(f"\033[92mSuccessfully computed sLORETA inverse solution for subject '{sub}', sample '{samp}'\033[0m")

if __name__ == "__main__":
    main(lam, overwrite)