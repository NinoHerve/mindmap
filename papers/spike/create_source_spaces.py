'''
This script generates cortical source spaces (brain sources are placed on the white/gray matter intersection only) for every subject of the 
SPIKE dataset. We also use the parcellation, in order to remove sources not belonging to gray matter areas.

Prerequisits
------------
- FreeSurfer reconstruction for all subjects.
- Parcellations for all subjects (Run `create_parcellations.py`)

Parameters
----------
- spacing: Spacing in [mm] between sources.
- overwrite: Whether to overwrite existing source spaces.
- n_jobs: Number of cores to use for the processing.

Outputs
-------
- {mm_dir}/{sub}/src/{sub}_src.fif: Cortical source space.

Notes
-----
- Subjects not working
'''


import mindmap as mm 
from pathlib import Path
import mne

########################## parameters #############################
# Source Space
spacing = 7

# Other options
overwrite = True
n_jobs    = 10
###################################################################

def main(spacing, overwrite, n_jobs):
    # config
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]

    # files & folders
    fsrc   = lambda sub: mm_dir/sub/"src"/f"{sub}_src.fif"
    fs_dir = reader.config["FS_DIR"]
    mm_dir = reader.config["MM_DIR"]

    for sub in subjects:
    
        # check file existence
        fname = fsrc(sub)
        if fname.exists() and not overwrite:
            print(f"\033[93mSurface-based source space already exists for subject {sub}\033[0m")
            continue 
        elif not fname.parent.exists():
            fname.parent.mkdir(parents=True)

        try: 
            # data
            data = reader(sub, "ctx", "cras")

            # create source space
            sources = mne.setup_source_space(
                subject=sub,
                spacing=spacing,
                surface="white",
                add_dist=True,      # geodesic distances between sources
                subjects_dir=fs_dir,
                n_jobs=n_jobs,
                verbose="warning",
            )

            # cortical constraint 
            sources = mm.utils.mask_sources(sources, data["ctx"], data["cras"])

        except Exception as e:
            print(f"\033[91mFailed to generate source space for subject {sub} due to: {e}\033[0m")
            continue

        # save
        sources.save(fname, overwrite=True, verbose="warning")
        print(f"\033[92mSuccessfully saved surface-based source space for subject {sub}\033[0m")

if __name__ == "__main__":
    main(spacing, overwrite, n_jobs)