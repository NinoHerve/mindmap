'''
This script generates the Multi Conn parcellation for all subjects in the SPIKE dataset.

Prerequisites
-------------
- Run the FreeSurfer reconstruction on every subject.

Parameters
----------
- `margin`: cortical margin spilling over white matter. Brain sources are placed on the white/gray matter interface. To make sure that
every source is labeled through the parcellation, we slightly spill the cortical areas over the white matter.
- `thresh`: Porbabilistic threshold. The Thalamic Nuclei is a probabilistic atlas. To descritize it we apply a threshold.
- `overwrite`: Whether to overwrite existing atlases.

Outputs
-------
- {mm_dir}/{sub}/parc/{sub}_multiConn4-dseg.nii.gz: Volumetric parcellation. 
- {mm_dir}/{sub}/parc/{sub}_multiConn4-dseg.txt: CLUT for parcellation.

Notes
-----
- Subject 06, 20 cannot be processed because they do cannot be processed through FreeSurfer.
- For more detailed information, or step by step process, check `mindmap/tutorials/anatomy/multi_conn_registration.ipynb`
'''

import mindmap as mm


############# Parameters #################
# Lausanne 2018 parcellation
margin = 2

# Thalamic Nuclei parcellation
thresh = 0.05

# other options
overwrite = True 
##########################################


def main(margin, thresh, overwrite):
    # configuration
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]
    mm_dir   = reader.config["MM_DIR"]

    # !!!!! check prerequisits !!!!!

    # output file generic name
    fname = lambda sub: (mm_dir/sub/"parc"/f"{sub}_multiConn4-dseg.nii.gz",  mm_dir/sub/"parc"/f"{sub}_multiConn4-dseg.txt")

    for sub in subjects:
        nii_fname, clut_fname = fname(sub)

        # check existence of folder/file
        if nii_fname.exists():
            if not overwrite:
                print(f"Skipping subject {sub} because parcellation already exists.")
                continue
        else:
            if not nii_fname.parent.exists():
                nii_fname.parent.mkdir(parents=True)    
    
        # create multiconn parcellation
        print(f"Generating MultiConn parcellation for subject {sub}...")
        try:
            multi_conn = mm.utils.parcelate(sub, "multiconn", margin=margin, threshold=thresh, verbose=0)
        except Exception as e:
            print(f"\033[91mMultiConn parcellation failed for subject {sub}\033[0m")
            print(f"Reason: {e}")
            continue

        # save
        multi_conn.save(nii_fname, clut_fname)
        print(f"\033[92mSuccessfully generated MultiConn Parcellation for subject {sub}\033[0m")


if __name__ == "__main__":
    main(margin, thresh, overwrite)