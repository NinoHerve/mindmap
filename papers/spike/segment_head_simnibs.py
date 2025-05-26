'''
This script generates head segmentations using the CHARM pipeline implemented via SimNIBS.

Prerequisites
-------------
- Run the FreeSurfer reconstruction on every subject.

Parameters
----------
- None

Outputs
-------
- {fs_dir}/{sub}/mri/m2m_[subID]/segmentation/labeling.nii.gz: Volumetric segmentation.
- {fs_dir}/{sub}/mri/m2m_[subID]/segmentation/labeling_LUT.txt: Segmentation's color LUT.

Notes
-----
- Subject 06, 20 cannot be processed because they do cannot be processed through FreeSurfer.
- For more detailed information, or step by step process, check `mindmap/tutorials/anatomy/head_segmentation.ipynb`
'''

def main()
    # configuration
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]

    for sub in subjects:
        try:
            T1 = fs_dir/sub/"mri"/"T1.mgz"

            print(f"\033[94mRunning charm segementation for subject {sub}\033[0m")
            run_charm(sub, T1)
        except Exception as e:
            print(f"\033[91mSegmentation failed for subject {sub} because: {e}\033[0m")
        else: 
            print(f"\033[92mSuccesffully segmented subject {sub}\033[0m")


if __name__ == "__main__":
    main(consistency_thr, length_thr, number_thr, distance_thr, overwrite)