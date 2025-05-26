'''
This script generates boundary element models (BEM) from the T1 images using FreeSurfer.

Prerequisites
-------------
- Run the FreeSurfer reconstruction on every subject.

Parameters
----------
- None

Outputs
-------
- {fs_dir}/{sub}/bem/outer_skin.surf: Surface of the outer skin.
- {fs_dir}/{sub}/bem/outer_skull.surf: Surface of the outer skull.
- {fs_dir}/{sub}/bem/inner_skull.surf: Surface of the inner skull.

Notes
-----
- Subject 06, 20 cannot be processed because they do cannot be processed through FreeSurfer.
- For more detailed information, or step by step process, check `mindmap/tutorials/forward/bem_simnibs.ipynb`
'''

def main()
    # configuration
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]

    for sub in subjects:
        fbem = mm_dir/data["sub"]/"fwd"/f"{data['sub']}_bem.fif"
        if fbem.exists() and not overwrite:
            print(f"\033[93mConductor model already exists for subject {sub}\033[0m")
            continue 

        elif not fbem.parent.exists():
            fbem.mkdir(parents=True)

        try:
            bem = mm.solve_volume_conduction(sub, method="bem", min_skull_thick=min_skull_thick, conductivities=conductivities)
            mne.write_bem_solution(fbem, bem, overwrite=overwrite, verbose="warning")
        except Exception as e:
            print(f"\033[91mFailed to generate conductor model for subject {sub} because: {e}\033[0m")
        else: 
            print(f"\033[92mSuccessfully saved conductor model for subject {data['sub']}\033[0m")


if __name__ == "__main__":
    main(consistency_thr, length_thr, number_thr, distance_thr, overwrite)