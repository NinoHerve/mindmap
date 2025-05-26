'''
This script generates boundary element models (BEM) from the SimNIBS segmentations.

Prerequisites
-------------
- Run the FreeSurfer reconstruction on every subject.
- Run CHARM (SimNIBS) on every T1 reconstructed image.

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
        try:
            data = reader(sub, "seg", verbose=0)

            # make volumetric masks of BEM layers
            skin, oskull, iskull = mm.utils.make_bem_volumetric_masks(data["seg"])

            # extract surfaces from volumes
            skin   = mm.utils.vol_to_surf(skin, subdivision=resolution)
            oskull = mm.utils.vol_to_surf(oskull, subdivision=resolution)
            iskull = mm.utils.vol_to_surf(iskull, subdivision=resolution)

            # save surfaces in [fs_dir]/[sub]/mri/bem
            fskin   = fs_dir/data["sub"]/"bem"/"outer_skin.surf"

            fskin_better   = fs_dir/data["sub"]/"bem"/"outer_skin.surf"
            foskull_better = fs_dir/data["sub"]/"bem"/"outer_skull.surf" 
            fiskull_better = fs_dir/data["sub"]/"bem"/"inner_skull.surf" 

            _,_,metadata = nib.freesurfer.read_geometry(fskin, read_metadata=True)
            mm.io.write_surface(fskin_better, skin_, metadata)
            mm.io.write_surface(foskull_better, oskull_, metadata)
            mm.io.write_surface(fiskull_better, iskull_, metadata)

            # create volume conductor
            model = mne.make_bem_model(
                subject      = data["sub"],
                subjects_dir = reader.config["FS_DIR"],
                conductivity = conductivities,
                ico          = None,
                verbose      = "warning",
            )

            bem = mne.make_bem_solution(model, verbose="warning")

            # save bem
            fbem = mm_dir/data["sub"]/"fwd"/f"{data['sub']}_bem.fif"

            if not fbem.parent.exists():
                fbem.parent.mkdir(parents=True)

            mne.write_bem_solution(fbem, bem, overwrite=overwrite, verbose="warning")

        except Exception as e:
            print(f"\033[91mMaking of conductor model of subject {sub} failed because: {e}\033[0m")
        else: 
            print(f"\033[92mSuccessfully saved conductor model for subject {data['sub']}\033[0m")


if __name__ == "__main__":
    main(consistency_thr, length_thr, number_thr, distance_thr, overwrite)