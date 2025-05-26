'''
This script solves the forward problem by computing maxwell's equations from the source space to the electrodes and
using the conductor model as medium.

Prerequisites
-------------
- Run the FreeSurfer reconstructions.
- Create source spaces.
- Create BEMs.
- Coregister electrode positions to BEM.

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
- For more detailed information, or step by step process, check `mindmap/tutorials/forward/fwd.ipynb`
'''

def main()
    # configuration
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]
    mm_dir   = reader.config["MM_DIR"]

    for sub in subjects:
        # check file existence
        ffwd = mm_dir/sub/"fwd"/f"{sub}_fwd.fif"
        try:
            data = mm.io.read_dataset(sub, "hmodel", "src_raw", "eeg", "trans")

            # compute forward solution
            forward = mne.make_forward_solution(
                info    = data["eeg"].info,
                trans   = data["trans"],
                src     = data["src_raw"],
                bem     = data["hmodel"],
                meg     = False,
                eeg     = True,
                verbose = "warning",
            )

            # save
            mne.write_forward_solution(ffwd, forward, overwrite=overwrite, verbose="warning")

        except Exception as e:
            print(f"\033[91mFailed to solve forward problem for subject {sub} because: {e}\033[0m")
        else: 
            print(f"\033[92mSuccessfully solved forward problem for subject {data['sub']}\033[0m")


if __name__ == "__main__":
    main()