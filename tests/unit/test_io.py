"""
This module performs unit tests for the io module.
"""

import pytest

from mindmap.io.readers import *
from mindmap.io.writers import *

DATA_DIR = Path(__file__).parent.parent/"data"
FS_DIR = Path(__file__).parent.parent/"data"/"derivatives"/"freesurfer"
MM_DIR = Path(__file__).parent.parent/"data"/"derivatives"/"mindmap"

@pytest.mark.parametrize("dtype, mri", [("nibabel", "nibabel"), ("ants", "ants")], indirect=["mri"])
def test_read_mri(dtype, mri):
    """
    Test read_mri.
    """
    # act
    file  = Path(__file__).parent.parent/"data"/"sample"/"anat"/"sample_T1w.nii.gz"
    print(str(file))
    test_mri = read_mri(file, dtype)

    # assert
    assert test_mri == mri

@pytest.mark.parametrize(
    "params, expected",
    [
        ({"root":DATA_DIR, "datatype":"eeg", "task":"rest", "run":1}, DATA_DIR/"sub-samp"/"eeg"/"sub-samp_task-rest_run-1_eeg.vhdr"),
        #({"root":DATA_DIR, "datatype":"ieeg"}, ""),
        ({"root":DATA_DIR, "datatype":"anat", "mne":False}, DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii.gz"),
        ({"root":MM_DIR, "datatype":"bem"}, MM_DIR/"sub-samp"/"bem"/"sub-samp_bem.fif"),
        ({"root":MM_DIR, "datatype":"fwd"}, MM_DIR/"sub-samp"/"fwd"/"sub-samp_fwd.fif"),
        ({"root":MM_DIR, "datatype":"src"}, MM_DIR/"sub-samp"/"src"/"sub-samp_src.fif"),
        ({"root":MM_DIR, "datatype":"cov"}, MM_DIR/"sub-samp"/"cov"/"sub-samp_cov.fif"),
        ({"root":MM_DIR, "datatype":"con"}, MM_DIR/"sub-samp"/"con"/"sub-samp_con.npy"),
        ({"root":MM_DIR, "datatype":"dseg"}, MM_DIR/"sub-samp"/"dseg"/"sub-samp_dseg.nii.gz"),
        ({"root":MM_DIR, "datatype":"ave"}, MM_DIR/"sub-samp"/"ave"/"sub-samp_ave.fif"),
        ({"root":MM_DIR, "datatype":"stc"}, MM_DIR/"sub-samp"/"stc"/"sub-samp_stc"),
        ({"root":MM_DIR, "datatype":"eig"}, MM_DIR/"sub-samp"/"eig"/"sub-samp_eigvec.npy"),
        ({"root":DATA_DIR, "datatype":"anat", "session":"post", "mne":False}, DATA_DIR/"sub-samp"/"ses-post"/"anat"/"sub-samp_ses-post_T1w.nii.gz"),
        ({"root":MM_DIR, "datatype":"stc", "acquisition":"night"}, MM_DIR/"sub-samp"/"stc"/"sub-samp_acq-night_stc"),
        ({"root":DATA_DIR, "datatype":"anat", "space":"mni", "mne":False}, DATA_DIR/"sub-samp"/"anat"/"sub-samp_space-mni_T1w.nii.gz"),
        ({"root":MM_DIR, "datatype":"stc", "description":"wcset"}, MM_DIR/"sub-samp"/"stc"/"sub-samp_desc-wcset_stc"),
        ({"root":MM_DIR, "datatype":"dseg", "atlas":"multiconn"}, MM_DIR/"sub-samp"/"dseg"/"sub-samp_atlas-multiconn_dseg.nii.gz"),
        ({"root":MM_DIR, "datatype":"dseg", "resolution":"scale4"}, MM_DIR/"sub-samp"/"dseg"/"sub-samp_res-scale4_dseg.nii.gz"),
        ({"root":DATA_DIR, "datatype":"anat", "suffix":"T2w", "mne":False}, DATA_DIR/"sub-samp"/"anat"/"sub-samp_T2w.nii.gz"),
        ({"root":DATA_DIR, "datatype":"anat", "extension":"nii", "mne":False}, DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii"),
    ]
)
def test_get_bidspath(params, expected):
    """
    Test get_bidspath.
    """
    # act
    file = get_bidspath(subject = "samp", **params)

    # assert
    assert file == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"fname":DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii.gz"}, DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii.gz"),
        ({"fname":DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii.gz", "instance":"str"}, str(DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii.gz")),
    ]
)
def test_check_file(params, expected):
    """
    Test check file.
    """
    # act
    file = check_file(**params)

    # assert
    assert file == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"fname":DATA_DIR/"FileNotFound"}, pytest.raises(FileNotFoundError)),
        ({"fname":DATA_DIR/"sub-samp"/"anat"/"sub-samp_T1w.nii.gz", "extension":"mgz"}, pytest.raises(ValueError)),
    ]
)
def test_exceptions_check_file(params, expected):
    """
    Test exceptions of check_file.
    """

    with expected:
        check_file(**params)
