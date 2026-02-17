
from pathlib import Path

import mne_bids


def build_raw_path(
    root      : str | Path,
    subject   : str,
    datatype  : str,
    session   : str = None,
    task      : str = None,
    run       : str | int = None,
    acq       : str = None,
    space     : str = None,
    desc      : str = None,
    suffix    : str = None,
) -> Path:
    """
    Build BIDS path for raw data.
    """
    fname = mne_bids.BIDSPath(
        root        = root,
        subject     = subject,
        datatype    = datatype,
        session     = session,
        task        = task,
        run         = run,
        acquisition = acq,
        space       = space,
        description = desc,
        suffix      = suffix,
    )

    return fname.fpath

def build_mm_path(
    root      : str | Path,
    subject   : str,
    datatype  : str,
    session   : str = None,
    task      : str = None,
    run       : str | int = None,
    acq       : str = None,
    space     : str = None,
    desc      : str = None,
    atlas     : str = None,
    res       : str = None,
    suffix    : str = None,
    extension : str = None,
) -> Path:
    """
    Constructs a BIDS-compliant path for MindMap derivatives.
    """
    # 1. Configuration registry
    MM_DATATYPES = {
        "bem"       : ("bem",       "fif"),
        "fwd"       : ("fwd",       "fif"),
        "src"       : ("src",       "fif"),
        "cov"       : ("cov",       "fif"),
        "con"       : ("con",       "npy"),
        "dseg"      : ("dseg",      "nii.gz"),
        "ave"       : ("ave",       "fif"),
        "i-ave"     : ("i-ave",     "fif"),
        "stc"       : ("stc",       ""),
        "eig"       : ("eig",       "npz"),
        "eeg-trans" : ("eeg-trans", "fif"),
        "anat"      : ("T1w",       "nii.gz"),
    }

    if datatype not in MM_DATATYPES:
        raise ValueError(f"Unknown datatype '{datatype}'. Valid: {list(MM_DATATYPES.keys())}")

    # 2. Cleanup inputs (ensure no redundant 'sub-' or 'ses-')
    sub_id = str(subject).replace("sub-", "")
    ses_id = str(session).replace("ses-", "") if session else None

    # 3. Build directory sturcture
    fpath = Path(root) / f"sub-{sub_id}"
    if ses_id:
        fpath /= f"ses-{ses_id}"
    fpath /= datatype

    # 4. Handle BIDS naming
    entities = {
        "ses"   : ses_id,
        "task"  : task,
        "acq"   : acq,
        "run"   : run,
        "space" : space,
        "atlas" : atlas,
        "res"   : res,
        "desc"  : desc,
    }

    # Construct the base filename
    parts = [f"sub-{sub_id}"]
    for key, val in entities.items():
        if val is not None:
            parts.append(f"{key}-{val}")

    # 5. Determine suffix and extension
    default_suffix, default_ext = MM_DATATYPES[datatype]

    final_suffix = suffix if suffix is not None else default_suffix
    final_ext    = extension if extension is not None else default_ext

    fname = f"{'_'.join(parts)}_{final_suffix}"
    if final_ext:
        fname += f".{final_ext}"

    return fpath / fname

def build_fs_path(
    root     : str | Path,
    subject  : str,
    datatype : str,
    hemi     : str = "lh",
) -> Path:
    """
    Build freesufer paths to objects.
    """
    # 1. Cleanup inputs (ensure no redundant 'sub-')
    sub_id = str(subject).replace("sub-", "")
    sub_dir = Path(root) / f"sub-{sub_id}"

    FS_MAP = {
        "T1"         : ("mri", "T1.mgz"),
        "brain"      : ("mri", "brain.mgz"),
        "orig"       : ("mri", "orig.mgz"),
        "white"      : ("surf", "{hemi}.white"),
        "pial"       : ("surf", "{hemi}.pial"),
        "inflated"   : ("surf", "{hemi}.inflated"),
        "vertex-ctx" : ("label", "{hemi}.cortex.label"),
    }

    if datatype not in FS_MAP:
        raise ValueError(f"Unknown FS datatype '{datatype}'. Available: {list(FS_MAP.keys())}")

    folder, file_tmpl = FS_MAP[datatype]

    filename = file_tmpl.format(hemi=hemi)

    return sub_dir/folder/filename



def build_charm_path(
        root     : str | Path,
        subject  : str,
        datatype : str,
) -> Path:
    """
    Build CHARM paths to objects.
    """
     # 1. Cleanup inputs (ensure no redundant 'sub-')
    sub_id = str(subject).replace("sub-", "")
    sub_dir = Path(root) / f"m2m_sub-{sub_id}"

    CHARM_MAP = {
        "seg" : ("segmentation", "labeling.nii.gz"),
        "lut" : ("segmentation", "labeling_LUT.txt"),
    }

    if datatype not in CHARM_MAP:
        raise ValueError(f"Unknown CHARM datatype '{datatype}'. Available: {list(CHARM_MAP.keys())}")

    folder, filename = CHARM_MAP[datatype]

    return sub_dir/folder/filename



def check_file(
        fname     : str | Path,
        extension : list[str] | str = None,
        instance  : str = None,
        ) -> str | Path:
    """
    Check file:
    1. Check file existance. Raises FileNotFound if false.
    2. Check file extension if one provided. Raises ValueError if false.

    Parameters
    ----------
    fname: str | Path
        File name.

    extension: str
        Required extension.

    instance: str
        "str" or "path". Instance in which to return the file name.
        If None, return the file name as it was given.

    Return:
    ------
    fname: str | Path
        File name returned as string or Path.
    """

    # check if file exists
    f = Path(fname)
    if not f.exists():
        raise FileNotFoundError(f"Following file was not found: {fname}")

    # check extension
    if extension is not None:
        if not isinstance(extension, list):
            extension = [extension]

        extension = [ext if ext.startswith(".") else "." + ext for ext in extension]

        if not f.suffix in extension:
            raise ValueError(f"Invalid file extension for {f}. Expected {extension}")

    # return requested instance
    if instance is not None:
        if instance == "str":
            return str(f)
        return f

    return f
