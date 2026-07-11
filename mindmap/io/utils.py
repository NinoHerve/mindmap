
from pathlib import Path

from bids import BIDSLayout

def build_bids_path(layout, subject, suffix, **kwargs):
    """
    Universal BIDS path builder for Raw and MindMap derivatives.
    """
    entities = {
        "subject": subject,
        "suffix": suffix,
        **kwargs
    }

    path = layout.build_path(entities, validate=False)

    if path is None:
        raise ValueError(f"Could not build path for suffix '{suffix}'.")

    return path

def build_raw_path(
    root : str | Path,
    entities: dict,
) -> Path:
    """
    Build BIDS path for raw data.
    """
    layout = BIDSLayout(root)

    file_path = layout.build_path(entities)

    return file_path

def build_mm_path(
    root      : str | Path,
    subject   : str,
    suffix    : str,
    session   : str = None,
    task      : str = None,
    run       : str | int = None,
    acq       : str = None,
    space     : str = None,
    desc      : str = None,
    atlas     : str = None,
    res       : str = None,
    extension : str = None,
    scale     : str = None,
) -> Path:
    """
    Constructs a BIDS-compliant path for MindMap derivatives.
    """
    # 1. Configuration registry (format: {suffix: (folder, extension)})
    MM_SUFFIX = {
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
        "v-ave"     : ("v-ave",     "fif"),
    }

    if suffix not in MM_SUFFIX:
        raise ValueError(f"Unknown datatype '{suffix}'. Valid: {list(MM_SUFFIX.keys())}")

    # 2. Cleanup inputs (ensure no redundant 'sub-' or 'ses-')
    sub_id = str(subject).replace("sub-", "")
    ses_id = str(session).replace("ses-", "") if session else None

    # 3. Build directory structure
    fpath = Path(root) / f"sub-{sub_id}"
    if ses_id:
        fpath /= f"ses-{ses_id}"
    fpath /= MM_SUFFIX[suffix][0]

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
        "scale" : scale,
    }

    # Construct the base filename
    parts = [f"sub-{sub_id}"]
    for key, val in entities.items():
        if val is not None:
            parts.append(f"{key}-{val}")

    # 5. Determine extension
    ext = extension if extension is not None else MM_SUFFIX[suffix][1]

    fname = f"{'_'.join(parts)}_{suffix}"
    if ext:
        fname += f".{ext}"

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
        "t1.mgz"     : ("mri", "T1.mgz"),
        "brain.mgz"  : ("mri", "brain.mgz"),
        "orig"       : ("mri", "orig.mgz"),
        "white"      : ("surf", "{hemi}.white"),
        "pial"       : ("surf", "{hemi}.pial"),
        "inflated"   : ("surf", "{hemi}.inflated"),
        "area"       : ("surf", "{hemi}.area"),
        "vertex-ctx" : ("label", "{hemi}.cortex.label"),
        "bem"        : ("bem", ""),
    }

    if datatype not in FS_MAP:
        raise ValueError(f"Unknown FS datatype '{datatype}'. Available: {list(FS_MAP.keys())}")

    folder, file_tmpl = FS_MAP[datatype]

    filename = file_tmpl.format(hemi=hemi)

    return sub_dir/folder/filename



def build_charm_path(
        root: str | Path,
        subject: str,
        datatype: str,
) -> Path:
    """
    Build CHARM paths to objects.
    """
     # 1. Cleanup inputs (ensure no redundant 'sub-')
    sub_id = str(subject).replace("sub-", "")
    sub_dir = Path(root) / f"sub-{sub_id}" / f"m2m_sub-{subject}"

    CHARM_MAP = {
        "seg"      : ("segmentation", "labeling.nii.gz"),
        "lut"      : ("segmentation", "labeling_LUT.txt"),
        "t1-charm" : ("", "T1.nii.gz"),
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


def parse_extension(filename):
    """Returns extension of filename."""
    path = Path(filename)

    suffixes = path.suffixes

    if len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]:
        return "".join(suffixes[-2:])

    return suffixes[-1] if suffixes else ""
