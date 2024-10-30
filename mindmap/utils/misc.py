from pathlib import Path
import os
import numpy as np
import pyvista as pv
import mne 
import nibabel as nib
import ants

def get_folders(
        func: str,
        fs_dir: str | Path | None = "pass",
        fs_home: str | Path | None = "pass", 
        mm_dir: str | Path | None = "pass",
) -> list[Path]:
    
    folders = []

    if fs_dir is None:
        try: fs_dir = os.environ["FS_DIR"]
        except KeyError:
            raise KeyError(f"FS_DIR (FreeSurfer subjects directory) env variable not found. Please provide '{func}' with subjects_dir or read conf file before calling '{func}'.")
 
    if fs_home is None:
        try: fs_home = os.environ["FS_HOME"]
        except KeyError:
            raise KeyError("FS_HOME (FreeSurfer home directory) env variable not found. Please provide '{func}' with fs_home or read conf file before calling '{func}'.")

    if mm_dir is None:
        try: mm_dir = os.environ["MM_DIR"]
        except KeyError:
            raise KeyError("MM_DIR (mindmap subject directory) env variable not found. Please provide '{func}' with mm_dir or read conf file before calling '{func}'.")

    for folder in [fs_dir, fs_home, mm_dir]:
        if folder == "pass":
            pass 
        else:
            if Path(folder).exists() == False:
                raise FileNotFoundError(f"Following folder does not exist: {folder}")
            folders.append(Path(folder))

    if len(folders) == 1:
        return folders[0]
    else:
        return folders


def surf_to_polydata(
        vertices: np.ndarray, 
        faces: np.ndarray,
) -> pv.PolyData:
    """
    Converts a list of vertices and faces to a PolyData structure.
    """
    
    # The faces for the pv.PolyData object needs to be a list of [..., n,v1,v2,v3, ...] 
    # where n indicates the number of vertices of the face and v1-vn indicate the 
    # vertices belonging to that face.
    pv_faces = np.empty_like(faces, shape=(faces.shape[0], faces.shape[1]+1))
    pv_faces[:,1:] = faces 
    pv_faces[:,0] = faces.shape[1]
    pv_faces = np.hstack(pv_faces)

    return pv.PolyData(vertices, pv_faces)


def compute_transformation(
        subject: str, 
        fs_dir: str | Path, 
        fro: str, 
        to:str, 
        scale:str="m"
) -> mne.transforms.Transform:
    """
    Compute the subject specific transformation to go from one space to the other.

    Parameters:
    ----------
    subject: str
        Subject's name.

    fs_dir: path-like
        FreeSurfer directory.

    fro: str ["head", "mri"]
        Initial space.

    to: str ["head", "mri"]
        Target space. 

    scale: str ['m', 'mm']
        Scale to use. 'm' for meters, 'mm' for millimeters.

    Returns:
    -------
    trans: mne.Transform
    """

    # files
    fs_dir   = Path(fs_dir)
    T1_fname = fs_dir/subject/"mri"/"T1.mgz"

    # data
    T1 = nib.load(T1_fname)

    if fro == "head":
        trans1 = np.linalg.inv(T1.affine)
    elif fro == "mri":
        trans1 = np.linalg.inv(T1.header.get_vox2ras_tkr())
    else:
        raise ValueError(f"Unknown space ('{fro}') for computing transformation.")
    
    if to == "head":
        trans2 = T1.affine
    elif to == "mri":
        trans2 = T1.header.get_vox2ras_tkr()
    else:
        raise ValueError(f"Unknown space ('{to}') for computing transformaiton.")
    
    if scale == "m":
        a = 1000
    elif scale == "mm":
        a = 1
    else:
        raise ValueError(f"Unknown scale metric ('{scale}')")

    # combine transformations into 1 transformation matrix
    A = trans2[:3,:3] @ trans1[:3,:3]
    B = (trans2[:3,:3] @ trans1[:3,3].reshape(-1,1)) + trans2[:3,3].reshape(-1,1)
    C = np.array([0,0,0,1]).reshape(1,-1)

    trans = np.concatenate([A,B/a], axis=1)
    trans = np.concatenate([trans, C], axis=0)

    return mne.transforms.Transform(fro=fro, to=to, trans=trans)


def ants_to_nibabel(
        ants_img: ants.core.ants_image.ANTsImage,
    ) -> nib.Nifti1Image:

    tmp = Path("./tmp.nii.gz")

    ants.image_write(ants_img, str(tmp))
    nib_img = nib.load(tmp)
    nib_img = nib.Nifti1Image(nib_img.get_fdata(), nib_img.affine)          # so that data is on the RAM. Else it accesses data only when using .get_fdata() but file will be deleted by then

    tmp.unlink()

    return nib_img


def _retrieve_path(*args):
        try:
            if len(args)==1:
                vars = os.environ[args[0]]
            
            else:
                vars = dict()
                for key in args:
                    vars[key] = os.environ[key]
        
        except KeyError as e:
            raise KeyError(f"Environment variable {e} could not be found. Read configuration file.")   

        return vars