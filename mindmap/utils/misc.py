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


def rolling_window(array, window_size):
    """
    Returns data of each window, when doing a rolling window.
    Reference: Erik Rigtorp - https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html 

    Parameters
    ----------
    array: np.ndarray (X, N) 
        Array to roll over. Roll over happens on last dimension.
    window_size: int 
        Window size.

    Returns
    -------
    out: np.ndarray (X, N-window_size+1, window_size)
        Windows. N-window+1 correponds to the number of windows. X can be replaced by multiple dimensions.
    """
    shape = array.shape[:-1] + (array.shape[-1] - window_size + 1, window_size)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)



def compute_snr(signal: np.ndarray, noise:np.ndarray) -> float:
    """
    Compute the signal to noise ratio.

    Parameters:
    ----------
    signal: np.ndarray (N, T)
        Signal of shape 'number of sensors' times 'number of time points'.

    noise: np.ndarray (N, T)
        Noise of shape 'number of sensors' times 'number of time points'.

    Returns:
    -------
    snr: np.ndarray (N,)
        Signal to noise ratio at each sensor.
    """
    assert signal.shape == noise.shape, "signal and noise must have the same size to compute SNR."

    signal_zero_mean = signal - signal.mean(axis=-1).reshape(-1,1)
    noise_zero_mean  = noise - noise.mean(axis=-1).reshape(-1, 1)

    Ps = np.sum(signal_zero_mean**2, axis=-1)/signal.shape[-1]
    Pn = np.sum(noise_zero_mean**2, axis=-1)/noise.shape[-1]

    snr = 10*np.log10(Ps/Pn)

    return snr 


def compute_eeg_snr(brain_signal: mne.SourceEstimate, fwd: mne.Forward, roi_mask: np.ndarray) -> float:
    """
    Compute the signal to noise ratio of EEG sensors.

    Parameters:
    ----------
    brain_signal: mne.SourceEstimate 
        Dipole activity.

    fwd: mne.Forward
        Forward solution.

    roi_mask: list | np.ndarray
        Mask for regions of interest.

    Returns:
    -------
    snr: np.ndarray (N,)
        Signal to noise ratio at each EEG electrode.
    """
    # extract data
    brain_signal = brain_signal.data 

    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
    fwd = fwd["sol"]["data"]

    signal     = fwd[:,roi_mask] @ brain_signal[roi_mask, :]
    background = fwd[:,~roi_mask] @ brain_signal[~roi_mask, :]

    snr = compute_snr(signal, background)

    return snr



def find_neighbor_regions(surface, parcellation, cras):
    """
    Defines neighboring regions. 

    Parameters:
    ----------
    surface: [np.ndarray, np.ndarray]
        Cortical surface (or white surface). First element indicate vertices.
        Second element indicate triangles.

    parcellation: Parcellation
        Volumetric parcellation with N regions.

    cras: np.ndarray
        Shift of the Talaraich transform.

    Returns:
    -------
    neighbors: (N,N)
        Neighborhood matrix. Like an adjacency matrix where a connection 
        means that two regions are neighbors.
    """
    labels    = parcellation.label_points(surface[0]+cras, id=True)
    regions   = parcellation.get_clut()["#No."].values
    neighbors = np.zeros((len(regions), len(regions)))

    # find triangles having vertices in different regions
    tri_regions = labels[surface[1]]                             # triangles where vertex id is replaced by region id
    tri_regions = tri_regions[np.sum(tri_regions==0, axis=1)==0] # delete triangles having vertex with id 0 (unknown)  
    info1 = np.abs(tri_regions[:,0] - tri_regions[:,1]) > 0      # True if vertex 1 is not in the same region as vertex 2
    info2 = np.abs(tri_regions[:,0] - tri_regions[:,2]) > 0      # True if vertex 1 is not in the same region as vertex 3

    # mark neighbors based on info1
    idx1 = np.searchsorted(regions, tri_regions[info1][:,0])
    idx2 = np.searchsorted(regions, tri_regions[info1][:,1])
    neighbors[idx1, idx2]= 1

    # mark neighbors based on info2
    idx1 = np.searchsorted(regions, tri_regions[info2][:,0])
    idx2 = np.searchsorted(regions, tri_regions[info2][:,2])
    neighbors[idx1, idx2]= 1

    return neighbors