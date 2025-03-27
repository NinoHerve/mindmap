from pathlib import Path
import os
import numpy as np
import pyvista as pv
import mne 
import nibabel as nib
import ants
import pandas as pd
import mindmap as mm
from mindmap.utils.sources import get_src_position, mask_sources


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


def _retrieve_env(*args):
        try:
            if len(args)==1:
                vars = os.environ[args[0]]

                if isinstance(vars, str) and ("/" in vars):
                    vars = Path(vars)
            
            else:
                vars = dict()

                for key in args:
                    v = os.environ[key]
                    if isinstance(v, str):
                        if (v == "True") or (v == "False"):
                            vars[key] = (v == "True")
                        elif "/" in v:
                            vars[key] = Path(v)
                        else:
                            vars[key] = v
        
        except KeyError as e:
            raise KeyError(f"Environment variable {e} was not found. Read configuration file or check configuration file.")  

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


def avg_power(signal):
    """
    Compute average power of a signal. 
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(1,-1)

    pow = signal - np.mean(signal, axis=1)[:,None]
    pow = np.sum(pow**2, axis=1)/signal.shape[-1]

    return pow


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


def correct_labels(labels, sources, parcel):
    """
    Correct labels that misplace left and right hemisphere dipoles because the parcellation is not perfect.
    """
    # map to switch from label id to label name
    clut = parcel.get_clut()
    map  = pd.Series(clut["Label name"].values, index=clut["#No."]).to_dict()
    
    label_names = [map[id] for id in labels]

    # left/right hemisphere seperation
    split = len(sources[0]["vertno"])

    # switch 'lh' labels in right hemisphere to 'rh' homologue and vice virsa
    label_names[:split] = [l.replace("-rh-", "-lh-") for l in label_names[:split]]
    label_names[split:] = [l.replace("-lh-", "-rh-") for l in label_names[split:]]

    # convert back to lable ids
    new_labels = [list(map.keys())[list(map.values()).index(n)] for n in label_names]

    return new_labels


def compute_distance(pos, area, m=1):
    """
    Compute distance between points and area.
    The algorithm computes the average vector distance to the m closest voxels of the area. If m=1, it's the distance to
    the closest point in the area, if m='all' it's the distance to the center of mass.

    Parameters
    ----------
    pos: np.ndarray (N,3)
        Position of points.

    area: nib.Nifti1Image
        Area as a nibabel image.

    m: int or 'all' 
        Number of voxels to take into account.

    Returns
    -------
    distances: np.ndarray (N,)
        Distances to area as norm of average distance vector.
    """
    if len(pos.shape) < 2:
        pos = pos.reshape(1,-1)

    voxels = np.argwhere(area.get_fdata()>0)
    voxels = mne.transforms.apply_trans(area.affine, voxels)
    
    if m == "all":
        m = voxels.shape[0]

    # compute vector distances
    vectors = voxels[:,None,:] - pos
    norms   = np.linalg.norm(vectors, ord=2, axis=-1)
    norms_i = np.argsort(norms, axis=0)
    ord_vec = np.take_along_axis(vectors, norms_i[:,:,None], axis=0)        # ordered from smallest to highest

    # compute norm of average of m closest distances
    avg  = np.mean(ord_vec[:m,:,:], axis=0)
    dist = np.linalg.norm(avg, axis=1)

    return dist



def select_sources(subject, spacing="ico1", mask=None, cras=np.zeros(3), n_jobs=1):
    fs_dir = _retrieve_env("FS_DIR")

    sources = mne.setup_source_space(
        subject=subject,
        spacing=spacing,
        surface="white",
        add_dist=True,      # geodesic distances between sources
        subjects_dir=str(fs_dir),
        n_jobs=n_jobs,
        verbose="warning",
    )

    if mask is not None:
        sources = mask_sources(sources, mask, cras)

    sources = get_src_position(sources)*1000 + cras

    return sources 


def files_exists(fnames:list | str | Path, mkdir:bool = False):
    """
    Check if file exists.

    Parameters
    ----------
    fnames: list(str | Path) | str | Path
        List of file names.

    mkdir: bool (default to False)
        Makes directory of parent if True.

    Returns
    -------
    exists: list(bool) | bool
        True when file exists.
    """
    # check input type
    if isinstance(fnames, str):
        fnames = [Path(fnames)]
    elif isinstance(fnames, Path):
        fnames = [fnames]
    
    # check existence of files
    exists = [False]*len(fnames)
    for i, f in enumerate(fnames):
        if f.exists():
            exists[i] = True 
        else:
            if not f.parent.exists() and mkdir:
                f.parent.mkdir(parents=True)

    if len(exists) == 1:
        exists = exists[0]

    return exists


def measure_distance(target_point, target_area, x=1):
    """
    Measures distance between a point and an area.

    Parameters
    ----------
    target_point: nd.array (3,)
        Target point

    target_area: nd.array (3,N) | nibabel.Nifti1
        Cloud point

    x: int
        Number of points from the cloud point to consider.

    Returns
    -------
    distance: float
        Distance in [m]

    Note: The distance is measured as the length of the average distance vector between the target point and the x closest points of the point cloud. 
    - x = 1: Distance to closest point of the area.
    - x = "all": Distance to the center of mass of the area.
    - x != "all" != 1: Distance to the center of mass of the portion consisting of the x closest points.
    """

    if isinstance(target_area, nib.Nifti1Image):
        affine      = target_area.affine
        target_area = np.argwhere(target_area.get_fdata()>0)
        target_area = nib.affines.apply_affine(affine, target_area)/1000


    distances = target_area-target_point

    if x == "all":
        mean_distance = np.mean(distances, axis=0)
        norm          = np.linalg.norm(mean_distance)

    else:
        norms         = np.linalg.norm(distances, ord=2, axis=1)
        nearest       = distances[np.argsort(norms)[:x],:]
        mean_distance = np.mean(nearest, axis=0)
        norm          = np.linalg.norm(mean_distance)

    return norm

def select_source_estimate(stc, sources, method="max", tp=None, tw=None):
    """
    Select the epileptic source according to method.

    Parammeters
    -----------
    stc: mne.SourceEstimate
        Source reconstruction.

    sources: mne.SourceSpace
        Sources. Includes source locations.

    method: str
        Method to use.
        - max: select maximum value. Requires a specific time point (tp).
        - energy: select source with maximum energy around spike time. Requires time window (tw).

    tp: int100
        Time point [s] to use.

    tw: list[int,int]
        Time range [s] to use.

    Returns:
    -------
    estimate: np.ndarray (3,)
        Location of Epileptic spike.
    """

    if method == "max":
        if tp is None:
            raise ValueError("Provide a time point (tp) to search for max.")
        
        time_idx = np.argmin(np.abs(stc.times-tp))
        src_idx  = np.argmax(np.abs(stc.data[:,time_idx]))

    elif method == "energy":
        raise NotImplementedError
    
    else:
        raise ValueError(f"Unknown method '{method}' for source estimation.")

    source_estimate = mm.utils.get_src_position(sources)[src_idx]

    return source_estimate