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

def nibabel_to_ants(
    nib_img: nib.Nifti1Image,
) -> ants.core.ants_image.ANTsImage:

    tmp = Path("./tmp.nii.gz")

    nib.save(nib_img, tmp)
    ants_img = ants.image_read(str(tmp))
    
    tmp.unlink()

    return ants_img


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

def estimate_snr(eeg:mne.Evoked, baseline:list[float] = None, signal:list[float] = None):
    """
    Estimate SNR from EEG. 

    Parameters
    ----------
    eeg: mne.Evoked
        EEG

    baseline: list[float]
        tmin and tmax of baseline. If `[]` then first 30% of time serie will be considered as baseline.

    signal: list[float]
        tmin and tmax of signal. If `[]` then only the max of recordings will be taken.

    Returns
    -------
        Estimated SNR at each electrode. 
    """
    if baseline is None:
        tmax     = int(0.3*len(eeg.times))
        baseline = [eeg.times[0], eeg.times[tmax]]

    eeg_      = eeg.copy()
    baseline  = eeg_.crop(tmin=baseline[0], tmax=baseline[1])
    pow_noise = avg_power(baseline.data)

    if signal is None:
        pow_sig = eeg.data.max(axis=1)**2

    else:
        eeg_    = eeg.copy()
        signal  = eeg_.crop(tmin=signal[0], tmax=signal[1])
        pow_sig = avg_power(signal.data)

    return 10*np.log10(pow_sig/pow_noise)

def avg_power(signal):
    """
    Compute average power of a signal. 
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(1,-1)

    avg_pow = signal - np.mean(signal, axis=1)[:,None]
    avg_pow = np.sum(avg_pow**2, axis=1)/signal.shape[-1]

    return avg_pow



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



def binary_classification(
    reference  :mne.SourceEstimate, 
    estimation :mne.SourceEstimate, 
    classifier :float=0.5,
) -> dict:
    """
    Measures precision, recall, sensitivity and accuracy.

    Parameters
    ----------
    reference: mne.SourceEstimate
        Reference image.

    estimation: mne.SourceEstimate
        Reconstructed image.

    classifier: float
        Percentage at which to classify.

    Returns
    -------
    metrics: dict
        Classification metrics.
    """
    if isinstance(reference, mne.SourceEstimate):
        reference = reference.data
    if isinstance(estimation, mne.SourceEstimate):
        estimation = estimation.data

    # process (normalize + classify)
    ref = np.abs(reference) / np.max(np.abs(reference))      
    ref = ref > classifier            

    est = np.abs(estimation) / np.max(np.abs(estimation))
    est = est > classifier

    # compute True/False Positives/Negatives
    TP = np.sum(ref & est,   axis=0)
    TN = np.sum(~ref & ~est, axis=0)
    FP = np.sum(~ref & est,  axis=0)
    FN = np.sum(ref & ~est,  axis=0)

    metrics = binary_classification_metrics(TP, TN, FP, FN)

    return metrics


def binary_classification_metrics(TP, TN, FP, FN):
    """
    Computes classification metrics.

    Parameters
    ----------
    - TP: int
        True positives.

    - TN: int
        True negatives.

    - FP: int
        False positives.

    - FN: int
        False negatives.

    Returns
    -------
    - metrics: dict
        - precision   : TP / (TP + FP)
        - recall      : TP / (TP + FN)
        - specificity : TN / (TN + FP)
        - accuracy    : (TP + TN) / (TP + TN + FP + FN)
        - F1 score    : 2 * precision * recall / (precision + recall) 
    """
    if isinstance(TP, list):
        assert (len(TP) == len(TN) == len(FP) == len(FN)), "Lists of TP,TN,FP and FN must have the same lengths."
    if isinstance(TP, (int, np.integer)):
        TP = [TP]
        TN = [TN]
        FP = [FP]
        FN = [FN]

    metrics = dict(
        precision   = [],
        sensitivity = [],
        specificity = [],
        accuracy    = [],
        f1_score    = [],
    )

    for tp, tn, fp, fn in zip(TP, TN, FP, FN):
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score    = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        metrics["precision"].append(precision)
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)
        metrics["accuracy"].append(accuracy)
        metrics["f1_score"].append(f1_score)
    
    return metrics



# others
# ------

def standardize_stc(stc, timepoint_wise=False):
    """
    Standardize image for visualization.

    Parameters
    ----------
    stc: mne.SourceEstimate
        Source estimation.

    timepoint_wise: bool
        Whether each time point should be standardized individually.

    Returns
    -------
    Standardized source estimate. 
    """

    stc = stc.copy()

    data = stc.data 
    if timepoint_wise:
        data = data / np.max(np.abs(data), axis=1)[:,None]
    else:
        data = data / np.max(np.abs(data))

    stc.data = data 

    return stc



def compute_correlation(array1, array2):
    """
    Compute correlation between two arrays
    """
    # return no correlation (0) if one of the standard deviations is 0
    if (np.std(array1)==0) or (np.std(array2)==0):
        return 0

    # standardize
    array1 = (np.array(array1) - np.mean(array1))/np.std(array1)
    array2 = (np.array(array2) - np.mean(array2))/np.std(array2)

    # correlation
    corr = np.sum(array1*array2)/array1.shape[0]

    return corr

def compute_stc_correlation(stc1, stc2, axis="space"):
    """
    Compute the correlation between two source estimates.

    Parameter
    ---------
    stc1: mne.SourceEstimate
        First source estimation.

    stc2: mne.SourceEstimate
        Second source estimation.

    axis: str ['space', 'time']
        Axis to compute correlation. Can be over space (spacial correlation) or over time (temporal correlation).
    """
    correlations = []
    nsource      = stc1.vertices[0].size + stc1.vertices[1].size

    assert np.all(stc1.times == stc2.times), "Source estimates do not have the same temporal axis."
    assert np.all(nsource == stc2.vertices[0].size+stc2.vertices[1].size), "Source estimates do not have the same number of sources."

    if axis == "space":
        for t in stc1.times:
            idx  = stc1.time_as_index(t)
            img1 = stc1.data[:,idx]         # absolute value ?
            img2 = stc2.data[:,idx]

            corr = compute_correlation(img1, img2)
            correlations.append(corr)

    else:
        for n in range(nsource):
            ts1 = stc1.data[n,:]
            ts2 = stc2.data[n,:]

            corr = compute_correlation(ts1, ts2)
            correlations.append(corr)

    return np.array(correlations)



def abs_stc(stc):
    """
    Returns source estimate with absolute values.
    """
    stc = stc.copy()
    stc.data = np.abs(stc.data)
    return stc

def squared_stc(stc):
    """
    Returns source estimate with squared values.
    """
    stc = stc.copy()
    stc.data = stc.data**2
    return stc


# generate weights
def harmonic_weights(eigvals, relaxation=1):
    """
    Generate harmonic weights from eigenvalues.

    Parameters
    ----------
    eigvals: np.ndarry (N,)
        Eigenvalues of harmonics.
    
    relaxation: float [0-1]
        Relaxes the small weights. The weight will give the probability of a harmonic to be chosen during reconstruction. If taking the raw weights, this can be pretty
        tough on middle harmonics. The relaxation allows middle harmonics to have more weigths. When relaxation = 0, all harmonics have a weigth of 1.
    """
    arr1 = np.abs(eigvals - np.roll(eigvals, -1))
    arr2 = np.abs(eigvals - np.roll(eigvals, 1))

    weights = (arr1 + arr2)/2

    # correct extremities
    weights[0]  = arr1[0]
    weights[-1] = arr2[-1]

    return weights**relaxation