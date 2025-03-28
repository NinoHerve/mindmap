import pywt 
import numpy as np
import scipy as sp
import mindmap as mm
import mne




def simulate_EEG(eeg_info, forward, labels, connectome, source_region, duration=1, spw=0.25, spd=50, target_snr=10, source_noise=1e-10, electric_noise=5e-7, n=1):
    """
    Simulate EEG data.
    M = number of channels, T = number of time samples, N = number of sources.

    Parameters
    ----------
    eeg_template: mne.Info (M,T)
        Info on EEG setup. 

    forward: mne.Forward (M,N)
        Forward solution.

    labels: list(str) (N,)
        Source region labels.

    connectome: np.ndarray (N,N)
        Source connectivity matrix.

    source_region: str
        Region to apply source.

    duration: float 
        Duratin in [s]

    spw: float [0-1]
        Spreading weight. 

    spd: float
        Spreading delay in [ms].starting_roi

    target_snr: float
        Maximum EEG SNR in [dB] to reach in the EEG data.

    source_noise: float
        Brownian noise step for source noise. 

    electric_noise: float
        White noise standard deviation for electric noise.

    n: int
        Number of simulation with given parameter set. 

    Returns:
    eeg: mne.Evoked 
        Averaged EEG data. 
    
    gt: mne.SourceEstimate 
        Ground truth activity.

    info: pd.DataFrame
        Info on parameters and obtained SNR.
    """

    # features 

    N = forward["nsource"]
    M = eeg_info["nchan"]
    T = int(eeg_info["sfreq"]*duration)

    vertno = [forward["src"][0]["vertno"], forward["src"][1]["vertno"]]     # to instanciate mne.SourceEstimate

    A = mne.convert_forward_solution(forward, force_fixed=True, verbose="warning")
    A = A["sol"]["data"]   

    # Source activity
    wavelet     = pywt.ContinuousWavelet("mexh")
    [signal, _] = wavelet.wavefun(level=5, length=T)

    shift = int(spd/1000*eeg_info["sfreq"])

    roi_mask   = np.isin(labels, source_region)
    Xs_default = spread_signal(signal, roi_mask, connectome, spw, shift)*1e-8

    # functions to estimate source activity scaling
    compute_scaling = lambda snr, Ps, Pn: np.sqrt(10**(snr/10) * Pn / Ps)

    # run simulations
    simulations   = []
    ground_truths = []
    max_snr       = []
    avg_snr       = []
    for _ in range(n):

        # background dipoles
        rng = np.random.default_rng()
        Xb  = rng.normal(loc=0, scale=source_noise, size=(N,T))
        Xb  = Xb.cumsum(axis=1)

        b, a = sp.signal.butter(5, 0.5, fs=eeg_info["sfreq"], btype="high", analog=False)
        Xb   = sp.signal.lfilter(b, a, Xb)

        # electric noise
        noise = rng.normal(loc=0, scale=electric_noise, size=(M,T))

        # scale source to reach maximum target SNR
        Ps  = mm.utils.avg_power(A@Xs_default)
        Pn  = mm.utils.avg_power(A@Xb + noise)
        idx = np.argmax(Ps)                         # id of electrode with maximum SNR

        scaling = compute_scaling(target_snr, Ps[idx], Pn[idx])
        Xs      = Xs_default * scaling

        # create ground truth
        brain_activity = np.zeros((N,T))
        brain_activity = Xs + Xb

        brain_activity = mne.SourceEstimate(brain_activity, vertno, tmin=0, tstep=1/eeg_info["sfreq"])

        # Generate EEGs
        eeg      = mne.apply_forward(forward, brain_activity, eeg_info, verbose="warning")
        eeg.data = eeg.data + noise

        # store data
        simulations.append(eeg)
        ground_truths.append(brain_activity)

        Ps  = mm.utils.avg_power(A@Xs)
        snr = 10*np.log10(Ps/Pn)
        max_snr.append(np.max(snr))
        avg_snr.append(np.mean(snr))

    return simulations, ground_truths, max_snr, avg_snr




def spread_signal(signal, starting_roi, adjacency, r, dt):
    Xa = np.zeros((len(starting_roi), len(signal)))
    Xa[starting_roi] = signal

    i       = 1
    cum_roi = starting_roi

    while True: 
        # compute next hop roi
        roi = (adjacency@cum_roi > 0) & (~cum_roi) 

        # check if roi is empty or wrong
        assert np.all((roi==0) | (roi==1))
        if np.all(roi==0):
            break 
        roi = roi.astype(bool)

        # update signal
        shifted_signal = np.roll(signal, i*dt)
        Xa[roi] = r**i*shifted_signal

        # update iteration
        cum_roi = cum_roi | roi
        i += 1

        if i > 20:
            raise RuntimeError("Signal still hasn't finish propagating after 20 repetitions")
        
    return Xa


def create_source_spike(duration, sfreq, stype="mex", amplitude=1e-8, width=None):
    """
    Creates source spike signal.

    Parameters
    ----------
    duration: float
        Duration of signal in [s]

    sfreq: int
        Sampling frequency in [Hz]

    stype: str
        Spike type
        - "mex": Mexican hat signal.
        - "ham": Double Ham filter.

    amplitude: float
        Maximum amplitude of signal.

    width: float
        Width of the spike in [s]

    Returns
    -------
        spike: np.ndarray (T,)
            Spike of size T (duration*sfreq)
    """
    if stype == "mex":
        wavelet        = pywt.ContinuousWavelet("mexh")
        [spike, _] = wavelet.wavefun(level=5, length=int(sfreq*duration))
        spike = spike

    elif stype == "ham":
        raise NotImplementedError("Double ham filter comming soon")
    
    else:
        raise ValueError(f"Unknown spike type '{stype}'")
    
    # scale spike
    spike = spike/np.max(np.abs(spike)) * amplitude

    return spike


def generate_spike_activity(src, sc, parc, spike, location="random", dw=0.25, dt=50, sfreq=250, cras=np.zeros(3)):
    """
    Generate spike activity inside brain.

    Parameters
    ----------
    src: np.ndarray | mne.SourceSpace (N,3)
        Source space. 

    sc: np.ndarray (N,N)
        Source connectome.

    parc: mm.Parcellation
        Brain parcellation.

    spike: np.ndarray (T,)
        Source spike. 

    location: list(str) | str
        Location(s) of the spike. If random, select a random location from parcellation. Location is a brain region.

    dw: float [0-1]
        Propagation weight. 1 propagates the full signal to all connections. 0 propagate nothing to connections.

    dt: float
        Propagation delay in [ms]

    sfreq: int
        Sampling frequency in [Hz] to convert time delay in index delay.

    Returns
    -------
    activity: mne.SourceEstimate
        Brain activity generated by spike. 
    """
    N = len(sc)
    T = len(spike)

    dt = int(dt/1000 * sfreq)

    # label dipoles by brain region
    dipole_labels = parc.label_points(src, cras = cras)

    if location == "random":
        labels   = np.unique(dipole_labels)
        location = np.random.choice(labels)

    starting_roi = np.array(dipole_labels) == location

    activity = np.zeros((N,T))
    activity[starting_roi] = spike

    i    = 1
    rois = starting_roi

    while True:
        # compute next hop roi 
        roi = (sc @ rois > 0) & (~rois)

        # check if roi is empty or wrong
        assert np.all((roi==0) | (roi==1))
        if np.all(roi==0):
            break 
        roi = roi.astype(bool)

        # update activity
        activity[roi] = dw**i*np.roll(spike, i*dt)

        # update iteration
        rois = rois | roi 
        i   += 1

        if i > 20:
            raise RuntimeError("Spike still hasn't finish propagating after 20 repetitions") 

    return activity


def generate_scaled_noise(data, fwd, target_snr, ratio=0.9, bn_type="pink", en_type="white"):
    """
    Generate noise such that highest EEG SNR corresponds to target snr.

    Parameter
    ---------
    data: np.ndarray | mne.SourceEstimate (N,T)
        Brain activity without noise.

    fwd: np.ndarray | mne.Forward 
        Forward matrix.

    target_snr: float
        Target SNR.

    ratio: float [0-1]
        Ratio between brain noise and eeg noise.

    Returns
    -------
    brain_noise: np.ndarray (N,T)
        Brain noise.

    eeg_noise: np.ndarray (M,T)
        EEG electrical noise.
    """
    
    if isinstance(data, mne.SourceEstimate):
        data = data.data

    if isinstance(fwd, mne.Forward):
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
        fwd = fwd["sol"]["data"]

    try:
        bn_noise = getattr(mm.utils.sim, f"{bn_type}_noise")
        en_noise = getattr(mm.utils.sim, f"{en_type}_noise")
    except AttributeError as e:
        raise AttributeError(f"{bn_type} or {en_type} of noise is not implemented.")

    N,T = data.shape
    M   = fwd.shape[0]

    # generate brain noise
    brain_noise = bn_noise(T, N)
    brain_noise /= np.sqrt(mm.utils.avg_power(brain_noise))[:,None] # Scale to unitary power

    # generate eeg noise
    eeg_noise = en_noise(T, M)
    eeg_noise /= np.sqrt(mm.utils.avg_power(eeg_noise))[:,None]     # Scale to unitary power

    # compute powers
    Pdata  = mm.utils.avg_power(fwd@data)
    Pbrain = mm.utils.avg_power(fwd@brain_noise)
    Peeg   = mm.utils.avg_power(eeg_noise)

    Pnoise = ratio*Pbrain + (1-ratio)*Peeg

    # Select electrode with highest SNR and define scaling factor
    idx   = np.argmax(Pdata)  
    scale = Pdata[idx]/(Pnoise[idx]*10**(target_snr/10))

    # apply scaling to noise
    brain_noise *= np.sqrt(scale*ratio)
    eeg_noise   *= np.sqrt(scale*(1-ratio))

    return brain_noise, eeg_noise


def simulate_eeg(brain_activity, fwd, eeg_temp, brain_noise=None, eeg_noise=None, sub=None):
    """
    Project source space to EEG space. Add noise.

    Parameters
    ---------- 
    brain_acitivity: np.ndarray (N,T)
        Brain activity.

    fwd: mne.Forward 
        Forward Operator

    eeg_temp: mne.Evoked
        EEG template, contains info such as sampling frequency.

    brain_noise: np.ndarray (N,T)
        Brain noise.

    eeg_noise: np.ndarray (M,T)
        EEG electrical noise.

    Returns
    -------
    brain_activity: mne.SourceEstimate 
        Brain activity + brain noise.

    eeg: mne.Evoked 
        Simulated EEG + EEG noise.
    """
    if brain_noise is None:
        brain_noise = np.zeros_like(brain_activity)
    if eeg_noise is None:
        eeg_noise = np.zeros_like(eeg_temp.data)
        
    info = eeg_temp.info

    # Convert brain activity to mne Object
    brain_activity = brain_activity + brain_noise 
    vertices       = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    brain_activity = mne.SourceEstimate(brain_activity, vertices, tmin=0, tstep=1/info["sfreq"], subject=sub)

    # Generate EEGs
    eeg = mne.apply_forward(fwd, brain_activity, info, verbose="warning")
    eeg.data += eeg_noise 

    return brain_activity, eeg



#########################
####     Noise       ####
#########################


def color_noise(T, N=1, filter=lambda f: 1):
    """
    Produces colored noise. Inspired from https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python

    Parameters
    ----------
    T: int
        Number of samples per series.

    N: int
        Number of series.

    filter: function
        Frequency filter (depends on color)
        - f* = f (= white noise)
        - f* = 1/f (= pink noise)
        - ...

    Returns
    -------
    color: np.ndarray(N,1)
        Colored noise 

    Note: To produce colored noise we generate a white noise signal and fix the frequencies according to the filter.
    """
    color = np.zeros((N,T))

    for n in range(N):
        # generate white noise
        white     = np.random.randn(T)
        white_fft = np.fft.rfft(white)

        # fix the frequency space
        scale     = filter(np.fft.rfftfreq(T))
        scale     = scale / np.sqrt(np.mean(scale**2))      # preserve white noise energy
        color_fft = white_fft * scale

        # inverse Fourier
        color[n,:] = np.fft.irfft(color_fft)

    return color.squeeze()

def coloredNoise(filter):
    return lambda T, N=1: color_noise(T, N, filter)

@coloredNoise
def white_noise(f):
    return 1

@coloredNoise 
def blue_noise(f):
    return np.sqrt(f)

@coloredNoise 
def violet_noise(f):
    return f

@coloredNoise 
def brownian_noise(f):
    return 1/np.where(f==0, float("inf"),f)

@coloredNoise 
def pink_noise(f):
    return 1/np.where(f==0, float("inf"), np.sqrt(f))