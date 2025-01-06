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