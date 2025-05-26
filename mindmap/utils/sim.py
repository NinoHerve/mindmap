import pywt 
import numpy as np
import scipy as sp
import mindmap as mm
import mne
import matplotlib.pyplot as plt
import seaborn as sns




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


class Simulation:
    """
    Attributes
    ----------
    fwd: mne.Forward 
        Forward (lead field) matrix. Also contains source space information.

    sc: np.ndarray (N,N)
        Dipole level structural-connectome, where N is the number of dipoles.

    lab: np.ndarray(str) | None
        Dipole brain region label if parcellation was provided.


    Methods
    ------- 
    """
    def __init__(self, forward, connectome, duration=1000, sfreq=250, parcellation=None, cras=None):
        """
        Parameters
        ----------
        forward: mne.Forward 
            Forward operator.

        connectome: np.ndarray (N,N)
            Source level structural-connectome.

        duration: float
            Duration of simulation in [ms]

        sfreq: int
            Sampling frequency.

        parcellation: mm.Parcellation.
            Parcellation.

        cras: np.ndarray (3,)
            Translation Subject -> FreeSurfer 

        Attributes
        ----------
        fwd: forward

        sc: connectome

        time: np.ndarray 
            Time array in [s]

        sfreq: sfreq

        lab: np.ndarray(str) | None
            Dipole brain region label if parcellation is provided.
        """
        self.fwd   = forward
        self.sc    = connectome
        self.sfreq = sfreq
        self.time  = np.linspace(0, duration/1000, int(duration/1000*sfreq))   # in [s] 

        if parcellation is None:
            self.lab = None 
        else: 
            if cras is None:
                print("\033[93mParcellation was provided without cras\033[0m")
                cras = np.zeros(3)

            points = mm.utils.get_src_position(self.fwd["src"])*1000 + cras
            labels = parcellation.label_points(points)

            self.lab = np.array(labels)

    def set_spike(self, location, onset=None, duration=20, amplitude=1e-9):
        """
        Define source spike.

        Parameters
        ----------
        location: int | str
            Dipole ID or Brain region name if self.parc is available.

        onset: float
            Onset in [ms]. If None, take half the simulation duration as onset.

        duration: float
            Signal duration in [ms]

        amplitude:
            Maximum spike amplitude in [A]

        width: float
            Spike duration in [ms]. 

        sfreq: int
            Sampling frequency.

        Attributes
        ----------
        onset: onset

        amp: amplitude

        width: spike duration

        loc: np.ndarray 
            List of source dipoles.

        spike: np.ndarray (T,)
            Source spike. The spike consists of two consecutive hanning windows.
        """
        # add attributes
        self.onset = onset if onset is not None else self.time[int(len(self.time)/2)]*1000
        self.amp   = amplitude
        self.width = duration
        self.loc   = location

        # location
        if isinstance(self.loc, int):
            self.loc_ids = np.array([self.loc])

        if isinstance(self.loc, str):
            if self.lab is None:
                raise AttributeError("No Parcellation provided, location can't be given by brain region.")

            if self.loc == "random":
                self.loc = np.random.choice(np.unique(self.lab))
            
            self.loc_ids = np.argwhere(self.lab == self.loc).squeeze()

        # signal
        width1 = int((2/3)*self.width/1000 * self.sfreq)           # width of first window (2/3 duration)
        width2 = int((1/3)*self.width/1000 * self.sfreq)           # width of second window (1/3 duration)
        
        t0 = int(self.onset/1000 * self.sfreq)   # spike begining
        t1 = t0 + width1                         # spike transition
        t2 = t1 + width2                         # spike ending

        bump1 = np.hanning(width1 + 1) * amplitude       # add an extra sample to overlap bump1 and bump2 at transition for smoother transition
        bump2 = np.hanning(width2 + 1) * (-0.5) * amplitude 

        signal1, signal2 = np.zeros_like(self.time), np.zeros_like(self.time)
        signal1[t0:t1+1] = bump1 
        signal2[t1-1:t2] = bump2 

        self.spike = signal1 + signal2 

    def run_propagation(self, conductance, delay):
        """
        Propagate source spike through a temporal dissusion process.
        
        Parameters
        ----------
        conductance: float [0-1]
            Signal conductance. e.g.
            - 1: The original signal is transfered to it's neighbors.
            - 0.5: The 50% downweighted signal is transfered.
            - 0: None of the signal is transfered.

        delay: float 
            Delay in [mm] for transferring the signal.

        Attributes
        ----------
        act: mne.SourceEstimate
            Brain activity of a spike propagating through a temporal diffusion process.
        """

        # add attributeds 
        self.cond = conductance
        self.dt   = delay

        # prepare features
        N = len(self.sc)
        T = len(self.spike)

        shift = int(self.dt/1000 * self.sfreq)

        starting_roi = np.zeros(N, dtype=bool)
        starting_roi[self.loc_ids] = True

        activity = np.zeros((N,T))
        activity[starting_roi] = self.spike

        i = 1 
        rois = starting_roi

        while True:
            # compute next hop roi
            roi = (self.sc @ rois > 0) & (~rois)

            # check if roi is empty or wrong
            assert np.all((roi==0) | (roi==1))
            if np.all(roi==0):
                break 
            roi = roi.astype(bool)

            # update activity
            activity[roi] = self.cond**i * np.roll(self.spike, i*shift)

            # update iteration
            rois = rois | roi 
            i += 1

            if i > 20:
                raise RuntimeError("Spike still hasn't finish propagating after 20 repetitions")
    
        vertices = [self.fwd["src"][0]["vertno"], self.fwd["src"][1]["vertno"]]
        self.act = mne.SourceEstimate(activity, vertices, tmin=0, tstep=1/self.sfreq)

    def run_propagation2(self, n_hops, conductance, delay, conductance_std=0, delay_std=0):
        N = len(self.sc)
        T = len(self.spike)
        X = np.zeros((N,T))

        # set source activity
        visited        = [self.loc_ids]
        X[visited[-1]] = self.spike

        # iterate through hops
        for _ in range(n_hops):
            last_visited = visited[-1]
            new_visited  = set()

            # iterate through current hop
            for node in last_visited:
                neighbors = np.argwhere(self.sc[node]==1).squeeze()    # find following hop
                
                for v in visited:
                    neighbors = list(set(neighbors)-set(v))         # select unvisited nodes
                new_visited = new_visited.union(set(neighbors))

                if len(neighbors) > 0:
                    weights = np.random.normal(conductance, conductance_std, len(neighbors))
                    delays  = delay + delay_std*np.random.pareto(10, len(neighbors))
                    delays  = delays/1000 * self.sfreq
                    new_X   = np.array([w * np.roll(X[node,:], int(d)) for w,d in zip(weights, delays)])
                
                    # add signal to following hop
                    X[neighbors] += new_X

            visited.append(list(new_visited))

        vertices   = [self.fwd["src"][0]["vertno"], self.fwd["src"][1]["vertno"]]
        self.act   = mne.SourceEstimate(X, vertices, tmin=0, tstep=1/self.sfreq)
        self.visit = visited

    def set_noise(self, target_snr, bnoise_type="pink", enoise_type="white", noise_ratio=0.9):
        """
        Generate noise such that highest EEG SNR corresponds to target snr.

        Parameter
        ---------
        target_snr: float
            Target SNR in [dB].

        bnoise_type: str
            Brain noise type (pink|brown|white|...)

        enoise_type: str
            EEG noise type (white|brown|pink|...)

        noise_ratio: float [0-1]
            Ratio between brain noise and eeg noise.

        Attributes
        ----------
        bnoise: np.ndarray (N,T)
            Brain noise.

        enoise: np.ndarray (M,T)
            EEG electrical noise.
        """
        assert hasattr(self, 'act'), "Propagate a spike source before generating noise."

        # save attributes
        self.snr         = target_snr
        self.bnoise_type = bnoise_type
        self.enoise_type = enoise_type
        self.nr          = noise_ratio

        # extract features
        activity = self.act.data

        fwd = mne.convert_forward_solution(self.fwd, force_fixed=True, verbose="warning")
        fwd = fwd["sol"]["data"]

        try:
            bn_noise = getattr(mm.utils.sim, f"{bnoise_type}_noise")
            en_noise = getattr(mm.utils.sim, f"{enoise_type}_noise")
        except AttributeError as e:
            raise AttributeError(f"{bnoise_type} or {enoise_type} of noise is not implemented.")

        N,T = activity.shape
        M   = fwd.shape[0]

        # generate brain noise
        bnoise  = bn_noise(T, N)
        bnoise /= np.sqrt(mm.utils.avg_power(bnoise))[:,None]        # Scale to unitary power

        # generate eeg noise
        enoise  = en_noise(T, M)
        enoise /= np.sqrt(mm.utils.avg_power(enoise))[:,None]     # Scale to unitary power

        # compute powers
        Pdata  = mm.utils.avg_power(fwd@activity)
        Pbrain = mm.utils.avg_power(fwd@bnoise)
        Peeg   = mm.utils.avg_power(enoise)

        Pnoise = self.nr*Pbrain + (1-self.nr)*Peeg

        # Select electrode with highest SNR and define scaling factor
        idx   = np.argmax(Pdata)  
        scale = Pdata[idx]/(Pnoise[idx]*10**(self.snr/10))

        # apply scaling to noise
        bnoise *= np.sqrt(scale*self.nr)
        enoise *= np.sqrt(scale*(1-self.nr))

        self.bnoise = bnoise
        self.enoise = enoise

    def run_eeg(self, eeg_info):
        """
        Generate EEG data from simulated brain activity.
        
        Parameters
        ----------
        eeg_info: mne.Info 
            EEG template from which to extract infos (such as electrode positions...)

        Attributes
        ----------
        eeg: mne.Evoked
            EEG activity.
        """
        assert hasattr(self, "act"), "Set propagation before running simulation."
        assert hasattr(self, "bnoise"), "Set noise before running simulation."

        activity = self.act + self.bnoise
        self.eeg = mne.apply_forward(self.fwd, activity, eeg_info, verbose="warning")


    def summary(self):
        """
        Print summary of simulation.
        """

        print("\033[92mSIMULATION SUMMARY\033[0m")
        
        print("\033[94mSource Space info:\033[0m")
        print(f"- Number of dipoles: {self.fwd['nsource']}")
        print(f"- Parcellation: {'unavailable' if self.lab is None else 'available'}")
        print(f"- Noise type: {self.bnoise_type if hasattr(self, 'bnoise_type') else None}")
        print(f"- Noise contribution: {self.nr if hasattr(self, 'nr') else None}")

        print("\033[94mEEG info:\033[0m")
        print(f"- Number of electrodes: {self.fwd['nchan']}")
        print(f"- Target SNR: {self.snr if hasattr(self, 'snr') else None} [dB]")
        print(f"- Noise type: {self.enoise_type if hasattr(self, 'enoise_type') else None}")
        print(f"- Noise contribution: {1-self.nr if hasattr(self, 'nr') else None}")

        print("\033[94mWindow info\033[0m")
        print(f"- Duration: {len(self.time)/self.sfreq if hasattr(self, 'time') else None} [s]")
        print(f"- Sampling rate: {self.sfreq if hasattr(self, 'sfreq') else None}")

        print("\033[94mSpike info:\033[0m")
        print(f"- Location: {self.loc if hasattr(self, 'loc') else None}")
        print(f"- Onset: {self.onset if hasattr(self, 'onset') else None} [ms]")
        print(f"- Amplitude: {self.amp if hasattr(self, 'amp') else None} [A]")
        print(f"- Duration: {self.width if hasattr(self, 'width') else None} [ms]")
        print(f"- Maximum: {self.get_max() if hasattr(self, 'spike') else None} [ms]")
        

        print("\033[94mPropagation info:\033[0m")
        print(f"- Conductance: {self.cond if hasattr(self, 'cond') else None}")
        print(f"- Delay: {self.dt if hasattr(self, 'dt') else None} [ms]")


    def get_spike(self, noise=True):
        return self.spike

    def get_activity(self, noise=True):
        assert hasattr(self, "act"), "Run propagation to get brain activity" 
        activity = self.act

        if noise:
            assert hasattr(self, "bnoise"), "Set noise to get brain activity with noise"
            activity += self.bnoise

        return activity

    def get_eeg(self, noise=True):
        assert hasattr(self, "eeg"), "Run eeg to get EEG activity"
        eeg = self.eeg 

        if noise:
            eeg.data += self.enoise

        return eeg

    def get_regions(self):
        assert self.lab is not None, "Parcellation was not provided. Impossible to get regions."
        return np.unique(self.lab)

    def get_max(self):
        assert self.spike is not None, "Spike was not defined yet. Run add_source_spike()"
        return self.time[np.argmax(self.spike)]

    def plot_spike(self, noise=False):
        if noise:
            assert hasattr(self, "bnoise"), "Generate noise before adding noise to plot."
            idx    = np.random.randint(self.bnoise.shape[0])
            y      = self.spike + self.bnoise[idx,:]
        else:
            y = self.spike

        ax    = sns.lineplot(x=self.time, y=y)
        title = f"Spike waveform with {self.bnoise_type} noise" if noise else "Spike waveform"
        ax.set(title = title, xlabel = "Time [s]", ylabel = "Ampères [A]")

        return ax

    def plot_brain(self, noise=False):
        assert hasattr(self, "act"), "Propagate spike source before plotting brain activity."
        if noise:
            assert hasattr(self, "bnoise"), "Generate noise before adding noise to plot"
            activity = self.act + self.bnoise
        else:
            activity = self.act

        return mm.viz.plot_brain(self.fwd["src"], activity, cmap="bwr")

    def plot_noise(self, n=3):
        """
        Parameters
        ----------
        n: int
            Number of examples to plot. 
        """
        assert hasattr(self, 'bnoise'), "Generate noise before plotting noise."
        
        # choose n random rows
        rows       = np.random.choice(range(self.bnoise.shape[0]), n)
        b_examples = self.bnoise[rows,:]

        rows       = np.random.choice(range(self.enoise.shape[0]), n)       
        e_examples = self.enoise[rows,:]

        fig, axes = plt.subplots(1,2, figsize=(20,5))
        fig.suptitle(f"Noise examples for target SNR = {self.snr} [dB]")

        axes[0].set_title(f"Brain noise ({self.bnoise_type})")
        for example in b_examples:
            sns.lineplot(ax=axes[0], x=self.time, y=example)
        ylim = (-np.max(np.abs(self.spike)), np.max(np.abs(self.spike)))
        axes[0].set(xlabel="time [s]", ylabel="Magnitude [A]", ylim=ylim)

        axes[1].set_title(f"Electric noise ({self.enoise_type})")
        for example in e_examples:
            sns.lineplot(ax=axes[1], x=self.time, y=example)
        axes[1].set(xlabel="time [s]", ylabel="Magnitude [A]", ylim=(-1e-6, 1e-6))

    def plot_eeg(self, noise=False):
        assert hasattr(self, "eeg"), "Generate EEG before plotting EEG"
        eeg = self.eeg.copy()

        if noise:
            eeg.data += self.enoise

        eeg.plot()


    def plot_summary(self):
        pass 




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