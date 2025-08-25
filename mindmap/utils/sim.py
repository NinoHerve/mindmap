import pywt 
import numpy as np
import scipy as sp
import mindmap as mm
import mne
import matplotlib.pyplot as plt
import seaborn as sns


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
    def __init__(
            self,
            forward:mne.Forward,
            connectome:np.ndarray,
            info:mne.Info,
            parcellation,
            cras:np.ndarray = np.zeros(3),
            duration:float  = 1000,
    ):
        """
        Parameters
        ----------
        forward: mne.Forward 
            Forward operator.

        connectome: np.ndarray (N,N)
            Source level structural-connectome.

        duration: float
            Duration of simulation in [ms]

        info: mne.Info
            EEG info.

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
        # simulation attributes
        self.t    = int(duration/1000 * info["sfreq"])      # duration in number of time samples
        self.n    = forward["nsource"]                      # number of sources
        self.m    = forward["nchan"]                        # number of EEG channels
        self.fwd  = forward                                 # forward solution
        self.sc   = connectome                              # structural connectome
        self.i    = info                                    # EEG info
        self.parc = parcellation                            # parcellation
        self.cras = cras                                    # ras translation (FreeSurfer -> RAS)

        if np.all(self.cras == np.zeros(3)):
            print("\033[93mNo cras applied\033[0m")

        points = mm.utils.get_src_position(self.fwd["src"])*1000 + self.cras
        labels = self.parc.label_points(points)

        self.lab = np.array(labels)     # dipole labels (brain region)

        # spike attributes
        self.loc     = "ctx-lh-inferiorparietal_2"                  # source spike location
        self.onset   = 500                                          # onset time in [ms]
        self.amp     = 1e-9                                         # amplitude in [V]
        self.width   = 100                                          # spike width in [ms]    
        self.loc_ids = np.argwhere(self.lab == self.loc).squeeze()  # source indices correponding to spike location
        self.spike   = np.zeros(self.t)
                      
        # propagation attributes
        self.cond  = 0.25                           # conductivity (connection weights)
        self.dt    = 50                             # delay in [ms]
        self.visit = []                             # visited nodes (info about 1 hop, 2 hop ... neighbors)

        activity = np.zeros((self.n, self.t))
        vertices = [self.fwd["src"][0]["vertno"], self.fwd["src"][1]["vertno"]]
        self.act = mne.SourceEstimate(activity, vertices, tmin=0, tstep=1/self.i["sfreq"])

        # noise attributes
        self.nr         = 0.9                         # ratio between brain and eeg noise
        self.snr        = 15                          # target snr
        self.bnoise_gen = mm.utils.sim.pink_noise     # brain noise generator
        self.enoise_gen = mm.utils.sim.white_noise    # electric noise generator
        self.bnoise     = np.zeros((self.n, self.t))  # brain noise
        self.enoise     = np.zeros((self.m, self.t))  # electric noise
        self.adj_snr    = 0                           # adjusted snr

        # final attributes
        self.eeg = None                             # eeg simulation

    def set_sim_attr(
            self,
            duration:float                     = None,
            forward:mne.Forward                = None,
            connectome:np.ndarray              = None,
            info:mne.Info                      = None,
            parcellation                       = None,
            cras:np.ndarray                    = None,
    ):
        """
        Change simulation attributes.

        Parameters
        ----------
        duration: float
            Simulation duration in [ms].

        forward: mne.Forward
            Forward solution.

        connectome: np.ndarray
            Structural connectome.

        info: mne.Info
            EEG infos.

        parcellation: mm.Parcellation
            Brain parcellation.

        cras: np.ndarray
            Ras translation (FreeSurfer -> RAS).
        """
        if duration is not None:
            self.t = duration

        if forward is not None:
            self.fwd = forward
            self.n   = forward["nsource"]

        if connectome is not None:
            self.sc = connectome

        if info is not None:
            self.i = info

        if parcellation is not None:
            self.parc = parcellation 
        
        if cras is not None: 
            self.cras = cras

    def set_spike_attr(
            self,
            location:str    = None,
            onset:float     = None,
            width:float     = None,
    ):
        """
        Change spike attributes.

        Parameters
        ----------
        location: str
            Spike location (brain region)

        onset: float
            Spike onset time in [ms]

        width: float
            Spike width in [ms] 
        """
        if location is not None:
            if location == "random":
                self.loc = np.random.choice(np.unique(self.lab))
            else:
                self.loc = location
            self.loc_ids = np.argwhere(self.lab == self.loc).squeeze()
        
        if onset is not None:
            self.onset = onset

        if width is not None:
            self.width = width

    def set_prop_attr(
            self,
            conduction:float = None,
            delay:float      = None,
    ):
        """
        Set propagation attributes.

        Parameters
        ----------
        conduction: float [0-1]
            Cunduction property (0.5 means 50% of signal will be transmitted in 1 hop neigbors).

        delay: float
            Propagation delay in [ms].
        """
        if conduction is not None:
            self.cond = conduction 

        if delay is not None:
            self.dt = delay


    def set_noise_attr(
            self,
            snr:float         = None,
            noise_ratio:float = None,
            bnoise_type:str   = None,
            enoise_type:str   = None,
    ):
        """
        Set noise attributes.

        Parameters
        ----------
        snr: float
            Target signal to noise ratio to achieve (averaged over all electrodes).

        noise_ratio: float [0-1]
            Noise ratio between brain noise and electric noise.

        bnoise_type: str
            Brain noise type (pink, white, brown, ...)

        enoise_type: str
            Electric noise type(pink, white, brown, ...)
        """
        if snr is not None:
            self.snr = snr 

        if noise_ratio is not None:
            self.nr = noise_ratio 

        if bnoise_type is not None:
            try:
                self.bnoise = getattr(mm.utils.sim, f"{bnoise_type}_noise")
            except AttributeError as  e: 
                raise AttributeError(f"{bnoise_type} type of noise is not implemented.") from e
            
        if enoise_type is not None:
            try:
                self.enoise = getattr(mm.utils.sim, f"{enoise_type}_noise")
            except AttributeError as e:
                raise AttributeError(f"'{enoise_type}' type of noise is not implemented.") from e

    def run(self):
        """
        Run simulation.
        """

        # create spike
        self.spike = self.generate_spike()

        # propagate spike
        self.act = self.propagate_spike(self.spike)

        # generate eeg
        self.eeg = self.generate_eeg()

    def generate_spike(self):
        """
        Generate spike signal.
        """

        sfreq = self.i["sfreq"]

        # signal
        width1 = int((2/3)*self.width/1000 * sfreq)
        width2 = int((1/3)*self.width/1000 * sfreq)

        t0 = int(self.onset/1000 * sfreq)
        t1 = t0 + width1
        t2 = t1 + width2

        bump1 = np.hanning(width1 + 1) * 5 * 1e-8               # Start at 50 nAmpère, will be adjusted with noise.
        bump2 = np.hanning(width2 + 1) * (-0.5) * 5 * 1e-8

        signal1, signal2 = np.zeros(self.t), np.zeros(self.t)
        signal1[t0:t1+1] = bump1
        signal2[t1-1:t2] = bump2

        spike = signal1 + signal2

        return spike
    
    # def propagate_spike(self, spike):
    #     """
    #     Propagate spike.
    #     """
    #     sfreq = self.i["sfreq"]
    #     shift = int(self.dt/1000 * sfreq)

    #     starting_roi = np.zeros(self.n, dtype=bool)
    #     starting_roi[self.loc_ids] = True
    #     self.visit.append(starting_roi)

    #     activity = np.zeros((self.n, self.t))
    #     activity[starting_roi] = spike

    #     i = 1
    #     rois = starting_roi

    #     while True:
    #         # compute next hop roi
    #         roi = (self.sc @ rois > 0) & (~rois)
    #         self.visit.append(roi)

    #         # check if roi is empty or wrong
    #         assert np.all((roi==0) | (roi==1))
    #         if np.all(roi==0):
    #             break 
    #         roi = roi.astype(bool)

    #         # update activity
    #         activity[roi] = self.cond**i * np.roll(spike, i*shift)

    #         # update iteration
    #         rois = rois | roi
    #         i += 1

    #         if i > 20:
    #             raise RuntimeError("Spike still hasn't finish propagating after 20 repetitions")

    #     vertices = [self.fwd["src"][0]["vertno"], self.fwd["src"][1]["vertno"]]

    #     return mne.SourceEstimate(activity, vertices, tmin=0, tstep=1/sfreq)

    

    def propagate_spike(self, spike):
        """
        Propagate spike.
        """
        sfreq = self.i["sfreq"]
        shift = int(self.dt/1000 * sfreq)

    def generate_eeg(self):
        """
        Generate EEG projection with noise.
        """

        fwd = mne.convert_forward_solution(self.fwd, force_fixed=True, verbose="warning")
        fwd = fwd["sol"]["data"]

        # generate brain noise
        self.bnoise  = self.bnoise_gen(self.t, self.n)
        self.bnoise  = self.bnoise / np.sqrt(mm.utils.avg_power(fwd@self.bnoise).mean()) * self.nr * 10e-6       # 10e-7 so that values are slightly below 1 nAmpère

        # generate eeg noise
        self.enoise  = self.enoise_gen(self.t, self.m)
        self.enoise  = np.sqrt(mm.utils.avg_power(self.enoise).mean()) * (1-self.nr) * 10e-6                     # same scale to maintain ratio between brain and electric noise

        # compute snr
        activity       = self.act.copy()
        activity.data += self.bnoise
        eeg            = mne.apply_forward(self.fwd, activity, self.i, verbose="warning")
        eeg.data      += self.enoise
        snr            = mm.utils.estimate_snr(eeg).mean()

        # adjust snr
        scale         = 10**((self.snr-snr)/20)
        self.act.data = self.act.data * scale

        # project final eeg
        activity       = self.act.copy()
        activity.data += self.bnoise
        eeg            = mne.apply_forward(self.fwd, activity, self.i, verbose="warning")
        eeg.data      += self.enoise

        self.adj_snr = mm.utils.estimate_snr(eeg).mean()

        return eeg
    

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

    def summary(self):
        """
        Print summary of simulation.
        """

        print("\033[92mSIMULATION SUMMARY\033[0m")

        print("\033[94mSimulation info\033[0m")
        print(f"- Duration: {self.t/self.i['sfreq']} [s]")
        print(f"- Sampling rate: {self.i['sfreq']}")
        
        print("\033[94mSource Space info:\033[0m")
        print(f"- Number of dipoles: {self.n}")
        print("- Noise type: pink")
        print(f"- Noise contribution: {self.nr}")

        print("\033[94mEEG info:\033[0m")
        print(f"- Number of electrodes: {self.m}")
        print(f"- Target SNR: {self.snr} [dB]")
        print(f"- Adjusted SNR: {self.adj_snr} [dB]")
        print("- Noise type: white")
        print(f"- Noise contribution: {1-self.nr}")

        print("\033[94mSpike info:\033[0m")
        print(f"- Location: {self.loc}")
        print(f"- Onset: {self.onset} [ms]")
        print(f"- Amplitude: {self.amp} [A]")
        print(f"- Duration: {self.width} [ms]")
        print(f"- Maximum: {self.get_max()} [ms]")
        

        print("\033[94mPropagation info:\033[0m")
        print(f"- Conductance: {self.cond}")
        print(f"- Delay: {self.dt} [ms]")


    def get_spike(self, noise=True):
        """
        Return spike as array.
        """
        if noise:
            idx   = np.random.choice(len(self.bnoise))
            spike = self.spike + self.bnoise[idx,:]
        else:
            spike = self.spike

        return spike

    def get_activity(self, noise=True):
        """
        Return brain activity as source estimate. 
        """
        activity = self.act.copy()

        if noise:
            activity.data += self.bnoise

        return activity

    def get_eeg(self, noise=True):
        """
        Return EEG as Evoked.
        """
        if not noise:
            activity  = self.act.copy()
            eeg       = mne.apply_forward(self.fwd, activity, self.i, verbose="warning")
            return eeg
        
        else:
            return self.eeg

    def get_regions(self):
        """
        Return region names.
        """
        return np.unique(self.lab)

    def get_max(self):
        """
        Return time of maximum.
        """
        times = self.eeg.times
        return times[np.argmax(self.spike)]

    def plot_spike(self, noise=False):
        """
        Plot spike waveform.
        """
        times = self.eeg.times
        y     = self.get_spike(noise=noise)
        
        ax    = sns.lineplot(x=times, y=y)
        title = "Spike waveform with pink noise" if noise else "Spike waveform"
        ax.set(title = title, xlabel = "Time [s]", ylabel = "Ampères [A]")

        return ax

    def plot_brain(self, noise=True):
        """
        Plot brain activity.
        """
        activity = self.get_activity(noise=noise)

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

        axes[0].set_title("Brain noise (pink)")
        times = self.eeg.times
        for example in b_examples:
            sns.lineplot(ax=axes[0], x=times, y=example)
        ylim = (-np.max(np.abs(self.spike)), np.max(np.abs(self.spike)))
        axes[0].set(xlabel="time [s]", ylabel="Magnitude [A]", ylim=ylim)

        axes[1].set_title("Electric noise (white)")
        for example in e_examples:
            sns.lineplot(ax=axes[1], x=times, y=example)
        axes[1].set(xlabel="time [s]", ylabel="Magnitude [A]", ylim=(-1e-6, 1e-6))

    def plot_eeg(self, noise=True):
        """
        Plot eeg.
        """
        eeg = self.get_eeg(noise=noise)
        eeg.plot()




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