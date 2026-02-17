import pywt 
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

from mindmap.utils import get_src_position, estimate_snr, avg_power, make_stc, Parcellation
from mindmap.viz import plot_brain

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

        points = get_src_position(self.fwd["src"])*1000 + self.cras
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
        self.bnoise_gen = pink_noise     # brain noise generator
        self.enoise_gen = white_noise    # electric noise generator
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
        self.bnoise  = self.bnoise / np.sqrt(avg_power(fwd@self.bnoise).mean()) * self.nr * 10e-6       # 10e-7 so that values are slightly below 1 nAmpère

        # generate eeg noise
        self.enoise  = self.enoise_gen(self.t, self.m)
        self.enoise  = np.sqrt(avg_power(self.enoise).mean()) * (1-self.nr) * 10e-6                     # same scale to maintain ratio between brain and electric noise

        # compute snr
        activity       = self.act.copy()
        activity.data += self.bnoise
        eeg            = mne.apply_forward(self.fwd, activity, self.i, verbose="warning")
        eeg.data      += self.enoise
        snr            = estimate_snr(eeg).mean()

        # adjust snr
        scale         = 10**((self.snr-snr)/20)
        self.act.data = self.act.data * scale

        # project final eeg
        activity       = self.act.copy()
        activity.data += self.bnoise
        eeg            = mne.apply_forward(self.fwd, activity, self.i, verbose="warning")
        eeg.data      += self.enoise

        self.adj_snr = estimate_snr(eeg).mean()

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

        return plot_brain(self.fwd["src"], activity, cmap="bwr")

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




class Source:
    loc   : int | str | list | np.ndarray   # stimulus locations (dipole ids or brain regions)
    width : dict                            # waveform width (in [s])
    on    : float                           # onset time (in [s])
    off   : float                           # offset time (in [s])
    rep   : float                           # time between repetitions (in [s])
    

    def __init__(self, locations, waveform_width=0.03, onset=None, offset=None, repeat=None):
        """
        Parameters
        ----------
        waveform_width: float
            Waveform width (in [s])

        onset: float
            Onset of stimulus in [s]. Default at starting time.

        offset: float
            Offset of stimulus in [s]. Default at ending time.

        repeat: float
            Time between repetitions in [s]
        """
        if not isinstance(locations, np.ndarray):
            locations = np.array(locations)
        
        self.loc   = locations
        self.width = waveform_width
        self.on    = onset 
        self.off   = offset 
        self.rep   = repeat


    def build_source_pattern(self, times, src, parc=None, cras=None):
        """
        Builds waveform and attributes activity to every dipole.

        Parameters
        ----------
        times: np.ndarray
            Time vector in [s]

        src: mne.SourceSpaces
            Source space

        parc: mm.Parcellation
            Parcellation. Required if location is a brain region (str). 
        """

        # build waveform
        waveform = self.build_waveform(times)

        # select target locations
        targets = self.select_targets(src, parc, cras)

        # generate stimulus pattern
        pattern = np.zeros((targets.shape[0], times.shape[0]))
        pattern[targets,:] = waveform["data"]

        out = dict(
            pattern  = pattern,
            waveform = waveform,
            targets  = np.argwhere(targets).squeeze(),        # mask to ids
        )
        
        return out


    def build_waveform(self, times, scale=1e-8):
        """
        Build spike waveform. The waveform is a double hanning window. 

        Parameters
        ----------
        times: np.ndarray 
            Time array (in [s])
        """
        sfreq = len(times) / (times[-1]-times[0])   # sampling frequency in [Hz]

        # define spike train
        onset  = self.on  if self.on  is not None else times[0]
        offset = self.off if self.off is not None else times[-1]

        train = np.zeros_like(times)
        train[int(onset*sfreq)] = 1 

        if self.rep is not None:
            assert (self.rep > 1/sfreq), f"Repetition ({self.rep} [s]) can't be smaller than the sampling time (currently: {1/sfreq} [s] <-> {sfreq} [Hz])"
            
            dt = self.rep
            t  = onset + dt

            while t < offset: 
                train[int(t*sfreq)] = 1
                t += dt 

        # build waveform (double hanning window) at each spiking event
        assert (self.width > 6/sfreq), f"Waveform width ({self.width} [s]) needs to be at least six times larger than sampling time (currently: {1/sfreq} [s] <-> {sfreq} [Hz])"

        width1 = int((2/3)*self.width * sfreq)
        width2 = int((1/3)*self.width * sfreq)

        window = np.zeros(width1 + width2)                                  
        window[:width1+1] = np.hanning(width1 + 1) * 5
        window[width1-1:] = np.hanning(width2 + 1) * (-0.5) * 5 
        
        waveform = np.convolve(train, window, mode="same") * scale

        out = dict(
            data    = waveform,
            wavelet = window,
            events  = train,
        )

        return out
    
    def select_targets(self, src, parc=None, cras=None):
        """
        Retrieves dipole ids belonging to source.
        """
        sources = get_src_position(src)

        if self.loc.dtype == int:
            targets = np.zeros(len(sources), dtype=bool)
            targets[self.loc] = True
        else:
            assert parc is not None, "Need parcellation to build stimulus because source locations are given as brain regions."
            if cras is None:
                print("\033[93mWarning: No CRAS translation for computing dipole labels.\033[0m")
                cras = np.zeros(3)

            sources = sources*1000 + cras
            labels  = parc.label_points(sources)
            targets = np.isin(labels, self.loc)

        return targets
        


    def plot_waveform(self, times, show_events=False, title="Source waveform"):
        """
        Plots the source waveform.
        """
        waveform = self.build_waveform(times)

        ax = sns.lineplot(x=times, y=waveform["data"])
        ax.set(title=title, xlabel = "Time [s]", ylabel = "Dipole activity [A]")
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")

        if show_events:
            val         = np.max(np.abs(waveform["data"])) * 0.1
            spike_times = times[np.argwhere(waveform["events"]>0)]

            ax.vlines(spike_times, -val, val, colors="k")

        return ax


    def plot_targets(self, src, parc=None, cras=None):
        """
        Plots the source locations. 
        """
        targets = self.select_targets(src, parc, cras)
        plot    = plot_brain(src, targets.astype(int), absolute=True).show()

        return plot

    def __str__(self):
        summary = "\033[94mSource:\033[0m\n" \
                  f"- locations: {self.loc}\n" \
                  f"- waveform width: {self.width} [ms]\n" \
                  f"- Onset: {self.on} [ms] \n" \
                  f"- Offset: {self.off} [ms] \n" \
                  f"- Repeat: {self.rep} [ms]\n"
        
        return summary
    
      

class HeadModel:
    sens: mne.Epochs | mne.Evoked
    fwd: mne.Forward
    weights: np.ndarray
    lenghts: np.ndarray
    speed_a: float
    d_syn: float
    parc: Parcellation
    cras: np.ndarray

    def __init__(self, sensors, forward, connectome_weights, connectome_lengths, axonal_speed=500, synaptic_delay=0.001, parcellation=None, cras=None):
        """
        Parameters
        ----------
        sensors: mne.info

        forward: mne.Forward
            Forward solution. Projects dipole activity to sensor space. Also contains source space information.

        connectome_weights: np.ndarray
            Connection weights between dipoles.

        connectome_lengths: np.ndarray
            Fiber lengths between dipoles.

        axonal_speed: float
            Axonal conduction speed in [mm/s]. Speeds can vary from 500 mm/s to 150000 mm/s.

        synaptic_delay: float
            Synaptic delay in [s]. Delays can vary from 0.0005 s to 0.001 s.

        parcellation: mm.utils.Parcellation
            Parcellation, needed if stimulating entire brain regions.

        cras: np.ndarray 
            RAS translation vector. Needed to align dipole space with parcellation space.
        """
        self.sens    = sensors
        self.fwd     = forward 
        self.weights = connectome_weights 
        self.lengths = connectome_lengths
        self.speed_a = axonal_speed
        self.d_syn   = synaptic_delay
        self.parc    = parcellation
        self.cras    = cras

    def compute_delay_matrix(self):
        return self.lengths/self.speed_a + (self.lengths>0)*self.d_syn

    def compute_distance_from_source(self, source, weight="delay", distance_in_hop=False):
        """
        Compute distance (delay, length or number of hops) between a target node and all other nodes.

        Parameters
        ----------
        source: int
            Dipole id of target node.

        weight: str
            Type of distance:
            - delay: will use the delay adjacency matrix.
            - length: will use the length adjacency matrix.
            - hop: will use the binarized adjacency matrix.

        distance_in_hop: bool
            Flag to return distance of shortest path in hops.
        """
        if weight == "delay":
            if not hasattr(self, "delays"):
                self.delays = self.compute_delay_matrix()
            D = self.delays
        elif weight == "length":
            D = self.lengths 
        elif weight == "hop":
            D = (self.lengths > 0).astype(int)

        else:
            raise ValueError(f"Unknown weight type {weight} for computing distance to source.")

        # compute shortest paths
        D     = nx.Graph(D) 
        paths = nx.shortest_path(D, source=source, weight="weight")
        
        # order paths by node id
        dist = np.ones(D.number_of_nodes())*np.inf  # isolated nodes are not in paths, set infinity as default.

        for n, path in paths.items():
            if distance_in_hop:
                dist[n] = len(path)-1       # don't count starting node
            else:
                dist[n] = nx.path_weight(D, path, weight="weight")

        return dist
    
    def __str__(self):
        summary = "\033[94mHead model:\033[0m\n" \
                  f"- Number of dipoles: {self.fwd['nsource']}\n" \
                  f"- Number of sensors: {len(self.sens.ch_names)}\n" \
                  f"- Recording duration: {self.sens.times[-1]-self.sens.times[0]} [s]\n" \
                  f"- Recording samples: {len(self.sens.times)}\n" \
                  f"- Axonal speed: {self.speed_a} [mm/s]\n" \
                  f"- Synaptic delay: {self.d_syn} [s]\n" \
                  f"- Parcellation: {None if self.parc is None else self.parc.n_regions}\n" \
                  f"- CRAS: {self.cras} \n" \
        
        return summary


class Noise:
    noises: pd.DataFrame

    def __init__(self, *noise_dicts):
        """
        Every noise object is a dictionary with the following information:
        - component: for which component this noise is aimed at ('dipole', 'sensor', 'delay', ...)
        - category: what type of noise to render ('white', 'brown', 'pink', ...)
        - scale: at which scale the noise should be rendered (1, 1e2, 1e3, ...) 
        """
        for noise in noise_dicts:
            assert np.all(np.isin(["component", "category", "scale"], list(noise.keys()))), "All noise dictionaries must have the keys ['component', 'category', 'scale']."

        self.noises = pd.DataFrame(noise_dicts)
        self.noises.set_index("component", inplace=True)

    def add_noise(self, signal:np.ndarray, component:str ="dipole"):
        """
        Adds noise depending on type of component.

        Parameters
        ----------
        signal: np.ndarray (T,N)
            Signal for which to add noise.

        component: str
            Type of component (e.g. dipole, sensor, ...)
        """

        if not component in self.noises.index:
            return signal

        # extract dimensions
        shape = signal.shape

        if len(shape) == 1:
            N = 1
            T = shape[0]
        else:
            N = shape[0]
            T = shape[1]

        return signal + self.generate_noise(T,N)


    def generate_noise(self, T, N=1, component="dipole"):
        
        assert component in pd.unique(self.noises.index), f"Only the noise of the following components have been provided to the Noise class: {pd.unique(self.noises.index)}"

        category = self.noises.loc[component]["category"]
        scale    = self.noises.loc[component]["scale"]

        try:
            noise_generator = getattr(mm.utils.sim, f"{category}_noise")
        except AttributeError as  e: 
            raise AttributeError(f"{category} noise has not been implemented in the mindmap package yet.") from e
        
        noise = noise_generator(T, N) * scale

        return noise
    
    def __str__(self):

        summary = "\033[94mNoise:\033[0m\n"

        for component in self.noises.index:
            summary += f"- '{component}' noise:\n" \
                f"  - category: {self.noises.loc[component]['category']}\n" \
                f"  - scale: {self.noises.loc[component]['scale']}\n"
            
        return summary


class Simulator:
    source: Source 
    hm: HeadModel 

    def __init__(self, head_model, source=None, noise=None):
        self.hm     = head_model
        self.source = source
        self.noise  = noise

    def compute_single_source_propagation(self, source, waveform, w=0.5):
        """
        Computes the propagation of a single dipole source. 

        Parameters
        ----------
        source: int
            Dipole id belonging to source.

        waveform: np.ndarray (T,)
            Waveform of source.

        w: float [0-1]
            Spatial weight decay.

        Returns
        -------
        data: np.ndarray (N,T)
            Propagated source. 
        """
        # delays
        delays = self.hm.compute_distance_from_source(source=source, weight="delay")
        delays[delays==np.inf] = 1e6                                                                # replace isolated nodes (distance is infinity) with very big number
        delays = np.round(delays * self.hm.sens.info["sfreq"]).astype(int)

        # weights
        hops    = self.hm.compute_distance_from_source(source=source, weight="hop", distance_in_hop=True)
        hops[hops==np.inf] = 1e6                                                                            
        weights = w**hops

        # dimensions
        N = self.hm.fwd["nsource"]
        T = len(self.hm.sens.times)

        # compute source propagation
        data = np.zeros((N,T))

        for dt in np.unique(delays):
            nodes = np.argwhere(delays==dt).squeeze()

            if dt == 0:
                signals = waveform 
            else:
                # delayed signal
                signals = np.pad(waveform[:-dt], (np.min([len(waveform), dt]),0), mode="constant", constant_values=0)    # pads new time points with 0. np.min([len(waveform), dt]) selects the size of the simulation instead of dt in case dt is larger than the simulation duration
        
            data[nodes,:] = signals.reshape(1,-1) * weights[nodes].reshape(-1,1)   # weighted + delayed signal

        return data
    
    def run(self, w, snr, return_stc_noise=False):
        """
        Simulate a source propagation.

        Parameters
        ----------
        w: float [0-1]
            Weight decay.

        snr: float
            Average peak to power snr to aim for in the EEG data. Single peak snr are measured as the ratio between the absolute maximum of a channel
            and the average power during the baseline of that channel. The final snr is an average across all channels.
        
        return_stc_noise
            Flag to return brain simulation with noise too.

        Returns
        -------
            ground_truth: mne.SourceEstimate 
                Dipole activity of simulated source propagation without noise.

            recordings: mne.Epochs | mne.Evoked
                Sensor activity of simulated source propagation.

            stc: mne.SourceEstimate (Optional)
                Dipole activity of simulated source propagation with noise. Only returned if
                `return_stc_noise` is True
        """
        # dimensions
        N = self.hm.fwd["nsource"]
        M = len(self.hm.sens.ch_names)
        T = len(self.hm.sens.times)


        # -----------------------------
        # Simulate source propagations
        # -----------------------------

        # output
        data = np.zeros((N,T))

        at_least_one_source = False
        if self.source is not None:
            # sources
            waveform = self.source.build_waveform(self.hm.sens.times)
            sources  = np.argwhere(self.source.select_targets(self.hm.fwd["src"], self.hm.parc, self.hm.cras)).squeeze()

            if len(sources) > 0:
                at_least_one_source = True

            for s in sources:
                data += self.compute_single_source_propagation(s, waveform["data"], w=w)

        # scale maximum to 1e-8 A (will later be adjusted to fit SNR)
        data = data / np.max(np.abs(data)) * 1e-8

        # ---------------
        # Generate noise
        # ---------------

        # generate noise seperately
        at_least_one_noise = False
        if self.noise is not None and "dipole" in self.noise.noises.index:
            noise_dipole = self.noise.generate_noise(T, N, component="dipole")  # shape T, N will be swapped during output (default N is 1 but T is mandatory so the function has to have mandatory first)
            at_least_one_noise = True
        else:
            noise_dipole = np.zeros((N,T))

        if self.noise is not None and "sensor" in self.noise.noises.index:
            noise_sensor = self.noise.generate_noise(T, M, component="sensor")
            at_least_one_noise = True
        else:
            noise_sensor = np.zeros((M,T))


        # ------------
        # Adjust SNR
        # ------------

        # check if SNR can be adjusted 
        if at_least_one_source and at_least_one_noise:

            # project the current simulation to sensors

            # convert to MNE objects
            stc = make_stc(
                src      = self.hm.fwd["src"],
                activity = data + noise_dipole,         # current dipole amplitudes with noise
                tmin     = self.hm.sens.times[0],
                tstep    = 1/self.hm.sens.info["sfreq"],
            )

            # apply forward
            recordings       = mne.apply_forward(self.hm.fwd, stc, self.hm.sens.info, verbose="warning")
            recordings.data += noise_sensor

            # compute average peak to power SNR
            current_snr = estimate_snr(recordings).mean()

            # adjust dipole amplitudes
            scale = 10**((snr-current_snr)/20)
            data  *= scale

        else:
            if not at_least_one_source:
                print("\033[93mSNR can't be adjusted due to missing sources.\033[0m")      
            if not at_least_one_noise:    
                print("\033[93mSNR can't be adjusted due to missing noises.\033[0m")


        # -------------------------
        # Project to sensor space
        # -------------------------

        # convert to MNE objects
        stc = make_stc(
            src      = self.hm.fwd["src"],
            activity = data + noise_dipole,         # dipole amplitudes with noise
            tmin     = self.hm.sens.times[0],
            tstep    = 1/self.hm.sens.info["sfreq"],
        )

        # apply forward
        recordings       = mne.apply_forward(self.hm.fwd, stc, self.hm.sens.info, verbose="warning")
        recordings.data += noise_sensor

        # ----------------
        # Manage outputs
        # ----------------

        ground_truth = make_stc(
            src      = self.hm.fwd["src"],
            activity = data,                            # dipole amplitudes without noise
            tmin     = self.hm.sens.times[0],
            tstep    = 1/self.hm.sens.info["sfreq"],
        )

        if not return_stc_noise:
            return ground_truth, recordings 
        else:
            return ground_truth, recordings, stc
    

    def __str__(self):
        summary = "\033[92mSimulator:\033[0m\n" \
                  "Simulator for source propagation\n\n"
        
        summary += self.hm.__str__() + "\n"

        if self.source is not None:
            summary += self.source.__str__() + "\n" 
        else:
            summary += "\033[94mSource:\033[0m\n" + "\033[91mNone\033[0m\n" + "\n"

        if self.noise is not None:
            summary += self.noise.__str__() + "\n" 
        else:
            summary += "\033[94mNoise:\033[0m\n" + "\033[91mNone\033[0m\n" + "\n"
        
        return summary