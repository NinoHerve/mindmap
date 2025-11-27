"""
Module Name: simulation.py
Description: Classes to run EEG simulations of source propagation.

This module provides classes to build your own EEG simulation of source propagation. The `Simulator`
class orchestrates the source propagation with the help of the following subclasses:
- `Source`: Manages the source location and waveform.
- `HeadModel`: Manages the head model, including dipole positions, sensor positions, 
dipole-to-sensor projections, and connections between dipoles.
- `Noise`: Manages the noise of the different components, such as dipole noise and sensor noise.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""
from typing import Callable

import numpy as np
import pandas as pd
import networkx as nx
import mne
import seaborn as sns

from mindmap.head import get_dipole_positions, align_freesurfer2subject, Parcellation, make_stc
from mindmap.metric import peak_baseline_snr
from mindmap.viz import plot_brain


# -------------
# Source class
# -------------


class Source:
    """
    Class to handle source information, including source locations and waveforms.
    """
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

        out = {
            "pattern"  : pattern,
            "waveform" : waveform,
            "targets"  : np.argwhere(targets).squeeze(),        # mask to ids
        }

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

        out = {
            "data"    : waveform,
            "wavelet" : window,
            "times"   : times,
            "events"  : train,
        }

        return out

    def select_targets(self, source_space, parc=None, tkras2ras=None):
        """
        Retrieves dipole ids belonging to source.
        """
        dipoles = get_dipole_positions(source_space)

        if self.loc.dtype == int:
            targets = np.zeros(len(dipoles), dtype=bool)
            targets[self.loc] = True
        else:
            assert parc is not None, "Need parcellation to build stimulus because source locations are given as brain regions."
            assert tkras2ras is not None, "Need tkras2ras transform to algin dipoles with T1-space."

            dipoles = align_freesurfer2subject(dipoles*1000, tkras2ras)
            labels  = parc.label_points(dipoles)
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
        plot_brain(src, targets.astype(int), absolute=True).show()


    def __str__(self):
        summary = "\033[94mSource:\033[0m\n" \
                  f"- locations: {self.loc}\n" \
                  f"- waveform width: {self.width} [ms]\n" \
                  f"- Onset: {self.on} [ms] \n" \
                  f"- Offset: {self.off} [ms] \n" \
                  f"- Repeat: {self.rep} [ms]\n"

        return summary


# -----------------
# Head Model class
# -----------------

class HeadModel:
    """
    The HeadModel class handles all physical properties of the model, such as dipole positions, 
    sensor positions, and dipole connections. This class is primarily used to measure distances 
    through the dipole network in order to predict time delays.
    """
    sens      : mne.Epochs | mne.Evoked
    fwd       : mne.Forward
    weights   : np.ndarray
    lenghts   : np.ndarray
    speed_a   : float
    d_syn     : float
    parc      : Parcellation
    tkras2ras : np.ndarray

    def __init__(self, sensors, forward, connectome_weights, connectome_lengths, axonal_speed=500, synaptic_delay=0.001, parcellation=None, tkras2ras=None):
        """
        Parameters
        ----------
        sensors: mne.info

        forward: mne.Forward
            Forward solution. Projects dipole activity to sensor space. Also contains source space 
            information.

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
        self.cras    = tkras2ras
        self.delays  = self.compute_delay_matrix()

    def compute_delay_matrix(self):
        """Computes a delay adjacency matrix."""
        return self.lengths/self.speed_a + (self.lengths>0)*self.d_syn

    def compute_distance_from_source(self, source, weight="delay", distance_in_hop=False):
        """
        Compute distance (delay, length or number of hops) between a target node and all other 
        nodes.

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
            graph_dist = self.delays
        elif weight == "length":
            graph_dist = self.lengths
        elif weight == "hop":
            graph_dist = (self.lengths > 0).astype(int)

        else:
            raise ValueError(f"Unknown weight type {weight} for computing distance to source.")

        # compute shortest paths
        graph_dist = nx.Graph(graph_dist)
        paths      = nx.shortest_path(graph_dist, source=source, weight="weight")

        # order paths by node id
        dist = np.ones(graph_dist.number_of_nodes())*np.inf  # isolated nodes are not in paths, set infinity as default.

        for n, path in paths.items():
            if distance_in_hop:
                dist[n] = len(path)-1       # don't count starting node
            else:
                dist[n] = nx.path_weight(graph_dist, path, weight="weight")

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
                  f"- tkras2ras: {True if self.cras is not None else False} \n" \

        return summary



# -------------
# Noise class
# -------------

class Noise:
    """
    Handles noise for every component. Noise is defined by a colour (white, pink, brown, ...) and
    a scale.
    """
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
            n = 1
            t = shape[0]
        else:
            n = shape[0]
            t = shape[1]

        return signal + self.generate_noise(t,n)


    def generate_noise(self, t, n=1, component="dipole"):
        """
        Generate n noise samples depending on type of component.

        Parameters
        ----------
        t: int
            Number of time samples.

        n: int
            Number of component samples.

        component: str
            Type of component (e.g. dipole, sensor, ...)
        """
        assert component in pd.unique(self.noises.index), f"Only the noise of the following components have been provided to the Noise class: {pd.unique(self.noises.index)}"

        category = self.noises.loc[component]["category"]
        scale    = self.noises.loc[component]["scale"]

        noise_generator = self.get_noise(category)

        noise = noise_generator(t, n) * scale

        return noise

    def __str__(self):

        summary = "\033[94mNoise:\033[0m\n"

        for component in self.noises.index:
            summary += f"- '{component}' noise:\n" \
                f"  - category: {self.noises.loc[component]['category']}\n" \
                f"  - scale: {self.noises.loc[component]['scale']}\n"

        return summary

    def get_noise(self, colour) -> Callable:
        """Get appropriate noise generator."""

        mapping = {
            "white": white_noise,
            "blue": blue_noise,
            "violet": violet_noise,
            "pink": pink_noise,
            "brown": brownian_noise,
            "brownian": brownian_noise,
        }

        try:
            noise_generator = mapping[colour]
        except KeyError as e:
            raise ValueError(f"'{colour}' noise has not been implemented in the mindmap package yet.") from e

        return noise_generator


def color_noise(t, n=1, filt=lambda f: 1):
    """
    Produces colored noise. Inspired from https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python

    Parameters
    ----------
    t: int
        Number of samples per series.

    n: int
        Number of series.

    filt: function
        Frequency filter (depends on color)
        - f* = f (= white noise)
        - f* = 1/f (= pink noise)
        - ...

    Returns
    -------
    color: np.ndarray(n,1)
        Colored noise 

    Note: To produce colored noise we generate a white noise signal and fix the frequencies according to the filter.
    """
    color = np.zeros((n,t))

    for i in range(n):
        # generate white noise
        white     = np.random.randn(t)
        white_fft = np.fft.rfft(white)

        # fix the frequency space
        scale     = filt(np.fft.rfftfreq(t))
        scale     = scale / np.sqrt(np.mean(scale**2))      # preserve white noise energy
        color_fft = white_fft * scale

        # inverse Fourier
        color[i,:] = np.fft.irfft(color_fft)

    return color.squeeze()

def colored_noise(filt):
    """Applies noise with given filter"""
    return lambda t, n=1: color_noise(t, n, filt)

@colored_noise
def white_noise(_):
    """White noise filter"""
    return 1

@colored_noise
def blue_noise(f):
    """Blue noise filter"""
    return np.sqrt(f)

@colored_noise
def violet_noise(f):
    """Violet noise filter"""
    return f

@colored_noise
def brownian_noise(f):
    """Brownian noise filter"""
    return 1/np.where(f==0, float("inf"),f)

@colored_noise
def pink_noise(f):
    """Pink noise filter"""
    return 1/np.where(f==0, float("inf"), np.sqrt(f))




# ----------------
# Simulator class
# ----------------

class Simulator:
    """
    Regroups a Source, a HeadModel and a Noise to orchestrate source propagation.
    """
    source : Source
    hm     : HeadModel
    noise  : Noise

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
        delays[delays==np.inf] = 1e6                                                    # replace isolated nodes (distance is infinity) with very big number
        delays = np.round(delays * self.hm.sens.info["sfreq"]).astype(int)

        # weights
        hops    = self.hm.compute_distance_from_source(source=source, weight="hop", distance_in_hop=True)
        hops[hops==np.inf] = 1e6
        weights = w**hops

        # dimensions
        n = self.hm.fwd["nsource"]
        t = len(self.hm.sens.times)

        # compute source propagation
        data = np.zeros((n,t))

        for dt in np.unique(delays):
            nodes = np.argwhere(delays==dt).squeeze()

            if dt == 0:
                signals = waveform
            else:
                # delayed signal
                signals = np.pad(waveform[:-dt], (np.min([len(waveform), dt]),0), mode="constant", constant_values=0)    # pads new time points with 0. np.min([len(waveform), dt]) selects the size of the simulation instead of dt in case dt is larger than the simulation duration

            data[nodes,:] = signals.reshape(1,-1) * weights[nodes].reshape(-1,1)   # weighted + delayed signal

        return data

    def run(self, w, snr, return_stc_noise=False, recording_type="raw"):
        """
        Simulate a source propagation.

        Parameters
        ----------
        w: float [0-1]
            Weight decay.

        snr: float
            Average peak to power snr to aim for in the EEG data. Single peak snr are measured as
            the ratio between the absolute maximum of a channel and the average power during the 
            baseline of that channel. The final snr is an average across all channels.
        
        return_stc_noise: bool
            Flag to return brain simulation with noise too.

        recording_type: str
            Type of MNE object to return:
            - raw: mne.io.Raw (raw recordings)
            - ave: mne.Evoked (averaged epochs)

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
        n = self.hm.fwd["nsource"]
        m = len(self.hm.sens.ch_names)
        t = len(self.hm.sens.times)


        # -----------------------------
        # Simulate source propagations
        # -----------------------------

        # output
        data = np.zeros((n,t))

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
            noise_dipole = self.noise.generate_noise(t, n, component="dipole")  # shape T, N will be swapped during output (default N is 1 but T is mandatory so the function has to have mandatory first)
            at_least_one_noise = True
        else:
            noise_dipole = np.zeros((n,t))

        if self.noise is not None and "sensor" in self.noise.noises.index:
            noise_sensor = self.noise.generate_noise(t, m, component="sensor")
            at_least_one_noise = True
        else:
            noise_sensor = np.zeros((m,t))


        # ------------
        # Adjust SNR
        # ------------

        # check if SNR can be adjusted
        if at_least_one_source and at_least_one_noise:

            # project the current simulation to sensors

            # convert to MNE objects
            stc = make_stc(
                source_space = self.hm.fwd["src"],
                activity     = data + noise_dipole,         # current dipole amplitudes with noise
                tmin         = self.hm.sens.times[0],
                tstep        = 1/self.hm.sens.info["sfreq"],
            )

            # apply forward
            if recording_type == "raw":
                recordings = self.apply_forward_raw(stc, noise_sensor, waveform["events"])
            else:
                recordings = self.apply_forward_evoked(stc, noise_sensor)

            # compute average peak to power SNR
            current_snr = peak_baseline_snr(recordings).mean() # average accross channels

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
            source_space = self.hm.fwd["src"],
            activity     = data + noise_dipole,         # dipole amplitudes with noise
            tmin         = self.hm.sens.times[0],
            tstep        = 1/self.hm.sens.info["sfreq"],
        )

        # apply forward
        if recording_type == "raw":
            events     = waveform["events"] if at_least_one_source else None
            recordings = self.apply_forward_raw(stc, noise_sensor, events)

        else:
            recordings = self.apply_forward_evoked(stc, noise_sensor)

        # ----------------
        # Manage outputs
        # ----------------

        ground_truth = make_stc(
            source_space = self.hm.fwd["src"],
            activity     = data,                            # dipole amplitudes without noise
            tmin         = self.hm.sens.times[0],
            tstep        = 1/self.hm.sens.info["sfreq"],
        )

        if not return_stc_noise:
            return ground_truth, recordings

        return ground_truth, recordings, stc

    def apply_forward_raw(
            self,
            stc          : mne.SourceEstimate,
            noise_sensor : np.ndarray,
            events       : np.ndarray = None,
            ) -> mne.io.Raw:
        """
        Apply forward model and make a io.Raw object.

        Parameters
        ----------
        stc: mne.SourceEstimates
            Source time currents.

        noise_sensor: np.ndarray
            Sensor noise.
            
        events: np.ndarray
            Spike envents.

        Returns
        -------
        recordings: mne.io.Raw
        """
        recordings  = mne.apply_forward_raw(
            fwd     = self.hm.fwd,
            stc     = stc,
            info    = self.hm.sens.info,
            verbose = "warning",
        )

        if events is not None:
            # build event annotation
            indices = np.argwhere(events == 1).squeeze()

            onsets       = self.hm.sens.times[indices]
            durations    = np.zeros_like(onsets)
            descriptions = np.ones_like(onsets)

            annotations = mne.Annotations(onsets, durations, descriptions)
            recordings.set_annotations(annotations)

        # add sensor noise
        recordings._data += noise_sensor

        return recordings


    def apply_forward_evoked(
        self,
        stc          : mne.SourceEstimate,
        noise_sensor : np.ndarray,
        ) -> mne.Evoked:
        """
        Apply forward model and make an Evoked object.
        
        Parameters
        ----------
        stc: mne.SourceEstimates
            Source time currents.

        noise_sensor: np.ndarray
            Sensor noise.

        Returns
        -------
        recordings: mne.Evoked
        """
        recordings = mne.apply_forward(
                fwd     = self.hm.fwd,
                stc     = stc,
                info    = self.hm.sens.info,
                verbose = "warning"
            )

        # add sensor noise
        recordings.data += noise_sensor

        return recordings



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
