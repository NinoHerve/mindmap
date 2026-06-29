"""
DOC String
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np

import mne

from mindmap.core.noise import pink_noise, white_noise
from mindmap.analysis.functional.eeg.metric import estimate_raw_snr
from mindmap.analysis.functional.inverse.io import make_stc
from mindmap.analysis.simulation.components import Source, HeadModel


@dataclass
class PropagationConfig:
    """Groups all variables related to how the signal spreads in space and time."""
    conduction_speed: float = 500    # [mm/ms]
    nodal_delay: float = 0.1         # [ms]
    spatial_decay: float = 0.5       # [0-1]
    jitter: float = 0.0              # [ms]

    def to_dict(self) -> dict:
        """To dictionary."""
        return asdict(self)


@dataclass
class EpochConfig:
    """Manages the dynamic timing windows for raw analysis, epoching, and SNR calculations."""
    tmin: float = -0.45                                     # [s]
    tmax: float = 0.45                                      # [s]
    baseline: tuple[float, float] = (-0.45, -0.2)           # [s]
    signal_window: tuple[float, float] = (-0.1, 0.45)       # [s]

    def to_dict(self) -> dict:
        """To dictionary."""
        return asdict(self)


class Simulator:
    """
    Coordinates the simulation pipeline by combining structural head models,
    source activations, and noise to output raw recordings and ground truths.
    """
    def __init__(self, head_model: HeadModel, epoch_config:EpochConfig = None):
        self.hm = head_model
        self.sfreq = head_model.sens.info["sfreq"]
        self.times = head_model.sens.times

        # If no config is provided, fall back to a default instance
        self.epoch_config = epoch_config if epoch_config is not None else EpochConfig()

    def save(
        self,
        folder: str | Path,
        sources: list[Source] | Source = None,
        prop_config: PropagationConfig = None,
    ):
        """Save Simulator."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # Save global simulator configuration
        global_meta = {
            "epoch_config": self.epoch_config.to_dict(),
            "has_run_history": sources is not None
        }
        with open(folder/"simulator_config.json", "w", encoding="utf-8") as f:
            json.dump(global_meta, f, indent=4)

        # 2. Delegate deep saving to HeadModel
        if self.hm:
            self.hm.save(folder)
        
        # 3. Handle optional Runtime Parameters (PropagationConfig)
        if prop_config is not None:
            with open(folder/"propagation_config.json", "w", encoding="utf-8") as f:
                json.dump(prop_config.to_dict(), f, indent=4)

        # 4. Handle optional Runtime Parameters (Sources List)
        if sources is not None:
            sources_dir = folder / "sources"
            sources_dir.mkdir(exist_ok=True)

            # Normalize single source instance to list
            sources_list = [sources] if isinstance(sources, Source) else sources

            for idx, src in enumerate(sources_list):
                src_sub_dir = sources_dir / f"source_{idx}.json"
                src.save(src_sub_dir)

    @classmethod
    def load(
        cls,
        folder_path: str | Path
    ) -> tuple["Simulator", list[Source] | None, PropagationConfig | None]:
        """
        Reconstructs the simulator state alongside any optional preserved runtime context.

        Returns
        -------
        simulator: Simulator
            The ready-to-run reconstructed environment.
        
        sources: list[Source] | None
            Preserved historical runtime sources (if exported).

        prop_config: PropagationConfig | None
            Preserved historical runtime propagation variable (if exported).
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Could not find folder '{folder}'")

        # 1. Parse global configurations
        with open(folder/"simulator_config.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        epoch_config = EpochConfig(**meta["epoch_config"])

        # 2. Re-instantiate the structural HeadModel
        head_model = HeadModel.load(folder)

        # 3. Instantiate the core Simulator shell
        simulator = cls(head_model=head_model, epoch_config=epoch_config)

        # 4. Extract runtime propagation configurations if present
        prop_config = None
        prop_file = folder / "propagation_config.json"
        if prop_file.exists():
            with open(prop_file, "r", encoding="utf-8") as f:
                prop_data = json.load(f)
            prop_config = PropagationConfig(**prop_data)

        # 5. Extract multiple recursive source signals if present
        sources = None
        sources_dir = folder/"sources"
        if meta["has_run_history"] and sources_dir.exists():
            sources = []
            # Find all source_*.json files and sort them sequentially
            source_files = sorted(sources_dir.glob("source_*.json"), key=lambda d: d.name)
            
            for src_file in source_files:
                loaded_source = Source.load(src_file)
                sources.append(loaded_source)

        return simulator, sources, prop_config

    def compute_propagation(
        self,
        source_indices: np.ndarray,
        waveform: np.ndarray,
        config: PropagationConfig,
    ):
        """
        Propagation with time-delays and spatial fading.

        Parameters
        ----------
        source_indices: np.ndarray
            Source indices.
        
        waveform: np.ndarray
            Source signal.

        config: PropagationConfig
            Groups all variables related to how the signal spreads in
            space and time.
            - conduction_velocity: travel speed along tract.
            - nodal_delay: processing time at node.
            - jitter: uncertainties in total delays.
            - spatial_decay: signal decay from node to node.
        """
        # 1. Get temporal delays and spatial hop-distances
        delays = self.hm.compute_delay_from_source(
            source_indices,
            config.conduction_speed,
            config.nodal_delay,
            config.jitter
        )
        hops = self.hm.compute_distance_from_source(source_indices, weight="hop")

        # handle isolated nodes (infinity)
        delays[np.isinf(delays)] = 1e6
        hops[np.isinf(hops)] = 1e6

        # Convert seconds to sample offsets
        sample_delays = np.round(delays/1000 * self.sfreq).astype(int)

        # Calculate spatial weights: w^hops
        weights = config.spatial_decay ** hops

        # Instanciate output
        n_dipoles = self.hm.fwd["nsource"]
        n_times = len(self.times)
        data = np.zeros((n_dipoles, n_times))

        # 2. Optimized Loop: Group by delay to save time
        unique_delays = np.unique(sample_delays)

        for dt in unique_delays:
            if dt >= n_times:
                continue

            # Get dipoles that share this delay
            nodes = np.where(sample_delays == dt)[0]

            # create shifted signal
            shifted_sig = np.zeros(n_times)

            if dt > 0:
                copy_len = n_times - dt
                shifted_sig[dt:] = waveform[:copy_len]
            else:
                copy_len = n_times + dt
                shifted_sig[:copy_len] = waveform[-dt:]

            # Apply spatial weights to all nodes sharing this delay
            # shape: (len(nodes), 1) * (1, n_times)
            data[nodes, :] = weights[nodes, np.newaxis] * shifted_sig[np.newaxis, :]

        return data

    def run(
        self,
        sources: Source | list[Source],
        prop_config: PropagationConfig = None,
        snr_target: float = None
    ):
        """
        Runs the simulation for multiple sources and handles SNR.

        Parameters
        ----------
        sources: Source | list[Source]
            Defines source activity (location and waveform).

        prop_config: PropagationConfig
            Propagation configuration.

        snr_target: float
            Single trial signal to noise ratio.
        """
        n_dipoles = self.hm.fwd["nsource"]
        n_sensors = len(self.hm.sens.ch_names)
        n_times = len(self.times)

        # -------------------------
        # 1. Generate Clean Signal
        # -------------------------

        activity = np.zeros((n_dipoles, n_times))
        events = np.zeros(n_times)

        if isinstance(sources, Source):
            sources = [sources]

        # compute feedforward propagation for each source and superpose results
        for src in sources:
            # initial signal
            res = src.signal.generate(self.times)

            # dipoles belonging to source
            targets = src.select_targets(self.hm.src, self.hm.parc, self.hm.ras2tkras)
            targets = np.argwhere(targets).squeeze()

            for targ in targets:
                activity += self.compute_propagation(targ, res["data"], config=prop_config)

            # Accumulate events for MNE Annotations
            events += res["events"]

        # Initial scaling to a standard physical unit (1e-8 Am)
        if np.max(np.abs(activity)) > 0:
            activity = (activity / np.max(np.abs(activity))) * 1e-8

        # -------------------
        # 2. Generate Noise
        # -------------------

        # We can look for NoiseSignals in our source or handle global noise
        # For simplicity, let's assum we generate a standard basline noise
        dipole_noise = pink_noise(n_times, n=n_dipoles) * 1e-10 # baseline brain
        sensor_noise = white_noise(n_times, n=n_sensors) * 1e-7 # baseline sensor

        # -------------------
        # 3. SNR Calibration
        # -------------------

        if snr_target is not None:
            # Temporary projection to measure current SNR
            # Signal + dipole noise + sensor noise
            temp_stc = make_stc(
                self.hm.src,
                activity + dipole_noise,
                tmin = self.hm.sens.times[0],
                tstep = 1/self.hm.sens.info["sfreq"],
            )
            temp_sensors = self._project_to_sensors(temp_stc, sensor_noise, events)

            # Use your peak_baseline_snr logic
            # current_snr = peak_baseline_snr(temp_sensors, self.times).mean()
            events_ = mne.events_from_annotations(temp_sensors, verbose="warning")[0]
            current_snr = estimate_raw_snr(
                temp_sensors,
                events = events_,
                event_id = 1,
                tmin = self.epoch_config.tmin,
                tmax = self.epoch_config.tmax,
                baseline = self.epoch_config.baseline,
                signal_window = self.epoch_config.signal_window,
            )   # FIND A WAY TO ADJUST THIS ACCORDING TO SIGNAL/SOURCE/ WHATEVER...

            # Adjust scaling
            scale_factor = snr_target / current_snr.max()
            activity *= scale_factor

        # -------------------------------------
        # 4. Final Projection & MNE Conversion
        # -------------------------------------

        # Ground Truth activity (clean averaged activity)
        ground_truth_stc = make_stc(
            self.hm.src,
            activity,
            tmin = self.hm.sens.times[0],
            tstep = 1/self.hm.sens.info["sfreq"],
        )
        ground_truth_stc = self._average_stc(ground_truth_stc, events)

        # Actual recording
        noisy_stc = make_stc(
            self.hm.src,
            activity + dipole_noise,
            tmin = self.hm.sens.times[0],
            tstep = 1/self.sfreq,
        )
        recordings = self._project_to_sensors(noisy_stc, sensor_noise, events)

        return ground_truth_stc, recordings


    def _project_to_sensors(
        self,
        stc : mne.SourceEstimate,
        noise_sensor : np.ndarray,
        events : np.ndarray = None,
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
            fwd = self.hm.fwd,
            stc = stc,
            info = self.hm.sens.info,
            verbose = "warning",
        )

        if events is not None:
            # build event annotation
            indices = np.argwhere(events == 1).squeeze()

            onsets = self.hm.sens.times[indices]
            durations = np.zeros_like(onsets)
            descriptions = np.ones_like(onsets)

            annotations = mne.Annotations(onsets, durations, descriptions)
            recordings.set_annotations(annotations)

        # add sensor noise
        recordings._data += noise_sensor

        return recordings


    def _average_stc(self, stc, events):
        """
        Crops, baselines, and averages the continuous source ground truth
        exactly how MNE-Python handles raw sensor epoching.
        """   
        # 1. Convert events vector to MNE's 3-column event matrix
        event_indices = np.argwhere(events == 1).squeeze()
        mne_events = np.zeros((len(event_indices), 3), dtype=int)
        mne_events[:, 0] = event_indices
        mne_events[:, 2] = 1    # Event ID

        # 2. Extract epochs

        # sample index relative to event
        start_samp = int(np.round(self.epoch_config.tmin * self.sfreq))
        stop_samp = int(np.round(self.epoch_config.tmax * self.sfreq))

        # calculate baseline for baseline correction
        b_start = int(np.round(self.epoch_config.baseline[0] * self.sfreq)) - start_samp
        b_stop = int(np.round(self.epoch_config.baseline[1] * self.sfreq)) - start_samp

        # allocate trial matrix: (trials, sources, time_samples)
        n_samples = stop_samp - start_samp + 1
        n_events = len(event_indices)
        n_sources = stc.data.shape[0]

        epoch_data = np.zeros((n_events, n_sources, n_samples))

        for i, onset in enumerate(event_indices):
            start_idx = onset + start_samp
            stop_idx = onset + stop_samp + 1

            # slice out the raw data block
            trial = stc.data[:, start_idx:stop_idx]

            # apply baseline correction
            baseline_mean = np.mean(trial[:, b_start:b_stop+1], axis=1, keepdims=True)
            epoch_data[i, :, :] = trial - baseline_mean

        # 3. Average across all trials
        evoked_data = np.mean(epoch_data, axis=0)

        # 4. Turn back into MNE object
        evoked_stc = mne.SourceEstimate(
            data = evoked_data,
            vertices = stc.vertices,
            tmin = start_samp / self.sfreq,
            tstep = 1/self.sfreq,
            subject = stc.subject,
        )

        return evoked_stc
