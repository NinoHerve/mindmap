"""
DOC String
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import warnings

import numpy as np

import mne

from mindmap.core.noise import get_noise_generator
from mindmap.analysis.functional.eeg.metric import (
    estimate_raw_snr_peak, estimate_raw_snr_peak_to_peak,
    estimate_raw_snr_gfp, estimate_raw_snr_power,
)
from mindmap.analysis.functional.inverse.io import make_stc
from mindmap.analysis.simulation.components import Source, HeadModel


@dataclass
class PropagationConfig:
    """Groups all variables related to how the signal spreads in space and time."""
    conduction_speed: float = 500              # [mm/ms] long-range tract speed
    u_fiber_speed: float | None = None         # [mm/ms] U-fiber speed; None disables the split (uniform conduction_speed everywhere)
    u_fiber_max_length: float | None = None    # [mm] edges with length <= this use u_fiber_speed instead of conduction_speed
    nodal_delay: float = 0.1                   # [ms]
    spatial_decay: float = 0.5                 # [0-1]
    jitter: float = 0.0                        # [ms] per-hop delay std; accumulates as jitter*sqrt(hops) along the path, not a flat total-path perturbation

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


@dataclass
class SNRConfig:
    """
    Groups all variables related to single-trial SNR estimation/calibration.
    `method`/window defaults reproduce the simulator's historical peak-amplitude
    estimator over the full `EpochConfig` windows, but `channel_selection`
    defaults to "auto" rather than the old (implicit) "all channels" search --
    that search is a channel-count-dependent "winner's curse" that makes low
    targets unreachable on many-channel montages (confirmed empirically: e.g.
    unreachable at target=2.5 with 203 channels, regardless of window). The
    calibration algorithm itself (same-noise-draw iterative correction, see
    `Simulator.run`) is always the new, corrected one regardless of these
    defaults.

    Note: `noise_cov`, when set, is not JSON-serializable -- don't call
    `to_dict()`/persist this config if it's set (mirrors how `Simulator.save`
    only ever persists `PropagationConfig`, never this class).
    """
    method: str = "peak"                               # "peak" | "peak_to_peak" | "gfp" | "power"
    spike_window: tuple[float, float] | None = None     # [s]; None -> epoch_config.signal_window
    baseline: tuple[float, float] | None = None         # [s]; None -> epoch_config.baseline
    whiten: bool = False                                # only used by method="gfp"
    noise_cov: "mne.Covariance | None" = None           # only used by method="gfp" when whiten=True
    channel_selection: str = "auto"                     # "auto" | "all" -- ignored by method="gfp"
    max_iter: int = 8
    tol: float = 0.05                                   # relative tolerance (fraction of snr_target)

    def to_dict(self) -> dict:
        """To dictionary."""
        return asdict(self)


@dataclass
class NoiseConfig:
    """
    Groups physical noise-GENERATION parameters (absolute scale/color), as
    opposed to SNRConfig, which governs SNR estimation only.

    `sensor_noise_scale_uv` is a small, FIXED instrument/electrode-noise
    floor (RMS, uV) -- a property of the recording hardware, independent of
    the forward model, so it's kept consistent across head models/sources
    rather than solved for. 2uV is a placeholder order-of-magnitude
    assumption (research-grade amplifier noise floors are typically sub-uV
    to a few uV) representing the smaller, hardware-consistent slice of
    real background EEG -- most of the ~10-20uV RMS literature figure for
    scalp background is genuine ongoing physiological activity, not
    instrument noise.

    `target_sensor_noise_uv` is the TOTAL desired sensor-level baseline
    noise (RMS, uV) -- sensor_noise's fixed contribution plus dipole_noise's
    forward-projected contribution combined, since the two are independent
    random sources and combine in quadrature at the delivered channel.
    Default 15uV is the literature midpoint of scalp EEG background noise
    (~10-20uV RMS). `Simulator.run` SOLVES for the dipole_noise Am-scale
    needed so this specific head model's forward-projected contribution
    makes up the remainder (`sqrt(target**2 - sensor_noise_scale_uv**2)`),
    rather than dipole_noise_scale being a free, independently-set field --
    dipole_noise's sensor-level contribution depends on forward gain (source
    depth/orientation/head geometry), which varies per head model, so a
    fixed Am constant can't be calibrated to a fixed uV target the way
    sensor-space noise can. The historical hardcoded sensor_noise scale
    (1e-7 V, ~0.1uV total) delivered ~2uV peaks at snr_target=5, ~1-2 orders
    of magnitude below real interictal spikes (~50-300uV peak-to-peak) --
    confirmed against the real MNE sample forward solution.

    The actual solved dipole_noise Am-scale used for a given run is exposed
    afterward as `SimulationResult.dipole_noise_scale`, mirroring how
    `SimulationResult.snr_picks` exposes the auto-selected SNR channel --
    useful for logging/reproducibility since it's head-model-dependent, not
    a fixed config value.
    """
    sensor_noise_scale_uv: float = 2.0     # [uV RMS] fixed instrument/electrode noise floor
    target_sensor_noise_uv: float = 15.0   # [uV RMS] TOTAL desired sensor-level baseline noise
    sensor_noise_color: str = "white"
    dipole_noise_color: str = "pink"

    def to_dict(self) -> dict:
        """To dictionary."""
        return asdict(self)


class SimulationResult:
    """
    Per-call container returned by Simulator.run(). Holds this run's clean
    activity and the two noise realizations already drawn, so both the
    noisy and clean variants of ground truth/EEG can be built on demand
    without re-running propagation or SNR/noise calibration (the expensive
    parts of a run). Never stored as mutable state on Simulator itself,
    which is reused across many run() calls.

    Each getter is lazy and memoized per with_noise value -- the first call
    for a given value builds and caches it; later calls with the same value
    return the same cached object (call .copy() first if you need to mutate
    it non-destructively, e.g. before raw.set_eeg_reference(...), which
    mutates its Raw in place).

    snr_picks: the channel(s) SNR calibration auto-selected for this run
    (SNRConfig.channel_selection="auto", the default for non-gfp methods).
    Re-measuring the delivered recording MUST use these same picks --
    searching all channels again reintroduces the channel-selection bias
    calibration was built to avoid, and can land on a different channel than
    the one actually calibrated.

    dipole_noise_scale: the actual dipole_noise Am-scale solved for by the
    NoiseConfig joint-calibration step -- head-model-dependent (varies with
    this source's forward gain), not a fixed config value, exposed here for
    logging/reproducibility.
    """
    def __init__(self, simulator, activity, dipole_noise, sensor_noise, events, snr_picks, dipole_noise_scale):
        self._simulator = simulator
        self._activity = activity
        self._dipole_noise = dipole_noise
        self._sensor_noise = sensor_noise
        self._events = events
        self.snr_picks = snr_picks
        self.dipole_noise_scale = dipole_noise_scale
        self._gt_cache = {}
        self._eeg_cache = {}

    def save(self, folder: str | Path):
        """Save this run's raw ingredients (activity, noise realizations,
        events) and per-run facts (snr_picks, dipole_noise_scale) to folder.
        Does NOT save the Simulator itself (use Simulator.save/load for
        that) -- reconstructing via SimulationResult.load needs that same
        Simulator passed back in, since the getters rely on its head
        model/epoch config."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # 1. Save the dense arrays in NumPy's native multi-array format
        np.savez(
            folder/"arrays.npz",
            activity=self._activity,
            dipole_noise=self._dipole_noise,
            sensor_noise=self._sensor_noise,
            events=self._events,
        )

        # 2. Save per-run scalar facts in a JSON file
        meta = {
            "snr_picks": self.snr_picks,
            "dipole_noise_scale": float(self.dipole_noise_scale),
        }
        with open(folder/"result_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    @classmethod
    def load(cls, folder: str | Path, simulator: "Simulator") -> "SimulationResult":
        """Reconstructs a SimulationResult from saved state. `simulator`
        must be the (already loaded, e.g. via Simulator.load) Simulator
        this result belongs to -- it isn't re-saved here, since it's
        reusable structural configuration, not a per-run fact."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Could not find folder '{folder}'")

        with open(folder/"result_config.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        arrays = np.load(folder/"arrays.npz")

        return cls(
            simulator = simulator,
            activity = arrays["activity"],
            dipole_noise = arrays["dipole_noise"],
            sensor_noise = arrays["sensor_noise"],
            events = arrays["events"],
            snr_picks = meta["snr_picks"],
            dipole_noise_scale = meta["dipole_noise_scale"],
        )

    def _make_source_stc(self, with_noise, tstep):
        source = self._activity + self._dipole_noise if with_noise else self._activity
        return make_stc(
            self._simulator.hm.src, source,
            tmin=self._simulator.hm.sens.times[0], tstep=tstep,
        )

    def get_ground_truth(self, with_noise: bool = False) -> mne.SourceEstimate:
        """
        Source-space ground truth, cropped/baselined/trial-averaged
        (Simulator._average_stc) regardless of with_noise, so both variants
        share identical shape/units/vertices and are directly comparable.

        with_noise=False (default): clean activity only -- reproduces the
        historical `gold` output exactly (no dipole noise at all).

        with_noise=True: `activity + dipole_noise`, averaged the same way,
        so ground truth can be compared against delivered EEG under a
        matched noise condition.
        """
        if with_noise not in self._gt_cache:
            stc = self._make_source_stc(with_noise, tstep=1/self._simulator.sfreq)
            self._gt_cache[with_noise] = self._simulator._average_stc(stc, self._events)
        return self._gt_cache[with_noise]

    def get_eeg(self, with_noise: bool = True) -> mne.io.Raw:
        """
        Sensor-space continuous recording via forward projection.

        with_noise=True (default): dipole_noise added source-side plus
        sensor_noise added sensor-side -- reproduces the historical `raw`
        output exactly.

        with_noise=False: fully clean forward projection of activity alone
        -- no dipole_noise, no sensor_noise.
        """
        if with_noise not in self._eeg_cache:
            stc = self._make_source_stc(with_noise, tstep=1/self._simulator.sfreq)
            noise = self._sensor_noise if with_noise else np.zeros_like(self._sensor_noise)
            self._eeg_cache[with_noise] = self._simulator._project_to_sensors(stc, noise, self._events)
        return self._eeg_cache[with_noise]


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
        # 1. Get temporal delays and hop counts, both along the same path: the
        # fastest (smallest-delay) one. The decay therefore counts the relays
        # the activity actually passes through, not the fewest-hop route,
        # which can differ (and be slower).
        delays, hops = self.hm.compute_delay_from_source(
            source_indices,
            config.conduction_speed,
            config.nodal_delay,
            config.jitter,
            config.u_fiber_speed,
            config.u_fiber_max_length,
            return_hops=True,
        )

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
        snr_target: float = None,
        snr_config: SNRConfig = None,
        noise_config: NoiseConfig = None,
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

        snr_config: SNRConfig
            Controls which SNR estimator/windows the calibration targets.
            Defaults to `SNRConfig()` (peak-amplitude estimator, `EpochConfig`
            windows) when None, reproducing historical behavior aside from
            the calibration algorithm itself, which is always the corrected
            same-noise-draw iterative version.

        noise_config: NoiseConfig
            Controls the absolute physical scale/color of the two noise
            sources. Defaults to `NoiseConfig()` (15uV RMS total sensor-level
            background, 2uV of which is a fixed instrument-noise floor and
            the rest solved for via dipole_noise's forward-projected
            contribution) when None.

        Returns
        -------
        result: SimulationResult
            Per-call container exposing get_ground_truth(with_noise) and
            get_eeg(with_noise), plus this run's snr_picks/dipole_noise_scale.
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
        noise_cfg = noise_config if noise_config is not None else NoiseConfig()
        dipole_gen = get_noise_generator(noise_cfg.dipole_noise_color)
        sensor_gen = get_noise_generator(noise_cfg.sensor_noise_color)

        sensor_noise_scale = noise_cfg.sensor_noise_scale_uv * 1e-6
        sensor_noise = sensor_gen(n_times, n=n_sensors) * sensor_noise_scale  # baseline sensor [V]

        # dipole_noise and sensor_noise are independent, so their contributions
        # to the total sensor-level noise floor combine in quadrature. Draw a
        # unit-scale dipole_noise realization, measure ITS forward-projected
        # contribution for this specific head model/source (project it alone,
        # no activity, no sensor_noise), then rescale that SAME realization
        # (same relative shape, just a different absolute scale) to whatever
        # Am level makes the total hit target_sensor_noise_uv -- rather than
        # assuming the dipole contribution stays negligible for every head
        # model (it varies with this source's forward gain).
        #
        # Measured within self.epoch_config.baseline (relative to each event),
        # NOT over the full recording duration: pink noise's variance is
        # scale-dependent (most of its power sits at low frequencies that only
        # complete a swing over long windows), so a full-duration std
        # systematically overestimates what a short ~0.2-0.3s baseline window
        # will actually show -- confirmed empirically (full-duration std ~17uV
        # vs. a 15uV target, but the SAME realization's baseline-window std
        # came in at ~10-14uV). Calibrating against the actual baseline
        # window's own statistics removes this mismatch.
        b0_offset = int(round(self.epoch_config.baseline[0] * self.sfreq))
        b1_offset = int(round(self.epoch_config.baseline[1] * self.sfreq))
        onset_indices = np.where(events == 1)[0]

        def _baseline_window_std(data_1d):
            """Median std within self.epoch_config.baseline, relative to each
            detected event onset (falls back to the full-duration std if no
            onset yields a valid segment, e.g. a noise-only run)."""
            stds = [
                data_1d[seg_start:seg_stop].std()
                for onset_idx in onset_indices
                for seg_start, seg_stop in [(max(onset_idx + b0_offset, 0), max(onset_idx + b1_offset, 0))]
                if seg_stop - seg_start > 1
            ]
            return np.median(stds) if stds else data_1d.std()

        dipole_noise_unit = dipole_gen(n_times, n=n_dipoles)  # unit scale
        probe_stc = make_stc(
            self.hm.src,
            dipole_noise_unit,
            tmin = self.hm.sens.times[0],
            tstep = 1/self.sfreq,
        )
        # dipole_noise_unit is a unit-variance-per-dipole reference array, not
        # a physical current -- across thousands of dipoles x samples its peak
        # routinely reaches several (unitless) sigma, which MNE's forward
        # model reads as an absurd multi-Am dipole moment and warns about.
        # This projection is purely internal (used only to measure the
        # forward-gain factor below, immediately discarded, never delivered),
        # so that specific, known-benign warning is suppressed here -- other
        # warnings still surface normally.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The maximum current magnitude", category=RuntimeWarning)
            probe_sensors = self._project_to_sensors(probe_stc, np.zeros((n_sensors, n_times)), events=None)
        probe_data = probe_sensors.get_data()
        contribution_per_unit = np.median([_baseline_window_std(probe_data[i]) for i in range(n_sensors)])

        target_total_v = noise_cfg.target_sensor_noise_uv * 1e-6
        if target_total_v**2 < sensor_noise_scale**2:
            warnings.warn(
                f"NoiseConfig.target_sensor_noise_uv ({noise_cfg.target_sensor_noise_uv}uV) is "
                f"smaller than sensor_noise_scale_uv ({noise_cfg.sensor_noise_scale_uv}uV) alone; "
                "dipole_noise scale clamped to 0 -- lower sensor_noise_scale_uv or raise the target.",
                stacklevel=2,
            )
        needed_contribution_v = np.sqrt(max(target_total_v**2 - sensor_noise_scale**2, 0.0))

        dipole_noise_scale = needed_contribution_v / contribution_per_unit if contribution_per_unit > 0 else 0.0
        dipole_noise = dipole_noise_unit * dipole_noise_scale  # baseline brain [Am]

        # -------------------
        # 3. SNR Calibration
        # -------------------

        snr_picks = None
        if snr_target is not None:
            snr_cfg = snr_config if snr_config is not None else SNRConfig()
            spike_window = snr_cfg.spike_window if snr_cfg.spike_window is not None else self.epoch_config.signal_window
            baseline_window = snr_cfg.baseline if snr_cfg.baseline is not None else self.epoch_config.baseline

            estimator_map = {
                "peak": estimate_raw_snr_peak,
                "peak_to_peak": estimate_raw_snr_peak_to_peak,
                "gfp": estimate_raw_snr_gfp,
                "power": estimate_raw_snr_power,
            }
            if snr_cfg.method not in estimator_map:
                raise ValueError(
                    f"Unknown SNRConfig.method '{snr_cfg.method}'. "
                    f"Expected one of {list(estimator_map)}."
                )
            estimator = estimator_map[snr_cfg.method]

            # On-the-fly noise covariance for GFP whitening: computed once
            # from this run's own noise realization's baseline segment, so
            # there's no dependency on a downstream mne.compute_covariance
            # call (which currently only happens after Simulator.run returns).
            noise_cov = None
            if snr_cfg.method == "gfp" and snr_cfg.whiten:
                noise_cov = snr_cfg.noise_cov
                if noise_cov is None:
                    # `baseline_window` is relative to each event, not an
                    # absolute raw time range, so epoch around the same
                    # events first (mirrors the mne.compute_covariance(...,
                    # tmax=...) pattern already used in pipeline.py/_run_one)
                    # rather than mne.compute_raw_covariance, which expects
                    # absolute tmin/tmax.
                    noise_only_stc = make_stc(
                        self.hm.src,
                        dipole_noise,
                        tmin = self.hm.sens.times[0],
                        tstep = 1/self.sfreq,
                    )
                    noise_only_sensors = self._project_to_sensors(noise_only_stc, sensor_noise, events)
                    noise_events = mne.events_from_annotations(noise_only_sensors, verbose="warning")[0]
                    noise_epochs = mne.Epochs(
                        noise_only_sensors,
                        noise_events,
                        event_id = 1,
                        tmin = self.epoch_config.tmin,
                        tmax = self.epoch_config.tmax,
                        baseline = baseline_window,
                        preload = True,
                        verbose = "warning",
                    )
                    noise_cov = mne.compute_covariance(
                        noise_epochs,
                        tmax = baseline_window[1],
                        method = "empirical",
                        verbose = "warning",
                    )

            def _measure_snr_raw(scale, picks="eeg"):
                """Project `scale * activity + dipole_noise` and return the estimator's
                raw output (per-channel array, or a single value for method='gfp'),
                using the SAME already-drawn dipole_noise/sensor_noise realization."""
                temp_stc = make_stc(
                    self.hm.src,
                    scale * activity + dipole_noise,
                    tmin = self.hm.sens.times[0],
                    tstep = 1/self.hm.sens.info["sfreq"],
                )
                temp_sensors = self._project_to_sensors(temp_stc, sensor_noise, events)
                events_ = mne.events_from_annotations(temp_sensors, verbose="warning")[0]
                kwargs = dict(
                    events = events_,
                    event_id = 1,
                    tmin = self.epoch_config.tmin,
                    tmax = self.epoch_config.tmax,
                    baseline = baseline_window,
                    signal_window = spike_window,
                )
                if snr_cfg.method == "gfp":
                    return estimator(temp_sensors, noise_cov=noise_cov, **kwargs)
                return estimator(temp_sensors, picks=picks, **kwargs)

            # Auto channel selection: identify the single channel with the best
            # SNR in a noise-negligible regime (activity scaled up until real
            # signal completely swamps the fixed noise realization, so which
            # channel "wins" reflects genuine forward-model gain, not which
            # channel got a lucky noise draw). Restricting calibration/delivery
            # to that one, pre-chosen channel removes the channel-selection
            # "winner's curse" that made low targets unreachable when searching
            # (via .max()) across every channel -- see the SNR-calibration
            # investigation notes. GFP is unaffected: it already aggregates
            # across channels via spatial std, so it has no such bias to fix.
            picks = "eeg"
            if snr_cfg.method != "gfp":
                if snr_cfg.channel_selection == "auto":
                    _CHANNEL_PROBE_SCALE = 1e5
                    # Same benign, internal-only "unrealistic current" warning
                    # as the noise-floor probe above -- this deliberately huge
                    # scale never leaves this function, it's only used to
                    # rank channels by true forward gain.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="The maximum current magnitude", category=RuntimeWarning)
                        huge_snr = _measure_snr_raw(_CHANNEL_PROBE_SCALE, picks="eeg")
                    eeg_idx = mne.pick_types(self.hm.sens.info, eeg=True, exclude=[])
                    eeg_ch_names = [self.hm.sens.info["ch_names"][i] for i in eeg_idx]
                    best_ch = eeg_ch_names[int(np.argmax(huge_snr))]
                    picks = [best_ch]
                elif snr_cfg.channel_selection != "all":
                    raise ValueError(
                        f"Unknown SNRConfig.channel_selection '{snr_cfg.channel_selection}'. "
                        "Expected 'auto' or 'all'."
                    )

                # Refine the noise-floor calibration now that a specific channel
                # is known: dipole_noise's forward-projected contribution varies
                # per channel (forward gain differs channel to channel), so the
                # whole-montage median used to solve dipole_noise_scale earlier
                # is only a rough initial estimate -- confirmed empirically to
                # systematically miss target_sensor_noise_uv on the actual
                # delivered channel by up to ~30-45%. Re-solve using this
                # channel's own measured contribution (reusing the already-
                # computed probe_sensors/probe_data, no extra projection needed,
                # same baseline-window-restricted measurement as above) for an
                # exact match on the channel that actually matters.
                if isinstance(picks, list):
                    ch_idx = probe_sensors.ch_names.index(picks[0])
                    contribution_this_channel = _baseline_window_std(probe_data[ch_idx])
                    dipole_noise_scale = (
                        needed_contribution_v / contribution_this_channel
                        if contribution_this_channel > 0 else 0.0
                    )
                    dipole_noise = dipole_noise_unit * dipole_noise_scale
            snr_picks = picks if snr_cfg.method != "gfp" else None

            def _measure_snr(scale):
                return _measure_snr_raw(scale, picks=picks).max() if snr_cfg.method != "gfp" \
                    else _measure_snr_raw(scale)

            # Same-noise-draw iterative fixed-point correction: reuses the
            # single, already-generated dipole_noise/sensor_noise realization
            # for this run (no new random draws, no Monte Carlo), converging
            # the true amplitude so THIS delivered instance reads out near
            # snr_target -- fixes the bias from the old one-shot linear
            # rescale, which extrapolated from an arbitrarily-large
            # pre-scaling amplitude far outside the target's own SNR regime.
            # Restricting to one channel above removes the dominant bias, but
            # a smaller one remains (the windowed max-in-time search within
            # that one channel is itself mildly nonlinear at low SNR), so
            # iterating a few more cheap steps is still needed to close it --
            # confirmed empirically: one-shot alone left ~34% error, a few
            # more iterations brought it under tol.
            scale = snr_target
            current = _measure_snr(scale)
            for _ in range(snr_cfg.max_iter):
                if current <= 0 or abs(current - snr_target) < snr_cfg.tol * snr_target:
                    break
                scale *= snr_target / current
                current = _measure_snr(scale)
            else:
                warnings.warn(
                    f"SNR calibration (method='{snr_cfg.method}') did not converge "
                    f"within {snr_cfg.max_iter} iterations (target={snr_target}, "
                    f"last measured={current:.3f}); this window/method combination "
                    "may sit at or below its noise floor for this run.",
                    stacklevel=2,
                )

            activity *= scale

        return SimulationResult(
            simulator=self,
            activity=activity,
            dipole_noise=dipole_noise,
            sensor_noise=sensor_noise,
            events=events,
            snr_picks=snr_picks,
            dipole_noise_scale=dipole_noise_scale,
        )


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
