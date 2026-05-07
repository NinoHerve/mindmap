"""
DOC String
"""

import numpy as np

from mindmap.core.noise import pink_noise, white_noise
from mindmap.analysis.functional.inverse.simulation import make_stc
from mindmap.analysis.simulation.components import Source, HeadModel


class Simulator:
    """
    DOC String:
    """
    def __init__(self, head_model: HeadModel):
        self.hm = head_model
        self.sfreq = head_model.sens.info["sfreq"]
        self.times = head_model.sens.times

    def compute_propagation(self, source_indices, waveform, spatial_decay=0.5):
        """
        Propagation with time-delays and spatial fading.
        """
        # 1. Get temporal delays and spatial hop-distances
        delays = self.hm.compute_distance_from_source(source_indices, weight="delay")
        hops = self.hm.compute_distance_from_source(source_indices, weight="hop")

        # handle isolated nodes (infinity)
        delays[np.isinf(delays)] = 1e6
        hops[np.isinf(hops)] = 1e6

        # Convert seconds to sample offsets
        sample_delays = np.round(delays * self.sfreq).astype(int)

        # Calculate spatial weights: w^hops
        weights = spatial_decay ** hops

        # Instanciate output
        n_dipoles = self.hm.fwd["nsource"]
        n_times = len(self.times)
        data = np.zeros((n_dipoles, n_times))

        # 2. Optimized Loop: Group by delay to save time
        unique_delays = np.unique(sample_delays)

        for dt in unique_delays:
            if dt >= n_times or dt < 0:
                continue

            # Get dipoles that share this delay
            nodes = np.where(sample_delays == dt)[0]

            # create shifted signal
            shifted_sig = np.zeros(n_times)
            copy_len = n_times - dt
            shifted_sig[dt:] = waveform[:copy_len]

            # Apply spatial weights to all nodes sharing this delay
            # shape: (len(nodes), 1) * (1, n_times)
            data[nodes, :] = weights[nodes, np.newaxis] * shifted_sig[np.newaxis, :]

        return data

    def run(self, sources, spatial_decay=0.5, snr=None):
        """
        Runs the simulation for multiple sources and handles SNR.
        """
        n_dipoles = self.hm.fwd["nsource"]
        n_sensors = len(self.hm.sens.ch_names)
        n_times = len(self.times)

        # -------------------------
        # 1. Generate Clean Signal
        # -------------------------
        
        clean_activity = np.zeros((n_dipoles, n_times))
        events = np.zeros(n_times)

        for src in sources:
            res = src.signal.generate(self.times)
            clean_activity += self.compute_propagation(
                src.indices, res["data"], spatial_decay=spatial_decay,
            )
            # Accumulate events for MNE Annotations
            events += res["events"]

        # Initial scaling to a standard physical unit (1e-8 Am)
        if np.max(np.abs(clean_activity)) > 0:
            clean_activity = (clean_activity / np.max(np.abs(clean_activity))) * 1e-8

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

        if snr is not None:
            # Temporary projection to measure current SNR
            # Signal (S) + Noise (N)
            temp_activity = clean_activity + dipole_noise
            temp_sensors = (self.hm.fwd["sol"]["data"] @ temp_activity) + sensor_noise

            # Use your peak_baseline_snr logic
            # current_snr = peak_baseline_snr(temp_sensors, self.times).mean()
            current_snr = self._claculate_current_snr(temp_sensors)

            # Adjust scaling: scale = 10^((target - current)/20)
            adjustment_db = snr - current_snr
            scale_factor = 10 ** (adjustment_db / 20)
            clean_activity *= scale_factor

        # -------------------------------------
        # 4. Final Projection & MNE Conversion
        # -------------------------------------

        noisy_activity = clean_activity + dipole_noise

        # Ground Truth STC
        stc_ground_truth = make_stc(clean_activity)

        # Sensor Recordings (Raw or Evoked)
        recordings = self._project_to_sensors(
            noisy_activity, sensor_noise, events, recording_type
        )

        if return_stc_noise:
            return stc_ground_truth, recordings, self._make_stc(noisy_activity)
    
        return stc_ground_truth, recordings