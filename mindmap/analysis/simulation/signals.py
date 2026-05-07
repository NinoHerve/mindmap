"""
DOC String
"""

from abc import ABC, abstractmethod

import numpy as np

from mindmap.core.noise import get_noise_generator

class Signal(ABC):
    """Base class for all temporal waveforms."""
    def __init__(self, name="Signal", event_id=1):
        self.name = name
        self.event_id = event_id

    def get_mne_metadata(self):
        """Returns metadata for MNE conversion."""
        return {
            "description": self.name,
            "event_id": self.event_id,
            "offset_id": self.event_id + 100,
        }

    @abstractmethod
    def generate(self, times: np.ndarray) -> dict:
        """
        Should return a dictionary containing at least:
        - 'data': the 1D time series
        - 'events': binary array of event onsets (if applicable)
        """
        raise NotImplementedError("Implement a `generate` method for Signal Childs.")

    def __str__(self):
        return f"Type: {self.name} (ID: {self.event_id})"
    
    def __add__(self, other):
        if isinstance(other, Signal):
            return CombinedSignal(self, other)
        return NotImplemented
    
class CombinedSignal(Signal):
    def __init__(self, sig1, sig2):
        super().__init__(name=f"{sig1.name} + {sig2.name}")
        self.signals = [sig1, sig2]

    def generate(self, times):
        data = np.zeros_like(times)
        events = np.zeros_like(times)
        for sig in self.signals:
            res = sig.generate(times)
            data += res["data"]
            events += res["events"]
        return {"data": data, "events": events}


class SpikeSignal(Signal):
    """
    Spike train with double Hanning window waveform.
    """
    def __init__(
        self,
        width = 0.03,
        onset = None,
        offset = None,
        repeat = None,
        scale = 1e-8,
        event_id = 1,
    ):
        super().__init__(name="Spike", event_id=event_id)
        self.width = width
        self.on = onset
        self.off = offset
        self.rep = repeat
        self.scale = scale

    def generate(self, times):
        sfreq = 1 / (times[1] - times[0])
        onset = self.on if self.on is not None else times[0]
        offset = self.off if self.off is not None else times[-1]

        # 1. Build Spike Train
        train = np.zeros_like(times)
        train[int(onset * sfreq)] = 1
        if self.rep:
            t = onset + self.rep
            while t < offset:
                idx = int(round(t * sfreq))
                if idx < len(train):
                    train[idx] = 1
                t += self.rep

        # 2. Build Double Hanning Window
        w1, w2 = int(2/3 * self.width * sfreq), int(1/3 * self.width * sfreq)
        window = np.zeros(w1 + w2)
        window[:w1+1] = np.hanning(w1 + 1) * 5
        window[w1-1:] = np.hanning(w2 + 1) * (-0.5) * 5

        # 3. Convolve
        data = np.convolve(train, window, mode="same") * self.scale
        return {"data": data, "events": train, "wavelet": window}

    def __str__(self):
        return super().__str__() + f", Width: {self.width}s, Repeat: {self.rep}s"


class OscillationSignal(Signal):
    """
    Oscillatory signal.
    """
    def __init__(
        self,
        frequency = 10,
        onset = None,
        offset = None,
        phase = 0,
        scale = 1e-8,
        taper = False,
        event_id = 2,
    ):
        super().__init__(name=f"Oscillation_{frequency}Hz", event_id=event_id)
        self.freq = frequency
        self.on = onset
        self.off = offset
        self.phase = phase
        self.scale = scale
        self.taper = taper

    def generate(self, times):
        # 1. base sine wave
        data = self.scale * np.sin(2*np.pi*self.freq*times + self.phase)

        # 2. create temporal mask
        onset = self.on if self.on is not None else times[0]
        offset = self.off if self.off is not None else times[-1]

        mask = (times >= onset) & (times <= offset)

        # 3. apply mask
        data[~mask] = 0

        # 4. Optional: Smooth the edges (Tapering)
        # Prevents "clicks" or spectral leakage at the start/end
        if self.taper:
            idx = np.where(mask)[0]
            if len(idx) > 0:
                window = np.hanning(len(idx))
                data[idx] *= window

        # 5. Define "events" as the start and end of the oscillation
        events = np.zeros_like(times)
        sfreq = 1 / (times[1] - times[0])
        events[int(onset * sfreq)] = 1
        events[int(offset * sfreq)] = -1 # -1 can signal the "offset" event

        return {"data": data, "events": events}

    def __str__(self):
        return super().__str__() + f", Freq: {self.freq}Hz, Taper: {self.taper}"


class ColoredNoise(Signal):
    """Colored Noise"""
    def __init__(self, color="pink", scale=1e-9, event_id=0):
        super().__init__(name=f"{color.capitalize()} Noise", event_id=event_id)
        self.color = color
        self.scale = scale

        # Map colors to your existing filtered functions
        self._generator = get_noise_generator(color)

    def generate(self, times):
        t = len(times)
        data = self._generator(t, n=1) * self.scale

        return {
            "data": data,
            "events": np.zeros_like(times)
        }
