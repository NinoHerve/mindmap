"""
DOC String
"""

from .signals import Signal, SpikeSignal, OscillationSignal
from .components import Source, HeadModel
from .engine import Simulator, SimulationResult, EpochConfig, PropagationConfig, SNRConfig, NoiseConfig
from .pipeline import generate_simulation_dataset, SimulationTask

__all__ = [
    "SpikeSignal",
    "OscillationSignal",
    "Source",
    "HeadModel",
    "Simulator",
    "SimulationResult",
    "EpochConfig",
    "PropagationConfig",
    "SNRConfig",
    "NoiseConfig",
    "generate_simulation_dataset",
    "SimulationTask",
]
