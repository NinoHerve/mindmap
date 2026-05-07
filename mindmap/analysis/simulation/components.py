"""
DOC string
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import nibabel as nib
import mne

from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.source_space.io import get_dipole_positions
from mindmap.analysis.simulation.signals import Signal

class Source:
    """
    Class to handle source information, including source locations and waveforms.
    """
    def __init__(self, locations, signal: Signal):
        self.loc = np.array(locations) if not isinstance(locations, np.ndarray) else locations
        self.signal = signal

    def build_source_pattern(self, times, src, parc=None, tkras2ras=None):
        """
        Create a brain activity with activity only at source location.
        """
        # waveform
        waveform = self.signal.generate(times)

        # spatial selection logic
        targets = self.select_targets(src, parc, tkras2ras)

        pattern = np.zeros((targets.shape[0], times.shape[0]))
        pattern[targets, :] = waveform["data"]

        return {
            "pattern": pattern,
            "waveform": waveform,
            "targets": np.argwhere(targets).squeeze(),
        }

    # --- finish class with the other stuff ---
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
            assert tkras2ras is not None, "Need tkras2ras transform to align dipoles with T1-space."

            dipoles = nib.affines.apply_affine(tkras2ras, dipoles*1000)
            labels = parc.label_points(dipoles)
            targets = np.isin(labels, self.loc)

        return targets

    def __str__(self):
        summary = [
            "\033[94mSource Information:\033[0m",
            f"- Locations: {self.loc}",
            f"- {self.signal}"
        ]
        return "\n".join(summary)


class HeadModel:
    """
    Head Model class. Stores anatomical information for the simulation.
    """
    def __init__(
        self,
        sensors: mne.Evoked,
        forward: mne.Forward,
        connectivity_weights: np.ndarray,
        connectivity_lengths: np.ndarray,
        conduction_speed: float = 500,
        node_delay: float = 0.001,
        parcellation: Parcellation = None,
        tkras2ras: np.ndarray = None,
    ):
        """
        Parameters:
        ----------
        - sensors: mne.Evoked
            Sensor object.
        - forward: mne.Forward
            Forward solution.
        - connectivity_weights: np.ndarray
            Connectome.
        - connectivity_lengths: np.ndarray
            Connectome with distance in [mm] as weights.
        - conduction_speed: float
            Conduction speed in [mm/ms]
        - node_delay: float
            Nodal delay in [ms].
        - parcellation: Parcellation (optional)
            Parcellation required if using brain region name as source.
        - tkras2ras: np.ndarray
            FreeSurfer to subject transform to align dipoles in Parcellation's
            space.
        """
        self.sens = sensors
        self.fwd = forward
        self.src = forward['src']
        self.parc = parcellation
        self.tkras2ras = tkras2ras

        # hyper-parameters
        self.speed = conduction_speed
        self.node_delay = node_delay

        # Store everything as Sparse CSR for memory efficiency
        # We only need the lengths and weights for the simulation
        self._weights = csr_matrix(connectivity_weights)
        self._lengths = csr_matrix(connectivity_lengths)
        self._unweighted = csr_matrix(self._lengths > 0)

        # Pre-calculate the 'delay' graph
        # Logic: (length / speed) + node_delay (if length > 0)
        # Use .data to only operate on existing connections
        self._delays = self._lengths / self.speed
        self._delays.data += self.node_delay

    def compute_distance_from_source(self, source_idx, weight="delay"):
        """
        Calculates shortest paths from a source to ALL other nodes using Dijkstra.
        """
        # Select the correct sparse representation
        if weight == "delay":
            graph = self._delays
        elif weight == "length":
            graph = self._lengths
        elif weight == "hop":
            graph = self._unweighted
        else:
            raise ValueError(f"Unknown weight type {weight}")

        # dijkstra returns a (N,) array of distances from the source_idx
        distances = dijkstra(
            csgraph=graph,
            directed=False,
            indices=source_idx,
            unweighted=(weight == "hop")
        )

        return distances
    
    def apply_jitter(self, amount=0.0001):
        """Adds random jitter to the conduction delays."""
        self._delays.data += np.random.normal(0, amount, size=self._delays.nnz)


    def __str__(self):
        # Using join for cleaner multi-line strings
        summary = [
            "\033[94mHead model:\033[0m",
            f"- Number of dipoles: {self.fwd['nsource']}",
            f"- Conduction speed: {self.speed} [mm/s]",
            f"- Node delay: {self.node_delay} [s]",
            f"- Parcellation: {getattr(self.parc, 'n_regions', None)}",
            f"- tkras2ras: {self.tkras2ras is not None}"
        ]
        return "\n".join(summary)
