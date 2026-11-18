"""
DOC string
"""
from pathlib import Path
import json

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.csgraph import dijkstra

import nibabel as nib
import mne

from mindmap.anatomy.parcellation.base import Parcellation
from mindmap.anatomy.source_space.io import get_dipole_positions
from mindmap.analysis.simulation.signals import Signal

def _hops_along_paths(predecessors: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Number of edges along each shortest path, from the predecessor and
    distance arrays of scipy's `dijkstra` (shape (N,) or (n_sources, N)).

    Follows every node's predecessor chain back to its source, all nodes at
    once (one vectorized step per hop). Sources get 0, unreachable nodes
    (infinite distance) `np.inf`.
    """
    pred = np.atleast_2d(predecessors)
    no_pred = -9999                        # scipy's marker for "no predecessor"
    hops = np.zeros(pred.shape, dtype=float)
    rows = np.arange(pred.shape[0])[:, None]

    cur = pred.copy()
    active = cur != no_pred
    while active.any():
        hops[active] += 1
        cur = np.where(active, pred[rows, np.where(active, cur, 0)], no_pred)
        active = cur != no_pred

    hops[~np.isfinite(np.atleast_2d(distances))] = np.inf
    return hops


class Source:
    """
    Class to handle source information, including source locations and waveforms.
    """
    def __init__(self, locations, signal: Signal):
        self.loc = np.array(locations) if not isinstance(locations, np.ndarray) else locations
        self.signal = signal

    def save(self, path: str | Path = "source_config.json"):
        """Saves Source configurations."""
        loc_data = self.loc.tolist() if isinstance(self.loc, np.ndarray) else self.loc
        is_index = np.issubdtype(self.loc.dtype, np.integer)

        metadata = {
            "locations": loc_data,
            "is_index": bool(is_index),
            "signal": self.signal.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    
    @classmethod
    def load(cls, path: str | Path):
        """Reconstructs Source from saved state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find file '{path}'")
        
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Re-verify tracking type format (convert back to appropriate numpy arrays)
        dtype = int if meta["is_index"] else None
        locations = np.array(meta["locations"], dtype=dtype)

        # Rebuild the exact nested Signal child
        signal = Signal.from_dict(meta["signal"])

        return cls(locations=locations, signal=signal)

    def build_source_pattern(self, times, src, parc=None, ras2tkras=None):
        """
        Create a brain activity with activity only at source location.
        """
        # waveform
        waveform = self.signal.generate(times)

        # spatial selection logic
        targets = self.select_targets(src, parc, ras2tkras)

        pattern = np.zeros((targets.shape[0], times.shape[0]))
        pattern[targets, :] = waveform["data"]

        return {
            "pattern": pattern,
            "waveform": waveform,
            "targets": np.argwhere(targets).squeeze(),
        }

    # --- finish class with the other stuff ---
    def select_targets(self, source_space, parc=None, ras2tkras=None):
        """
        Retrieves dipole ids belonging to source.
        """
        dipoles = get_dipole_positions(source_space)

        if self.loc.dtype == int:
            targets = np.zeros(len(dipoles), dtype=bool)
            targets[self.loc] = True
        else:
            assert parc is not None, "Need parcellation to build stimulus because source locations are given as brain regions."
            assert ras2tkras is not None, "Need tkras2ras transform to align dipoles with T1-space."

            dipoles = nib.affines.apply_affine(np.linalg.inv(ras2tkras), dipoles*1000)
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
        parcellation: Parcellation = None,
        ras2tkras: np.ndarray = None,
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
        - parcellation: Parcellation (optional)
            Parcellation required if using brain region name as source.
        - ras2tkras: np.ndarray
            FreeSurfer to subject transform to align dipoles in Parcellation's
            space.
        """
        self.sens = sensors
        self.fwd = forward
        self.src = forward['src']
        self.parc = parcellation
        self.ras2tkras = ras2tkras

        # Store everything as Sparse CSR for memory efficiency
        # We only need the lengths and weights for the simulation
        self._weights = csr_matrix(connectivity_weights)
        self._lengths = csr_matrix(connectivity_lengths)
        self._unweighted = csr_matrix(self._lengths > 0)
    
    def save(self, folder: str | Path = "./headmodel", verbose="warning"):
        """Save C-bound MNE objects and SciPy matrices natively."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # 1. Save MNE objects using robust formats
        mne.write_evokeds(folder/"sensors-ave.fif", self.sens, overwrite=True, verbose=verbose)
        mne.write_forward_solution(folder/"forward-fwd.fif", self.fwd, overwrite=True, verbose=verbose)

        # 2. Save SciPy CSR matrices using their native efficient formats
        save_npz(folder/"conn_weights.npz", self._weights)
        save_npz(folder/"conn_lengths.npz", self._lengths)

        # 3. Save your custom parcellation using your custom save option
        if self.parc is not None:
            self.parc.save(folder/"parcellation.nii.gz", folder/"parcellation_lut.txt")

        # 4. Save arrays and structural parameters in a JSON file
        meta = {
            "has_parc": self.parc is not None,
            "ras2tkras": self.ras2tkras.tolist() if self.ras2tkras is not None else None
        }
        with open(folder/"headmodel_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    @classmethod
    def load(cls, folder: str | Path, verbose="warning"):
        """Reconstructs the HeadModel components securely back into execution space."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Could not find folder '{folder}'.")
        
        # 1. Load basic properties
        with open(folder/"headmodel_config.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        ras2tkras = np.array(meta["ras2tkras"]) if meta["ras2tkras"] is not None else None

        # 2. Load MNE structures safely
        sensors = mne.read_evokeds(folder/"sensors-ave.fif", verbose=verbose)[0]
        forward = mne.read_forward_solution(folder/"forward-fwd.fif", verbose=verbose)

        # 3. Load Sparse arrays
        connectivity_weights = load_npz(folder/"conn_weights.npz")
        connectivity_lengths = load_npz(folder/"conn_lengths.npz")

        # 4. Load Parcellation
        parcellation = None
        if meta["has_parc"]:
            parcellation = Parcellation(folder/"parcellation.nii.gz", folder/"parcellation_lut.txt")

        return cls(
            sensors = sensors,
            forward = forward,
            connectivity_weights = connectivity_weights,
            connectivity_lengths = connectivity_lengths,
            parcellation = parcellation,
            ras2tkras = ras2tkras,
        )
    

    def compute_distance_from_source(self, source_idx, weight="hop"):
        """
        Calculates shortest paths from a source to ALL other nodes using Dijkstra.

        Parameters
        ----------
        source_idx: int
            Source index.

        weight: str ['length', 'hop']
            Type of weight to consider along the path.

        Returns
        -------
        distances: np.ndarray
            Shortest path between source_idx and all other nodes.
        """
        # Select the correct sparse representation
        if weight == "length":
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

        return distances.squeeze()
    
    def compute_delay_from_source(
        self,
        source_idx,
        conduction_speed: float = 500,
        nodal_delay: float = 0.1,
        jitter: float = 0.0,
        u_fiber_speed: float | None = None,
        u_fiber_max_length: float | None = None,
        return_hops: bool = False,
    ):
        """
        Compute delay with uncertainties.

        Parameter
        ---------
        source_idx: int
            Source index.

        - conduction_speed: float
            Conduction speed in [mm/ms], used for edges longer than
            `u_fiber_max_length` (or all edges if the U-fiber split is disabled).

        - node_delay: float
            Nodal delay in [ms].

        jitter: float
            Per-hop delay uncertainty (standard deviation, in [ms]). Applied
            once per graph edge on the shortest path and accumulated along
            it -- like `nodal_delay` -- rather than as a single perturbation
            of the total path delay. Independent per-hop noise sums in
            quadrature, so the realized total-delay std scales as
            `jitter * sqrt(hops)`, not `jitter` alone: a target reached in
            many hops picks up proportionally more timing uncertainty than
            one reached directly, matching how compounding relay-to-relay
            jitter would actually behave physically.

        - u_fiber_speed: float or None
            Conduction speed in [mm/ms] for short, U-fiber-like edges. If None
            (default), the U-fiber split is disabled and `conduction_speed` is
            applied uniformly to every edge.

        - u_fiber_max_length: float or None
            Edge length in [mm] at or below which `u_fiber_speed` is used
            instead of `conduction_speed`. Ignored if `u_fiber_speed` is None.

        - return_hops: bool
            If True, also return the number of edges along each node's
            fastest (smallest-delay) path, i.e. the path the delay is summed
            over. Unreachable nodes get `np.inf`.

        Returns
        -------
        delays: np.ndarray
            Delays from source with added uncertainties.

        hops: np.ndarray
            Only if `return_hops`: number of edges along each fastest path.
        """
        # edge delay matrix
        delays = self._lengths.copy()
        if u_fiber_speed is not None and u_fiber_max_length is not None:
            speeds = np.where(delays.data <= u_fiber_max_length, u_fiber_speed, conduction_speed)
            delays.data = delays.data / speeds
        else:
            delays.data = delays.data / conduction_speed
        delays.data += nodal_delay

        # dijkstra returns a (N,) array of distances from the source_idx, and
        # the predecessor of each node on its fastest path
        delays_from_source, predecessors = dijkstra(
            csgraph = delays,
            directed = False,
            indices = source_idx,
            unweighted = False,
            return_predecessors = True,
        )
        # number of edges along each fastest path -- the same path the delay
        # (and its nodal_delay terms) is summed over, NOT the fewest-hop path,
        # which can be shorter in edges but slower
        hops = _hops_along_paths(predecessors, delays_from_source).squeeze()
        delays_from_source = delays_from_source.squeeze()

        if jitter > 0:
            # Path SELECTION stays deterministic (based on the mean edge
            # weights above) -- only the total delay along that fixed,
            # anatomically-real path becomes noisy: one independent draw per
            # (source, node) pair, with variance hops * jitter**2, as if the
            # path's per-hop jitters were independent and summed.
            delays_from_source += np.random.normal(0, jitter, size=delays_from_source.size) * np.sqrt(hops)

        if return_hops:
            return delays_from_source, hops
        return delays_from_source

    def __str__(self):
        # Using join for cleaner multi-line strings
        summary = [
            "\033[94mHead model:\033[0m",
            f"- Number of dipoles: {self.fwd['nsource']}",
            f"- Parcellation: {getattr(self.parc, 'n_regions', None)}",
            f"- tkras2ras: {self.ras2tkras is not None}"
        ]
        return "\n".join(summary)
