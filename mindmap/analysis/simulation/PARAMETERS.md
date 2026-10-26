# Simulation parameter reference

Documents every free/configurable parameter in `mindmap.analysis.simulation`, the value
currently used in `tutorials/analysis/simulations/simulation_spike.ipynb`, the mechanism it
controls in the code, and the literature (or lack thereof) behind that value. Written to keep
a record of *why* each number was chosen, since several went through multiple rounds of
correction during development.

Two source documents feed this file: the development conversation that built/tuned this
module, and `/home/nino/Manuscripts/wCSET/parameter_estimates_review.md` (working citation-
review notes for the wCSET manuscript, which shares several of these parameters). Where the
two disagree or only partially overlap, both are noted explicitly rather than silently
resolved — see **Open items** at the end.

---

## 1. Propagation parameters (`PropagationConfig`, `engine.py`)

Control how a source's waveform spreads across the structural connectome: `HeadModel.
compute_delay_from_source` runs Dijkstra over the subject's connectome graph (edge weight =
length / speed + nodal_delay) to get each dipole's arrival time, and separately over the same
graph with **unweighted edges** (`weight="hop"`) to get each dipole's hop-count from the
source; `Simulator.compute_propagation` then shifts the waveform by the arrival time and
scales it by `spatial_decay ** hops`.

### `conduction_speed` — 4.5 mm/ms (= m/s)

Long-range (non-U-fiber) axonal conduction speed, used to convert a graph edge's physical
length into a delay.

- **Literature basis**: Two independent large-sample cortico-cortical evoked potential (CCEP)
  studies converge on ~3-6 m/s for adult long-range association fibers:
  - Lemaréchal, Jedynak, Trebaul et al. (2022), *Brain*, doi:10.1093/brain/awab362 — median
    axonal conduction delay 10.2 ms, median velocity **3.9 m/s**, fitted across the whole
    cortex from 774,445 CCEPs recorded in 780 epilepsy patients.
  - *"Developmental trajectory of transmission speed in the human brain,"* Nat. Neurosci.
    26(4), 537-541 (2023), doi:10.1038/s41593-023-01272-0 — adult long-range association
    fibers **~3-6 m/s**, typical adult latencies 20-30ms. (Author-attribution discrepancy —
    see Open items.)
- **Why not a diffusion-MRI/histology-derived value** (e.g. the classic 30 m/s figures seen in
  some tractography-based work): those target the fastest, largest-diameter axon
  subpopulation within one hand-picked tract. Since this simulator propagates activity across
  the *whole* connectome rather than one selected tract, the CCEP population-average estimate
  is the more representative anchor.
- **4.5 m/s** sits centrally within the ~3-6 m/s band both papers report.
- **Previously mis-cited value** (context, not used here): an earlier version of the wCSET
  manuscript used 30 mm/ms citing Caminiti et al. (2013, *J Neurosci*, "Diameter, Length,
  Speed, and Conduction Delay of Callosal Axons in Macaque Monkeys and Humans") — that paper
  reports 4.9-11.4 m/s (fastest measured pathway ~17-20 m/s) for **callosal** axons
  specifically, not general cortico-cortical fibers, and doesn't support 30 mm/ms at all.

### `u_fiber_speed` — 2.5 mm/ms, `u_fiber_max_length` — 35 mm

Splits short association/U-fibers from long-range tracts: edges at or below `u_fiber_max_length`
use `u_fiber_speed` instead of `conduction_speed`.

- **Literature basis**: same 2023 Nat. Neurosci. paper above reports short-range U-fibers at
  **~2-3 m/s**, distinctly slower than long-range association fibers. `u_fiber_max_length=35mm`
  uses the commonly-cited 30-40mm streamline-length cutoff for short association/U-fibers.
- `2.5 m/s` sits centrally in the ~2-3 m/s band.

### `nodal_delay` — 7 ms

Added once per graph edge on the shortest path (not per traversed node) — i.e. it accumulates
once for every relay a signal passes through between source and target, representing the time
a cortical population takes to receive an input, integrate it, and re-fire toward the next
node. This is a **broader quantity than pure synaptic transmission delay** — it also swallows
local dendritic integration time — a distinction that matters for picking the right citation.

- **Literature basis**: Lemaréchal et al. (2022, *Brain*, as above) fit dynamic-causal/neural-
  mass models to real CCEPs and report population-level *synaptic time constants* directly:
  median **5.8 ms** (excitatory), **7.3 ms** (inhibitory), adults >15y, with regional variation
  (sensorimotor/occipital/temporal shorter; frontal/insular/parietal/cingulate longer).
- **7ms** sits almost exactly at the inhibitory median, within the 5.8-7.3ms combined range.
  Kept at 7ms (considered, then explicitly rejected, raising it to 15ms — see below).
- **Considered and rejected: borrowing canonical neural-mass-model (e.g. Jansen-Rit) time
  constants (~10-20ms) instead.** These describe a mechanistically different quantity and
  don't transfer to this parameter. In Jansen-Rit-style models, the "time constant" (1/a≈10ms
  excitatory, 1/b≈20ms inhibitory) parameterizes an **alpha-function convolution kernel**,
  `h(t) = A·a·t·e^{-at}`, applied to incoming input to produce the postsynaptic response's
  *shape*: the response starts immediately and the time constant governs how it rises to peak
  and decays over time — it's a filter, not a delay. `nodal_delay` in this codebase does the
  opposite: `compute_propagation` shifts the waveform by a fixed number of samples and copies
  it unchanged (`shifted_sig[dt:] = waveform[:copy_len]`) — a pure additive translation, no
  reshaping. Using the Jansen-Rit figures to size a pure delay conflates "how long the
  postsynaptic response takes to rise and decay" with "how long before the next node reacts at
  all." Lemaréchal et al.'s figures remain the better-matched anchor despite also being
  convolution-kernel-based internally, because their modeling exercise explicitly decomposes
  each *observed* CCEP peak latency into an axonal (conduction) term plus a synaptic/intrinsic
  term — the same decomposition `edge delay = length/speed + nodal_delay` performs here — while
  Jansen-Rit-style constants are normally fit to reproduce local oscillatory dynamics (e.g.
  alpha rhythm generation), not inter-regional relay latency.
- **Previously mis-cited value** (context, not used here): an earlier wCSET manuscript version
  used 10ms citing Sabatini & Regehr (1996, *Nature*, "Timing of neurotransmission at fast
  synapses in the mammalian brain") — that paper measures **presynaptic vesicle release**
  specifically (<1ms, possibly <100μs), ~100x smaller than the value it was cited for, and
  doesn't measure population relay time at all.

### `spatial_decay` — 0.65 (per-hop multiplicative attenuation, `decay ** hops`)

Amplitude attenuation applied per graph hop (unweighted edge count on the shortest path,
independent of that edge's physical length).

- **No literature value exists** for a per-hop amplitude decay factor of this specific kind.
  Comparable published models (Raj et al. 2012 network-diffusion model of disease progression;
  epidemic/SIR-style seizure-propagation models) all treat their analogous decay/diffusivity
  parameter as a free, data-fitted quantity rather than a literature-derived constant. Kept as
  a tuned/free parameter here too, consistent with that precedent.
- **Empirical tuning rationale (this session)**: because `hops` is unweighted while conduction
  delay is weighted by real physical length, a distant hub reached via a single long-range
  tract edge (this connectome has edges up to 148mm; ~66% of dipoles have ≥1 edge >30mm) costs
  only `decay**1` — the same attenuation as crossing one ~2mm mesh edge — while spatially
  intervening cortex reachable only through many short mesh hops is decayed far more heavily.
  This produces a later-arriving (real conduction delay), still-strong, spatially **separate**
  secondary focus, confirmed by clustering the activated dipoles' actual 3D coordinates
  (Euclidean proximity), not graph-topological adjacency (which wrongly treats a hub linked by
  a direct tract edge as "connected" to the source even when 10cm away with nothing active in
  between).
- Swept `spatial_decay ∈ {0.2, 0.35, 0.5, 0.65, 0.8}` across 5 different source regions.
  Low values (≤0.35) barely spread past the immediate source before dissipating; high values
  (0.8) saturate a very large fraction of the cortex for an extended plateau. **0.65** was
  chosen as the best match to "clear focal onset → broad multi-region spread (including
  spatially disjoint, often bilateral secondary foci from long-range tracts) → full
  dissipation": it plateaus for ~30ms rather than instantly saturating (0.8) or barely
  spreading (≤0.35).

### `jitter` — 1 ms (per-hop delay std, `PropagationConfig.jitter`)

Gaussian noise on delay, applied **once per graph edge on the shortest path and accumulated
along it** — the same per-hop accounting `nodal_delay` uses — rather than as a single
perturbation of the total path delay.

- **Bug fixed this session**: the original implementation drew a *single* `N(0, jitter)`
  perturbation and added it to the already-summed total path delay
  (`delays_from_source += np.random.normal(0, jitter, size=...)`), so a dipole 20 hops away
  received the same jitter magnitude as one 1 hop away — physically implausible if jitter is
  meant to represent per-relay timing noise accumulating over a path. Fixed in
  `HeadModel.compute_delay_from_source` (`components.py`) to scale by the shortest path's hop
  count instead: `jitter * sqrt(hops)`. This is the correct scaling (not `jitter * hops`,
  linear) because independent per-hop noise sums in *variance*, not in std — `hops` independent
  `N(0, jitter²)` draws sum to `N(0, hops·jitter²)`, i.e. std `jitter·sqrt(hops)`. Verified
  empirically: measured jitter std at hops={1,2,5,7} matched the predicted `sqrt(hops)` scaling
  closely (e.g. hops=5: observed 2.21ms vs. predicted 2.24ms, jitter=1ms).
- **Path selection stays deterministic.** Only the total delay along the already-selected
  (mean-weight) shortest path becomes noisy — signal is still assumed to travel the one
  anatomically-real, fixed route Dijkstra finds; only relay-to-relay timing on that route is
  stochastic. Re-running Dijkstra on jitter-perturbed edge weights (letting jitter also change
  *which* path is shortest) was considered and rejected: anatomically, an axon's route doesn't
  change from one trial to the next due to synaptic timing noise, so perturbing path selection
  itself would model something unphysical.
- **No literature value used or sought** for the 1ms magnitude itself. A small, arbitrary
  smoothing term to avoid perfectly deterministic, sample-locked arrival times across repeated
  simulations; not intended to represent a specific physiological source of timing variability.
  Free parameter, same category as `spatial_decay`. Note its *meaning* changed with the fix
  above (per-hop std, not flat total-path std), so a target 20 hops from the source now
  realizes ~`sqrt(20)≈4.5`x more delay spread than before at the same `jitter=1ms` setting —
  worth re-checking if any downstream analysis depended on the old (flat) behavior.

---

## 2. Source signal parameters (`SpikeSignal`, `signals.py`)

### `width` — 0.05 s (50 ms)

Total duration of the double-Hanning-window waveform (`2/3 * width` main lobe + `1/3 * width`
after-going slow wave).

- **Literature basis**: the classic clinical convention distinguishes interictal **spikes**
  (20-70ms duration) from **sharp waves** (70-200ms) — e.g. Chatrian-style terminology still
  in common clinical use. 50ms sits centrally in the spike band.
- **Discrepancy with `parameter_estimates_review.md`**: that document defends the wCSND
  manuscript's existing **W=0.1s (100ms)** value, citing `kane_revised_2017` (IFCN revised
  terminology/glossary paper) as reporting a mean duration of **≈72ms, range 15-200ms** for
  scalp EEG spikes — a broader empirical distribution that does not exclude 100ms, and under
  which 100ms is close to the reported mean rather than an outlier. **This is not yet
  reconciled** with the tighter 20-70ms "spike-only" convention used to justify narrowing to
  50ms in this codebase's tutorial — see Open items.
- `SpikeSignal.generate` renormalizes the discrete peak of the double-Hanning window so `scale`
  remains a reliable amplitude control regardless of `width`/sampling-rate combination (a
  correction added this session — the raw discretized window's peak drifted up to ~30% off
  target before this fix, non-monotonically in `width`).

### `onset`, `repeat`, `scale`, `event_id`

Experiment-design/bookkeeping parameters, not physiologically anchored:

- `onset` (0.5s in the tutorial): time of the first spike within the periodic frame.
- `repeat` (1s): inter-spike interval in the underlying continuous raw recording, i.e. how many
  trials get generated for later epoching/averaging — a data-volume choice, not a literature
  quantity.
- `scale` (1e-8): **has no effect on the final delivered amplitude.** `Simulator.run`
  renormalizes total source activity to a canonical 1e-8 Am peak immediately after generation
  (`engine.py:397-399`), and the SNR-calibration step (see below) rescales again to hit
  `snr_target`. Any nonzero `scale` value produces an identical delivered recording.
- `event_id`: MNE annotation/event bookkeeping identifier, not physiological.

### `OscillationSignal` (`frequency`, `phase`, `taper`) — not used in the current tutorial

Present in the codebase as an alternative signal type but not exercised in this session's work;
no literature-grounded parameter values were established for it. Documented here for
completeness only.

---

## 3. Noise parameters (`NoiseConfig`, `engine.py`)

Two independent noise sources are generated — `dipole_noise` (source-space, forward-projected)
and `sensor_noise` (sensor-space, added directly) — and combine **in quadrature** at the
delivered channel (`total² = dipole_contribution² + sensor_noise_scale²`), since they're
statistically independent.

### `sensor_noise_scale_uv` — 2.0 µV RMS (fixed)

Represents instrument/electrode-noise floor: a property of recording hardware, independent of
the forward model, so it's kept fixed across head models/sources rather than solved for.

- **No specific paper pinned down this session** — 2µV is an order-of-magnitude engineering
  assumption (research-grade amplifier noise floors are typically sub-µV to a few µV), not a
  literature-derived value. Flagged as a placeholder in the code's own docstring.

### `target_sensor_noise_uv` — 15.0 µV RMS (total, solved-for)

Total desired sensor-level baseline noise (fixed `sensor_noise_scale_uv` contribution plus
`dipole_noise`'s forward-projected contribution, combined in quadrature). `Simulator.run`
**solves** for the `dipole_noise` Am-scale needed to make up the remainder for this specific
head model's forward gain, rather than treating `dipole_noise_scale` as a free, independently
set field (its sensor-level contribution depends on source depth/orientation/head geometry,
which varies per subject/source).

- **Literature basis**: 15µV is the midpoint of the commonly-cited scalp EEG background noise
  range, ~10-20µV RMS. This is a general clinical-EEG convention gathered during this session's
  literature pass rather than a single pinned primary source — **no specific paper identified**
  for this exact figure (unlike the propagation parameters above, which trace to specific CCEP
  studies). Treat as a textbook-level convention, not a hard citation.
- **Validation performed this session**: with the historical hardcoded scale (1e-7 V sensor
  noise, no quadrature solve), a `snr_target=5` run delivered only ~2µV peaks — 1-2 orders of
  magnitude below real interictal spikes. After fixing the noise model, the same target
  delivers ~85-90µV peak-to-peak and ~15µV baseline RMS, both within the literature bands
  quoted below for `snr_target`.

### `sensor_noise_color` (white), `dipole_noise_color` (pink)

- `sensor_noise_color="white"`: standard assumption for instrument/thermal noise.
- `dipole_noise_color="pink"`: standard assumption for ongoing background cortical activity
  (1/f-type spectrum), consistent with the broadly-observed 1/f scaling of resting EEG/MEG
  spectra. Neither has a specific citation attached this session — both are standard modeling
  conventions rather than fitted/measured values.

---

## 4. SNR calibration target (`snr_target` argument to `Simulator.run`, + `SNRConfig`)

### `snr_target` — 5 (used in the tutorial's `sim.run(..., snr_target=5, ...)`)

Single-trial peak-amplitude SNR the calibration loop iterates the source scale to hit (same-
noise-draw fixed-point iteration, not a fresh Monte Carlo draw each step).

- **Literature basis**: Gonsisko et al. (2024), *Epilepsia*, doi:10.1111/epi.18141,
  "Electroencephalographic source imaging of spikes with concurrent high-frequency
  oscillations is concordant with the clinical ground truth" — reports individual (single-
  trial) interictal spikes at **~3-5 dB** SNR, converting (SNR(dB) = 10·log₁₀(ratio)) to a
  linear ratio of **~2.0-3.2**. Averaged spikes reach >10 dB (ratio >10), corroborated by
  Sohrabpour et al.'s IRES paper (PMC5124544), which uses 10-20 dB (ratio 10-100) as
  representative of source-imaging studies that calibrate against the averaged evoked response.
- **Why the used value (5) is above the literature single-trial band (~2-3), not equal to
  it**: an earlier attempt at `snr_target=2.5` was tested directly against this simulator's
  noise floor and found to sit at or below the level noise can randomly generate on its own —
  i.e., too close to the floor to reliably read out as "a spike" in a single trial, an
  operational problem the literature figure alone doesn't address. `snr_target=5` was chosen
  as a practical compromise: still much closer to the single-trial literature range (~2-3) than
  the old, averaged-response-scale default of 10, while remaining comfortably above the
  simulator's own noise floor. This is a literature-**informed**, not literature-**derived**,
  choice — flagged explicitly since it doesn't fall inside the cited 2.0-3.2 band.
- `SNRConfig(method="peak", spike_window=(-0.03, 0.03))`: `method="peak"` (max|amplitude| /
  baseline std) was chosen as the default estimator after finding that whitened/GFP-based
  methods introduced calibration mismatches when the whitening noise covariance didn't exactly
  match the delivered noise; not literature-driven, an internal-consistency choice.
  `channel_selection="auto"` (restricting calibration to one pre-selected channel by forward
  gain) removes a channel-count-dependent "winner's curse" bias that made low targets like 2.5
  unreachable on a many-channel montage, regardless of window width — an artifact of the
  measurement procedure, not a physiological consideration.

---

## 5. Epoch/window parameters (`EpochConfig`, `engine.py`)

`tmin=-0.45s`, `tmax=0.45s`, `baseline=(-0.45,-0.2)s`, `signal_window=(-0.1,0.45)s` — purely
methodological choices (epoch length, baseline window used both for MNE baseline-correction and
for the noise-floor calibration's `_baseline_window_std` measurement, and the post-onset window
where signal is expected). **No literature citation applies** to these; they were sized to
comfortably contain the observed propagation timescales (spread patterns extending out to
~130ms post-onset at `spatial_decay=0.8`, the widest-spreading setting tested) with margin.

---

## 6. Structural connectome inputs (`HeadModel`, `components.py`)

`connectivity_weights`/`connectivity_lengths` are **not free simulation parameters** — they're
the subject's own dMRI-derived structural connectome (dipole-to-dipole graph with real
streamline lengths in mm), loaded from disk. They're listed here because they directly
determine how `nodal_delay`/`conduction_speed`/`spatial_decay` behave: this session found the
graph has edges up to 148mm (~66% of dipoles carry ≥1 edge >30mm), which is what makes the
long-range-tract "leapfrogging" behavior described under `spatial_decay` possible in the first
place.

---

## Open items (flagged, not resolved)

1. **Author-list discrepancy** for the conduction-speed/U-fiber paper: this session's earlier
   work cited it as "Fung et al. 2023," while `parameter_estimates_review.md` cites the same
   title/journal/volume/DOI (`Nat. Neurosci. 26(4), 537-541, doi:10.1038/s41593-023-01272-0`)
   as "van Blooijs, D., van den Boom, M.A., van der Aar, J.F., et al. 2023." Both can't be
   right for the same paper — worth checking the actual reference before this citation goes
   into any manuscript or is relied on further.
2. **Spike-width convention mismatch**: this codebase's tutorial narrows `width` to 50ms using
   the classic 20-70ms "spike" vs. 70-200ms "sharp wave" duration convention;
   `parameter_estimates_review.md` defends the wCSET manuscript's 100ms value using the same
   `kane_revised_2017` reference but reporting a broader empirical distribution (mean ≈72ms,
   range 15-200ms) under which 100ms isn't an outlier. These two conventions haven't been
   reconciled — the codebase and the manuscript may now disagree on the "correct" width for
   different reasons.
3. **`snr_target=5`** sits above the literature-derived single-trial band (~2.0-3.2) for
   practical/operational reasons (noise-floor visibility), not because the literature was
   re-interpreted — worth revisiting if a stricter single-trial-realistic target is needed.
4. **`sensor_noise_scale_uv`, `target_sensor_noise_uv`'s exact 10-20µV band, `dipole_noise`'s
   pink-noise assumption, and `sensor_noise`'s white-noise assumption** have no single pinned
   primary-source citation from this session — general clinical-EEG/modeling conventions only.
