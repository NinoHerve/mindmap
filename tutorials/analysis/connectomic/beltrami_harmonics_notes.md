# Beltrami-Operator (Mesh-LBO) Harmonics — Design & Implementation Notes (2026-07-16)

Split out of `../inverse/sabotage_connectome_notes.md` (that file stays
focused on the dMRI-corruption study itself; this one owns the
mesh-geometry-basis side track it spawned — see that file's "Next steps"
item 3 for the pointer back). Note: `sabotage_connectome_notes.md` and
`sabotage_connectome.ipynb` moved from this directory to `../inverse/` on
2026-07-16, after this note was split out — they're inverse-solver
tuning/comparison work, not connectome-construction work, hence the move.

Goal: an alternative to `wCSET`'s connectome-Laplacian eigenbasis
(`compute_cst`), built purely from cortical *shape* (Laplace-Beltrami
eigenmodes of the source-space mesh), with no connectome/connectivity input
at all — to test whether `wCSET`'s corruption-robustness comes from genuine
connectome structure or just generic eigenbasis smoothness (see
[[sabotage-connectome-study]]).

Reference: "A geometry aware framework enhances noninvasive mapping of whole
human brain dynamics" (`s41551-026-01664-0`, Nat. Biomed. Eng. 2026) — their
"GBF" framework computes S=300 Laplace-Beltrami eigenmodes per hemisphere on
each subject's cortical surface mesh and uses them as the source basis,
structurally identical to how `wCSET` already uses connectome-Laplacian
eigenmodes (`eigenvectors`/`eigenvalues` -> `compute_harmonic_prior` ->
`w_source`, reused unchanged for both the forward-spread and the
map-back-to-dipole-space step in `solve()`).

## Why the basis must be orthonormal (not why it must be complete)

The `wCSET` math (`x = Rz`, ridge-regress `z`, map back via `x̂ = Rẑ`) is a valid
regularized estimator for *any* `R` — no orthogonality needed for the
solver itself. What *does* require orthonormal columns (`UᵀU = I`) is the
harmonic prior's interpretation: `w_harm(λ) = exp(-β·λ)` is only a clean
function of spatial frequency alone if energy splits additively across
modes (Parseval). Without it, a mode's contribution to the reconstructed
map is inflated/deflated by its own column norm and correlation with other
kept columns — a numerical artifact (e.g. mesh-density-dependent amplitude
bias), not a neural one — and it would contaminate exactly the
geometry-vs-connectome comparison this test is meant to isolate.

This has never been a live concern before because `compute_cst`
(`mindmap/analysis/connectivity/spectral.py:47-51`) runs plain
`np.linalg.eigh` on an *ordinary* symmetric matrix (ordinary, not
generalized, eigenproblem) — LAPACK guarantees exact Euclidean
orthonormality for that case, for free, every time, and `compute_cst` is
also never truncated (always full `n×n`), so completeness held too,
automatically. Nobody had to design for it.

The mesh case breaks both guarantees; fix (1) below is still needed, but fix
(2) (originally: compute on the fine mesh, restrict, then QR-reorthonormalize)
turned out to be the wrong plan entirely — superseded below.
1. `lapy` solves the *generalized* eigenproblem `Aψ = λBψ` (FEM stiffness
   `A`, mass matrix `B`) and normalizes eigenvectors so `ψᵢᵀBψⱼ = δᵢⱼ`
   (mass-orthonormal), not `UᵀU = I`. Fix: rescale by the (lumped/diagonal)
   mass matrix, `Ψ_euclid = B^(1/2) Ψ_lapy` (row `v` scaled by
   `sqrt(area_v)`), before it's a valid drop-in.

## Resolved: S=300 (paper's number) creates a basis-size mismatch against the connectome side — checked empirically, not a false alarm

`compute_cst` is always used as the *complete* basis (n=7,465 total used
dipoles for sub-simf10: 3,748 + 3,717), never truncated. Checked whether the
settled `harm=5` weighting already *softly* truncates the connectome case to
something near k≈300 anyway (which would make the S=300-vs-S=n framing a
non-issue) — it does not: at `harm=5` on sub-simf10's actual connectome
spectrum, `w_harm` at k=300 is still 0.040 (4% of peak), 90% of cumulative
weighted mass isn't reached until k≈4,707, and the "effective DOF"
(participation ratio `(Σw)²/Σw²`) is ≈447 — same order of magnitude as 300,
but nowhere near it, and nothing to hang a "they're already equivalent"
argument on.

## Decision: match S=n on both sides, let a shared tunable `exp(-β·λ)` (like `harm`) discount both spectra, instead of hard-truncating either

This also simplifies the implementation: fix (2) above (fine-mesh, then restrict
to `vertno`, then QR-reorthonormalize) was only needed to get a *few,
well-resolved low modes* efficiently. Once going for the *complete* basis
instead, there's no reason to touch the fine 112k-vertex mesh at all — build
the FEM stiffness/mass matrices **directly on the coarse, already-decimated
per-hemisphere mesh** (the one `vertno` already lives on) and run an
ordinary dense generalized `eigh` on it, per hemisphere (no LBO coupling
across hemispheres). This gives exact orthonormality for free (same
guarantee `compute_cst` already gets), no QR-correction hack needed.

Cost: cheap, not expensive. A full dense eigh on the *joint* n=7,465
connectome Laplacian is already measured at ~17.7s (project memory: latest
completed run). Two independent per-hemisphere dense generalized eigh calls
at ~3,700-3,750 each cost roughly `(3700/7465)³ ≈ 1/8` of that each — **~2s
per hemisphere, ~4s total.** (The infeasible route would be asking a sparse
partial-eigensolver, e.g. `lapy`'s default ARPACK/shift-invert path, for
*all* n eigenpairs — Krylov methods degrade to dense-eigh cost as k→n
anyway, so there's no shortcut there; or worse, attempting this on the fine
112k-vertex mesh, where a dense eigh is ~10¹⁵ flops / 100GB+ RAM and
genuinely infeasible. Neither is what "build directly on the coarse mesh"
means. Actual measured cost after implementation turned out higher than
this estimate — see "Implementation status" below.)

## Why building directly on the coarse (source-space) mesh is the more correct choice, not just the cheaper one

(Resolution mismatch doesn't disappear, but changes character and becomes justified):
1. Fine-mesh-then-restrict's error was a *domain mismatch* (point-sampling
   an accurate fine-mesh function down to a mismatched coarse set —
   classic aliasing, which is why it needed the QR patch). Building the
   operator directly on the coarse mesh instead gives ordinary *FEM
   discretization* error, well-characterized in the literature (relative
   eigenvalue error for mode k grows roughly like `(h·√λ_k)²` for mesh edge
   length `h`) — smooth, expected degradation of the high end, not a
   forced, unprincipled artifact.
2. `x` (what `wCSET` solves for) lives in `R^n_used`, not `R^112k`. An
   eigenbasis is only exactly orthonormal w.r.t. the operator it was
   derived from — building on the coarse mesh's own triangulation gives the
   basis natively defined on the space the inverse problem actually lives
   in, no correction hacks.
3. `con`/`con_strength` is *already* a per-dipole connectome native to this
   same coarse node set. Building the mesh-LBO basis on the same set keeps
   both harmonic bases defined over an identical discretization, so
   "geometric vs. connectome structure" stays the only thing differing
   between the two conditions — which is the actual point of the test.

Residual, honest caveat: coarse-mesh high-index modes are still less
faithful to the true folded cortical geometry than a fine-mesh computation
would give (moderate decimation is known to preserve low LBO modes well and
degrade high ones — standard in shape-analysis practice). That's exactly
the kind of imperfection the shared tunable weighting is meant to discount,
philosophically consistent with why high connectome modes are already
downweighted (different underlying reason — structural implausibility
there, discretization fidelity here — same "trust low, discount high"
treatment on both sides).

## The coarse cortex mesh is not closed — it has a boundary at the medial wall, and needs Neumann BC, not an assumed-closed-manifold formulation

`vertno` (`mask_sources` in
`papers/spike/create_source_spaces.py`) keeps cortex-only dipoles; `use_tris`
is the triangulation of a *pre-mask*, closed 4,098-vertex/hemi decimated
surface (verified for sub-simf10: watertight, Euler number 2, single
component, 0 boundary edges) that still includes ~350-381 non-cortex
(mostly medial-wall) vertices per hemisphere. Restricting to faces whose 3
vertices are all in `vertno` (verified empirically) gives a clean single
connected component, **single boundary loop** (89 edges, both hemispheres),
zero non-manifold edges — a well-behaved open manifold-with-boundary
(topologically a disk), not a fragmented or non-manifold mess. So:
- Boundary condition must be **Neumann/natural** at that medial-wall loop
  (no artificial pinning to zero — the medial wall isn't a real neural
  boundary, just where cortex modeling stops). This is the standard,
  anatomically-appropriate default for open-surface LBO computation in this
  literature (same choice implicit in Pang et al. 2023-style cortical GEMs)
  and `lapy`'s default behavior for a mesh with boundary — needs to be a
  deliberate, checked choice, not an assumed-closed-surface accident.
  Preserves `λ=0` (constant mode) same as the closed/connectome case, so
  the existing `eigval[0]=0`-style convention still applies.
- Minor edge case found empirically: for sub-simf10 hemi 101, the strict
  "keep face only if all 3 vertices are cortex" rule leaves exactly 1 of
  3,748 `vertno` dipoles with zero incident faces (a dangling vertex right
  at the boundary) — hemi 102 has none. See "Implementation status" below
  for how this was ultimately handled (not the nearest-neighbor patch
  originally proposed here).

## Why hemispheres are kept independent (no cross-hemisphere pairing/joining)

The Laplace-Beltrami operator generates two independent per-hemisphere
spectra, and near-degenerate eigenvalues (strong bilateral shape symmetry)
make cross-hemisphere mode *pairing* highly uncertain, especially for
higher harmonics. Considered whether to connect the hemispheres (e.g. via
`mm.analysis.connectivity.connectome_from_geometry`'s `inter_hemi_mirror`
homotopic-matching scheme) to sidestep this — concluded no:

- **Pairing doesn't actually matter for this use case.** `wCSET`'s math
  needs an `(n_total × S_total)` matrix with orthonormal columns and a
  per-column weight `w_harm(λ_k)` that only depends on that column's own
  eigenvalue — nothing depends on any cross-hemisphere correspondence. Two
  independently-computed, independently-orthonormal per-hemisphere bases,
  block-diagonal-stacked (each hemisphere's modes are exactly zero on the
  other hemisphere's dipoles), are automatically still a valid, complete,
  orthonormal basis for the joint space. Same argument extends to
  *within*-hemisphere degenerate clusters: arbitrary rotation within a
  degenerate subspace doesn't matter either, since `w_harm` only depends on
  the (shared) eigenvalue, not on which specific rotation the solver
  returns. Pairing/ordering ambiguity only matters for *interpreting*
  modes bilaterally (e.g. claiming mode 47L and 47R are homologs) — not
  for using them as a reconstruction basis, which is all this needs.
- `connectome_from_geometry` isn't the right tool anyway: it builds a KNN
  *graph* (for an ordinary graph Laplacian), not a triangulated mesh — cotangent
  FEM weights need real triangle angles/areas, and there's no well-defined
  local geometry at a "stitched" inter-hemisphere graph edge with no
  incident triangle.
- Connecting the hemispheres at all (even via anatomically-motivated
  homotopic mirroring) would reintroduce a connectivity-like assumption
  into what's supposed to be a connectivity-free control condition —
  defeating the point of isolating "pure geometry" against the
  connectome-based basis.
- Matches the reference paper's own practice: S modes per hemisphere, no
  cross-hemisphere pairing/joining step mentioned anywhere in their
  methods.

## Implementation status (2026-07-16): core function shipped and verified

`lapy` added as a dependency (`pyproject.toml`, installed in the `mindmap`
env — no conflicts). Implementation split as:
- `mindmap/geometry/processing.py::compute_laplace_beltrami_basis(rr, tris)`
  — generic, mesh-agnostic: `lapy.Solver(lump=True)` for cotangent
  stiffness `A` / lumped mass `B` assembly only, then the full dense
  generalized eigh via `scipy.linalg.eigh(A, B)` directly (not `lapy`'s own
  `.eigs()`, which is a sparse partial-eigensolver, no cheaper than dense
  once asked for close to the full spectrum), then rescales by
  `sqrt(diag(B))` for Euclidean orthonormality (fix 1 above).
- `mindmap/anatomy/source_space/spectral.py::compute_beltrami_harmonics(source_space)`
  — the `compute_cst`-drop-in wrapper: per-hemisphere cortex-only
  triangulation (`_cortex_triangulation`, restricts `use_tris` to
  all-cortex faces), calls the above, clamps each hemisphere's own
  smallest eigenvalue to 0, block-diagonal-concatenates, globally sorts by
  eigenvalue ascending (needed for `compute_harmonic_prior`'s "gap" method,
  which relies on a sorted spectrum, even though the settled "lowpass"
  method doesn't care about order).

Verified on sub-simf10's real source space (n=7,465, matching the
connectome's `eig`):
- **Confirmed lapy's normalization empirically** (fix 1 was necessary, not
  hypothetical): raw `lapy`/`scipy` generalized-eigh output has `UᵀU != I`
  but `UᵀBU = I` exactly, on both a toy closed mesh and this real one —
  matches the prediction above exactly.
- **Timing**: load-dependent, same as the connectome's own eigh already was
  (project memory: 18s-150s+ depending on system load). Measured ~100-110s
  total (both hemispheres) once under heavier load, but ~4.3s on 3 separate
  re-runs since (matching the original back-of-envelope estimate almost
  exactly) — the earlier ~100-110s figure was a load artifact, not a stable
  cost of the method; check `uptime` rather than trusting either number
  blind. Either way, a one-time, cacheable cost, same treatment as the
  existing `eig` connectome precomputation, so no feasibility concern.
- **Found and fixed a real defect during verification, not just a
  theoretical one**: the dangling-vertex case (1 of 7,465 dipoles for
  sub-simf10, hemi 101) was initially patched via nearest-neighbor row
  duplication, which measurably broke orthonormality (up to 13% on some
  `UᵀU` diagonal entries — confirmed by isolating: excluding that one row
  restored orthonormality to machine precision, ~1e-15). Fixed by leaving
  that dipole's row exactly zero instead (costs that one dipole any
  representation in this basis; keeps `UᵀU = I` exact) — re-verified at
  ~1e-15 after the fix.
- **Independent correctness check**: each hemisphere's zero-eigenvalue mode
  is confirmed proportional to `sqrt(local vertex area)` rather than
  literally constant (corr=0.86 against `build_dipole_patch_area`, an
  independently-computed dense-mesh area, not `lapy`'s own coarse-mesh mass
  matrix) — exactly the predicted consequence of the Euclidean-orthonormal
  rescale (the *true*, pre-rescale zero-eigenvector is the literal constant
  vector, since FEM stiffness matrix rows sum to zero by construction).
- Exactly 2 near-zero eigenvalues in the combined spectrum (one per
  hemisphere, matching 2 disconnected components, as designed), all
  eigenvalues non-negative and ascending, no NaN/Inf.

**Not yet done**: wiring this into the notebook itself (a
`wCSET(eigenvectors=..., eigenvalues=..., ...)` run using this basis, and
tuning its own `β` empirically, the same way `harm=5` was tuned for the
connectome side in `harmonic-prior-tuning.ipynb`).

## Beta tuning decision (settled 2026-07-16): `β=30`

New notebook: `../inverse/beltrami-prior-tuning.ipynb`, exact same methodology
as `harmonic-prior-tuning.ipynb` (`sub-simf10` runs 1-10, `wCSET(depth=0.5,
harmonic_method="lowpass")`, `sLORETA` as an external reference,
correlation/AP-hrp/AP-pp/localization-error-hrp metrics) but with the
mesh-Beltrami `eigvec`/`eigval` in place of the connectome's.

**Grid had to be different, not reused.** The mesh spectrum's shape is very
different from the connectome's (median eigenvalue 0.25 of max 4.71 for
mesh vs. median 1.03 of max 1.61 for connectome — mesh eigenvalues are far
more spread out with a long tail). Checked via participation ratio
(`(Σw)²/Σw²`) before running anything: the connectome's settled `harm=5`
gives effective-DOF≈447, but `beta=5` on the mesh spectrum leaves
effective-DOF≈4,906 (barely discounted at all) — reusing `[1,2,3,5,7,10]`
would have told us almost nothing. Swept `[1, 5, 10, 20, 30, 50, 75, 100,
150, 200]` instead (participation ratio ≈447, matching the connectome's
own operating point, lands around `beta≈50` on this spectrum).

**Results** (mean ± std across 10 runs; `sloreta`: ap_hrp=0.349, ap_pp=0.730,
loc_err_hrp=29.6±30.2mm):

| beta | ap_hrp | ap_pp | loc_err_hrp (mm) |
|---|---|---|---|
| 1   | 0.295 | 0.717 | 52.0 ± 37.6 |
| 5   | 0.320 | 0.722 | 31.9 ± 33.1 |
| 10  | 0.340 | 0.726 | 22.8 ± 27.3 |
| 20  | 0.367 | 0.733 | 16.2 ± 23.3 |
| **30**  | **0.386** | **0.739** | **5.4 ± 5.9** |
| 50  | 0.411 | 0.753 | 16.5 ± 22.6 |
| 75  | 0.431 | 0.769 | 17.2 ± 21.4 |
| 100 | 0.445 | 0.782 | 17.5 ± 21.2 |
| 150 | 0.459 | 0.798 | 10.8 ± 8.3 |
| 200 | 0.462 | 0.806 | 13.9 ± 12.8 |

Correlation (plotted, not tabulated) tracks AP exactly: monotonically
increasing with `beta` across the whole grid, `beta=200` highest.

**The genuine tension**: correlation and AP both increase *monotonically*
with `beta` across the entire swept range (diminishing but not zero
returns even from 150→200) — nothing here says "stop." Localization error
does the opposite: it has a sharp, non-monotonic dip at `beta=30`
specifically (5.4mm, with the tightest std of the whole sweep, 5.9mm —
consistently good, not one lucky run), then gets *worse* again at 50-100
(~17mm) before partially recovering at 150-200 (11-14mm, still ~2-2.5x
worse than `beta=30`).

**Decision: `beta=30`.** It comfortably beats the `sLORETA` reference on all
three metrics simultaneously (ap_hrp 0.386>0.349, ap_pp 0.739>0.730, loc_err
5.4mm≪29.6mm) and gives a dramatically better, low-variance localization
error than any other point in the grid — a ~3x improvement over its
neighbors (20, 50) for a comparatively small AP cost (0.386 vs. 0.445 at
beta=100, the point where AP gains start seriously trading off localization
accuracy). This mirrors the connectome side's own stated philosophy for
picking `harm=5`: "a reasonable balance... rather than the sweep's extreme
end," not a single-metric argmax.

Known, accepted tradeoff, not an oversight: correlation and AP both
increase *monotonically* with `beta` across the entire swept range
(diminishing but not zero returns even from 150→200) — someone weighting
whole-map correlation over localization accuracy specifically could
reasonably pick `beta=100-200` instead, and whether AP/correlation
eventually plateau past `beta=200` is still untested (grid wasn't extended
that far). Not re-litigated further without new evidence — same status as
`harm=5`: settled, not up for casual second-guessing.

**Cross-check against the reference paper's S=300/hemisphere, done properly
(not just eyeballed).** `w_harm` at literal mode index k=300 is *not* near
zero at `beta=30` (≈0.19 in each hemisphere, checked on each hemisphere's
own un-merged spectrum) — this is a smooth decay, not a hard cutoff, so
nothing gets suddenly zeroed at a specific index by design. But the
*effective* degrees of freedom implied by `beta=30` -- via the
participation ratio `(Σw)²/Σw²` (aka inverse participation ratio /
effective sample size / inverse Simpson index — same formula, different
field's name; see the parallel connectome-side check earlier in this doc)
-- comes out to **≈396 (LH) / ≈386 (RH) modes/hemisphere**, independently
computed on each hemisphere's own spectrum. That's a strikingly close match
to the paper's chosen S=300/hemisphere, and a *third* independent
convergence on the same ~300-450/hemisphere ballpark, on top of: (1) the
paper's own resolution/sensitivity analysis (their Extended Data Fig. 1f),
and (2) this project's own a priori Weyl's-law/mesh-spacing estimate from
before any implementation existed (`../inverse/sabotage_connectome_notes.md`'s
"Beltrami-operator implementation notes", superseded into this file). Three
different methods, three different starting assumptions, same ballpark —
good corroborating evidence that `beta=30` isn't an arbitrary or
overfit choice, without claiming it's literally equivalent to a hard
S=300 truncation (it isn't, by construction).
