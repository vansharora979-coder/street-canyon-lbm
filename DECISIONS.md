# Design decisions

Each entry: the choice, the alternatives, and why. Append-only; supersede with
a new dated entry rather than rewriting history.

## D1 — Primary solver: from-scratch NumPy D2Q9 BGK
Per the brief. Robust, minimal dependencies, fully inspectable. Optional later
acceleration with `numba`; advanced path (`lbmpy`/`XLB`) only if the from-scratch
solver is validated first and speed becomes the bottleneck.

## D2 — Forcing scheme: Guo (2002)
For the body-force-driven Poiseuille check (and later any source/buoyancy term),
we use Guo forcing — a source term `F_i` in collision plus the `F/2` velocity
correction in the moments. Alternative: the simpler Shan–Chen velocity shift.
Guo cancels the discrete lattice-force artefacts, so a constant force reproduces
the exact parabola; measured rel. L2 error is **7.9e-4**, validating the choice.

## D3 — No-slip walls: halfway bounce-back via a boolean solid mask
Second-order accurate (wall sits halfway between last fluid and first solid
node) and generalises to arbitrary geometry — exactly what the canyon needs.
Alternative: full-way bounce-back (first order) or interpolated bounce-back
(more complex). The mask + precomputed per-direction index sets keep the hot
loop vectorised.

## D4 — D2Q9 velocity ordering matches Zou/He (1997) / Mocz
So the standard Zou/He inlet formulas apply directly (indices 1,5,8 are the
west-inlet unknowns) and the reference tutorial is easy to cross-check.

## D5 — Lattice↔physical mapping parameterised by (n_cells, u_lbm, Re)
`solver.LatticeUnits` fixes the three independent choices — resolution
(`cells_per_H`), lattice velocity (`u_lbm` ≤ 0.1 for low Mach), and the
*simulated* Reynolds number — and derives ν_lbm and τ. Physical `dx`, `dt`
follow from H and U_H only for reporting/diagnostics.

## D6 — Simulated Reynolds number and the stability ladder (PROVISIONAL — locked Phase 4/5)
Real-street canyon Re is ~1e6–1e7 and **cannot** be resolved by a 2D laptop LBM;
we do not claim to. The mean canyon vortex / three-regime structure is
approximately Reynolds-independent above Re ~ 1e4 (the basis for wind-tunnel
validation of full-scale flows), so we target a *simulated* building-height
**Re ~ 1e4** (CODASC order). At laptop resolution (n≈40, u_lbm≈0.05) plain BGK
implies τ ≈ 0.5006 — too close to 0.5 to be stable — confirming we must escalate:
**BGK → MRT/TRT → + Smagorinsky LES**. **Plan B** (per brief §8): if a turbulent
solution is unstable/too slow on the laptop, drop to a transitional Re (~5e2–1e3),
use plain MRT, and explicitly reframe as a laminar/transitional *proof-of-method*
study. The final collision model + Re are locked by the Phase 5 CODASC gate.

## D7 — CODASC validation case
Empty (no-trees) isolated street canyon, **H/W = 1**, **90° (perpendicular)**
wind, street-level line source; compare **normalized wall concentrations c+**
(vertical profiles on leeward & windward walls at the canyon mid-length, i.e.
the 3D tunnel's `y/L = 0` centre plane — the configuration a 2D model can
legitimately represent). COST 732 metrics FAC2 / NMSE / hit rate; acceptance
FAC2 ≥ 0.66. Exact `c+` normalization to be confirmed against the CODASC docs
before Phase 5 (see `data/validation/README.md`).

## D9 — Canyon boundary conditions (Phase 2)
Walls (ground + buildings): halfway bounce-back via the solid mask (D3). Top:
free-slip / specular reflection, applied link-wise from the *post-collision*
distribution (using the streamed array instead injects momentum and blows up the
top row). Inlet (west): **non-equilibrium extrapolation** velocity BC (Guo,
Zheng & Shi 2002) with a log-law profile — the Zou/He velocity inlet seeded a
growing disturbance at the inlet-top/-ground corners. Outlet (east):
**constant-pressure** non-equilibrium extrapolation (rho = 1); a velocity-in /
zero-gradient-out pair does not pin mass and drifts. The free-slip top was
validated independently against the half-channel analytic profile (zero
penetration, zero shear at the top).

## D10 — Outlet viscosity sponge
Even with the non-equilibrium BCs, the (mildly reflective) inlet/outlet sustain a
domain-spanning acoustic standing wave (~5-7% velocity oscillation) that prevents
a clean steady state. A **viscosity sponge** over the last few H before the
outlet (tau ramped smoothly up to ~1.0) absorbs the wake and outgoing acoustic
waves, breaking the resonance. Implemented as a spatially-varying tau field in
`lattice.collide_bgk` / `CanyonSimulation`.

## D11 — Unsteady flow => time-averaging (Phase 2 onward)
The canyon flow is genuinely unsteady above Re ~ O(100) (vortex shedding off the
wall-mounted buildings); no steady state exists, so `CanyonSimulation.run`
reports the **time-averaged mean** field over a window after the transient — the
rigorous choice for a statistically-stationary flow, and the same machinery the
production high-Re (MRT + Smagorinsky LES) runs will use. For a genuinely steady
laminar demonstration, Re below ~70 also works (early-stop on the L2 tolerance).

## D12 — Single-vortex diagnostics
Rotation sense is judged by the **area-integrated cavity vorticity** (negative =>
clockwise, the skimming sense for +x wind aloft), computed after zeroing the
non-physical velocities inside solid cells; plus street-floor reverse flow and a
single sign change of u_x down the canyon centreline. Pointwise upper/lower
samples are reported for context but are unreliable near the vortex centre at low
resolution.

## D13 — Passive scalar: D2Q5 advection-diffusion LBM (Phase 3)
A separate 5-velocity population `g` (rest + 4 neighbours) with the **linear**
advection-diffusion equilibrium `g_i^eq = w_i C (1 + e_i.u/c_s^2)`, one-way
coupled to the D2Q9 flow (the scalar is passive). Diffusivity `D = c_s^2(tau_g-1/2)`;
`tau_g` from the Schmidt number `Sc = nu/D` (default 0.72). Verified against
analytic cases: pure-diffusion variance growth `var = var0 + 2Dt` (0.0% error,
mass conserved to machine precision) and uniform advection (centre of mass
translates exactly at `u`).

## D14 — Scalar boundary conditions
Walls (ground + buildings): **zero-flux** (insulating, non-absorbing pollutant)
via bounce-back of `g`. Inlet (west): **C = 0** Dirichlet (clean approach air).
Outlet (east) and top: **open** (zero-gradient) so pollutant leaves. Source: a
continuous **line source at the street floor** (row 1, street columns), added as
`g_i += w_i S` each step. No sponge needed for the scalar (advection-diffusion,
no acoustics).

## D15 — Ventilation metrics
Reported on the **time-averaged** fields (the flux includes turbulent transport
because it is accumulated from instantaneous C and u, not the product of means):
- **retention** = canyon-mean concentration (higher => worse ventilation);
- **ventilation index** ACH* = w_e/U_H = `Q·H/(content·U_H)` (higher => better);
  these are inverse measures (the brief's "two equivalent ways").
- **conservation check**: the advective flux across the roof-opening plane should
  balance the source rate Q at steady state.
- **CODASC c+** = `C·U_H·H·L_src/Q` (a constant rescaling of C; exact L_src/units
  convention finalized against the CODASC docs in Phase 5, see D7).

## D16 — High-Re stability: MRT + Smagorinsky LES (Phase 4)
At the CODASC-scale simulated Re ~ 1e4 the relaxation time is tau ~ 0.5004-0.5007
(s_nu -> 2), where plain BGK is unconditionally unstable. We escalate to **MRT**
(Lallemand & Luo moment basis; non-hydrodynamic "magic" rates s_e=1.19,
s_eps=1.4, s_q=1.2 from d'Humieres 2002) plus a **Smagorinsky LES** eddy
viscosity (Cs=0.16; strain from central differences, added as a local tau field
on top of the sponge). Validated: MRT reduces to BGK exactly when all rates equal
1/tau (machine precision), conserves mass+momentum, M@M^-1=I; LES adds eddy
viscosity only in shear. Confirmed **stable at Re=1e4** on grids 24 and 48
(coarsest = hardest); the instantaneous field is turbulent (so single-vortex /
mass-balance are only clean after time-averaging, as at Re=100). Chosen by the
user over the laminar Plan B (HPC enables it).

## D17 — GPU backend via CuPy (Phase 4)
The from-scratch solver is pure array operations, so a thin backend module
(`backend.py`) selects NumPy (default, the validated reference) or CuPy (GPU) via
the `CANYON_LBM_BACKEND` env var at import. **NumPy stays the reference**; CuPy
is validated to reproduce it to round-off (BGK rel diff ~4e-14, MRT+LES ~2e-12 on
a deterministic laminar case). Measured GPU speedup on the RTX 4090: ~13x at
96 cells/H (165 -> 12.8 ms/step), turning the ~7-hour grid sweep into ~1 hour.
CuPy is an **optional** dependency (`pip install -e .[accel]`); the pinned
`requirements.txt` stays CPU-only so a no-GPU checkout still reproduces.

## D18 — Phase 4 pivot: laminar (no LES), superseding the Re~1e4 ambition (D16)
The high-Re grid-independence attempt failed for a **methodological** reason, not
just cost. Chronology, all evidenced in the run logs:
- **Re=1e4 (MRT+LES):** n=96 blew up (tau~0.5004, stability knife-edge); halving
  the Mach (u_lbm 0.05->0.025) + Cs 0.16->0.22 fixed stability, but then the
  poorly-ventilated canyon **never equilibrated** (content drifted +57% through
  the whole averaging window at the halved exchange velocity) and carried ~13%
  statistical uncertainty (lag-1 autocorr 0.99) vs a 3% gate.
- **Re=2000 (MRT+LES):** stable and better-behaved, but the metric **moved 3x
  between n=24 and n=48** (retention 1050->362). Root cause: **LES grid-
  independence is ill-posed** -- the Smagorinsky sub-grid viscosity is
  `nu_t=(Cs*dx)^2|S|`, so refining the grid shrinks the filter, lowers nu_t,
  and raises the *effective* Reynolds number. Each grid simulates a different
  flow by construction; a "does the metric stop changing" gate cannot be met.
- **Decision (user-confirmed):** do Phase 4 -- and the production study -- in the
  **laminar/transitional regime (Re=150, BGK, no sub-grid model)**, where the
  flow is fully resolved (grid-refinement converges properly), steady/weakly-
  unsteady (clean metric, negligible statistical noise), and BGK-stable
  (tau 0.524->0.596). This is the brief's **Plan B**, triggered by physics +
  method rather than hardware. The H/W question is still answered: the mean
  canyon vortex and three-regime transition are ~Reynolds-independent; turbulence
  and 3-D are stated limitations, with CODASC (Phase 5) comparing the
  geometry-determined normalized concentration pattern.
- D16 (MRT) and D17 (GPU backend) **remain valid infrastructure**: MRT is a
  more-stable drop-in, and the validated CuPy backend still accelerates the
  laminar production sweep. They are simply not used for the grid-independence
  gate (ill-posed under LES).

## D19 — Phase 5 CODASC: qualitative validation + Péclet insight (gate reframed)
The CODASC reference (AR=1, no trees, 90°; downloaded from umweltaerodynamik.de)
shows a strong leeward/windward asymmetry (c+ ratio ~3.0; leeward 42.6, windward
11.8 at street level) and a gentle vertical decay (street/roof ~2.1). The
grid-converged steady-laminar model (Re=25):
- **captures** the vertical decay (direction right) and, with Péclet matching,
  the leeward>windward asymmetry **direction**;
- but **cannot match it quantitatively**: at the physical Sc=0.72 the Péclet
  number is only Pe=Sc·Re≈18 (diffusion-dominated → near-symmetric, ratio 1.02);
  raising Pe recovers part of the asymmetry (1.69 at Pe=200) **confirming it is a
  Péclet/advection effect**, but the weak laminar vortex can't drive the strong
  mixing of the turbulent tunnel (decay too steep), so the ratio stalls ~1.7.
- FAC2≈0.79 on raw c+ "passes" 0.66 but is coincidental magnitude overlap, not
  correct physics — reported honestly, NOT claimed as a pass.
**Decision (user-confirmed):** the two hard gates (grid-independence ⇒ low Re;
turbulent validation ⇒ high Re) are mutually exclusive in 2-D LBM, so adopt the
brief's **Plan B** — a 2-D laminar **proof-of-method**. CODASC is a qualitative
comparison; the Péclet-controlled asymmetry is a genuine result; quantitative
turbulent validation is an explicit, stated limitation (no overclaiming). The
production H/W sweep runs Péclet-matched (advection-dominated) and clearly
labelled as an idealized laminar exploration.

## D20 — (2026-06-18) LES + GCI ("options 2+3") reconsidered and REJECTED
Mid-Phase-6 it was proposed to reopen the turbulent regime: MRT + Smagorinsky LES
at high Re, report a Grid Convergence Index (GCI) as "error bars", and present a
dual laminar/turbulent result. **Rejected** — it contradicts locked decisions for
reasons already in the repo:
1. **2-D LES grid-independence is ill-posed (D18, Phase 4b).** The Smagorinsky
   sub-grid filter scales with cell size, so refining the grid changes the
   effective Reynolds number; the ventilation metric never settles (it moved ~3x
   from n=24 to n=48 at Re=2000). Structural to 2-D LES, not a tuning issue.
2. **GCI is invalid here.** GCI assumes the solution is in the asymptotic,
   monotonic-convergence range. Our logs show the opposite for the turbulent/
   transitional cases (~3x LES drift; NEGATIVE Richardson order from shedding at
   Re=150). Reporting a GCI band on a non-convergent solution would overclaim and
   break the no-overclaim rule (CLAUDE.md).
3. **The regime is already locked and gated (D18/D19):** steady-laminar Re=25,
   BGK, no LES, 48 cells/H, Péclet-matched; grid-converged (48→96 = 0.0%); CODASC
   compared qualitatively; asymmetry shown to be a Péclet effect.
**Turbulent route, if ever wanted = steady RANS (k-epsilon / k-omega)** — it
grid-converges and is what CODASC-validated canyon studies use. But there is no
RANS in this codebase, adding a two-equation closure to the LBM is a major build,
and 2-D RANS still cannot capture 3-D along-canyon exchange. **Recorded as future
work; not built now.** The study stands as the sanctioned 2-D laminar
proof-of-method.

## D8 — Reproducibility plumbing
`requirements.txt` is pinned from `pip freeze` of the actually-installed
versions (guarantees the pins resolve). Package installed editable via
`pyproject.toml` (src layout). Every result is written through `io.save_result`,
which emits a `*.meta.json` sidecar (UTC timestamp, git SHA, platform, library
versions, config snapshot). Seeds are fixed in configs and tests.

## D21 — (2026-06-18) Phase 6.5: skimming collapse reproduced at high Péclet (OUTCOME A)
Phase 6 found *no* monotonic skimming collapse — but it swept at **Pe=50**, below
the advection-dominated regime. Phase 6.5 tested whether the collapse appears once
advection dominates, **staying entirely in the locked regime** (steady-laminar
Re=25, BGK, no LES/RANS/turbulence): only the scalar diffusivity changed, via the
Schmidt number. Exploiting the one-way scalar coupling (D13), the steady D2Q9 flow
was solved **once per H/W** and reused across the whole Péclet ladder; only the
cheap D2Q5 scalar re-solved. H/W ∈ {0.5, 1, 2, 3}, Pe ∈ {50, 72, 144, 200}.

**Result — the collapse emerges as Pe rises.** Retention is flat/non-monotonic at
Pe=50 (the Phase 6 finding) but rises **monotonically** with H/W at Pe≥144, and
ACH* falls; the collapse is fully developed by Pe=200 (retention 1721→2510→3596
for H/W=0.5→1→2, saturating ~H/W 2–3). So Phase 6's "no collapse" was a Péclet
artifact, **not** a property of the laminar model — Phase 6.5 refines, not
contradicts, Phase 6.

**Grid check (the gate), Pe=200, H/W=1, n=24/48/96 — PASSED.** retention_eq
0.2%, asymmetry 0.7%, retention_raw 0.3% (n=48→96); all CONVERGED, observed
(Richardson) order p≈+5. The naive per-step ACH showed "49.9%", but that was a
**units artifact**: ACH in [1/step] scales with the timestep (dt ∝ dx ∝ 1/n), so
it halves on each n-doubling by construction (8.30e-6→4.16e-6 = exactly ×0.5). The
grid-invariant **ACH\* = ACH·(H/u_lbm)** converges to **0.2%** (0.00797→0.00798).
The figure and paper report ACH\*, not per-step ACH.

**Grid-converged Péclet ceiling ≥ 200 (at 96 cells/H).** Despite Pe_cell≈2.1 at
n=96 (and ≈4.2 at n=48), the *integral/ratio* metrics — retention and ACH\* — are
grid-converged because under-resolution perturbs only the thin per-cell roof flux
detail, not the bulk canyon mean. retention is grid-robust even at n=48 (within
0.3% of n=96). The collapse is therefore a converged result, not a coarse-grid
artifact.

**VERDICT: OUTCOME A.** The skimming collapse IS reproduced and IS grid-converged,
in the advection-dominated regime, within the locked 2-D laminar model. The
controlling parameter is the **Péclet number** (advection vs diffusion of the
pollutant), consistent with D19's asymmetry finding. No turbulence was reopened;
D18/D20 stand. Figure `figures/peclet_hw_diagnostic.png`; data
`results/phase6_5_peclet.{json,summary.csv}`.

## D22 — (2026-06-19) Re=25 flow is weakly unsteady (residual acoustic mode); Phase 6.5b convergence result

### Part A — The Re=25 flow is NOT steady: "steady" → "time-averaged statistically-stationary"

**Discovery.** The instantaneous Re=25 canyon flow does not reach a steady state —
the cavity circulation oscillates at a fixed period and the L2 speed-change
plateaus at ~1e-1, never the 1e-6 a steady state would hit. Missed until now
because every prior result used `average_from` (time-averaging), so the
instantaneous field was never inspected. The "Re=25 < ~Re47 shedding onset →
steady" label (D11, D18) was an untested assumption, not a measurement.

**Check 1 — physical shedding vs numerical acoustic? → NUMERICAL (acoustic mode).**
Discriminators via `scripts/diagnose_unsteadiness.py` (8 cases, results in
`results/_diag_unsteady.log`):

**(Mach sweep — decisive.)** Period ~constant in steps as u varies (5025/5038/5556
at u=0.05/0.025/0.10), so St ∝ 1/u (0.190/0.380/0.095). Physical shedding has
St ≈ const; acoustic modes have T ≈ const in steps → period set by the lattice
sound speed, not by flow geometry. Verdict: **acoustic, not shedding**.

**(Domain)** 16H outflow shifts St→0.114 and cuts amplitude 10× (<0.001 u) →
domain-resonant, suppressible. Consistent with a standing acoustic wave pinned to
the domain length.

**(Sponge)** Doubling sponge length or strengthening to τ_sponge=1.9 leaves
frequency and amplitude unchanged → the D10 sponge cannot fully damp a standing
wave that re-enters from the interior.

**(Grid)** St≈0.19 at n=24/48/96; period ∝ n → wavelength ∝ grid spacing.

Amplitude: <1% of u_lbm throughout. The St≈0.19 resemblance to physical shedding
is coincidental; calling it shedding would overclaim.

**Check 2 — window-independence of the mean.** The time-averaged cavity circulation
converges as the window grows (from BASE case, n=48, u=0.05):

| window | periods | circ | Δ from prev |
|-------:|--------:|-----:|------------:|
| 7 500 steps | 1.5 | −0.24074 | — |
| 14 500 | 2.9 | −0.24710 | 2.64% |
| 21 750 | 4.3 | −0.23642 | 4.32% |
| 29 000 | 5.8 | −0.24404 | 3.22% |
| 36 250 | 7.2 | −0.23982 | 1.73% |
| 43 500 | 8.7 | −0.24160 | 0.74% |
| 50 500 | 10.0 | −0.24243 | 0.34% |

**By ~10 periods (≈3.5 flow-throughs) the mean is window-independent to <0.5%.**
Also domain-independent: −0.242 at 8H outflow vs −0.2435 at 16H.

**Re-justification of Re=25 (replaces D11/D18 "steady" rationale).** The regime is
**"time-averaged (statistically-stationary), weakly-unsteady laminar Re=25."** The
time-averaged mean is (a) window-independent ≥10 periods, (b) domain-independent,
(c) grid-robust in the ventilation metric (Phase 4b/6.5). Re=25's weak
single-frequency acoustic ripple averages out cleanly; Re=150's stronger genuine
unsteadiness diverges on refinement (D18). No physical shedding at Re=25; the
acoustic mode is a numerical artefact that is irrelevant to the mean vortex.

**Production recipe** (Phase 6.5b): BURN_FT=1.5 (discard startup transient) +
AVG_FT=3.5 (~10 periods); subsampled accumulation every samp=ft//200 steps (~70
samples/period); single-host-transfer at end. Validated: reproduces the diagnostic
cavity_circ to 4 significant figures.

**Correction superseding "steady" language in D11, D18, D19, D20, D21 and script
docstrings.** The label "steady laminar Re=25" should read "time-averaged
(statistically-stationary) laminar Re=25" throughout. No turbulence reopened; D18
and D20 stand in all other respects.

### Part B — Phase 6.5b: Pe=144 clean-resolution convergence result

**Purpose.** Phase 6.5's grid check was at a single point (Pe=200, H/W=1, n=48→96)
with all three grids sitting at Pe_cell≥2 (borderline resolved). Phase 6.5b tested
whether the HEADLINE Pe=144 metric converges at **both** grids in the
unambiguously-resolved regime (Pe_cell<2): n=96 (Pe_cell=1.50) and n=192
(Pe_cell=0.75), across all four H/W values.

**Results — Pe=144, n=96 vs n=192 (Pe_cell=1.50 vs 0.75):**

| H/W | ret_eq n=96 | ret_eq n=192 | Δret_eq | ΔACH* | Δasym |
|----:|------------:|-------------:|--------:|------:|------:|
| 0.5 | 1340.8 | 1125.9 | **16.03%** | **19.09%** | 6.18% |
| 1.0 | 1963.4 | 1964.2 | **0.04%** | **0.04%** | 5.11% |
| 2.0 | 2309.1 | 2227.1 | **3.55%** | **3.68%** | 0.77% |
| 3.0 | 2246.4 | 2092.9 | **6.83%** | **7.34%** | 0.05% |

Worst change: **19.09%** (H/W=0.5, ACH*). All hygiene flags clean (tau_g>0.5,
Pe_cell<2, minC≈0.000%).

**VERDICT: <1% criterion FAILS for H/W∈{0.5, 2.0, 3.0}. Outcome A is NOT locked
as "fully grid-converged at Pe=144 across all H/W."**

**What holds:**
- H/W=1.0 converges exceptionally (0.04% in ret/ACH* → the Phase 6.5 H/W=1 gate
  was representative for that geometry).
- The collapse *direction* is robust at both grids: monotonic rise H/W=0.5→2
  with saturation H/W=2→3, visible at n=96 AND n=192. The qualitative finding
  (Péclet drives the collapse) is not in doubt.
- The n=192 curve (Pe_cell=0.75) is the best-resolved estimate of the headline
  values: 1126 / 1964 / 2227 / 2093 for H/W=0.5/1/2/3.

**Why the non-convergence at H/W=0.5 (most likely cause).** The BGK flow solver
uses different τ_flow between grids (1.076 at n=96, 1.652 at n=192) because the
physical Re is held fixed while the lattice viscosity scales with n. The cavity
circulation ratio n=192/n=96 = 2.26/1.40 = 1.61 (expected ≈2.0 for pure grid
scaling); n=192 produces a proportionally weaker vortex, changing the advection
pattern seen by the scalar and therefore the flux balance. This flow non-equivalence
is widest at H/W=0.5 (ratio 1.61) and narrows toward H/W=3 (ratio 1.94), which
matches the convergence ordering.

**Pe=200 follow-on (optional, runs on cached flows):**
- H/W=1.0: Δret=0.85%, ΔACH*=0.85% (borderline)
- H/W=2.0: Δret=4.34%, ΔACH*=4.53% (fail)

**Implication for the paper narrative.** The Phase 6.5 headline result (Outcome A:
collapse emerges as Pe↑, Pe controls the regime) is not contradicted — the pattern
is robust. But the "grid-converged" qualifier is now limited to H/W=1. For H/W∈{0.5,
2, 3} at Pe=144, the two resolved grids (n=96, n=192) are not yet in the asymptotic
regime; the n=192 values are the best-available resolved estimate. No turbulence
reopened; D18/D20/D21 stand; paper framing in `results_limitations_framing.md`
NOT touched (per locked constraint). Data: `results/phase6_5b_peclet.{json,summary.csv}`.

## D23 — (2026-06-20) Pe=144 non-convergence is in the FLOW but NOT a bounce-back-wall (BGK-τ) artifact — TRT magic-parameter test REFUTES the wall hypothesis [NO-GO]

**TL;DR.** We hypothesised the H/W=0.5 Pe=144 grid non-convergence was BGK's
τ-dependent bounce-back wall, and tested it by re-solving the flow as pure TRT with
the magic parameter Λ=3/16 (τ-independent wall). **Result: the TRT flow is identical
to BGK at both grids (circ ratio 1.608× vs 1.614×; worst metric Δ 19.07% vs 19.09%).
The wall hypothesis is REFUTED.** The non-convergence is real and in the flow, but
its cause is NOT the wall BC. Honest fallback adopted (see OUTCOME).

**The non-convergence is in the FLOW, not the scalar.** The D22B Pe=144
non-convergence (16–19% n=96→192 at H/W=0.5) occurred while the *scalar* was clean
(Pe_cell<2, tau_g>0.5, minC≈0). The signal is in the time-averaged flow itself: the
cavity circulation scales by only 1.61× from n=96→192 at H/W=0.5, where pure grid
scaling demands **2.0×** (a fixed velocity field on a 2×-finer grid: per-cell
vorticity halves, cavity-cell count quadruples → Σω doubles). The deficit narrows
monotonically — 1.61 / 1.72 / 1.87 / 1.94 for H/W=0.5/1/2/3 — exactly matching the
convergence ordering (worst at H/W=0.5, best at H/W=1... and note H/W=1's
near-2.0-consistent metric convergence is partly luck of where its curve sits).

**HYPOTHESIS (tested below, REFUTED): BGK's bounce-back wall location is τ-dependent.**
In TRT language the wall position is governed by the magic parameter
Λ = (1/s_nu − ½)(1/s_q − ½). BGK sets every rate to 1/τ, so Λ_BGK = (τ − ½)². To
hold Re=25 fixed while n changes, τ must change (1.076 at n=96, 1.652 at n=192), so
**Λ_BGK jumps 0.33 → 1.33** — a 4× change. The hypothesis was that the effective
no-slip plane therefore sits at a different sub-cell location on each grid, so the
two grids solve subtly different geometries. Plausible (a known BGK trait; Ginzburg;
d'Humières & Ginzburg 2009) — but the test below shows it is NOT what drives this
flow's grid-dependence.

**Correction to the Phase-4b / Pe=50 "grid-converged" interpretation.** Phase 4b
(retention 48→96 = 0.0%) and Phase 6 (Pe=50, 48→96 = 1.1%) reported the flow as
"grid-converged." That was over-read: at low Péclet the scalar transport is
**diffusion-dominated and largely insensitive to the detailed flow**, so the metric
looked converged even though the underlying BGK flow was not (the same τ-dependence
was present — it simply did not show through a diffusion-dominated metric). The flow
non-convergence only becomes visible once advection dominates (high Pe), which is
precisely the regime Phase 6.5/6.5b operate in. So "metric stopped moving at Pe=50"
≠ "flow grid-converged"; it means "metric insensitive to the flow at Pe=50."

**Fix (root cause): re-solve the FLOW as pure TRT with Λ = 3/16.** The magic value
3/16 pins the bounce-back wall at the mid-link **independent of τ**, so both grids
solve the same geometry and the flow converges. Implemented via the existing MRT
routine (Phase 4a): all even non-conserved moments relax at s_nu = 1/τ (viscosity,
hence Re=25, unchanged — only the odd q-rate moves), and s_q is set so Λ=3/16. NO
LES; the laminar Re=25 regime is otherwise untouched. Verified: MRT with all rates
= 1/τ reproduces BGK to 1e-15; magic_s_q gives Λ=0.1875 at both τ.
`scripts/phase6_5c_mrt_test.py`.

**TEST (before committing to the full re-run): H/W=0.5 (worst case), n=96 & n=192,
Pe=144.** GO criterion: circulation ratio → ~2.0 AND retention_eq/ACH*/asymmetry
converge to <~3%. Also confirmed the time-averaging window is ~9.9 acoustic periods
at BOTH grids (analytically from T∝n, and measured in-run), so averaging is not a
second confound.

**OUTCOME: NO-GO (wall hypothesis REFUTED).** H/W=0.5, Pe=144, with the TRT magic
parameter Λ=3/16 enforced on the flow at both grids:

| metric | BGK n=96→192 | TRT n=96→192 |
|---|---|---|
| circ ratio (n192/n96) | 1.614× | **1.608×** |
| retention_eq Δ | 16.03% | **16.01%** |
| ACH* Δ | 19.09% | **19.07%** |
| asymmetry Δ | 6.18% | **6.21%** |

The TRT flow differs from BGK by only ~0.6% in circulation (n=192) despite a 7×
change in Λ, and the convergence is unchanged to 2 decimals. **Pinning the wall does
essentially nothing here** — at Re=25 the cavity circulation is set by the
large-scale recirculation balance, not by sub-cell wall slip, so the τ-dependent
wall was a red herring. (The averaging window was 13–17 acoustic periods at both
grids, so that is not the confound either.) Data: `results/phase6_5c_mrt_test.json`,
`results/_flowcache/flow_mrt_n{96,192}_ar0.5.npz`, `scripts/phase6_5c_mrt_test.py`.

**What the non-convergence actually is (best current understanding).** The flow
grid-dependence (circ ratio 1.61, not 2.0) is REAL but is NOT a wall-BC artifact.
The remaining candidates — not yet separated — are (a) the wide-canyon (H/W=0.5)
time-averaged vortex structure is genuinely still resolving (n=96→192 is not yet the
asymptotic range for this geometry; the shallow W=2H cavity has more corner/secondary
structure than the tall canyons, consistent with the deficit being worst at H/W=0.5
and mild at H/W=3), and/or (b) residual scalar numerical diffusion that halves with
the grid even at Pe_cell<2. A cheap follow-up that WOULD separate them: run one
frozen flow (e.g. the n=192 mean, restricted/coarsened) through the scalar at both
scalar resolutions — flow-fixed. Not done (user pre-committed to STOP on NO-GO).

**Honest fallback (adopted).** Report the **n=192 values as the best-resolved
estimate**, and state the residual grid-sensitivity at low H/W explicitly — now with
the stronger, tested statement that it is **NOT** a bounce-back-wall artifact (TRT
magic-parameter ruled that out to ~0.1%). The **directional result is robust**: the
skimming collapse (retention rising with H/W, saturating ~H/W 2–3) holds at n=96 AND
n=192 AND under both BGK and TRT. The Péclet-controls-the-collapse headline (D21)
stands; only the "grid-converged to <1% at every H/W" claim is retired (it holds at
H/W=1; elsewhere n=192 is best-resolved with a stated few-to-~16% grid band).

**Consequences.** The full four-H/W MRT re-run is CANCELLED — it would reproduce BGK
(TRT≈BGK), so it buys nothing. The MRT-collision speed/OOM optimisation is therefore
also moot (only mattered for that re-run). No turbulence reopened; D18/D20/D21 stand;
`results_limitations_framing.md` NOT touched (per the locked constraint — framing
revision is the user's call once these numbers land).
