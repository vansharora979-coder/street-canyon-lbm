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

## D8 — Reproducibility plumbing
`requirements.txt` is pinned from `pip freeze` of the actually-installed
versions (guarantees the pins resolve). Package installed editable via
`pyproject.toml` (src layout). Every result is written through `io.save_result`,
which emits a `*.meta.json` sidecar (UTC timestamp, git SHA, platform, library
versions, config snapshot). Seeds are fixed in configs and tests.
