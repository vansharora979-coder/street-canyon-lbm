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

## D8 — Reproducibility plumbing
`requirements.txt` is pinned from `pip freeze` of the actually-installed
versions (guarantees the pins resolve). Package installed editable via
`pyproject.toml` (src layout). Every result is written through `io.save_result`,
which emits a `*.meta.json` sidecar (UTC timestamp, git SHA, platform, library
versions, config snapshot). Seeds are fixed in configs and tests.
