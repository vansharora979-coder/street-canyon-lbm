# 2-D Lattice-Boltzmann Street-Canyon Ventilation Study

A reproducible, from-scratch 2-D lattice-Boltzmann (LBM) model that quantifies how
a street canyon's geometry and flow regime govern its ability to clear street-level
air pollution.

## The finding (in two sentences)

Using a validated 2-D D2Q9 lattice-Boltzmann model with a passive-scalar pollutant,
this study shows that street-canyon pollutant **trapping — both the leeward/windward
concentration asymmetry and the skimming-flow ventilation drop with aspect ratio
`H/W` — is controlled by the Péclet number** (the balance of advective vs. diffusive
scalar transport), not by canyon geometry alone. The classic aspect-ratio dependence
that echoes Oke's (1988) flow regimes emerges **only once advection dominates** and
is absent when diffusion dominates, identifying advective dominance as the
controlling parameter for canyon ventilation.

> **Scope & honesty (please read before citing).** This is an **idealized 2-D study
> with perpendicular wind** in a **time-averaged, weakly-unsteady laminar regime
> (Re = 25)** — the regime in which the lattice solution is provably grid-independent.
> The high effective Péclet numbers are reached by **lowering the scalar diffusivity
> in a laminar flow** — a *proxy for advective dominance, not turbulence*. Real
> canyons reach that regime through turbulence; **turbulence and 3-D (oblique-wind)
> effects are the stated realism gap and are documented as future work.** The study
> reports **trends and mechanisms in dimensionless terms** — it does **not** claim
> absolute real-street concentrations, causation, or health outcomes, and it does
> **not** claim that a 2-D laminar model "reproduces Oke."

## Method (brief)

- **Flow:** from-scratch NumPy D2Q9 lattice-Boltzmann, BGK collision (with an
  MRT/TRT + Smagorinsky-LES escalation path in the code for stability studies).
  Optional CuPy GPU backend (NumPy stays the validated reference).
- **Pollutant:** passive scalar via a D2Q5 advection-diffusion population, one-way
  coupled to the flow, with a continuous street-level line source.
- **Boundaries:** halfway bounce-back walls, non-equilibrium velocity inlet (log-law
  profile), constant-pressure outflow, free-slip top, outlet viscosity sponge;
  COST 732 domain sizing.
- **Validation:** Poiseuille channel (analytic, ~0.08 % L2 error) and the **CODASC**
  wind-tunnel database (qualitative — the model captures the vertical `c⁺` decay and
  the Péclet-controlled asymmetry; absolute turbulent values are out of scope).
- **Metrics:** canyon-mean retention, a grid-invariant air-exchange rate
  `ACH* = ACH·H/u`, and the leeward/windward concentration asymmetry.

## Key results

- **Wall asymmetry rises monotonically with Péclet number** (leeward/windward ratio
  climbs from ≈1.0 in the diffusion-dominated limit toward the wind-tunnel value as
  advection takes over) — `figures/pe_sensitivity_asymmetry.png`.
- **The skimming-flow ventilation collapse appears only at high Péclet:** retention
  rises (and `ACH*` falls) monotonically with `H/W` in the advection-dominated regime
  and is flat/non-monotonic when diffusion dominates — `figures/peclet_hw_diagnostic.png`.
- **Grid independence** is established for the laminar regime (Phase 4) and holds
  cleanly for the metric at `H/W = 1`; the wide-canyon (`H/W = 0.5`) metric retains a
  resolution band between the two finest grids that a dedicated test showed is **not**
  a bounce-back-wall artifact (see `DECISIONS.md` D23). The finest grid is reported as
  the best-resolved estimate; the *directional* result is robust across grids and
  collision models.

Full reasoning and every design decision are logged in `DECISIONS.md`; the running
phase-by-phase narrative is in `PROGRESS.md`.

## Install

Requires Python ≥ 3.11.

```bash
make setup          # create .venv, install pinned deps (requirements.txt) + package (editable)
make test           # run the test suite (D2Q9 moments, conservation, bounce-back, Poiseuille, metrics)
```

Exact, pinned dependency versions are in `requirements.txt`. The optional GPU
backend (`pip install -e .[accel]`, selected at runtime via
`CANYON_LBM_BACKEND=cupy`) needs a CUDA-capable GPU; NumPy is the validated default.

## Reproduce

```bash
make reproduce      # env -> tests -> demo cases -> regenerate figures
```

This regenerates the Poiseuille validation, the canyon schematic, the `H/W = 1` flow
and concentration demo, and the Péclet/asymmetry/aspect-ratio figures **from the
committed result summaries** (`results/*.summary.csv`). A few figures
(`grid_independence`, `codasc_validation`, `les_grid_divergence`) depend on inputs
that are *not* redistributed (large run outputs / KIT's CODASC raw data); the figures
are committed as the record, and `make figures` skips them gracefully with a printed
note explaining which script regenerates each. The production high-Péclet sweep
itself (`scripts/phase6_5*.py`) is a GPU job, not part of the quick `make reproduce`.

## Repository layout

```
src/canyon_lbm/   lattice, boundary, geometry, scalar, solver, metrics, io, viz
scripts/          run_case, grid_independence, validate_codasc, run_sweep, pe_sensitivity,
                  phase6_5{,b,c}_peclet, diagnose_unsteadiness, make_figures
configs/          base.yaml, canyon_demo.yaml, validation_codasc.yaml, sweep_aspect_ratio.yaml
tests/            pytest suite (equilibrium, conservation, bounce-back, Poiseuille, metrics)
data/validation/  how to obtain the CODASC reference data (not redistributed — see README there)
results/          committed result summaries (*.summary.csv); per-run blobs are git-ignored
figures/          generated figures (PNG; vector SVG for the small ones)
paper/            framing/scaffold for the IMRaD manuscript (in preparation)
DECISIONS.md      design decisions + rationale (append-only)
PROGRESS.md       phase-by-phase running log
```

## Figures & paper

- **`figures/`** — the figure set (PNG, plus SVG for the lighter ones). These are the
  results of record; `make figures` regenerates the ones whose inputs are committed.
- **`paper/`** — the manuscript is in preparation; `paper/README.md` and
  `paper/results_limitations_framing.md` hold the Results/Limitations scaffold. The
  final draft lands in Phase 8.

## Status

Phase-gated; **grid-independence (Phase 4)** and **CODASC validation (Phase 5)** are
hard gates that gate any production sweep.

| Phase | Description | Status |
|------:|-------------|--------|
| 0 | Bootstrap / scaffold | ✅ |
| 1 | D2Q9 BGK core + Poiseuille validation | ✅ |
| 2 | Canyon geometry + boundary conditions | ✅ |
| 3 | Passive-scalar pollutant + ventilation metric | ✅ |
| 4 | Grid-independence gate (laminar Re = 25) | ✅ |
| 5 | CODASC validation (qualitative) | ✅ |
| 6 | Production aspect-ratio sweep | ✅ |
| 6.5 | High-Péclet `H/W` diagnostic — Péclet control confirmed | ✅ |
| 7 | Figure set | ✅ |
| 8 | IMRaD paper draft | ⏳ in preparation |
| 9 | Reproducibility audit | ⏳ in progress |

## How to cite

A manuscript describing this work is in preparation (working title: *Quantifying
Street-Canyon Ventilation: A 2-D Lattice-Boltzmann Study of How Aspect Ratio and the
Péclet Number Govern Pollutant Trapping*; target: NHSJS). Until it appears, please
cite the repository — the formal paper citation will be added here on acceptance:

```
<Author(s)>. Quantifying Street-Canyon Ventilation: A 2-D Lattice-Boltzmann Study of
How Aspect Ratio and the Péclet Number Govern Pollutant Trapping. GitHub repository,
2026. <repository URL>
```

## License

MIT — see `LICENSE`.

## Key references

Oke (1988), street-canyon flow regimes; CODASC wind-tunnel database (Gromke & Ruck,
KIT — <https://www.codasc.de>); COST 732 best-practice guideline (Franke et al. 2007);
AIJ CFD guidelines (Tominaga et al. 2008); Mocz (2020), LBM-in-Python tutorial. Full
reference list in `paper/` and `DECISIONS.md`.
