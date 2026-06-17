# 2D Lattice-Boltzmann Street-Canyon Ventilation Study

A reproducible, validated 2D lattice-Boltzmann (LBM) simulation that quantifies
how a street canyon's **aspect ratio** (`H/W`, building height over street
width) governs how efficiently the canyon clears air pollution.

**One-sentence mission.** Measure a single dimensionless ventilation metric
(pollutant retention / air-exchange rate) as a function of `H/W`, reproduce the
classic three flow-regime transition (Oke 1988), validate against real
wind-tunnel data (CODASC), and present it as a clean curve with the
skimming-flow transition located — all on a CPU laptop with open-source Python.

> **Scope & honesty.** This is an *idealized 2D study* with perpendicular wind.
> It describes associations in a simplified canyon; it does **not** claim
> absolute real-street concentrations, causation, or health outcomes. True
> oblique-wind effects are 3D and are documented as future work.

## Method (brief)
- **Flow:** from-scratch NumPy D2Q9 lattice-Boltzmann (BGK, with an MRT/TRT +
  Smagorinsky-LES escalation path for stability at higher Reynolds number).
- **Pollutant:** passive scalar via a D2Q5 advection-diffusion population,
  continuous street-level line source.
- **Boundaries:** halfway bounce-back walls, Zou/He velocity inlet (log-law
  profile), zero-gradient outflow, free-slip top; COST 732 domain sizing.
- **Validation:** CODASC wind-tunnel database; COST 732 metrics (FAC2, NMSE,
  hit rate).

## Install
Requires Python ≥ 3.11.

```bash
make setup          # create .venv, install pinned deps + package (editable)
make test           # run the test suite
```

## Reproduce
```bash
make reproduce      # env -> tests -> regenerate every available figure
```
Individual steps:
```bash
make run            # Phase 1 Poiseuille validation case
make figures        # regenerate figures into figures/
python scripts/run_case.py --poiseuille
```

## Phase status
Work is phase-gated; **grid-independence (Phase 4)** and **CODASC validation
(Phase 5)** are hard gates that must pass before any production sweep.

| Phase | Description | Status |
|------:|-------------|--------|
| 0 | Bootstrap / scaffold | ✅ done |
| 1 | D2Q9 BGK core + Poiseuille validation | ✅ done |
| 2 | Canyon geometry + boundary conditions | ✅ done |
| 3 | Passive-scalar pollutant + ventilation metric | ✅ done |
| 4 | Grid-independence study (gate) | ⏳ next |
| 5 | CODASC validation (gate) | — |
| 6 | Production aspect-ratio sweep | — |
| 7 | Figures (≥5) | — |
| 8 | IMRaD paper draft | — |
| 9 | Reproducibility audit | — |

See `PROGRESS.md` for the running log and `DECISIONS.md` for design rationale.

## Repository layout
```
src/canyon_lbm/   lattice, boundary, geometry, scalar, solver, metrics, io, viz
scripts/          run_case, grid_independence, validate_codasc, run_sweep, make_figures
configs/          base.yaml, validation_codasc.yaml, sweep_aspect_ratio.yaml
tests/            pytest suite (equilibrium, conservation, bounce-back, Poiseuille, metrics)
data/validation/  CODASC reference data (see README there for how to obtain)
results/          generated CSV/JSON + run metadata
figures/          generated figures (PNG + SVG)
paper/            IMRaD draft
```

## License
MIT — see `LICENSE`.

## Key references
Oke (1988) flow regimes; CODASC wind-tunnel database (KIT,
<https://www.codasc.de>); COST 732 best-practice guideline; AIJ CFD guidelines
(Tominaga et al. 2008); Mocz (2020) LBM-in-Python tutorial. Full list in
`paper/` and `DECISIONS.md`.
