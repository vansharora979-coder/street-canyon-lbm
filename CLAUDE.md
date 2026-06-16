# Project: 2D LBM Street-Canyon Ventilation Study

## Goal
Quantify a dimensionless ventilation metric (pollutant retention / air-exchange rate)
vs street-canyon aspect ratio H/W using a 2D D2Q9 lattice-Boltzmann model with a
passive-scalar pollutant. Reproduce Oke's three flow regimes; validate against CODASC;
publish to NHSJS/JHSS.

## Golden rules
- Phase-gated work; grid-independence (Phase 4) and CODASC validation (Phase 5) are HARD gates.
- Reproducibility: config-driven YAML, pinned deps, fixed seeds, run metadata beside every result,
  figures regenerable via `make reproduce`.
- 2D + idealized only. No overclaiming. H/W is the primary deliverable; wind angle = future 3D work.
- Keep u_lbm ≲ 0.1, τ > 0.5; escalate BGK→TRT/MRT→Smagorinsky for stability.
- Plan B: if turbulent solution is unstable on a laptop, reframe as a laminar proof-of-method study.

## Layout
src/canyon_lbm (lattice, boundary, geometry, scalar, solver, metrics, io, viz);
scripts (run_case, grid_independence, validate_codasc, run_sweep, make_figures);
configs (base, validation_codasc, sweep_aspect_ratio); tests; paper (IMRaD); results; figures.

## Conventions (so future sessions stay consistent)
- Distribution array shape is `(9, Ny, Nx)`: axis 0 = direction, axis 1 = y (up), axis 2 = x (inlet→outlet).
- D2Q9 velocity ordering matches Zou/He (1997) / Mocz; see `lattice.C` and `lattice.OPP`.
- Walls are halfway bounce-back via a boolean `solid` mask (`boundary.precompute_bounceback`).
- Lattice↔physical mapping lives in `solver.LatticeUnits` (choose n_cells, u_lbm, Re → τ, ν).
- Every result goes through `io.save_result`, which writes a `*.meta.json` sidecar (git SHA, versions, config).

## Working method
- After each phase: print a status report vs the acceptance criteria, then PAUSE for the user.
- Do not run production sweeps (Phase 6) before BOTH rigor gates pass.
- Record design choices in DECISIONS.md; running progress in PROGRESS.md.

## Status
See PROGRESS.md and DECISIONS.md. (Phases 0–1 complete; awaiting go-ahead for Phase 2.)
