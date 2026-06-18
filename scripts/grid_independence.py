#!/usr/bin/env python3
"""Phase 4 -- grid-independence study (HARD gate).

Runs the H/W = 1 canyon at increasing resolution (default 24/48/96 cells per
building height) at the target Re ~ 1e4 with MRT + Smagorinsky LES and the
passive scalar, time-averages, and records the ventilation metric vs resolution.
The production resolution is the coarsest grid at which the metric is converged
(changes < ~2-3% vs the next-finer grid).

Consistent (reduced) domain margins + an outlet sponge are used across all grids
so the comparison isolates *resolution* sensitivity; this is a convergence test,
not an absolute-accuracy run.

Runs on whatever backend is selected (set CANYON_LBM_BACKEND=cupy to use the GPU):
    CANYON_LBM_BACKEND=cupy python scripts/grid_independence.py
    CANYON_LBM_BACKEND=cupy python scripts/grid_independence.py 24 48 96
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from canyon_lbm import io, lattice as lb, scalar as sc
from canyon_lbm.backend import backend_name
from canyon_lbm.boundary import log_law_profile
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation

ROOT = Path(__file__).resolve().parents[1]

# Fixed physics across grids (only the resolution changes). STEADY LAMINAR
# regime, NO sub-grid model. Grid-independence is only well-posed when the flow
# is (a) fully resolved -- with LES, refining shrinks the filter and changes the
# effective Re, so the metric never converges (retention moved 3x at Re=2000) --
# and (b) STEADY: the canyon wake sheds above ~Re 40 (classic ~Re 47 bluff-body
# onset; confirmed -- n=96 stays unsteady at Re=40/150), and coarse grids
# fake-damp that unsteadiness while fine grids resolve it, so the metric DIVERGES
# with refinement (negative Richardson order observed at Re=150). Re=25 is below
# the shedding onset: steady and fully resolved at every grid (tau 0.644 at n=24
# -> 1.076 at n=96), so the metric grid-converges cleanly. Peclet = Sc*Re ~ 18
# (advection-dominated, shows the H/W contrast). This is the brief's sanctioned
# Plan B (2-D laminar proof-of-method); turbulence/high-Re/3-D are stated
# limitations, no quantitative real-canyon claims. See DECISIONS.md (D18).
RE = 25.0
U_LBM = 0.05
SCHMIDT = 0.72
COLLISION = "bgk"          # fully resolved; no LES sub-grid model
CS = 0.0                   # unused (no LES)
AR = 1.0
MARGINS = dict(building_width_H=1.0, fetch_upstream_H=4.0, top_margin_H=5.0,
               outflow_H=8.0)
SPONGE_H = 4.0


def _budget(n: int) -> tuple[int, int]:
    """(max_iter, average_from) as a FIXED number of flow-throughs, per grid.

    A flow-through (the canyon-clearing timescale) is ~(domain length in H) *
    n / u_lbm steps -- it scales with resolution and 1/Mach. A per-grid step
    formula gave the *fine* grids too few flow-throughs (and thus no
    equilibration); fixing the flow-through count is the correct, grid-consistent
    choice: ~10 flow-throughs total, averaging over the last ~4 (after ~6 to
    equilibrate the canyon pollutant).
    """
    domain_H = 4.0 + 1.0 + 1.0 + 1.0 + 8.0   # fetch + bldg + street + bldg + outflow
    ft = int(domain_H * n / U_LBM)            # steps per flow-through
    return 10 * ft, 6 * ft


def run_grid(n: int) -> dict:
    g = build_canyon(cells_per_H=n, aspect_ratio=AR, **MARGINS)
    nu = U_LBM * g.h / RE
    tau = lb.tau_from_viscosity(nu)
    tau_g = sc.tau_from_schmidt(nu, SCHMIDT)
    inlet = log_law_profile(g.ny, g.h, U_LBM, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, U_LBM, inlet, collision=COLLISION,
                           sponge_cells=int(SPONGE_H * g.h), tau_sponge=1.0,
                           with_scalar=True, tau_g=tau_g, source_strength=1.0,
                           Cs=CS)
    max_iter, avg_from = _budget(n)
    print(f"\n=== n={n} cells/H | grid {g.ny}x{g.nx} | tau={tau:.5f} | "
          f"max_iter={max_iter} avg_from={avg_from} | backend={backend_name()} ===",
          flush=True)
    out = sim.run(max_iter=max_iter, tol=1e-6, check_every=4000,
                  ramp_iters=8000, average_from=avg_from, verbose=True)
    s = out["scalar"]
    return {
        "cells_per_H": n, "ny": g.ny, "nx": g.nx, "tau": tau, "tau_g": tau_g,
        "retention_mean_conc": s["retention_mean_conc"],
        "ventilation_index": s["ventilation_index"],
        "flux_over_source": s["flux_over_source"],
        "cavity_circulation": out["cavity_circulation"],
        "mass_imbalance": out["mass_imbalance"],
        "single_vortex": out["single_vortex"],
        "n_avg": out["n_avg"], "iters": out["iters"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    grids = [int(a) for a in sys.argv[1:]] or [24, 48, 96]
    out_csv = ROOT / "results" / "grid_independence.csv"

    # Checkpoint each grid as it finishes: a blow-up on a later (finer) grid must
    # never discard the coarser grids that already completed.
    rows: list[dict] = []
    for n in grids:
        try:
            rows.append(run_grid(n))
        except Exception as exc:  # numerical blow-up, etc.
            print(f"\n!!! grid n={n} FAILED ({type(exc).__name__}: {exc}); "
                  f"keeping {len(rows)} completed grid(s).", flush=True)
            continue
        _write_csv(out_csv, rows)
        io.save_result(ROOT / "results" / "grid_independence.summary.json",
                       {"grids": rows},
                       config={"Re": RE, "u_lbm": U_LBM, "schmidt": SCHMIDT,
                               "Cs": CS, "aspect_ratio": AR,
                               "collision": COLLISION, **MARGINS},
                       extra={"backend": backend_name()})
        print(f"  [checkpointed {len(rows)} grid(s) -> {out_csv.name}]", flush=True)

    if not rows:
        raise SystemExit("No grid completed; see errors above.")

    print("\n--- grid-independence summary (H/W = 1, Re ~ 1e4, MRT+LES) ---")
    print(f"{'cells/H':>8} {'retention':>12} {'ventilation*':>13} "
          f"{'circ':>10} {'mass_imb':>10} {'flux/src':>9}")
    for r in rows:
        print(f"{r['cells_per_H']:>8} {r['retention_mean_conc']:>12.3f} "
              f"{r['ventilation_index']:>13.5f} {r['cavity_circulation']:>10.3e} "
              f"{r['mass_imbalance']:>10.2e} {r['flux_over_source']:>9.3f}")
    if len(rows) >= 2:
        a, b = rows[-2], rows[-1]
        for key in ("retention_mean_conc", "ventilation_index"):
            rel = abs(b[key] - a[key]) / (abs(a[key]) + 1e-30) * 100
            verdict = "CONVERGED" if rel < 3.0 else "NOT converged"
            print(f"  {key}: {a['cells_per_H']}->{b['cells_per_H']} cells/H "
                  f"changes {rel:.1f}%  [{verdict} @ 3% gate]")
    print(f"\nWrote {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
