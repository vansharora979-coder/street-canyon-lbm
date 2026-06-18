#!/usr/bin/env python3
"""Phase 6 -- production aspect-ratio sweep (idealized laminar exploration).

Sweeps H/W in {0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0} at the production resolution
(48 cells/H), in the grid-converged steady-laminar regime (Re=25, BGK), with the
passive scalar at an advection-dominated Peclet number (Sc raised so dispersion
follows the canyon vortex -- see Phase 5 / DECISIONS D19). Locates the
skimming-flow transition in the ventilation-vs-H/W curve.

The cases are independent, so they run in PARALLEL across CPU cores (NumPy
backend). To avoid needing each canyon to fully equilibrate (slow at high Pe),
the ventilation metric is the FILL-INDEPENDENT equilibrium retention
  retention_eq = retention_mean_conc / flux_over_source
(the linear-fill correction: content/flux is constant once the flow is steady),
plus the air-exchange rate ACH = opening_flux / canyon_content.

SCOPE: 2-D, laminar, idealized. Trends and the regime transition are the
deliverable -- not quantitative real-canyon concentrations.

Usage:  python scripts/run_sweep.py            (7 production cases, parallel)
        python scripts/run_sweep.py 1.0 96     (single H/W at a given resolution)
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from pathlib import Path

from canyon_lbm import io, lattice as lb, scalar as sc
from canyon_lbm.boundary import log_law_profile
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation

ROOT = Path(__file__).resolve().parents[1]

# Production regime (Phase 4 gate + Phase 5 Peclet matching).
N = 48
RE = 25.0
U_LBM = 0.05
SCHMIDT = 2.0                 # Pe = Sc*Re = 50 (advection-dominated, scalar resolved at n=48)
CS = 0.0
MARGINS = dict(building_width_H=1.0, fetch_upstream_H=4.0, top_margin_H=5.0,
               outflow_H=8.0)
SPONGE_H = 4.0
AR_VALUES = [0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0]


def run_one(spec: tuple[float, int]) -> dict:
    ar, n = spec
    g = build_canyon(cells_per_H=n, aspect_ratio=ar, **MARGINS)
    nu = U_LBM * g.h / RE
    tau = lb.tau_from_viscosity(nu)
    tau_g = sc.tau_from_schmidt(nu, SCHMIDT)
    inlet = log_law_profile(g.ny, g.h, U_LBM, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, U_LBM, inlet, collision="bgk",
                           sponge_cells=int(SPONGE_H * g.h), tau_sponge=1.0,
                           with_scalar=True, tau_g=tau_g, source_strength=1.0)
    # ~8 flow-throughs; average over the last ~3 (flow steady => fill-independent metric).
    ft = int((4 + 1 + 1 + 1 + 8) * g.h / U_LBM)
    out = sim.run(max_iter=8 * ft, tol=1e-6, check_every=4000,
                  ramp_iters=8000, average_from=5 * ft)
    s = out["scalar"]
    fos = s["flux_over_source"]
    retention_eq = s["retention_mean_conc"] / fos if fos else float("nan")
    ach = s["opening_flux"] / s["canyon_content"] if s["canyon_content"] else float("nan")
    return {
        "aspect_ratio": ar, "cells_per_H": n, "w_cells": g.w,
        "retention_eq": retention_eq,
        "retention_raw": s["retention_mean_conc"],
        "ach_exchange_rate": ach,
        "ventilation_index": s["ventilation_index"],
        "cavity_circulation": out["cavity_circulation"],
        "flux_over_source": fos,
        "single_vortex": out["single_vortex"],
        "iters": out["iters"],
    }


def main() -> None:
    if len(sys.argv) >= 2:                      # single case: H/W [resolution]
        ar = float(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else N
        specs = [(ar, n)]
    else:                                        # full production sweep (CPU-parallel)
        # n=96 grid check at H/W=1 is run separately on the GPU (too slow on CPU).
        specs = [(ar, N) for ar in AR_VALUES]

    single = len(specs) == 1
    if single:
        # Run directly (no pool): a forked worker cannot initialise CUDA, so the
        # single-case path must stay in-process to allow the GPU backend.
        print(f"Running 1 case (Re={RE}, Pe={SCHMIDT*RE:.0f})...", flush=True)
        rows = [run_one(specs[0])]
    else:
        nproc = min(len(specs), mp.cpu_count())
        print(f"Running {len(specs)} cases on {nproc} processes "
              f"(Re={RE}, Pe={SCHMIDT*RE:.0f})...", flush=True)
        with mp.Pool(nproc) as pool:
            rows = pool.map(run_one, specs)
    rows.sort(key=lambda r: (r["cells_per_H"], r["aspect_ratio"]))

    if single:                                   # grid-check / one-off; don't touch the sweep CSV
        r = rows[0]
        io.save_result(ROOT / "results" /
                       f"sweep_single_AR{r['aspect_ratio']:g}_n{r['cells_per_H']}.json",
                       r, config={"Re": RE, "schmidt": SCHMIDT})
        print(f"H/W={r['aspect_ratio']} n={r['cells_per_H']}: "
              f"retention_eq={r['retention_eq']:.1f} ACH={r['ach_exchange_rate']:.3e} "
              f"circ={r['cavity_circulation']:+.3f} flux/src={r['flux_over_source']:.3f}")
        return

    out_csv = ROOT / "results" / "sweep_aspect_ratio.summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.save_result(ROOT / "results" / "sweep_aspect_ratio.json", {"cases": rows},
                   config={"Re": RE, "schmidt": SCHMIDT, "u_lbm": U_LBM, "n": N,
                           "collision": "bgk", **MARGINS})

    print(f"\n--- ventilation vs aspect ratio (steady laminar, Pe={SCHMIDT*RE:.0f}) ---")
    print(f"{'H/W':>5} {'w_cells':>7} {'retention_eq':>12} {'ACH':>10} "
          f"{'circ':>9} {'flux/src':>8}")
    for r in rows:
        print(f"{r['aspect_ratio']:>5} {r['w_cells']:>7} {r['retention_eq']:>12.1f} "
              f"{r['ach_exchange_rate']:>10.2e} {r['cavity_circulation']:>9.3f} "
              f"{r['flux_over_source']:>8.3f}")
    print(f"\nWrote {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
