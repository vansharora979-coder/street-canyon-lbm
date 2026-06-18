#!/usr/bin/env python3
"""Headline result: the leeward/windward asymmetry is advection-controlled.

Runs the H/W=1 canyon in the locked steady-laminar regime (Re=25, BGK, 48
cells/H) at several Schmidt numbers, i.e. several Peclet numbers Pe=Sc*Re, and
measures the leeward/windward wall-c+ asymmetry. It rises from ~1 (diffusion-
dominated, symmetric) toward the CODASC turbulent value (~3) as advection takes
over -- demonstrating that the canyon's signature asymmetry (and, by the same
mechanism, the skimming trapping) is a Peclet/advection effect, NOT reproduced
when diffusion dominates.

Note: the scalar boundary layer thins as ~H/sqrt(Pe); at 48 cells/H it is well
resolved up to Pe~70-90 and only marginal above, so the high-Pe points are
illustrative of the TREND, not grid-converged absolute values.

Run on the GPU sequentially (single process -- CUDA-safe):
    CANYON_LBM_BACKEND=cupy python scripts/pe_sensitivity.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from canyon_lbm import io, lattice as lb, scalar as sc
from canyon_lbm.backend import backend_name
from canyon_lbm.boundary import log_law_profile
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation

ROOT = Path(__file__).resolve().parents[1]

N = 48
RE = 25.0
U_LBM = 0.05
SCHMIDT_VALUES = [0.72, 1.44, 2.88, 5.76]      # Pe = Sc*Re = 18, 36, 72, 144
MARGINS = dict(building_width_H=1.0, fetch_upstream_H=4.0, top_margin_H=5.0,
               outflow_H=8.0)


def run_pe(schmidt: float) -> dict:
    g = build_canyon(cells_per_H=N, aspect_ratio=1.0, **MARGINS)
    nu = U_LBM * g.h / RE
    tau = lb.tau_from_viscosity(nu)
    tau_g = sc.tau_from_schmidt(nu, schmidt)
    inlet = log_law_profile(g.ny, g.h, U_LBM, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, U_LBM, inlet, collision="bgk",
                           sponge_cells=4 * g.h, tau_sponge=1.0,
                           with_scalar=True, tau_g=tau_g, source_strength=1.0)
    ft = int((4 + 1 + 1 + 1 + 8) * g.h / U_LBM)
    print(f"Pe={schmidt*RE:.0f} (Sc={schmidt}) tau_g={tau_g:.4f} backend={backend_name()}...",
          flush=True)
    out = sim.run(max_iter=6 * ft, tol=1e-6, check_every=4000,
                  ramp_iters=8000, average_from=4 * ft)
    C = out["mean_C"]
    h, (s0, s1) = g.h, g.street
    off = max(1, int(round((1 / 24) * h)))
    rows = np.arange(1, h + 1)
    west = C[rows, s0 + off]
    east = C[rows, s1 - 1 - off]
    lee, wind = (west, east) if west.mean() >= east.mean() else (east, west)
    asym = float(lee.mean() / max(wind.mean(), 1e-30))
    decay = float(lee[0] / max(lee[-1], 1e-30))
    print(f"   -> asymmetry={asym:.2f}  decay={decay:.2f}  "
          f"flux/src={out['scalar']['flux_over_source']:.2f}", flush=True)
    return {"Pe": schmidt * RE, "schmidt": schmidt, "asymmetry": asym,
            "decay_street_roof": decay,
            "flux_over_source": out["scalar"]["flux_over_source"]}


def main() -> None:
    rows = [run_pe(s) for s in SCHMIDT_VALUES]
    out_csv = ROOT / "results" / "pe_sensitivity.summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.save_result(ROOT / "results" / "pe_sensitivity.json", {"cases": rows},
                   config={"Re": RE, "n": N, "aspect_ratio": 1.0,
                           "collision": "bgk", **MARGINS},
                   extra={"codasc_reference_asymmetry": 2.98})
    print("\n--- leeward/windward asymmetry vs Peclet (H/W=1, laminar Re=25) ---")
    for r in rows:
        print(f"  Pe={r['Pe']:>4.0f}: asymmetry={r['asymmetry']:.2f}")
    print(f"Wrote {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
