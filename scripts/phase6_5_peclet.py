#!/usr/bin/env python3
"""Phase 6.5 -- high-Peclet, grid-checked H/W diagnostic.

Phase 6 swept at Pe=50, below the advection-dominated regime (asymmetry only
reaches ~1.2-1.5 by Pe=72-144). This tests the advection-dominated regime
properly: does ventilation worsen monotonically with H/W (the skimming collapse)
once advection dominates, and is that result grid-converged?

LOCKED PLAN (D6/D18/D20): steady-laminar Re=25, BGK, NO LES/RANS/turbulence.
Only the SCALAR diffusivity changes (Pe via Schmidt number). The D2Q9 flow is
one-way coupled (D13) and steady, so it is solved ONCE per H/W and reused for
every Pe -- only the cheap D2Q5 scalar re-solves.

Numerical hygiene: reports cell-Peclet Pe_cell = u_lbm*dx/D = Pe/n; flags any run
with tau_g < 0.51 or negative concentrations as under-resolved (not a result).

Usage:  CANYON_LBM_BACKEND=cupy python scripts/phase6_5_peclet.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from canyon_lbm import io, lattice as lb, scalar as sc
from canyon_lbm.backend import asnumpy, backend_name, xp
from canyon_lbm.boundary import log_law_profile
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation

ROOT = Path(__file__).resolve().parents[1]
RE = 25.0
U_LBM = 0.05
MARGINS = dict(building_width_H=1.0, fetch_upstream_H=4.0, top_margin_H=5.0,
               outflow_H=8.0)
SPONGE_H = 4.0
HW_SUBSET = [0.5, 1.0, 2.0, 3.0]
PE_LADDER = [50.0, 72.0, 144.0, 200.0]          # Sc = Pe / Re
GRID_LADDER = [24, 48, 96]                       # for the grid check at the top Pe


def _ft(g):
    return int((4 + 1 + 1 + 1 + 8) * g.h / U_LBM)   # steps per flow-through


def solve_flow(n: int, ar: float):
    """Solve the steady Re=25 flow once; return geometry + frozen (ux,uy), nu."""
    g = build_canyon(cells_per_H=n, aspect_ratio=ar, **MARGINS)
    nu = U_LBM * g.h / RE
    tau = lb.tau_from_viscosity(nu)
    inlet = log_law_profile(g.ny, g.h, U_LBM, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, U_LBM, inlet, collision="bgk",
                           sponge_cells=int(SPONGE_H * g.h), with_scalar=False)
    ft = _ft(g)
    out = sim.run(max_iter=5 * ft, tol=1e-6, check_every=4000,
                  ramp_iters=8000, average_from=3 * ft)
    f = out["fields"]
    return g, np.asarray(f["ux"]), np.asarray(f["uy"]), nu


def scalar_on_frozen_flow(g, ux_np, uy_np, nu, schmidt, n_ft=5) -> dict:
    """Equilibrate the D2Q5 scalar on a FROZEN steady velocity field; metrics."""
    tau_g = sc.tau_from_schmidt(nu, schmidt)
    D = sc.diffusivity_from_tau(tau_g)
    pe_cell = U_LBM / D                                  # = Pe / n
    ux, uy = xp.asarray(ux_np), xp.asarray(uy_np)        # frozen (solids already 0)
    solid = xp.asarray(g.solid)
    cav = xp.asarray(g.cavity_mask)
    masks = sc.precompute_bounceback_scalar(solid)
    ny, nx = g.ny, g.nx
    s0, s1 = g.street
    source = xp.zeros((ny, nx)); source[g.source_row, s0:s1] = 1.0
    source_rate = float(s1 - s0)
    G = sc.equilibrium_scalar(xp.zeros((ny, nx)), ux, uy)

    ft = _ft(g); max_iter = n_ft * ft; avg_from = int(0.5 * max_iter)
    orow, off = g.roof_row + 1, max(1, int(round((1 / 24) * g.h)))
    csum = None; content_sum = flux_sum = 0.0; nav = 0; min_C = 1e9
    for it in range(1, max_iter + 1):
        C = sc.scalar_concentration(G)
        Gp = sc.collide_scalar(G, sc.equilibrium_scalar(C, ux, uy), tau_g, source)
        G = sc.stream_scalar(Gp)
        sc.apply_bounceback_scalar(G, Gp, masks)
        sc.inlet_zero_concentration(G); sc.open_outlet(G); sc.open_top(G)
        if it >= avg_from:
            C = sc.scalar_concentration(G)
            csum = C.copy() if csum is None else csum + C
            content_sum += float(C[cav].sum())
            adv = C[orow, s0:s1] * uy[orow, s0:s1]
            diff = -D * (C[orow, s0:s1] - C[orow - 1, s0:s1])
            flux_sum += float(xp.sum(adv + diff)); nav += 1
            min_C = min(min_C, float(C.min()))
    Cm = asnumpy(csum / nav)
    content, opening_flux = content_sum / nav, flux_sum / nav
    n_cav = g.h * g.w
    fos = opening_flux / source_rate
    rows = np.arange(1, g.h + 1)
    west, east = Cm[rows, s0 + off], Cm[rows, s1 - 1 - off]
    lee, wind = (west, east) if west.mean() >= east.mean() else (east, west)
    return {
        "Pe": schmidt * RE, "schmidt": schmidt, "tau_g": tau_g,
        "pe_cell": pe_cell,
        "retention_eq": (content / n_cav) / fos if fos else float("nan"),
        "ach": opening_flux / content if content else float("nan"),
        "asymmetry": float(lee.mean() / max(wind.mean(), 1e-30)),
        "flux_over_source": fos, "min_C": min_C,
        "under_resolved": bool(min_C < -1e-6 or tau_g < 0.51),
    }


def main() -> None:
    print(f"Phase 6.5 (backend={backend_name()}): flow solved once per H/W, "
          f"reused across Pe ladder.", flush=True)

    # --- Part A: H/W x Pe matrix at n=48 (flow reused per H/W) ---
    rows = []
    for ar in HW_SUBSET:
        g, ux, uy, nu = solve_flow(48, ar)
        print(f"\n[H/W={ar}] flow solved (n=48, grid {g.ny}x{g.nx}); scalar ladder:",
              flush=True)
        for pe in PE_LADDER:
            m = scalar_on_frozen_flow(g, ux, uy, nu, pe / RE)
            m.update({"aspect_ratio": ar, "cells_per_H": 48})
            rows.append(m)
            print(f"   Pe={pe:>4.0f}: retention_eq={m['retention_eq']:7.1f} "
                  f"ACH={m['ach']:.3e} asym={m['asymmetry']:.2f} "
                  f"Pe_cell={m['pe_cell']:.1f} flux/src={m['flux_over_source']:.2f} "
                  f"{'UNDER-RESOLVED' if m['under_resolved'] else ''}", flush=True)

    # --- Part B: grid check at the top Pe (200), H/W=1 ---
    print(f"\n[GRID CHECK @ Pe={PE_LADDER[-1]:.0f}, H/W=1]:", flush=True)
    grid = []
    for n in GRID_LADDER:
        g, ux, uy, nu = solve_flow(n, 1.0)
        m = scalar_on_frozen_flow(g, ux, uy, nu, PE_LADDER[-1] / RE)
        m.update({"cells_per_H": n})
        grid.append(m)
        print(f"   n={n:>3}: retention_eq={m['retention_eq']:7.1f} ACH={m['ach']:.3e} "
              f"asym={m['asymmetry']:.2f} Pe_cell={m['pe_cell']:.2f} "
              f"{'UNDER-RESOLVED' if m['under_resolved'] else ''}", flush=True)

    # convergence between the two finest grids + observed (Richardson) order
    def conv(key):
        a, b, c = (grid[0][key], grid[1][key], grid[2][key])   # n=24,48,96
        rel = 100 * abs(c - b) / (abs(b) + 1e-30)
        denom = (b - a)
        order = (np.log(abs(denom) / (abs(c - b) + 1e-30)) / np.log(2.0)
                 if abs(c - b) > 1e-30 and abs(denom) > 1e-30 else float("nan"))
        return rel, order
    print("\n  convergence (two finest, 48->96):", flush=True)
    grid_ok = {}
    for key in ("retention_eq", "ach", "asymmetry"):
        rel, order = conv(key)
        grid_ok[key] = rel < 3.0
        print(f"    {key:13s}: {rel:5.1f}% change, observed order p={order:+.2f} "
              f"[{'CONVERGED' if rel < 3 else 'NOT'}]", flush=True)

    io.save_result(ROOT / "results" / "phase6_5_peclet.json",
                   {"matrix": rows, "grid_check": grid},
                   config={"Re": RE, "u_lbm": U_LBM, "n_matrix": 48,
                           "pe_ladder": PE_LADDER, "hw_subset": HW_SUBSET,
                           "collision": "bgk", **MARGINS})
    with open(ROOT / "results" / "phase6_5_peclet.summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    print("\nWrote results/phase6_5_peclet.{json,summary.csv}", flush=True)


if __name__ == "__main__":
    main()
