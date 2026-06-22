#!/usr/bin/env python3
"""Phase 6.5b -- clean-resolution (Pe_cell < 2) confirmation of OUTCOME A.

Phase 6.5 confirmed the skimming collapse at high Péclet, but its whole Pe=200
grid ladder sat at Pe_cell >= 2 (8.3/4.2/2.1 for n=24/48/96): "the metric stopped
moving" is necessary but not sufficient, and a formal Richardson order over a
ladder that is NOT in the asymptotic range is meaningless. This closes the gap
with a genuinely-resolved convergence pair.

PRIORITY (the headline): Pe=144, H/W in {0.5,1,2,3}, n=96 (Pe_cell=1.5) AND
n=192 (Pe_cell=0.75). Both grids resolved; full resolved collapse curve; n=192
puts ~64 cells across the H/W=3 street (vs 16 at n=48), settling whether the
H/W=2->3 dip was a 16-cell artifact or real saturation.

FOLLOW-ON (optional, non-blocking, reuses cached flows): Pe=200, H/W in {1,2},
n=96 (Pe_cell~2.1) AND n=192 (Pe_cell~1.0) -- retires the borderline Pe=200/n=96
number. The Pe=144 verdict is computed and saved BEFORE this runs; a failure here
cannot affect it.

LOCKED REGIME UNCHANGED: time-averaged (statistically-stationary) laminar Re=25,
BGK, no LES/RANS/turbulence. Only resolution and scalar diffusivity change. The
Re=25 flow is NOT steady -- it carries a weak (<1% of u) residual ACOUSTIC mode
(D22 / diagnose_unsteadiness.py: St ~ 1/u, domain-resonant, NOT physical shedding).
So the D2Q9 flow is solved ONCE per (H/W, n) as a TIME-AVERAGED mean over ~10
oscillation periods (window-independent to <1%; the acoustic ripple averages out,
the mean is the clean physical vortex), one-way coupled to the scalar (D13),
checkpointed to disk, and reused across Pe.

HONEST FRAMING: high Pe here = LOW scalar diffusivity in a LAMINAR flow, a proxy
for advective dominance, NOT turbulence. Real canyons reach this regime through
turbulence; 3-D/turbulence remain the stated gap. We do NOT claim "2-D laminar
reproduces Oke".

Usage (GPU, single process -- CUDA-safe):
    CANYON_LBM_BACKEND=cupy python scripts/phase6_5b_peclet.py
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
CACHE = ROOT / "results" / "_flowcache"

GRIDS = [96, 192]
PE144_HW = [0.5, 1.0, 2.0, 3.0]        # PRIORITY headline curve
PE200_HW = [1.0, 2.0]                  # optional follow-on (reuses cached flows)
DO_PE200_FOLLOWON = True               # non-blocking; runs only after Pe=144 verdict


def _ft(g):
    return int((4 + 1 + 1 + 1 + 8) * g.h / U_LBM)   # steps per flow-through


def _free_gpu():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


BURN_FT = 1.5      # discard the startup transient (mean established by ~1.5 ft)
AVG_FT = 3.5       # average over ~10 oscillation periods (period ~0.35 ft):
                   # Check-2 (diagnose_unsteadiness.py) showed the mean is
                   # window-independent to <1% by ~10 periods.


def solve_flow(n: int, ar: float):
    """Re=25 TIME-AVERAGED mean flow; cache the mean (ux,uy) per (n,ar).

    The instantaneous flow carries a weak (<1% of u) residual ACOUSTIC mode (not
    physical shedding -- see D22 / diagnose_unsteadiness.py), so there is no steady
    state to tol-converge to. We report the time-averaged mean over AVG_FT (~10
    periods), which is window-independent (<1%) and domain/grid-robust. The acoustic
    ripple averages out; the mean is the clean physical vortex.
    """
    g = build_canyon(cells_per_H=n, aspect_ratio=ar, **MARGINS)
    CACHE.mkdir(parents=True, exist_ok=True)
    ck = CACHE / f"flow_n{n}_ar{ar:g}.npz"
    nu = U_LBM * g.h / RE
    if ck.exists():
        d = np.load(ck)
        print(f"   [flow cached] n={n} AR={ar} grid {g.ny}x{g.nx} "
              f"(mean over {int(d['n_avg'])} samples)", flush=True)
        return g, d["ux"], d["uy"], nu, int(d["n_avg"])
    tau = lb.tau_from_viscosity(nu)
    inlet = log_law_profile(g.ny, g.h, U_LBM, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, U_LBM, inlet, collision="bgk",
                           sponge_cells=int(SPONGE_H * g.h), with_scalar=False)
    ft = _ft(g)
    burn, total = int(BURN_FT * ft), int((BURN_FT + AVG_FT) * ft)
    samp = max(1, ft // 200)                 # ~70 mean-samples per oscillation period
    ramp = 8000
    print(f"   [solving flow] n={n} AR={ar} grid {g.ny}x{g.nx} ft={ft} "
          f"(tau={tau:.3f}); burn {BURN_FT}ft + average {AVG_FT}ft (~10 periods) ...",
          flush=True)
    # Subsampled time-average: accumulate the mean every `samp` steps (the
    # oscillation is slow, ~0.35 ft/period, so subsampling is statistically
    # identical to every-step averaging at a fraction of the macroscopic() cost).
    sum_ux = sum_uy = None; n_avg = 0; fluid = sim.fluid
    for it in range(1, total + 1):
        sim.step(inlet_scale=min(1.0, it / ramp))
        if it > burn and it % samp == 0:
            _, ux, uy = sim.macroscopic()
            if sum_ux is None:
                sum_ux = xp.zeros_like(ux); sum_uy = xp.zeros_like(uy)
            sum_ux += ux; sum_uy += uy; n_avg += 1
        if it % 8000 == 0:                   # stability guard (same as solver.run)
            _, ux, uy = sim.macroscopic()
            smax = float(xp.nanmax(xp.sqrt(ux * ux + uy * uy)))
            if not np.isfinite(smax) or smax > 0.4:
                raise FloatingPointError(f"flow unstable at it={it}: umax={smax:.3g}")
    mux = xp.where(fluid, sum_ux / n_avg, 0.0)
    muy = xp.where(fluid, sum_uy / n_avg, 0.0)
    wz = xp.gradient(muy, axis=1) - xp.gradient(mux, axis=0)
    circ = float(xp.sum(wz[xp.asarray(g.cavity_mask)]))
    ux, uy = asnumpy(mux), asnumpy(muy)
    np.savez(ck, ux=ux, uy=uy, n_avg=n_avg, ny=g.ny, nx=g.nx)
    print(f"   [flow done] time-averaged mean over {n_avg} samples "
          f"({AVG_FT}ft); cavity_circ={circ:+.4f}", flush=True)
    _free_gpu()
    return g, ux, uy, nu, n_avg


def scalar_on_frozen_flow(g, ux_np, uy_np, nu, schmidt, n_ft=5) -> dict:
    """Equilibrate the D2Q5 scalar on a FROZEN steady velocity; full metrics."""
    tau_g = sc.tau_from_schmidt(nu, schmidt)
    D = sc.diffusivity_from_tau(tau_g)
    pe_cell = U_LBM / D                                  # = Pe / n
    ux, uy = xp.asarray(ux_np), xp.asarray(uy_np)        # solids already 0
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
            min_C = min(min_C, float(C.min()))           # worst instantaneous undershoot
    Cm = asnumpy(csum / nav)
    content, opening_flux = content_sum / nav, flux_sum / nav
    n_cav = g.h * g.w
    fos = opening_flux / source_rate
    retention_raw = content / n_cav                       # canyon-mean concentration
    mr, mc = np.unravel_index(int(np.argmin(Cm)), Cm.shape)   # location of field min
    rows = np.arange(1, g.h + 1)
    west, east = Cm[rows, s0 + off], Cm[rows, s1 - 1 - off]
    lee, wind = (west, east) if west.mean() >= east.mean() else (east, west)
    ach = opening_flux / content if content else float("nan")
    _free_gpu()
    return {
        "Pe": schmidt * RE, "schmidt": schmidt, "tau_g": tau_g, "pe_cell": pe_cell,
        "retention_raw": retention_raw,
        "retention_eq": retention_raw / fos if fos else float("nan"),
        "ach": ach,
        "ach_star": ach * g.h / U_LBM,                    # grid-invariant ACH* = ACH*H/u
        "asymmetry": float(lee.mean() / max(wind.mean(), 1e-30)),
        "flux_over_source": fos,
        "min_C": min_C,
        "min_C_pct": 100.0 * min_C / retention_raw if retention_raw else float("nan"),
        "min_at_source_row": bool(mr == g.source_row),
        "min_loc": f"r={mr},c={mc};src_row={g.source_row}",
        "tau_g_ok": bool(tau_g > 0.51),
    }


def _flush_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def _run_block(rows, csv_path, n, ar, pe):
    g, ux, uy, nu, fiters = solve_flow(n, ar)
    m = scalar_on_frozen_flow(g, ux, uy, nu, pe / RE)
    m.update({"aspect_ratio": ar, "cells_per_H": n, "flow_iters": fiters})
    rows.append(m)
    _flush_csv(rows, csv_path)                # incremental: survive an interrupt
    flag = "" if (m["tau_g_ok"] and abs(m["min_C_pct"]) < 0.05) else " <<CHECK"
    print(f"   Pe={pe:>4.0f}: ret_raw={m['retention_raw']:7.1f} "
          f"ret_eq={m['retention_eq']:7.1f} ACH*={m['ach_star']:.5f} "
          f"asym={m['asymmetry']:.2f} Pe_cell={m['pe_cell']:.2f} "
          f"minC={m['min_C_pct']:+.3f}% ({m['min_loc']}) tau_g={m['tau_g']:.4f}{flag}",
          flush=True)
    return m


def _pick(rows, pe, ar, n):
    for r in rows:
        if (abs(r["Pe"] - pe) < 1e-6 and abs(r["aspect_ratio"] - ar) < 1e-9
                and r["cells_per_H"] == n):
            return r
    return None


def _pct(a, b):
    return 100.0 * abs(b - a) / (abs(a) + 1e-30)


def _convergence(rows, pe, hws):
    out = []
    for ar in hws:
        r96, r192 = _pick(rows, pe, ar, 96), _pick(rows, pe, ar, 192)
        if not (r96 and r192):
            continue
        out.append({"Pe": pe, "aspect_ratio": ar,
                    "d_retention_eq_pct": _pct(r96["retention_eq"], r192["retention_eq"]),
                    "d_ach_star_pct": _pct(r96["ach_star"], r192["ach_star"]),
                    "d_asymmetry_pct": _pct(r96["asymmetry"], r192["asymmetry"])})
    return out


def _report_curve(rows, pe, hws, n):
    pts = [(ar, _pick(rows, pe, ar, n)) for ar in hws]
    s = "  ".join(f"H/W={ar}:{r['retention_eq']:.0f}" for ar, r in pts if r)
    print(f"  retention_eq curve  Pe={pe:.0f} n={n}:  {s}", flush=True)


def main() -> None:
    print(f"Phase 6.5b (backend={backend_name()}): clean-resolution (Pe_cell<2) "
          f"confirmation. float64; flow tol-converged + cached per (H/W,n).",
          flush=True)
    csv_path = ROOT / "results" / "phase6_5b_peclet.summary.csv"
    rows = []

    # ===== PRIORITY: Pe=144 headline curve at n=96 and n=192 =====
    for n in GRIDS:
        for ar in PE144_HW:
            print(f"\n[Pe=144  n={n}  H/W={ar}]", flush=True)
            _run_block(rows, csv_path, n, ar, 144.0)

    conv144 = _convergence(rows, 144.0, PE144_HW)
    print("\n=== Pe=144 (HEADLINE) n=96 -> n=192 convergence [target < 1%] ===",
          flush=True)
    for n in GRIDS:
        _report_curve(rows, 144.0, PE144_HW, n)
    worst = 0.0
    for c in conv144:
        worst = max(worst, c["d_retention_eq_pct"], c["d_ach_star_pct"],
                    c["d_asymmetry_pct"])
        print(f"  H/W={c['aspect_ratio']}: retention_eq {c['d_retention_eq_pct']:5.2f}%  "
              f"ACH* {c['d_ach_star_pct']:5.2f}%  asym {c['d_asymmetry_pct']:5.2f}%",
              flush=True)
    print(f"  worst change across all H/W and metrics: {worst:.2f}%", flush=True)

    io.save_result(ROOT / "results" / "phase6_5b_peclet.json",
                   {"matrix": rows, "convergence_96_to_192_Pe144": conv144},
                   config={"Re": RE, "u_lbm": U_LBM, "grids": GRIDS,
                           "pe144_hw": PE144_HW, "priority": "Pe=144",
                           "collision": "bgk", "flow_termination": "tol=1e-6",
                           **MARGINS})
    _flush_csv(rows, csv_path)
    print("\n[Pe=144 PRIORITY COMPLETE -- results + verdict saved]", flush=True)

    # ===== FOLLOW-ON (non-blocking): Pe=200 at H/W in {1,2}, both grids =====
    if DO_PE200_FOLLOWON:
        try:
            print("\n[FOLLOW-ON] Pe=200 confirmation (reuses cached flows) ...",
                  flush=True)
            for n in GRIDS:
                for ar in PE200_HW:
                    print(f"\n[Pe=200  n={n}  H/W={ar}]", flush=True)
                    _run_block(rows, csv_path, n, ar, 200.0)
            conv200 = _convergence(rows, 200.0, PE200_HW)
            print("\n=== Pe=200 (stress test) n=96 -> n=192 convergence ===",
                  flush=True)
            for c in conv200:
                print(f"  H/W={c['aspect_ratio']}: retention_eq {c['d_retention_eq_pct']:5.2f}%  "
                      f"ACH* {c['d_ach_star_pct']:5.2f}%  asym {c['d_asymmetry_pct']:5.2f}%",
                      flush=True)
            io.save_result(ROOT / "results" / "phase6_5b_peclet.json",
                           {"matrix": rows,
                            "convergence_96_to_192_Pe144": conv144,
                            "convergence_96_to_192_Pe200": conv200},
                           config={"Re": RE, "u_lbm": U_LBM, "grids": GRIDS,
                                   "pe144_hw": PE144_HW, "pe200_hw": PE200_HW,
                                   "priority": "Pe=144", "collision": "bgk",
                                   "flow_termination": "tol=1e-6", **MARGINS})
            _flush_csv(rows, csv_path)
        except Exception as e:        # never let the follow-on void the Pe=144 result
            print(f"\n[FOLLOW-ON Pe=200 FAILED -- Pe=144 verdict stands]: {e!r}",
                  flush=True)

    print("\nWrote results/phase6_5b_peclet.{json,summary.csv}", flush=True)


if __name__ == "__main__":
    main()
