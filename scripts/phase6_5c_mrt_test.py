#!/usr/bin/env python3
"""Phase 6.5c TEST -- does a TRT magic-parameter (Lambda=3/16) flow fix the
Pe=144 grid non-convergence at H/W=0.5?

DIAGNOSIS (D22 Part B -> D23). Pe=144 retention_eq/ACH* changed 16-19% from
n=96 to n=192 at H/W=0.5, while the *scalar* hygiene was clean (Pe_cell<2,
tau_g>0.5, minC~0). So the non-convergence is in the FLOW, not the scalar:

  The BGK bounce-back wall sits at a tau-DEPENDENT location. In TRT language the
  controlling magic parameter is Lambda = (1/s_nu - 1/2)(1/s_q - 1/2); for BGK
  every rate equals 1/tau, so Lambda = (tau - 1/2)^2. To hold Re=25 fixed, tau
  must change with the grid (1.076 at n=96, 1.652 at n=192), so Lambda jumps
  0.33 -> 1.33 and the two grids solve subtly DIFFERENT effective geometries.
  Symptom: the time-averaged cavity circulation scales as 1.61x from n=96->192
  at H/W=0.5 (pure grid scaling would give 2.0x), narrowing to 1.94x at H/W=3.

FIX (root cause). Re-solve the FLOW as pure TRT with the bounce-back-consistent
magic parameter Lambda = 3/16, which pins the wall at the mid-link INDEPENDENT
of tau. Implemented through the existing MRT routine (Phase 4a): all even
non-conserved moments relax at s_nu = 1/tau (so the viscosity, hence Re=25, is
UNCHANGED) and the odd q-moments at s_q chosen so Lambda = 3/16. NO LES; the
laminar Re=25 regime is otherwise untouched.

This TEST does ONLY H/W=0.5 (the worst case) at n=96 and n=192. GO criterion:
the circulation ratio moves toward ~2.0 AND retention_eq/ACH*/asymmetry converge
to <~3% (n=96->192). YES -> commit to the full four-H/W re-run; NO -> MRT is not
the cure, stop and report (honest fallback).

Usage:  CANYON_LBM_BACKEND=cupy python scripts/phase6_5c_mrt_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Reuse the EXACT Phase 6.5b scalar solver + constants so the only thing that
# changes vs the BGK baseline is the flow collision (clean apples-to-apples).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6_5b_peclet import (  # noqa: E402
    AVG_FT, BURN_FT, MARGINS, RE, SPONGE_H, U_LBM, _ft, _pct,
    scalar_on_frozen_flow,
)

from canyon_lbm import lattice as lb  # noqa: E402
from canyon_lbm.backend import asnumpy, backend_name, xp  # noqa: E402
from canyon_lbm.boundary import log_law_profile  # noqa: E402
from canyon_lbm.geometry import build_canyon  # noqa: E402
from canyon_lbm.solver import CanyonSimulation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "_flowcache"
LAMBDA = 3.0 / 16.0          # TRT magic parameter -> tau-independent mid-link wall
HW = 0.5                     # worst-case aspect ratio (BGK ratio 1.61, d 16-19%)
PE = 144.0                   # the headline Peclet

# BGK baseline (Phase 6.5b, H/W=0.5) for side-by-side reporting.
BGK = {"circ_ratio": 1.614, "d_retention_eq": 16.03, "d_ach_star": 19.09,
       "d_asym": 6.18, "circ96": -1.4025, "circ192": -2.2633}


def magic_s_q(tau_field):
    """Odd-moment rate enforcing Lambda=(1/s_nu-1/2)(1/s_q-1/2)=3/16 with
    s_nu=1/tau. Pins the bounce-back wall at the mid-link independent of tau."""
    return 1.0 / (0.5 + LAMBDA / (tau_field - 0.5))


def _free_pool():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _period_zc(sig, samp, burn_samples):
    """Dominant period (steps) via zero-crossings of the post-burn probe signal."""
    x = sig[burn_samples:]
    if x.size < 4:
        return float("nan")
    x = x - x.mean()
    nz = int(np.sum(np.abs(np.diff(np.sign(x))) > 0))
    if nz < 2:
        return float("nan")
    span = (x.size - 1) * samp
    return 2.0 * span / nz


def solve_flow_trt(n: int, ar: float):
    """Re=25 TIME-AVERAGED mean flow, TRT(Lambda=3/16). Cached separately from BGK.

    Same averaging recipe as Phase 6.5b (BURN_FT + AVG_FT ~ 10 acoustic periods);
    a canyon-centre uy probe measures the period so we can confirm the window
    really spans >=5-10 periods at BOTH grids (averaging is not a confound).
    """
    g = build_canyon(cells_per_H=n, aspect_ratio=ar, **MARGINS)
    CACHE.mkdir(parents=True, exist_ok=True)
    ck = CACHE / f"flow_mrt_n{n}_ar{ar:g}.npz"
    nu = U_LBM * g.h / RE
    if ck.exists():
        d = np.load(ck)
        print(f"   [flow cached MRT] n={n} AR={ar} (mean over {int(d['n_avg'])} "
              f"samples) circ={float(d['circ']):+.4f} T~{float(d['T_meas']):.0f}",
              flush=True)
        return (g, d["ux"], d["uy"], nu, int(d["n_avg"]),
                float(d["circ"]), float(d["T_meas"]))
    tau = lb.tau_from_viscosity(nu)
    inlet = log_law_profile(g.ny, g.h, U_LBM, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, U_LBM, inlet, collision="mrt",
                           sponge_cells=int(SPONGE_H * g.h), with_scalar=False)
    # ---- pure TRT with the magic parameter (override the MRT ghost rates) ----
    # even non-conserved moments (e, eps, pxx, pxy) relax at s_nu = 1/tau_field,
    # so viscosity -> Re=25 is preserved EXACTLY; only the odd q-rate moves, to
    # set Lambda = 3/16 (tau-independent wall). s_nu itself is applied in _collide.
    s_nu_field = 1.0 / sim.tau_field
    sim.s_e = s_nu_field
    sim.s_eps = s_nu_field
    sim.s_q = magic_s_q(sim.tau_field)

    ft = _ft(g)
    burn, total = int(BURN_FT * ft), int((BURN_FT + AVG_FT) * ft)
    samp = max(1, ft // 200)
    ramp = 8000
    fluid = sim.fluid
    h = g.h
    s0, s1 = g.street
    pr, pc = h // 2, (s0 + s1) // 2          # canyon-centre probe (uy)
    print(f"   [solving flow MRT-TRT] n={n} AR={ar} grid {g.ny}x{g.nx} ft={ft} "
          f"(tau={tau:.3f}, Lambda=3/16); burn {BURN_FT}ft + average {AVG_FT}ft ...",
          flush=True)
    sum_ux = sum_uy = None
    n_avg = 0
    probe = []                                # host floats (NO device views!)
    for it in range(1, total + 1):
        sim.step(inlet_scale=min(1.0, it / ramp))
        if it % samp == 0:
            _, ux, uy = sim.macroscopic()
            # Read ONE interior cell to host. Storing the 0-d device view
            # (uy[pr,pc]) instead would pin its 28 MB parent array alive per
            # sample -> ~22 GB leak -> OOM at n=192. float() detaches it.
            probe.append(float(uy[pr, pc]))
            if it > burn:
                if sum_ux is None:
                    sum_ux = xp.zeros_like(ux)
                    sum_uy = xp.zeros_like(uy)
                sum_ux += ux
                sum_uy += uy
                n_avg += 1
        if it % 8000 == 0:
            _, ux, uy = sim.macroscopic()
            smax = float(xp.nanmax(xp.sqrt(ux * ux + uy * uy)))
            if not np.isfinite(smax) or smax > 0.4:
                raise FloatingPointError(f"flow unstable it={it} umax={smax:.3g}")
            _free_pool()                      # defrag the CuPy pool (insurance)
    mux = xp.where(fluid, sum_ux / n_avg, 0.0)
    muy = xp.where(fluid, sum_uy / n_avg, 0.0)
    wz = xp.gradient(muy, axis=1) - xp.gradient(mux, axis=0)
    circ = float(xp.sum(wz[xp.asarray(g.cavity_mask)]))
    T_meas = _period_zc(np.array(probe), samp, burn // samp)
    ux, uy = asnumpy(mux), asnumpy(muy)
    np.savez(ck, ux=ux, uy=uy, n_avg=n_avg, circ=circ, T_meas=T_meas,
             ny=g.ny, nx=g.nx)
    nper = (AVG_FT * ft / T_meas) if T_meas == T_meas else float("nan")
    print(f"   [flow done MRT-TRT] mean over {n_avg} samples; "
          f"cavity_circ={circ:+.4f}; T~{T_meas:.0f} steps -> {nper:.1f} periods "
          f"averaged", flush=True)
    return g, ux, uy, nu, n_avg, circ, T_meas


def main() -> None:
    print(f"Phase 6.5c TEST (backend={backend_name()}): TRT magic Lambda=3/16 "
          f"flow; H/W={HW}, n=96 & n=192; Pe={PE:.0f}.", flush=True)
    rows = []
    for n in (96, 192):
        print(f"\n[MRT-TRT  n={n}  H/W={HW}]", flush=True)
        g, ux, uy, nu, navg, circ, T = solve_flow_trt(n, HW)
        m = scalar_on_frozen_flow(g, ux, uy, nu, PE / RE)
        m.update({"aspect_ratio": HW, "cells_per_H": n, "circ": circ,
                  "n_avg_flow": navg, "T_meas": T,
                  "periods_avg": (AVG_FT * _ft(g) / T) if T == T else float("nan")})
        rows.append(m)
        print(f"   Pe={PE:.0f}: ret_raw={m['retention_raw']:.1f} "
              f"ret_eq={m['retention_eq']:.1f} ACH*={m['ach_star']:.5f} "
              f"asym={m['asymmetry']:.2f} Pe_cell={m['pe_cell']:.2f} "
              f"minC={m['min_C_pct']:+.3f}% tau_g={m['tau_g']:.4f}", flush=True)

    r96, r192 = rows
    ratio = abs(r192["circ"]) / max(abs(r96["circ"]), 1e-30)
    d_ret = _pct(r96["retention_eq"], r192["retention_eq"])
    d_ach = _pct(r96["ach_star"], r192["ach_star"])
    d_asym = _pct(r96["asymmetry"], r192["asymmetry"])
    worst = max(d_ret, d_ach, d_asym)
    go = (ratio > 1.85) and (worst < 3.0)

    print(f"\n=== Phase 6.5c TEST verdict (H/W={HW}, Pe={PE:.0f}) ===", flush=True)
    print(f"  {'metric':14s} {'n=96':>10} {'n=192':>10} {'change':>9}   (BGK was)",
          flush=True)
    print(f"  {'circ ratio':14s} {r96['circ']:>10.4f} {r192['circ']:>10.4f} "
          f"{ratio:>8.3f}x   (1.614x; ideal 2.0)", flush=True)
    print(f"  {'retention_eq':14s} {r96['retention_eq']:>10.1f} "
          f"{r192['retention_eq']:>10.1f} {d_ret:>8.2f}%   ({BGK['d_retention_eq']}%)",
          flush=True)
    print(f"  {'ACH*':14s} {r96['ach_star']:>10.5f} {r192['ach_star']:>10.5f} "
          f"{d_ach:>8.2f}%   ({BGK['d_ach_star']}%)", flush=True)
    print(f"  {'asymmetry':14s} {r96['asymmetry']:>10.3f} {r192['asymmetry']:>10.3f} "
          f"{d_asym:>8.2f}%   ({BGK['d_asym']}%)", flush=True)
    print(f"  periods averaged: n96={r96['periods_avg']:.1f}  "
          f"n192={r192['periods_avg']:.1f}  (need >=5-10; averaging not a confound)",
          flush=True)
    print(f"\n  worst metric change {worst:.2f}%  |  circ ratio {ratio:.3f}x",
          flush=True)
    verdict = ("GO -- magic parameter restores flow grid-convergence; commit to "
               "the full four-H/W re-run" if go else
               "NO-GO -- MRT/TRT magic is not the cure; STOP and report (fallback)")
    print(f"  VERDICT: {verdict}", flush=True)

    out = ROOT / "results" / "phase6_5c_mrt_test.json"
    with open(out, "w") as fh:
        json.dump({"aspect_ratio": HW, "Pe": PE, "Lambda": LAMBDA, "rows": rows,
                   "circ_ratio": ratio, "d_retention_eq_pct": d_ret,
                   "d_ach_star_pct": d_ach, "d_asym_pct": d_asym,
                   "worst_change_pct": worst, "go": bool(go), "bgk_baseline": BGK},
                  fh, indent=2, default=lambda o: float(o))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
