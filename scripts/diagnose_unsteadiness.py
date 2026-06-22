#!/usr/bin/env python3
"""Diagnose the Re=25 canyon-flow oscillation: physical shedding vs numerical/BC mode.

The instantaneous canyon flow at Re=25 oscillates (it does NOT reach a steady
state) -- contradicting the project's own ~Re47 shedding-onset estimate (D11) and
its history of an acoustic BC mode (D10). St~=0.19 alone does not establish physical
shedding. This runs the discriminators:

  CHECK 1 (physical vs numerical):
   (a) sponge strength/length + downstream fetch -- if amplitude/freq move
       materially with BC treatment -> BC-driven.
   (b) canyon-interior probe vs downstream-wake probe -- is the canyon itself
       unsteady, or is the wake bleeding into the cavity metric?
   (c) Mach (u_lbm) at fixed Re=25 -- acoustic period is ~constant in steps
       (St ~ 1/u); convective shedding St is ~u-independent.
   (d) grid robustness of St -- physical St is ~grid-independent.

  CHECK 2 (re-justify Re=25): the mean must be averaging-WINDOW-independent --
   average over increasing numbers of periods and show the mean stops moving.

Diagnostic only (informs the DECISIONS entry); not part of `make reproduce`.
Usage:  CANYON_LBM_BACKEND=cupy python scripts/diagnose_unsteadiness.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from canyon_lbm import lattice as lb
from canyon_lbm.backend import asnumpy, backend_name, xp
from canyon_lbm.boundary import log_law_profile
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation

RE = 25.0
M0 = dict(building_width_H=1.0, fetch_upstream_H=4.0, top_margin_H=5.0, outflow_H=8.0)


def _free():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def run_case(n=48, ar=1.0, u=0.05, sponge_cells_H=4.0, tau_sponge=1.0,
             outflow_H=8.0, n_ft=5, transient_ft=1.5, samp=250):
    """Run a canyon flow; return probe time series + windowed-mean record.

    Sampling is SYNC-FREE: per-sample probes are stored as on-device 0-d arrays
    (no host transfer / no float() -> no GPU stall) and pulled to host once at the
    end. Per-sample cost is two point reads (no full-field gradient), so the GPU
    pipeline stays saturated.
    """
    M = dict(M0); M["outflow_H"] = outflow_H
    g = build_canyon(cells_per_H=n, aspect_ratio=ar, **M)
    nu = u * g.h / RE
    tau = lb.tau_from_viscosity(nu)
    inlet = log_law_profile(g.ny, g.h, u, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, u, inlet, collision="bgk",
                           sponge_cells=int(sponge_cells_H * g.h),
                           tau_sponge=tau_sponge, with_scalar=False)
    ft = int((4 + 1 + 1 + 1 + 8) * g.h / u)
    cav = xp.asarray(g.cavity_mask); fluid = sim.fluid
    h = g.h; s0, s1 = g.street
    cc_row, cc_col = h // 2, (s0 + s1) // 2                 # canyon-interior probe
    wk_col = min(s1 + h + int(1.5 * h), g.nx - int(sponge_cells_H * h) - 5)
    wk_row = h // 2                                         # downstream-wake probe
    ramp = 8000
    transient = int(transient_ft * ft)

    ts, cc_s, wk_s = [], [], []                             # device 0-d scalars
    sum_ux = sum_uy = None; navg = 0; meanrec = []
    next_mean_snap = transient + ft // 2
    for it in range(1, n_ft * ft + 1):
        sim.step(inlet_scale=min(1.0, it / ramp))
        if it % samp == 0:
            _, ux, uy = sim.macroscopic()
            uy = xp.where(fluid, uy, 0.0)
            ts.append(it)
            cc_s.append(uy[cc_row, cc_col]); wk_s.append(uy[wk_row, wk_col])
            if it > transient:
                ux = xp.where(fluid, ux, 0.0)
                if sum_ux is None:
                    sum_ux = xp.zeros_like(ux); sum_uy = xp.zeros_like(uy)
                sum_ux += ux; sum_uy += uy; navg += 1
                if it >= next_mean_snap:                    # rare -> sync OK here
                    mux, muy = sum_ux / navg, sum_uy / navg
                    wzm = xp.gradient(muy, axis=1) - xp.gradient(mux, axis=0)
                    meanrec.append((navg * samp, float(xp.sum(wzm[cav]))))
                    next_mean_snap += ft // 2
    cc = asnumpy(xp.stack(cc_s)); wk = asnumpy(xp.stack(wk_s))   # single transfer
    _free()
    return {"ft": ft, "h": h, "nx": g.nx, "transient": transient, "samp": samp,
            "ts": np.array(ts), "canyon_probe": cc, "wake_probe": wk,
            "meanrec": meanrec}


def _period_St(ts, sig, transient, samp, h, u):
    """Dominant period (steps) via FFT + zero-crossings; Strouhal St = h/(T*u)."""
    m = ts > transient
    x = sig[m] - sig[m].mean()
    N = x.size
    if N < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # FFT (uniform sampling at `samp` steps)
    F = np.abs(np.fft.rfft(x * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=samp)              # cycles per step
    k = 1 + int(np.argmax(F[1:]))
    T_fft = 1.0 / freqs[k] if freqs[k] > 0 else float("nan")
    # zero-crossings
    sgn = np.sign(x); nz = np.sum(np.abs(np.diff(sgn)) > 0)
    span = (ts[m][-1] - ts[m][0])
    T_zc = 2 * span / nz if nz > 0 else float("nan")
    St_fft = h / (T_fft * u) if T_fft == T_fft else float("nan")
    amp = x.std() / (abs(sig[m].mean()) + 1e-30)    # relative oscillation amplitude
    return T_fft, T_zc, St_fft, amp


def main():
    print(f"backend={backend_name()}  RE={RE}\n", flush=True)
    cases = [
        ("BASE            ", dict(n=48, u=0.05)),
        ("sponge_LEN_8H   ", dict(n=48, u=0.05, sponge_cells_H=8.0)),
        ("sponge_STRONG   ", dict(n=48, u=0.05, tau_sponge=1.9)),
        ("fetch_outflow16H", dict(n=48, u=0.05, outflow_H=16.0)),
        ("mach_u0.025     ", dict(n=48, u=0.025)),
        ("mach_u0.10      ", dict(n=48, u=0.10)),
        ("grid_n24        ", dict(n=24, u=0.05)),
        ("grid_n96        ", dict(n=96, u=0.05, n_ft=4)),
    ]
    print(f"{'case':16s} {'h':>3} {'u':>5} {'T_canyon':>8} {'T_wake':>7} {'St':>5} "
          f"{'mean_circ':>9} {'canyonAmp':>9} {'canyonAmp/u':>11} {'wakeAmp':>8}",
          flush=True)
    base = None
    for label, kw in cases:
        t0 = time.time()
        r = run_case(**kw)
        h, u = r["h"], kw.get("u", 0.05)
        _, T_zc, St, _ = _period_St(r["ts"], r["canyon_probe"], r["transient"], r["samp"], h, u)
        _, Tw_zc, _, _ = _period_St(r["ts"], r["wake_probe"], r["transient"], r["samp"], h, u)
        m = r["ts"] > r["transient"]
        cc = r["canyon_probe"][m]; wk = r["wake_probe"][m]
        cc_amp = cc.std(); wk_amp = wk.std()
        mean_circ = r["meanrec"][-1][1] if r["meanrec"] else float("nan")
        print(f"{label} {h:>3} {u:>5.3f} {T_zc:>8.0f} {Tw_zc:>7.0f} {St:>5.3f} "
              f"{mean_circ:>9.4f} {cc_amp:>9.2e} {cc_amp/u:>11.3f} {wk_amp:>8.2e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if label.startswith("BASE"):
            base = r

    # CHECK 2 -- window independence of the mean (from the BASE run)
    print("\n=== CHECK 2: mean-field cavity-circulation vs averaging window (BASE) ===",
          flush=True)
    T_fft, T_zc, St, _ = _period_St(base["ts"], base["canyon_probe"], base["transient"],
                                    base["samp"], base["h"], 0.05)
    period = T_zc if T_zc == T_zc else T_fft
    print(f"  (dominant period ~= {period:.0f} steps ~= {period/base['ft']:.3f} ft)",
          flush=True)
    prev = None
    for avg_steps, circ_of_mean in base["meanrec"]:
        nper = avg_steps / period if period else float("nan")
        dpct = (100 * abs(circ_of_mean - prev) / (abs(prev) + 1e-30)
                if prev is not None else float("nan"))
        print(f"  avg over {avg_steps:6d} steps (~{nper:4.1f} periods): "
              f"circ_of_mean={circ_of_mean:+.5f}  d_from_prev={dpct:5.2f}%", flush=True)
        prev = circ_of_mean


if __name__ == "__main__":
    main()
