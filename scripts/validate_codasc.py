#!/usr/bin/env python3
"""Phase 5 -- CODASC wind-tunnel validation.

Runs the production canyon (H/W=1, no trees, perpendicular wind, street-level
line source; steady-laminar Re=25, 48 cells/H) and compares the leeward/windward
wall normalized-concentration (c+) profiles against the CODASC reference data
(Gromke & Ruck, KIT; downloaded to data/validation/).

c+ normalization (CODASC):  c+ = c * U_H * H / Q_l   (Q_l = line-source rate).
c+ is sampled at x+ = 1/24 (x/H) in front of each wall, as in the experiment.

HONEST SCOPE: the wind tunnel is turbulent (Re ~ 3.7e4); this laminar model is
not expected to match the *absolute* c+. The validation target is the PATTERN
-- the leeward>>windward asymmetry and the vertical c+ decay -- which is set by
the canyon vortex and is the physically robust, regime-independent comparison.
COST 732 metrics (FAC2, NMSE, hit rate) are reported descriptively.

Usage (GPU recommended):  CANYON_LBM_BACKEND=cupy python scripts/validate_codasc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canyon_lbm import io, metrics as mt
from canyon_lbm.backend import backend_name
from canyon_lbm.solver import CanyonSimulation

ROOT = Path(__file__).resolve().parents[1]


def codasc_centreplane(path: Path):
    """Vertical c+ profile at the canyon mid-length (y/H ~ 0) from a CODASC file.

    Returns (z/H levels, c+) sorted by height. The file is a grid over y/H
    (along-canyon) x z/H (height); we take the column nearest y/H = 0.
    """
    d = np.genfromtxt(path, skip_header=1)
    yH, zH, cp = d[:, 0], d[:, 1], d[:, 2]
    levels = np.unique(zH)
    out = []
    for z in levels:
        m = d[zH == z]
        out.append(m[np.argmin(np.abs(m[:, 0]))][2])  # c+ at y/H nearest 0
    return levels, np.array(out)


def run_validation_case(config: dict) -> dict:
    sim = CanyonSimulation.from_config(config)
    g = sim.geom
    run = config.get("run", {})
    print(f"CODASC case: H/W={g.aspect_ratio}, {g.ny}x{g.nx}, tau={sim.tau:.4f}, "
          f"backend={backend_name()}", flush=True)
    out = sim.run(
        max_iter=int(run.get("max_iter", 144_000)),
        tol=float(run.get("steady_tol", 1e-6)),
        check_every=int(run.get("check_every", 4000)),
        ramp_iters=int(run.get("ramp_iters", 8000)),
        average_from=int(run["average_from"]) if run.get("average_from") else None,
        verbose=True,
    )
    C = out["mean_C"] if out.get("mean_C") is not None else None
    if C is None:
        raise RuntimeError("no scalar field -- enable the scalar block in the config")

    # --- wall c+ profiles, sampled x+ = 1/24 (x/H) in front of each wall ---
    h, (s0, s1) = g.h, g.street
    off = max(1, int(round(config["validation"]["wall_offset_xplus"] * h)))
    rows = np.arange(1, h + 1)
    zH_sim = (rows - 0.5) / h
    cplus = sim.u_lbm * h / sim.source_rate            # c+ = C * U_H * H / Q_l
    west_face = C[rows, s0 + off] * cplus              # west building's canyon face
    east_face = C[rows, s1 - 1 - off] * cplus          # east building's canyon face
    # Leeward = the higher-concentration wall (vortex deposits pollutant there).
    if west_face.mean() >= east_face.mean():
        lee, wind = west_face, east_face
        lee_label = "west face (leeward)"
    else:
        lee, wind = east_face, west_face
        lee_label = "east face (leeward)"
    return {"sim": dict(zH=zH_sim, lee=lee, wind=wind, lee_label=lee_label,
                        flux_over_source=out["scalar"]["flux_over_source"]),
            "iters": out["iters"]}


def main() -> None:
    cfg = io.load_config(ROOT / "configs" / "validation_codasc.yaml")
    res = run_validation_case(cfg)
    sim = res["sim"]

    zH_A, cpA = codasc_centreplane(ROOT / cfg["validation"]["reference_A"])  # leeward
    zH_B, cpB = codasc_centreplane(ROOT / cfg["validation"]["reference_B"])  # windward

    # Interpolate the sim profiles onto the CODASC z/H levels for comparison.
    lee_i = np.interp(zH_A, sim["zH"], sim["lee"])
    wind_i = np.interp(zH_B, sim["zH"], sim["wind"])

    obs = np.concatenate([cpA, cpB])          # CODASC (observed)
    simv = np.concatenate([lee_i, wind_i])    # model

    # Pattern signatures (normalization-independent):
    asym_sim = float(sim["lee"].mean() / max(sim["wind"].mean(), 1e-30))
    asym_ref = float(cpA.mean() / cpB.mean())
    # vertical decay: ratio street/roof on the leeward wall
    decay_sim = float(sim["lee"][0] / max(sim["lee"][-1], 1e-30))
    decay_ref = float(cpA[0] / cpA[-1])

    out = {
        "regime": {"Re": cfg["flow"]["Re"], "collision": cfg["flow"]["collision"],
                   "cells_per_H": cfg["resolution"]["cells_per_H"]},
        "flux_over_source": sim["flux_over_source"],
        "pattern": {
            "leeward_windward_ratio_sim": asym_sim,
            "leeward_windward_ratio_codasc": asym_ref,
            "leeward_streettoroof_decay_sim": decay_sim,
            "leeward_streettoroof_decay_codasc": decay_ref,
        },
        "cost732_raw_cplus": {     # absolute c+ -- expected to differ (laminar vs turbulent)
            "FAC2": mt.fac2(obs, simv),
            "NMSE": mt.nmse(obs, simv),
            "hit_rate": mt.hit_rate(obs, simv, dq=0.25),
        },
        "profiles": {
            "zH_codasc": zH_A.tolist(), "cplus_leeward_codasc": cpA.tolist(),
            "cplus_windward_codasc": cpB.tolist(),
            "zH_sim": sim["zH"].tolist(), "cplus_leeward_sim": sim["lee"].tolist(),
            "cplus_windward_sim": sim["wind"].tolist(),
        },
    }
    io.save_result(ROOT / "results" / "codasc_validation.json", out, config=cfg)

    from canyon_lbm import viz
    paths = viz.plot_codasc_validation(out, ROOT / "figures" / "codasc_validation")

    print("\n=== CODASC validation (H/W=1, no trees, 90 deg) ===")
    print(f"  leeward/windward asymmetry : sim {asym_sim:.2f}  vs  CODASC {asym_ref:.2f}")
    print(f"  leeward street/roof decay  : sim {decay_sim:.2f}  vs  CODASC {decay_ref:.2f}")
    print(f"  raw-c+ FAC2={out['cost732_raw_cplus']['FAC2']:.2f} "
          f"NMSE={out['cost732_raw_cplus']['NMSE']:.2f} "
          f"hit_rate={out['cost732_raw_cplus']['hit_rate']:.2f} "
          f"(absolute c+ differs: laminar model vs turbulent tunnel)")
    print(f"  figure -> {', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
