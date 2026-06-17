#!/usr/bin/env python3
"""Run a single simulation case from a YAML config.

Supported case types:
  * ``poiseuille`` -- Phase 1 solver validation (analytic channel).
  * ``canyon``     -- Phase 2 street-canyon flow (config-driven geometry + BCs).

Usage:
    python scripts/run_case.py --config configs/canyon_demo.yaml
    python scripts/run_case.py --poiseuille
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from canyon_lbm import io
from canyon_lbm.solver import CanyonSimulation, run_poiseuille

ROOT = Path(__file__).resolve().parents[1]


def _run_poiseuille(config: dict) -> None:
    params = config.get("poiseuille", {}) if config else {}
    res = run_poiseuille(**params)
    payload = {k: v for k, v in res.items()
               if k not in ("ux_profile", "ux_analytic", "y")}
    for k in ("ux_profile", "ux_analytic", "y"):
        payload[k] = res[k].tolist()
    out = ROOT / "results" / "poiseuille.json"
    io.save_result(out, payload, config=config or {"type": "poiseuille"})
    print(json.dumps({k: res[k] for k in
                      ("converged", "iters", "rel_l2_error", "max_rel_error", "tau")},
                     indent=2))
    print(f"\nResult + metadata written to {out.relative_to(ROOT)}")


def _run_canyon(config: dict) -> None:
    sim = CanyonSimulation.from_config(config)
    geom = sim.geom
    run = config.get("run", {})
    print(f"Canyon H/W={geom.aspect_ratio} (actual {geom.aspect_ratio_actual:.3f}): "
          f"grid {geom.ny}x{geom.nx} cells, tau={sim.tau:.4f}, collision={sim.collision}")
    avg_from = run.get("average_from", None)
    out = sim.run(
        max_iter=int(run.get("max_iter", 150_000)),
        tol=float(run.get("steady_tol", 1e-6)),
        check_every=int(run.get("check_every", 1000)),
        ramp_iters=int(run.get("ramp_iters", 8000)),
        average_from=int(avg_from) if avg_from is not None else None,
        verbose=True,
    )
    fields = out.pop("fields")
    out.pop("history", None)
    mean_C = out.pop("mean_C", None)

    ar_tag = f"{geom.aspect_ratio:g}".replace(".", "p")
    npz = ROOT / "results" / f"canyon_AR{ar_tag}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        ux=fields["ux"], uy=fields["uy"], rho=fields["rho"], solid=geom.solid,
        h=geom.h, w=geom.w, b=geom.b, aspect_ratio=geom.aspect_ratio,
        street=np.array(geom.street), west_building=np.array(geom.west_building),
        east_building=np.array(geom.east_building), roof_row=geom.roof_row,
        u_lbm=sim.u_lbm,
    )
    if mean_C is not None:
        arrays["C"] = mean_C
    np.savez_compressed(npz, **arrays)
    summary = ROOT / "results" / f"canyon_AR{ar_tag}.json"
    io.save_result(summary, out, config=config, extra={"fields_file": npz.name})

    keys = ["converged", "averaged", "n_avg", "iters", "single_vortex", "clockwise",
            "cavity_circulation", "mass_imbalance"]
    summ = {k: out.get(k) for k in keys}
    if "scalar" in out:
        summ["scalar"] = out["scalar"]
    print(json.dumps(summ, indent=2, default=float))
    print(f"\nFields -> {npz.relative_to(ROOT)}; summary+metadata -> "
          f"{summary.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--poiseuille", action="store_true",
                    help="Run the Phase 1 Poiseuille validation case.")
    args = ap.parse_args()

    config = io.load_config(args.config) if args.config else {}
    case_type = "poiseuille" if args.poiseuille else config.get("type", "poiseuille")

    if case_type == "poiseuille":
        _run_poiseuille(config)
    elif case_type == "canyon":
        _run_canyon(config)
    else:
        raise SystemExit(f"Unknown case type '{case_type}'.")


if __name__ == "__main__":
    main()
