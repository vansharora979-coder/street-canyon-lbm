#!/usr/bin/env python3
"""Regenerate paper figures. Single entry point for `make reproduce`.

As phases land, each figure is appended here so the whole set rebuilds with one
command. Phase 1 provides the Poiseuille validation figure.

Planned (by phase):
  Phase 1  poiseuille_validation        -- LBM vs analytic channel profile
  Phase 6  metric_vs_aspect_ratio       -- primary ventilation curve + regimes
  Phase 2  flow_field_regimes           -- streamline panels across regimes
  Phase 4  grid_independence            -- metric vs resolution
  Phase 5  codasc_validation            -- normalized c+ sim vs wind tunnel
  Phase 7  canyon_schematic             -- H/W definition sketch
"""

from __future__ import annotations

from pathlib import Path

from canyon_lbm import viz
from canyon_lbm.solver import run_poiseuille

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"


def fig_poiseuille_validation() -> None:
    res = run_poiseuille(ny=42, nx=8, tau=0.8, u_max=0.05, max_iter=200_000)
    paths = viz.plot_poiseuille(res, FIGDIR / "poiseuille_validation")
    print(
        f"  poiseuille_validation  (rel L2 err = {res['rel_l2_error']:.2e})"
        f" -> {', '.join(p.name for p in paths)}"
    )


def fig_canyon_flow() -> None:
    """Plot the H/W=1 canyon flow field from the saved run_case output."""
    npz = ROOT / "results" / "canyon_AR1.npz"
    if not npz.exists():
        print(
            "  canyon_flow_HW1  SKIPPED (no results/canyon_AR1.npz). "
            "Run: python scripts/run_case.py --config configs/canyon_demo.yaml"
        )
        return
    paths = viz.plot_canyon_flow(npz, FIGDIR / "canyon_flow_HW1",
                                 title="Street-canyon flow field, H/W = 1")
    print(f"  canyon_flow_HW1 -> {', '.join(p.name for p in paths)}")


def fig_canyon_concentration() -> None:
    """Plot the time-averaged pollutant field from the saved scalar-enabled run."""
    npz = ROOT / "results" / "canyon_AR1.npz"
    if not npz.exists():
        print("  canyon_concentration_HW1  SKIPPED (no results/canyon_AR1.npz).")
        return
    import numpy as np
    if "C" not in np.load(npz).files:
        print("  canyon_concentration_HW1  SKIPPED (run canyon with a scalar config).")
        return
    paths = viz.plot_canyon_concentration(
        npz, FIGDIR / "canyon_concentration_HW1",
        title="Time-averaged pollutant concentration, H/W = 1")
    print(f"  canyon_concentration_HW1 -> {', '.join(p.name for p in paths)}")


def fig_grid_independence() -> None:
    """Plot the ventilation metric vs resolution from the grid-independence CSV."""
    csv = ROOT / "results" / "grid_independence.csv"
    if not csv.exists():
        print("  grid_independence  SKIPPED (run scripts/grid_independence.py first).")
        return
    paths = viz.plot_grid_independence(csv, FIGDIR / "grid_independence")
    print(f"  grid_independence -> {', '.join(p.name for p in paths)}")


def fig_codasc_validation() -> None:
    """Plot the sim-vs-CODASC wall c+ comparison from the saved validation JSON."""
    import json
    j = ROOT / "results" / "codasc_validation.json"
    if not j.exists():
        print("  codasc_validation  SKIPPED (run scripts/validate_codasc.py first).")
        return
    paths = viz.plot_codasc_validation(json.load(open(j)),
                                       FIGDIR / "codasc_validation")
    print(f"  codasc_validation -> {', '.join(p.name for p in paths)}")


FIGURES = {
    "poiseuille_validation": fig_poiseuille_validation,
    "canyon_flow_HW1": fig_canyon_flow,
    "canyon_concentration_HW1": fig_canyon_concentration,
    "grid_independence": fig_grid_independence,
    "codasc_validation": fig_codasc_validation,
}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "names",
        nargs="*",
        help="Figure names to build (default: all currently available).",
    )
    args = ap.parse_args()
    names = args.names or list(FIGURES)
    print("Building figures:")
    for name in names:
        if name not in FIGURES:
            raise SystemExit(f"Unknown figure '{name}'. Available: {list(FIGURES)}")
        FIGURES[name]()
    print(f"Done. Figures in {FIGDIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
