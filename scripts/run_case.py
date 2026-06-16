#!/usr/bin/env python3
"""Run a single simulation case from a YAML config.

Phase 1: only the ``poiseuille`` validation case is wired up. The canyon case
(``type: canyon``) is added in Phase 2.

Usage:
    python scripts/run_case.py --config configs/base.yaml
    python scripts/run_case.py --poiseuille
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canyon_lbm import io
from canyon_lbm.solver import run_poiseuille

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument(
        "--poiseuille",
        action="store_true",
        help="Run the Phase 1 Poiseuille validation case.",
    )
    args = ap.parse_args()

    config = io.load_config(args.config) if args.config else {}
    case_type = "poiseuille" if args.poiseuille else config.get("type", "poiseuille")

    if case_type == "poiseuille":
        params = config.get("poiseuille", {}) if config else {}
        res = run_poiseuille(**params)
        # Strip the large arrays for the JSON summary; keep scalars + profiles.
        payload = {
            k: v
            for k, v in res.items()
            if k not in ("ux_profile", "ux_analytic", "y")
        }
        payload["ux_profile"] = res["ux_profile"].tolist()
        payload["ux_analytic"] = res["ux_analytic"].tolist()
        payload["y"] = res["y"].tolist()
        out = ROOT / "results" / "poiseuille.json"
        io.save_result(out, payload, config=config or {"type": "poiseuille"})
        print(json.dumps(
            {k: res[k] for k in
             ("converged", "iters", "rel_l2_error", "max_rel_error", "tau", "nu")},
            indent=2,
        ))
        print(f"\nResult + metadata written to {out.relative_to(ROOT)}")
    else:
        raise SystemExit(
            f"Case type '{case_type}' is not available until a later phase "
            "(canyon flow = Phase 2, scalar = Phase 3)."
        )


if __name__ == "__main__":
    main()
