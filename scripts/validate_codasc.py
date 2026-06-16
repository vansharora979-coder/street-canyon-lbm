#!/usr/bin/env python3
"""Phase 5 -- CODASC wind-tunnel validation (HARD gate).

Reproduces an empty-canyon CODASC case (matching H/W, perpendicular wind,
normalized street-level line source), compares normalized wall concentrations
c+ against the wind-tunnel data, and computes COST 732 metrics (FAC2, NMSE, hit
rate). Acceptance: FAC2 >= 0.66 (or clearly justified). Do not proceed to the
production sweep otherwise.

Not yet implemented: requires the canyon solver (Phase 2) and scalar metric
(Phase 3), plus the CODASC reference data in data/validation/ (see that
directory's README for how to obtain it).
"""

import sys


def main() -> None:
    raise SystemExit(
        "validate_codasc.py runs in Phase 5. Build Phases 2-3 first and place the "
        "CODASC reference data in data/validation/ (see data/validation/README.md)."
    )


if __name__ == "__main__":
    sys.exit(main())
