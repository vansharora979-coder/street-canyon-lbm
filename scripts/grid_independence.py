#!/usr/bin/env python3
"""Phase 4 -- grid-independence study (HARD gate).

Runs H/W = 1 at increasing resolution (e.g. 20, 40, 80 cells per building
height), plots the ventilation metric vs resolution, and selects the production
resolution where the metric changes < ~2-3% between the two finest grids.

Not yet implemented: requires the canyon flow solver (Phase 2) and the scalar
metric (Phase 3).
"""

import sys


def main() -> None:
    raise SystemExit(
        "grid_independence.py runs in Phase 4, after the canyon solver (Phase 2) "
        "and scalar metric (Phase 3) are built and CODASC-validated (Phase 5)."
    )


if __name__ == "__main__":
    sys.exit(main())
