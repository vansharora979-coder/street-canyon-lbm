#!/usr/bin/env python3
"""Phase 6 -- production aspect-ratio sweep.

Sweeps H/W in {0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0} (perpendicular wind) at the
production resolution chosen in Phase 4, saving every result + metadata to CSV.
Produces the metric-vs-H/W table from which the skimming-flow transition is
located.

Gated: only runs after BOTH grid-independence (Phase 4) and CODASC validation
(Phase 5) pass.
"""

import sys


def main() -> None:
    raise SystemExit(
        "run_sweep.py runs in Phase 6, and only after the grid-independence "
        "(Phase 4) and CODASC validation (Phase 5) gates have passed."
    )


if __name__ == "__main__":
    sys.exit(main())
