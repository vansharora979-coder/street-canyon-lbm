"""Parameterized street-canyon geometry (solid masks, domain sizing).

Phase 2 deliverable. Builds, from physical parameters (building height H, street
width W, COST-732 domain margins) and a resolution, the boolean solid mask plus
the index sets needed by the boundary module.

Planned API
-----------
build_canyon(h_cells, aspect_ratio, ...) -> CanyonGeometry
    with .solid (Ny, Nx bool), .canyon_opening_row, .street_source_cells, etc.

Domain sizing follows COST 732 / AIJ best practice:
  * top boundary >= 5-6 H above rooftops,
  * outflow >= 15 H downstream of the canyon,
  * adequate upstream fetch (>= 5 H).
"""

from __future__ import annotations


def build_canyon(*args, **kwargs):  # pragma: no cover - Phase 2
    raise NotImplementedError(
        "Canyon geometry is implemented in Phase 2 (see PROGRESS.md)."
    )
