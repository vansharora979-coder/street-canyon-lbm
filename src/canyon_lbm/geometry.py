"""Parameterized street-canyon geometry (solid mask + index sets).

Builds, from physical parameters and a resolution, the boolean ``solid`` mask
plus the index sets the solver/boundary/scalar modules need. Everything is
config-driven; no geometry magic numbers live in the solver.

Coordinate convention (see :mod:`canyon_lbm.lattice`)
-----------------------------------------------------
Arrays are ``(Ny, Nx)``: axis 0 = y (row, up), axis 1 = x (col, inlet->outlet,
i.e. west->east, the wind direction).

Layout (cells), wind blowing in +x::

      top (free-slip)                         row Ny-1
      . . . . . . . . . . . . . . . . . . . .
      .            free stream             .   top margin = top_margin_H * h
      .   ____                  ____       .
      .  |wind|     street      |lee |     .   building height h  (rows 1..h)
   in |  |ward|   (canyon, w)   |ward|     | out
      .  |____|________________|____|      .
      #########  ground (row 0, solid)  ####   row 0
       <-L_up->  <-B-><--w--><-B->  <--L_down-->

* Ground: row 0 is solid everywhere (no-slip floor).
* Buildings: height ``h`` cells, occupying rows ``1..h`` over their footprint.
  Building tops are at row ``h``; the canyon opening plane is ``y = h + 0.5``.
* Aspect ratio ``AR = H/W`` controls the street width ``w = round(h / AR)``.

Domain sizing follows COST 732 / AIJ best practice (units of building height H):
upstream fetch >= 5 H, top margin >= 5-6 H, outflow >= 15 H downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CanyonGeometry:
    """Immutable canyon geometry: the solid mask and all derived index sets."""

    solid: np.ndarray            # (Ny, Nx) bool, True in ground + buildings
    ny: int
    nx: int
    h: int                       # building height [cells]
    w: int                       # street width [cells]  (= round(h / AR))
    b: int                       # building width [cells]
    aspect_ratio: float          # requested H/W
    L_up: int                    # upstream fetch [cells]
    L_down: int                  # downstream outflow length [cells]
    L_top: int                   # top margin above roofs [cells]
    # Column ranges (Python half-open [start, stop)):
    west_building: tuple[int, int]
    street: tuple[int, int]
    east_building: tuple[int, int]
    ground_row: int = 0
    roof_row: int = 0            # topmost solid building row (= h)
    source_row: int = 1          # street-floor row for the pollutant line source
    cavity_mask: np.ndarray = field(default=None, repr=False)  # (Ny,Nx) bool

    @property
    def aspect_ratio_actual(self) -> float:
        """Aspect ratio after integer rounding of w (h/w)."""
        return self.h / self.w

    @property
    def street_cols(self) -> np.ndarray:
        return np.arange(self.street[0], self.street[1])

    def opening_plane(self):
        """(row_below, row_above, col_slice) bounding the roof-level opening.

        Vertical flux across the canyon opening is evaluated between fluid rows
        ``h`` (top of cavity) and ``h + 1`` (free stream), over the street.
        """
        return self.roof_row, self.roof_row + 1, slice(*self.street)


def build_canyon(
    cells_per_H: int,
    aspect_ratio: float,
    building_width_H: float = 1.0,
    fetch_upstream_H: float = 5.0,
    top_margin_H: float = 6.0,
    outflow_H: float = 15.0,
) -> CanyonGeometry:
    """Construct a :class:`CanyonGeometry` from physical (H-relative) parameters.

    Parameters
    ----------
    cells_per_H : int
        Lattice resolution: cells per building height H. Sets ``h``.
    aspect_ratio : float
        H/W. Street width is ``w = round(h / AR)``.
    building_width_H, fetch_upstream_H, top_margin_H, outflow_H : float
        Building width and COST 732 domain margins, in units of H.
    """
    h = int(round(cells_per_H))
    w = int(round(cells_per_H / aspect_ratio))
    b = int(round(building_width_H * cells_per_H))
    L_up = int(round(fetch_upstream_H * cells_per_H))
    L_down = int(round(outflow_H * cells_per_H))
    L_top = int(round(top_margin_H * cells_per_H))
    if min(h, w, b, L_up, L_down, L_top) < 1:
        raise ValueError("All geometry extents must be >= 1 cell.")

    nx = L_up + b + w + b + L_down
    ny = h + L_top + 1  # +1 for the ground row at index 0

    west0, west1 = L_up, L_up + b
    str0, str1 = west1, west1 + w
    east0, east1 = str1, str1 + b

    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True                       # ground floor
    solid[1 : h + 1, west0:west1] = True      # windward (upstream) building
    solid[1 : h + 1, east0:east1] = True      # leeward (downstream) building

    cavity = np.zeros((ny, nx), dtype=bool)   # fluid cells inside the canyon
    cavity[1 : h + 1, str0:str1] = True

    return CanyonGeometry(
        solid=solid,
        ny=ny,
        nx=nx,
        h=h,
        w=w,
        b=b,
        aspect_ratio=aspect_ratio,
        L_up=L_up,
        L_down=L_down,
        L_top=L_top,
        west_building=(west0, west1),
        street=(str0, str1),
        east_building=(east0, east1),
        roof_row=h,
        source_row=1,
        cavity_mask=cavity,
    )
