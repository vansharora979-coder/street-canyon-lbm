"""Boundary conditions for the D2Q9 solver.

Implemented and validated in Phase 1:
  * Halfway bounce-back no-slip walls for arbitrary solid masks.

Implemented for the canyon problem, exercised/validated in Phase 2:
  * Zou/He velocity inlet (west boundary).
  * Zero-gradient (Neumann) outflow (east boundary).
  * Free-slip / symmetry top boundary.

All functions operate on the ``(9, Ny, Nx)`` distribution array described in
:mod:`canyon_lbm.lattice`.
"""

from __future__ import annotations

import numpy as np

from .lattice import CX, CY, OPP, NQ


def precompute_bounceback(solid: np.ndarray) -> list[np.ndarray]:
    """Precompute, for each direction, the fluid nodes that bounce back in it.

    Halfway bounce-back rule: after streaming, a fluid node ``x`` receives in
    direction ``j`` the population that *would* have arrived from ``x - c_j``.
    If ``x - c_j`` is solid, that population is instead the post-collision
    population the node sent toward the wall, ``f_post[opp(j)](x)``.

    ``masks[j]`` is the boolean array (Ny, Nx), True at fluid nodes whose
    upstream neighbour in direction ``j`` is solid. ``np.roll(solid, c_j)``
    places ``solid[x - c_j]`` at index ``x``.

    Parameters
    ----------
    solid : ndarray (Ny, Nx) of bool
        True where the node is a solid obstacle/wall.

    Returns
    -------
    masks : list of 9 boolean ndarrays (Ny, Nx)
    """
    fluid = ~solid
    masks = []
    for j in range(NQ):
        upstream_solid = np.roll(solid, shift=(CY[j], CX[j]), axis=(0, 1))
        masks.append(upstream_solid & fluid)
    return masks


def apply_bounceback(
    f: np.ndarray, fpost: np.ndarray, masks: list[np.ndarray]
) -> None:
    """Apply halfway bounce-back in place after streaming.

    ``f`` is the streamed distribution (modified in place); ``fpost`` is the
    pre-streaming post-collision distribution; ``masks`` is from
    :func:`precompute_bounceback`.
    """
    for j in range(NQ):
        m = masks[j]
        if m.any():
            f[j][m] = fpost[OPP[j]][m]


def zou_he_velocity_west(f: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> None:
    """Zou/He (1997) prescribed-velocity inlet on the west (x = 0) column.

    Sets density and the three unknown populations streaming into the domain
    (directions 1, 5, 8) from the known ones, enforcing the prescribed inlet
    velocity profile ``(ux, uy)`` (each of shape (Ny,)).
    Modifies ``f`` in place. Validated in Phase 2.
    """
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (f[i, :, 0] for i in range(NQ))
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)
    f[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux
    f[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy
    f[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy


def outflow_east_zerogradient(f: np.ndarray) -> None:
    """Zero-gradient (Neumann) outflow: copy the penultimate column to the last.

    Cheap, stable convective-style outlet for the populations leaving the
    domain. Modifies ``f`` in place. Validated in Phase 2.
    """
    f[:, :, -1] = f[:, :, -2]


def freeslip_top(f: np.ndarray) -> None:
    """Free-slip (specular reflection) top boundary at the highest row.

    Reflects the wall-normal (y) component while preserving the tangential (x)
    component: incoming downward populations at the top row are set from the
    matching upward ones. Modifies ``f`` in place. Validated in Phase 2.
    """
    # At top row, the populations entering the domain point downward (cy < 0):
    # 4 (S) <- 2 (N), 7 (SW) <- 6 (NW), 8 (SE) <- 5 (NE).
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[6, -1, :]
    f[8, -1, :] = f[5, -1, :]
