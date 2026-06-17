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


def log_law_profile(ny: int, h: int, u_ref: float, z0_cells: float) -> np.ndarray:
    """Logarithmic approach-flow profile u(y), normalized so u(H) = u_ref.

    u(y) = u_ref * ln((y + z0)/z0) / ln((H + z0)/z0), with the ground no-slip
    plane at row 0.5 (halfway bounce-back). Heights are y = row - 0.5; the
    ground row (and anything below the plane) is set to zero.

    Returns an array of shape (ny,) giving the streamwise inlet velocity per row.
    """
    y = np.arange(ny, dtype=np.float64) - 0.5
    u = np.zeros(ny, dtype=np.float64)
    valid = y > 0
    den = np.log((h + z0_cells) / z0_cells)
    u[valid] = u_ref * np.log((y[valid] + z0_cells) / z0_cells) / den
    return u


def power_law_profile(ny: int, h: int, u_ref: float, alpha: float = 0.25) -> np.ndarray:
    """Power-law approach-flow profile u(y) = u_ref (y/H)^alpha, u(H) = u_ref.

    Ground no-slip plane at row 0.5; heights y = row - 0.5. Returns shape (ny,).
    """
    y = np.arange(ny, dtype=np.float64) - 0.5
    u = np.zeros(ny, dtype=np.float64)
    valid = y > 0
    u[valid] = u_ref * (y[valid] / h) ** alpha
    return u


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


def inlet_velocity_neq(f: np.ndarray, ux_b: np.ndarray, uy_b: np.ndarray) -> None:
    """Non-equilibrium extrapolation velocity inlet on the west (x=0) column.

    Guo, Zheng & Shi (2002): the boundary node takes the prescribed velocity
    with the density extrapolated from the first interior column, and copies the
    neighbour's non-equilibrium part::

        f(0) = f_eq(rho_1, u_b) + [ f(1) - f_eq(rho_1, u_1) ]

    Second-order, and far more robust at the inlet-top/-ground corners than the
    Zou/He formula (which seeds a growing corner disturbance here). Modifies
    ``f`` in place. ``ux_b, uy_b`` are the prescribed inlet velocity per row.
    """
    from .lattice import equilibrium

    col1 = f[:, :, 1]                       # (9, ny) first interior column
    rho1 = col1.sum(axis=0)
    ux1 = (CX[:, None] * col1).sum(axis=0) / rho1
    uy1 = (CY[:, None] * col1).sum(axis=0) / rho1
    f[:, :, 0] = equilibrium(rho1, ux_b, uy_b) + (col1 - equilibrium(rho1, ux1, uy1))


def outlet_pressure_neq(f: np.ndarray, rho_b: float = 1.0) -> None:
    """Non-equilibrium extrapolation constant-pressure outlet on the east column.

    Prescribes density ``rho_b`` (default 1) with the velocity extrapolated from
    the penultimate column::

        f(-1) = f_eq(rho_b, u_-2) + [ f(-2) - f_eq(rho_-2, u_-2) ]

    Pinning the outlet pressure stops the slow mass drift a velocity-in /
    gradient-out pair would otherwise accumulate. Modifies ``f`` in place.
    """
    from .lattice import equilibrium

    col = f[:, :, -2]                       # (9, ny) penultimate column
    rho2 = col.sum(axis=0)
    ux2 = (CX[:, None] * col).sum(axis=0) / rho2
    uy2 = (CY[:, None] * col).sum(axis=0) / rho2
    rb = np.full_like(rho2, rho_b)
    f[:, :, -1] = equilibrium(rb, ux2, uy2) + (col - equilibrium(rho2, ux2, uy2))


def outflow_east_zerogradient(f: np.ndarray) -> None:
    """Zero-gradient (Neumann) outflow: copy the penultimate column to the last.

    Cheap, stable convective-style outlet for the populations leaving the
    domain. Modifies ``f`` in place. Validated in Phase 2.
    """
    f[:, :, -1] = f[:, :, -2]


def freeslip_top(f: np.ndarray, fpost: np.ndarray) -> None:
    """Free-slip (specular reflection) top boundary at the highest row.

    Reflects the wall-normal (y) component while preserving the tangential (x)
    component. Like bounce-back, this is a link-wise reflection applied after
    streaming using the *post-collision* distribution ``fpost`` (using the
    already-streamed ``f`` instead injects spurious momentum and blows up the
    top row). The up-going populations leaving the top reflect into the matching
    down-going ones with the same x-component:
        4 (S)  <- fpost 2 (N),   7 (SW) <- fpost 6 (NW),   8 (SE) <- fpost 5 (NE).
    Modifies ``f`` in place.
    """
    f[4, -1, :] = fpost[2, -1, :]
    f[7, -1, :] = fpost[6, -1, :]
    f[8, -1, :] = fpost[5, -1, :]
