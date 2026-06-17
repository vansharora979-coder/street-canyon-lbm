"""Boundary conditions for the D2Q9 solver (NumPy/CuPy backend-agnostic).

Implemented and validated:
  * Halfway bounce-back no-slip walls for arbitrary solid masks (Phase 1).
  * Free-slip / specular top (Phase 2).
  * Non-equilibrium-extrapolation velocity inlet + constant-pressure outlet,
    log-/power-law inlet profiles (Phase 2). Zou/He inlet kept for reference.

All functions operate on the ``(9, Ny, Nx)`` distribution array.
"""

from __future__ import annotations

from .backend import xp
from .lattice import CXi, CYi, OPPi, NQ, equilibrium


def precompute_bounceback(solid):
    """Per-direction fluid nodes that bounce back (halfway no-slip).

    ``masks[j]`` is True at fluid nodes whose upstream neighbour in direction j
    is solid; ``roll(solid, c_j)`` places ``solid[x - c_j]`` at index x.
    """
    fluid = ~solid
    masks = []
    for j in range(NQ):
        upstream_solid = xp.roll(solid, shift=(CYi[j], CXi[j]), axis=(0, 1))
        masks.append(upstream_solid & fluid)
    return masks


def apply_bounceback(f, fpost, masks) -> None:
    """Apply halfway bounce-back in place after streaming."""
    for j in range(NQ):
        m = masks[j]
        if bool(m.any()):
            f[j][m] = fpost[OPPi[j]][m]


def log_law_profile(ny: int, h: int, u_ref: float, z0_cells: float):
    """Logarithmic approach-flow profile u(y), normalized so u(H) = u_ref.

    u(y) = u_ref * ln((y+z0)/z0) / ln((H+z0)/z0); ground no-slip plane at row 0.5
    (heights y = row - 0.5); ground/below-plane rows are zero. Shape (ny,).
    """
    y = xp.arange(ny, dtype=float) - 0.5
    u = xp.zeros(ny, dtype=float)
    valid = y > 0
    den = float(xp.log((h + z0_cells) / z0_cells))
    u[valid] = u_ref * xp.log((y[valid] + z0_cells) / z0_cells) / den
    return u


def power_law_profile(ny: int, h: int, u_ref: float, alpha: float = 0.25):
    """Power-law approach-flow profile u(y) = u_ref (y/H)^alpha, u(H) = u_ref."""
    y = xp.arange(ny, dtype=float) - 0.5
    u = xp.zeros(ny, dtype=float)
    valid = y > 0
    u[valid] = u_ref * (y[valid] / h) ** alpha
    return u


def inlet_velocity_neq(f, ux_b, uy_b) -> None:
    """Non-equilibrium extrapolation velocity inlet on the west (x=0) column.

    f(0) = f_eq(rho_1, u_b) + [ f(1) - f_eq(rho_1, u_1) ] (Guo, Zheng & Shi 2002):
    prescribed velocity, density extrapolated from x=1, neighbour non-eq copied.
    Robust at the inlet corners where the Zou/He formula seeds a disturbance.
    """
    col1 = f[:, :, 1]                       # (9, ny)
    rho1 = col1.sum(axis=0)
    ux1 = sum(CXi[i] * col1[i] for i in range(NQ)) / rho1
    uy1 = sum(CYi[i] * col1[i] for i in range(NQ)) / rho1
    f[:, :, 0] = equilibrium(rho1, ux_b, uy_b) + (col1 - equilibrium(rho1, ux1, uy1))


def outlet_pressure_neq(f, rho_b: float = 1.0) -> None:
    """Non-equilibrium extrapolation constant-pressure outlet on the east column.

    f(-1) = f_eq(rho_b, u_-2) + [ f(-2) - f_eq(rho_-2, u_-2) ]. Pinning the outlet
    density stops the slow mass drift a velocity-in / gradient-out pair accrues.
    """
    col = f[:, :, -2]
    rho2 = col.sum(axis=0)
    ux2 = sum(CXi[i] * col[i] for i in range(NQ)) / rho2
    uy2 = sum(CYi[i] * col[i] for i in range(NQ)) / rho2
    rb = xp.full_like(rho2, rho_b)
    f[:, :, -1] = equilibrium(rb, ux2, uy2) + (col - equilibrium(rho2, ux2, uy2))


def zou_he_velocity_west(f, ux, uy) -> None:
    """Zou/He (1997) prescribed-velocity inlet (kept for reference; not the
    default -- see ``inlet_velocity_neq``). Sets density and the three unknown
    incoming populations (1, 5, 8) from the known ones. Modifies ``f`` in place.
    """
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (f[i, :, 0] for i in range(NQ))
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)
    f[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux
    f[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy
    f[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy


def outflow_east_zerogradient(f) -> None:
    """Zero-gradient (Neumann) outflow: copy the penultimate column to the last."""
    f[:, :, -1] = f[:, :, -2]


def freeslip_top(f, fpost) -> None:
    """Free-slip (specular) top boundary, applied link-wise from the
    post-collision distribution (using the streamed array injects momentum and
    blows up the top row): 4<-2, 7<-6, 8<-5 at the top row."""
    f[4, -1, :] = fpost[2, -1, :]
    f[7, -1, :] = fpost[6, -1, :]
    f[8, -1, :] = fpost[5, -1, :]
