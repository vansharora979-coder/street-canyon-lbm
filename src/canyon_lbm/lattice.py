"""D2Q9 lattice-Boltzmann primitives: velocity set, equilibrium, collision, forcing.

All quantities here are in *lattice units* (dx = dt = 1, lattice speed of sound
c_s^2 = 1/3). Conversion to physical units is handled in :mod:`canyon_lbm.solver`.

Array layout convention used throughout the package
---------------------------------------------------
Distribution functions ``f`` have shape ``(9, Ny, Nx)``:
    axis 0 -> lattice direction i = 0..8
    axis 1 -> y (row),  increasing upward in physical space
    axis 2 -> x (col),  increasing in the streamwise (inlet->outlet) direction

Macroscopic fields ``rho, ux, uy`` have shape ``(Ny, Nx)``.

D2Q9 velocity ordering (matches Zou/He 1997 and Mocz 2020)::

    i :   0    1    2    3    4    5    6    7    8
    cx:   0    1    0   -1    0    1   -1   -1    1
    cy:   0    0    1    0   -1    1    1   -1   -1
"""

from __future__ import annotations

import numpy as np

# --- D2Q9 velocity set ------------------------------------------------------
# Discrete velocities c_i = (cx_i, cy_i).
C = np.array(
    [
        [0, 0],   # 0  rest
        [1, 0],   # 1  E
        [0, 1],   # 2  N
        [-1, 0],  # 3  W
        [0, -1],  # 4  S
        [1, 1],   # 5  NE
        [-1, 1],  # 6  NW
        [-1, -1], # 7  SW
        [1, -1],  # 8  SE
    ],
    dtype=np.int64,
)
CX = C[:, 0]
CY = C[:, 1]

# Lattice weights w_i (sum to 1).
W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)

# Opposite direction for each i (used for bounce-back).
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

# Squared lattice speed of sound for the standard D2Q9 lattice.
CS2 = 1.0 / 3.0

NQ = 9


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order (incompressible-NS) Maxwellian equilibrium distribution.

    f_i^eq = w_i * rho * [1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5 |u|^2]

    Parameters
    ----------
    rho, ux, uy : ndarray, shape (Ny, Nx)

    Returns
    -------
    feq : ndarray, shape (9, Ny, Nx)
    """
    usq = ux * ux + uy * uy
    feq = np.empty((NQ,) + rho.shape, dtype=np.float64)
    for i in range(NQ):
        cu = CX[i] * ux + CY[i] * uy
        feq[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq


def macroscopic(
    f: np.ndarray, fx: np.ndarray | float = 0.0, fy: np.ndarray | float = 0.0
):
    """Density and (force-corrected) velocity moments of ``f``.

    With a body force ``F`` present, the consistent fluid velocity is
    ``u = (sum_i c_i f_i + F/2) / rho`` (Guo et al. 2002).

    Parameters
    ----------
    f : ndarray, shape (9, Ny, Nx)
    fx, fy : float or ndarray (Ny, Nx)
        Body-force density components (lattice units). Default 0.

    Returns
    -------
    rho, ux, uy : ndarray, shape (Ny, Nx)
    """
    rho = f.sum(axis=0)
    jx = (CX[:, None, None] * f).sum(axis=0)
    jy = (CY[:, None, None] * f).sum(axis=0)
    ux = (jx + 0.5 * fx) / rho
    uy = (jy + 0.5 * fy) / rho
    return rho, ux, uy


def guo_forcing(
    ux: np.ndarray,
    uy: np.ndarray,
    fx: np.ndarray | float,
    fy: np.ndarray | float,
    tau: float,
) -> np.ndarray:
    """Guo (2002) forcing source term added during collision.

    F_i = (1 - 1/(2 tau)) * w_i * [ 3 (c_i - u).F + 9 (c_i.u)(c_i.F) ]

    The (1 - 1/(2 tau)) prefactor together with the F/2 velocity correction in
    :func:`macroscopic` removes the discrete lattice force artefacts, so a
    constant body force reproduces the exact parabolic Poiseuille profile.

    Returns
    -------
    Fi : ndarray, shape (9, Ny, Nx)
    """
    pref = 1.0 - 1.0 / (2.0 * tau)
    shape = ux.shape
    Fi = np.empty((NQ,) + shape, dtype=np.float64)
    for i in range(NQ):
        ci_dot_u = CX[i] * ux + CY[i] * uy
        ci_dot_F = CX[i] * fx + CY[i] * fy
        ci_minus_u_dot_F = (CX[i] - ux) * fx + (CY[i] - uy) * fy
        Fi[i] = pref * W[i] * (3.0 * ci_minus_u_dot_F + 9.0 * ci_dot_u * ci_dot_F)
    return Fi


def collide_bgk(
    f: np.ndarray, feq: np.ndarray, tau: float, Fi: np.ndarray | None = None
) -> np.ndarray:
    """Single-relaxation-time (BGK) collision, optionally with a forcing term.

    f_i^post = f_i - (f_i - f_i^eq)/tau [+ F_i]
    """
    fpost = f - (f - feq) / tau
    if Fi is not None:
        fpost = fpost + Fi
    return fpost


def stream(fpost: np.ndarray) -> np.ndarray:
    """Advect each population by its lattice velocity (periodic via np.roll).

    Non-periodic boundaries are corrected afterwards by the boundary module.
    """
    f = np.empty_like(fpost)
    for i in range(NQ):
        f[i] = np.roll(fpost[i], shift=(CY[i], CX[i]), axis=(0, 1))
    return f


def viscosity_from_tau(tau: float) -> float:
    """Kinematic viscosity in lattice units: nu = c_s^2 (tau - 1/2)."""
    return CS2 * (tau - 0.5)


def tau_from_viscosity(nu: float) -> float:
    """Relaxation time from kinematic viscosity (inverse of viscosity_from_tau)."""
    return nu / CS2 + 0.5
