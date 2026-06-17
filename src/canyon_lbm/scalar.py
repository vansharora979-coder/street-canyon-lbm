"""Passive-scalar pollutant transport via a D2Q5 advection-diffusion LBM.

A separate population ``g`` of 5 velocities is advected by the D2Q9 flow
velocity field and relaxed toward its advection-diffusion equilibrium, solving

    dC/dt + u . grad C = D laplacian C + S

with diffusivity ``D = c_s^2 (tau_g - 1/2)`` (c_s^2 = 1/3) and a source ``S``
(the street-level line source). One-way coupling: the scalar does not affect the
flow. Array ops go through :mod:`canyon_lbm.backend` (NumPy or CuPy).

D2Q5 velocity ordering (rest + 4 nearest neighbours)::

    i :   0    1    2    3    4
    cx:   0    1    0   -1    0
    cy:   0    0    1    0   -1
"""

from __future__ import annotations

from .backend import xp

# Host-side Python tuples (roll shifts, indexing, scalar multiplies).
CX5i = (0, 1, 0, -1, 0)
CY5i = (0, 0, 1, 0, -1)
W5f = (1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6)   # sum to 1 => c_s^2 = 1/3
OPP5i = (0, 3, 4, 1, 2)

# Backend arrays (vectorized use / tests on the NumPy backend).
C5 = xp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
CX5 = C5[:, 0]
CY5 = C5[:, 1]
W5 = xp.array(W5f, dtype=float)
OPP5 = xp.array(OPP5i, dtype=int)
_W5v = xp.asarray(W5f, dtype=float)[:, None, None]
CS2_5 = 1.0 / 3.0
NQ5 = 5


def equilibrium_scalar(C, ux, uy):
    """Linear advection-diffusion equilibrium  g_i^eq = w_i C (1 + e_i.u / c_s^2)."""
    geq = xp.empty((NQ5,) + C.shape, dtype=float)
    for i in range(NQ5):
        eu = CX5i[i] * ux + CY5i[i] * uy
        geq[i] = W5f[i] * C * (1.0 + 3.0 * eu)
    return geq


def scalar_concentration(g):
    """C = sum_i g_i."""
    return g.sum(axis=0)


def diffusivity_from_tau(tau_g):
    return CS2_5 * (tau_g - 0.5)


def tau_from_diffusivity(D):
    return D / CS2_5 + 0.5


def tau_from_schmidt(nu, schmidt):
    """Scalar relaxation time from the flow viscosity and Schmidt number Sc=nu/D."""
    return tau_from_diffusivity(nu / schmidt)


def collide_scalar(g, geq, tau_g, source=None):
    """BGK collision for the scalar, optionally with a source term.

    g_i^post = g_i - (g_i - g_i^eq)/tau_g [+ w_i S]. ``source`` is a (Ny, Nx)
    field of S; ``tau_g`` may be scalar or field.
    """
    if getattr(tau_g, "ndim", 0) == 2:
        tau_g = tau_g[None, :, :]
    gpost = g - (g - geq) / tau_g
    if source is not None:
        gpost = gpost + _W5v * source[None, :, :]
    return gpost


def stream_scalar(gpost):
    g = xp.empty_like(gpost)
    for i in range(NQ5):
        g[i] = xp.roll(gpost[i], shift=(CY5i[i], CX5i[i]), axis=(0, 1))
    return g


def precompute_bounceback_scalar(solid):
    """Zero-flux (insulating, Neumann) walls via bounce-back of ``g`` (D2Q5)."""
    fluid = ~solid
    masks = []
    for j in range(NQ5):
        masks.append(xp.roll(solid, shift=(CY5i[j], CX5i[j]), axis=(0, 1)) & fluid)
    return masks


def apply_bounceback_scalar(g, gpost, masks) -> None:
    for j in range(NQ5):
        m = masks[j]
        if bool(m.any()):
            g[j][m] = gpost[OPP5i[j]][m]


def inlet_zero_concentration(g) -> None:
    """Clean-air Dirichlet inlet (C = 0) on the west column: g(:, :, 0) = 0."""
    g[:, :, 0] = 0.0


def open_outlet(g) -> None:
    """Zero-gradient (open) scalar outlet on the east column."""
    g[:, :, -1] = g[:, :, -2]


def open_top(g) -> None:
    """Zero-gradient (open) scalar top boundary."""
    g[:, -1, :] = g[:, -2, :]
