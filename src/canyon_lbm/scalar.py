"""Passive-scalar pollutant transport via a D2Q5 advection-diffusion LBM.

A separate population ``g`` of 5 velocities is advected by the D2Q9 flow
velocity field and relaxed toward its advection-diffusion equilibrium. This
solves

    dC/dt + u . grad C = D laplacian C + S

with diffusivity ``D = c_s^2 (tau_g - 1/2)`` (c_s^2 = 1/3) and a source ``S``
(the street-level line source). One-way coupling: the scalar does not affect the
flow.

Array layout matches :mod:`canyon_lbm.lattice`: ``g`` has shape ``(5, Ny, Nx)``,
the concentration ``C`` has shape ``(Ny, Nx)``.

D2Q5 velocity ordering (rest + 4 nearest neighbours)::

    i :   0    1    2    3    4
    cx:   0    1    0   -1    0
    cy:   0    0    1    0   -1
"""

from __future__ import annotations

import numpy as np

C5 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)
CX5 = C5[:, 0]
CY5 = C5[:, 1]
# D2Q5 weights (sum to 1); give c_s^2 = 1/3.
W5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
OPP5 = np.array([0, 3, 4, 1, 2], dtype=np.int64)
CS2_5 = 1.0 / 3.0
NQ5 = 5


def equilibrium_scalar(C: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Linear advection-diffusion equilibrium  g_i^eq = w_i C (1 + e_i.u / c_s^2)."""
    geq = np.empty((NQ5,) + C.shape, dtype=np.float64)
    for i in range(NQ5):
        eu = CX5[i] * ux + CY5[i] * uy
        geq[i] = W5[i] * C * (1.0 + 3.0 * eu)
    return geq


def scalar_concentration(g: np.ndarray) -> np.ndarray:
    """C = sum_i g_i."""
    return g.sum(axis=0)


def diffusivity_from_tau(tau_g: float) -> float:
    return CS2_5 * (tau_g - 0.5)


def tau_from_diffusivity(D: float) -> float:
    return D / CS2_5 + 0.5


def tau_from_schmidt(nu: float, schmidt: float) -> float:
    """Scalar relaxation time from the flow viscosity and Schmidt number Sc=nu/D."""
    return tau_from_diffusivity(nu / schmidt)


def collide_scalar(g, geq, tau_g, source=None):
    """BGK collision for the scalar, optionally with a source term.

    g_i^post = g_i - (g_i - g_i^eq)/tau_g [+ w_i S].  ``source`` is a (Ny, Nx)
    field of S (concentration added per step); ``tau_g`` may be scalar or field.
    """
    if np.ndim(tau_g) == 2:
        tau_g = tau_g[None, :, :]
    gpost = g - (g - geq) / tau_g
    if source is not None:
        gpost = gpost + W5[:, None, None] * source[None, :, :]
    return gpost


def stream_scalar(gpost: np.ndarray) -> np.ndarray:
    g = np.empty_like(gpost)
    for i in range(NQ5):
        g[i] = np.roll(gpost[i], shift=(CY5[i], CX5[i]), axis=(0, 1))
    return g


def precompute_bounceback_scalar(solid: np.ndarray) -> list[np.ndarray]:
    """Zero-flux (insulating, Neumann) walls via bounce-back of ``g``.

    Non-absorbing pollutant walls => zero normal flux; bounce-back of the scalar
    populations enforces it. Same construction as the flow bounce-back, with the
    D2Q5 stencil.
    """
    fluid = ~solid
    masks = []
    for j in range(NQ5):
        masks.append(np.roll(solid, shift=(CY5[j], CX5[j]), axis=(0, 1)) & fluid)
    return masks


def apply_bounceback_scalar(g, gpost, masks) -> None:
    for j in range(NQ5):
        m = masks[j]
        if m.any():
            g[j][m] = gpost[OPP5[j]][m]


def inlet_zero_concentration(g: np.ndarray) -> None:
    """Clean-air Dirichlet inlet (C = 0) on the west column: g(:, :, 0) = 0."""
    g[:, :, 0] = 0.0


def open_outlet(g: np.ndarray) -> None:
    """Zero-gradient (open) scalar outlet on the east column."""
    g[:, :, -1] = g[:, :, -2]


def open_top(g: np.ndarray) -> None:
    """Zero-gradient (open) scalar top boundary."""
    g[:, -1, :] = g[:, -2, :]
