"""D2Q9 lattice-Boltzmann primitives: velocity set, equilibrium, collision, forcing.

All quantities here are in *lattice units* (dx = dt = 1, lattice speed of sound
c_s^2 = 1/3). Conversion to physical units is handled in :mod:`canyon_lbm.solver`.

Array operations go through the selected backend (:mod:`canyon_lbm.backend`),
so the same code runs on NumPy (CPU, the validated reference) or CuPy (GPU).
Per-direction constants come in two forms: plain Python tuples ``CXi/CYi/...``
(used for ``roll`` shifts, integer indexing, and scalar multiplies -- these must
stay host-side) and backend arrays ``C/CX/CY/W/OPP`` (used in vectorized ops).

Array layout convention used throughout the package
---------------------------------------------------
Distribution functions ``f`` have shape ``(9, Ny, Nx)``:
    axis 0 -> lattice direction i = 0..8
    axis 1 -> y (row),  increasing upward in physical space
    axis 2 -> x (col),  increasing in the streamwise (inlet->outlet) direction

D2Q9 velocity ordering (matches Zou/He 1997 and Mocz 2020)::

    i :   0    1    2    3    4    5    6    7    8
    cx:   0    1    0   -1    0    1   -1   -1    1
    cy:   0    0    1    0   -1    1    1   -1   -1
"""

from __future__ import annotations

from .backend import xp

# --- D2Q9 velocity set ------------------------------------------------------
# Host-side Python tuples: used for roll shifts, integer indexing, and as scalar
# multipliers (a Python int/float times a backend array stays on the backend).
CXi = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CYi = (0, 0, 1, 0, -1, 1, 1, -1, -1)
Wf = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
OPPi = (0, 3, 4, 1, 2, 7, 8, 5, 6)

# Backend arrays for vectorized use (and for tests on the NumPy backend).
C = xp.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
     [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
CX = C[:, 0]
CY = C[:, 1]
W = xp.array(Wf, dtype=float)
OPP = xp.array(OPPi, dtype=int)
_CXv = xp.asarray(CXi, dtype=float)[:, None, None]   # (9,1,1) for moments
_CYv = xp.asarray(CYi, dtype=float)[:, None, None]

CS2 = 1.0 / 3.0
NQ = 9


def equilibrium(rho, ux, uy):
    """Second-order (incompressible-NS) Maxwellian equilibrium distribution.

    f_i^eq = w_i * rho * [1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5 |u|^2]
    Returns ``feq`` of shape (9, Ny, Nx).
    """
    usq = ux * ux + uy * uy
    feq = xp.empty((NQ,) + rho.shape, dtype=float)
    for i in range(NQ):
        cu = CXi[i] * ux + CYi[i] * uy
        feq[i] = Wf[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq


def macroscopic(f, fx=0.0, fy=0.0):
    """Density and (force-corrected) velocity moments of ``f``.

    With a body force ``F`` present, the consistent fluid velocity is
    ``u = (sum_i c_i f_i + F/2) / rho`` (Guo et al. 2002).
    """
    rho = f.sum(axis=0)
    jx = (_CXv * f).sum(axis=0)
    jy = (_CYv * f).sum(axis=0)
    ux = (jx + 0.5 * fx) / rho
    uy = (jy + 0.5 * fy) / rho
    return rho, ux, uy


def guo_forcing(ux, uy, fx, fy, tau):
    """Guo (2002) forcing source term added during collision.

    F_i = (1 - 1/(2 tau)) * w_i * [ 3 (c_i - u).F + 9 (c_i.u)(c_i.F) ]
    """
    pref = 1.0 - 1.0 / (2.0 * tau)
    Fi = xp.empty((NQ,) + ux.shape, dtype=float)
    for i in range(NQ):
        ci_dot_u = CXi[i] * ux + CYi[i] * uy
        ci_dot_F = CXi[i] * fx + CYi[i] * fy
        ci_minus_u_dot_F = (CXi[i] - ux) * fx + (CYi[i] - uy) * fy
        Fi[i] = pref * Wf[i] * (3.0 * ci_minus_u_dot_F + 9.0 * ci_dot_u * ci_dot_F)
    return Fi


def collide_bgk(f, feq, tau, Fi=None):
    """Single-relaxation-time (BGK) collision, optionally with a forcing term.

    f_i^post = f_i - (f_i - f_i^eq)/tau [+ F_i]. ``tau`` may be a scalar or a
    per-cell (Ny, Nx) field (broadcast over the 9 directions).
    """
    if getattr(tau, "ndim", 0) == 2:
        tau = tau[None, :, :]
    fpost = f - (f - feq) / tau
    if Fi is not None:
        fpost = fpost + Fi
    return fpost


# --- MRT (multiple-relaxation-time) collision -------------------------------
#
# Moment basis (Lallemand & Luo 2000), ordering matched to C above:
#   m = (rho, e, eps, jx, qx, jy, qy, pxx, pxy)
# With every rate equal to 1/tau, MRT reduces exactly to BGK; the freedom in the
# non-hydrodynamic rates buys stability near tau -> 1/2 (high Reynolds number).
M_MRT = xp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=float,
)
MINV_MRT = xp.linalg.inv(M_MRT)


def collide_mrt(f, feq, s_nu, s_e=1.19, s_eps=1.4, s_q=1.2):
    """Multiple-relaxation-time collision.

    Relaxes the shear moments (pxx, pxy) at ``s_nu = 1/tau`` (the viscosity; may
    be a per-cell field for LES) and the non-hydrodynamic moments at the fixed
    "magic" rates. ``meq = M @ feq`` reuses the validated equilibrium.
    """
    m = xp.einsum("ij,jyx->iyx", M_MRT, f)
    meq = xp.einsum("ij,jyx->iyx", M_MRT, feq)
    dm = m - meq
    out = m.copy()
    out[1] -= s_e * dm[1]
    out[2] -= s_eps * dm[2]
    out[4] -= s_q * dm[4]
    out[6] -= s_q * dm[6]
    out[7] -= s_nu * dm[7]
    out[8] -= s_nu * dm[8]
    return xp.einsum("ij,jyx->iyx", MINV_MRT, out)


def smagorinsky_tau(ux, uy, tau0_field, Cs=0.16, fluid=None):
    """Effective relaxation time with a Smagorinsky sub-grid eddy viscosity.

    nu_t = (Cs * Delta)^2 |S|,  Delta = 1 cell,  |S| = sqrt(2 S_ij S_ij) from
    central differences of u. tau_eff = (nu0 + nu_t)/c_s^2 + 1/2, with nu0 from
    the base ``tau0_field`` (which already carries the outlet sponge). Velocities
    are zeroed in solids first so wall-adjacent strains are not contaminated.
    """
    if fluid is not None:
        ux = xp.where(fluid, ux, 0.0)
        uy = xp.where(fluid, uy, 0.0)
    dux_dx = xp.gradient(ux, axis=1)
    dux_dy = xp.gradient(ux, axis=0)
    duy_dx = xp.gradient(uy, axis=1)
    duy_dy = xp.gradient(uy, axis=0)
    sxy = 0.5 * (dux_dy + duy_dx)
    smag = xp.sqrt(2.0 * (dux_dx ** 2 + 2.0 * sxy ** 2 + duy_dy ** 2))
    nu0 = CS2 * (tau0_field - 0.5)
    nu_t = (Cs ** 2) * smag
    return (nu0 + nu_t) / CS2 + 0.5


def stream(fpost):
    """Advect each population by its lattice velocity (periodic via roll)."""
    f = xp.empty_like(fpost)
    for i in range(NQ):
        f[i] = xp.roll(fpost[i], shift=(CYi[i], CXi[i]), axis=(0, 1))
    return f


def viscosity_from_tau(tau):
    """Kinematic viscosity in lattice units: nu = c_s^2 (tau - 1/2)."""
    return CS2 * (tau - 0.5)


def tau_from_viscosity(nu):
    """Relaxation time from kinematic viscosity (inverse of viscosity_from_tau)."""
    return nu / CS2 + 0.5
