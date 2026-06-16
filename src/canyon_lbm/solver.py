"""Time-stepping, unit conversion, and canonical-flow drivers.

Phase 1 provides:
  * :class:`LatticeUnits` -- explicit lattice <-> physical unit mapping.
  * :func:`run_poiseuille` -- forced channel flow used to validate the BGK core
    against the analytic parabolic profile.

The canyon driver (general inlet/outlet/top BCs + passive scalar) is added in
Phases 2-3 on top of the same :mod:`canyon_lbm.lattice` primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lattice as lb
from .boundary import apply_bounceback, precompute_bounceback


@dataclass(frozen=True)
class LatticeUnits:
    """Explicit lattice-units <-> physical-units mapping.

    The simulation is defined by three independent dimensionless/lattice
    choices and the physical reference scales:

    * ``n_cells`` : lattice cells resolving the reference length ``L_ref_phys``
      (here the building height H). Sets ``dx``.
    * ``u_lbm``   : lattice speed corresponding to ``u_ref_phys`` (keep <= 0.1
      for low Mach number / weak compressibility error). Sets ``dt``.
    * ``Re``      : target Reynolds number ``Re = u_ref * L_ref / nu``.

    From these, the lattice viscosity and relaxation time follow:
        nu_lbm = u_lbm * n_cells / Re
        tau    = nu_lbm / c_s^2 + 1/2 = 3 nu_lbm + 1/2
    """

    n_cells: int          # cells per reference length (building height H)
    u_lbm: float          # lattice velocity for u_ref_phys
    Re: float             # target Reynolds number
    L_ref_phys: float     # reference length [m] (building height H)
    u_ref_phys: float     # reference velocity [m/s] (U_H at building height)

    @property
    def dx(self) -> float:
        """Physical length of one lattice cell [m]."""
        return self.L_ref_phys / self.n_cells

    @property
    def dt(self) -> float:
        """Physical duration of one lattice step [s]."""
        return self.u_lbm * self.dx / self.u_ref_phys

    @property
    def nu_lbm(self) -> float:
        """Kinematic viscosity in lattice units."""
        return self.u_lbm * self.n_cells / self.Re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return lb.tau_from_viscosity(self.nu_lbm)

    @property
    def nu_phys(self) -> float:
        """Kinematic viscosity in physical units [m^2/s]."""
        return self.nu_lbm * self.dx * self.dx / self.dt

    def summary(self) -> dict:
        return {
            "n_cells": self.n_cells,
            "u_lbm": self.u_lbm,
            "Re": self.Re,
            "tau": self.tau,
            "nu_lbm": self.nu_lbm,
            "dx_m": self.dx,
            "dt_s": self.dt,
            "L_ref_m": self.L_ref_phys,
            "u_ref_mps": self.u_ref_phys,
            "nu_phys_m2s": self.nu_phys,
        }


def poiseuille_analytic(ny_fluid: int, g: float, nu: float) -> np.ndarray:
    """Analytic steady velocity for a force-driven channel (lattice units).

    With halfway bounce-back, the ``ny_fluid`` interior rows sit at distances
    ``y = r - 0.5`` (r = 1..ny_fluid) from the bottom wall plane, and the
    effective channel width is ``H = ny_fluid``. The momentum balance
    ``nu u'' = -g`` gives ``u(y) = g/(2 nu) * y * (H - y)``.

    Returns the analytic ``u_x`` at the interior fluid rows (length ny_fluid).
    """
    H = ny_fluid
    y = np.arange(1, ny_fluid + 1) - 0.5
    return g / (2.0 * nu) * y * (H - y)


def run_poiseuille(
    ny: int = 32,
    nx: int = 8,
    tau: float = 0.8,
    u_max: float = 0.05,
    max_iter: int = 200_000,
    tol: float = 1e-9,
    check_every: int = 200,
):
    """Force-driven Poiseuille channel; periodic in x, bounce-back walls in y.

    Walls are the first and last rows (solid); the ``ny - 2`` interior rows are
    fluid. A constant body force ``g`` is chosen so the analytic centreline
    speed equals ``u_max``. Runs to steady state (relative L2 change in u_x
    below ``tol``) or ``max_iter`` steps.

    Returns
    -------
    dict with keys: ``ux_profile`` (numeric, interior rows), ``ux_analytic``,
    ``y`` (row centres from bottom wall), ``g``, ``nu``, ``tau``, ``iters``,
    ``converged``, ``rel_l2_error``, ``max_rel_error``, ``mass0``, ``mass1``.
    """
    nu = lb.viscosity_from_tau(tau)
    ny_fluid = ny - 2
    H = ny_fluid
    # Body force so that g/(2 nu) * (H/2)^2 = u_max  ->  g = 8 nu u_max / H^2.
    g = 8.0 * nu * u_max / (H * H)

    # Geometry: solid top and bottom rows.
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    masks = precompute_bounceback(solid)

    fx = np.full((ny, nx), g, dtype=np.float64)
    fy = 0.0

    # Initialise at rest, rho = 1.
    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.zeros((ny, nx), dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    f = lb.equilibrium(rho, ux, uy)
    mass0 = float(f.sum())

    prev = ux.copy()
    converged = False
    iters = max_iter
    for it in range(1, max_iter + 1):
        rho, ux, uy = lb.macroscopic(f, fx, fy)
        feq = lb.equilibrium(rho, ux, uy)
        Fi = lb.guo_forcing(ux, uy, fx, fy, tau)
        fpost = lb.collide_bgk(f, feq, tau, Fi)
        f = lb.stream(fpost)
        apply_bounceback(f, fpost, masks)

        if it % check_every == 0:
            denom = np.linalg.norm(ux) + 1e-30
            change = np.linalg.norm(ux - prev) / denom
            prev = ux.copy()
            if change < tol:
                converged = True
                iters = it
                break

    rho, ux, uy = lb.macroscopic(f, fx, fy)
    mass1 = float(f.sum())

    ux_profile = ux[1:-1, nx // 2]
    ux_analytic = poiseuille_analytic(ny_fluid, g, nu)
    y = np.arange(1, ny_fluid + 1) - 0.5

    rel_l2 = float(
        np.linalg.norm(ux_profile - ux_analytic) / np.linalg.norm(ux_analytic)
    )
    max_rel = float(
        np.max(np.abs(ux_profile - ux_analytic)) / np.max(np.abs(ux_analytic))
    )

    return {
        "ux_profile": ux_profile,
        "ux_analytic": ux_analytic,
        "y": y,
        "g": g,
        "nu": nu,
        "tau": tau,
        "u_max": u_max,
        "iters": iters,
        "converged": converged,
        "rel_l2_error": rel_l2,
        "max_rel_error": max_rel,
        "mass0": mass0,
        "mass1": mass1,
    }
