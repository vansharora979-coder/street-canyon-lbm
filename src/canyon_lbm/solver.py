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
from . import metrics as mt
from . import scalar as sc
from .backend import asnumpy, xp
from .boundary import (
    apply_bounceback,
    freeslip_top,
    inlet_velocity_neq,
    log_law_profile,
    outlet_pressure_neq,
    power_law_profile,
    precompute_bounceback,
)
from .geometry import CanyonGeometry, build_canyon


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


class CanyonSimulation:
    """D2Q9 flow solver for the street canyon (Phase 2).

    Wires the lattice primitives to the canyon geometry and boundary conditions:
    bounce-back walls (ground + buildings), a Zou/He velocity inlet with a
    log-law (or power-law) approach profile, a zero-gradient outflow, and a
    free-slip top. BGK collision; the MRT/Smagorinsky escalation (for higher Re)
    plugs into :meth:`_collide`.
    """

    def __init__(
        self,
        geom: CanyonGeometry,
        tau: float,
        u_lbm: float,
        inlet_ux: np.ndarray,
        collision: str = "bgk",
        sponge_cells: int = 0,
        tau_sponge: float = 1.0,
        with_scalar: bool = False,
        tau_g: float = 1.0,
        source_strength: float = 1.0,
        Cs: float = 0.16,
    ):
        self.geom = geom
        self.tau = float(tau)
        self.u_lbm = float(u_lbm)
        # State arrays live on the active backend (NumPy or CuPy/GPU).
        self.inlet_ux = xp.asarray(inlet_ux, dtype=float)
        self.inlet_uy = xp.zeros_like(self.inlet_ux)
        self.collision = collision
        # MRT non-hydrodynamic relaxation rates (d'Humieres 2002) + LES constant.
        self.s_e, self.s_eps, self.s_q = 1.19, 1.4, 1.2
        self.Cs = float(Cs)
        self.solid = xp.asarray(geom.solid)
        self.fluid = ~self.solid
        self.cavity_mask = xp.asarray(geom.cavity_mask)
        self.bb_masks = precompute_bounceback(self.solid)

        # Passive-scalar pollutant (D2Q5 advection-diffusion), one-way coupled.
        self.with_scalar = bool(with_scalar)
        if self.with_scalar:
            self.tau_g = float(tau_g)
            self.scalar_masks = sc.precompute_bounceback_scalar(self.solid)
            s0, s1 = geom.street
            self.source = xp.zeros((geom.ny, geom.nx), dtype=float)
            self.source[geom.source_row, s0:s1] = source_strength
            self.n_source_cells = s1 - s0
            self.source_rate = float(source_strength * self.n_source_cells)
            z = xp.zeros((geom.ny, geom.nx), dtype=float)
            self.g = sc.equilibrium_scalar(z, z, z)   # clean start, C = 0

        # Outlet viscosity sponge: tau ramps smoothly from the bulk value up to
        # tau_sponge over the last `sponge_cells` columns, absorbing the wake and
        # outgoing acoustic waves so they do not reflect off the outlet and
        # sustain a domain-spanning standing wave. Quadratic ramp -> no abrupt
        # impedance jump at the sponge entrance.
        self.sponge_cells = int(sponge_cells)
        self.tau_field = xp.full((geom.ny, geom.nx), self.tau, dtype=float)
        if self.sponge_cells > 0:
            x = xp.arange(geom.nx)
            start = geom.nx - self.sponge_cells
            s = xp.clip((x - start) / self.sponge_cells, 0.0, 1.0)
            self.tau_field[:, :] = self.tau + (tau_sponge - self.tau) * s[None, :] ** 2

        # Initialise from rest (rho = 1, u = 0). The inlet is ramped up smoothly
        # in run(), avoiding the startup shock an impulsive profile would slam
        # into the building faces (a classic low-tau BGK blow-up).
        rho = xp.ones((geom.ny, geom.nx), dtype=float)
        uz = xp.zeros((geom.ny, geom.nx), dtype=float)
        self.f = lb.equilibrium(rho, uz, uz)

    @classmethod
    def from_config(cls, config: dict) -> "CanyonSimulation":
        """Build a fully config-driven canyon simulation from a parsed YAML dict."""
        n = int(config["resolution"]["cells_per_H"])
        g = config.get("geometry", {})
        geom = build_canyon(
            cells_per_H=n,
            aspect_ratio=float(g.get("aspect_ratio", 1.0)),
            building_width_H=float(g.get("building_width_H", 1.0)),
            fetch_upstream_H=float(g.get("fetch_upstream_H", 5.0)),
            top_margin_H=float(g.get("top_margin_H", 6.0)),
            outflow_H=float(g.get("outflow_H", 15.0)),
        )
        flow = config.get("flow", {})
        u_lbm = float(flow.get("u_lbm", 0.05))
        Re = float(flow.get("Re", 1.0e3))
        nu = u_lbm * n / Re
        tau = lb.tau_from_viscosity(nu)
        if flow.get("inlet_profile", "log") == "power":
            inlet = power_law_profile(geom.ny, geom.h, u_lbm,
                                      alpha=float(flow.get("alpha", 0.25)))
        else:
            z0 = float(flow.get("z0_over_H", 0.01)) * n
            inlet = log_law_profile(geom.ny, geom.h, u_lbm, z0_cells=z0)
        sponge_cells = int(round(float(flow.get("sponge_H", 0.0)) * n))

        # Optional passive scalar (Phase 3): enabled by a `scalar` config block.
        scfg = config.get("scalar", {})
        with_scalar = bool(scfg) and scfg.get("enabled", True)
        tau_g = 1.0
        if with_scalar:
            schmidt = float(scfg.get("schmidt", 0.72))
            tau_g = sc.tau_from_schmidt(nu, schmidt)
        collision = flow.get("collision", "bgk")
        les = flow.get("les", {})
        if collision == "mrt" and les.get("enabled", False):
            collision = "mrt_les"
        Cs = float(les.get("Cs", 0.16))
        return cls(geom, tau, u_lbm, inlet, collision=collision,
                   sponge_cells=sponge_cells,
                   tau_sponge=float(flow.get("tau_sponge", 1.0)),
                   with_scalar=with_scalar, tau_g=tau_g,
                   source_strength=float(scfg.get("source_strength", 1.0)),
                   Cs=Cs)

    def _collide(self, f, feq, ux, uy):
        if self.collision == "bgk":
            return lb.collide_bgk(f, feq, self.tau_field)
        if self.collision in ("mrt", "mrt_les"):
            if self.collision == "mrt_les":
                tau_eff = lb.smagorinsky_tau(ux, uy, self.tau_field, self.Cs,
                                             self.fluid)
                s_nu = 1.0 / tau_eff
            else:
                s_nu = 1.0 / self.tau_field
            return lb.collide_mrt(f, feq, s_nu, self.s_e, self.s_eps, self.s_q)
        raise NotImplementedError(f"unknown collision '{self.collision}'")

    def step(self, inlet_scale: float = 1.0) -> None:
        f = self.f
        rho, ux, uy = lb.macroscopic(f)
        feq = lb.equilibrium(rho, ux, uy)
        fpost = self._collide(f, feq, ux, uy)
        fnew = lb.stream(fpost)
        apply_bounceback(fnew, fpost, self.bb_masks)       # ground + buildings
        inlet_velocity_neq(                                # inlet (west)
            fnew, self.inlet_ux * inlet_scale, self.inlet_uy
        )
        outlet_pressure_neq(fnew, rho_b=1.0)               # constant-pressure outlet
        freeslip_top(fnew, fpost)                          # free-slip top
        self.f = fnew

        if self.with_scalar:
            # Advect the pollutant with the just-updated velocity field.
            _, ux, uy = lb.macroscopic(self.f)
            C = sc.scalar_concentration(self.g)
            geq = sc.equilibrium_scalar(C, ux, uy)
            gpost = sc.collide_scalar(self.g, geq, self.tau_g, source=self.source)
            gnew = sc.stream_scalar(gpost)
            sc.apply_bounceback_scalar(gnew, gpost, self.scalar_masks)  # zero-flux walls
            sc.inlet_zero_concentration(gnew)              # clean air in
            sc.open_outlet(gnew)                           # pollutant advects out
            sc.open_top(gnew)
            self.g = gnew

    def macroscopic(self):
        return lb.macroscopic(self.f)

    def run(
        self,
        max_iter: int = 200_000,
        tol: float = 1e-7,
        check_every: int = 500,
        ramp_iters: int = 10_000,
        average_from: int | None = None,
        verbose: bool = False,
    ) -> dict:
        """Iterate the canyon flow; return fields + single-vortex diagnostics.

        The inlet velocity is linearly ramped from 0 to full over the first
        ``ramp_iters`` steps. Two termination modes:

        * ``average_from`` is None: iterate to a steady state (relative L2 change
          of the speed field below ``tol``) and report the final snapshot.
        * ``average_from`` set: the canyon flow is unsteady (vortex shedding), so
          run to ``max_iter`` and report the **time-averaged** mean field
          accumulated over ``[average_from, max_iter]`` -- the rigorous choice
          for a statistically-stationary flow. ``tol`` early-stop is disabled.

        Aborts on NaN / velocity blow-up.
        """
        prev = None
        converged = False
        iters = max_iter
        history = []
        averaging = average_from is not None
        sums = None
        n_avg = 0
        scal_C_sum = None
        scal_content_sum = 0.0
        scal_flux_sum = 0.0
        orow = self.geom.roof_row + 1  # first fluid row above the cavity
        s0, s1 = self.geom.street
        D_scal = sc.diffusivity_from_tau(self.tau_g) if self.with_scalar else 0.0
        for it in range(1, max_iter + 1):
            scale = min(1.0, it / ramp_iters) if ramp_iters > 0 else 1.0
            self.step(inlet_scale=scale)

            if averaging and it >= average_from:
                rho, ux, uy = self.macroscopic()
                if sums is None:
                    sums = [xp.zeros_like(rho) for _ in range(3)]
                sums[0] += rho; sums[1] += ux; sums[2] += uy
                n_avg += 1
                if self.with_scalar:
                    C = sc.scalar_concentration(self.g)
                    if scal_C_sum is None:
                        scal_C_sum = xp.zeros_like(C)
                    scal_C_sum += C
                    scal_content_sum += float(C[self.cavity_mask].sum())
                    # Total scalar flux across the roof opening = advective +
                    # diffusive (Fick). At this laminar Re the roof exchange is
                    # diffusion-dominated (mean vertical velocity ~0 in the
                    # recirculation), so both terms are needed for the budget.
                    adv = C[orow, s0:s1] * uy[orow, s0:s1]
                    diff = -D_scal * (C[orow, s0:s1] - C[orow - 1, s0:s1])
                    scal_flux_sum += float(xp.sum(adv + diff))

            if it % check_every == 0:
                _, ux, uy = self.macroscopic()
                speed = xp.sqrt(ux * ux + uy * uy)
                smax = float(xp.nanmax(speed))
                if not np.isfinite(smax) or smax > 0.4:
                    raise FloatingPointError(
                        f"Unstable at iter {it}: max speed = {smax:.3g} "
                        f"(tau={self.tau:.4f}). Lower Re / raise resolution / use MRT."
                    )
                cur = speed[self.fluid]
                if prev is not None and it > ramp_iters:
                    change = float(
                        xp.linalg.norm(cur - prev) / (xp.linalg.norm(cur) + 1e-30)
                    )
                    history.append((it, change))
                    if verbose:
                        tag = " [avg]" if (averaging and it >= average_from) else ""
                        extra = ""
                        if self.with_scalar:
                            content = float(
                                sc.scalar_concentration(self.g)[self.cavity_mask].sum()
                            )
                            extra = f"  content={content:.3e}"
                        print(f"  it={it:7d}  d(speed)={change:.3e}  umax={smax:.4f}"
                              f"{extra}{tag}", flush=True)
                    if not averaging and change < tol:
                        converged = True
                        iters = it
                        break
                prev = cur.copy()

        if averaging and n_avg > 0:
            mean = {"rho": sums[0] / n_avg, "ux": sums[1] / n_avg,
                    "uy": sums[2] / n_avg}
            diag = self.diagnostics(mean)
            diag.update({"averaged": True, "n_avg": n_avg,
                         "average_from": average_from})
            if self.with_scalar and scal_C_sum is not None:
                diag["scalar"] = self._scalar_metrics(
                    scal_content_sum / n_avg, scal_flux_sum / n_avg)
                diag["mean_C"] = asnumpy(scal_C_sum / n_avg)
        else:
            diag = self.diagnostics()
            diag.update({"averaged": False})
            if self.with_scalar:
                C = sc.scalar_concentration(self.g)
                content = float(C[self.cavity_mask].sum())
                diag["scalar"] = self._scalar_metrics(content, float("nan"))
                diag["mean_C"] = asnumpy(C)
        diag.update({"iters": iters, "converged": converged, "tau": self.tau,
                     "history": history})
        return diag

    def _scalar_metrics(self, content: float, opening_flux: float) -> dict:
        """Ventilation metrics from the (time-averaged) canyon pollutant content."""
        g = self.geom
        n_cav = int(g.cavity_mask.sum())
        vent = mt.ventilation_index(self.source_rate, content, g.h, self.u_lbm)
        return {
            "canyon_content": content,
            "retention_mean_conc": content / max(n_cav, 1),
            "ventilation_index": vent,                  # ACH* = w_e/U_H
            "opening_flux": opening_flux,
            "source_rate": self.source_rate,
            "flux_over_source": (opening_flux / self.source_rate
                                 if self.source_rate else float("nan")),
            "tau_g": self.tau_g,
        }

    def diagnostics(self, fields: dict | None = None) -> dict:
        """Mass balance and single-vortex diagnostics for a velocity field.

        Pass ``fields`` (dict with ``rho, ux, uy``) to diagnose a time-averaged
        field; otherwise the current instantaneous state is used.

        Velocities inside solid cells are non-physical (they come from
        bounced-back populations), so they are zeroed before any spatial
        gradient or cavity average — otherwise wall-adjacent gradients are
        contaminated and the circulation sign is unreliable.
        """
        geom = self.geom
        if fields is None:
            rho, ux, uy = self.macroscopic()
        else:
            rho, ux, uy = fields["rho"], fields["ux"], fields["uy"]
        ux = xp.where(self.fluid, ux, 0.0)
        uy = xp.where(self.fluid, uy, 0.0)

        # Mass balance: streamwise mass flux near the inlet vs near the outlet.
        inflow = float(xp.sum((rho * ux)[:, 1][self.fluid[:, 1]]))
        outflow = float(xp.sum((rho * ux)[:, -2][self.fluid[:, -2]]))
        mass_imbalance = abs(inflow - outflow) / (abs(inflow) + 1e-30)

        # Canyon-cavity vorticity (clockwise => negative w_z for +x wind aloft).
        wz = xp.gradient(uy, axis=1) - xp.gradient(ux, axis=0)
        cav = self.cavity_mask
        circulation = float(xp.sum(wz[cav]))
        u_ref = max(self.u_lbm, 1e-12)

        # Rotation sense: the area-integrated cavity vorticity is the robust
        # measure (negative => clockwise, the skimming-flow sense for +x wind
        # aloft). Pointwise upper/lower samples are reported for context but are
        # unreliable near the roof / vortex centre at low resolution.
        h = geom.h
        s0, s1 = geom.street
        cols = slice(s0 + 1, s1 - 1)
        ux_upper = float(xp.mean(ux[2 * h // 3 : h, cols]))
        ux_lower = float(xp.mean(ux[1 : max(2, h // 3), cols]))
        clockwise = bool(circulation < 0)

        floor_ux = float(xp.mean(ux[geom.source_row, s0:s1]))  # street-floor reverse flow
        xc = (s0 + s1) // 2
        col = ux[1 : h + 1, xc]
        nz = col[xp.abs(col) > 1e-9 * u_ref]
        sign_changes = int(xp.sum(xp.diff(xp.sign(nz)) != 0)) if int(nz.size) else 0

        return {
            "inflow": inflow,
            "outflow": outflow,
            "mass_imbalance": mass_imbalance,
            "cavity_circulation": circulation,
            "ux_upper_over_uref": ux_upper / u_ref,
            "ux_lower_over_uref": ux_lower / u_ref,
            "floor_ux_over_uref": floor_ux / u_ref,
            "centreline_sign_changes": sign_changes,
            "clockwise": clockwise,
            # One recirculation cell, rotating clockwise, with reverse flow along
            # the street floor -- the hallmarks of the skimming-flow vortex.
            "single_vortex": bool(sign_changes == 1 and clockwise and floor_ux < 0),
            "fields": {"rho": asnumpy(rho), "ux": asnumpy(ux), "uy": asnumpy(uy)},
        }

