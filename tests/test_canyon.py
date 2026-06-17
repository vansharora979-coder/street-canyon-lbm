"""Phase 2: canyon solver stability, mass balance, and single-vortex sense.

A small, short run (not fully converged) is enough to assert the qualitative
outcomes that appear early: it stays stable (no blow-up), inflow ~ outflow, and
the recirculation has the correct skimming-flow sense (downstream flow aloft,
reverse flow along the street floor).
"""

import numpy as np
import pytest

from canyon_lbm import lattice as lb
from canyon_lbm.boundary import log_law_profile
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation


@pytest.fixture(scope="module")
def small_canyon_run():
    # Low Reynolds number => genuinely steady laminar flow with a clean single
    # vortex (moderate Re sheds vortices and a short run gives unsteady snapshots).
    g = build_canyon(cells_per_H=16, aspect_ratio=1.0, building_width_H=1.0,
                     fetch_upstream_H=3.0, top_margin_H=3.0, outflow_H=6.0)
    u_lbm, Re = 0.04, 50.0
    tau = lb.tau_from_viscosity(u_lbm * g.h / Re)
    inlet = log_law_profile(g.ny, g.h, u_lbm, z0_cells=0.01 * g.h)
    # Outlet sponge suppresses the reflective-BC standing wave so the flow
    # actually reaches a clean steady state.
    sim = CanyonSimulation(g, tau, u_lbm, inlet, collision="bgk",
                           sponge_cells=3 * g.h, tau_sponge=1.0)
    out = sim.run(max_iter=40000, tol=1e-5, check_every=2000, ramp_iters=5000)
    return out


def test_canyon_runs_stably(small_canyon_run):
    out = small_canyon_run
    # run() raises on NaN / blow-up; reaching here means it stayed bounded.
    assert np.isfinite(out["cavity_circulation"])
    assert out["iters"] > 0


def test_canyon_mass_balance(small_canyon_run):
    # At/near steady state, streamwise inflow ~ outflow.
    assert small_canyon_run["mass_imbalance"] < 0.05


def test_canyon_single_clockwise_vortex(small_canyon_run):
    out = small_canyon_run
    # Robust skimming-flow signatures: one recirculation cell rotating clockwise
    # (negative area-integrated cavity vorticity for +x wind aloft) with reverse
    # flow along the street floor.
    assert out["cavity_circulation"] < 0          # clockwise
    assert out["clockwise"]
    assert out["floor_ux_over_uref"] < 0          # street-floor return flow
    assert out["centreline_sign_changes"] == 1    # single vortex (one ux reversal)
    assert out["single_vortex"]


def test_canyon_config_driven():
    """from_config builds geometry + tau from a parsed YAML-style dict."""
    cfg = {
        "type": "canyon",
        "resolution": {"cells_per_H": 16},
        "geometry": {"aspect_ratio": 2.0, "fetch_upstream_H": 2.0,
                     "top_margin_H": 3.0, "outflow_H": 3.0},
        "flow": {"Re": 200.0, "u_lbm": 0.05, "collision": "bgk"},
    }
    sim = CanyonSimulation.from_config(cfg)
    assert sim.geom.h == 16 and sim.geom.w == 8           # AR = 2 -> w = h/2
    nu = 0.05 * 16 / 200.0
    assert sim.tau == pytest.approx(lb.tau_from_viscosity(nu))
