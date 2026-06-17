"""Phase 3 acceptance: D2Q5 advection-diffusion scalar on known cases."""

import numpy as np
import pytest

from canyon_lbm import scalar as sc


def test_d2q5_weights_and_opposites():
    assert sc.W5.sum() == pytest.approx(1.0)
    second = np.einsum("i,ia,ib->ab", sc.W5, sc.C5, sc.C5)
    assert np.allclose(second, sc.CS2_5 * np.eye(2))
    assert np.array_equal(sc.OPP5[sc.OPP5], np.arange(5))


def test_equilibrium_sums_to_concentration():
    C = 1.0 + np.random.default_rng(0).random((6, 7))
    u = 0.05 * np.ones((6, 7))
    geq = sc.equilibrium_scalar(C, u, -u)
    assert np.allclose(geq.sum(axis=0), C)
    # First moment gives the advective flux C u.
    jx = (sc.CX5[:, None, None] * geq).sum(axis=0)
    assert np.allclose(jx, C * u)


def test_pure_diffusion_matches_analytic_variance():
    """A Gaussian under pure diffusion spreads as var = var0 + 2 D t (exactly)."""
    ny = nx = 120
    tau_g = 1.0
    D = sc.diffusivity_from_tau(tau_g)
    Y, X = np.mgrid[0:ny, 0:nx]
    s0 = 4.0
    C = np.exp(-((X - nx / 2) ** 2 + (Y - ny / 2) ** 2) / (2 * s0 ** 2))
    C /= C.sum()
    u = np.zeros((ny, nx))
    g = sc.equilibrium_scalar(C, u, u)
    N = 300
    for _ in range(N):
        Cn = sc.scalar_concentration(g)
        g = sc.stream_scalar(sc.collide_scalar(g, sc.equilibrium_scalar(Cn, u, u), tau_g))
    C = sc.scalar_concentration(g)
    m = C.sum()
    mx = (C * X).sum() / m
    varx = (C * (X - mx) ** 2).sum() / m
    expected = s0 ** 2 + 2 * D * N
    assert m == pytest.approx(1.0, abs=1e-9)            # mass conserved
    assert varx == pytest.approx(expected, rel=2e-3)    # diffusivity correct


def test_uniform_advection_translates_at_u():
    ny = nx = 160
    tau_g = 0.8
    ux0 = 0.05
    Y, X = np.mgrid[0:ny, 0:nx]
    C = np.exp(-((X - 40) ** 2 + (Y - 80) ** 2) / (2 * 5.0 ** 2))
    u = np.full((ny, nx), ux0)
    v = np.zeros((ny, nx))
    g = sc.equilibrium_scalar(C, u, v)
    N = 400
    for _ in range(N):
        Cn = sc.scalar_concentration(g)
        g = sc.stream_scalar(sc.collide_scalar(g, sc.equilibrium_scalar(Cn, u, v), tau_g))
    C = sc.scalar_concentration(g)
    mx = (C * X).sum() / C.sum()
    assert mx == pytest.approx(40 + ux0 * N, abs=1.0)


def test_source_injects_at_constant_rate():
    """With an insulating box and a source, total scalar grows by S*ncells/step."""
    ny = nx = 30
    solid = np.zeros((ny, nx), bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    masks = sc.precompute_bounceback_scalar(solid)
    src = np.zeros((ny, nx))
    src[ny // 2, nx // 2] = 1.0
    u = np.zeros((ny, nx))
    g = sc.equilibrium_scalar(np.zeros((ny, nx)), u, u)
    tau_g = 0.8
    for _ in range(20):
        C = sc.scalar_concentration(g)
        gpost = sc.collide_scalar(g, sc.equilibrium_scalar(C, u, u), tau_g, source=src)
        g = sc.stream_scalar(gpost)
        sc.apply_bounceback_scalar(g, gpost, masks)
    total = sc.scalar_concentration(g).sum()
    # 20 steps * 1.0 per step injected into a closed (insulating) box.
    assert total == pytest.approx(20.0, rel=1e-6)


def test_coupled_canyon_scalar_produces_sane_metrics():
    """A scalar-enabled canyon run: pollutant accumulates, ventilation metric is
    finite and positive, and concentration is higher at the street source than at
    the canyon opening (a physical gradient)."""
    from canyon_lbm import lattice as lb
    from canyon_lbm.boundary import log_law_profile
    from canyon_lbm.geometry import build_canyon
    from canyon_lbm.solver import CanyonSimulation

    g = build_canyon(cells_per_H=14, aspect_ratio=1.0, fetch_upstream_H=2.0,
                     top_margin_H=3.0, outflow_H=4.0)
    u_lbm, Re = 0.04, 60.0
    nu = u_lbm * g.h / Re
    tau = lb.tau_from_viscosity(nu)
    tau_g = sc.tau_from_schmidt(nu, 0.72)
    inlet = log_law_profile(g.ny, g.h, u_lbm, z0_cells=0.01 * g.h)
    sim = CanyonSimulation(g, tau, u_lbm, inlet, sponge_cells=2 * g.h,
                           with_scalar=True, tau_g=tau_g, source_strength=1.0)
    out = sim.run(max_iter=14000, tol=1e-5, check_every=2000, ramp_iters=4000,
                  average_from=7000)
    s = out["scalar"]
    assert np.isfinite(s["canyon_content"]) and s["canyon_content"] > 0
    assert np.isfinite(s["ventilation_index"]) and s["ventilation_index"] > 0
    assert s["retention_mean_conc"] > 0
    # Pollutant gradient: stronger at the street floor than at the opening.
    C = out["mean_C"]
    s0, s1 = g.street
    floor = C[g.source_row, s0:s1].mean()
    opening = C[g.roof_row, s0:s1].mean()
    assert floor > opening > 0
