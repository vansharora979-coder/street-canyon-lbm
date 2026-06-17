"""D2Q9 core invariants: equilibrium moments, conservation, viscosity mapping."""

import numpy as np
import pytest

from canyon_lbm import lattice as lb


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_weights_and_velocities():
    assert lb.W.sum() == pytest.approx(1.0)
    # sum_i w_i c_i = 0
    assert np.allclose((lb.W[:, None] * lb.C).sum(axis=0), 0.0)
    # sum_i w_i c_ia c_ib = c_s^2 delta_ab
    second = np.einsum("i,ia,ib->ab", lb.W, lb.C, lb.C)
    assert np.allclose(second, lb.CS2 * np.eye(2))
    # opposite-of-opposite is identity
    assert np.array_equal(lb.OPP[lb.OPP], np.arange(9))


def test_equilibrium_sums_to_density():
    ny, nx = 6, 7
    rho = 1.0 + 0.1 * np.random.rand(ny, nx)
    ux = 0.05 * (np.random.rand(ny, nx) - 0.5)
    uy = 0.05 * (np.random.rand(ny, nx) - 0.5)
    feq = lb.equilibrium(rho, ux, uy)
    assert np.allclose(feq.sum(axis=0), rho, atol=1e-13)


def test_equilibrium_recovers_momentum():
    ny, nx = 6, 7
    rho = 1.0 + 0.1 * np.random.rand(ny, nx)
    ux = 0.05 * (np.random.rand(ny, nx) - 0.5)
    uy = 0.05 * (np.random.rand(ny, nx) - 0.5)
    feq = lb.equilibrium(rho, ux, uy)
    jx = (lb.CX[:, None, None] * feq).sum(axis=0)
    jy = (lb.CY[:, None, None] * feq).sum(axis=0)
    assert np.allclose(jx, rho * ux, atol=1e-13)
    assert np.allclose(jy, rho * uy, atol=1e-13)


def test_zero_velocity_equilibrium_is_weights_times_rho():
    rho = np.full((4, 4), 1.3)
    feq = lb.equilibrium(rho, np.zeros((4, 4)), np.zeros((4, 4)))
    for i in range(9):
        assert np.allclose(feq[i], lb.W[i] * rho)


def test_mass_conserved_periodic_collide_stream():
    """Collision + streaming with no walls conserves total mass to ~machine eps."""
    ny, nx = 16, 16
    rho = 1.0 + 0.05 * np.random.rand(ny, nx)
    ux = 0.05 * (np.random.rand(ny, nx) - 0.5)
    uy = 0.05 * (np.random.rand(ny, nx) - 0.5)
    f = lb.equilibrium(rho, ux, uy)
    m0 = f.sum()
    tau = 0.9
    for _ in range(50):
        r, u, v = lb.macroscopic(f)
        feq = lb.equilibrium(r, u, v)
        f = lb.stream(lb.collide_bgk(f, feq, tau))
    assert f.sum() == pytest.approx(m0, rel=0, abs=1e-9)


def test_viscosity_tau_roundtrip():
    for tau in (0.55, 0.8, 1.2):
        nu = lb.viscosity_from_tau(tau)
        assert lb.tau_from_viscosity(nu) == pytest.approx(tau)
    assert lb.viscosity_from_tau(0.8) == pytest.approx((0.8 - 0.5) / 3.0)


def _consistent_feq(f):
    rho = f.sum(0)
    ux = (lb.CX[:, None, None] * f).sum(0) / rho
    uy = (lb.CY[:, None, None] * f).sum(0) / rho
    return lb.equilibrium(rho, ux, uy)


def test_mrt_matrix_invertible():
    assert np.allclose(lb.M_MRT @ lb.MINV_MRT, np.eye(9))


def test_mrt_reduces_to_bgk_when_rates_equal():
    """With every relaxation rate = 1/tau, MRT is identical to BGK."""
    f = 0.1 + np.random.default_rng(1).random((9, 8, 9))
    feq = _consistent_feq(f)
    tau = 0.8
    bgk = lb.collide_bgk(f, feq, tau)
    mrt = lb.collide_mrt(f, feq, s_nu=1 / tau, s_e=1 / tau, s_eps=1 / tau, s_q=1 / tau)
    assert np.allclose(bgk, mrt, atol=1e-13)


def test_mrt_conserves_mass_and_momentum():
    f = 0.1 + np.random.default_rng(2).random((9, 8, 9))
    feq = _consistent_feq(f)
    post = lb.collide_mrt(f, feq, s_nu=1 / 0.7)  # magic non-hydro rates
    assert np.allclose(post.sum(0), f.sum(0))
    assert np.allclose((lb.CX[:, None, None] * post).sum(0),
                       (lb.CX[:, None, None] * f).sum(0))
    assert np.allclose((lb.CY[:, None, None] * post).sum(0),
                       (lb.CY[:, None, None] * f).sum(0))


def test_smagorinsky_adds_eddy_viscosity_only_in_shear():
    ny, nx = 8, 9
    tau0 = np.full((ny, nx), 0.51)
    # Uniform flow -> zero strain -> tau unchanged.
    uniform = lb.smagorinsky_tau(np.full((ny, nx), 0.05), np.zeros((ny, nx)),
                                 tau0, Cs=0.16)
    assert np.allclose(uniform, tau0)
    # Sheared flow -> positive eddy viscosity -> larger tau.
    yy = np.arange(ny)[:, None] * np.ones(nx)
    shear = lb.smagorinsky_tau(0.01 * yy, np.zeros((ny, nx)), tau0, Cs=0.16)
    assert np.all(shear >= tau0) and np.max(shear - tau0) > 0
