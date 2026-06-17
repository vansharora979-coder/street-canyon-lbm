"""Bounce-back bookkeeping and no-slip behaviour."""

import numpy as np

from canyon_lbm import lattice as lb
from canyon_lbm.boundary import (
    apply_bounceback,
    freeslip_top,
    precompute_bounceback,
)


def test_bounceback_masks_single_solid():
    """For a lone solid node, direction-j bounce-back fires exactly at solid+c_j."""
    solid = np.zeros((5, 5), dtype=bool)
    solid[2, 2] = True
    masks = precompute_bounceback(solid)
    for j in range(9):
        m = masks[j]
        expected = np.zeros_like(solid)
        ty, tx = 2 + lb.CY[j], 2 + lb.CX[j]
        if j != 0:  # rest direction never bounces
            expected[ty, tx] = True
        assert np.array_equal(m, expected), f"direction {j}"


def test_bounceback_excludes_solid_nodes():
    """Bounce-back masks are only ever True on fluid nodes."""
    solid = np.zeros((6, 6), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    fluid = ~solid
    for m in precompute_bounceback(solid):
        assert not np.any(m & solid)
        assert np.all((~m) | fluid)


def test_poiseuille_no_slip_symmetry():
    """The converged channel profile is symmetric and ~zero at the walls."""
    from canyon_lbm.solver import run_poiseuille

    res = run_poiseuille(ny=34, nx=6, tau=0.8, u_max=0.04, max_iter=60_000)
    u = res["ux_profile"]
    # Symmetric about the channel centre.
    assert np.allclose(u, u[::-1], rtol=1e-3, atol=1e-6)
    # Near-wall speed is far below the centreline speed (no-slip enforced).
    assert u[0] < 0.1 * u.max()
    # Linear extrapolation to the wall plane (y = 0) is ~0.
    wall_extrap = 1.5 * u[0] - 0.5 * u[1]
    assert abs(wall_extrap) < 0.02 * u.max()


def test_freeslip_top_zero_penetration_and_zero_shear():
    """No-slip floor + free-slip top, body-force driven: a half-channel profile
    with zero wall-normal velocity and ~zero shear at the free-slip top."""
    ny, nx = 20, 12
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    masks = precompute_bounceback(solid)
    tau = 0.8
    nu = lb.viscosity_from_tau(tau)
    g = 2e-5
    fx = np.full((ny, nx), g)
    f = lb.equilibrium(np.ones((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx)))
    for _ in range(12000):
        rho, ux, uy = lb.macroscopic(f, fx, 0.0)
        feq = lb.equilibrium(rho, ux, uy)
        fpost = lb.collide_bgk(f, feq, tau, lb.guo_forcing(ux, uy, fx, 0.0, tau))
        f = lb.stream(fpost)
        apply_bounceback(f, fpost, masks)
        freeslip_top(f, fpost)
    rho, ux, uy = lb.macroscopic(f, fx, 0.0)
    # No flow through the free-slip top.
    assert np.abs(uy[~solid]).max() < 1e-6
    # Near-zero shear at the top (free-slip): du/dy small vs the profile scale.
    prof = ux[:, nx // 2]
    assert abs(prof[-1] - prof[-2]) < 0.02 * prof.max()
    # Peak velocity is at the top (no-slip bottom, free-slip top -> monotone rise).
    assert prof.argmax() >= ny - 2
