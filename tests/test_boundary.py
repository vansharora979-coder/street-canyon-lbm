"""Bounce-back bookkeeping and no-slip behaviour."""

import numpy as np

from canyon_lbm import lattice as lb
from canyon_lbm.boundary import precompute_bounceback


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
