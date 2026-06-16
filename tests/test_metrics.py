"""COST 732 validation metrics (used as a hard gate in Phase 5)."""

import numpy as np

from canyon_lbm.metrics import fac2, hit_rate, nmse


def test_fac2_perfect_and_boundary():
    obs = np.array([1.0, 2.0, 4.0, 8.0])
    assert fac2(obs, obs) == 1.0
    # Exactly a factor of two is inside the band.
    assert fac2(obs, 2.0 * obs) == 1.0
    assert fac2(obs, 0.5 * obs) == 1.0
    # Just outside the band on every point.
    assert fac2(obs, 2.0001 * obs) == 0.0


def test_nmse_zero_for_identical():
    obs = np.array([1.0, 3.0, 5.0])
    assert nmse(obs, obs) == 0.0
    assert nmse(obs, obs * 1.0 + 0.5) > 0.0


def test_hit_rate_bounds():
    obs = np.array([1.0, 2.0, 4.0])
    assert hit_rate(obs, obs) == 1.0
    # 30% off with dq=0.25 and no absolute allowance -> all miss.
    assert hit_rate(obs, obs * 1.3, dq=0.25, w=0.0) == 0.0
