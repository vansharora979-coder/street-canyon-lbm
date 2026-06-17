"""Ventilation metrics (Phase 3) and COST 732 validation metrics (Phase 5)."""

import numpy as np
import pytest

from canyon_lbm.metrics import (
    canyon_content,
    canyon_mean_concentration,
    codasc_cplus,
    fac2,
    hit_rate,
    nmse,
    ventilation_index,
)


def test_canyon_content_and_mean():
    C = np.zeros((5, 5))
    cav = np.zeros((5, 5), bool)
    cav[1:4, 1:4] = True
    C[cav] = 2.0
    assert canyon_content(C, cav) == pytest.approx(2.0 * 9)
    assert canyon_mean_concentration(C, cav) == pytest.approx(2.0)


def test_ventilation_index_inverse_to_content():
    # ACH* = Q H / (content U).  Doubling content halves ventilation.
    v1 = ventilation_index(source_rate=10.0, content=100.0, h=20, u_lbm=0.05)
    v2 = ventilation_index(source_rate=10.0, content=200.0, h=20, u_lbm=0.05)
    assert v1 == pytest.approx(10.0 * 20 / (100.0 * 0.05))
    assert v2 == pytest.approx(v1 / 2)


def test_codasc_cplus_is_linear_rescaling():
    C = np.array([1.0, 2.0, 4.0])
    cp = codasc_cplus(C, u_lbm=0.05, h=20, L_src=3.0, Q=6.0)
    assert np.allclose(cp, C * (0.05 * 20 * 3.0) / 6.0)


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
