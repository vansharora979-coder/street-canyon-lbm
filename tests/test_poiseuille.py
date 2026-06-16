"""Phase 1 acceptance gate: forced channel matches the analytic parabola."""

import numpy as np

from canyon_lbm.solver import run_poiseuille


def test_poiseuille_matches_analytic_within_1pct():
    res = run_poiseuille(ny=42, nx=8, tau=0.8, u_max=0.05, max_iter=200_000)
    assert res["converged"], "Poiseuille run did not reach steady state"
    # Phase 1 acceptance criterion: velocity within ~1% of analytic.
    assert res["rel_l2_error"] < 1e-2
    assert res["max_rel_error"] < 1e-2


def test_poiseuille_mass_conserved():
    res = run_poiseuille(ny=42, nx=8, tau=0.8, u_max=0.05, max_iter=200_000)
    rel_mass_change = abs(res["mass1"] - res["mass0"]) / res["mass0"]
    assert rel_mass_change < 1e-9


def test_poiseuille_centreline_speed():
    """Peak speed should match the target u_max used to set the body force."""
    res = run_poiseuille(ny=42, nx=8, tau=0.8, u_max=0.05, max_iter=200_000)
    assert np.isclose(np.max(res["ux_profile"]), 0.05, rtol=2e-2)
