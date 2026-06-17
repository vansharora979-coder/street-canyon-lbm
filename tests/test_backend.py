"""Backend abstraction: NumPy default, and CuPy reproduces NumPy to round-off."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from canyon_lbm import backend

# A tiny, deterministic, laminar canyon run that prints a checksum. Laminar (low
# Re) so round-off does not amplify chaotically -- the two backends must agree.
_SNIPPET = """
import numpy as np
from canyon_lbm.backend import asnumpy
from canyon_lbm.geometry import build_canyon
from canyon_lbm.solver import CanyonSimulation
from canyon_lbm import lattice as lb, scalar as sc
from canyon_lbm.boundary import log_law_profile
g = build_canyon(14, 1.0, fetch_upstream_H=2, top_margin_H=3, outflow_H=3)
u, Re = 0.05, 40.0
nu = u*g.h/Re; tau = lb.tau_from_viscosity(nu); tg = sc.tau_from_schmidt(nu, 0.72)
inlet = log_law_profile(g.ny, g.h, u, z0_cells=0.14)
sim = CanyonSimulation(g, tau, u, inlet, collision="mrt_les", sponge_cells=2*g.h,
                       with_scalar=True, tau_g=tg, Cs=0.16)
for it in range(120):
    sim.step(min(1.0, it/60))
_, ux, uy = lb.macroscopic(sim.f); C = sc.scalar_concentration(sim.g)
print(repr((float(np.abs(asnumpy(ux)).sum()), float(asnumpy(C).sum()))))
"""


def _run(env_backend):
    import os
    env = dict(os.environ, CANYON_LBM_BACKEND=env_backend)
    out = subprocess.check_output([sys.executable, "-c", _SNIPPET], env=env)
    return eval(out.decode().strip())


def test_default_backend_is_numpy():
    assert backend.backend_name() == "numpy"
    a = np.arange(3)
    assert backend.asnumpy(a) is a  # no-op on the host backend


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None,
                    reason="CuPy not installed (CPU-only environment)")
def test_cupy_reproduces_numpy():
    cpu = _run("numpy")
    gpu = _run("cupy")
    assert np.allclose(cpu, gpu, rtol=1e-9, atol=1e-9), f"cpu={cpu} gpu={gpu}"
