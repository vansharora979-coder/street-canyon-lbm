"""Passive-scalar pollutant transport (D2Q5 advection-diffusion LBM).

Phase 3 deliverable. A separate D2Q5 population g_i is advected by the velocity
field from the D2Q9 flow solver and relaxed toward its advection-diffusion
equilibrium, with a continuous line/area source at street level.

Planned API
-----------
d2q5_equilibrium(c, ux, uy) -> geq
collide_stream_scalar(g, ux, uy, tau_g, source, solid_masks) -> g
The scalar diffusivity is D = c_s^2 (tau_g - 1/2) with c_s^2 = 1/3 (D2Q5).
"""

from __future__ import annotations


def d2q5_equilibrium(*args, **kwargs):  # pragma: no cover - Phase 3
    raise NotImplementedError(
        "Passive scalar transport is implemented in Phase 3 (see PROGRESS.md)."
    )
