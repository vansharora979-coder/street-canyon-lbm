"""Array-backend selection: NumPy (default, the validated reference) or CuPy (GPU).

The backend is chosen once at import time from the ``CANYON_LBM_BACKEND``
environment variable (values ``numpy`` | ``cupy``). This keeps the test suite and
all validation on NumPy while letting the expensive production sweeps run on the
GPU by setting the variable before importing the package:

    CANYON_LBM_BACKEND=cupy python scripts/run_sweep.py ...

The two backends run identical math (the solver is pure array operations), so a
GPU result must match the NumPy reference to round-off -- this is checked in the
tests and before any production GPU run.

Use ``xp`` for array operations, ``asnumpy(a)`` to bring a result back to the
host (e.g. for saving or plotting), and ``GPU`` to branch only when unavoidable.
"""

from __future__ import annotations

import os

_name = os.environ.get("CANYON_LBM_BACKEND", "numpy").lower()

if _name == "cupy":
    import cupy as xp  # type: ignore

    GPU = True
elif _name == "numpy":
    import numpy as xp

    GPU = False
else:  # pragma: no cover
    raise ValueError(
        f"CANYON_LBM_BACKEND must be 'numpy' or 'cupy', got '{_name}'."
    )


def asnumpy(a):
    """Return ``a`` as a host NumPy array (no-op on the NumPy backend)."""
    if GPU:
        return xp.asnumpy(a)
    return a


def backend_name() -> str:
    return "cupy" if GPU else "numpy"
