"""Ventilation metrics and CODASC normalization.

Phase 3 deliverable. Computes the primary outcome (a dimensionless ventilation
measure) two equivalent ways, plus the CODASC normalized concentration used for
validation:

  * Pollutant retention: canyon-averaged normalized tracer concentration
    (higher = worse ventilation).
  * Air-exchange rate / exchange velocity: net tracer flux leaving across the
    roof-opening plane divided by canyon tracer content.
  * CODASC normalized concentration:  c+ = c * U_H * H * L_src / Q
    (exact form to be confirmed against codasc.de docs in Phase 5).

COST 732 validation metrics (Phase 5): FAC2, NMSE, hit rate.
"""

from __future__ import annotations

import numpy as np


def fac2(obs: np.ndarray, sim: np.ndarray) -> float:
    """Fraction of predictions within a factor of two of observations.

    FAC2 = fraction of points with 0.5 <= sim/obs <= 2.0. Standard COST 732
    metric; a common acceptance target is FAC2 >= 0.66. (Used in Phase 5.)
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = obs != 0
    ratio = np.full(obs.shape, np.nan)
    ratio[mask] = sim[mask] / obs[mask]
    within = (ratio >= 0.5) & (ratio <= 2.0)
    return float(np.count_nonzero(within) / np.count_nonzero(mask))


def nmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Normalized mean square error: <(obs-sim)^2> / (<obs><sim>)."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return float(np.mean((obs - sim) ** 2) / (np.mean(obs) * np.mean(sim)))


def hit_rate(obs: np.ndarray, sim: np.ndarray, dq: float = 0.25, w: float = 0.0) -> float:
    """COST 732 hit rate q: fraction of points within relative dq or absolute w."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    rel = np.abs((sim - obs) / np.where(obs != 0, obs, np.nan))
    absdiff = np.abs(sim - obs)
    hits = (rel <= dq) | (absdiff <= w)
    return float(np.count_nonzero(hits) / obs.size)
