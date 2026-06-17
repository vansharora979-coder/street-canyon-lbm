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

# --- Primary ventilation metrics (Phase 3) ---------------------------------
#
# The canyon clears the street-level pollutant by exchanging it across the roof
# opening. At a statistically-stationary state the removal rate equals the source
# rate Q, so the canyon's stored pollutant content settles at the level where
# Q = (removal). We report the outcome two equivalent ways:
#
#   * retention  -- the canyon-averaged concentration (higher => worse ventilation)
#   * ventilation -- a normalized air-exchange rate ACH* = w_e/U_H
#                    = Q * H / (content * U_H)   (higher => better ventilation)
#
# These are inverse measures (ACH* ∝ 1/content), as in the brief.


def canyon_content(C: np.ndarray, cavity_mask: np.ndarray) -> float:
    """Total pollutant stored in the canyon: sum of C over the cavity cells."""
    return float(C[cavity_mask].sum())


def canyon_mean_concentration(C: np.ndarray, cavity_mask: np.ndarray) -> float:
    """Canyon-averaged concentration (the retention measure, raw units)."""
    return float(C[cavity_mask].mean())


def ventilation_index(source_rate: float, content: float, h: int,
                      u_lbm: float) -> float:
    """Normalized air-exchange rate ACH* = w_e/U_H = Q*H / (content*U_H).

    ``source_rate`` Q is the pollutant injected per step (S * n_source_cells);
    ``content`` is the canyon pollutant content. Dimensionless (per advective
    timescale H/U_H). Higher => better ventilation.
    """
    return float(source_rate * h / (content * u_lbm))


def opening_flux(C: np.ndarray, uy: np.ndarray, opening_row: int,
                 street: tuple[int, int], D: float = 0.0) -> float:
    """Total scalar flux across the roof-opening plane: advective + diffusive.

    flux = sum_street [ C*uy  -  D (C[opening_row] - C[opening_row-1]) ]

    evaluated over the street columns at the first fluid row above the cavity.
    At a statistically-stationary state this balances the source rate Q -- a
    conservation check on the ventilation budget. At laminar/transitional Re the
    diffusive term dominates (the mean roof-level vertical velocity is ~0 in the
    recirculation), so it must be included.
    """
    s0, s1 = street
    adv = C[opening_row, s0:s1] * uy[opening_row, s0:s1]
    diff = -D * (C[opening_row, s0:s1] - C[opening_row - 1, s0:s1])
    return float(np.sum(adv + diff))


def codasc_cplus(C: np.ndarray, u_lbm: float, h: int, L_src: float,
                 Q: float) -> np.ndarray:
    """CODASC normalized concentration  c+ = C * U_H * H * L_src / Q.

    Implemented for the validation comparison (Phase 5). c+ rescales C by a
    constant, so the spatial pattern and the H/W trend are independent of the
    absolute source strength. The exact L_src / units convention is reconciled
    against the CODASC documentation in Phase 5 (see DECISIONS.md D7).
    """
    return C * (u_lbm * h * L_src) / Q


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
