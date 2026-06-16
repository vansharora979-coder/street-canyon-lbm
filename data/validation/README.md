# CODASC validation data

This directory holds the wind-tunnel reference data used in the Phase 5
validation gate. The data are **not redistributed here**; obtain them directly
from the source so the licence/attribution stays with KIT.

## Source
**CODASC — Concentration Data of Street Canyons**, Karlsruhe Institute of
Technology (KIT), Laboratory of Building- and Environmental Aerodynamics.
Web: <https://www.codasc.de>

Underlying experiments: Gromke, C. & Ruck, B., wind-tunnel studies of traffic
pollutant dispersion in street canyons (isolated canyon, line source at street
level, various aspect ratios, wind directions, and tree configurations).

## Case we validate
- Geometry: **isolated street canyon, H/W = 1** (street width = building height).
- Trees: **none** (empty-canyon reference case).
- Wind direction: **90° (perpendicular)** — the configuration a 2D model can
  legitimately represent (the centre plane `y/L = 0` of the 3D tunnel).
- Quantity: **normalized concentration `c+`** on the leeward and windward canyon
  walls (vertical profiles at the canyon mid-length).

## Normalization (confirm against the CODASC docs before use)
The database reports a dimensionless concentration of the form

    c+ = c · U_H · H · L_src / Q

where `c` is concentration, `U_H` the reference velocity at building height,
`H` the building height, `L_src` the line-source length, and `Q` the source
strength. **Verify the exact definition and units in the CODASC documentation**
and match them in `canyon_lbm.metrics` before computing FAC2/NMSE/hit rate.

## How to add the data
1. Download the empty-canyon, 90°, H/W = 1 wall-concentration dataset from
   <https://www.codasc.de>.
2. Save it here as `codasc_AR1_notrees_90deg.csv` with columns documented in a
   short header (e.g. `wall, z_over_H, c_plus`).
3. `scripts/validate_codasc.py` (Phase 5) reads the path set in
   `configs/validation_codasc.yaml`.

## Acceptance (COST 732)
FAC2 ≥ 0.66 (primary), with NMSE and hit rate reported alongside. If the gate
fails, debug boundary conditions / normalization — **do not tune the answer**.
