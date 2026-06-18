# CODASC validation data

The Phase 5 validation compares against the **CODASC** (Concentration Data of
Street Canyons) wind-tunnel database — Gromke & Ruck, KIT Laboratory of
Building- and Environmental Aerodynamics. The data are **KIT's; not redistributed
here** (git-ignored). Download them yourself with the commands below.

## Source
Canonical address: <https://www.codasc.de> — but that host currently has a TLS
(SNI) misconfiguration, so fetch from the lab's mirror, which is where it
redirects:
`https://www.umweltaerodynamik.de/bilder-originale/CODA/`

## The case we validate
Isolated street canyon, **H/W = 1**, **no trees**, **90° (perpendicular) wind**,
street-level line source. File naming: `AR_winddir_tree_density_wall`, so the
reference case is `1_90_0,0_000` with walls `A` and `B`.

## Download (run from the repo root)
```bash
base="https://www.umweltaerodynamik.de/bilder-originale/CODA/conzdata_dat"
curl -L -A "Mozilla/5.0" -o data/validation/codasc_AR1_notrees_90deg_A.txt "$base/1_90_0,0_000_A.txt"
curl -L -A "Mozilla/5.0" -o data/validation/codasc_AR1_notrees_90deg_B.txt "$base/1_90_0,0_000_B.txt"
```
(Excel versions are under `.../condata_xls/1_90_0,0_000_A.xls`.)

## File format
Tab-separated, header `"y/H" "z/H" "c+"`, 700 grid points = 100 along-canyon
positions (`y/H` ∈ [−5, 5]) × 7 wall heights (`z/H` ∈ {0, 0.167, …, 1.0}).
- **Wall A = leeward** (pollutant accumulates; c⁺ up to ~42 at street level).
- **Wall B = windward** (c⁺ up to ~12).
The 2-D model compares to the canyon **mid-length** profile (`y/H ≈ 0`).

## c⁺ normalization (from the CODASC docs)
`c⁺ = cₘ · U_H · H / Q_l`  (= c · U_H · H · L_src / Q), with cₘ the measured
volume concentration, U_H the velocity at building height, H the building height,
and Q_l the line-source emission rate per unit length. c⁺ is sampled at
**x⁺ = 1/24 ≈ 0.0417 (x/H)** in front of each wall — `validate_codasc.py` samples
the model at the same offset.

## Honest scope (Phase 5)
The tunnel is **turbulent** (Re ≈ 3.7×10⁴); this study runs a **steady-laminar**
model (Re = 25 — the regime where grid-independence is well-posed, Phase 4). The
absolute c⁺ is therefore **not** expected to match. The validation target is the
**pattern**: the leeward≫windward asymmetry and the vertical c⁺ decay, both set
by the canyon vortex. COST 732 metrics (FAC2, NMSE, hit rate) are reported
descriptively, not as a strict pass/fail.
