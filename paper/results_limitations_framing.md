# NHSJS paper — Results & Limitations framing
*Working scaffold for the street-canyon ventilation study. Built around the verified steady-laminar model; the central contribution is the **Péclet control of canyon trapping**. All numbers below are from the repo (DECISIONS.md / PROGRESS.md) — confirm against the latest run outputs before submission.*

## Framing thesis (the one sentence the paper defends)
Using a verified, grid-converged 2-D lattice-Boltzmann model, we show that street-canyon pollutant **trapping and the leeward/windward concentration asymmetry are governed by the Péclet number** — i.e. by advective transport from the canyon vortex, not by diffusion — and we map how the ventilation metric varies with aspect ratio H/W through the Oke flow regimes. The claim is deliberately scoped to trends and mechanisms in an idealized 2-D laminar regime, the regime in which the solution is provably grid-independent.

This is a **proof-of-method + mechanism** paper. We do not claim absolute real-canyon concentrations; we claim a verified model, a reproduced regime structure, and an identified control parameter.

---

## Results (suggested order)

**R1 — Solver verification (short, establishes credibility).**
Body-force Poiseuille reproduces the analytic parabola to relative L2 = 4.45×10⁻⁴ (Guo forcing); the D2Q5 passive-scalar solver matches analytic advection–diffusion (pure-diffusion variance growth at ~0% error, mass conserved to machine precision; exact translation under uniform advection). Establishes that flow and scalar transport are correct before any canyon claim.

**R2 — Grid independence (the credibility gate).**
In the steady-laminar regime (Re = 25, BGK, no LES) the canyon ventilation metric is grid-independent: canyon-mean retention = 298.8 / 299.0 / 299.0 at 24 / 48 / 96 cells per building height (48→96 change = 0.0%, well inside the <3% gate). Production resolution = 48 cells/H. State plainly that this convergence is *achievable only in the steady regime* and forward-reference R5/Limitations for why turbulent and transitional runs do not converge.

**R3 — Flow-regime transition vs H/W (Oke reproduction).**
Across H/W ∈ {0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0}, the canyon organizes into a single trapped clockwise recirculation as H/W increases (judged by area-integrated cavity vorticity and street-floor reverse flow, per D12). This qualitatively reproduces Oke's (1988) regime sequence — isolated-roughness (H/W ≲ 0.3) → wake-interference (~0.3–0.7) → skimming (≳ 0.7); mark the skimming onset on the curve. Language: "qualitatively reproduces," "consistent with" — not "proves."

**R4 — The Péclet headline (the contribution).**
At the physical Schmidt number (Pe ≈ 18) scalar transport is diffusion-influenced and the leeward/windward asymmetry is weak (ratio ≈ 1.02). Raising the Péclet number drives advection-dominated transport and **recovers the asymmetry** (ratio ≈ 1.69 at Pe = 200, trending toward the wind-tunnel's ≈ 3.0). Interpretation: the canyon trapping/asymmetry is an **advection (Péclet) phenomenon carried by the recirculating vortex**, not a diffusive one. Present the Péclet-matched ventilation-vs-H/W curve (retention rising / ACH* falling into the skimming regime) as the main result — it shows the trapping *trend* via vortex transport while remaining grid-converged.

**R5 — CODASC comparison (honest, qualitative).**
The steady-laminar model captures the **vertical c⁺ decay structure** of the CODASC AR=1, 90°, no-trees case, but not the full magnitude/asymmetry, because the tunnel flow is 3-D and turbulent. Report FAC2 ≈ 0.79 explicitly as **not a rigorous pass** (coincidental, per D19); frame the comparison as qualitative validation of *structure* plus confirmation of the *Péclet mechanism*. Do not present a FAC2 number as if it met the 0.66 acceptance gate.

---

## Limitations (own them plainly — this is what NHSJS rewards)

1. **2-D idealization.** A 2-D model omits the three-dimensional along-canyon vortices and lateral exchange that dominate ventilation in real, finite-length canyons. Results are 2-D "infinite-canyon" trends.
2. **Reynolds regime.** Re = 25 lies below the Reynolds-independent range (≳ 10⁴) on which wind-tunnel/full-scale equivalence rests. We use it because it is the regime in which the solution is provably grid-converged (D6/D18); the flow is therefore a laminar analogue, and we claim trends/mechanisms, not absolute high-Re values.
3. **Why not turbulence/LES (state this as a finding, not an apology).** 2-D LES is grid-ill-posed here: the Smagorinsky sub-grid filter scales with cell size, so the metric does not converge (it moved ≈3× from n=24→48). Transitional laminar sheds above ~Re 40, so coarse grids fake a steady state and fine grids resolve the shedding — yielding a negative Richardson order. Reporting a turbulent result — even with a Grid Convergence Index "error bar" — would overclaim, because GCI presumes asymptotic convergence the solution does not have. We therefore restrict quantitative claims to the verified steady-laminar regime and present the LES divergence itself as a documented methodological result (see Fig. on grid behaviour).
4. **Péclet matching.** The asymmetry-recovery uses an elevated Péclet number to mimic advection-dominated transport; this isolates the mechanism but is not a calibration of turbulent diffusivity.
5. **Validation scope.** The CODASC comparison is qualitative (structure + mechanism), not a quantitative FAC2 pass.

---

## Discussion angle (1–2 paragraphs)
Position the contribution as **operationalizing** the trapping question: turning Oke's qualitative regimes and the informal "narrow streets trap pollution" intuition into a quantified, mechanism-level statement — that the trapping and wall asymmetry are *Péclet-controlled* — delivered with a verified, reproducible, open-source 2-D model. Then lay out the principled path to absolute values: a 3-D simulation with a steady RANS closure (k-ε / k-ω), which grid-converges and is the standard for CODASC-validated canyon work, as the natural next study.

---

## Suggested figures (NHSJS needs ≥5)
1. Canyon schematic defining H, W, H/W, source, and the roof-opening plane.
2. Verification panel: Poiseuille vs analytic + scalar ADE check.
3. Grid independence: converged laminar metric (24/48/96) **with an inset** showing the LES non-convergence (3× drift) — credibility + the methodological finding in one figure.
4. Flow-field panels across H/W with vorticity / streamlines, regime transition annotated.
5. **Headline:** ventilation (retention & ACH*) vs H/W at Péclet-matched conditions, Oke regime bands shaded, skimming onset marked.
6. Péclet/asymmetry recovery (ratio vs Pe: 1.02 → 1.69 …) beside the CODASC vertical-profile structure comparison.

---

## Language checklist (NHSJS reviewer pet peeves)
- No causal verbs the design can't support — use "associated with," "consistent with," "governed by" (the Péclet claim is a controlled-parameter result, so "governs/controls" is defensible there).
- Reserve "significant" for statistical significance only; otherwise "substantial," "marked."
- Every numeric claim carries its uncertainty or the resolution it was measured at.
- Say "qualitatively reproduces Oke (1988)," never "proves Oke."
- State the 2-D / Re=25 scope in the abstract itself, so no reviewer feels a claim was oversold.
