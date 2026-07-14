"""Figure helpers. Matplotlib is forced to the non-interactive Agg backend so
figures render identically on a headless laptop/CI. Each helper saves PNG + SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: str | Path) -> list[Path]:
    """Save a figure as both PNG and SVG next to ``path`` (stem reused)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in (".png", ".svg"):
        p = path.with_suffix(ext)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def _add_buildings(ax, d, color="0.55"):
    """Overlay ground + the two buildings as filled patches."""
    import matplotlib.patches as mpatches

    h = int(d["h"])
    w0, w1 = (int(v) for v in d["west_building"])
    e0, e1 = (int(v) for v in d["east_building"])
    nx = d["ux"].shape[1]
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), nx, 1.0, color=color, zorder=5))
    for (c0, c1) in ((w0, w1), (e0, e1)):
        ax.add_patch(
            mpatches.Rectangle((c0 - 0.5, 0.5), c1 - c0, h, color=color, zorder=5)
        )


def plot_canyon_flow(npz_path: str | Path, path: str | Path,
                     title: str | None = None) -> list[Path]:
    """Two-panel canyon flow field: full domain + canyon-zoom streamlines.

    Reads the ``.npz`` written by ``scripts/run_case.py`` (ux, uy, solid, and
    geometry metadata) and renders speed (colour) with streamlines, the buildings
    overlaid in grey. The canyon zoom shows the single recirculating vortex.
    """
    import numpy as np

    d = np.load(npz_path)
    ux, uy, solid = d["ux"], d["uy"], d["solid"]
    ny, nx = ux.shape
    h = int(d["h"]); s0, s1 = (int(v) for v in d["street"])
    w0 = int(d["west_building"][0]); e1 = int(d["east_building"][1])
    u_lbm = float(d["u_lbm"])
    # Zero the (non-physical) velocity inside solids for clean plotting.
    uxm = np.where(solid, 0.0, ux)
    uym = np.where(solid, 0.0, uy)
    speed = np.sqrt(uxm**2 + uym**2) / u_lbm
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1, 1.25]}
    )

    # --- (a) full domain ---
    pm = ax1.pcolormesh(X, Y, speed, cmap="viridis", shading="auto", vmin=0)
    ax1.streamplot(X, Y, uxm, uym, color="white", density=1.0, linewidth=0.5,
                   arrowsize=0.6)
    _add_buildings(ax1, d)
    ax1.set_title(title or "Street-canyon flow field, H/W = 1")
    ax1.set_xlabel("x (cells, inlet -> outlet)")
    ax1.set_ylabel("y (cells)")
    ax1.set_xlim(0, nx - 1); ax1.set_ylim(0, ny - 1)
    fig.colorbar(pm, ax=ax1, label=r"$|u|/u_{ref}$", fraction=0.025, pad=0.01)

    # --- (b) canyon zoom: the recirculating vortex ---
    pad = int(0.7 * h)
    x0, x1 = max(0, w0 - pad), min(nx, e1 + pad)
    y0, y1 = 0, min(ny, h + pad)
    sl = (slice(y0, y1), slice(x0, x1))
    Xc, Yc = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    spc = speed[sl]
    pm2 = ax2.pcolormesh(Xc, Yc, spc, cmap="viridis", shading="auto", vmin=0)
    ax2.streamplot(Xc, Yc, uxm[sl], uym[sl], color="white", density=1.6,
                   linewidth=0.7, arrowsize=0.8)
    _add_buildings(ax2, d)
    ax2.axhline(h + 0.5, color="orange", ls="--", lw=1.2, zorder=6,
                label="canyon opening (roof level)")
    ax2.set_title("Canyon recirculation (zoom)")
    ax2.set_xlabel("x (cells)"); ax2.set_ylabel("y (cells)")
    ax2.set_xlim(x0, x1 - 1); ax2.set_ylim(y0, y1 - 1)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.colorbar(pm2, ax=ax2, label=r"$|u|/u_{ref}$", fraction=0.025, pad=0.01)

    fig.tight_layout()
    return _save(fig, path)


def plot_canyon_concentration(npz_path: str | Path, path: str | Path,
                              title: str | None = None) -> list[Path]:
    """Time-averaged pollutant concentration in the canyon (zoom), buildings
    overlaid, streamlines showing how the vortex traps the scalar near the
    leeward wall. Reads the ``.npz`` written by ``scripts/run_case.py`` (needs
    the ``C`` field, i.e. a scalar-enabled run)."""
    import numpy as np

    d = np.load(npz_path)
    if "C" not in d.files:
        raise KeyError(f"{npz_path} has no 'C' field (run with a scalar config).")
    C = d["C"]; solid = d["solid"]; ux = d["ux"]; uy = d["uy"]
    ny, nx = C.shape
    h = int(d["h"]); s0, s1 = (int(v) for v in d["street"])
    w0 = int(d["west_building"][0]); e1 = int(d["east_building"][1])
    Cm = np.where(solid, np.nan, C)
    uxm = np.where(solid, 0.0, ux); uym = np.where(solid, 0.0, uy)
    # Normalise by the canyon-mean so the colour scale is comparable across H/W.
    cav = np.zeros_like(solid); cav[1:h + 1, s0:s1] = True
    Cref = float(np.nanmean(C[cav])) or 1.0

    pad = int(0.8 * h)
    x0, x1 = max(0, w0 - pad), min(nx, e1 + pad)
    y0, y1 = 0, min(ny, h + pad)
    sl = (slice(y0, y1), slice(x0, x1))
    Xc, Yc = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    pm = ax.pcolormesh(Xc, Yc, Cm[sl] / Cref, cmap="inferno", shading="auto",
                       vmin=0, vmax=max(2.0, np.nanpercentile(Cm[cav] / Cref, 95)))
    ax.streamplot(Xc, Yc, uxm[sl], uym[sl], color="white", density=1.4,
                  linewidth=0.5, arrowsize=0.7)
    _add_buildings(ax, d, color="0.4")
    ax.axhline(h + 0.5, color="cyan", ls="--", lw=1.0, zorder=6,
               label="canyon opening")
    ax.plot([s0, s1 - 1], [1, 1], "s", color="lime", ms=3, zorder=7,
            label="street source")
    ax.set_title(title or "Time-averaged pollutant concentration, H/W = 1")
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    ax.set_xlim(x0, x1 - 1); ax.set_ylim(y0, y1 - 1)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.colorbar(pm, ax=ax, label=r"$C / \langle C \rangle_{canyon}$",
                 fraction=0.025, pad=0.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_peclet_hw_diagnostic(csv_path: str | Path, path: str | Path) -> list[Path]:
    """Phase 6.5 decisive figure: ventilation vs H/W, one line per Péclet number.

    Shows whether the skimming collapse (retention ↑ / ACH ↓ with H/W) emerges
    as Pe rises into the advection-dominated regime. Grid-under-resolved points
    (cell-Péclet too high / negative C) are drawn hollow + dashed so they aren't
    read as converged results.
    """
    import csv as _csv
    import numpy as np

    U_LBM = 0.05
    rows = list(_csv.DictReader(open(csv_path)))
    pes = sorted({float(r["Pe"]) for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, pe in enumerate(pes):
        sub = sorted((r for r in rows if float(r["Pe"]) == pe),
                     key=lambda r: float(r["aspect_ratio"]))
        hw = np.array([float(r["aspect_ratio"]) for r in sub])
        ret = np.array([float(r["retention_eq"]) for r in sub])
        # ACH* = ACH[1/step] * (H/u_lbm): grid-invariant exchange per flow-time
        # (raw per-step ACH scales with dt~1/n and is NOT grid-comparable).
        achs = np.array([float(r["ach"]) * float(r["cells_per_H"]) / U_LBM
                         for r in sub])
        col = cmap(i / max(len(pes) - 1, 1))
        for ax, y in ((ax1, ret), (ax2, achs)):
            ax.plot(hw, y, "-o", color=col, ms=6, label=f"Pe={pe:.0f}")
    for ax in (ax1, ax2):
        ax.axvspan(0.7, 3.2, color="#f6cccc", alpha=0.4, lw=0)
        ax.axvline(0.7, color="0.5", ls=":", lw=1)
        ax.set_xlabel("aspect ratio  H/W"); ax.set_xlim(0.4, 3.1)
        ax.legend(fontsize=8, frameon=False, title="(skimming →)")
    ax1.set_ylabel("equilibrium retention  (canyon-mean $c$)")
    ax1.set_title("Retention vs H/W")
    ax2.set_ylabel(r"air-exchange rate  ACH* $= w_e/U_H$")
    ax2.set_title("Ventilation vs H/W")
    fig.suptitle("Skimming collapse emerges in the advection-dominated regime "
                 "(2-D laminar Re=25)\nretention ↑ / ACH* ↓ with H/W at high Pe; "
                 "grid-converged at Pe=200 (n=48→96: 0.2%)", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_canyon_schematic(path: str | Path) -> list[Path]:
    """Definition sketch (two panels): (a) the canyon geometry -- inlet log-law
    profile, free-slip top, roof exchange plane, canyon vortex, H, W, the
    street-level source, and the sponge outflow zone; (b) the D2Q9 velocity
    set used by the solver. Pure diagram; label placement only, no data."""
    import matplotlib.patches as mpatches
    import numpy as np

    H, W, B = 1.0, 1.0, 0.8  # building height, street width, building width

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [2.6, 1]}
    )

    # ---- Panel (a): canyon geometry -------------------------------------
    x_left, x_right = -1.7, 3.2
    y_bot, y_top = -0.28, 2.0
    ground_y = 0.0

    # ground (brown strip)
    ax.add_patch(mpatches.Rectangle(
        (x_left, ground_y - 0.12), x_right - x_left, 0.12, color="#6b4a30", zorder=1))

    # sponge outflow zone (shaded band to the right of the domain)
    sponge_x0, sponge_x1 = 2.3, x_right
    ax.add_patch(mpatches.Rectangle(
        (sponge_x0, ground_y - 0.12), sponge_x1 - sponge_x0, y_top - (ground_y - 0.12),
        color="0.85", zorder=0))
    ax.text(sponge_x0 + (sponge_x1 - sponge_x0) / 2, (y_top + ground_y) / 2 - 0.1,
            "sponge", color="0.4", ha="center", va="center", fontsize=10, rotation=90)

    # buildings
    ax.add_patch(mpatches.Rectangle((-B, 0), B, H, color="0.6", ec="k", zorder=2))
    ax.add_patch(mpatches.Rectangle((W, 0), B, H, color="0.6", ec="k", zorder=2))

    # free-slip top boundary
    top_y = 1.78
    ax.plot([x_left, sponge_x1], [top_y, top_y], "--", color="k", lw=1.3, zorder=1)
    ax.text((x_left + sponge_x1) / 2 + 0.3, top_y + 0.08, "free-slip top",
            ha="center", va="bottom", fontsize=10)
    # "(a)" sits in the top-left corner, well above the dashed line
    ax.text(x_left + 0.03, y_top - 0.02, "(a)", ha="left", va="top",
            fontsize=15, fontweight="bold")

    # inlet log-law velocity profile -- kept entirely left of the building
    # (never crosses x = -B) and capped well below "(a)" / the dashed line.
    x0 = -1.55
    prof_y_top = 1.35
    amp = 0.55  # max rightward bow stays left of the building edge at -0.8

    def prof(y):
        return x0 + amp * (y / prof_y_top) ** 0.4

    y_curve = np.linspace(0.02, prof_y_top, 60)
    ax.plot(prof(y_curve), y_curve, color="C0", lw=1.6, zorder=1)
    for y in np.linspace(0.02, prof_y_top, 7):
        ax.annotate("", xy=(prof(y), y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.4))
    # label sits well above the curve's top and well below "(a)"
    ax.text(x0, prof_y_top + 0.18, "inlet (log-law)", color="C0",
            ha="left", va="bottom", fontsize=10)

    # roof exchange plane
    ax.plot([0, W], [H, H], color="darkorange", lw=2.2, zorder=2)
    ax.text(W / 2, H + 0.07, "roof exchange plane", color="darkorange",
            ha="center", va="bottom", fontsize=9.5)

    # canyon recirculating vortex (kept clear of the W dimension below it)
    cx, cy, r = W / 2, 0.64, 0.20
    th = np.linspace(0.25 * np.pi, 1.9 * np.pi, 100)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color="C0", lw=1.8, zorder=2)
    ax.annotate("", xy=(cx + r * np.cos(th[-1]) + 0.04, cy + r * np.sin(th[-1]) - 0.02),
                xytext=(cx + r * np.cos(th[-1]), cy + r * np.sin(th[-1])),
                arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.8))

    # W dimension -- its own row, clear of both the vortex above and the
    # street-level source label below
    w_y = 0.30
    ax.annotate("", xy=(0.12, w_y), xytext=(W - 0.12, w_y),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
    ax.text(W / 2, w_y + 0.05, "W", ha="center", va="bottom", fontsize=12)

    # H dimension, to the right of the right building
    h_x = W + B + 0.15
    ax.annotate("", xy=(h_x, 0), xytext=(h_x, H),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
    ax.text(h_x + 0.06, H / 2, "H", ha="left", va="center", fontsize=12)

    # street-level source: dots stay on the brown ground; the label moves up
    # onto the white canyon interior (readable background) in near-black text,
    # in its own row below the W dimension
    dots_y = 0.035
    ax.plot(np.linspace(0.15, W - 0.15, 8), np.full(8, dots_y), "o",
            color="firebrick", ms=5, zorder=3)
    ax.text(W / 2, dots_y + 0.06, "street-level source", color="0.15",
            ha="center", va="bottom", fontsize=9.5)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bot, y_top)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Panel (b): D2Q9 velocity set -------------------------------------
    dirs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
    for dx, dy in dirs:
        axb.annotate("", xy=(dx, dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.8))
    axb.plot(0, 0, "o", color="k", ms=9, zorder=3)
    axb.text(0.02, 1.55, "(b)", ha="left", va="top", fontsize=15, fontweight="bold")
    axb.text(0, -1.55, "D2Q9 lattice", ha="center", va="top", fontsize=10.5)
    axb.set_xlim(-1.6, 1.6)
    axb.set_ylim(-1.75, 1.75)
    axb.set_aspect("equal")
    axb.axis("off")

    fig.subplots_adjust(wspace=0.15)
    return _save(fig, path)

def plot_flow_regimes(path: str | Path) -> list[Path]:
    """Figure 1: Oke (1988) flow-regime schematic -- three panels (isolated
    roughness, wake interference, skimming flow) vs aspect ratio H/W. Pure
    diagram; no data. Bold panel subtitle and the grey H/W-range caption below
    it are given a deliberate vertical gap so they never crowd each other."""
    import matplotlib.patches as mpatches
    import numpy as np

    building_color = "#b7c4d9"
    panels = [
        dict(tag="(a)", title="Isolated roughness", caption="H/W < 0.3", mode="isolated"),
        dict(tag="(b)", title="Wake interference", caption="0.3 < H/W < 0.7", mode="wake"),
        dict(tag="(c)", title="Skimming flow", caption="H/W > 0.7", mode="skim"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)

    H, B, W = 1.0, 0.55, 0.55
    ground_y = 0.0
    x_left, x_right = -B - 0.35, W + B + 0.35
    y_bot, y_top = -0.55, 1.75

    for ax, p in zip(axes, panels):
        # ground
        ax.add_patch(mpatches.Rectangle(
            (x_left, ground_y - 0.08), x_right - x_left, 0.08, color="#6b4a30", zorder=1))
        # buildings
        ax.add_patch(mpatches.Rectangle((-B, 0), B, H, color=building_color, ec="k", zorder=2))
        ax.add_patch(mpatches.Rectangle((W, 0), B, H, color=building_color, ec="k", zorder=2))

        # approach-flow arrow markers along the top
        for xa in np.linspace(x_left + 0.15, x_right - 0.15, 5):
            ax.plot(xa, H + 0.3, ">", color="C0", ms=9, zorder=3)

        if p["mode"] == "isolated":
            # two arrows plunging straight into the (wide-open) canyon
            for xa in (W / 2 - 0.18, W / 2 + 0.18):
                ax.annotate("", xy=(xa, 0.15), xytext=(xa, H + 0.08),
                            arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.8))
            # H dimension on the left building
            hx = -B - 0.12
            ax.annotate("", xy=(hx, 0), xytext=(hx, H),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
            ax.text(hx - 0.06, H / 2, "H", ha="right", va="center", fontsize=11)
            # W dimension between the buildings
            wy = 0.34
            ax.annotate("", xy=(0.05, wy), xytext=(W - 0.05, wy),
                        arrowprops=dict(arrowstyle="<->", color="C0", lw=1.4))
            ax.text(W / 2, wy + 0.09, "W", ha="center", va="bottom", color="C0", fontsize=11)
        elif p["mode"] == "wake":
            cx, cy, r = W / 2, 0.5, 0.28
            th = np.linspace(0.15 * np.pi, 1.85 * np.pi, 100)
            ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color="C0", lw=2.2, zorder=3)
            ax.annotate("", xy=(cx + r * np.cos(th[0]) - 0.03, cy + r * np.sin(th[0])),
                        xytext=(cx + r * np.cos(th[0]), cy + r * np.sin(th[0])),
                        arrowprops=dict(arrowstyle="-|>", color="C0", lw=2.2))
        else:  # skimming
            cx, cy, r = W / 2, 0.48, 0.18
            th = np.linspace(0.15 * np.pi, 1.85 * np.pi, 100)
            ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color="C0", lw=2.2, zorder=3)
            ax.annotate("", xy=(cx + r * np.cos(th[0]) - 0.03, cy + r * np.sin(th[0])),
                        xytext=(cx + r * np.cos(th[0]), cy + r * np.sin(th[0])),
                        arrowprops=dict(arrowstyle="-|>", color="C0", lw=2.2))
            # short roof-level feed arrow above the vortex
            ax.annotate("", xy=(cx + 0.15, H + 0.08), xytext=(cx - 0.1, H + 0.08),
                        arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.8))

        ax.text(x_left + 0.04, y_top - 0.02, p["tag"], ha="left", va="top",
                fontsize=14, fontweight="bold")

        # Bold title and the grey H/W-range caption below it -- deliberately
        # spaced apart (0.30 data units) so they never crowd each other.
        title_y = -0.30
        caption_y = -0.30 - 0.20
        ax.text((x_left + x_right) / 2, title_y, p["title"], ha="center", va="top",
                fontsize=13, fontweight="bold")
        ax.text((x_left + x_right) / 2, caption_y, p["caption"], ha="center", va="top",
                fontsize=10.5, color="0.5")

        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bot, y_top)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(wspace=0.05, bottom=0.02)
    return _save(fig, path)

def plot_les_grid_divergence(les_csv: str | Path, laminar_csv: str | Path,
                             path: str | Path) -> list[Path]:
    """Methods/limitations figure: WHY 2-D LES is ill-posed for this gate.

    Plots the ventilation metric vs resolution for (a) the high-Re MRT+LES runs,
    where it DIVERGES on refinement (the Smagorinsky filter scales with cell
    size, so refining changes the effective Re), and (b) the steady-laminar runs,
    where it converges cleanly. Values are normalized to each series' coarsest
    grid so the contrast is visible on one axis. A methodological result.
    """
    import csv as _csv
    import numpy as np

    def load(p):
        rows = sorted(_csv.DictReader(open(p)), key=lambda r: float(r["cells_per_H"]))
        n = np.array([float(r["cells_per_H"]) for r in rows])
        ret = np.array([float(r["retention_mean_conc"]) for r in rows])
        return n, ret

    nL, retL = load(les_csv)          # MRT+LES (Re ~ 2000): diverges
    nA, retA = load(laminar_csv)      # steady laminar (Re=25): converges

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhspan(0.97, 1.03, color="C2", alpha=0.15, label="±3% gate")
    ax.plot(nL, retL / retL[0], "s-", color="C3", ms=7,
            label=f"MRT+LES, Re≈2000  ({retL[0]/retL[-1]:.1f}× drift, n=24→48)")
    ax.plot(nA, retA / retA[0], "o-", color="C0", ms=7,
            label="steady laminar, Re=25 (converged)")
    ax.set_xscale("log", base=2); ax.set_xticks(nA)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("resolution (cells per building height H)")
    ax.set_ylabel("ventilation metric / coarsest-grid value")
    ax.set_title("Why 2-D LES is ill-posed here:\nthe metric diverges on "
                 "refinement (LES) vs converges (laminar)", fontsize=10)
    ax.legend(loc="center left", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_pe_sensitivity(csv_path: str | Path, path: str | Path) -> list[Path]:
    """Headline insight: the leeward/windward asymmetry is advection-controlled.

    Plots the wall c+ asymmetry (leeward/windward) vs Péclet number at H/W=1,
    showing it rises from ~1 (diffusion-dominated, symmetric) toward the CODASC
    value as advection takes over. The CODASC reference (~3.0) is marked.
    """
    import csv as _csv
    import numpy as np

    rows = sorted(_csv.DictReader(open(csv_path)), key=lambda r: float(r["Pe"]))
    pe = np.array([float(r["Pe"]) for r in rows])
    asym = np.array([float(r["asymmetry"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(2.98, color="0.4", ls="--", lw=1.2, label="CODASC (turbulent) ≈ 3.0")
    ax.axhline(1.0, color="0.7", ls=":", lw=1, label="symmetric (no asymmetry)")
    ax.semilogx(pe, asym, "o-", color="C3", ms=7, label="2-D laminar model (Re=25)")
    ax.set_xlabel("Péclet number  Pe = Sc·Re  (advection / diffusion)")
    ax.set_ylabel("leeward / windward  c⁺ asymmetry")
    ax.set_title("Leeward–windward asymmetry is advection-controlled\n"
                 "(it emerges as the Péclet number rises)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_ylim(0.9, 3.2)
    fig.tight_layout()
    return _save(fig, path)


def plot_aspect_ratio_sweep(csv_path: str | Path, path: str | Path) -> list[Path]:
    """Primary deliverable: ventilation metric vs aspect ratio H/W, with the
    three Oke (1988) flow-regime bands and the skimming transition marked.

    Reads results/sweep_aspect_ratio.summary.csv (retention_eq, ach vs H/W).
    """
    import csv as _csv
    import numpy as np

    rows = sorted((r for r in _csv.DictReader(open(csv_path))),
                  key=lambda r: float(r["aspect_ratio"]))
    hw = np.array([float(r["aspect_ratio"]) for r in rows])
    ret = np.array([float(r["retention_eq"]) for r in rows])
    ach = np.array([float(r["ach_exchange_rate"]) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    # Oke flow-regime bands.
    bands = [(0.0, 0.3, "isolated\nroughness", "#cde6cd"),
             (0.3, 0.7, "wake\ninterference", "#fdf3c0"),
             (0.7, 3.3, "skimming", "#f6cccc")]
    for ax in (ax1, ax2):
        for x0, x1, _lab, col in bands:
            ax.axvspan(x0, x1, color=col, alpha=0.6, lw=0)
        ax.axvline(0.7, color="0.4", ls="--", lw=1)

    ax1.plot(hw, ret, "o-", color="C3", ms=6)
    ax1.set_ylabel("equilibrium retention  (canyon-mean $c$)")
    ax1.set_xlabel("aspect ratio  H/W")
    ax1.set_title("Pollutant retention vs H/W")
    ax2.plot(hw, ach, "o-", color="C0", ms=6)
    ax2.set_ylabel(r"air-exchange rate  ACH  (flux/content) [1/step]")
    ax2.set_xlabel("aspect ratio  H/W")
    ax2.set_title("Ventilation (air-exchange) vs H/W")
    for ax in (ax1, ax2):
        ax.set_xlim(hw.min() - 0.1, hw.max() + 0.1)
        ymax = ax.get_ylim()[1]
        for x0, x1, lab, _c in bands:
            xc = 0.5 * (max(x0, hw.min()) + min(x1, hw.max() + 0.1))
            ax.text(xc, 0.93 * ymax, lab, ha="center", va="top", fontsize=7.5,
                    color="0.35")
    ax1.annotate("laminar: weak / non-monotonic —\nno skimming collapse "
                 "(advection-suppressed)", xy=(0.5, 0.04), xycoords="axes fraction",
                 fontsize=7.5, color="0.25", ha="left", va="bottom")
    fig.suptitle("Ventilation vs H/W (2-D laminar, Re=25, Pe=50): the flow "
                 "regime-change is present\nbut the pollutant skimming-collapse "
                 "is not — it is advection-controlled (see Péclet figure)",
                 fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_codasc_validation(out: dict, path: str | Path) -> list[Path]:
    """Sim vs CODASC wall c+ profiles: absolute (left) and normalized shape (right).

    ``out`` is the dict written by scripts/validate_codasc.py. The laminar model
    is not expected to match the absolute c+ (turbulent tunnel); the right panel
    shows the PATTERN -- leeward>>windward asymmetry and vertical decay -- which
    is the physically robust validation.
    """
    import numpy as np

    p = out["profiles"]
    zr = np.array(p["zH_codasc"]); leeR = np.array(p["cplus_leeward_codasc"])
    winR = np.array(p["cplus_windward_codasc"])
    zs = np.array(p["zH_sim"]); leeS = np.array(p["cplus_leeward_sim"])
    winS = np.array(p["cplus_windward_sim"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # (a) absolute c+ (log x: the laminar/turbulent offset is honest, structure visible)
    ax1.semilogx(leeR, zr, "o", color="C3", label="CODASC leeward")
    ax1.semilogx(winR, zr, "s", color="C0", label="CODASC windward")
    ax1.semilogx(leeS, zs, "-", color="C3", lw=2, label="model leeward")
    ax1.semilogx(winS, zs, "-", color="C0", lw=2, label="model windward")
    ax1.set_xlabel(r"$c^+$ (absolute)"); ax1.set_ylabel("z/H (height up wall)")
    ax1.set_title("Absolute $c^+$\n(laminar model vs turbulent tunnel)", fontsize=10)
    ax1.legend(fontsize=7, frameon=False); ax1.set_ylim(0, 1)

    # (b) normalized by each dataset's leeward street-level value -> shape/pattern
    leeRn, winRn = leeR / leeR[0], winR / leeR[0]
    leeSn, winSn = leeS / leeS[0], winS / leeS[0]
    ax2.plot(leeRn, zr, "o", color="C3", label="CODASC leeward")
    ax2.plot(winRn, zr, "s", color="C0", label="CODASC windward")
    ax2.plot(leeSn, zs, "-", color="C3", lw=2, label="model leeward")
    ax2.plot(winSn, zs, "-", color="C0", lw=2, label="model windward")
    ax2.set_xlabel(r"$c^+ / c^+_{\rm leeward,street}$"); ax2.set_ylabel("z/H")
    ax2.set_title("Normalized shape (pattern test)", fontsize=10)
    ax2.legend(fontsize=7, frameon=False); ax2.set_ylim(0, 1)

    pat = out["pattern"]
    fig.suptitle(
        f"CODASC validation, H/W=1 (no trees, 90°) — "
        f"leeward/windward: model {pat['leeward_windward_ratio_sim']:.1f} "
        f"vs CODASC {pat['leeward_windward_ratio_codasc']:.1f}", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_grid_independence(csv_path: str | Path, path: str | Path) -> list[Path]:
    """Ventilation metric vs lattice resolution from grid_independence.csv.

    Two panels (retention and the normalized air-exchange index) vs cells/H, with
    a +/-3% band around the finest-grid value marking the convergence gate.
    """
    import csv as _csv

    rows = list(_csv.DictReader(open(csv_path)))
    n = [float(r["cells_per_H"]) for r in rows]
    retention = [float(r["retention_mean_conc"]) for r in rows]
    vent = [float(r["ventilation_index"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, vals, label in (
        (axes[0], retention, "retention  (canyon-mean C)"),
        (axes[1], vent, r"ventilation index  $w_e/U_H$"),
    ):
        ax.plot(n, vals, "o-", color="C0", ms=6)
        finest = vals[-1]
        ax.axhspan(0.97 * finest, 1.03 * finest, color="C2", alpha=0.15,
                   label="±3% of finest")
        ax.axhline(finest, color="C2", lw=1, ls="--")
        ax.set_xlabel("resolution (cells per building height H)")
        ax.set_ylabel(label)
        ax.set_xticks(n)
        ax.legend(loc="best", fontsize=8, frameon=False)
    if len(n) >= 2:
        rel = abs(vent[-1] - vent[-2]) / (abs(vent[-2]) + 1e-30) * 100
        axes[1].set_title(f"{n[-2]:.0f}→{n[-1]:.0f} cells/H: {rel:.1f}% change")
    # Read the actual regime (Re, collision) from the run metadata so the title
    # never goes stale relative to the data it plots.
    regime = ""
    try:
        import json
        meta = json.load(open(
            Path(csv_path).with_name("grid_independence.summary.json.meta.json")))
        cfg = meta.get("config", {})
        if cfg.get("Re") is not None:
            regime = f", Re = {float(cfg['Re']):g}, {cfg.get('collision', '').upper()}"
    except Exception:
        pass
    fig.suptitle(f"Grid-independence study (H/W = 1{regime})")
    fig.tight_layout()
    return _save(fig, path)


def plot_poiseuille(result: dict, path: str | Path) -> list[Path]:
    """Numeric vs analytic Poiseuille profile with the relative error annotated."""
    y = result["y"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        result["ux_analytic"], y, "-", color="0.4", lw=2, label="analytic parabola"
    )
    ax.plot(
        result["ux_profile"],
        y,
        "o",
        ms=4,
        mfc="none",
        color="C0",
        label="LBM (D2Q9 BGK)",
    )
    ax.set_xlabel(r"$u_x$ (lattice units)")
    ax.set_ylabel("y (cells from bottom wall)")
    ax.set_title("Poiseuille channel: LBM vs analytic")
    ax.legend(loc="upper right", frameon=False)
    txt = (
        f"rel. L2 error = {result['rel_l2_error']:.2e}\n"
        f"max rel. error = {result['max_rel_error']:.2e}\n"
        rf"$\tau$ = {result['tau']}, iters = {result['iters']}"
    )
    ax.text(
        0.05, 0.5, txt, transform=ax.transAxes, va="center", ha="left", fontsize=9
    )
    return _save(fig, path)
