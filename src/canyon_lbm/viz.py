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
        0.03, 0.03, txt, transform=ax.transAxes, va="bottom", ha="left", fontsize=9
    )
    return _save(fig, path)
