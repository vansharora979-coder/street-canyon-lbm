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
