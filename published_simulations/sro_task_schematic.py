"""Panel A of the SRO figure: what the task is.

Drawn, not measured -- every number in it is read off sro.py so the sketch and
the simulation cannot drift apart:

    SRO          the 4 x 2 stimulus-response-outcome table
    SWITCH_MIN/MAX   block length
    outcome()    the three payoff levels

Three stacked blocks:
    (a) one trial:   stimulus -> operant -> outcome
    (b) the table:   which operant is correct, and whether it pays
    (c) the blocks:  context switches, uncued
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import sro

OUT = os.path.join("figures", "main", sro.TASK_DIR, "pooled_seeds")
# one colour per stimulus.  The stimuli are arbitrary images in the monkey task,
# so they get arbitrary but distinguishable colours here.
SCOL = ["#4a7fb5", "#5aab6e", "#d4813e", "#9b6bb5"]
CCOL = ["#f2d9d6", "#d8e4ef"]                    # the two context blocks
OPNAME = {1: "release", 0: "hold"}
INK = "#1a1a1a"


def _box(ax, x, y, w, h, fc, ec=INK, lw=1.2, r=0.02, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _arrow(ax, x0, y0, x1, y1, lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=INK, zorder=3))


def draw(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── (a) one trial ────────────────────────────────────────────────────────
    ax.text(0.0, 0.985, "one trial", fontsize=9.5, style="italic",
            color="#555", va="top")
    ytr, htr = 0.845, 0.10
    _box(ax, 0.045, ytr, 0.085, htr, SCOL[0])
    ax.text(0.088, ytr + htr / 2, "stim", ha="center", va="center",
            fontsize=8.5, color="w", weight="bold")
    _arrow(ax, 0.145, ytr + htr / 2, 0.245, ytr + htr / 2)
    _box(ax, 0.26, ytr, 0.20, htr, "w")
    ax.text(0.36, ytr + htr / 2, "hold  or  release", ha="center", va="center",
            fontsize=8.5)
    _arrow(ax, 0.475, ytr + htr / 2, 0.575, ytr + htr / 2)
    _box(ax, 0.59, ytr, 0.365, htr, "w")
    ax.text(0.772, ytr + htr / 2,
            "$+1$ correct & pays   $0$ correct   $-1$ wrong",
            ha="center", va="center", fontsize=8.2)

    # ── (b) the mapping table ────────────────────────────────────────────────
    ax.text(0.0, 0.775, "which operant is correct, and whether it pays",
            fontsize=9.5, style="italic", color="#555", va="top")
    x0, cw, y0, rh = 0.215, 0.185, 0.40, 0.145
    for j in range(sro.N_STIM):                       # stimulus column headers
        cx = x0 + j * cw
        _box(ax, cx + cw / 2 - 0.036, y0 + 2 * rh + 0.022, 0.072, 0.062, SCOL[j])
        ax.text(cx + cw / 2, y0 + 2 * rh + 0.053, f"{j + 1}", ha="center",
                va="center", fontsize=8.5, color="w", weight="bold")
    ax.text(x0 + 2 * cw, y0 + 2 * rh + 0.115, "stimulus", ha="center",
            fontsize=9, color="#555")

    for i in range(sro.N_CTX):                        # one row per context
        ry = y0 + (1 - i) * rh
        ax.add_patch(Rectangle((x0 - 0.205, ry), 0.19, rh * 0.86, fc=CCOL[i],
                               ec="none", zorder=1))
        ax.text(x0 - 0.11, ry + rh * 0.43, f"context {i + 1}", ha="center",
                va="center", fontsize=9)
        for j in range(sro.N_STIM):
            op, pays = sro.SRO[i, j]
            cx = x0 + j * cw
            _box(ax, cx + 0.012, ry, cw - 0.024, rh * 0.86, CCOL[i], ec="#999", lw=.8)
            ax.text(cx + cw / 2, ry + rh * 0.56, OPNAME[int(op)], ha="center",
                    va="center", fontsize=9, weight="bold")
            ax.text(cx + cw / 2, ry + rh * 0.22,
                    "pays $+1$" if pays else "pays $0$", ha="center",
                    va="center", fontsize=7.8, color="#444")

    ax.text(0.5, y0 - 0.055,
            "context 2 is a full reversal: operant AND payout flip on all four",
            ha="center", fontsize=8.2, color="#444")
    ax.text(0.5, y0 - 0.115,
            "so operant and reward are orthogonal --\nneither alone identifies the context",
            ha="center", va="top", fontsize=8.2, color="#444")

    # ── (c) the block structure ──────────────────────────────────────────────
    ax.text(0.0, 0.165, "context switches, never cued", fontsize=9.5,
            style="italic", color="#555", va="top")
    yb, hb = 0.045, 0.062
    rng = np.random.default_rng(3)
    edges, t, c = [0.0], 0.0, 0
    while t < 1.0:                                     # blocks to scale
        t += rng.integers(sro.SWITCH_MIN, sro.SWITCH_MAX + 1) / sro.SEQ_LEN
        edges.append(min(t, 1.0)); c = 1 - c
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        ax.add_patch(Rectangle((a, yb), b - a, hb, fc=CCOL[k % 2],
                               ec=INK, lw=.9, zorder=2))
        ax.text((a + b) / 2, yb + hb / 2, f"{k % 2 + 1}", ha="center",
                va="center", fontsize=8, color="#555")
        if k:
            ax.plot([a, a], [yb + hb, yb + hb + 0.035], color="#e67e22", lw=1.4)
    ax.text(0.5, yb - 0.028,
            f"{sro.SWITCH_MIN}-{sro.SWITCH_MAX} trials per block, "
            f"{sro.SEQ_LEN} trials per session",
            ha="center", va="top", fontsize=8.2, color="#444")


def figure(path=None):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    draw(ax)
    fig.tight_layout(pad=0.3)
    os.makedirs(OUT, exist_ok=True)
    path = path or os.path.join(OUT, "sro_task_schematic.png")
    fig.savefig(path, dpi=200); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    figure()
