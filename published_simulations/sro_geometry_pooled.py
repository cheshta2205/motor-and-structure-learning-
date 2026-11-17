"""Pooled across-seed version of sro_geometry_conditions.

Reads the per-seed JSON that sro_geometry_conditions.py already wrote -- no
recompute, no checkpoints loaded.  Every seed is drawn as its own dot because
the finding is about SPREAD, not mean: the main arm's context code is abstract
in every network, while the control arm's ranges from chance to ceiling.  A
mean with error bars would hide exactly that.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sro

ROOT = os.path.join("figures", "main", sro.TASK_DIR)
OUT = os.path.join(ROOT, "pooled_seeds")

COND = [("main", "main", "#1a1a1a"),
        ("control", "control", "#2f7ab8"),
        ("main, $\\delta$=0", "main, $\\delta$=0", "#c0392b"),
        ("main, $\\delta$ shuffled", "main, $\\delta$ shuffled", "#e0883a")]
POPS = [("h", "MTL hidden state ($h$)"),
        ("cb_l1", "cerebellum, layer 1"),
        ("cb_l2", "cerebellum, layer 2")]
NAMED = ["context", "value", "action"]
METRICS = [("dec", "decoding", .5), ("ccgp", "CCGP", .5), ("ps", "parallelism score", 0.)]


def load():
    """{(cond, pop, var, metric): [per-seed values]} plus sd."""
    files = sorted(glob.glob(os.path.join(ROOT, "seed*/sro_geometry_conditions.json")),
                   key=lambda p: int(p.split("seed")[1].split("/")[0]))
    D, SD = {}, {}
    for f in files:
        d = json.load(open(f))
        for key, val in d.items():
            cond, pop = key.split("|")
            SD.setdefault((cond, pop), []).append(val["sd"])
            for var, mm in val["named"].items():
                for m, v in mm.items():
                    D.setdefault((cond, pop, var, m), []).append(v)
    return D, SD, len(files)


def figure(path=None):
    D, SD, n = load()
    fig, AX = plt.subplots(len(POPS), 4, figsize=(16.5, 9.4))
    rng = np.random.default_rng(0)

    for r, (pop, poplab) in enumerate(POPS):
        for c, (m, mlab, chance) in enumerate(METRICS):
            ax = AX[r, c]
            for vi, var in enumerate(NAMED):
                for ci, (cond, clab, col) in enumerate(COND):
                    y = np.asarray(D.get((cond, pop, var, m), []), float)
                    if not y.size:
                        continue
                    x = vi + (ci - 1.5) * 0.19
                    ax.scatter(x + rng.uniform(-.045, .045, y.size), y, s=17,
                               color=col, alpha=.55, edgecolors="none",
                               zorder=2, label=clab if (r == 0 and c == 0 and vi == 0) else None)
                    ax.plot([x - .075, x + .075], [y.mean()] * 2, color=col,
                            lw=2.4, solid_capstyle="butt", zorder=3)
            ax.axhline(chance, color="#999", lw=1, ls="--", zorder=1)
            ax.set_xticks(range(len(NAMED))); ax.set_xticklabels(NAMED)
            ax.set_xlim(-.55, len(NAMED) - .45)
            ax.set_ylim(-.55, 1.06) if m == "ps" else ax.set_ylim(-.02, 1.06)
            if r == 0:
                ax.set_title(mlab, fontsize=11.5)
            if c == 0:
                ax.set_ylabel(f"{poplab}\n\naccuracy", fontsize=10)

        ax = AX[r, 3]                                   # shattering dimensionality
        for ci, (cond, clab, col) in enumerate(COND):
            y = np.asarray(SD.get((cond, pop), []), float)
            if not y.size:
                continue
            ax.scatter(ci + rng.uniform(-.1, .1, y.size), y, s=17, color=col,
                       alpha=.55, edgecolors="none", zorder=2)
            ax.plot([ci - .22, ci + .22], [y.mean()] * 2, color=col, lw=2.4, zorder=3)
        ax.axhline(.5, color="#999", lw=1, ls="--", zorder=1)
        ax.set_xticks(range(len(COND)))
        ax.set_xticklabels([c[1] for c in COND], rotation=32, ha="right", fontsize=8)
        ax.set_ylim(-.02, 1.06)
        if r == 0:
            ax.set_title("shattering dimensionality\n(mean decode, 35 dichotomies)",
                         fontsize=10.5)

    for a in AX.ravel():
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    AX[0, 0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle("Representational geometry of previous-trial variables, "
                 f"pooled across {n} seeds\n"
                 "one dot per network, bar = mean; dashed = chance", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT, exist_ok=True)
    path = path or os.path.join(OUT, "sro_geometry_pooled.png")
    fig.savefig(path, dpi=170); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"saved -> {path}  ({n} seeds)")


def table():
    D, SD, n = load()
    print(f"\n{'pop':7s} {'variable':8s} {'metric':6s} " +
          " ".join(f"{c[1][:11]:>12s}" for c in COND))
    for pop, _ in POPS:
        for var in NAMED:
            for m, _, _ in METRICS:
                cells = []
                for cond, _, _ in COND:
                    y = np.asarray(D.get((cond, pop, var, m), []), float)
                    cells.append(f"{y.mean():.2f}[{y.min():.2f},{y.max():.2f}]"
                                 if y.size else "--")
                print(f"{pop:7s} {var:8s} {m:6s} " + " ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    import sys
    table() if "--table" in sys.argv else figure()
