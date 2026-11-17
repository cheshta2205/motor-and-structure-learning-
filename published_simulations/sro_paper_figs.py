"""Paper-styled versions of the three pooled SRO figures.

Reads only what the per-seed runs already wrote -- no checkpoints are loaded and
nothing is recomputed -- and redraws it in the layout used in the manuscript:
panel letters, minimal ticks, column headers, cleaned spines.  The point is that
a rerun no longer needs an Illustrator pass to get back to the published look.

    SRO_VARIANT=bernardi python3 sro_paper_figs.py --seeds 0 1 ... 9

Writes, into <figures>/main/<TASK_DIR>/pooled_seeds/:
    paper_behaviour.{png,svg}   context decode + inference signature   (b, c)
    paper_geometry.{png,svg}    decoding/CCGP, PS, SD vs CCGP          (d, e, f)
    paper_layers.{png,svg}      the same measures across MTL and CB    (Ext Data)
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sro

FONT = 22.0                  # manuscript body size
FAMILY = "Source Sans 3"
S = FONT / 8.5               # everything below is authored at 8.5pt

VCOL = {"context": "#c0392b", "value": "#e084c4", "action": "#f1a512"}
VLAB = {"context": "context", "value": "previous value", "action": "previous action"}
# the four ablation conditions, keyed as sro_geometry_conditions.json writes them
COND = [("main", "main", "#1a1a1a"),
        ("control", "control", "#2f7ab8"),
        ("main, $\\delta$=0", "main, $\\delta$ = 0", "#7a4fa3"),
        ("main, $\\delta$ shuffled", "main, $\\delta$ = shuffled", "#8d8d8d")]
POPS = [("h", "MTL hidden state ($h$)"),
        ("cb_l1", "Cerebellum Layer 1"),
        ("cb_l2", "Cerebellum Layer 2")]
NAMED = ["context", "value", "action"]
METRICS = [("dec", "Decoding", .5), ("ccgp", "CCGP", .5), ("ps", "Parallelism Score", 0.)]
# the named variables land within ~0.02 of each other, so overlaid markers
# occlude; dodge them sideways instead of shrinking them
DODGE = {"context": -0.13, "value": 0.0, "action": 0.13}
REF_COL = "#c8c8c8"          # chance / zero rules, kept paler than any series


def style():
    plt.rcParams.update({
        "font.family": [FAMILY, "sans-serif"],
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
        "xtick.labelsize": FONT, "ytick.labelsize": FONT,
        "legend.fontsize": FONT * .85,
        "axes.linewidth": .8 * S, "xtick.major.width": .8 * S,
        "ytick.major.width": .8 * S, "xtick.major.size": 3.5 * S,
        "ytick.major.size": 3.5 * S,
        "lines.linewidth": 1.4 * S, "legend.frameon": False,
        "svg.fonttype": "none",          # keep text editable in Illustrator
    })


def bare(ax, yticks=(0, 1), ylim=None):
    """Two spines, two y ticks -- the manuscript's axis convention."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if ylim is not None:
        ax.set_ylim(*ylim)


def letter(ax, ch, dx=-.16, dy=1.06):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontsize=FONT * 1.25,
            fontweight="bold", va="top", ha="left")


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(np.isfinite(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


# ── loaders ────────────────────────────────────────────────────────────────
def load_behaviour(seeds):
    out = {}
    for s in seeds:
        p = os.path.join(sro.fig_dir(s), "sro_behaviour.npz")
        if not os.path.exists(p):
            print(f"[seed {s}] no behaviour npz, skipping"); continue
        z = np.load(p, allow_pickle=True)
        for arm in ("main", "control"):
            for mode in ("intact", "zero", "shuffle"):
                k = f"{arm}|{mode}"
                if f"{k}|dec" not in z:
                    continue
                d = out.setdefault(k, {"ks": z[f"{k}|ks"], "dec": [], "f1c": []})
                d["dec"].append(z[f"{k}|dec"])
                d["f1c"].append(z[f"{k}|f1c"])
    return out


def load_geometry(seeds, arm="main", key="h"):
    acc = {}
    for s in seeds:
        p = os.path.join(sro.fig_dir(s), f"sro_geometry_{arm}_{key}.npz")
        if not os.path.exists(p):
            print(f"[seed {s}] no geometry npz, skipping"); continue
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            acc.setdefault(k, []).append(z[k])
    return {k: np.array(v) for k, v in acc.items()}


def load_layers(seeds):
    acc = {}
    for s in seeds:
        p = os.path.join(sro.fig_dir(s), "sro_geometry_conditions.json")
        if not os.path.exists(p):
            print(f"[seed {s}] no conditions json, skipping"); continue
        j = json.load(open(p))
        for k, v in j.items():
            acc.setdefault(k, []).append(v)
    return acc


# ── figure 1: behaviour ────────────────────────────────────────────────────
def fig_behaviour(seeds, out):
    B = load_behaviour(seeds)
    if not B:
        print("no behaviour data"); return
    fig, AX = plt.subplots(1, 2, figsize=(7.4 * S, 2.5 * S))

    ARMC = {"main": "#2f7ab8", "control": "#1a1a1a"}
    LS = {"intact": "-", "zero": "--", "shuffle": ":"}
    LBL = {"intact": "intact", "zero": "CB = 0", "shuffle": "CB = shuffle"}

    ax = AX[0]
    for arm in ("main", "control"):
        for mode in ("intact", "zero", "shuffle"):
            k = f"{arm}|{mode}"
            if k not in B:
                continue
            ks, d = B[k]["ks"], np.array(B[k]["dec"])
            mu, se = np.nanmean(d, 0), sem(d)
            ax.plot(ks, mu, ls=LS[mode], color=ARMC[arm], marker="o", ms=2.2 * S,
                    label=f"{arm} ({LBL[mode]})")
            ax.fill_between(ks, mu - se, mu + se, color=ARMC[arm], alpha=.18, lw=0)
    ax.axhline(.5, color="#999", lw=.7 * S, ls="--")
    ax.axvline(0, color="#e67e22", lw=1.0 * S, ls=":")
    ax.set_xlabel("trials relative to context switch")
    ax.set_ylabel("context decode accuracy")
    bare(ax, yticks=(0.2, 1), ylim=(0.18, 1.04))
    ax.legend(ncol=2, loc="lower right", handlelength=1.6, columnspacing=1.0)
    letter(ax, "b")

    ax = AX[1]
    x = np.arange(1, 5)
    for arm in ("main", "control"):
        for mode in ("intact", "zero", "shuffle"):
            k = f"{arm}|{mode}"
            if k not in B:
                continue
            f = np.array(B[k]["f1c"])
            ax.errorbar(x, np.nanmean(f, 0), yerr=sem(f), ls=LS[mode],
                        color=ARMC[arm], marker="s", ms=4 * S, capsize=2.5 * S, lw=1.4 * S)
    ax.axhline(.5, color="#999", lw=.7 * S, ls="--")
    ax.set_xticks(x)
    ax.set_xlabel("stimulus appearance order")
    ax.set_ylabel("operant accuracy")
    bare(ax, yticks=(0, 1), ylim=(-.05, 1.08))
    letter(ax, "c")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"paper_behaviour.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}/paper_behaviour.png  ({len(seeds)} seeds)")


# ── figure 2: main-arm geometry ────────────────────────────────────────────
def fig_geometry(seeds, out):
    G = load_geometry(seeds)
    if not G:
        print("no geometry data"); return
    fig, AX = plt.subplots(1, 3, figsize=(7.4 * S, 2.5 * S))

    def named(metric):
        return {nm: G[f"{nm}|{metric}"] for nm in NAMED if f"{nm}|{metric}" in G}

    # (d) decoding and CCGP, all 35 dichotomies grey, named highlighted
    ax = AX[0]
    for i, metric in enumerate(("dec", "ccgp")):
        cloud = np.nanmean(G[metric], 0)                   # (35,)
        ax.scatter(np.full(cloud.shape, i) + np.random.default_rng(0)
                   .uniform(-.09, .09, cloud.size), cloud,
                   s=7 * S**2, color="#9a9a9a", alpha=.7, lw=0,
                   label="other dichotomies" if i == 0 else None, zorder=1)
        for nm, v in named(metric).items():
            ax.errorbar(i + DODGE[nm], np.nanmean(v), yerr=sem(v), fmt="o",
                        ms=7 * S, color=VCOL[nm], mec="#2b2b2b", mew=.7 * S,
                        capsize=2.5 * S, zorder=3,
                        label=VLAB[nm] if i == 0 else None)
    ax.axhline(.5, color="#999", lw=.7 * S, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["decoding", "CCGP"])
    ax.set_xlim(-.45, 1.45); ax.set_ylabel("accuracy")
    bare(ax, yticks=(0, 1), ylim=(-.05, 1.08))
    ax.legend(loc="lower left", handletextpad=.3)
    letter(ax, "d")

    # (e) parallelism score
    ax = AX[1]
    cloud = np.nanmean(G["ps"], 0)
    ax.scatter(np.random.default_rng(1).uniform(-.33, .33, cloud.size), cloud,
               s=7 * S**2, color="#9a9a9a", alpha=.7, lw=0, zorder=1)
    for nm, v in named("ps").items():
        ax.errorbar(DODGE[nm] * 3, np.nanmean(v), yerr=sem(v), fmt="o", ms=7 * S,
                    color=VCOL[nm], mec="#2b2b2b", mew=.7 * S, capsize=2.5 * S, zorder=3)
    ax.axhline(0, color="#999", lw=.7 * S, ls="--")
    ax.set_xticks([]); ax.set_xlim(-.5, .5)
    ax.set_ylabel("parallelism score")
    bare(ax, yticks=(0, 1), ylim=(-.45, 1.08))
    letter(ax, "e")

    # (f) shattering dimensionality against context CCGP
    ax = AX[2]
    sd = np.asarray(G["sd"], float).ravel()
    ax.bar([0], [np.nanmean(sd)], width=.55, color="#d0d0d0",
           yerr=[sem(sd)], capsize=3 * S, edgecolor="none", zorder=1)
    for nm, v in named("ccgp").items():
        ax.errorbar(1 + DODGE[nm], np.nanmean(v), yerr=sem(v), fmt="o", ms=7 * S,
                    color=VCOL[nm], mec="#2b2b2b", mew=.7 * S, capsize=2.5 * S, zorder=3)
    ax.axhline(.5, color="#999", lw=.7 * S, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SD", "CCGP"])
    ax.set_xlim(-.55, 1.55)
    bare(ax, yticks=(0, 1), ylim=(-.05, 1.08))
    letter(ax, "f")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"paper_geometry.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}/paper_geometry.png  ({len(seeds)} seeds)")


# ── figure 3: across MTL and CB layers ─────────────────────────────────────
def fig_layers(seeds, out):
    L = load_layers(seeds)
    if not L:
        print("no layer data"); return
    fig, AX = plt.subplots(3, 4, figsize=(9.6 * S, 6.2 * S))
    rng = np.random.default_rng(2)

    for r, (pop, poplab) in enumerate(POPS):
        for c, (metric, metlab, chance) in enumerate(METRICS):
            ax = AX[r, c]
            for vi, nm in enumerate(NAMED):
                for ci, (key, clab, col) in enumerate(COND):
                    rows = L.get(f"{key}|{pop}")
                    if not rows:
                        continue
                    v = np.array([x["named"][nm][metric] for x in rows])
                    xc = vi * 1.6 + (ci - 1.5) * .26
                    ax.scatter(xc + rng.uniform(-.055, .055, v.size), v,
                               s=8 * S**2, color=col, alpha=.75, lw=0, zorder=2)
                    ax.plot([xc - .1, xc + .1], [np.nanmean(v)] * 2,
                            color=col, lw=1.8 * S, zorder=3)
            ax.axhline(chance, color=REF_COL, lw=.7 * S,
                       ls="-" if metric == "ps" else "--")
            ax.set_xticks([vi * 1.6 for vi in range(3)])
            ax.set_xticklabels(NAMED if r == 2 else [])
            lo = -.45 if metric == "ps" else -.05
            bare(ax, yticks=(0, 1), ylim=(lo, 1.08))
            if c == 0:
                ax.set_ylabel(f"{poplab}\naccuracy")
            if r == 0:
                ax.set_title(metlab)
                letter(ax, "abcd"[c], dx=-.22)

        # shattering dimensionality, one point per condition
        ax = AX[r, 3]
        for ci, (key, clab, col) in enumerate(COND):
            rows = L.get(f"{key}|{pop}")
            if not rows:
                continue
            v = np.array([x["sd"] for x in rows])
            ax.scatter(np.full(v.size, ci) + rng.uniform(-.09, .09, v.size), v,
                       s=8 * S**2, color=col, alpha=.75, lw=0, zorder=2)
            ax.plot([ci - .22, ci + .22], [np.nanmean(v)] * 2, color=col,
                    lw=1.8 * S, zorder=3)
        ax.axhline(.5, color=REF_COL, lw=.7 * S, ls="--")
        # the legend already names the four conditions; the rotated duplicate
        # underneath only cost a line of type
        ax.set_xticks([])
        ax.set_xlim(-.6, len(COND) - .4)
        bare(ax, yticks=(0, 1), ylim=(-.05, 1.08))
        if r == 0:
            ax.set_title("Shattering dimensionality")
            letter(ax, "d", dx=-.22)
            h = [plt.Line2D([], [], ls="", marker="o", ms=5 * S, color=c[2],
                            label=c[1]) for c in COND]
            ax.legend(handles=h, loc="lower left", bbox_to_anchor=(.02, .02),
                      handletextpad=.3, labelspacing=.35)

    fig.suptitle("Representational geometry across MTL and CB layers", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(out, f"paper_layers.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}/paper_layers.png  ({len(seeds)} seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--only", choices=["behaviour", "geometry", "layers"],
                    nargs="+", default=["behaviour", "geometry", "layers"])
    a = ap.parse_args()

    style()
    out = sro.pooled_dir()
    if "behaviour" in a.only:
        fig_behaviour(a.seeds, out)
    if "geometry" in a.only:
        fig_geometry(a.seeds, out)
    if "layers" in a.only:
        fig_layers(a.seeds, out)
