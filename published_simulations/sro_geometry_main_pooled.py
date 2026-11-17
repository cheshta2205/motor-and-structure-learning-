"""Pooled across-seed version of sro.fig_geometry (the 5-panel Bernardi figure).

Panels A, B and E come straight from the per-seed npz that sro.py --figures
already writes.  Panels C and D (CCGP and PS against trials since the switch)
are NOT in that npz, so they are recomputed here and cached, the same pattern
used for the behaviour pooling.
"""
import os, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sro

ARM, KEY = "main", "h"
# The three named variables land within 0.02 of each other in the decoding
# column and within 0.01 in PS, so overlaid markers occlude one another.  Dodge
# them sideways inside their column rather than shrinking the markers.
DODGE = {"context": -0.10, "value": 0.0, "action": 0.10}
OUT = os.path.join("figures", "main", sro.TASK_DIR, "pooled_seeds")
TSS_NPZ = "sro_geometry_bytss_{arm}_{key}.npz"


def measure(seeds, force=False):
    """Cache the time-resolved CCGP/PS per seed (panels C and D)."""
    for s in seeds:
        p = os.path.join(sro.fig_dir(s), TSS_NPZ.format(arm=ARM, key=KEY))
        if os.path.exists(p) and not force:
            print(f"[seed {s}] cached"); continue
        o = sro.collect(ARM, s)
        if o is None:
            print(f"[seed {s}] no checkpoint"); continue
        ks, gt = sro.geometry_by_tss(o, key=KEY)
        store = {"ks": np.asarray(ks, float)}
        for nm in gt:
            for m in ("ccgp", "ps"):
                store[f"{nm}|{m}"] = np.asarray(gt[nm][m], float)
        np.savez(p, **store)
        print(f"[seed {s}] saved -> {os.path.basename(p)}", flush=True)


def load():
    A, T, ks = {}, {}, None
    for s in range(10):
        d = sro.fig_dir(s)
        f = os.path.join(d, f"sro_geometry_{ARM}_{KEY}.npz")
        if os.path.exists(f):
            z = np.load(f)
            for k in z.files:
                A.setdefault(k, []).append(z[k])
        g = os.path.join(d, TSS_NPZ.format(arm=ARM, key=KEY))
        if os.path.exists(g):
            z = np.load(g)
            ks = z["ks"]
            for k in z.files:
                if k != "ks":
                    T.setdefault(k, []).append(z[k])
    return A, T, ks


def _unnamed_idx():
    """Indices of the 32 dichotomies that are NOT a task variable.

    Context, value and action are themselves three of the 35 balanced splits,
    so leaving them in the grey cloud invites the reading that the spread comes
    from them.  It does not: it comes from splits with no task interpretation,
    which is exactly what makes them the reference distribution.
    """
    named = {frozenset(P) for P, _ in sro._named().values()}
    named |= {frozenset(N) for _, N in sro._named().values()}
    return [i for i, (P, N) in enumerate(sro._all_dichotomies())
            if frozenset(P) not in named and frozenset(N) not in named]


def figure(path=None):
    A, T, ks = load()
    n = max(len(v) for v in A.values())
    keep = _unnamed_idx()
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.42, wspace=.32)
    axA, axB, axE = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                     fig.add_subplot(gs[:, 2]))
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)

    # A -- each of the 35 dichotomies as its across-seed mean.  NOT one point
    # per dichotomy per seed: 350 points is unreadable, and the seed spread is
    # already carried by the error bars on the named three.  The grey cloud is
    # the reference distribution that makes context CCGP interpretable, so it
    # is thinned rather than dropped.
    for i, key in enumerate(("dec", "ccgp")):
        vals = np.vstack(A[key]).mean(0)[keep]
        axA.scatter(np.full(len(vals), i) + rng.normal(0, .04, len(vals)), vals,
                    s=14, facecolor="#aaa", edgecolor="#aaa", lw=.7, zorder=1,
                    label="other dichotomies (no task variable)" if i == 0 else None)
        for nm, c in sro.VCOL.items():
            v = np.array(A[f"{nm}|{key}"])
            axA.errorbar([i + DODGE[nm]], [v.mean()],
                         yerr=[v.std(ddof=1) / np.sqrt(len(v))],
                         fmt="o", ms=9, color=c, mec="k", mew=.6, capsize=3,
                         zorder=3, label=sro.VLAB[nm] if i == 0 else None)
    axA.axhline(.5, ls="--", color="#888", lw=1)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["decoding", "CCGP"])
    # 0, not 0.15: pooled value CCGP is 0.06 and was being clipped off the axis
    axA.set_ylabel("accuracy"); axA.set_ylim(0, 1.02)
    axA.set_title("A  decoding & CCGP\n(32 unnamed dichotomies in grey, "
                  "each averaged over seeds)", fontsize=10)
    axA.legend(frameon=False, fontsize=8)

    # B -- parallelism
    vals = np.vstack(A["ps"]).mean(0)[keep]
    axB.scatter(rng.normal(0, .04, len(vals)), vals, s=14, color="#aaa")
    for nm, c in sro.VCOL.items():
        v = np.array(A[f"{nm}|ps"])
        axB.errorbar([DODGE[nm]], [v.mean()], yerr=[v.std(ddof=1) / np.sqrt(len(v))],
                     fmt="o", ms=9, color=c, mec="k", mew=.6, capsize=3, zorder=3)
    axB.axhline(0, ls="--", color="#888", lw=1)
    axB.set_xticks([]); axB.set_ylabel("parallelism score")
    axB.set_title("B  parallelism score", fontsize=10)

    # C, D -- the inference dynamics
    for ax, m, ttl in ((axC, "ccgp", "C  CCGP over time"),
                       (axD, "ps", "D  PS over time")):
        for nm, c in sro.VCOL.items():
            k = f"{nm}|{m}"
            if k not in T:
                continue
            a = np.vstack(T[k])
            mu = np.nanmean(a, 0)
            se = np.nanstd(a, 0, ddof=1) / np.sqrt(a.shape[0])
            ax.plot(ks, mu, "-o", ms=3, color=c, label=sro.VLAB[nm])
            ax.fill_between(ks, mu - se, mu + se, color=c, alpha=.2, lw=0)
        ax.axhline(.5 if m == "ccgp" else 0, ls="--", color="#888", lw=1)
        ax.axvline(0, ls=":", color="#e67e22", lw=1.5)
        ax.set_xlabel("trials relative to context switch")
        ax.set_ylabel(m.upper()); ax.set_title(ttl, fontsize=10)

    # E -- shattering dimensionality against context CCGP
    sd = np.array(A["sd"])
    axE.bar([0], [sd.mean()], yerr=[sd.std(ddof=1) / np.sqrt(len(sd))],
            color="#cccccc", width=.6, capsize=4)
    for j, nm in enumerate(sro.VCOL):
        v = np.array(A[f"{nm}|ccgp"])
        axE.errorbar([1 + DODGE[nm]], [v.mean()],
                     yerr=[v.std(ddof=1) / np.sqrt(len(v))],
                     fmt="o", ms=9, color=sro.VCOL[nm], mec="k", mew=.6, capsize=3)
    axE.axhline(.5, ls="--", color="#888", lw=1)
    axE.set_xticks([0, 1]); axE.set_xticklabels(["SD", "CCGP"])
    axE.set_ylim(0, 1.02)
    axE.set_title(f"E  shattering dim & abstraction\nSD = {sd.mean():.2f} "
                  f"$\\pm$ {sd.std(ddof=1)/np.sqrt(len(sd)):.2f}", fontsize=10)

    for a_ in (axA, axB, axC, axD, axE):
        sro._despine(a_)
    fig.suptitle(f"model {ARM} arm, {KEY} -- geometry of abstraction, pooled over "
                 f"{n} seeds (mean $\\pm$ s.e.m.; grey = the 32 dichotomies that "
                 "are not task variables)",
                 fontsize=11.5)
    os.makedirs(OUT, exist_ok=True)
    path = path or os.path.join(OUT, f"sro_geometry_{ARM}_{KEY}_pooled.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig); print(f"saved -> {path}  ({n} seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    a = ap.parse_args()
    if a.measure:
        measure(a.seeds, force=a.force)
    if a.figure or not a.measure:
        figure()
