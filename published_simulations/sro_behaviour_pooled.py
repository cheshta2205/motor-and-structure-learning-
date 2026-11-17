"""Pooled across-seed version of sro.fig_behaviour.

sro.py draws this figure but never saves its arrays -- only the PNG -- so
pooling it means recomputing the greedy rollouts.  This script caches each
seed's curves to npz on the way through, so every later re-pool is instant.

Left  : context decoding from z, aligned to the switch.
Right : Bernardi Fig 1C -- operant accuracy on the FIRST post-switch
        appearance of each stimulus, by appearance order.
Both  : 2 arms x 3 ablation modes.  The claim is the 2x2 -- zeroing the RPE
        flattens `main` and leaves `control` alone, because `control` is handed
        the outcome directly and does not need the error.
"""
import os, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sro

ARMS = ("main", "control")
MODES = ("intact", "zero", "shuffle")
LS = {"intact": "-", "zero": "--", "shuffle": ":"}
ORDERS = (1, 2, 3, 4)
OUT = os.path.join("figures", "main", sro.TASK_DIR, "pooled_seeds")


def npz_path(seed):
    return os.path.join(sro.fig_dir(seed), "sro_behaviour.npz")


def measure(seed, force=False):
    """Compute (and cache) one seed's curves."""
    p = npz_path(seed)
    if os.path.exists(p) and not force:
        print(f"[seed {seed}] cached"); return
    store = {}
    for arm in ARMS:
        for mode in MODES:
            o = sro.collect(arm, seed, e_mode=mode)
            if o is None:
                print(f"[seed {seed}] no checkpoint for {arm}"); continue
            ks, dec = sro.decode_by_tss(o)
            f1c = sro.fig1c(arm, seed, e_mode=mode)
            store[f"{arm}|{mode}|ks"] = np.asarray(ks, float)
            store[f"{arm}|{mode}|dec"] = np.asarray(dec, float)
            store[f"{arm}|{mode}|f1c"] = np.array(
                [np.nan if f1c is None else f1c[k] for k in ORDERS], float)
            print(f"[seed {seed}] {arm}/{mode}: settled decode "
                  f"{np.nanmean(dec[np.asarray(ks) >= 5]):.3f}", flush=True)
    if store:
        np.savez(p, **store); print(f"  saved -> {p}", flush=True)


def load_all():
    acc, ks = {}, None
    for f in sorted(glob.glob(os.path.join("figures", "main", sro.TASK_DIR,
                                        "seed*", "sro_behaviour.npz")),
                    key=lambda q: int(q.split("seed")[1].split("/")[0])):
        z = np.load(f)
        for k in z.files:
            arm, mode, what = k.split("|")
            if what == "ks":
                ks = z[k]; continue
            acc.setdefault((arm, mode, what), []).append(z[k])
    return acc, ks


def figure(path=None):
    acc, ks = load_all()
    if ks is None:
        raise SystemExit("no cached seeds -- run --measure first")
    n = max(len(v) for v in acc.values())
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 4.9))
    for arm in ARMS:
        for mode in MODES:
            col = sro.ACOL[arm]
            d = acc.get((arm, mode, "dec"))
            if d is not None:
                a = np.vstack(d)
                m, e = np.nanmean(a, 0), np.nanstd(a, 0, ddof=1) / np.sqrt(len(a))
                ax[0].plot(ks, m, LS[mode], color=col, lw=1.9, marker="o", ms=3,
                           label=f"{arm} / {mode}")
                ax[0].fill_between(ks, m - e, m + e, color=col, alpha=.17, lw=0)
            g = acc.get((arm, mode, "f1c"))
            if g is not None:
                a = np.vstack(g)
                m, e = np.nanmean(a, 0), np.nanstd(a, 0, ddof=1) / np.sqrt(len(a))
                ax[1].errorbar(ORDERS, m, yerr=e, fmt=LS[mode] + "s", color=col,
                               lw=1.9, ms=6, capsize=3)
    ax[0].axvline(0, ls=":", color="#e67e22", lw=1.5)
    ax[0].axhline(.5, ls="--", color="#888", lw=1)
    ax[0].set_xlabel("trials relative to context switch")
    ax[0].set_ylabel("context decode accuracy")
    ax[0].set_title("how fast does MTL infer the new context?\n"
                    "decoded from $z$; chance 0.5", fontsize=10.5)
    ax[0].legend(fontsize=8, frameon=False, ncol=2)
    ax[1].axhline(.5, ls="--", color="#888", lw=1)
    ax[1].set_xticks(ORDERS); ax[1].set_xlim(.7, 4.3); ax[1].set_ylim(-.03, 1.05)
    ax[1].set_xlabel("stimulus appearance order after the switch")
    ax[1].set_ylabel("operant accuracy, first appearance")
    ax[1].set_title("inference signature (cf. Bernardi Fig 1C)\n"
                    "order 1 = stale context; 2-4 never seen in the new one",
                    fontsize=10.5)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("SRO task: context inferred from a REWARD prediction error — "
                 f"mean of {n} seeds, shading/bars s.e.m.", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(OUT, exist_ok=True)
    path = path or os.path.join(OUT, "sro_behaviour_pooled.png")
    fig.savefig(path, dpi=170); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig); print(f"saved -> {path}  ({n} seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    a = ap.parse_args()
    if a.measure:
        for s in a.seeds:
            measure(s, force=a.force)
    if a.figure or not a.measure:
        figure()
