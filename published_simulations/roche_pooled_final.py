"""Pooled Rochefort figures for the FINAL configuration.

Why this exists: rochefort_final.py writes pooled figures into fig_dir(seeds[0]),
which is seed0's own folder -- the same confusion the old Rochefort_2011 README
records.  Here they go to ROCHEFORT_FINAL/pooled_ac_seeds/ instead, and seed 0's
own per-seed set is generated too (the training queue began at seed 1, so seed0/
was left empty).

NOTE ON PROVENANCE: figures/main/Rochefort_2011/pooled_seeds0-9/ are from the
OLD script -- the version in which the cerebellum substituted for the cue in the
light as well as the dark.  They are NOT the final model and must not be used.
"""
import argparse
import os
import time

import numpy as np

import rochefort_final as R

# NOT R.FIG_ROOT -- that is figures/, the parent of every paradigm's folder.
# The pooled set belongs beside the per-seed folders it summarises.
POOL = os.path.join(os.path.dirname(os.path.normpath(R.fig_dir(0))),
                    "pooled_ac_seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(10)))
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    fov = np.deg2rad(R.FOV_DEG)
    os.makedirs(POOL, exist_ok=True)
    R._POOLED_TAG = None                 # no sigma suffix; the folder says it

    for name, fn in (("bars_2x2", R.plot_bars_2x2),
                     ("map_similarity", R.plot_map_similarity)):
        t = time.time()
        print(f"[{name}] pooling {len(a.seeds)} seeds ...", flush=True)
        fn(seeds=tuple(a.seeds), sigma=a.sigma, fov=fov, n=a.n,
           path=os.path.join(POOL, f"{name}.png"))
        print(f"[{name}] {time.time() - t:.0f}s -> {POOL}", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The two figures rochefort_final.py cannot pool: they take a single `models`
# dict rather than a list of seeds.  Here the per-seed curves are collected and
# averaged, with the spread ACROSS independently trained networks as the error
# bar -- which is the one that matters.
# ---------------------------------------------------------------------------
def _models_for(seed, sigma, fov):
    return R.get_models(fov, seed, R.CKPT_DIR, R.EPOCHS, force=False,
                        cb_noise=sigma)


def pooled_session_length(seeds, sigma, fov, n=40, Ts=(300, 600, 1200, 2400),
                          path=None):
    """Stability vs dark session length, and CB's open-loop drift, per seed."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, drift = {}, {}
    for seed in seeds:
        models = _models_for(seed, sigma, fov)
        for tag, (m, c) in models.items():
            nz = getattr(c, "cb_noise", 0.0)
            for p_ in ("cue_full", "cue_none"):
                res.setdefault((tag, p_), []).append(
                    [R.arena_session_test(m, c, p_, n=n, T=T, fov=fov,
                                          cb_noise=nz)["stability"][3]
                     for T in Ts])
            d = R.cb_accuracy(m, c, n=128, T=max(Ts) + 80, gap_at=50, fov=fov,
                              cb_noise=nz,
                              ks=(10, 50, 150, 300, 600, 1200, 2400))["drift_rel"]
            drift.setdefault(tag, []).append(d)
        print(f"   seed {seed} done", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, col in (("WT", "#2f7ab8"), ("L7PKCI", "#c0392b")):
        for p_, style, lab in (("cue_full", "s--", "light"),
                               ("cue_none", "o-", "dark")):
            v = np.array(res[(tag, p_)])
            axes[0].errorbar(Ts, v.mean(0), yerr=v.std(0) / np.sqrt(len(v)),
                             fmt=style, color=col, ms=5, capsize=3,
                             label=f"{tag} {lab}")
        ks = list(drift[tag][0].keys())
        v = np.array([[d[k] for k in ks] for d in drift[tag]])
        axes[1].errorbar(ks, v.mean(0), yerr=v.std(0) / np.sqrt(len(v)),
                         fmt="o-", color=col, ms=5, capsize=3, label=tag)
    axes[0].set_ylabel("place field stability  (S2 vs S4)")
    axes[0].set_xlabel("dark session length (steps)")
    axes[0].set_xscale("log"); axes[0].set_xticks(Ts)
    axes[0].set_xticklabels([str(t) for t in Ts])
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_ylabel("CB open-loop error (relative)")
    axes[1].set_xlabel("steps since the last anchored observation")
    axes[1].set_xscale("log"); axes[1].legend(fontsize=8, frameon=False)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Stability falls with unanchored time, and CB's error is why\n"
                 f"{len(seeds)} seeds, mean $\\pm$ SEM across networks",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=170)
    fig.savefig(str(path).replace(".png", ".svg"))
    plt.close(fig)
    print(f"saved -> {path}", flush=True)
