"""Robustness of the OLM/ORM dissociation to the observation model.

THE QUESTION.  Three numbers in the observation model are free choices: how far
the agent can see (SENSE_R), how far off its heading it can see (BETA0), and how
far it can feel (R_ID).  Beyond each, noise is added.  Does the dissociation --
cerebellar damage costs location and spares identity -- depend on where those
lines were drawn?

WHAT IS AND IS NOT SWEPT.  The STRUCTURAL asymmetry is not a free choice and is
not swept: bearing degrades with distance AND angle, shape with distance alone,
because whiskers are omnidirectional at contact and vision is not.  That follows
from sensor physics.  What is swept is where each falloff begins.

    angular clean edge   50   70   90  [109]  130       beta0 68 96 123 [150] 178
    whisker reach       0.2  0.4  0.6  [0.8]  1.0  1.2
    vision radius       0.4  0.6  [0.9]  1.2  1.6

Bracketed values are the setting used throughout the paper; they are ALREADY
trained (olm_orm_final_s0.pt) and are not retrained here, so each row costs one
training fewer than it looks.

RETRAINING IS REQUIRED.  The network learns against its observation model.
Measuring a network trained at reach 0.8 under reach 0.4 is a mismatch, not a
sensitivity -- so every setting is trained from scratch.  At ~55 min each, the
13 new settings are about twelve hours.

olm_orm_final.py IS NOT MODIFIED.  Its sensor constants are module globals read
inside ObjectArena.observe, so this script sets them before constructing
anything, and passes its own tagged checkpoint path to train().
"""
import argparse
import os

import numpy as np
import torch

import olm_orm_final as O

SWEEP_DIR = os.path.join(O.CKPT_DIR, "robustness")
FIG_DIR = os.path.join(O.FIG_DIR, "robustness")
OUT_NPZ = os.path.join(FIG_DIR, "sweep.npz")     # legacy single-seed file


def out_npz(seed):
    """One file per seed, so a second seed never overwrites the first."""
    return os.path.join(FIG_DIR, f"sweep_s{seed}.npz")

# (axis, value, beta0_deg, sense_r, r_id).  None means "leave at the default".
DEFAULT = dict(beta0_deg=np.degrees(O.BETA0), sense_r=O.SENSE_R, r_id=O.R_ID)
EDGE_TO_BETA0 = {50: 68, 70: 96, 90: 123, 109: 150, 130: 178}

SETTINGS = (
    [("angle", e, dict(beta0_deg=b)) for e, b in EDGE_TO_BETA0.items()] +
    [("reach", r, dict(r_id=r)) for r in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)] +
    [("vision", v, dict(sense_r=v)) for v in (0.4, 0.6, 0.9, 1.2, 1.6)]
)


def is_default(kw):
    return all(abs(kw[k] - DEFAULT[k]) < 1e-9 for k in kw)


def tag(kw):
    if is_default(kw):
        return ""                      # the paper's setting: use the main ckpt
    return "_" + "_".join(f"{k[:2]}{v:.2f}" for k, v in sorted(kw.items()))


def ckpt_for(seed, kw):
    return (O.ckpt_path(seed) if is_default(kw)
            else os.path.join(SWEEP_DIR, f"olm_orm_sweep_s{seed}{tag(kw)}.pt"))


def apply(kw):
    """Set the sensor geometry.  Must precede any arena construction."""
    O.BETA0 = np.radians(kw.get("beta0_deg", DEFAULT["beta0_deg"]))
    O.SENSE_R = float(kw.get("sense_r", DEFAULT["sense_r"]))
    return float(kw.get("r_id", DEFAULT["r_id"]))          # r_id is an argument


def do_train(seed, epochs):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for axis, val, kw in SETTINGS:
        full = {**DEFAULT, **kw}
        p = ckpt_for(seed, full)
        if os.path.exists(p):
            print(f"[{axis}={val}] exists, skipping  ({os.path.basename(p)})",
                  flush=True)
            continue
        r_id = apply(full)
        print(f"[{axis}={val}] training -> {os.path.basename(p)}", flush=True)
        O.train(seed=seed, epochs=epochs, r_id=r_id, out=p)


def do_measure(seed, n=384):
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = {}
    for axis, val, kw in SETTINGS:
        full = {**DEFAULT, **kw}
        p = ckpt_for(seed, full)
        if not os.path.exists(p):
            print(f"[{axis}={val}] no checkpoint, skipping"); continue
        r_id = apply(full)
        torch.manual_seed(0)
        mtl, cb = O.load(p)
        for task in ("OLM", "ORM"):
            for site in ("cb", "h"):
                for sig in O.SIGMAS:
                    pi = O.preference_index(mtl, cb, task, sig, site, seed,
                                            n=n, r_id=r_id, k=O.PI_WINDOW)
                    rows[(axis, val, task, site, sig)] = pi
        print(f"[{axis}={val}] measured", flush=True)
    dest = out_npz(seed)
    np.savez(dest, **{"|".join(map(str, k)): np.array([v])
                      for k, v in rows.items()})
    print(f"saved -> {dest}")


# ---------------------------------------------------------------------------
AXES = (("angle", r"angular clean edge (deg)", 109),
        ("vision", r"vision radius", 0.9),
        ("reach", r"whisker reach", 0.8))


def figure(path=None, sigma=0.8):
    """Two rows.  Top: does the task still work at this setting?  Bottom: is the
    dissociation preserved?  The top row is the validity gate for the bottom --
    a cost expressed as a fraction of an intact preference that has itself
    collapsed is a fraction of nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import glob
    files = sorted(glob.glob(os.path.join(FIG_DIR, "sweep_s*.npz")))
    if not files:
        files = [OUT_NPZ]
    acc = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            ax_, val, task, site, sig = k.split("|")
            acc.setdefault((ax_, float(val), task, site, float(sig)),
                           []).append(float(z[k][0]))
    d = {k: float(np.mean(v)) for k, v in acc.items()}
    nseed = max(len(v) for v in acc.values())
    print(f"pooling {len(files)} seed file(s): "
          + ", ".join(os.path.basename(f) for f in files))
    vals = {a: sorted({v for (ax_, v, *_) in d if ax_ == a}) for a, _, _ in AXES}

    fig, AX = plt.subplots(2, 3, figsize=(12.6, 6.6))
    for c, (axis, xlab, default) in enumerate(AXES):
        xs = vals[axis]
        # row 1 -- intact PI
        ax = AX[0, c]
        for task, col in (("OLM", "#2f7ab8"), ("ORM", "#c0392b")):
            y = [d.get((axis, x, task, "cb", 0.0), np.nan) for x in xs]
            e = [np.std(acc.get((axis, x, task, "cb", 0.0), [np.nan]), ddof=0)
                 / max(1, len(acc.get((axis, x, task, "cb", 0.0), [0]))) ** .5
                 for x in xs]
            ax.errorbar(xs, y, yerr=e, fmt="-o", ms=4, color=col, lw=1.8,
                        capsize=2, label=task)
        ax.axhline(0.5, ls=":", color="#888", lw=1)
        ax.set_ylim(0.45, 1.03)
        ax.set_title(xlab, fontsize=10.5)
        if c == 0:
            ax.set_ylabel("intact preference index\n(does the task work?)")
            ax.legend(fontsize=8.5, frameon=False)
        # row 2 -- cost of the lesion
        ax2 = AX[1, c]
        for task, col in (("OLM", "#2f7ab8"), ("ORM", "#c0392b")):
            for site, ls in (("cb", "-"), ("h", "--")):
                y, e = [], []
                for x in xs:
                    b = np.array(acc.get((axis, x, task, site, 0.0), [np.nan]))
                    cur = np.array(acc.get((axis, x, task, site, sigma), [np.nan]))
                    m = min(len(b), len(cur))
                    per = np.where(b[:m] - .5 > 1e-9,
                                   100.0 * (1.0 - (cur[:m] - .5) / (b[:m] - .5)),
                                   np.nan)
                    y.append(np.nanmean(per) if m else np.nan)
                    e.append(np.nanstd(per) / max(1, m) ** .5 if m > 1 else 0.0)
                ax2.errorbar(xs, y, yerr=e, fmt=ls + "o", ms=4, color=col, lw=1.8,
                         capsize=2, alpha=1.0 if site == "cb" else 0.55,
                         label=f"{task}, {'cerebellar' if site=='cb' else 'MTL'}"
                               if c == 0 else None)
        ax2.axhline(0.0, ls=":", color="#888", lw=1)
        ax2.set_ylim(-5, 105)
        ax2.set_xlabel(xlab)
        if c == 0:
            ax2.set_ylabel(f"% of above-chance PI lost\nat $\\sigma$={sigma}")
            ax2.legend(fontsize=8, frameon=False)
        for a in (ax, ax2):
            a.axvline(default, color="#1a1a1a", lw=1.0, ls="-", alpha=.35)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)

    fig.suptitle("Does the dissociation depend on where the observation model "
                 "starts adding noise?\nvertical line = the setting used "
                 f"throughout; mean of {nseed} network(s) per point, "
                 "error bars s.e.m. across seeds", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = path or os.path.join(FIG_DIR, "robustness.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"saved -> {path}")


# ---------------------------------------------------------------------------
DECOMP_NPZ = os.path.join(FIG_DIR, "decompose.npz")   # legacy single-seed file


def decomp_npz(seed):
    """Per seed, so the mechanism test can be replicated across networks."""
    return os.path.join(FIG_DIR, f"decompose_s{seed}.npz")


def do_decompose(seed=0, n=384, sigma=0.8):
    import torch
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = {}
    for axis, val, kw in SETTINGS:
        full = {**DEFAULT, **kw}
        p = ckpt_for(seed, full)
        if not os.path.exists(p):
            print(f"[{axis}={val}] no checkpoint"); continue
        r_id = apply(full)
        torch.manual_seed(0)
        mtl, cb = O.load(p)
        for task in ("OLM", "ORM"):
            for cond, kw2 in (("intact", {}), ("lesion", {"cb_noise": sigma})):
                ec, _ = O.two_sessions(mtl, cb, task, True, seed, n=n,
                                       r_id=r_id, **kw2)
                eu, _ = O.two_sessions(mtl, cb, task, False, seed, n=n,
                                       r_id=r_id, **kw2)
                k = O.PI_WINDOW
                rows[(axis, val, task, cond, "changed")] = float(
                    np.nanmean(ec[:, :k]))
                rows[(axis, val, task, cond, "unchanged")] = float(
                    np.nanmean(eu[:, :k]))
        print(f"[{axis}={val}] decomposed", flush=True)
    np.savez(decomp_npz(seed), **{"|".join(map(str, k)): np.array([v])
                            for k, v in rows.items()})
    print(f"saved -> {decomp_npz(seed)}")


def figure_decompose(path=None):
    """The two errors behind the ratio, so a change in PI can be attributed.

    PI is e_changed/(e_changed + e_unchanged), and a ratio can move for two
    reasons.  Plotting the parts shows that the angle and vision columns of the
    robustness sweep move for DIFFERENT reasons: angle acts on the unchanged
    object under lesion, vision on the changed object when intact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(DECOMP_NPZ)
    d = {}
    for k in z.files:
        ax_, val, task, cond, which = k.split("|")
        d[(ax_, float(val), task, cond, which)] = float(z[k][0])

    SER = (("intact", "changed", "#1a1a1a", "-", "o"),
           ("intact", "unchanged", "#1a1a1a", "--", "s"),
           ("lesion", "changed", "#c0392b", "-", "o"),
           ("lesion", "unchanged", "#c0392b", "--", "s"))

    fig, AX = plt.subplots(2, 3, figsize=(12.6, 6.8), sharey="row")
    for r, task in enumerate(("OLM", "ORM")):
        for c, (axis, xlab, default) in enumerate(AXES):
            ax = AX[r, c]
            xs = sorted({v for (a, v, *_) in d if a == axis})
            for cond, which, col, ls, mk in SER:
                y = [d.get((axis, x, task, cond, which), np.nan) for x in xs]
                ax.plot(xs, y, ls, marker=mk, ms=4, color=col, lw=1.7,
                        alpha=1.0 if which == "changed" else 0.55,
                        label=f"{cond}, {which}" if (r == 0 and c == 0) else None)
            ax.set_yscale("log")
            ax.axvline(default, color="#1a1a1a", lw=1.0, alpha=.3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(xlab, fontsize=10.5)
            if r == 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(f"{task}\nprediction error (first "
                              f"{O.PI_WINDOW} steps)")
    AX[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("What moves the preference: the changed object, or the "
                 "unchanged one?\nPI is a ratio of these two; the vertical line "
                 "is the setting used throughout", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = path or os.path.join(FIG_DIR, "decompose.png")
    fig.savefig(path, dpi=170); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--fig-decompose", action="store_true",
                    dest="fig_decompose")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=O.EPOCHS)
    ap.add_argument("--n", type=int, default=384)
    a = ap.parse_args()
    if a.train:
        do_train(a.seed, a.epochs)
    if a.measure:
        do_measure(a.seed, n=a.n)
    if a.figure:
        figure()
    if a.decompose:
        do_decompose(a.seed, n=a.n)
    if a.fig_decompose:
        figure_decompose()


# ---------------------------------------------------------------------------
# Figure


# ---------------------------------------------------------------------------
# Decomposition: PI is a ratio, so it can move because the CHANGED object's
# error moved, because the UNCHANGED object's did, or both.  The sweep saved
# only the ratio, which is why the angle column is uninterpretable: cost rises
# with a WIDER clean field, and neither "the observation does the work" nor
# "the estimate does the work" predicts that sign.  Here we record the two
# errors separately.
