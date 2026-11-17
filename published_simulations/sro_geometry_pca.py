"""The eight conditions in the MTL hidden state, in the model's own PCA space.

Why PCA and not MDS: classical MDS places the 8 condition means and nothing
else.  PCA is a linear map, so single trials go into the SAME space and the
spread of each condition can be drawn, not just its mean.  (On the means alone
the two are the same thing -- classical MDS on Euclidean distances IS PCA on
the centred means -- so nothing is lost by the switch.)

Three kinds of line, each making a different claim:

    grey dashed   same stimulus, across contexts.  Parallel = context is one
                  shared displacement = abstraction.
    colour A      within a context, the two stimuli needing the SAME operant
                  (1-2 and 3-4).  Short.
    colour B      within a context, stimuli needing OPPOSITE operants
                  (1-3, 1-4, 2-3, 2-4).  Long.

A and B holding in BOTH contexts is the point: the internal geometry of a
context survives the reversal, even though which operant {1,2} calls for flips.

Stimulus numbering and colours are panel (a)'s, which are NOT sro.py's
ordering -- see FIG_OF_SRO.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sro

TSS_MIN = 5
N_CLOUD = 200                      # trials drawn per condition for the spread

# panel (a) numbers the stimuli so context 1's two "release" stimuli come
# first.  sro index -> figure index, and the same array inverts it.
FIG_OF_SRO = [0, 2, 1, 3]
STIM_COL = ["#4a7fb5", "#3f7a47", "#e8b04b", "#8c6e4f"]   # 1 2 3 4, as panel a
CTX_MARK = ["o", "s"]
TOGETHER = [(0, 1), (2, 3)]                               # figure numbering
CROSS = [(0, 2), (0, 3), (1, 2), (1, 3)]
A_COL, B_COL, DASH_COL = "#6a3d9a", "#9fb3bd", "#9e9e9e"


def collect(arm="main", seed=0, e_mode="intact", rng=None):
    """Z-scored trials, their condition ids, and the 8 condition means."""
    out = sro.collect(arm, seed, e_mode=e_mode)
    if out is None:
        raise SystemExit(f"no checkpoint for {arm} seed {seed}")
    X = out["h"].reshape(-1, out["h"].shape[-1]).numpy()
    cond = sro._cond_ids(out, prev=True)
    keep = out["tss"].ravel() >= TSS_MIN
    X, cond = X[keep], cond[keep]
    sd = X.std(0, keepdims=True)                 # as in Bernardi's MDS recipe
    X = (X - X.mean(0, keepdims=True)) / np.where(sd < 1e-8, 1.0, sd)
    M = np.stack([X[cond == c].mean(0) for c in range(8)])
    return X, cond, M


def pca(A, k=3):
    """Axes from the CONDITION MEANS, not from single trials.

    Fitting on trials puts the leading components where the trial-to-trial
    noise is, which is an order of magnitude larger than the differences
    between conditions -- the eight means then collapse into two dots and the
    within-context geometry is invisible.  Fitting on the means puts the axes
    where the structure is; because PCA is a linear map, single trials still
    project into the same space, so the spread can be drawn on top.
    """
    mu = A.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return mu, Vt[:k], var[:k]


def _pairs(P, ctx, idx, col, lw, alpha, ax, dims):
    for a, b in idx:
        p, q = P[ctx, a], P[ctx, b]
        seg = np.stack([p, q])[:, :dims]
        ax.plot(*seg.T, color=col, lw=lw, alpha=alpha, zorder=2)


def _plane(ax, w, b, Y, dims):
    """Context decoder fitted on the plotted coordinates, drawn across them."""
    lo, hi = Y[:, :dims].min(0), Y[:, :dims].max(0)
    rng = np.maximum(hi - lo, 1e-9)
    lo, hi = lo - .08 * rng, hi + .08 * rng   # hugging the data:
    # a plane drawn well past the points reads, in perspective, as cutting
    # THROUGH the ones it is merely in front of
    if dims == 2:
        ys = np.array([lo[1], hi[1]])
        ax.plot(-(w[1] * ys + b) / w[0], ys, color="#c9a227", lw=1.5, zorder=1)
        return
    j = int(np.argmax(np.abs(w)))
    oth = [i for i in range(3) if i != j]
    g1, g2 = np.meshgrid(np.linspace(lo[oth[0]], hi[oth[0]], 2),
                         np.linspace(lo[oth[1]], hi[oth[1]], 2))
    gj = -(w[oth[0]] * g1 + w[oth[1]] * g2 + b) / w[j]
    V = np.empty(g1.shape + (3,))
    V[..., oth[0]], V[..., oth[1]], V[..., j] = g1, g2, gj
    ax.add_collection3d(Poly3DCollection([[V[0, 0], V[0, 1], V[1, 1], V[1, 0]]],
                                         facecolor="#e8d9a0", alpha=.18,
                                         edgecolor="#c9a227", linewidths=.8))


def figure(arm="main", seed=0, e_mode="intact", dims=2, cloud=True,
           plane=None, holdout=3, path=None):
    plane = True if plane is None else plane
    rng = np.random.default_rng(0)
    X, cond, M = collect(arm, seed, e_mode)
    mu, W, var = pca(M, 3)
    Y = (M - mu) @ W.T                                   # the 8 means

    ctx = np.array([c // sro.N_STIM for c in range(8)])
    stm = np.array([c % sro.N_STIM for c in range(8)])
    fig_i = np.array([FIG_OF_SRO[s] for s in stm])
    pays = sro.SRO[ctx, stm, 1]
    P = np.zeros((2, 4, 3))                              # [context, fig stim]
    for c in range(8):
        P[ctx[c], fig_i[c]] = Y[c]

    # the held-out-stimulus context decoder, fitted in plotted coordinates
    w = b = None
    if plane:
        s_out = [s for s in range(4) if FIG_OF_SRO[s] == holdout][0]
        tr = stm != s_out
        A = np.c_[Y[tr][:, :dims], np.ones(tr.sum())]
        t = np.where(ctx[tr] == 1, 1.0, -1.0)
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        w, b = coef[:dims], coef[dims]
        pred = (Y[:, :dims] @ w + b > 0).astype(int)
        ok = bool((pred[~tr] == ctx[~tr]).all())
        print(f"  held-out stimulus {holdout + 1}: pair "
              f"{'correct' if ok else 'MISCLASSIFIED'}")

    if dims == 2:
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
    else:
        fig = plt.figure(figsize=(7.4, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            a_.pane.set_visible(False); a_.line.set_color("#444")

    if plane:
        _plane(ax, w, b, Y, dims)

    if cloud:
        for c in range(8):
            ix = np.flatnonzero(cond == c)
            ix = rng.choice(ix, min(N_CLOUD, len(ix)), replace=False)
            T = (X[ix] - mu) @ W.T
            ax.scatter(*T[:, :dims].T, s=4, alpha=.10, lw=0,
                       color=STIM_COL[FIG_OF_SRO[c % 4]], zorder=1)

    for c in range(2):                                   # within-context lines
        _pairs(P, c, CROSS, B_COL, 1.0, .85, ax, dims)
        _pairs(P, c, TOGETHER, A_COL, 2.2, 1.0, ax, dims)
    for f in range(4):                                   # across contexts
        seg = np.stack([P[0, f], P[1, f]])[:, :dims]
        ax.plot(*seg.T, color=DASH_COL, lw=1.2, ls=(0, (4, 3)), zorder=2)

    for c in range(8):
        f = fig_i[c]
        # the pair the decoder never saw is hollow, as in the MDS panel: the
        # fill carries stimulus identity here, so "held out" has to be the
        # ABSENCE of fill rather than a colour of its own
        held = plane and f == holdout
        ax.scatter(*Y[c, :dims], s=150, marker=CTX_MARK[ctx[c]],
                   facecolors="white" if held else STIM_COL[f],
                   edgecolors=STIM_COL[f] if held else "white",
                   lw=2.2 if held else 1.4, zorder=6)
        lab = f"{f + 1}{'+' if pays[c] else '-'}"
        off = [(-16, 12), (16, 12), (-16, -16), (16, -16)][f]
        if dims == 2:
            ax.annotate(lab, Y[c, :2], textcoords="offset points", xytext=off,
                        ha="center", fontsize=10, color=STIM_COL[f])
        else:
            sp = np.ptp(Y, axis=0)
            ax.text(Y[c, 0] + off[0] / 150 * sp[0], Y[c, 1],
                    Y[c, 2] + off[1] / 150 * sp[2], lab, fontsize=9.5,
                    color=STIM_COL[f], ha="center")

    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    if dims == 3:
        ax.set_zlabel("PC 3"); ax.view_init(elev=14, azim=-78)
    else:
        for q in ("top", "right"):
            ax.spines[q].set_visible(False)
    ttl = ("MTL hidden state: the eight conditions\n"
           "circles context 1, squares context 2")
    if plane:
        ttl += f"; boundary fitted without stimulus {holdout + 1} (hollow)"
    ax.set_title(ttl, fontsize=10.5)
    fig.tight_layout()

    d = sro.pooled_dir()
    tag = "" if dims == 2 else "_3d"
    path = path or os.path.join(
        d, f"sro_geometry_pca_{arm}_{e_mode}_seed{seed}{tag}.png")
    fig.savefig(path, dpi=200); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)

    D = np.linalg.norm(M[:, None] - M[None], axis=-1)
    def _d(pairs):
        return np.mean([D[c * 4 + FIG_OF_SRO.index(a), c * 4 + FIG_OF_SRO.index(b)]
                        for c in range(2) for a, b in pairs])
    print(f"  variance explained: {np.round(var, 3)}  (sum {var.sum():.3f})")
    print(f"  mean distance  same-operant pairs {_d(TOGETHER):6.2f}   "
          f"different-operant pairs {_d(CROSS):6.2f}   "
          f"across contexts {D[:4, 4:].mean():6.2f}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="main")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--e-mode", default="intact",
                   choices=["intact", "zero", "shuffle"])
    p.add_argument("--dims", type=int, default=2, choices=[2, 3])
    p.add_argument("--no-cloud", action="store_true")
    p.add_argument("--plane", action="store_true", default=None)
    p.add_argument("--no-plane", dest="plane", action="store_false")
    a = p.parse_args()
    figure(a.arm, a.seed, a.e_mode, a.dims, not a.no_cloud, a.plane)
