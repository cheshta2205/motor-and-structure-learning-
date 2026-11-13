"""
switch_dynamics.py -- how z moves through its context space after a switch.

heldout_representation.py shows where each context SETTLES in z: rotations on
PC1, curls on PC2, drifts on a ring in PC3-PC4 of the all-context PCA. This
draws the path z takes to get there, step by step, around a context switch,
in those same planes (user, 2026-10-15). Trained contexts only for now; the
held-out contexts come later. One seed, z only.

Switches (clean ones only: at least PRE steps of the old context before, and
POST steps of the new one after):

  within family   rot -60 -> rot +60    curl -13 -> curl +13    drift 0 -> drift 180
  across family   rot +60 -> curl +13   curl +13 -> rot +60     rot +60 -> drift 90

Each panel: the mean trajectory over every clean switch of that pair, from
PRE steps before the switch to POST after, a marker per step coloured by
time since the switch, with the 95% interval of the mean as an ellipse at
each step; every trained centroid in grey,
the old and new ones labelled. Rotation and curl panels in PC1-PC2; drift in
PC3-PC4; rot -> drift on PC1 against the drift-90 direction in PC3-PC4.

    python switch_dynamics.py --seed 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe

import model as M
import task as T
from data import generate_sequences
from plotting_utils import (use_paper_style, offset_spines, figsize, colorbar, label_panels,
                            save, INK, MUTED)

use_paper_style()

PRE, POST = 3, 12
FAM_MARKER = {0: "o", 1: "s", 2: "^"}       # rotation, curl, drift
N_SEQ = 4800            # ~80 clean switches per ordered pair
PAIRS = {
    "within": [("rot -60", "rot +60"), ("curl -13", "curl +13"), ("drift 0", "drift 180")],
    "across": [("rot +60", "curl +13"), ("curl +13", "rot +60"), ("rot +60", "drift 90")],
}
ID = {lab: c for c, lab in T.C_LABEL.items() if T.is_table_context(c)}


def _fam_val(label):
    """'rot -60' -> ('Rotation', '−60°'); 'curl +13' -> ('Curl', '+13')."""
    kind, _, val = label.partition(" ")
    fam = {"rot": "Rotation", "curl": "Curl", "drift": "Drift"}[kind]
    return fam, val.replace("-", "−") + ("°" if kind in ("rot", "drift") else "")


def rollout(arm, seed, n=N_SEQ):
    loaded = M.load(arm, seed)
    if loaded is None:
        return None
    ec, cb = loaded
    trans, ctx, tss = generate_sequences(n, seed=123)
    with torch.no_grad():
        _, _, _, sites = M.rollout(ec, cb, trans, arm, "intact",
                                   torch.Generator().manual_seed(0), sites=True)
    z = sites["z"].numpy()
    return z.reshape(n, -1, z.shape[-1]), ctx[:, :, 0].numpy().astype(int), tss.numpy()


def clean_switches(cid, tss, a, b):
    """(sequence, step) of every switch a -> b with PRE steps of a before it
    and POST steps of b from it on."""
    out = []
    B, L = cid.shape
    for s in range(B):
        for i in range(PRE, L - POST):
            if (cid[s, i - 1] == a and cid[s, i] == b and np.all(cid[s, i - PRE:i] == a)
                    and np.all(cid[s, i:i + POST + 1] == b)):
                out.append((s, i))
    return out


def plane(pair, V, cent):
    """The two axes a pair's panel is drawn in, as rows of a (2, dim) matrix,
    and their labels. Rotation and curl: PC1-PC2. Drift: PC3-PC4. Rotation to
    drift: PC1 against the unit direction of the drift centroid in PC3-PC4."""
    fams = {T.FAMILY[ID[pair[0]]], T.FAMILY[ID[pair[1]]]}
    if fams <= {0, 1}:
        return V[[0, 1]], ("PC1", "PC2")
    if fams == {2}:
        return V[[2, 3]], ("PC3", "PC4")
    d = [ID[p] for p in pair if T.FAMILY[ID[p]] == 2][0]
    u = cent[d] @ V[[2, 3]].T
    u = u / np.linalg.norm(u)
    return np.vstack([V[0], u @ V[[2, 3]]]), ("PC1", f"{T.C_LABEL[d]} direction (PC3–PC4)")


def fig_switch_dynamics(seed, z, cid, tss, path, into=None, start="a"):
    """Draw the switch-dynamics figure; `into` and `start` as in
    heldout_representation.fig_heldout (nothing saved, caption returned)."""
    keep = tss >= T.TSS_MIN
    Zs = z[keep]
    mu = Zs.mean(0)
    _, S, Vt = np.linalg.svd(Zs - mu, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    from heldout_representation import orient_pcs
    V = orient_pcs(Vt[:4], mu, z[keep], cid[keep])       # the held-out figures' orientation
    cent = {c: z[keep & (cid == c)].mean(0) - mu for c in range(T.N_CONTEXTS)}
    rng = np.random.default_rng(0)
    steps = np.arange(-PRE, POST + 1)
    # time along the trajectory in greys, so it cannot be confused with the
    # contexts' parameter colours
    cmap, norm = plt.get_cmap("Greys"), plt.Normalize(-PRE - 4, POST)
    from heldout_representation import _norm_for, FAM_IDX, heldout_ids
    fam_name = {i: n for n, i in FAM_IDX.items()}
    # the colour range over trained AND held-out values, exactly as in the
    # held-out figures, so a context has the same colour in both
    fam_ids = {f: v["trained"] + v["interp"] + v["extrap"] for f, v in heldout_ids().items()}
    def ref_col(c):
        f = fam_name[T.FAMILY[c]]
        cm, lo, hi = _norm_for(f, fam_ids[f])
        v = float(T.C_PARAM[c]) % 360.0 if f == "drift" else float(T.C_PARAM[c])
        return plt.get_cmap(cm)(plt.Normalize(lo, hi)(v))

    if into is not None:
        fig, axes = into, into.subplots(2, 3)
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize("double", height=5.4), layout="constrained")
    counts = {}
    for r, kind in enumerate(("within", "across")):
        for k, (a_lab, b_lab) in enumerate(PAIRS[kind]):
            ax = axes[r, k]
            a, b = ID[a_lab], ID[b_lab]
            P, labels = plane((a_lab, b_lab), V, cent)
            sw = clean_switches(cid, tss, a, b)
            counts[(a_lab, b_lab)] = len(sw)
            # every trained centroid, coloured as in the held-out figures
            # (user, 2026-10-15): rotation by angle, curl by gain, drift by
            # direction; the two in play larger and labelled
            for c in range(T.N_CONTEXTS):
                q = cent[c] @ P.T
                # the old and new contexts on TOP of the trajectory, so its
                # first and last steps cannot hide them
                # family by marker shape (user, 2026-10-15); the two contexts
                # of the switch enlarged with an ink edge -- the arrows say
                # which is the start, so they are styled alike
                mk = FAM_MARKER[T.FAMILY[c]]
                if c in (a, b):
                    ax.scatter(*q, s=85, marker=mk, facecolor=ref_col(c), edgecolor=INK,
                               linewidth=1.0, zorder=7)
                else:
                    ax.scatter(*q, s=18, marker=mk, color=ref_col(c), lw=0, zorder=1)
            for c, lab in ((a, "start"), (b, "end")):
                q = cent[c] @ P.T
                # "Start:\nRotation\n(−60°)" (user, 2026-10-15)
                kind_, _, val = T.C_LABEL[c].partition(" ")
                fam = {"rot": "Rotation", "curl": "Curl", "drift": "Drift"}[kind_]
                val = val.replace("-", "−") + ("°" if kind_ in ("rot", "drift") else "")
                ax.annotate(f"{fam}\n({val})", q, xytext=(7, 0),
                            textcoords="offset points", va="center", linespacing=1.0,
                            fontsize=plt.rcParams["xtick.labelsize"], color=INK, zorder=8)
            if not sw:
                continue
            traj = np.stack([(z[s, i - PRE:i + POST + 1] - mu) @ P.T for s, i in sw])  # (n, T, 2)
            m = traj.mean(0)
            # the 95% interval of the mean at each step, as an ellipse
            for t_ in range(len(steps)):
                C = np.cov(traj[:, t_, :].T) / len(traj)
                w, v = np.linalg.eigh(C)
                ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
                ax.add_patch(Ellipse(m[t_], 2 * 1.96 * np.sqrt(w[1]), 2 * 1.96 * np.sqrt(w[0]),
                                     angle=ang, facecolor=cmap(norm(steps[t_])), alpha=0.25,
                                     lw=0, zorder=3))
            # the path as a chain of arrows, one per step (user, 2026-10-15),
            # so its direction of travel reads at every step; steps too short
            # to hold an arrowhead (z settled) are drawn as plain segments
            span = float(np.ptp(m, axis=0).max()) or 1.0
            for t_ in range(len(m) - 1):
                long_enough = np.linalg.norm(m[t_ + 1] - m[t_]) > 0.04 * span
                ax.annotate("", xy=m[t_ + 1], xytext=m[t_], zorder=4,
                            arrowprops=dict(arrowstyle="-|>" if long_enough else "-", color=INK,
                                            lw=0.9, mutation_scale=8, shrinkA=3, shrinkB=3))
            sc = ax.scatter(m[:, 0], m[:, 1], c=steps, cmap=cmap, norm=norm, s=16,
                            edgecolor=INK, linewidth=0.4, zorder=5)
            # the switch itself: the first step of the new context
            ax.scatter(*m[PRE], s=46, facecolor="none", edgecolor=INK, linewidth=1.0, zorder=6)
            # family over perturbation (user, 2026-10-15): "Rotation\n(−60° → +60°)"
            # within a family, "Rotation → Curl\n(+60° → +13)" across
            fa, va_ = _fam_val(a_lab)
            fb, vb_ = _fam_val(b_lab)
            ax.set_title(f"{fa}\n({va_} → {vb_})" if fa == fb else f"{fa} → {fb}\n({va_} → {vb_})")
            ax.set_xlabel(labels[0] if not labels[0].startswith("PC") else
                          f"{labels[0]} ({var[int(labels[0][2:]) - 1]:.0%})")
            ax.set_ylabel(labels[1] if not labels[1].startswith("PC") else
                          f"{labels[1]} ({var[int(labels[1][2:]) - 1]:.0%})")
            ax.autoscale_view()
    colorbar(sc, ax=axes[:, -1], label="steps since switch", ticks=[-PRE, 0, POST // 2, POST])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker=FAM_MARKER[i], ls="", color=MUTED, label=n)
                        for i, n in enumerate(T.FAM_NAMES)]
               , loc="outside upper center", ncol=3)
    for a_ in axes.ravel():
        offset_spines(a_)
    label_panels(fig, list(axes[:, 0]), start=start)
    n_txt = ", ".join(f"{a} → {b} {n}" for (a, b), n in counts.items())
    caption = (
        f"How z moves after a context switch, main circuit seed {seed}. Top row (a): switches "
        f"within a family; bottom row (b): across families. Each panel: the mean trajectory of "
        f"z over every clean switch of that pair (at least {PRE} steps of the old context "
        f"before and {POST} of the new one after; {n_txt}), from {PRE} steps before the switch "
        f"to {POST} after, a marker per step coloured by steps since the switch (ringed: the "
        f"first step of the new context; arrows, the direction of travel at each step), with the 95% interval of the mean as an ellipse at "
        f"each step. Coloured points are the twelve trained contexts' settled centroids, "
        f"coloured as in the held-out figures (rotation by angle, curl by gain, drift by "
        f"direction; marker shape the family: circle rotation, square curl, triangle drift); the switch's two contexts enlarged and labelled (tss >= {T.TSS_MIN}), the old and new ones labelled. Axes "
        f"from the PCA of all settled steps: rotation and curl panels in PC1-PC2, drift in "
        f"PC3-PC4, rotation to drift on PC1 against the drift-90 direction in PC3-PC4.")
    if into is not None:
        return caption
    save(fig, path, caption=caption)
    print(f"  saved -> {path}  ({n_txt})")
    return caption


def fig_combined(seed, arm, path):
    """The held-out geometry (top, a-f) and the switch dynamics (bottom, g-h)
    in one figure (user, 2026-10-15); both are also saved on their own."""
    import heldout_representation as H
    from representation_analysis import collect
    ids = H.heldout_ids()
    pool = list(range(T.N_CONTEXTS)) + [c for f in ids.values() for k in ("interp", "extrap")
                                        for c in f[k]]
    d = collect(arm, seed, n=H.N_SEQ, contexts=pool)
    z, cid, tss = rollout(arm, seed)
    # condensed (user, 2026-10-15): 8.2 in rather than the two figures'
    # 10.6 stacked; the panels give up height, not width
    fig = plt.figure(figsize=figsize("double", height=8.2), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, hspace=0.01)
    top, bottom = fig.subfigures(2, 1, height_ratios=[4.0, 4.2], hspace=0.0)
    cap_top = H.fig_heldout(seed, d, ids, None, into=top, start="a")
    cap_bot = fig_switch_dynamics(seed, z, cid, tss, None, into=bottom, start="g")
    caption = ("Held-out contexts and switch dynamics in z. (a-f) " + cap_top
               + " (g, h) " + cap_bot.replace("Top row (a)", "Top row (g)", 1)
               .replace("bottom row (b)", "bottom row (h)", 1))
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="main")
    ap.add_argument("--combined", action="store_true",
                    help="also draw the held-out geometry and the switch dynamics as one figure")
    a = ap.parse_args()
    out = rollout(a.arm, a.seed)
    if out is None:
        raise SystemExit(f"no checkpoint for {a.arm} seed {a.seed}")
    z, cid, tss = out
    fig_switch_dynamics(a.seed, z, cid, tss, os.path.join(M.fig_dir(a.arm, a.seed),
                                                         "switch_dynamics_z.png"))
    if a.combined:
        fig_combined(a.seed, a.arm, os.path.join(M.fig_dir(a.arm, a.seed),
                                                 "heldout_and_switch_z.png"))
