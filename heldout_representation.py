"""
heldout_representation.py -- where do contexts the circuit never saw land in z?

The default circuit is fitted on the table's twelve contexts. This rolls it
out on those PLUS off-table contexts of every family (task.rotation_id,
curl_id, drift_id) and draws where the held-out contexts sit in z relative
to the trained ones. Plots only for now (user, 2026-10-15); the
quantification comes later, once the protocol has converged. One seed.

Held-out sets (user-approved 2026-10-15):

  rotation   interpolation +-15, +-45 deg       extrapolation +-90 deg
  curl       interpolation +-3, +-10            extrapolation +-20
  drift      interpolation 45/135/225/315 deg   extrapolation every direction,
             trained and diagonal, at half and double the table's drift speed
             (direction lives on a circle, so speed is the only way out of the
             range)

Figure heldout_pca_z, one row per family:

  left    PC1 vs PC2 of z WITHIN the family: the PCA fitted on the family's
          trained contexts only; each context as its centroid -- filled
          (trained), open circles (interpolation), open diamonds
          (extrapolation) -- and a 1-SD covariance ellipse of its steps
  middle  the same centroids in the PCA of all twelve trained contexts, the
          other families' trained centroids in grey: does a held-out context
          stay in its family's region? Rotations and curls on PC1-PC2, drifts
          on PC3-PC4, where their ring lies
  right   the centroid's coordinate on the family's PC1 against its
          parameter (rotation, curl); for drift, the ring angle in the global
          PC3-PC4 plane against the direction

    python heldout_representation.py --seed 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import model as M
import task as T
from plotting_utils import (use_paper_style, offset_spines, figsize, colorbar, label_panels,
                            save, INK, MUTED, PALETTE)
from representation_analysis import collect

use_paper_style()

HELD = {
    "rotation": {"interp": [-45.0, -15.0, 15.0, 45.0], "extrap": [-90.0, 90.0]},
    "curl": {"interp": [-10.0, -3.0, 3.0, 10.0], "extrap": [-20.0, 20.0]},
    # every direction at every speed (user, 2026-10-15): the trained and the
    # diagonal directions each at half and double the table speed; the
    # diagonals at table speed are the direction interpolation
    "drift": {"interp": [45.0, 135.0, 225.0, 315.0],
              "extrap": [(phi, sp) for sp in (0.5, 2.0)
                         for phi in list(T.DRIFT_ANGLES) + [45.0, 135.0, 225.0, 315.0]]},
}
FAM_IDX = {name: i for i, name in enumerate(T.FAM_NAMES)}
N_SEQ = 4400           # 44 contexts: three times the usual rollout keeps ~the samples per context


def heldout_ids():
    """{family: {"trained": [ids], "interp": [ids], "extrap": [ids]}}, registering
    the off-table contexts."""
    out = {}
    for fam, sets in HELD.items():
        trained = [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] == FAM_IDX[fam]]
        if fam == "rotation":
            interp = [T.rotation_id(a) for a in sets["interp"]]
            extrap = [T.rotation_id(a) for a in sets["extrap"]]
        elif fam == "curl":
            interp = [T.curl_id(k) for k in sets["interp"]]
            extrap = [T.curl_id(k) for k in sets["extrap"]]
        else:
            interp = [T.drift_id(a) for a in sets["interp"]]
            extrap = [T.drift_id(a, sp) for a, sp in sets["extrap"]]
        out[fam] = {"trained": trained, "interp": interp, "extrap": extrap}
    return out


def _pca(X, n=3):
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:n], (S ** 2) / (S ** 2).sum()


def _param(c):
    return float(T.C_PARAM[c])


def _speed(c):
    return float(T.DRIFT_SPEED.get(c, 1.0))


# rotation by LIGHTNESS, not hue (user, 2026-10-15): the drift map cycles
# through every hue, so a red-blue rotation map collided with drift 0 (red)
# and drift 225 (blue). Brown to near-white to slate: neither end is a drift
# hue, and the curl map (pink-green) stays distinct.
from matplotlib.colors import LinearSegmentedColormap
ROT_CMAP = LinearSegmentedColormap.from_list("rot_div", ["#6b4423", "#c9a27e", "#f2efe9",
                                                         "#8a9aa5", "#2f3e46"])


def _norm_for(fam, ids):
    """Colour scale for a family's parameter: diverging around zero for
    rotation and curl, the cyclic map over 0-360 for drift direction."""
    if fam == "drift":
        return "paper_cyc", 0.0, 360.0
    v = max(abs(_param(c)) for c in ids)
    return (ROT_CMAP if fam == "rotation" else "PiYG"), -v, v


# ONE ORIENTATION for every figure of z's all-context PCA (user, 2026-10-16):
# a PC's sign is arbitrary, so two PCAs of the same code can come out
# mirrored. Each component is flipped so a named context sits on its
# positive side: PC1 rot +60, PC2 curl +13, PC3 drift 0, PC4 drift 90.
PC_ANCHORS = ("rot +60", "curl +13", "drift 0", "drift 90")


def orient_pcs(V, mu, z, cid):
    """Flip the rows of V (n_pc x dim) so each anchor context's centroid has
    a positive coordinate on its PC (PC_ANCHORS)."""
    V = np.array(V, copy=True)
    lab = {l: c for c, l in T.C_LABEL.items() if T.is_table_context(c)}
    for k, name in enumerate(PC_ANCHORS[:len(V)]):
        c = lab[name]
        if (z[cid == c].mean(0) - mu) @ V[k] < 0:
            V[k] = -V[k]
    return V


def _colour_axis(ax, cmap, norm, label):
    """Replace `ax`'s x axis with its colour key (user, 2026-10-15): a thin
    gradient strip directly below the plot over the same data range, holding
    the ticks and the label, so the parameter is keyed once."""
    lo, hi = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9]
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)
    cax = ax.inset_axes([0.0, -0.075, 1.0, 0.045])
    xs = np.linspace(lo, hi, 512)
    cax.imshow(xs[None, :], aspect="auto", cmap=cmap, norm=norm, extent=[lo, hi, 0, 1])
    cax.set_xlim(lo, hi)
    cax.set_yticks([])
    cax.set_xticks(ticks)
    cax.set_xlabel(label)
    for sp in cax.spines.values():
        sp.set_visible(False)
    return cax


def fig_heldout(seed, d, ids, path, into=None, start="a"):
    """Draw the held-out figure. `into`: a (sub)figure to draw the main panels
    in instead of a new figure (switch_dynamics' combined figure); then
    nothing is saved and the caption is returned. `start`: the first letter."""
    z, cid = d["z"], d["cid"]
    trained_all = [c for c in range(T.N_CONTEXTS)]
    mask_tr = np.isin(cid, trained_all)
    g_mu, g_V, g_var = _pca(z[mask_tr], 4)
    g_V = orient_pcs(g_V, g_mu, z, cid)
    # which global PCs each family's row shows (user, 2026-10-15): rotations
    # load on PC1 and curls on PC2, but the drift ring lies in PC3-PC4 (seed 0:
    # drift centroids spread 0.31 on each, ~0.02 on PC1-PC2)
    G_PCS = {"rotation": (0, 1), "curl": (0, 1), "drift": (2, 3)}
    cent = {c: z[cid == c].mean(0) for c in np.unique(cid)}
    rng = np.random.default_rng(0)

    # two figures (user, 2026-10-15): the main one, a row per family with the
    # all-contexts PCA and the parameter panel; and the within-family PCAs
    # with their ellipses on their own, side by side
    # rotations and curls share PC1-PC2 (user, 2026-10-15): one panel for
    # both, beside their two parameter panels; drift (PC3-PC4) below
    fig = into if into is not None else plt.figure(figsize=figsize("double", height=5.0),
                                                       layout="constrained")
    gs = fig.add_gridspec(2, 6)
    axG = {"rotation": fig.add_subplot(gs[0, 0:2])}
    axG["curl"] = axG["rotation"]
    axG["drift"] = fig.add_subplot(gs[1, 0:2])
    axP = {"rotation": fig.add_subplot(gs[0, 2:4]), "curl": fig.add_subplot(gs[0, 4:6]),
           "drift": fig.add_subplot(gs[1, 2:4])}
    # drift recovery split into its two parameters (user, 2026-10-15)
    axSpeed = fig.add_subplot(gs[1, 4:6])
    colour_axes = []
    axes = np.array([axG["rotation"], axP["rotation"], axP["curl"], axG["drift"], axP["drift"],
                     axSpeed])
    figW, axW = plt.subplots(1, 3, figsize=figsize("double", height=2.6), layout="constrained")
    for r, fam in enumerate(["rotation", "curl", "drift"]):
        sets = ids[fam]
        fam_ids = sets["trained"] + sets["interp"] + sets["extrap"]
        cmap, vmin, vmax = _norm_for(fam, fam_ids)
        norm = plt.Normalize(vmin, vmax)
        col = lambda c: plt.get_cmap(cmap)(norm(_param(c) % 360.0 if fam == "drift" else _param(c)))
        f_mu, f_V, f_var = _pca(z[np.isin(cid, sets["trained"])], 2)
        proj = lambda X, mu, V: (np.atleast_2d(X) - mu) @ V.T

        # left: within-family PCA. Each context's cloud as its centroid and a
        # 1-SD covariance ellipse (user, 2026-10-15; replaced the faint step
        # clouds): trained filled lightly, interpolation solid, extrapolation
        # dashed
        ax = axW[r]
        style = {c: k for k in ("trained", "interp", "extrap") for c in sets[k]}
        for c in fam_ids:
            P = proj(z[cid == c], f_mu, f_V)
            w, v = np.linalg.eigh(np.cov(P.T))
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            k = style[c]
            ax.add_patch(Ellipse(P.mean(0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang,
                                 facecolor=col(c) if k == "trained" else "none",
                                 alpha=0.15 if k == "trained" else 1.0, lw=0, zorder=1))
            ax.add_patch(Ellipse(P.mean(0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang,
                                 facecolor="none", edgecolor=col(c), lw=0.8,
                                 ls="--" if k == "extrap" else "-", zorder=2))
        ax.autoscale_view()
        for kind, marker, fill in (("trained", "o", True), ("interp", "o", False),
                                   ("extrap", "D", False)):
            for c in sets[kind]:
                p = proj(cent[c], f_mu, f_V)[0]
                size = 22 * (_speed(c) if fam == "drift" else 1.0)
                ax.scatter(*p, s=size, marker=marker, zorder=3,
                           facecolor=col(c) if fill else "white", edgecolor=col(c) if fill else col(c),
                           linewidth=1.2)
        ax.set_xlabel(f"PC1 ({f_var[0]:.0%})"); ax.set_ylabel(f"PC2 ({f_var[1]:.0%})")
        ax.set_title(fam.capitalize())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_label = {"rotation": "angle (°)", "curl": "gain k", "drift": "direction (°)"}[fam]
        colorbar(sm, ax=ax, label=cbar_label, location="bottom")

        # middle: the global PCA of the twelve trained contexts, on the
        # components that carry this family
        ax = axG[fam]
        gi, gj = G_PCS[fam]
        gV = g_V[[gi, gj]]
        shown = [f for f in G_PCS if axG[f] is ax]
        for c in trained_all:
            if T.FAMILY[c] not in [FAM_IDX[f] for f in shown] and fam == shown[0]:
                p = proj(cent[c], g_mu, gV)[0]
                ax.scatter(*p, s=14, color=MUTED, alpha=0.6, lw=0, zorder=1)
        for kind, marker, fill in (("trained", "o", True), ("interp", "o", False),
                                   ("extrap", "D", False)):
            for c in sets[kind]:
                p = proj(cent[c], g_mu, gV)[0]
                size = 22 * (_speed(c) if fam == "drift" else 1.0)
                ax.scatter(*p, s=size, marker=marker, zorder=3,
                           facecolor=col(c) if fill else "white", edgecolor=col(c), linewidth=1.2)
        ax.set_xlabel(f"PC{gi + 1} ({g_var[gi]:.0%})")
        ax.set_ylabel(f"PC{gj + 1} ({g_var[gj]:.0%})")
        # the colour key sits under the family's parameter panel, where the
        # parameter is also the x axis; a shared panel cannot hold two bars
        # the colour key IS the parameter panel's x axis (user, 2026-10-15):
        # drawn after the layout, in _colour_axis
        colour_axes.append((axP[fam], cmap, norm, cbar_label))
        ax.set_title(" and ".join(f.capitalize() if i == 0 else f for i, f in enumerate(shown))
                     + f" (PC{gi + 1}–PC{gj + 1})")

        # right: position against the parameter. Rotation and curl: the
        # centroid's coordinate on the family's PC1. Drift (user, 2026-10-15):
        # POLAR -- angle and magnitude both -- in the global PC3-PC4 plane,
        # where the ring is cleanest. Radius is the distance from the trained
        # drifts' centre; the ring's zero and handedness are fitted on the
        # TRAINED directions only (sign that best matches, then the circular
        # mean offset), so held-out contexts get no say.
        if fam == "drift":
            gD = g_V[[2, 3]]
            centre = np.mean([proj(cent[c], g_mu, gD)[0] for c in sets["trained"]], axis=0)
            def rel(c):
                return proj(cent[c], g_mu, gD)[0] - centre
            def raw(c):
                q = rel(c)
                return np.degrees(np.arctan2(q[1], q[0]))
            best = None
            for sgn in (1.0, -1.0):
                diffs = np.radians([sgn * raw(c) - _param(c) for c in sets["trained"]])
                R = abs(np.mean(np.exp(1j * diffs)))
                off = np.degrees(np.angle(np.mean(np.exp(1j * diffs))))
                if best is None or R > best[0]:
                    best = (R, sgn, off)
            _, sgn, off = best
            # the ring angle against the drift direction (user, 2026-10-15:
            # direction only; the speed panel and the polar plot were dropped).
            # Speed shows as marker size.
            axA = axP["drift"]
            def ang(c):
                a_ = sgn * raw(c) - off
                x_ = _param(c) % 360.0
                return x_ + ((a_ - x_ + 180.0) % 360.0 - 180.0)
            for kind, marker, fill in (("trained", "o", True), ("interp", "o", False),
                                       ("extrap", "D", False)):
                for c in sets[kind]:
                    # the three speeds of a direction side by side, not stacked:
                    # half speed nudged left, double right (the true direction
                    # is the middle one)
                    dx = {0.5: -9.0, 1.0: 0.0, 2.0: 9.0}.get(_speed(c), 0.0)
                    axA.scatter(_param(c) % 360.0 + dx, ang(c), s=22 * _speed(c), marker=marker,
                                zorder=3, facecolor=col(c) if fill else "white",
                                edgecolor=col(c), linewidth=1.2)
            axA.plot([0, 360], [0, 360], color=MUTED, ls=":", lw=0.8, zorder=0)
            axA.set_xticks([0, 90, 180, 270, 360]); axA.set_yticks([0, 90, 180, 270, 360])
            axA.set_xlabel("\n\n"); axA.set_ylabel("PC3–PC4 (°)")
            axA.set_title("Drift direction")
            # magnitude: the centroid's distance from the trained drifts'
            # centre in PC3-PC4 against the drift speed (the proportional
            # reference line was dropped, user 2026-10-15)
            for kind, marker, fill in (("trained", "o", True), ("interp", "o", False),
                                       ("extrap", "D", False)):
                for c in sets[kind]:
                    axSpeed.scatter(_speed(c), np.linalg.norm(rel(c)), s=22, marker=marker,
                                    zorder=3, facecolor=col(c) if fill else "white",
                                    edgecolor=col(c), linewidth=1.2)
            sp = np.array([0.5, 1.0, 2.0])
            axSpeed.set_xscale("log", base=2)
            axSpeed.set_xticks(sp, ["½", "1", "2"])
            axSpeed.minorticks_off()
            axSpeed.set_xlim(0.42, 2.4)
            axSpeed.set_ylim(0, None)
            axSpeed.set_xlabel("drift speed (× table)")
            axSpeed.set_ylabel("PC3–PC4")
            axSpeed.set_title("Drift speed")
            ax = axA
            continue
        # the family's axis IN THE ALL-CONTEXT PCA (user, 2026-10-15):
        # rotation on PC1, curl on PC2 -- the same coordinates as panel a
        ax = axP[fam]
        k_pc = {"rotation": 0, "curl": 1}[fam]
        for kind, marker, fill in (("trained", "o", True), ("interp", "o", False),
                                   ("extrap", "D", False)):
            for c in sets[kind]:
                p = proj(cent[c], g_mu, g_V)[0]
                ax.scatter(_param(c), p[k_pc], s=22, marker=marker, zorder=3,
                           facecolor=col(c) if fill else "white", edgecolor=col(c), linewidth=1.2)
        ax.set_xlabel("\n\n")                 # room for the colour axis below
        ax.set_ylabel(f"PC{k_pc + 1} ({g_var[k_pc]:.0%})")
        ax.set_title(fam.capitalize())

    handles = [Line2D([], [], marker="o", ls="", markeredgecolor=INK, markerfacecolor=INK,
                      label="trained"),
               Line2D([], [], marker="o", ls="", markeredgecolor=INK, markerfacecolor="white",
                      markeredgewidth=1.2, label="held out, interpolation"),
               Line2D([], [], marker="D", ls="", markeredgecolor=INK, markerfacecolor="white",
                      markeredgewidth=1.2, label="held out, extrapolation"),
               Line2D([], [], marker="o", ls="", color=MUTED, markeredgewidth=0,
                      label="other families (trained)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    for a in axes.ravel():
        offset_spines(a)
    for spec_ in colour_axes:
        _colour_axis(*spec_)
    label_panels(fig, list(axes), start=start)
    heldout_txt = (
        f"The circuit was fitted on the twelve table contexts; here it is rolled out on those "
        f"and on off-table contexts of every family: rotations at +-15 and +-45 deg "
        f"(interpolation) and +-90 (extrapolation); curls with gain +-3 and +-10 "
        f"(interpolation) and +-20 (extrapolation); drifts towards 45/135/225/315 deg "
        f"(interpolation), and every one of those eight directions at half and double the "
        f"table speed (extrapolation; marker size is the speed). Markers are context centroids of the "
        f"settled steps: filled trained, open circles interpolation, open diamonds "
        f"extrapolation, coloured by the context's parameter.")
    caption = (
        f"Held-out contexts in z, main circuit seed {seed} (tss >= {T.TSS_MIN}). " + heldout_txt +
        f" (a) Rotations and curls together in PC1-PC2 of the PCA of all twelve trained "
        f"contexts, where they lie on orthogonal axes; the trained drifts in grey. (d) Drifts "
        f"in PC3-PC4 of the same PCA, where their ring lies; the other families' trained "
        f"centroids in grey. (f) Each drift centroid's distance from the trained drifts' "
        f"centre in PC3-PC4 against the drift speed. (b, c, e) Each centroid's coordinate on "
        f"its family's axis in the all-context PCA (rotation PC1, curl PC2, as in a) against "
        f"its parameter; for drift, "
        f"the centroid's angle on the ring in the global PC3-PC4 plane against the drift "
        f"direction (zero and handedness fitted on the trained directions only; dotted: "
        f"identity; the half- and double-speed markers are nudged left and right of their "
        f"direction so all three speeds show). The within-family PCAs are in heldout_within_family_z.")
    captionW = (
        f"Held-out contexts in z within each family, main circuit seed {seed} "
        f"(tss >= {T.TSS_MIN}): PC1 vs PC2 of the PCA fitted on the family's trained contexts "
        f"only, each context as its centroid and a 1-SD covariance ellipse of its settled steps "
        f"(lightly filled trained, solid interpolation, dashed extrapolation). " + heldout_txt)
    for a in axW:
        offset_spines(a)
    label_panels(figW, list(axW))
    if into is not None:
        plt.close(figW)
        return caption
    save(figW, path.replace("heldout_pca_z", "heldout_within_family_z"), caption=captionW)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")
    return caption


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="main")
    a = ap.parse_args()
    ids = heldout_ids()
    pool = list(range(T.N_CONTEXTS)) + [c for f in ids.values() for k in ("interp", "extrap")
                                        for c in f[k]]
    d = collect(a.arm, a.seed, n=N_SEQ, contexts=pool)
    if d is None:
        raise SystemExit(f"no checkpoint for {a.arm} seed {a.seed}")
    out = M.fig_dir(a.arm, a.seed)
    fig_heldout(a.seed, d, ids, os.path.join(out, "heldout_pca_z.png"))
