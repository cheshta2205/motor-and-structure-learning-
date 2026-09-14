"""
multimodality.py -- THE PREMISE OF THE WHOLE TASK, checked rather than assumed.

Context inference only buys anything if the next state is genuinely ambiguous
without the context.  That is a claim about two distributions:

    p(s' | s, a)        marginalizing over contexts   -- should be MULTIMODAL
    p(s' | s, a, c)     conditioning on one           -- should be UNIMODAL

If the marginal were unimodal, a context-blind forward model could fit it with
a single prediction and the cerebellar error signal would carry nothing a
simpler model could not get.  If the conditional were multimodal, knowing the
context would not be enough and the task would be asking for something the
architecture cannot deliver.

BOTH ARE EXACT HERE, not empirical -- which is a consequence of the design
rather than luck.  Noise enters the motor command and the context then
transforms it:

    F    = A(a + sigma|a| eta) + Bv  =  (Aa + Bv)  +  sigma|a| A eta
    vel' = v + F/MASS * DT
    Cov  = (sigma |a| DT / MASS)^2 * A A^T

and EVERY level of factor A is orthogonal (rotations and reflections both),
so A A^T = I.  The sixteen conditionals are therefore isotropic Gaussians with
IDENTICAL covariance, differing only in their means.  So:

  * each conditional is unimodal by construction -- it is one Gaussian;
  * the marginal is an equal-weight, equal-covariance Gaussian mixture, whose
    modes can be counted exactly by mean-shift (for this family the mean-shift
    fixed points ARE the modes), with no density estimation and no smoothing
    parameter to argue about.

WHAT THE COUNT SHOWS.  Separation grows with SPEED, because the field factor acts on
the velocity and contributes nothing at rest:

      |vel|   min sep/sd   modes
      0.000        0.00        4      field invisible; only aim's 4 levels separate
      0.020        1.07        6
      0.040        1.00       14
      0.080        2.51       16      all cells resolved

So how informative a transition is depends on the state it is taken from, and
an agent has to be MOVING for the field factor to be identifiable at all.
That is worth knowing before reading any decoding curve: early steps of an
episode are near-uninformative about half the design.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import task as T

FIG_DIR = os.path.join("figures", "multimodality")


def _despine(a):
    # copied rather than imported from figures.py: this module depends only on
    # task.py by design, so it stays runnable while the rest of the repo is
    # being migrated to the factorial contexts
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def conditionals(pos, vel, act, contexts=None):
    """Means of p(s' | s, a, c) for each context, and the SHARED noise sd.

    Returns (mus, sd) with mus[i] = [pos', vel'] (4-vector) for contexts[i].
    One sd for all of them, because the covariance is context-independent --
    see the module docstring.  Computed through task.clip_speed so the speed
    cap is respected; near the cap the conditional is a truncated Gaussian
    rather than a Gaussian, still unimodal but no longer exactly isotropic.
    """
    contexts = range(T.N_CONTEXTS) if contexts is None else contexts
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    mus = []
    for c in contexts:
        A, B = T.context_matrices(c)
        vn = np.asarray(T.clip_speed(
            (vel + ((A @ act + B @ vel) / T.MASS) * T.DT).astype(np.float32)),
            dtype=np.float64)
        mus.append(np.concatenate([pos + vn * T.DT, vn]))
    sd = T.CMD_NOISE * float(np.linalg.norm(act)) * T.DT / T.MASS
    return np.stack(mus), sd


def count_modes(mus, sd, n_start=4000, seed=0, max_iter=300):
    """Number of modes of the equal-covariance isotropic mixture over `mus`.

    Mean-shift from many starts.  For a Gaussian mixture with a single shared
    isotropic covariance the mean-shift update is exactly the gradient-ascent
    fixed-point map of the density, so converged points are modes -- this is a
    count, not an estimate.  Two limits kept honest: starts are spread over the
    means' bounding box padded by 4 sd, and points are merged at sd/2, well
    below the smallest separation that can produce distinct modes (2 sd).
    """
    rng = np.random.default_rng(seed)
    lo, hi = mus.min(0) - 4 * sd, mus.max(0) + 4 * sd
    X = rng.uniform(lo, hi, size=(n_start, mus.shape[1]))
    for _ in range(max_iter):
        q = -((X[:, None, :] - mus[None]) ** 2).sum(-1) / (2 * sd ** 2)
        w = np.exp(q - q.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        Xn = w @ mus
        if np.abs(Xn - X).max() < 1e-12:
            X = Xn
            break
        X = Xn
    modes = []
    for x in X:
        if not any(np.linalg.norm(x - m) < sd * 0.5 for m in modes):
            modes.append(x)
    return len(modes), np.array(modes)


def mixture_density(grid, mus, sd, weights=None):
    """Exact mixture density on `grid` (n, d).  No KDE anywhere in this file."""
    w = np.full(len(mus), 1.0 / len(mus)) if weights is None else np.asarray(weights)
    q = -((grid[:, None, :] - mus[None]) ** 2).sum(-1) / (2 * sd ** 2)
    return (np.exp(q) * w).sum(1) / (2 * np.pi * sd ** 2) ** (mus.shape[1] / 2)


def speed_sweep(act, speeds, direction=(1.0, 0.4)):
    """Mode count and separation against speed -- the table in the docstring."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    rows = []
    for sp in speeds:
        mus, sd = conditionals(np.zeros(2), d * sp, act)
        v = mus[:, 2:]
        D = np.linalg.norm(v[:, None] - v[None], axis=-1)
        off = D[~np.eye(len(mus), dtype=bool)]
        k, _ = count_modes(v, sd)
        rows.append((sp, off.min() / sd, off.mean() / sd, k))
    return rows


def context_colors():
    """One colour per context: HUE is the family, LIGHTNESS the angle.

    The design is flat now -- rotation and reflection, each an angle sweep --
    so the two visual channels are family and position within the family
    rather than two crossed factors.
    """
    ramps = ["Blues", "Reds"]
    per = T.N_CONTEXTS // len(T.FAM_NAMES)
    cols = {}
    for c in range(T.N_CONTEXTS):
        fam = T.FAMILY[c]
        cm = plt.get_cmap(ramps[fam % len(ramps)])
        cols[c] = cm(0.35 + 0.55 * (c % per) / max(per - 1, 1))
    return cols


def _marginal_strip(ax, axis, mus, sd, grid1d, cols, contexts, horizontal=True):
    """Per-context 1-d marginals plus their mixture, on a strip axis.

    Each curve is the EXACT projected Gaussian (a projection of an isotropic
    Gaussian is Gaussian with the same sd), weighted 1/N so the curves sum to
    the mixture drawn over them -- no estimation, and the black total is
    genuinely the sum of the coloured parts rather than a separate fit.
    """
    total = np.zeros_like(grid1d)
    for c in contexts:
        y = (np.exp(-(grid1d - mus[c, axis]) ** 2 / (2 * sd ** 2))
             / (sd * np.sqrt(2 * np.pi)) / len(contexts))
        total += y
        if horizontal:
            ax.plot(grid1d, y, color=cols[c], lw=1.1)
        else:
            ax.plot(y, grid1d, color=cols[c], lw=1.1)
    # hide only the DENSITY axis, and only the tick LABELS on the shared one --
    # set_xticks([]) on a sharex strip would strip them from the joint panel too
    if horizontal:
        ax.plot(grid1d, total, color="k", lw=1.4)
        ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
    else:
        ax.plot(total, grid1d, color="k", lw=1.4)
        ax.set_xticks([])
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def plot(act=(0.8, 0.3), speed=0.08, direction=(1.0, 0.4), shown=None,
         out_dir=FIG_DIR):
    """The marginal and one conditional as joint plots with per-context
    marginal strips, plus the speed dependence.

    Marginal strips sit on TOP (the v'_x marginal) and on the RIGHT (v'_y),
    each sharing its axis with the joint panel so a peak lines up with the
    cluster that produced it.  Per-context curves are drawn in the context's
    colour and the black curve over them is their sum, so the total really is
    the sum of the coloured parts.  Everything is in the VELOCITY plane of s':
    position differs only by the constant `pos` and a factor DT, so it carries
    the same information on a less readable scale.
    """
    os.makedirs(out_dir, exist_ok=True)
    # `shown` picks which context's conditional gets its own panel.  DERIVED,
    # not a literal: it defaulted to 10, which pointed off the end once the
    # table shrank from 16 contexts to 8.  A literal id in a default argument
    # is only loud when it happens to be out of range.
    shown = (T.N_CONTEXTS - 1) // 2 if shown is None else shown
    act = np.asarray(act, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    mus, sd = conditionals(np.zeros(2), d * speed, act)
    v = mus[:, 2:]
    k, _ = count_modes(v, sd)
    cols = context_colors()

    pad = 6 * sd
    lo, hi = v.min(0) - pad, v.max(0) + pad
    gx = np.linspace(lo[0], hi[0], 400)
    gy = np.linspace(lo[1], hi[1], 400)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], 1)

    fig = plt.figure(figsize=(15.0, 5.0))
    # column 4 is an empty spacer: gridspec wspace is uniform, and without it
    # the sweep panel's y-label lands on top of the conditional's right strip
    gs = fig.add_gridspec(2, 6, width_ratios=[5, 1.1, 5, 1.1, 0.9, 5.4],
                          height_ratios=[1.1, 5], wspace=0.08, hspace=0.08)

    panels = [(0, range(T.N_CONTEXTS), f"p(v' | s, a)   marginal\n{k} modes"),
              (2, [shown], f"p(v' | s, a, c = {T.C_LABEL[shown]})\n1 mode")]
    for col, ctxs, title in panels:
        axm = fig.add_subplot(gs[1, col])
        axt = fig.add_subplot(gs[0, col], sharex=axm)
        axr = fig.add_subplot(gs[1, col + 1], sharey=axm)

        Z = mixture_density(grid, v[list(ctxs)], sd).reshape(GX.shape)
        axm.contourf(GX, GY, Z, levels=40, cmap="Greys")
        for c in ctxs:
            axm.plot(v[c, 0], v[c, 1], "o", ms=5.5, color=cols[c],
                     mec="k", mew=0.4)
        axm.set_xlim(lo[0], hi[0]); axm.set_ylim(lo[1], hi[1])
        axm.set_xlabel("v'_x")
        if col == 0:
            axm.set_ylabel("v'_y")
        _despine(axm)

        _marginal_strip(axt, 2, mus, sd, gx, cols, ctxs, horizontal=True)
        _marginal_strip(axr, 3, mus, sd, gy, cols, ctxs, horizontal=False)
        axt.set_title(title, fontsize=10)

    ax2 = fig.add_subplot(gs[1, 5])
    rows = speed_sweep(act, [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10])
    ax2.plot([r[0] for r in rows], [r[3] for r in rows], "o-", color="k")
    ax2.axhline(T.N_CONTEXTS, ls=":", c="0.6")
    ax2.axvline(T.V_MAX, ls=":", c="0.6")
    ax2.set_xlabel("|vel|"); ax2.set_ylabel("modes in the marginal")
    ax2.set_title("field is invisible at rest;\nthe speed cap re-merges modes",
                  fontsize=10)
    _despine(ax2)

    handles = [plt.Line2D([], [], color=cols[c], lw=2.2, label=T.C_LABEL[c])
               for c in range(T.N_CONTEXTS)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    path = os.path.join(out_dir, "next_state_multimodality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, k, rows


def position_invariance(act=(0.8, 0.3), speed=0.08, direction=(1.0, 0.4),
                        positions=((0., 0.), (-0.5, 0.35), (0.6, -0.6)),
                        speeds=(0.0, 0.03, 0.08)):
    """Is the next-POSITION structure a function of velocity alone?

    Exactly yes in the interior, and for an arithmetic reason rather than an
    empirical one:  pos' = pos + vel' * DT.  The position mixture is the
    velocity mixture translated by `pos`, so CENTERING the sixteen means on
    their own centroid must give a configuration that does not depend on pos
    at all.  This measures that residual, which should be 0 to float epsilon,
    and contrasts it against varying the velocity, which must change it.

    THE ONE EXCEPTION IS THE WALL.  env_step clips pos' to [-POS_LIM, POS_LIM],
    which truncates the mixture and can merge modes -- so position is a pure
    translation in the interior and nothing of the sort near a boundary.
    Reported separately rather than hidden.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    ref = None
    out = {"by_position": [], "by_speed": []}
    for p in positions:
        mus, sd = conditionals(np.asarray(p, float), d * speed, act)
        cfg = mus[:, :2] - mus[:, :2].mean(0)          # centred configuration
        if ref is None:
            ref = cfg
        k, _ = count_modes(mus[:, :2], sd)
        margin = T.POS_LIM - np.abs(mus[:, :2]).max()
        out["by_position"].append((p, float(np.abs(cfg - ref).max()), k, margin))
    ref = None
    for sp in speeds:
        mus, sd = conditionals(np.zeros(2), d * sp, act)
        cfg = mus[:, :2] - mus[:, :2].mean(0)
        if ref is None:
            ref = cfg
        k, _ = count_modes(mus[:, :2], sd)
        out["by_speed"].append((sp, float(np.abs(cfg - ref).max()), k))
    return out


def plot_position(act=(0.8, 0.3), direction=(1.0, 0.4),
                  positions=((0., 0.), (-0.5, 0.35), (0.6, -0.6)),
                  speeds=(0.0, 0.03, 0.08), speed=0.08, out_dir=FIG_DIR):
    """The same joint density in POSITION space.

    Top row varies the starting position at fixed velocity; bottom row varies
    the starting velocity at fixed position.  Every panel uses the SAME axis
    EXTENT, centred on its own mean, so "identical picture" means "identical
    structure" -- the top row should be three copies of one figure and the
    bottom row three different ones.  Each panel's start position is printed
    on it, since that is the only thing separating the top row's panels.
    """
    os.makedirs(out_dir, exist_ok=True)
    act = np.asarray(act, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    cols = context_colors()
    _, sd = conditionals(np.zeros(2), d * speed, act)
    half = 6 * sd + 0.045          # one window size for all six panels

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.6))
    rows = [[(np.asarray(p, float), d * speed, f"pos = ({p[0]:+.2f}, {p[1]:+.2f})")
             for p in positions],
            [(np.zeros(2), d * sp, f"|vel| = {sp:.2f}") for sp in speeds]]
    subtitles = ["varying START POSITION at |vel| = %.2f" % speed,
                 "varying START VELOCITY at pos = (0, 0)"]

    for r, row in enumerate(rows):
        for cx, (p0, v0, lab) in enumerate(row):
            ax = axes[r][cx]
            mus, sdl = conditionals(p0, v0, act)
            P = mus[:, :2]
            k, _ = count_modes(P, sdl)
            ctr = P.mean(0)
            gx = np.linspace(ctr[0] - half, ctr[0] + half, 260)
            gy = np.linspace(ctr[1] - half, ctr[1] + half, 260)
            GX, GY = np.meshgrid(gx, gy)
            Z = mixture_density(np.stack([GX.ravel(), GY.ravel()], 1),
                                P, sdl).reshape(GX.shape)
            ax.contourf(GX, GY, Z, levels=40, cmap="Greys")
            for c in range(T.N_CONTEXTS):
                ax.plot(P[c, 0], P[c, 1], "o", ms=5, color=cols[c],
                        mec="k", mew=0.4)
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            # when means coincide the markers stack and only the last-drawn
            # context is visible -- say so rather than let the panel read as
            # "only the purple cells are here"
            note = "" if k == T.N_CONTEXTS else f"  ({T.N_CONTEXTS} means stacked on {k})"
            ax.set_title(f"{lab}   -   {k} modes{note}", fontsize=9)
            ax.set_xlabel("pos'_x", fontsize=8)
            if cx == 0:
                ax.set_ylabel("pos'_y", fontsize=8)
            ax.tick_params(labelsize=7)
            _despine(ax)
        axes[r][1].text(0.5, 1.22, subtitles[r], transform=axes[r][1].transAxes,
                        ha="center", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, "next_position_multimodality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    path, k, rows = plot()
    print(f"{'|vel|':>8} {'min sep/sd':>11} {'mean sep/sd':>12} {'modes':>7}")
    for sp, lo, me, kk in rows:
        print(f"{sp:8.3f} {lo:11.2f} {me:12.2f} {kk:7d}")
    print(f"\nmarginal has {k} modes; every conditional has exactly 1 "
          f"(one isotropic Gaussian)")
    print(f"figure -> {path}")

    inv = position_invariance()
    print("\n\npos' = pos + vel'*DT, so the POSITION mixture should be the "
          "velocity\nmixture translated -- centred on its own centroid it must "
          "not depend on pos.\n")
    print(f"{'start position':>22} {'config deviation':>18} {'modes':>7} "
          f"{'wall margin':>12}")
    for p, dev, kk, margin in inv["by_position"]:
        print(f"{'(%+.2f, %+.2f)' % p:>22} {dev:18.2e} {kk:7d} {margin:12.3f}")
    print(f"\n{'start speed':>22} {'config deviation':>18} {'modes':>7}")
    for sp, dev, kk in inv["by_speed"]:
        print(f"{sp:22.2f} {dev:18.2e} {kk:7d}")
    print("\nconfig deviation = max |centred means - centred means of the first "
          "row|.\n~0 across positions and nonzero across speeds is the claim.")
    print(f"position figure -> {plot_position()}")
