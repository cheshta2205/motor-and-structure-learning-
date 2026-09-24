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

and EVERY A is orthogonal (rotations; the identity for curl and drift), so
A A^T = I.  The drift's velocity increment w and the plant's damping shift
the means and touch nothing else.  The twelve conditionals are therefore
isotropic Gaussians with IDENTICAL covariance, differing only in their
means.  So:

  * each conditional is unimodal by construction -- it is one Gaussian;
  * the marginal is an equal-weight, equal-covariance Gaussian mixture, whose
    modes can be counted exactly by mean-shift (for this family the mean-shift
    fixed points ARE the modes), with no density estimation and no smoothing
    parameter to argue about.

WHAT THE COUNT SHOWS.  The noise sd scales with |a|; the rotation separations
do too, but the curl (k|v|) and drift (w) separations do not, so a SMALLER
action and a FASTER agent resolve more of the design.  At the figure's
operating point, |a| = 0.3 with the velocity at 45 degrees to the action
(DEFAULT_ACT / DEFAULT_DIRECTION, chosen by sweeping |a|, |v| and the angle
on 2026-09-21):

      |vel|   min sep/sd   modes
      0.000        0.00        5      curl invisible at rest; drift and rotation separate
      0.030        1.29        8
      0.060        1.80        9
      0.080        1.85       10
      0.100        0.68        9      the speed cap starts to re-merge modes

At a typical policy action (|a| ~ 0.8) the same state gives 2 modes: the
command noise then swamps the curl and drift offsets.  So how informative a
transition is depends on the state AND the action it is taken from -- an
agent has to be moving, and not pushing hard, for the field families to be
identifiable from one step.  That is worth knowing before reading any
decoding curve: early steps of an episode are near-uninformative about two
of the three families.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import task as T
# plotting_utils depends on matplotlib alone, so importing it keeps this
# module's "task.py only" dependence on the rest of the repo
from plotting_utils import (use_paper_style, figsize, offset_spines, label_panels,
                            save, PALETTE, COLORS, INK)

FIG_DIR = os.path.join("figures", "multimodality")
use_paper_style()

# The state/action the figures are drawn at. Chosen by sweeping |a|, |v| and
# the angle between them for the marginal's mode count (2026-09-21): the
# command noise scales with |a| while the curl and drift separations do not,
# so a SMALLER action resolves more contexts. |a| = 0.3 (rms |a| under the GP
# policy is ~0.8), |v| = 0.08 (rms ~0.07) at 45 degrees to the action gives
# 10 modes with the closest pair 1.85 sd apart; the old (0.8, 0.3) at |v| =
# 0.08 gave 2.
DEFAULT_ACT = (0.3, 0.0)
DEFAULT_SPEED = 0.08
DEFAULT_DIRECTION = (1.0, 1.0)

# one hue per family, as in figures.FCOL (not imported: see above)
FAMILY_HUE = {"rotation": PALETTE["navy"], "reflection": PALETTE["rose"],
              "curl": PALETTE["teal"], "drift": PALETTE["amber"]}


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
        # the WHOLE law, as env_step_ab applies it: F = A a + B v - DAMPING v,
        # then the drift increment w (zero for every family but drift). Using
        # context_matrices here dropped w, so the four drift contexts collapsed
        # onto one mean and the figure showed a single dot for the family.
        A, B, w = T.context_affine(c)
        F = A @ act + B @ vel - T.DAMPING * vel
        vn = np.asarray(T.clip_speed(
            (vel + (F / T.MASS) * T.DT + w).astype(np.float32)), dtype=np.float64)
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


def speed_sweep(act, speeds, direction=DEFAULT_DIRECTION):
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


def _tint(color, f):
    """`color` mixed toward white: f = 1 is the colour itself, 0 is white."""
    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(1 - f * (1 - rgb))


def context_colors():
    """One colour per context: HUE is the family (the paper palette's family
    hue), LIGHTNESS the position within the family's angle sweep, darkest last.
    """
    per = T.N_CONTEXTS // len(T.FAM_NAMES)
    cols = {}
    for c in range(T.N_CONTEXTS):
        fam = T.FAM_NAMES[T.FAMILY[c]]
        hue = FAMILY_HUE.get(fam, COLORS[T.FAMILY[c] % len(COLORS)])
        cols[c] = _tint(hue, 0.45 + 0.55 * (c % per) / max(per - 1, 1))
    return cols


def _marginal_strip(ax, axis, mus, sd, grid1d, cols, contexts, horizontal=True):
    """Per-context 1-d marginals plus their mixture, on a strip axis.

    Each curve is the EXACT projected Gaussian (a projection of an isotropic
    Gaussian is Gaussian with the same sd), each integrating to 1 so the
    conditionals are drawn at their true height. The black curve is the
    mixture, their equal-weight average, which also integrates to 1 -- no
    estimation anywhere, and the black curve is genuinely the average of the
    coloured parts rather than a separate fit.
    """
    total = np.zeros_like(grid1d)
    for c in contexts:
        y = (np.exp(-(grid1d - mus[c, axis]) ** 2 / (2 * sd ** 2))
             / (sd * np.sqrt(2 * np.pi)))
        total += y / len(contexts)
        if horizontal:
            ax.plot(grid1d, y, color=cols[c], lw=0.8)
        else:
            ax.plot(y, grid1d, color=cols[c], lw=0.8)
    # hide only the DENSITY axis, and only the tick LABELS on the shared one --
    # set_xticks([]) on a sharex strip would strip them from the joint panel too
    if horizontal:
        ax.plot(grid1d, total, color=INK, lw=1.2)
        ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
    else:
        ax.plot(total, grid1d, color=INK, lw=1.2)
        ax.set_xticks([])
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def plot(act=DEFAULT_ACT, speed=DEFAULT_SPEED, direction=DEFAULT_DIRECTION,
         out_dir=FIG_DIR, lims=(0.03, 0.09)):
    """The marginal p(v' | s, a) as a joint plot with per-context marginal
    strips -- one panel, single-column width (user, 2026-09-21: only this
    panel is needed; the conditional is one Gaussian by construction and the
    speed dependence is the printed table).

    Marginal strips sit on TOP (the v'_x marginal) and on the RIGHT (v'_y),
    each sharing its axis with the joint panel so a peak lines up with the
    cluster that produced it.  Per-context curves are drawn in the context's
    colour, each integrating to 1, and the black curve is the mixture, their
    average.  Everything is in the VELOCITY plane of s': position differs only
    by the constant `pos` and a factor DT, so it carries the same information
    on a less readable scale.

    `lims` fixes both axes (user, 2026-09-22: 0.03-0.09, the same square window
    in v'_x and v'_y so the geometry is not distorted); None autoscales to the
    means plus 6 sd. Each family's means are joined in table order by a thin
    line in the family hue, so the family reads as one sweep: the rotations
    trace the arc of the rotated action, the curls a line along R(90) v, and
    the drifts a closed diamond of the four directions.

    Returns (path, modes, speed-sweep rows).
    """
    os.makedirs(out_dir, exist_ok=True)
    act = np.asarray(act, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    mus, sd = conditionals(np.zeros(2), d * speed, act)
    v = mus[:, 2:]
    k, _ = count_modes(v, sd)
    cols = context_colors()

    if lims is None:
        pad = 6 * sd
        lo, hi = v.min(0) - pad, v.max(0) + pad
    else:
        lo, hi = np.array([lims[0], lims[0]]), np.array([lims[1], lims[1]])
    gx = np.linspace(lo[0], hi[0], 400)
    gy = np.linspace(lo[1], hi[1], 400)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], 1)

    # Single-column print width: the joint panel with its two strips. The
    # height is chosen so the joint panel comes out close to square (its two
    # axes span the same window), without set_aspect, which would shrink the
    # panel inside its cell and pull it out of line with the shared strips.
    fig = plt.figure(figsize=(2.5, 3), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[7, 1.1], height_ratios=[1.1, 7],
                          wspace=0.02, hspace=0.02)
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)

    ctxs = range(T.N_CONTEXTS)
    Z = mixture_density(grid, v, sd).reshape(GX.shape)
    axm.contourf(GX, GY, Z, levels=40, cmap="Greys")
    # the family sweeps: one thin line through each family's means, in table
    # order (angle / gain / direction), under the markers
    for fam_i, fam in enumerate(T.FAM_NAMES):
        members = [c for c in ctxs if T.FAMILY[c] == fam_i]
        if len(members) > 1:
            if fam == "drift":                 # a full 360 sweep: close the loop
                members = members + members[:1]
            hue = FAMILY_HUE.get(fam, COLORS[fam_i % len(COLORS)])
            axm.plot(v[members, 0], v[members, 1], "-", color=hue, lw=0.8, alpha=0.7,
                     zorder=2)
    for c in ctxs:
        axm.plot(v[c, 0], v[c, 1], "o", ms=4, color=cols[c], mec=INK, mew=0.4, zorder=3)
    axm.set_xlim(lo[0], hi[0]); axm.set_ylim(lo[1], hi[1])
    if lims is not None:
        # fixed window: ticks from its lower edge in round steps, so the spine
        # runs the full window (0.03, 0.05, ..., 0.09) and the trim has no
        # reason to stretch the axis to enclose the density's background square
        step = 0.01 if (lims[1] - lims[0]) <= 0.08 else 0.02
        ticks = np.round(np.arange(lims[0], lims[1] + 1e-9, step), 10)
        axm.set_xticks(ticks); axm.set_yticks(ticks)
    axm.set_xlabel("$v'_x$"); axm.set_ylabel("$v'_y$")
    _marginal_strip(axt, 2, mus, sd, gx, cols, ctxs, horizontal=True)
    _marginal_strip(axr, 3, mus, sd, gy, cols, ctxs, horizontal=False)
    axt.set_title(f"$p(v' \\mid s, a)$: {k} mode{'s' if k != 1 else ''}")

    handles = [plt.Line2D([], [], color=cols[c], lw=2, label=T.C_LABEL[c])
               for c in ctxs]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    offset_spines(axm)

    rows = speed_sweep(act, [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10],
                       direction=direction)
    caption = (
        f"Next-state multimodality, exact from the task's noise model. The "
        f"marginal p(v' | s, a) over all {T.N_CONTEXTS} contexts at |vel| = {speed:.2f} "
        f"along ({d[0]:.2f}, {d[1]:.2f}), action ({act[0]:.1f}, {act[1]:.1f}): an "
        f"equal-weight mixture of isotropic Gaussians with identical covariance "
        f"(sd {sd:.4f}), {k} modes by mean-shift. Dots are the per-context means "
        f"(hue = family, lightness = angle), joined within a family in table order; "
        f"strips are the exact 1-d marginals per "
        f"context, each integrating to 1, with the mixture (their average) in "
        f"black. Every conditional p(v' | s, a, c) is a single Gaussian. Modes vs "
        f"speed: " + "; ".join(f"|v| {r[0]:.2f}: {r[3]}" for r in rows) + ".")
    path = os.path.join(out_dir, "next_state_multimodality.png")
    save(fig, path, caption=caption)
    return path, k, rows


def position_invariance(act=DEFAULT_ACT, speed=DEFAULT_SPEED, direction=DEFAULT_DIRECTION,
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


def plot_position(act=DEFAULT_ACT, direction=DEFAULT_DIRECTION,
                  positions=((0., 0.), (-0.5, 0.35), (0.6, -0.6)),
                  speeds=(0.0, 0.03, 0.08), speed=DEFAULT_SPEED, out_dir=FIG_DIR):
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

    # Double-column print width. Each panel is centred on its own mean, so the
    # axes cannot be shared; what is shared is the WINDOW SIZE, which is what
    # makes "same picture" mean "same structure".
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", height=4.8),
                             layout="constrained")
    rows = [[(np.asarray(p, float), d * speed, f"pos = ({p[0]:+.2f}, {p[1]:+.2f})")
             for p in positions],
            [(np.zeros(2), d * sp, f"|vel| = {sp:.2f}") for sp in speeds]]
    row_titles = [f"start position varied, |vel| = {speed:.2f}",
                  "start velocity varied, pos = (0, 0)"]

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
                ax.plot(P[c, 0], P[c, 1], "o", ms=4, color=cols[c], mec=INK, mew=0.4)
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            # modes RESOLVED out of the N means: a pair closer than ~2 sd merges
            # into one mode (and at rest the curl means coincide outright), so
            # the count is the honest summary of what the panel separates
            ax.set_title(f"{lab}: {k}/{T.N_CONTEXTS} resolved")
            ax.set_xlabel("$\\mathrm{pos}'_x$")
            if cx == 0:
                ax.set_ylabel("$\\mathrm{pos}'_y$")
        # the row's title sits above the middle panel's own title
        axes[r][1].annotate(row_titles[r], (0.5, 1.0), xycoords="axes fraction",
                            xytext=(0, 16), textcoords="offset points", ha="center",
                            va="bottom", fontsize=plt.rcParams["figure.titlesize"])

    offset_spines(fig)
    label_panels(fig)
    caption = (
        f"The same next-state density in position space. Top row: start position "
        f"varied at |vel| = {speed:.2f}; bottom row: start velocity varied at "
        f"pos = (0, 0). Every panel uses the same window size centred on its own "
        f"mean, so identical pictures mean identical structure: pos' = pos + vel' DT, "
        f"so the position mixture is the velocity mixture translated, independent "
        f"of the start position (top row, three copies) and not of the start speed "
        f"(bottom row). Dots are per-context means (hue = family, lightness = "
        f"angle); each title gives the modes resolved out of the {T.N_CONTEXTS} "
        f"means, since a pair closer than about 2 sd merges into one mode and at "
        f"rest the curl means coincide. Action ({act[0]:.1f}, {act[1]:.1f}). The "
        f"wall (pos clipped to ±{T.POS_LIM}) is the one exception, not shown.")
    path = os.path.join(out_dir, "next_position_multimodality.png")
    save(fig, path, caption=caption)
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
