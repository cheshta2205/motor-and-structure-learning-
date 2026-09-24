"""
zero_shot_analysis.py -- zero-shot generalization to novel rotations, across
configs, from the curves of `zero_shot`-schedule runs (sac.zero_shot_blocks).

THE QUESTION. Trained on four rotations (the WIDE set, +-30 and +-90), what
does each agent do in a rotation it has never met, before learning anything
there? Two contrasts share the figure. Against COIN, whose only route to a
novel context is a posterior over the contexts it knows: at an exactly
ambiguous angle that posterior commits to one anchor per trial, where an
axis code sits between them -- a difference in the DISTRIBUTION of the
per-trial code, not in its mean. And against the same circuit fitted on
UNSTRUCTURED transforms (random linear maps at the same effect sizes): what
pretraining on the family bought. See the ZERO-SHOT comment in sac.py.

WHAT A RUN HOLDS. After the cycle, ZS_PROBE_EP frozen reaches at each of
sac.ZS_TEST_ANGLES (curve["kind"] == "zs", MC frozen, deterministic action,
belief carried reach to reach within the block), then one few-shot block of
ordinary training at config.zs_learn_angle ("fs"). The trained anchors are
in the sweep as ordinary points, so the curve through them is one curve.

THE FIGURE, two rows of three. Every sweep point is the mean +- s.e.m. over seeds
of the per-seed block mean over the SETTLED reaches (the block's first reach
is dropped: its belief is the previous block's, so it measures the switch,
not the context). The trained angles are marked on the axis.
  a  the DECODED angle against the true one: what the context code MC
     receives "means". For an inferred oracle it is the posterior-mean
     angle over its rotation hypotheses (curve["belief_angle"]); for a
     circuit config it is a ridge decoder from the reach-mean z to the
     angle, fitted per seed on that seed's TRAINING reaches (the anchors)
     and applied to the sweep. A staircase through the anchors is a lookup;
     the diagonal is an axis.
  b  the FIRST ACTION's angle from the goal direction against the rotation:
     the feedforward compensation the agent carries INTO a reach, before any
     feedback -- a blind policy aims straight everywhere, a mixture code
     aims in steps, an axis code aims in proportion. Full compensation is
     aim = -rotation (dashed).
  c  the per-reach DISTRIBUTION of compensation at the midpoint angles
     (+-60): whether the code commits to one anchor per trial (two modes,
     COIN: its posterior at an exactly ambiguous angle lands on one
     hypothesis or the other, reach by reach) or sits between them (one
     mode, the circuit). The mean can agree while the distributions do not,
     and it is the per-trial code that a policy learns from and that a
     human experiment measures.
  d  the MODE of the per-reach aim at each test angle against the angle's
     signed offset from the nearest trained one, as the signed error from
     full compensation. A code that commits to the nearest anchor per trial
     has its mode on that anchor, so its error equals the offset (the
     identity, on both sides); an axis code's mode follows the angle and its
     error stays near zero.
     Baseline is left out: with no context signal its error tracks the
     angle itself.
  e  the SHARE of reaches aimed within 15 degrees of full compensation,
     against the same offset: how much of the distribution is on target. A
     code on the nearest anchor is on target only at the anchors and
     collapses toward the midpoints; an axis code stays flat.
  f  the FEW-SHOT learning curve at each learn angle present: the block's
     return per reach (smoothed), mean +- s.e.m. over seeds. Zero-shot
     RETURN is not drawn: outcomes
     are matched to within a few percent (summary(), --metric), because a
     first-step misaim moves the hand five percent of the reach and
     feedback absorbs it.

RUN NAMES. A zero-shot run is <base>[_train<set>]_rotw_zs<angle>[_h<N>][_carry]
[_clearbuf]: the circuit's training set when not the default (trainrotw,
trainlin), the wide context set, the few-shot angle, the test horizon. Runs
that differ only in the few-shot angle have IDENTICAL frozen sweeps and are
pooled for the sweep panels; runs differ in horizon are different tasks and
get separate figures (zero_shot.pdf, zero_shot_h12.pdf).

    python zero_shot_analysis.py                       # every *_zs* run, one figure per horizon
    python zero_shot_analysis.py --horizon 12
    python zero_shot_analysis.py --configs baseline_rotw_zs60_h12 coin_rotw_zs60_h12
    python zero_shot_analysis.py --metric steps        # the return panel's measure
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sac
import task as T
import make_ablations_plot as MAP
from plotting_utils import (use_paper_style, figsize, plot_band, save, trim_spines,
                            label_panels, MUTED, INK)

METRICS = {
    "return": ("return", "return per reach"),
    "first_angle": ("first_angle", "first-action angle (deg)"),
    "steps": ("steps", "steps to the goal"),
    "path_ratio": ("path_ratio", "path ratio (arrived reaches)"),
    "dist_norm": ("dist_norm", "mean distance / start distance"),
    "success": ("success", "success rate"),
}
ZS_NAME_RE = re.compile(r"_zs(-?[0-9]+(?:\.[0-9]+)?)")
SMOOTH_W = 15            # reaches, for panel D


def find_runs(run_dir=None, configs=None):
    """{run name: [seed dirs]} for the zero-shot runs on disk (or the named
    ones), skipping seeds without a curve."""
    root = run_dir or sac.RUN_DIR
    names = configs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*_zs*")))
    found = {}
    for name in names:
        if not ZS_NAME_RE.search(name):
            raise ValueError(f"{name!r} is not a zero_shot run (no _zs<angle> in the name)")
        seeds = sorted(glob.glob(os.path.join(root, name, "seed_*", "curve.npz")))
        if seeds:
            found[name] = seeds
    return found


def phase_set_of(name):
    """The sac.PHASE_SETS entry a run name carries (its _rot / _rotw token);
    the trained anchors are read off it."""
    tokens = set(name.split("_"))
    for ps, suffix in sorted(sac.PHASE_SET_SUFFIX.items(), key=lambda kv: -len(kv[1])):
        if suffix and suffix.lstrip("_") in tokens:
            return ps
    return sac.DEFAULT_PHASE_SET


TRAIN_TAG_RE = re.compile(r"_train([a-z]+)")
HORIZON_RE = re.compile(r"_h([0-9]+)(?:_|$)")


def horizon_of(name):
    """The test blocks' horizon a run name carries (_h12), else MAX_STEPS."""
    m = HORIZON_RE.search(name)
    return int(m.group(1)) if m else int(sac.MAX_STEPS)


def variant_of(name):
    """base, plus ' [train<set>]' when the run's circuit was fitted on a
    non-default context set (the _train<tag> infix)."""
    base = MAP.base_config(name)
    m = TRAIN_TAG_RE.search(name)
    return base if m is None else f"{base} [train{m.group(1)}]"


def anchors_of(df):
    """The trained rotation angles of the runs in `df`, which must all share
    one context set: a figure mixes anchor sets only by mistake."""
    sets = sorted(df.phase_set.unique())
    if len(sets) != 1:
        raise ValueError(f"runs from several context sets {sets}; pass --configs for one set")
    return sorted(T.C_PARAM[c] for c in sac.phase_ids_for(sets[0]) if T.FAMILY[c] == 0)


def load_curves(run_dirs):
    """One frame per run name, all seeds, the z_mean column kept as arrays."""
    frames = []
    for name, paths in run_dirs.items():
        for p in paths:
            seed = int(os.path.basename(os.path.dirname(p)).split("_")[1])
            curve = np.load(p, allow_pickle=True)["curve"].item()
            if "kind" not in curve:
                raise ValueError(f"{p} predates the zero_shot columns; rerun it")
            arrays = {k for k in ("z_mean", "z0") if k in curve}
            df = pd.DataFrame({k: v for k, v in curve.items() if k not in arrays})
            for k in arrays:
                df[k] = list(curve[k])
            df["run"] = name
            df["seed"] = seed
            # `base` is the preset the run derives from, `variant` the base
            # plus the circuit it was driven by when that is not the default
            # (main [trainrotw] against main [trainlin]): the structure-
            # learning contrast is between two circuits under ONE preset, so
            # everything downstream groups by variant
            df["base"] = MAP.base_config(name)
            df["variant"] = variant_of(name)
            df["learn_angle"] = float(ZS_NAME_RE.search(name).group(1))
            df["phase_set"] = phase_set_of(name)
            df["horizon"] = horizon_of(name)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("no zero_shot runs found")
    return pd.concat(frames, ignore_index=True)


def _settled(df):
    """Drop each block's first reach: it starts on the previous block's belief."""
    return df[df.ep > 1]


def sweep_table(df, metric="return"):
    """Per (base, seed, angle) block means over the settled frozen reaches,
    pooled over the runs of one base (identical sweeps). Returns a frame with
    columns base, seed, param, value."""
    col = METRICS[metric][0]
    zs = _settled(df[df.kind == sac.ZS_KIND])
    # a seed of the same variant appearing under several learn angles is one
    # sweep repeated; average those first so no seed counts twice
    g = zs.groupby(["variant", "seed", "param"])[col].mean().reset_index()
    return g.rename(columns={col: "value"})


def decoded_angles(df):
    """Per (base, seed, angle): the angle the context code reads as.

    Inferred oracle -> belief_angle. Circuit config (z_mean present) -> ridge
    from z_mean to the angle, fitted on that seed's training reaches and
    applied to its frozen reaches. Others -> nothing."""
    from sklearn.linear_model import RidgeCV
    rows = []
    for (base, run, seed), g in df.groupby(["variant", "run", "seed"]):
        zs = _settled(g[g.kind == sac.ZS_KIND])
        if zs.belief_angle.notna().any():
            for ang, gg in zs.groupby("param"):
                rows.append(dict(variant=base, seed=seed, param=ang,
                                 decoded=float(gg.belief_angle.mean()), source="posterior"))
            continue
        train = g[(g.kind == sac.TRAIN_KIND) & g.z_mean.map(lambda v: isinstance(v, np.ndarray))]
        if train.empty:
            continue
        X = np.stack(train.z_mean.values)
        y = train.param.values.astype(float)
        dec = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y)
        for ang, gg in zs.groupby("param"):
            zz = gg[gg.z_mean.map(lambda v: isinstance(v, np.ndarray))]
            if zz.empty:
                continue
            rows.append(dict(variant=base, seed=seed, param=ang,
                             decoded=float(dec.predict(np.stack(zz.z_mean.values)).mean()),
                             source="z decoder"))
    return pd.DataFrame(rows, columns=["variant", "seed", "param", "decoded", "source"])


def few_shot_curves(df, metric="return"):
    """{(base, learn_angle): (reach index, mean, sem)} of the few-shot block."""
    col = METRICS[metric][0]
    out = {}
    fs = df[df.kind == sac.FS_KIND]
    for (base, ang), g in fs.groupby(["variant", "learn_angle"]):
        mat = g.pivot_table(index="seed", columns="ep", values=col)
        m = mat.mean(axis=0)
        s = (mat.std(axis=0, ddof=1) / np.sqrt(mat.notna().sum(axis=0))
             if len(mat) > 1 else m * 0)
        out[(base, float(ang))] = (m.index.values, m.values, s.values)
    return out


def trained_level(df, metric="return"):
    """{base: (mean, sem)} of the anchors' last-occurrence blocks under the
    cycle -- the fully-trained reference for panel D."""
    col = METRICS[metric][0]
    tr = df[df.kind == sac.TRAIN_KIND]
    last = tr[tr.occurrence == tr.occurrence.max()]
    per_seed = last.groupby(["variant", "run", "seed"])[col].mean().reset_index()
    per_seed = per_seed.groupby(["variant", "seed"])[col].mean()
    out = {}
    for base, g in per_seed.groupby(level=0):
        v = g.values
        out[base] = (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    return out


def midpoints_of(anchors):
    """The angles halfway between consecutive same-sign anchors (+-60 for the
    wide set): where a discrete code is most ambiguous."""
    pos = sorted(a for a in anchors if a > 0)
    mids = [(a + b) / 2 for a, b in zip(pos, pos[1:])]
    return sorted({m for m in mids} | {-m for m in mids})


def midpoint_compensation(df, anchors):
    """{variant: per-reach COMPENSATION at the midpoint angles} from the
    settled frozen reaches: -first_angle * sign(angle), so full compensation
    reads as +|midpoint| whichever side the rotation is on. Per reach, not
    per seed: the point is the DISTRIBUTION -- a code that picks one anchor
    per trial gives two modes at the anchors' compensation, an axis code one
    mode at the midpoint's."""
    mids = midpoints_of(anchors)
    zs = _settled(df[(df.kind == sac.ZS_KIND) & df.param.isin(mids)])
    out = {}
    for v, g in zs.groupby("variant"):
        out[v] = (-g.first_angle * np.sign(g.param)).to_numpy(dtype=float)
    return out


def _smooth(y, w=SMOOTH_W):
    y = np.asarray(y, float)
    if len(y) < 3 or w <= 1:
        return y
    w = min(w, len(y))
    k = np.ones(w) / w
    pad = np.pad(y, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _base_of(variant):
    return variant.split(" [")[0]


# How the circuits driving one preset are told apart, by the _train<tag>
# in the run name: (line style, fraction toward black). The wide-rotation
# circuit is the experiment's own and draws solid in the preset's hue; the
# unstructured control is dashed and darkened; the default circuit (fitted
# on the table) is dotted and lightened, so that where it appears beside
# the others it reads as the reference, not the result.
_CIRCUIT_STYLE = {"trainrotw": ("-", 0.0), "trainlin": ("--", 0.45), "": (":", -0.45)}


def _tag_of(variant):
    return variant.split(" [")[1].rstrip("]") if " [" in variant else ""


def _colour(variant, siblings=()):
    """The preset's colour, shaded for the circuit; the default circuit is
    lightened only when another circuit of the same preset is drawn too."""
    base = MAP.CONFIG_COLORS.get(_base_of(variant), MUTED)
    tag = _tag_of(variant)
    if tag == "" and not any(_base_of(s) == _base_of(variant) and s != variant for s in siblings):
        return base
    _, frac = _CIRCUIT_STYLE.get(tag, ("--", 0.45))
    return MAP._shade(base, frac) if frac > 0 else MAP._lighten(base, -frac) if frac < 0 else base


def _style(variant, siblings=()):
    tag = _tag_of(variant)
    if tag == "" and not any(_base_of(s) == _base_of(variant) and s != variant for s in siblings):
        return "-"
    return _CIRCUIT_STYLE.get(tag, ("--", 0.45))[0]


def _order(variants):
    return sorted(variants, key=lambda v: (MAP.CONFIG_ORDERS.get(_base_of(v), 99), v))


def _mean_sem(tbl, value="value"):
    """(angles, mean, sem) over seeds from a (seed, param, value) frame."""
    piv = tbl.pivot_table(index="seed", columns="param", values=value)
    m = piv.mean(axis=0)
    s = (piv.std(axis=0, ddof=1) / np.sqrt(piv.notna().sum(axis=0))
         if len(piv) > 1 else m * 0)
    return m.index.values.astype(float), m.values, s.values


def _mark_anchors(ax, anchors, ls=":", label=None):
    """The trained angles on an x axis: a faint vertical guide through the
    panel and a small filled triangle sitting on the axis line at each one
    (x in data units, y pinned to the axes), so they read as 'trained here'
    at a glance and the guides stay quiet."""
    import matplotlib.transforms as mtransforms
    for a in anchors:
        ax.axvline(a, color=MUTED, lw=0.6, ls=ls, zorder=0)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(list(anchors), [0.0] * len(anchors), marker="^", ls="none", ms=4.5,
            color=INK, mec="none", transform=trans, clip_on=False, zorder=6,
            label=label)


def make_figure(df, metric="return", out_path=None):
    """Zero-shot generalization to novel rotations. Every sweep point is the
    mean +- s.e.m. over seeds of each seed's block mean over the settled
    frozen reaches (the block's first reach dropped: its belief is the
    previous block's); the trained angles are marked by a triangle on the
    axis and a faint vertical guide. a: the angle the context code reads as
    -- the inferred oracle's posterior-mean angle, or a ridge decoder from
    the reach-mean z fitted on that seed's training reaches -- against the
    true angle; the diagonal is perfect. b: the FIRST action's angle from
    the goal direction against the rotation, the feedforward compensation
    carried into a reach before any feedback; the dashed diagonal is full
    compensation. c: the per-reach distribution of compensation at the
    midpoint angles (sign-folded so full compensation reads +|midpoint|,
    dotted; dashed: the anchors' compensation), a kernel density over a
    faint histogram of the same reaches. A code that picks one anchor per
    trial gives two modes at the anchors' levels; an axis code gives one
    mode at the midpoint. d: the mode of the per-reach aim at each test angle
    against the angle's signed offset from the nearest trained one, as the
    signed error from full compensation; the line is the mean over the angles
    at each offset with its 95% bootstrap band over reaches, the dashed
    identity a mode that sits on the nearest anchor. e: the share of reaches
    aimed within 15 degrees of full compensation against the same offset,
    mean and 95% interval across seeds. f (and on): the few-shot block's return per reach
    (smoothed over 15 reaches) at each learn angle, mean +- s.e.m. over
    seeds."""
    use_paper_style()
    anchors = anchors_of(df)
    bases = _order(df.variant.unique())
    curves = few_shot_curves(df, metric)
    learn_angles = sorted({a for _, a in curves})
    n_panels = 5 + max(1, len(learn_angles))
    n_cols = (n_panels + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=figsize("double", height=4.4),
                             layout="constrained", squeeze=False)
    flat = list(axes.ravel())
    axA, axB, axC, axM, axS = flat[:5]
    fs_axes = flat[5:5 + len(learn_angles)]
    lim = max(abs(a) for a in sac.ZS_TEST_ANGLES)
    ticks = [-90, -45, 0, 45, 90]

    # a: the decoded angle
    dec = decoded_angles(df)
    axA.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.6, ls="--", zorder=0)
    for base in bases:
        t = dec[dec.variant == base]
        if t.empty:
            continue
        x, m, s = _mean_sem(t, "decoded")
        src = t.source.iloc[0]
        plot_band(axA, x, m, s, color=_colour(base, bases), label=f"{base} ({src})", lw=1.2, ms=3,
                  ls=_style(base, bases))
    _mark_anchors(axA, anchors)
    axA.set_xlabel("True rotation (deg)")
    axA.set_ylabel("Decoded rotation (deg)")
    axA.set_xticks(ticks)
    axA.set_yticks(ticks)

    # b: the first action's aim -- feedforward compensation
    tbl = sweep_table(df, "first_angle")
    for base in bases:
        t = tbl[tbl.variant == base]
        if t.empty:
            continue
        x, m, s = _mean_sem(t)
        plot_band(axB, x, m, s, color=_colour(base, bases), label=base, lw=1.2, ms=3,
                  ls=_style(base, bases))
    _mark_anchors(axB, anchors)
    axB.plot([-lim, lim], [lim, -lim], color=MUTED, lw=0.6, ls="--", zorder=0)
    axB.set_xlabel("Rotation (deg)")
    axB.set_ylabel("First-action Angle (deg)")
    axB.set_xticks(ticks)
    axB.set_yticks(ticks)
    # axB.set_title("Zero-shot Aim")

    # c: the per-reach distribution of compensation at the midpoints -- one
    # mode or two (draw_midpoint_density)
    draw_midpoint_density(axC, df, anchors, bases, with_counts=False, legend=False)

    # d: where the aim's mode sits, against the distance from an anchor;
    # e: how much of the aim is on target, against the same offset. Either
    # is dropped from the figure when a cut-down run has too few reaches
    # per angle to estimate a mode (an empty axes would break trim_spines)
    empty = []
    if not draw_mode_error(axM, df, anchors, bases, legend=False):
        empty.append(axM)
    if not draw_near_share(axS, df, anchors, bases, legend=False):
        empty.append(axS)

    # f...: few-shot learning, one panel per learn angle
    for ax, ang in zip(fs_axes, learn_angles):
        for base in bases:
            if (base, ang) not in curves:
                continue
            x, m, s = curves[(base, ang)]
            plot_band(ax, x, _smooth(m), _smooth(s), color=_colour(base, bases), label=base,
                      lw=1.2, marker=None, ls=_style(base, bases))
        ax.set_xlabel("Reach")
        ax.set_ylabel(METRICS[metric][1])
        # ax.set_title(f"Few-shot at {ang:+g} deg")

    # an unused panel is REMOVED, not hidden: a hidden axes with no data has
    # infinite limits and trim_spines would choke on it
    used = [a for a in [axA, axB, axC, axM, axS] if a not in empty] + fs_axes
    for ax in flat:
        if ax not in used:
            fig.delaxes(ax)

    # ONE legend for the figure: every panel shares the encoding (a colour
    # and line style per config, the triangle for a trained angle), so it is
    # stated once, below the panels, rather than six times
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=_colour(b, bases), ls=_style(b, bases), lw=1.2, label=b)
               for b in bases]
    handles.append(Line2D([], [], marker="^", ls="none", ms=4.5, color=INK, mec="none",
                          label="trained angle"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=6)

    trim_spines(fig)
    label_panels(fig, used)
    if out_path is not None:
        save(fig, out_path, caption=make_figure.__doc__)
    return fig


MIDPOINT_RANGE = (-120.0, 150.0)      # compensation axis of the density panel
MIDPOINT_TICK = 30.0                  # evenly spaced ticks on it
MIDPOINT_BIN = 10.0                   # the faint histogram's bin width
MIDPOINT_BW = 0.15                    # the kernel density's bandwidth factor


def draw_midpoint_density(ax, df, anchors, bases, with_counts=True, legend=True):
    """Panel c: the distribution of per-reach compensation at the midpoint
    angles, one density per config -- a kernel density estimate over a faint
    histogram of the same reaches. Dashed verticals: the anchors'
    compensation; dotted: full compensation at the midpoint."""
    from scipy.stats import gaussian_kde
    mids = midpoints_of(anchors)
    comp = midpoint_compensation(df, anchors)
    lo, hi = MIDPOINT_RANGE
    edges = np.arange(lo, hi + MIDPOINT_BIN, MIDPOINT_BIN)
    grid = np.linspace(lo, hi, 400)
    for b in bases:
        x = comp.get(b, np.array([]))
        x = x[np.isfinite(x)]
        if len(x) < 5:
            continue
        col, ls = _colour(b, bases), _style(b, bases)
        ax.hist(x, bins=edges, density=True, color=col, alpha=0.12, lw=0)
        kde = gaussian_kde(x, bw_method=MIDPOINT_BW)
        ax.plot(grid, kde(grid), color=col, ls=ls, lw=1.2,
                label=f"{b} (n={len(x)})" if with_counts else b)
    # every trained angle's compensation, marked like the trained angles
    # elsewhere -- the negative ones are where a reach lands that compensated
    # for the OPPOSITE anchor (the blind policy's second lump); the midpoint's
    # full compensation dotted
    _mark_anchors(ax, sorted(anchors), ls="--", label="trained" if with_counts else None)
    for m in sorted({abs(m) for m in mids}):
        ax.axvline(m, color=MUTED, lw=0.8, ls=":", zorder=0)
    # a margin past the data on both sides, as the other panels get from
    # matplotlib's default margins: without it the histogram's first bar and
    # the density sit flush against the y spine and the end ticks crowd the
    # corners
    ax.set_xlim(lo - MIDPOINT_TICK / 2, hi + MIDPOINT_TICK / 2)
    first = np.ceil(lo / MIDPOINT_TICK) * MIDPOINT_TICK
    ax.set_xticks(np.arange(first, hi + 1e-9, MIDPOINT_TICK))
    ax.set_xlabel("Compensation (deg)")
    ax.set_ylabel("Density")
    # ax.set_title(f"Aim at {'/'.join(f'{abs(m):g}' for m in sorted({abs(m) for m in mids}))} deg, per reach")
    if legend:
        ax.legend(frameon=False, fontsize=6, loc="upper left")


def make_midpoint_figure(df, out_path=None):
    """Distribution of per-reach compensation at the midpoint angles (the
    frozen reaches, sign-folded so full compensation reads +|midpoint|),
    one density per config: a kernel density estimate over a faint
    histogram of the same reaches, pooled over seeds and both midpoint
    signs. Dashed verticals: the anchors' compensation; dotted: full
    compensation at the midpoint. A code that commits to one anchor per
    trial is bimodal at the anchors; an axis code is unimodal at the
    midpoint. A blind policy's fixed aim compensates one sign and not the
    other, so it appears at plus and minus its aim."""
    use_paper_style()
    bases = _order(df.variant.unique())
    fig, ax = plt.subplots(figsize=figsize("single", height=2.4), layout="constrained")
    draw_midpoint_density(ax, df, anchors_of(df), bases)
    trim_spines(fig)
    if out_path is not None:
        save(fig, out_path, caption=make_midpoint_figure.__doc__)
    return fig


MODE_GRID = np.linspace(-180.0, 180.0, 1441)     # the mode search grid, quarter-degree
N_BOOT = 100                                     # bootstrap resamples for the per-angle intervals
BOOT_GRID = np.linspace(-180.0, 180.0, 361)       # a coarser mode grid for the resamples
BOOT_SEED = 0


def _kde_mode(x, bw, grid=MODE_GRID):
    from scipy.stats import gaussian_kde
    return float(grid[np.argmax(gaussian_kde(x, bw_method=bw)(grid))])


def _boot_replicates(x, stat, n_boot=N_BOOT, seed=BOOT_SEED):
    """`stat` over n_boot resamples (with replacement) of the reaches x. The
    same seed for every angle, so replicate i is a matched draw across angles
    and per-offset means of replicates are valid replicates themselves."""
    rng = np.random.default_rng(seed)
    return np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])


def mode_error_table(df, anchors, bw=MIDPOINT_BW, min_n=10, boot=True):
    """Per (variant, angle): the MODE of the per-reach first-action aim over
    the settled frozen reaches (the peak of a kernel density), its error from
    full compensation (-angle), and the angle's distance to the nearest
    trained anchor. The mode, not the mean: at an ambiguous angle a code that
    commits to one anchor per trial has its mode ON that anchor and its mean
    between the two, so the mode's error grows with the distance from the
    anchor while the mean's does not."""
    from scipy.stats import gaussian_kde
    zs = _settled(df[df.kind == sac.ZS_KIND])
    rows = []
    for v, g in zs.groupby("variant"):
        for ang, gg in g.groupby("param"):
            x = gg.first_angle.to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < min_n:
                continue
            mode = _kde_mode(x, bw)
            full = -float(ang)
            reps = (_boot_replicates(x, lambda r: _kde_mode(r, bw, BOOT_GRID)) - full if boot
                    else None)
            # the nearest anchor: by angle, with a tie (the midpoints, 0)
            # broken toward the anchor whose compensation the mode is nearer
            dmin = min(abs(ang - a) for a in anchors)
            near = [a for a in anchors if abs(abs(ang - a) - dmin) < 1e-9]
            a_near = min(near, key=lambda a: abs(mode - (-a)))
            # both axes SIGNED: the angle's offset from that anchor, and the
            # mode's error from full compensation. A mode ON the anchor has
            # error (-a) - (-ang) = ang - a = the offset, so a categorical
            # code lies on the identity whichever side of the anchor it is
            rows.append(dict(variant=v, angle=float(ang), dist=float(dmin), anchor=float(a_near),
                             offset=float(ang - a_near), mode=mode, mode_err=mode - full,
                             err_reps=reps, n=len(x)))
    return pd.DataFrame(rows, columns=["variant", "angle", "dist", "anchor", "offset", "mode",
                                       "mode_err", "err_reps", "n"])


def draw_mode_error(ax, df, anchors, bases, exclude=("baseline",), legend=True):
    """The mode of the first-action aim against the distance from the nearest
    trained angle (mode_error_table), both axes signed: the line is the
    mean over the angles at each offset with its bootstrap band, the dashed
    identity is a mode that sits on the nearest anchor."""
    tbl = mode_error_table(df, anchors)
    shown = [b for b in bases if _base_of(b) not in exclude]
    if tbl.empty:                       # too few reaches per angle (a cut-down run)
        return False
    dmax = max(tbl.dist.max(), 1.0)
    ax.plot([-dmax, dmax], [-dmax, dmax], color=MUTED, lw=0.6, ls="--", zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.6, zorder=0)
    ax.axvline(0.0, color=MUTED, lw=0.6, zorder=0)
    for i, b in enumerate(shown):
        t = tbl[tbl.variant == b]
        if t.empty:
            continue
        col, ls = _colour(b, bases), _style(b, bases)
        # the line: the mean over the angles at each offset. The band: the
        # 95% interval of that same mean over matched bootstrap replicates
        # (resampling reaches within each angle), where replicates exist
        offs = np.array(sorted(t.offset.unique()))
        m = np.array([t[t.offset == o].mode_err.mean() for o in offs])
        lo = hi = None
        if t.err_reps.map(lambda r: r is not None).all():
            band = np.array([np.mean(np.stack(t[t.offset == o].err_reps.values), axis=0)
                             for o in offs])                      # (offsets, n_boot)
            lo, hi = np.percentile(band, 2.5, axis=1), np.percentile(band, 97.5, axis=1)
        plot_band(ax, offs, m, lo=lo, hi=hi, color=col, ls=ls, lw=1.2, marker="o", ms=3.0,
                  label=b)
    ax.set_xlabel(r"$\Delta$ Angle (deg)")
    ax.set_ylabel(r"$\Delta$ Mode (deg)")
    ax.set_xticks(sorted(tbl.offset.unique()))
    if legend:
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    return True


def make_mode_error_figure(df, out_path=None, exclude=("baseline",)):
    """The mode of the first-action aim against the distance from the nearest
    trained angle. Per test angle, the mode of the per-reach aim distribution
    (frozen reaches, kernel density peak) is compared with full compensation,
    and the signed error plotted against the angle's signed offset from its
    nearest anchor; the line is the mean over the angles at each offset
    with its 95% bootstrap band over reaches. A code that commits to the nearest anchor per
    trial has its mode on that anchor, so its error EQUALS the offset (the
    dashed identity) on either side; an axis code's mode follows the angle
    and its error stays near zero. Baseline is not drawn: it has no
    context signal, so its error tracks the angle itself."""
    use_paper_style()
    bases = _order(df.variant.unique())
    fig, ax = plt.subplots(figsize=figsize("single", height=2.4), layout="constrained")
    draw_mode_error(ax, df, anchors_of(df), bases, exclude)
    trim_spines(fig)
    if out_path is not None:
        save(fig, out_path, caption=make_mode_error_figure.__doc__)
    return fig


NEAR_TOL = 15.0          # degrees: a reach "near full compensation" is within this


def near_share_table(df, anchors, tol=NEAR_TOL):
    """Per (variant, angle): the share of settled frozen reaches whose first-
    action aim is within `tol` degrees of full compensation (-angle), with
    the angle's signed offset from its nearest anchor (ties as in
    mode_error_table: by the mode). Panel d says where the mode sits; this
    says how much of the distribution sits on the target."""
    zs = _settled(df[df.kind == sac.ZS_KIND])
    modes = mode_error_table(df, anchors, boot=False).set_index(["variant", "angle"])
    rows = []
    for v, g in zs.groupby("variant"):
        for ang, gg in g.groupby("param"):
            x = gg.first_angle.to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 10 or (v, float(ang)) not in modes.index:
                continue
            near = lambda r, a=float(ang): float((np.abs(r - (-a)) <= tol).mean())
            by_seed = {int(sd): near(gs.first_angle.to_numpy(dtype=float))
                       for sd, gs in gg.groupby("seed")}
            rows.append(dict(variant=v, angle=float(ang),
                             offset=float(modes.loc[(v, float(ang)), "offset"]),
                             share=near(x), share_by_seed=by_seed, n=len(x)))
    return pd.DataFrame(rows, columns=["variant", "angle", "offset", "share", "share_by_seed", "n"])


def draw_near_share(ax, df, anchors, bases, exclude=("baseline",), tol=NEAR_TOL, legend=True):
    """The share of reaches aimed within `tol` degrees of full compensation
    against the angle's signed offset from its nearest anchor: the line is
    the mean over seeds and angles at each offset, the band its 95%
    interval across seeds."""
    tbl = near_share_table(df, anchors, tol)
    shown = [b for b in bases if _base_of(b) not in exclude]
    if tbl.empty:                       # too few reaches per angle (a cut-down run)
        return False
    ax.axvline(0.0, color=MUTED, lw=0.6, zorder=0)
    for i, b in enumerate(shown):
        t = tbl[tbl.variant == b]
        if t.empty:
            continue
        col, ls = _colour(b, bases), _style(b, bases)
        # per seed, the share pooled over the angles at an offset; the line
        # is the mean over seeds and the band its 95% interval (1.96 s.e.m.)
        offs = np.array(sorted(t.offset.unique()))
        per_seed = []
        for o in offs:
            d = {}
            for bs in t[t.offset == o].share_by_seed:
                for sd, v_ in bs.items():
                    d.setdefault(sd, []).append(v_)
            per_seed.append(np.array([np.mean(v_) for v_ in d.values()]))
        m = np.array([v_.mean() for v_ in per_seed])
        sem = np.array([v_.std(ddof=1) / np.sqrt(len(v_)) if len(v_) > 1 else 0.0 for v_ in per_seed])
        plot_band(ax, offs, m, s=1.96 * sem, color=col, ls=ls, lw=1.2, marker="o", ms=3.0,
                  label=b)
    ax.set_xlabel("$\\Delta$ Angle (deg)")
    ax.set_ylabel(f"Fraction near compensation\n(" + r"$\pm$" + f"{tol:g})")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(tbl.offset.unique()))
    if legend:
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    return True


def make_near_share_figure(df, out_path=None, exclude=("baseline",), tol=NEAR_TOL):
    """The share of per-reach first-action aims within 15 degrees of full
    compensation (settled frozen reaches), per test angle, against the
    angle's signed offset from its nearest trained anchor; the line is the
    mean at each offset, the band its 95% interval across seeds. A code that commits to
    the nearest anchor per trial puts its reaches ON the anchor, so the
    share is high at the anchors and collapses toward the midpoints; an axis
    code keeps a single mode on the target and its share stays flat.
    Baseline is not drawn: it has no context signal."""
    use_paper_style()
    bases = _order(df.variant.unique())
    fig, ax = plt.subplots(figsize=figsize("single", height=2.4), layout="constrained")
    draw_near_share(ax, df, anchors_of(df), bases, exclude, tol)
    trim_spines(fig)
    if out_path is not None:
        save(fig, out_path, caption=make_near_share_figure.__doc__)
    return fig


def summary(df, metric="return"):
    """A per-config table: zero-shot value at the anchors, at the interior
    angles and beyond the anchors, mean over seeds; and the seed count."""
    anchors = set(anchors_of(df))
    outer = max(anchors)
    tbl = sweep_table(df, metric)
    groups = {"anchors": lambda a: a in anchors,
              "interior": lambda a: (a not in anchors) and abs(a) < outer,
              "beyond": lambda a: abs(a) > outer}
    rows = []
    for base, t in tbl.groupby("variant"):
        row = {"config": base, "seeds": t.seed.nunique()}
        for name, f in groups.items():
            sel = t[t.param.map(f)]
            row[name] = float(sel.value.mean()) if not sel.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", default=None, help="run names (default: every *_zs* run)")
    ap.add_argument("--metric", default="return", choices=list(METRICS))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None,
                    help="figure path without extension (default figures/sac/zero_shot[_<metric>])")
    ap.add_argument("--contexts", default=None, choices=list(sac.PHASE_SETS),
                    help="only the runs of this context set (default: every set found, "
                         "one figure each)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="only the runs whose test blocks ran under this horizon (_h<N>; "
                         f"{sac.MAX_STEPS} for runs without the tag)")
    args = ap.parse_args(argv)
    runs = find_runs(args.run_dir, args.configs)
    # ONE FIGURE PER (CONTEXT SET, HORIZON): different anchors cannot share
    # axes, and a shorter test horizon is a different task, so the runs found
    # are split by both -- zero_shot.pdf, zero_shot_h12.pdf, zero_shot_rot.pdf
    by_key = {}
    for name, paths in runs.items():
        by_key.setdefault((phase_set_of(name), horizon_of(name)), {})[name] = paths
    if args.contexts is not None:
        by_key = {k: v for k, v in by_key.items() if k[0] == args.contexts}
    if args.horizon is not None:
        by_key = {k: v for k, v in by_key.items() if k[1] == args.horizon}
    if not by_key:
        raise SystemExit("no zero_shot runs found for that selection")
    if args.out is not None and len(by_key) > 1:
        raise SystemExit(f"--out names one figure but {len(by_key)} (context set, horizon) "
                         "groups were found; add --contexts / --horizon")
    for (ps, h), grp_runs in sorted(by_key.items()):
        print(f"\n== context set {ps}, test horizon {h} ==")
        for name, paths in grp_runs.items():
            print(f"{name}: {len(paths)} seed(s)")
        df = load_curves(grp_runs)
        print(summary(df, args.metric).to_string(float_format=lambda v: f"{v:.3f}"))
        # the wide set at the run's horizon is THE experiment and takes the
        # plain name; another context set or horizon carries its suffix
        stem = "zero_shot" + ("" if ps == sac.WIDE_PHASE_SET
                              else (sac.PHASE_SET_SUFFIX.get(ps) or f"_{ps}"))
        stem += "" if h == sac.MAX_STEPS else f"_h{h}"
        out = args.out or os.path.join(sac.FIG_DIR, stem + ("" if args.metric == "return"
                                                            else f"_{args.metric}"))
        make_figure(df, args.metric, out)
        print(f"figure -> {out}.pdf / .png")
        # the companion: panel c's midpoint distribution as densities
        mid_out = os.path.join(os.path.dirname(out), stem + "_midpoint")
        make_midpoint_figure(df, mid_out)
        print(f"figure -> {mid_out}.pdf / .png")
        # ... and the mode's error against the distance from an anchor
        me_out = os.path.join(os.path.dirname(out), stem + "_modeerr")
        make_mode_error_figure(df, me_out)
        print(f"figure -> {me_out}.pdf / .png")
        ns_out = os.path.join(os.path.dirname(out), stem + "_nearshare")
        make_near_share_figure(df, ns_out)
        print(f"figure -> {ns_out}.pdf / .png")


if __name__ == "__main__":
    main()
