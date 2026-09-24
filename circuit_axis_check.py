"""
circuit_axis_check.py -- does a frozen circuit carry ROTATION AS AN AXIS
across the gaps between anchors, and how fast does it identify the angle?

The supervised-level gate for the zero-shot experiment (sac.py, "zero_shot";
zero_shot_analysis.py), run BEFORE an RL sweep is spent on a circuit: drive
the checkpoint through sac's own rollout (sac.rollout_z, uniform random
actions, no policy) in a set of ANCHOR rotations and in the off-table test
angles, fit a ridge decoder from z to the angle on the anchor steps only,
and read what it says at the held-out angles.

THE ANCHORS ARE THE SAME FOR EVERY CIRCUIT (--anchors, default the wide
set +-30/+-90: the RL agent's training contexts), whatever the circuit was
fitted on. That is what makes the structure-learning contrast fair: the
rotation-trained circuit (rotations_wide) and the unstructured control
(linear_random, random linear transforms at the same effect sizes) are both
decoded from the same four rotations, and the difference between them is
what pretraining on the family bought. A circuit that has learned the
family as an axis decodes a novel angle near its true value between the
anchors; a lookup decodes it as whichever anchor is nearest; a circuit that
never saw rotations has to build the axis from the four anchors' z alone.

IDENTIFICATION SPEED is the second panel: the decoder's absolute error as a
function of the step within an episode, averaged over the novel angles. z at
step 0 holds nothing (EC's hidden state is reset, the first prediction error
is zero) and the angle becomes readable over the next few steps; a circuit
that represents one parameter should get there in fewer steps than one that
has to pin down four. The decoder is fitted on steps >= MIN_STEP of the
anchors, as in the first panel.

    python circuit_axis_check.py                                      # wide circuit, main+control, seeds 0 1
    python circuit_axis_check.py --train-contexts rotations_wide linear_random   # the contrast
    python circuit_axis_check.py --train-contexts all --anchors rotations        # the default circuit at its own anchors
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model as M
import sac
import task as T
from plotting_utils import (use_paper_style, figsize, plot_band, save, trim_spines,
                            label_panels, MUTED, PALETTE)

MIN_STEP = 3
DEFAULT_ANCHORS = sac.WIDE_PHASE_SET
ARM_COLOUR = {"main": PALETTE["salmon"], "control": PALETTE["rose"]}


def decode_tables(arm, ckpt_seed, train_contexts, anchors_set=DEFAULT_ANCHORS,
                  test_angles=sac.ZS_TEST_ANGLES, n_episodes=8, min_step=MIN_STEP, seed=0):
    """One (arm, checkpoint seed, training set) -> (by_angle, by_step), or
    None when the checkpoint is absent.

    by_angle: angle, decoded (mean over steps >= min_step), sd (over the
              episodes' means), anchor (bool)
    by_step:  t, abs_err_novel, abs_err_anchor (mean |decoded - true| at
              that step over the novel angles / the anchors)"""
    from sklearn.linear_model import RidgeCV
    spec = M.CircuitSpec(train_contexts=train_contexts)
    if not os.path.exists(M.ckpt_path(arm, ckpt_seed, spec)):
        return None
    anchors = [c for c in sac.phase_ids_for(anchors_set) if T.FAMILY[c] == 0]
    tests = [T.rotation_id(a) for a in test_angles]
    contexts = list(dict.fromkeys(anchors + tests))
    r = sac.rollout_z(arm, ckpt_seed, contexts, n_episodes=n_episodes, seed=seed, spec=spec)
    ctx = np.asarray(contexts)[r["y"]]
    angle = np.array([T.C_PARAM[c] for c in ctx], dtype=float)
    keep = r["t"] >= min_step
    is_anchor = np.isin(ctx, anchors)
    dec = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(r["z"][keep & is_anchor],
                                                     angle[keep & is_anchor])
    pred = dec.predict(r["z"])
    meta = dict(arm=arm, ckpt_seed=ckpt_seed, train_contexts=train_contexts)

    rows = []
    for c in contexts:
        m = keep & (ctx == c)
        per_ep = pd.Series(pred[m]).groupby(r["episode"][m]).mean()
        rows.append(dict(**meta, angle=float(T.C_PARAM[c]), decoded=float(per_ep.mean()),
                         sd=float(per_ep.std(ddof=1)) if len(per_ep) > 1 else 0.0,
                         anchor=bool(c in anchors)))
    by_angle = pd.DataFrame(rows)

    err = np.abs(pred - angle)
    steps = []
    for t in range(int(r["t"].max()) + 1):
        m = r["t"] == t
        steps.append(dict(**meta, t=t,
                          abs_err_novel=float(err[m & ~is_anchor].mean()),
                          abs_err_anchor=float(err[m & is_anchor].mean())))
    return by_angle, pd.DataFrame(steps)


def make_figure(by_angle, by_step, anchors_set=DEFAULT_ANCHORS, out_path=None):
    """Top: decoded rotation against the true one, from a ridge decoder fitted
    on the anchors' z alone (dotted verticals: the anchors, the same set for
    every circuit), mean +- s.e.m. over checkpoint seeds; one column per
    circuit training set, one line per arm. The diagonal is a perfect axis;
    a staircase through the anchors is a lookup. Bottom: the decoder's
    absolute error at each step of an episode, averaged over the novel
    angles (solid) and the anchors (dotted): how many steps the circuit needs
    to identify the angle. Uniform random actions through sac.rollout_z."""
    use_paper_style()
    sets = list(dict.fromkeys(by_angle.train_contexts))
    n = len(sets)
    fig, axes = plt.subplots(2, n, figsize=figsize("double" if n > 1 else "single", height=4.4),
                             layout="constrained", squeeze=False)
    lim = max(abs(a) for a in sac.ZS_TEST_ANGLES)
    ticks = [-90, -45, 0, 45, 90]
    for j, ts in enumerate(sets):
        ax, ax2 = axes[0, j], axes[1, j]
        sub = by_angle[by_angle.train_contexts == ts]
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.6, ls="--", zorder=0)
        for a in sorted(sub[sub.anchor].angle.unique()):
            ax.axvline(a, color=MUTED, lw=0.6, ls=":", zorder=0)
        for arm, g in sub.groupby("arm"):
            piv = g.pivot_table(index="ckpt_seed", columns="angle", values="decoded")
            m = piv.mean(axis=0)
            s = (piv.std(axis=0, ddof=1) / np.sqrt(len(piv))) if len(piv) > 1 else m * 0
            plot_band(ax, m.index.values, m.values, s.values, color=ARM_COLOUR.get(arm),
                      label=arm, lw=1.2, ms=3)
        ax.set_title(f"circuit fitted on {ts}")
        ax.set_xlabel("true rotation (deg)")
        ax.set_ylabel("decoded rotation (deg)")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.legend(frameon=False, fontsize=6, loc="upper left")

        st = by_step[by_step.train_contexts == ts]
        for arm, g in st.groupby("arm"):
            for col, ls in (("abs_err_novel", "-"), ("abs_err_anchor", ":")):
                piv = g.pivot_table(index="ckpt_seed", columns="t", values=col)
                m = piv.mean(axis=0)
                s = (piv.std(axis=0, ddof=1) / np.sqrt(len(piv))) if len(piv) > 1 else m * 0
                plot_band(ax2, m.index.values, m.values, s.values, color=ARM_COLOUR.get(arm),
                          label=f"{arm}, {'novel' if col.endswith('novel') else 'anchors'}",
                          lw=1.2, ls=ls, marker=None)
        ax2.set_xlabel("step in episode")
        ax2.set_ylabel("|decoded - true| (deg)")
        ax2.set_title("identification speed")
        ax2.legend(frameon=False, fontsize=6, loc="upper right")
    trim_spines(fig)
    label_panels(fig, list(axes.ravel()))
    if out_path is not None:
        save(fig, out_path, caption=make_figure.__doc__)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", nargs="+", default=["main", "control"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--train-contexts", nargs="+", default=[M.WIDE_TRAIN_SET],
                    choices=list(M.TRAIN_CONTEXT_SETS))
    ap.add_argument("--anchors", default=DEFAULT_ANCHORS, choices=list(sac.PHASE_SETS),
                    help="the rotations the decoder is fitted on, for EVERY circuit")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    angles, steps = [], []
    for ts in args.train_contexts:
        for arm in args.arms:
            for s in args.seeds:
                r = decode_tables(arm, s, ts, anchors_set=args.anchors, n_episodes=args.episodes)
                if r is None:
                    print(f"no checkpoint: {arm} seed {s} ({ts})")
                    continue
                angles.append(r[0]); steps.append(r[1])
    if not angles:
        raise SystemExit("nothing to check: no checkpoints found")
    by_angle, by_step = pd.concat(angles, ignore_index=True), pd.concat(steps, ignore_index=True)
    show = (by_angle.groupby(["train_contexts", "arm", "angle"])["decoded"].mean()
            .unstack("angle").round(1))
    print(show.to_string(sparsify=False))
    print("\nmean |error| at novel angles by step (deg):")
    print(by_step.groupby(["train_contexts", "arm", "t"])["abs_err_novel"].mean()
          .unstack("t").round(1).to_string(sparsify=False))
    out = args.out or os.path.join(sac.FIG_DIR, "circuit_axis_check")
    make_figure(by_angle, by_step, args.anchors, out)
    print(f"figure -> {out}.pdf / .png")
    return by_angle, by_step


if __name__ == "__main__":
    main()
