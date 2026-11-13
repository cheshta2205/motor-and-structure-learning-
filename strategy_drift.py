"""
strategy_drift.py -- does an implicit plan override an explicit strategy?
The Mazzoni & Krakauer (2006, J Neurosci) paradigm on the frozen agents.

THE HUMAN RESULT. Under a 45-degree visuomotor rotation, participants told to
aim at the neighbouring target (an explicit strategy that cancels the
rotation) hit the target at once -- and then DRIFTED past it, toward the aimed
side, over dozens of trials, while still aiming where told. An implicit
process kept learning from the SENSORY PREDICTION ERROR between the aimed
direction and the cursor, though task error was zero. Cerebellar patients
drift less (Taylor, Klemfuss & Ivry 2010).

WHY THIS MODEL SHOULD SHOW IT. EC infers the context from CB's prediction
error, not from task error. Under the strategy the hand lands on the target,
but CB -- conditioned on a z that has not read the rotation -- predicts the
hand moving where MC pushed it, 45 degrees away. EC moves z toward the
rotation, MC adds its own compensation on top of the strategy, and the hand
drifts past the target. Silencing the error into EC (the lesion) removes it.

THE PROBE. A trained agent, MC FROZEN (deterministic action). Centre-out
reaches to eight targets. Each trial:
  * the STRATEGY: MC is shown the goal rotated by -theta about the start, so
    an agent that believes there is no rotation pushes exactly so that the
    rotated force carries the hand to the true target;
  * NO ONLINE CORRECTION: MC makes ONE decision and the plant holds it for
    `hold` steps (EC and CB run every plant step, so `hold` sets how much
    evidence a trial carries). hold=0 is the closed-loop reference: MC acts
    every step up to CLOSED_LOOP_STEPS.
  * the CARRY: EC's hidden state (or COIN's posterior) at the end of a trial
    is the next trial's start, as in training -- deterministic here, no leak.
Block design after Mazzoni & Krakauer: N_BASE trials with no rotation, N_PRE
rotation trials WITHOUT the strategy, then N_STRAT rotation trials WITH it.

THE MEASURE. The hand's INTENDED initial direction -- MC's single action
rotated by the plant, before command noise (whose ~16 degrees per step would
otherwise swamp it; the noisy first-step displacement is kept as
`hand_err_noisy`) -- relative to the true target, signed so that positive is
PAST the target toward the aimed side, and BASELINE-SUBTRACTED: each agent's
mean error over its own no-rotation trials is removed, as in the human
studies. That matters here because an agent whose error input is silenced
sits at a z it never trained on and aims with a constant bias; the
subtraction leaves only what the rotation changed. 0 = the strategy works,
+theta = the implicit compensation has fully stacked on it. Called `drift`.

CONDITIONS. hold in HOLDS; lesion (EC's error input zeroed; for COIN, its
observer gets no evidence); strategy on (the paradigm) and off (classic
adaptation, for reference).

    python strategy_drift.py                      # every config, 10 seeds, theta -45
    python strategy_drift.py --theta 45 --holds 2 3
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sac
import task as T
import zero_shot_analysis as ZA
from plotting_utils import use_paper_style, figsize, plot_band, save, trim_spines, label_panels, MUTED, INK

RUNS = {                      # display name -> run directory
    "main [trainrotw]": "main_trainrotw_rotw_zs60_h12_carry",
    "main [trainlin]": "main_trainlin_rotw_zs60_h12_carry",
    "coin": "coin_rotw_zs60_h12",
    "baseline": "baseline_rotw_zs60_h12",
}
THETA = -45.0                 # the rotation; -45 is novel to every agent (the few-shot block was +60)
HOLDS = (4,)                  # 0 = closed loop. 2 and 3 were run and changed nothing
                              # (onset in one trial at every hold); 4 is the cleanest
CLOSED_LOOP_STEPS = 12
N_BASE, N_PRE, N_STRAT = 16, 2, 80
SEED_BASE = 90_000
PLOT_TRIALS = 11              # strategy trials drawn (0..10): the onset is the whole story


def _unit_angle(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _rot(v, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


class Probe:
    """One reloaded agent and its circuit (or observer), carried across trials."""

    def __init__(self, agent_path, device="cpu"):
        self.agent = sac.load_agent(agent_path, device)
        self.cfg = self.agent.config
        self.device = device
        self.ec, self.cb = sac.build_ec_cb(self.cfg, device)
        self.observer = None
        if self.cfg.use_oracle and self.cfg.oracle_source != "given":
            self.observer = sac.make_observer(self.cfg, self.agent.oracle_embed, device)
            self.observer.count_prior = float(self.cfg.observer_count_prior)
        self.reset()

    def reset(self):
        self.h = (torch.zeros(1, sac.EC_HIDDEN, device=self.device) if self.ec is not None else None)
        if self.observer is not None:
            self.observer.reset()

    def trial(self, context_id, theta_shown, target, hold, lesion, rng):
        """One centre-out trial. Returns the hand's initial direction (deg) and
        the first action's angle (deg), both absolute."""
        cfg, dev = self.cfg, self.device
        pos = np.zeros(2, dtype=np.float32)
        vel = np.zeros(2, dtype=np.float32)
        shown_goal = _rot(target, theta_shown)
        h = self.h
        s_tm1 = torch.zeros(1, sac.SDIM, device=dev)
        a_tm1 = torch.zeros(1, sac.ACTION_DIM, device=dev)
        s_np = sac.circuit_state(pos, vel)
        s_t = torch.tensor(s_np, dtype=torch.float32, device=dev).unsqueeze(0)
        e_t = torch.zeros(1, sac.SDIM, device=dev)
        cb_h2 = torch.zeros(1, sac.CB_HIDDEN, device=dev)
        oracle_vec = self.observer.code() if self.observer is not None else (
            sac.oracle_code(None, self.agent.oracle_embed) if cfg.use_oracle else None)
        n_steps = CLOSED_LOOP_STEPS if hold == 0 else int(hold)
        action, first_dir, first_action_angle = None, None, None
        with torch.no_grad():
            for t in range(n_steps):
                e_in = torch.zeros_like(e_t) if lesion else e_t
                obs, z_t, h = sac.augmented_obs_step(
                    self.ec, cfg, h, s_tm1, a_tm1, s_t, e_in, cb_h2, self.agent.cb_proj,
                    pos, vel, shown_goal, oracle_vec, dev, self.agent.ec_offset, rng=rng)
                if t == 0 or hold == 0:
                    o = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                    _, _, a = self.agent.actor.get_action(o)
                    action = a.squeeze(0).cpu().numpy().astype(np.float32)
                    if t == 0:
                        first_action_angle = _unit_angle(action)
                a_full = torch.tensor(action, dtype=torch.float32, device=dev).unsqueeze(0)
                pos_new, vel_new, imp = sac.env_step_params(pos, vel, action, context_id, rng,
                                                            return_impulse=True)
                if t == 0:
                    first_dir = _unit_angle(pos_new - pos)
                if self.observer is not None:
                    self.observer.update(action, vel, vel_new, imp, s_prev=s_np,
                                         s_next=sac.circuit_state(pos_new, vel_new),
                                         evidence=not lesion)
                    oracle_vec = self.observer.code()
                s_pred, cb_h2 = sac.cb_predict(self.cb, s_t, a_full, z_t, dev)
                s_np = sac.circuit_state(pos_new, vel_new)
                s_next = torch.tensor(s_np, dtype=torch.float32, device=dev).unsqueeze(0)
                s_tm1, a_tm1, s_t, e_t = s_t, a_full, s_next, s_next - s_pred
                pos, vel = pos_new, vel_new
        self.h = h
        return first_dir, first_action_angle


def run_block(probe, theta, hold, lesion, strategy, seed, n_pre=N_PRE):
    """The Mazzoni-Krakauer block on one agent: rows of per-trial results."""
    rng = np.random.default_rng(SEED_BASE + seed)
    probe.reset()
    _, targets = sac.eval_battery()
    rot_id, null_id = T.rotation_id(theta), T.rotation_id(0.0)
    phases = ([("base", null_id, 0.0)] * N_BASE + [("pre", rot_id, 0.0)] * int(n_pre)
              + [("strat", rot_id, (-theta if strategy else 0.0))] * N_STRAT)
    order = np.concatenate([rng.permutation(len(targets)) for _ in range(len(phases) // len(targets) + 1)])
    rows = []
    for i, (phase, cid, shown) in enumerate(phases):
        tgt = targets[order[i]]
        hand, aim = probe.trial(cid, shown, tgt, hold, lesion, rng)
        rot_now = theta if phase != "base" else 0.0
        hand_noisy = _wrap(hand - _unit_angle(tgt))
        hand_err = _wrap(aim + rot_now - _unit_angle(tgt))      # intended, noise-free
        rows.append(dict(trial=i, phase=phase, hold=hold, lesion=lesion, strategy=strategy,
                         seed=seed, theta=theta, hand_err=hand_err, hand_err_noisy=hand_noisy,
                         aim_rel_target=_wrap(aim - _unit_angle(tgt)), rotation=rot_now))
    base = np.mean([r["hand_err"] for r in rows if r["phase"] == "base"])
    for r in rows:
        e = _wrap(r["hand_err"] - base)
        # strategy: + is past the target toward the aimed side (-theta).
        # no strategy: + is compensation achieved (theta = full), i.e. how far
        # the hand has come back from where the raw rotation puts it
        r["drift"] = (-e * np.sign(theta) if strategy else
                      (abs(theta) - (e * np.sign(theta))) if r["phase"] != "base" else 0.0)
    return rows


def collect(theta=THETA, holds=HOLDS, seeds=range(10), run_dir=None, configs=None, n_pre=N_PRE):
    root = run_dir or sac.RUN_DIR
    out = []
    for name, d in RUNS.items():
        if configs and name not in configs:
            continue
        for seed in seeds:
            p = os.path.join(root, d, f"seed_{seed}", "agent.pt")
            if not os.path.exists(p):
                continue
            probe = Probe(p)
            lesions = (False, True) if (probe.ec is not None or probe.observer is not None) else (False,)
            for hold in holds:
                for lesion in lesions:
                    for strategy in (True, False):
                        out += [dict(r, variant=name, n_pre=int(n_pre))
                                for r in run_block(probe, theta, hold, lesion, strategy, seed, n_pre)]
    return pd.DataFrame(out)


def _colour(v, bases):
    return ZA._colour(v, bases)


def _label(v):
    """The figure name of a variant: the presets' display names
    (make_ablations_plot.DISPLAY_NAME -- Full Circuit, Motor Cortex, COIN),
    with the circuit pretrained on unstructured transforms marked, since it
    shares its preset with the Full Circuit."""
    import make_ablations_plot as MAP
    base = ZA._base_of(v)
    name = MAP.DISPLAY_NAME.get(base, base)
    return name + (" (Random Linear)" if ZA._tag_of(v) == "trainlin" else "")


SCHEMATIC_VARIANT = "main [trainrotw]"
SETTLED = (2, 10)             # strategy trials averaged for the settled direction


def draw_schematic(ax, d, theta, lesion, variant=SCHEMATIC_VARIANT, colour="k"):
    """Centre-out geometry in the TARGET's frame (target straight up): the
    target, the direction MC was told to aim, and the hand's initial
    direction on the first strategy trial (open arrow) and once settled
    (thin arrows: one per seed; thick: their mean)."""
    g = d[(d.variant == variant) & (d.lesion == lesion)]
    aim_off = -theta                                   # the strategy: aim theta the other way
    sgn = np.sign(aim_off)
    up = 90.0

    def vec(deg, r=1.0):
        t = np.deg2rad(deg)
        return r * np.cos(t), r * np.sin(t)

    def arrow(deg, r, **kw):
        x, y = vec(deg, r)
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, **kw))

    # the target and the aimed direction
    tx, ty = vec(up)
    ax.add_patch(plt.Circle((tx, ty), 0.09, fc="none", ec=MUTED, lw=1.0))
    ax.text(tx, ty + 0.16, "Target", ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.plot(*zip((0, 0), vec(up + aim_off, 1.05)), color=MUTED, lw=0.8, ls="--")
    ax.text(*vec(up + aim_off, 1.2), "Instructed\nAim", ha="center", va="center", fontsize=6,
            color=MUTED, linespacing=0.9, multialignment="center")
    ax.plot(0, 0, "o", color=MUTED, ms=3)
    # the rotation: what the plant does to a push along the instructed aim --
    # a curved arrow from the aim line onto the target line, labelled
    from matplotlib.patches import Arc, FancyArrowPatch
    r_arc = 0.30
    lo_, hi_ = sorted((up, up + aim_off))
    ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=lo_, theta2=hi_, color=INK, lw=0.7))
    # the arrowhead, at the target end, pointing along the arc toward it
    ax.add_patch(FancyArrowPatch(vec(up + 0.06 * aim_off, r_arc), vec(up, r_arc),
                                 arrowstyle="-|>", mutation_scale=6, color=INK, lw=0.7))
    lx, ly = vec(up + aim_off / 2, r_arc + 0.12)
    ax.text(lx, ly, f"{theta:+g}\N{DEGREE SIGN}".replace("-", "\N{MINUS SIGN}"), ha="center",
            va="center", fontsize=6, color=INK)
    # the first strategy trial (mean over seeds)
    first = g[g.k == 0].drift.mean()
    arrow(up + sgn * first, 0.86, color=MUTED, lw=1.2, mutation_scale=8)
    # settled: per seed, then the mean
    settled = g[(g.k >= SETTLED[0]) & (g.k <= SETTLED[1])].groupby("seed").drift.mean()
    for v_ in settled.values:
        arrow(up + sgn * v_, 0.86, color=colour, lw=0.5, mutation_scale=4, alpha=0.35)
    arrow(up + sgn * settled.mean(), 0.86, color=colour, lw=1.8, mutation_scale=10)
    # direct labels, placed just past each mean arrow's tip
    labels = [(up + sgn * first, "first trial", MUTED),
              (up + sgn * settled.mean(), f"trials {SETTLED[0]}-{SETTLED[1]}", colour)]
    if abs(first - settled.mean()) < 15:          # the arrows coincide: one label for both
        labels = [(up + sgn * settled.mean(), "every trial", colour)]
    for ang, txt, col in labels:
        x, y = vec(ang, 0.62)
        ax.text(x + (0.05 if np.cos(np.deg2rad(ang)) >= -0.05 else -0.05), y, txt, fontsize=5.5,
                color=col, ha=("left" if np.cos(np.deg2rad(ang)) >= -0.05 else "right"), va="center")
    # tight to what is drawn: the aim side carries the "Instructed Aim" label
    ax.set_xlim(*((-1.05, 0.38) if sgn > 0 else (-0.38, 1.05))); ax.set_ylim(-0.06, 1.36)
    ax.set_axis_off()
    # no title: the caption names the panels (a intact, b silenced)
    return first, settled.mean()


def draw_straight(ax, df_all, theta, hold, variant=SCHEMATIC_VARIANT, colour="k"):
    """The perturbation alone: no strategy, the first rotation trial. The
    agent aims straight at the target (dashed) and the rotation carries the
    hand theta off it (grey arrow; the mean over seeds, baseline-subtracted)."""
    g = df_all[(df_all.variant == variant) & (~df_all.strategy) & (~df_all.lesion)
               & (df_all.hold == hold)]
    base = g[g.phase == "base"].groupby("seed").hand_err.mean()
    rot = g[g.phase != "base"]
    first = rot[rot.trial == rot.trial.min()].set_index("seed").hand_err
    per_seed = (first - base.reindex(first.index)).apply(_wrap)
    hand = float(per_seed.mean())
    print(f"straight reach under {theta:+g}: first-trial hand {hand:+.1f} deg "
          f"(seeds: {np.round(per_seed.values, 0).tolist()})")
    up = 90.0

    def vec(deg, r=1.0):
        t = np.deg2rad(deg)
        return r * np.cos(t), r * np.sin(t)

    tx, ty = vec(up)
    ax.add_patch(plt.Circle((tx, ty), 0.09, fc="none", ec=MUTED, lw=1.0))
    ax.text(tx, ty + 0.16, "Target", ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.plot(*zip((0, 0), vec(up, 1.05)), color=MUTED, lw=0.8, ls="--")
    ax.plot(0, 0, "o", color=MUTED, ms=3)
    for v_ in per_seed.values:
        xs, ys = vec(up + v_, 0.86)
        ax.annotate("", xy=(xs, ys), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color=MUTED, lw=0.5,
                                    mutation_scale=4, alpha=0.4))
    x, y = vec(up + hand, 0.86)
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color=MUTED, lw=1.6,
                                mutation_scale=9))
    xl, yl = vec(up + hand, 0.62)
    right = np.cos(np.deg2rad(up + hand)) >= -0.05
    ax.text(xl + (0.05 if right else -0.05), yl, "first trial", fontsize=5.5, color=MUTED,
            ha=("left" if right else "right"), va="center")
    from matplotlib.patches import Arc, FancyArrowPatch
    r_arc = 0.30
    lo_, hi_ = sorted((up, up + theta))
    ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=lo_, theta2=hi_, color=INK, lw=0.7))
    ax.add_patch(FancyArrowPatch(vec(up + 0.94 * theta, r_arc), vec(up + theta, r_arc),
                                 arrowstyle="-|>", mutation_scale=6, color=INK, lw=0.7))
    lx, ly = vec(up + theta / 2, r_arc + 0.12)
    ax.text(lx, ly, f"{theta:+g}\N{DEGREE SIGN}".replace("-", "\N{MINUS SIGN}"), ha="center",
            va="center", fontsize=6, color=INK)
    sgn = np.sign(theta)
    # tight to what is drawn: the hand falls on the rotation side
    ax.set_xlim(*((-1.05, 0.38) if sgn > 0 else (-0.38, 1.05))); ax.set_ylim(-0.06, 1.36)
    ax.set_axis_off()
    return hand


def _fit_equal(ax, fig):
    """Equal aspect WITHOUT shrinking the box: after layout, widen whichever
    data range is short so the data's aspect matches the box's. The box then
    fills its grid cell (so its panel label lines up with the plot beside
    it) and nothing drawn inside the original limits is clipped -- which
    adjustable="datalim" did, by NARROWING the x range."""
    fig.canvas.draw()
    bb = ax.get_window_extent()
    box = bb.width / bb.height
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    if w / h < box:
        pad = (h * box - w) / 2; ax.set_xlim(x0 - pad, x1 + pad)
    else:
        # the spare height goes BELOW the drawing, so it sits right under
        # the panel's title instead of floating mid-panel
        pad = w / box - h; ax.set_ylim(y0 - pad, y1)


def make_figure(df, out_path=None, n_trials=PLOT_TRIALS, holds=HOLDS):
    """Mazzoni & Krakauer on the frozen agents. a: the perturbation alone --
    no strategy, the first rotation trial: the Full Circuit aims straight at
    the target (dashed) and the -45 degree rotation (curved arrow) carries
    the hand off it (thin arrows: one per seed; thick: their mean). b, c: the strategy, for
    the rotation-trained circuit, in the target's frame (target straight up, a
    -45 degree rotation, the strategy to aim 45 degrees the other way,
    dashed; the curved arrow is the rotation, which carries a push along the
    instructed aim onto the target). Grey arrow: the hand's initial direction on the first
    strategy trial, mean over seeds -- the strategy works. Thin arrows: each
    seed's direction once settled (strategy trials 2-10, numbered as in c); thick: their mean.
    Intact (b), the hand drifts past the target toward the aim by about the
    whole rotation; with the error into EC silenced (c) it stays on the
    target. d: drift of the hand's initial direction past the target, toward
    the aimed side, across the first eleven strategy trials, mean +- s.e.m.
    over seeds. One decision per trial, held for 4 plant steps: no online
    correction (holds of 2 and 3 steps and closed-loop control gave the same
    one-trial onset). Solid: intact; dotted: the error into EC silenced
    (COIN: its observer gets no evidence). Dashed grey: the full rotation,
    where implicit compensation has fully stacked on the strategy; zero: the
    strategy works. Initial direction is the intended one (MC's action
    rotated by the plant, before command noise), baseline-subtracted."""
    use_paper_style()
    d = df[(df.strategy) & (df.phase == "strat")].copy()
    d["k"] = d.trial - d.trial.min()
    d = d[(d.k < n_trials) & d.hold.isin(holds)]
    holds = sorted(d.hold.unique(), key=lambda h: (h == 0, h))
    variants = ZA._order(d.variant.unique())
    theta = float(d.theta.iloc[0])
    with_schematic = len(holds) == 1 and SCHEMATIC_VARIANT in variants
    n_sch = 3 if with_schematic else 0
    fig = plt.figure(figsize=figsize("double", height=2.5), layout="constrained")
    gs = fig.add_gridspec(1, n_sch + len(holds), width_ratios=[1.35] * n_sch + [1.25] * len(holds))
    sch_axes = [fig.add_subplot(gs[0, i]) for i in range(n_sch)]
    axes = [fig.add_subplot(gs[0, n_sch + i]) for i in range(len(holds))]
    if with_schematic:
        draw_straight(sch_axes[0], df, theta, holds[0], colour=_colour(SCHEMATIC_VARIANT, variants))
        for ax, lesion in zip(sch_axes[1:], (False, True)):
            draw_schematic(ax, d[d.hold == holds[0]], theta, lesion,
                           colour=_colour(SCHEMATIC_VARIANT, variants))
            # drawn as text at the top of the panel, not set_title: a title
            # would shrink these axes and misalign their labels with d's
            ax.text(0.5, 1.0, "Without Prediction Error" if lesion else "With Prediction Error",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=plt.rcParams["axes.titlesize"])
    for ax, hold in zip(axes, holds):
        for v in variants:
            for lesion, ls in ((False, "-"), (True, ":")):
                g = d[(d.hold == hold) & (d.variant == v) & (d.lesion == lesion)]
                if g.empty:
                    continue
                piv = g.pivot_table(index="seed", columns="k", values="drift")
                m = piv.mean(axis=0)
                s = piv.std(axis=0, ddof=1) / np.sqrt(len(piv)) if len(piv) > 1 else m * 0
                plot_band(ax, m.index.values, m.values, s.values, color=_colour(v, variants),
                          ls=ls, lw=1.1, marker="o", ms=2.5, label=(v if not lesion else None))
        a_ = abs(theta)
        ax.set_ylim(-0.75 * a_, 1.75 * a_)
        ax.plot([0, n_trials - 1], [a_, a_], color=MUTED, lw=0.6, ls="--", zorder=0)
        ax.plot([0, n_trials - 1], [0, 0], color=MUTED, lw=0.6, zorder=0)
        if len(holds) > 1:
            ax.set_title("closed loop" if hold == 0 else f"one decision, {hold} steps")
        ax.set_xlabel("Trial after the strategy")
        ax.set_xticks(range(0, n_trials, 2 if n_trials <= 12 else 10))
        ax.set_xlim(-0.5, n_trials - 1 + 0.5)
    axes[0].set_ylabel("Drift past the target (deg)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=_colour(v, variants), lw=1.2, label=_label(v)) for v in variants]
    handles += [Line2D([], [], color=MUTED, ls=":", lw=1.2, label="No Prediction error")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=6)
    for ax in axes:
        trim_spines(ax)
    for ax in sch_axes:
        _fit_equal(ax, fig)
    label_panels(fig, sch_axes + axes)
    if out_path is not None:
        save(fig, out_path, caption=make_figure.__doc__)
    return fig


def summary(df):
    d = df[(df.strategy) & (df.phase == "strat")].copy()
    d["k"] = d.trial - d.trial.min()
    win = {"trials 1-2": (0, 1), "3-10": (2, 9), "41-80": (40, 79)}
    rows = []
    for (v, hold, lesion), g in d.groupby(["variant", "hold", "lesion"]):
        r = dict(variant=v, hold=hold, lesion=lesion)
        for name, (a, b) in win.items():
            r[name] = g[(g.k >= a) & (g.k <= b)].drift.mean()
        # trials to reach half of the final drift (from the per-trial mean)
        m = g.groupby("k").drift.mean()
        final = m[m.index >= 40].mean()
        r["half_trial"] = int((m >= final / 2).idxmax()) if final > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--holds", type=int, nargs="+", default=list(HOLDS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--configs", nargs="+", default=None, choices=list(RUNS))
    ap.add_argument("--n-pre", type=int, default=N_PRE,
                    help=f"rotation trials BEFORE the strategy (Mazzoni & Krakauer: {N_PRE}); "
                         "0 gives the strategy from the first rotation trial")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    df = collect(a.theta, tuple(a.holds), range(a.seeds), configs=a.configs, n_pre=a.n_pre)
    tag = f"_rot{a.theta:+g}".replace("+", "p").replace("-", "m") + ("" if a.n_pre == N_PRE else f"_pre{a.n_pre}")
    csv = os.path.join(sac.RUN_DIR, f"strategy_drift{tag}.csv")
    df.to_csv(csv, index=False)
    print(summary(df).round(1).to_string(index=False))
    out = a.out or os.path.join(sac.FIG_DIR, f"strategy_drift{tag}")
    make_figure(df, out)
    print(f"data -> {csv}\nfigure -> {out}.pdf / .png")


if __name__ == "__main__":
    main()
