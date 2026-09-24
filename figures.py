"""
figures.py -- every figure we need. 

SIX PER SEED, into figures/<arm>/seed<N>/ :

  epoch_curves.png      training history: MSE and per-family decoding vs epoch
  tss_curves.png        PER FACTOR LEVEL vs steps-since-switch: velocity MSE against
                        that family's two oracle rungs, and its decoding curve
                        -- how fast the circuit recovers after a switch
  ablation.png          THE MAIN FIGURE.  intact / zero e / shuffle e, decoding
                        vs tss, oandhow it depends on cb error. 
  decoding_global.png   those two honest readouts on their own, larger
  switch_aligned.png    e lesioned ONLY in a window around each switch - to see if e only infers or is needed continuously
  forces.png            angle between CB's predicted force and the true one,
                        against the context-blind reference

TWO POOLED, into figures/<arm>/across_seeds/ :

  ablation_seeds.png       the main figure, banded across TRAINING SEEDS
  switch_aligned_seeds.png the same for the switch-aligned window

WHICH ERROR BAR.  A per-seed band is +-1 SD over decoder train/test SPLITS: how
precisely context can be read out of THIS network. 

TOLD THE FAMILY, OR NOT.  The four coloured panels hand the decoder the family
and ask which member it is -- chance 1/4.  The last two do not: "which family?"
(chance 1/4) and "which of 16?" (chance 1/16).  Per-family numbers are
systematically optimistic, because MTL itself has to infer the family.  The
16-way readout is the honest one and the ablation looks much larger there.

LINEAR DECODERS ONLY, and the MLP code is now GONE rather than merely unplotted.
It ran alongside every linear decode throughout the earlier work and sat on top
of the linear curve, so the question it answered -- is the code linearly
readable, or merely present -- came back settled.  It also dominated the cost
(~40x per fit, 16.5s against 0.4s), which is what made these figures slow
enough to avoid running.  See decode._clf.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from plotting_utils import (use_paper_style, offset_spines, PALETTE, INK, MUTED, figsize,
                            plot_band, chance_line, label_panels, save)

import model as M
import oracle as O
from data import generate_sequences
from decode import by_tss, family_by_tss
from task import (CONTEXTS, N_CONTEXTS, FAMILY, FAM_NAMES, DT, MASS,
                  POS_SCALE, VEL_SCALE, POS_LIM, TSS_MIN, RHO, force_of)

N_SEQ    = 1200
N_SPLITS = 8
KS       = [0, 1, 2, 3, 4, 6, 8, 10, 12]
# The three cerebellar-channel conditions, plus a control that never touches
# the cerebellum: noise into MTL's own recurrent state, scaled by that state's
# sd.  Without it, "zeroing e hurts" only shows that perturbing something hurts.
COND = [("intact", "intact", INK, "-"),
        ("zero", "$e$ zeroed", PALETTE["rose"], "-"),
        ("shuffle", "$e$ shuffled", PALETTE["amber"], "--"),
        ("mtl_noise", "MTL noise", MUTED, ":")]
FCOL = {"rotation": PALETTE["navy"], "reflection": PALETTE["rose"],
        "curl": PALETTE["teal"], "drift": PALETTE["amber"]}
# ONE grouping: the design is flat again, so a "level" is a family.
LEVELS = list(FAM_NAMES)
LEVEL_INDEX = {n: i for i, n in enumerate(FAM_NAMES)}
# decode the context within each family, decode the family, decode all 16
READOUTS = LEVELS + ["family", "all"]
ARM_TITLE = {"main": "main (MTL denied $s_t$)",
             "control": "control (MTL given $s_t$)"}

use_paper_style()

def _handles():
    return [Line2D([], [], color=c, ls=ls, marker="o", ms=4, lw=1.8, label=l)
            for _, l, c, ls in COND]


def _chance(readout):
    """Chance for each readout.

    A context decode WITHIN one family is over that family's members, so
    chance is 1/8 rather than the 1/4 the crossed design had.
    """
    if readout == "all":
        return 1.0 / N_CONTEXTS
    if readout == "family":
        return 1.0 / len(FAM_NAMES)
    return len(FAM_NAMES) / N_CONTEXTS


def _level_mask(d, name):
    """Boolean mask selecting one family out of a rollout dict."""
    return d["fam"] == LEVEL_INDEX[name]


def _despine(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def ang(u, v):
    """Angle in degrees between two stacks of 2-vectors."""
    du = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12
    c = np.clip((u * v).sum(-1) / du, -1, 1)
    return np.degrees(np.arccos(c))


# ── data ── load ckpts 
def collect(arm, seed, n=N_SEQ, action_mode=M.DEFAULT_ACTION_MODE):
    """Roll the trained arm out under all three conditions.  One place, so every
    figure below is looking at the same rollout.

    `action_mode` picks WHICH circuit -- the two live at different checkpoint
    paths -- and the rollout distribution with it.  Both, together: describing
    a gp-trained circuit on fitted-mode sequences would characterise it on a
    distribution it was never fitted for, and the figures would be labelled gp
    while showing something else.
    """
    loaded = M.load(arm, seed, action_mode)
    if loaded is None:
        return None
    ec, cb = loaded
    trans, ctx, tss = generate_sequences(n, seed=123, action_mode=action_mode)
    out = {"trans": trans,
           "cid": ctx[:, :, 0].numpy().astype(int),
           "fam": ctx[:, :, 1].numpy().astype(int),
           "tss": tss.numpy(),
           "sid": np.repeat(np.arange(n)[:, None], trans.shape[1], 1)}
    for mode, _, _, _ in COND:
        with torch.no_grad():
            z, pred, tgt = M.rollout(ec, cb, trans, arm, mode,
                                     torch.Generator().manual_seed(0))
        e = (pred[:, :-1] - tgt[:, :-1]) ** 2
        # VELOCITY, NOT POSITION.  `se` was the position half, which under this
        # plant is the LEAST informative thing available: pos' = pos + vel'*DT,
        # so echoing the input scores R2 = 0.9966 and 99.7% of the metric is
        # free.  Every ablation then looked similar because the metric was
        # mostly measuring a copy.  All the context lives in the velocity --
        # see multimodality.position_invariance -- so that is what `se` is.
        out[mode] = {"z": z.numpy(), "pred": pred.numpy(),
                     "se": e[..., 2:].mean(-1).numpy(),
                     "se_pos": e[..., :2].mean(-1).numpy()}
        out[mode]["mse"] = float(out[mode]["se"].mean())
        # kept for the convergence check: the two should sit in the ratio
        # (VEL_SCALE/POS_SCALE)^2 once the heads tie -- see model.ALPHA
        out[mode]["mse_pos"] = float(out[mode]["se_pos"].mean())
        print(f"    [{arm} seed {seed}] {mode} rolled out", flush=True)
    return out


def curves(d):
    """Linear decoding vs tss for every (condition, readout), as (mean, sd) over
    N_SPLITS sequence-level splits."""
    res = {}
    for mode, _, _, _ in COND:
        zf = d[mode]["z"].reshape(-1, d[mode]["z"].shape[-1])
        for r in READOUTS:
            y = d["fam"] if r == "family" else d["cid"]
            mask = _level_mask(d, r).ravel() if r in LEVELS else None
            runs = [by_tss(zf, y.ravel(), d["sid"].ravel(), d["tss"].ravel(),
                           KS, mask=mask, seed=s)
                    for s in range(N_SPLITS)]
            R = np.array(runs, float)
            res[(mode, r)] = (np.nanmean(R, 0), np.nanstd(R, 0))
        # each FACTOR derived from the 16-way decoder -- see decode.family_by_tss.
        # Both, because a code can support one factor and not the other, which
        # is exactly the case the crossed design exists to expose.
        grp = np.array([FAMILY[c] for c in range(N_CONTEXTS)])
        runs = [family_by_tss(zf, d["cid"].ravel(), d["fam"].ravel(),
                              d["sid"].ravel(), d["tss"].ravel(), KS, grp,
                              seed=s) for s in range(N_SPLITS)]
        R = np.array(runs, float)
        res[(mode, "family_derived")] = (np.nanmean(R, 0), np.nanstd(R, 0))
    return res


# ── the six per-seed figures ─────
def fig_mtl_control(arm, seed, r, path):
    """The four conditions on one axis, per seed: is the decoding loss specific
    to the CEREBELLAR channel, or does any perturbation do it?

    intact          the ceiling
    e zeroed        the channel carries no information
    e shuffled      the channel carries the WRONG information
    MTL noise       an equal-magnitude hit that never touches the cerebellum

    MTL_NOISE is calibrated (model.py) so it costs the forward model the same
    prediction accuracy that zeroing e does.  Without that matching the
    comparison says nothing except that a bigger perturbation hurts more.

    READ WITH CARE.  Decoding is read out of z, and z is MTL's own output, so
    noise in h sits one step from the readout while zeroing e acts upstream and
    reaches z only through the recurrence.  The MTL curve therefore faces an
    easier route to the measure, and a large drop there is not by itself
    evidence against cerebellar specificity.
    """
    ros = [ro for ro in ("family", "all") if (COND[0][0], ro) in r]
    fig, axes = plt.subplots(1, len(ros), figsize=figsize("double" if len(ros) > 1 else "single",
                                                          height=2.6),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, ro in zip(axes[0], ros):
        for mode, lab, col, ls in COND:
            mu, sd = r[(mode, ro)]
            plot_band(ax, KS, mu, sd, color=col, ls=ls, label=lab)
        ch = _chance(ro)
        chance_line(ax, ch, f"chance {ch:.2f}")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("steps since context switch")
        ax.set_title("which family?" if ro == "family"
                     else f"which context? (all {N_CONTEXTS})")
    axes[0][0].set_ylabel("linear decoding of context")
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=False)
    fig.legend(handles=_handles(), ncol=len(COND), loc="outside upper center")
    who = ARM_TITLE.get(arm, arm) + ("" if seed is None else f", seed {seed}")
    caption = (f"{who}: is the decoding loss specific to the cerebellar channel? "
               f"Linear decoding of context from z against steps since the switch, "
               f"under the intact channel, e zeroed, e shuffled and an equal-magnitude "
               f"noise injected into MTL's own state (which never touches the "
               f"cerebellum). Dotted line is chance; band +-1 SD across {N_SPLITS} "
               f"decoder splits.")
    offset_spines(fig)
    if len(ros) > 1:
        label_panels(fig)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save(fig, path, caption=caption)
    print(f"    saved -> {path}", flush=True)


def fig_ablation(arm, seed, d, r, path, band="split", n_seed=None):
    """The main figure: MSE per condition, then one decoding panel per readout.

    Built at double-column width with the style's type (no per-call font
    sizes), one y label per row with inner tick labels dropped, and the
    which-error-bar note in <path>_caption.txt rather than a suptitle
    (plotting_utils follow-ups 1, 2, 5). Written as PDF + PNG."""
    ncol = int(np.ceil((1 + len(READOUTS)) / 2))
    fig, axes = plt.subplots(2, ncol, figsize=figsize("double", height=4.4),
                             layout="constrained")
    ax = axes.ravel()
    for k in range(1 + len(READOUTS), len(ax)):
        ax[k].axis("off")
    # d[m] is a rollout dict (per seed: one MSE), a float, or an ARRAY of
    # per-seed MSEs (across_seeds): then the bar is the mean with a +-1 SD
    # whisker and one dot per seed, and the label quotes both.
    per_seed = [np.atleast_1d(d[m]["mse"] if isinstance(d[m], dict) else d[m]).astype(float)
                for m, _, _, _ in COND]
    ms = [v.mean() for v in per_seed]
    ax[0].bar(range(len(COND)), ms, color=[c for _, _, c, _ in COND], width=.62, zorder=1)
    for k, v in enumerate(per_seed):
        if len(v) > 1:
            ax[0].errorbar(k, v.mean(), v.std(ddof=1), color=INK, lw=0.8, capsize=0, zorder=3)
            ax[0].scatter(k + np.linspace(-0.18, 0.18, len(v)), v, s=4, color=INK,
                          alpha=0.5, lw=0, zorder=2)
    ax[0].set_xticks(range(len(COND)), [l for _, l, _, _ in COND])
    ax[0].set_ylabel("velocity MSE")
    ax[0].set_title("prediction")
    for k, v in enumerate(per_seed):
        top = v.mean() + (v.std(ddof=1) if len(v) > 1 else 0.0)
        top = max(top, v.max())
        text = f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f}\n±{v.std(ddof=1):.4f}"
        ax[0].annotate(text, (k, top), xytext=(0, 1.5), textcoords="offset points",
                       ha="center", va="bottom", fontsize=plt.rcParams["xtick.labelsize"])
    # LEVEL readouts are titled in their level's colour; the whole-factor
    # readouts (family / all contexts) are not levels and stay in ink
    ttl = {"family": "which family?", "all": f"which context? (all {N_CONTEXTS})"}
    dec = list(ax[1:1 + len(READOUTS)])
    for a, ro in zip(dec, READOUTS):
        if a is not dec[0]:
            a.sharey(dec[0])
        for mode, lab, col, ls in COND:
            mu, sd = r[(mode, ro)]
            plot_band(a, KS, mu, sd, color=col, ls=ls, label=lab)
        ch = _chance(ro)
        chance_line(a, ch, f"chance {ch:.2f}")
        a.set_ylim(0, 1)
        a.set_title(ttl.get(ro, ro), color=INK if ro in ttl else FCOL.get(ro, INK))
        a.set_xlabel("steps since context switch")
    # one y label per row (the first decoding panel of each), inner labels off
    row_first = {axes[i, 1 if i == 0 else 0] for i in range(axes.shape[0])}
    for a in dec:
        if a in row_first:
            a.set_ylabel("decoding accuracy")
        else:
            a.tick_params(labelleft=False)
    fig.legend(handles=_handles(), ncol=len(COND), loc="outside upper center")
    if band == "split":
        err = (f"Line: mean over {N_SPLITS} sequence-level decoder splits; band: +-1 SD "
               f"across splits, an internal check and not the error bar to quote "
               f"(see across_seeds/).")
    else:
        err = (f"Line: mean over {n_seed} training seeds; band: +-1 SD across them, "
               f"the error bar to quote. In (a) the bar is the mean MSE over seeds, "
               f"the whisker +-1 SD and each dot one seed.")
    caption = (f"{ARM_TITLE.get(arm, arm)}: is the cerebellar error necessary? "
               f"(a) Held-out velocity MSE per condition. (b-f) Linear decoding of "
               f"context from z against steps since the context switch, per readout; "
               f"dotted line is chance. {err}")
    offset_spines(fig)
    label_panels(fig, [a for a in ax if a.axison])
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


def fig_global(arm, seed, r, path):
    """The readouts that are NOT handed a factor level, on their own.

    THREE PANELS NOW, NOT TWO.  "which family" was one question; a crossing
    poses two, and they are answered to very different degrees -- on the first
    trained circuit the aim decodes at 0.88 and the field at 0.40 -- so
    collapsing them into one curve would hide the main finding.
    """
    keys = ["family", "all"]
    fig, ax = plt.subplots(1, 2, figsize=figsize("double", height=2.6), sharey=True,
                           layout="constrained")
    ttl = {"family": "which family? " + " / ".join(FAM_NAMES),
           "all": f"which context? (all {N_CONTEXTS})"}
    for i, k in enumerate(keys):
        for mode, lab, col, ls in COND:
            mu, sd = r[(mode, k)]
            plot_band(ax[i], KS, mu, sd, color=col, ls=ls)
            if k == "family" and (mode, f"{k}_derived") in r:
                mu2, _ = r[(mode, f"{k}_derived")]
                ax[i].plot(KS, mu2, ":", color=col, marker="^", ms=3, lw=1, alpha=.85)
        ch = _chance(k)
        chance_line(ax[i], ch, f"chance {ch:.2f}")
        ax[i].set_title(ttl[k])
        ax[i].set_xlabel("steps since context switch")
        ax[i].set_ylim(0, 1.02)
    ax[0].set_ylabel("linear decoding")
    ax[1].tick_params(labelleft=False)
    h = _handles() + [
        Line2D([], [], color=MUTED, ls="-", marker="o", ms=4, lw=1.8,
               label="factor: direct 4-way fit"),
        Line2D([], [], color=MUTED, ls=":", marker="^", ms=4, lw=1.5,
               label="factor: derived from the 16-way fit")]
    fig.legend(handles=h, ncol=3, loc="outside upper center")
    pooled = str(seed).startswith("pooled")
    band = ("+-1 SD across training seeds" if pooled
            else f"+-1 SD across {N_SPLITS} decoder splits")
    caption = (f"{ARM_TITLE.get(arm, arm)}{' (' + str(seed) + ')' if pooled else ''}: "
               f"context decoding with nothing given away. (a) Which family, decoded "
               f"directly (solid) or derived from the 16-way fit (dotted, triangles). "
               f"(b) Which of the {N_CONTEXTS} contexts. Dotted grey line is chance; "
               f"band {band}.")
    offset_spines(fig)
    label_panels(fig)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


def fig_tss(arm, seed, d, path):
    """PER FAMILY, against time since switch: prediction error and decoding.

    Two panels, both for the INTACT network -- the ablation lives in
    ablation.png.  This figure asks what the finished model DOES, family by
    family; the epoch figure asks how it came to be.

    WHY tss ON THE X-AXIS AT ALL.  Every MSE plotted against epoch is POOLED
    over tss, and ~12% of steps sit at tss = 0, where MTL is at chance BY
    CONSTRUCTION: the context has just changed and z was built from the previous
    context's outcome.  Those steps carry near-blind error and drag the pooled
    number well above the settled value, so the model looks permanently worse
    than the oracle when in fact it goes BELOW the one-hot rung once settled.
    """
    lad = O.load()
    fig, ax = plt.subplots(1, 2, figsize=figsize("double", height=2.6), layout="constrained")
    se = d["intact"]["se"]
    tc = d["tss"][:, :-1]
    for i, f in enumerate(LEVELS):
        # se and tc are (B, T-1) -- the last step has no target -- so the mask
        # has to be sliced to match.  _level_mask returns the full (B, T).
        m = _level_mask(d, f)[:, :-1]
        y = [se[m & (tc == k)].mean() if (m & (tc == k)).sum() else np.nan
             for k in KS]
        ax[0].plot(KS, y, "o-", ms=3, color=FCOL[f], label=f)
        if lad:
            # both rungs per family, in that family's colour: dotted = knows
            # nothing, dashed = told the context.  The model should sit between
            # them, and drop toward the dashed line as tss grows.
            ax[0].axhline(lad["blind"][f], ls=":", color=FCOL[f], lw=.8, alpha=.8)
            ax[0].axhline(lad["one-hot"][f], ls=(0, (5, 2)), color=FCOL[f],
                          lw=.8, alpha=.9)
    if lad:
        ax[0].plot([], [], ls=":", color=MUTED, lw=.8, label="context-blind CB")
        ax[0].plot([], [], ls=(0, (5, 2)), color=MUTED, lw=.8,
                   label="oracle: told the context (one-hot)")
    else:
        print("  (no oracle ladder -- run `python oracle.py`; rungs omitted)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("steps since context switch")
    ax[0].set_ylabel("held-out velocity MSE")
    ax[0].set_title("prediction error, per family")
    ax[0].legend()

    zf = d["intact"]["z"].reshape(-1, d["intact"]["z"].shape[-1])
    for i, f in enumerate(LEVELS):
        ax[1].plot(KS, by_tss(zf, d["cid"].ravel(), d["sid"].ravel(),
                              d["tss"].ravel(), KS,
                              mask=_level_mask(d, f).ravel()),
                   "o-", ms=3, color=FCOL[f], label=f)
    ch = _chance(LEVELS[0])                     # within-family: over that family's members
    chance_line(ax[1], ch, f"chance {ch:.2f}")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("steps since context switch")
    ax[1].set_ylabel("within-family context decoding")
    ax[1].set_title("linear decoder, per family")
    ax[1].legend()
    caption = (f"{ARM_TITLE.get(arm, arm)} (AR(1), rho={RHO}), the intact network, per "
               f"family, against time since the context switch rather than epoch: "
               f"tss=0 is the step the context changes, where MTL is at chance by "
               f"construction. (a) Held-out velocity MSE; dotted rung: a context-blind "
               f"CB, dashed rung: an oracle told the context (one-hot). (b) Within-family "
               f"linear decoding of context from z; dotted grey line is chance.")
    offset_spines(fig)
    label_panels(fig)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


def fig_epochs(arm, seed, path, action_mode=M.DEFAULT_ACTION_MODE):
    """Training history.  Needs track.npz, which only train() writes."""
    p = os.path.join(M.fig_dir(arm, seed, action_mode=action_mode), "track.npz")
    if not os.path.exists(p):
        print(f"  (no track.npz for {arm} seed {seed} -- only written during "
              f"training; skipping epoch_curves)")
        return
    z = np.load(p)
    fig, ax = plt.subplots(1, 2, figsize=figsize("double", height=2.6), layout="constrained")
    ep = np.arange(1, len(z["hist"]) + 1)
    ax[0].plot(ep, z["hist"], color=INK, lw=1.2, label="overall")
    pf = z["per_fam"]
    for i, f in enumerate(LEVELS[:pf.shape[1]]):
        ax[0].plot(ep, pf[:, i], color=FCOL[f], lw=.9, label=f)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("held-out velocity MSE")
    ax[0].legend()
    for f in LEVELS:
        # track() writes the level name ALONE.  It was "<level>_linear" while an
        # MLP decoder existed alongside; dropping the MLP dropped the suffix.
        if f in z.files:
            ax[1].plot(z["dec_ep"], z[f], "o-", ms=3, color=FCOL[f], label=f)
    # within a family the decode is over that family's members, so chance is
    # 1/members -- not a hardcoded 0.25, which was right only when the families
    # happened to have four members each
    ch = _chance(LEVELS[0])
    chance_line(ax[1], ch, f"chance {ch:.2f}")
    ax[1].set_ylim(0, 1.03); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("within-family linear decoding")
    ax[1].legend()
    caption = (f"{ARM_TITLE.get(arm, arm)}, seed {seed}: training history. (a) Held-out "
               f"velocity MSE by epoch, overall and per family. (b) Within-family linear "
               f"decoding of context from z at the tracked epochs; dotted line is chance.")
    offset_spines(fig)
    label_panels(fig)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


def fig_forces(arm, seed, d, path):
    """Angle between CB's predicted force and the true one.

    Force geometry is scale-free, so unlike MSE it stays comparable regardless
    of the target's dimension.  The context-blind reference is the force
    averaged over every context: curl's four gains are +-.5/+-.25 and sum to
    zero, so the curl family contributes exactly `a` to that mean whatever the
    velocity -- the vel=0 default is exact here, not an approximation.
    """
    trans = np.asarray(d["trans"])
    p = trans[:, :, 0:2]; v = trans[:, :, 2:4]
    a = trans[:, :, 4:6]
    nxt_p = trans[:, :, 6:8]; nxt_v = trans[:, :, 8:10]
    # F = MASS (v' - v) / DT, read from the STORED velocities.  Differencing
    # positions would give the same answer in the interior and the wrong one at
    # a wall, where the position clips and the velocity is zeroed.
    Ft = MASS * (nxt_v - v) / DT
    Fb = np.stack([[np.mean([force_of(aa, k) for k in range(N_CONTEXTS)], 0)
                    for aa in row] for row in a])
    # the prediction is in build()'s scaled units; undo the VELOCITY scale
    # collect() stores numpy, not tensors
    pr_v = np.asarray(d["intact"]["pred"])[..., 2:] / VEL_SCALE
    Fc = MASS * (pr_v - v) / DT
    # drop wall steps: there the plant zeroed v' and F is not recoverable
    interior = np.all(np.abs(nxt_p) < POS_LIM - 1e-6, axis=-1)
    keep = (d["tss"] >= TSS_MIN) & interior
    fig, ax = plt.subplots(1, len(LEVELS), figsize=figsize("double", height=2.2),
                           sharey=True, layout="constrained")
    cb_col, blind_col = PALETTE["rose"], MUTED
    for i, fm in enumerate(LEVELS):
        m = keep & _level_mask(d, fm)
        A, B = ang(Fc, Ft)[m], ang(Fb, Ft)[m]
        bins = np.linspace(0, 180, 37)
        ax[i].hist(B, bins, color=blind_col, alpha=.55, label=f"blind ({B.mean():.0f}°)")
        ax[i].hist(A, bins, color=cb_col, alpha=.55, label=f"CB ({A.mean():.0f}°)")
        ax[i].axvline(A.mean(), color=cb_col, lw=1)
        ax[i].axvline(B.mean(), color=blind_col, lw=1, ls="--")
        ax[i].set_title(fm, color=FCOL[fm])
        ax[i].legend()
        ax[i].set_xlabel("predicted vs true force (deg)")
        ax[i].set_xlim(0, 180); ax[i].set_xticks([0, 90, 180])   # an angle: not "200"
        if i:
            ax[i].tick_params(labelleft=False)
    ax[0].set_ylabel("count")
    caption = (f"{ARM_TITLE.get(arm, arm)}: does CB's predicted force track the context? "
               f"Per family, the angle between CB's predicted force and the true one on "
               f"settled interior steps (tss >= {TSS_MIN}, no wall contact), against a "
               f"context-blind reference: the force averaged over every context. Vertical "
               f"lines mark the means, quoted in the legends.")
    offset_spines(fig)
    label_panels(fig)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


# Window around a context switch, for switch_aligned: PRE steps before and
# POST after, so OFFSETS runs -PRE..POST-1 with 0 the switch step itself.
# (These sat between two MLP functions and were deleted with them when the MLP
# decoder was removed by splitting the file on `def` boundaries -- a reminder
# that a block-level edit takes whatever module-level code happens to follow
# the block.)
PRE, POST = 4, 4
OFFSETS = list(range(-PRE, POST))


def switch_aligned(arm, seed, n=N_SEQ, action_mode=M.DEFAULT_ACTION_MODE):
    """Lesion e ONLY in a window around each switch, and score the same window.

    Separates re-inferring a NEW context from maintaining one already known.
    Decoding falls on BOTH sides, which says the error is not just a change
    detector -- it feeds a continuous state-estimation loop.
    """
    loaded = M.load(arm, seed, action_mode)
    if loaded is None:
        return None
    ec, cb = loaded
    trans, ctx, tss = generate_sequences(n, seed=123, action_mode=action_mode)
    t = tss.numpy(); cid = ctx[:, :, 0].numpy().astype(int)
    fam = ctx[:, :, 1].numpy().astype(int)
    B, T = t.shape
    mask = np.zeros((B, T), bool)
    sw = []
    for i in range(B):
        for j in range(1, T - POST):
            if t[i, j] == 0 and t[i, j - 1] >= PRE - 1:
                mask[i, j - PRE: j + POST] = True
                sw.append((i, j))
    print(f"  [{arm} seed {seed}] {len(sw)} usable switches", flush=True)
    em = torch.tensor(mask)
    out = {}
    for mode, _, _, _ in COND:
        with torch.no_grad():
            z, _, _ = M.rollout(ec, cb, trans, arm, mode,
                                torch.Generator().manual_seed(0), e_mask=em)
        zn = z.numpy()
        for k, off in enumerate(OFFSETS):
            idx = [(i, j + off) for i, j in sw if 0 <= j + off < T]
            bi = np.array([i for i, _ in idx]); bj = np.array([j for _, j in idx])
            Z = zn[bi, bj]; Y = cid[bi, bj]; F = fam[bi, bj]
            for f in LEVELS:
                m = F == LEVEL_INDEX[f]
                if m.sum() < 60:
                    continue
                from decode import acc
                out[(mode, f, off)] = acc(Z[m], Y[m], bi[m], 0)
    return out


def fig_switch(arm, seed, out, path, band=None, n_seed=None):
    fig, ax = plt.subplots(1, len(LEVELS), figsize=figsize("double", height=2.2),
                           sharey=True, layout="constrained")
    for i, f in enumerate(LEVELS):
        for mode, lab, col, ls in COND:
            if band is None:
                y = [out.get((mode, f, o), np.nan) for o in OFFSETS]
                ax[i].plot(OFFSETS, y, ls, color=col, marker="o", ms=3, lw=1)
            else:
                mu, sd = band[(mode, f)]
                plot_band(ax[i], OFFSETS, mu, sd, color=col, ls=ls)
        ax[i].axvline(-.5, color=MUTED, lw=.8)       # the switch
        ch = _chance(f)
        chance_line(ax[i], ch, f"chance {ch:.2f}" if i == len(LEVELS) - 1 else None)
        ax[i].set_xticks(OFFSETS); ax[i].set_ylim(0, 1.05)
        ax[i].set_title(f, color=FCOL[f])
        ax[i].set_xlabel("steps relative to switch")
        if i == 0:
            ax[i].set_ylabel("linear decoding of context")
        else:
            ax[i].tick_params(labelleft=False)
    fig.legend(handles=_handles(), ncol=len(COND), loc="outside upper center")
    err = ("" if band is None else
           f" Line: mean over {n_seed} training seeds; band +-1 SD across them.")
    caption = (f"{ARM_TITLE.get(arm, arm)}: e lesioned only in the window shown, "
               f"{PRE} steps before to {POST - 1} after a context switch (grey line), "
               f"and decoded in that window. Decoding falls on both sides of the switch, "
               f"so the error feeds a continuous state estimate rather than a change "
               f"detector alone. Dotted line is chance.{err}")
    offset_spines(fig)
    label_panels(fig)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


# ── drivers ──────────────────────────────────────────────────────────────────


def per_seed(arm, seed, action_mode=M.DEFAULT_ACTION_MODE):
    d = collect(arm, seed, action_mode=action_mode)
    if d is None:
        print(f"  no checkpoint for {arm} seed {seed} "
              ); return
    out = M.fig_dir(arm, seed, action_mode=action_mode)
    r = curves(d)
    np.savez(os.path.join(out, "decoding.npz"),
             ks=np.array(KS, float),
             **{f"{m}|mse": d[m]["mse"] for m, _, _, _ in COND},
             **{f"{m}|{ro}|{w}": r[(m, ro)][j]
                for m, _, _, _ in COND
                for ro in READOUTS + ["family_derived"]
                for j, w in ((0, "mu"), (1, "sd"))})
    fig_ablation(arm, seed, d, r, os.path.join(out, "ablation.png"))
    fig_mtl_control(arm, seed, r, os.path.join(out, "mtl_control.png"))
    fig_global(arm, seed, r, os.path.join(out, "decoding_global.png"))
    fig_tss(arm, seed, d, os.path.join(out, "tss_curves.png"))
    fig_epochs(arm, seed, os.path.join(out, "epoch_curves.png"),
               action_mode=action_mode)
    fig_forces(arm, seed, d, os.path.join(out, "forces.png"))
    sa = switch_aligned(arm, seed, action_mode=action_mode)
    if sa:
        np.savez(os.path.join(out, "switch_aligned.npz"),
                 offsets=np.array(OFFSETS),
                 **{f"{m}|{f}": np.array([sa.get((m, f, o), np.nan)
                                          for o in OFFSETS])
                    for m, _, _, _ in COND for f in LEVELS})
        fig_switch(arm, seed, sa, os.path.join(out, "switch_aligned.png"))
    print(f"\n{arm} seed {seed}, linear decoding, mean +- SD over {N_SPLITS} splits")
    print(f"  {'readout':<11}{'mode':<10}" + "".join(f"{k:>13}" for k in KS))
    for ro in READOUTS:
        for m, _, _, _ in COND:
            mu, sd = r[(m, ro)]
            print(f"  {ro:<11}{m:<10}" + "".join(
                f"{a:>8.3f}+-{b:.3f}" for a, b in zip(mu, sd)))


def across_seeds(arm, seeds, action_mode=M.DEFAULT_ACTION_MODE):
    """Pool per-seed npz into figures banded ACROSS SEEDS -- the error bar to
    quote.  Missing seeds are skipped, never assumed, and the count actually
    used is stamped into the figure so an arm at n=2 and one at n=6 cannot be
    silently averaged as if equivalent."""
    d = {}
    for s in seeds:
        p = os.path.join(M.fig_dir(arm, s, action_mode=action_mode), "decoding.npz")
        if os.path.exists(p):
            d[s] = np.load(p)
    if not d:
        print(f"  {arm}: nothing to pool"); return
    # the CIRCUIT'S tag, not the bare arm: this used to be figures/<arm>/, so
    # pooling the ar1 (or any noisy) circuit overwrote the default's panels
    out = os.path.join(M.FIG_DIR, M.arm_tag(arm, action_mode), "across_seeds")
    os.makedirs(out, exist_ok=True)
    n = len(d)
    # per-seed values, not their mean: fig_ablation draws mean, SD and the dots
    mse = {m: np.array([float(z[f"{m}|mse"]) for z in d.values()])
           for m, _, _, _ in COND}
    r = {}
    for m, _, _, _ in COND:
        for ro in READOUTS + ["family_derived"]:
            key = f"{m}|{ro}|mu"
            if not all(key in z.files for z in d.values()):
                continue
            R = np.array([z[key] for z in d.values()], float)
            r[(m, ro)] = (R.mean(0), R.std(0))
    fig_ablation(arm, None, mse, r, os.path.join(out, "ablation_seeds.png"),
                 band="seed", n_seed=n)
    # the two honest readouts, pooled the same way.  Bands here are ACROSS
    # SEEDS, unlike the per-seed version where they are across decoder splits.
    if all((m, "family_derived") in r for m, _, _, _ in COND):
        fig_global(arm, f"pooled over {n} seeds", r,
                   os.path.join(out, "decoding_global_seeds.png"))
    else:
        print("  (no family_derived in the per-seed npz -- regenerate figures "
              "to pool decoding_global)")
    sw = {}
    for s in seeds:
        p = os.path.join(M.fig_dir(arm, s, action_mode=action_mode), "switch_aligned.npz")
        if os.path.exists(p):
            sw[s] = np.load(p)
    if sw:
        band = {}
        for m, _, _, _ in COND:
            for f in LEVELS:
                R = np.array([z[f"{m}|{f}"] for z in sw.values()], float)
                band[(m, f)] = (np.nanmean(R, 0), np.nanstd(R, 0))
        fig_switch(arm, None, None, os.path.join(out, "switch_aligned_seeds.png"),
                   band=band, n_seed=len(sw))
    print(f"\n{arm}: pooled over {n} seeds {sorted(d)}")
    print(f"  {'readout':<11}{'mode':<10}" + "".join(f"{k:>13}" for k in KS))
    for ro in READOUTS:
        for m, _, _, _ in COND:
            mu, sd = r[(m, ro)]
            print(f"  {ro:<11}{m:<10}" + "".join(
                f"{a:>8.3f}+-{b:.3f}" for a, b in zip(mu, sd)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arms", nargs="+", default=list(M.ARMS),
                    help="main (MTL denied s_t) and/or control (MTL given s_t)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pool", action="store_true",
                    help="also write the across-seed figures")
    M.add_circuit_args(ap)
    a = ap.parse_args()
    spec = M.circuit_from_args(a)
    for arm in a.arms:
        for s in a.seeds:
            per_seed(arm, s, action_mode=spec)
        if a.pool:
            across_seeds(arm, a.seeds, action_mode=spec)
