"""
figures.py -- every figure we need. 

SIX PER SEED, into figures/<arm>/seed<N>/ :

  epoch_curves.png      training history: MSE and per-family decoding vs epoch
  tss_curves.png        PER FAMILY vs steps-since-switch: prediction MSE against
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

LINEAR DECODERS ONLY.  An MLP was run alongside throughout and sat on top of the
linear curve, so the question it answered -- is the code linearly readable, or
merely present -- is settled.  Plotting both now costs clarity and adds nothing.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import model as M
import oracle as O
from data import generate_sequences
from decode import by_tss, family_by_tss
from task import (CONTEXTS, N_CONTEXTS, FAMILY, FAMS, DT, LAMBDA, S_SCALE,
                  TSS_MIN, RHO, force_of)

N_SEQ    = 1200
N_SPLITS = 8
KS       = [0, 1, 2, 3, 4, 6, 8, 10, 12]
# The three cerebellar-channel conditions, plus a control that never touches
# the cerebellum: noise into MTL's own recurrent state, scaled by that state's
# sd.  Without it, "zeroing e hurts" only shows that perturbing something hurts.
COND = [("intact", "intact", "#1a1a1a", "-"),
        ("zero", "$e$ zeroed", "#c0392b", "-"),
        ("shuffle", "$e$ shuffled", "#e67e22", "--"),
        ("mtl_noise", "MTL noise", "#7f8c8d", ":")]
FCOL = {"rotation": "#4a6fa5", "reflection": "#c0392b",
        "drift": "#1e8449", "curl": "#8e44ad"}
READOUTS = FAMS + ["family", "all"]
ARM_TITLE = {"main": "main (MTL denied $s_t$)",
             "control": "control (MTL given $s_t$)"}


def _handles():
    return [Line2D([], [], color=c, ls=ls, marker="o", ms=4, lw=1.8, label=l)
            for _, l, c, ls in COND]


def _chance(readout):
    if readout == "all":
        return 1.0 / N_CONTEXTS
    return 1.0 / len(FAMS) if readout == "family" else 0.25


def _despine(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def ang(u, v):
    """Angle in degrees between two stacks of 2-vectors."""
    du = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12
    c = np.clip((u * v).sum(-1) / du, -1, 1)
    return np.degrees(np.arccos(c))


# ── data ── load ckpts 
def collect(arm, seed, n=N_SEQ):
    """Roll the trained arm out under all three conditions.  One place, so every
    figure below is looking at the same rollout."""
    loaded = M.load(arm, seed)
    if loaded is None:
        return None
    ec, cb = loaded
    trans, ctx, tss = generate_sequences(n, seed=123)
    out = {"trans": trans,
           "cid": ctx[:, :, 0].numpy().astype(int),
           "fam": ctx[:, :, 1].numpy().astype(int),
           "tss": tss.numpy(),
           "sid": np.repeat(np.arange(n)[:, None], trans.shape[1], 1)}
    for mode, _, _, _ in COND:
        with torch.no_grad():
            z, pred, tgt = M.rollout(ec, cb, trans, arm, mode,
                                     torch.Generator().manual_seed(0))
        out[mode] = {"z": z.numpy(), "pred": pred.numpy(),
                     "se": ((pred[:, :-1, :2] - tgt[:, :-1, :2]) ** 2)
                            .mean(-1).numpy()}
        out[mode]["mse"] = float(out[mode]["se"].mean())
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
            mask = (d["fam"].ravel() == FAMS.index(r)) if r in FAMS else None
            runs = [by_tss(zf, y.ravel(), d["sid"].ravel(), d["tss"].ravel(),
                           KS, "linear", mask=mask, seed=s)
                    for s in range(N_SPLITS)]
            R = np.array(runs, float)
            res[(mode, r)] = (np.nanmean(R, 0), np.nanstd(R, 0))
        # family DERIVED from the 16-way decoder -- see decode.family_by_tss
        fam_of = np.array([FAMILY[c] for c in range(N_CONTEXTS)])
        runs = [family_by_tss(zf, d["cid"].ravel(), d["fam"].ravel(),
                              d["sid"].ravel(), d["tss"].ravel(), KS, fam_of,
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
    import matplotlib.pyplot as plt
    ros = [ro for ro in ("family", "all") if (COND[0][0], ro) in r]
    fig, axes = plt.subplots(1, len(ros), figsize=(4.6 * len(ros), 4.2),
                             squeeze=False)
    for ax, ro in zip(axes[0], ros):
        for mode, lab, col, ls in COND:
            mu, sd = r[(mode, ro)]
            ax.plot(KS, mu, ls, color=col, marker="o", ms=4, lw=1.8, label=lab)
            ax.fill_between(KS, mu - sd, mu + sd, color=col, alpha=.18, lw=0)
        ch = _chance(ro)
        ax.axhline(ch, ls=":", color="#888", lw=1)
        ax.text(KS[-1], ch + .015, f"chance {ch:.2f}", fontsize=7,
                color="#888", ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("steps since context switch")
        ax.set_title("which family?" if ro == "family"
                     else f"which context?  (all {N_CONTEXTS})", fontsize=11)
        _despine(ax)
    axes[0][0].set_ylabel("linear decoding of context")
    axes[0][0].legend(frameon=False, fontsize=8.5)
    ttl = f"{ARM_TITLE.get(arm, arm)}"
    if seed is not None:
        ttl += f"   seed {seed}"
    fig.suptitle("Is the decoding loss specific to the cerebellar channel?\n"
                 + ttl, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"    saved -> {path}", flush=True)


def fig_ablation(arm, seed, d, r, path, band="split", n_seed=None):
    """The main figure: MSE bars plus one decoding panel per readout."""
    ncol = int(np.ceil((1 + len(READOUTS)) / 2))
    fig, axes = plt.subplots(2, ncol, figsize=(3.3 * ncol, 8.2))
    ax = axes.ravel()
    for k in range(1 + len(READOUTS), len(ax)):
        ax[k].axis("off")
    ms = [d[m]["mse"] if isinstance(d[m], dict) else d[m] for m, _, _, _ in COND]
    ax[0].bar(range(len(COND)), ms, color=[c for _, _, c, _ in COND], width=.62)
    ax[0].set_xticks(range(len(COND)))
    ax[0].set_xticklabels([l for _, l, _, _ in COND], fontsize=8.5)
    ax[0].set_ylabel("position MSE")
    ax[0].set_title("prediction", fontsize=10)
    for k, v in enumerate(ms):
        ax[0].text(k, v * 1.02, f"{v:.4f}", ha="center", fontsize=7.5)
    for i, ro in enumerate(READOUTS):
        a = ax[1 + i]
        for mode, lab, col, ls in COND:
            mu, sd = r[(mode, ro)]
            a.plot(KS, mu, ls, color=col, marker="o", ms=3.5, lw=1.6)
            a.fill_between(KS, mu - sd, mu + sd, color=col, alpha=.20, lw=0)
        ch = _chance(ro)
        a.axhline(ch, ls=":", color="#888", lw=1)
        a.text(KS[-1], ch + .015, f"chance {ch:.2f}", fontsize=6.5,
               color="#888", ha="right")
        a.set_ylim(0, 1.05)
        ttl = {"family": "which family?", "all": f"which context?  (all {N_CONTEXTS})"}
        a.set_title(ttl.get(ro, ro), fontsize=11,
                    color="#1a1a1a" if ro in ttl else FCOL[ro])
        a.set_xlabel("steps since context switch")
        if i == 0:
            a.set_ylabel("linear decoding of context")
        if ro in ("family", "all"):
            a.set_ylabel(f"linear decoding, {'family' if ro=='family' else 'all contexts'}",
                         fontsize=9)
    for a in ax:
        _despine(a)
    fig.legend(handles=_handles(), ncol=3, frameon=False, fontsize=10,
               loc="upper center", bbox_to_anchor=(.5, .935))
    if band == "split":
        sub = (f"line is the mean over {N_SPLITS} sequence-level decoder splits, "
               f"band is $\\pm$1 SD ACROSS SPLITS -- an internal check, not the "
               f"error bar to quote (see across_seeds/)")
    else:
        sub = (f"line is the mean over {n_seed} TRAINING SEEDS, band is $\\pm$1 SD "
               f"across them -- this is the error bar to quote")
    fig.suptitle(f"{ARM_TITLE.get(arm, arm)} -- is the cerebellar error necessary?\n"
                 f"{sub}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def fig_global(arm, seed, r, path):
    """The two readouts that are NOT handed the family, on their own."""
    keys = ["family", "all"]
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ttl = {"family": "which FAMILY?   rotation / reflection / drift / curl",
           "all": f"which CONTEXT?   all {N_CONTEXTS}"}
    for i, k in enumerate(keys):
        for mode, lab, col, ls in COND:
            mu, sd = r[(mode, k)]
            ax[i].plot(KS, mu, ls, color=col, marker="o", ms=4, lw=1.8)
            ax[i].fill_between(KS, mu - sd, mu + sd, color=col, alpha=.20, lw=0)
            if k == "family" and (mode, "family_derived") in r:
                mu2, _ = r[(mode, "family_derived")]
                ax[i].plot(KS, mu2, ":", color=col, marker="^", ms=4, lw=1.5,
                           alpha=.85)
        ch = _chance(k)
        ax[i].axhline(ch, ls=":", color="#888", lw=1)
        ax[i].text(KS[-1], ch + .02, f"chance {ch:.2f}", fontsize=8,
                   color="#888", ha="right")
        ax[i].set_title(ttl[k], fontsize=11)
        ax[i].set_xlabel("steps since context switch")
        ax[i].set_ylabel("linear decoding", fontsize=10)
        ax[i].set_ylim(0, 1.02)
        _despine(ax[i])
    h = _handles() + [
        Line2D([], [], color="#555", ls="-", marker="o", ms=4, lw=1.8,
               label="family: direct 4-way fit"),
        Line2D([], [], color="#555", ls=":", marker="^", ms=4, lw=1.5,
               label="family: derived from the 16-way fit")]
    fig.legend(handles=h, ncol=3, frameon=False, fontsize=9,
               loc="upper center", bbox_to_anchor=(.5, .93))
    band = ("band is $\\pm$1 SD ACROSS TRAINING SEEDS"
            if str(seed).startswith("pooled")
            else f"band is $\\pm$1 SD across {N_SPLITS} decoder splits")
    fig.suptitle(f"{ARM_TITLE.get(arm, arm)}"
                 f"{'  (' + str(seed) + ')' if str(seed).startswith('pooled') else ''}"
                 f": context decoding WITHOUT being told the family\n"
                 f"the decoder faces the same problem MTL does -- nothing is "
                 f"given away;  {band}.\nDotted: family is not linearly "
                 f"separable as 4 regions (each is 4 clusters), but $z$ still "
                 f"determines it.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .84))
    fig.savefig(path, dpi=170); plt.close(fig)
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
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    se = d["intact"]["se"]
    fam, tc = d["fam"][:, :-1], d["tss"][:, :-1]
    for i, f in enumerate(FAMS):
        m = fam == i
        y = [se[m & (tc == k)].mean() if (m & (tc == k)).sum() else np.nan
             for k in KS]
        ax[0].plot(KS, y, "o-", ms=4, color=FCOL[f], label=f)
        if lad:
            # both rungs per family, in that family's colour: dotted = knows
            # nothing, dashed = told the context.  The model should sit between
            # them, and drop toward the dashed line as tss grows.
            ax[0].axhline(lad["blind"][f], ls=":", color=FCOL[f], lw=1.1, alpha=.8)
            ax[0].axhline(lad["one-hot"][f], ls=(0, (5, 2)), color=FCOL[f],
                          lw=1.2, alpha=.9)
    if lad:
        ax[0].plot([], [], ls=":", color="#555", lw=1.1, label="context-blind CB")
        ax[0].plot([], [], ls=(0, (5, 2)), color="#555", lw=1.2,
                   label="oracle: told the context (one-hot)")
    else:
        print("  (no oracle ladder -- run `python oracle.py`; rungs omitted)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("steps since context switch")
    ax[0].set_ylabel("held-out position MSE")
    ax[0].set_title("prediction error, per family", fontsize=10)
    ax[0].legend(frameon=False, fontsize=7.5)

    zf = d["intact"]["z"].reshape(-1, d["intact"]["z"].shape[-1])
    for i, f in enumerate(FAMS):
        ax[1].plot(KS, by_tss(zf, d["cid"].ravel(), d["sid"].ravel(),
                              d["tss"].ravel(), KS, "linear",
                              mask=(d["fam"].ravel() == i)),
                   "o-", ms=4, color=FCOL[f], label=f)
    ax[1].axhline(.25, ls=":", color="#555", lw=1)
    ax[1].text(KS[-1], .265, "chance 0.25", fontsize=7.5, color="#888", ha="right")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("steps since context switch")
    ax[1].set_ylabel("within-family context decoding")
    ax[1].set_title("linear decoder, per family", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7.5)
    for a_ in ax:
        _despine(a_)
    fig.suptitle(
        f"{ARM_TITLE.get(arm, arm)}  (AR(1), rho={RHO}).  x is time since "
        f"switch, not epoch:\ntss=0 is the step the context changes, where MTL "
        f"is at chance by construction.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def fig_epochs(arm, seed, path):
    """Training history.  Needs track.npz, which only train() writes."""
    p = os.path.join(M.fig_dir(arm, seed), "track.npz")
    if not os.path.exists(p):
        print(f"  (no track.npz for {arm} seed {seed} -- only written during "
              f"training; skipping epoch_curves)")
        return
    z = np.load(p)
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ep = np.arange(1, len(z["hist"]) + 1)
    ax[0].plot(ep, z["hist"], color="#1a1a1a", lw=1.6, label="overall")
    pf = z["per_fam"]
    for i, f in enumerate(FAMS[:pf.shape[1]]):
        ax[0].plot(ep, pf[:, i], color=FCOL[f], lw=1.2, label=f)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("held-out position MSE")
    ax[0].legend(frameon=False, fontsize=8)
    for f in FAMS:
        k = f"{f}_linear"
        if k in z.files:
            ax[1].plot(z["dec_ep"], z[k], "o-", ms=3, color=FCOL[f], label=f)
    ax[1].axhline(.25, ls=":", color="#888", lw=1)
    ax[1].set_ylim(0, 1.03); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("within-family linear decoding")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        _despine(a)
    fig.suptitle(f"{ARM_TITLE.get(arm, arm)}: training history (seed {seed})",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def fig_forces(arm, seed, d, path):
    """Angle between CB's predicted force and the true one.

    Force geometry is scale-free, so unlike MSE it stays comparable regardless
    of the target's dimension.  The context-blind reference is the force
    averaged over every context: curl's four gains are +-.5/+-.25 and sum to
    zero, so the curl family contributes exactly `a` to that mean whatever the
    velocity -- the vel=0 default is exact here, not an approximation.
    """
    trans = d["trans"]
    p = trans[:, :, :2].numpy(); a = trans[:, :, 2:4].numpy()
    nxt = trans[:, :, 4:6].numpy()
    v = np.zeros_like(p); v[:, 1:] = (p[:, 1:] - p[:, :-1]) / DT
    Ft = (nxt - p) / DT - LAMBDA * v
    Fb = np.stack([[np.mean([force_of(aa, k) for k in range(N_CONTEXTS)], 0)
                    for aa in row] for row in a])
    pr = d["intact"]["pred"][..., :2] / S_SCALE
    Fc = (pr - p) / DT - LAMBDA * v
    keep = d["tss"] >= TSS_MIN
    fig, ax = plt.subplots(1, len(FAMS), figsize=(4.35 * len(FAMS), 4.0))
    for i, fm in enumerate(FAMS):
        m = keep & (d["fam"] == i)
        A, B = ang(Fc, Ft)[m], ang(Fb, Ft)[m]
        bins = np.linspace(0, 180, 37)
        ax[i].hist(B, bins, color="#7f8c8d", alpha=.55,
                   label=f"blind ({B.mean():.0f}$\\degree$)")
        ax[i].hist(A, bins, color="#c0392b", alpha=.55,
                   label=f"CB ({A.mean():.0f}$\\degree$)")
        ax[i].axvline(A.mean(), color="#c0392b", lw=1.5)
        ax[i].axvline(B.mean(), color="#7f8c8d", lw=1.5, ls="--")
        ax[i].set_title(fm, fontsize=10)
        ax[i].legend(frameon=False, fontsize=8)
        ax[i].set_xlabel("angle: predicted vs true force (deg)")
        _despine(ax[i])
    ax[0].set_ylabel("count")
    fig.suptitle(f"{ARM_TITLE.get(arm, arm)} -- does CB's predicted force track "
                 f"the context?", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def mlp_curves(d, n_splits=3):
    """Per-family MLP decoding vs tss, for every condition.

    Separated out because MLP fits are ~40x slower than linear (16.5 s vs 0.4 s
    for a 4-way family fit), so they run at n_splits=3 against linear's 8 and
    are computed ONCE here, then reused by both figures that need them.
    """
    out = {}
    zs = {m: d[m]["z"].reshape(-1, d[m]["z"].shape[-1]) for m, _, _, _ in COND}
    for mode, _, _, _ in COND:
        for i, f in enumerate(FAMS):
            runs = [by_tss(zs[mode], d["cid"].ravel(), d["sid"].ravel(),
                           d["tss"].ravel(), KS, "mlp",
                           mask=(d["fam"].ravel() == i), seed=sp)
                    for sp in range(n_splits)]
            out[(mode, f)] = np.nanmean(np.array(runs, float), 0)
        print(f"    mlp: {mode} done", flush=True)
    return out


def fig_decoding_curves(arm, seed, r, mlp, path):
    """What can be read out of z: the same four families under each decoder.

    INTACT only -- the ablation lives in ablation.png.  Two panels rather than
    one overlay so the families stay legible; the point is that the two panels
    look the same, i.e. the context is not merely present in z but linearly
    readable.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for j, (kind, lab) in enumerate((("linear", "linear decoder"),
                                     ("mlp", "MLP decoder"))):
        for i, f in enumerate(FAMS):
            y = r[("intact", f)][0] if kind == "linear" else mlp[("intact", f)]
            ax[j].plot(KS, y, "o-", ms=4, color=FCOL[f],
                       label=f if j == 0 else None)
        ax[j].axhline(.25, ls=":", color="#555", lw=1)
        ax[j].set_ylim(0, 1.03)
        ax[j].set_xlabel("steps since context switch")
        ax[j].set_title(lab, fontsize=10)
        _despine(ax[j])
    ax[0].set_ylabel("within-family context decoding")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{ARM_TITLE.get(arm, arm)}  (AR(1) actions, rho={RHO}) -- "
                 f"what can be read out of $z$", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def fig_linear_vs_mlp(arm, seed, r, mlp, path, n_mlp_splits=3):
    """Both decoders under ABLATION -- the evidence for plotting linear only.

    The MLP asks whether context is PRESENT in z at all; the linear decoder asks
    whether it is present in an explicit, linearly readable format.  Those can
    come apart: an ablation could leave the information intact while destroying
    the format, and a linear-only figure would then overstate the damage.  This
    is the check that they do not come apart here -- including under zeroed and
    shuffled e, which decoding_curves.png does not show.
    """
    fig, ax = plt.subplots(1, len(FAMS), figsize=(3.5 * len(FAMS), 4.2))
    diffs = []
    for i, f in enumerate(FAMS):
        for mode, lab, col, ls in COND:
            mu_lin, mu_mlp = r[(mode, f)][0], mlp[(mode, f)]
            ax[i].plot(KS, mu_lin, "-", color=col, marker="o", ms=3.5, lw=1.7)
            ax[i].plot(KS, mu_mlp, ":", color=col, marker="^", ms=4, lw=1.5,
                       alpha=.9)
            diffs.append(np.nanmean(np.abs(mu_mlp - mu_lin)))
        ax[i].axhline(.25, ls=":", color="#888", lw=1)
        ax[i].set_ylim(0, 1.05)
        ax[i].set_title(f, fontsize=11, color=FCOL[f])
        ax[i].set_xlabel("steps since context switch")
        if i == 0:
            ax[i].set_ylabel("within-family context decoding")
        _despine(ax[i])
    md = float(np.nanmean(diffs))
    h = _handles() + [
        Line2D([], [], color="#555", ls="-", marker="o", ms=4, lw=1.7,
               label=f"linear ({N_SPLITS} splits)"),
        Line2D([], [], color="#555", ls=":", marker="^", ms=4, lw=1.5,
               label=f"MLP ({n_mlp_splits} splits)")]
    fig.legend(handles=h, ncol=5, frameon=False, fontsize=9,
               loc="upper center", bbox_to_anchor=(.5, .96))
    fig.suptitle(
        f"{ARM_TITLE.get(arm, arm)}: why the other figures plot LINEAR only\n"
        f"the two decoders coincide (mean |MLP - linear| = {md:.3f}), so the "
        f"context is not merely present in $z$ -- it is linearly readable, and "
        f"the ablation destroys the format as well as the information.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .85))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}   mean |MLP-linear| = {md:.3f}")
    return md


# ── switch-aligned lesion ────────────────────────────────────────────────────
PRE, POST = 4, 4
OFFSETS = list(range(-PRE, POST))


def switch_aligned(arm, seed, n=N_SEQ):
    """Lesion e ONLY in a window around each switch, and score the same window.

    Separates re-inferring a NEW context from maintaining one already known.
    Decoding falls on BOTH sides, which says the error is not just a change
    detector -- it feeds a continuous state-estimation loop.
    """
    loaded = M.load(arm, seed)
    if loaded is None:
        return None
    ec, cb = loaded
    trans, ctx, tss = generate_sequences(n, seed=123)
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
            for fi, f in enumerate(FAMS):
                m = F == fi
                if m.sum() < 60:
                    continue
                from decode import acc
                out[(mode, f, off)] = acc(Z[m], Y[m], bi[m], "linear", 0)
    return out


def fig_switch(arm, seed, out, path, band=None, n_seed=None):
    fig, ax = plt.subplots(1, len(FAMS), figsize=(3.4 * len(FAMS), 4.0))
    for i, f in enumerate(FAMS):
        for mode, lab, col, ls in COND:
            if band is None:
                y = [out.get((mode, f, o), np.nan) for o in OFFSETS]
                ax[i].plot(OFFSETS, y, ls, color=col, marker="o", ms=4, lw=1.6)
            else:
                mu, sd = band[(mode, f)]
                ax[i].plot(OFFSETS, mu, ls, color=col, marker="o", ms=4, lw=1.6)
                ax[i].fill_between(OFFSETS, mu - sd, mu + sd, color=col,
                                   alpha=.20, lw=0)
        ax[i].axvline(-.5, color="#888", lw=1)
        ax[i].axhline(.25, ls=":", color="#888", lw=1)
        ax[i].set_xticks(OFFSETS); ax[i].set_ylim(0, 1.05)
        ax[i].set_title(f, fontsize=11, color=FCOL[f])
        ax[i].set_xlabel("steps relative to switch")
        if i == 0:
            ax[i].set_ylabel("linear decoding of context")
        _despine(ax[i])
    fig.legend(handles=_handles(), ncol=3, frameon=False, fontsize=10,
               loc="upper center", bbox_to_anchor=(.5, .965))
    tail = ("" if band is None else
            f"\nmean $\\pm$1 SD across {n_seed} training seeds")
    fig.suptitle(f"{ARM_TITLE.get(arm, arm)}: $e$ lesioned only in the window shown "
                 f"-- decoding falls on BOTH sides of the switch{tail}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


# ── drivers ──────────────────────────────────────────────────────────────────
def per_seed(arm, seed, with_mlp=False):
    d = collect(arm, seed)
    if d is None:
        print(f"  no checkpoint for {arm} seed {seed}"); return
    out = M.fig_dir(arm, seed)
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
    fig_epochs(arm, seed, os.path.join(out, "epoch_curves.png"))
    fig_forces(arm, seed, d, os.path.join(out, "forces.png"))
    if with_mlp:
        mlp = mlp_curves(d)
        np.savez(os.path.join(out, "mlp.npz"),
                 ks=np.array(KS, float),
                 **{f"{m}|{f}": mlp[(m, f)] for m, _, _, _ in COND for f in FAMS})
        fig_decoding_curves(arm, seed, r, mlp,
                            os.path.join(out, "decoding_curves.png"))
        fig_linear_vs_mlp(arm, seed, r, mlp,
                          os.path.join(out, "linear_vs_mlp.png"))
    sa = switch_aligned(arm, seed)
    if sa:
        np.savez(os.path.join(out, "switch_aligned.npz"),
                 offsets=np.array(OFFSETS),
                 **{f"{m}|{f}": np.array([sa.get((m, f, o), np.nan)
                                          for o in OFFSETS])
                    for m, _, _, _ in COND for f in FAMS})
        fig_switch(arm, seed, sa, os.path.join(out, "switch_aligned.png"))
    print(f"\n{arm} seed {seed}, linear decoding, mean +- SD over {N_SPLITS} splits")
    print(f"  {'readout':<11}{'mode':<10}" + "".join(f"{k:>13}" for k in KS))
    for ro in READOUTS:
        for m, _, _, _ in COND:
            mu, sd = r[(m, ro)]
            print(f"  {ro:<11}{m:<10}" + "".join(
                f"{a:>8.3f}+-{b:.3f}" for a, b in zip(mu, sd)))


def across_seeds(arm, seeds):
    """Pool per-seed npz into figures banded ACROSS SEEDS -- the error bar to
    quote.  Missing seeds are skipped, never assumed, and the count actually
    used is stamped into the figure so an arm at n=2 and one at n=6 cannot be
    silently averaged as if equivalent."""
    d = {}
    for s in seeds:
        p = os.path.join(M.fig_dir(arm, s), "decoding.npz")
        if os.path.exists(p):
            d[s] = np.load(p)
    if not d:
        print(f"  {arm}: nothing to pool"); return
    out = os.path.join(M.FIG_DIR, arm, "across_seeds")
    os.makedirs(out, exist_ok=True)
    n = len(d)
    mse = {m: float(np.mean([float(z[f"{m}|mse"]) for z in d.values()]))
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
        p = os.path.join(M.fig_dir(arm, s), "switch_aligned.npz")
        if os.path.exists(p):
            sw[s] = np.load(p)
    if sw:
        band = {}
        for m, _, _, _ in COND:
            for f in FAMS:
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
    ap.add_argument("--mlp", action="store_true",
                    help="also make linear_vs_mlp.png (slow: ~10 min per seed)")
    ap.add_argument("--pool", action="store_true",
                    help="also write the across-seed figures")
    a = ap.parse_args()
    for arm in a.arms:
        for s in a.seeds:
            per_seed(arm, s, with_mlp=a.mlp)
        if a.pool:
            across_seeds(arm, a.seeds)
