"""
representation_analysis.py -- HOW is context represented, not just whether it
can be read out.

`figures.py` answers two questions: can context be decoded from z, and does
removing the cerebellar error break that. This module asks what the code looks
like. Four sites, all returned by `model.rollout(..., sites=True)`:

    z            MTL's output, 64-d      -- what CB actually receives
    mtl_hidden   MTL's GRU state, 256-d  -- what MTL holds internally
    cb_layer1    CB post-ReLU 1, 256-d   -- DOWNSTREAM of z
    cb_layer2    CB post-ReLU 2, 256-d   -- DOWNSTREAM of z

On the CB sites: context being present there is guaranteed by construction,
since z is an input. The question worth asking is whether CB REFORMATS it --
entangles context with action so the force prediction becomes linear.

ANALYSES (each writes an npz beside its png, so figures can be restyled without
refitting anything):

  --confusion  which contexts get mistaken for which, and does the confusion
               respect family boundaries.  Two views of the same thing: the
               decoder's actual errors, and the cosine similarity between the
               decoder's weight vectors for each pair of contexts.  The first is
               behavioural, the second geometric; when they agree the confusion
               reflects representational overlap rather than a quirk of the
               classifier.
               CAUTION on the cosine matrix: multinomial logistic weights are
               defined only up to an additive constant across classes (the
               softmax gauge), and L2 picks the centered solution -- the K rows
               sum to exactly zero in every z dimension.  K zero-sum vectors
               have mean pairwise cosine -1/(K-1), so NEUTRAL IS NOT ZERO and
               the faint blue background is arithmetic rather than a finding.
               At K = 16 that was -0.067 (measured -0.065); at the current
               K = 8 it is -0.143, twice as far from zero.  The figure prints
               the value it should be read against, so trust the figure over
               any number quoted here.
  --pca        top PCs at each site, coloured by context, family, position and
               velocity.  The kinematic colourings matter: z carries velocity as
               well as context (that is why the oracle ladder needed velocity to
               be a fair ceiling), so PC1-2 may well be kinematics with context
               living elsewhere.
  --topk       decoding accuracy from the top-k PCs, k = 2..64.  This is the
               QUANTITATIVE version of "is context a major or a minor axis" --
               eyeballing a PCA cannot settle it.  A curve that stays flat and
               then climbs late means context lives in low-variance directions,
               i.e. mixed selectivity.

EVERYTHING IS RESTRICTED TO tss >= TSS_MIN by default.  Early in a block the
context genuinely is not established yet, so including those steps measures how
FAST the circuit infers rather than what it ends up representing, and smears
every geometry.

  python representation_analysis.py --confusion --arms main control --seeds 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import model as M
from data import generate_sequences
from decode import _clf, seq_split
from task import (CONTEXTS, C_LABEL, FAMILY, FAM_NAMES, N_CONTEXTS, TSS_MIN)
from plotting_utils import (use_paper_style, offset_spines, bracket_xticks, PALETTE, INK,
                            figsize, colorbar, label_panels, save)

N_SEQ = 1200
N_SPLITS = 8
SITES = ("z", "mtl_hidden", "cb_layer1", "cb_layer2")
FCOL = {"rotation": PALETTE["navy"], "reflection": PALETTE["rose"],
        "curl": PALETTE["teal"], "drift": PALETTE["amber"]}
# contexts grouped by family, which is the order that makes the block
# structure visible in the confusion matrix
ORDER = sorted(range(N_CONTEXTS), key=lambda c: (FAMILY[c], c))
LABELS = [C_LABEL[c] for c in ORDER]


def _despine(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def collect(arm, seed, mode="intact", n=N_SEQ, tss_min=TSS_MIN,
            action_mode=M.DEFAULT_ACTION_MODE):
    """Roll out and return the four sites plus labels, restricted to settled
    steps.

    `action_mode` selects WHICH supervised circuit -- the two live at different
    checkpoint paths. It also selects the rollout distribution below, because
    measuring a gp-trained circuit's geometry on fitted-mode sequences would
    describe it on a distribution it was not fitted for.
    """
    loaded = M.load(arm, seed, action_mode)
    if loaded is None:
        return None
    ec, cb = loaded
    trans, ctx, tss = generate_sequences(n, seed=123, action_mode=action_mode)
    with torch.no_grad():
        _, _, _, sites = M.rollout(ec, cb, trans, arm, mode,
                                   torch.Generator().manual_seed(0), sites=True)
    t = tss.numpy()
    keep = (t >= tss_min).ravel()
    cid = ctx[:, :, 0].numpy().astype(int).ravel()
    fam = ctx[:, :, 1].numpy().astype(int).ravel()
    sid = np.repeat(np.arange(n)[:, None], t.shape[1], 1).ravel()
    out = {k: v.numpy().reshape(-1, v.shape[-1])[keep] for k, v in sites.items()}
    out.update(cid=cid[keep], fam=fam[keep], sid=sid[keep],
               tss=t.ravel()[keep])
    # kinematics, for the PCA colourings: the same observation the network sees
    obs, act, _ = M.build(trans)
    obs = obs.numpy().reshape(-1, obs.shape[-1])[keep]
    out["pos"], out["vel"] = obs[:, :2], obs[:, 2:]
    # THE ACTION. It is the variable the contexts actually transform
    # (F = A(c) a), so it is where an interaction between context and state is
    # theoretically expected.
    out["act"] = act.numpy().reshape(-1, act.shape[-1])[keep]
    return out


# ── 1. confusion 
def confusion(d, site="z", n_splits=N_SPLITS):
    """KxK row-normalised confusion (K = N_CONTEXTS), plus the decoder
    weight-cosine matrix.

    Predictions are ACCUMULATED across splits rather than averaging per-split
    matrices, so every row has enough counts to be read.  The weight matrix is
    averaged over splits.
    """
    X, y, sid = d[site], d["cid"], d["sid"]
    C = np.zeros((N_CONTEXTS, N_CONTEXTS), float)
    W = []
    for s in range(n_splits):
        te = seq_split(sid, seed=s)
        if te.all() or (~te).all():
            continue
        clf = _clf(s).fit(X[~te], y[~te])
        pred = clf.predict(X[te])
        for a, b in zip(y[te], pred):
            C[a, b] += 1
        W.append(clf[-1].coef_)                  # (N_CONTEXTS, n_features)
    C = C[np.ix_(ORDER, ORDER)]
    C = C / np.maximum(C.sum(1, keepdims=True), 1)    # row-normalise
    Wm = np.mean(W, 0)
    Wn = Wm / (np.linalg.norm(Wm, axis=1, keepdims=True) + 1e-12)
    S = (Wn @ Wn.T)[np.ix_(ORDER, ORDER)]             # cosine similarity
    return C, S


def fig_confusion(arm, seed, C, S, path, site="z"):
    """Two separate figures: the annotated confusion, and the weight cosine.

    Separate rather than side by side because a KxK matrix with a number in
    every cell needs the space to stay legible.
    """
    acc = float(np.mean(np.diag(C)))
    within = _within_family_mass(C)
    full = [C_LABEL[c] for c in ORDER]

    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    im = ax.imshow(C, cmap="paper_seq", vmin=0, vmax=1)
    for i in range(N_CONTEXTS):
        for j in range(N_CONTEXTS):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > .55 else INK)
    ax.set_xticks(range(N_CONTEXTS)); ax.set_yticks(range(N_CONTEXTS))
    ax.set_xticklabels(full, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(full, fontsize=7.5)
    ax.set_xlabel("predicted context"); ax.set_ylabel("true context")
    bnd = np.cumsum([sum(1 for c in ORDER if FAMILY[c] == f)
                     for f in range(len(FAM_NAMES))])[:-1] - .5
    for x in bnd:
        ax.axhline(x, color="w", lw=1.4); ax.axvline(x, color="w", lw=1.4)
    colorbar(im, ax=ax, label="P(predicted | true)")
    ax.set_title(f"{arm} {seed if str(seed).startswith('pooled') else f'seed {seed}'}"
                 f" -- context confusion in {site}  "
                 f"(tss >= {TSS_MIN})\n"
                 f"rows sum to 1; accuracy {acc:.3f}; {within['family']:.0%} "
                 f"of errors stay inside the true family "
                 f"(chance {within['chance']:.0%})", fontsize=10.5)
    fig.tight_layout()
    offset_spines(fig)
    save(fig, path)
    print(f"  saved -> {path}")

    wpath = path.replace("confusion_", "weightcos_")
    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    im = ax.imshow(S, cmap="paper_div", vmin=-1, vmax=1)
    for i in range(N_CONTEXTS):
        for j in range(N_CONTEXTS):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(S[i, j]) > .6 else INK)
    ax.set_xticks(range(N_CONTEXTS)); ax.set_yticks(range(N_CONTEXTS))
    ax.set_xticklabels(full, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(full, fontsize=7.5)
    for x in bnd:
        ax.axhline(x, color=INK, lw=1.0); ax.axvline(x, color=INK, lw=1.0)
    colorbar(im, ax=ax, label="weight cosine")
    ax.set_title(
        f"{arm} {seed if str(seed).startswith('pooled') else f'seed {seed}'}"
        f" -- decoder weight cosine in {site}\n"
        f"correlation between the decoder's weight vector for each pair of "
        f"contexts.  +1 = both read off the same\ndirection in {site}, so they "
        f"overlap and will confuse.  Neutral is {-1/(N_CONTEXTS-1):+.3f}, not 0: "
        f"the {N_CONTEXTS} rows sum to\nzero by construction, so only departures "
        f"from that mean anything.\n\n"
        f"WHY THIS AS WELL AS THE CONFUSION.  Confusion is behavioural -- which "
        f"errors actually happen.\nThis is geometric -- where the contexts sit "
        f"in {site} -- and uses no test data at all.  A pair that\nconfuses AND "
        f"has aligned weights overlaps in the representation; if the weights "
        f"were unrelated,\nthe fault would be the classifier instead.",
        fontsize=8.5, loc="left")

    fig.tight_layout()
    offset_spines(fig)
    save(fig, wpath)
    print(f"  saved -> {wpath}")
    return acc, within


def _within_family_mass(C):
    """Fraction of OFF-DIAGONAL mass that stays inside the true family, and
    the chance level it has to beat.

    CHANCE IS NOT ZERO.  With 8 members per family, a uniformly wrong guess
    stays inside the family 7/15 of the time, so 0.47 is neutral and only
    departures from it mean anything.  Reporting the raw fraction without the
    reference reads as a strong result when it is arithmetic.
    """
    fam = np.array([FAMILY[c] for c in ORDER])
    off = C.copy()
    np.fill_diagonal(off, 0)
    tot = max(off.sum(), 1e-12)
    per_fam = N_CONTEXTS // len(FAM_NAMES)
    return {"family": float(off[fam[:, None] == fam[None, :]].sum() / tot),
            "chance": (per_fam - 1) / (N_CONTEXTS - 1)}


# ── 2. PCA: what explains the variance at each site ──────────────────────────
def pca_sites(d, n_comp=64):
    """PCA at every site, plus the kinematic variables to colour by.

    The colourings matter as much as the projection.  z carries velocity as well
    as context -- that is why the oracle ladder needed velocity to be a fair
    ceiling -- so if PC1-2 are kinematics with context elsewhere, that is most
    of the answer to "is context a major axis", before --topk quantifies it.
    """
    out = {}
    for site in SITES:
        X = d[site]
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2) / (S ** 2).sum()
        out[site] = {"proj": Xc @ Vt[:n_comp].T, "var": var,
                     "comp": Vt[:n_comp]}
    return out


def fig_pca(arm, seed, d, P, path, n_show=4000, rng_seed=0):
    """One row per site: cumulative variance, then PC1-2 coloured four ways."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(d["cid"]), size=min(n_show, len(d["cid"])),
                     replace=False)
    pos = d["pos"][idx]
    spd = np.linalg.norm(d["vel"][idx], axis=1)
    colourings = [
        ("context", d["cid"][idx], "tab20", None),
        ("family", d["fam"][idx], None, None),
        ("position (x)", pos[:, 0], "viridis", None),
        ("speed |v|", spd, "plasma", (np.percentile(spd, 1),
                                      np.percentile(spd, 99))),
    ]
    fig, axes = plt.subplots(len(SITES), 1 + len(colourings),
                             figsize=(4.2 * (1 + len(colourings)),
                                      4.1 * len(SITES)))
    for si, site in enumerate(SITES):
        v = P[site]["var"]
        a = axes[si, 0]
        a.plot(np.arange(1, 21), np.cumsum(v[:20]), "o-", ms=3, color="#333")
        a.axhline(.9, ls=":", color="#888", lw=1)
        a.set_xlabel("PC"); a.set_ylabel("cumulative variance")
        a.set_ylim(0, 1.02)
        a.set_title(f"{site}\nPC1 {v[0]:.0%}, PC1-2 {v[:2].sum():.0%}, "
                    f"PC1-10 {v[:10].sum():.0%}", fontsize=9)
        _despine(a)
        pr = P[site]["proj"][idx]
        for ci, (name, c, cmap, lim) in enumerate(colourings):
            a = axes[si, 1 + ci]
            if name == "family":
                for fi, f in enumerate(FAM_NAMES):
                    m = c == fi
                    a.scatter(pr[m, 0], pr[m, 1], s=2, alpha=.35,
                              color=FCOL[f], label=f if si == 0 else None)
                if si == 0:
                    a.legend(frameon=False, fontsize=7, markerscale=4)
            else:
                kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
                a.scatter(pr[:, 0], pr[:, 1], s=2, alpha=.35, c=c, cmap=cmap, **kw)
            a.set_xlabel("PC1"); a.set_ylabel("PC2")
            a.set_title(name, fontsize=9)
            _despine(a)
    fig.suptitle(
        f"{arm} seed {seed} -- what explains the variance at each site  "
        f"(tss >= {TSS_MIN})\nif PC1-2 separate by position or speed but not by "
        f"context, context is a MINOR axis -- see the top-k decoding figure for "
        f"the quantitative version", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    offset_spines(fig)
    save(fig, path)
    print(f"  saved -> {path}")


def pc_variance(d, P, n_pc=10):
    """For each PC, how much of ITS variance is explained by context, by
    position, and by velocity.

    This is the paper-ready version of the PCA question.  A scatter can only
    suggest what a component encodes; this measures it.  For each principal
    component (a 1-D signal over samples):

        context   eta^2 -- between-context variance / total.  Exactly the R^2 of
                  regressing the PC on an N_CONTEXTS-way one-hot, no linearity
                  assumed.
        family    eta^2 over the two families (rotation / reflection). Family
                  is NESTED in context, so family <= context on every PC and
                  the gap is what the PC encodes about WHICH member of a
                  family. A PC with family ~ context has abstracted the
                  family and nothing else; one with family ~ 0 and context
                  high separates members but not families.
        position  R^2 of an OLS fit on (x, y)
        velocity  R^2 of an OLS fit on (vx, vy)

    TWO CAVEATS, both real.  (1) These do not sum to 1 and may overlap: drift
    contexts systematically push the agent around, so context and position are
    genuinely correlated in this task and the same variance can be claimed
    twice.  Read them as "how much does each variable explain on its own", not
    as a partition.  (2) The position and velocity fits are LINEAR, so a
    nonlinear dependence would be understated; context is not, since eta^2 is
    assumption-free.  That asymmetry favours context, so treat a context
    advantage as suggestive unless it is large.
    """
    out = {}
    for site in SITES:
        pr = P[site]["proj"][:, :n_pc]
        var = P[site]["var"][:n_pc]
        r2 = {"context": [], "family": [], "position": [], "velocity": []}
        for k in range(n_pc):
            y = pr[:, k]
            yv = y.var()
            # eta^2: 1 - within-group variance / total, for each grouping
            for name, labels in (("context", d["cid"]), ("family", d["fam"])):
                within = 0.0
                for g in np.unique(labels):
                    m = labels == g
                    if m.sum() > 1:
                        within += m.sum() * y[m].var()
                r2[name].append(1 - within / (len(y) * yv + 1e-12))
            for name, X in (("position", d["pos"]), ("velocity", d["vel"])):
                A = np.column_stack([X, np.ones(len(X))])
                beta, *_ = np.linalg.lstsq(A, y, rcond=None)
                resid = y - A @ beta
                r2[name].append(1 - resid.var() / (yv + 1e-12))
        out[site] = {"var": var, **{k: np.array(v) for k, v in r2.items()}}
    return out


def _pc_variance_panels(axes, per_site, cols, n_pc, err=False):
    """The shared body of fig_pc_variance and pool_pc_variance: one panel per
    site, grouped bars of what each PC encodes (`per_site[site][name]` is a
    vector, or a seeds x PCs array when `err`, drawn as mean +- SD) and the
    scree curve on a twin axis. Each PC's bars sit under one bracket
    (bracket_xticks; the group is 0.8 wide, so neighbours stay 0.2 apart)."""
    x = np.arange(n_pc)
    nb = len(cols)
    w = 0.8 / nb
    for ai, (a, site) in enumerate(zip(axes, SITES)):
        v = per_site[site]
        for bi, (name, c) in enumerate(cols.items()):
            if name not in v:                       # an older npz without family
                continue
            R = np.asarray(v[name])[..., :n_pc]
            kw = (dict(yerr=R.std(0), capsize=1, error_kw={"lw": .6, "ecolor": INK})
                  if err else {})
            a.bar(x + (bi - (nb - 1) / 2) * w, R.mean(0) if err else R, width=w * 0.95,
                  color=c, label=name if ai == 0 else None, **kw)
        var = np.asarray(v["var"])[..., :n_pc]
        var = var.mean(0) if err else var
        a2 = a.twinx()
        a2.plot(x, var * 100, "k.--", ms=3, lw=.8, alpha=.6)
        a2.set_ylim(0, max(40, var[0] * 115))
        # the twin duplicates the host's left spine and, on all but the last
        # panel, would show an unlabelled right spine between panels: one
        # labelled scree axis on the right of the row is enough
        a2.spines["top"].set_visible(False)
        a2.spines["left"].set_visible(False)
        if ai == len(SITES) - 1:
            a2.set_ylabel("% of the site's variance (dashed)")
        else:
            a2.spines["right"].set_visible(False)
            a2.tick_params(right=False, labelright=False)
        a.set_xticks(x, [f"{i + 1}" for i in x])
        a.set_xlabel("principal component")
        a.set_title(site)
        a.set_ylim(0, 1)
        _despine(a)
    axes[0].set_ylabel("variance of that PC explained ($R^2$)")
    axes[0].legend(frameon=False)


# the four encodings' colours, shared by the per-seed and pooled figures
PC_VARIANCE_COLS = {"context": PALETTE["violet"], "family": "#c39bd3",
                    "position": PALETTE["teal"], "velocity": PALETTE["amber"]}


def fig_pc_variance(arm, seed, V, path, n_pc=10):
    """One panel per site: grouped bars of what each PC encodes. Built at
    double-column width; the methods prose goes to the caption file."""
    fig, axes = plt.subplots(1, len(SITES), figsize=figsize("double", height=2.2),
                             sharey=True, layout="constrained")
    _pc_variance_panels(axes, V, PC_VARIANCE_COLS, n_pc)
    caption = (
        f"{arm} seed {seed}: what each principal component encodes (tss >= {TSS_MIN}). "
        f"context is eta^2 over the {N_CONTEXTS} contexts, family eta^2 over the 2 "
        f"families (nested in context: family <= context, the gap is within-family "
        f"identity); position and velocity are linear R^2. They may overlap. "
        f"Dashed line: how much of the site's total variance that PC carries.")
    offset_spines(fig)
    bracket_xticks(fig)
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


def gif_pca(arm, seed, d, P, site, path, n_show=3000, frames=72, rng_seed=0):
    """Rotating 3D view of the top three PCs, coloured four ways at once.

    A static 2D projection hides whether an apparent cluster is real or an
    artefact of the viewing angle -- rotation is the cheapest way to tell.  The
    four panels share one rotation so the SAME points are compared: if the
    context panel looks structured only where the position panel also does, the
    structure is kinematic, not contextual.
    """
    from matplotlib.animation import PillowWriter
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(d["cid"]), size=min(n_show, len(d["cid"])),
                     replace=False)
    pr = P[site]["proj"][idx][:, :3]
    var = P[site]["var"]
    spd = np.linalg.norm(d["vel"][idx], axis=1)
    panels = [("context", d["cid"][idx], "tab20", None),
              ("family", d["fam"][idx], None, None),
              ("position (x)", d["pos"][idx][:, 0], "viridis", None),
              ("speed |v|", spd, "plasma",
               (np.percentile(spd, 1), np.percentile(spd, 99)))]
    fig = plt.figure(figsize=(4.25 * len(panels), 4.6))
    axes = [fig.add_subplot(1, len(panels), i + 1, projection="3d")
            for i in range(len(panels))]
    for a, (name, c, cmap, lim) in zip(axes, panels):
        if name == "family":
            for fi, f in enumerate(FAM_NAMES):
                m = c == fi
                a.scatter(pr[m, 0], pr[m, 1], pr[m, 2], s=2, alpha=.5,
                          color=FCOL[f], label=f)
            a.legend(frameon=False, fontsize=7, markerscale=4, loc="upper left")
        else:
            kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
            a.scatter(pr[:, 0], pr[:, 1], pr[:, 2], s=2, alpha=.5, c=c,
                      cmap=cmap, **kw)
        a.set_title(name, fontsize=10)
        a.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=8)
        a.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=8)
        a.set_zlabel(f"PC3 ({var[2]:.0%})", fontsize=8)
        a.tick_params(labelsize=6)
    fig.suptitle(f"{arm} seed {seed} -- {site}, top 3 PCs  "
                 f"({var[:3].sum():.0%} of variance, tss >= {TSS_MIN})",
                 fontsize=11)
    offset_spines(fig)
    fig.tight_layout(rect=(0, 0, 1, .93))
    w = PillowWriter(fps=12)
    with w.saving(fig, path, dpi=70):
        for ang in np.linspace(0, 360, frames, endpoint=False):
            for a in axes:
                a.view_init(elev=18, azim=ang)
            w.grab_frame()
    plt.close(fig)
    print(f"  saved -> {path}")


# ── 2b. is context a MAJOR or a MINOR axis? ──────────────────────────────────
def topk(d, P, ks=(1, 2, 3, 5, 8, 12, 20, 32, 48, 64), n_splits=3):
    """16-way decoding from the top-k PCs only, at each site.

    THE QUANTITATIVE VERSION of "major or minor axis".  Eyeballing a PCA cannot
    settle it: a scatter can look unstructured while the information sits in a
    handful of low-variance directions.  If the curve climbs immediately,
    context lives in the high-variance directions; if it stays flat and climbs
    late, context is a minor axis carried by low-variance directions -- mixed
    selectivity, and the reason to project before doing geometry.
    """
    out = {}
    for site in SITES:
        pr, var = P[site]["proj"], P[site]["var"]
        full = P[site].get("full")
        acc = []
        for k in ks:
            if k > pr.shape[1]:
                acc.append(np.nan); continue
            a = [_clf(sp).fit(pr[~te][:, :k], d["cid"][~te]).score(
                     pr[te][:, :k], d["cid"][te])
                 for sp in range(n_splits)
                 for te in [seq_split(d["sid"], seed=sp)]]
            acc.append(float(np.mean(a)))
        out[site] = {"ks": np.array(ks, float), "acc": np.array(acc, float),
                     "cumvar": np.array([var[:k].sum() for k in ks])}
        print(f"    topk: {site} done", flush=True)
    return out


def fig_topk(arm, seed, T, full_acc, path):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for site in SITES:
        t = T[site]
        ax[0].plot(t["ks"], t["acc"], "o-", ms=4, label=site)
        ax[1].plot(t["cumvar"], t["acc"], "o-", ms=4, label=site)
    for a, xl, xs in ((ax[0], "number of top PCs used", "log"),
                      (ax[1], "cumulative variance those PCs explain", "linear")):
        a.axhline(1 / N_CONTEXTS, ls=":", color="#888", lw=1)
        a.set_xlabel(xl); a.set_xscale(xs)
        a.set_ylabel(f"{N_CONTEXTS}-way context decoding")
        a.set_ylim(0, 1.02)
        a.legend(frameon=False, fontsize=8)
        _despine(a)
    if full_acc is not None:
        ax[0].axhline(full_acc, ls="--", color="#c25b6b", lw=1)
        ax[0].text(t["ks"][-1], full_acc + .02, f"all dims ({full_acc:.2f})",
                   fontsize=8, color="#c25b6b", ha="right")
    fig.suptitle(
        f"{arm} seed {seed} -- is context a MAJOR or a MINOR axis?\n"
        f"left: {N_CONTEXTS}-way decoding from the top-k PCs only.\n"
        f"right: the same, against how much variance those PCs explain.\n"
        f"context needing PCs that carry little variance = a minor axis "
        f"(mixed selectivity)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .84))
    offset_spines(fig)
    save(fig, path)
    print(f"  saved -> {path}")


def fig_clusters(arm, seed, d, P, site, path, n_show=4000, rng_seed=0):
    """The gif's content, in a form that can go in a paper.

    A gif conveys cluster structure by rotating; a figure cannot.  Showing all
    THREE pairwise projections of the top 3 PCs is the static equivalent and is
    strictly more honest -- there is no viewing angle to choose, and a cluster
    that only appears in one projection is visibly only in one projection.

    Rows are colourings, columns are PC pairs, so reading down a column asks
    "is this structure contextual or kinematic?" on identical axes.
    """
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(d["cid"]), size=min(n_show, len(d["cid"])),
                     replace=False)
    pr = P[site]["proj"][idx][:, :3]
    var = P[site]["var"]
    spd = np.linalg.norm(d["vel"][idx], axis=1)
    rows = [("context", d["cid"][idx], "tab20", None),
            ("family", d["fam"][idx], None, None),
            ("position (x)", d["pos"][idx][:, 0], "viridis", None),
            ("speed |v|", spd, "plasma",
             (np.percentile(spd, 1), np.percentile(spd, 99)))]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 4.1 * len(rows)))
    for ri, (name, c, cmap, lim) in enumerate(rows):
        for ci, (i, j) in enumerate(pairs):
            a = axes[ri, ci]
            if name == "family":
                for fi, f in enumerate(FAM_NAMES):
                    m = c == fi
                    a.scatter(pr[m, i], pr[m, j], s=2.5, alpha=.4,
                              color=FCOL[f], label=f if (ri, ci) == (1, 0) else None)
                if (ri, ci) == (1, 0):
                    a.legend(frameon=False, fontsize=7, markerscale=5)
            else:
                kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
                a.scatter(pr[:, i], pr[:, j], s=2.5, alpha=.4, c=c, cmap=cmap,
                          **kw)
            a.set_xlabel(f"PC{i+1} ({var[i]:.0%})", fontsize=9)
            a.set_ylabel(f"PC{j+1} ({var[j]:.0%})" if ci == 0 else "",
                         fontsize=9)
            if ri == 0:
                a.set_title(f"PC{i+1} vs PC{j+1}", fontsize=11)
            if ci == 0:
                a.text(-.28, .5, name, transform=a.transAxes, rotation=90,
                       va="center", ha="center", fontsize=12, weight="bold")
            _despine(a)
    fig.suptitle(
        f"{arm} seed {seed} -- cluster structure in {site}, top 3 PCs "
        f"({var[:3].sum():.0%} of variance, tss >= {TSS_MIN})\n"
        f"all three pairwise projections, so no viewing angle is chosen for "
        f"you.  Read DOWN a column: is the structure contextual, or kinematic?",
        fontsize=11.5)
    offset_spines(fig)
    fig.tight_layout(rect=(0, 0, 1, .955))
    save(fig, path)
    print(f"  saved -> {path}")


def fig_paper_pca_z(arm, seeds, d, P, path, n_show=4000, rng_seed=0,
                    action_mode=M.DEFAULT_ACTION_MODE):
    """PAPER FIGURE.  What z encodes, and how consistent it is across seeds.

    Top row: PC1 vs PC2 of z for ONE seed, coloured by context, position and
    speed.  Scatters cannot be averaged across seeds -- each seed has its own PC
    basis and its own sign conventions -- so a single seed is shown and the
    across-seed claim is carried by the bottom row instead.

    Bottom row: for each of the top 10 PCs, how much of ITS variance is
    explained by context (eta^2 over the contexts), by position and by
    velocity (linear R^2), as mean +- SD ACROSS SEEDS.  This is the quantitative
    statement and it is what should be quoted.

    Caveat kept in the caption: context is measured assumption-free while
    position and velocity are linear fits, so a nonlinear kinematic dependence
    would be understated.  The same linear measure reaches R^2 ~ 1.0 in CB's
    layers, which shows it detects kinematics perfectly well when they are
    there.
    """
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(d["cid"]), size=min(n_show, len(d["cid"])),
                     replace=False)
    pr = P["z"]["proj"][idx]
    var = P["z"]["var"]
    spd = np.linalg.norm(d["vel"][idx], axis=1)
    # pool the per-PC variance decomposition over whatever seeds have it
    V = {k: [] for k in ("context", "family", "position", "velocity")}
    got = []
    for sd_ in seeds:
        f = os.path.join(M.fig_dir(arm, sd_, action_mode=action_mode),
                         "pc_variance.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for k in V:
            if f"{k}|z" in z.files:             # family is absent from older npz
                V[k].append(z[f"{k}|z"])
        got.append(sd_)
    V = {k: v for k, v in V.items() if v}
    # Built at double-column width, type from the style, methods prose in the
    # caption file rather than a suptitle (plotting_utils follow-ups 1, 2, 5).
    fig = plt.figure(figsize=figsize("double", height=4.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, .8])
    panels = [("context", d["cid"][idx], "tab20", None),
              ("position (x)", d["pos"][idx][:, 0], "paper_seq", None),
              ("speed |v|", spd, "paper_seq",
               (np.percentile(spd, 1), np.percentile(spd, 99)))]
    top = []
    for i, (name, c, cmap, lim) in enumerate(panels):
        # one PC basis, so the three scatters share both axes and only the
        # first carries the y label
        a = fig.add_subplot(gs[0, i], sharex=top[0] if top else None,
                            sharey=top[0] if top else None)
        kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
        sc = a.scatter(pr[:, 0], pr[:, 1], s=2, alpha=.45, c=c, cmap=cmap, lw=0, **kw)
        a.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
        if i == 0:
            a.set_ylabel(f"PC2 ({var[1]:.0%})")
        else:
            a.tick_params(labelleft=False)
        a.set_title(name)
        if name != "context":
            colorbar(sc, ax=a)
        top.append(a)
    ax = fig.add_subplot(gs[1, :])
    if got:
        n_pc = len(V["context"][0])
        x = np.arange(n_pc)
        cols = {"context": PALETTE["violet"], "position": PALETTE["teal"],
                "velocity": PALETTE["amber"]}
        for bi, (name, col) in enumerate(cols.items()):
            A = np.array(V[name])
            ax.bar(x + (bi - 1) * .27, A.mean(0), yerr=A.std(0), width=.26,
                   color=col, label=name, error_kw={"lw": .6, "ecolor": INK})
        ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_xlabel("principal component of $z$")
        ax.set_ylabel("variance explained ($R^2$)")
        ax.set_ylim(0, 1)
        ax.legend(ncol=3, loc="upper right")
        ax.set_title(f"what each PC encodes (mean $\\pm$ SD, {len(got)} seeds)")
    caption = (
        f"{arm}: z encodes context in its dominant components (tss >= {TSS_MIN}; "
        f"scatter from seed {seeds[0]}; pooled seeds {got}). Top: PC1 vs PC2 of z "
        f"for one seed, coloured by context, x position and speed. Bottom: for each "
        f"of the top PCs, the share of its variance explained by context (eta^2 "
        f"over the {N_CONTEXTS} contexts, assumption-free), position and velocity "
        f"(linear R^2), mean +- SD across seeds. The shares need not sum to 1: "
        f"drift contexts move the agent, so context and position are correlated.")
    offset_spines(fig)
    if got:
        bracket_xticks(ax, half_width=0.4)        # three .26 bars at .27 spacing: 0.8 wide
    label_panels(fig, top + [ax])
    save(fig, path, caption=caption)
    print(f"  saved -> {path}")


# ── 3. MDS geometry ──────────────────────────────────────────────────────────
def _mds(dist, seed=0):
    from sklearn.manifold import MDS
    return MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
               n_init=8, normalized_stress="auto").fit_transform(dist)


def mds_weights(d, site="z", n_splits=N_SPLITS, rank=None):
    """Geometry of the decoder's READOUT DIRECTIONS.

    Fit the N_CONTEXTS-way decoder, take its weight matrix W, low-rank
    truncate it, normalise the rows and use 1 - cosine as a distance.  Each
    point is then one context, positioned by how similar its detection RECIPE is
    to every other context's.

    The rank truncation (default N/2, following the earlier figure) denoises W:
    N_CONTEXTS contexts cannot need more than a handful of directions to
    separate, and
    the trailing singular values are fit noise that inflates every pairwise
    distance.  It does NOT depend on whether context is a major or minor axis of
    the variance -- the decoder finds its directions by supervised fitting
    either way -- so this view is valid at every site.
    """
    rank = rank or N_CONTEXTS // 2
    W = []
    for sp in range(n_splits):
        te = seq_split(d["sid"], seed=sp)
        W.append(_clf(sp).fit(d[site][~te], d["cid"][~te])[-1].coef_)
    W = np.mean(W, 0)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    Wt = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank]
    Wn = Wt / (np.linalg.norm(Wt, axis=1, keepdims=True) + 1e-9)
    dp = Wn @ Wn.T
    return _mds(1 - dp), dp


def context_centroids(d, site="z", residualise=False, center=True,
                      context_ids=None):
    """The per-context mean of `site`, as a (n_contexts, n_features) array.

    Centroids, not all samples -- any geometry over ~28k points would be
    dominated by within-context spread and is not what the question asks.

    residualise=True first regresses position and velocity out of the site,
    which matters because drift contexts systematically push the agent around:
    their centroids differ partly because the agent IS somewhere else, not only
    because the context differs.  Comparing the two tells you whether a
    separation is contextual or kinematic.

    BEWARE: residualise=True ALSO CENTRES, whether or not `center` is set.  Its
    design matrix carries an intercept column, so the residual has zero mean by
    construction and the contexts' shared component goes with it.  Measured on
    main seed 0, mean off-diagonal cosine of the centroids:

        residualise=0 center=0   +0.645        <- the only uncentered combination
        residualise=0 center=1   -0.142
        residualise=1 center=0   -0.142        <- centered despite center=False
        residualise=1 center=1   -0.142

    So a caller that wants the geometry WITH its common component -- which is
    what a downstream reader receives -- must pass residualise=False as well as
    center=False.  Asking for one and not the other silently returns the
    centered geometry under the uncentered name.

    center=True subtracts the mean OVER CONTEXTS.  It is off for the MDS
    callers, which use distances and so are already translation-invariant, and
    on for anything angular -- see centroid_cosine.

    context_ids restricts to a SUBSET of the contexts, and the restriction
    happens BEFORE centering, which is the only order that means anything: an
    agent that meets four of the eight contexts has the mean of those four as
    its shared component, not the mean of all eight.
    """
    ids = range(N_CONTEXTS) if context_ids is None else list(context_ids)
    X = d[site]
    if residualise:
        K = np.column_stack([d["pos"], d["vel"], np.ones(len(X))])
        beta, *_ = np.linalg.lstsq(K, X, rcond=None)
        X = X - K @ beta
    cen = np.stack([X[d["cid"] == c].mean(0) for c in ids])
    return cen - cen.mean(0) if center else cen


def centroid_cosine(d, site="z", residualise=True, center=True,
                    context_ids=None):
    """Pairwise cosine between the context centroids.

    Returns (S, eigenvalues of S).  This is the representational geometry in
    the form sac.make_oracle_embed consumes: a Gram matrix it draws a context
    code from, so that the RL oracle's code has the circuit's geometry instead
    of a near-orthogonal one.  Angular rather than the Euclidean distances
    mds_centroids uses, because a cosine matrix is scale-free -- it says where
    the contexts sit RELATIVE to each other and carries no unit that would have
    to be matched against another code's.

    CENTRING IS THE ONE CHOICE HERE THAT CHANGES THE ANSWER, and which way to
    take it depends on what the caller is going to do with the matrix.  The
    centroids share a large common component (z is a tanh readout and its mean
    is nowhere near the origin), so the RAW cosines are all strongly positive
    -- main seed 0 measures off-diagonal mean +0.645, min +0.289, with a single
    eigenvalue carrying 5.53 of the trace of 8.

      center=False  the geometry a DOWNSTREAM READER actually receives, common
                    mean included.  That mean is not inert to learning even
                    though it carries no context information: it is shared by
                    every context's input and therefore by every context's
                    gradient, which is the gradient-sharing sac.center_ec
                    exists to test.  This is what an RL oracle should match.
      center=True   the DISCRIMINATIVE geometry alone, with the shared offset
                    a first-layer bias absorbs removed.  This is the one to
                    read when asking what distinguishes the contexts.

    NEUTRAL IS NOT ZERO once centered, for the same arithmetic as the decoder
    weight cosines in confusion(): K centered vectors sum to zero, so their mean
    pairwise INNER PRODUCT is exactly -1/(K-1) of their mean squared norm, and
    the mean COSINE lands there too up to the spread in those norms -- exactly
    only if the centroids are equally long.  At K = 8 that is -0.143 against a
    measured -0.142, and at the K = 4 sac uses, -0.3333 against -0.3329.  Those
    agreements are near-arithmetic, not findings; only departures from them
    mean anything.

    WHAT THE CENTRED MATRIX MEASURES, on the trained circuit (main, seeds 0
    and 1 agree):

        rot -90 vs rot +90   -0.91      refl 0  vs refl 90    -0.94
        rot -90 vs rot -45   +0.60      refl 45 vs refl 135   -0.95
        rot -45 vs rot +45   -0.19      any rotation vs any reflection  ~0

    which is the geometry of the ACTION TRANSFORMS THEMSELVES.  Under the
    Frobenius inner product <R(a), R(b)> = 2 cos(a - b) and <M(a), M(b)> =
    2 cos(2(a - b)), while <R(a), M(b)> = 0 for every pair -- so rotations lie
    on one circle, reflections on another at double angle, and the two planes
    are orthogonal.  Every number above is that prediction to within ~0.1.

    Hence the SPECTRUM: 3.21, 1.96, 1.85, 0.96, then 0.013, 0.003, 0.001,
    0.000.  Four directions hold 99.8% of the trace and the gap to the fifth is
    74x, so the eight contexts occupy a 4-DIMENSIONAL code -- exactly the
    dimension of the 2x2 matrices they are.  The circuit has not learned eight
    labels; it has learned the space they live in.
    """
    cen = context_centroids(d, site, residualise=residualise, center=center,
                            context_ids=context_ids)
    n = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12)
    S = n @ n.T
    S = 0.5 * (S + S.T)                      # exactly symmetric, for eigh
    return S, np.sort(np.linalg.eigvalsh(S))[::-1]


def mds_centroids(d, site="z", residualise=False):
    """MDS on the Euclidean distances between context centroids.

    Distances, so no centering is needed -- see context_centroids.
    """
    cen = context_centroids(d, site, residualise=residualise, center=False)
    D = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    return _mds(D / (D.max() + 1e-12)), D


def fig_mds(arm, seed, panels, path, site="z"):
    """One panel per geometry, contexts coloured by family."""
    fig, ax = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.2))
    ax = np.atleast_1d(ax)
    for i, (name, coords, note) in enumerate(panels):
        for c in range(N_CONTEXTS):
            ax[i].scatter(coords[c, 0], coords[c, 1], s=170,
                          color=FCOL[FAM_NAMES[FAMILY[c]]],
                          edgecolor="k", lw=.7, zorder=3)
            ax[i].annotate(C_LABEL[c],
                           (coords[c, 0], coords[c, 1]),
                           textcoords="offset points", xytext=(8, 6),
                           fontsize=7, color="#333")
        ax[i].set_xlabel("MDS dim 1"); ax[i].set_ylabel("MDS dim 2")
        ax[i].set_title(f"{name}\n{note}", fontsize=10)
        ax[i].set_aspect("equal", adjustable="datalim")
        ax[i].grid(alpha=.3)
        _despine(ax[i])
    fig.suptitle(f"{arm} seed {seed} -- context geometry in {site}  "
                 f"(tss >= {TSS_MIN});  colour = family", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    offset_spines(fig)
    save(fig, path)
    print(f"  saved -> {path}")


def pool_pc_variance(arm, seeds, tag="", n_pc=10,
                     action_mode=M.DEFAULT_ACTION_MODE):
    """The per-PC decomposition at ALL FOUR sites, pooled across seeds.

    The paper figure pools this too but only for z.  This is the version that
    answers "is MTL context-dominated as well?", because the claim is about the
    ORDERING of the sites and needs all four side by side with error bars.

    Bars are mean +- SD across seeds; the dashed line is the scree curve -- how
    much of that site's TOTAL variance each PC carries.  The bars say what a
    component encodes, the dashed line says how much it matters.
    """
    got, A = [], {s_: {k: [] for k in ("context", "family", "position", "velocity",
                                       "var")} for s_ in SITES}
    for sd_ in seeds:
        f = os.path.join(M.fig_dir(arm, sd_, action_mode=action_mode),
                         f"pc_variance{tag}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for s_ in SITES:
            for k in A[s_]:
                if f"{k}|{s_}" in z.files:      # family is absent from older npz
                    A[s_][k].append(z[f"{k}|{s_}"])
        got.append(sd_)
    if not got:
        print(f"  {arm}: no pc_variance to pool"); return
    cols = {k: c for k, c in PC_VARIANCE_COLS.items() if A["z"][k]}
    fig, axes = plt.subplots(1, len(SITES), figsize=figsize("double", height=2.2),
                             sharey=True, layout="constrained")
    _pc_variance_panels(axes, A, cols, n_pc, err=True)
    caption = (
        f"{arm}: what each principal component encodes, mean +- SD across {len(got)} "
        f"seeds {got} (tss >= {TSS_MIN}). context is eta^2 over the {N_CONTEXTS} contexts "
        f"(assumption-free); position and velocity are linear R^2. Dashed line: how "
        f"much of the site's total variance that PC carries. Only z leads with context.")
    offset_spines(fig)
    bracket_xticks(fig)
    d = os.path.join(M.FIG_DIR, arm, "across_seeds")
    os.makedirs(d, exist_ok=True)
    p_ = os.path.join(d, f"pc_variance{tag}.png")
    save(fig, p_, caption=caption)
    print(f"  saved -> {p_}   (pooled over {len(got)} seeds)")


def pool_topk(arm, seeds, tag="",
              action_mode=M.DEFAULT_ACTION_MODE):
    """Pool the top-k curves.  These ARE comparable across seeds -- k and
    accuracy mean the same thing in every network -- so this is the version to
    quote, and the per-seed figure becomes a check rather than a result."""
    got, A = [], {site: [] for site in SITES}
    ks = cum = None
    for sd_ in seeds:
        f = os.path.join(M.fig_dir(arm, sd_, action_mode=action_mode),
                         f"topk{tag}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for site in SITES:
            A[site].append(z[f"acc|{site}"])
        ks = z["ks|z"]
        cum = {site: z[f"cumvar|{site}"] for site in SITES}
        got.append(sd_)
    if not got:
        print(f"  {arm}: no topk to pool"); return
    d = os.path.join(M.FIG_DIR, arm, "across_seeds")
    os.makedirs(d, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for site in SITES:
        R = np.array(A[site], float)
        m, sd = R.mean(0), R.std(0)
        for i, xs in enumerate((ks, cum[site])):
            ax[i].plot(xs, m, "o-", ms=4, label=site if i == 0 else None)
            ax[i].fill_between(xs, m - sd, m + sd, alpha=.18, lw=0)
    for a_, xl, xs_ in ((ax[0], "number of top PCs used", "log"),
                        (ax[1], "cumulative variance those PCs explain",
                         "linear")):
        a_.axhline(1 / N_CONTEXTS, ls=":", color="#888", lw=1)
        a_.set_xlabel(xl); a_.set_xscale(xs_)
        a_.set_ylabel(f"{N_CONTEXTS}-way context decoding"); a_.set_ylim(0, 1.02)
        _despine(a_)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{arm} -- is context a MAJOR or a MINOR axis?  "
                 f"mean $\\pm$1 SD across {len(got)} seeds {got}\n"
                 f"context needing PCs that carry little variance = a minor "
                 f"axis (mixed selectivity)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .87))
    offset_spines(fig)
    p = os.path.join(d, f"topk{tag}.png")
    save(fig, p)
    print(f"  saved -> {p}   (pooled over {len(got)} seeds)")


def pool_mds(arm, seeds, site="z", tag="",
             action_mode=M.DEFAULT_ACTION_MODE):
    """Pool MDS by averaging the DISTANCE MATRICES, then embedding once.

    MDS coordinates cannot be averaged across seeds: each embedding is defined
    only up to rotation, reflection and scale, so averaging raw coordinates
    would blur an identical geometry into mush.  The distance matrices carry no
    such freedom, so averaging those and embedding the average is the correct
    pooling.
    """
    dps, Ds, Drs, got = [], [], [], []
    for sd_ in seeds:
        f = os.path.join(M.fig_dir(arm, sd_, action_mode=action_mode),
                         f"mds_{site}{tag}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if "D" not in z.files:
            continue
        dps.append(z["dp"]); Ds.append(z["D"]); Drs.append(z["D_resid"])
        got.append(sd_)
    if not got:
        print(f"  {arm}: no mds to pool (regenerate -- older npz lacks D)")
        return
    dp = np.mean(dps, 0)
    Dc, Dr = np.mean(Ds, 0), np.mean(Drs, 0)
    cw = _mds(1 - dp)
    cc = _mds(Dc / (Dc.max() + 1e-12))
    cr = _mds(Dr / (Dr.max() + 1e-12))
    d = os.path.join(M.FIG_DIR, arm, "across_seeds")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"mds_{site}{tag}.npz"), weights=cw, centroids=cc,
             centroids_resid=cr, dp=dp, D=Dc, D_resid=Dr,
             seeds=np.array(got))
    fig_mds(arm, f"pooled over {len(got)} seeds", [
        ("MDS on decoder WEIGHTS", cw,
         f"1 - cosine, rank {N_CONTEXTS//2} truncated"),
        ("MDS on CENTROIDS", cc, f"distance between mean {site} per context"),
        ("MDS on CENTROIDS, kinematics removed", cr,
         "position and velocity regressed out first")],
        os.path.join(d, f"mds_{site}{tag}.png"), site)


def pool_confusion(arm, seeds, site="z", tag="",
                   action_mode=M.DEFAULT_ACTION_MODE):
    """Average the row-normalised confusion and the weight-cosine across seeds.

    Both are already normalised per seed, so a plain mean is the right pooling
    -- no seed dominates by having more test rows.  Missing seeds are skipped
    and the count actually used is stamped into the figure.
    """
    Cs, Ss, got = [], [], []
    for s in seeds:
        p = os.path.join(M.fig_dir(arm, s, action_mode=action_mode),
                         f"confusion_{site}{tag}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        Cs.append(z["C"]); Ss.append(z["S"]); got.append(s)
    if not Cs:
        print(f"  {arm}: nothing to pool"); return
    C, S = np.mean(Cs, 0), np.mean(Ss, 0)
    d = os.path.join(M.FIG_DIR, arm, "across_seeds")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"confusion_{site}{tag}.npz"),
             C=C, S=S, order=np.array(ORDER), labels=np.array(LABELS),
             seeds=np.array(got))
    acc, wf = fig_confusion(arm, f"pooled over {len(got)} seeds", C, S,
                            os.path.join(d, f"confusion_{site}{tag}.png"), site)
    print(f"  {arm}: pooled over {len(got)} seeds {got} -- "
          f"accuracy {acc:.3f}; {wf['family']:.0%} of errors stay in family "
          f"(chance {wf['chance']:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arms", nargs="+", default=list(M.ARMS), choices=list(M.ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mode", default="intact", choices=["intact", "zero", "shuffle"])
    ap.add_argument("--site", default="z", choices=SITES)
    # --action-mode / --readout / --rate-beta / --noise-sigma: which supervised
    # circuit to analyse. The checkpoint, the rollout distribution and the
    # figure directory (figures/<arm_tag>/) all follow the one spec.
    M.add_circuit_args(ap)
    ap.add_argument("--confusion", action="store_true")
    ap.add_argument("--pca", action="store_true",
                    help="PCA at all four sites, coloured by context, family, "
                         "position and speed")
    ap.add_argument("--paper", action="store_true",
                    help="the paper figure: z's PCA coloured by context, "
                         "position and speed, plus the per-PC decomposition "
                         "pooled across seeds")
    ap.add_argument("--mds", action="store_true",
                    help="MDS geometry: decoder weights, and z centroids with "
                         "and without kinematics regressed out")
    ap.add_argument("--gif", action="store_true",
                    help="rotating 3D gif of the top 3 PCs at each site")
    ap.add_argument("--topk", action="store_true",
                    help="context decoding from the top-k PCs -- major or minor "
                         "axis")
    ap.add_argument("--pool", action="store_true",
                    help="also write the across-seed average")
    a = ap.parse_args()

    use_paper_style()

    if not (a.confusion or a.pca or a.topk or a.gif or a.mds or a.paper):
        ap.error("nothing to do: pass --confusion, --pca, --topk, --mds, --paper and/or --gif")
    # the spec travels in the action_mode slot everywhere below (see model's
    # CircuitSpec section), so the noisy circuits need no further plumbing
    a.action_mode = M.circuit_from_args(a)
    for arm in a.arms:
        # the label every figure carries: "main" or "main_gp"
        atag = M.arm_tag(arm, a.action_mode)
        if a.paper:
            d0 = collect(arm, a.seeds[0], a.mode, action_mode=a.action_mode)
            if d0 is not None:
                P0 = pca_sites(d0)
                fig_paper_pca_z(atag, a.seeds, d0, P0,
                                os.path.join(M.FIG_DIR, atag,
                                             f"paper_pca_z{'' if a.mode=='intact' else '_'+a.mode}.png"),
                                action_mode=a.action_mode)
        for s in a.seeds:
            d = collect(arm, s, a.mode, action_mode=a.action_mode)
            if d is None:
                print(f"  no checkpoint for {arm} seed {s} ({a.action_mode})")
                continue
            print(f"  [{atag} seed {s}] {d['cid'].size} settled steps", flush=True)
            out = M.fig_dir(arm, s, action_mode=a.action_mode)
            tag = "" if a.mode == "intact" else f"_{a.mode}"
            if a.confusion:
                C, S = confusion(d, a.site)
                np.savez(os.path.join(out, f"confusion_{a.site}{tag}.npz"),
                         C=C, S=S, order=np.array(ORDER),
                         labels=np.array(LABELS))
                fig_confusion(atag, s, C, S,
                              os.path.join(out, f"confusion_{a.site}{tag}.png"),
                              a.site)
            if a.mds:
                cw, dp = mds_weights(d, a.site)
                cc, _ = mds_centroids(d, a.site)
                cr, _ = mds_centroids(d, a.site, residualise=True)
                _, Dc = mds_centroids(d, a.site)
                _, Dr = mds_centroids(d, a.site, residualise=True)
                np.savez(os.path.join(out, f"mds_{a.site}{tag}.npz"),
                         weights=cw, centroids=cc, centroids_resid=cr,
                         dp=dp, D=Dc, D_resid=Dr)
                fig_mds(atag, s, [
                    ("MDS on decoder WEIGHTS",   cw,
                     f"1 - cosine, rank {N_CONTEXTS//2} truncated"),
                    ("MDS on CENTROIDS",         cc,
                     f"distance between mean {a.site} per context"),
                    ("MDS on CENTROIDS, kinematics removed", cr,
                     "position and velocity regressed out first")],
                    os.path.join(out, f"mds_{a.site}{tag}.png"), a.site)
            if a.pca or a.topk or a.gif:
                P = pca_sites(d)
                np.savez(os.path.join(out, f"pca{tag}.npz"),
                         **{f"var|{k}": P[k]["var"] for k in SITES})
            if a.pca:
                fig_pca(atag, s, d, P, os.path.join(out, f"pca{tag}.png"))
                for site in SITES:
                    fig_clusters(atag, s, d, P, site,
                                 os.path.join(out, f"clusters_{site}{tag}.png"))
                V = pc_variance(d, P)
                np.savez(os.path.join(out, f"pc_variance{tag}.npz"),
                         **{f"{w}|{k}": V[k][w] for k in SITES
                            for w in ("var", "context", "family", "position", "velocity")})
                fig_pc_variance(atag, s, V,
                                os.path.join(out, f"pc_variance{tag}.png"))
            if a.gif:
                for site in SITES:
                    gif_pca(atag, s, d, P, site,
                            os.path.join(out, f"pca_{site}{tag}.gif"))
            if a.topk:
                T = topk(d, P)
                np.savez(os.path.join(out, f"topk{tag}.npz"),
                         **{f"{w}|{k}": T[k][w] for k in SITES
                            for w in ("ks", "acc", "cumvar")})
                te = seq_split(d["sid"], seed=0)
                full = float(_clf(0).fit(d["z"][~te], d["cid"][~te])
                             .score(d["z"][te], d["cid"][te]))
                fig_topk(atag, s, T, full, os.path.join(out, f"topk{tag}.png"))
        if a.pool:
            tg = "" if a.mode == "intact" else f"_{a.mode}"
            if a.confusion:
                pool_confusion(arm, a.seeds, a.site, tg,
                               action_mode=a.action_mode)
            if a.pca:
                pool_pc_variance(arm, a.seeds, tg,
                                 action_mode=a.action_mode)
            if a.topk:
                pool_topk(arm, a.seeds, tg, action_mode=a.action_mode)
            if a.mds:
                pool_mds(arm, a.seeds, a.site, tg,
                         action_mode=a.action_mode)
