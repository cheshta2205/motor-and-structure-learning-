"""conjunctivity.py -- is the context code the SAME EVERYWHERE, or does it move
with what the agent is doing?

An ADDITIVE (abstract) code puts context along one fixed direction whatever the
state and action.  A CONJUNCTIVE code encodes (state, action, context) jointly,
so the direction separating two contexts rotates as you move.  The distinction
is not cosmetic downstream: MC's first layer is a FIXED linear map, fit on
whatever states one block visited and then applied in the next block's states.
A code that is only locally readable is one MC cannot key a memory on.

WHY A GENERALISATION KERNEL rather than a single number.  Train a context
decoder in one bin of (action direction | position) and test it in another,
then plot accuracy against the DISTANCE between bins.  Additive gives a flat
kernel; conjunctive gives a decaying one; and the WIDTH of the decay says over
what range the code is locally additive, which no scalar summary carries.

THE CONJUNCTIVE CASE HAS AN EXACT REFERENCE CURVE, which is what makes this
quantitative rather than a shape-eyeballing exercise.  Suppose z encoded the
EXPECTED FORCE -- maximally conjunctive.  Then the vector separating contexts
c and c' at action angle phi is proportional to (A_c - A_c') a(phi), and for a
SAME-FAMILY pair that matrix difference is a scaled rotation (rotations:
R(t1) - R(t2) = 2 sin(dt/2) R(mid + 90deg)) or a scaled reflection.  Either way
its angle moves one-for-one with phi, so

    parallelism(dphi) = cos(dphi)          fully conjunctive
    parallelism(dphi) = 1                  fully abstract

Verified against the context table to sd 0.000 across absolute action angle for
all four same-family pairs.  MIXED-FAMILY PAIRS HAVE NO SUCH CURVE -- their
difference is a general 2x2, so the shape depends on the absolute angle, not
the difference (measured sd 1.000 across phi0) -- so the kernel fit is
restricted to same-family pairs and mixed ones are reported separately.

THE SIGNATURE WORTH MORE THAN THE REST OF THE FIGURE.  The conjunctive
reference goes NEGATIVE past 90 degrees, so a decoder trained at phi and tested
at phi + 180 should be SYSTEMATICALLY WRONG -- below chance.  Noise can only
degrade a decoder to chance; nothing stochastic pushes it past.  Below-chance
transfer therefore establishes rotation in a way no noise account can mimic,
and its absence says the code is degraded rather than rotated.  Accuracy is
kept SIGNED for that reason; folding it would destroy the one measure that
cannot be faked.

POSITION IS THE NEGATIVE CONTROL.  B = 0 for every context and every A is
norm-preserving, so position enters the force law not at all -- it reaches the
dynamics only through the walls.  There is no principled conjunctive reference
for it and none is drawn: a flat kernel is the prediction, and a decaying one
means the code is carrying something it has no business carrying.

NULLS ARE NOT OPTIONAL.  Shuffled bins gave 11.1 degrees of apparent rotation
and a 1.02x conditioning gain from sampling noise alone in an earlier version
of this analysis.  Note what the shuffled control does and does not do: it
destroys the bin structure, so its kernel is FLAT AT THE CEILING -- which
coincides with the abstract prediction.  It validates the estimator; it does
not discriminate the hypotheses.  The label permutation is what gives the floor.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model as M
import task as T
from plotting_utils import make_axis_nice, set_plt_params
from representation_analysis import SITES, _despine

N_DIR_BINS = 8            # action-direction bins: distances are 0,45,...,180
N_POS_BINS = 3            # per axis, so a 3x3 grid
MIN_CELL = 80             # rows per (pair, bin) below which a cell is dropped
TEST_FRAC = 0.3


# ── binning ──────────────────────────────────────────────────────────────

def bin_action_direction(d, n_bins=N_DIR_BINS):
    """Angular bins of the action, with the CIRCULAR distance between centres.

    The action is the variable the contexts transform, so this is where an
    interaction is theoretically expected -- and the only conditioning variable
    with an analytic conjunctive reference.
    """
    ang = np.arctan2(d["act"][:, 1], d["act"][:, 0])
    idx = np.clip(((ang + np.pi) / (2 * np.pi) * n_bins).astype(int), 0, n_bins - 1)
    centres = (np.arange(n_bins) + 0.5) * (2 * np.pi / n_bins) - np.pi
    diff = np.abs(np.subtract.outer(centres, centres))
    dist = np.degrees(np.minimum(diff, 2 * np.pi - diff))     # circular
    edges = np.degrees(np.arange(n_bins + 1) * (2 * np.pi / n_bins) - np.pi)
    labels = [f"{edges[i]:.0f}:{edges[i+1]:.0f}" for i in range(n_bins)]
    return idx, dist, labels, "action direction (deg)"


def bin_position(d, n_bins=N_POS_BINS):
    """A grid on position, with the Euclidean distance between bin centres.

    QUANTILE edges, not equal-width: sequences start in the middle half, so
    equal-width bins would leave the outer ring nearly empty and the distance
    axis would be carried by a handful of rows.
    """
    pos = d["pos"]
    edges = [np.quantile(pos[:, k], np.linspace(0, 1, n_bins + 1)) for k in (0, 1)]
    ix = [np.clip(np.searchsorted(edges[k][1:-1], pos[:, k], side="right"),
                  0, n_bins - 1) for k in (0, 1)]
    idx = ix[0] * n_bins + ix[1]
    cx = [0.5 * (edges[k][:-1] + edges[k][1:]) for k in (0, 1)]
    cen = np.array([[cx[0][i], cx[1][j]] for i in range(n_bins) for j in range(n_bins)])
    dist = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=-1)
    labels = [f"x[{edges[0][i]:+.2f},{edges[0][i+1]:+.2f}]\n"
              f"y[{edges[1][j]:+.2f},{edges[1][j+1]:+.2f}]"
              for i in range(n_bins) for j in range(n_bins)]
    return idx, dist, labels, "position (grid units)"


BINNERS = {"action_dir": bin_action_direction, "position": bin_position}


# ── the analytic references ──────────────────────────────────────────────

def same_family_pairs():
    fam = {c: T.FAMILY[c] for c in range(T.N_CONTEXTS)}
    return [(a, b) for a in range(T.N_CONTEXTS) for b in range(a + 1, T.N_CONTEXTS)
            if fam[a] == fam[b]]


def mixed_family_pairs():
    fam = {c: T.FAMILY[c] for c in range(T.N_CONTEXTS)}
    return [(a, b) for a in range(T.N_CONTEXTS) for b in range(a + 1, T.N_CONTEXTS)
            if fam[a] != fam[b]]


def conjunctive_reference(dist_deg):
    """cos(dphi): the parallelism a force-coding (fully conjunctive) z gives.

    Only defined for action-direction distances; see the module docstring on
    why mixed-family pairs are excluded from the fit.
    """
    return np.cos(np.radians(np.asarray(dist_deg, dtype=float)))


def mixing_from_parallelism(dist_deg, para):
    """One number per site: where the empirical curve sits between the two
    references.  para ~= alpha * 1 + (1 - alpha) * cos(dphi), least squares.

    alpha = 1 fully abstract, alpha = 0 fully conjunctive.  Fit rather than
    read off a single distance so one noisy bin cannot set it.
    """
    dist_deg = np.asarray(dist_deg, float)
    para = np.asarray(para, float)
    ok = np.isfinite(para) & (dist_deg > 1e-9)
    if ok.sum() < 2:
        return np.nan
    c = conjunctive_reference(dist_deg[ok])
    # para = alpha + (1-alpha) c  =>  para - c = alpha (1 - c)
    num = float(((para[ok] - c) * (1.0 - c)).sum())
    den = float(((1.0 - c) ** 2).sum())
    return num / den if den > 1e-12 else np.nan


# ── measures ─────────────────────────────────────────────────────────────

def _clf():
    # regularised, because mtl_hidden is 256-d against a few hundred rows per
    # cell and an unregularised fit would make the site comparison a
    # comparison of overfitting
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.1))


def _seq_split(sid, frac=TEST_FRAC, seed=0):
    """Held out BY SEQUENCE. Different bins hold different rows, but rows from
    one sequence appear in many bins, so a row-wise split would leak across
    the off-diagonal and inflate exactly the number being measured."""
    g = np.unique(sid)
    rng = np.random.default_rng(seed)
    te = set(rng.choice(g, size=max(1, int(frac * len(g))), replace=False).tolist())
    return np.isin(sid, list(te))


def decode_transfer(d, site, idx, n_bins, pairs=None, seed=0, shuffle_bins=False,
                    permute_labels=False):
    """(n_bins, n_bins) SIGNED binary decode accuracy, train bin -> test bin.

    Binary over same-family context pairs rather than K-way, because chance is
    then exactly 0.5 and BELOW-chance transfer is interpretable -- which is the
    signature the module docstring is about.
    """
    X, cid, sid = d[site], d["cid"], d["sid"]
    pairs = same_family_pairs() if pairs is None else pairs
    rng = np.random.default_rng(seed + 99)
    if shuffle_bins:
        idx = rng.integers(0, n_bins, len(idx))
    if permute_labels:
        cid = rng.permutation(cid)
    te = _seq_split(sid, seed=seed)

    acc = np.full((n_bins, n_bins), np.nan)
    cnt = np.zeros((n_bins, n_bins))
    for c0, c1 in pairs:
        m_pair = (cid == c0) | (cid == c1)
        y = (cid == c0).astype(int)
        for b in range(n_bins):
            tr = m_pair & (idx == b) & ~te
            if tr.sum() < MIN_CELL or len(np.unique(y[tr])) < 2:
                continue
            f = _clf().fit(X[tr], y[tr])
            for b2 in range(n_bins):
                ts = m_pair & (idx == b2) & te
                if ts.sum() < MIN_CELL or len(np.unique(y[ts])) < 2:
                    continue
                a = f.score(X[ts], y[ts])
                acc[b, b2] = a if np.isnan(acc[b, b2]) else acc[b, b2] + a
                cnt[b, b2] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def parallelism_transfer(d, site, idx, n_bins, pairs=None):
    """(n_bins, n_bins) cosine between the CODING VECTORS of two bins.

    The coding vector for a pair in bin b is the difference of that pair's two
    context means inside b. 1 = the same direction everywhere (additive), 0 =
    rotated away, negative = inverted.
    """
    X, cid = d[site], d["cid"]
    pairs = same_family_pairs() if pairs is None else pairs
    out = np.full((n_bins, n_bins), np.nan)
    cnt = np.zeros((n_bins, n_bins))
    for c0, c1 in pairs:
        vecs = {}
        for b in range(n_bins):
            m0 = (cid == c0) & (idx == b)
            m1 = (cid == c1) & (idx == b)
            if m0.sum() < MIN_CELL or m1.sum() < MIN_CELL:
                continue
            v = X[m0].mean(0) - X[m1].mean(0)
            n = np.linalg.norm(v)
            if n > 1e-12:
                vecs[b] = v / n
        for b, vb in vecs.items():
            for b2, vb2 in vecs.items():
                c = float(vb @ vb2)
                out[b, b2] = c if np.isnan(out[b, b2]) else out[b, b2] + c
                cnt[b, b2] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)


def kernel(mat, dist, normalise=False):
    """Collapse a (bin, bin) matrix onto its distance axis.

    Returns (distances, mean, sem). `normalise` divides by the distance-0
    value, which is what makes sites with different within-bin ceilings
    comparable -- z is 32-d and mtl_hidden 256-d, so an unnormalised site
    comparison is a comparison of how much each one overfits.
    """
    mat = np.asarray(mat, float)
    ds = np.unique(np.round(dist, 6))
    xs, ms, es = [], [], []
    for dd in ds:
        sel = (np.round(dist, 6) == dd) & np.isfinite(mat)
        if not sel.any():
            continue
        v = mat[sel]
        xs.append(dd); ms.append(v.mean())
        es.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
    xs, ms, es = np.array(xs), np.array(ms), np.array(es)
    if normalise and len(ms) and np.isfinite(ms[0]) and abs(ms[0]) > 1e-9:
        ms, es = ms / ms[0], es / abs(ms[0])
    return xs, ms, es


def variance_decomposition(d, site, idx, n_bins):
    """Two-way on the cell means: context, bin, interaction, within-cell.

    Balanced -- only bins populated for EVERY context are used, because an
    unbalanced cell table turns selection into apparent interaction.
    """
    X, cid = d[site], d["cid"]
    ctxs = list(range(T.N_CONTEXTS))
    cells = np.full((len(ctxs), n_bins, X.shape[1]), np.nan)
    resid = []
    for i, c in enumerate(ctxs):
        for b in range(n_bins):
            m = (cid == c) & (idx == b)
            if m.sum() < MIN_CELL:
                continue
            cells[i, b] = X[m].mean(0)
            resid.append(((X[m] - X[m].mean(0)) ** 2).sum(1).mean())
    ok = [b for b in range(n_bins) if not np.isnan(cells[:, b, 0]).any()]
    if len(ok) < 2:
        return None
    Mc = cells[:, ok, :]
    mu = Mc.mean((0, 1))
    a = Mc.mean(1) - mu
    g = Mc.mean(0) - mu
    e = Mc - mu - a[:, None, :] - g[None, :, :]
    return {"context": float((a ** 2).sum(1).mean()),
            "bin": float((g ** 2).sum(1).mean()),
            "interaction": float((e ** 2).sum(2).mean()),
            "within_cell": float(np.mean(resid)),
            "n_bins_used": len(ok)}


# ── the figure ───────────────────────────────────────────────────────────

def analyse(d, conditioning="action_dir", sites=SITES, seed=0):
    """Every measure for one conditioning variable, plus its nulls."""
    idx, dist, labels, axis_name = BINNERS[conditioning](d)
    n_bins = int(dist.shape[0])
    # how many rows actually landed in each bin -- an unpopulated bin makes a
    # whole row and column of the transfer matrix NaN, and the reader should be
    # able to see that rather than infer it from a gap
    counts = np.bincount(idx, minlength=n_bins)
    idx_shuf = np.random.default_rng(seed + 7).integers(0, n_bins, len(idx))
    out = {"conditioning": conditioning, "dist": dist, "n_bins": n_bins,
           "labels": labels, "axis_name": axis_name, "counts": counts,
           "sites": {}}
    for site in sites:
        acc = decode_transfer(d, site, idx, n_bins, seed=seed)
        par = parallelism_transfer(d, site, idx, n_bins)
        xs_a, ma, ea = kernel(acc, dist, normalise=False)
        xs_p, mp, ep = kernel(par, dist)
        out["sites"][site] = {
            "acc_mat": acc, "par_mat": par,
            "acc_k": (xs_a, ma, ea), "par_k": (xs_p, mp, ep),
            "alpha": (mixing_from_parallelism(xs_p, mp)
                      if conditioning == "action_dir" else np.nan),
            "var": variance_decomposition(d, site, idx, n_bins),
            "mixed_par_k": kernel(parallelism_transfer(
                d, site, idx, n_bins, pairs=mixed_family_pairs()), dist),
            # FLOORS. Both statistics are biased by finite cell sizes, in
            # OPPOSITE directions, and neither is readable without its floor:
            #   parallelism is attenuated DOWN, because the two cell means
            #     carry independent noise, so E<v_b, v_b'> < cos(true angle)
            #     for b != b' while the diagonal is exactly 1 by construction
            #   interaction is biased UP, because e_cb inherits the noise in
            #     every cell mean it is built from
            # Shuffling the bin labels destroys any true dependence while
            # keeping the cell sizes, so what comes back is the bias alone.
            "par_k_null": kernel(parallelism_transfer(
                d, site, idx_shuf, n_bins), dist),
            "var_null": variance_decomposition(d, site, idx_shuf, n_bins),
        }
    out["null_shuffled"] = kernel(decode_transfer(
        d, "z", idx, n_bins, seed=seed, shuffle_bins=True), dist)
    out["null_permuted"] = kernel(decode_transfer(
        d, "z", idx, n_bins, seed=seed, permute_labels=True), dist)
    return out


def fig_conjunctivity(arm, seed, res, path, tag=None):
    """Four panels: the raw transfer matrix, the two kernels, the decomposition."""
    set_plt_params()
    tag = tag or arm          # so a gp run says so on the figure, not just in its path
    cond = res["conditioning"]
    unit = "deg" if cond == "action_dir" else "grid units"
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

    # A. the raw object the kernels summarise, for z
    z = res["sites"]["z"]["acc_mat"]
    # FIXED 0-1, DIVERGING ABOUT CHANCE, and saturation is a result rather than
    # a plotting failure: a matrix that is uniformly dark red IS the finding,
    # because it says a decoder fit in any one bin transfers everywhere. An
    # adaptive scale would turn that into a picture of sampling noise and
    # invite reading structure into it.
    im = ax[0].imshow(z, cmap="RdBu_r", vmin=0.0, vmax=1.0)
    cbar_lab = "accuracy (white = chance 0.5)"
    ax[0].set_title("z: decode transfer\n(train bin -> test bin)")
    labs, cnts = res["labels"], res["counts"]
    small = 7 if len(labs) <= 8 else 5.5
    ax[0].set_xticks(range(len(labs)))
    ax[0].set_xticklabels(labs, rotation=90, fontsize=small)
    ax[0].set_yticks(range(len(labs)))
    ax[0].set_yticklabels([f"{l}  (n={c})" for l, c in zip(labs, cnts)],
                          fontsize=small)
    ax[0].set_xlabel(f"test bin: {res['axis_name']}")
    ax[0].set_ylabel(f"train bin: {res['axis_name']}")
    plt.colorbar(im, ax=ax[0], fraction=0.046, label=cbar_lab)

    # B. the accuracy kernel
    for site in res["sites"]:
        xs, m, e = res["sites"][site]["acc_k"]
        ax[1].errorbar(xs, m, yerr=e, marker="o", ms=4, capsize=2, label=site)
    xs, m, _ = res["null_shuffled"]
    ax[1].plot(xs, m, ls=":", color="grey", label="shuffled bins (ceiling, flat)")
    xs, m, _ = res["null_permuted"]
    ax[1].plot(xs, m, ls=":", color="k", label="permuted labels (floor)")
    ax[1].axhline(0.5, color="k", lw=0.8, alpha=0.4)
    if cond == "action_dir":
        g = np.linspace(0, 180, 100)
        # a force-coding readout inverts past 90 deg: 0.5 + (ceiling-0.5)*cos
        ceil = res["sites"]["z"]["acc_k"][1][0]
        ax[1].plot(g, 0.5 + (ceil - 0.5) * conjunctive_reference(g),
                   ls="--", color="crimson", label="fully conjunctive")
    ax[1].set_xlabel(f"bin distance ({unit})")
    ax[1].set_ylabel("pairwise decode accuracy")
    ax[1].set_title("generalisation kernel\nbelow 0.5 = systematically inverted")
    ax[1].legend(fontsize=5.5, loc="lower left", framealpha=0.85)

    # C. the parallelism kernel, with both references
    par_title = "parallelism kernel (same-family pairs)"
    for site in res["sites"]:
        xs, m, e = res["sites"][site]["par_k"]
        ax[2].errorbar(xs, m, yerr=e, marker="o", ms=4, capsize=2, label=site)
    xs, m, _ = res["sites"]["z"]["par_k_null"]
    ax[2].plot(xs, m, ls=":", color="grey",
               label="shuffled bins = attenuation floor")
    ax[2].axhline(1.0, ls="--", color="seagreen", label="fully abstract")
    if cond == "action_dir":
        g = np.linspace(0, 180, 100)
        ax[2].plot(g, conjunctive_reference(g), ls="--", color="crimson",
                   label="fully conjunctive  cos(dphi)")
        # in the TITLE: every corner of these axes is occupied -- the curves
        # fill the top, the legend the lower left, the conjunctive reference
        # the lower right
        txt = "  ".join(f"{s_.replace('_layer','')} {res['sites'][s_]['alpha']:+.2f}"
                        for s_ in res["sites"])
        par_title += f"\nalpha (1 abstract, 0 conjunctive):  {txt}"
    ax[2].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax[2].set_xlabel(f"bin distance ({unit})")
    ax[2].set_ylabel("coding-vector cosine")
    ax[2].set_title(par_title, fontsize=8)
    ax[2].legend(fontsize=5.5, loc="lower left", framealpha=0.85)

    # D. variance decomposition
    keys = ["context", "bin", "interaction", "within_cell"]
    w = 0.2
    for i, site in enumerate(res["sites"]):
        v = res["sites"][site]["var"]
        if v is None:
            continue
        tot = sum(v[k] for k in keys)
        ax[3].bar(np.arange(len(keys)) + i * w, [v[k] / tot for k in keys],
                  width=w, label=site)
        vn = res["sites"][site]["var_null"]
        if vn is not None:
            # the shuffled floor, drawn ON the interaction bar: anything not
            # standing above its own outline is bias, not signal
            ax[3].bar(keys.index("interaction") + i * w,
                      vn["interaction"] / tot, width=w, facecolor="none",
                      edgecolor="k", lw=1.0, hatch="///", zorder=5,
                      label="shuffled floor" if i == 0 else None)
    ax[3].set_xticks(np.arange(len(keys)) + 1.5 * w)
    ax[3].set_xticklabels(keys, rotation=20, fontsize=7)
    ax[3].set_ylabel("fraction of variance")
    ax[3].set_title("two-way decomposition")
    ax[3].legend(fontsize=6)

    for a_ in ax[1:]:
        make_axis_nice(a_); _despine(a_)
    fig.suptitle(f"{tag} seed {seed} -- context coding vs {cond}", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


SUBDIR = "conjunctivity"
N_POLICY_EP = 120          # per context; ~35 steps each, so ~34k rows over 8


class _ScriptedActor:
    """An EXOGENOUS stand-in for the policy: the deployed action statistics
    (rho 0.92, tanh-saturated) with no dependence on the observation at all.

    This is the control that separates "the policy's closed loop" from "the
    action statistics it happens to produce".  A closed-loop policy chooses
    actions partly on the basis of context, so conditioning on action direction
    conditions on a context-dependent variable; this cannot, so any conjunctivity
    it reproduces is attributable to the statistics alone.
    """

    def __init__(self, rho=0.92, gain=2.2, seed=0):
        import numpy as _np
        self.rho, self.gain = rho, gain
        self.rng = _np.random.default_rng(seed)
        self.u = None

    def get_action(self, obs):
        import torch as _t
        u = (self.rng.standard_normal(2) if self.u is None else
             self.rho * self.u +
             np.sqrt(1 - self.rho ** 2) * self.rng.standard_normal(2))
        self.u = u
        a = np.tanh(self.gain * u).astype(np.float32)
        return _t.tensor(a).unsqueeze(0), None, None


def collect_policy(arm, seed, agent_path, n_episodes=N_POLICY_EP,
                   action_mode=M.DEFAULT_ACTION_MODE, tss_min=None, device="cpu",
                   actor=None):
    """The same dict collect() returns, but driven by a TRAINED SAC POLICY.

    Every step goes through sac's own functions -- circuit_state,
    augmented_obs_step, cb_predict's inputs, env_step_params -- rather than a
    private copy of the loop.  That is not fastidiousness: sac.py once fed the
    circuit raw positions where model.build fed scaled ones, z's context decode
    fell 1.00 -> 0.28, and both frozen arms looked exactly like the no-circuit
    baseline.  A private rollout here would reproduce that silently.

    ALL EIGHT CONTEXTS, not just the four the policy trained on.  The policy is
    a DRIVER here -- its job is to generate the deployed action distribution,
    not to be competent -- and using the same eight keeps every pair, every
    bin and every decomposition directly comparable with the supervised figure.
    It does perform badly in the four it never saw; that costs nothing the
    measurement depends on.

    `tss` is steps since the EPISODE start, because the context is constant
    within an episode under a policy.  It plays the same role the supervised
    tss does -- dropping the steps before z has resolved -- and is filtered the
    same way.
    """
    import sac
    import torch
    tss_min = T.TSS_MIN if tss_min is None else tss_min
    loaded = M.load(arm, seed, action_mode)
    if loaded is None:
        return None
    ec, cb = loaded
    sac.freeze_module(ec); sac.freeze_module(cb)
    agent = sac.load_agent(agent_path)
    cfg = agent.config
    # THE AGENT AND THE CIRCUIT MUST BE THE SAME PAIR. An RL policy is adapted
    # to whatever z its own frozen circuit emitted; driving it with the other
    # circuit measures a policy reading a channel it never saw, and nothing
    # about the result would look wrong. The two are identical in shape, so
    # only the recorded mode can catch it.
    # ONLY WHEN THE ACTOR IS ACTUALLY DRIVING. Under source="scripted" the
    # actor is ignored and the agent supplies just cb_proj (a fixed random
    # projection) and the channel flags, so a mismatched pair is harmless
    # there -- and refusing it would block measuring a circuit that has no RL
    # run of its own yet, which is exactly when the scripted drive is useful.
    # normalised on both sides: a saved blob may still say "fitted", and an
    # older blob has no readout fields (deterministic). The WHOLE circuit has
    # to match -- a noisy circuit and its tanh sibling are shape-identical.
    agent_circuit = M.CircuitSpec(getattr(cfg, "action_mode", "ar1"),
                                  getattr(cfg, "ec_readout", M.DEFAULT_READOUT),
                                  getattr(cfg, "rate_beta", M.RATE_BETA),
                                  getattr(cfg, "noise_sigma", M.NOISE_SIGMA))
    if actor is None and agent_circuit != M.circuit(action_mode):
        raise ValueError(
            f"{agent_path} was trained against the {agent_circuit} circuit but "
            f"{M.circuit(action_mode)} was requested. Pass --agent for a run "
            f"of the matching circuit (the default preset 'main' is "
            f"{M.DEFAULT_ACTION_MODE!r}; an 'ar1' agent needs a run made with "
            f"action_mode='ar1').")
    rng = np.random.default_rng(0)

    cols = {k: [] for k in ("z", "mtl_hidden", "cb_layer1", "cb_layer2",
                            "cid", "fam", "sid", "tss", "pos", "vel", "act")}
    ep_id = 0
    for cid in range(T.N_CONTEXTS):
        for _ in range(n_episodes):
            ep_id += 1
            pos, goal = sac.sample_pos_goal(rng)
            vel = np.zeros(2, np.float32); vo = vel.copy()
            h = torch.zeros(1, sac.EC_HIDDEN, device=device)
            s_tm1 = torch.zeros(1, sac.SDIM, device=device)
            a_tm1 = torch.zeros(1, sac.ACTION_DIM, device=device)
            e_t = torch.zeros(1, sac.SDIM, device=device)
            cb_h2 = torch.zeros(1, sac.CB_HIDDEN, device=device)
            s_t = torch.tensor(sac.circuit_state(pos, vo), device=device).unsqueeze(0)
            for t in range(sac.MAX_STEPS):
                if np.linalg.norm(pos - goal) < sac.GOAL_RADIUS:
                    break
                obs, z_t, h = sac.augmented_obs_step(
                    ec, cfg, h, s_tm1, a_tm1, s_t, e_t, cb_h2, agent.cb_proj,
                    pos, vo, goal, None, device)
                drive = agent.actor if actor is None else actor
                with torch.no_grad():
                    a_t, _, _ = drive.get_action(
                        torch.tensor(obs, device=device).unsqueeze(0))
                act = a_t.squeeze(0).cpu().numpy().astype(np.float32)
                at = torch.tensor(act, device=device).unsqueeze(0)
                # all four sites at THIS step's (s, a, z), matching the order
                # model.rollout records them in
                with torch.no_grad():
                    s_pred, l1, l2 = cb(s_t, at, z_t, return_hidden=True)
                if t >= tss_min:
                    cols["z"].append(z_t.squeeze(0).cpu().numpy())
                    cols["mtl_hidden"].append(h.squeeze(0).cpu().numpy())
                    cols["cb_layer1"].append(l1.squeeze(0).cpu().numpy())
                    cols["cb_layer2"].append(l2.squeeze(0).cpu().numpy())
                    # SCALED, as model.build scales them, so the position bins
                    # here and in the supervised figure mean the same thing
                    sc = sac.circuit_state(pos, vo)
                    cols["pos"].append(sc[:2]); cols["vel"].append(sc[2:])
                    cols["act"].append(act.copy())
                    cols["cid"].append(cid); cols["fam"].append(T.FAMILY[cid])
                    cols["sid"].append(ep_id); cols["tss"].append(t)
                pn, vn = sac.env_step_params(pos, vel, act, cid, rng)
                s_next = torch.tensor(sac.circuit_state(pn, vn),
                                      device=device).unsqueeze(0)
                s_tm1, a_tm1 = s_t, at
                s_t = s_next
                e_t = s_next - s_pred
                cb_h2 = l2
                pos, vel, vo = pn, vn, vn
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("z", "mtl_hidden", "cb_layer1", "cb_layer2", "pos", "vel", "act"):
        out[k] = out[k].astype(np.float64 if k in ("pos", "vel") else np.float32)
    return out


def main(arm="main", seed=0, action_mode=M.DEFAULT_ACTION_MODE, out_dir=None,
         source="supervised", agent_path=None):
    """Both figures for one (arm, seed), into model.fig_dir's conjunctivity
    subdirectory so they sit beside that seed's other panels.

    `source` selects the DISTRIBUTION the circuit is driven with:
      "supervised"  data.generate_sequences, what it was fitted on
      "policy"      a trained SAC actor in closed loop, what it is deployed on
    The same circuit measured two ways; the difference between the figures is
    the whole question.
    """
    from representation_analysis import collect
    # "main" is the frozen default-circuit preset (was main_frozen)
    if source == "scripted":
        agent_path = agent_path or os.path.join(
            "sac_runs", "main", f"seed_{seed}", "agent.pt")
        d = collect_policy(arm, seed, agent_path, action_mode=action_mode,
                           actor=_ScriptedActor())
    elif source == "policy":
        agent_path = agent_path or os.path.join(
            "sac_runs", "main", f"seed_{seed}", "agent.pt")
        if not os.path.exists(agent_path):
            print(f"no trained agent at {agent_path}")
            return None
        d = collect_policy(arm, seed, agent_path, action_mode=action_mode)
    else:
        d = collect(arm, seed, action_mode=action_mode)
    if d is None:
        print(f"no checkpoint for {arm} seed {seed} ({action_mode})")
        return None
    out_dir = out_dir or M.fig_dir(arm, seed, SUBDIR, action_mode)
    results = {}
    for cond in ("action_dir", "position"):
        res = analyse(d, cond)
        p = fig_conjunctivity(arm, seed, res,
                              os.path.join(out_dir, f"{cond}_{source}.png"),
                              tag=f"{M.arm_tag(arm, action_mode)} [{source}]")
        results[cond] = res
        zk = res["sites"]["z"]
        xs, m, _ = zk["par_k"]
        fmt = "{:.0f}" if cond == "action_dir" else "{:.2f}"
        print(f"\n{cond}: z parallelism  " +
              "  ".join(f"{fmt.format(x)}:{v:+.2f}" for x, v in zip(xs, m)))
        if cond == "action_dir":
            print(f"  alpha (1 abstract, 0 conjunctive) per site: " +
                  "  ".join(f"{s} {res['sites'][s]['alpha']:+.2f}"
                            for s in res["sites"]))
        v, vn = zk["var"], zk["var_null"]
        if v:
            tot = v["context"] + v["bin"] + v["interaction"] + v["within_cell"]
            # AGAINST ITS FLOOR, always. Both statistics are biased by finite
            # cell sizes and an unqualified interaction number reads as a small
            # real effect when it is usually the bias.
            ratio = (v["interaction"] / vn["interaction"]) if vn else float("nan")
            print(f"  z variance: context {v['context']/tot:.3f}  "
                  f"bin {v['bin']/tot:.3f}  "
                  f"interaction {v['interaction']/tot:.5f} "
                  f"(floor {vn['interaction']/tot:.5f}, {ratio:.2f}x)")
        print(f"  -> {p}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="main")
    ap.add_argument("--seed", type=int, default=0)
    M.add_circuit_args(ap)      # --action-mode --readout --rate-beta --noise-sigma
    ap.add_argument("--source", default="supervised",
                    choices=("supervised", "policy", "scripted"),
                    help="distribution to drive the circuit with")
    ap.add_argument("--agent", default=None,
                    help="agent.pt for --source policy (default: "
                         "sac_runs/main/seed_<seed>/agent.pt)")
    a = ap.parse_args()
    main(a.arm, a.seed, M.circuit_from_args(a), source=a.source, agent_path=a.agent)
