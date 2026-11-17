"""
sro.py -- the same circuit on Bernardi et al. 2020's serial-reversal SRO task.

THE POINT.  In the motor task the cerebellar error is a SENSORY prediction
error, s_t - s_hat_t.  Here the forward model predicts REWARD, so its error is a
reward-prediction error, r_t - r_hat_t. want to check if mtl's context inference 
depends on what CB prediction error is or is it agnostic to it.

THE TASK (Bernardi et al. 2020).  4 stimuli x 2 contexts.  Each
(context, stimulus) fixes a Stimulus-Response-Outcome mapping: which operant
(hold/release) is correct, and whether performing it correctly pays out.  Within
a context the 4 stimuli span {R+, R-, H+, H-}, so operant and reward are
ORTHOGONAL -- neither alone identifies the context.  Context is never cued and
switches every 50-70 trials, flipping all four mappings at once.

OUTCOMES ARE ACTION-DEPENDENT (hold vs release).

    correct on a "+" stimulus  ->  r = +1   (reward)
    correct on a "-" stimulus  ->  r =  0   (nothing, but no timeout)
    incorrect                  ->  r = -1   (timeout)

Three levels, not two, and for a reason: with a binary reward, both operants on
a "-" stimulus give 0, so nothing orders them and a greedy agent is at chance on
half the stimuli.  The timeout is what makes the correct response worth making
there -- exactly as in the monkey task.

THE TWO ARMS, mirroring model.py:

    main     MTL <- [x_{t-1}, O_{t-1},          delta]     outcome only via the error
    control  MTL <- [x_{t-1}, O_{t-1}, r_{t-1}, delta]     outcome given directly

MTL is one step behind on everything except the error, exactly as in the motor
model (`parts = [s_tm1, a_tm1, e_t]`).  It never sees x_t -- that is CB's input
for the current trial.  O_{t-1} matters here in a way it did not before: the
outcome now depends on the action, so an RPE cannot be interpreted without
knowing which action produced it.

CB IS A PURE FORWARD MODEL.  r_hat = CB(x_t, O_t, z_t), one scalar, one head.
The earlier version had CB emit the operant as well, which made it a policy AND
a value predictor; here the operant is an INPUT, so CB does the same job its
motor counterpart does.

ACTIONS ARE RANDOM DURING TRAINING, greedy only at test.  Random operants are
the exogenous-action setup the motor task uses, and they give unbiased coverage
of both operants at every stimulus -- a greedy policy would stop sampling the
branch it believes is wrong, which is precisely the branch it needs after a
switch.  At test the operant is chosen by querying the forward model twice and
taking the higher r_hat.  No policy is ever trained: the behaviour in the Fig-1C
analog falls out of predicting consequences.

  python sro.py --train --arms main control --seeds 0
  python sro.py --figures --arms main control --seeds 0 --pool
"""
import argparse
import os
from itertools import combinations, permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── task ───
N_STIM, N_CTX = 4, 2
SEQ_LEN = 240
SWITCH_MIN, SWITCH_MAX = 50, 70

# SRO[context, stimulus] = (correct operant, does correct performance pay out)
#   context 0:  R+  R-  H+  H-      spans the 2x2
#   context 1:  full reversal -- operant AND payout flip for all four
# each row = C 
# SRO[0, 0] = [1, 1] → in context 0, stimulus 0 needs release, and doing it correctly pays. SRO[1, 0] = [0, 0] → in context 1 the same image needs hold, and correct performance doesn't pay.

# VARIANT.  Set SRO_VARIANT=bernardi to use the table actually run in the
# experiment (their Fig 1B).  There the OPERANT reverses for all four stimuli
# but the REINFORCEMENT reverses for only two (A and C); B stays rewarded and D
# stays unrewarded in both contexts.  Our default "full" table reverses both for
# all four.  Both tables keep context, operant and reinforcement mutually
# orthogonal across the eight conditions, so the geometry is comparable; the
# experiment's version is the harder test, because on B and D trials reward is
# identical in the two contexts and the context must be held rather than re-read.
VARIANT = os.environ.get("SRO_VARIANT", "full").lower()
if VARIANT not in ("full", "bernardi"):
    raise ValueError(f"SRO_VARIANT must be 'full' or 'bernardi', got {VARIANT!r}")

_SRO_FULL = np.array([
    [[1, 1], [1, 0], [0, 1], [0, 0]],
    [[0, 0], [0, 1], [1, 0], [1, 1]],
], dtype=np.int64)

# Bernardi et al. 2020, Fig 1B.  R=1 H=0, +=1 -=0.
#   ctx 1:  A R+   B H+   C R-   D H-
#   ctx 2:  A H-   B R+   C H+   D R-
_SRO_BERNARDI = np.array([
    [[1, 1], [0, 1], [1, 0], [0, 0]],
    [[0, 0], [1, 1], [0, 1], [1, 0]],
], dtype=np.int64)

SRO = _SRO_BERNARDI if VARIANT == "bernardi" else _SRO_FULL

# every file this run writes is kept under its own variant folder, so the two
# task versions can never be mixed up with each other
CKPT_SUBDIR = "sro_full" if VARIANT == "full" else "sro_match_bernardi"
TASK_DIR = "SRO_Bernardi" if VARIANT == "full" else "SRO_match_bernardi"

N_TRAIN, N_VAL, BATCH, EPOCHS, LR = 2000, 300, 128, 60, 1e-3
EC_HIDDEN, CB_HIDDEN, LATENT_DIM = 256, 256, 64      # 64 to match model.py
ARMS = {"main": True, "control": False}              # True = outcome withheld

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
FIG_ROOT = os.path.join(HERE, "figures")


def sro_ckpt_dir():
    """checkpoints/<sro_full|sro_match_bernardi>/ -- one folder per task table."""
    return os.path.join(CKPT_DIR, CKPT_SUBDIR)


def ckpt_path(arm, seed):
    return os.path.join(sro_ckpt_dir(), f"sro_{arm}_seed{seed}.pt")


def pooled_dir():
    """figures/main/<TASK_DIR>/pooled_seeds -- the across-seed figures."""
    d = os.path.join(FIG_ROOT, "main", TASK_DIR, "pooled_seeds")
    os.makedirs(d, exist_ok=True)
    return d


def fig_dir(seed):
    """figures/main/SRO_Bernardi/seed<N>/

    Its own subfolder rather than loose in the seed directory: this is a
    separate task with its own trained networks, so mixing its figures in with
    the motor ones invites reading an `sro_` file as a result about the motor
    model.  Under `main` because sro_behaviour.png plots BOTH arms on one axis
    -- it is a comparison, not a per-arm result.
    """
    d = os.path.join(FIG_ROOT, "main", TASK_DIR, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def outcome(ctx, stim, op):
    """+1 correct and rewarded, 0 correct and unrewarded, -1 incorrect."""
    correct = op == SRO[ctx, stim, 0]
    pays = SRO[ctx, stim, 1] == 1
    return np.where(correct, np.where(pays, 1.0, 0.0), -1.0).astype(np.float32)


def generate(n_seqs, seq_len=SEQ_LEN, seed=0):
    """Stimulus and context schedules.  Operants are NOT drawn here -- they are
    chosen during the rollout, randomly in training and greedily at test."""
    rng = np.random.default_rng(seed)
    stim = rng.integers(0, N_STIM, size=(n_seqs, seq_len))
    ctx = np.zeros((n_seqs, seq_len), dtype=np.int64)
    tss = np.zeros((n_seqs, seq_len), dtype=np.int64)
    for n in range(n_seqs):
        c, t = int(rng.integers(N_CTX)), 0
        while t < seq_len:
            blk = int(rng.integers(SWITCH_MIN, SWITCH_MAX + 1))
            end = min(t + blk, seq_len)
            ctx[n, t:end] = c
            tss[n, t:end] = np.arange(end - t)
            t, c = end, 1 - c
    return stim, ctx, tss


# ── model ────────────────────────────────────────────────────────────────────
class EC(nn.Module):
    """MTL.  Input width depends on the arm: the control also receives the
    previous outcome, which makes the RPE informationally redundant."""

    def __init__(self, withhold_outcome):
        super().__init__()
        self.withhold = withhold_outcome
        w = N_STIM + 1 + 1 + (0 if withhold_outcome else 1)   # x, O, delta [, r]
        self.cell = nn.GRUCell(w, EC_HIDDEN)
        self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())

    def init_hidden(self, B):
        return torch.zeros(B, EC_HIDDEN)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.readout(h), h


class CB(nn.Module):
    """Cerebellum: a forward model.  (stimulus, operant, context code) -> reward.

    ONE head.  Taking the operant as an INPUT is what lets the same network be
    queried twice at test to choose an action, and is what makes it a forward
    model rather than a policy.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STIM + 1 + LATENT_DIM, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, 1))

    def forward(self, x, o, z):
        return self.net(torch.cat([x, o, z], -1)).squeeze(-1)


def rollout(ec, cb, stim, ctx, arm, policy="random", e_mode="intact",
            gen=None, rng=None):
    """One pass over the trial sequence.

    policy="random"  operants are coin flips           (training)
    policy="greedy"  argmax over CB's two predictions  (test read-out)
    e_mode           intact | zero | shuffle -- the ablation, on delta
    """
    withhold = ARMS[arm]
    B, T = stim.shape
    rng = rng or np.random.default_rng(0)
    h = ec.init_hidden(B)
    eye = torch.eye(N_STIM)
    x_prev = torch.zeros(B, N_STIM)
    o_prev = torch.zeros(B, 1)
    r_prev = torch.zeros(B, 1)
    delta = torch.zeros(B, 1)
    Z, R, O, RH, D, H = [], [], [], [], [], []
    for t in range(T):
        d = delta
        if e_mode in ("zero", "shuffle"):
            d = (torch.zeros_like(delta) if e_mode == "zero"
                 else delta[torch.randperm(B, generator=gen)])
        parts = [x_prev, o_prev, d] if withhold else [x_prev, o_prev, r_prev, d]
        z, h = ec.step(torch.cat(parts, -1), h)
        x_t = eye[torch.as_tensor(stim[:, t])]
        if policy == "greedy":
            with torch.no_grad():
                r0 = cb(x_t, torch.zeros(B, 1), z)
                r1 = cb(x_t, torch.ones(B, 1), z)
            op = (r1 > r0).float().unsqueeze(-1)
        else:
            op = torch.as_tensor(rng.integers(0, 2, size=(B, 1)),
                                 dtype=torch.float32)
        r_hat = cb(x_t, op, z)
        r = torch.as_tensor(outcome(ctx[:, t], stim[:, t],
                                    op.squeeze(-1).long().numpy()))
        delta = (r - r_hat).detach().unsqueeze(-1)
        Z.append(z); R.append(r); O.append(op.squeeze(-1)); RH.append(r_hat)
        D.append(delta.squeeze(-1)); H.append(h)
        x_prev, o_prev, r_prev = x_t, op, r.unsqueeze(-1)
    return dict(z=torch.stack(Z, 1), r=torch.stack(R, 1),
                o=torch.stack(O, 1), r_hat=torch.stack(RH, 1),
                delta=torch.stack(D, 1), h=torch.stack(H, 1))


PARTIAL_EVERY = 5          # epochs between resumable snapshots


def partial_path(arm, seed):
    return ckpt_path(arm, seed).replace(".pt", ".partial.pt")


def train(arm, seed=0, epochs=EPOCHS):
    p = ckpt_path(arm, seed)
    if os.path.exists(p):
        print(f"== sro/{arm} seed {seed}: exists, skipping"); return
    torch.manual_seed(seed)
    stim, ctx, _ = generate(N_TRAIN, seed=seed)
    ec, cb = EC(ARMS[arm]), CB()
    opt = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    rng = np.random.default_rng(seed)

    # RESUME.  A run can be stopped at any point; every PARTIAL_EVERY epochs we
    # write weights, optimiser and all three RNG streams, so picking up again
    # continues the same trajectory rather than restarting the network.
    start_ep = 1
    pp = partial_path(arm, seed)
    if os.path.exists(pp):
        try:
            ck = torch.load(pp, weights_only=False)
            ec.load_state_dict(ck["ec"]); cb.load_state_dict(ck["cb"])
            opt.load_state_dict(ck["opt"])
            np.random.set_state(ck["np_state"])
            rng.bit_generator.state = ck["rng_state"]
            torch.set_rng_state(ck["torch_state"])
            start_ep = ck["epoch"] + 1
            print(f"== sro/{arm} seed {seed}: resuming at epoch {start_ep}",
                  flush=True)
        except Exception as e:
            print(f"== sro/{arm} seed {seed}: partial unreadable ({e}), "
                  f"starting over", flush=True)
            start_ep = 1

    for ep in range(start_ep, epochs + 1):
        perm = np.random.permutation(N_TRAIN); tot = 0.
        for i in range(0, N_TRAIN, BATCH):
            j = perm[i:i + BATCH]
            out = rollout(ec, cb, stim[j], ctx[j], arm,
                          policy="random", rng=rng)
            loss = ((out["r_hat"] - out["r"]) ** 2).mean()   # forward model only
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        if ep % 10 == 0 or ep == 1:
            print(f"  [sro/{arm} s{seed}] ep {ep:3d} | reward MSE {tot/N_TRAIN:.5f}",
                  flush=True)
        if ep % PARTIAL_EVERY == 0 and ep != epochs:
            os.makedirs(sro_ckpt_dir(), exist_ok=True)
            tmp = pp + ".tmp"          # write-then-rename: never a torn file
            torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(),
                        "opt": opt.state_dict(), "epoch": ep,
                        "np_state": np.random.get_state(),
                        "rng_state": rng.bit_generator.state,
                        "torch_state": torch.get_rng_state()}, tmp)
            os.replace(tmp, pp)
    os.makedirs(sro_ckpt_dir(), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict()}, p)
    if os.path.exists(pp):
        os.remove(pp)
    print(f"  saved -> {p}", flush=True)


def load(arm, seed):
    p = ckpt_path(arm, seed)
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu")
    ec, cb = EC(ARMS[arm]), CB()
    ec.load_state_dict(sd["ec"]); cb.load_state_dict(sd["cb"])
    ec.eval(); cb.eval()
    return ec, cb


# ── analysis ─────────────────────────────────────────────────────────────────
COND = dict(context=lambda c, s, o, r: c,
            value=lambda c, s, o, r: r,
            action=lambda c, s, o, r: o)
VCOL = {"context": "#c0392b", "value": "#e084c4", "action": "#f1a512"}
# value and action are the PREVIOUS trial's -- h_t has not seen the current
# stimulus or response, so those are the only ones it could carry.  Labelled
# explicitly in the legends so the panels cannot be misread.
VLAB = {"context": "context", "value": "value of previous trial",
        "action": "action of previous trial"}
MCOL = {"intact": "#1a1a1a", "zero": "#c0392b", "shuffle": "#e67e22"}
ACOL = {"main": "#2c7fb8", "control": "#1a1a1a"}


def collect(arm, seed, n=N_VAL, e_mode="intact", policy="random", data_seed=123):
    """Roll the trained arm out on held-out sequences."""
    got = load(arm, seed)
    if got is None:
        return None
    ec, cb = got
    stim, ctx, tss = generate(n, seed=data_seed)
    with torch.no_grad():
        out = rollout(ec, cb, stim, ctx, arm, policy=policy, e_mode=e_mode,
                      gen=torch.Generator().manual_seed(0),
                      rng=np.random.default_rng(7))
    out.update(stim=stim, ctx=ctx, tss=tss)
    return out


def _seq_split(n_seq, T, frac=.3, seed=0):
    """Test mask, split by SEQUENCE -- trials within a sequence share a context
    schedule, so a trial-level split would leak."""
    rng = np.random.default_rng(seed)
    te = rng.choice(n_seq, max(1, int(frac * n_seq)), replace=False)
    m = np.zeros(n_seq, bool); m[te] = True
    return np.repeat(m[:, None], T, 1).ravel()


def decode_by_tss(out, key="z", ks=range(-5, 21), seed=0):
    """Context decoding from z, aligned to the switch.

    ONE decoder fitted on all trials and scored per bin -- the same choice as
    decode.by_tss in the motor pipeline, and for the same reason: fitting inside
    each bin starves the late ones and manufactures a decay.- see if by looking at beh of a trial it can deode C/ HOW MANY TRIALS 
    OR STIM IT NEEDS TO DECODE C FROM BEHAVIOR
    """
    X = out[key].reshape(-1, out[key].shape[-1]).numpy()
    y = out["ctx"].ravel()
    tss = out["tss"].ravel()
    B, T = out["ctx"].shape
    te = _seq_split(B, T, seed=seed)
    clf = LogisticRegression(max_iter=2000).fit(X[~te], y[~te])
    T2 = tss.reshape(B, T).astype(int)
    out_acc = []
    for k in ks:
        if k >= 0:
            sel = te & (tss == k)
        else:
            # k trials BEFORE a switch: trial t qualifies if t+|k| starts a new
            # block.  np.roll WRAPS, so the last |k| columns would pick up the
            # start of the same sequence -- mask them out explicitly.
            nxt = np.roll(T2, -abs(k), axis=1)
            valid = np.ones((B, T), bool)
            valid[:, T - abs(k):] = False
            sel = te & ((nxt == 0) & valid).ravel()
        out_acc.append(clf.score(X[sel], y[sel]) if sel.sum() > 20 else np.nan)
    return np.array(list(ks), float), np.array(out_acc, float)


def fig1c(arm, seed, n=N_VAL, e_mode="intact"):
    """Bernardi Fig 1C: operant accuracy on the FIRST post-switch appearance of
    each stimulus, by appearance order.

    Order 1 is answered with the stale context and should be at or below chance.
    Orders 2-4 are stimuli NOT YET SEEN in the new context -- above chance there
    means the context was inferred, not that associations were relearned.

    Actions are GREEDY here: the operant is whichever of hold/release the
    forward model predicts will pay more.  No policy is trained; the behaviour
    is a read-out of CB plus the inferred context.
    """
    out = collect(arm, seed, n, e_mode=e_mode, policy="greedy")
    if out is None:
        return None
    stim, ctx, tss = out["stim"], out["ctx"], out["tss"]
    op = out["o"].numpy().astype(int)
    B, T = stim.shape
    acc = {k: [] for k in (1, 2, 3, 4)}
    for b in range(B):
        starts = np.flatnonzero(tss[b] == 0)
        for st in starts[1:]:                      # skip the first block
            seen, order = set(), 0
            for t in range(st, min(st + 40, T)):
                s_ = stim[b, t]
                if s_ in seen:
                    continue
                seen.add(s_); order += 1
                if order > 4:
                    break
                correct = op[b, t] == SRO[ctx[b, t], s_, 0]
                acc[order].append(float(correct))
    return {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}


# ── geometry: Bernardi's CCGP / SD / PS ──────────────────────────────────────
def _conditions():
    """8 conditions = 2 contexts x 4 stimuli, and their task variables."""
    ctx = np.array([c // N_STIM for c in range(8)])
    stim = np.array([c % N_STIM for c in range(8)])
    return ctx, stim, SRO[ctx, stim, 0], SRO[ctx, stim, 1]


def _all_dichotomies():
    """The 35 balanced 4-vs-4 splits of the 8 conditions."""
    items, seen, out = list(range(8)), set(), []
    for combo in combinations(items, 4):
        P = frozenset(combo)
        if P in seen:
            continue
        seen.add(P); seen.add(frozenset(items) - P)
        out.append((sorted(P), sorted(frozenset(items) - P)))
    return out


def _named():
    ctx, stim, op, rew = _conditions()
    f = lambda v: ([c for c in range(8) if v[c] == 1],
                   [c for c in range(8) if v[c] == 0])
    return {"context": f(ctx), "value": f(rew), "action": f(op)}


def _decode(X, cond, P):
    y = np.isin(cond, P).astype(int)
    # shuffled folds: the rows are grouped by condition, and unshuffled folds
    # would hold out whole conditions -- turning "decoding" into CCGP by accident
    cv = StratifiedKFold(4, shuffle=True, random_state=0)
    return cross_val_score(LogisticRegression(max_iter=500), X, y, cv=cv).mean()


def _ccgp(X, cond, P, N, k=3):
    """Train on a subset of conditions, test on the held-out ones.  This is the
    abstraction measure: it asks whether the split generalises to conditions the
    decoder never saw, which requires the geometry to be clustered rather than
    arbitrary."""
    accs = []
    for trP in combinations(P, k):
        teP = [c for c in P if c not in trP]
        for trN in combinations(N, k):
            teN = [c for c in N if c not in trN]
            tr = np.isin(cond, list(trP) + list(trN))
            te = np.isin(cond, teP + teN)
            if tr.sum() < 10 or te.sum() < 10:
                continue
            clf = LogisticRegression(max_iter=500).fit(
                X[tr], np.isin(cond[tr], trP).astype(int))
            accs.append((clf.predict(X[te]) ==
                         np.isin(cond[te], teP).astype(int)).mean())
    return float(np.mean(accs)) if accs else np.nan


def _ps(means, P, N):
    """Parallelism: the best mean cosine over pairings of the two sides.  High PS
    means the coding directions are parallel across conditions -- the geometric
    signature of abstraction."""
    best = -1.
    for perm in permutations(N):
        vs = []
        for i, p in enumerate(P):
            v = means[p] - means[perm[i]]
            vs.append(v / (np.linalg.norm(v) + 1e-9))
        cs = [vs[i] @ vs[j] for i in range(len(vs)) for j in range(i + 1, len(vs))]
        best = max(best, float(np.mean(cs)))
    return best


def _cond_ids(out, prev=True):
    """Condition id per trial = context*4 + stimulus.

    prev=True uses the PREVIOUS trial's stimulus, which is what h_t can
    possibly encode: h_t is built from [x_{t-1}, O_{t-1}, delta] and has not
    seen the current stimulus or made the current response.  Decoding
    CURRENT-trial value and action from it is not a null result, it is a
    mislabelled axis -- they cannot be there.

    This is Bernardi's own distinction: their Fig 3 is the PRE-stimulus window
    and decodes previous-trial variables; Fig 5 is post-stimulus and decodes
    current-trial ones.  Our h updates once per trial, so it is the Fig-3 case.

    Context is unaffected either way -- it is constant within a block, so the
    previous trial's context is almost always the current one.
    """
    ctx, stim = out["ctx"], out["stim"]
    if prev:
        ctx = np.concatenate([ctx[:, :1], ctx[:, :-1]], 1)
        stim = np.concatenate([stim[:, :1], stim[:, :-1]], 1)
    return (ctx * N_STIM + stim).ravel()


def geometry(out, key="h", tss_min=5, max_per_cond=400, seed=0, prev=True):
    """Decoding, CCGP and PS for all 35 dichotomies, plus SD."""
    X = out[key].reshape(-1, out[key].shape[-1]).numpy()
    cond = _cond_ids(out, prev)
    keep = out["tss"].ravel() >= tss_min          # settled trials only
    X, cond = X[keep], cond[keep]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(cond == c),
                                     min(max_per_cond, (cond == c).sum()),
                                     replace=False) for c in range(8)])
    X, cond = X[idx], cond[idx]
    means = np.stack([X[cond == c].mean(0) for c in range(8)])
    dec, ccgp_, ps_ = [], [], []
    for P, N in _all_dichotomies():
        dec.append(_decode(X, cond, P))
        ccgp_.append(_ccgp(X, cond, P, N))
        ps_.append(_ps(means, P, N))
    named, out_named = _named(), {}
    for nm, (P, N) in named.items():
        out_named[nm] = dict(dec=_decode(X, cond, P), ccgp=_ccgp(X, cond, P, N),
                             ps=_ps(means, P, N))
    return dict(dec=np.array(dec), ccgp=np.array(ccgp_), ps=np.array(ps_),
                sd=float(np.mean(dec)), named=out_named)


def geometry_by_tss(out, key="h", ks=range(-5, 16), seed=0, prev=True):
    """CCGP and PS for the three named variables, against trials since the
    switch -- the inference-dynamics panels (C and D).

    NEGATIVE k = trials BEFORE a switch, i.e. the settled end of the previous
    block, labelled with that block's context.  Including them matters: without
    them the panel only shows the abstraction RECOVERING, and you cannot see
    that it was high beforehand and collapsed.  Same x-range as the behaviour
    figure, so the two can be read together.
    """
    X = out[key].reshape(-1, out[key].shape[-1]).numpy()
    cond = _cond_ids(out, prev)
    tss = out["tss"].ravel()
    B, T = out["tss"].shape
    T2 = out["tss"].reshape(B, T).astype(int)
    named = _named()
    res = {nm: {"ccgp": [], "ps": []} for nm in named}
    for k in ks:
        if k >= 0:
            sel = tss == k
        else:
            # trial t is |k| trials before a switch if t+|k| starts a block.
            # np.roll wraps, so mask the final |k| columns of each sequence.
            nxt = np.roll(T2, -abs(k), axis=1)
            valid = np.ones((B, T), bool)
            valid[:, T - abs(k):] = False
            sel = ((nxt == 0) & valid).ravel()
        Xk, ck = X[sel], cond[sel]
        ok = len(np.unique(ck)) == 8 and sel.sum() > 100
        means = (np.stack([Xk[ck == c].mean(0) for c in range(8)])
                 if ok else None)
        for nm, (P, N) in named.items():
            res[nm]["ccgp"].append(_ccgp(Xk, ck, P, N) if ok else np.nan)
            res[nm]["ps"].append(_ps(means, P, N) if ok else np.nan)
    return np.array(list(ks)), res


# ── figures ──────────────────────────────────────────────────────────────────
def _despine(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def fig_behaviour(res, path, seed=0):
    """FIGURE 1.  Left: how long MTL takes to infer the context after a switch.
    Right: the Bernardi Fig-1C signature.

    Both arms and all three ablation conditions.  The claim is the 2x2: zeroing
    the RPE should flatten `main` and leave `control` alone, because `control`
    is handed the outcome directly and does not need the error.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for arm, per_mode in res.items():
        for mode, r in per_mode.items():
            if r is None:
                continue
            ls = {"intact": "-", "zero": "--", "shuffle": ":"}[mode]
            ax[0].plot(r["ks"], r["dec"], ls, color=ACOL[arm], lw=1.9,
                       marker="o", ms=3,
                       label=f"{arm} / {mode}" if mode == "intact" else None)
            if r["fig1c"] is not None:
                ax[1].plot([1, 2, 3, 4], [r["fig1c"][k] for k in (1, 2, 3, 4)],
                           ls, color=ACOL[arm], lw=1.9, marker="s", ms=6)
    ax[0].axvline(0, ls=":", color="#e67e22", lw=1.5)
    ax[0].axhline(.5, ls="--", color="#888", lw=1)
    ax[0].set_xlabel("trials relative to context switch")
    ax[0].set_ylabel("context decode accuracy")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_title("how fast does MTL infer the new context?\n"
                    "decoded from $z$; chance 0.5", fontsize=10)
    ax[1].axhline(.5, ls="--", color="#888", lw=1)
    ax[1].set_xticks([1, 2, 3, 4]); ax[1].set_ylim(0, 1.03)
    ax[1].set_xlabel("stimulus appearance order after the switch")
    ax[1].set_ylabel("operant accuracy, first appearance")
    ax[1].set_title("inference signature (cf. Bernardi Fig 1C)\n"
                    "order 1 = stale context; 2-4 never seen in the new one",
                    fontsize=10)
    from matplotlib.lines import Line2D
    h = ([Line2D([], [], color=ACOL[a], lw=2, label=a) for a in ACOL] +
         [Line2D([], [], color="#666", ls=l, lw=1.8, label=m)
          for m, l in (("intact", "-"), ("$\\delta$ zeroed", "--"),
                       ("$\\delta$ shuffled", ":"))])
    fig.legend(handles=h, ncol=5, frameon=False, fontsize=9,
               loc="upper center", bbox_to_anchor=(.5, 1.0))
    for a in ax:
        _despine(a)
    fig.suptitle(f"SRO task (Bernardi et al. 2020): context inferred from a "
                 f"REWARD prediction error, seed {seed}", fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {path}")


def fig_decode_populations(res, path, seed=0):
    """Context decoding against trials-since-switch, for BOTH populations.

    h is MTL's 256-d recurrent state, z the 64-d readout CB consumes.  Plotting
    them together answers a question the geometry panels cannot: does the
    hidden state track the context faster than the code it emits, or are they
    locked together?  In the motor task the two came apart -- z pulled context
    into its dominant components while h kept it mixed with kinematics -- so it
    is worth checking rather than assuming they agree here.
    """
    fig, ax = plt.subplots(1, len(res), figsize=(6.2 * len(res), 4.6),
                           squeeze=False)
    ax = ax[0]
    KCOL = {"h": "#7b3294", "z": "#2c7fb8"}
    for i, (arm, per_key) in enumerate(res.items()):
        for key, per_mode in per_key.items():
            for mode, r in per_mode.items():
                if r is None:
                    continue
                ls = {"intact": "-", "zero": "--"}[mode]
                ax[i].plot(r["ks"], r["dec"], ls, color=KCOL[key], lw=1.9,
                           marker="o", ms=3,
                           label=f"{key}, {mode}" if i == 0 else None)
        ax[i].axvline(0, ls=":", color="#e67e22", lw=1.5)
        ax[i].axhline(.5, ls="--", color="#888", lw=1)
        ax[i].set_ylim(0, 1.03)
        ax[i].set_xlabel("trials relative to context switch")
        ax[i].set_ylabel("context decode accuracy" if i == 0 else "")
        ax[i].set_title(arm, fontsize=11)
        _despine(ax[i])
    ax[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"context decoding from MTL's hidden state (h) vs its readout "
                 f"(z), seed {seed}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def fig_geometry(g, ks, gt, path, arm="main", key="h", seed=0):
    """FIGURE 2.  Bernardi's geometry panels on the model's MTL hidden state.

    A  decoding and CCGP for all 35 dichotomies, the three named ones highlighted
    B  parallelism score, same
    C  CCGP vs trials since switch -- the inference dynamics
    D  PS vs trials since switch
    E  shattering dimensionality (mean decode over all 35) against context CCGP

    SD is a property of the whole population, not of a variable: it is the mean
    decodability over every dichotomy.  High SD with high CCGP is the
    combination Bernardi reports as surprising, because they normally trade off.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.42, wspace=.32)
    axA, axB, axE = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                     fig.add_subplot(gs[:, 2]))
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    for i, (vals, lab) in enumerate(((g["dec"], "decoding"), (g["ccgp"], "CCGP"))):
        axA.scatter(np.full(len(vals), i) + rng.normal(0, .05, len(vals)), vals,
                    s=16, facecolor="none" if i == 0 else "#888",
                    edgecolor="#888", lw=.8, zorder=1)
        for nm, c in VCOL.items():
            axA.scatter([i], [g["named"][nm]["dec" if i == 0 else "ccgp"]],
                        s=90, color=c, edgecolor="k", lw=.6, zorder=3,
                        label=VLAB[nm] if i == 0 else None)
    axA.axhline(.5, ls="--", color="#888", lw=1)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["decoding", "CCGP"])
    axA.set_ylabel("accuracy"); axA.set_ylim(.15, 1.02)
    axA.set_title("A  decoding & CCGP\n(all 35 dichotomies)", fontsize=10)
    axA.legend(frameon=False, fontsize=8)
    axB.scatter(rng.normal(0, .05, len(g["ps"])), g["ps"], s=16, color="#888")
    for nm, c in VCOL.items():
        axB.scatter([0], [g["named"][nm]["ps"]], s=90, color=c, edgecolor="k",
                    lw=.6, zorder=3)
    axB.axhline(0, ls="--", color="#888", lw=1)
    axB.set_xticks([]); axB.set_ylabel("parallelism score")
    axB.set_title("B  parallelism score", fontsize=10)
    for a_, m, ttl in ((axC, "ccgp", "C  CCGP over time"),
                       (axD, "ps", "D  PS over time")):
        for nm, c in VCOL.items():
            a_.plot(ks, gt[nm][m], "-o", ms=3, color=c, label=VLAB[nm])
        a_.axhline(.5 if m == "ccgp" else 0, ls="--", color="#888", lw=1)
        a_.axvline(0, ls=":", color="#e67e22", lw=1.5)
        a_.set_xlabel("trials relative to context switch")
        a_.set_ylabel(m.upper()); a_.set_title(ttl, fontsize=10)
    axE.bar([0], [g["sd"]], color="#cccccc", width=.6)
    axE.scatter([1] * 3, [g["named"][nm]["ccgp"] for nm in VCOL],
                s=90, c=[VCOL[nm] for nm in VCOL], edgecolor="k", lw=.6)
    axE.axhline(.5, ls="--", color="#888", lw=1)
    axE.set_xticks([0, 1]); axE.set_xticklabels(["SD", "CCGP"])
    axE.set_ylim(0, 1.02)
    axE.set_title(f"E  shattering dim & abstraction\nSD = {g['sd']:.2f}",
                  fontsize=10)
    for a_ in (axA, axB, axC, axD, axE):
        _despine(a_)
    fig.suptitle(f"model {arm} arm, {key} -- geometry of abstraction "
                 f"(cf. Bernardi et al. 2020 Fig 3: PREVIOUS-trial variables), "
                 f"seed {seed}", fontsize=11.5)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--key", default="h", choices=["h", "z"],
                    help="population for the geometry: h = MTL hidden state "
                         "(primary, the HPC analog), z = the 64-d readout")
    a = ap.parse_args()
    if not (a.train or a.figures):
        ap.error("nothing to do: pass --train and/or --figures")
    for s in a.seeds:
        if a.train:
            for arm in a.arms:
                train(arm, s)
        if a.figures:
            res = {}
            for arm in a.arms:
                res[arm] = {}
                for mode in ("intact", "zero", "shuffle"):
                    o = collect(arm, s, e_mode=mode)
                    if o is None:
                        res[arm][mode] = None; continue
                    ks, dec = decode_by_tss(o)
                    res[arm][mode] = dict(ks=ks, dec=dec,
                                          fig1c=fig1c(arm, s, e_mode=mode))
                    print(f"    [{arm}/{mode}] context decode at tss>=5: "
                          f"{np.nanmean(dec[ks >= 5]):.3f}", flush=True)
            d = fig_dir(s)
            fig_behaviour(res, os.path.join(d, "sro_behaviour.png"), s)
            # the same decode, from both populations
            byk = {}
            for arm in a.arms:
                byk[arm] = {}
                for key in ("h", "z"):
                    byk[arm][key] = {}
                    for mode in ("intact", "zero"):
                        o = collect(arm, s, e_mode=mode)
                        if o is None:
                            byk[arm][key][mode] = None; continue
                        ks_, dec_ = decode_by_tss(o, key=key)
                        byk[arm][key][mode] = dict(ks=ks_, dec=dec_)
            fig_decode_populations(byk,
                                   os.path.join(d, "sro_decode_h_vs_z.png"), s)
            for arm in a.arms:
                o = collect(arm, s)
                if o is None:
                    continue
                g = geometry(o, key=a.key)
                ks2, gt = geometry_by_tss(o, key=a.key)
                np.savez(os.path.join(d, f"sro_geometry_{arm}_{a.key}.npz"),
                         dec=g["dec"], ccgp=g["ccgp"], ps=g["ps"], sd=g["sd"],
                         **{f"{nm}|{m}": g["named"][nm][m]
                            for nm in VCOL for m in ("dec", "ccgp", "ps")})
                fig_geometry(g, ks2, gt,
                             os.path.join(d, f"sro_geometry_{arm}_{a.key}.png"),
                             arm, a.key, s)
                print(f"    [{arm}] SD {g['sd']:.3f} | context CCGP "
                      f"{g['named']['context']['ccgp']:.3f} "
                      f"PS {g['named']['context']['ps']:.3f}", flush=True)
