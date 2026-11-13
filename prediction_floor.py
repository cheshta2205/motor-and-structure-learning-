"""
prediction_floor.py -- what is the FORWARD MODEL's error floor, and how much of
it is context ignorance?

The prediction task reports one number, held-out velocity MSE (model.track,
figures.collect), and on its own that number is uninterpretable. 0.0139 for the
intact main circuit is good or bad only against two references that the task
itself fixes:

  the NOISE FLOOR    what nothing can beat.  The plant injects command noise
                     (task.CMD_NOISE = 0.28 of |a|, before A is applied), so
                     even a predictor that knows the context exactly and runs
                     the true plant forward still misses by that much.
  the CONTEXT-FREE   what a cerebellum ALONE can reach -- a forward model given
  FLOOR              (s_t, a_t) and no context signal at all.  Every ablation
                     in figures.COND is an attempt to push the circuit toward
                     this, and none of them reaches it, because EC still sees
                     s_{t-1} and a_{t-1} and can infer the context without the
                     cerebellar error (see "WHY e-ZEROED IS NOT CONTEXT-FREE").

The gap between them is the HEADROOM: the most that knowing which context is
active can buy on this metric.  It is the prediction task's analogue of
blind_ceiling.py's specialist-minus-generalist, and it exists for the same
reason -- to stop a comparison being reported on a metric with no room in it.

THE LADDER, all in the units model.track and figures report (velocity scaled
by task.VEL_SCALE, mean over both components, steps [:-1]):

  carry        v' = v.  The zero-effort reference: velocity's own persistence.
  affine       the best affine map (s_t, a_t) -> v', in closed form.  Not a
               floor, a sanity rung -- the plant is affine in (v, a) WITHIN a
               context, but a mixture over contexts is not, so this lands
               ABOVE the context-free floor and that is the expected sign.
  cb_blind     CB trained with z == 0.  THE CEREBELLUM-ONLY MODEL, and the
               context-free floor for the actual architecture.
  cb_prior     analytic: the noiseless plant's prediction AVERAGED over the
               context pool.  This is the Bayes-optimal context-free
               prediction when (s_t, a_t) says nothing about which context is
               active.  Since it does say a little -- a drift context shifts
               the velocity marginal, a curl context correlates v with its own
               perpendicular -- cb_blind can come in BELOW cb_prior, and the
               difference is how much context leaks into a single observation.
               Compare the two to check cb_blind actually converged: a
               cb_blind far ABOVE cb_prior is an undertrained network, not a
               floor.
  cb_oracle    CB trained with z == one-hot(context): perfect context, same
               network, same loss.  The best the forward model can do with the
               context handed to it, and therefore the target the circuit is
               being asked to approach.
  plant        the noiseless plant under the TRUE context.  Command noise
               alone.  No predictor can beat this; cb_oracle sits above it by
               whatever the network cannot represent.

WHY e-ZEROED IS NOT CONTEXT-FREE, which is the reason this file exists.  In
the main arm EC's input is [s_{t-1}, a_{t-1}, e_t].  Zeroing e_t removes the
cerebellar error, but EC is recurrent and still receives the state and action
stream, from which the context is recoverable directly -- consecutive states
give the realised force, the action gives what was commanded.  So "e zeroed"
measures the loss of ONE route to the context, not the absence of context, and
it lands below the context-free floor.  Reading it as "the circuit without
context" overstates what the cerebellar channel contributes.

WHAT TO DO WITH THE NUMBERS.  Two derived quantities, printed at the end:

  headroom  = cb_blind - cb_oracle     the room the metric has
  realised  = (cb_blind - intact) / headroom

`realised` is the fraction of the available context benefit the circuit
actually delivers, on [0, 1].  It is the number to report when asked "how good
is 0.0139" -- and the number to check before running an ablation, since an
ablation can only move within the headroom.

MEASURED, main arm, seed 0, gp actions, 12 contexts (2026-10-13):

                       all  settled
  carry              0.0574   0.0567
  affine             0.0346   0.0351
  cb_blind           0.0227   0.0224     <- CEREBELLUM ONLY
  cb_prior           0.0224   0.0223
  cb_oracle          0.0068   0.0068
  plant              0.0056   0.0055
  circuit intact     0.0137   0.0100     0.56 / 0.79 of the headroom
  circuit e-zeroed   0.0206   0.0189     0.14 / 0.23
  circuit e-shuffled 0.0254   0.0240    -0.17 / -0.10
  circuit MTL noise  0.0167   0.0136     0.37 / 0.57

FOUR THINGS THOSE NUMBERS SAY.

1. cb_blind AND cb_prior AGREE to 1.3%.  The trained cerebellum-only network
   lands on the prior-mean prediction, which means a single (s_t, a_t) carries
   essentially NO information about which context is active -- exactly the
   property task.py built the families for ("both are norm-preserving and
   zero-mean over actions, so no statistic of the observations alone identifies
   either").  It also means 0.0224 is a real floor and not an optimisation
   artefact: two independent routes reach it.

2. THE ARCHITECTURE IS NOT THE BINDING CONSTRAINT.  cb_oracle - plant = 0.0012,
   under 8% of the headroom.  Handed the context as a one-hot, CB reproduces
   the plant's force law to within a tenth of what context ignorance costs.  So
   every interesting number here is about the CONTEXT SIGNAL, not about CB's
   capacity, and a figure that improves prediction is reporting better context
   inference rather than a better forward model.

3. e-ZEROED IS NOT CONTEXT-FREE, by 0.0021 -- it sits BELOW the floor a
   context-free predictor can reach.  See the note below on why.  Reporting it
   as "the circuit without context" overstates the cerebellar channel: the
   channel is worth 0.56 - 0.14 = 0.42 of the headroom, not all of it.

4. e-SHUFFLED IS WORSE THAN KNOWING NOTHING, by 0.0027 (a negative realised
   fraction).  A permuted error is not an absent error, it is a confidently
   WRONG one, and the circuit acts on it.  That is a stronger statement than
   "shuffling hurts" and worth making as one.

    python prediction_floor.py                 # the ladder, main arm
    python prediction_floor.py --arm control
    python prediction_floor.py --quick         # fewer sequences/epochs, for a smoke test
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

import model as M
import task as T
from data import generate_sequences

# Defaults. SMALLER THAN model.train's: CB is feedforward and the map it has to
# fit here has six inputs, so it converges on a fraction of the data the
# recurrent circuit needs. Checked by raising both -- cb_blind and cb_oracle
# move in the fourth decimal between (2048, 120) and (8192, 256), which is
# inside the seed-to-seed spread of the intact numbers they are compared to.
# The ablation modes model.rollout accepts, in the order figures.COND lists
# them. Named here rather than imported from figures, which pulls matplotlib
# and the paper style in with it.
ABLATIONS = ("intact", "zero", "shuffle", "mtl_noise")

N_TRAIN_SEQ = 2048
N_VA_SEQ = 300            # matches model.train's validation set exactly
EPOCHS = 120
BATCH = 4096
LR, WD = 1e-3, 1e-4


# ── the data, once ───────────────────────────────────────────────────────────
def sequences(n, seed, spec):
    """(trans, context_id, family, time-since-switch) for one draw.

    The SPEC goes to the generator, exactly as model.train does it: the action
    process and the fitted context pool are properties of the circuit being
    characterised, and a floor computed on a different action distribution is
    a floor for a different task.
    """
    trans, ctx, tss = generate_sequences(n, seed=seed, action_mode=spec)
    return (trans, ctx[:, :, 0].numpy().astype(int),
            ctx[:, :, 1].numpy().astype(int), tss.numpy())


def vel_mse(pred_raw, trans, mask=None):
    """The reported metric: velocity error in VEL_SCALE units, steps [:-1].

    `pred_raw` is next-velocity in RAW plant units, matching trans; the scaling
    happens here so the analytic predictors can stay in plant units and the
    network predictors (which work in scaled units) divide once on the way in.
    """
    tgt = trans[:, :, 8:10].numpy()
    e = ((pred_raw[:, :-1] - tgt[:, :-1]) * T.VEL_SCALE) ** 2
    per_step = e.mean(-1)
    return float(per_step.mean() if mask is None else per_step[mask].mean())


# ── analytic rungs ───────────────────────────────────────────────────────────
def plant_predictions(trans, pool):
    """[len(pool), B, T, 2]: the NOISELESS next velocity under every context in
    the pool, from each step's actual (pos, vel, action).

    task.env_step_ab is shape-polymorphic, so one call does the whole array per
    context -- and it is the plant itself, not a copy of it, which is the
    mistake blind_ceiling.py and structure.py each made once.
    """
    pos = trans[:, :, 0:2].numpy()
    vel = trans[:, :, 2:4].numpy()
    act = trans[:, :, 4:6].numpy()
    out = np.empty((len(pool),) + vel.shape, np.float32)
    for i, c in enumerate(pool):
        A, B, w = T.context_affine(c)
        _, vn, _ = T.env_step_ab(pos, vel, act, A, B, rng=None,
                                 drift=(w if w.any() else None))
        out[i] = vn
    return out


def affine_fit(tr_trans, va_trans):
    """Least-squares affine map (pos, vel, act, 1) -> next velocity, fitted on
    the training set and scored on validation.  Closed form, no training."""
    def design(trans):
        obs, act, _ = M.build(trans)
        x = torch.cat([obs, act, torch.ones_like(act[..., :1])], -1)
        return x.numpy().reshape(-1, x.shape[-1])
    Xtr, Xva = design(tr_trans), design(va_trans)
    Ytr = (tr_trans[:, :, 8:10].numpy() * T.VEL_SCALE).reshape(-1, 2)
    W, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    return (Xva @ W).reshape(va_trans.shape[0], va_trans.shape[1], 2) / T.VEL_SCALE


# ── the trained rungs ────────────────────────────────────────────────────────
def _z_zero(cid):
    return torch.zeros(cid.shape + (T.LATENT_DIM,))


def _z_onehot(cid, pool):
    """One-hot context in the first len(pool) slots of a LATENT_DIM vector.

    PERFECT CONTEXT, not a better one: the slots carry no geometry, so the
    network has to learn each context's force law separately -- which is the
    right control, since the question is what the context IDENTITY is worth,
    not what a well-shaped code is worth.
    """
    idx = {c: i for i, c in enumerate(pool)}
    z = torch.zeros(cid.shape + (T.LATENT_DIM,))
    flat = np.vectorize(idx.get)(cid).reshape(-1)
    z.view(-1, T.LATENT_DIM)[np.arange(flat.size), flat] = 1.0
    return z


def fit_cb(tr_trans, tr_z, va_trans, va_z, epochs=EPOCHS, batch=BATCH,
           seed=0, label="", log_every=20):
    """Train a CB alone on (s_t, a_t, z_t) -> s_{t+1}, with model.weighted_loss.

    FEEDFORWARD MEANS THE SEQUENCE STRUCTURE IS IRRELEVANT: CB has no
    recurrence (see model.CB), so a step is an independent sample and the whole
    thing is an MLP regression over flattened steps.  That is why this trains
    in seconds while model.train takes hours -- and why the floor it reports is
    a property of the task, not of an optimisation that might not have run long
    enough.

    THE SAME LOSS as the circuit, deliberately.  model.weighted_loss upweights
    the velocity half by ALPHA on top of build()'s per-channel scaling; a floor
    fitted on plain velocity MSE would be a floor for a different objective
    than the one the intact number comes from.
    """
    torch.manual_seed(seed)
    cb = M.CB(M.CB_ACTIVATION)
    obs, act, tgt = M.build(tr_trans)
    X = torch.cat([obs, act, tr_z], -1).reshape(-1, obs.shape[-1] + act.shape[-1] + T.LATENT_DIM)
    Y = tgt.reshape(-1, tgt.shape[-1])
    vo, va_, vt = M.build(va_trans)
    Xv = torch.cat([vo, va_, va_z], -1)

    opt = torch.optim.AdamW(cb.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = X.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = cb.net(X[idx])
            # weighted_loss on a flat batch: it slices [:, :-1] off a [B, T, D]
            # array, which a flat batch has no analogue of, so the formula is
            # applied here directly. SAME formula -- if ALPHA or DT moves there
            # it must move here, which is why both are read from the module.
            e = (pred - Y[idx]) ** 2
            loss = (e[..., :2] / T.DT ** 2).mean() + M.ALPHA * e[..., 2:].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if log_every and (ep % log_every == 0 or ep == 1):
            with torch.no_grad():
                p = cb.net(Xv)
            m = vel_mse(p[..., 2:].numpy() / T.VEL_SCALE, va_trans)
            print(f"    [{label}] ep {ep:3d}  held-out vel MSE {m:.5f}", flush=True)
    with torch.no_grad():
        p = cb.net(Xv)
    return p[..., 2:].numpy() / T.VEL_SCALE


# ── the context-free cerebellum's TRAINING CURVE (2026-10-23) ────────────────
#
# For the circuit figure's training panel: the same cerebellum with its context
# input clamped to zero, trained EXACTLY as model.train trains the circuit --
# the same training and validation sequences for the seed, batches of
# M.BATCH sequences, M.EPOCHS epochs, AdamW at the same rate, the same warm
# restarts, the same loss -- and scored each epoch on the same held-out
# metric (model.train's hist). Only the context differs, so the gap between
# the two curves is what context bought, epoch by epoch. Not a floor: the
# floor (fit_cb on a single cosine decay) is the asymptote this approaches.
def blind_track_path(spec, seed):
    return os.path.join(M.FIG_DIR, f"prediction_floor_blind_track_{spec}_seed{seed}.npz")


def blind_track(seed, action_mode=M.DEFAULT_ACTION_MODE, epochs=None):
    spec = M.circuit(action_mode)
    epochs = epochs or M.EPOCHS
    torch.manual_seed(seed)
    trans, _, _ = generate_sequences(M.N_TRAIN, seed=seed, action_mode=spec)
    va, _, _ = generate_sequences(300, seed=seed + 500, action_mode=spec)
    obs, act, tgt = M.build(trans)
    z0 = torch.zeros(*obs.shape[:2], T.LATENT_DIM)
    X = torch.cat([obs, act, z0], -1)                       # [N, T, D], per sequence
    vo, va_, _ = M.build(va)
    Xv = torch.cat([vo, va_, torch.zeros(*vo.shape[:2], T.LATENT_DIM)], -1)
    cb = M.CB(M.CB_ACTIVATION)
    opt = torch.optim.AdamW(cb.parameters(), lr=M.LR, weight_decay=M.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=M.LR_RESTART_T0, T_mult=M.LR_RESTART_TMULT, eta_min=1e-5)
    hist = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(M.N_TRAIN)
        for i in range(0, M.N_TRAIN, M.BATCH):
            idx = perm[i:i + M.BATCH]
            pred = cb.net(X[idx])
            # model.weighted_loss's formula, on [B, T, D]: steps [:-1]
            e = (pred[:, :-1] - tgt[idx][:, :-1]) ** 2
            loss = (e[..., :2] / T.DT ** 2).mean() + M.ALPHA * e[..., 2:].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        with torch.no_grad():
            p = cb.net(Xv)
        hist.append(vel_mse(p[..., 2:].numpy() / T.VEL_SCALE, va))
        if ep % 32 == 0 or ep == 1:
            print(f"    [blind track {seed}] ep {ep:3d}  held-out vel MSE {hist[-1]:.5f}",
                  flush=True)
    path = blind_track_path(spec, seed)
    np.savez(path, hist=np.array(hist), seed=np.array(seed), epochs=np.array(epochs))
    print(f"  blind track -> {path}")
    return np.array(hist)


def load_blind_tracks(spec):
    """[n_seed, n_epoch] context-free training curves, or None if none run."""
    paths = sorted(glob.glob(os.path.join(M.FIG_DIR, f"prediction_floor_blind_track_{spec}_seed*.npz")))
    if not paths:
        return None
    H = [np.load(p)["hist"] for p in paths]
    n = min(len(h) for h in H)
    return np.array([h[:n] for h in H], float)


# ── the measured circuit, for the comparison ─────────────────────────────────
def measured(arm, action_mode=M.DEFAULT_ACTION_MODE):
    """{mode: (mean, sd, n)} over seeds, from the decoding.npz figures.per_seed
    writes.  Empty when none are on disk -- the ladder is still meaningful
    without them, it just has nothing to place inside it.

    CHECK THE SEED COUNT in the printout: these files live in a Dropbox folder
    that has been observed to truncate and restore them mid-session.

    BUILT FROM FIG_DIR AND arm_tag RATHER THAN model.fig_dir, which creates the
    directory it names -- a read-only report must not leave an empty
    figures/<arm>/seed0 behind when there is nothing on disk to read.
    """
    pat = os.path.join(M.FIG_DIR, M.arm_tag(arm, action_mode),
                       "seed*", "decoding.npz")
    out = {}
    for p in sorted(glob.glob(pat)):
        try:
            d = np.load(p)
        except Exception:
            continue
        for k in d.files:
            if k.endswith("|mse"):
                out.setdefault(k.split("|")[0], []).append(float(d[k]))
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in out.items()}


# ── the floors across DATA DRAWS ─────────────────────────────────────────────
def floor_spread(seeds, n_train=N_TRAIN_SEQ, n_va=N_VA_SEQ, epochs=EPOCHS,
                 action_mode=M.DEFAULT_ACTION_MODE, oracle=True):
    """The floors re-measured on independent draws of the task, one per seed.

    WHAT THE ERROR BAR IS, and it is not the usual one. Every other +- in this
    repo is across TRAINING SEEDS of a network. These floors have no training
    seed worth varying -- they are properties of the task -- so the spread here
    is across DRAWS OF THE SEQUENCES: a fresh training set and a fresh
    validation set per seed, with the fit re-run on each. It answers "how well
    is this floor pinned down", which is the question a reader has when the
    circuit is said to end below it.

    The circuit_* rows are deliberately absent: those ARE per-checkpoint, they
    have their own across-seed numbers in the saved decoding.npz, and averaging
    the two kinds of seed into one number would be a category error.
    """
    spec = M.circuit(action_mode)
    pool = list(spec.context_ids or range(T.N_CONTEXTS))
    keys = ["carry", "affine", "cb_blind", "cb_prior"] + (["cb_oracle"] if oracle else []) + ["plant"]
    acc = {k: {"all": [], "settled": []} for k in keys}
    for sd in seeds:
        print(f"\n[draw {sd}]", flush=True)
        tr_trans, tr_cid, _, _ = sequences(n_train, sd, spec)
        va_trans, va_cid, _, va_tss = sequences(n_va, sd + 500, spec)
        preds = plant_predictions(va_trans, pool)
        true_i = np.vectorize({c: i for i, c in enumerate(pool)}.get)(va_cid)
        rungs = {
            "carry": va_trans[:, :, 2:4].numpy(),
            "affine": affine_fit(tr_trans, va_trans),
            "cb_blind": fit_cb(tr_trans, _z_zero(tr_cid), va_trans, _z_zero(va_cid),
                               epochs=epochs, seed=sd, label=f"cb_blind/{sd}",
                               log_every=epochs),
            "cb_prior": preds.mean(0),
            "plant": np.take_along_axis(preds, true_i[None, ..., None], 0)[0],
        }
        if oracle:
            rungs["cb_oracle"] = fit_cb(tr_trans, _z_onehot(tr_cid, pool), va_trans,
                                        _z_onehot(va_cid, pool), epochs=epochs,
                                        seed=sd, label=f"cb_oracle/{sd}",
                                        log_every=epochs)
        settled = va_tss[:, :-1] >= T.TSS_MIN
        for k in keys:
            acc[k]["all"].append(vel_mse(rungs[k], va_trans))
            acc[k]["settled"].append(vel_mse(rungs[k], va_trans, settled))
    return {k: {c: np.array(v, float) for c, v in d.items()} for k, d in acc.items()}


def report_spread(sp, seeds):
    n = len(seeds)
    print("\n" + "\u2500" * 72)
    print(f"{'':12}{'all steps':>22}{'settled steps':>22}   over {n} independent draws")
    for k, v in sp.items():
        a, s_ = v["all"], v["settled"]
        print(f"{k:12}{a.mean():>13.5f} +-{a.std(ddof=1):8.5f}"
              f"{s_.mean():>13.5f} +-{s_.std(ddof=1):8.5f}   {DESC.get(k, k)}")
    print("\u2500" * 72)


# ── the cache the figures read ───────────────────────────────────────────────
#
# NOT PER ARM, and not per seed. cb_blind, cb_prior, cb_oracle and plant are all
# CB-ONLY -- no EC anywhere in them -- so the arm, which is nothing but EC's
# input width, cannot move them, and the sequence draw is the only thing a seed
# changes. What they DO depend on is everything a CircuitSpec names: the action
# process, and the contexts in the pool. So the spec is the filename, and the
# arm and seed that happened to measure it are provenance fields inside.
def floor_path(spec):
    return os.path.join(M.FIG_DIR, f"prediction_floor_{spec}.npz")


def save_floors(spec, mse, mse_settled, arm, seed, n_train, epochs):
    """Write the floors for figures.floor_lines. ONLY THE FLOORS: the circuit_*
    rows are a property of one checkpoint, and a figure that drew them as
    reference lines would be comparing a seed to itself."""
    path = floor_path(spec)
    np.savez(path,
             rungs=np.array(ORDER),
             all=np.array([mse[k] for k in ORDER]),
             settled=np.array([mse_settled[k] for k in ORDER]),
             arm=np.array(arm), seed=np.array(seed),
             n_train=np.array(n_train), epochs=np.array(epochs),
             tss_min=np.array(T.TSS_MIN))
    print(f"\n  floors -> {path}")
    return path


def floor_spread_path(spec):
    return os.path.join(M.FIG_DIR, f"prediction_floor_spread_{spec}.npz")


def save_floor_spread(spec, sp, seeds):
    """Write the floors' spread across independent task draws (floor_spread),
    for the bands figures.floor_lines draws (user, 2026-10-23)."""
    keys = list(sp)
    path = floor_spread_path(spec)
    np.savez(path, rungs=np.array(keys),
             all=np.array([sp[k]["all"] for k in keys]),
             settled=np.array([sp[k]["settled"] for k in keys]),
             seeds=np.array(seeds))
    print(f"\n  floor spread -> {path}")
    return path


def load_floor_spread(spec):
    """{rung: (all_draws, settled_draws)} arrays, or None when not measured."""
    path = floor_spread_path(spec)
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    return {str(k): (np.asarray(a, float), np.asarray(s_, float))
            for k, a, s_ in zip(z["rungs"], z["all"], z["settled"])}


def load_floors(spec):
    """{rung: (all, settled)} from the cache, or None when it has not been run.

    Returns None rather than raising: a figure must still render without it,
    the way fig_epochs renders without a track.npz.
    """
    path = floor_path(spec)
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    return {str(k): (float(a), float(s)) for k, a, s
            in zip(z["rungs"], z["all"], z["settled"])}


# ── driver ───────────────────────────────────────────────────────────────────
def ladder(arm="main", seed=0, n_train=N_TRAIN_SEQ, n_va=N_VA_SEQ,
           epochs=EPOCHS, action_mode=M.DEFAULT_ACTION_MODE):
    spec = M.circuit(action_mode)
    pool = list(spec.context_ids or range(T.N_CONTEXTS))
    print(f"arm {arm}  seed {seed}  {len(pool)} contexts  "
          f"{n_train} train / {n_va} val sequences  {epochs} epochs")
    tr_trans, tr_cid, _, _ = sequences(n_train, seed, spec)
    # THE SAME VALIDATION SET model.train scores on: seed + 500, 300 sequences.
    # A floor measured on different sequences than the intact number it is
    # compared to would differ by sampling alone.
    va_trans, va_cid, va_fam, va_tss = sequences(n_va, seed + 500, spec)

    preds = plant_predictions(va_trans, pool)
    true_i = np.vectorize({c: i for i, c in enumerate(pool)}.get)(va_cid)
    p_true = np.take_along_axis(preds, true_i[None, ..., None], 0)[0]
    p_prior = preds.mean(0)

    rungs = {
        "carry": va_trans[:, :, 2:4].numpy(),
        "affine": affine_fit(tr_trans, va_trans),
        "cb_blind": fit_cb(tr_trans, _z_zero(tr_cid), va_trans, _z_zero(va_cid),
                           epochs=epochs, seed=seed, label="cb_blind"),
        "cb_prior": p_prior,
        "cb_oracle": fit_cb(tr_trans, _z_onehot(tr_cid, pool), va_trans,
                            _z_onehot(va_cid, pool), epochs=epochs, seed=seed,
                            label="cb_oracle"),
        "plant": p_true,
    }
    # THE CIRCUIT ITSELF, rolled out from its checkpoint, so the comparison does
    # not rest on a saved npz and so the SETTLED split below can be taken on it
    # too. The multi-seed decoding.npz numbers still print beside it; this row
    # is the one seed, all four ablation modes, measured here and now.
    circ = circuit_rungs(arm, seed, va_trans, action_mode)
    rungs.update(circ)

    settled = va_tss[:, :-1] >= T.TSS_MIN
    mse = {k: vel_mse(v, va_trans) for k, v in rungs.items()}
    mse_settled = {k: vel_mse(v, va_trans, settled) for k, v in rungs.items()}
    per_fam = {k: {f: vel_mse(v, va_trans, va_fam[:, :-1] == i)
                   for i, f in enumerate(T.FAM_NAMES)} for k, v in rungs.items()}
    rows = [f"circuit_{m}" for m in ABLATIONS if f"circuit_{m}" in circ]
    return (mse, mse_settled, per_fam, measured(arm, action_mode),
            float(settled.mean()), rows)


def circuit_rungs(arm, seed, va_trans, action_mode=M.DEFAULT_ACTION_MODE):
    """{f"circuit_{mode}": next-velocity prediction} for the trained checkpoint,
    or {} when there is none on disk.

    SAME ROLLOUT AS figures.collect, same ablation modes, same generator seed
    for the shuffle permutation -- the point is to put the circuit on exactly
    the rows above it, not to re-measure it slightly differently.
    """
    loaded = M.load(arm, seed, action_mode)
    if loaded is None:
        print(f"  no checkpoint for {arm} seed {seed}: the ladder prints "
              f"without the circuit's own rows")
        return {}
    ec, cb = loaded
    out = {}
    for mode in ABLATIONS:
        with torch.no_grad():
            _, pred, _ = M.rollout(ec, cb, va_trans, arm, mode,
                                   torch.Generator().manual_seed(0))
        out[f"circuit_{mode}"] = pred[..., 2:].numpy() / T.VEL_SCALE
        print(f"    [circuit] {mode} rolled out", flush=True)
    return out


DESC = {
    "carry": "v' = v, no model",
    "affine": "best affine (s,a) -> v'",
    "cb_blind": "CEREBELLUM ONLY (z = 0)",
    "cb_prior": "plant averaged over contexts",
    "cb_oracle": "CB + one-hot context",
    "plant": "true plant, true context: COMMAND NOISE",
    "circuit_intact": "the trained circuit, this seed",
    "circuit_zero": "  e zeroed",
    "circuit_shuffle": "  e shuffled",
    "circuit_mtl_noise": "  MTL noise",
}
ORDER = ["carry", "affine", "cb_blind", "cb_prior", "cb_oracle", "plant"]


def report(mse, mse_settled, per_fam, meas, settled_frac, circuit_rows):
    w = max(6, max(len(f) for f in T.FAM_NAMES))
    rule = "\u2500" * 86
    print("\n" + rule)
    head = "  ".join(f"{f:>{w}}" for f in T.FAM_NAMES)
    print(f"{'':18}{'all':>8}{'settled':>9}   {head}   what it is")
    for k in ORDER + circuit_rows:
        if circuit_rows and k == circuit_rows[0]:
            print(rule)
        fam = "  ".join(f"{per_fam[k][f]:>{w}.4f}" for f in T.FAM_NAMES)
        print(f"{k:18}{mse[k]:>8.5f}{mse_settled[k]:>9.5f}   {fam}   {DESC.get(k, k)}")
    print(rule)
    print(f"'settled' is tss >= TSS_MIN = {T.TSS_MIN}, {settled_frac:.0%} of the scored steps. "
          f"The analytic\nand z-free rungs are flat across it by construction -- "
          f"they have nothing to\nre-infer after a switch -- so the column is where "
          f"the circuit's inference cost\nshows up, and the honest comparison.")
    if meas:
        print(rule)
        for k in ABLATIONS:
            if k in meas:
                m, sd, n = meas[k]
                print(f"  {k:16}{m:>8.5f}{'':9}   "
                      f"{'':{(w + 2) * len(T.FAM_NAMES) - 2}}   saved decoding.npz, "
                      f"{n} seed{'s' if n != 1 else ''} (sd {sd:.5f})")
    print(rule)
    _derived(mse, "all steps")
    _derived(mse_settled, f"settled steps (tss >= {T.TSS_MIN})")


def _derived(m, which):
    """headroom, the network's own cost, the single-step leakage, and the
    fraction of the headroom each circuit condition realises."""
    room = m["cb_blind"] - m["cb_oracle"]
    print(f"\n{which}:")
    print(f"  headroom   cb_blind - cb_oracle   {room:.5f}"
          "   the most context identity can buy")
    print(f"  net cost   cb_oracle - plant      {m['cb_oracle'] - m['plant']:.5f}"
          "   what CB cannot represent")
    leak = m["cb_prior"] - m["cb_blind"]
    note = ("context recoverable from ONE (s,a)" if leak > 0 else
            "<= 0: cb_blind did not quite reach the analytic optimum, "
            "so this is the residual fit gap, not leakage")
    print(f"  leakage    cb_prior - cb_blind    {leak:+.5f}   {note}")
    if room <= 0:
        return
    for k in ABLATIONS:
        key = f"circuit_{k}"
        if key in m:
            print(f"  realised   {k:20} {(m['cb_blind'] - m[key]) / room:>6.2f}"
                  f"   of the headroom")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--arm", default="main", choices=list(M.ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=N_TRAIN_SEQ)
    ap.add_argument("--n-va", type=int, default=N_VA_SEQ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--spread", type=int, nargs="+", metavar="SEED",
                    help="re-measure the FLOORS on this many independent draws "
                         "of the task and report mean +- SD; no circuit rows; "
                         "saved to prediction_floor_spread_<spec>.npz for the "
                         "figures' floor bands (the point floors are untouched)")
    ap.add_argument("--blind-track", type=int, nargs="+", metavar="SEED",
                    help="train the context-free cerebellum on the circuit's own schedule "
                         "for these seeds and save its held-out curve for the figures")
    ap.add_argument("--no-oracle", action="store_true",
                    help="with --spread, skip the cb_oracle fit (halves the time)")
    ap.add_argument("--quick", action="store_true",
                    help="256 train sequences and 20 epochs: a smoke test, NOT a floor")
    a = ap.parse_args()
    if a.quick:
        a.n_train, a.epochs = 256, 20
    if a.blind_track:
        for sd in a.blind_track:
            blind_track(sd)
        return
    if a.spread:
        sp = floor_spread(a.spread, a.n_train, a.n_va, a.epochs, oracle=not a.no_oracle)
        report_spread(sp, a.spread)
        # the spread IS cached (2026-10-23), in its own file, for the floor
        # bands; the point floors in prediction_floor_<spec>.npz are untouched
        save_floor_spread(M.circuit(), sp, a.spread)
        return
    out = ladder(a.arm, a.seed, a.n_train, a.n_va, a.epochs)
    report(*out)
    # --quick DOES NOT WRITE. Its networks are visibly undertrained (cb_blind
    # lands above cb_prior), and a smoke test that overwrote the cache would put
    # wrong reference lines on every figure that reads it.
    if a.quick:
        print("\n  --quick: cache NOT written (these are not floors)")
    else:
        save_floors(M.circuit(), out[0], out[1], a.arm, a.seed, a.n_train, a.epochs)


if __name__ == "__main__":
    main()
