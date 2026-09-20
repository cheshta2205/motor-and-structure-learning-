"""
circuit_probe.py -- does the circuit's z actually distinguish the contexts
UNDER THE SAC ROLLOUT?

decode.py already answers "can context be read out of z" on the PRETRAINING
distribution: model.rollout over data.generate_sequences. That is the right
question for model.py and the wrong one for sac.py, because a frozen
checkpoint is only as useful as it is on the distribution it is actually
driven with. The two came apart badly once: sac.py fed the circuit raw
positions where model.build feeds per-channel-scaled ones, and z went from a 1.00
decode to 0.28 (chance 0.17) without anything raising an error. Both frozen
arms then behaved like the no-circuit baseline, which is the symptom that is
easy to misread as a result about the circuit rather than about the wiring.

So this module rolls out through sac.py's OWN functions -- circuit_state,
augmented_obs_step, cb_predict, sample_pos_goal, env_step_params -- rather
than reimplementing the loop. If sac.py drives the circuit wrongly, this
probe is driven wrongly in the same way and the number drops. A private copy
of the loop would keep reading 1.00 while training quietly read 0.28.

METHOD, following decode.py:
  multinomial logistic regression on z, K = len(contexts), chance 1/K,
  SPLIT BY EPISODE so no timestep leaks across (steps within an episode share
  a context and a trajectory; a timestep split inflates every number -- this
  bit us before, at a false 0.60 on a probe that was reading episode identity
  off the goal). Steps before TSS_MIN are dropped by default: early in an
  episode the context genuinely is not established yet, since e_t is zero on
  step 0 and EC's hidden state is freshly reset.

Two controls worth keeping in the output:
  raw obs   decoding [pos, vel, goal] must sit at chance. It is the check
            that the probe is not leaking episode identity.
  by t      accuracy against step number. A circuit that is INFERRING shows a
            rising curve; a flat one at chance-plus-epsilon means the state
            is not accumulating evidence at all.

  python circuit_probe.py --arms main control --ckpt-seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import model as M
import sac
import task as T
from decode import _clf, seq_split

N_EPISODES = 140          # per context; 6 contexts x 140 x 50 steps = 42k rows
N_SPLITS = 3
T_BINS = ((0, 5), (5, 15), (15, 30), (30, 50))


def rollout_z(arm, ckpt_seed, contexts=None, *, n_episodes=N_EPISODES,
              max_steps=None, seed=0, device="cpu"):
    """Drive a frozen checkpoint through sac.py's rollout and collect z.

    Actions are uniform random -- what sac.run_sac itself takes before
    learning_starts, and the distribution the circuit has to work on when the
    policy is still exploring. A trained policy narrows the action
    distribution, so this is the harder and more honest case.

    Returns arrays aligned on one row per (episode, step):
      z, cb_h2, base_obs, y (context index), episode, t

    cb_h2 is the CB hidden state AS THE OBSERVATION SAW IT -- one step stale,
    zero on an episode's first step -- so it is also what sac's fixed input
    scales must be calibrated against (see source_rms).
    """
    contexts = list(sac.PHASE_CONTEXT_IDS if contexts is None else contexts)
    max_steps = sac.MAX_STEPS if max_steps is None else max_steps

    config = sac.AblationConfig(name="probe", use_ec=True, use_cb=True,
                                freeze=True, arm=arm)
    cb_proj = sac.make_cb_proj(seed, device)
    ec, cb = M.load(arm, ckpt_seed)
    sac.freeze_module(ec)
    sac.freeze_module(cb)

    rng = np.random.default_rng(seed)
    z_rows, h2_rows, base_rows, y, episode, step = [], [], [], [], [], []

    for ctx_i, context_id in enumerate(contexts):
        for ep in range(n_episodes):
            pos, goal = sac.sample_pos_goal(rng)
            vel_sim = np.zeros(2, dtype=np.float32)
            vel_obs = np.zeros(2, dtype=np.float32)
            h = torch.zeros(1, sac.EC_HIDDEN, device=device)
            s_tm1 = torch.zeros(1, sac.SDIM, device=device)
            a_tm1 = torch.zeros(1, sac.ACTION_DIM, device=device)
            s_t = torch.tensor(sac.circuit_state(pos, vel_obs),
                               device=device).unsqueeze(0)
            e_t = torch.zeros(1, sac.SDIM, device=device)
            cb_h2 = torch.zeros(1, sac.CB_HIDDEN, device=device)

            for t in range(max_steps):
                obs, z_t, h = sac.augmented_obs_step(
                    ec, config, h, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
                    pos, vel_obs, goal, None, device)
                z_rows.append(z_t.squeeze(0).detach().cpu().numpy())
                h2_rows.append(cb_h2.squeeze(0).detach().cpu().numpy())
                base_rows.append(obs[:sac.SDIM + 2].copy())
                y.append(ctx_i)
                episode.append(ctx_i * 1_000_000 + ep)
                step.append(t)

                action = rng.uniform(-1.0, 1.0, size=sac.ACTION_DIM).astype(np.float32)
                action_t = torch.tensor(action, device=device).unsqueeze(0)
                pos_new, vel_sim = sac.env_step_params(
                    pos, vel_sim, action, context_id, rng)
                vel_obs_new = ((pos_new - pos) / sac.DT).astype(np.float32)

                s_pred, cb_h2 = sac.cb_predict(cb, s_t, action_t, z_t, device)
                s_next = torch.tensor(sac.circuit_state(pos_new, vel_obs_new),
                                      device=device).unsqueeze(0)
                s_tm1, a_tm1, s_t = s_t, action_t, s_next
                e_t = s_next - s_pred
                pos, vel_obs = pos_new, vel_obs_new

    return {"z": np.asarray(z_rows, dtype=np.float32),
            "cb_h2": np.asarray(h2_rows, dtype=np.float32),
            "base_obs": np.asarray(base_rows, dtype=np.float32),
            "y": np.asarray(y), "episode": np.asarray(episode),
            "t": np.asarray(step)}


def source_rms(arm="main", ckpt_seed=0, *, skip_first_step=True, **kw):
    """RMS of each of the actor's input sources, for sac's fixed input scales.

    RMS rather than std because a zero-mean weight's contribution to the
    pre-activation variance is n * var(w) * mean(x^2).

    skip_first_step drops t == 0, where cb_h2 and e_t are zero by convention
    and would drag the CB estimate down for a reason that has nothing to do
    with its scale.

    sac.KIN_RMS / EC_STATE_RMS / CB_STATE_RMS are literals calibrated from
    this; tests/test_sac.py recalibrates and fails if they have drifted.
    """
    data = rollout_z(arm, ckpt_seed, **kw)
    keep = data["t"] > 0 if skip_first_step else slice(None)

    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))

    return {"kin": rms(data["base_obs"][keep]),
            "ec": rms(data["z"][keep]),
            "cb": rms(data["cb_h2"][keep])}


def decode_accuracy(X, y, episode, *, kind="linear", n_splits=N_SPLITS):
    """Mean/sd accuracy over n_splits episode-grouped train/test splits.

    NaN when a split degenerates (a class missing from one side), rather than
    a misleading number.
    """
    accs = []
    for split_seed in range(n_splits):
        test = seq_split(episode, seed=split_seed)
        if test.all() or (~test).all():
            continue
        if len(np.unique(y[~test])) < 2 or len(np.unique(y[test])) < 2:
            continue
        accs.append(_clf(kind, split_seed).fit(X[~test], y[~test]).score(X[test], y[test]))
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.std(accs))


def probe_arm(arm, ckpt_seed, contexts=None, *, tss_min=T.TSS_MIN, **kw):
    """One arm, one checkpoint: pooled z accuracy, the raw-obs control, and
    the accuracy-by-step curve."""
    contexts = list(sac.PHASE_CONTEXT_IDS if contexts is None else contexts)
    data = rollout_z(arm, ckpt_seed, contexts, **kw)
    keep = data["t"] >= tss_min
    z_mean, z_sd = decode_accuracy(data["z"][keep], data["y"][keep], data["episode"][keep])
    raw_mean, _ = decode_accuracy(data["base_obs"][keep], data["y"][keep],
                                  data["episode"][keep])

    by_t = {}
    for lo, hi in T_BINS:
        m = (data["t"] >= lo) & (data["t"] < hi)
        if m.sum() > 500:
            by_t[(lo, hi)] = decode_accuracy(data["z"][m], data["y"][m],
                                             data["episode"][m])[0]

    return {"arm": arm, "ckpt_seed": ckpt_seed, "chance": 1.0 / len(contexts),
            "z": z_mean, "z_sd": z_sd, "raw_obs": raw_mean, "by_t": by_t}


def main(arms=("main", "control"), ckpt_seeds=(0,), contexts=None, **kw):
    contexts = list(sac.PHASE_CONTEXT_IDS if contexts is None else contexts)
    chance = 1.0 / len(contexts)
    print(f"{len(contexts)}-way decode of z under the SAC rollout, "
          f"contexts {contexts}, chance {chance:.3f}, split by episode")
    print(f"{'arm':9s} {'ckpt':>4s}  {'z':>13s} {'raw obs':>8s}   " +
          "  ".join(f"t{lo}-{hi}" for lo, hi in T_BINS))
    results = []
    for arm in arms:
        for ckpt_seed in ckpt_seeds:
            r = probe_arm(arm, ckpt_seed, contexts, **kw)
            results.append(r)
            by_t = "  ".join(f"{r['by_t'].get((lo, hi), float('nan')):.3f} "
                             for lo, hi in T_BINS)
            print(f"{arm:9s} {ckpt_seed:4d}  {r['z']:.3f} +-{r['z_sd']:.3f} "
                  f"{r['raw_obs']:8.3f}   {by_t}")
    print("\nraw obs is the leakage control and must sit at chance; a z column "
          "near chance\nmeans the circuit is not carrying context on the "
          "distribution training actually uses.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["main", "control"])
    ap.add_argument("--ckpt-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--contexts", nargs="+", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    args = ap.parse_args()
    main(arms=args.arms, ckpt_seeds=args.ckpt_seeds, contexts=args.contexts,
         n_episodes=args.episodes)
