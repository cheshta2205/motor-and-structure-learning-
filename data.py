"""
data.py -- THE DATA: rolls the task out into sequences to train and test on.
Draws an action, applies the current context's force law through task.env_step,
occasionally switches context, and records what the agent did and where it
ended up.  Returns three aligned tensors -- the transitions the network sees,
the context labels (used ONLY for decoding and scoring, never as input), and
the steps since the last switch.

FAST-switching contexts, AR(1) Gaussian actions.  Both the action process and
its scale now live in task.py (task.next_action / task.ACTION_SD), because the
plant's effect-size calibration is measured at that scale and the two must not
drift apart -- they did once, leaving the two factors' marginals 20% apart with
nothing complaining.

VELOCITY IS STORED EXPLICITLY, which it was not before.  The old format kept
only positions and let model.build reconstruct V = (s_t - s_{t-1})/DT.  That
identity IS EXACT in the interior of the new plant -- pos' = pos + vel'*DT --
but it BREAKS at a wall: env_step clips the position and zeroes the velocity on
the clipping axis, so the position difference is not the velocity there.  It is
RARE AND SEVERE rather than common and mild -- measured on these sequences it
hits 0.7% of steps (far below the ~13% of a uniform start, because sequences
begin in the middle half), but where it hits, the reconstruction is off by
0.0795 against a typical speed of 0.0251, i.e. wrong by 3x.  Since every bit of
context information lives in the velocity (see multimodality.position_invariance)
those steps would corrupt exactly the signal the circuit exists to extract, and
they are unmarked.  Storing velocity costs four floats per step and removes the
reconstruction entirely.

Blocks: STAY_PROB 0.9 over SEQ_LEN 40 gives mean block ~8 steps and ~4.9 blocks
per sequence, with 59% of steps at tss >= TSS_MIN.

THE NOISE IS ON.  env_step is given the sequence rng, so the command noise that
makes context inference require history (task.CMD_NOISE) is present in the
training data.  Passing rng=None here would generate a deterministic plant that
an ideal observer solves in two steps -- see tests/test_task.py.
"""
import functools
import os

import numpy as np
import torch

from task import (CONTEXTS, N_CONTEXTS, POS_LIM, SEQ_LEN, STAY_PROB, RHO,
                  FAMILY, ACTION_SD, env_step, next_action,
                  initial_action, action_sequence, sample_action_params)

GAUSS_SD = ACTION_SD          # kept as a name for callers that imported it

ACTION_MODES = ("gp", "ar1", "iid", "policy")

# ── the "policy" action process: a trained oracle agent's recorded actions ──
#
# Built by action_bank.py (which needs sac; data cannot import it). The file
# holds one action stream, reach after reach in random order, and a stamp that
# identifies its contents; model.convention_for puts the stamp into every
# circuit trained on it. Missing bank -> a clear error naming the builder, not
# a fallback to another process.
ACTION_BANK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "checkpoints", "action_bank_oracle.npz")


@functools.lru_cache(maxsize=None)
def load_action_bank(path=None):
    # resolved at CALL time, so a test (or a caller) that repoints
    # ACTION_BANK_PATH is honoured; a def-time default would freeze the path
    path = ACTION_BANK_PATH if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"action_mode='policy' needs the recorded action bank at {path}; "
            f"build it with `python action_bank.py` (records the oracle agents' "
            f"actions in their four contexts).")
    z = np.load(path)
    return {"actions": np.asarray(z["actions"], dtype=np.float32),
            "episode_start": np.asarray(z["episode_start"]),
            "stamp": str(z["stamp"])}


def action_bank_stamp(path=None) -> str:
    return load_action_bank(path)["stamp"]


def bank_window(bank, seq_len, rng):
    """A random `seq_len` window of the recorded stream. Windows may span a
    reach boundary -- a change of goal is part of what a policy's actions do."""
    n = len(bank["actions"])
    if n < seq_len:
        raise ValueError(f"action bank has {n} steps, fewer than seq_len {seq_len}")
    start = int(rng.integers(0, n - seq_len + 1))
    return bank["actions"][start:start + seq_len]


def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0, stay=STAY_PROB,
                       rho=RHO, contexts=None, action_mode="gp"):
    """Returns (transitions, labels, time-since-switch).

        transitions[b, t] = [x, y, vx, vy, ax, ay, x', y', vx', vy']   (10)
        labels[b, t]      = [context_id, family]
        tss[b, t]         = steps since the last context switch

    LABELS CARRY THE FAMILY as well as the context, because "which of the 16"
    and "which family" are different questions with different chance levels
    (1/16 and 1/2) and the second is the one the rotation/reflection contrast
    is about.

    `contexts` restricts which contexts are drawn -- a subset for a held-out
    experiment.  Default None means all of them.

    `action_mode` selects the action process:

      "gp"      THE DEFAULT (model.DEFAULT_ACTION_MODE, since 2026-09-16): a
                DIFFERENT process per sequence, drawn from task's Matern x
                tanh-warp family (see task.sample_action_params).  Marginal sd
                is ACTION_SD in both, so the plant's effect-size calibration is
                untouched; correlation and marginal SHAPE vary.
      "ar1"     task.next_action -- the AR(1) at RHO with a Gaussian marginal,
                bit for bit what the pre-2026-09-16 checkpoints were trained
                on ("fitted" is the legacy spelling).
      "iid"     the AR(0) control: ar1 at rho = 0.
      "policy"  windows of a trained oracle agent's recorded actions (the bank
                action_bank.py writes): the deployment distribution itself,
                replayed open loop -- see that module. Drawn before the loop
                like a GP draw, so still independent of the context stream.

    WHY "gp" EXISTS: the circuit is fitted under one action distribution and
    deployed under a policy whose actions look nothing like it, and under that
    shift z's context code stops being additive.  Training across a range is
    what would make additivity a property of the circuit rather than of the
    distribution it happened to be fitted on.  See task's GP section.

    A NEW CONTEXT IS ALWAYS A DIFFERENT ONE -- the resample below draws from
    the other cells only -- so tss = 0 always marks a real change.
    """
    # a model.CircuitSpec in the action_mode slot also names the contexts the
    # circuit was fitted on (its .context_ids, None = all); an explicit
    # `contexts` argument still wins
    if contexts is None:
        contexts = getattr(action_mode, "context_ids", None)
    pool = list(range(N_CONTEXTS)) if contexts is None else list(contexts)
    if len(pool) < 2:
        raise ValueError("need at least two contexts to switch between")
    # model.CircuitSpec in the slot: the rollout distribution follows the
    # circuit's mode (duck-typed -- data cannot import model). And VALIDATE:
    # the branches below are string compares, so an unrecognised value used to
    # fall silently into the ar1 path.
    action_mode = getattr(action_mode, "action_mode", action_mode)
    action_mode = {"fitted": "ar1"}.get(action_mode, action_mode)
    if action_mode not in ACTION_MODES:
        raise KeyError(f"unknown action_mode {action_mode!r}; expected one of {ACTION_MODES}")
    if action_mode == "iid":
        # the AR(0) control: the ar1 path with no temporal correlation at all;
        # next_action holds the marginal at ACTION_SD, so only rho moves
        rho = 0.0
    bank = load_action_bank() if action_mode == "policy" else None
    rng = np.random.default_rng(seed)
    T_, C_, S_ = [], [], []
    for _ in range(n_seqs):
        # start in the middle half, away from the walls: a sequence that begins
        # pinned spends its first steps on clipped transitions, which carry no
        # context information at all
        pos = rng.uniform(-0.5 * POS_LIM, 0.5 * POS_LIM, 2).astype(np.float32)
        vel = np.zeros(2, np.float32)
        # GP actions are drawn JOINTLY over the sequence, before the loop, so
        # the action stream cannot depend on the context stream at all -- the
        # property the comment below maintains by convention in "fitted" mode.
        if action_mode == "gp":
            acts = action_sequence(seq_len, rng)
        elif action_mode == "policy":
            acts = bank_window(bank, seq_len, rng)
        else:
            acts = None
        a = initial_action(rng) if acts is None else acts[0]
        c, tss = int(pool[rng.integers(len(pool))]), 0
        st, sc, ss = [], [], []
        for t in range(seq_len):
            if t > 0 and rng.random() > stay:
                others = [x for x in pool if x != c]
                c = int(others[rng.integers(len(others))])
                tss = 0
            # The action process is NOT reset at a switch: it is a property of
            # the agent, not the environment, so resetting it would leak the
            # switch time into the actions themselves.
            if t > 0:
                a = next_action(a, rng, rho=rho) if acts is None else acts[t]
            pn, vn, _F = env_step(pos, vel, a, c, rng)
            st.append([pos[0], pos[1], vel[0], vel[1], a[0], a[1],
                       pn[0], pn[1], vn[0], vn[1]])
            sc.append([float(c), float(FAMILY[c])])
            ss.append(float(tss))
            pos, vel = pn, vn
            tss += 1
        T_.append(st); C_.append(sc); S_.append(ss)
    return (torch.tensor(np.array(T_), dtype=torch.float32),
            torch.tensor(np.array(C_), dtype=torch.float32),
            torch.tensor(np.array(S_), dtype=torch.float32))
