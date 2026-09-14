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
import numpy as np
import torch

from task import (CONTEXTS, N_CONTEXTS, POS_LIM, SEQ_LEN, STAY_PROB, RHO,
                  FAMILY, ACTION_SD, env_step, next_action,
                  initial_action)

GAUSS_SD = ACTION_SD          # kept as a name for callers that imported it


def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0, stay=STAY_PROB,
                       rho=RHO, contexts=None):
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

    A NEW CONTEXT IS ALWAYS A DIFFERENT ONE -- the resample below draws from
    the other cells only -- so tss = 0 always marks a real change.
    """
    pool = list(range(N_CONTEXTS)) if contexts is None else list(contexts)
    if len(pool) < 2:
        raise ValueError("need at least two contexts to switch between")
    rng = np.random.default_rng(seed)
    T_, C_, S_ = [], [], []
    for _ in range(n_seqs):
        # start in the middle half, away from the walls: a sequence that begins
        # pinned spends its first steps on clipped transitions, which carry no
        # context information at all
        pos = rng.uniform(-0.5 * POS_LIM, 0.5 * POS_LIM, 2).astype(np.float32)
        vel = np.zeros(2, np.float32)
        a = initial_action(rng)
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
                a = next_action(a, rng, rho=rho)
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
