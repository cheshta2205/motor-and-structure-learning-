"""
data.py -- THE DATA: rolls the task out into sequences to train and test on.
Draws an action, applies the current context's transformation through
task.env_step, occasionally switches context, and records what the agent did and
where it ended up.  Returns three aligned tensors -- the transitions the network
sees, the context labels (used ONLY for decoding and scoring, never as input),
and the steps since the last switch.

FAST-switching contexts, AR(1) Gaussian actions.

ACTIONS ARE AR(1) AND GAUSSIAN, both deliberately.

  a_t = rho*a_{t-1} + sqrt(1-rho^2)*eps,   rho = 0.35,  eps ~ N(0, 1/3) - correlated actions. unbounded. 

The sqrt(1-rho^2) keeps the marginal variance fixed at 1/3 whatever rho is, so
rho changes ONLY the temporal correlation and not the action scale.  The
Gaussian sd of 1/sqrt(3) matches the variance of the uniform[-1,1] this replaced
-- so the change is the shape of the support (unbounded vs a square), not the
scale.  A square support leaks rotation: the corners tell you the axes.

Blocks: STAY_PROB 0.9 over SEQ_LEN 40 gives mean block ~8 steps and ~4.9 blocks
per sequence, with 59% of steps at tss >= TSS_MIN.
"""
import numpy as np
import torch

from task import (CONTEXTS, N_CONTEXTS, GRID_SIZE, SEQ_LEN, STAY_PROB, RHO,
                  env_step)

GAUSS_SD = float(np.sqrt(1 / 3))


def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0, stay=STAY_PROB, rho=RHO):
    """Returns (transitions, labels, time-since-switch).

        transitions[b, t] = [x, y, ax, ay, x', y']
        labels[b, t]      = [context_id, family]
        tss[b, t]         = steps since the last context switch

    VELOCITY IS NOT STORED, only positions and actions.  model.build
    reconstructs V_{t-1} = (s_t - s_{t-1})/DT for the observation and
    V_{t+1} = (s_{t+1} - s_t)/DT for the target.  Keeping them separate is what
    lets velocity appear in the LOSS without appearing in MTL's INPUT.

    A NEW CONTEXT IS ALWAYS A DIFFERENT ONE -- `nxt + (nxt >= c)` samples from
    the other 15 -- so tss = 0 always marks a real change.
    """
    rng = np.random.default_rng(seed)
    sd = np.sqrt(1 - rho ** 2)
    T_, C_, S_ = [], [], []
    for _ in range(n_seqs):
        pos = rng.uniform(.25 * GRID_SIZE, .75 * GRID_SIZE, 2).astype(np.float32)
        vel = np.zeros(2, np.float32)
        a = rng.normal(0.0, GAUSS_SD, 2).astype(np.float32)
        c, tss = int(rng.integers(N_CONTEXTS)), 0
        st, sc, ss = [], [], []
        for t in range(seq_len):
            if t > 0 and rng.random() > stay:
                nxt = int(rng.integers(N_CONTEXTS - 1))
                c = nxt + (nxt >= c)          # sample a DIFFERENT context
                tss = 0
            # The action process is NOT reset at a switch: it is a property of
            # the agent, not the environment, so resetting it would leak the
            # switch time into the actions themselves.
            if t > 0:
                # the draw is cast to float32 BEFORE the sd multiply, not
                # after: doing it the other way rounds differently and the
                # trajectories drift apart by ~5e-07, enough that the trained
                # checkpoints no longer see the data they were trained on
                eps = rng.normal(0.0, GAUSS_SD, 2).astype(np.float32)
                a = (rho * a + sd * eps).astype(np.float32)
            kind, param, bx, by, c_id, _, fam = CONTEXTS[c]
            pn, vn, _ = env_step(pos, vel, a, kind, param, bx, by)
            st.append([pos[0], pos[1], a[0], a[1], pn[0], pn[1]])
            sc.append([float(c_id), float(fam)])
            ss.append(float(tss))
            pos, vel = pn, vn
            tss += 1
        T_.append(st); C_.append(sc); S_.append(ss)
    return (torch.tensor(np.array(T_), dtype=torch.float32),
            torch.tensor(np.array(C_), dtype=torch.float32),
            torch.tensor(np.array(S_), dtype=torch.float32))
