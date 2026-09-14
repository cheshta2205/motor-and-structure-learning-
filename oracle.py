"""
oracle.py -- the MSE reference ladder.

What CB scores knowing NOTHING about the context, what it scores being TOLD it
exactly, and what it scores being told THE TWO FACTORS separately.  Without
these the raw MSEs mean little: "0.041" only becomes interpretable as a
fraction of the gain that was available.

THREE RUNGS.  CB only -- no MTL, no recurrence, no cerebellar error.  The
context is handed in directly, so these bound what the forward model can do,
not what the circuit can infer.

    blind        [pos, vel, a]                 no context at all
    family-only  [pos, vel, a, family2]        told rotation vs reflection
    one-hot      [pos, vel, a, onehot16]       a perfect context label

FAMILY-ONLY MEASURES WHAT THE COARSE DISTINCTION IS WORTH.  Rotation and
reflection are the load-bearing pair -- same magnitude, opposite determinant --
so knowing WHICH family without knowing which member is exactly the coarse
half of the problem.  blind -> family-only against family-only -> one-hot
splits the available gain into "which kind" and "which member".

SCORED ON VELOCITY, NOT POSITION.  The old ladder predicted the next position
and reported position MSE.  Under this plant that is close to meaningless:
pos' = pos + vel'*DT, so echoing the input scores R2 = 0.9966 and 99.7% of the
metric is free.  All the context information is in the velocity
(multimodality.position_invariance), so that is what the ladder measures.
Position MSE is still reported beside it, because the two should sit in the
ratio VEL_SCALE^2 once the heads tie -- a discrepancy there means the forward
model has not converged.

BROKEN DOWN BY FACTOR LEVEL, not by "family".  The design is a 4x4 crossing,
so "how well is drag_hi predicted" and "how well is mirror predicted" are two
different questions and the marginal over each factor is the natural cut.

  python oracle.py
"""
import os

import numpy as np
import torch
import torch.nn as nn

from task import (N_CONTEXTS, FAM_NAMES, CB_HIDDEN, LR, N_TRAIN,
                  BATCH, VEL_SCALE, POS_SCALE)
from data import generate_sequences
import model as M

EPOCHS = 250
LADDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "oracle_ladder.npz")
RUNGS = ("blind", "family-only", "one-hot")


class OracleCB(nn.Module):
    """Same shape and same observation as the real CB, minus z.

    Input [pos(2), vel(2), a(2)] plus whatever context code the rung supplies;
    predicts the next STATE (pos and vel), exactly as model.CB does, so the
    numbers are directly comparable to model.track's.
    """

    def __init__(self, ctx_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 + ctx_dim, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, 4))

    def forward(self, s, v, a, c):
        parts = [s, v, a] + ([c] if c is not None else [])
        return self.net(torch.cat(parts, -1))


def context_code(rung, ctx):
    """The context information a rung is handed, as a [B, T, d] tensor.

    aim and field come from the LABELS rather than from divmod on the id, so
    this reads the same columns every decoder does and cannot disagree with
    them.
    """
    if rung == "blind":
        return None
    if rung == "family-only":
        return torch.nn.functional.one_hot(ctx[:, :, 1].long(),
                                           len(FAM_NAMES)).float()
    if rung == "one-hot":
        return torch.nn.functional.one_hot(ctx[:, :, 0].long(),
                                           N_CONTEXTS).float()
    raise KeyError(f"unknown rung {rung!r}; known: {RUNGS}")


def run(rung, seed=0, epochs=EPOCHS):
    """Train one rung.  Returns (velocity MSE, position MSE, per-level dicts)."""
    torch.manual_seed(seed)
    trans, ctx, _ = generate_sequences(N_TRAIN, seed=seed)
    # build() is what the real circuit is fed, so the ladder shares its input
    # conventions by construction rather than by a copied block that can drift
    obs, act, tgt = M.build(trans)
    C = context_code(rung, ctx)
    fam = ctx[:, :, 1].numpy().astype(int)

    cb = OracleCB(0 if C is None else C.shape[-1])
    opt = torch.optim.Adam(cb.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-5)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_TRAIN)
        tot = 0.0
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            pred = cb(obs[idx][:, :, :2], obs[idx][:, :, 2:], act[idx],
                      None if C is None else C[idx])
            # the ladder is a FORWARD-MODEL bound, so it is trained on the same
            # weighted loss the circuit is, not on a bare mean
            loss = M.weighted_loss(pred, tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        if ep % 100 == 0:
            print(f"    [{rung}] ep {ep:3d} | loss {tot / N_TRAIN:.5f}", flush=True)

    with torch.no_grad():
        pred = cb(obs[:, :, :2], obs[:, :, 2:], act, C)
    e = (pred[:, :-1] - tgt[:, :-1]) ** 2
    vel_se = e[..., 2:].mean(-1).numpy()
    pos_se = e[..., :2].mean(-1).numpy()
    per_fam = {FAM_NAMES[i]: float(vel_se[fam[:, :-1] == i].mean())
               for i in range(len(FAM_NAMES))}
    return float(vel_se.mean()), float(pos_se.mean()), per_fam


def load():
    """The saved ladder as {rung: {level: vel MSE}}, or None if never built.

    figures.py draws each rung as a horizontal reference on the per-level MSE
    panels, so this returns the PER-LEVEL breakdown rather than the scalars.
    Returns None rather than raising when the file is absent: the ladder takes
    ~15 minutes to build and every figure below it should still render without
    one.
    """
    if not os.path.exists(LADDER):
        return None
    z = np.load(LADDER)
    out = {}
    for r in RUNGS:
        d = {n: float(z[f"{r}_fam_{n}"]) for n in FAM_NAMES
             if f"{r}_fam_{n}" in z.files}
        if d:
            out[r] = d
    return out or None


def main(seed=0, epochs=EPOCHS):
    res = {}
    for rung in RUNGS:
        print(f"  [{rung}] training...", flush=True)
        res[rung] = run(rung, seed=seed, epochs=epochs)

    w = 13
    print(f"\n{'rung':<14}{'vel MSE':>10}{'pos MSE':>10}{'ratio':>8}   "
          + "".join(f"{n:>{w}}" for n in FAM_NAMES))
    for rung in RUNGS:
        v, p, pf = res[rung]
        print(f"{rung:<14}{v:>10.5f}{p:>10.5f}{(v / p if p else float('nan')):>8.1f}   "
              + "".join(f"{pf[n]:>{w}.5f}" for n in FAM_NAMES))

    exp = (VEL_SCALE / POS_SCALE) ** 2
    print(f"\nratio = vel MSE / pos MSE; it should approach "
          f"(VEL_SCALE/POS_SCALE)^2 = {exp:.0f} once the two heads tie.")
    print("A ratio near 1 means they are erring INDEPENDENTLY and the forward "
          "model has not converged -- see model.ALPHA.")
    print("\nfactored vs one-hot is the factorial check: the force law is "
          "exactly\nA(aim) @ a + B(field) @ v, so being told both factors is "
          "being told the\ncontext, and the two rungs should agree.")

    np.savez(LADDER,
             **{f"{r}_vel": res[r][0] for r in RUNGS},
             **{f"{r}_pos": res[r][1] for r in RUNGS},
             **{f"{r}_fam_{n}": res[r][2][n] for r in RUNGS for n in FAM_NAMES})
    print(f"\nsaved -> {LADDER}")
    return res


if __name__ == "__main__":
    main()
