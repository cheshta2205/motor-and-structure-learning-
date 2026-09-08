"""
oracle.py -- the MSE reference ladder.

What CB scores knowing NOTHING about the context, and what it scores being TOLD
it exactly.  Without these the raw MSEs mean little: "0.041" only becomes
interpretable as a fraction of the gain that was available.

TWO RUNGS.  CB only -- no MTL, no recurrence, no cerebellar error.  The context
is handed in directly, so these bound what the forward model can do, not what
the circuit can infer.

    blind      [pos, vel, a_t]           no context at all
    one-hot    [pos, vel, a_t, onehot]   a perfect context label

THE ORACLE'S STATE MATCHES THE MODEL'S but gets clear explicit C label.

Both still predict POSITION and are scored on position MSE, which is also what
model.track reports, so the numbers are directly comparable.

The residual-output rungs from the exploratory version are gone: both arms here
predict the next state absolutely, so only the absolute ladder applies.


  python oracle.py
"""
import os

import numpy as np
import torch
import torch.nn as nn

from task import (CONTEXTS, N_CONTEXTS, FAMS, FAM_NAME, CB_HIDDEN, S_SCALE,
                  DT, LR, N_TRAIN, BATCH)
from data import generate_sequences

EPOCHS = 250
LADDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "oracle_ladder.npz")
RUNGS = ("blind", "one-hot")


class OracleCB(nn.Module):
    """Same shape and same observation as the real CB, minus z.

    Input [pos(2), vel(2), a(2)] + one-hot; predicts the next POSITION
    absolutely.  Matching the model's 4-d state is what makes this a ceiling
    rather than a handicapped comparison -- see the module docstring.
    """

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 2 + d, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, 2))

    def forward(self, s, v, a, c):
        parts = [s, v, a] + ([c] if c is not None else [])
        return self.net(torch.cat(parts, -1))


def run(table, tag, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    trans, ctx, _ = generate_sequences(N_TRAIN, seed=seed)
    t = trans.clone()
    t[:, :, :2] *= S_SCALE
    t[:, :, 4:6] *= S_SCALE
    cid = ctx[:, :, 0].long()
    fam = ctx[:, :, 1].numpy().astype(int)
    C = table[cid] if table is not None else None
    # V_{t-1} = (s_t - s_{t-1})/DT, exactly as model.build derives it
    pos = t[:, :, :2]
    V = torch.zeros_like(pos)
    V[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / DT
    cb = OracleCB(0 if table is None else table.shape[1])
    opt = torch.optim.Adam(cb.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-5)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_TRAIN)
        tot = 0.
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            b = t[idx]
            pred = cb(b[:, :, :2], V[idx], b[:, :, 2:4],
                      C[idx] if C is not None else None)
            loss = torch.mean((pred[:, :-1] - b[:, 1:, :2]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        if ep % 100 == 0:
            print(f"    [{tag}] ep {ep:3d} | MSE {tot/N_TRAIN:.5f}", flush=True)
    with torch.no_grad():
        pred = cb(t[:, :, :2], V, t[:, :, 2:4], C)
        se = ((pred[:, :-1] - t[:, 1:, :2]) ** 2).mean(-1).numpy()
    fams = sorted({c[6] for c in CONTEXTS})
    return float(se.mean()), {FAM_NAME[f]: float(se[fam[:, :-1] == f].mean())
                              for f in fams}


def load():
    """The ladder as {rung: {'overall': x, family: x}}, or None if not built."""
    if not os.path.exists(LADDER):
        return None
    z = np.load(LADDER)
    return {r: {"overall": float(z[f"{r}_overall"]),
                **{f: float(z[f"{r}_{f}"]) for f in FAMS}} for r in RUNGS}


def main(seed=0):
    res = {}
    for name, tab in (("blind", None), ("one-hot", torch.eye(N_CONTEXTS))):
        print(f"  === {name} ===", flush=True)
        res[name] = run(tab, name, seed)
    print(f"\n{'rung':<16}{'overall':>10}" + "".join(f"{f:>12}" for f in FAMS))
    for k, (o, per) in res.items():
        print(f"{k:<16}{o:>10.4f}" + "".join(f"{per[f]:>12.4f}" for f in FAMS))
    np.savez(LADDER,
             **{f"{k}_{f}": res[k][1][f] for k in res for f in FAMS},
             **{f"{k}_overall": res[k][0] for k in res})
    print(f"\nsaved -> {LADDER}")


if __name__ == "__main__":
    main()
