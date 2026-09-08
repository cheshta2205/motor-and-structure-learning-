"""
model.py -- the two networks of MTL and CB, the forward loop, the MSE loss, and training.

THE TWO ARMS.  They are separately trained networks that differ in ONE thing:
whether MTL is shown s_t, the state its own action produced or not.

    main      MTL <- [s_{t-1}, a_{t-1},      e_t]        10-d input
    control   MTL <- [s_{t-1}, a_{t-1}, s_t, e_t]        14-d input

Everything else is identical -- same alpha, same loss, same CB inputs, same
task, same seeds -- so the contrast isolates that one input.  This is to show how the control, 
when MTL knows next S, it doesnt need to rely on CB pred eror. 
TRAINING-time contrast about what the architecture forces the network to learn.

also, S= [S,V]. for our task. 

TIMING -- the one convention to get right.  Within iteration t:
    1.  observe s_t, a_t
    2.  e_t = s_t - s_hat_t,  where s_hat_t was written by CB at t-1
        (read-before-write on s_pred)
    3.  z_t       = MTL(s_{t-1}, a_{t-1}, [s_t,] e_t)
    4.  s_hat_t+1 = CB(s_t, a_t, z_t)

THE ABLATIONS are test-time only, on frozen weights,
    intact   unmodified
    zero     e_t = 0.  The INFORMATION test: does MTL need what e carries?
             Note it is not "CB disconnected": during training e ~ 0 means "CB
             predicted correctly", so it reads as confirmation of the current
             belief.
    mtl_noise  Gaussian noise straight into the GRU hidden state, scaled by
             that state's own sd.  Never passes through the cerebellum, so it
             is the control that makes "zeroing e hurts" a claim about the
             CEREBELLAR CHANNEL rather than about perturbing the network at
             all.  The analogue of Zeidler's hippocampal PV-interneuron
             control, which leaves OLM intact where cerebellar drive abolishes
             it.  Scaling by the state's own sd matters: h is 256-d while e is
             4-d, so a matched NOMINAL magnitude is not a matched perturbation.
    shuffle  e_t taken from a random other sequence, redrawn each timestep.  A
             PERTURBATION test: it injects an actively wrong signal, and any GRU
             degrades on garbage whether or not it needs the channel.
The design claim rests on ZERO, where control nulls and main does not.  Control
still loses a lot under shuffle (all-16 .812 -> .447), so shuffle cannot carry
"control does not need e" -- report it instead as evidence that the intact main
circuit RELIES on e rather than merely tolerating it.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from task import (CONTEXTS, FAMS, FAM_NAME, DT, S_SCALE, EC_HIDDEN, CB_HIDDEN,
                  LATENT_DIM, N_TRAIN, BATCH, EPOCHS, LR, TSS_MIN, RHO,
                  rot_mat, refl_mat)
from data import generate_sequences
from decode import acc

# ── The two arms.  ALPHA is fixed at 25  the code this
# was extracted from parsed it out of arm names to run a sweep, which is gone.
ALPHA = 25.0
# arm name -> IS MTL DENIED s_t, the state its own action produced?
# Read the boolean as "denied", NOT as "enabled" -- control: False means the
# control is not denied s_t, i.e. it DOES receive it.
#   main    True   MTL never sees s_t, so the cerebellar error is its only
#                  account of what just happened            -> 10-d MTL input
#   control False  MTL is handed s_t as well, can compute the transformation
#                  itself, and e becomes redundant          -> 14-d MTL input
ARMS = {"main": True, "control": False}
SDIM = ODIM = 4                              # state is position AND velocity [x, y, vx, vy]

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def ckpt_path(arm, seed):
    """Checkpoint path.  `seed` is REQUIRED on purpose: a default would let a
    caller that forgot it silently read seed 0's weights while writing into
    another seed's folder.  That exact bug cost three figure scripts in the
    exploratory version, and it is invisible -- the figures render fine, they
    are just of the wrong network."""
    return os.path.join(CKPT_DIR, f"{arm}_seed{seed}.pt")


def fig_dir(arm, seed):        # seed required -- see ckpt_path
    d = os.path.join(FIG_DIR, arm, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def _check(arm):
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    return ARMS[arm]


def mtl_input_width(arm):
    """Width of MTL's input for this arm: 10 if denied s_t, 14 if given it."""
    return (1 if ARMS[arm] else 2) * SDIM + 2 + SDIM


# Guard against the two arms being silently swapped.  The boolean above is easy
# to misread ("control: False" looks like the control being switched off), and
# flipping it would exchange the headline result with its own control while
# every filename, figure title and log line kept the old names -- the failure
# would be invisible.  Cheap arithmetic check, no modules built.
assert mtl_input_width("main") == 10 and mtl_input_width("control") == 14, (
    "arm definitions are inverted: 'main' must DENY s_t (10-d MTL input) and "
    "'control' must receive it (14-d)")


class EC(nn.Module):
    """MTL: GRU over the observation stream -> z_t (64-d, tanh)."""

    def __init__(self, no_current):
        super().__init__()
        self.no_current = no_current
        n_state = 1 if no_current else 2          # s_{t-1} alone, or with s_t
        self.cell = nn.GRUCell(n_state * SDIM + 2 + SDIM, EC_HIDDEN)
        self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())

    def init_hidden(self, B):
        return torch.zeros(B, EC_HIDDEN)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.readout(h), h


class CB(nn.Module):
    """Cerebellum: a forward model (s_t, a_t, z_t) -> s_hat_{t+1}.

    Feedforward on purpose -- no recurrence, so it can never hold two
    consecutive states and therefore can never identify the context itself.
    It predicts the NEXT STATE absolutely, not a displacement or a residual.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SDIM + 2 + LATENT_DIM, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, ODIM))

    def forward(self, s, a, z, return_hidden=False):
        """return_hidden also returns the two post-ReLU layer activations.

        Used by representation_analysis to ask what CB does with the context it
        is handed.  Note these sit DOWNSTREAM of z, so context being present in
        them is guaranteed by construction -- the question worth asking is
        whether CB REFORMATS it, e.g. entangles context with action so that the
        force prediction becomes linear.
        """
        x = torch.cat([s, a, z], -1)
        if not return_hidden:
            return self.net(x)
        h1 = self.net[1](self.net[0](x))      # Linear -> ReLU
        h2 = self.net[3](self.net[2](h1))     # Linear -> ReLU
        return self.net[4](h2), h1, h2


def build(trans):
    """Observation stream, actions, and prediction target.

    Velocity in the OBSERVATION is V_{t-1} = (s_t - s_{t-1})/DT, which the agent
    could know at t; using V_t would leak a_t.  The velocity TARGET is
    V_{t+1} = (s_{t+1} - s_t)/DT -- a displacement over dt, so a velocity loss
    is a displacement loss in different units.
    """
    t = trans.clone()
    t[:, :, :2] *= S_SCALE
    t[:, :, 4:6] *= S_SCALE
    pos, act, nxt = t[:, :, :2], t[:, :, 2:4], t[:, :, 4:6]
    vel = torch.zeros_like(pos)
    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / DT
    obs = torch.cat([pos, vel], -1)
    tgt = torch.cat([nxt, (nxt - pos) / DT], -1)
    return obs, act, tgt


# Magnitude of the direct-MTL perturbation, as a fraction of the GRU state's
# own standard deviation.
#
# CALIBRATED, not chosen: 0.5 costs the forward model the same accuracy that
# zeroing e does (position MSE 0.0923 vs 0.0948, against an intact 0.0425).
# That matching is what makes it a control.  At 1.0 the MSE is 0.557 -- six
# times the damage -- so "MTL noise also hurts decoding" would say nothing
# except that the network had been broken.  With the damage matched, the
# question becomes the interesting one: at EQUAL cost to prediction, does
# perturbing MTL cost as much CONTEXT DECODING as removing the cerebellar
# error does?
MTL_NOISE = 0.5


def rollout(ec, cb, trans, arm, e_mode="intact", gen=None, e_mask=None,
            sites=False):
    """One forward pass over a batch of sequences.  See the module docstring for
    the timing convention and the ablation modes.

    e_mask, when given, restricts the ablation to selected (sequence, timestep)
    entries -- used to lesion the cerebellar channel only in a window around a
    context switch, which separates re-inferring a new context from maintaining
    one already known.
    """
    no_current = _check(arm)
    obs, act, target = build(trans)
    B, T, _ = obs.shape
    h = ec.init_hidden(B)
    s_pred = torch.zeros(B, ODIM)
    Z, P = [], []
    # sites: the four representations representation_analysis looks at --
    # z (MTL output, 64-d), h (MTL GRU hidden, 256-d), and CB's two post-ReLU
    # layers (256-d each).  Off by default so the normal path allocates nothing.
    H, L1, L2 = [], [], []
    for t in range(T):
        s_t, a_t = obs[:, t], act[:, t]
        if t == 0:
            s_tm1 = torch.zeros(B, SDIM)
            a_tm1 = torch.zeros(B, 2)
            e_t = torch.zeros(B, ODIM)
        else:
            s_tm1, a_tm1 = obs[:, t - 1], act[:, t - 1]
            e_t = s_t - s_pred            # s_pred is from t-1: read before write
        if e_mode in ("zero", "shuffle"):
            e_new = (torch.zeros_like(e_t) if e_mode == "zero"
                     else e_t[torch.randperm(B, generator=gen)])
            if e_mask is None:
                e_t = e_new
            else:
                m = e_mask[:, t].unsqueeze(-1).to(e_t.dtype)
                e_t = m * e_new + (1 - m) * e_t
        # ── the entire difference between the two arms 
        parts = ([s_tm1, a_tm1, e_t] if no_current
                 else [s_tm1, a_tm1, s_t, e_t])
        z_t, h = ec.step(torch.cat(parts, -1), h)
        if e_mode == "mtl_noise":
            nz = MTL_NOISE * h.std() * torch.randn(h.shape, generator=gen)
            if e_mask is not None:
                nz = e_mask[:, t].unsqueeze(-1).to(h.dtype) * nz
            h = h + nz
        if sites:
            s_pred, l1, l2 = cb(s_t, a_t, z_t, return_hidden=True)
            H.append(h); L1.append(l1); L2.append(l2)
        else:
            s_pred = cb(s_t, a_t, z_t)
        Z.append(z_t); P.append(s_pred)
    Z, P = torch.stack(Z, 1), torch.stack(P, 1)
    if sites:
        return Z, P, target, {"z": Z, "mtl_hidden": torch.stack(H, 1),
                              "cb_layer1": torch.stack(L1, 1),
                              "cb_layer2": torch.stack(L2, 1)}
    return Z, P, target


def weighted_loss(pred, tgt):
    """L = mean((Se/dt)^2) + ALPHA * mean(Ve^2)

    WHY THE 1/dt.  F is a per-step velocity impulse, so a force error dF appears
    as Ve = dF in the velocity residual but only Se = dt*dF in the position one
    -- the same mistake is 25x smaller in position, 625x once squared.  Dividing
    Se by dt puts both residuals in the same units, after which ALPHA is a
    genuine relative weight and ALPHA = 1 would weight the halves equally.
    Without it, a plain mean over the four components is 99.84% velocity: a
    velocity-only loss dressed up as a neutral one.
    """
    e = (pred[:, :-1] - tgt[:, :-1]) ** 2
    return (e[..., :2] / DT ** 2).mean() + ALPHA * e[..., 2:].mean()


def load(arm, seed):
    """Load a trained arm.  Returns (ec, cb) in eval mode, or None if absent."""
    p = ckpt_path(arm, seed)
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu")
    ec, cb = EC(_check(arm)), CB()
    ec.load_state_dict(sd["ec"]); cb.load_state_dict(sd["cb"])
    ec.eval(); cb.eval()
    return ec, cb


def track(ec, cb, arm, va, va_ctx, va_tss, va_fam, seed=0):
    """Held-out position MSE (overall + per family) and per-family context
    decoding, linear and MLP.  MSE is the POSITION half only, so both arms stay
    on the same scale."""
    ec.eval(); cb.eval()
    with torch.no_grad():
        z, pred, tgt = rollout(ec, cb, va, arm)
    se = ((pred[:, :-1, :2] - tgt[:, :-1, :2]) ** 2).mean(-1).numpy()
    zn = z.numpy()
    ok = va_tss.numpy() >= TSS_MIN
    cid = va_ctx[:, :, 0].numpy().astype(int)
    sid = np.repeat(np.arange(zn.shape[0])[:, None], zn.shape[1], 1)
    out = {"mse": float(se.mean())}
    for i, f in enumerate(FAMS):
        out[f"mse_{f}"] = float(se[va_fam[:, :-1] == i].mean())
    dec = {}
    for kind in ("linear", "mlp"):
        for i, f in enumerate(FAMS):
            m = ok & (va_fam == i)
            dec[f"{f}_{kind}"] = acc(zn[m], cid[m], sid[m], kind, seed)
    ec.train(); cb.train()
    return out, dec


def train(arm, seed, epochs=EPOCHS, decode_every=25):
    """Train one arm.  Writes the ONLY checkpoint, and track.npz beside it.

    track.npz holds the per-epoch history and CANNOT be rebuilt from a
    checkpoint -- recovering one seed's per-family history once cost a full
    retrain -- so do not delete it.
    """
    _check(arm)
    torch.manual_seed(seed)
    trans, _, _ = generate_sequences(N_TRAIN, seed=seed)
    va, va_ctx, va_tss = generate_sequences(300, seed=seed + 500)
    va_fam = va_ctx[:, :, 1].numpy().astype(int)
    ec, cb = EC(ARMS[arm]), CB()
    opt = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-5)
    hist, per_fam, dec_ep, dec = [], [], [], []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_TRAIN); tot = 0.
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            _, pred, tgt = rollout(ec, cb, trans[idx], arm)
            loss = weighted_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        ec.eval()
        with torch.no_grad():
            _, pr, tg = rollout(ec, cb, va, arm)
        se = ((pr[:, :-1, :2] - tg[:, :-1, :2]) ** 2).mean(-1).numpy()
        ec.train()
        hist.append(float(se.mean()))
        per_fam.append([float(se[va_fam[:, :-1] == i].mean())
                        for i in range(len(FAMS))])
        if ep % decode_every == 0 or ep == 1:
            _, dd = track(ec, cb, arm, va, va_ctx, va_tss, va_fam, seed)
            dec_ep.append(ep); dec.append(dd)
            print(f"  [{arm}] ep {ep:3d} | train {tot/N_TRAIN:.5f} | "
                  f"held-out pos MSE {hist[-1]:.5f} | "
                  + " ".join(f"{k[:3]}-{k.split('_')[1][:3]} {v:.2f}"
                             for k, v in dd.items()), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": hist},
               ckpt_path(arm, seed))
    np.savez(os.path.join(fig_dir(arm, seed), "track.npz"),
             hist=np.array(hist), per_fam=np.array(per_fam),
             dec_ep=np.array(dec_ep),
             **{k: np.array([d[k] for d in dec]) for k in dec[0]})
    print(f"  saved -> {ckpt_path(arm, seed)}", flush=True)
