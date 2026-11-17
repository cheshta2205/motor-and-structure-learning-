#!/usr/bin/env python
"""
Object-location and object-recognition memory in an MTL-cerebellar circuit.

SELF-CONTAINED.  This file needs nothing else from the repository.  It holds one
configuration -- the one every figure in the paper comes from -- rather than the
switches and variants that were tried along the way.

    python olm_orm_final.py --train   --seeds 0 1 2 3
    python olm_orm_final.py --figures --seeds 0 1 2 3


THE CIRCUIT

    e_t     = s_t - s_hat_t                      cerebellar prediction error
    z_t, h  = MTL([s_{t-1}, a_{t-1}, e_t], h)    latent inferred from that error
    s_hat   = s_in + CB([s_in, a_t, z_t])        forward model, conditioned on it

Trained end to end on next-state prediction alone.  MTL never sees the current
observation -- the cerebellar prediction error is its only account of what just
happened -- and nothing supervises z.


THE OBSERVATION MODEL

Earlier versions of this model gated perception: inside whisker reach you got
the shape, outside it the channel was zeroed; and outside the visual field the
cerebellum's own prediction was substituted for the missing observation.  Both
hard boundaries are gone here.  Every channel is present at every step, degraded
in proportion to how well it can currently be sensed:

    s_obs = sqrt(p) * s_true + sqrt(1-p) * sigma_c * eps,    eps ~ N(0,1)

sigma_c is the channel's own standard deviation, so the mixture is
variance-preserving: the observation keeps its scale at any precision, and p is
exactly the fraction of the observed variance that is signal.  Without that, low
precision would also shrink the numbers and the network could read reliability
straight off the input's magnitude instead of having to accumulate.

Precision is a plateau -- exactly 1 inside the sensor's range, then decaying:

    p(d) = 1                                        d <= R
    p(d) = 0.05 + 0.95 / (1 + ((d - R)/0.7)^2)      d >  R

R is the sensor's range: 0.80 for shape (whisker reach) and 0.90 for vision.
These are the boundaries the gated model used, now crossed continuously rather
than in one step.  LAM = 0.7 and the exponent 2 were chosen so precision falls
to half at roughly 0.7 beyond the range -- placing the transition inside the
arena, since the animal is never more than 1.75 from an object -- and so that the
decay is gradual rather than a cliff: a reading at d = 1.2 still retains about
half its signal variance.  Neither was tuned against the lesion results.

Vision is additionally scaled by a graded field of view,

    A(beta) = beta0^12 / (|beta|^12 + beta0^12),     beta0 = 150 deg = FOV/2

so the object fades at the rim of the 300-degree field instead of blinking out.

The floor of 0.05 means no channel is ever entirely absent: a distant object
leaves a weak trace rather than vanishing, and the network never sees an input
distribution it could read as a boundary crossing.

Both sensors report their own reliability.  SEEN carries the bearing precision
and FELT the shape precision, both undegraded.  Without FELT a whisker reading
taken at contact and one taken from across the arena look identical, so the
network cannot learn to trust a clean reading and must blend with the stored
estimate at every step -- which keeps z load-bearing even where perception alone
would suffice.

The loss is always taken against the TRUE state and every step is scored, so
nothing is masked out and a perfect predictor scores zero.


THE LESIONS

Three sites, applied at test on frozen weights:

    cb   noise on what the cerebellum transmits
    e    noise on the prediction error, after it is computed
    h    noise into MTL's recurrent state

sigma is a fraction of the perturbed quantity's own standard deviation at every
site, so it means the same thing at each.  The MTL injection is additionally
divided by sqrt(MTL_HIDDEN / S_DIM) so that the 256-dimensional hidden state and
the 20-dimensional state vector carry the same total injected variance; without
it the MTL site is perturbed about 13 times harder at nominal parity, which can
produce MTL dependence on its own.

CAVEAT worth stating in any writeup: that correction assumes total injected
variance is the right invariant.  The 256 units feed a 64-dimensional tanh
readout, so the effective number of independent channels is lower and unknown,
and sqrt(256/20) stands in for it rather than measuring it.  Report the
unnormalised numbers alongside (set DIM_MATCH = False).

The lesion follows Zeidler's 3-on / 60-off duty cycle.


WHAT IS MEASURED

Two sessions.  The agent explores for 200 steps; one object is then moved (OLM)
or changed in identity (ORM); it explores for a further 200 steps with the MTL
state and the cerebellar prediction carried across.  A change is detectable only
against a stored expectation, so we compare prediction error on the altered
object with error on the untouched one.  That contrast is the preference:

    PI = e_changed / (e_changed + e_unchanged)

which is 0.5 when the change goes unnoticed, and is Zeidler's discrimination
index rescaled: DI = 100 * (2*PI - 1).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_HERE, "checkpoints")
FIG_DIR = os.path.join(_HERE, "figures", "main", "OLM_ORM_FINAL")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- the world --
R_ARENA = 1.0
STEP = 0.04                  # commanded speed
TURN_MOM, TURN_SD = 0.80, 0.35
SPEED_MOM, SPEED_JITTER = 0.80, 0.30
OBJ_MIN_SEP = 0.95           # keeps the intensity radius from covering both
R_TOUCH = 0.15               # contact flag
R_ID = 0.80                  # whisker reach: shape is felt, not seen
SENSE_R = 0.90               # vision range
INT_R = 0.32                 # apparent size is informative within this
FOV_DEG = 300.0              # mouse visual field, 60 deg blind zone behind

# ------------------------------------------------------------ the observation --
P_FLOOR = 0.05               # nothing is ever entirely absent
PLAT_LAM = 0.7               # precision falls to half ~0.7 beyond the range
PLAT_N = 2.0                 # gradual decay rather than a cliff
BETA0 = np.radians(FOV_DEG / 2.0)
BETA_M = 12.0                # sharpness of the angular falloff
SD_BEAR, SD_INT, SD_SHAPE = 0.707, 0.300, 0.727   # each channel's own sd

# ------------------------------------------------------------- the state map --
S_DIM, A_DIM, SHAPE_D = 20, 2, 3
SEEN = [2, 6]                # bearing precision, per object
OBJ = [slice(3, 6), slice(7, 10)]        # bearing sin, cos, apparent size
SHAPE_OBJ = [slice(10, 13), slice(13, 16)]
FELT = [18, 19]              # shape precision, per object
TGT = {"OLM": [3, 4, 5],     # object 0 bearing sin, cos, apparent size
       "ORM": [10, 11, 12]}  # object 0 curvature, elongation, angularity
CH_NAMES = ["head sin", "head cos", "seen0", "bear0 sin", "bear0 cos", "int0",
            "seen1", "bear1 sin", "bear1 cos", "int1",
            "curv0", "elong0", "angul0", "curv1", "elong1", "angul1",
            "touch", "at wall", "felt0", "felt1"]
SHAPE_HALFWIDTH = 1.26       # shape features ~ U(-1.26, 1.26)

# ------------------------------------------------------------- the networks --
MTL_HIDDEN, LATENT_DIM, CB_HIDDEN = 256, 64, 256
CB_INIT_STD = 0.05
BATCH, SEQ_LEN, EPOCHS, STEPS_EP, LR = 64, 200, 60, 100, 1e-3
P_SWITCH = 0.05              # fraction of training sequences containing a change
RESUME_EVERY = 2

# --------------------------------------------------------------- the lesions --
PI_WINDOW = 4                # steps after the change that PI is read over
PI_WINDOWS = (1, 4, 8, 16)   # the window sweep: is the dissociation k-specific?
LESION_SIGMA = 0.8           # the dose the four-condition and quality figures use
DIM_MATCH = True             # equal total injected variance across sites


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def precision(dist, R):
    """1 inside the sensor's range, decaying outside it."""
    u = np.maximum(dist - R, 0.0) / PLAT_LAM
    return np.where(dist <= R, 1.0,
                    P_FLOOR + (1.0 - P_FLOOR) / (1.0 + u ** PLAT_N))


class ObjectArena:
    """Circular arena, two objects, an agent that does not choose where to go.

    The agent is a random walk: step() never references object positions, so a
    behavioural discrimination index is undefined here -- Zeidler's DI measures
    a CHOICE.  Everything measured below is read off prediction error.
    """

    def __init__(self, batch, rng, r_id=R_ID):
        self.B, self.rng, self.r_id = batch, rng, r_id
        self._mrng = np.random.default_rng(rng.integers(2 ** 63))
        # a separate stream for observation noise, so changing it cannot shift
        # the agent's path and confound a comparison
        self._orng = np.random.default_rng(rng.integers(2 ** 63))
        self.reset_objects()
        self.reset_agent()

    def reset_objects(self):
        self.obj = np.zeros((self.B, 2, 2))
        for b in range(self.B):
            while True:
                p = self.rng.uniform(-0.75, 0.75, size=(2, 2))
                if (np.linalg.norm(p[0]) < 0.75 and np.linalg.norm(p[1]) < 0.75
                        and np.linalg.norm(p[0] - p[1]) > OBJ_MIN_SEP):
                    self.obj[b] = p
                    break
        self.shape = self.rng.uniform(-SHAPE_HALFWIDTH, SHAPE_HALFWIDTH,
                                      (self.B, 2, SHAPE_D))

    def reset_agent(self):
        a = self.rng.uniform(-np.pi, np.pi, self.B)
        r = np.sqrt(self.rng.random(self.B)) * 0.85
        self.p = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        self.theta = self.rng.uniform(-np.pi, np.pi, self.B)
        self.omega = np.zeros(self.B)
        self.vdrive = np.zeros(self.B)

    def move_object(self, idx=0, rng=None, who=None):
        """OLM: resample one object's position from the TRAINING layout.

        Not `angle + pi` -- that fixes object separation at 180 degrees where
        training saw a spread, and the visibility and travel statistics change
        with it.  That produced a spurious effect once.
        """
        rng = self._mrng if rng is None else rng
        who = np.ones(self.B, bool) if who is None else np.asarray(who, bool)
        for b in range(self.B):
            if not who[b]:
                continue
            other, old = self.obj[b, 1 - idx], self.obj[b, idx].copy()
            best, best_d = None, -np.inf
            for min_disp in (0.5, 0.25, 0.0):
                for _ in range(200):
                    p = rng.uniform(-0.75, 0.75, size=2)
                    if (np.linalg.norm(p) < 0.75
                            and np.linalg.norm(p - other) > OBJ_MIN_SEP
                            and np.linalg.norm(p - old) > min_disp):
                        best = p
                        break
                    d = np.linalg.norm(p - other)
                    if np.linalg.norm(p) < 0.75 and d > best_d:
                        best, best_d = p, d
                if best is not None and np.linalg.norm(best - other) > OBJ_MIN_SEP:
                    break
            self.obj[b, idx] = best if best is not None else old

    def change_identity(self, idx=0, rng=None, who=None):
        """ORM: resample one object's shape, positions untouched.

        last_delta records how large the change was, which is what lets you ask
        whether the deficit scales with how different the new object is.
        """
        rng = self._mrng if rng is None else rng
        who = np.ones(self.B, bool) if who is None else np.asarray(who, bool)
        old = self.shape[:, idx].copy()
        new = rng.uniform(-SHAPE_HALFWIDTH, SHAPE_HALFWIDTH, (self.B, SHAPE_D))
        self.shape[who, idx] = new[who]
        self.last_delta = np.linalg.norm(self.shape[:, idx] - old, axis=1)

    def step(self):
        """Returns the commanded action, which here IS the applied motion: the
        same speed and turn are handed back and used to move the agent.

        Dead reckoning from the efference copy still fails, but through the
        WALL, not through motor noise.  A collision rewrites theta by specular
        reflection and snaps the position to R_ARENA * 0.98, and none of that
        appears in the command.  Integrating the commands alone is exact for
        the first few steps (1e-9) and then diverges as soon as the agent
        touches the boundary: 3e-3 by step 10, 0.17 by step 20, 1.9 by step 60
        -- wider than the arena.  So position cannot be held open-loop and must
        be maintained from sensation, which is the property the forward model
        is being asked for.

        NOTE: self.vdrive below is computed and never read -- the speed line is
        STEP * clip(1.0, ...), i.e. STEP.  Speed therefore has per-step jitter
        but no momentum, despite SPEED_MOM.  It applies identically to every
        condition, task and lesion, so it affects nothing that is measured."""
        self.omega = (TURN_MOM * self.omega
                      + (1 - TURN_MOM) * self.rng.normal(0, TURN_SD, self.B))
        self.vdrive = (SPEED_MOM * self.vdrive
                       + (1 - SPEED_MOM) * self.rng.normal(0, 1.0, self.B))
        speed = STEP * np.clip(1.0, 0.1, 2.0)
        speed = speed * (1 + SPEED_JITTER * self.rng.normal(0, 1, self.B))
        speed = np.clip(speed, 0.2 * STEP, 2.0 * STEP)
        turn = self.omega
        cmd = np.stack([speed, turn], 1)
        self.theta = self.theta + turn
        d = np.stack([np.cos(self.theta), np.sin(self.theta)], 1)
        nxt = self.p + speed[:, None] * d
        rad = np.linalg.norm(nxt, axis=1)
        out = rad > R_ARENA
        if out.any():                       # reflect off the wall
            n = nxt[out] / rad[out, None]
            dr = d[out] - 2.0 * np.sum(d[out] * n, axis=1, keepdims=True) * n
            self.theta[out] = np.arctan2(dr[:, 1], dr[:, 0])
            nxt[out] = n * (R_ARENA * 0.98)
        self.p = nxt
        return cmd.astype(np.float32)

    def observe(self):
        """(obs, true, precision).

        `true` is the CLEAN state and is what the model is scored against, so a
        perfect predictor scores zero -- there is no irreducible noise floor.
        `obs` is what the model is handed.  Heading, contact and wall proximity
        are proprioceptive and are never degraded: the animal always knows which
        way it is pointing and what it is touching.
        """
        true = np.zeros((self.B, S_DIM))
        obs = np.zeros((self.B, S_DIM))
        prec = np.zeros((self.B, 4))          # bear0, bear1, shape0, shape1
        true[:, 0], true[:, 1] = np.sin(self.theta), np.cos(self.theta)
        touch = np.zeros(self.B)

        def degrade(clean, p, sd):
            """variance-preserving: keeps the channel's own scale at any p"""
            p = np.clip(p, 0.0, 1.0)
            return (np.sqrt(p) * clean
                    + np.sqrt(1.0 - p) * sd * self._orng.normal(size=clean.shape))

        for i in range(2):
            delta = self.obj[:, i] - self.p
            dist = np.linalg.norm(delta, axis=1)
            beta = wrap(np.arctan2(delta[:, 1], delta[:, 0]) - self.theta)
            # graded field of view: flat inside, fading at the rim
            A = BETA0 ** BETA_M / (np.abs(beta) ** BETA_M + BETA0 ** BETA_M)
            vis = P_FLOOR + (1.0 - P_FLOOR) * A
            p_bear = np.minimum(precision(dist, SENSE_R), vis)
            # shape has no angular term: whisker contact is not heading-selective
            p_shp = precision(dist, self.r_id)
            prec[:, i], prec[:, 2 + i] = p_bear, p_shp

            b = OBJ[i].start
            true[:, SEEN[i]] = obs[:, SEEN[i]] = p_bear    # reliability, exact
            true[:, FELT[i]] = obs[:, FELT[i]] = p_shp
            true[:, b] = np.sin(beta)
            true[:, b + 1] = np.cos(beta)
            true[:, b + 2] = np.clip(INT_R / np.maximum(dist, 1e-3), 0.0, 2.0)
            obs[:, b] = degrade(true[:, b], p_bear, SD_BEAR)
            obs[:, b + 1] = degrade(true[:, b + 1], p_bear, SD_BEAR)
            obs[:, b + 2] = degrade(true[:, b + 2], p_bear, SD_INT)
            true[:, SHAPE_OBJ[i]] = self.shape[:, i]
            obs[:, SHAPE_OBJ[i]] = degrade(self.shape[:, i], p_shp[:, None],
                                           SD_SHAPE)
            touch = np.maximum(touch, (dist < R_TOUCH).astype(float))

        for arr in (true, obs):
            arr[:, 0], arr[:, 1] = true[:, 0], true[:, 1]      # heading exact
            arr[:, 16] = touch
            arr[:, 17] = (R_ARENA - np.linalg.norm(self.p, axis=1)) < 0.08
        return (obs.astype(np.float32), true.astype(np.float32),
                prec.astype(np.float32))


def rollout(env, T):
    """(obs, true, precision, action) -- `true` is the target."""
    S, U, V, A = [], [], [], []
    for _ in range(T):
        s, u, v = env.observe()
        S.append(s), U.append(u), V.append(v)
        A.append(env.step())
    return (np.stack(S, 1), np.stack(U, 1), np.stack(V, 1),
            np.stack(A, 1).astype(np.float32))


def to_t(*a):
    return [torch.from_numpy(x).to(DEVICE) for x in a]


class MTL(nn.Module):
    """z_t = MTL([s_{t-1}, a_{t-1}, e_t], h).

    No current observation: the cerebellar prediction error is MTL's only
    account of what just happened.
    """

    def __init__(self, hidden=MTL_HIDDEN, latent=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden
        self.cell = nn.GRUCell(S_DIM + A_DIM + S_DIM, hidden)
        self.readout = nn.Sequential(nn.Linear(hidden, latent), nn.Tanh())

    def init_hidden(self, B, device):
        return torch.zeros(B, self.hidden_dim, device=device)

    def step(self, s_prev, a_prev, e, h):
        h = self.cell(torch.cat([s_prev, a_prev, e], -1), h)
        return self.readout(h), h


class Cerebellum(nn.Module):
    """The forward model: belief plus action to expected observation.

    It is not the integrator and does not need to be.  What a lesion destroys is
    the informativeness of the correction -- a prediction error only tells you
    about the world if the predictor was accurate.
    """

    def __init__(self, latent=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S_DIM + A_DIM + latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, S_DIM))
        nn.init.normal_(self.net[-1].weight, std=CB_INIT_STD)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s_in, a, z, noise=0.0, off=False):
        """`noise` may be a float or a (B, 1) gate, so the lesion can follow the
        duty cycle.  off=True kills CB's contribution entirely -- the control
        that asks what the circuit does with no forward model at all.

        The CLEAN copy carries the recursion so the belief stays bounded; only
        what CB TRANSMITS is corrupted.
        """
        d = self.net(torch.cat([s_in, a, z], -1))
        clean = s_in + torch.zeros_like(d) if off else s_in + d
        if isinstance(noise, torch.Tensor) or noise > 0:
            return clean, clean + noise * clean.std() * torch.randn_like(clean)
        return clean, clean


def run_sequence(mtl, cb, S, U, V, A, h0=None, s_hat0=None, collect=False,
                 cb_noise=0.0, e_noise=0.0, h_noise=0.0, mask=None,
                 cb_off=False):
    """One session.  Nothing is substituted and nothing is masked.

    s_in is always the observation, so the cerebellum never stands in for a
    missing input and its output never re-enters its own input.  The loss is
    taken against the TRUE state at every step.
    """
    B, T, _ = S.shape
    h = mtl.init_hidden(B, S.device) if h0 is None else h0
    s_hat = S[:, 0].clone() if s_hat0 is None else s_hat0
    s_prev = S[:, 0].clone()
    a_prev = torch.zeros(B, A_DIM, device=S.device)
    preds, targs, zs, es = [], [], [], []

    for t in range(T):
        s_t, a_t = S[:, t], A[:, t]
        s_in = s_t                       # no substitution, ever
        e_t = s_t - s_hat
        gate = 1.0 if mask is None else mask[:, t].unsqueeze(-1)

        if e_noise > 0:                  # sd-scaled, like every other site
            e_t = e_t + e_noise * gate * e_t.std() * torch.randn_like(e_t)

        z, h = mtl.step(s_prev, a_prev, e_t, h)

        if h_noise > 0:
            # divide by sqrt(hidden / S_DIM) so the 256-d hidden state and the
            # 20-d state vector carry equal TOTAL injected variance
            dm = float(np.sqrt(S_DIM / h.shape[-1])) if DIM_MATCH else 1.0
            h = h + h_noise * dm * gate * h.std() * torch.randn_like(h)

        nz = (cb_noise * gate) if cb_noise > 0 else 0.0
        s_hat_clean, s_hat = cb(s_in, a_t, z, noise=nz, off=cb_off)
        s_prev, a_prev = s_t, a_t

        if t < T - 1:
            preds.append(s_hat_clean)
            targs.append(U[:, t + 1])    # the CLEAN next state
        if collect:
            zs.append(z), es.append(e_t)

    out = {"pred": torch.stack(preds, 1), "target": torch.stack(targs, 1),
           "h_last": h, "s_hat_last": s_hat}
    if collect:
        out["z"] = torch.stack(zs, 1)
        out["e"] = torch.stack(es, 1)
    return out


def mse(res):
    return ((res["pred"] - res["target"]) ** 2).mean()


def duty_cycle(B, T, on=3, off=60, device=DEVICE):
    """Zeidler's stimulation protocol: 3 s on, every 60 s."""
    m = torch.zeros(B, T, device=device)
    for k in range(0, T, off):
        m[:, k:k + on] = 1.0
    return m


# =========================================================================
# Training
# =========================================================================
def ckpt_path(seed, d=None):
    return os.path.join(d or CKPT_DIR, f"olm_orm_final_s{seed}.pt")


def _resume(path, mtl, cb, opt, sched, rng):
    p = path + ".resume"
    if not (path and os.path.exists(p)):
        return 0
    ck = torch.load(p, map_location=DEVICE)
    mtl.load_state_dict(ck["mtl"]), cb.load_state_dict(ck["cb"])
    opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
    rng.bit_generator.state = ck["rng"]
    print(f"resuming from epoch {ck['ep']}", flush=True)
    return ck["ep"] + 1


def train(seed=0, epochs=EPOCHS, r_id=R_ID, out=None):
    """Next-state prediction only.  On P_SWITCH of sequences one object is
    moved or its identity changed partway through, so the network learns that
    these things sometimes change and it has to revise."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mtl, cb = MTL().to(DEVICE), Cerebellum().to(DEVICE)
    params = list(mtl.parameters()) + list(cb.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * STEPS_EP, eta_min=1e-5)
    start = _resume(out, mtl, cb, opt, sched, rng)

    for ep in range(start, epochs):
        run = 0.0
        for _ in range(STEPS_EP):
            env = ObjectArena(BATCH, rng, r_id=r_id)
            who = rng.random(BATCH) < P_SWITCH
            if who.any():
                t0 = int(rng.integers(40, SEQ_LEN - 40))
                S1, U1, V1, A1 = rollout(env, t0)
                mv = who & (rng.random(BATCH) < 0.5)   # half move, half swap
                env.move_object(0, rng=rng, who=mv)
                env.change_identity(0, rng=rng, who=who & ~mv)
                S2, U2, V2, A2 = rollout(env, SEQ_LEN - t0)
                S, U = np.concatenate([S1, S2], 1), np.concatenate([U1, U2], 1)
                V, A = np.concatenate([V1, V2], 1), np.concatenate([A1, A2], 1)
            else:
                S, U, V, A = rollout(env, SEQ_LEN)
            loss = mse(run_sequence(mtl, cb, *to_t(S, U, V, A)))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step()
            run += loss.item()
        if out and (ep % RESUME_EVERY == 0 or ep == epochs - 1):
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            torch.save({"mtl": mtl.state_dict(), "cb": cb.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "ep": ep},
                       out + ".resume")
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[seed {seed}] epoch {ep:3d}  lr {sched.get_last_lr()[0]:.1e}"
                  f"  mse {run / STEPS_EP:.5f}", flush=True)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"mtl": mtl.state_dict(), "cb": cb.state_dict()}, out)
        print(f"saved -> {out}", flush=True)
        if os.path.exists(out + ".resume"):
            os.remove(out + ".resume")
    return mtl, cb


def load(path):
    ck = torch.load(path, map_location=DEVICE)
    mtl, cb = MTL().to(DEVICE), Cerebellum().to(DEVICE)
    mtl.load_state_dict(ck["mtl"]), cb.load_state_dict(ck["cb"])
    mtl.eval(), cb.eval()
    return mtl, cb


# =========================================================================
# Measurement
# =========================================================================
_VAR = {}


def channel_variance(r_id=R_ID, seed=0, n=384, T=200):
    """Variance of each TRUE channel, used to normalise the errors.

    Without it the two tasks are not comparable: OLM is scored on bearing and
    apparent size (mean variance ~0.39) and ORM on the shape features (~0.53),
    so the same relative error reads larger for one than the other.  Measured on
    the true state over every step, since that is what the loss compares against.
    """
    if r_id not in _VAR:
        env = ObjectArena(n, np.random.default_rng(seed), r_id=r_id)
        _, U, _, _ = rollout(env, T)
        _VAR[r_id] = U.reshape(-1, S_DIM).var(0)
    return _VAR[r_id]


COND = [("intact", {}, "#1a1a1a"),
        ("cerebellar output", {"cb_noise": 0.8}, "#2f7ab8"),
        ("direct MTL", {"h_noise": 0.8}, "#7f8c8d"),
        ("CB copies its input forward", {"cb_off": True}, "#c0392b")]
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)


@torch.no_grad()
def two_sessions(mtl, cb, task, changed, seed, n=384, T=200, r_id=R_ID, **kw):
    """Explore, apply the manipulation, explore again with state carried over.

    Returns (per-step error on the task's channels, precision array).  Both
    conditions use the SAME seed, so the only difference between them is the
    manipulation itself.
    """
    rng, mrng = np.random.default_rng(seed), np.random.default_rng(seed + 99991)
    env = ObjectArena(n, rng, r_id=r_id)
    S, U, V, A = rollout(env, T)
    r0 = run_sequence(mtl, cb, *to_t(S, U, V, A), mask=duty_cycle(n, T), **kw)
    if changed:
        if task == "OLM":
            env.move_object(0, rng=mrng)
        else:
            env.change_identity(0, rng=mrng)
    S, U, V, A = rollout(env, T)
    r = run_sequence(mtl, cb, *to_t(S, U, V, A), h0=r0["h_last"],
                     s_hat0=r0["s_hat_last"], mask=duty_cycle(n, T), **kw)
    chans = TGT[task]
    err = ((r["pred"] - r["target"]) ** 2).cpu().numpy()[..., chans]
    err = err / channel_variance(r_id)[chans]      # comparable across tasks
    return err.mean(-1), V


def by_step(err, nmax=60):
    """Error at the n-th STEP since the change.  Every step, no filtering.

    An earlier version indexed by the n-th READING instead -- the n-th step
    whose precision cleared a threshold -- on the grounds that arenas differ
    enormously in how often they see the object, so time-aligning smears the
    curves.  That is true, but it put the figures on a different basis from the
    preference index, which is anchored to the change with no precision
    criterion, and the two then disagreed: the figure showed the cerebellar
    lesion closing the location gap while the index showed almost nothing.

    One basis for both.  PI is now literally the first PI_WINDOW points of these
    curves, so a reader can check it off the figure.
    """
    return err[:, :nmax]


def curves(mtl, cb, seed, n=384, r_id=R_ID):
    """{(task, condition, changed): error at the n-th step since the change}"""
    out = {}
    for task in ("OLM", "ORM"):
        for lab, kw, _ in COND:
            for changed in (True, False):
                e, V = two_sessions(mtl, cb, task, changed, seed, n=n,
                                    r_id=r_id, **kw)
                out[(task, lab, changed)] = by_step(e)
    return out


def ratio(cur, task, lab, upto=PI_WINDOW):
    """The preference: how much more surprising the changed object is."""
    c = np.nanmean(cur[(task, lab, True)][:, :upto])
    u = np.nanmean(cur[(task, lab, False)][:, :upto])
    return c, u, c / max(u, 1e-12)




N_READINGS = 5               # how much evidence the preference is read over








@torch.no_grad()
def preference_index(mtl, cb, task, sigma=0.0, site="cb", seed=0, n=256,
                     T=200, r_id=R_ID, k=PI_WINDOW):
    """PI = e_changed / (e_changed + e_unchanged) over the first k steps after
    the change.  0.5 means the change went unnoticed.

    ANCHORED TO THE CHANGE, WITH NO PRECISION CRITERION.  That is what the
    experiment does: the object is moved or swapped while the animal is out of
    the arena, it is put back, and exploration is counted from that moment.
    Nothing in the experiment locates a "re-encounter".

    Earlier versions anchored the window on the first step whose precision
    crossed a threshold.  That asks a finer question -- is the model surprised
    at the instant it senses the object again -- but the answer moves with the
    threshold, because the cerebellar contribution depends on how much the model
    must predict rather than see: score a moment of poor perception and the cost
    looks large, score a good one and it looks small.  Worse, a threshold loose
    enough to keep location measurable put identity at 0.997, where the index
    cannot move.  Anchoring on the change removes the choice.

    Why a short window: the novelty signal is a transient.  The model relearns
    the changed value within about ten readings, so averaging the whole session
    is mostly averaging recovered behaviour and pulls the index to chance.  The
    animal's session is longer than this, and the correspondence rests on
    exploration accumulating early because novelty wanes -- worth stating rather
    than leaving implicit.  The dissociation holds for k between 4 and 40.
    """
    kw = {f"{site}_noise": sigma} if sigma > 0 else {}
    ec, _ = two_sessions(mtl, cb, task, True, seed, n=n, T=T, r_id=r_id, **kw)
    eu, _ = two_sessions(mtl, cb, task, False, seed, n=n, T=T, r_id=r_id, **kw)
    c = float(np.nanmean(ec[:, :k]))
    u = float(np.nanmean(eu[:, :k]))
    return c / (c + u + 1e-12)


def dose_response(mtl, cb, seed, n=128, r_id=R_ID, sigmas=SIGMAS):
    """{(task, site, sigma): PI}"""
    return {(t, s, g): preference_index(mtl, cb, t, g, s, seed, n=n, r_id=r_id)
            for t in ("OLM", "ORM") for s in ("cb", "e", "h") for g in sigmas}


@torch.no_grad()
def preference_index_windows(mtl, cb, task, sigma=0.0, site="cb", seed=0,
                             n=256, T=200, r_id=R_ID, ks=PI_WINDOWS):
    """PI at several post-change windows, from ONE pair of sessions.

    The windows are nested -- k=1 is the first step, k=16 the first sixteen --
    so taking a different number of leading steps out of the same rollout costs
    nothing beyond the single-window call it replaces.  Returns {k: PI}.

    This exists to test whether the dissociation is an artefact of the k=4
    choice.  It is not a free parameter we tuned: the novelty signal is a
    transient and a long window averages in recovered behaviour, so the
    question is whether the answer is stable over the range where the signal
    is still present.
    """
    kw = {f"{site}_noise": sigma} if sigma > 0 else {}
    ec, _ = two_sessions(mtl, cb, task, True, seed, n=n, T=T, r_id=r_id, **kw)
    eu, _ = two_sessions(mtl, cb, task, False, seed, n=n, T=T, r_id=r_id, **kw)
    out = {}
    for k in ks:
        c = float(np.nanmean(ec[:, :k]))
        u = float(np.nanmean(eu[:, :k]))
        out[k] = c / (c + u + 1e-12)
    return out


def dose_response_windows(mtl, cb, seed, n=128, r_id=R_ID, sigmas=SIGMAS,
                          ks=PI_WINDOWS):
    """{(task, site, sigma, k): PI}.  Same cost as dose_response."""
    out = {}
    for t in ("OLM", "ORM"):
        for s in ("cb", "e", "h"):
            for g in sigmas:
                d = preference_index_windows(mtl, cb, t, g, s, seed, n=n,
                                             r_id=r_id, ks=ks)
                for k, v in d.items():
                    out[(t, s, g, k)] = v
    return out


def at_window(accw, k):
    """Slice a window-sweep dict down to the {(task, site, sigma): PI} form the
    existing figures and report expect."""
    return {(t, s, g): v for (t, s, g, kk), v in accw.items() if kk == k}


# =========================================================================
# Figures
# =========================================================================
def _save(fig, path, dpi=170):
    """Write the figure as PNG and as SVG beside it.

    SVG because the figures go into a manuscript: text stays selectable and the
    lines stay sharp at any zoom, which a raster cannot do.
    """
    fig.savefig(path, dpi=dpi)
    root = path[:-4] if path.lower().endswith(".png") else path
    fig.savefig(root + ".svg", format="svg")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_gap(cur, path, subtitle=""):
    """The gap between changed and unchanged IS the preference.

    Plotting them in separate panels with independent axes hides exactly what
    the figure is for, so they share an axis per task and the gap is shaded.
    """
    import matplotlib.pyplot as plt
    show = [c for c in COND if c[0] in ("intact", "cerebellar output")]
    fig, AX = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, task in zip(AX, ("OLM", "ORM")):
        for lab, _, col in show:
            ch = np.nanmean(cur[(task, lab, True)], 0)
            un = np.nanmean(cur[(task, lab, False)], 0)
            x = np.arange(1, len(ch) + 1)
            ax.plot(x, ch, color=col, lw=2.0, label=f"{lab} - changed")
            ax.plot(x, un, color=col, lw=2.0, ls=":", label=f"{lab} - unchanged")
            ax.fill_between(x, un, ch, color=col, alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("steps since the change")
        ax.set_title(f"{task}   (object "
                     f"{'location' if task == 'OLM' else 'identity'} changed)",
                     fontsize=11)
        _style(ax)
    AX[0].set_ylabel("squared error / channel variance")
    AX[0].legend(fontsize=8, frameon=False)
    fig.suptitle("The gap between changed and unchanged is the preference "
                 "signal\n" + subtitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, path)
    plt.close(fig)


def fig_four_conditions(cur, path, subtitle=""):
    """All four conditions, both tasks, changed and unchanged."""
    import matplotlib.pyplot as plt
    fig, AX = plt.subplots(2, 2, figsize=(12.5, 7.4), sharex=True)
    for i, task in enumerate(("OLM", "ORM")):
        for j, changed in enumerate((False, True)):
            ax = AX[i, j]
            for lab, _, col in COND:
                m = np.nanmean(cur[(task, lab, changed)], 0)
                ax.plot(np.arange(1, len(m) + 1), m, color=col,
                        lw=2.0 if lab == "intact" else 1.4, label=lab)
            ax.set_yscale("log")
            ax.set_title(f"{task}   "
                         f"{'CHANGED' if changed else 'control (unchanged)'}",
                         fontsize=10)
            _style(ax)
            if (i, j) == (0, 0):
                ax.legend(fontsize=8, frameon=False)
            if j == 0:
                ax.set_ylabel("squared error / channel variance")
    for ax in AX[1]:
        ax.set_xlabel("steps since the change")
    fig.suptitle("Error vs evidence, all four conditions\n" + subtitle,
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, path)
    plt.close(fig)


def fig_dose(acc, path, subtitle="", sigmas=SIGMAS):
    """PI vs lesion strength, by injection site.

    Plotted as PI itself, with chance drawn at 0.5.  Note that the raw index
    compresses: 0.5 is a total loss of preference, so a fall from 0.89 to 0.55
    is near-total even though it reads as a 38% change on this axis.  The cost
    table in numbers.txt expresses the same numbers as a fraction of the
    above-chance index, which is the undistorted version.
    """
    import matplotlib.pyplot as plt
    sites = (("cb", "#2f7ab8", "cerebellar output"),
             ("e", "#c0392b", "error channel only"),
             ("h", "#7f8c8d", "direct MTL site"))
    fig, AX = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, task in zip(AX, ("OLM", "ORM")):
        for s, col, lab in sites:
            y = [np.mean(acc[(task, s, g)]) for g in sigmas]
            e = [np.std(acc[(task, s, g)]) / max(np.sqrt(len(np.atleast_1d(
                acc[(task, s, g)]))), 1) for g in sigmas]
            ax.errorbar(sigmas, y, yerr=e, marker="o", ms=5, color=col, lw=1.9,
                        ls="--" if s == "h" else "-", label=lab, capsize=2)
        ax.axhline(0.5, color="#c0392b", lw=1.0, ls=":")     # chance
        ax.set_xlabel(r"lesion strength  $\sigma$")
        ax.set_title(f"{task}   (object "
                     f"{'location' if task == 'OLM' else 'identity'} changed)",
                     fontsize=11)
        _style(ax)
    AX[0].set_ylabel("preference index")
    AX[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Preference index above chance vs lesion strength\n" + subtitle,
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save(fig, path)
    plt.close(fig)


@torch.no_grad()
def forward_model_value(mtl, cb, seed, n=384, T=200, r_id=R_ID):
    """Next-state prediction error, cerebellum on vs replaced by identity.

    The identity condition is the right null for a forward model rather than a
    mere sanity check.  Since s_hat = s_in + CB(s_in, a, z), switching CB off
    leaves s_hat = s_in -- the prediction that the next state equals the current
    one.  Beating it is exactly the claim that the cerebellum predicts the
    sensory consequences of movement, and it is the same identity-cerebellum
    baseline that appears among the lesion conditions later.

    Returns {condition: {group: MSE}} over channel groups, normalised by each
    channel's own variance so groups on different scales are comparable.
    """
    rng = np.random.default_rng(seed)
    env = ObjectArena(n, rng, r_id=r_id)
    S, U, V, A = rollout(env, T)
    args = to_t(S, U, V, A)
    var = channel_variance(r_id)
    groups = {"all channels": list(range(S_DIM)),
              "object location\n(OLM channels)": TGT["OLM"],
              "object identity\n(ORM channels)": TGT["ORM"]}
    out = {}
    for lab, kw in (("cerebellar forward model", {}),
                    ("CB copies its input forward", {"cb_off": True})):
        r = run_sequence(mtl, cb, *args, mask=duty_cycle(n, T), **kw)
        e = ((r["pred"] - r["target"]) ** 2).cpu().numpy()
        out[lab] = {g: float(np.mean(e[..., idx] / var[idx]))
                    for g, idx in groups.items()}
    return out


def fig_forward_value(per_seed, path, subtitle=""):
    """Does the forward model beat assuming the world is static?"""
    import matplotlib.pyplot as plt
    conds = ["cerebellar forward model", "CB copies its input forward"]
    groups = list(per_seed[0][conds[0]].keys())
    cols = {conds[0]: "#1a1a1a", conds[1]: "#c0392b"}
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, c in enumerate(conds):
        v = np.array([[p[c][g] for g in groups] for p in per_seed])
        m, e = v.mean(0), v.std(0) / max(np.sqrt(len(v)), 1)
        ax.bar(x + (i - 0.5) * w, m, w, yerr=e, color=cols[c], label=c,
               capsize=3, edgecolor="none")
    a = np.array([[p[conds[0]][g] for g in groups] for p in per_seed]).mean(0)
    b = np.array([[p[conds[1]][g] for g in groups] for p in per_seed]).mean(0)
    for xi, (lo, hi) in enumerate(zip(a, b)):
        ax.text(xi, hi * 1.15, f"{hi / max(lo, 1e-12):.0f}$\\times$",
                ha="center", fontsize=10, color="#444")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel("next-state prediction error\n(variance-normalised MSE)")
    ax.legend(fontsize=9, frameon=False)
    _style(ax)
    ax.set_title("The cerebellum predicts the consequences of movement\n"
                 + subtitle, fontsize=10.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


@torch.no_grad()
def r2_by_channel(mtl, cb, seed, n=384, T=200, r_id=R_ID):
    """Per-channel R^2 for the forward model and for persistence.

    R^2 = 1 - MSE/var, so 0 is predicting the channel's average and 1 is exact.
    Persistence -- s_hat = s_in, the prediction that nothing changes -- is the
    right null for a forward model, and because there is no substitution in this
    configuration it is also "copy the previous observation": the two baselines
    of earlier versions collapse into one.

    R^2 compresses at the top, where all the interesting differences are: 0.92
    to 0.98 reads as nothing but is a fourfold cut in residual variance.  We
    therefore return the residual ratio alongside it and annotate the figure
    with it.
    """
    rng = np.random.default_rng(seed)
    env = ObjectArena(n, rng, r_id=r_id)
    S, U, V, A = rollout(env, T)
    args = to_t(S, U, V, A)
    out = {}
    for lab, kw in (("cb", {}), ("lesion", {"cb_noise": LESION_SIGMA}),
                    ("persist", {"cb_off": True})):
        r = run_sequence(mtl, cb, *args, mask=duty_cycle(n, T), **kw)
        pr = r["pred"].cpu().numpy()
        tg = r["target"].cpu().numpy()
        m = ((pr - tg) ** 2).mean((0, 1))
        v = tg.reshape(-1, tg.shape[-1]).var(0)
        out[lab] = 1.0 - m / np.maximum(v, 1e-12)
        out[lab + "_mse"] = m / np.maximum(v, 1e-12)
    return out


def fig_r2_by_channel(per_seed, path, subtitle=""):
    """How well the forward model predicts each channel, against persistence."""
    import matplotlib.pyplot as plt
    R = {k: np.array([p[k] for p in per_seed]) for k in per_seed[0]}
    x = np.arange(S_DIM); w = 0.27
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    for i, (k, col, lab) in enumerate(
            (("cb", "#c0392b", "cerebellar forward model"),
             ("lesion", "#2f7ab8",
              f"cerebellar lesion  ($\\sigma$={LESION_SIGMA})"),
             ("persist", "#7f8c8d", "copy the observation forward"))):
        m = R[k].mean(0); e = R[k].std(0) / max(np.sqrt(len(R[k])), 1)
        ax.bar(x + (i - 1) * w, m, w, yerr=e, color=col, label=lab,
               capsize=2, edgecolor="none")
    fold = R["persist_mse"].mean(0) / np.maximum(R["cb_mse"].mean(0), 1e-12)
    top = np.maximum(R["cb"].mean(0), R["persist"].mean(0))
    for xi, (f, t) in enumerate(zip(fold, top)):
        ax.text(xi, t + 0.035, f"{f:.0f}$\\times$", ha="center", fontsize=8.5,
                color="#c0392b")
    ax.axhline(0.0, color="#1a1a1a", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CH_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("$R^2$  --  fraction of the channel's\nvariance explained")
    ax.set_ylim(min(0.0, float(R["persist"].mean(0).min()) - 0.08), 1.34)
    ax.text(-0.4, 0.03, "0  =  predicting the channel's average",
            fontsize=8.5, color="#666", ha="left")
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    _style(ax)
    ax.set_title("How well the forward model predicts each channel\n"
                 + subtitle + "\nred = fold reduction in residual variance "
                 "over persistence ($R^2$ compresses; this does not)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def fig_r2_targets(per_seed, path, subtitle="", conds=None):
    """The six channels the two tasks are scored on, forward model vs persistence.

    One panel, not two: the OLM and ORM measures are built from different
    channels of the SAME state vector, so putting them on one axis is the
    comparison.  Only object 0 appears -- it is the object the manipulation
    moves or swaps.  Per-seed points are drawn over the bars because with ten
    networks the individual values carry the claim better than an error bar.
    """
    import matplotlib.pyplot as plt
    idx = list(TGT["OLM"]) + list(TGT["ORM"])
    names = [CH_NAMES[i] for i in idx]
    R = {k: np.array([p[k] for p in per_seed])[:, idx] for k in per_seed[0]}
    x = np.arange(len(idx)); w = 0.27 if (conds and len(conds) > 2) else 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ALL = (("cb", "#c0392b", "cerebellar forward model"),
           ("lesion", "#2f7ab8",
            f"cerebellar lesion  ($\\sigma$={LESION_SIGMA})"),
           ("persist", "#7f8c8d", "copy the observation forward"))
    # The lesion bar is OPTIONAL and off by default.  Its R^2 here is averaged
    # over the whole session, of which the 3-on/60-off duty cycle leaves ~94%
    # unstimulated, so it understates the lesion by fivefold -- during a burst
    # location residual rises 24x, not 4.5x.  Preference is read inside a burst,
    # so putting the diluted number beside the preference result would invite
    # the conclusion that prediction survives the lesion.  It does not; only
    # IDENTITY prediction does.
    use = conds or ("cb", "persist")
    bars = [c for c in ALL if c[0] in use]
    off = (len(bars) - 1) / 2.0
    for i, (k, col, lab) in enumerate(bars):
        m = R[k].mean(0)
        ax.bar(x + (i - off) * w, m, w, color=col, label=lab, edgecolor="none",
               zorder=2)
        if len(R[k]) > 1:                      # every network, not just the mean
            for j in range(len(R[k])):
                ax.plot(x + (i - off) * w, R[k][j], ls="none", marker="o",
                        ms=2.6, mfc="white", mec="#1a1a1a", mew=0.6, zorder=3,
                        alpha=0.85)
    fold = R["persist_mse"].mean(0) / np.maximum(R["cb_mse"].mean(0), 1e-12)
    top = np.maximum(R["cb"].mean(0), R["persist"].mean(0))
    for xi, (f, t) in enumerate(zip(fold, top)):
        ax.text(xi, t + 0.03, f"{f:.0f}$\\times$", ha="center", fontsize=9.5,
                color="#c0392b")
    lo = float(R["persist"].mean(0).min())
    ax.set_ylim(min(0.0, lo - 0.10), 1.30)
    for a, b, lab in ((0, 2, "object location  (OLM)"),
                      (3, 5, "object identity  (ORM)")):
        y = 1.17
        ax.plot([a - 0.42, b + 0.42], [y, y], color="#1a1a1a", lw=1.1)
        ax.text((a + b) / 2.0, y + 0.02, lab, ha="center", fontsize=9.5)
    ax.axhline(0.0, color="#1a1a1a", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("$R^2$  --  fraction of the channel's variance explained")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    _style(ax)
    ax.set_title("The forward model predicts what movement will do to the "
                 "senses\n" + subtitle, fontsize=10.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


@torch.no_grad()
def mtl_by_precision(mtl, cb, seed, n=384, T=200, r_id=R_ID, sigma=LESION_SIGMA,
                     bins=(0.0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01)):
    """Does the forward model lean on MTL's latent when the observation is poor?

    The architecture forces the configuration to live in MTL -- the cerebellar
    module is feedforward, so nothing else can persist -- but that is a deduction
    rather than a measurement.  Here we measure it.  We bin every step by how
    precisely object 0 is currently sensed and compare prediction error with and
    without noise on the MTL recurrent state.  If the latent is carrying a
    remembered estimate, corrupting it should cost nothing where the object is
    seen clearly and progressively more as the reading degrades, because only
    then must the prediction come from memory rather than from the input.

    Returns {task: {"centres", "intact", "lesion", "n"}}.
    """
    rng = np.random.default_rng(seed)
    env = ObjectArena(n, rng, r_id=r_id)
    S, U, V, A = rollout(env, T)
    args = to_t(S, U, V, A)
    var = channel_variance(r_id)
    out = {}
    err = {}
    for lab, kw in (("intact", {}), ("lesion", {"h_noise": sigma})):
        r = run_sequence(mtl, cb, *args, mask=duty_cycle(n, T), **kw)
        e = ((r["pred"] - r["target"]) ** 2).cpu().numpy()
        err[lab] = e
    # precision of object 0 at the step being PREDICTED (t+1), bearing and shape
    prec = {"OLM": V[:, 1:, 0], "ORM": V[:, 1:, 2]}
    for task in ("OLM", "ORM"):
        idx = TGT[task]
        p = prec[task]
        c, a, b, cnt = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() < 200:
                continue
            c.append(0.5 * (lo + min(hi, 1.0)))
            for lab, acc in (("intact", a), ("lesion", b)):
                v = (err[lab][..., idx] / var[idx]).mean(-1)
                acc.append(float(v[m].mean()))
            cnt.append(int(m.sum()))
        out[task] = {"centres": np.array(c), "intact": np.array(a),
                     "lesion": np.array(b), "n": np.array(cnt)}
    return out


def fig_mtl_by_precision(per_seed, path, subtitle=""):
    """Prediction error against how well the object is currently sensed."""
    import matplotlib.pyplot as plt
    fig, AX = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, task in zip(AX, ("OLM", "ORM")):
        c = per_seed[0][task]["centres"]
        for k, col, lab in (("intact", "#1a1a1a", "intact"),
                            ("lesion", "#7f8c8d",
                             f"noise on the MTL state  ($\\sigma$={LESION_SIGMA})")):
            v = np.array([p[task][k] for p in per_seed])
            m = v.mean(0); e = v.std(0) / max(np.sqrt(len(v)), 1)
            ax.errorbar(c, m, yerr=e, marker="o", ms=5, lw=2.0, color=col,
                        label=lab, capsize=3,
                        ls="--" if k == "lesion" else "-")
        ax.set_yscale("log")
        ax.set_xlabel("how precisely the object is currently sensed")
        ax.set_title(f"{task} channels "
                     f"({'bearing and size' if task == 'OLM' else 'shape'})",
                     fontsize=11)
        _style(ax)
    AX[0].set_ylabel("next-state prediction error\n(variance-normalised)")
    AX[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("The forward model falls back on MTL's latent when the "
                 "observation is poor\n" + subtitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _save(fig, path)
    plt.close(fig)


def fig_window_sweep(accw, path, subtitle="", sigmas=SIGMAS, ks=PI_WINDOWS):
    """PI vs lesion strength, one column per post-change window.

    The k=4 window used everywhere else is one column of this figure.  If the
    dissociation were an artefact of that choice the columns would disagree.
    """
    import matplotlib.pyplot as plt
    sites = (("cb", "#2f7ab8", "cerebellar output"),
             ("e", "#c0392b", "error channel only"),
             ("h", "#7f8c8d", "direct MTL site"))
    fig, AX = plt.subplots(2, len(ks), figsize=(3.1 * len(ks), 6.4),
                           sharey="row", sharex=True)
    for r, task in enumerate(("OLM", "ORM")):
        for c, k in enumerate(ks):
            ax = AX[r, c]
            for site, col, lab in sites:
                v = [np.atleast_1d(accw[(task, site, g, k)]) for g in sigmas]
                y = [np.mean(a) for a in v]
                e = [np.std(a) / max(np.sqrt(len(a)), 1) for a in v]
                ax.errorbar(sigmas, y, yerr=e, marker="o", ms=4, color=col,
                            lw=1.8, ls="--" if site == "h" else "-",
                            label=lab, capsize=2)
            ax.axhline(0.5, color="#c0392b", lw=1.0, ls=":")
            _style(ax)
            if r == 0:
                ax.set_title(f"window = {k} step{'s' if k > 1 else ''}",
                             fontsize=10.5)
            if r == 1:
                ax.set_xlabel(r"lesion strength  $\sigma$")
            if c == 0:
                ax.set_ylabel(f"{task}\npreference index", fontsize=10)
    AX[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Preference index vs lesion strength, by post-change window\n"
                 + subtitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, path)
    plt.close(fig)


def fig_window_cost(accw, path, subtitle="", ks=PI_WINDOWS, sigma=0.8):
    """The dissociation itself against window length.

    y is the fraction of ABOVE-CHANCE preference lost, which is the undistorted
    version of the effect -- PI alone compresses, badly so for ORM where the
    intact index sits at 0.999.  A flat pair of curves means the k=4 result is
    not a window artefact.
    """
    import matplotlib.pyplot as plt
    sites = (("cb", "#2f7ab8", "cerebellar output"),
             ("h", "#7f8c8d", "direct MTL site"))
    fig, AX = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, task in zip(AX, ("OLM", "ORM")):
        for site, col, lab in sites:
            y, e = [], []
            for k in ks:
                base = np.atleast_1d(accw[(task, "cb", 0.0, k)]) - 0.5
                cur = np.atleast_1d(accw[(task, site, sigma, k)]) - 0.5
                c = 100.0 * (1.0 - cur / np.maximum(base, 1e-12))
                y.append(np.mean(c))
                e.append(np.std(c) / max(np.sqrt(len(c)), 1))
            ax.errorbar(ks, y, yerr=e, marker="o", ms=5, color=col, lw=2.0,
                        ls="--" if site == "h" else "-", label=lab, capsize=3)
        ax.axhline(0.0, color="#999", lw=1.0, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(ks))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("post-change window (steps)")
        ax.set_title(f"{task}   (object "
                     f"{'location' if task == 'OLM' else 'identity'} changed)",
                     fontsize=11)
        _style(ax)
    AX[0].set_ylabel(f"% of above-chance PI lost at $\\sigma$={sigma}")
    AX[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Is the dissociation specific to the 4-step window?\n"
                 + subtitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save(fig, path)
    plt.close(fig)


def fig_channel_variance(path, seed=0, n=384, r_id=R_ID):
    """Which channels each task is scored on, and their variance.

    The answer to "does shape dominate the variance, and is that why ORM is
    robust?" -- a number rather than an argument.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    env = ObjectArena(n, np.random.default_rng(seed), r_id=r_id)
    _, U, _, _ = rollout(env, 200)
    v = U.reshape(-1, S_DIM).var(0)
    col = ["#d5d8da"] * S_DIM
    for i in TGT["OLM"]:
        col[i] = "#2f7ab8"
    for i in TGT["ORM"]:
        col[i] = "#c0392b"
    fig, ax = plt.subplots(figsize=(10.5, 4))
    ax.bar(np.arange(S_DIM), v, color=col)
    ax.set_xticks(np.arange(S_DIM))
    ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=8)
    ax.set_ylabel("variance of the true state")
    _style(ax)
    ax.legend(handles=[
        Patch(fc="#2f7ab8", label=f"scored for OLM (mean "
                                  f"{v[TGT['OLM']].mean():.2f})"),
        Patch(fc="#c0392b", label=f"scored for ORM (mean "
                                  f"{v[TGT['ORM']].mean():.2f})"),
        Patch(fc="#d5d8da", label="not scored")], fontsize=8.5, frameon=False)
    ax.set_title("Per-channel variance and which channels each task is scored on",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# =========================================================================
# Across seeds
# =========================================================================
def pool_curves(per_seed):
    """Mean the error curves across seeds.

    Each seed is a separately trained network, so the spread across them is the
    error bar that matters -- varying only the arena draw within one network
    says the effect is reliable given that network, not that it survives
    retraining.
    """
    keys = per_seed[0].keys()
    return {k: np.concatenate([p[k] for p in per_seed], 0) for k in keys}


def pool_dose(per_seed):
    """{(task, site, sigma): array of one PI per seed}"""
    return {k: np.array([p[k] for p in per_seed]) for k in per_seed[0]}


def report(cur, acc, seeds, fh=None, accw=None, wsig=0.8):
    """The numbers behind the figures, printed and optionally written out."""
    def emit(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")

    emit(f"seeds: {list(seeds)}")
    emit(f"\nerror over the first {PI_WINDOW} steps since the change")
    emit(f"{'task':>4} {'condition':>30} {'changed':>9} {'unchanged':>11} "
         f"{'ratio':>8}")
    for task in ("OLM", "ORM"):
        for lab, _, _ in COND:
            c, u, r = ratio(cur, task, lab)
            emit(f"{task:>4} {lab:>30} {c:9.3f} {u:11.3f} {r:8.1f}")
    emit("\npreference index (mean over seeds)")
    emit(f"{'task':>4} {'site':>5} " + "".join(f"{g:>8}" for g in SIGMAS))
    for task in ("OLM", "ORM"):
        for s in ("cb", "e", "h"):
            emit(f"{task:>4} {s:>5} "
                 + "".join(f"{np.mean(acc[(task, s, g)]):8.3f}" for g in SIGMAS))
    emit("\ncost: % of above-chance PI lost")
    emit(f"{'sigma':>6} | {'OLM CB':>7} {'OLM MTL':>8} | {'ORM CB':>7} "
         f"{'ORM MTL':>8}")
    for g in SIGMAS:
        def cost(t, s):
            base = np.mean(acc[(t, "cb", 0.0)]) - 0.5
            return 100 * (1 - (np.mean(acc[(t, s, g)]) - 0.5) / max(base, 1e-12))
        emit(f"{g:6.1f} | {cost('OLM','cb'):6.1f}% {cost('OLM','h'):7.1f}% |"
             f" {cost('ORM','cb'):6.1f}% {cost('ORM','h'):7.1f}%")
    if accw is not None:
        emit(f"\nPI by post-change window (mean over seeds, sigma={wsig})")
        emit(f"{'task':>4} {'site':>5} " + "".join(f"{k:>8}" for k in PI_WINDOWS)
             + "   | % of above-chance PI lost")
        for task in ("OLM", "ORM"):
            for st in ("cb", "e", "h"):
                pis = [np.mean(accw[(task, st, wsig, k)]) for k in PI_WINDOWS]
                cost = [100 * (1 - (np.mean(accw[(task, st, wsig, k)]) - .5)
                        / max(np.mean(accw[(task, "cb", 0.0, k)]) - .5, 1e-12))
                        for k in PI_WINDOWS]
                emit(f"{task:>4} {st:>5} " + "".join(f"{v:8.3f}" for v in pis)
                     + "   | " + "".join(f"{v:7.1f}%" for v in cost))
    if len(seeds) == 1:
        emit("\nCAVEAT: one seed, no error bars across networks.")


def figures(seeds, n=384, n_pi=384, r_id=R_ID, out=None):
    """One folder per seed, plus pooled_ac_seeds/ when there is more than one.
    Everything lands under figures/main/OLM_ORM_FINAL/."""
    out = out or FIG_DIR
    os.makedirs(out, exist_ok=True)
    # SEED TORCH TOO.  The arena draw is seeded, but the lesions use
    # torch.randn_like, so without this every run draws different noise and the
    # figures are not reproducible -- PI at sigma=0.5 moved by 0.035 between two
    # runs of the same command before this was added.
    torch.manual_seed(0)
    per_curves, per_dose, per_fv, per_r2, done = [], [], [], [], []
    for s in seeds:
        p = ckpt_path(s)
        if not os.path.exists(p):
            print(f"no checkpoint for seed {s} -- run --train first ({p})")
            continue
        mtl, cb = load(p)
        print(f"[seed {s}] curves ...", flush=True)
        c = curves(mtl, cb, s, n=n, r_id=r_id)
        # the window sweep runs the same sessions the single-window dose
        # response would have, so this costs no more than dose_response did
        print(f"[seed {s}] dose-response (windows {list(PI_WINDOWS)}) ...",
              flush=True)
        aw = dose_response_windows(mtl, cb, s, n=n_pi, r_id=r_id)
        a = at_window(aw, PI_WINDOW)
        fv = forward_model_value(mtl, cb, s, n=n, r_id=r_id)
        r2 = r2_by_channel(mtl, cb, s, n=n, r_id=r_id)
        per_curves.append(c), per_dose.append(aw), per_fv.append(fv)
        per_r2.append(r2)
        done.append(s)
        d = os.path.join(out, f"seed{s}")
        os.makedirs(d, exist_ok=True)
        one = {k: np.array([v]) for k, v in a.items()}
        onew = {k: np.array([v]) for k, v in aw.items()}
        sub = f"seed {s}, reach {r_id}, n={n} arenas"
        fig_forward_value([fv], os.path.join(d, "forward_model_value.png"), sub)
        fig_r2_by_channel([r2], os.path.join(d, "r2_by_channel.png"), sub)
        fig_r2_targets([r2], os.path.join(d, "r2_targets.png"), sub)
        fig_gap(c, os.path.join(d, "gap_ALL_X.png"), sub)
        fig_four_conditions(c, os.path.join(d, "four_conditions_ALL_X.png"), sub)
        fig_dose(one, os.path.join(d, "pi_dose_ALL_X.png"), sub)
        fig_window_sweep(onew, os.path.join(d, "pi_window_sweep.png"), sub)
        fig_window_cost(onew, os.path.join(d, "pi_window_cost.png"), sub)
        fig_channel_variance(os.path.join(d, "channel_variance.png"),
                             seed=s, n=n, r_id=r_id)
        with open(os.path.join(d, "numbers_ALL_X.txt"), "w") as fh:
            report(c, one, [s], fh, accw=onew)
        print(f"[seed {s}] figures -> {d}", flush=True)

    if not per_curves:
        return
    if len(per_curves) > 1:
        cur = pool_curves(per_curves)
        accw = pool_dose(per_dose)              # {(task, site, sigma, k): [per seed]}
        acc = at_window(accw, PI_WINDOW)
        pool = os.path.join(out, "pooled_ac_seeds")
        os.makedirs(pool, exist_ok=True)
        sub = (f"{len(per_curves)} seeds, reach {r_id}, n={n} arenas per seed; "
               f"error bars are across independently trained networks")
        fig_forward_value(per_fv, os.path.join(pool,
                          "forward_model_value.png"), sub)
        fig_r2_by_channel(per_r2, os.path.join(pool, "r2_by_channel.png"), sub)
        fig_r2_targets(per_r2, os.path.join(pool, "r2_targets.png"), sub)
        fig_gap(cur, os.path.join(pool, "gap_ALL_X.png"), sub)
        fig_four_conditions(cur, os.path.join(pool,
                            "four_conditions_ALL_X.png"), sub)
        fig_dose(acc, os.path.join(pool, "pi_dose_ALL_X.png"), sub)
        fig_window_sweep(accw, os.path.join(pool, "pi_window_sweep.png"), sub)
        fig_window_cost(accw, os.path.join(pool, "pi_window_cost.png"), sub)
        fig_channel_variance(os.path.join(pool, "channel_variance.png"),
                             seed=done[0], n=n, r_id=r_id)
        with open(os.path.join(pool, "numbers_ALL_X.txt"), "w") as fh:
            report(cur, acc, done, fh, accw=accw)
        np.savez(os.path.join(pool, "pooled.npz"),
                 **{f"cur|{k}": v for k, v in cur.items()},
                 **{"dose|" + "|".join(map(str, k)): v
                    for k, v in accw.items()})
        print(f"\npooled over {len(done)} seeds -> {pool}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="OLM/ORM in an MTL-cerebellar circuit (final configuration)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--n", type=int, default=384, help="arenas for the curves")
    ap.add_argument("--n-pi", type=int, default=384,
                    help="arenas for the PI sweep.  PI is read off the same "
                         "curves as the figures, so there is no reason for this "
                         "to differ from --n; at 96 the ratio swings 40% "
                         "between draws, at 384 about 11%.")
    ap.add_argument("--r-id", type=float, default=R_ID, help="whisker reach")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not (args.train or args.figures):
        ap.print_help()
        return
    if args.train:
        for s in args.seeds:
            p = ckpt_path(s)
            if os.path.exists(p):
                print(f"seed {s}: checkpoint exists, skipping -> {p}")
                continue
            train(seed=s, epochs=args.epochs, r_id=args.r_id, out=p)
    if args.figures:
        figures(args.seeds, n=args.n, n_pi=args.n_pi, r_id=args.r_id,
                out=args.out)


if __name__ == "__main__":
    main()
