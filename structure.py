"""
structure.py -- does the circuit learn the FAMILY as a continuous axis?

Train on a few anchor perturbations, then test on values never seen.  If the
network has learned "rotation" rather than three separate rotations, prediction
should stay good between the anchors and z should vary smoothly with the angle.

    rotation   train  theta in {0, 45, 90} deg      test  -30..120 in 5 deg steps
    drift      train  d     in {0.4, 0.8, 1.2}      test  0.0..1.6 in 0.05 steps

Both sweeps include INTERPOLATION (between anchors) and EXTRAPOLATION (outside
them), and the drift sweep includes d = 0, i.e. no perturbation at all.

WHY THIS IS A REWRITE, NOT A PORT.  The earlier version of this experiment
(paper_code/structure_learning_{rot,drift}.py) was built on the task as it stood
before several changes, and two of them change what the result means:

  * it gave MTL the current state -- torch.cat([s_tm1, a_tm1, s_t, e_t]) -- so
    it is our CONTROL architecture, the one in which the cerebellar error is
    informationally redundant (zeroing e costs it 0.011).  Whatever structure
    learning it showed, it showed in the arm where the loop is not load-bearing.
  * context was CONSTANT within a sequence, so it tested "identify one fixed
    angle from a stream", not contextual inference over time.

It also predated damping (vel_new = vel + f, undamped, with DT = 1.0 in a
4-unit box -- the regime that pins the agent against a wall most of the time),
correlated actions, the [pos, vel] state, and the dt-normalised weighted loss.

THIS VERSION inherits everything from the manuscript task by construction: the
generator emits transitions in the same (N, T, 6) format, so model.build,
model.rollout, model.EC/CB and model.weighted_loss are reused unchanged.  Only
the CONTEXT SET differs -- a handful of anchors from one family instead of the
16 fixed contexts.  And it runs BOTH arms, which is the question the earlier
version could not ask: is structure learning a property of the cerebellar loop,
or of the readout?

TRAINING USES SWITCHING, TESTING DOES NOT.  Training draws hazard-switching
sequences over the anchors, matching the regime the main result is trained in.
Each test value is then presented in its own constant-context sequences, which
is the clean way to ask "what does the circuit do at this angle" without
confounding it with how fast it detects a switch.

  python structure.py --family rotation --train --arms main control --seeds 0
  python structure.py --family rotation --sweep --arms main control --seeds 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import model as M
from task import (POS_LIM, SEQ_LEN, STAY_PROB, RHO, TSS_MIN, DT, MASS, V_MAX,
                  DAMPING, N_TRAIN, BATCH, EPOCHS, LR,
                  rot_mat, refl_mat, env_step_ab, next_action,
                  initial_action)
from data import GAUSS_SD
from decode import seq_split

FAMILIES = {
    "rotation": {"train": [0.0, 45.0, 90.0],
                 "test": np.arange(-30.0, 121.0, 5.0),
                 "unit": "deg", "label": r"rotation angle $\theta$ (deg)"},
    # Same sweep as rotation so the two are directly comparable: same units,
    # same anchors, same held-out values.  The difference is that reflection is
    # orientation-REVERSING (det -1) and its angular offset depends on the
    # action's direction, where rotation's is constant -- see task.refl_mat.
    "reflection": {"train": [0.0, 45.0, 90.0],
                   "test": np.arange(-30.0, 121.0, 5.0),
                   "unit": "deg", "label": r"mirror-line angle $\phi$ (deg)"},
    # DRAG AND CURL ARE NO LONGER CONTEXTS, and sweeping them here is now a
    # GENERALIZATION probe rather than an interpolation one.  The context set
    # is flat -- rotations and reflections only -- and a light drag is baked
    # into the plant, so a circuit trained on it has never seen a
    # velocity-dependent field VARY.  Asking whether it nonetheless represents
    # a curl gain as an axis is a strictly harder question than the rotation
    # sweep, and the two should not be read as the same test.
    #
    # Both sweeps ride ON TOP of the plant's constant DAMPING, so `drag` here
    # means EXTRA drag beyond it and 0.0 means the plant's own value.
    "drag":     {"train": [0.5 * float(DAMPING), float(DAMPING),
                           2.0 * float(DAMPING)],
                 "test": np.round(np.arange(0.0, 2.51 * DAMPING, DAMPING / 8), 2),
                 "unit": "", "label": "extra drag coefficient $b$"},
    "curl":     {"train": [10.0, 20.0, 30.0],
                 "test": np.round(np.arange(0.0, 75.1, 2.5), 2),
                 "unit": "", "label": "curl gain $k$"},
}


def speed_diagnostics(family, params, n_seqs=40, seed=0):
    """Mean speed and cap-saturated fraction at each parameter value.

    REPORT THIS BESIDE ANY SWEEP RESULT.  The field factor is only identifiable
    while the agent is moving and below the cap -- at |v| = 0 only the aim
    levels separate (4 modes of 16) and at |v| = V_MAX the contexts merge again
    (13 of 16).  Curl's sweep leaves that band at the top by saturating, drag's
    by stalling, and in both cases the accuracy drop looks identical to a
    failure to extrapolate.  Reporting it makes the difference visible instead
    of arguable.
    """
    out = []
    for p in np.atleast_1d(np.asarray(params, dtype=float)):
        tr, _ = gen(family, [float(p)], n_seqs, seed=seed, switch=False)
        spd = np.sqrt((tr[..., 8:10].numpy() ** 2).sum(-1))
        out.append((float(p), float(spd.mean()),
                    float(np.mean(spd >= V_MAX * 0.999))))
    return out


def out_dir(seed):
    """figures/main/seed<N>/ -- the same folders as every other figure.

    Filenames are prefixed `structure_` so they sit alongside ablation.png,
    pca.png and the rest without colliding.

    They go under `main` rather than being duplicated into `control` because a
    single structure figure plots BOTH arms plus the floor and ceiling: it is
    one comparison, not a per-arm result, and `main` is the headline arm.
    """
    return M.fig_dir("main", seed)


def matrices(family, param):
    """(A, B) for a continuous parameter, in task's own F = A a + B v form.

    This is the whole point of the rewrite: the families ARE the two factors of
    the factorial, swept continuously instead of at four levels.  rotation and
    reflection move A with B = 0; drag and curl move B with A = I.  Feeding
    these to task.env_step_ab means the sweep runs on the SAME plant as
    everything else rather than on a copy -- structure.py used to carry its own
    integration, and that copy still had the damping term and the [0, GRID]
    clip long after both were removed from the task.
    """
    I = np.eye(2, dtype=np.float32)
    Z = np.zeros((2, 2), dtype=np.float32)
    if family == "rotation":
        return rot_mat(np.deg2rad(param)), Z
    if family == "reflection":
        return refl_mat(np.deg2rad(param)), Z
    if family == "curl":
        # the exact discrete rotation, not param * R(90): the naive form is
        # expansive under explicit Euler and the sweep would measure the cap
        return I, (MASS * (rot_mat(param / MASS) - I)).astype(np.float32)
    if family == "drag":
        return I, (-param * I).astype(np.float32)
    raise KeyError(f"unknown family {family!r}; known: {list(FAMILIES)}")


def gen(family, params, n_seqs, seed=0, seq_len=SEQ_LEN, switch=True,
        stay=STAY_PROB, rho=RHO):
    """Sequences over a set of parameter values, in the (N, T, 10) format
    model.build expects.  switch=False holds one value for the whole sequence.

    VELOCITY IS STORED, not reconstructed from positions -- see data.py: the
    identity pos' = pos + vel'*DT is exact in the interior but wrong at a wall,
    and all the context information lives in the velocity.

    NOISE IS ON.  env_step_ab gets the sequence rng, so these sequences carry
    the same command noise the main task does.  Without it an ideal observer
    identifies the parameter from two steps and the whole question of whether
    structure is learned over time becomes moot.

    The action process is AR(1) Gaussian and is NOT reset at a switch, exactly
    as in data.generate_sequences -- resetting it would leak the switch time.
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(params, dtype=float)
    mats = [matrices(family, float(p)) for p in P]
    T_, L_ = [], []
    for _ in range(n_seqs):
        pos = rng.uniform(-0.5 * POS_LIM, 0.5 * POS_LIM, 2).astype(np.float32)
        vel = np.zeros(2, np.float32)
        a = initial_action(rng)
        k = int(rng.integers(len(P)))
        st, lb = [], []
        for t in range(seq_len):
            if switch and t > 0 and len(P) > 1 and rng.random() > stay:
                nxt = int(rng.integers(len(P) - 1))
                k = nxt + (nxt >= k)
            if t > 0:
                a = next_action(a, rng, rho=rho)
            A, B = mats[k]
            pos_new, vel_new, _F = env_step_ab(pos, vel, a, A, B, rng)
            st.append([pos[0], pos[1], vel[0], vel[1], a[0], a[1],
                       pos_new[0], pos_new[1], vel_new[0], vel_new[1]])
            lb.append(P[k])
            pos, vel = pos_new, vel_new
        T_.append(st); L_.append(lb)
    return (torch.tensor(np.array(T_), dtype=torch.float32),
            np.array(L_, dtype=float))


def _tag(anchors):
    """Anchors go in the filename. this is what are the anchors on which you wanna train."""
    return "-".join(f"{a:g}" for a in anchors).replace(".", "p")


class CodeCB(torch.nn.Module):
    """model.CB with a configurable code width.

    The real CB takes a 64-d z.  The one-hot baseline hands it a 3-d categorical
    code instead, so the input layer has to change.  Everything else -- widths,
    activations, absolute next-state output -- matches model.CB exactly.
    """

    def __init__(self, code_dim):
        super().__init__()
        from task import CB_HIDDEN
        self.net = torch.nn.Sequential(
            torch.nn.Linear(M.SDIM + 2 + code_dim, CB_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(CB_HIDDEN, CB_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(CB_HIDDEN, M.ODIM))

    def forward(self, s, a, code):
        return self.net(torch.cat([s, a, code], -1))


class OracleMLP(torch.nn.Module):
    """Told the parameter, emits a 64-d code for the SAME CB.

    NOT A CEILING -- A MATCHED COMPARISON.  It is trained on the SAME anchors
    for the SAME number of epochs as the arms, so the only difference is TOLD
    the parameter vs INFERRING it.  That matters for how the result reads: the
    arms beating it between anchors is a finding, not a broken baseline.

    Why it loses there: z is computed from the observed DYNAMICS, which vary
    continuously with the parameter, so at 22.5 deg the inferred code naturally
    lands between the 0 and 45 codes.  This oracle's code comes from an MLP
    whose input is the parameter itself, fitted at three values -- nothing makes
    it interpolate between them.  Structure comes from the dynamics being
    continuous, not from knowing the parameter.

    (Training it on a DENSE parameter range would make it a genuine ceiling, but
    it would then see values the arms never did and stop being a comparison.
    The earlier version of this experiment trained it on the anchors for 40
    epochs against the model's 50; at that point it had not yet fitted the three
    values sharply, so it LOOKED smooth -- a ceiling by undertraining.)

    INPUTS MIRROR THE ARM IT IS COMPARED TO.  A single oracle cannot be a fair
    ceiling for both arms: `main` is denied s_t while `control` receives it, so
    an oracle given s_t has TWO advantages over main -- told the parameter AND
    shown the current state -- and only one over control.  So there are two:

        for_arm="main"     [s_{t-1}, a_{t-1},      theta]   ->  7-d in
        for_arm="control"  [s_{t-1}, a_{t-1}, s_t, theta]   -> 11-d in

    s is [pos, vel] throughout, exactly as the arms receive it, so SDIM=4 and
    the widths are 4+2+1 and 4+2+4+1.

    CAPACITY IS MATCHED ON PURPOSE.  A scalar-theta oracle would hand CB a 1-d
    context where our circuit hands it 64-d, so any gap could be capacity rather
    than knowledge.  The output is 64-d with a tanh, matching EC's readout.

    KNOWN REMAINING ASYMMETRY: this is feedforward while EC is recurrent, so z
    can carry accumulated state on top of the parameter.  In the earlier version
    of this experiment that made the inferred code BEAT the told one -- not a
    finding, an unmatched baseline.
    """

    def __init__(self, for_arm="control"):
        super().__init__()
        from task import EC_HIDDEN, LATENT_DIM
        self.for_arm = for_arm
        n_state = 1 if M.ARMS[for_arm] else 2       # main denied s_t, control not
        self.net = torch.nn.Sequential(
            torch.nn.Linear(M.SDIM * n_state + 2 + 1, EC_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(EC_HIDDEN, EC_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(EC_HIDDEN, LATENT_DIM), torch.nn.Tanh())

    def forward(self, s_tm1, a_tm1, s_t, p_norm):
        parts = ([s_tm1, a_tm1, p_norm] if M.ARMS[self.for_arm]
                 else [s_tm1, a_tm1, s_t, p_norm])
        return self.net(torch.cat(parts, -1))


def _code_inputs(trans, params, family, anchors):
    """Observation stream plus the two context codes the baselines use."""
    obs, act, tgt = M.build(trans)
    B, T, _ = obs.shape
    s_tm1 = torch.cat([torch.zeros(B, 1, M.SDIM), obs[:, :-1]], 1)
    a_tm1 = torch.cat([torch.zeros(B, 1, 2), act[:, :-1]], 1)
    P = torch.tensor(params, dtype=torch.float32)
    scale = 90.0 if family in ("rotation", "reflection") else 1.0
    p_norm = (P / scale).unsqueeze(-1)
    A = np.asarray(anchors, dtype=float)
    # one-hot of the NEAREST anchor: the baseline can only ever express a
    # trained value, which is exactly the "no structure" hypothesis
    idx = np.argmin(np.abs(params[..., None] - A[None, None, :]), axis=-1)
    oh = torch.tensor(np.eye(len(A), dtype=np.float32)[idx])
    return obs, act, tgt, s_tm1, a_tm1, p_norm, oh


def baseline_ckpt(family, kind, seed, anchors):
    return os.path.join(M.CKPT_DIR,
                        f"structure_{family}_{_tag(anchors)}_{kind}_seed{seed}.pt")


def train_baseline(family, kind, seed, anchors=None, epochs=EPOCHS):
    """Train CB with an EXPLICIT context code instead of MTL's inferred z.

    kind = "onehot"          nearest-anchor categorical -- cannot express an
                             unseen value, so it is the no-structure floor
    kind = "oracle_main"     TOLD the parameter, inputs matched to `main`
    kind = "oracle_control"  TOLD the parameter, inputs matched to `control`
                             -- same anchors and epochs as the arms, so this is
                             a matched comparison, not a ceiling

    Two oracles because a single one cannot be a fair ceiling for both arms --
    see OracleMLP.
    """
    anchors = list(anchors if anchors is not None else FAMILIES[family]["train"])
    p = baseline_ckpt(family, kind, seed, anchors)
    if os.path.exists(p):
        print(f"== {family}/{kind} {anchors} seed {seed}: exists, skipping")
        return
    torch.manual_seed(seed)
    trans, params = gen(family, anchors, N_TRAIN, seed=seed)
    obs, act, tgt, s_tm1, a_tm1, p_norm, oh = _code_inputs(
        trans, params, family, anchors)
    is_oh = kind == "onehot"
    cb = CodeCB(len(anchors)) if is_oh else M.CB()
    oracle = None if is_oh else OracleMLP(kind.split("_", 1)[1])
    ps = list(cb.parameters()) + (list(oracle.parameters()) if oracle else [])
    opt = torch.optim.Adam(ps, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-5)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_TRAIN); tot = 0.
        for i in range(0, N_TRAIN, BATCH):
            j = perm[i:i + BATCH]
            code = (oh[j] if is_oh
                    else oracle(s_tm1[j], a_tm1[j], obs[j], p_norm[j]))
            pred = cb(obs[j], act[j], code)
            loss = M.weighted_loss(pred, tgt[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        sch.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  [{family}/{kind} s{seed}] ep {ep:3d} | "
                  f"train {tot/N_TRAIN:.5f}", flush=True)
    torch.save({"cb": cb.state_dict(),
                "oracle": oracle.state_dict() if oracle else None,
                "anchors": anchors}, p)
    print(f"  saved -> {p}", flush=True)


def sweep_baseline(family, kind, seed, anchors=None, n_per=200):
    anchors = list(anchors if anchors is not None else FAMILIES[family]["train"])
    p = baseline_ckpt(family, kind, seed, anchors)
    if not os.path.exists(p):
        print(f"  no checkpoint: {p}"); return None
    sd = torch.load(p, map_location="cpu")
    is_oh = kind == "onehot"
    cb = CodeCB(len(anchors)) if is_oh else M.CB()
    cb.load_state_dict(sd["cb"]); cb.eval()
    oracle = None
    if not is_oh:
        oracle = OracleMLP(kind.split("_", 1)[1])
        oracle.load_state_dict(sd["oracle"]); oracle.eval()
    mse = []
    for v in FAMILIES[family]["test"]:
        trans, params = gen(family, [v], n_per, seed=_vseed(v),
                            switch=False)
        obs, act, tgt, s_tm1, a_tm1, p_norm, oh = _code_inputs(
            trans, params, family, anchors)
        with torch.no_grad():
            code = oh if is_oh else oracle(s_tm1, a_tm1, obs, p_norm)
            pred = cb(obs, act, code)
        mse.append(float(((pred[:, :-1, :2] - tgt[:, :-1, :2]) ** 2).mean()))
    return {"vals": np.asarray(FAMILIES[family]["test"], float),
            "mse": np.array(mse), "anchors": np.array(anchors)}

#fit decoder and test on held out 
def param_decode(res, anchors):
    """Fit a LINEAR z -> parameter map on the ANCHORS ONLY, predict everywhere.

    """
    from sklearn.linear_model import LinearRegression
    A = set(np.round(anchors, 3))
    m = np.array([round(float(v), 3) in A for v in res["vals"]])
    if not m.any():
        return None
    reg = LinearRegression().fit(res["centroids"][m], res["vals"][m])
    return reg.predict(res["centroids"])


def _vseed(v):
    """A per-value RNG seed that is always a NON-NEGATIVE int.

    The obvious `1000 + int(v * 100)` goes negative for rotation's -30 deg and
    numpy rejects it, which killed the sweep after all four models had trained.
    Offsetting before scaling keeps every value in range and still gives each
    one its own stream.
    """
    return int(round((v + 1000.0) * 100))


def ckpt(family, arm, seed, anchors):
    return os.path.join(
        M.CKPT_DIR, f"structure_{family}_{_tag(anchors)}_{arm}_seed{seed}.pt")


def train(family, arm, seed, anchors=None, epochs=EPOCHS):
    """Same optimiser, schedule, loss and architecture as model.train -- only
    the context set differs."""
    anchors = list(anchors if anchors is not None else FAMILIES[family]["train"])
    p = ckpt(family, arm, seed, anchors)
    if os.path.exists(p):
        print(f"== {family}/{arm} anchors {anchors} seed {seed}: exists, skipping")
        return
    torch.manual_seed(seed)
    trans, _ = gen(family, anchors, N_TRAIN, seed=seed)
    va, _ = gen(family, anchors, 300, seed=seed + 500)
    ec, cb = M.EC(M.ARMS[arm]), M.CB()
    opt = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-5)
    hist = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_TRAIN); tot = 0.
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            _, pred, tgt = M.rollout(ec, cb, trans[idx], arm)
            loss = M.weighted_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        ec.eval()
        with torch.no_grad():
            _, pr, tg = M.rollout(ec, cb, va, arm)
        hist.append(float(((pr[:, :-1, :2] - tg[:, :-1, :2]) ** 2).mean()))
        ec.train()
        if ep % 25 == 0 or ep == 1:
            print(f"  [{family}/{arm} {_tag(anchors)} s{seed}] ep {ep:3d} | train {tot/N_TRAIN:.5f}"
                  f" | held-out pos MSE {hist[-1]:.5f}", flush=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": hist,
                "anchors": anchors}, p)
    print(f"  saved -> {p}", flush=True)


def sweep(family, arm, seed, anchors=None, n_per=200):
    """Prediction error and z at every test value, each in constant-context
    sequences."""
    anchors = list(anchors if anchors is not None else FAMILIES[family]["train"])
    p = ckpt(family, arm, seed, anchors)
    if not os.path.exists(p):
        print(f"  no checkpoint: {p}"); return None
    sd = torch.load(p, map_location="cpu")
    ec, cb = M.EC(M.ARMS[arm]), M.CB()
    ec.load_state_dict(sd["ec"]); cb.load_state_dict(sd["cb"])
    ec.eval(); cb.eval()
    vals = FAMILIES[family]["test"]
    mse, cen = [], []
    for v in vals:
        trans, _ = gen(family, [v], n_per, seed=_vseed(v),
                       switch=False)
        with torch.no_grad():
            z, pred, tgt = M.rollout(ec, cb, trans, arm)
        mse.append(float(((pred[:, :-1, :2] - tgt[:, :-1, :2]) ** 2).mean()))
        cen.append(z[:, TSS_MIN:].reshape(-1, z.shape[-1]).numpy().mean(0))
    return {"vals": np.asarray(vals, float), "mse": np.array(mse),
            "centroids": np.stack(cen), "anchors": np.array(sd["anchors"])}


def fig_sweep(family, res, base, path, seed=0):
    """(a) prediction error across the sweep, against a floor and a ceiling.
        (b) the parameter decoded from z by a map fitted on the ANCHORS ONLY.

    Panel (a) needs both baselines to mean anything.

    ONE-HOT is the no-structure floor: CB is handed a categorical code for the
    nearest trained value, so it cannot express anything in between.

    TOLD-THETA is a MATCHED COMPARISON, not a ceiling.  Same anchors, same
    epochs, same 64-d code, same CB -- the only difference is told vs inferred.
    Read the SHAPE, not the height: at the anchors the explicit codes win easily
    (no inference cost), and the question is how far each degrades away from
    them.  The arms rise ~1.2x off their anchor value where one-hot rises ~8x
    and told-theta ~13x.  Told-theta losing is a result: nothing makes an MLP
    fitted at three parameter values interpolate between them, whereas z
    inherits its smoothness from the dynamics.

    Panel (b) is the sharper test, because low error at a held-out value could
    just mean nearby values are easy.  A LINEAR map fitted on the anchors alone,
    predicting held-out values correctly, says z places them on a metric axis.
    """
    L = FAMILIES[family]["label"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
    a0 = next(r for r in res.values() if r is not None)
    lo, hi = a0["anchors"].min(), a0["anchors"].max()
    ax[0].axvspan(lo, hi, color="#eeeeee", zorder=0, label="trained range")
    for v in a0["anchors"]:
        ax[0].axvline(v, ls="--", color="#999", lw=1, zorder=1)
    if base.get("onehot") is not None:
        ax[0].plot(base["onehot"]["vals"], base["onehot"]["mse"], "-o", ms=3,
                   color="tomato", lw=1.5, label="one-hot (no structure)")
    ACOL = {"main": "#2c7fb8", "control": "#1a1a1a"}
    for arm, r in res.items():
        if r is None:
            continue
        c = ACOL[arm]
        ax[0].plot(r["vals"], r["mse"], "-", lw=2, color=c,
                   label=f"{arm} (inferred)")
        # each arm against ITS OWN ceiling -- inputs matched to that arm
        b = base.get(f"oracle_{arm}")
        if b is not None:
            ax[0].plot(b["vals"], b["mse"], "--", lw=1.6, color=c, alpha=.75,
                       label=f"{arm}: told $\\theta$ (same anchors)")
    ax[0].set_xlabel(L); ax[0].set_ylabel("CB prediction MSE")
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=8.5)
    ax[0].set_title("does structure enable generalisation?\n"
                    "dashed = told the parameter instead of inferring it, "
                    "same anchors and epochs", fontsize=10)
    for arm, r in res.items():
        if r is None:
            continue
        pred = param_decode(r, r["anchors"])
        if pred is None:
            continue
        c = "#2c7fb8" if arm == "main" else "#1a1a1a"
        A = set(np.round(r["anchors"], 3))
        m = np.array([round(float(v), 3) in A for v in r["vals"]])
        ax[1].scatter(r["vals"][~m], pred[~m], s=26, color=c, alpha=.8,
                      label=f"{arm}: held-out")
        ax[1].scatter(r["vals"][m], pred[m], s=90, marker="D", color="tomato",
                      edgecolor="k", lw=.6, zorder=3,
                      label="trained anchors" if arm == "main" else None)
    lim = [a0["vals"].min(), a0["vals"].max()]
    ax[1].plot(lim, lim, ":", color="#666", lw=1, label="identity")
    ax[1].set_xlabel(f"true {L}"); ax[1].set_ylabel(f"decoded from $z$")
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title(f"does $z$ encode it as a continuous axis?\n"
                    f"(decoder fitted on the trained anchors only)",
                    fontsize=11)
    for a_ in ax:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle(f"structure learning, {family} -- trained on "
                 f"{list(a0['anchors'])}, seed {seed}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, .92))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")


def fig_overlay(family, seed, sets, path):
    """The leave-one-out comparison: the same sweep, trained with and without an
    interior anchor, on one axis.

    THIS IS THE TEST THAT THE SINGLE SWEEP CANNOT DO.  With anchors {0, 45, 90}
    a smooth curve through 45 proves nothing -- 45 was trained.  Removing it and
    retraining asks whether the circuit can still handle 45 having never seen
    it.  A BUMP at the omitted value means the anchors were memorised; NO bump
    means the family was learned as an axis.

    The bump has to be read against the two neighbours, not against zero: error
    rises away from any anchor simply because the perturbation is larger, so
    what matters is a local peak AT the omitted value.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
    L = FAMILIES[family]["label"]
    styles = {0: ("-", 2.0, "#2c7fb8"), 1: ("--", 1.8, "#c0392b")}
    omitted = None
    for si, (tag, anch, res) in enumerate(sets):
        ls, lw, col = styles[si % 2]
        for arm, r in res.items():
            if r is None or arm != "main":
                continue
            ax[0].plot(r["vals"], r["mse"], ls, lw=lw, color=col,
                       label=f"trained on {list(anch)}")
            pred = param_decode(r, anch)
            if pred is not None:
                ax[1].plot(r["vals"], pred, ls, lw=lw, color=col,
                           label=f"trained on {list(anch)}")
        for v in anch:
            ax[0].axvline(v, ls=":", color=col, lw=1, alpha=.5)
    # the value present in one anchor set and not the other
    a0, a1 = set(np.round(sets[0][1], 3)), set(np.round(sets[1][1], 3))
    for v in sorted(a0 ^ a1):
        omitted = v
        for a_ in ax:
            a_.axvline(v, color="#e67e22", lw=2, alpha=.6, zorder=0)
            a_.annotate("omitted", (v, a_.get_ylim()[1]), fontsize=9,
                        color="#e67e22", ha="center", va="top")
    ax[0].set_xlabel(L); ax[0].set_ylabel("CB prediction MSE")
    ax[0].set_yscale("log"); ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("prediction error: does a bump appear where the\n"
                    "anchor was removed?", fontsize=11)
    lim = [sets[0][2]["main"]["vals"].min(), sets[0][2]["main"]["vals"].max()]
    ax[1].plot(lim, lim, ":", color="#666", lw=1, label="identity")
    ax[1].set_xlabel(f"true {L}"); ax[1].set_ylabel("decoded from $z$")
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title("and does the decoded axis stay straight\n"
                    "through the omitted value?", fontsize=11)
    for a_ in ax:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle(f"structure learning, {family}: leave-one-out at "
                 f"{omitted:g} (main arm, seed {seed})", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  saved -> {path}")
    # Quantify by comparing the TWO CURVES TO EACH OTHER, not each curve to its
    # own neighbours.  The neighbour ratio was the first thing tried and it is
    # structurally blind to the failure that actually occurs: removing an
    # interior anchor lifts the WHOLE span between the outer anchors, so the
    # neighbours rise with the omitted point and the ratio reads ~1.0 while the
    # curve is 15x worse.  Anchors and extrapolation are reported too, as the
    # control -- they should be unchanged.
    r_full = sets[0][2].get("main")
    r_loo = sets[1][2].get("main")
    if r_full is not None and r_loo is not None and omitted is not None:
        v = r_full["vals"]
        lo, hi = min(sets[1][1]), max(sets[1][1])
        inside = (v > lo) & (v < hi)
        i = int(np.argmin(np.abs(v - omitted)))
        print(f"    at {omitted:g}          : "
              f"{r_loo['mse'][i] / r_full['mse'][i]:5.2f}x worse without the anchor")
        print(f"    interior mean   : "
              f"{r_loo['mse'][inside].mean() / r_full['mse'][inside].mean():5.2f}x")
        at_anch = np.isin(np.round(v, 3), np.round(sets[1][1], 3))
        if at_anch.any():
            print(f"    at the anchors  : "
                  f"{r_loo['mse'][at_anch].mean() / r_full['mse'][at_anch].mean():5.2f}x"
                  f"   (should be ~1)")
        out = (v < lo) | (v > hi)
        if out.any():
            print(f"    extrapolation   : "
                  f"{r_loo['mse'][out].mean() / r_full['mse'][out].mean():5.2f}x"
                  f"   (should be ~1)")
        for tag, anch, res in sets:
            r = res.get("main")
            pred = param_decode(r, anch)
            if pred is not None:
                print(f"    theta decoded from z, anchors {list(anch)}: "
                      f"|err| at {omitted:g} = {abs(pred[i] - v[i]):.1f}, "
                      f"interior mean {np.abs(pred - v)[inside].mean():.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--family", default="rotation", choices=list(FAMILIES))
    ap.add_argument("--arms", nargs="+", default=list(M.ARMS),
                    choices=list(M.ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--anchors", type=float, nargs="+", default=None,
                    help="override the training anchors, e.g. --anchors 0 90 "
                         "for the leave-one-out control (train 0 and 90, test "
                         "the omitted 45).  Goes into the checkpoint name, so "
                         "it cannot overwrite the full-anchor run.")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true",
                    dest="skip_baselines",
                    help="train only the arms, not the one-hot and told-theta "
                         "baselines.  Use for extra SEEDS: the baselines "
                         "establish the mechanism at seed 0 and cost 11 min per "
                         "anchor set, so repeating them triples the run for "
                         "variance we are not reporting on them.")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--overlay", nargs="+", type=float, default=None,
                    metavar="ANCHOR",
                    help="overlay the default-anchor sweep with the one trained "
                         "on these anchors, e.g. --overlay 0 90")
    a = ap.parse_args()
    if not (a.train or a.sweep or a.overlay):
        ap.error("nothing to do: pass --train, --sweep and/or --overlay")
    for s in a.seeds:
        if a.overlay:
            sets = []
            for anch in (FAMILIES[a.family]["train"], a.overlay):
                f = os.path.join(out_dir(s), f"structure_{a.family}_{_tag(anch)}.npz")
                if not os.path.exists(f):
                    print(f"  missing {f} -- run --sweep for those anchors first")
                    sets = []; break
                z = np.load(f)
                arms = sorted({k.split("|")[0] for k in z.files})
                sets.append((_tag(anch), np.asarray(anch, float),
                             {arm: {k.split("|")[1]: z[k] for k in z.files
                                    if k.startswith(arm + "|")} for arm in arms}))
            if len(sets) == 2:
                fig_overlay(a.family, s, sets,
                            os.path.join(out_dir(s), f"structure_{a.family}_overlay.png"))
            continue
        if a.train:
            for arm in a.arms:
                train(a.family, arm, s, a.anchors)
            if not a.skip_baselines:
                for kind in ("onehot", "oracle_main", "oracle_control"):
                    train_baseline(a.family, kind, s, a.anchors)
        if a.sweep:
            res = {arm: sweep(a.family, arm, s, a.anchors) for arm in a.arms}
            base = {k: sweep_baseline(a.family, k, s, a.anchors)
                    for k in ("onehot", "oracle_main", "oracle_control")}
            if any(v is not None for v in res.values()):
                tg = _tag(a.anchors if a.anchors is not None else FAMILIES[a.family]['train'])
                np.savez(os.path.join(out_dir(s), f"structure_{a.family}_{tg}.npz"),
                         **{f"{arm}|{k}": v for arm, r in res.items()
                            if r is not None for k, v in r.items()})
                fig_sweep(a.family, res, base,
                          os.path.join(out_dir(s), f"structure_{a.family}_{tg}.png"), s)
