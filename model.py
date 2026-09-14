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

also, S= [Position, Velocity]. for our task. 

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
import math
import os
import pdb
import numpy as np
import torch
import torch.nn as nn

from task import (CONTEXTS, N_CONTEXTS, FAM_NAMES, FAMILY, C_LABEL,
                  DT, POS_LIM, V_MAX, POS_SCALE, VEL_SCALE, GRID_SIZE,
                  EC_HIDDEN, CB_HIDDEN, LATENT_DIM, N_TRAIN, BATCH, EPOCHS, LR, WD,
                  TSS_MIN, RHO, rot_mat, refl_mat)
from data import generate_sequences
from decode import acc

# ── The two arms.
# ALPHA UPWEIGHTS THE VELOCITY HALF OF THE FORWARD-MODEL LOSS -- but MOST OF
# THAT REWEIGHTING NOW HAPPENS IN build(), NOT HERE, and the two must be
# counted together or alpha overshoots by two orders of magnitude.
#
# The argument for upweighting is unchanged and is in weighted_loss: both heads
# need the same quantity (dvel), position only says WHERE the same structure
# sits (multimodality.position_invariance), and in RAW units the shared signal
# is 280x more dilute in the position target.
#
# WHAT CHANGED IS THE UNITS.  build() scales velocity by VEL_SCALE = 1/V_MAX =
# 10 so both channels arrive on [-1, 1].  Squared, that is 100x, so the loss
# never sees the raw 280x -- in the units it actually operates on:
#
#     Var(pos' * POS_SCALE)  0.30455
#     Var(vel' * VEL_SCALE)  0.10882      ratio 2.80x, not 280x
#
# so VARIANCE-MATCHING IS ALPHA = 2.8, and the 100 first put here was a ~100x
# overshoot that would have driven the position term to 0.01% of the loss.
# ALPHA = 3 is variance-matching rounded, which leaves position at 0.33%.
#
# CHECK AFTER THE FIRST RETRAIN.  At the tied optimum a single error dvel shows
# up as dvel*POS_SCALE*DT in position and dvel*VEL_SCALE in velocity, so
#     rms(e_vel) / rms(e_pos)  ->  VEL_SCALE / POS_SCALE  =  10
#     corr(e_pos, e_vel)       ->  1
# (an earlier note here said the ratio goes to 1; that was the raw-unit
# statement and is wrong for the scaled residuals the loss sees.)  If the ratio
# stays near 1, the two heads are erring INDEPENDENTLY -- the position head has
# not converged and alpha is too high.
#
# Recalibrate from the block in weighted_loss if MASS, V_MAX, DT, POS_LIM or
# either of the build() scales change -- this is a measurement, not a setting.
ALPHA = 3.0

# arm name -> IS MTL DENIED s_t, the state its own action produced?
# Read the boolean as "denied", NOT as "enabled" -- control: False means the
# control is not denied s_t, i.e. it DOES receive it.
#   main    True   MTL never sees s_t, so the cerebellar error is its only
#                  account of what just happened            -> 10-d MTL input
#   control False  MTL is handed s_t as well, can compute the transformation
#                  itself, and e becomes redundant          -> 14-d MTL input

ARMS = {"main": True, "control": False}
SDIM = ODIM = 4                              # state is position AND velocity [x, y, vx, vy]
ADIM = 2                                      # action is a 2-d force vector [fx, fy]

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ── Architecture and convention ──────────────────────────────────────────
#
# THE INPUT-VARIANT REGISTRY IS GONE.  It existed because two input conventions
# coexisted and a checkpoint driven under the wrong one fails SILENTLY -- z's
# context decodability fell 1.00 -> 0.28 with nothing raising an error, and
# both frozen arms then looked exactly like the no-circuit baseline.  Its keys
# were pos_divisor, center_position, s_scale and scale_action, every one of
# which was a workaround for the old plant: position ran [0, 60] and the
# velocity channel was 97x smaller, so the input needed conditioning before a
# network could see it.  The factorial plant puts position in [-1, 1] and
# velocity in [-V_MAX, V_MAX] natively, each with its own scale, so there is
# one convention and nothing to choose.
#
# THE GUARD SURVIVES THE REGISTRY.  A stamp is written into every checkpoint
# and checked on load, because the failure mode it caught is real and does not
# depend on there being alternatives.
#
# BUMP THIS WHENEVER THE CONTEXTS OR THE PLANT CHANGE, not only when the
# ARCHITECTURE does -- that distinction is the trap.  Going from the 4x4
# factorial to flat rotation/reflection families left every parameter shape
# identical (EC still takes 10 inputs, CB still 70), so a checkpoint trained on
# the old contexts loads without a murmur and is simply driven on a context set
# it has never seen.  Shapes cannot catch that; only this string can.
#
#   factorial-v1       4x4 aim x field crossing, drag/curl as contexts
#   flat-families-v1   rotation and reflection sweeps, damping in the plant
CONVENTION = "flat-families-v1"

# CB's hidden nonlinearity.  MEASURED, not chosen: with tanh the circuit tops
# out at 0.51-0.65 rotation decoding against ReLU's 0.91, because tanh bounds
# h1/h2 to [-1, 1] and caps the z-gates-action product that separating rotation
# from reflection requires.
#
# A NOTE ON state_dict SAFETY that outlives the registry: ReLU and Tanh carry
# no parameters and sit at the same Sequential indices, so a checkpoint trained
# under one loads into the other WITHOUT ERROR and predicts nonsense.  Shapes
# cannot catch it; CONVENTION can.
CB_ACTIVATION = "relu"

# EC's recurrent cell.  GRU for now.
#
# THE EVIDENCE IS SPLIT AND ALL OF IT PREDATES THIS TASK.  On the old task the
# RNN matched the GRU on every supervised measure -- decoding 0.91/0.99/0.78/
# 0.86 against 0.95/0.99/0.65/0.83, best drift and curl of anything tested --
# with a third of the parameters and half the training time.  But downstream
# its z protected RL learning about a QUARTER as well (+0.041 against +0.163,
# non-overlapping across seeds), which is the dissociation that made supervised
# decoding a poor predictor of what MC can use.
#
# None of that transfers: the plant, the contexts, the loss and the noise have
# all changed.  So this is the conservative default rather than a finding --
# the GRU is what the prior downstream result was measured on, which makes it
# the better baseline to re-establish against.  One line to flip, and worth
# re-running the pair once there is a retrained circuit to compare.
CELL_TYPE = "gru" # "rnn"


def ckpt_path(arm, seed):
    """Checkpoint path.  `seed` is REQUIRED on purpose: a default would let a
    caller that forgot it silently read seed 0's weights while writing into
    another seed's folder.  That exact bug cost three figure scripts in the
    exploratory version, and it is invisible -- the figures render fine, they
    are just of the wrong network.

    Flat layout again, now that there is one convention.  Checkpoints from
    before the factorial rewrite live at these same paths and are INVALID --
    load() rejects them on CONVENTION rather than on shape, because their
    shapes still match.
    """
    return os.path.join(CKPT_DIR, f"{arm}_seed{seed}.pt")


def fig_dir(arm, seed):                      # seed required -- see ckpt_path
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
    """MTL: a recurrent cell over the observation stream -> z_t (64-d, tanh).

    cell_type SELECTS WHETHER GATING EXISTS AT ALL, which is the control for
    the claim that rotation and reflection need a MULTIPLICATIVE interaction
    between the action and the error while drift does not.

      gru   h' = (1-u)*n + u*h, with u = sigmoid(p) ~ 0.5 + p/4. The u*h term
            carries a (W_iu x)*h product that is FIRST ORDER in the input.
      rnn   h' = tanh(W_ih x + W_hh h + b). No multiplication anywhere; the
            same product appears only through tanh's cubic term,
            tanh(q) ~ q - q^3/3, so it is THIRD ORDER and suppressed by q^2.

    An RNNCell holds a third of a GRUCell's parameters at the same width. The
    hidden size is held fixed rather than the parameter count, so the
    manipulation is "remove the gates" rather than "remove the gates and
    change the width"; report the parameter difference alongside any result.
    """

    CELL_TYPES = {"gru": nn.GRUCell, "rnn": nn.RNNCell}

    def __init__(self, no_current, cell_type="gru"):
        super().__init__()
        if cell_type not in self.CELL_TYPES:
            raise KeyError(f"unknown cell_type {cell_type!r}; "
                           f"expected one of {list(self.CELL_TYPES)}")
        self.no_current = no_current
        self.cell_type = cell_type
        n_state = 1 if no_current else 2          # s_{t-1} alone, or with s_t
        self.cell = self.CELL_TYPES[cell_type](
            n_state * SDIM + ADIM + SDIM, EC_HIDDEN)
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

    ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(self, activation="relu"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise KeyError(f"unknown cb activation {activation!r}; "
                           f"expected one of {list(self.ACTIVATIONS)}")
        self.activation = activation
        act = self.ACTIVATIONS[activation]
        # the nonlinearity carries no parameters and keeps the same Sequential
        # indices either way, so a checkpoint trained under one activation
        # LOADS CLEANLY into the other -- see INPUT_VARIANTS on why load()
        # checks the recorded CONVENTION instead of trusting shapes
        self.net = nn.Sequential(
            nn.Linear(SDIM + ADIM + LATENT_DIM, CB_HIDDEN), act(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), act(),
            nn.Linear(CB_HIDDEN, ODIM))

    def forward(self, s, a, z, return_hidden=False):
        """return_hidden also returns the two post-nonlinearity activations.

        Used by representation_analysis to ask what CB does with the context it
        is handed.
        """
        x = torch.cat([s, a, z], -1)

        if not return_hidden:
            return self.net(x)
        
        h1 = self.net[1](self.net[0](x))      # Linear -> activation
        h2 = self.net[3](self.net[2](h1))     # Linear -> activation
        return self.net[4](h2), h1, h2


@torch.no_grad()
def circuit_operating_point(ec, cb, trans, arm):
    """Where EC's nonlinearities are running, over a real rollout.

    Returns (gate_pre_std, readout_pre_std, S, A, Z). A gate pre-activation
    std near 0 means both sigmoids sit at 0.5 and cannot gate, leaving the GRU
    a leaky linear integrator; a readout std near 0 confines z to a sliver of
    its tanh range. Use it to check a change BEFORE spending a retrain --
    a trained v1 circuit sits at gate std ~2.5, readout ~0.88.
    """
    no_current = _check(arm)
    obs, act, _ = build(trans)
    B, L, _ = obs.shape
    h = ec.init_hidden(B)
    s_pred = torch.zeros(B, ODIM)
    gate_pre, read_pre, S, A, Z = [], [], [], [], []
    W_ih, W_hh = ec.cell.weight_ih, ec.cell.weight_hh
    b_ih, b_hh = ec.cell.bias_ih, ec.cell.bias_hh
    for t in range(L):
        s_t, a_t = obs[:, t], act[:, t]
        if t == 0:
            s_tm1, a_tm1 = torch.zeros(B, SDIM), torch.zeros(B, ADIM)
            e_t = torch.zeros(B, ODIM)
        else:
            s_tm1, a_tm1, e_t = obs[:, t - 1], act[:, t - 1], s_t - s_pred
        parts = [s_tm1, a_tm1, e_t] if no_current else [s_tm1, a_tm1, s_t, e_t]
        x = torch.cat(parts, -1)
        gi, gh = x @ W_ih.T + b_ih, h @ W_hh.T + b_hh
        if ec.cell_type == "gru":
            i_r, i_u, i_n = gi.chunk(3, 1)
            h_r, h_u, h_n = gh.chunk(3, 1)
            r = torch.sigmoid(i_r + h_r)
            gate_pre.append(torch.cat([i_r + h_r, i_u + h_u, i_n + r * h_n], -1))
        else:
            # an RNNCell has ONE pre-activation and no gates, so there is
            # nothing to chunk -- the tanh argument is the whole story
            gate_pre.append(gi + gh)
        z, h = ec.step(x, h)
        read_pre.append(ec.readout[0](h))
        if t > 0:
            S.append(s_t); A.append(a_t); Z.append(z)
        s_pred = cb(s_t, a_t, z)
    return (float(torch.cat(gate_pre).std()), float(torch.cat(read_pre).std()),
            torch.cat(S), torch.cat(A), torch.cat(Z))


@torch.no_grad()
def ec_source_slices(arm):
    """EC's input columns grouped by SOURCE rather than by concatenation block.

    rollout() builds the input as [s_tm1, a_tm1, (s_t,) e_t], but the block
    seams are the wrong place to normalize. Position and velocity sit inside
    the SAME state block at different scales, and the action -- its own block
    -- is the one that is actually starved. Measured std as a ratio to
    position, main seed 0 over 600 sequences:

        position 1.00   velocity 1.14   action 0.064

    build() rescales the position columns and, through them, the velocity
    derived from them, but never touches the action columns, so any s_scale > 1
    makes that 0.064 progressively worse: at s_scale 15 the commanded force --
    the single most informative input to a forward model -- enters 26x quieter
    than position.
    """
    no_current = _check(arm)
    out = [("pos", 0, 2), ("vel", 2, 4), ("act", 4, 6)]
    i = 6
    if not no_current:                       # control is additionally given s_t
        out += [("cur_pos", i, i + 2), ("cur_vel", i + 2, i + 4)]
        i += 4
    out += [("e_pos", i, i + 2), ("e_vel", i + 2, i + 4)]
    assert out[-1][2] == mtl_input_width(arm), "source slices do not tile"
    return [(n, slice(a, b)) for n, a, b in out]


def cb_source_slices():
    """CB's input columns by source: forward() concatenates [s, a, z]."""
    return [("pos", slice(0, 2)), ("vel", slice(2, 4)),
            ("act", slice(4, 6)), ("z", slice(6, 6 + LATENT_DIM))]


# THE INIT EXPERIMENTS ARE GONE.  _whiten, apply_source_whitened_init,
# apply_bias_spread and apply_per_source_init all lived here and all three
# manipulations were MEASURED AND REJECTED on the old task:
#
#   per_source   harmful -- it scales ALL of weight_ih, action columns
#                included, so CB starts predicting at 1/15 of target scale
#   whitened     harmful -- 2x2 matched at init, the scaled arms learn
#                (1.043 -> 2.30) and the whitened ones do not (1.030 -> 1.33).
#                Drive has to live in the INPUT, not the weights; it is Adam,
#                and it is cell-independent
#   bias_spread  null at convergence (transient at epoch 50, gone by 100)
#
# They were kept only so the tables above stayed reproducible, and every one of
# them was a response to the old plant's input conditioning, which no longer
# exists.  ec_source_slices and cb_source_slices survive because they describe
# the semantic input layout, which analysis code reads on its own account.


def build(trans):
    """Observation stream, actions, and prediction target.

    trans = [B, T, 10] = [x, y, vx, vy, ax, ay, x', y', vx', vy'], straight
    from data.generate_sequences.  VELOCITY IS READ, NOT RECONSTRUCTED: the
    old build derived V = (s_t - s_{t-1})/DT from positions, which is exact in
    the plant's interior but wrong at a wall, where the position is clipped and
    the velocity zeroed.  Rare (0.7% of steps) and severe (3x) -- and since all
    context information lives in the velocity, those steps corrupted precisely
    the signal the circuit exists to extract.  See data.py.

    PER-CHANNEL SCALING, replacing the old uniform s_scale.  Position and
    velocity have different natural bounds (POS_LIM and V_MAX), and a single
    scalar can only lift both, never correct the ratio between them -- which
    was the whole pathology of the old plant, where position ran 97x larger
    than a step's displacement.  Each channel is divided by its own bound, so
    both arrive on [-1, 1].

    ACTIONS ARE NOT RESCALED and no longer need to be.  After scaling, the
    three sources measure pos 0.379, vel 0.224, act 0.577 -- within 2.6x of
    each other, against the 16x imbalance the old uniform scale created.  That
    is what retired the `scale_action` question rather than answering it.
    """
    t = trans.clone()
    pos = t[:, :, 0:2] * POS_SCALE
    vel = t[:, :, 2:4] * VEL_SCALE
    act = t[:, :, 4:6]
    nxt_pos = t[:, :, 6:8] * POS_SCALE
    nxt_vel = t[:, :, 8:10] * VEL_SCALE
    obs = torch.cat([pos, vel], -1)
    tgt = torch.cat([nxt_pos, nxt_vel], -1)
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
        if t == 0: # at t=0 initialize the previous state, error, and action to zeros, since there is no t-1
            s_tm1 = torch.zeros(B, SDIM) # previous state
            a_tm1 = torch.zeros(B, ADIM) # previous action
            e_t = torch.zeros(B, ODIM) # error from the previous time step
        else:
            s_tm1, a_tm1 = obs[:, t - 1], act[:, t - 1]
            e_t = s_t - s_pred            # s_pred is from t-1: read before write

        if e_mode in ("zero", "shuffle"):
            # zero sets e_t to zero. Shuffle permutes the errors across the batch
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


def weighted_loss(pred, tgt, alpha=None):
    """L = mean((Se/dt)^2) + alpha * mean(Ve^2)   [alpha defaults to ALPHA]

    THE 1/dt IS NOW A NO-OP and is kept only so the loss stays correct if DT
    ever moves again.  It existed to put both residuals in the same units: a
    force error dF showed up as dF in velocity but only dt*dF in position, so
    at the old DT = 0.04 the same mistake was 625x smaller once squared.  At
    DT = 1 the two residuals are already commensurate, and ALL of the weighting
    now lives in alpha.

    BOTH HEADS NEED EXACTLY ONE THING, and it is not position.  Writing the
    plant out,

        pos' = pos + vel + dvel        vel' = vel + dvel
        dvel = (A a + B v) / MASS      <- the only context-dependent term

    every other term in both targets is an INPUT to be copied.  So the two
    heads pose the same learning problem, and differ only in how much copying
    is bundled with it:

                            target variance    dvel as a fraction of it
        pos' head                  0.30455                      0.0014
        vel' head                  0.00109                      0.3835
        (raw units; build() then scales velocity by VEL_SCALE)

        echo baselines:   pos' = pos  ->  R2 = 0.9966
                          vel' = vel  ->  R2 = 0.8286

    99.7% of the position target is had by echoing the input, and the shared
    signal is 299x more dilute there.  Under alpha = 1 the position head's
    gradient is therefore almost entirely a demand that h2 REPRESENT POSITION
    -- in a hidden layer the velocity head also reads.

    AND POSITION PROVABLY CARRIES NO CONTEXT.  multimodality.position_invariance
    shows pos' = pos + vel'*DT makes the next-state mixture a rigid translate of
    the velocity mixture: centred on its own centroid the 16-mean configuration
    is bit-identical across start positions (max deviation 5.6e-17) and changes
    only with velocity.  All context information in a transition lives in the
    velocity; position only says where the same structure sits.  So the position
    head is spending a shared trunk on a nuisance coordinate.

    THOSE ARE RAW-UNIT NUMBERS.  build() scales velocity by VEL_SCALE = 10
    before the loss sees it, which is 100x squared, so the loss operates on a
    2.80x ratio rather than 280x and the variance-matched weight is ALPHA =
    2.8, not 280.  Counting the two reweightings separately is how a 100x
    overshoot got into this file once already -- see the note at ALPHA.

    The stronger option, not taken: pos' = pos + vel'*DT is DETERMINISTIC given
    vel', so the position head is exactly redundant and could be dropped. That
    would change CB's output width and hence e_t and EC's input width, so it is
    a structural change rather than a reweighting -- worth doing deliberately,
    not as a side effect of this.
    """
    alpha = ALPHA if alpha is None else alpha
    e = (pred[:, :-1] - tgt[:, :-1]) ** 2
    return (e[..., :2] / DT ** 2).mean() + alpha * e[..., 2:].mean()


def load(arm, seed):
    """Load a trained arm.  Returns (ec, cb) in eval mode, or None if absent.

    REFUSES A CHECKPOINT FROM A DIFFERENT CONVENTION.  Shapes cannot catch
    this: a pre-factorial checkpoint has the same parameter shapes, and ReLU
    and Tanh have no parameters at all, so the wrong network loads cleanly and
    predicts nonsense.  A checkpoint with no stamp predates the rewrite and is
    rejected for that reason, not treated as current.
    """
    p = ckpt_path(arm, seed)
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu")
    stored = sd.get("convention")
    if stored != CONVENTION:
        raise ValueError(
            f"{p} was trained under convention {stored!r} but {CONVENTION!r} "
            f"is current. Driving a checkpoint with the wrong convention is "
            f"silent -- it costs z its context and looks like a result. "
            f"Retrain it.")
    ec, cb = EC(_check(arm), CELL_TYPE), CB(CB_ACTIVATION)
    ec.load_state_dict(sd["ec"]); cb.load_state_dict(sd["cb"])
    ec.eval(); cb.eval()
    return ec, cb


# ONE grouping now, not two: the design is flat, so a "level" is a family.
LEVELS = list(FAM_NAMES)
LEVEL_OF = {n: i for i, n in enumerate(FAM_NAMES)}


def track(ec, cb, arm, va, va_ctx, va_tss, seed=0):
    """Held-out VELOCITY MSE (overall + per factor level) and per-level context
    decoding.

    VELOCITY, NOT POSITION.  The old version reported the position half on the
    grounds that it kept both arms on one scale.  Under this plant that metric
    is close to vacuous -- pos' = pos + vel'*DT, so echoing the input scores
    R2 = 0.9966 and 99.7% of it is free -- and all the context information is
    in the velocity (multimodality.position_invariance).  Position MSE is still
    returned beside it, because the two should sit in the ratio
    (VEL_SCALE/POS_SCALE)^2 once the heads tie; a ratio near 1 means they are
    erring independently and the forward model has not converged.

    PER FAMILY.  Two groups, rotation and reflection.  The context decode
    inside one is over that family's eight members, so chance is 1/8.
    """
    ec.eval(); cb.eval()
    with torch.no_grad():
        z, pred, tgt = rollout(ec, cb, va, arm)
    e = (pred[:, :-1] - tgt[:, :-1]) ** 2
    se_v = e[..., 2:].mean(-1).numpy()
    se_p = e[..., :2].mean(-1).numpy()
    zn = z.numpy()
    ok = va_tss.numpy() >= TSS_MIN
    cid = va_ctx[:, :, 0].numpy().astype(int)
    fam = va_ctx[:, :, 1].numpy().astype(int)
    sid = np.repeat(np.arange(zn.shape[0])[:, None], zn.shape[1], 1)
    out = {"mse": float(se_v.mean()), "mse_pos": float(se_p.mean())}
    for f in LEVELS:
        out[f"mse_{f}"] = float(se_v[fam[:, :-1] == LEVEL_OF[f]].mean())
    dec = {}
    for f in LEVELS:
        m = ok & (fam == LEVEL_OF[f])
        dec[f] = acc(zn[m], cid[m], sid[m], seed)
    ec.train(); cb.train()
    return out, dec


def train(arm, seed, epochs=EPOCHS, decode_every=25,
          save=True):
    """Train one arm.  Writes the ONLY checkpoint, and track.npz beside it.

    `save=False` runs without writing anything.  Use it for a short diagnostic
    run -- a truncated `epochs` otherwise leaves a checkpoint that run.py then
    SKIPS, so the real training silently never happens and the figures come
    from a ten-epoch network.

    track.npz holds the per-epoch history and CANNOT be rebuilt from a
    checkpoint -- recovering one seed's per-family history once cost a full
    retrain -- so do not delete it.
    """
    _check(arm)
    torch.manual_seed(seed)
    trans, _, _ = generate_sequences(N_TRAIN, seed=seed)
    va, va_ctx, va_tss = generate_sequences(300, seed=seed + 500)
    ec, cb = EC(ARMS[arm], CELL_TYPE), CB(CB_ACTIVATION)
    opt = torch.optim.AdamW(list(ec.parameters()) + list(cb.parameters()), lr=LR, weight_decay=WD)
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
            tot += float(loss.detach()) * len(idx)
        sch.step()
        ec.eval()
        with torch.no_grad():
            _, pr, tg = rollout(ec, cb, va, arm)
        # velocity, matching track() -- see there for why position is vacuous
        se = ((pr[:, :-1, 2:] - tg[:, :-1, 2:]) ** 2).mean(-1).numpy()
        ec.train()
        hist.append(float(se.mean()))
        _fam = va_ctx[:, :, 1].numpy().astype(int)
        per_fam.append([float(se[_fam[:, :-1] == LEVEL_OF[f]].mean())
                        for f in LEVELS])
        if ep % decode_every == 0 or ep == 1:
            _, dd = track(ec, cb, arm, va, va_ctx, va_tss, seed)
            dec_ep.append(ep); dec.append(dd)
            print(f"  [{arm}] ep {ep:3d} | train {tot/N_TRAIN:.3e} | "
                  f"held-out vel MSE {hist[-1]:.3e} | "
                  # keyed by the FULL level name, and by the level name ALONE:
                  # track() returned "<level>_linear"/"<level>_mlp" while both
                  # decoders existed, and dropping the MLP dropped the suffix
                  + "  ".join(f"{f} {dd[f]:.2f}" for f in LEVELS),
                  flush=True)
    if not save:
        print(f"  [{arm}] save=False: no checkpoint written", flush=True)
        return hist

    os.makedirs(CKPT_DIR, exist_ok=True)
    # the convention is recorded IN the file: load() refuses a checkpoint driven
    # with a convention it was not fitted under
    os.makedirs(os.path.dirname(ckpt_path(arm, seed)), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": hist,
                "convention": CONVENTION},
               ckpt_path(arm, seed))
    np.savez(os.path.join(fig_dir(arm, seed), "track.npz"),
             levels=np.array(LEVELS),
             hist=np.array(hist), per_fam=np.array(per_fam),
             dec_ep=np.array(dec_ep),
             **{k: np.array([d[k] for d in dec]) for k in dec[0]})
    print(f"  saved -> {ckpt_path(arm, seed)}", flush=True)
