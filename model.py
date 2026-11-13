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
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from task import (CONTEXTS, N_CONTEXTS, FAM_NAMES, FAMILY, C_LABEL,
                  DT, POS_LIM, V_MAX, POS_SCALE, VEL_SCALE, GRID_SIZE,
                  EC_HIDDEN, CB_HIDDEN, LATENT_DIM, N_TRAIN, BATCH, EPOCHS, LR, WD,
                  TSS_MIN, RHO, rot_mat, refl_mat,
                  ACTION_ELL_RANGE, ACTION_GAIN_RANGE, ACTION_SCALE_RANGE)
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
# v2 (2026-09-18): the context table changed from rotation + reflection to
# rotation + curl + drift (task.py's family section). Every v1 checkpoint is
# refused by load(); the old artefacts were moved aside under */flat-families-v1/.
CONVENTION = "flat-families-v2"

# Which action process the sequences were drawn from.  A SEPARATE CHECKPOINT
# PATH AND A SEPARATE CONVENTION STAMP for each, because the two differ in
# nothing a shape check can see: same architecture, same weights layout, same
# everything except the distribution they were fitted on.  Driving a circuit
# with the wrong one is exactly the silent failure load() exists to catch --
# it costs z its context and reads as a result.
#
#   "gp"   a different Matern x tanh-warp action process per sequence (task's
#          GP family).  THE DEFAULT since 2026-09-16: unsuffixed checkpoint
#          (main_seed0.pt), unsuffixed figure dir (figures/main).
#   "ar1"  task.next_action's single AR(1) at RHO.  Was the default, and was
#          called "fitted"; now the suffixed variant (main_ar1_seed0.pt,
#          figures/main_ar1).  Its convention stamp is unchanged, so an AR(1)
#          checkpoint written before the rename still loads -- only its file
#          name moved.
#
# "fitted" is accepted everywhere as a legacy spelling of "ar1", because saved
# agent blobs (sac.save_agent's asdict(config)) carry the old string and must
# keep loading.
#   "iid"  the AR(1) path at rho = 0: independent actions every step. The
#          control for WHY z carries the previous action -- at rho = 0 the
#          previous realised force says nothing about the current one, so any
#          reliance CB shows on z's action content there cannot be the
#          autocorrelation shortcut, and any a_{t-1} still present in z is pure
#          leakage. Checkpoint main_iid_seed*.pt, figures/main_iid.
DEFAULT_ACTION_MODE = "gp"
#   "policy"  windows of a trained oracle agent's recorded actions -- the
#             deployment distribution itself (see data's bank section; the
#             builder was removed in the 2026-09-20 cleanup and is in git
#             history). Tag main_policy; the bank's stamp is in the convention.
ACTION_MODES = ("gp", "ar1", "iid", "policy")
LEGACY_ACTION_MODES = {"fitted": "ar1"}

# ── EC's readout to CB: deterministic, or a fixed-noise channel with a rate cost
#
# WHY. z carries the previous action -- 55-61% of its within-context variance
# is linearly a_{t-1}, for every circuit on every action stream -- because EC
# must process a_{t-1} to interpret the error, the GRU state keeps it, and a
# linear readout has no reason to exclude it. That content is a one-step
# context estimate CB profits from whenever a_t ~ a_{t-1}, so it is never
# removed; and under a closed-loop policy it is what makes z conjunctive and
# stops it from partitioning MC. A DIMENSIONALITY bottleneck cannot fix this:
# the content is four numbers (a_{t-1} and the force it produced), and 32
# dims or 8 pass it trivially.
#
# A RATE cost can. z = tanh(mu + NOISE_SIGMA * eps) in training, tanh(mu) at
# inference, with beta * mean ||mu||^2 added to the loss. With the noise
# FIXED, magnitude IS information: capacity per dimension is
# 0.5 log(1 + mu^2/sigma^2), and CB cannot amplify its way out because
# amplifying mu amplifies the noise with it. Without the noise the same
# penalty is a scale cost the next layer undoes for free. The variance is
# deliberately NOT learned (unlike a VIB): a learned sigma only lets EC
# switch dimensions off, at the cost of a second knob and the chance to shrink
# sigma back toward a noiseless channel.
#
# The belief is four well-separated points sent at low rate with a large
# margin; the action content is continuous and needs high SNR to be worth
# anything. The noise taxes the second disproportionately, which is the point.
#
# A THIRD CIRCUIT AXIS beside arm and action_mode: own checkpoint tag
# (main_iid_noisy0.001_seed0.pt), own stamp (sigma and beta both in it, so a
# mismatch raises in load), own figure dir.
#
# SIGMA IS A PARAMETER, NOT A CONSTANT (2026-09-17). At sigma = 1 no beta gave
# an additive code: beta >= 1e-3 collapsed the channel (mu = 0), beta = 1e-4
# and 1e-5 produced a purely conjunctive z (zero per-context mean), beta = 0
# saturated every dimension and stayed context-blind. Additive noise on an
# input acts as a Tikhonov penalty sigma^2 ||W_z||^2 on the weights reading
# it, so sigma = 1 against a signal that starts at rms ~0.1 per dimension was
# a hundredfold weight decay on CB's z weights -- the collapse was the noise
# scale, not beta. The optimum is invariant to (sigma, beta) -> (1, beta
# sigma^2) in the linear regime, but the DYNAMICS are not, which is what a
# sigma sweep at fixed beta tests. The default keeps every existing tag: the
# tag carries sigma only when it differs from NOISE_SIGMA, the stamp always did.
#
# THE NOISY CHANNEL IS THE DEFAULT since 2026-09-18 (user's decision after the
# sigma sweeps): sigma = 0.3, beta = 1e-3 gave the largest between-context
# fraction of z on GP data (0.45-0.48 against 0.15 deterministic) and the best
# of the noisy circuits under a trained policy. The unsuffixed tag ("main")
# now names that circuit; the deterministic readout is the "_tanh" variant.
# Stamps describe CONTENT, not defaults, so both are explicit.
READOUTS = ("tanh", "noisy")
DEFAULT_READOUT = "noisy"
NOISE_SIGMA = 0.3        # DEFAULT sd of the channel noise on the pre-tanh mean
RATE_BETA = 1e-3         # weight on mean ||mu||^2 per step; sweep it


def normalize_readout(readout):
    if readout not in READOUTS:
        raise KeyError(f"unknown readout {readout!r}; expected one of {READOUTS}")
    return readout


def readout_stamp(readout=DEFAULT_READOUT, beta=RATE_BETA, sigma=NOISE_SIGMA) -> str:
    """The convention suffix a readout adds -- explicit for BOTH readouts,
    since a stamp describes what the checkpoint is, not what the default was
    when it was written."""
    if normalize_readout(readout) == "tanh":
        return "+tanh-readout"
    return f"+noisy-sigma{float(sigma):g}-beta{float(beta):g}"


def readout_tag(readout=DEFAULT_READOUT, beta=RATE_BETA, sigma=NOISE_SIGMA) -> str:
    """The name suffix a readout adds, relative to the DEFAULT: "" for the
    default noisy channel at (RATE_BETA, NOISE_SIGMA), "_tanh" for the
    deterministic readout, "_noisy<beta>" for another beta at the default
    sigma, "_noisy<beta>_sigma<sigma>" otherwise. ONE definition for arm_tag
    (checkpoints, figure dirs) and for sac's run names, so a run directory
    and the circuit it was driven with agree."""
    if normalize_readout(readout) == "tanh":
        return "_tanh"
    if float(beta) == RATE_BETA and float(sigma) == NOISE_SIGMA:
        return ""
    tag = f"_noisy{float(beta):g}"
    if float(sigma) != NOISE_SIGMA:
        tag += f"_sigma{float(sigma):g}"
    return tag


# ── which circuit: one object instead of four loose knobs ─────────────────
#
# Every analysis in this repo (figures, representation_analysis,
# sac's frozen load) selects the circuit through an `action_mode` argument that
# it hands to load()/fig_dir()/generate_sequences(). When the readout became a
# second axis, those call sites kept passing the mode alone -- so `run.py
# --figures --noise-sigma 0.1` would have rendered the DEFAULT circuit's panels
# under the noisy circuit's flags, silently, the same failure the convention
# stamp exists to prevent. Threading three more keyword arguments through
# twenty signatures is the honest fix and the fragile one; instead the
# `action_mode` slot accepts a CircuitSpec. A bare string still means "that
# mode, deterministic readout", so nothing existing changes meaning, and a
# spec carries the whole selection to every function already receiving the
# slot. data.generate_sequences reads .action_mode off it (duck-typed; data
# cannot import model), so the rollout distribution follows the circuit too.
# ── the FOURTH circuit axis: which contexts the circuit was fitted on ──────
#
# The default is the whole table. "rotations" / "reflections" fit on one
# family only (2026-09-18, the user's test of "the contexts we trained on are
# the problem": z's code is a ring per family, additive WITHIN a family and
# not across, so a circuit that never saw reflections should carry a purely
# additive code). The set is in the tag (_trainrot) and the stamp
# (+train-rotations); data.generate_sequences reads the ids off the spec, so
# every analysis of such a circuit rolls it out on its own contexts.
_FAMILY_SET = {f"{name}s": name for name in FAM_NAMES}        # "rotations" -> "rotation"
# "rotations_wide" (2026-09-21): the table with its four rotations REPLACED
# by task.WIDE_ROT_ANGLES (+-30, +-90) -- curl and drift as before, so the
# circuit differs from the default one in the rotation anchors alone. The
# zero-shot experiment's circuit; see task.wide_rotation_ids for why.
WIDE_TRAIN_SET = "rotations_wide"
# "linear_random" (2026-09-21): the structure-learning CONTROL -- the table
# with its rotations replaced by four random linear action transforms at the
# wide rotations' effect sizes (task.random_linear_ids). Same count, same
# effect sizes, no one-parameter axis; see the task.py comment.
LINEAR_TRAIN_SET = "linear_random"
TRAIN_CONTEXT_SETS = ("all",) + tuple(_FAMILY_SET) + (WIDE_TRAIN_SET, LINEAR_TRAIN_SET)
DEFAULT_TRAIN_CONTEXTS = "all"
_SHORT_FAMILY = {"rotation": "rot", "reflection": "refl", "curl": "curl", "drift": "drift"}
_TRAIN_TAG = {"all": "", **{s: f"_train{_SHORT_FAMILY.get(name, name)}"
                            for s, name in _FAMILY_SET.items()},
              WIDE_TRAIN_SET: "_trainrotw", LINEAR_TRAIN_SET: "_trainlin"}


def train_context_ids(train_contexts=DEFAULT_TRAIN_CONTEXTS):
    """Context ids a training-set name selects; None means the whole table."""
    if train_contexts not in TRAIN_CONTEXT_SETS:
        raise KeyError(f"unknown train_contexts {train_contexts!r}; "
                       f"expected one of {TRAIN_CONTEXT_SETS}")
    if train_contexts == "all":
        return None
    if train_contexts == WIDE_TRAIN_SET:
        import task as _T
        return _T.wide_rotation_ids() + [c for c in range(N_CONTEXTS) if FAMILY[c] != 0]
    if train_contexts == LINEAR_TRAIN_SET:
        import task as _T
        return _T.random_linear_ids() + [c for c in range(N_CONTEXTS) if FAMILY[c] != 0]
    fam = FAM_NAMES.index(_FAMILY_SET[train_contexts])
    return [c for c in range(N_CONTEXTS) if FAMILY[c] == fam]


def train_contexts_tag(train_contexts=DEFAULT_TRAIN_CONTEXTS) -> str:
    train_context_ids(train_contexts)                 # validates
    return _TRAIN_TAG[train_contexts]


def train_contexts_stamp(train_contexts=DEFAULT_TRAIN_CONTEXTS) -> str:
    train_context_ids(train_contexts)
    if train_contexts == "all":
        return ""
    if train_contexts == WIDE_TRAIN_SET:
        # THE ANGLES ARE IN THE STAMP: a checkpoint fitted on one wide set
        # must not load as if fitted on another once the constant moves
        import task as _T
        angles = ",".join(f"{a:g}" for a in _T.WIDE_ROT_ANGLES)
        return f"+train-{train_contexts}[{angles}]"
    if train_contexts == LINEAR_TRAIN_SET:
        # the draw is in the stamp: its seed and the effect sizes it matched
        import task as _T
        effs = ",".join(f"{_T.effect_size(_T.LINEAR_MATS[c]):.2f}" for c in _T.random_linear_ids())
        return f"+train-{train_contexts}[seed{_T.RANDOM_LINEAR_SEED};{effs}]"
    return f"+train-{train_contexts}"


@dataclass(frozen=True)
class CircuitSpec:
    action_mode: str = DEFAULT_ACTION_MODE
    readout: str = DEFAULT_READOUT
    beta: float = RATE_BETA
    sigma: float = NOISE_SIGMA
    train_contexts: str = DEFAULT_TRAIN_CONTEXTS

    def __post_init__(self):
        object.__setattr__(self, "action_mode", normalize_action_mode(self.action_mode))
        object.__setattr__(self, "readout", normalize_readout(self.readout))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "sigma", float(self.sigma))
        train_context_ids(self.train_contexts)        # validates

    @property
    def context_ids(self):
        """The contexts this circuit was fitted on (None = all). data reads it."""
        return train_context_ids(self.train_contexts)

    def __str__(self):
        return arm_tag("", self)[1:] or DEFAULT_ACTION_MODE


def add_circuit_args(ap):
    """The four circuit-selecting flags, identically on every CLI that loads a
    supervised circuit (run.py, figures.py, representation_analysis.py).
    Pair with circuit_from_args."""
    ap.add_argument("--action-mode", default=DEFAULT_ACTION_MODE,
                    choices=list(ACTION_MODES) + list(LEGACY_ACTION_MODES),
                    help="action process the circuit was fitted on; selects the "
                         "checkpoint, the rollout distribution and the figure "
                         "directory figures/<arm_tag>/")
    ap.add_argument("--readout", choices=list(READOUTS), default=None,
                    help="EC's readout: 'tanh' (deterministic) or 'noisy' (fixed-"
                         "noise channel with rate cost). --rate-beta or "
                         "--noise-sigma alone imply 'noisy'.")
    ap.add_argument("--rate-beta", type=float, default=None,
                    help="beta of the noisy readout (default model.RATE_BETA)")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="channel-noise sd of the noisy readout (default "
                         "model.NOISE_SIGMA; in the tag only when not default)")
    ap.add_argument("--train-contexts", default=DEFAULT_TRAIN_CONTEXTS,
                    choices=list(TRAIN_CONTEXT_SETS),
                    help="which contexts the circuit was (or is to be) fitted on: "
                         "the whole table, one family, rotations_wide (the table with "
                         "+-30/+-90 as its rotations) or linear_random (random linear "
                         "transforms in their place). Tag _trainrot/_trainrotw/_trainlin.")
    return ap


def circuit_from_args(a) -> "CircuitSpec":
    """The CircuitSpec a parsed add_circuit_args namespace selects. A beta or
    sigma without --readout implies the noisy readout (as sac.py's CLI does):
    a sigma on a tanh circuit would otherwise be accepted and ignored."""
    readout = a.readout
    if readout is None:
        readout = ("noisy" if (a.rate_beta is not None or a.noise_sigma is not None)
                   else DEFAULT_READOUT)
    return CircuitSpec(a.action_mode, readout,
                       RATE_BETA if a.rate_beta is None else a.rate_beta,
                       NOISE_SIGMA if a.noise_sigma is None else a.noise_sigma,
                       getattr(a, "train_contexts", DEFAULT_TRAIN_CONTEXTS))


def circuit(action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
            beta=RATE_BETA, sigma=NOISE_SIGMA,
            train_contexts=DEFAULT_TRAIN_CONTEXTS) -> CircuitSpec:
    """Resolve the (action_mode, readout, beta, sigma, train_contexts) slots
    into a CircuitSpec. `action_mode` may already BE one; the loose knobs must
    then be left at their defaults -- a spec that says noisy beside a
    `readout` that says tanh is a contradiction, not a merge."""
    if isinstance(action_mode, CircuitSpec):
        loose = (normalize_readout(readout), float(beta), float(sigma), train_contexts)
        if loose != (DEFAULT_READOUT, RATE_BETA, NOISE_SIGMA, DEFAULT_TRAIN_CONTEXTS):
            raise ValueError(f"got both a CircuitSpec {action_mode} and loose "
                             f"readout/beta/sigma/train_contexts {loose}; pass one or the other")
        return action_mode
    return CircuitSpec(action_mode, readout, beta, sigma, train_contexts)


def normalize_action_mode(action_mode):
    """Canonical mode name, or KeyError. Maps the legacy "fitted" to "ar1".
    A CircuitSpec in the slot yields its mode."""
    if isinstance(action_mode, CircuitSpec):
        return action_mode.action_mode
    mode = LEGACY_ACTION_MODES.get(action_mode, action_mode)
    if mode not in ACTION_MODES:
        raise KeyError(f"unknown action_mode {action_mode!r}; "
                       f"expected one of {ACTION_MODES} (or legacy "
                       f"{tuple(LEGACY_ACTION_MODES)})")
    return mode


def convention_for(action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
                   beta=RATE_BETA, sigma=NOISE_SIGMA):
    """The stamp a checkpoint carries: the action process it was fitted on,
    plus the readout's suffix. load() refuses anything else."""
    c = circuit(action_mode, readout, beta, sigma)
    return (_action_convention(c.action_mode) + train_contexts_stamp(c.train_contexts)
            + readout_stamp(c.readout, c.beta, c.sigma))


def _action_convention(action_mode=DEFAULT_ACTION_MODE):
    action_mode = normalize_action_mode(action_mode)
    if action_mode == "ar1":
        return CONVENTION
    if action_mode == "iid":
        # its own stamp: an iid circuit is shape-identical to an ar1 one and
        # nothing but the stamp could tell them apart
        return f"{CONVENTION}+iid-actions"
    if action_mode == "policy":
        # THE BANK'S IDENTITY IS IN THE STAMP: a circuit fitted on one recording
        # of the oracle's actions must not load as if fitted on another
        import data
        return f"{CONVENTION}+policy-actions+{data.action_bank_stamp()}"
    # THE RANGES ARE IN THE STAMP, derived rather than hand-maintained.  Two
    # circuits trained on different gp parameter ranges are the same shapes and
    # the same architecture, so nothing else could tell them apart -- and the
    # ranges DO move: ACTION_ELL_RANGE was truncated from (0.3, 30) to (0.3, 3)
    # after the wide version measurably made the circuit worse.  Deriving the
    # stamp means that truncation invalidated the checkpoints trained under it
    # automatically, instead of leaving them loadable and wrong.
    el, eh = ACTION_ELL_RANGE
    gl, gh = ACTION_GAIN_RANGE
    sl, sh = ACTION_SCALE_RANGE
    return (f"{CONVENTION}+gp-actions"
            f"+ell{el:g}-{eh:g}+gain{gl:g}-{gh:g}+scale{sl:g}-{sh:g}")

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


def arm_tag(arm, action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
            beta=RATE_BETA, sigma=NOISE_SIGMA):
    """`arm` with the action-mode infix and the readout suffix: "main" for the
    defaults, "main_ar1" for the AR(1) circuit, "main_iid_noisy0.001" for a
    noisy-readout circuit at beta = 1e-3 on i.i.d. actions at the default
    sigma, "main_iid_noisy0.001_sigma0.1" at sigma = 0.1.

    ONE DEFINITION, used by both ckpt_path and fig_dir.  They have to agree:
    figures written under "main" from a circuit loaded out of "main_ar1" are
    mislabelled results, and nothing downstream could tell.  That is the same
    failure the convention stamp exists to prevent, one directory level up.
    """
    c = circuit(action_mode, readout, beta, sigma)
    tag = arm if c.action_mode == DEFAULT_ACTION_MODE else f"{arm}_{c.action_mode}"
    return tag + train_contexts_tag(c.train_contexts) + readout_tag(c.readout, c.beta, c.sigma)


def ckpt_path(arm, seed, action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
              beta=RATE_BETA, sigma=NOISE_SIGMA):
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
    return os.path.join(CKPT_DIR,
                        f"{arm_tag(arm, action_mode, readout, beta, sigma)}_seed{seed}.pt")


def fig_dir(arm, seed, sub=None,             # seed required -- see ckpt_path
            action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
            beta=RATE_BETA, sigma=NOISE_SIGMA):
    """Per-seed figure directory, optionally one level deeper.

    SPLIT BY ACTION MODE, via the same arm_tag as the checkpoint: the AR(1)
    circuit writes to figures/main_ar1/seed0 rather than overwriting the
    default (gp) circuit's panels at figures/main/seed0.  The checkpoints were
    separated for exactly this reason and the figure tree has to follow, or
    two circuits' results end up in one directory distinguishable only by
    mtime.

    `sub` keeps a family of related panels together without inventing a
    parallel tree somewhere else -- a sub-analysis's figures live at
    fig_dir(arm, seed, "<sub>", mode) rather than under a figures/ root of
    their own, so everything for one (arm, seed, mode) stays in one place.
    """
    d = os.path.join(FIG_DIR, arm_tag(arm, action_mode, readout, beta, sigma), f"seed{seed}")
    if sub:
        d = os.path.join(d, sub)
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

    def __init__(self, no_current, cell_type="gru", readout=DEFAULT_READOUT,
                 beta=RATE_BETA, sigma=NOISE_SIGMA):
        super().__init__()
        if cell_type not in self.CELL_TYPES:
            raise KeyError(f"unknown cell_type {cell_type!r}; "
                           f"expected one of {list(self.CELL_TYPES)}")
        self.no_current = no_current
        self.cell_type = cell_type
        self.readout_kind = normalize_readout(readout)
        self.beta = float(beta)
        self.sigma = float(sigma)                 # fixed, not a Parameter
        n_state = 1 if no_current else 2          # s_{t-1} alone, or with s_t
        self.cell = self.CELL_TYPES[cell_type](
            n_state * SDIM + ADIM + SDIM, EC_HIDDEN)
        if self.readout_kind == "tanh":
            # unchanged, parameter names included: every existing checkpoint
            self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())
        else:
            # the channel MEAN; the noise is self.sigma, fixed -- see the
            # readout section above
            self.mu = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.reset_rate()

    def init_hidden(self, B):
        return torch.zeros(B, EC_HIDDEN)

    # ── rate bookkeeping (noisy readout only) ─────────────────────────────
    def reset_rate(self):
        """Call before each training rollout; rate() then covers that rollout."""
        self._rate_sum = 0.0
        self._rate_n = 0

    def rate(self):
        """Mean ||mu||^2 per step over the steps since reset_rate(), with
        gradient. 0 for the deterministic readout or outside training."""
        if self._rate_n == 0:
            return torch.zeros(())
        return self._rate_sum / self._rate_n

    def rate_penalty(self):
        return self.beta * self.rate() if self.readout_kind == "noisy" else torch.zeros(())

    def read(self, h):
        """z from a hidden state, under whichever readout this circuit has:
        the deterministic tanh, or the noisy channel (sampled and rate-counted
        in training, its mean at inference). ONE place, so a caller that reads
        z from a stored hidden state (sac's end-to-end trainer re-reads the
        previous step's) gets the same readout step() applies."""
        if self.readout_kind == "tanh":
            return self.readout(h)
        mu = self.mu(h)
        if self.training:
            self._rate_sum = self._rate_sum + (mu ** 2).sum(-1).mean()
            self._rate_n += 1
            g = mu + self.sigma * torch.randn_like(mu)
        else:
            g = mu                                # deterministic at inference
        return torch.tanh(g)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.read(h), h


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
        # the readout's pre-activation, whichever readout this circuit has
        read_pre.append((ec.readout[0] if ec.readout_kind == "tanh" else ec.mu)(h))
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


def load(arm, seed, action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
         beta=RATE_BETA, sigma=NOISE_SIGMA):
    """Load a trained arm.  Returns (ec, cb) in eval mode, or None if absent.

    REFUSES A CHECKPOINT FROM A DIFFERENT CONVENTION.  Shapes cannot catch
    this: a pre-factorial checkpoint has the same parameter shapes, and ReLU
    and Tanh have no parameters at all, so the wrong network loads cleanly and
    predicts nonsense.  A checkpoint with no stamp predates the rewrite and is
    rejected for that reason, not treated as current.
    """
    c = circuit(action_mode, readout, beta, sigma)
    p = ckpt_path(arm, seed, c)
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu")
    stored = sd.get("convention")
    expected = convention_for(c)
    if stored != expected:
        raise ValueError(
            f"{p} was trained under convention {stored!r} but {expected!r} "
            f"is current. Driving a checkpoint with the wrong convention is "
            f"silent -- it costs z its context and looks like a result. "
            f"Retrain it.")
    ec = EC(_check(arm), CELL_TYPE, readout=c.readout, beta=c.beta, sigma=c.sigma)
    cb = CB(CB_ACTIVATION)
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
        # a circuit fitted on one family has no validation steps in the other
        # (CircuitSpec.train_contexts): report nan rather than decode nothing
        dec[f] = acc(zn[m], cid[m], sid[m], seed) if m.sum() > 0 else float("nan")
    ec.train(); cb.train()
    return out, dec


# the learning-rate schedule's warm restarts: back to LR after LR_RESTART_T0
# epochs, then after cycles LR_RESTART_TMULT times longer each time. Named so
# figures can mark them (figures.lr_restart_epochs); unchanged values.
LR_RESTART_T0 = 50
LR_RESTART_TMULT = 2


def lr_restart_epochs(n_epochs):
    """The epochs after which the learning rate jumps back to LR."""
    out, t, cyc = [], LR_RESTART_T0, LR_RESTART_T0
    while t < n_epochs:
        out.append(t)
        cyc *= LR_RESTART_TMULT
        t += cyc
    return out


def train(arm, seed, epochs=EPOCHS, decode_every=32,
          save=True, action_mode=DEFAULT_ACTION_MODE, readout=DEFAULT_READOUT,
          beta=RATE_BETA, sigma=NOISE_SIGMA):
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
    c = circuit(action_mode, readout, beta, sigma)
    torch.manual_seed(seed)
    # the SPEC goes to the generator: it carries the action process AND the
    # contexts this circuit is fitted on (CircuitSpec.context_ids)
    trans, _, _ = generate_sequences(N_TRAIN, seed=seed, action_mode=c)
    # VALIDATION TRACKS TRAINING. A gp-trained circuit scored on fitted-mode
    # sequences would report held-out numbers from a distribution it is not
    # being asked to work on, which is the comparison this whole change exists
    # to stop making.
    va, va_ctx, va_tss = generate_sequences(300, seed=seed + 500, action_mode=c)
    ec = EC(ARMS[arm], CELL_TYPE, readout=c.readout, beta=c.beta, sigma=c.sigma)
    cb = CB(CB_ACTIVATION)
    opt = torch.optim.AdamW(list(ec.parameters()) + list(cb.parameters()), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=LR_RESTART_T0, T_mult=LR_RESTART_TMULT, eta_min=1e-5)
    hist, per_fam, dec_ep, dec = [], [], [], []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_TRAIN); tot = 0.
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            ec.reset_rate()                       # rate() covers this rollout
            _, pred, tgt = rollout(ec, cb, trans[idx], arm)
            # + beta * mean ||mu||^2 for the noisy readout, 0 otherwise
            loss = weighted_loss(pred, tgt) + ec.rate_penalty()
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
    path = ckpt_path(arm, seed, c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": hist,
                "convention": convention_for(c),
                "action_mode": c.action_mode, "readout": c.readout,
                "beta": c.beta, "sigma": c.sigma},
               path)
    # THE CIRCUIT'S OWN figure dir. This used to be fig_dir(arm, seed) -- the
    # DEFAULT mode's dir whatever mode was being trained -- so training one
    # circuit overwrote another's track.npz, which the docstring above says
    # cannot be rebuilt.
    np.savez(os.path.join(fig_dir(arm, seed, action_mode=c), "track.npz"),
             levels=np.array(LEVELS),
             hist=np.array(hist), per_fam=np.array(per_fam),
             dec_ep=np.array(dec_ep),
             **{k: np.array([d[k] for d in dec]) for k in dec[0]})
    print(f"  saved -> {path}", flush=True)
