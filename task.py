"""
task.py -- THE TASK: what a context is, how a step evolves, and the constants
every other module reads.  Defines the 16 contexts as a FACTORIAL design, the
force law each one applies to the agent's action, and the plant that turns
that force into the next state.  data.py draws sequences from it; model.py
learns to predict them.

THE FACTORIAL.  Every force law here is linear in (action, velocity):

    F = A @ action + B @ vel

and the old design's four "families" were nothing more than four disjoint
corners of that one space -- rotation and reflection varied A with B = 0, drag
and curl varied B with A = I.  Sixteen contexts, but sixteen arbitrary labels
with no structure between them, so there was nothing for a model to discover
and no way to ask whether it had.

Making (A, B) the design instead gives the same sixteen contexts as a 4 x 4
crossing of two factors:

    AIM, on the action                 FIELD, on the velocity
      direct   I         identity         free      0            identity
      rot45    R(45)     rotation         drag_lo   -b_lo * I    weak drag
      rot90    R(90)     rotation         drag_hi   -b_hi * I    strong drag
      mirror   M(0)      reflection       curl      +k * R(90)   curl field

  A context is then read as "aim | field" -- `rot90 | drag_hi` is a 90-degree
  rotation of your command inside a strongly viscous medium.

The two factors have deliberately PARALLEL structure -- each is {identity, two
graded levels of one type, one level of a contrasting type}.  Within AIM the
contrast is det +1 against det -1 (rotation against reflection, which the
motor-learning literature finds are learned and consolidated by separate
mechanisms).  Within FIELD it is parallel against perpendicular to the velocity
(drag against curl).

WHAT THIS BUYS.  Train on twelve cells, hold out four, and ask whether the
model handles combinations it has never seen.  A factored latent generalizes
to them; a sixteen-way lookup cannot.  That question is unaskable with flat
labels, and it is the one this repo is named for.

BALANCING means matching the two factors' MARGINALS, not equalizing cells --
cells are supposed to differ, that is the design.  AIM's marginals are locked
by geometry (|R(theta)a - a| = 2|sin(theta/2)||a|, and |M(phi)a - a| =
sqrt(2)|a| for EVERY axis), so FIELD's three coefficients are solved onto
them.  Measured on
the one-step transition, rms|dpos_c - dpos_null| over rms|dpos_null| at matched
state and action:

      AIM:   direct 0.000  rot45 0.150  rot90 0.277  mirror 0.276
      FIELD: free   0.000  drag_lo 0.150  drag_hi 0.277  curl 0.276

Note that rot90 and mirror are equal in MAGNITUDE by construction (0.277 vs
0.276).
They differ in kind, not in size, which is the whole point of pairing them --
effect size will not separate them, a decoder will.

EFFECT SIZE DOES NOT ADD ACROSS CELLS and that is expected: rot90|drag_hi is
super-additive (0.839), rot90|curl is sub-additive (0.431) because curl's matrix
is also R(90) and the two partly align.  The FACTORIZATION is still exact --
it lives in (A, B), not in the scalar summary.

THE PLANT is a pure double integrator: force is the independent variable,
acceleration the dependent one.

    F   = A @ action + B @ vel        the context's force law, plant-free
    acc = F / MASS                    Newton, and the only place MASS appears
    vel = clip_speed(vel + acc * DT)  a speed cap, NOT damping
    pos = clip(pos + vel * DT, -1, 1) normalized units

No damping: the null plant is frictionless, and drag is a CONTEXT rather than
a property of the world.  The speed cap does the bounding damping used to do,
which is what keeps curl's effect size stationary -- an undamped velocity is a
random walk, so without the cap a curl field would strengthen as sqrt(t)
through an episode without bound.

WHY DT = 1 AND POSITION IN [-1, 1].  The previous plant used DT = 0.04 to stop
a constant drift force from pinning the agent against a wall, and paid for it:
one step moved the agent 1/97th of its own coordinate, so the velocity channel
was invisible beside position and a uniform x15 input scale was bolted on to
compensate -- which cannot fix a RATIO, and pushed position to [0, 60].  Drift
is gone (drag replaced it, and drag cannot integrate into a wall because it
opposes motion), so the workaround goes with it.  Position is now natively
zero-centered and velocity has its own scale, VEL_SCALE, instead of a shared
one.
"""
import numpy as np

# ── Plant ────────────────────────────────────────────────────────────────
MASS    = 100.0   # force -> acceleration.  Sets the ballistic fraction:
                  # V_MAX/(rms|F|/MASS) steps to saturate against 2/V_MAX
                  # steps to cross, which is 0.35 here -- accelerate, cruise,
                  # brake, with braking that has to be planned.  At ~1 the
                  # plant would be speed-controlled and "acceleration control"
                  # a fiction; at <<1 it never reaches the far wall.
DT      = 1.0
V_MAX   = 0.10    # speed cap.  Travel across the grid is 2/V_MAX = 20 steps.
POS_LIM = 1.0     # position lives in [-POS_LIM, +POS_LIM] on both axes
GRID_SIZE = 2.0 * POS_LIM        # the extent, for anything that needs a width

# Circuit input scales.  PER CHANNEL, replacing the old uniform S_SCALE: a
# single scalar cannot correct a ratio between two channels, only lift both.
# Each is 1/(that channel's natural bound), so both arrive on [-1, 1].
POS_SCALE = 1.0 / POS_LIM
VEL_SCALE = 1.0 / V_MAX

# ── Damping is PLANT, not context ────────────────────────────────────────
#
# Every context runs under the same light damping.  It was briefly a context
# dimension (a "field" factor crossing drag and curl with the action
# transforms) and that did not work: the field acts on VELOCITY, which starts
# at zero and has to build, so an ideal observer was at exactly chance from one
# step and needed ~5 to reach 0.81, against 0.83 from a SINGLE step for the
# action transforms.  The first trained circuit decoded the action factor at
# 0.88 and the velocity factor at 0.40.
#
# So it is a plant property now.  DAMPING keeps the value the drag_lo context
# had -- calibrated, back then, to match rot45's effect size, so as a plant
# constant the number carries no meaning beyond being mild.  MEASURED, it is
# mild indeed: lambda_eff = 1 - DAMPING/MASS = 0.823, and the integrated
# autocorrelation time of an AR(1) is (1+lambda)/(1-lambda) = 10.3, so
# velocity's memory is 12.4 steps against 12.1 UNDAMPED -- barely changed.
# (An earlier version of this comment said 5.6 steps, from 1/(1-lambda), which
# is the wrong formula.)
#
# What it does change is speed and wall contact, and there the effect is large:
# mean |v| falls 0.034 -> 0.0164 and wall time 13% -> 2%.  Those are the
# reasons to keep it; shortening velocity's memory is not one of them, and no
# longer needs to be now that no context depends on velocity.
DAMPING = 17.70

# ── The contexts: action transforms, in two families ─────────────────────
#
# FLAT, not crossed.  Each family is a graded sweep of one geometric
# parameter, which is what the repo had before the factorial and what
# structure.py's continuous-axis question needs.
#
#   rotation    F = R(theta) a    det +1, orientation-preserving
#   reflection  F = M(phi) a      det -1, orientation-REVERSING
#
# THE PAIR IS LOAD-BEARING and the reason both families are angles.  Both are
# norm-preserving and zero-mean over actions, so no statistic of the
# observations alone identifies either -- you have to relate the force to the
# action you chose, which is exactly what a forward-model prediction error
# provides.  And they cannot be told apart from a single (action, force) pair:
# for any one action a reflection looks like the rotation theta = 2(phi -
# alpha).  Separating them needs several action DIRECTIONS, i.e. accumulated
# structured evidence rather than a threshold.
FAM_NAMES = ["rotation", "reflection"]

# FOUR OF EACH, the original set.  Rotations at +-45/+-90 and reflections on
# four axes at 45-degree spacing, which span the family since M(phi) =
# M(phi + pi).
#
# The two sweeps are different IN KIND and that is geometric, not a choice.
# |R(th)a - a| = 2|sin(th/2)||a| so rotations are GRADED -- +-45 and +-90 sit
# at 0.77 and 1.41 in units of |a|, a 1.85x spread.  |M(phi)a - a| = sqrt(2)|a|
# for EVERY axis, so reflections are FLAT and differ only in direction.  The
# families meet exactly at +-90, where 2 sin(45) = sqrt(2).
ROT_ANGLES  = [-90.0, -45.0, 45.0, 90.0]
REFL_ANGLES = [0.0, 45.0, 90.0, 135.0]


def rot_mat(theta):
    """Rotation by theta.  det = +1, norm-preserving."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]], dtype=np.float32)


def refl_mat(phi):
    """Reflection across the line at angle phi.  det = -1, norm-preserving.

    phi is the angle of the MIRROR LINE.  Reflecting is "rotate the line onto
    the x-axis, flip y, rotate back":

        M(phi) = R(phi) @ diag(1, -1) @ R(-phi)

    and the two rotations compose to double the angle, which is where the
    2*phi entries come from.  M(phi) = M(phi + pi), so four axes span it.

    THE ASYMMETRY THIS CREATES, which is the point of including reflection.
    Rotation maps a vector at angle alpha to alpha + theta -- the SAME angular
    offset for every action.  Reflection maps it to 2*phi - alpha, so the
    offset is 2*(phi - alpha) and DEPENDS on the action's direction:

               alpha=0   alpha=30   alpha=60   alpha=90
      rotation   +30       +30        +30        +30
      reflection +60         0        -60       -120

    Consequently a SINGLE observed (action, force) pair cannot separate the
    two: for any one action, a reflection is indistinguishable from the
    rotation theta = 2*(phi - alpha).  Telling them apart requires sampling
    several action DIRECTIONS -- structured accumulated evidence rather than a
    threshold, and exactly the discrimination a forward-model error is good
    for.

    NOTE THE EFFECT-SIZE ASYMMETRY BETWEEN THE FAMILIES, which is geometric
    and cannot be tuned away.  |M(phi) a - a| = sqrt(2)|a| for EVERY axis, so
    all reflections have identical effect size and differ only in direction;
    |R(th) a - a| = 2|sin(th/2)||a|, so rotations are GRADED and only th = +-90
    matches a reflection.  The reflection family is therefore a pure direction
    sweep and the rotation family a magnitude-and-direction sweep.
    """
    return np.array([[np.cos(2 * phi),  np.sin(2 * phi)],
                     [np.sin(2 * phi), -np.cos(2 * phi)]], dtype=np.float32)


CONTEXTS = ([(0, th, i, f"rot {th:+g}") for i, th in enumerate(ROT_ANGLES)]
            + [(1, ph, len(ROT_ANGLES) + i, f"refl {ph:g}")
               for i, ph in enumerate(REFL_ANGLES)])
N_CONTEXTS = len(CONTEXTS)
C_LABEL = {c[2]: c[3] for c in CONTEXTS}
FAMILY  = {c[2]: c[0] for c in CONTEXTS}
C_PARAM = {c[2]: c[1] for c in CONTEXTS}       # the angle, in DEGREES
FAMS    = list(FAM_NAMES)


def context_matrix(c):
    """The action transform A for context id `c`.

    Angles live in the table in DEGREES and become radians only here, so the
    table stays readable and there is one place the conversion happens.

    c=None IS THE NULL FIELD: the identity, F = a.  It is NOT a member of the
    table -- no context is the identity, deliberately, since one would sit at
    effect size zero and be indistinguishable from "no perturbation".  But
    sac.py's warmup block needs exactly that condition, to fill a replay
    buffer without teaching a context, so it is a sentinel rather than a
    context id.  The guard lives HERE, in the one function that indexes the
    table, so every caller gets it.
    """
    if c is None:
        return np.eye(2, dtype=np.float32)
    kind, param = FAMILY[int(c)], C_PARAM[int(c)]
    return (rot_mat(np.deg2rad(param)) if kind == 0
            else refl_mat(np.deg2rad(param)))


# ── Sequences ────────────────────────────────────────────────────────────
SEQ_LEN   = 40       # steps per sequence.  Two grid crossings at the cap.
# fast switch p(switch) = 1 - STAY_PROB = 0.10, so block lengths are
# GEOMETRIC, not typical.  The consequence that matters downstream: few steps
# survive to tss >= 12, so late bins in every decoding curve hold ~3x less
# data than early ones, which is why the decoder is fit ONCE and scored per
# bin -- see decode.py.
STAY_PROB = 0.9
TSS_MIN   = 4
RHO       = 0.35     # AR(1) action correlation

# THE ACTION PROCESS LIVES HERE, not in data.py, because the plant's
# calibration DEPENDS ON IT and the two must not drift apart.  Calibrating the
# field coefficients against a standard normal while data.py drew sd = sqrt(1/3)
# left the two factors' marginals 20% apart, silently -- the aim marginals move
# with the action scale and the field marginals do not (see DRAG_LO).
#
# sd = sqrt(1/3) matches Var(uniform[-1, 1]), i.e. what a tanh-squashed policy
# produces, so the circuit is fitted on the action distribution it is deployed
# under.  Gaussian rather than uniform because a square support LEAKS ROTATION:
# its corners tell you the axes.
ACTION_SD = float(np.sqrt(1.0 / 3.0))


def next_action(a, rng, rho=RHO, sd=ACTION_SD):
    """One AR(1) step.  sqrt(1-rho^2) holds the marginal variance fixed, so rho
    changes only the temporal correlation and never the action scale.

    The draw is cast to float32 BEFORE the sd multiply, not after: the other
    order rounds differently and trajectories drift by ~5e-07, which is enough
    that trained checkpoints no longer see the data they were trained on.
    """
    eps = rng.normal(0.0, sd, 2).astype(np.float32)
    return (rho * np.asarray(a, dtype=np.float32)
            + np.sqrt(1.0 - rho ** 2) * eps).astype(np.float32)


def initial_action(rng, sd=ACTION_SD):
    """The AR(1) process at its stationary marginal."""
    return rng.normal(0.0, sd, 2).astype(np.float32)

# ── Stochasticity ────────────────────────────────────────────────────────
# WITHOUT IT THERE IS NOTHING TO INTEGRATE.  The plant is linear and was
# deterministic, so an ideal observer reached 100% on TWO steps: one
# (action, vel, dvel) triple is two exact equations, and two triples
# over-determine a 16-way choice.  A model could identify the context from the
# immediate prediction error and never use history at all.
#
# NOISE GOES IN THE MOTOR COMMAND, not on the velocity or the next state:
#
#     F = A @ (action + eta) + B @ vel,   eta ~ N(0, (CMD_NOISE*|action|)^2)
#
# which is the physiological model (noise scales with the command) AND the
# only placement whose magnitude cannot leak the aim factor -- every A level is
# orthogonal, so |A eta| = |eta| exactly.  Measured over the four aim levels the
# injected noise rms is 0.00615 / 0.00619 / 0.00615 / 0.00612, under 1.2%
# spread.
#
# The FIELD factor DOES shape the noise (drag damps accumulated noise, curl
# rotates it: rms 0.0056 / 0.0056 / 0.0086 against 0.0062 at B=0, a 1.6x
# spread).  That is unavoidable for any velocity-level noise, because B is
# literally a velocity feedback term, and it is benign here: reading it
# requires estimating a variance, which needs history -- the point of the
# exercise.  A speed-scaled noise would make it 4x and is rejected for that.
#
# CMD_NOISE IS CALIBRATED AGAINST THE BLOCK LENGTH.  Ideal-observer accuracy
# (16-way, chance 6.2%) after k steps at 0.28:
#
#     k = 1     2     5    10    20
#        21%   50%   86%   98%  100%
#
# Blocks are geometric with mean 1/(1-STAY_PROB) = 10 steps, so the context
# becomes identifiable on exactly the timescale it survives: not in one step,
# and not so slowly that it switches before it can be known.
CMD_NOISE = 0.28
# Additive process noise, both OFF by default -- kept because they are the
# other two placements one might want, not because they are recommended.
# VEL_NOISE perturbs velocity after the force is applied; POS_NOISE perturbs
# the next position directly.  Either can be combined with CMD_NOISE.
VEL_NOISE = 0.0
POS_NOISE = 0.0

# ── Network sizes ────────────────────────────────────────────────────────
EC_HIDDEN, CB_HIDDEN, LATENT_DIM = 256, 256, 32
N_TRAIN, BATCH, EPOCHS, LR, WD = 4000, 256, 250, 1e-3, 1e-4


def context_matrices(c):
    """(A, B) for context id `c`, in the plant's F = A a + B v form.

    B IS ZERO FOR EVERY CONTEXT.  Damping is a plant property now, applied
    inside env_step_ab, so a context is purely a transform of the action.  The
    (A, B) shape is kept because structure.py sweeps velocity-dependent fields
    (curl) as continuous parameters and needs the general form.
    """
    return context_matrix(c), np.zeros((2, 2), dtype=np.float32)


def clip_speed(vel):
    """Cap the SPEED, not each component, so the cap is isotropic and cannot
    quietly turn a diagonal velocity by shortening one axis more than the
    other -- which would be an extra, unlabelled force field.

    Shape-polymorphic: (2,) or (..., 2).  The norm is taken over the LAST axis
    so a stack of velocities is capped per-row, not by its Frobenius norm --
    which is what a naive float(norm) would do, silently, to a batch.
    """
    vel = np.asarray(vel, dtype=np.float32)
    n = np.linalg.norm(vel, axis=-1, keepdims=True)
    return np.where(n > V_MAX, vel * (V_MAX / np.maximum(n, 1e-12)),
                    vel).astype(np.float32)


def context_force(action, vel, c):
    """The context's force law.  Plant-free by construction: no MASS, no DT,
    no V_MAX appear here, so a context is a statement about forces and the
    plant alone says what forces do."""
    A, B = context_matrices(c)
    return (A @ np.asarray(action, dtype=np.float32)
            + B @ np.asarray(vel, dtype=np.float32)).astype(np.float32)


def env_step_ab(pos, vel, action, A, B, rng=None):
    """One step under an ARBITRARY force law F = A @ action + B @ vel.

    The plant, separated from the context table.  env_step is this with (A, B)
    looked up from a context id; analyses that sweep a CONTINUOUS parameter --
    structure.py's interpolation and extrapolation tests -- need the same plant
    at matrices that are not in the table, and must not carry their own copy of
    the integration.  structure.py used to, and its copy still had the damping
    term and the [0, GRID] clip long after both were gone here.

    Returns (pos_new, vel_new, F), where F is the force ACTUALLY applied -- the
    noisy one when rng is given, since that is what produced the motion and
    therefore what a forward model would have had to predict.

    rng=None is NOISELESS, for tests and for any analysis that needs the clean
    force law.  Every training and evaluation path should pass one: an agent
    evaluated without the noise it trained under is being handed a different
    plant.

    VELOCITY IS ZEROED ON THE AXIS THAT CLIPS.  With no damping a pinned agent
    would otherwise keep integrating force into the wall and need as many
    steps to escape as it spent arriving; the collision is inelastic instead.
    This is what makes wall time horizon-independent (~13% at both 40 and 300
    steps rather than 49% and 90%).

    NOTE for callers passing their own A: the command noise is injected BEFORE
    A is applied, so its magnitude is |eta| only when A is ORTHOGONAL.  Every
    aim level here is; a caller sweeping non-orthogonal matrices gets a noise
    amplitude that varies with the matrix, which would leak the parameter.
    """
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    # SHAPE-POLYMORPHIC: (2,) or (..., 2).  `x @ M.T` is M @ x for a single
    # vector and a per-row M @ x_i for a stack, so one expression serves both
    # and there is no batched copy of the plant to drift out of step.  This is
    # not hypothetical -- blind_ceiling.py and structure.py each carried their
    # own copy, and both still had the damping term and the [0, GRID] clip long
    # after those were removed here.
    if rng is not None and CMD_NOISE:
        scale = CMD_NOISE * np.linalg.norm(action, axis=-1, keepdims=True)
        action = action + scale * rng.standard_normal(action.shape).astype(np.float32)

    # DAMPING is PLANT, not context: every context runs under the same light
    # damping, so it is added here rather than folded into any context's B.
    # Keeping it out of context_force is what lets that function stay a
    # statement about the context alone.
    F = (action @ A.T + vel @ B.T - DAMPING * vel).astype(np.float32)
    vel_new = vel + (F / MASS) * DT
    if rng is not None and VEL_NOISE:
        vel_new = vel_new + VEL_NOISE * rng.standard_normal(vel_new.shape).astype(np.float32)
    vel_new = clip_speed(vel_new)

    raw = pos + vel_new * DT
    if rng is not None and POS_NOISE:
        raw = raw + POS_NOISE * rng.standard_normal(raw.shape).astype(np.float32)
    out = (raw < -POS_LIM) | (raw > POS_LIM)
    pos_new = np.clip(raw, -POS_LIM, POS_LIM)
    vel_new = np.where(out, np.float32(0.0), vel_new)
    return (pos_new.astype(np.float32), vel_new.astype(np.float32),
            F.astype(np.float32))


def env_step(pos, vel, action, c, rng=None):
    """One step in context `c`.  See env_step_ab for the plant itself."""
    A, B = context_matrices(c)
    return env_step_ab(pos, vel, action, A, B, rng)


def force_of(action, c, vel=None):
    """Ground-truth F for context id `c`, for the force-angle diagnostics.

    vel defaults to ZERO, which is exact for the one thing this is used for --
    the context-BLIND reference force, i.e. F averaged over every context.
    The field factor's four levels do not sum to zero (two drags and one curl
    against one identity), so pass a real velocity whenever a single context's
    force is what you want rather than the family mean.
    """
    v = np.zeros(2, dtype=np.float32) if vel is None else vel
    return context_force(action, v, c)
