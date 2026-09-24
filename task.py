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
MASS    = 32.0    # force -> acceleration (was 100; 32 since 2026-09-18 so a
                  # context's per-step effect on velocity is ~3x larger relative
                  # to the velocity state, while a typical action still takes
                  # ~3 steps from rest to the speed cap -- braking must still
                  # be planned).  Sets the ballistic fraction:
                  # V_MAX/(rms|F|/MASS) steps to saturate against 2/V_MAX
                  # steps to cross, which is 0.35 here -- accelerate, cruise,
                  # brake, with braking that has to be planned.  At ~1 the
                  # plant would be speed-controlled and "acceleration control"
                  # a fiction; at <<1 it never reaches the far wall.
DT      = 1.0
V_MAX   = 0.10    # speed cap.  Travel across the grid is 2/V_MAX = 20 steps.
POS_LIM = 1.0     # position lives in [-POS_LIM, +POS_LIM] on both axes
GRID_SIZE = 10 * POS_LIM        # the extent, for anything that needs a width

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
DAMPING = 1

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
# ── THREE FAMILIES since 2026-09-18: rotation, curl, drift ─────────────────
#
# The reflection family was replaced (kept below as refl_mat for the
# structure sweeps and the record). Measured on the rotation/reflection table,
# z coded each family as a ring of directions centred on the origin -- the
# two rings in orthogonal planes -- so every context was linearly separable
# but the FAMILY was not (which plane carries the energy is a quadratic), and
# a reflection is indistinguishable from some rotation under any single
# action direction, which is exactly what a competent policy supplies. The
# original repo's families were all continuous perturbations connected to
# the identity -- rotation, curl, drift -- each switching on a different
# linear feature of the force law, so family was a DIRECTION there and the
# code was additive throughout. These are those kinds of family, built in
# this plant:
#
#   rotation  F = R(theta) a            turns the commanded force
#   curl      F = a + k R(90) v         a force perpendicular to the velocity
#   drift     v <- v + w u(phi)         a constant velocity increment along
#                                       direction phi, added AFTER the force
#                                       has acted (not a force: the plant is
#                                       carried, the way a current carries a
#                                       swimmer)
#
# FOUR OF EACH. Rotations at +-30 and +-60 (user's choice, 2026-09-18; the
# rotation/reflection table had +-45/+-90).
#
# CALIBRATED TO ONE ANOTHER by effect size, |F - a| relative to the action:
# |R(th)a - a| = 2|sin(th/2)||a| puts +-30 at 0.52|a| and +-60 at 1.00|a|.
# Curl's effect is k|v|, so it DEPENDS ON MASS AND DAMPING through the plant's
# speed: measured in the null field on the GP action process
# (task.action_sequence, settled steps) at MASS = 32, DAMPING = 1, rms|v| =
# 0.0735 and rms|a| = 0.98, so k = 7 reproduces the +-30 effect (0.51) and
# k = 13 the +-60 one (0.98). (At MASS 100 / DAMPING 17.7, rms|v| was 0.0257
# and the matched gains 20 and 40; at MASS 100 / DAMPING 1, 0.0474 and 10 /
# 20 -- re-measure whenever either moves.) A velocity increment w per step is
# what a force MASS w / DT would produce, so w = 0.016 is the +-30 effect
# (0.51) in force units, per step; four directions at 90 degrees make it a
# pure direction sweep at one magnitude, as the reflections were. NOTE that
# with DAMPING = 1 the drift ACCUMULATES: its terminal speed w MASS / DAMPING
# = 0.5 is above V_MAX, so a drift context rides the speed cap and pins
# against walls more than the other families do. At this mass the cap binds
# on ~26% of settled null-field steps already. Curl's realised effect on the
# GP generator also runs above its null-field calibration because the field
# itself raises |v|.
#
# NOTE ON IDENTIFIABILITY. Rotation is identifiable from one (action, force)
# pair. Curl and drift act on velocity and as a constant: a curl needs a
# non-zero velocity, which starts at zero and builds (an ideal observer
# needed ~5 steps to reach 0.81 on the old drag/curl factor), and a drift
# is separable from a rotation only once the action changes direction or
# magnitude. That is the structure the families are meant to have.
FAM_NAMES = ["rotation", "curl", "drift"]

ROT_ANGLES  = [-60.0, -30.0, 30.0, 60.0]      # degrees
CURL_GAINS  = [-13.0, -7.0, 7.0, 13.0]        # k in F = a + k R(90) v (MASS 32, DAMPING 1)
DRIFT_ANGLES = [0.0, 90.0, 180.0, 270.0]      # degrees; the drift direction
DRIFT_VEL = 0.016                             # w, velocity units per step (MASS 32)
DRIFT_FORCE_EQUIV = DRIFT_VEL * MASS / DT      # the force that would produce w, in action units
REFL_ANGLES = [0.0, 45.0, 90.0, 135.0]        # the retired family, for structure.py


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


# (family index, parameter, context id, label). Parameters: rotation angle in
# DEGREES, curl gain k, drift direction in DEGREES (magnitude DRIFT_FORCE).
CONTEXTS = ([(0, th, i, f"rot {th:+g}") for i, th in enumerate(ROT_ANGLES)]
            + [(1, k, len(ROT_ANGLES) + i, f"curl {k:+g}")
               for i, k in enumerate(CURL_GAINS)]
            + [(2, ph, len(ROT_ANGLES) + len(CURL_GAINS) + i, f"drift {ph:g}")
               for i, ph in enumerate(DRIFT_ANGLES)])
N_CONTEXTS = len(CONTEXTS)
C_LABEL = {c[2]: c[3] for c in CONTEXTS}
FAMILY  = {c[2]: c[0] for c in CONTEXTS}
C_PARAM = {c[2]: c[1] for c in CONTEXTS}       # angle (deg) or curl gain
FAMS    = list(FAM_NAMES)


def unit(angle_deg):
    th = np.deg2rad(angle_deg)
    return np.array([np.cos(th), np.sin(th)], dtype=np.float32)


def context_affine(c):
    """(A, B, w) for context id `c`: the force law F = A a + B v, and the
    velocity increment w the plant adds after the force has acted
    (env_step_ab's `drift`). Zero w for every family but drift.

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
    I = np.eye(2, dtype=np.float32)
    Z = np.zeros((2, 2), dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    if c is None:
        return I, Z, z
    kind, param = FAMILY[int(c)], C_PARAM[int(c)]
    if kind == 0:                                     # rotation
        return rot_mat(np.deg2rad(param)), Z, z
    if kind == 1:                                     # curl: k R(90) v
        return I, (np.float32(param) * rot_mat(np.pi / 2)).astype(np.float32), z
    if kind == LINEAR_KIND:                           # a general action transform
        return LINEAR_MATS[int(c)], Z, z
    return I, Z, (DRIFT_VEL * unit(param)).astype(np.float32)     # drift


def context_matrix(c):
    """The action transform A alone (identity for curl and drift). Kept for
    callers that only ask how the ACTION is mapped; the full law is
    context_affine."""
    return context_affine(c)[0]


# ── Off-table rotations, for the zero-shot experiment (2026-09-21) ──────────
#
# sac.py's "zero_shot" schedule drops an agent trained on the table's four
# rotations into rotations the table does not hold -- 45 degrees, between two
# anchors; 90, beyond them -- and asks what it does before it has learned
# anything there. Those angles have to be contexts in every sense the rest of
# the code relies on: an integer id the plant, the experience bank and the
# logged curve can carry, a FAMILY entry that routes context_affine to
# rot_mat, a C_PARAM entry holding the angle, a C_LABEL entry naming it. So
# they are REGISTERED into the same tables the fixed contexts live in.
#
# Ids start at EXTRA_ID_BASE, far above the table, so nothing can mistake one
# for a table context; N_CONTEXTS is unchanged and still counts the table
# alone, and everything that enumerates range(N_CONTEXTS) -- the supervised
# data, the phase sets, the decoders -- never sees them. The id is a function
# of the angle (tenths of a degree, modulo 360), not of registration order,
# so a spawned worker and its parent agree on it without talking. An angle
# the table already holds returns the TABLE id: "the rotation at +30" is one
# context however it is asked for, which is what lets the zero-shot sweep
# include the trained anchors as ordinary points.
#
# A rotation of 0 is the identity plant, i.e. the same force law as the
# c=None null field -- but as a context with an id, not a sentinel, so the
# sweep can put "no perturbation" on the same axis as the rest.
EXTRA_ID_BASE = 1000


def rotation_id(theta_deg):
    """The context id of the rotation by `theta_deg`, registering it if the
    table does not hold it. Idempotent."""
    theta = float(theta_deg)
    for c, kind in FAMILY.items():
        if kind == 0 and c < EXTRA_ID_BASE and np.isclose(C_PARAM[c], theta):
            return c
    c = EXTRA_ID_BASE + int(round((theta % 360.0) * 10.0))
    if c not in FAMILY:
        FAMILY[c] = 0
        C_PARAM[c] = theta
        C_LABEL[c] = f"rot {theta:+g}"
    return c


def is_table_context(c) -> bool:
    """False for a registered off-table context (and for the None sentinel)."""
    return c is not None and 0 <= int(c) < N_CONTEXTS


# THE WIDE, SPARSE ROTATION SET (2026-09-21, user). With the table's anchors
# 30 degrees apart, a 50/50 mixture of two neighbouring policies is close to
# the policy for the angle between them, so interpolation cannot tell a
# mixture code (COIN) from an axis code (the circuit): both looked alike at
# +-45 in the first zero-shot sweep. Doubling the gap is the test: anchors at
# +-30 and +-90, midpoints at +-60 where averaging the two policies should
# stop being a good policy while a point on the axis between them should
# not. Within 90 degrees so the feedback sign never flips. +-30 is kept so
# the near-anchor tests at 15 and 45 stay comparable with the table's.
#
# NOT a change to the table: that would silently redefine the COIN paradigms
# (P+ is the largest rotation), the mixed context set and every checkpoint.
# It is a named set -- model.TRAIN_CONTEXT_SETS "rotations_wide" for the
# circuit, sac.PHASE_SETS "rotations_wide" for the agent -- built from the
# registry above, so the +-90 contexts are real ids everywhere.
WIDE_ROT_ANGLES = [-90.0, -30.0, 30.0, 90.0]


def wide_rotation_ids():
    """The four wide rotations' context ids (table ids for +-30, registered
    ids for +-90), in WIDE_ROT_ANGLES order."""
    return [rotation_id(th) for th in WIDE_ROT_ANGLES]


# ── Unstructured action transforms: the structure-learning CONTROL ──────────
#
# (2026-09-21, user.) Braun, Aertsen, Wolpert & Mehring (2009) established
# structural learning with a contrast: exposure to a STRUCTURED family of
# transformations (rotations, one parameter) against exposure to
# UNSTRUCTURED ones (random linear transforms), measured as how fast a novel
# family member is learned afterwards. The zero-shot experiment has the
# structured arm -- a circuit fitted on the wide rotations -- and this is the
# other: a circuit fitted on the same NUMBER of action transforms at the same
# EFFECT SIZES, drawn at random from the general linear group, so nothing
# about them lies on a one-parameter axis. A novel rotation lies inside that
# family too (every rotation is a linear map), so the unstructured circuit
# can represent it -- but it has to identify four numbers rather than one.
# Whatever the rotation-trained circuit does better at novel rotations is
# what learning the structure bought. And it is where the circuit parts
# from COIN most sharply: the inferred oracle has no circuit, so which family
# was seen in pretraining cannot reach it at all.
#
# EFFECT SIZE of an action transform A, as the module measures it (the rms
# over unit action directions of |A u - u|): ||A - I||_F / sqrt(2), which for
# a rotation is 2 |sin(theta/2)| -- 0.52 at 30 degrees, 1.41 at 90. The
# random set draws A - I uniformly on the Frobenius sphere of exactly those
# radii, two at each, so the two sets differ in structure alone.
#
# A fourth context KIND, outside FAM_NAMES on purpose: the three families
# are the table's design and everything that enumerates them (the phase
# sets, the decoders' per-family scores) must keep seeing three. These
# contexts exist only through the registry, under ids from LINEAR_ID_BASE.
# C_PARAM holds the draw index; the matrix lives in LINEAR_MATS.
LINEAR_KIND = 3
LINEAR_ID_BASE = 10_000
LINEAR_MATS = {}
RANDOM_LINEAR_SEED = 2026


def effect_size(A) -> float:
    """rms over unit action directions of |A u - u|: ||A - I||_F / sqrt(2)."""
    return float(np.linalg.norm(np.asarray(A, dtype=np.float64) - np.eye(2)) / np.sqrt(2.0))


def register_linear(A, index, label=None):
    """Register the action transform `A` (2x2) as a context of LINEAR_KIND
    under draw index `index`; idempotent for the same index."""
    c = LINEAR_ID_BASE + int(index)
    if c not in FAMILY:
        FAMILY[c] = LINEAR_KIND
        C_PARAM[c] = float(index)
        C_LABEL[c] = label or f"lin {int(index)}"
        LINEAR_MATS[c] = np.asarray(A, dtype=np.float32).reshape(2, 2)
    return c


def random_linear_ids(effects=None, seed=RANDOM_LINEAR_SEED):
    """The unstructured control set: one random linear transform per entry
    of `effects` (default: the wide rotations' effect sizes, so two at 0.52
    and two at 1.41), A - I uniform on the Frobenius sphere of radius
    sqrt(2) * effect. Deterministic in `seed`; registered on first call."""
    if effects is None:
        effects = [effect_size(rot_mat(np.deg2rad(th))) for th in WIDE_ROT_ANGLES]
    rng = np.random.default_rng(seed)
    ids = []
    for i, eff in enumerate(effects):
        E = rng.standard_normal(4)
        E = E / np.linalg.norm(E) * np.sqrt(2.0) * float(eff)
        A = np.eye(2) + E.reshape(2, 2)
        ids.append(register_linear(A, i, label=f"lin {i} ({eff:.2f})"))
    return ids


# ── Sequences ────────────────────────────────────────────────────────────
SEQ_LEN   = 128       # steps per sequence.  Two grid crossings at the cap.
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


# ── A GP FAMILY OF ACTION PROCESSES ──────────────────────────────────────
#
# next_action above is ONE point in this family and the reason the family
# exists.  The circuit is fitted on it (rho = 0.35, Gaussian marginal) and
# deployed under a trained SAC policy, whose actions are nothing like it:
# measured on main_frozen seed 0, lag-1 correlation 0.92 against 0.35, mean
# |a| 1.24 against 0.72, and 44.7% of steps with BOTH components past 0.9
# against 1.3%.  Under that shift z stops being an additive context code and
# becomes conjunctive in (s, a, c) -- conditioning on (s, a) recovers 7.3x more
# context separation, and the context subspace rotates 43 degrees across state
# bins.  Additivity, in other words, was never a property of the circuit; it
# was a property of the circuit ON ONE INPUT DISTRIBUTION, and nothing in the
# supervised objective asked it to hold anywhere else.
#
# So: draw a DIFFERENT action process per trajectory, wide enough that no
# single regime can be relied on.
#
#   kernel   Matern(nu, ell) over the time axis.  nu = 1/2 IS the OU process,
#            i.e. exactly AR(1), so next_action's rho = 0.35 is the point
#            (nu = 1/2, ell = -1/ln(0.35) = 0.95) and this family strictly
#            contains what the circuit is fitted on today.
#   warp     a = tanh(g * u) / k(g), renormalised so the marginal sd stays
#            ACTION_SD whatever g is.  g -> 0 is Gaussian; g ~ 3.4 reproduces
#            the policy's corner mass.
#
# SCALE VARIES TOO, AND THAT IS SAFE HERE -- checked, not assumed.  ACTION_SD's
# warning is about balancing the AIM and FIELD factors against each other, and
# FIELD is gone: B == 0 for every context, every A is norm-preserving, so the
# plant is LINEAR in the action and a scale factor cancels out of the effect
# size.  Measured, rms|dpos_c - dpos_null| / rms|dpos_null| is identical to four
# decimals at scale 1.0 and 1.6 for all eight contexts.  What the scale does
# change is the overall SNR of the inference problem, which is a robustness axis
# worth covering rather than a confound.
#
# It has to vary, because holding it fixed makes the saturated end unreachable:
# renormalising tanh(5u) to ACTION_SD caps |a| at 0.60, so the family would span
# saturation SHAPE while never producing the magnitude the policy actually uses
# (per-axis sd 0.885 against ACTION_SD's 0.577, a factor of 1.53).
#
# AND IT SETTLES THE ISOTROPY PROBLEM RATHER THAN CHOOSING A SIDE.  A saturated
# per-axis marginal is square-supported, and a square support LEAKS ROTATION
# (see ACTION_SD) -- its corners tell you the axes, which is precisely the
# distinction the rotation/reflection families exist to test.  Drawing g per
# trajectory means some sequences are isotropic Gaussian and some are square:
# the axis cue is present only sometimes, so relying on it is punished rather
# than rewarded.
#
# WHAT THIS DOES NOT FIX, and it is accepted deliberately: at long ell a
# sequence carries very little context evidence, because evidence comes from
# independent (action, response) pairs.  n_eff = T (1 - rho) / (1 + rho) is
# 19.3 probes over SEQ_LEN = 128 at rho = 0.35 and 1.7 at rho = 0.92, against
# the ~5 the ideal observer needs for 86% (see CMD_NOISE).  So the long-ell
# tail of this family is close to unidentifiable IN PRINCIPLE, not merely
# hard.  Those draws are kept anyway -- they are what deployment looks like,
# and a circuit that learns to be uncertain there is behaving correctly.
ACTION_NU_CHOICES = (0.5, 1.5, 2.5, np.inf)   # inf = squared-exponential
# TRUNCATED FROM (0.3, 30.0), AND THE MEASUREMENT IS WHY.  The wide range put
# HALF the draws above ell = 3, and that half is the worst of both worlds --
# stratified over the family, an ell band of [10, 30) has n_eff = 0.6
# independent action probes and context decodes at 0.568, while [0.3, 1.0) has
# 28.5 probes and decodes at 0.833.  CMD_NOISE is calibrated so an ideal
# observer needs ~5 probes for 86%, so above ell ~ 3 the context is close to
# unidentifiable IN PRINCIPLE rather than merely hard.
#
# And those sequences are the HARDEST TO PREDICT (vel MSE 4.47e-3 against
# 2.55e-3 at short ell), so in an unweighted mean they carry ~1.9x the loss
# while providing the least pressure to build a context code.  The
# uninformative half dominated the gradient.
#
# MEASURED CONSEQUENCE of getting this wrong: a circuit trained on the wide
# range came out MORE conjunctive than the fitted one under deployment
# (conditioning gain 31.2x against 8.15x, 8% of its context code additive
# against 25%), and decoded context on its OWN distribution at 0.635 while the
# fitted circuit managed 0.735 on that same data.  Training on a distribution
# made it worse at that distribution.
#
# WHAT THIS GIVES UP, deliberately: rho now reaches 0.717, so the family no
# longer covers the deployed policy's 0.92.  That is the point -- covering a
# regime where the evidence is not there to extract buys nothing, and if the
# deployed correlation really is unidentifiable then the fix belongs on the
# POLICY side (an entropy floor, or squashing the action norm instead of each
# component) rather than in the circuit's training data.
ACTION_ELL_RANGE = (0.3, 3.0)                 # log-uniform; rho 0.036 .. 0.717
ACTION_GAIN_RANGE = (0.0, 5.0)                # Gaussian .. hard-saturated
ACTION_SCALE_RANGE = (0.6, 1.8)               # x ACTION_SD; fitted is 1.0 and
                                              # the trained policy is 1.53


def matern_kernel(lags, nu, ell):
    """Matern correlation at integer `lags`.  nu = inf is squared-exponential.

    Correlation, not covariance: k(0) = 1, so the marginal is set entirely by
    the warp's renormalisation and never by the kernel.
    """
    d = np.abs(np.asarray(lags, dtype=np.float64)) / float(ell)
    if not np.isfinite(nu):
        return np.exp(-0.5 * d ** 2)
    if nu == 0.5:
        return np.exp(-d)
    if nu == 1.5:
        r = np.sqrt(3.0) * d
        return (1.0 + r) * np.exp(-r)
    if nu == 2.5:
        r = np.sqrt(5.0) * d
        return (1.0 + r + r ** 2 / 3.0) * np.exp(-r)
    raise ValueError(f"unsupported nu {nu!r}; expected one of {ACTION_NU_CHOICES}")


def _tanh_sd(gain, n_quad=129):
    """sd of tanh(gain * u) for u ~ N(0, 1), by Gauss-Hermite quadrature.

    Quadrature rather than Monte Carlo so the normaliser is DETERMINISTIC: a
    sampled one would make the marginal sd depend on an rng draw, and the
    plant's calibration is written against a fixed ACTION_SD.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n_quad)
    t = np.tanh(float(gain) * x)
    return float(np.sqrt(np.sum(w * t ** 2) / np.sum(w)))


def sample_action_params(rng):
    """One trajectory's action process: (nu, ell, gain).

    ell is LOG-uniform because timescales are multiplicative -- uniform would
    put almost all its mass at the slow end and never sample the fast one.
    """
    lo, hi = ACTION_ELL_RANGE
    return {
        "nu": float(ACTION_NU_CHOICES[rng.integers(len(ACTION_NU_CHOICES))]),
        "ell": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "gain": float(rng.uniform(*ACTION_GAIN_RANGE)),
        "scale": float(rng.uniform(*ACTION_SCALE_RANGE)),
    }


def _chol(K):
    """Cholesky with escalating jitter, falling back to an eigen-clip.

    A smooth kernel at long ell is badly conditioned -- nu = inf at ell = 30
    over 40 steps is numerically rank ~3 -- so a bare cholesky fails on draws
    this family is meant to produce, not on exotic ones.
    """
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(len(K)))
        except np.linalg.LinAlgError:
            continue
    w, V = np.linalg.eigh(K)
    return V * np.sqrt(np.maximum(w, 0.0))


def action_sequence(n_steps, rng, params=None, sd=ACTION_SD):
    """A whole trajectory's actions, (n_steps, 2) float32.

    SEQUENCE-LEVEL, not a step recursion, because a GP draw is joint over time.
    That also makes the action stream independent of the context stream BY
    CONSTRUCTION -- data.generate_sequences maintains that by convention today
    ("the action process is NOT reset at a switch"), and a joint draw cannot
    violate it.

    Marginal sd is `sd * params["scale"]`.  The scale is free because the
    plant is linear in the action and every context is norm-preserving, so it
    moves the inference problem's SNR without touching the contexts' relative
    effect sizes -- see the section header.
    """
    params = sample_action_params(rng) if params is None else params
    lags = np.abs(np.subtract.outer(np.arange(n_steps), np.arange(n_steps)))
    K = matern_kernel(lags, params["nu"], params["ell"])
    L = _chol(K)
    u = L @ rng.standard_normal((n_steps, 2))          # unit marginal, per axis
    g = params["gain"]
    target = sd * params.get("scale", 1.0)
    if g < 1e-6:                                       # the Gaussian limit
        a = u * target
    else:
        a = np.tanh(g * u) * (target / _tanh_sd(g))
    return a.astype(np.float32)

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
N_TRAIN, BATCH, EPOCHS, LR, WD = 8192, 256, 256, 1e-3, 1e-4


def context_matrices(c):
    """(A, B) for context id `c`: the force law F = A a + B v.

    The drift increment is NOT in this pair -- callers that need the whole
    context use context_affine. Kept because structure.py was written against
    the (A, B) form; B is non-zero for the curl family only.
    """
    A, B, _w = context_affine(c)
    return A, B


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


def env_step_ab(pos, vel, action, A, B, rng=None, return_impulse=False, drift=None):
    """One step under an ARBITRARY force law F = A @ action + B @ vel, with an
    optional velocity increment `drift` added after the force has acted.

    `drift` is zero when omitted, so every caller written against the (A, B)
    form is unchanged. It is NOT part of F: F is what the action and the
    field did, the drift is what carried the plant regardless.

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
    if drift is not None:
        vel_new = (vel_new + np.asarray(drift, dtype=np.float32)).astype(np.float32)
    if rng is not None and VEL_NOISE:
        vel_new = vel_new + VEL_NOISE * rng.standard_normal(vel_new.shape).astype(np.float32)
    vel_new = clip_speed(vel_new)

    raw = pos + vel_new * DT
    if rng is not None and POS_NOISE:
        raw = raw + POS_NOISE * rng.standard_normal(raw.shape).astype(np.float32)
    out = (raw < -POS_LIM) | (raw > POS_LIM)
    pos_new = np.clip(raw, -POS_LIM, POS_LIM)
    # the momentum the wall takes out, BEFORE the velocity is zeroed -- the
    # collision is inelastic, so the impulse is the whole normal component.
    # Per-axis, because `out` is per-axis: a corner contact removes both.
    removed = np.where(out, vel_new, np.float32(0.0))
    vel_new = np.where(out, np.float32(0.0), vel_new)
    stepped = (pos_new.astype(np.float32), vel_new.astype(np.float32),
               F.astype(np.float32))
    if not return_impulse:
        return stepped
    return stepped + (wall_impulse(removed),)


def wall_impulse(removed_velocity):
    """|J| = MASS * |v_removed|, the impulse an inelastic wall contact delivers.

    A SEPARATE FUNCTION so the normaliser below has something to name, and so
    the quantity is defined once. Norm over the LAST axis, matching
    clip_speed: a stack of contacts is scored per row.
    """
    v = np.asarray(removed_velocity, dtype=np.float32)
    return (MASS * np.linalg.norm(v, axis=-1)).astype(np.float32)


# The largest impulse the plant can deliver: a head-on contact at the speed
# cap. Dividing by it puts wall contact on [0, 1] regardless of MASS and V_MAX,
# so a reward weight written against it does not silently change meaning when
# the plant is retuned -- which is exactly how ACTION_SD and the field
# calibration came apart once (see next_action).
MAX_WALL_IMPULSE = float(MASS * V_MAX)


def env_step(pos, vel, action, c, rng=None, return_impulse=False):
    """One step in context `c`.  See env_step_ab for the plant itself.

    `return_impulse` appends |J| from any wall contact as a FOURTH return
    value. Opt-in because every existing caller unpacks exactly three, and a
    silent arity change is the class of breakage this module keeps warning
    about."""
    A, B, w = context_affine(c)
    return env_step_ab(pos, vel, action, A, B, rng, return_impulse, drift=w)


def force_of(action, c, vel=None):
    """Ground-truth F for context id `c`, for the force-angle diagnostics.

    vel defaults to ZERO, which is exact for the one thing this is used for --
    the context-BLIND reference force, i.e. F averaged over every context.
    Curl's force is velocity-dependent, so pass a real velocity whenever a
    single curl context's force is what you want rather than the family mean.
    """
    v = np.zeros(2, dtype=np.float32) if vel is None else vel
    return context_force(action, v, c)
