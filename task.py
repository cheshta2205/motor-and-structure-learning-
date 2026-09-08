"""
task.py -- THE TASK: what a context is, how a step evolves, and the constants
every other module reads.  Defines the 16 contexts and the four families they
fall into, the transformation each one applies to the agent's action, and the
dynamics that turn that force into the next state.  data.py draws sequences from
it; model.py learns to predict them.

ONE TASK,  16 contexts in 4 families, CARDINAL drift offsets, curl
acts on V, rotation and reflection. AR(1) Gaussian actions at rho = 0.35, correlated.  

THE FOUR FAMILIES

  rotation    F = R(theta) a          orientation-preserving, det +1
  reflection  F = M(phi) a            orientation-REVERSING, det -1
  drift       F = a + g*b             a constant offset
  curl        F = a + k*R(90deg) v    turns the current VELOCITY

Rotation and reflection are the load-bearing pair: both are norm-preserving
(|Ma| = |a|) and ZERO-MEAN over actions (E[R(theta)a] = R(theta)E[a] = 0), so no
statistic of the observations alone identifies them -- you must extract from the 
the action you took.  That is exactly what a forward-model
prediction error provides, which is why these two show the strongest cerebellar
dependence.  

Drift is the opposite end: easier, add as a liar sum, temporal averaging identifies it and
its dep on prediction error is less.  

"""
import numpy as np

# ── Contexts: (kind, param, bx, by, c_id, label, family) 
#  0 rotation   param = theta   F = R(theta) a
#  1 reflection param = phi     F = M(phi) a      (mirror line at phi)
#  2 drift      param = g       F = a + g*(bx,by)
#  3 curl       param = k       F = a + k*R(bx) v
# Reflections repeat at phi + pi, so four axes span that family.
#
# THE DRIFT OFFSETS ARE COLLINEAR -- all four on the x-axis at two magnitudes
# (+-1.6, +-1.0).  This is deliberate and it is what the trained checkpoints
# use, but it was NOT the intent, and the history matters because every figure
# and filename in the exploratory code says "cardinal":
#
CONTEXTS = [
    (0, +np.pi / 2,   0.0,  0.0,  0, "C0: rot +90",    0),
    (0, +np.pi / 4,   0.0,  0.0,  1, "C1: rot +45",    0),
    (0, -np.pi / 2,   0.0,  0.0,  2, "C2: rot -90",    0),
    (0, -np.pi / 4,   0.0,  0.0,  3, "C3: rot -45",    0),
    (1, 0.0,          0.0,  0.0,  4, "C4: refl 0",     1),
    (1, np.pi / 4,    0.0,  0.0,  5, "C5: refl 45",    1),
    (1, np.pi / 2,    0.0,  0.0,  6, "C6: refl 90",    1),
    (1, 3 * np.pi / 4, 0.0, 0.0,  7, "C7: refl 135",   1),
    (2, 0.25,        +1.6,  0.0,  8, "C8: drift ++",   2),
    (2, 0.25,        +1.0,  0.0,  9, "C9: drift +",    2),
    (2, 0.25,        -1.6,  0.0, 10, "C10: drift --",  2),
    (2, 0.25,        -1.0,  0.0, 11, "C11: drift -",   2),
    (3, +0.50, +np.pi / 2,  0.0, 12, "C12: curl +.5",  3),
    (3, +0.25, +np.pi / 2,  0.0, 13, "C13: curl +.25", 3),
    (3, -0.50, +np.pi / 2,  0.0, 14, "C14: curl -.5",  3),
    (3, -0.25, +np.pi / 2,  0.0, 15, "C15: curl -.25", 3),
]
N_CONTEXTS = len(CONTEXTS)
C_LABEL  = {c[4]: c[5] for c in CONTEXTS}
FAMILY   = {c[4]: c[6] for c in CONTEXTS}
FAM_NAME = {0: "rotation", 1: "reflection", 2: "drift", 3: "curl"}
FAMS     = [FAM_NAME[i] for i in sorted(set(FAMILY.values()))]

# ── Dynamics 
# GRID_SIZE stays at 4 to match the navigation/savings task.  A constant drift
# force integrates twice, so its excursion over T steps is c/(1-LAMBDA)*DT*T;
# at the original DT=1.0 that is ~32 units and a drift agent spends 89% of its
# steps pinned against a wall, 
GRID_SIZE = 4.0
DT        = 0.04     # set by the excursion constraint above
LAMBDA    = 0.3      # heavier damping -> lower steady-state |V|, and it is what
                     # makes the curl loop contracting rather than explosive
# S_SCALE is INPUT CONDITIONING, not physics.  
S_SCALE   = 15.0

# ── Sequences ──
SEQ_LEN   = 40       # steps per sequence
# fast switch p(switch) = 1 - STAY_PROB = 0.10, so block lengths are
# GEOMETRIC, not typical. 
# The consequence that matters downstream: few steps survive to
# tss >= 12,  Late bins in every decoding curve
# therefore hold ~3x less data than early ones, which is why the decoder is fit
# ONCE and scored per bin -- see decode.py.
STAY_PROB = 0.9
TSS_MIN   = 4        #
RHO       = 0.35     # AR(1) action correlation.  
# ── Network sizes 
EC_HIDDEN, CB_HIDDEN, LATENT_DIM = 256, 256, 64
# Taken from the AR(1) pipeline (ar1_sweep), NOT from the legacy 12-context
# module that sits beside it in the old tree -- that one says 2000/128/60 and
# using its numbers trains on half the data.  Caught 2026-08-25 by comparing a
# fresh run against the original checkpoints: MSE came out 0.056 instead of
# 0.041, i.e. visibly worse but not obviously broken.
N_TRAIN, BATCH, EPOCHS, LR = 4000, 256, 350, 1e-3


def rot_mat(theta):
    """Rotation by theta.  det = +1, norm-preserving."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


def refl_mat(phi):
    """Reflection across the line at angle phi.  det = -1, norm-preserving.

    phi is the angle of the MIRROR LINE.  Reflecting
    is "rotate the line onto the x-axis, flip y, rotate back":

        M(phi) = R(phi) @ diag(1, -1) @ R(-phi)

    and the two rotations compose to double the angle, which is where the 2*phi
    entries come from.  M(phi) = M(phi + pi), so four axes span the family.

    THE ASYMMETRY THIS CREATES, which is the point of including reflection.
    Rotation maps a vector at angle alpha to alpha + theta -- the SAME angular
    offset for every action.  Reflection maps it to 2*phi - alpha, so the offset
    is 2*(phi - alpha) and DEPENDS on the action's direction:

               alpha=0   alpha=30   alpha=60   alpha=90
      rotation   +30       +30        +30        +30
      reflection +60         0        -60       -120

    Consequently a SINGLE observed (action, force) pair cannot separate the two:
    for any one action, a reflection is indistinguishable from the rotation
    theta = 2*(phi - alpha).  Telling them apart requires sampling several
    action DIRECTIONS -- structured accumulated evidence rather than a
    threshold, and exactly the discrimination a forward-model error is good for.
    """
    return np.array([[np.cos(2 * phi),  np.sin(2 * phi)],
                     [np.sin(2 * phi), -np.cos(2 * phi)]])


def env_step(pos, vel, action, kind, param, bx, by):
    """One step.  V <- LAMBDA*V + F,  S <- clip(S + DT*V, 0, GRID_SIZE).

    F depends on the ACTION for rotation and reflection, is a constant offset
    for drift, and turns the VELOCITY for curl.  Only curl feeds back through V;
    damping keeps that loop contracting -- see the module docstring.
    """
    if kind == 0:
        F = rot_mat(param) @ action
    elif kind == 1:
        F = refl_mat(param) @ action
    elif kind == 3:
        F = action + param * (rot_mat(bx) @ vel)
    else:
        F = action + param * np.array([bx, by], dtype=np.float32)
    vel_new = LAMBDA * vel + F
    pos_new = np.clip(pos + DT * vel_new, 0.0, GRID_SIZE)
    return pos_new.astype(np.float32), vel_new.astype(np.float32), F


def force_of(action, c, vel=None):
    """Ground-truth F for context id `c`, for the force-angle diagnostics.

    vel defaults to ZERO, which is exact for the one thing this is used for --
    the context-BLIND reference force, i.e. F averaged over every context.
    Curl's four gains are +.5/+.25/-.5/-.25 and sum to zero, so the curl family
    contributes exactly `action` to that mean whatever the velocity is.  Pass a
    real velocity if you ever want a single curl context's force on its own.
    """
    kind, param, bx, by = CONTEXTS[c][:4]
    if kind == 0:
        return rot_mat(param) @ action
    if kind == 1:
        return refl_mat(param) @ action
    if kind == 3:
        v = np.zeros(2, dtype=np.float32) if vel is None else vel
        return action + param * (rot_mat(bx) @ v)
    return action + param * np.array([bx, by], dtype=np.float32)
