"""
Cerebellar contribution to the hippocampal spatial code (Rochefort et al., 2011).


    python rochefort_final.py --train   --seeds 0 1 2 3
    python rochefort_final.py --figures --seeds 0 1 2 3


THE CIRCUIT

    e_t     = g * (s_t - s_hat_t)                cerebellar prediction error
    z_t, h  = MTL([s_{t-1}, a_{t-1}, e_t], h)    pose belief inferred from it
    s_hat   = s_in + CB([s_in, a_t, z_t])        forward model, conditioned on it

Trained end to end on next-state prediction.  The animal has no access to its true position.  The
arena is circular so it has no intrinsic orientation, which makes the cue the
only allocentric reference and removing it genuinely remove the reference frame.


THE OBSERVATION MODEL

The cerebellum never substitutes for the cue while the lights are on.  Every cue
channel is present and degraded by how well it can currently be sensed:

    s_obs = sqrt(p) * s_true + sqrt(1-p) * sigma_c * eps

    p(d)    = 1                                     d <= R
    p(d)    = 0.05 + 0.95 / (1 + ((d - R)/0.7)^2)   d >  R
    A(beta) = b0^12 / (|beta|^12 + b0^12),          b0 = FOV/2 = 150 deg
    p       = min(p(d), 0.05 + 0.95 * A(beta))

so the cue fades at the rim of the 300-degree field instead of blinking out.

SAME ANGULAR GATE AS THE OLM/ORM MODEL, NO RADIAL ONE.  beta0 = FOV/2 = 150 deg
is identical there, so the cue is clean to ~110 deg and falls to half precision
at 150.  What differs is the radius: there the objects are small things inside
the arena and vision (0.90) and whisker reach (0.80) both bite, whereas here the
sensing radius (2.2) exceeds the largest possible cue distance (2.0), so p(d) is
always 1 and the distance term never fires -- the cue is a large wall card meant
to be readable arena-wide, and Rochefort's gate was always angular.  Angle alone
degrades it.

THE DARK IS DIFFERENT ON PURPOSE.  With no light there is no signal to degrade,
so the cue channels are substituted by the cerebellum's own prediction, and what
persists is its dead-reckoned estimate.  Both conditions have memory; they
differ in whether there is a live signal to combine it with.

Heading is gated by the LIGHT, not by the cue: in a lit arena you know which way
you are facing from the walls and optic flow whether or not that one cue happens
to be behind you.  Contact channels are never gated -- whiskers work in the dark.


THE LESION

Noise on what the cerebellum TRANSMITS, applied during training.  The clean copy
carries the recursion so the belief stays bounded; propagating the noisy one
makes the dark an unbounded random walk.  Three earlier implementations failed
and are not offered here: freezing CB at zero-init leaves MTL with no gradient;
freezing it at a small init lets MTL invert the fixed map; and putting the noise
in the loss degrades the mutant in every condition rather than only the dark.


WHAT IS MEASURED

The five-session protocol -- cue, cue, dark, dark, cue -- with the animal removed
and reinserted only at the first and last transitions, as in the experiment.
Rate maps are built in cue-centred coordinates, since absolute angle decodes at
R^2 ~ 0: it is arbitrary from session to session.  Each arena is one mouse, and
pooled figures average across independently trained networks.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

_HERE    = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_HERE, "checkpoints")
FIG_ROOT = os.path.join(_HERE, "figures")


_POOLED_TAG = None      # set by main() for multi-seed runs


def fig_dir(seed):
    """One folder per seed under ROCHEFORT_FINAL, plus a pooled folder when
    several seeds are averaged.  Keeping them apart matters: a pooled figure
    written into seed0/ under the same filename silently overwrote the
    single-seed one once."""
    sub = _POOLED_TAG or f"seed{seed}"
    d = os.path.join(FIG_ROOT, "main", "ROCHEFORT_FINAL", sub)
    os.makedirs(d, exist_ok=True)
    return d


# The world

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- arena 
R_WALL       = 1.0
STEP         = 0.04
SPEED_JITTER = 0.30
TURN_MOM     = 0.80
TURN_SD      = 0.35

# -- perceptibility 
SENSE_FULL = 0.90         # vision range, MATCHED TO THE OLM/ORM MODEL.  Was 2.2
                          # (arena-wide, distance term inert); now the cue's
                          # vision channels degrade beyond 0.90 exactly as an
                          # object's do there.  The cue sits ON the wall while
                          # those objects sit inside r<0.75, so the same number
                          # is STRICTER here: clean over ~33% of the arena
                          # against ~46% there.  Erring strict is the safe
                          # direction.  Dark is untouched -- it is a gate
                          # (g=0 -> CB substitution), not a precision term.
FOV_DEG    = 300.0        # alpha: mouse visual field, 60 deg blind zone behind
K_ANG      = 0.15         # softness of the FOV rim (rad)
K_RAD      = 0.08         # softness of the range edge
CUE_GAIN   = 0.30
CUE_SOFT   = 0.15
BEAR_NOISE = 0.0          # difficulty should come from partial observability,
                          # not from noise we injected; put back later to find
                          # how much the forward model tolerates
INT_NOISE  = 0.0

S_DIM   = 7               # sin(beta) cos(beta) intensity | sin(th) cos(th) | - the state input vector 
                          #                                 at_wall at_cue
                          #
            
# Channel groups.  CUE_DIM are the egocentric cue variables; HEAD_SLICE is
# Earth-referenced heading; the rest is whisker contact.
#   [0:3] sin(beta) cos(beta) intensity   -- the cue, seen - replace these w CB s^ if out of FOV OR if DARK 
#   [3:5] sin(theta) cos(theta)           -- heading
#   [5:7] at_wall at_cue                  -- contact
CUE_DIM = 3
HEAD_SLICE = slice(3, 5)
VIS_DIM = 5               # the first VIS_DIM are the VISION channels: what
                          # darkness removes, and what CB substitutes for.  The
                          # last two are CONTACT and are never substituted --
                          # whiskers work in the dark, so they stay real and
                          # keep e alive in exactly the condition where the
                          # visual error signal is gone.
WALL_TOUCH = 0.08         # boundary contact.  Pins r and says NOTHING about phi
                          # (a circle looks the same from every point on its
                          # rim), so the residual drift is a pure rotation of
                          # the map -- the coherent remapping the paper reports.
CUE_TOUCH  = 0.15         # touching the cue.  It is a card with a bottle
                          # attached, deliberately readable by touch and smell,
                          # which is why dark+cue-preserved shows no deficit.
                          # Being one point on the wall it pins BOTH r and phi:
                          # a full re-anchor.
A_DIM = 2                 # speed, turn rate

# -- networks 
MTL_HIDDEN = 256 
LATENT_DIM = 64
CB_HIDDEN  = 256
CB_INIT_STD = 0.05        # see Cerebellum.__init__ -- must not be 0
INSERT_JITTER    = 0.05   # sd of where the animal is put back, in arena radii
INSERT_JITTER_TH = 0.20   # sd of the heading it is put back at (rad)

# Correlation time of the lesion noise, in steps.  0 = white (the original).
# White noise AVERAGES OUT of a rate map: ~10 visits per bin over a session cuts
# it by ~sqrt(10), so most of the corruption never reaches the measure.  That is
# one reason our field-rate deficit is ~half the paper's.  A correlated
# perturbation is wrong the SAME way for a stretch, so binning cannot cancel it
# -- and it is arguably the better model of blocked LTD, which leaves a
# persistently mis-adapted mapping rather than fresh static every step.
# Resampled per sequence, so it cannot be inverted the way a frozen CB was.

CB_NOISE    = 0.30        # L7-PKCI: sd of the noise on CB's prediction,
                          # in units of the signal.  Swept -- too small does
                          # nothing, too large breaks the cued condition too
                          # and the interaction is lost.

# -- training 
BATCH    = 64
SEQ_LEN  = 200
EPOCHS   = 60
STEPS_EP = 100    # 2,400 updates plateaued at R^2~0.94 on a deterministic
                  # system; 6,000 with cosine decay reaches 0.0163 and is
                  # still falling
LR       = 1e-3

# Development is cued -- the mice lived in the light.  Uncued episodes give no
# gradient at all (nothing to predict), so they are omitted as no-ops.  Gaps
# arise on their own from the FOV and the sensing radius as the animal turns
# and wanders, and those gaps are where the forward model is learned.
TRAIN_SENSE = np.array([SENSE_FULL, SENSE_FULL, 1.4, 1.0])   # test value twice
                          # over, plus two wider ones: the gaps a varying radius
                          # opens are where the forward model is learned.

# Blackouts: long uncued stretches injected into training episodes.  The only
# naturally occurring gaps are the few steps the cue spends in the blind zone,
# which teach dead reckoning over ~5 steps; the dark test asks for SESSION_T.
# GAP_HI is deliberately on the order of a session so the horizon CB is trained
# at matches the horizon it is tested at.
N_GAPS = 1
GAP_LO = 30
GAP_HI = 80      # ~30% of an episode dark, so cued experience still dominates

# -- protocol ----------------------------------------------------------------
PROTOCOL  = ["cue", "cue", "probe", "probe", "cue"]
# Fig. 1A labels each transition, and the animal is NOT removed at every one:
#
#   S1 --"Mouse removed"--> S2 --"Light extinction and object removal"--> S3
#      --(nothing)--> S4 --"Mouse removed"--> S5
#
# So the mouse stays in the arena for the whole dark period; the lights simply
# go off around it.  It is never disoriented in the dark.
REMOVE_BEFORE = [False, True, False, False, True]   # index i = before session i
# (sense_r, cue_present).  cue_none removes the cue PHYSICALLY, so it cannot be
# touched either -- otherwise "cue removed" still leaves the bottle there and
# the at_cue channel keeps re-anchoring the belief, which is the opposite of
# what that condition is supposed to test.
PROBE     = {"cue_full":  (SENSE_FULL, True),
             "cue_none":  (0.0,        False)}
SESSION_T = 100          # = SEQ_LEN, so the dark horizon at test matches the
                         # longest blackout seen in training.  Nothing pins a
                         # step to real time, so there is no reason to test at a
                         # horizon the forward model was never trained for.
N_TRIALS  = 256

# -- rate maps 
N_R, N_PHI = 5, 24        # polar bins: an arena rotation is a roll along phi
MIN_OCC    = 8
SEL_FRAC   = 0.5


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))

# Environment

class CueArena:
    """Circular arena, one cue on the wall, agent forages.  A circle is used
    rather than a square followng the paper 
    """

    def __init__(self, batch, rng, sense_r=SENSE_FULL, fov=np.deg2rad(FOV_DEG),
                 cue_present=True):
        self.B, self.rng = batch, rng
        self.sense_r, self.fov = sense_r, fov
        self.cue_present = cue_present
        self.cue_phi = rng.uniform(0, 2 * np.pi, size=batch)
        self.reset_agent()

    def reset_agent(self, consistent=True):
        """Mouse removed and reinserted between sessions.

        The experimenter puts the animal back in at roughly the same place, and
        the animal feels itself being carried.  A uniformly random pose is a
        teleport, and in the dark that is fatal in a way the experiment is not:
        the agent wakes up somewhere unknowable with no cue to work it out, so
        its belief is wrong from the first step and never recovers. 

        """
        if consistent and hasattr(self, "insert_p"):
            self.p     = self.insert_p + self.rng.normal(0, INSERT_JITTER,
                                                         size=(self.B, 2))
            self.theta = self.insert_th + self.rng.normal(0, INSERT_JITTER_TH,
                                                          size=self.B)
            self.omega = np.zeros(self.B)
            return
        self._random_agent()
        self.insert_p, self.insert_th = self.p.copy(), self.theta.copy()

    def _random_agent(self):
        r  = R_WALL * np.sqrt(self.rng.uniform(0, 1, size=self.B))
        th = self.rng.uniform(0, 2 * np.pi, size=self.B)
        self.p     = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        self.theta = self.rng.uniform(0, 2 * np.pi, size=self.B)
        self.omega = np.zeros(self.B)

    def rotate_cue(self, angle):
        self.cue_phi = (self.cue_phi + angle) % (2 * np.pi)

    @property
    def cue_xy(self):
        return R_WALL * np.stack([np.cos(self.cue_phi), np.sin(self.cue_phi)], 1)

    def step(self):
        """Egocentric action [speed, turn rate].  Integrating it needs heading
        tracking and rotation -- nonlinear, not a running sum."""
        self.omega = (TURN_MOM * self.omega
                      + (1 - TURN_MOM) * self.rng.normal(0, TURN_SD, size=self.B))
        speed = np.clip(STEP * (1 + SPEED_JITTER * self.rng.normal(0, 1, self.B)),
                        0.2 * STEP, 2.0 * STEP)

        self.theta = self.theta + self.omega
        d   = np.stack([np.cos(self.theta), np.sin(self.theta)], 1)
        nxt = self.p + speed[:, None] * d

        rad = np.linalg.norm(nxt, axis=1)
        out = rad > R_WALL
        if out.any():                       # reflect about the inward normal
            n     = nxt[out] / rad[out, None]
            d_out = d[out] - 2.0 * np.sum(d[out] * n, axis=1, keepdims=True) * n
            self.theta[out] = np.arctan2(d_out[:, 1], d_out[:, 0])
            nxt[out] = n * (R_WALL * 0.98)

        self.p = nxt
        return np.stack([speed, self.omega], 1)

    def observe(self):
        """Returns (s, g).  s is the cue seen egocentrically; g in [0,1] is how
        well it is seen -- soft in both bearing and range.

        A hard cut would make g flicker as the heading jitters near the rim, and
        that flicker is itself a heading signal.
        """
        delta = self.cue_xy - self.p
        dist  = np.linalg.norm(delta, axis=1)
        beta  = wrap(np.arctan2(delta[:, 1], delta[:, 0]) - self.theta
                     + self.rng.normal(0, BEAR_NOISE, size=self.B))

        # g is BINARY: the cue is either perceived or it is not.  The soft
        # sigmoids only place the boundary (and, through BEAR_NOISE, make
        # detection near the rim stochastic rather than knife-edged); the
        # network never sees a partial blend, so s_in is always either the real
        # observation or CB's prediction, never a mixture of the two.
        # DARK: no light, so no signal to degrade -- the cue channels are
        # substituted by CB's own prediction, and what persists is its
        # dead-reckoned estimate.  LIT: the cue is ALWAYS an observation
        # (g = 1, never substituted); its reliability is carried by the noise.
        g = np.zeros(self.B) if self.sense_r <= 0.0 else np.ones(self.B)

        inten = CUE_GAIN / (dist + CUE_SOFT) + self.rng.normal(0, INT_NOISE,
                                                               size=self.B)
        if self.sense_r > 0.0:
            # p = 1 inside the sensing radius, decaying outside it, multiplied
            # by a graded field of view so the cue fades at the rim of the
            # 300 deg field instead of blinking out.  The DISTANCE term is inert
            # SENSE_FULL is 0.90, matched to the OLM/ORM vision range, so the
            # distance term DOES engage: the cue is clean within 0.90 and fades
            # beyond it, reaching precision 0.33 at the far wall (d = 2.0).
            b0 = self.fov / 2.0
            A  = b0 ** BETA_M / (np.abs(beta) ** BETA_M + b0 ** BETA_M)
            u  = np.maximum(dist - self.sense_r, 0.0) / PLAT_LAM
            pd = np.where(dist <= self.sense_r, 1.0,
                          P_FLOOR + (1.0 - P_FLOOR) / (1.0 + u ** PLAT_N))
            p  = np.minimum(pd, P_FLOOR + (1.0 - P_FLOOR) * A)

            def _deg(x, sd):
                return (np.sqrt(p) * x
                        + np.sqrt(1.0 - p) * sd * self.rng.normal(size=x.shape))
            sb, cb_ = _deg(np.sin(beta), SD_BEARC), _deg(np.cos(beta), SD_BEARC)
            inten   = _deg(inten, SD_INTC)
            beta_s, beta_c = sb, cb_
        else:
            beta_s, beta_c = np.sin(beta), np.cos(beta)
        at_wall = (R_WALL - np.linalg.norm(self.p, axis=1)) < WALL_TOUCH
        at_cue  = (dist < CUE_TOUCH) & self.cue_present   # only if it is there

        s = np.stack([beta_s, beta_c, inten,
                      np.sin(self.theta), np.cos(self.theta),
                      at_wall.astype(float), at_cue.astype(float)], 1)
        # `lit` is the LIGHT, not the cue.  g can be 0 in a lit arena -- the
        # cue is behind the animal, or out of range -- and the two must be kept
        # apart, because heading is available whenever there is light while the
        # cue's bearing is available only when the cue itself is visible.
        lit = np.full(self.B, float(self.sense_r > 0.0))
        return s.astype(np.float32), g.astype(np.float32), lit.astype(np.float32)


def rollout(env, T):
    """(S, G, LIT, A, P).  P is true position -- ANALYSIS ONLY, never an input.

    G is cue perceptibility, LIT is whether the arena is lit at all.  They are
    returned separately because they gate different channels (run_sequence)."""
    S, G, LIT, A, P = [], [], [], [], []
    for _ in range(T):
        s, g, lit = env.observe()
        S.append(s); G.append(g); LIT.append(lit); P.append(env.p.copy())
        A.append(env.step())
    return (np.stack(S, 1), np.stack(G, 1), np.stack(LIT, 1),
            np.stack(A, 1).astype(np.float32), np.stack(P, 1).astype(np.float32))


def add_blackouts(G, LIT, rng, n_gaps=N_GAPS, lo=GAP_LO, hi=GAP_HI):
    """Force long uncued stretches into a training episode by zeroing g.

    Without this the only gaps are the few steps the cue spends in the blind
    zone, so CB learns to dead-reckon over ~5 steps -- and the dark test then
    asks it to integrate over SESSION_T, a hundredfold extrapolation it has
    never been trained for.  WT drifts to zero and the genotype contrast
    vanishes.

    A blackout still trains: the observation at the far end of it produces a
    large error if CB integrated badly, and BPTT carries that error back across
    the whole stretch.  So the gap is supervised by its own reappearance, at
    the horizon the test actually uses.  Each episode ends cued, so no blackout
    runs off the end unsupervised.

    A blackout is the LIGHTS going out, not the cue slipping behind the animal,
    so it zeroes LIT as well as G.  Otherwise heading would stay observable
    right through the gap and the model would never have to dead-reckon it.
    """
    B, T = G.shape
    G, LIT = G.copy(), LIT.copy()
    for b in range(B):
        for _ in range(n_gaps):
            L     = int(rng.integers(lo, hi))
            start = int(rng.integers(0, max(1, T - L - 1)))
            G[b, start:start + L]   = 0.0
            LIT[b, start:start + L] = 0.0
    return G, LIT


def to_t(*arrs):
    return [torch.from_numpy(a).to(DEVICE) for a in arrs]


# ============================================================================
# The two networks
# ============================================================================
class MTL(nn.Module):
    """GRU over [s_{t-1}, a_{t-1}, s_t, e_t] -> z_t, the pose belief.

    Same input convention as ECtoCB_train_MSE_12C.py and olm_mtl_cb.py, with s
    read as s_in (the observation where there is one, CB's prediction where
    there is not).  The gain is not an input: it lives in the environment, where
    it sets the substitution blend and masks the loss.
    """

    def __init__(self, use_action=True, hidden=MTL_HIDDEN, latent=LATENT_DIM):
        """MTL sees [s_{t-1}, a_{t-1}, e] -- no current observation.

        The light/dark dissociation survives this, because s_prev IS the
        previous step's substituted input (see the rollout).  Lit, that anchor
        is a real observation and one step of integration from it is easy, so a
        corrupted CB costs little.  Dark, the anchor is CB's own drifted output
        and the corruption compounds.  So the difference moves from "is CB's
        output in MTL's input" to "how good is the anchor MTL is handed".
        """
        super().__init__()
        self.use_action = use_action
        self.hidden_dim = hidden
        in_dim = S_DIM + (A_DIM if use_action else 0) + S_DIM
        self.cell    = nn.GRUCell(in_dim, hidden)
        self.readout = nn.Sequential(nn.Linear(hidden, latent), nn.Tanh())

    def init_hidden(self, B, device):
        return torch.zeros(B, self.hidden_dim, device=device)

    def step(self, s_prev, a_prev, s_in, e, h):
        # s_in is accepted and ignored, so run_sequence's call site is
        # unchanged and the substitution still reaches MTL -- one step later,
        # as s_prev.
        parts = [s_prev] + ([a_prev] if self.use_action else []) + [e]
        h = self.cell(torch.cat(parts, dim=-1), h)
        return self.readout(h), h


# ---------------------------------------------------------------------------
# MTL MUST NOT RECEIVE THE CURRENT OBSERVATION.
# "main" in model.py: MTL is denied the current observation.
# The width is the only thing that catches a regression: MTL.step accepts s_in
# and ignores it (rochefort) or does not take it at all (olm_orm), so adding it
# back to `parts` would run without error and silently train a different model
# whose checkpoints look identical from the outside.  Checked at import.
assert MTL().cell.weight_ih.shape[1] == 7 + 2 + 7, (
    "MTL input width is not s_prev + a_prev + e -- the current observation has "
    "leaked back in")


class Cerebellum(nn.Module):
    """Residual forward model: s_hat_{t+1} = s_in + delta(s_in, a_t, z_t).

    Zero-init on the output layer means an unlearned CB predicts "nothing
    changes" -- an innate default rather than noise, and the reason the mutant's
    uncued drift is coherent rather than explosive.
    """

    def __init__(self, latent=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(S_DIM + A_DIM + latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, S_DIM),
        )
        # SMALL but non-zero.  Zero-init makes an unlearned CB predict "nothing
        # changes", which is a nice innate default -- but combined with freezing
        # it sets d(s_hat)/dz identically to zero, so MTL receives NO gradient
        # and never trains at all.  The lesioned model was a randomly
        # initialised network, not a mouse with an unadaptable cerebellum.
        #
        # It also contradicts the biology: L7-PKCI Purkinje cells fire normally
        # and keep transmitting; only LTD is blocked.  A zero-output cerebellum
        # transmits nothing.  Small random weights give a cerebellum that
        # transmits an unadapted mapping -- which is the lesion we want -- and
        # keeps the gradient path to MTL open.  Identical for both genotypes, so
        # freezing is the only difference between them.
        nn.init.normal_(self.delta[-1].weight, std=CB_INIT_STD)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, s_in, a, z, noise=0.0):
        """noise > 0 is the L7-PKCI lesion: CB's prediction is unreliable.

        Applied HERE, at the source, rather than to e.  Since e = s - s_hat,
        corrupting s_hat is algebraically the same as corrupting e wherever the
        observation is real (which is what olm_mtl_cb.py does) -- but it also
        covers the dark, where s_in IS s_hat and e is zero, so corrupting e
        alone would do nothing in the one condition that matters.

        Why noise rather than freezing.  A frozen CB is broken but PREDICTABLE:
        the same (s_in, a, z) always gives the same wrong answer, and since MTL
        chooses z it can learn which z makes a broken cerebellum output what it
        wants.  Measured: a frozen CB with an untrained MTL drifts to 95x the
        signal sd over 99 steps; with a trained MTL, 0.41.  MTL cut the drift
        200-fold with CB frozen -- it was not relying on the cerebellum, it was
        inverting it, and the lesion had nothing left to break.

        Noise has no z in it, so there is no z that cancels it.  And CB still
        LEARNS, so with the cue present the mutant's representation is not
        handicapped -- the real observation overwrites the noisy prediction.
        The deficit appears only in the dark, where s_in = s_hat and there is
        nothing to overwrite it.  That condition-dependence is the 2x2.
        """
        out = s_in + self.delta(torch.cat([s_in, a, z], dim=-1))
        if not noise:
            return out, out
        # Two copies.  The CLEAN one trains CB (loss = ||s_hat - s||^2), so the
        # lesion does not corrupt the teaching signal and CB still learns a
        # proper forward model.  The NOISY one is what MTL sees -- through the
        # substitution and, automatically, through e = s - s_hat_noisy.
        #
        # Keeping them separate matters: with the noise inside the loss, CB
        # learns a worse model and MTL -- whose only training signal IS that
        # loss -- ends up with a degraded representation in EVERY condition,
        # including the light where CB is barely used.  Measured: the mutant's
        # cued coherence was 0.55-0.58 regardless of sigma, against WT's 0.67,
        # and it did not scale with sigma at all, which is what gave it away.
        return out, out + noise * torch.randn_like(out)


# ============================================================================
# The loop
# ============================================================================
# ===========================================================================
# THE OBSERVATION MODEL
# ===========================================================================
# The cerebellum never substitutes for the cue while the lights are on.  Every
# cue channel is present and degraded by how well it can currently be sensed:
#
#     s_obs = sqrt(p) * s_true + sqrt(1-p) * sigma_c * eps
#
# sigma_c is the channel's own sd, so the mixture is variance-preserving and p
# is exactly the fraction of the observed variance that is signal.
#
#     p(d)    = 1                                     d <= R
#     p(d)    = 0.05 + 0.95 / (1 + ((d - R)/0.7)^2)   d >  R
#     A(beta) = b0^12 / (|beta|^12 + b0^12),          b0 = FOV/2 = 150 deg
#     p       = min(p(d), 0.05 + 0.95 * A(beta))
#
# LAM = 0.7 and the exponent 2 put the half-precision point about 0.7 beyond
# the sensing radius and make the decay gradual rather than a cliff.  In this
# arena the DISTANCE term never engages -- SENSE_FULL is 2.2 and the cue can be
# at most 2.0 away, because it is a large wall card meant to be readable from
# anywhere -- so what degrades the cue here is angle alone.
#
# THE DARK IS DELIBERATELY DIFFERENT.  With sense_r = 0 there is no light and so
# no signal to degrade; the cue channels are substituted by CB's prediction and
# what persists is its dead-reckoned estimate, which is the memory in that
# condition.  Both conditions have memory; they differ only in whether there is
# a live signal to combine it with.
P_FLOOR  = 0.05           # nothing is ever entirely absent
PLAT_LAM = 0.7            # precision falls to half ~0.7 beyond the radius
PLAT_N   = 2.0            # gradual decay rather than a cliff
BETA_M   = 12.0           # sharpness of the angular falloff
SD_BEARC, SD_INTC = 0.707, 0.300      # scale of the cue channels


def obs_avail(g_t, lit_t, n_ch=None):
    """Per-channel observability at one timestep -> (B, S_DIM), in [0, 1].

    THE SINGLE DEFINITION.  The rollout uses it to decide which channels come
    from the senses and which from CB, and the loss uses it to decide which
    channels CB is scored on.  They must agree: a channel the animal can
    observe is a channel it can compute an error for and therefore learn from,
    and a channel it cannot observe supplies neither.  Keeping two copies of
    this rule is how they came apart -- the loss went on gating heading by cue
    perceptibility after the rollout had stopped, so CB received a live heading
    error on out-of-FOV lit steps and was never trained on it.

        cue     [0:CUE_DIM]  g    -- egocentric, needs to SEE the cue
        heading [3:5]        lit  -- allocentric, needs light only
        contact [5:7]        1    -- whiskers work in the dark
    """
    w = torch.ones(g_t.shape[0], S_DIM if n_ch is None else n_ch,
                   dtype=g_t.dtype, device=g_t.device)
    w[:, :CUE_DIM]   = g_t.unsqueeze(-1)
    w[:, HEAD_SLICE] = lit_t.unsqueeze(-1)
    return w


def run_sequence(mtl, cb, S, G, LIT, A, h0=None, s_hat0=None, collect=False,
                 cb_noise=0.0):
    """S (B,T,S_DIM)  G (B,T)  A (B,T,A_DIM).

    Returns masked prediction/target pairs plus, optionally, z and the errors.
    """
    B, T, _ = S.shape
    cb._eps = None                        # fresh OU draw per sequence
    h     = mtl.init_hidden(B, S.device) if h0 is None else h0
    s_hat = S[:, 0].clone() if s_hat0 is None else s_hat0
    s_hat_clean = s_hat.clone()   # the copy that carries the recursion
    s_prev = torch.zeros(B, S_DIM, device=S.device)
    a_prev = torch.zeros(B, A_DIM, device=S.device)

    preds, targs, masks, zs, errs, gains, hs = [], [], [], [], [], [], []

    for t in range(T):
        s_t, g_t, a_t = S[:, t], G[:, t], A[:, t]
        gc = g_t.unsqueeze(-1)

        # Substitution applies to the VISION channels only.  Contact channels
        # (at_wall, at_cue) are whisker signals -- they work in the dark, so
        # they are never replaced by CB's guess and their prediction error is
        # always live.  That is what keeps e non-zero when the cue is gone, and
        # it is what bounds the drift: without it, every channel goes open-loop
        # together and the error compounds without limit.
        # HEADING IS GATED BY THE LIGHTS, NOT BY THE CUE.  In a lit arena you
        # know which way you are facing from the walls and optic flow, whether
        # or not that one cue happens to be behind you -- so turning away from
        # it should not disorient you.  Gating heading by g made the
        # light-condition baseline depend on FOV, which it should not: at FOV
        # 300 the cue is behind the animal on ~17% of lit steps, and on every
        # one of those the model was losing its orientation and having CB
        # supply it.
        blend = obs_avail(g_t, LIT[:, t])

        # Two substituted states.  The CLEAN one carries the recursion, so CB
        # feeds on its own uncorrupted output and the belief stays bounded.  The
        # NOISY one is what MTL receives, and e inherits the noise from it.
        #
        # Propagating the noisy copy instead makes the dark an unbounded random
        # walk -- sigma compounds over the session (0.6 across 200 steps reaches
        # sd ~8.5) and training diverges, loss 517 at epoch 0.  So the lesion is
        # on what the cerebellum TRANSMITS; its internal estimate stays coherent.
        s_in       = blend * s_t + (1 - blend) * s_hat_clean
        s_in_noisy = blend * s_t + (1 - blend) * s_hat
        e_t        = blend * (s_t - s_hat)

        z, h  = mtl.step(s_prev, a_prev, s_in_noisy, e_t, h)
        s_hat_clean, s_hat = cb(s_in, a_t, z, noise=cb_noise)
        s_prev, a_prev = s_in_noisy, a_t

        if t < T - 1:
            preds.append(s_hat_clean)   # loss on the CLEAN prediction
            targs.append(S[:, t + 1])
            # No observation, no gradient -- per channel, and by the same
            # rule the rollout used.  Heading is supervised on every lit step,
            # including the ones where the cue is out of view.
            masks.append(obs_avail(G[:, t + 1], LIT[:, t + 1]))
        if collect:
            zs.append(z); errs.append(e_t); gains.append(g_t); hs.append(h)

    out = {"pred": torch.stack(preds, 1), "target": torch.stack(targs, 1),
           "mask": torch.stack(masks, 1), "h_last": h, "s_hat_last": s_hat}
    if collect:
        out["z"]    = torch.stack(zs, 1)
        out["h"]    = torch.stack(hs, 1)   # GRU state, for decoding checks
        out["err"]  = torch.stack(errs, 1)
        out["gain"] = torch.stack(gains, 1)
    return out


# HOW MUCH DO UNOBSERVED CHANNELS COUNT IN THE LOSS?
#
# Substitution and supervision are separate questions.  The GATE decides what
# CB's prediction stands in for as an INPUT -- unchanged.  This decides what CB
# is SCORED on, and the target is always the environment's ground truth, never
# CB's own guess, so the error is real and often large exactly where the animal
# could not have measured it.
#
#   0.0   only where observable.  "No observation, no gradient" -- the dark is
#         supervised indirectly, by backprop from the observation at the far
#         end of a blackout.
#   1.0   every channel equally.  MEASURED WORSE on every axis: WT dark
#         stability -41% -> -61%, and the mutant stopped matching WT in the
#         light (0.745 -> 0.702), which was the strongest match with the paper.
#   0.1   the middle ground.  CB is still asked to predict the cue and heading
#         through a blackout, but that term cannot outgrow the one it can
#         actually learn from.  Unobserved entries are 24.7% of all entries and
#         already carry 34.6% of the squared error at weight 1.
UNOBS_W = 0.0

def masked_mse(res):
    """w = mask + UNOBS_W * (1 - mask).  See UNOBS_W above."""
    m = res["mask"]
    w = m + UNOBS_W * (1.0 - m)
    return (w * (res["pred"] - res["target"]) ** 2).sum() / (w.sum() + 1e-8)


# ============================================================================
# Training
# ============================================================================
RESUME_EVERY = 2          # epochs between resume snapshots.  A snapshot is
                          # a few MB and under a second to write, while an
                          # epoch is ~75 s -- so a tighter interval costs
                          # almost nothing and caps what an unexpected
                          # restart can destroy at ~2.5 minutes.


def _resume_save(path, ep, mtl, cb, opt, sched, rng):
    """Snapshot everything needed to continue EXACTLY where we stopped.

    Both RNG streams go in: numpy drives the arena and torch drives weight
    init and the lesion noise.  Without them a resumed run continues from the
    right weights but a different data stream, which is not the same run.
    """
    torch.save({"ep": ep + 1, "mtl": mtl.state_dict(), "cb": cb.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "np_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state()}, path + ".resume")


def _resume_load(path, mtl, cb, opt, sched, rng):
    """Returns the epoch to start from; 0 if there is nothing to resume."""
    rp = path + ".resume" if path else None
    if not rp or not os.path.exists(rp):
        return 0
    ck = torch.load(rp, map_location=DEVICE)
    mtl.load_state_dict(ck["mtl"]); cb.load_state_dict(ck["cb"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    rng.bit_generator.state = ck["np_rng"]
    torch.set_rng_state(ck["torch_rng"])
    print(f"resuming from epoch {ck['ep']}  ({os.path.basename(rp)})", flush=True)
    return ck["ep"]


def train(seed=0, epochs=EPOCHS, lesion=False, fov=np.deg2rad(FOV_DEG),
          use_action=True, out=None, quiet=False, cb_noise=None):
    """lesion=True is L7-PKCI: CB's weights never update, during development,
    and MTL learns around a cerebellum that cannot adapt."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    mtl = MTL(use_action=use_action).to(DEVICE)
    cb  = Cerebellum().to(DEVICE)

    # The lesion is NOISE on CB's prediction, applied during training so it is
    # developmental -- MTL grows up with an unreliable cerebellum.  CB itself
    # still learns; only its output is corrupted.  Freezing was tried first and
    # failed: it is a deterministic handicap, and MTL simply learned to invert
    # it (see Cerebellum.forward).
    noise = (CB_NOISE if cb_noise is None else cb_noise) if lesion else 0.0
    params = list(mtl.parameters()) + list(cb.parameters())

    opt = torch.optim.Adam(params, lr=LR)
    # constant LR never settles -- Adam orbits the minimum.  Cosine decay is what
    # converts extra updates into a lower loss rather than more bouncing.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * STEPS_EP, eta_min=1e-5)
    tag = f"L7PKCI(s={noise})" if lesion else "WT"

    start_ep = _resume_load(out, mtl, cb, opt, sched, rng)
    for ep in range(start_ep, epochs):
        running = 0.0
        for _ in range(STEPS_EP):
            env = CueArena(BATCH, rng,
                           sense_r=TRAIN_SENSE[rng.integers(len(TRAIN_SENSE))],
                           fov=fov)
            S, G, LIT, A, _ = rollout(env, SEQ_LEN)
            G, LIT = add_blackouts(G, LIT, rng)   # train at the test's horizon
            loss = masked_mse(run_sequence(mtl, cb, *to_t(S, G, LIT, A),
                                           cb_noise=noise))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            running += loss.item()

        if out and (ep % RESUME_EVERY == 0 or ep == epochs - 1):
            _resume_save(out, ep, mtl, cb, opt, sched, rng)
        if not quiet and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[{tag} a={np.rad2deg(fov):.0f}] epoch {ep:4d}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  "
                  f"mse {running / STEPS_EP:.5f}", flush=True)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"mtl": mtl.state_dict(), "cb": cb.state_dict(),
                    "use_action": use_action, "cb_noise": noise}, out)
        print(f"saved -> {out}", flush=True)
        if os.path.exists(out + ".resume"):
            os.remove(out + ".resume")   # run finished; nothing to resume
    # Stamp the lesion on the returned objects, exactly as load() does.  Without
    # this a model that was TRAINED in this process (rather than loaded from
    # disk) carries no cb_noise attribute, and every downstream
    # `getattr(cb, "cb_noise", 0.0)` -- plot_bars_paper's included -- reads 0.0
    # and evaluates the mutant with a healthy cerebellum.  The lesion is
    # permanent; it must not be curable by having just finished training.
    mtl.cb_noise = cb.cb_noise = float(noise)
    return mtl, cb


def load(path):
    ck  = torch.load(path, map_location=DEVICE)
    mtl = MTL(use_action=ck.get("use_action", True)).to(DEVICE)
    cb  = Cerebellum().to(DEVICE)
    mtl.load_state_dict(ck["mtl"]); cb.load_state_dict(ck["cb"])
    mtl.eval(); cb.eval()
    # the lesion is permanent -- a model trained with noisy CB output must be
    # evaluated with it too, or the mutant is silently cured at test time
    mtl.cb_noise = cb.cb_noise = float(ck.get("cb_noise", 0.0))
    return mtl, cb


def ckpt_path(tag, fov, seed, d=None, use_action=True, cb_noise=None):
    """cb_noise MUST be in the name.  Without it every sigma in a dose-response
    sweep writes to the same file, the second sigma finds it already on disk and
    loads it instead of training, and the curve comes out flat -- which reads as
    a null result rather than as the bug it is."""
    d = CKPT_DIR if d is None else d
    sig = ("" if (cb_noise is None or not tag.lower().startswith("l7"))
           else f"_sig{cb_noise:.2f}")
    return os.path.join(d, f"rochefort_final_{tag.lower()}_s{seed}{sig}.pt")


def get_models(fov, seed, d=None, epochs=EPOCHS, force=False, use_action=True,
               cb_noise=None):
    models = {}
    for tag, lesion in (("WT", False), ("L7PKCI", True)):
        p = ckpt_path(tag, fov, seed, d, use_action, cb_noise)
        models[tag] = (train(seed=seed, epochs=epochs, lesion=lesion, fov=fov,
                             use_action=use_action, out=p, cb_noise=cb_noise)
                       if force or not os.path.exists(p) else load(p))
        models[tag][0].eval(); models[tag][1].eval()
    return models

# ============================================================================
# Is CB load-bearing?
# ============================================================================
@torch.no_grad()
def cb_accuracy(mtl, cb, seed=3, n=256, T=150, gap_at=50,
                fov=np.deg2rad(FOV_DEG), cb_noise=0.0, ks=None):
    # cb_noise must be accepted here: the lesion is permanent, so a mutant
    # evaluated without it is silently cured.
    """Split CB's prediction error into the two things it confounds.

    one_step : |s_hat - s| on a FULLY CUED sequence, so every prediction is made
               from a real observation.  This is CB's actual accuracy, and on a
               deterministic environment it should approach zero.  If it does
               not, CB is underfit.

    drift    : |s_hat - s| by steps into a blackout, where s_hat is propagating
               open-loop off its own output.  This cannot go to zero -- it IS
               the dead reckoning error -- and it is what decides whether the
               dark condition survives.

    Both are reported against the signal's own sd, so 1.0 means "no better than
    guessing the mean".
    """
    rng = np.random.default_rng(seed)
    env = CueArena(n, rng, sense_r=SENSE_FULL, fov=fov)
    S, G, LIT, A, _ = rollout(env, T)
    sd = S.std(axis=(0, 1)).mean()

    cued = run_sequence(mtl, cb, *to_t(S, G, LIT, A), collect=True,
                        cb_noise=cb_noise)
    e_c  = np.abs(cued["pred"].cpu().numpy() - cued["target"].cpu().numpy())
    m    = G[:, 1:] > 0.5                       # only steps with a real target
    one_step = float(e_c.mean(axis=2)[m].mean())

    G2, LIT2 = G.copy(), LIT.copy()
    G2[:, gap_at:] = 0.0; LIT2[:, gap_at:] = 0.0   # the lights go out at gap_at
    dark = run_sequence(mtl, cb, *to_t(S, G2, LIT2, A), collect=True,
                        cb_noise=cb_noise)
    e_d  = np.abs(dark["pred"].cpu().numpy()
                  - dark["target"].cpu().numpy()).mean(axis=(0, 2))

    # the default horizon stops at 99 steps; the session-length figure needs
    # it to reach the longest dark session it plots, so it passes its own
    ks = [k for k in (ks or (1, 5, 10, 25, 50, 99))
          if gap_at - 1 + k < len(e_d)]
    return {"one_step": one_step, "one_step_rel": one_step / sd, "sd": float(sd),
            "drift": {k: float(e_d[gap_at - 1 + k]) for k in ks},
            "drift_rel": {k: float(e_d[gap_at - 1 + k] / sd) for k in ks}}


def report_cb(mtl, cb, **kw):
    r = cb_accuracy(mtl, cb, **kw)
    print(f"\nCB one-step error (fully cued)  {r['one_step']:.4f}"
          f"   = {r['one_step_rel']:.3f} of signal sd"
          f"   {'OK' if r['one_step_rel'] < 0.05 else 'UNDERFIT'}")
    print("open-loop drift into a blackout, relative to sd:")
    print("  steps " + "".join(f"{k:>8}" for k in r["drift_rel"]))
    print("  err   " + "".join(f"{v:>8.2f}" for v in r["drift_rel"].values()))
    return r


# PER-ARENA analysis
# ============================================================================
# Every batch element is an independent arena -- one mouse.  Pooling them into a
# single rate map averages 256 animals together, and in the dark each one's
# internal frame drifts a different way, so their fields land at different
# angles and cancel.  WT then looks like it lost its map when every individual
# map was intact.  Build one map per arena and average the RESULTS, exactly as
# an experimenter analyses each animal before pooling across animals.
# Coarse bins and long sessions: a per-arena map gets only T samples, not
# 256*T, so 4x16 bins over 100 steps leaves ~1.6 samples per bin and almost
# everything is dropped.  3x8 over 300 steps gives ~12.
PA_R, PA_PHI, PA_MIN_OCC = 3, 8, 4
# A per-arena map gets T samples, not 256*T, so it is easy to be sampling- rather
# than signal-limited.  Measured: WT cued stability reads 0.425 at T=300 (12.5
# samples/bin), 0.578 at 600, 0.726 at 1200 -- still climbing.  If drift were the
# limiter it would FALL with T.  At 300 the numbers were mostly noise, and the
# 0.726 at T=1200 is where the paper's range sits (0.55-0.82).
PA_T = 1200


def arena_maps(Z, P, cue_phi, nr=PA_R, nphi=PA_PHI, min_occ=PA_MIN_OCC):
    """(B, U, nr, nphi) -- one rate map per arena per unit, cue-centred."""
    B, T, U = Z.shape
    r   = np.linalg.norm(P, axis=-1)
    phi = (np.arctan2(P[..., 1], P[..., 0]) - cue_phi[:, None]) % (2 * np.pi)
    b   = (np.clip((r / R_WALL * nr).astype(int), 0, nr - 1) * nphi
           + np.clip((phi / (2 * np.pi) * nphi).astype(int), 0, nphi - 1))

    out = np.full((B, U, nr * nphi), np.nan)
    for i in range(B):
        occ = np.bincount(b[i], minlength=nr * nphi).astype(float)
        tot = np.stack([np.bincount(b[i], weights=Z[i, :, u],
                                    minlength=nr * nphi) for u in range(U)])
        with np.errstate(invalid="ignore", divide="ignore"):
            m = tot / occ[None, :]
        m[:, occ < min_occ] = np.nan
        out[i] = m
    return out.reshape(B, U, nr, nphi)


def arena_modulation(M):
    """Mean spatial modulation (sd across bins), averaged over units and arenas.

    This is the field-strength measure -- the analogue of the paper's mean field
    rate / peak rate / spatial coherence (Fig. 1E/F), which is what drops in
    L7PKCI in the dark.  
    """
    return float(np.nanmean(np.nanstd(M.reshape(*M.shape[:2], -1), axis=2)))


def arena_stability(A, B):
    """compaes ac maps, same locations, same units. so how stable is a units fring wrt a particular area within maps. 
    Per-arena, per-unit map correlation, averaged.  Unlike modulation this
    DOES require the frame to be stable, so a drifting map scores low even with
    sharp fields."""
    v = []
    for i in range(A.shape[0]):
        for u in range(A.shape[1]):
            a, b = A[i, u].ravel(), B[i, u].ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10 or a[ok].std() < 1e-9 or b[ok].std() < 1e-9:
                continue
            v.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.nanmean(v)) if v else np.nan


@torch.no_grad()
def arena_session_test(mtl, cb, probe, seed=1234, n=N_TRIALS, T=None,
                       fov=np.deg2rad(FOV_DEG), cb_noise=0.0, lit_sense=None):
    """The protocol, analysed per arena.

    lit_sense overrides the sensing radius during the LIT sessions.  The default
    (SENSE_FULL = 2.2) exceeds the largest possible cue distance, so distance
    never degrades the cue and only angle does.  Setting it to a value that
    bites (1.4 or 1.0, both inside TRAIN_SENSE) makes the lit cue fade with
    distance as the OLM/ORM objects do -- the check on whether the light/dark
    interaction needs an arena-wide cue.
    """
    T = T or PA_T
    lit = SENSE_FULL if lit_sense is None else lit_sense
    rng = np.random.default_rng(seed)
    env = CueArena(n, rng, sense_r=lit, cue_present=True, fov=fov)
    h, s_hat, maps, Zs, Ps = None, None, [], [], []

    for i, phase in enumerate(PROTOCOL):
        env.sense_r, env.cue_present = ((lit, True) if phase == "cue"
                                        else PROBE[probe])
        if REMOVE_BEFORE[i]:            # only where Fig. 1A says "Mouse removed"
            env.reset_agent()
        S, G, LIT, A, P = rollout(env, T)
        res = run_sequence(mtl, cb, *to_t(S, G, LIT, A), h0=h, s_hat0=s_hat,
                           collect=True, cb_noise=cb_noise)
        h, s_hat = res["h_last"], res["s_hat_last"]
        Z = res["z"].cpu().numpy()
        maps.append(arena_maps(Z, P, env.cue_phi)); Zs.append(Z); Ps.append(P)

    return {"maps": maps, "cue_phi": env.cue_phi,
            "modulation": np.array([arena_modulation(m) for m in maps]),
            "stability":  np.array([arena_stability(maps[1], m) for m in maps])}


def report_arena(models, fov=np.deg2rad(FOV_DEG), seed=0):
    print(f"\n{'='*70}\nPER-ARENA analysis  (each arena = one mouse)\n{'='*70}")
    # cb_noise MUST be passed: a mutant trained with a noisy cerebellum and
    # evaluated without one is silently cured, and reads as WT.
    res = {(t, p): arena_session_test(m, c, p, fov=fov,
                                      cb_noise=getattr(c, "cb_noise", 0.0))
           for t, (m, c) in models.items() for p in ("cue_full", "cue_none")}

    print("\n-- field strength (spatial modulation) --   paper Fig.1E/F")
    print(f"   {'':20}" + "".join(f"{'S%d' % (i+1):>9}" for i in range(5)))
    for t in ("WT", "L7PKCI"):
        for p in ("cue_full", "cue_none"):
            print(f"   {t:<8}{p:<12}"
                  + "".join(f"{v:>9.4f}" for v in res[(t, p)]["modulation"]))

    print("\n-- the 2x2 --   field strength at S4")
    print(f"   {'':10}{'cue':>10}{'no cue':>10}{'change':>10}")
    for t in ("WT", "L7PKCI"):
        a = res[(t, "cue_full")]["modulation"][3]
        b = res[(t, "cue_none")]["modulation"][3]
        print(f"   {t:<10}{a:>10.4f}{b:>10.4f}{100*(b-a)/a:>9.0f}%")

    print("\n-- map stability vs S2 --   paper Fig.1J  (needs a stable frame,")
    print("   so a map that stays sharp but rotates still scores low)")
    print(f"   {'':20}" + "".join(f"{'S%d' % (i+1):>9}" for i in range(5)))
    for t in ("WT", "L7PKCI"):
        for p in ("cue_full", "cue_none"):
            print(f"   {t:<8}{p:<12}"
                  + "".join(f"{v:>9.3f}" for v in res[(t, p)]["stability"]))
    return res


def plot_arena(models, fov=np.deg2rad(FOV_DEG), path="figs/rochefort_2x2.png",
               arena=0, n_units=3):
    """Per-arena rate maps + the 2x2.

    Single-arena maps are the point: pooled across arenas the WT fields cancel,
    because each arena's belief frame differs.  Colours follow the convention
    in the paper -- dark = silent, bright = that map's peak, each map scaled to
    its own range, with the peak printed above it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    CMAP = "turbo"          # jet-like, as in the paper, but perceptually saner
    C_CUE, C_DARK = "#2f7ab8", "#c0392b"

    # cb_noise MUST be passed: a mutant trained with a noisy cerebellum and
    # evaluated without one is silently cured, and reads as WT.
    res = {(t, p): arena_session_test(m, c, p, fov=fov,
                                      cb_noise=getattr(c, "cb_noise", 0.0))
           for t, (m, c) in models.items() for p in ("cue_full", "cue_none")}
    tags = list(models)

    fig = plt.figure(figsize=(14, 3.1 * len(tags) * n_units / 2 + 4))
    nrow = len(tags) * 2 * n_units
    gs = fig.add_gridspec(nrow + 3, 6, height_ratios=[1] * nrow + [.35, 1.6, .2],
                          hspace=.5, wspace=.2)
    r_e = np.linspace(0, R_WALL, PA_R + 1)
    p_e = np.linspace(0, 2 * np.pi, PA_PHI + 1)

    row = 0
    for tag in tags:
        for probe, col in (("cue_full", C_CUE), ("cue_none", C_DARK)):
            M0 = res[(tag, probe)]["maps"][1][arena]
            units = np.argsort(-np.nan_to_num(
                np.nanstd(M0.reshape(len(M0), -1), 1)))[:n_units]
            for u in units:
                for j in range(5):
                    ax = fig.add_subplot(gs[row, j], projection="polar")
                    m = res[(tag, probe)]["maps"][j][arena][u]
                    ax.pcolormesh(p_e, r_e, np.ma.masked_invalid(m), cmap=CMAP,
                                  vmin=np.nanmin(m), vmax=np.nanmax(m),
                                  shading="flat")
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.spines["polar"].set_visible(False)
                    ax.set_title(f"{np.nanmax(m):.2f}", fontsize=6, pad=1)
                    if row == 0:
                        ax.text(.5, 1.55, f"S{j+1} ({PROTOCOL[j]})", fontsize=9,
                                ha="center", transform=ax.transAxes)
                    if j == 0:
                        ax.text(-.5, .5, f"{tag}\n{probe}\nunit {u}",
                                transform=ax.transAxes, fontsize=7, color=col,
                                ha="center", va="center", rotation=90)
                ax = fig.add_subplot(gs[row, 5])
                ax.plot(range(1, 6), res[(tag, probe)]["modulation"], "o-",
                        color=col, ms=3, lw=1.4)
                ax.set_ylim(0, .12); ax.set_xticks(range(1, 6))
                ax.tick_params(labelsize=6)
                if row == 0:
                    ax.set_title("field strength", fontsize=8)
                for sp in ("top", "right"): ax.spines[sp].set_visible(False)
                row += 1

    # colourbar: what the colours mean
    cax = fig.add_subplot(gs[nrow, 1:5])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["silent", "peak"])
    cb.set_label("activity, each map scaled to its own range "
                 "(number above each map = its peak)", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # the 2x2
    ax = fig.add_subplot(gs[nrow + 1, :])
    w, x = .35, np.arange(2)
    for k, tag in enumerate(tags):
        vals = [arena_coherence(res[(tag, p)]["maps"][3])
                for p in ("cue_full", "cue_none")]
        c = C_CUE if tag == "WT" else C_DARK
        ax.bar(x + k * w, vals, w, color=c, label=tag)
        for xi, v in zip(x + k * w, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w * (len(tags) - 1) / 2)
    ax.set_xticklabels(["cue available", "cue removed"])
    ax.set_ylabel("spatial coherence at S4")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle("Per-arena rate maps (one arena = one mouse). "
                 "S3/S4 are the manipulated sessions.", fontsize=11)
    fig.subplots_adjust(left=.07, right=.98, top=.95, bottom=.04)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")
    return res



def arena_peak(M):
    """Peak rate (Fig. 1E)."""
    return float(np.nanmean(np.nanmax(M.reshape(*M.shape[:2], -1), axis=2)))


def arena_field_rate(M):
    """Mean rate INSIDE the field (Fig. 1E), the field taken as bins above
    mean + 1 sd of that unit's map."""
    F, out = M.reshape(*M.shape[:2], -1), []
    for i in range(F.shape[0]):
        for u in range(F.shape[1]):
            m = F[i, u][np.isfinite(F[i, u])]
            if len(m) < 6:
                continue
            f = m[m >= m.mean() + m.std()]
            if len(f):
                out.append(f.mean())
    return float(np.nanmean(out)) if out else np.nan


def arena_coherence(M):
    """Spatial coherence (Fig. 1F) - correlate each bin against
    the mean of its 8 neighbours.  A real field is a contiguous blob, so bin and
    surround agree and coherence is high.
    """
    out = []
    for i in range(M.shape[0]):
        for u in range(M.shape[1]):
            m = M[i, u]
            acc, cnt = np.zeros_like(m), np.zeros_like(m)
            for dr in (-1, 0, 1):
                for dp in (-1, 0, 1):
                    if dr == dp == 0:
                        continue
                    sh = np.roll(m, dp, axis=1)              # phi wraps
                    if dr != 0:
                        sh = np.roll(sh, dr, axis=0)
                        sh[0 if dr == 1 else -1] = np.nan    # r does not
                    ok = np.isfinite(sh); acc[ok] += sh[ok]; cnt[ok] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                nb = acc / cnt
            ok = np.isfinite(m) & np.isfinite(nb)
            if ok.sum() < 8 or m[ok].std() < 1e-9 or nb[ok].std() < 1e-9:
                continue
            out.append(np.corrcoef(m[ok], nb[ok])[0, 1])
    return float(np.nanmean(out)) if out else np.nan


def report_paper_measures(models, fov=np.deg2rad(FOV_DEG)):
    """Fig. 1E/F/I/J, per arena."""
    print(f"\n{'='*74}\nthe paper's measures, per arena\n{'='*74}")
    print("paper: WT flat on every measure; L7PKCI drops in the dark and\n"
          "       recovers at S5 (coherence 0.79 -> 0.60)\n")
    print(f"{'':8}{'':11}{'peak 1E':>10}{'field 1E':>10}{'coher 1F':>10}"
          f"{'stab 1J':>10}")
    for t in ("WT", "L7PKCI"):
        for p in ("cue_full", "cue_none"):
            r = arena_session_test(*models[t], p, fov=fov,
                                   cb_noise=getattr(models[t][1], "cb_noise", 0.0))
            M = r["maps"][3]
            print(f"{t:<8}{p:<11}{arena_peak(M):>10.4f}"
                  f"{arena_field_rate(M):>10.4f}{arena_coherence(M):>10.3f}"
                  f"{r['stability'][3]:>10.3f}")


# Per-arena values (for error bars and statistics)
# ============================================================================
def per_arena_values(M, kind):
    """One value per arena, so arenas can serve as the unit of analysis --
    they are our 'animals'.  Averaging them away first makes error bars
    impossible."""
    out = []
    for i in range(M.shape[0]):
        v = arena_coherence(M[i:i + 1]) if kind == "coherence" else \
            arena_field_rate(M[i:i + 1])
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def per_arena_stability(A, B):
    out = []
    for i in range(A.shape[0]):
        v = arena_stability(A[i:i + 1], B[i:i + 1])
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def plot_bars_paper(models, path="figs/rochefort_bars_paper.png",
                    fov=np.deg2rad(FOV_DEG), n=40):
    """Fig. 1E/F/J layout exactly: open bars = WT, filled = L7PKCI, grouped by
    light and dark, SEM error bars, asterisks from a two-sample t-test.

    Arenas are the unit of analysis -- one arena is one animal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    tags = list(models)
    vals = {}
    for t, (m, c) in models.items():
        nz = getattr(c, "cb_noise", 0.0)
        for p in ("cue_full", "cue_none"):
            r = arena_session_test(m, c, p, n=n, fov=fov, cb_noise=nz)
            vals[(t, p, "field")]     = per_arena_values(r["maps"][3], "field")
            vals[(t, p, "coherence")] = per_arena_values(r["maps"][3], "coherence")
            vals[(t, p, "stability")] = per_arena_stability(r["maps"][1],
                                                            r["maps"][3])

    def stars(a, b):
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        return ("***" if p < .001 else "**" if p < .01 else
                "*" if p < .05 else "n.s."), p

    # FIELD RATE is normalised within each seed to its own light value.  Raw
    # values are not comparable ACROSS sigma: those are separately trained
    # networks, and the mutant's units simply run hotter (mean|z| 0.124 -> 0.150
    # at sigma 0.5, i.e. +21%), so a raw light line rises ~18% while spatial
    # tuning improves only ~4%.  Normalising cancels that gain, and it also
    # uncovers an effect the raw panel hid: raw dark looked flat across sigma
    # (0.097 -> 0.101) when the cost of cue removal is in fact growing
    # (0.874 -> 0.765).  Coherence and stability are correlations, immune to
    # gain, and are left absolute.
    # (The source script's copy of this function had the dose-response
    # normalisation pasted in verbatim here, referring to `sigmas` and `acc`,
    # neither of which exists in this scope -- it raises NameError.  Rewritten
    # against `vals`, which is what this function actually builds.)
    #
    # Both genotypes are divided by the SAME number, WT's light mean.  Dividing
    # each by its own would force both light bars to 1.0 and so manufacture the
    # paper's "light properties are unaffected" result instead of testing it.
    base = vals[(tags[0], "cue_full", "field")].mean()
    for t in tags:
        for p in ("cue_full", "cue_none"):
            vals[(t, p, "field")] = vals[(t, p, "field")] / base

    panels = [("field", "mean field rate\nrelative to light"),
              ("coherence", "spatial coherence"),
              ("stability", "place field stability")]
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * 3, 4.2))
    x = np.arange(2)          # light, dark
    w = 0.34

    for ax, (k, title) in zip(axes, panels):
        for i, t in enumerate(tags):
            mu = [vals[(t, p, k)].mean() for p in ("cue_full", "cue_none")]
            se = [vals[(t, p, k)].std() / np.sqrt(len(vals[(t, p, k)]))
                  for p in ("cue_full", "cue_none")]
            ax.bar(x + i * w, mu, w, yerr=se, capsize=3,
                   facecolor="white" if i == 0 else "black",
                   edgecolor="black", lw=1.1,
                   label=t if k == "field" else "")
            # light vs dark, within genotype
            s, _ = stars(vals[(t, "cue_full", k)], vals[(t, "cue_none", k)])
            y = max(mu[0] + se[0], mu[1] + se[1]) * 1.06
            ax.plot([x[0] + i * w, x[1] + i * w], [y, y], "k-", lw=.8)
            ax.text(x.mean() + i * w, y, s, ha="center", va="bottom",
                    fontsize=9)

        ax.set_xticks(x + w / 2); ax.set_xticklabels(["light", "dark"])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, None)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[0].set_ylabel(f"per arena (n={n}), session 4   mean $\\pm$ SEM")

    fig.suptitle("Suppression of external cues affects the mutant specifically",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")

    print(f"\n{'measure':<12}{'genotype':<9}{'light':>16}{'dark':>16}"
          f"{'drop':>8}  light-vs-dark")
    for k, _ in panels:
        for t in tags:
            a, b = vals[(t, "cue_full", k)], vals[(t, "cue_none", k)]
            s, p = stars(a, b)
            print(f"{k:<12}{t:<9}{a.mean():>9.3f}+-{a.std()/np.sqrt(len(a)):<5.3f}"
                  f"{b.mean():>9.3f}+-{b.std()/np.sqrt(len(b)):<5.3f}"
                  f"{100*(b.mean()-a.mean())/a.mean():>7.0f}%  {s} (p={p:.1e})")
    return vals


# ============================================================================
# Controls: does the effect depend on how long the dark lasts, and on sigma?
# ============================================================================
def plot_session_length(models, path="figs/stability_vs_session.png",
                        Ts=(300, 600, 1200, 2400), n=40,
                        fov=np.deg2rad(FOV_DEG)):
    """LEFT: stability vs dark session length.  RIGHT: CB's open-loop error vs
    steps unanchored -- the drift that CAUSES the stability loss.

    The right panel used to show "cost of removing the cue, %", which is just
    the left panel's two curves divided by each other and adds nothing.  CB
    error is an independent measurement of the mechanism: in the dark s_in IS
    s_hat, so CB predicts from its own output with e = 0 on the visual
    channels and nothing to correct it.  The error compounds, and it compounds
    faster in the mutant.  Wall contact pins radius but says NOTHING about
    angle, so what accumulates is a rotation of the whole map -- which is why
    coherence barely moves while stability collapses.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, drift = {}, {}
    for tag, (m, c) in models.items():
        nz = getattr(c, "cb_noise", 0.0)
        for p_ in ("cue_full", "cue_none"):
            res[(tag, p_)] = [arena_session_test(m, c, p_, n=n, T=T, fov=fov,
                                                 cb_noise=nz)["stability"][3]
                              for T in Ts]
        # open-loop drift, out to the longest session on the left panel
        drift[tag] = cb_accuracy(m, c, n=128, T=max(Ts) + 80, gap_at=50,
                                 fov=fov, cb_noise=nz,
                                 ks=(10, 50, 150, 300, 600, 1200, 2400)
                                 )["drift_rel"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, col in (("WT", "#2f7ab8"), ("L7PKCI", "#c0392b")):
        axes[0].plot(Ts, res[(tag, "cue_full")], "s--", color=col, ms=5,
                     label=f"{tag} light")
        axes[0].plot(Ts, res[(tag, "cue_none")], "o-", color=col, ms=5,
                     label=f"{tag} dark")
        d = drift[tag]
        axes[1].plot(list(d.keys()), list(d.values()), "o-", color=col, ms=5,
                     label=tag)

    axes[0].set_ylabel("place field stability  (S2 vs S4)")
    axes[0].set_xlabel("dark session length (steps)")
    axes[0].set_xscale("log"); axes[0].set_xticks(Ts)
    axes[0].set_xticklabels([str(t) for t in Ts])
    axes[1].set_ylabel("CB open-loop error / signal sd")
    axes[1].set_xlabel("steps since the last real observation")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle("Stability falls with time in the dark (left) because CB's "
                 "belief drifts open-loop (right)\n"
                 "WT drifts slowly; the mutant is already severe -- the "
                 "genotypes differ in RATE, not in kind", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")

    print(f"\n{'T':>7}{'WT_lt':>10}{'WT_dk':>10}{'mut_lt':>10}{'mut_dk':>10}")
    for i, T in enumerate(Ts):
        print(f"{T:>7}" + "".join(f"{res[(t,p_)][i]:>10.3f}"
              for t in ("WT", "L7PKCI") for p_ in ("cue_full", "cue_none")))
    print(f"\n{'steps':>7}" + "".join(f"{k:>8}" for k in drift["WT"]))
    for tag in ("WT", "L7PKCI"):
        print(f"{tag:>7}" + "".join(f"{v:>8.2f}" for v in drift[tag].values()))

def plot_dose_response(paths, path="figs/dose_response.png",
                       fov=np.deg2rad(FOV_DEG), n=40):
    """Every measure against lesion severity, averaged over seeds.

    WT is not a separate genotype here -- it is sigma = 0, the zero point of the
    same sweep.  That makes the prediction explicit and falsifiable: the LIGHT
    line should be flat across sigma (the cue overwrites CB's signal, so
    corrupting it should cost nothing), while the DARK line falls.  A light line
    that slopes means the lesion is doing something it should not.

    paths : {sigma: [checkpoint, ...]}   one entry per seed
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = sorted(paths)
    acc = {(s, p, k): [] for s in sigmas
           for p in ("cue_full", "cue_none")
           for k in ("field", "coherence", "stability")}

    for s in sigmas:
        for ck in paths[s]:
            m, c = load(ck)
            for p in ("cue_full", "cue_none"):
                r = arena_session_test(m, c, p, n=n, fov=fov, cb_noise=s)
                M = r["maps"][3]
                acc[(s, p, "field")].append(arena_field_rate(M))
                acc[(s, p, "coherence")].append(arena_coherence(M))
                acc[(s, p, "stability")].append(r["stability"][3])

    # FIELD RATE is normalised within each seed to its own light value.  Raw
    # values are not comparable ACROSS sigma: those are separately trained
    # networks, and the mutant's units simply run hotter (mean|z| 0.124 -> 0.150
    # at sigma 0.5, i.e. +21%), so a raw light line rises ~18% while spatial
    # tuning improves only ~4%.  Normalising cancels that gain, and it also
    # uncovers an effect the raw panel hid: raw dark looked flat across sigma
    # (0.097 -> 0.101) when the cost of cue removal is in fact growing
    # (0.874 -> 0.765).  Coherence and stability are correlations, immune to
    # gain, and are left absolute.
    for s in sigmas:
        base = list(acc[(s, "cue_full", "field")])          # per seed
        for p in ("cue_full", "cue_none"):
            acc[(s, p, "field")] = [v / b for v, b in
                                    zip(acc[(s, p, "field")], base)]

    panels = [("field", "mean field rate\nrelative to light"),
              ("coherence", "spatial coherence"),
              ("stability", "place field stability")]
    fig, axes = plt.subplots(1, 3, figsize=(4.3 * 3, 4))
    for ax, (k, title) in zip(axes, panels):
        for p, col, lab in (("cue_full", "#2f7ab8", "light"),
                            ("cue_none", "#c0392b", "dark")):
            mu = [np.mean(acc[(s, p, k)]) for s in sigmas]
            se = [np.std(acc[(s, p, k)]) / max(1, np.sqrt(len(acc[(s, p, k)])))
                  for s in sigmas]
            ax.errorbar(sigmas, mu, yerr=se, fmt="o-", color=col, capsize=3,
                        ms=5, label=lab if k == "field" else "")
        ax.set_xlabel("lesion severity  σ   (0 = WT)")
        ax.set_title(title, fontsize=10)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    n_seeds = len(next(iter(paths.values())))
    axes[0].set_ylabel(f"session 4, mean $\\pm$ SEM over {n_seeds} seeds")

    fig.suptitle("Every measure at session 4 against lesion severity  "
                 "(sigma = 0 is WT, the same sweep)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")

    print(f"\n{'sigma':>7}" + "".join(f"{c:>22}" for c, _ in panels))
    for s in sigmas:
        row = "".join(
            f"{np.mean(acc[(s,'cue_full',k)]):>10.3f}"
            f"{np.mean(acc[(s,'cue_none',k)]):>10.3f}  " for k, _ in panels)
        print(f"{s:>7.1f}{row}")
    return acc

# ============================================================================
# Light-on vs light-off map similarity, measured four ways
# ============================================================================
# Ported from paper_code/selectivity_change.py.  The point is robustness: mean
# field rate is invariant to DISPLACEMENT by construction, so a map that stays
# perfectly sharp and merely moves scores unchanged on it.  These four measures
# compare the light map with the dark map directly, and they disagree with each
# other about almost everything except the genotype ordering -- which is what
# makes the agreement worth having.
S_LIGHT, S_DARK = 1, 3          # S2 vs S4, the severe contrast
MIN_BINS = 20                   # a Cartesian map has ~380 in-circle pixels

PIX      = 22        # Cartesian grid across the arena
SMOOTH_S = 1.3       # Gaussian sd in pixels.  Real place-field maps are always
                     # smoothed -- a per-arena map has only T samples over ~380
                     # in-circle pixels, so unsmoothed it is speckle.


def _gauss_smooth(m, s=SMOOTH_S):
    """Gaussian smoothing that ignores NaN (unvisited pixels)."""
    from math import ceil
    r = int(ceil(3 * s))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x ** 2 + y ** 2) / (2 * s * s))
    ok = np.isfinite(m).astype(float)
    v = np.where(np.isfinite(m), m, 0.0)
    num = np.zeros_like(v); den = np.zeros_like(v)
    H, W = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = k[dy + r, dx + r]
            num += w * np.roll(np.roll(v, dy, 0), dx, 1)
            den += w * np.roll(np.roll(ok, dy, 0), dx, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out


def cartesian_maps(Z, P, cue_phi, pix=PIX, reduce="mean"):
    """(B, U, pix, pix) cue-centred Cartesian rate maps, smoothed.

    Cue-centred means each arena is rotated so its cue sits at the same place,
    which is the only frame in which maps from different arenas are comparable.

    reduce="mean"   average over every visit to a bin -- the standard map, and
                    what every population measure uses.
    reduce="first"  take only the FIRST visit to each bin.  A drifting belief
                    makes the same place look different on the tenth pass than
                    on the first, and averaging blends the two.  A first-visit
                    map is what the cell looked like before that bin's estimate
                    had a chance to move, so comparing it with the mean map
                    separates "the field is genuinely broad" from "the field is
                    sharp but was recorded at several different offsets".
                    Noisier by construction: one sample per bin, no averaging.
    """
    B, T, U = Z.shape
    ca, sa = np.cos(-cue_phi)[:, None], np.sin(-cue_phi)[:, None]
    x = P[..., 0] * ca - P[..., 1] * sa            # rotate cue to a fixed bearing
    y = P[..., 0] * sa + P[..., 1] * ca
    xi = np.clip(((x + R_WALL) / (2 * R_WALL) * pix).astype(int), 0, pix - 1)
    yi = np.clip(((y + R_WALL) / (2 * R_WALL) * pix).astype(int), 0, pix - 1)
    b = yi * pix + xi

    out = np.full((B, U, pix, pix), np.nan)
    for i in range(B):
        occ = np.bincount(b[i], minlength=pix * pix).astype(float)
        if reduce == "first":
            # first index at which each bin was entered; np.unique returns the
            # FIRST occurrence for each value when the input is in time order
            m = np.full((U, pix * pix), np.nan)
            bins, first = np.unique(b[i], return_index=True)
            m[:, bins] = Z[i, first, :].T
        else:
            tot = np.stack([np.bincount(b[i], weights=Z[i, :, u],
                                        minlength=pix * pix) for u in range(U)])
            with np.errstate(invalid="ignore", divide="ignore"):
                m = tot / occ[None, :]
        m[:, occ < 1] = np.nan
        for u in range(U):
            out[i, u] = _gauss_smooth(m[u].reshape(pix, pix))

    gx = (np.arange(pix) + .5) / pix * 2 - 1
    mask = (gx[None, :] ** 2 + gx[:, None] ** 2) > 1.0     # outside the arena
    out[:, :, mask] = np.nan
    return out


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na < 1e-9 or nb < 1e-9 else float(a @ b / (na * nb))


def _per_unit(A, B, fn, min_bins=MIN_BINS):
    """fn(a, b) over every (arena, unit) pair, on the bins finite in both."""
    out = []
    for i in range(A.shape[0]):
        for u in range(A.shape[1]):
            a, b = A[i, u].ravel(), B[i, u].ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < min_bins:
                continue
            v = fn(a[ok], b[ok])
            if v is not None and np.isfinite(v):
                out.append(v)
    return float(np.mean(out)) if out else np.nan


def cos_raw(A, B, **kw):
    """Uncentred dot product."""
    return _per_unit(A, B, _cos, **kw)


def pearson(A, B, **kw):
    """Centred dot product -- what arena_stability computes, on this grid."""
    return _per_unit(A, B, lambda a, b: _cos(a - a.mean(), b - b.mean()), **kw)


def cos_rect(A, B, **kw):
    """Rectify BOTH maps at the LIGHT map's own mean, then cosine.

    Using the light map's mean for both is the point: the field region is then
    defined ONCE, from the light session, rather than re-found in each map --
    which is exactly the property mean field rate lacks.  After the relu the
    maps are sparse and positive, so this evaluates similarity in the regime
    field rate assumes.
    """
    def f(a, b):
        mu = a.mean()
        return _cos(np.maximum(a - mu, 0), np.maximum(b - mu, 0))
    return _per_unit(A, B, f, **kw)


def pv_corr(A, B, min_units=8):
    """Population-vector correlation: per spatial bin, correlate the activity
    vector ACROSS units between light and dark, then average over bins.  No
    per-unit field definition anywhere, so neither the sparsity nor the sign
    objection applies to it."""
    out = []
    for i in range(A.shape[0]):
        a = A[i].reshape(A.shape[1], -1); b = B[i].reshape(B.shape[1], -1)
        for k in range(a.shape[1]):
            ok = np.isfinite(a[:, k]) & np.isfinite(b[:, k])
            if ok.sum() < min_units:
                continue
            x, y = a[ok, k], b[ok, k]
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            out.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(out)) if out else np.nan


SIM = [("cos_raw", cos_raw), ("pearson", pearson),
       ("cos_rect", cos_rect), ("pv_corr", pv_corr)]


@torch.no_grad()
def collect_cart(mtl, cb, probe, seed=1234, n=64, T=None,
                 fov=np.deg2rad(FOV_DEG), cb_noise=0.0, reduce="mean"):
    """Cartesian maps, one per session.  Same protocol as arena_session_test:
    belief carried across sessions, agent removed only where Fig. 1A says so."""
    T = T or PA_T
    rng = np.random.default_rng(seed)
    env = CueArena(n, rng, sense_r=SENSE_FULL, cue_present=True, fov=fov)
    h, s_hat, car = None, None, []
    for i, phase in enumerate(PROTOCOL):
        env.sense_r, env.cue_present = ((SENSE_FULL, True) if phase == "cue"
                                        else PROBE[probe])
        if REMOVE_BEFORE[i]:
            env.reset_agent()
        S, G, LIT, A, P = rollout(env, T)
        res = run_sequence(mtl, cb, *to_t(S, G, LIT, A), h0=h, s_hat0=s_hat,
                           collect=True, cb_noise=cb_noise)
        h, s_hat = res["h_last"], res["s_hat_last"]
        car.append(cartesian_maps(res["z"].cpu().numpy(), P, env.cue_phi,
                                  reduce=reduce))
    return car


def plot_map_similarity(seeds=(0,), sigma=CB_NOISE, fov=np.deg2rad(FOV_DEG),
                        n=64, path=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = {}
    for seed in seeds:
        for tag, nz in (("WT", 0.0), ("L7PKCI", sigma)):
            pth = ckpt_path(tag, fov, seed,
                            cb_noise=(None if tag == "WT" else sigma))
            if not os.path.exists(pth):
                print(f"   missing {os.path.basename(pth)}"); return
            m, c = load(pth)
            for probe in ("cue_full", "cue_none"):
                car = collect_cart(m, c, probe, n=n, fov=fov, cb_noise=nz)
                A, B = car[S_LIGHT], car[S_DARK]
                d = acc.setdefault((tag, probe), {})
                for name, fn in SIM:
                    d.setdefault(name, []).append(fn(A, B))
        print(f"   seed {seed} done", flush=True)

    ns = len(seeds)
    sem = lambda v: (np.nanstd(v) / np.sqrt(ns)) if ns > 1 else 0.0
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    print(f"\n{'measure':<11}{'genotype':<9}{'light':>9}{'dark':>9}{'cost':>9}")
    for ax, (k, _) in zip(axes, SIM):
        x = np.arange(2)
        for j, probe in enumerate(("cue_full", "cue_none")):
            vals = [acc[(t, probe)][k] for t in ("WT", "L7PKCI")]
            ax.bar(x + j * 0.38 - 0.19, [np.mean(v) for v in vals], 0.36,
                   yerr=[sem(v) for v in vals], capsize=3,
                   color=["#2f7ab8", "#c0392b"],
                   alpha=1.0 if probe == "cue_full" else 0.45,
                   edgecolor="#333", lw=.7,
                   label="light" if probe == "cue_full" else "dark")
        for t in ("WT", "L7PKCI"):
            lt = np.mean(acc[(t, "cue_full")][k])
            dk = np.mean(acc[(t, "cue_none")][k])
            print(f"{k:<11}{t:<9}{lt:>9.3f}{dk:>9.3f}{100*(dk-lt)/lt:>8.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(["WT", "L7PKCI"])
        ax.set_title(k, fontsize=10)
        ax.set_ylabel("light-on vs light-off similarity")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Light-on vs light-off map similarity (S2 vs S4), measured "
                 "directly: the mutant loses more on every variant\n"
                 "(solid = both sessions lit, the sampling ceiling; faded = "
                 "second session dark)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")


@torch.no_grad()
def paper_ratemaps(mtl, cb, probe="cue_none", n=32, T=1200,
                   fov=np.deg2rad(FOV_DEG), cb_noise=0.0, seed=1234,
                   reduce="mean"):
    """Cartesian maps for the Fig. 1C/D layout, one entry per session."""
    return collect_cart(mtl, cb, probe, seed=seed, n=n, T=T, fov=fov,
                        cb_noise=cb_noise, reduce=reduce)


def plot_ratemaps_2x2(seed=0, sigma=CB_NOISE, fov=np.deg2rad(FOV_DEG),
                      arena=0, path=None, shared_scale=False, reduce="mean"):
    """Four rows -- WT light, WT dark, mutant light, mutant dark -- one figure.

    Splitting light and dark into separate files hid the effect: the protocol is
    cue/cue/probe/probe/cue, so S1, S2 and S5 are the SAME cued run in both and
    only S3/S4 differ.  A reader had to flip between images to do the
    subtraction.  With each genotype's dark row directly under its own light
    row, the mutant's S4 collapse reads off the page and WT's two near-identical
    rows are the built-in control.

    ONE SCALE PER GENOTYPE, shared across its two rows -- not one global scale.
    A global scale is tempting for comparability and wrong here: the example
    cells differ ~2.8x in peak while the POPULATION field rate differs only
    ~1.19x.  "Most modulated cell" takes each network's extreme, and those tails
    do not scale together, so a global scale makes an unrepresentative ratio the
    most salient thing on the page and flattens WT into one colour.  The two
    cells are different units of different networks; there is no cross-genotype
    comparison to preserve.  Per genotype keeps the only real comparison --
    light vs dark within a genotype -- and still shows the S4 collapse.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    CMAP = "turbo"

    # The mutant's noise is drawn fresh on every call (torch.randn_like); the
    # seed inside the protocol fixes only the trajectory.  Without this the
    # example cell is not reproducible -- the same model gave an S4 peak of 0.17
    # on one run and 0.46 on the next.  The SIZE of the effect must still come
    # from the population measures, never from this example.
    torch.manual_seed(0)

    rows = []
    for tag in ("WT", "L7PKCI"):
        pth = ckpt_path(tag, fov, seed,
                        cb_noise=(None if tag == "WT" else sigma))
        if not os.path.exists(pth):
            print(f"   missing {os.path.basename(pth)}"); return
        m, c = load(pth)
        nz = getattr(c, "cb_noise", 0.0)
        M = {pr: paper_ratemaps(m, c, pr, fov=fov, cb_noise=nz, reduce=reduce)
             for pr in ("cue_full", "cue_none")}
        # the most spatially modulated cell of the CUED reference session, and
        # the same cell for both rows so they are comparable
        ref = M["cue_full"][1][arena]
        unit = int(np.argmax(np.nan_to_num(
            np.nanstd(ref.reshape(len(ref), -1), 1))))
        # "cue kept" / "cue REMOVED" describes what fills the S3-S4 PROBE
        # WINDOW for that row, not the individual session.
        for pr, lab in (("cue_full", "cue kept"), ("cue_none", "cue REMOVED")):
            rows.append((tag, lab, unit, [M[pr][j][arena][unit] for j in range(5)]))
        print(f"   {tag}: cell {unit}, cb_noise={nz}", flush=True)

    lim = {}
    for tag, lab, unit, ms in rows:
        lo, hi = float(np.nanmin(ms)), float(np.nanmax(ms))
        pv = lim.get(tag, (lo, hi))
        lim[tag] = (min(pv[0], lo), max(pv[1], hi))
    if shared_scale:
        allm = [m for _, _, _, ms in rows for m in ms]
        g = (float(np.nanmin(allm)), float(np.nanmax(allm)))
        lim = {t: g for t in lim}
    for t, (lo, hi) in lim.items():
        print(f"   {t} colour scale: [{lo:.3f}, {hi:.3f}]")

    fig, axes = plt.subplots(len(rows), 5, figsize=(1.75 * 5, 1.8 * len(rows)))
    for i, (tag, lab, unit, ms) in enumerate(rows):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            vmin, vmax = lim[tag]
            ax.imshow(np.ma.masked_invalid(m), cmap=CMAP, origin="lower",
                      vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            removed = (lab == "cue REMOVED") and j in (2, 3)
            for sp in ax.spines.values():
                sp.set_visible(removed)      # outline only the panels that
                sp.set_color("#c0392b")      # actually had the cue taken away
                sp.set_linewidth(2.0)
            ax.set_title(f"{np.nanmax(m):.2f}", fontsize=7, pad=2)
            if i == 0:
                ax.text(.5, 1.45, f"S{j+1}", fontsize=9, ha="center",
                        transform=ax.transAxes)
            if j == 0:
                ax.text(-.20, .5, f"{tag}\n{lab}\ncell {unit}",
                        transform=ax.transAxes, fontsize=8, ha="right",
                        va="center",
                        color="#c0392b" if "REMOVED" in lab else "#2f7ab8")

    cax = fig.add_axes([.35, .065, .32, .012])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["silent", "peak"])
    cb.set_label("one scale per genotype, shared across its two rows  "
                 "(peak of each map printed above; cells are NOT comparable "
                 "across genotypes)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Red outline = the cue was removed for that session (S3-S4 "
                 "only).  Every other panel is cued.\n"
                 "The mutant remaps at S4 and recovers at S5.  WT also blurs "
                 "at S3-S4 -- a known mismatch: the paper's WT is flat",
                 fontsize=11)
    fig.subplots_adjust(left=.11, right=.98, top=.84, bottom=.09,
                        hspace=.35, wspace=.06)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")


# ============================================================================
# Rotation-corrected stability: how much of the loss is JUST a turned map?
# ============================================================================
# arena_stability compares two maps bin for bin, so a map that stays perfectly
# sharp and merely rotates scores badly -- which is the right behaviour for the
# paper's Fig. 1J, and is exactly what we think happens in the dark (wall
# contact pins r and says nothing about phi).
#
# But it conflates two things we would like to separate: the map DEGRADED, and
# the map MOVED.  Since the maps are polar and cue-centred, a rotation of the
# animal's internal frame is a roll along phi, so we can search over rolls and
# take the best:
#
#     raw        = corr(A, B)                      what Fig. 1J reports
#     corrected  = max over k of corr(A, roll(B, k))
#     offset     = the k that achieves it
#
# corrected >> raw means the field survived and the frame turned.  corrected
# ~ raw means the map genuinely fell apart.  The recovered offset should differ
# from arena to arena, since each one's frame drifts independently -- that
# independence is also why pooling arenas destroys the signal.
ROT_PHI = 24          # finer than PA_PHI=8: a 45 deg quantum is too coarse to
                      # call a rotation.  3 x 24 = 72 bins, so T=1200 leaves
                      # ~17 samples per bin.


def rotation_stability(A, B):
    """(raw, corrected, |offset| in degrees) -- per arena, then averaged."""
    nphi = A.shape[-1]
    raw = arena_stability(A, B)
    best, off = [], []
    for i in range(A.shape[0]):
        sc = [arena_stability(A[i:i+1], np.roll(B[i:i+1], k, axis=-1))
              for k in range(nphi)]
        sc = np.array(sc)
        if not np.isfinite(sc).any():
            continue
        k = int(np.nanargmax(sc))
        best.append(sc[k])
        off.append(min(k, nphi - k) * 360.0 / nphi)      # signed-free magnitude
    return raw, float(np.mean(best)), float(np.mean(off))


@torch.no_grad()
def report_rotation(models, fov=np.deg2rad(FOV_DEG), n=48, T=None):
    print(f"\n{'='*74}\nROTATION-CORRECTED STABILITY  (S2 vs S4, {n} arenas)\n{'='*74}")
    print(f"{'':20}{'raw':>9}{'rot-corrected':>16}{'recovered turn':>17}"
          f"{'  (shuffled-arena null)':>24}")
    out = {}
    for tag, (m, c) in models.items():
        nz = getattr(c, "cb_noise", 0.0)
        for pr in ("cue_full", "cue_none"):
            rng = np.random.default_rng(1234)
            env = CueArena(n, rng, sense_r=SENSE_FULL, cue_present=True, fov=fov)
            h, s_hat, maps = None, None, []
            for i, phase in enumerate(PROTOCOL):
                env.sense_r, env.cue_present = ((SENSE_FULL, True) if phase == "cue"
                                                else PROBE[pr])
                if REMOVE_BEFORE[i]:
                    env.reset_agent()
                S, G, LIT, A, P = rollout(env, T or PA_T)
                r = run_sequence(m, c, *to_t(S, G, LIT, A), h0=h, s_hat0=s_hat,
                                 collect=True, cb_noise=nz)
                h, s_hat = r["h_last"], r["s_hat_last"]
                maps.append(arena_maps(r["z"].cpu().numpy(), P, env.cue_phi,
                                       nr=PA_R, nphi=ROT_PHI))
            raw, cor, deg = rotation_stability(maps[1], maps[3])
            # NULL: each arena's S2 against a DIFFERENT arena's S4.  Realistic
            # map statistics, no true alignment.  Essential, because taking the
            # best of 24 rolls inflates any correlation -- mismatched arenas
            # still score 0.36-0.50 corrected.  Without this line on the figure
            # the recovered stability looks far more meaningful than it is.
            _, ncor, ndeg = rotation_stability(maps[1], np.roll(maps[3], 1, axis=0))
            out[(tag, pr)] = (raw, cor, deg, ncor, ndeg)
            lab = "light" if pr == "cue_full" else "DARK"
            print(f"{tag:<10}{lab:<10}{raw:>9.3f}{cor:>16.3f}{deg:>15.0f} deg"
                  f"    null {ncor:.3f} / {ndeg:.0f} deg")
    print("\n  corrected >> raw  ->  the field survived, the frame turned")
    print("  corrected ~= raw  ->  the map genuinely degraded")
    return out


def plot_rotation(models, fov=np.deg2rad(FOV_DEG), n=48, path=None):
    """LEFT: stability before and after correcting for a rigid rotation.
    RIGHT: how far the frame turned."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = report_rotation(models, fov=fov, n=n)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    x, w = np.arange(2), 0.36                       # light, dark

    for gi, tag in enumerate(("WT", "L7PKCI")):
        raw = [out[(tag, p_)][0] for p_ in ("cue_full", "cue_none")]
        cor = [out[(tag, p_)][1] for p_ in ("cue_full", "cue_none")]
        deg = [out[(tag, p_)][2] for p_ in ("cue_full", "cue_none")]
        ncor = [out[(tag, p_)][3] for p_ in ("cue_full", "cue_none")]
        ndeg = [out[(tag, p_)][4] for p_ in ("cue_full", "cue_none")]
        pos = x + (gi - .5) * w
        # raw as the solid bar, the rotation-recovered part stacked on top
        axes[0].bar(pos, raw, w * .9, color=GCOL[tag], edgecolor="black", lw=.8,
                    label=f"{tag}  as measured")
        axes[0].bar(pos, np.array(cor) - np.array(raw), w * .9, bottom=raw,
                    color=GCOL[tag], edgecolor="black", lw=.8, alpha=.35,
                    hatch="///",
                    label="recovered by rotation" if gi == 0 else None)
        # the null, drawn as a capped marker on each bar
        axes[0].plot(pos, ncor, "_", color="#c0392b", ms=16, mew=2.2,
                     label="chance (shuffled arenas)" if gi == 0 else None)
        axes[1].bar(pos, deg, w * .9, color=GCOL[tag], edgecolor="black", lw=.8,
                    label=tag)
        axes[1].plot(pos, ndeg, "_", color="#c0392b", ms=16, mew=2.2,
                     label="chance" if gi == 0 else None)

    for ax, ylab in zip(axes, ("place field stability  (S2 vs S4)",
                               "recovered rotation of the map  (deg)")):
        ax.set_xticks(x); ax.set_xticklabels(["light", "dark"])
        ax.set_ylabel(ylab)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("WT's dark map TURNS; the mutant's falls apart\n"
                 "red bars = chance, from pairing each arena's S2 with another "
                 "arena's S4.  WT clears it; the mutant does not.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"saved -> {path}")


# ============================================================================
# Within-session drift: is the dark map SMEARED or just MOVED?
# ============================================================================
# Every other figure builds one map per session over all PA_T steps, so a frame
# that rotates DURING a session appears as a smear and cannot be told apart
# from a sharp map sitting at a new angle.  The population coherence says the
# latter (WT loses only ~4% there while stability falls ~41%), but that is an
# inference, not a measurement.
#
# Splitting each session in half measures it directly:
#   stability(S2, S4-first)  vs  stability(S2, S4-second)
#       -> if the second half is worse, the rotation is PROGRESSIVE within the
#          session, and part of our WT mismatch is a session-length artefact
#   stability(S4-first, S4-second)
#       -> internal consistency.  High means a stable offset (moved); low means
#          the frame is still turning (smeared).
@torch.no_grad()
def half_session(mtl, cb, probe, seed=1234, n=N_TRIALS, T=None,
                 fov=np.deg2rad(FOV_DEG), cb_noise=0.0):
    """Maps for each session AND for each half of each session."""
    T = T or PA_T
    rng = np.random.default_rng(seed)
    env = CueArena(n, rng, sense_r=SENSE_FULL, cue_present=True, fov=fov)
    h, s_hat, full, halves = None, None, [], []
    for i, phase in enumerate(PROTOCOL):
        env.sense_r, env.cue_present = ((SENSE_FULL, True) if phase == "cue"
                                        else PROBE[probe])
        if REMOVE_BEFORE[i]:
            env.reset_agent()
        S, G, LIT, A, P = rollout(env, T)
        res = run_sequence(mtl, cb, *to_t(S, G, LIT, A), h0=h, s_hat0=s_hat,
                           collect=True, cb_noise=cb_noise)
        h, s_hat = res["h_last"], res["s_hat_last"]
        Z = res["z"].cpu().numpy()
        full.append(arena_maps(Z, P, env.cue_phi))
        k = T // 2
        halves.append((arena_maps(Z[:, :k], P[:, :k], env.cue_phi),
                       arena_maps(Z[:, k:], P[:, k:], env.cue_phi)))
    return full, halves


def plot_half_session(models, fov=np.deg2rad(FOV_DEG), n=40, path=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref, out = {}, {}
    for tag, (m, c) in models.items():
        nz = getattr(c, "cb_noise", 0.0)
        for pr in ("cue_full", "cue_none"):
            full, halves = half_session(m, c, pr, n=n, fov=fov, cb_noise=nz)
            ref[(tag, pr)] = full[1]                       # S2, the reference
            a, b = halves[3]                              # S4, first and second
            out[(tag, pr)] = (arena_stability(ref[(tag, pr)], a),
                              arena_stability(ref[(tag, pr)], b),
                              arena_stability(a, b))
        print(f"   {tag} done", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    labs = ["S2 vs\nS4 first half", "S2 vs\nS4 second half", "S4 first vs\nS4 second"]
    x = np.arange(3)
    for ax, pr, title in zip(axes, ("cue_full", "cue_none"),
                             ("cue kept", "cue REMOVED")):
        for gi, tag in enumerate(("WT", "L7PKCI")):
            ax.bar(x + (gi - .5) * .38, out[(tag, pr)], .36, color=GCOL[tag],
                   edgecolor="black", lw=.8, label=tag)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
        ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=11)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("place field stability")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Is the dark map SMEARED or just MOVED?\n"
                 "a high third bar means the frame settled at a new angle; a "
                 "low one means it was still turning", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)

    print(f"\n{'':18}{'S2 vs 1st':>11}{'S2 vs 2nd':>11}{'1st vs 2nd':>12}")
    for pr in ("cue_full", "cue_none"):
        for tag in ("WT", "L7PKCI"):
            print(f"{tag:<8}{pr:<10}" + "".join(f"{v:>11.3f}" for v in out[(tag, pr)]))
    print(f"saved -> {path}")


# ============================================================================
# The 2x2 as bars -- the paper's own contrast, session-wise
# ============================================================================
# Three differences from plot_bars_paper, all of them deliberate:
#
# 1. LIGHT vs DARK IS POOLED WITHIN ONE cue_none RUN.  The protocol is
#    cue/cue/probe/probe/cue, so a single cue-removal run already contains both
#    -- S1/S2/S5 lit, S3/S4 dark -- and that is the paper's own x-axis in Fig.
#    1E-H ("during the dark sessions", plural).  Reading S4 alone off a separate
#    cue_full run, which plot_bars_paper does, is the deepest point of the dark
#    period and therefore the most severe reading available: pooling S3 with S4
#    raises every dark value and shrinks every deficit.  Matching their analysis
#    costs apparent effect size and removes a "computed differently" objection.
#
# 2. FIELD RATE IS RELATIVE to each genotype's own light value.  The mutant's
#    units run ~21% hotter overall (mean|z| 0.124 -> 0.150), and once field rate
#    is normalised by each unit's own activation sd the genotype difference
#    collapses from +18% to +3.8%.  z is a tanh, not Hz.  Coherence and
#    stability are correlations, immune to gain, and stay absolute.
#
# 3. NO SIGNIFICANCE TESTS.  Seeds are retrainings of one architecture on one
#    data distribution, far more homogeneous than 40 mice; at SEMs of 0.003-0.020
#    almost any consistent effect clears p<0.001.  WT coherence fell 2.9% and
#    still came out p=2e-04, which certifies reproducibility, not meaning.  The
#    figure reports the SIZE of each change and lets the reader judge.
LIT_S, DK_S = (0, 1, 4), (2, 3)        # S1,S2,S5 cued;  S3,S4 probe
# Fig. 1J is a PROGRESSION, not one number -- the shape is the claim (WT flat
# 0.60/0.57/0.55; mutant declining from a higher start 0.82/0.57/0.42).  S2-S5
# is added beyond their panel: it separates progressive degradation from simple
# decay, since a model merely running down could not recover when the cue returns.
JIDX, JLAB = [0, 2, 3, 4], ["S1-S2", "S2-S3", "S2-S4", "S2-S5"]
PAPER_J = {"WT": [0.60, 0.57, 0.55], "L7PKCI": [0.78, 0.57, 0.42]}
GCOL = {"WT": "white", "L7PKCI": "#2b2b2b"}      # colour = GENOTYPE everywhere


def _pooled(m, c, nz, fov, n):
    """One cue-removal run, sessions pooled by condition, as the paper does."""
    r = arena_session_test(m, c, "cue_none", n=n, fov=fov, cb_noise=nz)
    M = r["maps"]
    f = lambda idx: (np.mean([arena_field_rate(M[i]) for i in idx]),
                     np.mean([arena_coherence(M[i]) for i in idx]))
    return f(LIT_S), f(DK_S), [r["stability"][i] for i in JIDX]


def plot_bars_2x2(seeds=(0,), sigma=CB_NOISE, fov=np.deg2rad(FOV_DEG), n=40,
                  path=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw, jraw = {}, {}
    for seed in seeds:
        # The mutant's noise is redrawn on every call (torch.randn_like); the
        # seed inside arena_session_test fixes only the trajectory.  Without
        # this the numbers move ~0.01 between rebuilds and the figure stops
        # matching whatever is quoted in the text.
        torch.manual_seed(seed)
        for tag, nz in (("WT", 0.0), ("L7PKCI", sigma)):
            pth = ckpt_path(tag, fov, seed, cb_noise=(None if tag == "WT" else sigma))
            if not os.path.exists(pth):
                print(f"   missing {os.path.basename(pth)} -- skipping seed {seed}")
                break
            m, c = load(pth)
            lt, dk, prog = _pooled(m, c, nz, fov, n)
            raw.setdefault((tag, "light"), []).append(lt)
            raw.setdefault((tag, "dark"), []).append(dk)
            jraw.setdefault(tag, []).append(prog)
        print(f"   seed {seed} done", flush=True)

    arr  = {k: np.array(v) for k, v in raw.items()}
    jarr = {k: np.array(v) for k, v in jraw.items()}
    for tag in ("WT", "L7PKCI"):                 # field rate -> own light value
        base = arr[(tag, "light")][:, 0].copy()
        for lab in ("light", "dark"):
            arr[(tag, lab)][:, 0] /= base

    ns = len(seeds)
    sem = lambda v: (v.std(0) / np.sqrt(ns)) if ns > 1 else np.zeros_like(v.mean(0))
    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.2))
    print(f"\n{'measure':<12}{'genotype':<9}{'cue present':>16}"
          f"{'cue removed':>16}{'change':>12}")
    for ax, (k, title) in zip(axes, [(0, "mean field rate (1E)\nrelative to light"),
                                     (1, "spatial coherence (1F)")]):
        top, notes = 0.0, []
        for gi, tag in enumerate(("WT", "L7PKCI")):
            lt, dk = arr[(tag, "light")][:, k], arr[(tag, "dark")][:, k]
            d = 100 * (dk - lt) / lt
            notes.append(f"{tag} {d.mean():.1f}%")
            for ci, v in enumerate((lt, dk)):
                e = v.std() / np.sqrt(ns) if ns > 1 else 0.0
                ax.bar(ci * 2.6 + gi, v.mean(), yerr=e, width=.9,
                       color=GCOL[tag], edgecolor="black", lw=.8, capsize=4,
                       label=tag if ci == 0 else "")
                top = max(top, v.mean() + e)
            print(f"{title[:11]:<12}{tag:<9}{lt.mean():>16.3f}"
                  f"{dk.mean():>16.3f}{d.mean():>11.1f}%")
        ax.set_ylim(0, top * 1.28)
        ax.set_title(f"{title}\ncue removal costs  " + ",  ".join(notes), fontsize=9)
        ax.set_xticks([.5, 3.1])
        ax.set_xticklabels(["light\nS1,S2,S5", "dark\nS3,S4"])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # -- panel 3: the Fig. 1J progression, cue removed --------------------
    ax, xj = axes[2], np.arange(len(JIDX))
    print()
    for gi, tag in enumerate(("WT", "L7PKCI")):
        v = jarr[tag]
        # Bars, not a line: S2-S5 is a DIFFERENT CONDITION (the cue is back), so
        # a line through it would imply interpolation through states that do not
        # exist.  The dashed rule marks the break.
        ax.bar(xj + (gi - .5) * .38, v.mean(0), yerr=sem(v), width=.36,
               color=GCOL[tag], edgecolor="black", lw=.7, capsize=3, label=tag)
        print(f"{'stability':<12}{tag:<9}" + "".join(f"{q:>10.3f}" for q in v.mean(0)))
    ax.axvline(2.5, color="#0006", lw=.9, ls="--")
    ax.text(3.0, ax.get_ylim()[1] * .93, "cue back", fontsize=8, ha="center",
            color="#555")
    ax.set_xticks(xj); ax.set_xticklabels(JLAB, fontsize=9)
    ax.set_title("place field stability (1J)\ncue removed", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for tag in ("WT", "L7PKCI"):
        print(f"{'paper':<12}{tag:<9}" + "".join(f"{q:>10.2f}" for q in PAPER_J[tag]))

    axes[0].set_ylabel(f"mean $\\pm$ SEM over {ns} seed" + ("s" if ns > 1 else
                       " (no error bars: one seed)"))
    axes[0].legend(frameon=False, fontsize=8)
    # No FOV in the title: the number is the HALF-POWER point of a graded
    # angular falloff, not a field width, and quoting it as "FOV 300" invites
    # the reading that there is a 60 deg blind zone behind.  There is not --
    # precision is clean to ~90 deg, half at 150 deg, and 85% noise directly
    # behind, never absent.
    fig.suptitle(f"Removing the cue costs the mutant more on every measure  "
                 f"(sigma {sigma})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    # SVG twin: vector for figure assembly, PNG for viewing.
    # Costs a few hundred ms -- the figure is already rendered.
    fig.savefig(str(path).replace('.png', '.svg'))
    plt.close(fig)
    print(f"\nsaved -> {path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Rochefort 2011 in an MTL-cerebellar circuit "
                    "(final configuration: no cerebellar substitution in the light)")
    ap.add_argument("--train", action="store_true",
                    help="train WT and L7-PKCI per seed; skips what exists")
    ap.add_argument("--figures", action="store_true",
                    help="every figure, one folder per seed plus a pooled set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--n", type=int, default=40,
                    help="arenas per condition -- each one is a mouse")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="L7-PKCI lesion strength: sd of the noise on what CB "
                         "transmits, as a fraction of its own scale")
    args = ap.parse_args()
    if not (args.train or args.figures):
        ap.print_help()
        return

    fov = np.deg2rad(FOV_DEG)
    global _POOLED_TAG

    if args.train:
        for s in args.seeds:
            _POOLED_TAG = None
            get_models(fov, s, CKPT_DIR, args.epochs, force=False,
                       cb_noise=args.sigma)      # trains only what is missing
    if not args.figures:
        return

    for s in args.seeds:
        _POOLED_TAG = None
        models = get_models(fov, s, CKPT_DIR, args.epochs, force=False,
                            cb_noise=args.sigma)
        sig = getattr(models["L7PKCI"][1], "cb_noise", CB_NOISE)
        d = fig_dir(s)
        print(f"\n=== seed {s} ===", flush=True)
        print(f"MTL input = s_prev({S_DIM}) + a_prev({A_DIM}) + e({S_DIM})"
              f"   [no current observation]")
        print("lit:  cue always observed, degraded by angle -- never substituted")
        print("dark: no signal to degrade, so the cue IS CB's prediction")
        print(f"lesion: noise on what CB transmits, sigma = {sig:.2f} "
              f"(WT = 0.00)\n", flush=True)

        report_arena(models, fov=fov, seed=s)
        report_paper_measures(models, fov=fov)
        plot_bars_paper(models, fov=fov, n=args.n,
                        path=os.path.join(d, "bars.png"))
        plot_rotation(models, fov=fov, n=args.n,
                      path=os.path.join(d, "rotation.png"))
        plot_half_session(models, fov=fov, n=args.n,
                          path=os.path.join(d, "half_session.png"))
        plot_session_length(models, fov=fov, n=args.n,
                            path=os.path.join(d, "stability_vs_session.png"))
        plot_ratemaps_2x2(seed=s, sigma=sig, fov=fov, reduce="first",
                          path=os.path.join(d, "ratemaps_first.png"))
        plot_ratemaps_2x2(seed=s, sigma=sig, fov=fov,
                          path=os.path.join(d, "ratemaps.png"))
        plot_map_similarity(seeds=(s,), sigma=sig, fov=fov, n=args.n,
                            path=os.path.join(d, "map_similarity.png"))
        plot_bars_2x2(seeds=(s,), sigma=sig, fov=fov, n=args.n,
                      path=os.path.join(d, "bars_2x2.png"))
        print(f"[seed {s}] figures -> {d}", flush=True)

    if len(args.seeds) > 1:
        # pooled figures average across independently trained networks, which is
        # the error bar that matters -- varying only the arena draw within one
        # network says the effect is reliable given that network, not that it
        # survives retraining
        _POOLED_TAG = f"pooled_seeds{min(args.seeds)}-{max(args.seeds)}"
        d = fig_dir(args.seeds[0])
        plot_map_similarity(seeds=tuple(args.seeds), sigma=args.sigma, fov=fov,
                            n=args.n, path=os.path.join(d, "map_similarity.png"))
        plot_bars_2x2(seeds=tuple(args.seeds), sigma=args.sigma, fov=fov,
                      n=args.n, path=os.path.join(d, "bars_2x2.png"))
        print(f"\npooled over {len(args.seeds)} seeds -> {d}", flush=True)


if __name__ == "__main__":
    main()
