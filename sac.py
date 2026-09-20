"""
sac.py -- SAC goal-reaching agent whose observations can be augmented with the
existing EC/CB circuit (model.py), frozen-pretrained or trained end-to-end.

task.py has no goal or reward -- it exists to be inferred from, not navigated
to. Everything here that is genuinely new is the goal-reaching wrapper around
task.env_step (sample_pos_goal, goal_env_step, step_reward): a random goal per
episode, reward shaped by distance, success on entering GOAL_RADIUS. The SAC
math (Actor/SoftQNetwork/SACTrainer) is the standard CleanRL-style recipe.

THE QUESTION THIS ASKS.  model.py's checkpoints were trained to predict
s_{t+1} from a forward-model error alone (main) or with s_t handed over
(control) -- see model.py's docstring. Ported here as a FROZEN feature
extractor for an unrelated downstream task (goal-reaching, not context
prediction), does the "denied s_t" design still produce useful context
features for control, compared to "control", an oracle (true context
one-hot), no augmentation, or an end-to-end-trained variant?

TIMING, carried over from model.py's rollout with one difference: the RL loop
needs a one-step-ahead lookahead to build the replay buffer's "next
observation" (SAC's Q-target needs it). See run_sac's per-step loop comment
for why this recomputes the EC step twice per timestep rather than trying to
carry a memoized lookahead across iterations -- it's a deliberate simplicity
trade, not an oversight; the GRUCell call is cheap.

VELOCITY, again as in model.py: model.build() reconstructs velocity from
CONSECUTIVE POSITIONS (V_t = (s_t - s_{t-1})/DT) rather than trusting the
physics simulator's own velocity state. Those are mathematically identical
whenever the position isn't clipped at a grid boundary (pos_new - pos =
DT*vel_new exactly, by construction of task.env_step) -- but the finite
difference form is EXACTLY what the frozen checkpoints were trained on even
at a boundary, so this file tracks position history and reconstructs velocity
the same way rather than reusing task.env_step's returned velocity, which
would silently diverge from the training distribution right at a wall.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import model as M
import task as T

import pdb

SDIM = M.SDIM                       # 4: [x, y, vx, vy]
ACTION_DIM = 2
HIDDEN_DIM = 256
LATENT_DIM = M.LATENT_DIM           # 64
EC_HIDDEN = M.EC_HIDDEN             # 256
CB_HIDDEN = M.CB_HIDDEN             # 256
DT = T.DT
POS_LIM = T.POS_LIM                 # position lives in [-1, +1]
GRID_SIZE = T.GRID_SIZE             # the EXTENT, 2*POS_LIM

# ── What the controller reads from the circuit ───────────────────────────
#
# PORTED FROM THE REFERENCE REPO (_ablations_sweep.py::_current_features), so
# this leg of the experiment is not a confound when comparing against it:
#
#   EC -> MC   z_t, passed straight through. The SAME tensor CB is
#              conditioned on; no second readout, no bottleneck.
#   CB -> MC   h2 @ cb_proj, where cb_proj is a FIXED RANDOM CB_HIDDEN x
#              CB_PROJ_DIM matrix with unit-norm columns, drawn once per run
#              from the run's seed and never trained.
#
# Both are computed under no_grad while the observation is BUILT, not inside
# the networks -- which is what makes one projection shared by the actor and
# both critics, and what puts the projected feature into the replay buffer.
# The RL loss therefore never reaches EC or CB, and "frozen" stays frozen.
#
# WIDTHS ARE MATCHED ACROSS CONTEXT CHANNELS, deliberately departing from the
# reference here: CB_PROJ_DIM is LATENT_DIM rather than the reference's 32, so
# z_ec, z_cb and the oracle's code are all 64-d. The reference could use 32 for
# both because its EC latent was also 32; this repo's is 64 (a property of the
# pretrained checkpoints), so keeping CB at 32 would have left the channels
# unmatched. Equal widths mean no channel is advantaged or handicapped by
# dimensionality alone, and init_by_source then weights them identically.
#
# WHAT THIS GIVES UP, deliberately, to match: the learned per-structure
# bottleneck and the per-source input prescale that preceded it. The prescale
# existed because S_SCALE = 15 inflates this repo's h2 to RMS ~5.85 against
# the kinematic block's 1.78 -- the reference had no such scaling, so its CB
# channel arrived at a comparable size on its own. Porting the construction
# faithfully therefore does NOT reproduce the reference's input BALANCE; see
# the RMS constants below, which are kept as measurements and are what the
# drift test still checks.
CB_PROJ_DIM = LATENT_DIM      # 64: matched to z_ec and the oracle code
CB_PROJ_SEED_BASE = 0        # cb_proj is drawn from (this + run seed)


def make_cb_proj(seed: int = 0, device="cpu", n_in: int = None, n_out: int = CB_PROJ_DIM):
    """The fixed random CB -> MC projection, columns normalized to unit norm.

    Per-run rather than global, exactly as the reference drew it from that
    run's rng: the projection is a fixed but arbitrary readout, and varying it
    across seeds keeps a single unlucky draw from becoming a result.
    """
    n_in = CB_HIDDEN if n_in is None else n_in
    g = torch.Generator().manual_seed(CB_PROJ_SEED_BASE + int(seed))
    P = torch.randn(n_in, n_out, generator=g)
    P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
    return P.to(device)


# ── The oracle's context code ────────────────────────────────────────────
#
# A DENSE, near-orthogonal embedding of the context label rather than a raw
# one-hot, living in the SAME space as EC's latent: ORACLE_EMBED_DIM is
# LATENT_DIM, because the oracle is the idealized z -- what EC would emit if
# it identified the context perfectly.
#
# WHY NOT A ONE-HOT. It differed from z on three axes at once: 4 dims against
# 64, a single active unit against a dense code, and exact orthogonality
# against a measured pairwise cosine of ~0.62 between z's context means. Any
# of the three could explain why the oracle retains across blocks and
# main_frozen does not. Matching width and density leaves near-orthogonality
# as the one structural difference remaining, which is the property worth
# isolating.
#
# Rows are unit-norm, so a code carries the same total energy the one-hot did,
# spread over 64 dimensions instead of concentrated in one. Four random unit
# vectors in 64-d have expected pairwise cosine 0 with sd ~1/sqrt(64) = 0.125:
# near-orthogonal, not exactly so.
#
# Drawn per run, like cb_proj, so one unlucky draw cannot become a result.
#
# ── AND NOW THE LAST DIFFERENCE CAN BE CLOSED TOO ────────────────────────
#
# The paragraph above names near-orthogonality as "the one structural
# difference remaining" and leaves it there. oracle_kernel closes it: the code
# is drawn from a GAUSSIAN PROCESS over the contexts whose KERNEL is a chosen
# pairwise-cosine matrix, and the near-orthogonal draw above is exactly that
# construction with the kernel set to the IDENTITY. Setting the kernel to z's
# own measured geometry instead gives an oracle that differs from z in nothing
# structural at all -- same width, same density, same pairwise cosines -- and
# still knows the context perfectly.
#
#   identity    K = I.  What the oracle has always used, bit for bit.
#   z           K = the cosine matrix of z's context means, common component
#               INCLUDED. This is the geometry MC actually receives, and the
#               one to use: the shared component is not inert to learning even
#               though it carries no context information, because it puts a
#               shared term in every context's gradient (see center_ec; the
#               overlap calibration came from the since-removed
#               policy_sensitivity probe). Measured +0.645 mean
#               off-diagonal, which is the "~0.62" the paragraph above cites.
#   z_centered   the same with that shared component removed: the DISCRIMINATIVE
#               geometry alone, mean off-diagonal -0.142. The control that
#               separates "the circuit's code is confusable" from "the
#               circuit's code is dominated by a common vector".
#
# THE COMPARISON IS FAIR BY CONSTRUCTION because all three go through the one
# function below. The identity oracle is not a differently-built code that
# happens to be near-orthogonal -- it is the same GP draw, the same
# renormalisation, the same width, with the kernel as the only difference. That
# is also why the identity path must stay bit-identical to what finished runs
# were trained on, which gp_code's two ordering details exist to guarantee.
#
# ONLY THE PHASE CONTEXTS. The agent meets N_PHASE_FAMILIES of the context
# table (PHASE_CONTEXT_IDS), so the kernel is measured over those and centered
# over those -- the shared component an agent sees is the mean of the contexts
# it actually meets, not of the full table.
ORACLE_EMBED_DIM = LATENT_DIM
ORACLE_EMBED_SEED_BASE = 1000        # distinct from cb_proj's stream
ORACLE_KERNELS = ("identity", "z", "z_centered")

ORACLE_GEOM_SITE = "z"               # which representation's geometry to copy
ORACLE_GEOM_N_SEQ = 600              # sequences rolled out to estimate it


def gp_code(K, dim: int, seed: int) -> np.ndarray:
    """`dim`-wide unit-norm codes drawn from the GP with kernel K.

    Write K = L L^T with L (n, r); then with G an (r, dim) standard normal,
    C = L G has each COLUMN an independent draw from N(0, K) over the contexts,
    and E[C C^T] = dim * K. Rows are renormalised to unit length -- which is
    why the 1/sqrt(dim) that would make that exactly K is not written and then
    divided away -- so no context drives MC harder than another and the kernel
    is all that distinguishes one code table from the next.

    r IS CHOSEN FROM THE SPECTRUM: the positive eigenvalues, with negatives
    clipped (a measured cosine matrix need not be quite PSD). A rank-deficient
    kernel therefore confines every context to an r-dimensional subspace
    however wide the code is, which is the part of a geometry that a wider code
    cannot paper over.

    A SINGLE DRAW REALISES K ONLY TO O(1/sqrt(dim)), and that is accepted
    rather than corrected -- it is what a code with this kernel looks like at
    the width z actually has. Measured mean |realised - K| over the 4x4 matrix:
    0.09 at dim 32, 0.06 at 64, 0.03 at 256. Callers that need to report the
    geometry a run actually had should measure it from the returned table
    rather than quoting K.

    TWO ORDERING DETAILS EXIST SO THAT K = I REPRODUCES A PLAIN NORMALISED
    GAUSSIAN DRAW BIT FOR BIT -- which is what this function replaced, and what
    every finished run in sac_runs/ was trained on:

      the eigenvalue sort is STABLE, so tied eigenvalues keep their original
      order. Under argsort(w)[::-1] the identity's equal eigenvalues come back
      REVERSED, L is a permutation rather than I, and every context silently
      swaps codes with another -- runs that should be identical differ by 0.6.
      the float32 cast happens BEFORE the renormalisation. Casting after moves
      the result by 3e-08, and task.next_action documents what a 5e-07 drift
      already costs in this repo.
    """
    K = np.asarray(K, dtype=np.float64)
    K = 0.5 * (K + K.T)                       # exactly symmetric, for eigh
    w, V = np.linalg.eigh(K)
    order = np.argsort(-w, kind="stable")     # stable: see the docstring
    w, V = np.maximum(w[order], 0.0), V[:, order]
    r = max(1, int((w > 1e-12 * max(w[0], 1.0)).sum()))
    L = V[:, :r] * np.sqrt(w[:r])
    g = np.random.default_rng(int(seed)).standard_normal((r, dim))
    C = (L @ g).astype(np.float32)            # cast BEFORE normalising
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


def oracle_kernel_matrix(kernel: str = "identity", n_contexts: int = None,
                         arm: str = "main", arm_seed: int = 0,
                         site: str = ORACLE_GEOM_SITE,
                         n_seq: int = ORACLE_GEOM_N_SEQ,
                         context_ids=None,
                         action_mode: str = "fitted") -> np.ndarray:
    """The (n, n) GP kernel the oracle's codes are drawn from.

    RAISES rather than falling back when a geometry kernel is asked for and no
    checkpoint is there. A silent fall back to the identity would produce a run
    named for the circuit's geometry that was trained on an orthogonal code --
    a result that is wrong rather than missing, and invisible in every figure
    downstream.

    residualise is FALSE for kernel="z" on purpose, and this is the trap the
    flag names: representation_analysis.context_centroids regresses out
    position and velocity with an INTERCEPT, so residualising alone already
    removes the contexts' shared component and would hand back the centered
    geometry under the uncentered name.
    """
    n = N_PHASE_FAMILIES if n_contexts is None else n_contexts
    if kernel == "identity":
        return np.eye(n)
    if kernel not in ORACLE_KERNELS:
        raise KeyError(f"unknown oracle_kernel {kernel!r}; "
                       f"expected one of {ORACLE_KERNELS}")
    # local: representation_analysis pulls in matplotlib and sklearn, and the
    # identity oracle -- the default -- needs neither
    from representation_analysis import collect, centroid_cosine
    d = collect(arm, arm_seed, n=n_seq, action_mode=action_mode)
    if d is None:
        raise FileNotFoundError(
            f"oracle_kernel={kernel!r} needs the {arm} seed {arm_seed} "
            f"({action_mode}) checkpoint to measure the geometry from, and "
            f"{M.ckpt_path(arm, arm_seed, action_mode)} is not there. "
            f"Train it first.")
    ids = PHASE_CONTEXT_IDS if context_ids is None else list(context_ids)
    if len(ids) != n:
        raise ValueError(f"oracle_kernel={kernel!r} was asked for {n} contexts "
                         f"but {len(ids)} context ids were given: {ids}")
    center = kernel == "z_centered"
    S, _ = centroid_cosine(d, site, residualise=center, center=center,
                           context_ids=ids)
    return S


def make_oracle_embed(seed: int = 0, n_contexts: int = None,
                      dim: int = ORACLE_EMBED_DIM, kernel: str = "identity",
                      arm: str = "main", arm_seed: int = 0,
                      action_mode: str = "fitted") -> np.ndarray:
    """Fixed dense codes for the context labels, one unit-norm row each.

    `kernel` selects the geometry those codes have -- see ORACLE_KERNELS above.
    The default reproduces the near-orthogonal draw this function has always
    returned, bit for bit.
    """
    n = N_PHASE_FAMILIES if n_contexts is None else n_contexts
    K = oracle_kernel_matrix(kernel, n, arm, arm_seed, action_mode=action_mode)
    return gp_code(K, dim, ORACLE_EMBED_SEED_BASE + int(seed))


# ── Matched-SNR oracle: a perfect label, degraded to the circuit's SNR ────
#
# Landmarks measured on main seed 0 over the four phase contexts, as the SNR a
# LINEAR readout can reach inside the code's own subspace (see
# AblationConfig.oracle_snr on why that is the right quantity):
#
#   1.00   the oracle as it has always been -- no within-context variance
#   0.61   the circuit on the distribution model.py fitted it on,  rho 0.60
#   0.19   the circuit under a trained SAC policy,                 rho 0.84
#
# Re-measure with the scratch diagnostics rather than trusting these if the
# circuit is retrained; they are a property of a checkpoint, not a constant.
ORACLE_SNR_AS_FITTED = 0.61
ORACLE_SNR_DEPLOYED = 0.19


def oracle_code_subspace(embed: np.ndarray) -> np.ndarray:
    """Orthonormal basis (r, dim) of the code table's DISCRIMINATIVE subspace.

    The centroids minus their mean, so r is at most n_contexts - 1: the shared
    component is deliberately excluded, because noise along it carries no
    context and would inflate the denominator without being confusable with
    anything. Rows of `embed` are the per-context codes.
    """
    disc = np.asarray(embed, dtype=np.float64)
    disc = disc - disc.mean(0)
    _u, s, vt = np.linalg.svd(disc, full_matrices=False)
    r = max(1, int((s > 1e-9 * max(s[0], 1e-12)).sum()))
    return vt[:r].astype(np.float32)


def snr_from_ratio(r: float) -> float:
    """Signal-to-noise RATIO (V/P, unbounded) -> the variance FRACTION
    AblationConfig.oracle_snr takes (V/(V+P), in [0, 1]).

    Rungs are specified in r because that is the axis the effect lives on --
    see oracle_snr. The noiseless oracle is r = infinity, which no finite
    ladder reaches: a sweep can bound the cliff from below but can never show
    convergence to the oracle, and a result of "retention only at the top
    rung" means the requirement is effectively a noiseless label rather than
    some particular finite SNR.
    """
    if r <= 0:
        raise ValueError(f"signal-to-noise ratio must be positive, got {r}")
    return float(r / (1.0 + r))


def ratio_from_snr(s: float) -> float:
    """The inverse of snr_from_ratio. Infinite at s == 1, by construction."""
    if not (0.0 < s <= 1.0):
        raise ValueError(f"oracle_snr must be in (0, 1], got {s}")
    return float("inf") if s == 1.0 else float(s / (1.0 - s))


def oracle_noise_scale(embed: np.ndarray, snr: float, rank: int = None) -> float:
    """Noise sd per basis dimension that puts total noise power at the level
    `snr` implies, given this code table's own between-context spread.

    With codes m_c, mean mbar and V = mean||m_c - mbar||^2, total noise power
    P = V (1 - snr) / snr, spread evenly over `rank` directions.

    `rank` DEFAULTS TO THE DISCRIMINATIVE SUBSPACE but is passed explicitly by
    OracleNoise, because the isotropic mode spreads the same P over all `dim`
    directions. Both modes then carry equal RAW noise power, which is what
    makes them comparable; they differ in how much of it lands where it can
    interfere. Note only the subspace mode has snr equal to the readable SNR --
    under isotropic noise a linear readout sees a higher effective SNR, and
    that asymmetry is the manipulation, not a flaw.
    """
    if not (0.0 < snr <= 1.0):
        raise ValueError(f"oracle_snr must be in (0, 1], got {snr}")
    if snr == 1.0:
        return 0.0
    disc = np.asarray(embed, dtype=np.float64)
    disc = disc - disc.mean(0)
    V = float((disc ** 2).sum(1).mean())
    r = oracle_code_subspace(embed).shape[0] if rank is None else int(rank)
    return float(np.sqrt(V * (1.0 - snr) / snr / r))


class OracleNoise:
    """AR(1) noise in the code's discriminative subspace, reset per episode.

    HOLDS MAGNITUDE FIXED ACROSS THE SWEEP. The channel is
    (code + noise) / sqrt(1 + P), where P is the noise power the target SNR
    implies, so its expected squared norm stays that of the noiseless code.
    A constant scalar cannot change between/total variance, so the sweep varies
    SNR and nothing else -- without it the block would get sqrt(1/snr) louder
    as SNR fell and the actor's input drive would move with the manipulation.

    At snr == 1.0 this is the identity: scale 0, divisor 1, so every finished
    oracle run stays reproducible bit for bit.
    """

    def __init__(self, embed: np.ndarray, snr: float, rho: float = 0.0,
                 rng=None, isotropic: bool = False):
        self.snr = float(snr)
        self.rho = float(rho)
        self.isotropic = bool(isotropic)
        # isotropic spreads the SAME total power over all `dim` directions, so
        # both modes carry equal raw noise power and differ only in where it
        # points -- which is the whole manipulation. oracle_noise_scale divides
        # by the basis rank, so rebuild it against the wider basis.
        self.basis = (np.eye(embed.shape[1], dtype=np.float32) if isotropic
                      else oracle_code_subspace(embed))          # (r, dim)
        self.scale = oracle_noise_scale(embed, snr,
                                        rank=self.basis.shape[0])
        self.rng = np.random.default_rng(0) if rng is None else rng
        # the renormaliser: total noise power is scale^2 * r
        self.divisor = float(np.sqrt(1.0 + self.scale ** 2 * self.basis.shape[0]))
        self.u = np.zeros(self.basis.shape[0], dtype=np.float32)

    @property
    def active(self) -> bool:
        return self.scale > 0.0

    def reset(self) -> None:
        """Draw the AR(1) process at its STATIONARY marginal, so step 0 of an
        episode is no quieter than step 30 -- starting it at zero would make
        the channel cleanest exactly where the circuit's is noisiest."""
        self.u = self.rng.standard_normal(self.basis.shape[0]).astype(np.float32)

    def advance(self) -> None:
        """One AR(1) step. CALLED ONCE PER ENV STEP, not once per observation.

        The rollout builds TWO observations per step -- the one it acts on and
        the lookahead's next_obs -- and they are timesteps t and t+1 of the
        same process, not two draws from it. So the lookahead advances and the
        next iteration reads the SAME sample back through `current`, which is
        what makes the stored (obs, next_obs) pair a real transition of this
        channel. cb_channel draws independently instead, and is right to: its
        premise is that the block carries nothing across steps. This one's
        premise is the opposite.
        """
        if not self.active:
            return
        eps = self.rng.standard_normal(self.basis.shape[0]).astype(np.float32)
        self.u = (self.rho * self.u
                  + np.sqrt(1.0 - self.rho ** 2) * eps).astype(np.float32)

    @property
    def current(self) -> np.ndarray:
        """The noise vector in code space, at the current timestep."""
        if not self.active:
            return np.zeros(self.basis.shape[1], dtype=np.float32)
        return ((self.scale * self.u) @ self.basis).astype(np.float32)

    def apply(self, oracle_vec) -> np.ndarray:
        """`oracle_vec` at this timestep's noise; a no-op when snr == 1.0.

        The warmup block passes a zero code (there is no context to encode),
        and it stays zero: noise on a label that says nothing would invent a
        context for the null field.
        """
        if oracle_vec is None or not self.active:
            return oracle_vec
        if not np.any(oracle_vec):
            return oracle_vec
        return ((np.asarray(oracle_vec, dtype=np.float32) + self.current)
                / self.divisor).astype(np.float32)


# ── Ideal-observer oracle: the context INFERRED, exactly, from the evidence ──
#
# The one-hot oracle is handed the context. EC infers it, but through a learned
# GRU whose readout also carries the last action -- and under a closed-loop
# policy that action-dependence is what makes z conjunctive and stops it from
# partitioning MC. This oracle sits between the two: it must infer the context
# from the same evidence EC gets, but its state is the exact Bayesian posterior
# over the phase contexts -- a sufficient statistic, action-free by
# construction. Against the one-hot it prices INFERENCE; against `main` it
# prices EC's READOUT. It is the finite-context ideal observer that COIN's
# contextual-inference account reduces to when the contexts are known, so it
# is the natural COIN baseline for this task.
#
# EVIDENCE. The plant is vel' = vel + (A_c a_noisy - DAMPING vel) DT/MASS with
# a_noisy = a + CMD_NOISE |a| eta, so off the walls and below the speed cap the
# realised force is recovered EXACTLY from what the agent observes:
#
#     y = (vel' - vel) MASS/DT + DAMPING vel = A_c a_noisy
#     y | c ~ N(A_c a, (CMD_NOISE |a|)^2 I)       (A_c orthogonal)
#
# Wall contacts zero the velocity and the speed cap rescales it, so those steps
# carry no usable evidence and are skipped rather than misread. VEL_NOISE and
# POS_NOISE are 0, so the inversion has no other error term.
#
# TRANSITION. Contexts switch between blocks and the observer is not told
# when, so it runs a sticky HMM: per step, stay with probability 1 - hazard or
# jump uniformly to another context. Evidence is strong -- rotations 45/90
# degrees apart give (1 - cos theta)/CMD_NOISE^2 = 3.7/12.8 nats per step --
# so at hazard 1e-3 (log((1-h)/h) = 6.9 nats to overturn) a switch is noticed
# in one or two steps. The cost of inference is therefore SMALL when the
# posterior is exact, which is exactly what makes the comparison to `main`
# informative: whatever `main` loses relative to this, it loses to its code.
BAYES_SWITCH_HAZARD = 1e-3


ORACLE_SOURCES = ("given", "bayes", "cb")


class _ContextPosterior:
    """Shared machinery of the inferred oracles: a sticky-HMM posterior over
    the phase contexts and its code -- the posterior-weighted mixture of the
    one-hot oracle's context codes, so at certainty it IS the one-hot code and
    lives in the same channel. Subclasses supply the evidence model."""

    def __init__(self, embed, context_ids=None, hazard=BAYES_SWITCH_HAZARD,
                 null_context=False):
        self.ids = list(PHASE_CONTEXT_IDS if context_ids is None else context_ids)
        self.embed = np.asarray(embed, dtype=np.float32)
        # THE NULL FIELD AS A HYPOTHESIS (2026-09-18). Without it a P0 block
        # has no context that fits: a zero rotation is equidistant from +-30,
        # so the belief sits at 50/50 between P+ and P- for ever, the code MC
        # gets is the midpoint of their codes, and the P0 transitions overwrite
        # the P+ slot (coin washed out to 0.5, not 0, with replay -- see the
        # memory of 2026-09-18). COIN treats the baseline as a context in its
        # own right. Appended LAST so the phase indices are unchanged, with the
        # ZERO code -- what the given oracle emits outside a context
        # (oracle_code(None)) -- so a null-certain code sits at the origin.
        self.null_context = bool(null_context)
        if self.null_context:
            self.ids.append(None)
            self.embed = np.vstack([self.embed, np.zeros((1, self.embed.shape[1]), np.float32)])
        self.K = len(self.ids)
        self.hazard = float(hazard)
        # the visit-count prior (AblationConfig.observer_count_prior): inert
        # at weight 0, where _predict is the original sticky-uniform step
        self.count_prior = 0.0
        self.counts = np.zeros(self.K, dtype=np.float64)
        self.reset()

    def reset(self):
        self.log_p = np.full(self.K, -np.log(self.K))

    def fresh(self):
        """A copy at the uniform prior, sharing every static part (matrices,
        models, calibrations) and carrying the visit counts -- what a probe
        starts from. The counts are copied, not shared: a probe must not
        write into the run's."""
        import copy
        c = copy.copy(self)
        c.counts = np.array(self.counts, copy=True)
        c.reset()
        return c

    def snapshot(self):
        """A copy CARRYING the current posterior and counts -- what the
        per-reach readout probe starts from, so its first action reads the
        belief the agent brings into its next reach rather than a uniform
        one. Copies, not shares: the probe's updates must not touch the run."""
        import copy
        c = copy.copy(self)
        c.counts = np.array(self.counts, copy=True)
        c.log_p = np.array(self.log_p, copy=True)
        return c

    def note_visit(self):
        """End of a reach with evidence: add its posterior to the counts."""
        self.counts = self.counts + self.posterior

    @property
    def posterior(self) -> np.ndarray:
        return np.exp(self.log_p)

    @staticmethod
    def evidence_usable(action, vel_new, wall_impulse=0.0) -> bool:
        """False on a wall contact, at the speed cap, or for a null action --
        neither the force law nor CB describes those steps."""
        if wall_impulse > 0.0:
            return False
        if np.linalg.norm(vel_new) >= T.V_MAX * (1.0 - 1e-6):
            return False
        return float(np.linalg.norm(action)) > 1e-6

    def _predict(self) -> np.ndarray:
        """Sticky transition: stay with 1 - hazard, else jump -- uniformly
        over the other contexts (count_prior 0, the original), or toward a
        (1-w) uniform + w visit-count mixture."""
        p = self.posterior
        if self.K > 1:
            if self.count_prior <= 0.0 or self.counts.sum() <= 0.0:
                p = (1.0 - self.hazard) * p + self.hazard * (1.0 - p) / (self.K - 1)
            else:
                w = min(float(self.count_prior), 1.0)
                q = (1.0 - w) / self.K + w * self.counts / self.counts.sum()
                p = (1.0 - self.hazard) * p + self.hazard * q
        return np.log(np.maximum(p, 1e-300))

    def _condition(self, log_prior, loglik):
        log_p = log_prior if loglik is None else log_prior + loglik
        self.log_p = log_p - np.logaddexp.reduce(log_p)

    def loglik(self, a, vel_prev, vel_new, s_prev=None, s_next=None):
        raise NotImplementedError

    def update(self, action, vel_prev, vel_new, wall_impulse=0.0, *,
               s_prev=None, s_next=None, evidence=True):
        """One step: predict, then condition if the step carries evidence.
        `s_prev`/`s_next` are the circuit-unit states (circuit_state); the
        plant observer ignores them, the CB observer needs them. evidence=False
        (a channel reach) is the predict step alone: the belief relaxes under
        the transition prior and nothing else."""
        log_prior = self._predict()
        a = np.asarray(action, dtype=np.float64)
        ll = (self.loglik(a, vel_prev, vel_new, s_prev, s_next)
              if evidence and self.evidence_usable(a, vel_new, wall_impulse) else None)
        self._condition(log_prior, ll)

    def code(self) -> np.ndarray:
        return (self.posterior.astype(np.float32) @ self.embed).astype(np.float32)


class BayesObserver(_ContextPosterior):
    """The ideal observer through the TRUE plant. The velocity change is
    dv = (A_c a_noisy + B_c v - DAMPING v) DT / MASS + w_c, so the quantity
    y = dv MASS / DT + DAMPING v equals A_c a_noisy + B_c v + w_c MASS / DT
    exactly (off the walls and the speed cap; see the comment above). A is
    orthogonal or the identity for every family, so the command noise reaches
    y with its own magnitude, CMD_NOISE |a|, whatever the context."""

    def __init__(self, embed, context_ids=None, hazard=BAYES_SWITCH_HAZARD,
                 cmd_noise=None, null_context=False):
        super().__init__(embed, context_ids, hazard, null_context)
        # context_affine(None) is the identity plant: the null hypothesis
        affine = [T.context_affine(c) for c in self.ids]
        self.A = np.stack([np.asarray(A, dtype=np.float64) for A, _, _ in affine])   # (K, 2, 2)
        self.B = np.stack([np.asarray(B, dtype=np.float64) for _, B, _ in affine])   # (K, 2, 2)
        # the drift is a velocity increment; in force units it is w MASS / DT
        self.d = np.stack([np.asarray(w, dtype=np.float64) * (T.MASS / T.DT)
                           for _, _, w in affine])                                    # (K, 2)
        self.sigma = float(T.CMD_NOISE if cmd_noise is None else cmd_noise)

    def loglik(self, a, vel_prev, vel_new, s_prev=None, s_next=None):
        vel_prev = np.asarray(vel_prev, dtype=np.float64)
        vel_new = np.asarray(vel_new, dtype=np.float64)
        y = (vel_new - vel_prev) * (T.MASS / T.DT) + T.DAMPING * vel_prev
        mu = self.A @ a + self.B @ vel_prev + self.d                # (K, 2)
        s2 = (self.sigma * np.linalg.norm(a)) ** 2
        return -np.sum((y - mu) ** 2, axis=1) / (2.0 * s2)


# ── The same observer, scored through the LEARNED forward model ──────────
#
# `coin` inverts the true plant, so its gap to `main` bundles two things: EC's
# readout carrying the action, and CB being an imperfect model. This variant
# separates them. Each context hypothesis c is CB conditioned on that
# context's z-centroid -- the per-context mean of z on the supervised data,
# exactly the quantity oracle_geom measures (residualise=False, center=False)
# -- and the observed transition is scored under each with a diagonal Gaussian
# whose std is CB's OWN residual std under centroid conditioning, calibrated
# once on the supervised data. Using the plant's command-noise variance
# instead would make a learned model overconfident about its own errors.
#
# This is also the more literal COIN: one predictor per context, responsibility
# from each predictor's error, inference and memory in the same loop.
#
#     coin    - coin_cb  =  the cost of CB's model error to inference
#     coin_cb - main     =  the cost of EC's readout, CB held identical
CB_OBSERVER_N_SEQ = 200          # transitions for the residual calibration


def circuit_of(config) -> "M.CircuitSpec":
    """The supervised circuit a config's frozen fields name -- ONE object for
    every load, geometry measurement and rollout, so a config cannot select
    the checkpoint with one set of fields and the rollout distribution with
    another. Older saved blobs lack the newer fields; they get the defaults
    those blobs were written under."""
    return M.CircuitSpec(config.action_mode,
                         getattr(config, "ec_readout", M.DEFAULT_READOUT),
                         getattr(config, "rate_beta", M.RATE_BETA),
                         getattr(config, "noise_sigma", M.NOISE_SIGMA),
                         getattr(config, "train_contexts", M.DEFAULT_TRAIN_CONTEXTS))


def cb_observer_parts(config: AblationConfig, device="cpu"):
    """(cb, centroids (K, LATENT), resid_std (SDIM,)) for a CBObserver, from
    the config's circuit (circuit_of) checkpoint. Raises if absent."""
    # local: representation_analysis pulls in matplotlib and sklearn
    from representation_analysis import collect, context_centroids
    from data import generate_sequences
    spec = circuit_of(config)
    loaded = M.load(config.arm, config.arm_seed, spec)
    if loaded is None:
        raise FileNotFoundError(
            f"oracle_source='cb' needs the {config.arm} seed {config.arm_seed} "
            f"({spec}) checkpoint; {M.ckpt_path(config.arm, config.arm_seed, spec)} "
            f"is not there.")
    _ec, cb = loaded
    cb = cb.to(device).eval()
    d = collect(config.arm, config.arm_seed, n=ORACLE_GEOM_N_SEQ, action_mode=spec)
    cen = context_centroids(d, "z", residualise=False, center=False,
                            context_ids=PHASE_CONTEXT_IDS).astype(np.float32)
    # residuals of CB under TRUE-context centroid conditioning, pooled over
    # the phase contexts, per state dimension
    trans, labels, _tss = generate_sequences(CB_OBSERVER_N_SEQ, seed=321,
                                             contexts=list(PHASE_CONTEXT_IDS),
                                             action_mode=spec)
    obs, act, tgt = M.build(trans)
    cid = labels[..., 0].reshape(-1).numpy()
    obs = obs.reshape(-1, SDIM); act = act.reshape(-1, ACTION_DIM); tgt = tgt.reshape(-1, SDIM)
    res = []
    with torch.no_grad():
        for k, c in enumerate(PHASE_CONTEXT_IDS):
            m = torch.as_tensor(cid == c)
            if not bool(m.any()):
                continue
            z = torch.as_tensor(cen[k], device=device).expand(int(m.sum()), -1)
            s_pred, _ = cb_predict(cb, obs[m].to(device), act[m].to(device), z, device)
            res.append((tgt[m].to(device) - s_pred).cpu().numpy())
    resid_std = np.maximum(np.concatenate(res).std(0), 1e-3).astype(np.float32)
    return cb, cen, resid_std


class CBObserver(_ContextPosterior):
    """Posterior over the phase contexts scored through the frozen CB, each
    context represented by its z-centroid. See the comment above."""

    def __init__(self, embed, cb, centroids, resid_std, context_ids=None,
                 hazard=BAYES_SWITCH_HAZARD, device="cpu", null_context=False):
        if null_context:
            # the null field has no z-centroid on the supervised data (it is
            # not a training context), so this observer cannot represent it
            raise ValueError("CBObserver has no null-field hypothesis; run the "
                             "paradigm schedules with oracle_source='bayes'")
        super().__init__(embed, context_ids, hazard)
        if len(centroids) != self.K:
            raise ValueError(f"{len(centroids)} centroids for {self.K} contexts")
        self.cb = cb
        self.device = device
        self.centroids = torch.as_tensor(np.asarray(centroids, dtype=np.float32),
                                         device=device)                  # (K, LATENT)
        self.resid_std = np.asarray(resid_std, dtype=np.float64)          # (SDIM,)

    def loglik(self, a, vel_prev, vel_new, s_prev=None, s_next=None):
        if s_prev is None or s_next is None:
            return None
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(s_prev, dtype=np.float32),
                                device=self.device).view(1, -1).expand(self.K, -1)
            at = torch.as_tensor(a.astype(np.float32),
                                 device=self.device).view(1, -1).expand(self.K, -1)
            s_pred, _ = cb_predict(self.cb, s, at, self.centroids, self.device)
        r = (np.asarray(s_next, dtype=np.float64) - s_pred.cpu().numpy()) / self.resid_std
        return -0.5 * np.sum(r ** 2, axis=1)


def observer_has_null(config) -> bool:
    """Whether the config's inferred oracle carries the null-field hypothesis:
    AblationConfig.observer_null_context when set, else yes under the COIN
    paradigm schedules (which have P0 blocks) and no under "cycle" (whose
    runs stay as they were)."""
    explicit = getattr(config, "observer_null_context", None)
    if explicit is not None:
        return bool(explicit)
    schedule = getattr(config, "schedule", DEFAULT_SCHEDULE)
    # only where the schedule's baseline IS the null field (P0 blocks);
    # savings-washout runs on a trained baseline and needs no such hypothesis
    return schedule != "cycle" and schedule_contexts(schedule)["base"] is None


def make_observer(config: AblationConfig, oracle_embed, device="cpu"):
    """The inferred oracle the config asks for, or None for the given one."""
    if not config.use_oracle or config.oracle_source == "given":
        return None
    null = observer_has_null(config)
    if config.oracle_source == "bayes":
        return BayesObserver(oracle_embed, null_context=null)
    if config.oracle_source == "cb":
        cb, cen, resid_std = cb_observer_parts(config, device)
        return CBObserver(oracle_embed, cb, cen, resid_std, device=device, null_context=null)
    raise KeyError(f"unknown oracle_source {config.oracle_source!r}; "
                   f"expected one of {ORACLE_SOURCES}")


def make_bayes_observer(config: AblationConfig, oracle_embed):
    """Kept for callers of the first version: the plant observer or None."""
    return make_observer(config, oracle_embed) if config.oracle_source == "bayes" else None


def make_oracle_noise(config: AblationConfig, oracle_embed, seed=0):
    """The run's OracleNoise, or None when the config has no oracle channel."""
    if not config.use_oracle or oracle_embed is None:
        return None
    return OracleNoise(oracle_embed, config.oracle_snr, config.oracle_noise_rho,
                       rng=np.random.default_rng(ORACLE_EMBED_SEED_BASE + int(seed)),
                       isotropic=config.oracle_noise_iso)

# THE RL LOSS NEVER REACHES EC OR CB. Their states enter the replay buffer as
# numpy, which severs the graph, so "frozen" still means frozen and the
# end-to-end configs' circuits are still shaped by prediction alone. What
# every config with a circuit learns is how to READ it.

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

LR_ACTOR = 3e-4
LR_Q = 3e-4
LR_EC_CB = 1e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 10000

# Recency-weighted replay, ported from the previous iteration's
# _ablations_sweep.py:  p(e) proportional to exp(-dt / gamma_t),  then mixed
# with a uniform floor.  dt is how many env steps ago the transition was
# stored, so recent experience dominates and old experience decays smoothly
# rather than being cut off.

PRIORITY_GAMMA_T = 2000.0
PRIORITY_GAMMA_TD = 1.0
PRIORITY_MIX = 0.02

# EPISODE-INDEXED replay (AblationConfig.replay_units = "episodes").
#
# The transition-indexed buffer above couples memory span to COMPETENCE: a
# fixed 10000 transitions is ~190 episodes for the baseline at ~53 steps each
# but ~400 for the oracle at ~25, and gamma_t = 2000 steps is ~38 episodes for
# one and ~80 for the other. So the agent that succeeds fastest replays the
# most old-context data -- the oracle's third-cycle revisits were sampling its
# own stale transitions at up to 22% while the circuit's were at 0% -- and a
# retention contrast between configs is confounded by how much there was to
# replay. In episode units the span and the recency constant are the same for
# every config, and every episode carries equal sampling mass, so a 60-step
# failure no longer outweighs a 20-step success three to one.
#
# Was CALIBRATED TO THE BASELINE, so that it was the competent agents that
# moved: 200 episodes ~ 10000 transitions / 53 steps, 40 episodes ~ 2000 / 53.
# RAISED TO 500 (2026-09-18) once the rehearsal budget below made capacity and
# batch composition independent. Under recency alone, capacity set how much
# old data there was AND (through the tail) how much of the batch it took; now
# the budget fixes the shares and capacity only sets HOW FAR BACK the old pool
# reaches. At 200 the paradigms could not hold their own history -- P0 100 +
# P+ 300 + P- 50 is 450 reaches, so entering the channel the buffer had
# already evicted half the P+ block it was meant to be protecting. 500 covers
# every paradigm's learned blocks (savings is the longest at 100 + 300 + 75 +
# 300, where only the base block ages out).
BUFFER_CAPACITY_EPISODES = 500
PRIORITY_GAMMA_EP = 40.0
# A FIXED REHEARSAL BUDGET (2026-09-18). One recency constant cannot do the
# two jobs replay is asked for: the 40 episodes that keep P+ alive through the
# P- block are the same 40 that starve P- of gradient. Measured composition
# under recency alone: the block being learned is 0.22 of the batch 10 reaches
# in and 0.71 at reach 50, so a 50-reach P- block never sees the buffer the
# 300-reach P+ block had to itself. Splitting the batch decouples them: the
# CURRENT block gets a fixed majority whatever its length, and everything
# before it shares a fixed budget, so protection no longer decays and
# dilution no longer depends on block length. Recency still orders each pool.
# Inert at 0.0 (every run before this); the paradigm schedules set it.
PARADIGM_REHEARSAL = 0.15
REPLAY_UNITS = ("transitions", "episodes")

# DEFAULT for AblationConfig.clear_buffer_on_switch, which is what the run loop
# actually reads. False matches the previous iteration: nothing STRICTLY
# prevents one context's transitions from being replayed into the next phase
# -- the recency weighting above plus capacity eviction only make it
# progressively unlikely. The *_clearbuf presets in sweep_ablations set it
# True per config (see ReplayBuffer.clear), at the cost of a short no-data gap
# after a switch. Flip this constant only to change the default for EVERY
# preset that does not pin its own.
CLEAR_BUFFER_ON_PHASE_SWITCH = False
BATCH_SIZE = 256

# LEARNING_STARTS gates gradient updates on having collected this many steps.
# It is BATCH_SIZE because that is the real floor -- the update also requires
# len(buffer) >= BATCH_SIZE -- so anything larger only buys extra random
# exploration. The reference repo's 5000 was calibrated for MAX_STEPS=100 at
# DT=1.0; here it consumed the first 100 episodes (20% of a 500-episode phase)
# with no learning at all, and ONLY the first phase paid that cost, which
# inflated the apparent savings for whichever context happened to come first.
# WARMUP_EPISODES below is the other half of the fix.
LEARNING_STARTS = BATCH_SIZE

# A null-field baseline block run BEFORE the phase sequence, in
# NULL_FIELD (F = a, no transformation). It fills the replay
# buffer and lets the agent learn to reach, so every logged phase then begins
# under identical conditions -- learning active from its first episode -- and
# the first context is no longer handicapped relative to the rest. This
# mirrors the null-field practice block that precedes perturbation in motor
# adaptation experiments. Its episodes are NOT logged into the curve; set to 0
# to skip it.
WARMUP_EPISODES = 0
UPDATE_EVERY = 1
GRAD_CLIP_NORM = 1.0

GOAL_RADIUS = 0.1
# Floor on the start-goal distance in sample_pos_goal: a shorter reach is
# nearly all first action, carries almost no adaptation to measure, and still
# pays a full SUCCESS_BONUS. 0.5 drops the shortest 15% of draws and moves the
# median distance 1.10 -> 1.16; see sample_pos_goal.
MIN_GOAL_DIST = 0.5
MAX_STEPS = 30
SUCCESS_BONUS = 5.0

assert GOAL_RADIUS < GRID_SIZE, (
    "GOAL_RADIUS must be smaller than the grid, or sample_pos_goal cannot "
    "find a start/goal pair outside it")

# Reward weights -- see step_reward. DIST_WEIGHT=0.1 is the previous
# iteration's absolute-distance coefficient. PROGRESS_WEIGHT scales the
# RELATIVE per-step change in distance, (d_prev - d)/d_prev, which is a
# fraction of the remaining gap rather than an absolute displacement -- its
# natural magnitude is DT*|v|/d_prev, at most ~0.19 when d_prev is at the
# GOAL_RADIUS floor.
#
# PROGRESS_PENALTY_SCALE multiplies the term when progress is NEGATIVE, so
# moving away costs more than the same movement toward the goal earns. 1.0 is
# symmetric; >1 makes not knowing which way to go expensive, which is the
# point -- but see step_reward on the stalling incentive it creates.
#
# DIST_WEIGHT 0.1 -> 0.3 (2026-09-18). On the v2 plant (MASS 32, DAMPING 1)
# SAC collapsed mid-block in every run: success 0.9 at reach 40 of P+, 0.04
# by reach 125, the deterministic action shrunk to |a| ~ 0.03 -- the agent
# learned to PARK, far from the goal, although parking returns -10.8 against
# +6 for a noisy PD law. The absolute term is the only part of the reward
# that says parking is bad, and at 0.1 d it was too weak an anchor once the
# plant had momentum (heavy damping had hidden this on the v1 plant). Single-
# seed diagnostics at that plant: 0.1 -> 0.04 by block end, 0.3 -> 0.80;
# making the penalty symmetric instead changed nothing (0.08). Every v1-plant
# run predates this and is not return-comparable.
DIST_WEIGHT = 0.3
# Per-step effort cost -EFFORT_WEIGHT |a|^2, applied only where a config asks
# for it (AblationConfig.effort_weight); the paradigm schedules do. See
# step_reward for the measurements behind the value.
EFFORT_WEIGHT = 0.1
PROGRESS_WEIGHT = 1.0
PROGRESS_PENALTY_SCALE = 2.0

# Cost of running into a wall, scaled by the IMPULSE the contact delivers --
# task.wall_impulse, normalised by task.MAX_WALL_IMPULSE so the weight is the
# cost of a head-on contact AT THE SPEED CAP and stays meaningful if MASS or
# V_MAX are retuned.
#
# IMPULSE RATHER THAN PROXIMITY, deliberately. sample_pos_goal draws goals
# uniformly over the whole grid, so goals sit near walls and the agent HAS to
# approach them; a proximity penalty would fight the task. Scaling by impact
# speed leaves a slow approach free and prices only the slam, which is the
# behaviour that actually destroys the transition.
#
# WHY AT ALL. Measured on main_frozen seed 0, 64.6% of the trained policy's
# steps end in a wall contact, against 2.2% under the action distribution
# data.generate_sequences uses -- a 29x mismatch. Wall contact zeroes velocity
# ON THE CLIPPING AXIS ONLY, so it is an axis-aligned nonlinearity that the
# circuit's training data almost never contains, and sample_pos_goal's
# docstring signs it off as costing "realism-of-path only". At two thirds of
# all transitions that is not a cosmetic cost.
#
# NOT CALIBRATED AGAINST BEHAVIOUR. 1.0 makes a full-speed impact cost ~5x the
# largest single progress term (~0.19) and a fifth of SUCCESS_BONUS, which is
# a starting point and not a measured one. Watch curve["wall_impulse"]: if
# wall contact does not fall, raise it; if episodes start ending short of
# goals near the boundary, lower it.
#
# SET TO 0.0 TO DISABLE. Any nonzero value changes the objective, so runs made
# under it are NOT comparable with anything already in sac_runs/.
WALL_IMPULSE_WEIGHT = 1.0

def _phase_ids(per_family=2):
    """Two contexts from each family, evenly spaced within it.

    DERIVED, NOT HARDCODED.  A literal list of context ids goes stale silently
    every time the context table is resized -- [1, 6, 8, 12] pointed off the
    end of an 8-context table and would have raised at import, which is the
    lucky case; a list that still indexed VALID but different contexts would
    simply have run the wrong experiment.
    """
    ids = []
    for f in range(len(T.FAM_NAMES)):
        mem = [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] == f]
        step = max(1, len(mem) // per_family)
        ids += mem[::step][:per_family]
    return ids


PHASE_CONTEXT_IDS = _phase_ids()

PHASE_LABELS = [T.C_LABEL[i] for i in PHASE_CONTEXT_IDS]
N_PHASE_FAMILIES = len(PHASE_CONTEXT_IDS)

# ── WHICH contexts the RL task uses: a task variant (2026-09-18) ────────────
#
# z's context code is a ring of directions per family (rotations in one
# 2-D plane, reflections in an orthogonal one), so WITHIN a family the code is
# exactly additive while ACROSS families it is not (family is which plane
# carries the energy -- a quadratic). And under one action direction a
# reflection is indistinguishable from some rotation, which the default set
# (two of each family) exposes the agent to on every reach. The prediction
# this knob exists to test: with rotations only, the frozen circuit should
# retain without replay.
#
# The set is MODULE STATE, not a parameter threaded through every function:
# PHASE_CONTEXT_IDS / PHASE_LABELS / N_PHASE_FAMILIES are read at call time by
# the observers, the oracle embed, the block schedule and the probes, so
# rebinding them once per process is enough -- and a spawned worker
# re-imports this module, so the choice travels in AblationConfig.phase_set
# and run_sac applies it first thing. Each set has four contexts, so
# N_PHASE_FAMILIES, the schedule length and every observation width are
# unchanged; only the identities differ. Run names get _rot / _refl.
# "mixed" is the derived default (two per family, _phase_ids); "<family>s" is
# all four of one family. Names follow task.FAM_NAMES, so a family change in
# task.py changes the knob with it.
_PHASE_FAMILY = {f"{name}s": name for name in T.FAM_NAMES}
PHASE_SETS = ("mixed",) + tuple(_PHASE_FAMILY)
# run-name suffix per set: "" for the default, the family's short name else
_SHORT = {"rotation": "rot", "reflection": "refl", "curl": "curl", "drift": "drift"}
PHASE_SET_SUFFIX = {"mixed": "", **{s: f"_{_SHORT.get(name, name)}" for s, name in _PHASE_FAMILY.items()}}
DEFAULT_PHASE_SET = "mixed"


def phase_ids_for(phase_set: str) -> list:
    if phase_set not in PHASE_SETS:
        raise KeyError(f"unknown phase_set {phase_set!r}; expected one of {PHASE_SETS}")
    if phase_set == DEFAULT_PHASE_SET:
        return _phase_ids()
    fam = T.FAM_NAMES.index(_PHASE_FAMILY[phase_set])
    return [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] == fam]


def apply_phase_set(phase_set: str) -> list:
    """Rebind the module's RL context set. Idempotent; returns the ids."""
    global PHASE_CONTEXT_IDS, PHASE_LABELS, N_PHASE_FAMILIES
    ids = phase_ids_for(phase_set)
    if ids != list(PHASE_CONTEXT_IDS):
        PHASE_CONTEXT_IDS = ids
        PHASE_LABELS = [T.C_LABEL[i] for i in ids]
        N_PHASE_FAMILIES = len(ids)
    return ids


# ── The block SCHEDULE: our cycle, or COIN's own paradigms (2026-09-18) ────
#
# "cycle" is what every run so far used and REMAINS THE DEFAULT, built by the
# same rule as before: the phase contexts in order, n_phase_ep each,
# phase_reps times round. Savings there is faster relearning on a revisit,
# with the other perturbations as the washout -- which confounds recognition
# with interference from, and transfer out of, the blocks in between.
#
# The COIN paper (Heald, Lengyel & Wolpert 2021) tests contextual inference
# with the classic adaptation paradigms. These are those, one reach per trial,
# block lengths from the standard design (Smith et al. 2006):
#
#   spontaneous_recovery   P0 | P+ (long) | P- (short) | CHANNEL
#       A single-state learner ends P- adapted to P- and stays there. A
#       contextual learner drifts back toward P+ during the channel, because
#       with no error to discriminate contexts its belief relaxes toward the
#       more frequent one. HERE that relaxation comes from the experience
#       prior over EC's start state (ExperienceBank) and, for the inferred
#       oracles, from the visit-count prior (_ContextPosterior.count_prior).
#   evoked_recovery        ... | CHANNEL (short) | P+ (two reaches) | CHANNEL
#   savings_washout        P0 | P+ | P0 (washout) | P+
#       Savings with a NULL washout: recognition shows as the second P+
#       starting higher and rising faster; weight-space transfer as merely
#       steeper. Nothing transfers out of a null block.
#
# THE CHANNEL. In a force channel the hand is guided straight to the target
# whatever force it produces, so the learner sees a successful reach and no
# error, and the force on the channel wall IS the expressed memory. Here a
# channel reach clamps the kinematics to the straight path at cruising speed
# (channel_step), which makes CB's prediction error what it is under a
# correct model of a guided hand; MC is frozen (no replay writes, no
# updates); the observers get no evidence and only relax; and the action is
# the readout -- the actor's DETERMINISTIC mean, so that with MC frozen the
# only thing that can move the channel trace is the context signal. Read the
# FIRST action (feedforward): the reach mean is dominated by braking.
#
# P+ is the smallest positive rotation in the table and P- its negation
# (task.ROT_ANGLES: +-30), so both must be in the run's context set: pass
# --contexts rotations. P0 is the null field. Savings-washout is the
# exception: its baseline is the TRAINED rot +30 and its P+ rot +60
# (savings_pair; schedule_contexts holds every schedule's roles).
SCHEDULES = ("cycle", "spontaneous_recovery", "evoked_recovery", "savings_washout")
DEFAULT_SCHEDULE = "cycle"
SCHEDULE_SUFFIX = {"cycle": "", "spontaneous_recovery": "_sr", "evoked_recovery": "_er",
                   "savings_washout": "_savw"}
# Block lengths in reaches. The human design is 50 / 125 / 15 / 150, calibrated
# to human adaptation rates, where 15 P- trials drive expressed adaptation
# through zero. This learner is slower -- P+ took 25-40 reaches to reach high
# success and 125 to get its first angle to ~-15 of -30 deg -- so after a
# 15-reach P- block MC had barely turned (baseline first angle +3 deg) and the
# channel read out "P+ with a little P-": nothing to recover from. Lengthened
# 2026-09-18 (user): P- long enough to adapt, P+ kept ~4x longer so the visit
# ratio (which sets the recovery plateau through the bank's count term and the
# observer's count prior) stays lopsided, baseline 100 so MC is competent in
# the null field before P+. The criterion for "P- long enough" is that its
# first-angle trace crosses zero toward the P- level before the channel.
# P+ 200 -> 300 (2026-09-18, user): with replay of the P0 block every model
# adapts to P+ more slowly (~0.3 of the full effect at reach 200) and `main`
# slowest (0.53); the paradigm needs every model fully adapted before P-, so
# the block is long rather than the P0 block short. P+ : P- is now 6 : 1,
# near the human 125 : 15.
COIN_BLOCK_EP = {"baseline": 100, "pplus": 300, "pminus": 50, "channel": 150,
                 "evoke_after": 10, "evoke": 1, "washout": 75}
NULL_LABEL = "P0"
CHANNEL_LABEL = "channel"


def coin_pair():
    """(P+ id, P- id): the LARGEST rotation and its negation -- +-60 on the
    current table.

    Was the smallest (+-30) until 2026-09-19. The change is about how much
    reward there is for knowing the context at all. A context-blind feedback
    law still commands cos(theta) of its force along the intended direction,
    so it converges for any |theta| < 90 and merely takes a curved path; the
    only question is what that costs. Measured with PD laws whose gains are
    tuned separately for each condition, so the blind one is a ceiling rather
    than a straw man: at 30 degrees perfect feedforward compensation is worth
    7% of return (4.50 -> 4.87) and both controllers finish in ~13 steps; at
    60 degrees it is worth 33% (3.28 -> 4.92) because the blind law needs
    21-25 steps. Success is 100% either way, so this is a cost in efficiency,
    not in failing. +-60 is also the best-separated pair in z (centroid
    distance 3.04 against 1.71 for +-30) and is already in the supervised
    training set, so no checkpoint changes."""
    rots = [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] == 0]
    pplus = max((c for c in rots if T.C_PARAM[c] > 0), key=lambda c: T.C_PARAM[c])
    pminus = next(c for c in rots if T.C_PARAM[c] == -T.C_PARAM[pplus])
    return pplus, pminus


def savings_pair():
    """(base id, P+ id) of the savings-washout schedule: P- of the other
    schedules (rot -30) as the BASELINE and P+ (rot +30) as the perturbation.
    A trained context as the baseline (2026-09-18, user's choice) rather than
    the null field: the circuit never saw the null, so z there is an
    extrapolation, and the observers need no null hypothesis to tell the two
    blocks apart. First tried as rot +30 / rot +60; that pair is the one the
    circuit cannot resolve -- the two z centroids sit 0.70 apart at cosine
    0.96 (main seed 0), so a retrieved P+ state read no differently from a
    base state -- while +-30 are 1.71 apart at cosine -0.92 (2026-09-18). The
    savings contexts are now the recovery contexts, so one pair serves every
    paradigm."""
    pplus, pminus = coin_pair()
    return pminus, pplus


def schedule_contexts(schedule):
    """The roles the contexts play in a schedule: {"base", "pplus", "pminus"}
    -> context id or None (None for the null field as base; None for a role
    the schedule has no context for). ONE place for what P+, P- and the
    baseline mean, read by the block builder, the logged belief / probe
    columns, the CLI check and the plot's adaptation index."""
    if schedule == "savings_washout":
        base, pplus = savings_pair()
        return {"base": base, "pplus": pplus, "pminus": None}
    pplus, pminus = coin_pair()
    return {"base": None, "pplus": pplus, "pminus": pminus}


def schedule_required_ids(schedule):
    """The context ids a schedule's blocks use (the null field excluded)."""
    return [c for c in schedule_contexts(schedule).values() if c is not None]


def schedule_blocks(schedule, n_phase_ep, phase_reps, warmup_episodes, block_ep=None):
    """The blocks run_sac runs, in order. Each carries context_id (None = null
    field), label, n_ep, log, phase_idx, occurrence, family_idx (row of the
    oracle embed; None for the null field), learn and channel. "cycle"
    reproduces the pre-schedule block list exactly (tests pin it)."""
    if schedule not in SCHEDULES:
        raise KeyError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    blocks = []
    if warmup_episodes > 0:
        blocks.append({"context_id": None, "label": "null field", "n_ep": warmup_episodes,
                       "log": False, "phase_idx": -1, "occurrence": -1, "family_idx": 0,
                       "learn": True, "channel": False})
    if schedule == "cycle":
        for phase_idx in range(N_PHASE_FAMILIES * phase_reps):
            family_idx = phase_idx % N_PHASE_FAMILIES
            blocks.append({"context_id": PHASE_CONTEXT_IDS[family_idx],
                           "label": PHASE_LABELS[family_idx], "n_ep": n_phase_ep,
                           "log": True, "phase_idx": phase_idx,
                           "occurrence": phase_idx // N_PHASE_FAMILIES,
                           "family_idx": family_idx, "learn": True, "channel": False})
        return blocks

    ep = {**COIN_BLOCK_EP, **(block_ep or {})}
    roles = schedule_contexts(schedule)
    for c in schedule_required_ids(schedule):
        if c not in PHASE_CONTEXT_IDS:
            raise KeyError(
                f"schedule {schedule!r} needs {T.C_LABEL[c]!r} in the run's context set "
                f"{[T.C_LABEL[i] for i in PHASE_CONTEXT_IDS]}; pass --contexts rotations")

    def ctx_block(cid, prefix):
        if cid is None:
            return dict(context_id=None, label=NULL_LABEL, family_idx=None, learn=True, channel=False)
        return dict(context_id=cid, label=f"{prefix} {T.C_LABEL[cid]}",
                    family_idx=PHASE_CONTEXT_IDS.index(cid), learn=True, channel=False)

    # the baseline block: the null field (P0) or, for savings, a trained
    # context (label "base rot +30" -- see savings_pair)
    PB = ctx_block(roles["base"], "base")
    PP = ctx_block(roles["pplus"], "P+")
    CH = dict(context_id=None, label=CHANNEL_LABEL, family_idx=None, learn=False, channel=True)
    if schedule == "spontaneous_recovery":
        PM = ctx_block(roles["pminus"], "P-")
        seq = [(PB, ep["baseline"]), (PP, ep["pplus"]), (PM, ep["pminus"]), (CH, ep["channel"])]
    elif schedule == "evoked_recovery":
        PM = ctx_block(roles["pminus"], "P-")
        seq = [(PB, ep["baseline"]), (PP, ep["pplus"]), (PM, ep["pminus"]),
               (CH, ep["evoke_after"]), (PP, ep["evoke"]),
               (CH, max(ep["channel"] - ep["evoke_after"], 1))]
    else:                                                   # savings_washout
        seq = [(PB, ep["baseline"]), (PP, ep["pplus"]), (PB, ep["washout"]), (PP, ep["pplus"])]
    seen = {}
    for phase_idx, (proto, n_ep) in enumerate(seq):
        occ = seen.get(proto["label"], 0); seen[proto["label"]] = occ + 1
        blocks.append({**proto, "n_ep": int(n_ep), "log": True, "phase_idx": phase_idx,
                       "occurrence": occ})
    return blocks


# THE PER-REACH READOUT PROBE (2026-09-18). A training reach's first action
# is one start geometry, one exploration sample and -- with the experience
# prior -- one draw of EC's start state: index sd ~1 per reach. Under a
# paradigm schedule every logged reach is therefore followed by an 8-target
# centre-out probe (eval_center_out) of the block's plant with the actor's
# tanh-mean, the observer carried from the run's current posterior and a
# fresh bank draw per target, and the MEAN first-action angle over the 8
# directions is logged as curve["probe_first_angle"] -- the expectation over
# the start-state prior rather than one sample of it. Only the first action
# is wanted, so the probe runs READOUT_PROBE_STEPS steps per target (the
# forward pass costs eight actor calls per reach). Side-effect free by
# eval_center_out's contract: the probe has its own rng, seeded by reach, and
# never touches the run's; the training trajectory of a paradigm run is
# unchanged by it. Off under "cycle": those runs stay bit-identical.
READOUT_PROBE_STEPS = 1


def channel_step(pos, goal):
    """The channel plant: the straight path to the goal at cruising speed,
    whatever the action. Returns (pos_new, vel_new, wall_impulse=0.0)."""
    gd = np.asarray(goal, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
    d = float(np.linalg.norm(gd))
    if d < 1e-8:
        return np.asarray(pos, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0
    step = min(T.V_MAX * T.DT, d)
    vel_new = (gd / d * (step / T.DT)).astype(np.float32)
    pos_new = (np.asarray(pos, dtype=np.float32) + vel_new * T.DT).astype(np.float32)
    return pos_new, vel_new, 0.0


# ── The experience prior over EC's start state ─────────────────────────────
#
# EC's hidden state is zeroed at every reach, so its start-of-reach belief is
# one fixed point and MC's first action is whatever the last gradient steps
# made it: spontaneous recovery is impossible by construction. COIN's start-
# of-trial belief is instead the previous posterior through a transition
# prior that leaks toward contexts in proportion to how often they were
# visited. The analogue: a bank of end-of-reach hidden states, and at reach
# start a draw from it with probability
#
#     p(i)  propto  (1 + age_i / tau) ^ (-alpha)
#
# in episodes: ONE power-law recency kernel (2026-09-18, user's call; it
# replaced exp(-age/tau) + a uniform term). Recency and count-based recall
# are the same kernel here. Its heavy tail is what recovers: in a channel
# block nothing enters the bank (a channel reach only holds the state it was
# dealt), every stored state ages together, and a power law's weight RATIOS
# move as ages grow -- the just-finished P- states are dealt first and, as
# they age, the 300 P+ states behind them regain weight, continuously, with
# no switch. An exponential cannot do this: exp(-age/tau) ratios are
# invariant to a common age shift, so it would stay pinned on the newest
# entries for ever, which is why the old mixture needed its uniform term and
# why that mixture flipped over a few tau. Every reach with evidence enters,
# the null field included. The bank is EC's memory, distinct from the replay
# buffer (MC's): clear_buffer_on_switch never clears it.
EC_PRIORS = ("zero", "experience", "carry")
# "carry" (2026-09-19, the default for an EC config): PROPAGATE the belief
# instead of redrawing it. Each reach starts from the PREVIOUS reach's end
# state, except with probability EC_PRIOR_HAZARD, when it starts from a state
# drawn uniformly over the bank -- COIN's sticky transition prior, which keeps
# the last posterior and otherwise leaks toward the global visit distribution,
# one reach at a time instead of one step.
#
# "experience" redrew from the bank on EVERY reach, and that is what made main
# lag at a block switch. Measured: EC is NOT sticky -- a state from the wrong
# context is overridden by genuine feedback within two steps (decoder says the
# true context with p = 0.01 -> 0.56 -> 0.97 over steps 0, 1, 2) -- so the cost
# of a wrong deal is one extra step of learning but the WHOLE first action,
# which is the entire readout. Redrawing kept dealing stale states long after
# the switch (at tau 1, alpha 1.5 the share of P- reaches still handed a P+
# state was 75% at reach 2, 46% at reach 5, 30% at reach 10), so main's early
# adaptation index reported the bank's mixture rather than what MC had learned,
# and both constants moved the readout directly. Carrying makes the previous
# reach's end state -- already the true context after one reach -- the default,
# so the lag matches the observer's single reach.
#
# In a CHANNEL block nothing corrects the state (the plant is clamped, so an
# end-of-reach state there is about the clamp, not the context), so the carry
# is left where it was and only a leak moves it. The dealt context then drifts
# from the just-finished block toward the bank's counts as 1 - (1-h)^k, which
# is spontaneous recovery with a cleaner shape than ageing gave.
EC_PRIOR_HAZARD = 0.02    # per reach: ~1 leak per 50, so a 150-reach channel
                          # has drifted ~95% of the way by its end
# tau and alpha control different things. tau sets how fast the newest states
# take over at a BLOCK START (the bank only learns a new context one finished
# reach at a time, so main's expressed adaptation lags coin's carried belief
# by however long that takes); alpha sets how the tail returns old states in
# the CHANNEL (the rebound's time course), and the channel barely moves with
# tau. Share of dealt states from the new block at reach 2 / 5 / 10, and P-
# share in the spontaneous-recovery channel at reach 1 / 50 / 150:
#   tau 3, alpha 2:    0.23 / 0.55 / 0.74    0.94 / 0.54 / 0.33
#   tau 1, alpha 2:    0.39 / 0.73 / 0.86    0.97 / 0.55 / 0.33
#   tau 1, alpha 1.5:  0.25 / 0.54 / 0.70    0.88 / 0.43 / 0.27   <- default
#   tau 1, alpha 1.25: 0.18 / 0.42 / 0.58    0.77 / 0.36 / 0.24
#   tau 1, alpha 3:    0.62 / 0.92 / 0.98    1.00 / 0.76 / 0.47
# The tail is the same in both columns, so a heavier tail (SMALLER alpha)
# returns P+ faster in the channel AND deals more old-context states at a
# block start; tau only trades off the first few reaches. tau 1 (2026-09-18,
# user): the last reach is the strongest prior -- the per-reach analogue of
# the observer carrying its last posterior. alpha 1.5 (same day, user): a
# faster channel rebound, bought with 0.30 fewer new-block deals at reach 5.
EC_PRIOR_TAU = 1.0        # episodes
EC_PRIOR_ALPHA = 4.5      # power-law exponent
PRIOR_SRC_NONE, PRIOR_SRC_NULL = -2, -1     # curve["prior_src"] codes


class ExperienceBank:
    def __init__(self, tau=EC_PRIOR_TAU, alpha=EC_PRIOR_ALPHA, device="cpu"):
        self.tau, self.alpha, self.device = float(tau), float(alpha), device
        self.h, self.episode, self.context = [], [], []

    def __len__(self):
        return len(self.h)

    def add(self, h, episode, context_id):
        self.h.append(np.asarray(h.detach().cpu().numpy(), dtype=np.float32).reshape(-1).copy())
        self.episode.append(int(episode))
        self.context.append(PRIOR_SRC_NULL if context_id is None else int(context_id))

    def probs(self, ep_now):
        age = np.maximum(ep_now - np.asarray(self.episode, dtype=np.float64), 0.0)
        w = (1.0 + age / self.tau) ** (-self.alpha)
        return w / w.sum()

    def sample(self, rng, ep_now=None):
        """(h0 as a (1, EC_HIDDEN) tensor, source context code, source episode).
        An empty bank yields the zero state, as with no prior. ep_now defaults
        to just after the newest entry (what a probe between reaches sees)."""
        if not self.h:
            return torch.zeros(1, EC_HIDDEN, device=self.device), PRIOR_SRC_NONE, -1
        if ep_now is None:
            ep_now = max(self.episode) + 1
        i = int(rng.choice(len(self.h), p=self.probs(ep_now)))
        return self._at(i)

    def sample_uniform(self, rng):
        """The same, drawn UNIFORMLY over stored states -- so each context is
        represented by how often it was visited and not by how recently. What
        a "carry" leak lands on: COIN's transition prior jumps to the global
        visit distribution, and a context reached 300 times holds 300 states."""
        if not self.h:
            return torch.zeros(1, EC_HIDDEN, device=self.device), PRIOR_SRC_NONE, -1
        return self._at(int(rng.integers(len(self.h))))

    def _at(self, i):
        h0 = torch.tensor(self.h[i], dtype=torch.float32, device=self.device).unsqueeze(0)
        return h0, self.context[i], self.episode[i]

PHASE_REPS = 3
N_PHASE_EP = 150

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "sac_runs")
FIG_DIR = os.path.join(BASE_DIR, "figures", "sac")


def run_dir(config_name, seed):
    d = os.path.join(RUN_DIR, config_name, f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def fig_dir(config_name, seed):
    d = os.path.join(FIG_DIR, config_name, f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    return d


# ── Ablation config ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class AblationConfig:
    """One point in the ablation sweep.

    use_ec/use_cb/freeze/arm generalize to all six presets (see
    sweep_ablations.CONFIG_PRESETS) rather than one flag per preset --
    cb_only_end_to_end is just (use_ec=False, use_cb=True, freeze=False), not
    new plumbing.
    """
    name: str
    use_ec: bool = False
    use_cb: bool = False
    freeze: bool = False
    use_oracle: bool = False
    oracle_kernel: str = "identity"
    """Geometry of the oracle's context code -- see ORACLE_KERNELS.

    "identity" is the near-orthogonal draw the oracle has always used; "z" and
    "z_centered" draw from the circuit's own measured pairwise cosines instead,
    so the oracle knows the context perfectly in the SHAPE the circuit's code
    has. Only meaningful with use_oracle=True; the geometry kernels read the
    (arm, arm_seed) checkpoint, which those configs otherwise leave at its
    default.
    """
    schedule: str = DEFAULT_SCHEDULE
    """The block schedule: "cycle" (the phase contexts in order, phase_reps
    times; the default and every run before 2026-09-18, unchanged) or one of
    COIN's paradigms -- "spontaneous_recovery", "evoked_recovery",
    "savings_washout" -- built from P+/P- (the +-30 rotations), the null
    field and channel blocks. See SCHEDULES. Run name: _sr / _er / _savw."""
    ec_prior: str = "zero"
    ec_prior_tau: float = EC_PRIOR_TAU
    ec_prior_alpha: float = EC_PRIOR_ALPHA
    ec_prior_hazard: float = EC_PRIOR_HAZARD
    """EC's start-of-reach hidden state. "zero" is every run before the
    paradigms. "carry" (the default for an EC config since 2026-09-19)
    propagates the previous reach's end state and leaks to a uniform bank
    draw with probability ec_prior_hazard; run name _carry. "experience"
    redraws from the bank every reach under the power-law recency kernel
    (1 + age/tau)^-alpha; run name _prior. Needs use_ec. See EC_PRIORS for
    why carrying replaced redrawing."""
    relabel_z: bool = False
    """Hindsight relabelling of stored transitions (default off): each
    transition's observation and next-observation carry the z from the END
    of its reach rather than the z at the step. The analogue of COIN
    crediting a trial's error to contexts by their POSTERIOR responsibility;
    without it, the first steps of a reach begun under the wrong prior teach
    MC the new context's compensation in the old context's z-slot. Only
    changes what the replay buffer stores. Run name: _relabel."""
    observer_null_context: Optional[bool] = None
    """Whether the inferred oracle's hypothesis set includes the NULL FIELD
    (identity plant, zero code) beside the phase contexts. None (the default)
    resolves to yes under the COIN paradigm schedules and no under "cycle"
    (observer_has_null), so every run before 2026-09-18 is unchanged. Without
    it a P0 block leaves the belief split 50/50 between P+ and P-. Only the
    plant observer ("bayes") can carry it. No run-name suffix: the saved
    config records it. Savings-washout runs on a trained baseline
    (savings_pair) and resolves to no."""
    observer_count_prior: float = 0.0
    """Weight (0..1) of a visit-count term in the inferred oracles' transition
    prior: with 0 (the default) the jump target is uniform, as before; with w
    it is (1-w) uniform + w counts/sum, where counts accumulate each reach's
    posterior (_ContextPosterior.note_visit). COIN's learned global
    transition distribution. Needs an inferred oracle_source. Run name:
    _cprior."""
    phase_set: str = DEFAULT_PHASE_SET
    """Which contexts the RL task cycles through: "mixed" (two per family,
    derived by _phase_ids -- six contexts under the three-family table) or
    "<family>s" (all four of one family). See PHASE_SETS. Applied to the
    module's context set by run_sac, so it reaches spawned workers; the run
    name carries the family's suffix (_rota / _curl / _drif)."""
    oracle_source: str = "given"
    """Where the oracle's context comes from.

      "given"  the true context, handed over -- every oracle config so far.
      "bayes"  the exact Bayesian posterior over the phase contexts, inferred
               step by step from the realised force (see BayesObserver). The
               code is the posterior-weighted mixture of the "given" codes, so
               at certainty the two channels are identical. This is the
               finite-context ideal observer -- the COIN baseline: it prices
               inference against the one-hot, and EC's readout against `main`.
               The belief persists across episodes and blocks; the observer is
               not told when the context switches. During probes it starts
               uniform in each probed context, so probe retention includes the
               cost of re-inferring the context.
      "cb"     the same posterior, but scored through the frozen CB with each
               context represented by its z-centroid (see CBObserver). Needs
               the (arm, arm_seed, action_mode) checkpoint. coin - coin_cb is
               the cost of CB's model error; coin_cb - main is EC's readout.

    Only meaningful with use_oracle=True.
    """
    oracle_snr: float = 1.0
    """Fraction of the oracle channel's variance that is CONTEXT. 1.0 is the
    noiseless code every oracle config has used, bit for bit.

    A FRACTION, NOT A RATIO, and the difference decides where rungs belong.
    This field is s = V / (V + P) over signal power V and noise power P, so it
    is bounded in [0, 1] and s = 1 means P = 0 -- an INFINITE signal-to-noise
    ratio, not a ratio of one. The ratio convention is r = V / P = s / (1 - s),
    and snr_from_ratio converts.

    Space rungs geometrically in r, never uniformly in s. For a linear readout
    separating K contexts d'^2 is proportional to r, so uniform steps in s
    crowd together at the low end and leave everything above r = 4 unsampled:
    the first ladder run here was s = 0.61 / 0.40 / 0.19, which reads like a
    wide sweep and is r = 1.56 / 0.67 / 0.23 -- a factor of 7, all of it at or
    below parity, against an oracle at infinity. Every one of those rungs came
    back at baseline retention, which bounds the cliff below but says nothing
    about where above r = 1.56 it sits.

    THIS IS THE CONTROL oracle_kernel COULD NOT BE. oracle_geom matches z's
    centroid pairwise cosines and nothing else, and the docstring above once
    claimed that left "nothing structural" between them. It leaves the
    property that decides learnability: a gp_code table has ONE code per
    context and therefore zero within-context variance, while z's context
    signal rides on 5-25x more moment-to-moment variation. Measured on main
    seed 0 over the four phase contexts, between/total variance of z is 0.15
    on the distribution model.py fitted it on and 0.04 under a trained policy,
    against exactly 1.00 for the oracle's code.

    So this adds noise to a PERFECT context label until its SNR matches the
    circuit's, and nothing else changes -- same geometry, same width, same
    magnitude (see oracle_noise_scale, which renormalises). Sweeping it turns
    "is the circuit's disadvantage its SNR?" into a dose-response curve with
    both circuit arms marked on the axis.

    THE NOISE LIVES IN THE CODE'S OWN SUBSPACE, rank n_contexts-1, which is
    the conservative choice and the reason this is parameterised by SNR alone.
    Any component orthogonal to the centroid differences is removable by a
    linear projection, so a first layer can and does filter it: measured, z's
    within-context noise is only 1.26x more aligned with the discriminative
    subspace than isotropic would be, and the SNR a linear readout can reach
    INSIDE that subspace is 0.61 as-fitted (0.19 under a trained policy), far
    above the raw 0.15/0.04. Putting all the noise in-subspace makes the raw
    and readable SNR the same number, so a run at oracle_snr=0.61 is matched
    to what MC can actually extract from the as-fitted circuit rather than to
    a statistic it can partly discard.
    """
    oracle_noise_iso: bool = False
    """Draw the noise ISOTROPICALLY in the full code space instead of inside
    the code's discriminative subspace.

    THE CONTROL FOR THE SUBSPACE CHOICE, which turned out to be load-bearing.
    Confining noise to the rank K-1 subspace was argued above as conservative:
    an orthogonal component is linearly filterable, so leaving it out keeps the
    raw and readable SNR equal. But it also puts every noise sample in the span
    of the other contexts' codes, so the first-layer update for context c,
    which runs along (m_c + n), moves the readout of every other context too.
    That is gradient interference, not lost information, and it predicts harm
    that barely improves as the code becomes more decodable.

    Which is what the ladder showed. At r >= 9 a single step already decodes
    the context at 1.000, and retention sat at 0.09 against 0.33 for the same
    code with no noise -- a plateau, not an information limit. Isotropic noise
    at matched RAW snr spreads the same power over all 32 dimensions, so only
    ~3/32 of it lands in the subspace where it can interfere. If retention
    returns under this flag, the mechanism is interference; if it does not,
    something else distinguishes an exactly-constant code.

    NOTE the two are not matched on readable SNR, and cannot be: that is the
    manipulation. Compare at equal raw snr and report both numbers.
    """
    oracle_noise_rho: float = 0.0
    """AR(1) temporal correlation of the oracle's noise, within an episode.

    0.0 is iid, which is the EASY case: independent draws average out over a
    reach, so a policy can integrate them away and SNR looks less binding than
    it is. z's in-subspace noise is correlated at +0.60 as-fitted and +0.84
    under a trained policy -- slower, and correspondingly harder to average --
    so a matched control has to carry it. Reset per episode, exactly as EC's
    hidden state is.
    """
    cb_gain: float = 1.0
    """Scalar on the CB block at MC's input. 1.0 leaves it untouched; see
    augment_obs on why this is not the same manipulation as mute_cb_to_mc."""
    arm: str = "main"          # only meaningful when use_ec/use_cb
    arm_seed: int = 0          # which checkpoint seed to load when freeze=True
    action_mode: str = "gp"
    """Which supervised circuit to freeze: the one fitted on the GP family that
    draws a different action process per sequence ("gp", the default -- see
    task's GP section and model.DEFAULT_ACTION_MODE), or on task.next_action's
    single AR(1) process ("ar1"; the legacy spelling "fitted" is accepted).

    A CONFIG FIELD RATHER THAN A FLAG, so it lands in save_agent's blob and in
    the run directory name, and a finished run says on its face which circuit
    it was driven with. The two live at different checkpoint paths under
    different convention stamps, so a mismatch raises in model.load rather than
    quietly producing a result about the wrong network -- but only if the mode
    actually reaches it, which is what this field is for.

    Meaningless unless freeze=True: an end-to-end config trains its own."""
    ec_readout: str = M.DEFAULT_READOUT
    rate_beta: float = M.RATE_BETA
    noise_sigma: float = M.NOISE_SIGMA
    train_contexts: str = M.DEFAULT_TRAIN_CONTEXTS
    """Which EC readout the frozen checkpoint has -- "noisy" (fixed-noise
    channel, noise sd noise_sigma, rate cost rate_beta; the default since
    2026-09-18 at sigma 0.3, beta 1e-3) or "tanh" (deterministic) -- and
    which contexts it was fitted on (train_contexts, model.TRAIN_CONTEXT_SETS).
    Together with arm/arm_seed/action_mode these are the circuit
    (model.CircuitSpec, see circuit_of); all land in the saved blob. Only
    meaningful with freeze=True."""
    center_ec: bool = False
    """Subtract a fixed offset from z_ec before it reaches MC.

    REPRESENTATIONALLY FREE, DYNAMICALLY NOT. A first layer can absorb any
    constant input offset into its bias -- W(z - m) + b = Wz + (b - Wm) -- so
    centering changes nothing about which policies are expressible. What it
    changes is the GRADIENT. The first layer's gradient is dL/d(pre) . z^T, so
    a large component shared by every context's z puts a large shared
    component in every context's gradient, and a step taken in one context
    then moves the others. z's context means carry 71% of their energy in one
    common vector, which is the regime that produces near-total gradient
    sharing (first-layer overlap ~0.95 for
    collinear codes against ~0.00 for orthogonal ones).

    So this is a test of the MECHANISM, not a fix bolted on: it removes the
    shared component while preserving every bit of discriminative information.

    The offset is calibrated once per run from the frozen circuit and then
    held fixed, so the observation distribution stays stationary -- a running
    mean would make stored transitions disagree with current ones, the same
    class of problem as stale replay features. Only defined for freeze=True;
    an end-to-end circuit moves under it.
    """
    normalize_circuit: bool = False
    """Standardize BOTH circuit blocks per unit before MC reads them:
    (x - mean) / std, from statistics calibrated once on the frozen circuit.

    THE POINT IS SCALE, NOT JUST OFFSET. Measured against the reference repo
    on an identical metric (mean rms / std rms over a random-action rollout):

                          state into circuit    z_ec           z_cb
        reference          2.10 / 1.79       0.30 / 0.44    0.38 / 0.33
        this, v1_abspos   21.27 / 7.22       0.49 / 0.37    5.92 / 1.19
        this, v3_rnn       0.41 / 7.22       0.34 / 0.60    1.58 / 0.88

    Against the kinematics MC reads beside it (std 1.79 there, 0.75 here),
    CB entered the reference 5x BELOW kinematics and enters here 1.6x above
    -- a ~9x flip in the varying part alone. That is S_SCALE=15, which this
    repo needs because DT=0.04 (the reference's DT=1.0 made its states O(1)
    unaided) and which CB's hidden layers then inherit and hand to MC.

    Centering alone does not fix it, and v3_scaled_rnn is the proof: it
    zero-centers the state almost exactly (0.3% DC) and still emits a z_cb
    that is 76% DC at 4x the reference's magnitude, because centering touches
    position's offset and not velocity's x15 scale.

    Unit variance rather than a tuned target: kinematics sits at std 0.75, so
    a standardized block enters at 1.33x it -- parity, with no fitted
    constant. Frozen-only and calibrated once, for the same stationarity
    reason as center_ec.
    """
    center_cb: bool = False
    """Subtract a fixed per-unit offset from the CB block before it reaches MC.

    The CB analogue of center_ec, and for the same reason -- but the case for
    it is far stronger here, because CB's block is more DC than signal:

        v1_abspos (main_frozen)   block mean rms 5.80  std rms 1.16   96% DC
        v3_scaled_rnn (centered)  block mean rms 1.58  std rms 0.85   78% DC

    A constant shared by every observation is representationally free -- the
    first layer absorbs it into its bias -- but it is not free to LEARN. The
    first-layer gradient is dL/d(pre) . x^T, so a component every context's
    input shares puts a matching component in every context's gradient, and a
    step taken in one context moves all the others. That is the mechanism
    center_ec was written to test, and CB's channel carries it 96/4.

    NOTE WHY INPUT CENTERING IS NOT ENOUGH. center_position (a variant key)
    zero-centers the state EC and CB both read, and it does cut the block
    from 5.80 to 1.58 -- but 78% of what is left is still constant, because
    that part is generated INSIDE CB by its own biases and by z_ec's own DC,
    not inherited from the input. Only a readout-side offset removes it.

    Calibrated once per run from the frozen circuit and then held fixed, so
    the observation distribution stays stationary; frozen-only, like
    center_ec, since an end-to-end CB moves under a fixed offset.
    """
    cb_noise: bool = False
    """Replace CB's channel into MC with matched-moment Gaussian noise.

    THE CONTENT CONTROL. mute_cb_to_mc and cb_gain both showed that CB's
    block hurts and that its magnitude is not why: muting takes lag 0 from
    0.40 to 0.80, while scaling it down to init-matched (cb_gain=0.31) only
    reaches 0.44. That leaves "what CB says" as the culprit but does not say
    which property of what it says.

    Two very different things are still confounded. Either ANY high-variance
    channel that varies with nothing the policy needs is harmful -- a
    distractor the network must spend capacity suppressing -- or it is
    something SPECIFIC to CB's signal: that it is state-correlated, so the
    actor can lock onto it and read the field off a channel that will mean
    something else in the next block.

    Noise separates them. The substituted block is drawn iid per step with
    CB's own per-dimension mean and std (see calibrate_cb_stats), so it
    enters the actor at the same width and the same scale and carries
    nothing. If lag 0 stays near 0.40 the harm is generic to the extra
    variance; if it recovers toward 0.80 the harm is in CB's structure.

    Iid is the weaker of the two possible controls -- it destroys the
    temporal and cross-dimension correlation along with the information, so
    a recovery says "structure matters" without saying which structure. A
    shuffle control (real block, wrong timestep) would separate those, and is
    the follow-up if this one recovers.
    """
    mute_cb_to_mc: bool = False
    """Zero CB's channel INTO MC while leaving CB itself running.

    The lesion is on the readout, not the structure: CB still predicts, so
    e_t = s_next - s_pred still feeds EC and z_t is unchanged. What MC loses
    is only the 64-d cb block, which stays present and correctly shaped so the
    network is architecturally identical to the un-muted config -- same obs
    width, same first-layer parameter count, same init_by_source
    decomposition. Only its contents are zero.

    That separates two things the existing configs conflate. use_cb=False
    removes CB entirely, which also silences EC's error input and so changes
    z; this changes what the CONTROLLER reads and nothing else.
    """
    clear_buffer_on_switch: bool = CLEAR_BUFFER_ON_PHASE_SWITCH
    """Wipe the replay buffer at every phase switch, so no transition from a
    previous context is ever sampled while a new one trains.

    THIS IS THE KNOB THAT SEPARATES RETENTION FROM SAVINGS, and the two
    versions of the experiment differ in nothing else:

      False  replay carries the just-left context into the next block --
             0.35-0.42 of every minibatch there. Tagged by the context code,
             those transitions keep the old policy alive in MC's weights, and
             the run shows RETENTION: probe success on earlier contexts held at
             ~0.4 for the frozen circuit, ~0.9 for the one-hot oracle.
      True   the buffer is wiped, the frozen circuit's retention collapses to
             baseline (0.03), and what is left is SAVINGS: revisits start at
             the same level as first visits and climb ~0.3 further within the
             block, in every seed and every context, with the revisit curve
             indistinguishable from the replay condition's. Faster
             readaptation after complete washout -- the classic result,
             produced with no replay support. The oracle keeps ~60% of its
             retention here and never had rate savings to lose.

    A CONFIG FIELD, not the module flag alone, so the two versions can sit in
    sweep_ablations side by side and run in parallel: a spawned worker
    re-imports sac.py, so a module-level patch made in the parent never
    reaches it, and a control run made that way would silently be the wrong
    condition. The field travels in the pickled config. Default is the module
    constant, so every existing run and preset is unchanged.
    """
    effort_weight: float = 0.0
    """Per-step action cost -w |a|^2 in step_reward. 0.0 is every run before
    2026-09-19; the paradigm schedules set EFFORT_WEIGHT. It exists to make
    knowing the context worth reward -- see coin_pair and step_reward."""
    rehearsal_fraction: float = 0.0
    """Fraction of each batch drawn from blocks BEFORE the one being learned
    (ReplayBuffer.rehearsal). 0.0 -- every run before 2026-09-18 -- leaves the
    recency rule alone, so how much of the batch is the current block depends
    on how long that block has run. The paradigm schedules set
    PARADIGM_REHEARSAL, which fixes both shares. Inert under clear_buffer_on_
    switch, where there is nothing older to draw. No run-name suffix: it is
    the paradigms' default and the saved config records it."""
    replay_units: str = "transitions"
    """What the replay buffer counts: "transitions" (capacity BUFFER_CAPACITY,
    recency in steps -- every existing run) or "episodes" (capacity
    BUFFER_CAPACITY_EPISODES, recency in episodes, equal mass per episode).

    See the comment at BUFFER_CAPACITY_EPISODES for why: in transition units
    a competent agent's short episodes buy it a longer memory, so how much
    previous-context data gets replayed is an OUTCOME of performance rather
    than a property of the design, and the retention version's contrast
    between configs inherits that.

    A CONFIG FIELD because it changes within-block learning for every config
    (short successful episodes are no longer under-weighted), so it must be
    set IDENTICALLY on a parent and its *_clearbuf twin or their contrast is
    no longer one knob -- tests pin that. Default keeps every existing run
    bit-identical; the CLI's --episode-buffer selects the other and suffixes
    the run name so the two never share a directory.
    """


@dataclass(frozen=True)
class ObsSpec:
    """How a config's observation is laid out.

    The circuit channels are projected while the observation is BUILT, so
    unlike the earlier bottleneck design there is no second width here: what
    the rollout stores is what the trunk reads.

      ec       z_t, LATENT_DIM wide, passed through unchanged
      cb       CB_PROJ_DIM wide, h2 already through the fixed random cb_proj

    Kinematic and oracle blocks are tracked separately even though they are
    adjacent, because init_by_source weights each INPUT SOURCE by its own
    width and they are different sources.
    """
    kin: int                        # [pos, vel, goal]
    oracle: int = 0                 # dense context code, or 0
    ec: int = 0                     # LATENT_DIM, or 0 when the config has no EC
    cb: int = 0                     # CB_PROJ_DIM, or 0 when the config has no CB

    @property
    def base(self) -> int:
        return self.kin + self.oracle

    @property
    def raw_dim(self) -> int:
        return self.base + self.ec + self.cb

    # the trunk reads exactly what the buffer stores
    feature_dim = raw_dim

    @property
    def feature_widths(self) -> list:
        """The trunk's input sources, in order, as widths. Zeros included so
        the decomposition lines up with the slice layout for absent blocks."""
        return [self.kin, self.oracle, self.ec, self.cb]


def obs_spec(config: AblationConfig, cb_proj_dim: int = CB_PROJ_DIM) -> ObsSpec:
    return ObsSpec(
        kin=SDIM + 2,
        oracle=ORACLE_EMBED_DIM if config.use_oracle else 0,
        ec=LATENT_DIM if config.use_ec else 0,      # z_t, the same code CB gets
        cb=cb_proj_dim if config.use_cb else 0,     # h2 through cb_proj
    )


def obs_dim(config: AblationConfig) -> int:
    """Width of the observation the rollout builds and the buffer stores."""
    return obs_spec(config).raw_dim


def circuit_state(pos, vel) -> np.ndarray:
    """[pos, vel] in the units EC and CB were trained on.

    PER CHANNEL, matching model.build exactly: position by POS_SCALE and
    velocity by VEL_SCALE, each 1/(that channel's own bound), so both arrive on
    [-1, 1].  The old uniform S_SCALE is gone with the plant that needed it --
    a single scalar can only lift both channels, never correct the ratio
    between them, and the ratio was the problem (position ran 97x larger than a
    step's displacement).

    This is INPUT CONDITIONING, not physics: the environment, the reward, the
    goal test and the ACTOR's own observation all stay in raw units.  Only what
    EC and CB see is scaled.

    GETTING IT WRONG IS SILENT AND EXPENSIVE, which is the whole reason this
    lives in one function that rollout_z also drives through.  When
    sac.py once fed raw positions where model.build fed scaled ones, z's
    context decode fell from 1.00 to 0.28 (chance 0.17) with nothing raising an
    error, and both frozen arms looked exactly like the no-circuit baseline --
    a wiring bug that reads as a result about the circuit.
    """
    pos = np.asarray(pos, dtype=np.float32) * T.POS_SCALE
    vel = np.asarray(vel, dtype=np.float32) * T.VEL_SCALE
    return np.concatenate([pos, vel], axis=0).astype(np.float32)


def build_base_obs(pos, vel, goal) -> np.ndarray:
    """[pos, vel, goal] -- deliberately WITHOUT the previous action.

    The actor is meant to be reactive: any integration over time should come
    from EC's recurrent state, not from history handed to the policy in its
    own observation. Including prev_action would smuggle in a one-step memory,
    and worse, it completes the (action, resulting force) pair that identifies
    the transformation -- with it, a memoryless MLP recovers the context from a
    single observation (~65% over 6 contexts vs 17% chance), which is what made
    every ablation config tie.

    task.py's rotation and reflection families are zero-mean over actions and
    norm-preserving, so per its docstring NO statistic of the observations
    alone identifies them -- you have to know the action you took. Withholding
    it therefore leaves those families reachable only through the cerebellar
    error channel, which is the asymmetry this experiment is about. Velocity
    stays: curl acts on it (F = a + k*R(90)*v), so the task is not
    controllable without it.
    """
    return np.concatenate([pos, vel, goal], axis=0).astype(np.float32)


def oracle_one_hot(phase_idx: int, n_phases: int = N_PHASE_FAMILIES) -> np.ndarray:
    """The sparse code. Kept as the control for the dense one -- see
    make_oracle_embed for why the dense code is what the oracle now uses."""
    v = np.zeros(n_phases, dtype=np.float32)
    v[int(phase_idx)] = 1.0
    return v


def oracle_code(phase_idx, embed: np.ndarray) -> np.ndarray:
    """The oracle's channel: a dense row of `embed`, or zeros when there is no
    context to encode (the null-field warmup block)."""
    if phase_idx is None:
        return np.zeros(embed.shape[1], dtype=np.float32)
    return np.asarray(embed[int(phase_idx)], dtype=np.float32)


def augment_obs(base_obs, config: AblationConfig, z_ec=None, z_cb=None,
                 oracle_vec=None) -> np.ndarray:
    """base_obs plus whichever circuit channels the config exposes.

    `z_ec` is EC's readout at t -- the same tensor CB is conditioned on.
    `z_cb` is CB's last hidden layer from the step t-1 forward pass, already
    through the fixed random cb_proj; one step stale by necessity, since a CB
    pass needs a_t and a_t is chosen from this very observation.
    """
    parts = [base_obs]
    if config.use_oracle:
        parts.append(oracle_vec)
    if config.use_ec:
        parts.append(z_ec)
    if config.use_cb:
        # muted here rather than at the projection, because this is the one
        # point BOTH the observation and the one-step lookahead pass through
        #
        # cb_gain SEPARATES MAGNITUDE FROM CONTENT, which mute_cb_to_mc
        # confounds. Measured on a trained main_frozen actor, CB contributes
        # 7.87 to the first-layer pre-activation against kinematics' 1.45 --
        # 5.4x -- so the hidden units' ReLU pattern is set mostly by CB and
        # kinematics is a perturbation on top. Muting removes that magnitude
        # AND everything CB was saying; a gain removes only the magnitude. If
        # scaling rescues learning the problem is scale, if it does not then
        # CB's CONTENT is what hurts.
        cb_block = np.zeros_like(z_cb) if config.mute_cb_to_mc else z_cb
        if config.cb_gain != 1.0:
            cb_block = cb_block * config.cb_gain
        parts.append(cb_block)
    return np.concatenate(parts, axis=0).astype(np.float32)


# ── Goal-reaching wrapper around task.env_step ───────────────────────────

def sample_pos_goal(rng):
    """Start and goal both uniform over the whole grid, independently.

    Distance is now free: it ranges from GOAL_RADIUS to the diagonal (~2.83),
    with the density of |g - p| for two uniform points in a square -- most mass
    around 0.7-1.5, a thin tail at both ends.  Direction stays exactly uniform
    by symmetry of the square, which is the property that matters for the
    ablation: it interacts with the context (under rotation the required action
    is the desired direction rotated by -theta, and per task.py reflection's
    angular offset DEPENDS on action direction), so a direction-biased sample
    would make some contexts systematically easier and confound the comparison.

    TWO REJECTIONS, both isotropic about the start, so neither biases
    direction -- the property this sampler exists to protect.

    The first is degenerate-only: a goal drawn inside GOAL_RADIUS of the start
    would count as success at step 0, giving a zero-length episode with the
    full SUCCESS_BONUS.  That is not a distance constraint on where goals may
    sit, it is the definition of a reachable episode.

    The second is MIN_GOAL_DIST (2026-09-19), a floor on how short a reach may
    be.  A reach of 0.2 finishes in two or three steps, so it is nearly all
    first action and carries almost no adaptation to measure, while still
    paying a full SUCCESS_BONUS and adding its own variance to every
    per-block mean.  At 0.5 it drops the shortest 15% of draws and moves the
    median distance only 1.10 -> 1.16, so the distance distribution is
    otherwise intact.  It does not threaten the step budget: a competent
    controller takes 12 steps at the median and 20 at the 95th percentile
    either side of the floor.

    WHAT THIS GIVES UP, so it is on the record rather than rediscovered.  The
    ring this replaces existed to kill a nuisance variable: with independent
    sampling the start distance previously ranged over 11x, and since reward
    accumulates over the trajectory, that alone drove ~75% of the variance in
    episode return -- against an effect being measured worth ~0.1 sd.  Under
    the relative-progress term the damage is smaller than it was (the
    telescoped progress contribution grows like ln(d0/GOAL_RADIUS), not
    linearly in d0), but it is no longer zero: return, steps and dist_norm now
    partly reflect how far the goal happened to be.  If arm-vs-arm contrasts
    look noisy, CONDITION ON OR REGRESS OUT the start distance before
    concluding anything -- it is logged per episode for exactly that reason.

    Wall contact also returns -- the ring kept trajectories off the walls at
    every radius tested.  That is no longer an input-convention hazard (the
    rollout reads vel_sim directly instead of finite-differencing positions,
    which diverges from the simulator exactly at a velocity clip), so it costs
    realism-of-path only.
    """
    # the grid is [-POS_LIM, POS_LIM] on both axes, not [0, GRID_SIZE]
    while True:
        pos = rng.uniform(-POS_LIM, POS_LIM, size=2).astype(np.float32)
        goal = rng.uniform(-POS_LIM, POS_LIM, size=2).astype(np.float32)
        d = np.linalg.norm(goal - pos)
        if d > GOAL_RADIUS and d >= MIN_GOAL_DIST:
            return pos, goal


# The null field: kind 2 is drift, F = a + param*(bx,by), so param=0 gives
# F = a exactly -- an identity transformation. task.CONTEXTS has no such entry
# (it exists to be inferred from, and an identity context carries nothing to
# infer), so it lives here as explicit parameters rather than a context id.
# THE NULL FIELD IS A SENTINEL, NOT A CONTEXT.  It used to be
# (2, 0.0, 0.0, 0.0) -- "drift with zero gain", i.e. the identity -- back when
# a context was a (kind, param, bx, by) tuple.  There is no identity context
# now (an identity context would sit at effect size zero and be
# indistinguishable from no perturbation), so task.context_matrices takes
# c=None to mean F = a.
NULL_FIELD = None


def env_step_params(pos, vel, action, context_id, rng=None, return_impulse=False):
    """One plant step.  `context_id` is now the whole context -- the factorial
    replaced the (kind, param, bx, by) tuple with a single cell index, and
    task.context_matrices is the only place it becomes matrices.

    PASS AN RNG.  task.CMD_NOISE is what makes context inference require
    history at all; without it an ideal observer identifies the context in two
    steps.  rng=None is the noiseless plant and is for tests only -- an agent
    evaluated without the noise it trained under is on a different plant.
    """
    if return_impulse:
        pos_new, vel_new, _F, imp = T.env_step(pos, vel, action, context_id, rng,
                                                return_impulse=True)
        return pos_new, vel_new, imp
    pos_new, vel_new, _F = T.env_step(pos, vel, action, context_id, rng)
    return pos_new, vel_new


def goal_env_step(pos, vel, action, context_id, rng=None, return_impulse=False):
    return env_step_params(pos, vel, action, context_id, rng, return_impulse)


def step_reward(pos_new, goal, pos_prev, wall_impulse=0.0, action=None,
                effort_weight=0.0):
    """Absolute distance plus asymmetric RELATIVE progress, plus the bonus:

        progress = (d_prev - d) / d_prev            fraction of the gap closed
        gain     = PROGRESS_WEIGHT * (PROGRESS_PENALTY_SCALE if progress < 0
                                       else 1.0)
        r        = -DIST_WEIGHT * d  +  gain * progress   [+ SUCCESS_BONUS]

    RELATIVE rather than absolute progress. Closing 10% of the remaining gap
    pays the same whether the goal is 4 units away or 0.4, which is what makes
    this scale-free. Two consequences:

      - It compresses the start-distance nuisance. Summed along a trajectory,
        sum(dd/d) is approximately ln(d_start/d_end), so episode return grows
        with the LOG of the starting distance instead of linearly with it --
        the spread across sampled starts narrows from ~27x to ~6x. Return is
        ~92% driven by start distance under the absolute form, so this is a
        real improvement to measurability -- and now the ONLY thing doing that
        work, since sample_pos_goal draws start and goal independently and no
        longer holds the distance fixed.
      - The denominator is safe: run_sac ends the episode as soon as
        d < GOAL_RADIUS, so d_prev is always >= GOAL_RADIUS at a step that
        gets scored, bounding the term by DT*|v|/GOAL_RADIUS ~ 0.19. The eps
        floor below is belt-and-braces for anyone lowering GOAL_RADIUS to 0.

    EFFORT. With effort_weight > 0 the step also pays -w |a|^2 (AblationConfig.
    effort_weight, EFFORT_WEIGHT; 0 and no action = every run before
    2026-09-19). It is there because a context-blind law must command
    1/cos(theta) times the force to get the same useful component and must
    then cancel the perpendicular part, so effort is one of the few costs that
    falls on it and not on a compensating one. Measured on tuned PD laws, it
    barely separates them at +-30 (7% of return -> 12% at w = 0.4, which costs
    everyone a third of their return) but sharpens +-60 a lot: 33% at w = 0,
    39% at 0.1, 50% at 0.2. Kept at 0.1: at 0.4 the per-step effort cost
    approaches DIST_WEIGHT * d and standing still starts to look attractive,
    which is the collapse DIST_WEIGHT was raised to prevent.

    ASYMMETRIC penalty. With PROGRESS_PENALTY_SCALE > 1, moving away costs more
    than the same movement toward the goal earns. That is the point: an agent
    that knows the transformation aims correctly and never pays it, while a
    context-blind one must probe and does -- so it sharpens exactly the
    feedforward signal the ablation is trying to measure.

    THE COST, worth watching. Asymmetry makes wrong movement cost more than
    NOT MOVING, so it can teach the agent to stall when uncertain -- and
    stalling suppresses precisely the probing needed to discover the context.
    The absolute term is what prevents that: standing still still pays
    -DIST_WEIGHT*d every step. Freezing is discouraged as long as

        DIST_WEIGHT * d  >  PROGRESS_WEIGHT * PROGRESS_PENALTY_SCALE * (worst
                            fractional step backwards)

    which at the current weights holds comfortably far from the goal but not
    close to it, where the fractional term is largest and the distance cost
    smallest. If runs show episodes ending with the agent parked short of the
    goal, this ratio is the first thing to check.

    NOT FARMABLE. Unlike max(0, d_prev - d), which pays for the forward half of
    an oscillation and nothing for the return leg, both of these forms lose on
    a round trip. Relative progress is anti-farming even when symmetric: the
    retreat leg divides by the smaller d_prev, so its magnitude exceeds the
    approach leg's and a cycle nets negative (second order in step size). The
    asymmetry only deepens that.

    The absolute term keeps what the sigmoid variants lacked: it never
    saturates, and is <= 0 everywhere, 0 only at the goal.
    """
    d = float(np.linalg.norm(pos_new - goal))
    d_prev = float(np.linalg.norm(pos_prev - goal))
    done = d < GOAL_RADIUS

    progress = (d_prev - d) / max(d_prev, 1e-8)
    gain = PROGRESS_WEIGHT * (PROGRESS_PENALTY_SCALE if progress < 0.0 else 1.0)

    reward = -DIST_WEIGHT * d + gain * progress
    if effort_weight and action is not None:
        reward -= float(effort_weight) * float(np.dot(action, action))
    if WALL_IMPULSE_WEIGHT and wall_impulse:
        reward -= WALL_IMPULSE_WEIGHT * float(wall_impulse) / T.MAX_WALL_IMPULSE
    if done:
        reward += SUCCESS_BONUS
    return reward, done, d


# ── EC/CB augmentation ────────────────────────────────────────────────────

def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def build_ec_cb(config: AblationConfig, device):
    """Construct (ec, cb), honoring use_ec/use_cb/freeze.

    freeze=True loads model.load(arm, arm_seed) -- the existing supervised
    checkpoints -- and freezes whichever of ec/cb the config actually wants
    (model.load always returns both; drop the unused one). freeze=False
    builds fresh, trainable modules with no checkpoint (there is nothing to
    load for cb_only_end_to_end, since a context-blind CB was never trained
    supervised)."""
    if config.freeze:
        spec = circuit_of(config)
        loaded = M.load(config.arm, config.arm_seed, spec)
        if loaded is None:
            raise FileNotFoundError(
                f"no checkpoint for arm={config.arm!r} seed={config.arm_seed} "
                f"circuit {spec}; expected {M.ckpt_path(config.arm, config.arm_seed, spec)}")
        ec, cb = loaded
        ec, cb = ec.to(device), cb.to(device)
        if config.use_ec:
            freeze_module(ec)
        else:
            ec = None
        if config.use_cb:
            freeze_module(cb)
        else:
            cb = None
        return ec, cb

    ec = (M.EC(no_current=(config.arm == "main"),
               cell_type=M.CELL_TYPE).to(device)
          if config.use_ec else None)
    cb = (M.CB(M.CB_ACTIVATION).to(device)
          if config.use_cb else None)
    return ec, cb


def ec_step_or_zero(ec, s_tm1, a_tm1, s_t, e_t, h, arm, device):
    """One EC step, or a zero latent when no EC exists (cb_only_end_to_end
    and the no-augmentation configs). Returning zeros rather than None keeps
    cb_predict's call signature uniform -- model.CB always expects a
    LATENT_DIM z argument, context-blind or not."""
    if ec is None:
        return torch.zeros(s_t.shape[0], LATENT_DIM, device=device), h
    no_current = (arm == "main")
    parts = [s_tm1, a_tm1, e_t] if no_current else [s_tm1, a_tm1, s_t, e_t]
    x = torch.cat(parts, dim=-1)
    z_t, h_new = ec.step(x, h)
    return z_t, h_new


def ec_channel(z_t, config, ec_offset, ec_std=None):
    """z_ec as MC receives it: raw, centered, or standardized.

    Used by the rollout, the one-step lookahead and the eval probe so none of
    the three can disagree about what the channel is.
    """
    z = z_t.squeeze(0).detach().cpu().numpy()
    if config.normalize_circuit:
        if ec_offset is None or ec_std is None:
            raise ValueError("normalize_circuit needs ec stats; see "
                             "calibrate_ec_stats")
        return ((z - ec_offset) / ec_std).astype(np.float32)
    if config.center_ec and ec_offset is not None and len(ec_offset):
        z = z - ec_offset
    return z


def cb_channel(z_cb, config, cb_stats, rng):
    """CB's block as MC receives it: the real projection, or matched noise.

    Drawn fresh on every call, so the observation stored in the buffer and the
    lookahead's next-observation get INDEPENDENT samples -- which is right,
    they are different timesteps of a channel whose whole premise is that it
    carries no information across them.

    Used by the rollout, the lookahead and the eval probe so none of the three
    can end up reading a different channel than the others.
    """
    if z_cb is None or not (config.cb_noise or config.center_cb
                            or config.normalize_circuit):
        return z_cb
    if cb_stats is None:
        raise ValueError("cb_noise/center_cb/normalize_circuit need cb_stats; "
                         "call calibrate_cb_stats once per run and thread it through")
    mean, std = cb_stats
    if config.cb_noise:
        return (mean + std * rng.standard_normal(mean.shape[0])).astype(np.float32)
    if config.normalize_circuit:
        return ((z_cb - mean) / std).astype(np.float32)
    return (z_cb - mean).astype(np.float32)


def augmented_obs_step(ec, config, h, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
                        pos, vel_obs, goal, oracle_vec, device, ec_offset=None,
                        cb_stats=None, rng=None, ec_std=None,
                        oracle_noise=None):
    """One EC step plus the observation built from the resulting channels.

    Note which signal goes where: EC is fed the raw prediction error `e_t`;
    the ACTOR is fed z_t -- the very tensor CB is about to be conditioned on
    -- together with CB's previous-step hidden layer through the fixed random
    `cb_proj`. Both projections happen HERE rather than inside the networks,
    matching the reference: one projection is then shared by the actor and
    both critics, and the projected feature is what lands in the buffer.

    Shared by run_sac's training loop and eval_center_out so the two cannot
    drift apart. Returns (obs, z_t, h_new).
    """
    z_t, h_new = ec_step_or_zero(ec, s_tm1, a_tm1, s_t, e_t, h, config.arm, device)
    z_cb = None
    if config.use_cb:
        with torch.no_grad():
            z_cb = (cb_h2 @ cb_proj).squeeze(0).cpu().numpy()
        z_cb = cb_channel(z_cb, config, cb_stats, rng)
    obs = augment_obs(
        build_base_obs(pos, vel_obs, goal), config,
        z_ec=(ec_channel(z_t, config, ec_offset, ec_std) if config.use_ec else None),
        z_cb=z_cb,
        oracle_vec=(oracle_noise.apply(oracle_vec) if oracle_noise is not None
                    else oracle_vec),
    )
    return obs, z_t, h_new


def cb_predict(cb, s_t, a_t, z_t, device):
    """CB's next-state prediction AND the hidden state the actor reads.

    Returns (s_pred, h2) where h2 is the second post-ReLU layer -- the one
    immediately upstream of CB's linear readout, so s_pred is a linear
    function of it. Gradients are kept: the end-to-end configs need s_pred
    differentiable for ec_cb_loss. h2 is detached where it enters the
    observation, not here.
    """
    if cb is None:
        return (torch.zeros(s_t.shape[0], SDIM, device=device),
                torch.zeros(s_t.shape[0], CB_HIDDEN, device=device))
    s_pred, _h1, h2 = cb(s_t, a_t, z_t, return_hidden=True)
    return s_pred, h2


# ── Networks (CleanRL-style, adapted to a config-dependent obs width) ────

def _as_spec(spec) -> ObsSpec:
    """Accept a bare width as well as an ObsSpec, so a network can still be
    built as Actor(8) in a test that has no circuit to describe."""
    return spec if isinstance(spec, ObsSpec) else ObsSpec(kin=int(spec))


def init_by_source(linear: nn.Linear, widths) -> None:
    """Re-initialize an input layer so each INPUT SOURCE is weighted by its own
    dimensionality: the columns reading a source of width n get std 1/sqrt(n).

    WHY NOT THE DEFAULT. nn.Linear initializes every column from one bound set
    by the TOTAL fan-in, so a source's contribution to the pre-activation grows
    with how many dimensions it happens to span. For main_frozen the trunk sees
    6 kinematic dims beside 64 circuit dims, so at initialization the circuit
    supplied an order of magnitude more pre-activation variance than position,
    velocity and goal combined -- MC swamped by EC and CB before learning
    anything. Per-source scaling makes each source contribute
    n * (1/sqrt(n))^2 = 1 unit of variance per unit of input power, independent
    of its width.

    Normal rather than uniform, so "standard deviation 1/sqrt(n)" is exact
    rather than off by the sqrt(3) that a uniform bound carries.

    SCOPE. This equalizes DIMENSIONALITY, not magnitude: a source whose entries
    are larger still contributes more, which is intended -- position in grid
    units genuinely is a bigger signal than a tanh latent. Applied to
    input-facing layers only; the hidden layers have a single source, where the
    default is already fan-in scaled and well tested for ReLU stacks.

    THE 1/sqrt(n_sources) FACTOR. Per-source scaling alone would make every
    source contribute a FULL unit of pre-activation variance instead of
    sharing one budget, so the total would grow as sqrt(number of sources) --
    measured, main_frozen's first hidden layer went from std 0.73 to 3.14 on
    three sources, a 1.7x inflation over the sqrt(3) expected. That breaks the
    criterion the default init exists to satisfy (pre-activation variance
    ~= input variance, so scale is preserved with depth) and shows up
    downstream as inflated Q values and a different effective learning rate.
    Dividing by n_sources restores the total while leaving the RATIOS between
    sources -- the whole point of the per-source scaling -- untouched.

    Zero widths are allowed and skipped, so the caller can pass a fixed
    decomposition whether or not a config has that block.
    """
    if sum(widths) != linear.weight.shape[1]:
        raise ValueError(f"source widths {list(widths)} sum to {sum(widths)}, "
                         f"but the layer reads {linear.weight.shape[1]}")
    n_sources = sum(1 for width in widths if width)
    with torch.no_grad():
        start = 0
        for width in widths:
            if width:
                nn.init.normal_(linear.weight[:, start:start + width], mean=0.0,
                                std=1.0 / math.sqrt(n_sources * width))
            start += width


class Actor(nn.Module):
    def __init__(self, spec, action_dim=ACTION_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        spec = _as_spec(spec)
        self.backbone = nn.Sequential(
            nn.Linear(spec.feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # init_by_source(self.backbone[0], spec.feature_widths)
        self.fc_mean = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        h = self.backbone(obs)
        mean = self.fc_mean(h)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return y_t, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, spec, action_dim=ACTION_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        spec = _as_spec(spec)
        self.net = nn.Sequential(
            nn.Linear(spec.feature_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # the action is a source in its own right, appended after the features
        # init_by_source(self.net[0], spec.feature_widths + [action_dim])

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


@dataclass(slots=True)
class SACBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class SACTrainer:
    """CleanRL-style SAC: critic update -> actor update -> auto-tuned alpha
    -> Polyak target update. Inlined here as one class (no BaseTrainer/
    package split) to match this repo's flat-module convention."""

    def __init__(self, actor, qf1, qf2, qf1_target, qf2_target,
                 actor_optimizer, q_optimizer, alpha_optimizer=None, *,
                 alpha=0.2, gamma=GAMMA, tau=TAU, target_entropy=None,
                 autotune=True, device=DEVICE, grad_clip_norm=GRAD_CLIP_NORM):
        self.actor, self.qf1, self.qf2 = actor, qf1, qf2
        self.qf1_target, self.qf2_target = qf1_target, qf2_target
        self.actor_optimizer, self.q_optimizer = actor_optimizer, q_optimizer
        self.gamma, self.tau, self.autotune = gamma, tau, autotune
        self.target_entropy = target_entropy
        self.device = device
        self.grad_clip_norm = grad_clip_norm

        self.log_alpha: Optional[torch.Tensor]
        if autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = alpha_optimizer or torch.optim.Adam([self.log_alpha], lr=LR_Q)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
        self.alpha = float(alpha)

    @property
    def current_alpha(self) -> float:
        if self.autotune and self.log_alpha is not None:
            return float(self.log_alpha.exp().item())
        return self.alpha

    def act(self, obs):
        return self.actor.get_action(obs)

    def compute_q_target(self, batch: SACBatch) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(batch.next_observations)
            q1t = self.qf1_target(batch.next_observations, next_actions)
            q2t = self.qf2_target(batch.next_observations, next_actions)
            min_q = torch.min(q1t, q2t) - self.current_alpha * next_log_pi
            return batch.rewards.flatten() + (1.0 - batch.dones.flatten()) * self.gamma * min_q.view(-1)

    def soft_update_targets(self) -> None:
        for p, tp in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        for p, tp in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def update(self, batch: SACBatch) -> dict:
        next_q_value = self.compute_q_target(batch)
        q1 = self.qf1(batch.observations, batch.actions).view(-1)
        q2 = self.qf2(batch.observations, batch.actions).view(-1)
        qf1_loss = F.mse_loss(q1, next_q_value)
        qf2_loss = F.mse_loss(q2, next_q_value)
        q_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        # clip_grad_norm_ returns the PRE-clip total norm. Recorded because
        # clipping rescales by the total across all parameters, so one input
        # block with large gradients shrinks the update to every other block
        # too -- and whether that is happening is invisible from the losses.
        q_gnorm = float(nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.grad_clip_norm if self.grad_clip_norm else float("inf")))
        self.q_optimizer.step()

        pi, log_pi, _ = self.actor.get_action(batch.observations)
        qf1_pi = self.qf1(batch.observations, pi)
        qf2_pi = self.qf2(batch.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.current_alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gnorm = float(nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.grad_clip_norm if self.grad_clip_norm else float("inf")))
        self.actor_optimizer.step()

        alpha_loss_val = 0.0
        if self.autotune and self.target_entropy is not None:
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = float(alpha_loss.detach().item())

        self.soft_update_targets()

        return {
            "qf1_loss": float(qf1_loss.detach().item()),
            "qf2_loss": float(qf2_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha": float(self.current_alpha),
            "alpha_loss": alpha_loss_val,
            "q_target_mean": float(next_q_value.detach().mean().item()),
            "log_pi_mean": float(log_pi.detach().mean().item()),
            "q_gnorm": q_gnorm,
            "actor_gnorm": actor_gnorm,
        }


def ec_cb_loss(pred: torch.Tensor, tgt: torch.Tensor,
               ) -> torch.Tensor:
    """Same weighting as model.weighted_loss (position/DT^2 + alpha*velocity)
    but on a flat [B, SDIM] batch of single transitions rather than
    model.py's [B, T, SDIM] sequences -- there's no trailing timestep to drop
    here, every sampled transition already has a well-defined next state.

    alpha comes from M.ALPHA -- one convention now, so there is nothing to
    reweight per-run.  (Historically this read the weight
    the halves (v3_whitened_a625) would otherwise train its end-to-end circuit
    under a different objective than the one its frozen checkpoints were fitted
    with, and nothing would raise."""
    e = (pred - tgt) ** 2
    alpha = M.ALPHA
    return (e[..., :2] / DT ** 2).mean() + alpha * e[..., 2:].mean()


class ECCBTrainer:
    """Joint EC/CB training for the two end-to-end configs, run alongside
    each SAC update. Owns only cb.parameters() when ec is None
    (cb_only_end_to_end)."""

    def __init__(self, ec, cb, optimizer, *, arm="main",
                 grad_clip_norm=GRAD_CLIP_NORM, device=DEVICE):
        self.ec, self.cb = ec, cb
        self.optimizer = optimizer
        self.arm = arm
        self.grad_clip_norm = grad_clip_norm
        self.device = device

    def recompute_error(self, s_tm1, a_tm1, s_t, h_ec_prev, e_t_stored):
        """Rebuild e_t at update time so the gradient can reach CB THROUGH the
        error EC consumes, not only through CB's final prediction.

        WHY. model.rollout's loop is un-detached: e_t = s_t - s_pred where
        s_pred is the live tensor from t-1, so pretraining backpropagates
        through the error into CB. Replaying the stored e_t instead makes it a
        constant and severs that path, which trained the end-to-end configs on
        a structurally different objective from the one that produced the
        frozen checkpoints -- end_to_end vs main_frozen then conflated "online
        vs pretrained" with "with vs without the error gradient".

        HOW, without storing or sampling anything new. h_ec_prev is the hidden
        state AFTER the step at t-1, so ec.readout(h_ec_prev) is exactly
        z_{t-1}: the latent CB was conditioned on when it made the prediction
        that e_t was measured against. Every other term is already in the
        buffer. Gradient therefore reaches CB and EC's readout, and through
        ec_step_or_zero the GRU cell as well.

        Deliberately NOT the reference repo's version, which recomputed the
        error as cb(s_t, a_t, z_prev) - s_next and fed it to the EC step for
        the same transition. That puts s_next inside the latent CB then uses
        to predict s_next. Conditioning on (s_tm1, a_tm1) keeps the causal
        order model.rollout has.

        TWO CASES DEFER TO THE STORED VALUE:
          t == 0    there is no previous transition, so the error is undefined
                    and model.rollout forces it to zero. Detected by s_tm1
                    being exactly zero, which is how the rollout initialises
                    it; a real state cannot be, since circuit_state scales a
                    ring-sampled position that never reaches the origin.
          ec None   cb_only_end_to_end has no EC, so ec_step_or_zero ignores
                    e_t entirely and there is nothing for the recomputed value
                    to feed.

        This is still a ONE-STEP unroll against pretraining's 40. That is
        inherent to replaying single transitions and is not fixed here.
        """
        if self.ec is None or self.cb is None:
            return e_t_stored
        # EC.read, not ec.readout: the noisy channel has no `readout` module
        # (it has `mu`), and read() applies whichever readout the circuit has
        z_tm1 = self.ec.read(h_ec_prev)
        e_recomputed = s_t - self.cb(s_tm1, a_tm1, z_tm1)
        first_step = (s_tm1.abs().sum(dim=-1, keepdim=True) == 0)
        return torch.where(first_step, e_t_stored, e_recomputed)

    def update(self, batch) -> dict:
        s_tm1, a_tm1, s_t, a_t, s_next, e_t_stored, h_ec_prev = batch
        e_t = self.recompute_error(s_tm1, a_tm1, s_t, h_ec_prev, e_t_stored)
        z_t, _ = ec_step_or_zero(self.ec, s_tm1, a_tm1, s_t, e_t, h_ec_prev, self.arm, self.device)
        pred = self.cb(s_t, a_t, z_t)
        loss = ec_cb_loss(pred, s_next)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm:
            params = list(self.cb.parameters()) + (list(self.ec.parameters()) if self.ec is not None else [])
            nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        self.optimizer.step()

        return {"ec_cb_loss": float(loss.detach().item())}


# ── Replay buffer ─────────────────────────────────────────────────────────

class ReplayBuffer:
    """Recency-weighted replay: p(e) proportional to exp(-dt / gamma_t), mixed
    with a uniform floor. Ported from the previous iteration's
    TimeTDEPrioritizedReplayBuffer (_ablations_sweep.py).

    Also stores the extra raw-transition fields (s_tm1/a_tm1/s_t/a_t/s_next/
    e_t/h_ec_prev) alongside the usual SAC fields -- one buffer class for all
    six configs, since the extra fields are cheap and only the end-to-end
    configs' ECCBTrainer ever reads them.
    """

    _FIELDS = ("observations", "actions", "rewards", "next_observations", "dones",
                "s_tm1", "a_tm1", "s_t", "a_t", "s_next", "e_t", "h_ec_prev",
                "timesteps", "abs_tdes", "episodes")

    def __init__(self, capacity=BUFFER_CAPACITY, *, gamma_t=PRIORITY_GAMMA_T,
                 gamma_td=PRIORITY_GAMMA_TD, mix=PRIORITY_MIX, seed=0,
                 units="transitions", capacity_episodes=BUFFER_CAPACITY_EPISODES,
                 gamma_ep=PRIORITY_GAMMA_EP):
        from collections import deque
        if units not in REPLAY_UNITS:
            raise ValueError(f"units must be one of {REPLAY_UNITS}, got {units!r}")
        self.units = units
        # the fixed rehearsal budget (PARADIGM_REHEARSAL): fraction of each
        # batch drawn from BEFORE the current block. 0 = pure recency.
        self.rehearsal = 0.0
        self.block_start_ep = None
        self.capacity = capacity
        self.capacity_episodes = int(capacity_episodes)
        self.gamma_t = float(gamma_t)
        self.gamma_ep = float(gamma_ep)
        self.gamma_td = float(gamma_td)
        self.mix = float(mix)
        self.rng = np.random.default_rng(seed)

        # In transition units the deques evict for us, exactly as before. In
        # episode units eviction is by WHOLE episode (see add), so the deques
        # are unbounded and the count is kept in _ep_counts.
        maxlen = capacity if units == "transitions" else None
        for field_name in self._FIELDS:
            setattr(self, field_name, deque(maxlen=maxlen))
        self._ep_counts = {}          # episode id -> transitions held, insertion-ordered

    def clear(self):
        """Drop every stored transition.

        Off by default (AblationConfig.clear_buffer_on_switch, whose default
        is CLEAR_BUFFER_ON_PHASE_SWITCH), matching the previous iteration,
        where recency weighting and capacity eviction only make cross-phase
        replay unlikely rather than impossible. The *_clearbuf presets enable
        it, which forbids that outright, at the cost of a gap after each
        switch: there is nothing to sample until batch_size transitions
        accumulate (~1 episode), and a phase's first updates then draw on a
        small, highly correlated buffer. Measured cost on learning the
        CURRENT block: none (end-of-block success 0.833 vs 0.839).
        """
        for field_name in self._FIELDS:
            getattr(self, field_name).clear()
        self._ep_counts.clear()
        self.block_start_ep = None

    def __len__(self):
        return len(self.observations)

    @property
    def n_episodes(self) -> int:
        return len(self._ep_counts)

    def _evict_oldest_episode(self):
        """Drop every transition of the oldest episode held. Episodes are
        appended contiguously, so they are the leftmost n entries."""
        oldest = next(iter(self._ep_counts))
        n = self._ep_counts.pop(oldest)
        for field_name in self._FIELDS:
            d = getattr(self, field_name)
            for _ in range(n):
                d.popleft()

    def _sampling_probs(self, t_now: int, ep_now=None) -> np.ndarray:
        """Transition units: p(e) proportional to exp(-dt / gamma_t), mixed
        with uniform. Ported verbatim from _ablations_sweep.py, including the
        max-subtraction before exponentiating and the uniform fallback for a
        degenerate sum. Note abs_tdes is deliberately absent here -- see
        PRIORITY_GAMMA_TD.

        Episode units: each EPISODE gets mass proportional to
        exp(-d_episodes / gamma_ep), split equally over its transitions, and
        the uniform floor is uniform over episodes too. So an episode's total
        sampling mass depends on its age and nothing else -- not on how many
        steps it took, which is what makes the amount of previous-context
        replay the same for every config.
        """
        if self.units == "episodes":
            if ep_now is None:
                raise ValueError("episode-unit buffer needs ep_now to sample")
            eps = np.asarray(self.episodes, dtype=np.int64)
            uniq, inv, counts = np.unique(eps, return_inverse=True, return_counts=True)
            age = np.maximum(0.0, float(ep_now) - uniq.astype(np.float64))
            w_ep = np.exp(-(age - age.min()) / max(self.gamma_ep, 1e-8))
            w_ep = w_ep / w_ep.sum()
            probs = (w_ep / counts)[inv]
            if self.mix > 0.0:
                floor = (np.full(len(uniq), 1.0 / len(uniq)) / counts)[inv]
                probs = (1.0 - self.mix) * probs + self.mix * floor
            return self._split_by_block(probs / probs.sum())

        dt = np.maximum(0.0, float(t_now) - np.asarray(self.timesteps, dtype=np.float64))

        energy = dt / max(self.gamma_t, 1e-8)
        logits = -energy
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        if self.mix > 0.0:
            probs = (1.0 - self.mix) * probs + self.mix * (1.0 / len(probs))
            probs = probs / probs.sum()

        return self._split_by_block(probs)

    def mark_block_start(self, episode: int) -> None:
        """The first episode of the block now being learned. Splits the buffer
        into the current block and everything before it; see self.rehearsal."""
        self.block_start_ep = int(episode)

    def _split_by_block(self, probs: np.ndarray) -> np.ndarray:
        """Reweight `probs` so the current block holds 1 - rehearsal of the
        mass and older blocks hold rehearsal, each pool keeping its own
        recency order. A no-op when the budget is off, no block is marked, or
        one pool is empty (a block's first update, or a cleared buffer)."""
        if not self.rehearsal or self.block_start_ep is None:
            return probs
        cur = np.asarray(self.episodes, dtype=np.int64) >= self.block_start_ep
        cur_sum, old_sum = probs[cur].sum(), probs[~cur].sum()
        if cur_sum <= 0.0 or old_sum <= 0.0:
            return probs
        out = np.empty_like(probs)
        out[cur] = probs[cur] / cur_sum * (1.0 - self.rehearsal)
        out[~cur] = probs[~cur] / old_sum * self.rehearsal
        return out / out.sum()

    def update_abs_tde(self, idxs: np.ndarray, abs_tde_values: np.ndarray):
        """Keep |TD error| current for the sampled transitions.

        Inert while _sampling_probs ignores abs_tdes, exactly as in the file
        this is ported from -- kept so the term can be switched on without
        re-deriving the bookkeeping."""
        for idx, val in zip(idxs, abs_tde_values):
            self.abs_tdes[int(idx)] = float(val)

    def add(self, obs, action, reward, next_obs, done, *,
            s_tm1, a_tm1, s_t, a_t, s_next, e_t, h_ec_prev,
            t_step: int, init_abs_tde: float = 0.0, episode=None):
        """`episode` is the global episode index the transition belongs to.
        Required in episode units (it is what capacity and recency count);
        recorded but unused in transition units."""
        if self.units == "episodes":
            if episode is None:
                raise ValueError("episode-unit buffer needs `episode` on add")
            episode = int(episode)
            self._ep_counts[episode] = self._ep_counts.get(episode, 0) + 1
            # a new episode past capacity evicts the oldest WHOLE episode, so
            # the buffer never holds a partial one
            while self.n_episodes > self.capacity_episodes:
                self._evict_oldest_episode()
        self.episodes.append(-1 if episode is None else int(episode))
        self.timesteps.append(float(t_step))
        self.abs_tdes.append(float(init_abs_tde))
        self.observations.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(np.asarray([reward], dtype=np.float32))
        self.next_observations.append(np.asarray(next_obs, dtype=np.float32).copy())
        self.dones.append(np.asarray([done], dtype=np.float32))
        self.s_tm1.append(np.asarray(s_tm1, dtype=np.float32).copy())
        self.a_tm1.append(np.asarray(a_tm1, dtype=np.float32).copy())
        self.s_t.append(np.asarray(s_t, dtype=np.float32).copy())
        self.a_t.append(np.asarray(a_t, dtype=np.float32).copy())
        self.s_next.append(np.asarray(s_next, dtype=np.float32).copy())
        self.e_t.append(np.asarray(e_t, dtype=np.float32).copy())
        self.h_ec_prev.append(np.asarray(h_ec_prev, dtype=np.float32).copy())

    def sample(self, batch_size, device, t_now: int, ep_now=None):
        probs = self._sampling_probs(t_now, ep_now)
        idxs = self.rng.choice(len(self.observations), size=batch_size, replace=True, p=probs)

        def stacked(field):
            return torch.tensor(np.stack([field[i] for i in idxs]), device=device)

        sac_batch = SACBatch(
            observations=stacked(self.observations),
            actions=stacked(self.actions),
            rewards=stacked(self.rewards),
            next_observations=stacked(self.next_observations),
            dones=stacked(self.dones),
        )
        eccb_batch = (
            stacked(self.s_tm1), stacked(self.a_tm1), stacked(self.s_t),
            stacked(self.a_t), stacked(self.s_next), stacked(self.e_t),
            stacked(self.h_ec_prev),
        )
        return sac_batch, eccb_batch, idxs


# ── Center-out behavioral probe ──────────────────────────────────────────

EVAL_N_TARGETS = 8
# The probe is deliberately FIXED while training sampling is not: it is the
# held-constant yardstick across arms and seeds, so it keeps the ring radius
# training used to have (0.75) even though episodes now draw goals anywhere.
#
# IT MUST FIT INSIDE THE WORKSPACE, which the assert below enforces. It was
# 2.5 against a POS_LIM of 1.0 (found 2026-09-20), so every one of the eight
# goals sat outside the box env_step clips position to: no reach could ever
# get within GOAL_RADIUS of one, every probe scored success 0.00 in every
# context and every block, path_ratio was NaN throughout, and the whole
# cross_context matrix and every center_out panel disagreed with the training
# curves they sit next to. A stale value from the pre-rewrite plant (task.py
# 2026-09-13), where position was not on [-1, 1]. 0.75 leaves a 0.25 margin
# to the wall so a curved or overshooting reach has room to bend without
# clipping -- curvature being what the trajectory panel exists to show --
# and is the same order as a training reach (median 1.14).
EVAL_RADIUS = 0.75

assert EVAL_RADIUS + GOAL_RADIUS <= T.POS_LIM, (
    f"EVAL_RADIUS={EVAL_RADIUS} puts the probe's goals outside the workspace "
    f"[-{T.POS_LIM}, {T.POS_LIM}]: the plant clips position to the box, so no "
    f"probe reach can reach them and every probe figure reads zero")


# CENTRE-OUT, and it stays that way.  A fixed start and equally spaced targets
# at one radius are what make trajectory CURVATURE readable: every reach is the
# same question, so a bend in the path is the field acting and not a different
# reach.  Off-centre starts were tried here and reverted -- they turn the eight
# reaches into eight different questions of length 0.22 to 1.64, which costs
# the comparison that the probe exists to provide and shifts the success level
# for reasons unrelated to the policy.
#
# Randomising the START DISTRIBUTION is the right thing in a different place:
# a representational analysis draws its starts from sample_pos_goal, so the
# data its bins are built from covers the state space the way the supervised
# rollout does.  That is a question about the CODE; this is a
# yardstick for BEHAVIOUR, and they want opposite things.
EVAL_START_ANYWHERE = False
EVAL_BATTERY_SEED = 12345


def eval_battery(n_targets=EVAL_N_TARGETS, radius=EVAL_RADIUS,
                 anywhere=EVAL_START_ANYWHERE, seed=EVAL_BATTERY_SEED):
    """The fixed (starts, goals) set the probe runs, both (n_targets, 2).

    anywhere=False -- the DEFAULT, see EVAL_START_ANYWHERE -- is the centre-out
    fan: every start at the origin, n_targets goals equally spaced at `radius`.

    anywhere=True draws (start, goal) pairs from sample_pos_goal instead, which
    covers the state space but makes every reach a different question. It is
    off by default and kept only so the contrast can be run deliberately; the
    per-block battery_seed in run_sac exists for it and is inert while this is
    False, because the centre-out fan does not depend on a seed.
    """
    if not anywhere:
        centre = np.zeros(2, dtype=np.float32)
        ang = np.arange(n_targets) * (2.0 * np.pi / n_targets)
        goals = (centre + radius * np.stack([np.cos(ang), np.sin(ang)], 1)
                 ).astype(np.float32)
        return np.repeat(centre[None, :], n_targets, axis=0), goals
    rng = np.random.default_rng(seed)
    starts, goals = [], []
    for _ in range(n_targets):
        p, g = sample_pos_goal(rng)
        starts.append(p); goals.append(g)
    return (np.asarray(starts, dtype=np.float32),
            np.asarray(goals, dtype=np.float32))



def eval_center_out(trainer, ec, cb, config, context_id, family_idx, *,
                     max_steps=None, n_targets=EVAL_N_TARGETS, radius=EVAL_RADIUS,
                     oracle_vec=None, cb_proj=None, oracle_embed=None,
                     ec_offset=None, cb_stats=None, ec_std=None, device=DEVICE,
                     oracle_noise=None, battery_seed=EVAL_BATTERY_SEED,
                     observer=None, bank=None, bank_ep_now=None,
                     channel=False, observer_start="uniform", probe_seed=0,
                     carry_h=None, carry_src=PRIOR_SRC_NONE):
    """Center-out reaching probe: start at the grid center, reach to n_targets
    equally spaced on a circle, and return the trajectories.

    The three keyword knobs added for the per-reach READOUT PROBE (see
    READOUT_PROBE_STEPS), all inert at their defaults:
      channel         step the channel plant (channel_step) instead of the
                      context's force field, with the observer predict-only
                      -- the probe of a channel block.
      observer_start  "uniform" (a fresh copy at the uniform prior, the
                      block-end probes) or "carry" (a snapshot of the run's
                      CURRENT posterior: the first action then reads the
                      belief the agent brings into its next reach).
      probe_seed      seed of the probe's own rng (bank draws, command
                      noise). Fixed at 0 for the block-end probes; the
                      readout probe seeds it by reach so successive points
                      draw independently while the run stays reproducible.
    The result carries `prior_src` per target: which context's end state
    each target's EC was dealt (PRIOR_SRC_* codes without a bank).

    A FIXED probe set is the point -- identical start and targets for every
    seed, config, context and occurrence, so trajectories are directly
    comparable and curvature is interpretable.  `battery_seed` only does
    anything under EVAL_START_ANYWHERE, which is off. An agent that compensates for
    the force field feedforward reaches straight; one that only corrects online
    reaches on a curve that straightens late. Success rate cannot show that
    difference; the trajectory shape can.

    Deterministic: uses the actor's tanh-mean rather than a sample, so what is
    plotted is the learned policy and not exploration noise. Side-effect free
    -- no replay writes, no optimizer steps, no global_step advance -- so
    calling it mid-training cannot perturb the run.

    EC hidden state and the CB prediction error reset per reach, exactly as
    they do per training episode, so context is not carried between reaches.

    Returns dict with:
        trajectories [n_targets, max_steps+1, 2], NaN-padded after termination
        goals        [n_targets, 2]
        success      [n_targets]
        steps        [n_targets]
        first_angle  [n_targets]   signed angle of the first action vs the
                                   goal direction, in degrees
        path_ratio   [n_targets]   path length / straight-line distance, NaN
                                   for reaches that never arrived

    NOTE ON first_angle. The first action is taken from the first observation,
    which is [center, zero velocity, goal] with e_t zero and a freshly reset EC
    hidden state -- it carries NO context information, by construction. So
    first_angle is identical whichever field the agent is dropped into: it
    reports the policy's DEFAULT aim, i.e. which context its weights are
    currently tuned toward, not per-context compensation. Context only becomes
    observable from the second step, once the agent has felt the field, so
    path_ratio is the measure that actually varies per context.
    """
    max_steps = MAX_STEPS if max_steps is None else max_steps
    starts, battery_goals = eval_battery(n_targets, radius, seed=battery_seed)
    if oracle_vec is None and config.use_oracle:
        if oracle_embed is None:
            # the config's kernel, not the default: a probe that rebuilds an
            # identity code for an agent trained on a geometry one is feeding
            # it a channel it has never seen
            oracle_embed = make_oracle_embed(
                0, kernel=config.oracle_kernel, arm=config.arm,
                arm_seed=config.arm_seed, action_mode=config.action_mode)
        oracle_vec = oracle_code(family_idx, oracle_embed)
    # The inferred oracle is NOT handed the probed context: it starts uniform
    # and has to infer it over the probe's reaches, so probe retention for
    # this config includes the cost of re-inference. Belief carries across the
    # reaches of one probe context (one visit), not across contexts.
    # `observer` is the run's template (its calibrations are expensive for the
    # CB one); the probe takes a fresh copy at the uniform prior. Without a
    # template it is built here -- the load-agent analyses call this directly.
    if observer_start not in ("uniform", "carry"):
        raise ValueError(f"observer_start must be 'uniform' or 'carry', got {observer_start!r}")
    carry = observer_start == "carry" and observer is not None
    probe_observer = None
    if config.use_oracle and config.oracle_source != "given" and (family_idx is not None or carry):
        if oracle_embed is None:
            oracle_embed = make_oracle_embed(
                0, kernel=config.oracle_kernel, arm=config.arm,
                arm_seed=config.arm_seed, action_mode=config.action_mode)
        if carry:
            probe_observer = observer.snapshot()
        else:
            probe_observer = (observer.fresh() if observer is not None
                              else make_observer(config, oracle_embed, device))
        oracle_vec = probe_observer.code()
    # `context_id` is a task context id, or NULL_FIELD (None) for the
    # no-perturbation warmup condition -- task.context_matrices handles both,
    # so the same probe serves every block.

    traj = np.full((n_targets, max_steps + 1, 2), np.nan, dtype=np.float32)
    goals = np.zeros((n_targets, 2), dtype=np.float32)
    success = np.zeros(n_targets, dtype=np.float32)
    steps_taken = np.zeros(n_targets, dtype=np.float32)
    first_angles = np.zeros(n_targets, dtype=np.float32)
    angles_by_step = np.full((n_targets, max_steps), np.nan, dtype=np.float32)
    prior_src = np.full(n_targets, PRIOR_SRC_NONE, dtype=np.int64)

    if cb_proj is None:
        cb_proj = make_cb_proj(0, device)
    # ITS OWN seed, not the run's rng: the probe has to feed the same channel
    # training fed -- a zero block would be off-distribution for an actor
    # trained on noise -- but it must stay reproducible and side-effect free,
    # and drawing from the run's rng would advance it mid-training.
    eval_rng = np.random.default_rng(probe_seed)
    # Same for the oracle's noise: a probe that fed a CLEAN code to an actor
    # trained on a noisy one would measure a channel the policy never had, and
    # would read as the snr manipulation doing nothing. Built here on its own
    # seed rather than taking the run's object, so the probe cannot advance the
    # training stream -- eval_center_out is side-effect free by contract.
    if (oracle_noise is None and config.use_oracle and config.oracle_snr < 1.0
            and oracle_embed is not None):
        oracle_noise = OracleNoise(oracle_embed, config.oracle_snr,
                                    config.oracle_noise_rho,
                                    rng=np.random.default_rng(0),
                                    isotropic=config.oracle_noise_iso)

    trainer.actor.eval()
    with torch.no_grad():
        for k in range(n_targets):
            goal = battery_goals[k]
            goals[k] = goal

            pos = starts[k].copy()
            vel_sim = np.zeros(2, dtype=np.float32)
            vel_obs = np.zeros(2, dtype=np.float32)
            # the same start-state prior the training reaches use (the bank
            # is read, never written, so the probe stays side-effect free)
            if ec is None:
                h = None
            elif carry_h is not None:
                # "carry": the probe must start where the NEXT reach would,
                # which is the propagated belief -- not a fresh bank draw, or
                # it would measure the bank's contents rather than behaviour
                h = carry_h.clone().to(device)
                prior_src[k] = carry_src
            elif bank is not None:
                # the CURRENT clock, not the bank's last write: after a channel
                # block (which writes nothing) the recency term must have
                # decayed exactly as it has for the training reaches
                h, prior_src[k], _src_ep = bank.sample(eval_rng, bank_ep_now)
            else:
                h = torch.zeros(1, EC_HIDDEN, device=device)
            if oracle_noise is not None:
                oracle_noise.reset()     # per reach, as the hidden state is
            s_tm1 = torch.zeros(1, SDIM, device=device)
            a_tm1 = torch.zeros(1, ACTION_DIM, device=device)
            s_t_np = circuit_state(pos, vel_obs)
            s_t = torch.tensor(s_t_np, dtype=torch.float32, device=device).unsqueeze(0)
            e_t = torch.zeros(1, SDIM, device=device)
            # CB's hidden state is one step stale by necessity (a CB pass
            # needs a_t); zero on the first step, exactly like e_t.
            cb_h2 = torch.zeros(1, CB_HIDDEN, device=device)

            traj[k, 0] = pos
            for t in range(max_steps):
                if np.linalg.norm(pos - goal) < GOAL_RADIUS:
                    success[k] = 1.0
                    break

                obs_np, z_t, h = augmented_obs_step(
                    ec, config, h, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
                    pos, vel_obs, goal, oracle_vec, device, ec_offset,
                    cb_stats=cb_stats, rng=eval_rng, ec_std=ec_std,
                    oracle_noise=oracle_noise)
                if oracle_noise is not None:
                    oracle_noise.advance()      # once per step, as in run_sac
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

                # third return value is the tanh-squashed MEAN: deterministic
                _, _, action_t = trainer.actor.get_action(obs_t)
                action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
                action_t_full = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

                # signed angle from the goal direction to the action taken,
                # recorded at EVERY step rather than only the first.
                #
                # first_angle alone cannot show context sensitivity for any
                # circuit config: at t=0 the probe has pos=center, vel=0,
                # e_t=0, cb_h2=0 and h=0, so z is a fixed vector independent of
                # context and the whole observation is identical across
                # contexts. The first action is therefore the same everywhere
                # BY CONSTRUCTION. The oracle escapes this only because its
                # code is present at t=0 with no inference delay.
                #
                # Context reaches EC a few steps in (circuit_probe: decodability
                # 0.74 over t in [0,5), 0.99 by [5,15)), so the aim TRAJECTORY
                # is where a circuit config's context dependence can appear.
                gd = goal - pos
                cross = gd[0] * action_np[1] - gd[1] * action_np[0]
                dot = action_np[0] * gd[0] + action_np[1] * gd[1]
                angle = float(np.degrees(np.arctan2(cross, dot)))
                angles_by_step[k, t] = angle
                if t == 0:
                    first_angles[k] = angle

                if channel:
                    pos_new, vel_sim_new, probe_wall = channel_step(pos, goal)
                else:
                    pos_new, vel_sim_new, probe_wall = env_step_params(
                        pos, vel_sim, action_np, context_id, eval_rng,
                        return_impulse=True)
                if probe_observer is not None:
                    # a channel step carries no evidence, as in the run loop
                    probe_observer.update(action_np, vel_obs, vel_sim_new, probe_wall,
                                          s_prev=s_t_np,
                                          s_next=circuit_state(pos_new, vel_sim_new),
                                          evidence=not channel)
                    oracle_vec = probe_observer.code()
                # THE SIMULATOR'S VELOCITY, not a finite difference.  The two
                # agree exactly in the interior (pos' = pos + vel'*DT) and
                # DIVERGE AT A WALL, where the plant clips the position and
                # zeroes the velocity on that axis while the difference still
                # reports the clipped step.  data.py stores the true velocity,
                # so that is what the circuit was fitted on -- feeding it a
                # difference makes the RL input convention depend on episodes
                # never touching a wall, which is a sampling choice rather
                # than a guarantee.  Measured: 0.00% of 24000 RL steps diverge
                # under ring sampling -- but free goal sampling lets
                # trajectories reach the walls again, so this now matters in
                # practice and not just in principle.
                vel_obs_new = vel_sim_new.astype(np.float32)

                s_pred_new, cb_h2 = cb_predict(cb, s_t, action_t_full, z_t, device)
                s_next_np = circuit_state(pos_new, vel_obs_new)
                s_next = torch.tensor(s_next_np, dtype=torch.float32, device=device).unsqueeze(0)

                s_tm1, a_tm1 = s_t, action_t_full
                s_t, s_t_np = s_next, s_next_np
                e_t = s_next - s_pred_new
                pos, vel_sim, vel_obs = pos_new, vel_sim_new, vel_obs_new

                traj[k, t + 1] = pos
                steps_taken[k] = t + 1
                if np.linalg.norm(pos - goal) < GOAL_RADIUS:
                    success[k] = 1.0
                    break
    trainer.actor.train()

    # Path length relative to the straight line, over reaches that ARRIVED --
    # a truncated failure has a short path and would masquerade as efficient.
    path_ratio = np.full(n_targets, np.nan, dtype=np.float32)
    for k in range(n_targets):
        if not success[k]:
            continue
        xy = traj[k][~np.isnan(traj[k][:, 0])]
        if len(xy) > 1:
            walked = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            # the reach ENDS on entering the goal circle, so a perfectly
            # straight one covers (radius - GOAL_RADIUS), not the full radius;
            # dividing by the radius would put ideal reaches below 1.0
            straight = float(np.linalg.norm(goals[k] - starts[k])) - GOAL_RADIUS
            path_ratio[k] = walked / max(straight, 1e-6)

    return {
        # `starts` is per-reach now. `center` is kept because every probes.npz
        # already on disk has it and the plotting reads it; it is the mean
        # start, which is meaningless as a reach origin, so plotters must
        # prefer `starts` when it is present.
        "trajectories": traj, "goals": goals, "starts": starts,
        "center": starts.mean(0),
        "success": success, "steps": steps_taken, "first_angle": first_angles,
        "angle_by_step": angles_by_step, "prior_src": prior_src,
        "path_ratio": path_ratio,
        "radius": float(radius), "goal_radius": float(GOAL_RADIUS),
    }


# ── Driving a frozen circuit, for calibration ─────────────────────────────
#
# These two came from circuit_probe.py, which was removed in the 2026-09-20
# cleanup once its diagnostic half (the context decoder over z) had answered
# its question: the input-scaling mismatch it was built to catch is fixed, and
# CB's relu-vs-tanh, not any scaling, turned out to control context learning.
# The rollout itself is load-bearing and stays -- every frozen-circuit run
# calibrates against it -- so it lives here now, next to its only callers,
# which also retires the circular import the split forced.

ROLLOUT_N_EPISODES = 140      # per context; 6 x 140 x 50 steps = 42k rows


def rollout_z(arm, ckpt_seed, contexts=None, *, n_episodes=ROLLOUT_N_EPISODES,
              max_steps=None, seed=0, device="cpu"):
    """Drive a frozen checkpoint through sac's OWN rollout and collect z.

    Through sac's own step functions -- circuit_state, augmented_obs_step,
    cb_predict, sample_pos_goal, env_step_params -- rather than a private copy
    of the loop, so that if sac drives the circuit wrongly this is driven
    wrongly in the same way. A private copy is exactly how the scaling bug hid:
    it kept reading a 1.00 decode while training quietly read 0.28.

    Actions are uniform random -- what run_sac itself takes before
    learning_starts, and the distribution the circuit has to work on while the
    policy is still exploring.

    Returns arrays aligned on one row per (episode, step):
      z, cb_h2, base_obs, y (context index), episode, t

    cb_h2 is the CB hidden state AS THE OBSERVATION SAW IT -- one step stale,
    zero on an episode's first step -- so it is also what the fixed input
    scales must be calibrated against (see source_rms).
    """
    contexts = list(PHASE_CONTEXT_IDS if contexts is None else contexts)
    max_steps = MAX_STEPS if max_steps is None else max_steps

    config = AblationConfig(name="probe", use_ec=True, use_cb=True,
                            freeze=True, arm=arm)
    cb_proj = make_cb_proj(seed, device)
    ec, cb = M.load(arm, ckpt_seed)
    freeze_module(ec)
    freeze_module(cb)

    rng = np.random.default_rng(seed)
    z_rows, h2_rows, base_rows, y, episode, step = [], [], [], [], [], []

    for ctx_i, context_id in enumerate(contexts):
        for ep in range(n_episodes):
            pos, goal = sample_pos_goal(rng)
            vel_sim = np.zeros(2, dtype=np.float32)
            vel_obs = np.zeros(2, dtype=np.float32)
            h = torch.zeros(1, EC_HIDDEN, device=device)
            s_tm1 = torch.zeros(1, SDIM, device=device)
            a_tm1 = torch.zeros(1, ACTION_DIM, device=device)
            s_t = torch.tensor(circuit_state(pos, vel_obs),
                               device=device).unsqueeze(0)
            e_t = torch.zeros(1, SDIM, device=device)
            cb_h2 = torch.zeros(1, CB_HIDDEN, device=device)

            for t in range(max_steps):
                obs, z_t, h = augmented_obs_step(
                    ec, config, h, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
                    pos, vel_obs, goal, None, device)
                z_rows.append(z_t.squeeze(0).detach().cpu().numpy())
                h2_rows.append(cb_h2.squeeze(0).detach().cpu().numpy())
                base_rows.append(obs[:SDIM + 2].copy())
                y.append(ctx_i)
                episode.append(ctx_i * 1_000_000 + ep)
                step.append(t)

                action = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
                action_t = torch.tensor(action, device=device).unsqueeze(0)
                pos_new, vel_sim = env_step_params(
                    pos, vel_sim, action, context_id, rng)
                vel_obs_new = ((pos_new - pos) / DT).astype(np.float32)

                s_pred, cb_h2 = cb_predict(cb, s_t, action_t, z_t, device)
                s_next = torch.tensor(circuit_state(pos_new, vel_obs_new),
                                      device=device).unsqueeze(0)
                s_tm1, a_tm1, s_t = s_t, action_t, s_next
                e_t = s_next - s_pred
                pos, vel_obs = pos_new, vel_obs_new

    return {"z": np.asarray(z_rows, dtype=np.float32),
            "cb_h2": np.asarray(h2_rows, dtype=np.float32),
            "base_obs": np.asarray(base_rows, dtype=np.float32),
            "y": np.asarray(y), "episode": np.asarray(episode),
            "t": np.asarray(step)}


# circuit_probe also carried source_rms, which calibrated the per-source input
# scales KIN_RMS / EC_STATE_RMS / CB_STATE_RMS. Those constants are gone --
# per-channel POS_SCALE/VEL_SCALE in circuit_state replaced the whole scheme --
# so it had no remaining caller and was dropped with the rest.


# ── Per-seed training loop ────────────────────────────────────────────────

def calibrate_ec_offset(config: AblationConfig, *, n_episodes=10, max_steps=30,
                        device="cpu") -> np.ndarray:
    """Mean z_ec over a random-action rollout across the phase contexts.

    Reuses rollout_z rather than rolling out here, so the calibration is
    driven through sac's own step functions and cannot drift from what
    training does.
    """
    if not config.use_ec:
        return np.zeros(0, dtype=np.float32)
    if not config.freeze:
        raise ValueError("center_ec is only defined for a frozen circuit; an "
                         "end-to-end EC moves under a fixed offset")
    data = rollout_z(config.arm, config.arm_seed,
                     n_episodes=n_episodes, max_steps=max_steps,
                     device=device)
    keep = data["t"] > 0        # t=0 has a reset hidden state and a zero error
    return data["z"][keep].mean(0).astype(np.float32)


def calibrate_ec_stats(config: AblationConfig, *, n_episodes=10, max_steps=30,
                       device="cpu"):
    """(mean, std) of z_ec, for normalize_circuit. calibrate_ec_offset is the
    mean half of this, kept separate because center_ec predates it and the
    two must not silently diverge -- so this reuses the same rollout call."""
    if not config.use_ec:
        return None
    if not config.freeze:
        raise ValueError("normalize_circuit is only defined for a frozen "
                         "circuit; an end-to-end EC moves under fixed stats")
    data = rollout_z(config.arm, config.arm_seed,
                     n_episodes=n_episodes, max_steps=max_steps,
                     device=device)
    keep = data["t"] > 0
    z = data["z"][keep]
    # floor the divisor: a unit that never varies would otherwise divide by
    # ~0 and turn a constant into a huge constant, the opposite of the intent
    return (z.mean(0).astype(np.float32),
            np.maximum(z.std(0), 1e-3).astype(np.float32))


def calibrate_cb_stats(config: AblationConfig, cb_proj, *, n_episodes=10,
                       max_steps=30, device="cpu"):
    """Per-dimension mean and std of the CB block, for cb_noise.

    MATCHED ON THE MARGINALS, deliberately. A first layer whose columns are
    drawn independently of the input turns matched marginal variance into
    matched drive -- E[var(Wx)] = w^2 tr(Cov), the cross-terms averaging out
    over random columns -- so noise matched this way enters a fresh actor at
    the same scale CB does. That is the same matching point cb_gain=0.31 was
    set at, which keeps the two manipulations on one axis.

    What it does NOT preserve is the cross-dimension covariance or the
    temporal correlation. That is the manipulation, not an oversight.

    Uses the RUN'S OWN cb_proj rather than rollout_z's, because the block MC
    sees is h2 @ cb_proj and cb_proj is drawn per run seed; calibrating
    against a different projection would match the wrong moments.
    """
    if not config.use_cb:
        return None
    if not config.freeze:
        raise ValueError("cb_noise/center_cb are only defined for a frozen "
                         "circuit; an end-to-end CB's statistics move during "
                         "training")
    if config.cb_noise and config.center_cb:
        raise ValueError("cb_noise already draws from the calibrated mean; "
                         "centering it on top is not a distinct manipulation")
    data = rollout_z(config.arm, config.arm_seed,
                     n_episodes=n_episodes, max_steps=max_steps,
                     device=device)
    h2 = torch.as_tensor(np.asarray(data["cb_h2"], dtype=np.float32),
                         device=device)
    blk = (h2 @ cb_proj).cpu().numpy()
    keep = data["t"] > 0    # t=0's cb_h2 is zero by construction, as in
    blk = blk[keep]         # calibrate_ec_offset
    return (blk.mean(0).astype(np.float32),
            np.maximum(blk.std(0), 1e-3).astype(np.float32))


AGENT_FILE = "agent.pt"


@dataclass(slots=True)
class TrainedAgent:
    """A reloaded run: the networks plus everything needed to feed them."""
    actor: "Actor"
    qf1: "SoftQNetwork"
    qf2: "SoftQNetwork"
    qf1_target: "SoftQNetwork"
    qf2_target: "SoftQNetwork"
    spec: ObsSpec
    config: AblationConfig
    cb_proj: torch.Tensor
    oracle_embed: np.ndarray
    seed: int
    alpha: float
    ec_offset: Optional[np.ndarray] = None


def save_agent(path, trainer, config, spec, cb_proj, oracle_embed, seed,
               ec_offset=None) -> None:
    """Persist the trained networks AND the input pipeline that produced them.

    BOTH CRITICS, not just the actor: the actor's sensitivity to a channel and
    the critic's can differ, and a policy that ignores a channel its value
    function does use is a different finding from neither using it.

    cb_proj and oracle_embed travel WITH the weights because both are drawn
    from the run seed. Regenerating them at load time would work only by
    coincidence of matching seeds, and would silently feed a reloaded actor
    channels it never saw during training.
    """
    torch.save({
        "actor": trainer.actor.state_dict(),
        "qf1": trainer.qf1.state_dict(),
        "qf2": trainer.qf2.state_dict(),
        "qf1_target": trainer.qf1_target.state_dict(),
        "qf2_target": trainer.qf2_target.state_dict(),
        "spec": asdict(spec),
        "config": asdict(config),
        "cb_proj": cb_proj.detach().cpu().clone(),
        "oracle_embed": np.asarray(oracle_embed, dtype=np.float32).copy(),
        "ec_offset": (None if ec_offset is None
                      else np.asarray(ec_offset, dtype=np.float32).copy()),
        "alpha": float(trainer.current_alpha),
        "seed": int(seed),
    }, path)


def load_agent(path, device="cpu") -> TrainedAgent:
    """Rebuild a run from disk. Networks come back in eval mode."""
    blob = torch.load(path, map_location=device, weights_only=False)
    spec = ObsSpec(**blob["spec"])
    config = AblationConfig(**blob["config"])
    # the run's context set becomes the process's: an analysis that probes
    # this agent (eval_center_out, the observers) must see the contexts it
    # was trained on, and those functions read the module's set
    apply_phase_set(getattr(config, "phase_set", DEFAULT_PHASE_SET))

    def _net(cls, key):
        net = cls(spec).to(device)
        net.load_state_dict(blob[key])
        net.eval()
        return net

    return TrainedAgent(
        actor=_net(Actor, "actor"),
        qf1=_net(SoftQNetwork, "qf1"), qf2=_net(SoftQNetwork, "qf2"),
        qf1_target=_net(SoftQNetwork, "qf1_target"),
        qf2_target=_net(SoftQNetwork, "qf2_target"),
        spec=spec, config=config,
        cb_proj=blob["cb_proj"].to(device),
        oracle_embed=np.asarray(blob["oracle_embed"], dtype=np.float32),
        seed=int(blob["seed"]), alpha=float(blob["alpha"]),
        ec_offset=(None if blob.get("ec_offset") is None
                   else np.asarray(blob["ec_offset"], dtype=np.float32)),
    )


def run_sac(seed, config: AblationConfig, *, n_phase_ep=N_PHASE_EP, phase_reps=PHASE_REPS,
            max_steps=MAX_STEPS, buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE,
            learning_starts=LEARNING_STARTS, device=DEVICE,
            priority_gamma_t=PRIORITY_GAMMA_T, priority_gamma_td=PRIORITY_GAMMA_TD,
            priority_mix=PRIORITY_MIX, warmup_episodes=WARMUP_EPISODES,
            cb_proj_dim=CB_PROJ_DIM, agent_path=None,
            buffer_capacity_episodes=BUFFER_CAPACITY_EPISODES,
            priority_gamma_ep=PRIORITY_GAMMA_EP, coin_block_ep=None):
    # FIRST: which contexts. Everything below -- embed, observers, schedule,
    # probes -- reads the module's set at call time (see PHASE_SETS), and a
    # spawned worker arrives here with the module at its import default.
    apply_phase_set(config.phase_set)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # one fixed random CB -> MC projection per run, shared by the actor, both
    # critics and the probe, exactly as the reference drew it
    cb_proj = make_cb_proj(seed, device, n_out=cb_proj_dim)
    oracle_embed = make_oracle_embed(seed, kernel=config.oracle_kernel,
                                     arm=config.arm, arm_seed=config.arm_seed,
                                     action_mode=config.action_mode)
    # its own rng stream, not `rng`: an oracle_snr run must not consume draws
    # that would otherwise have gone to sample_pos_goal, or the snr sweep
    # would also be a different-episodes sweep
    oracle_noise = make_oracle_noise(config, oracle_embed, seed=seed)
    observer = make_observer(config, oracle_embed, device)
    # normalize_circuit needs the mean AND the std of both blocks; the
    # older center_* flags need only the means, and share the same rollout
    ec_stats = (calibrate_ec_stats(config, device=device)
                if config.normalize_circuit else None)
    ec_offset = (ec_stats[0] if ec_stats is not None
                 else calibrate_ec_offset(config, device=device)
                 if config.center_ec else None)
    ec_std = ec_stats[1] if ec_stats is not None else None
    cb_stats = (calibrate_cb_stats(config, cb_proj, device=device)
                if (config.cb_noise or config.center_cb
                    or config.normalize_circuit) else None)

    ec, cb = build_ec_cb(config, device)
    spec = obs_spec(config, cb_proj_dim=cb_proj_dim)

    actor = Actor(spec).to(device)
    qf1 = SoftQNetwork(spec).to(device)
    qf2 = SoftQNetwork(spec).to(device)
    qf1_target = SoftQNetwork(spec).to(device)
    qf2_target = SoftQNetwork(spec).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=LR_Q)

    trainer = SACTrainer(
        actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer,
        gamma=GAMMA, tau=TAU, target_entropy=-float(ACTION_DIM), autotune=True,
        device=device, grad_clip_norm=GRAD_CLIP_NORM,
    )

    eccb_trainer = None
    if not config.freeze and (ec is not None or cb is not None):
        params = ([p for p in ec.parameters()] if ec is not None else []) + \
                 ([p for p in cb.parameters()] if cb is not None else [])
        eccb_optimizer = torch.optim.Adam(params, lr=LR_EC_CB)
        eccb_trainer = ECCBTrainer(ec, cb, eccb_optimizer, arm=config.arm,
                                    grad_clip_norm=GRAD_CLIP_NORM, device=device)

    # units come from the CONFIG, not a run kwarg, so they land in the saved
    # blob and the run name -- see AblationConfig.replay_units
    replay_buffer = ReplayBuffer(
        buffer_capacity, gamma_t=priority_gamma_t, gamma_td=priority_gamma_td,
        mix=priority_mix, seed=seed, units=config.replay_units,
        capacity_episodes=buffer_capacity_episodes, gamma_ep=priority_gamma_ep)
    replay_buffer.rehearsal = float(config.rehearsal_fraction)

    n_phases = N_PHASE_FAMILIES * phase_reps
    curve = {k: [] for k in (
        "ep", "global_ep", "phase", "occurrence", "label", "success", "steps",
        # init_dist is logged because goals are sampled freely now: the start
        # distance varies ~30x across episodes and partly drives return and
        # steps, so any arm-vs-arm contrast wants it as a covariate.  Under the
        # old ring sampling it was a constant and carried no information.
        "init_dist", "dist_norm", "angle", "first_angle",
        "return", "q_loss", "actor_loss", "alpha", "q_target_mean",
        "log_pi_mean", "ec_cb_loss", "q_gnorm", "actor_gnorm",
        # wall contact: mean normalised impulse per step, and the fraction of
        # steps that touched a wall at all. See WALL_IMPULSE_WEIGHT.
        "wall_impulse", "wall_rate",
        # COIN schedules: channel reach flag, EC's start-state source, and the
        # inferred oracle's end-of-reach posterior on P+, P- and the null
        # field (nan without an observer, when the pair is not in the context
        # set, or -- for the null -- when the observer has no null hypothesis)
        "channel", "prior_src", "belief_pplus", "belief_pminus", "belief_base",
        # the per-reach readout probe (READOUT_PROBE_STEPS): mean first-action
        # angle over the 8 directions, and over the targets dealt a P+ / P- /
        # baseline start state (schedule_contexts: the null field or, under
        # savings, the base context); nan under "cycle" and where no target
        # drew the source
        "probe_first_angle", "probe_fa_pplus", "probe_fa_pminus", "probe_fa_base",
    )}

    probes = []
    global_step = 0
    global_ep = 0

    # Blocks, in order: an optional unlogged null-field warmup, then the
    # phases. One loop over both so warmup runs through exactly the same step
    # implementation as training rather than a parallel copy that could drift.
    # the schedule: our cycle (the default, same block list as before) or one
    # of COIN's paradigms -- see SCHEDULES
    blocks = schedule_blocks(config.schedule, n_phase_ep, phase_reps, warmup_episodes,
                             coin_block_ep)
    # EC's start-of-reach prior (see ExperienceBank); None means the zero state
    if config.ec_prior not in EC_PRIORS:
        raise KeyError(f"unknown ec_prior {config.ec_prior!r}; expected one of {EC_PRIORS}")
    bank = (ExperienceBank(config.ec_prior_tau, config.ec_prior_alpha, device)
            if config.ec_prior in ("experience", "carry") and ec is not None else None)
    # "carry": the belief propagated from the previous reach, and whose context
    # it came from. None until the first reach ends. See EC_PRIORS.
    carry_h, carry_src = None, PRIOR_SRC_NONE
    if observer is not None:
        observer.count_prior = float(config.observer_count_prior)
    # hindsight relabelling (AblationConfig.relabel_z): the ec block of a
    # stored observation, replaced at reach end with the end-of-reach z
    ec_slice = slice(spec.base, spec.base + spec.ec)

    for block_idx, block in enumerate(blocks):
        context_id = block["context_id"]
        family_idx = block["family_idx"]
        occurrence = block["occurrence"]
        label = block["label"]
        phase_idx = block["phase_idx"]
        log_block = block["log"]
        learn = block.get("learn", True)
        channel = block.get("channel", False)
        # the warmup and the null field have no context to encode, so the
        # given oracle sees zeros there (family_idx None -> the zero code)
        oracle_vec = (oracle_code(family_idx if log_block else None, oracle_embed)
                      if config.use_oracle else None)
        if observer is not None:
            # the inferred oracle carries its belief across the switch and has
            # to notice it from evidence; zeros in the warmup, as above
            oracle_vec = observer.code() if log_block else oracle_code(None, oracle_embed)

        # Replay must not carry one context's transitions into the next: each
        # phase learns from its own data only. Read from the CONFIG, not the
        # module flag -- see AblationConfig.clear_buffer_on_switch for why.
        if config.clear_buffer_on_switch and log_block:
            replay_buffer.clear()
        # THE NULL-FIELD BLOCK IS NEVER REPLAYED (2026-09-18, user's call, for
        # every config): its transitions carry a z the circuit was never
        # trained on -- low-norm and central, closer to every context centroid
        # than they are to each other -- so replaying them taught "aim
        # straight" across the whole code space and slowed P+ learning for
        # every model, main most (first-50-reach index 0.12 vs 0.49 with an
        # empty buffer). The buffer is cleared ONCE, on leaving a LOGGED
        # null-field block (P0). The cycle schedule's warmup is a null-field
        # block too but unlogged, so those runs are untouched; the trained
        # baseline of savings-washout is a real context and IS replayed.
        prev = blocks[block_idx - 1] if block_idx else None
        if (log_block and prev is not None and prev["log"]
                and prev["label"] == NULL_LABEL and not config.clear_buffer_on_switch):
            replay_buffer.clear()

        # the batch split follows the block boundary (ReplayBuffer.rehearsal)
        replay_buffer.mark_block_start(global_ep + 1)

        kind_note = ("" if log_block else " [warmup, not logged]") + \
                    (" [channel: clamped, MC frozen]" if channel else "")
        print(f"\nPhase {label} ({block['n_ep']} episodes){kind_note}...", flush=True)

        for local_ep in range(1, block["n_ep"] + 1):
            global_ep += 1

            pos, goal = sample_pos_goal(rng)
            vel_sim = np.zeros(2, dtype=np.float32)
            vel_obs = np.zeros(2, dtype=np.float32)

            # EC's start state: zero, or a draw from the experience bank
            prior_src = PRIOR_SRC_NONE
            if ec is None:
                h = None
            elif bank is None:
                h = torch.zeros(1, EC_HIDDEN, device=device)
            elif config.ec_prior == "carry":
                # keep the previous reach's belief; leak to the bank's visit
                # distribution with probability ec_prior_hazard (see EC_PRIORS)
                if carry_h is None or float(rng.random()) < config.ec_prior_hazard:
                    h, prior_src, _src_ep = bank.sample_uniform(rng)
                else:
                    h, prior_src = carry_h.clone(), carry_src
                carry_h, carry_src = h.clone(), prior_src
            else:
                h, prior_src, _src_ep = bank.sample(rng, global_ep)
            pending, z_last = [], None      # relabel_z: this reach's transitions, end z
            if oracle_noise is not None:
                oracle_noise.reset()        # per episode, as EC's hidden is
            s_tm1 = torch.zeros(1, SDIM, device=device)
            a_tm1 = torch.zeros(1, ACTION_DIM, device=device)
            s_t_np = circuit_state(pos, vel_obs)
            s_t = torch.tensor(s_t_np, dtype=torch.float32, device=device).unsqueeze(0)
            # e_t is forced to zero at the first step of an episode -- there is
            # no CB prediction to compare against yet, exactly model.py's t==0.
            e_t = torch.zeros(1, SDIM, device=device)
            cb_h2 = torch.zeros(1, CB_HIDDEN, device=device)

            init_dist = float(np.linalg.norm(pos - goal))
            rewards, dist_seq, angle_seq, wall_imps = [], [], [], []
            first_angle: Optional[float] = None
            success, steps = False, 0
            latest = {"q_loss": 0.0, "actor_loss": 0.0, "alpha": trainer.current_alpha,
                      "q_target_mean": 0.0, "log_pi_mean": 0.0}
            latest_ec_cb = float("nan")

            for _ in range(max_steps):
                d = float(np.linalg.norm(pos - goal))
                dist_seq.append(d)
                if d < GOAL_RADIUS:
                    success = True
                    break

                h_before = h
                obs_np, z_t, h = augmented_obs_step(
                    ec, config, h, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
                    pos, vel_obs, goal, oracle_vec, device, ec_offset,
                    cb_stats=cb_stats, rng=rng, ec_std=ec_std,
                    oracle_noise=oracle_noise)

                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                # random actions only while LEARNING has not started; a channel
                # block reads the policy out whatever the buffer holds
                if learn and (global_step < learning_starts or len(replay_buffer) < batch_size):
                    action_np = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
                else:
                    with torch.no_grad():
                        action_t, _, action_mean = trainer.act(obs_t)
                    # A CHANNEL REACH READS THE POLICY OUT: the actor's tanh-mean,
                    # not a sample. MC is frozen there, so exploration noise is
                    # pure measurement noise -- with a stochastic readout a frozen
                    # baseline's channel wandered by 0.4 of the full effect over
                    # 150 reaches (first-action index sd ~0.9 per reach). A force
                    # channel measures the expressed memory; this is that.
                    action_np = ((action_mean if channel else action_t)
                                 .squeeze(0).cpu().numpy().astype(np.float32))
                action_t_full = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

                if channel:
                    # the channel plant: straight to the goal whatever the
                    # action; the action is the readout (see SCHEDULES)
                    pos_new, vel_sim_new, wall_imp = channel_step(pos, goal)
                else:
                    pos_new, vel_sim_new, wall_imp = env_step_params(
                        pos, vel_sim, action_np, context_id, rng,
                        return_impulse=True)
                # THE SIMULATOR'S VELOCITY, not a finite difference.  The two
                # agree exactly in the interior (pos' = pos + vel'*DT) and
                # DIVERGE AT A WALL, where the plant clips the position and
                # zeroes the velocity on that axis while the difference still
                # reports the clipped step.  data.py stores the true velocity,
                # so that is what the circuit was fitted on -- feeding it a
                # difference makes the RL input convention depend on episodes
                # never touching a wall, which is a sampling choice rather
                # than a guarantee.  Measured: 0.00% of 24000 RL steps diverge
                # under ring sampling -- but free goal sampling lets
                # trajectories reach the walls again, so this now matters in
                # practice and not just in principle.
                vel_obs_new = vel_sim_new.astype(np.float32)
                # t -> t+1 for the inferred oracle, ONCE per env step and
                # BEFORE the lookahead below, so next_obs carries the updated
                # belief and the next iteration reads the same one back --
                # the same contract OracleNoise.advance keeps.
                if observer is not None and log_block:
                    # a channel reach carries no evidence: the belief relaxes
                    # under the transition prior alone (evidence=False)
                    observer.update(action_np, vel_obs, vel_obs_new, wall_imp,
                                    s_prev=s_t_np,
                                    s_next=circuit_state(pos_new, vel_obs_new),
                                    evidence=not channel)
                    oracle_vec = observer.code()
                reward, done, _ = step_reward(pos_new, goal, pos,
                                              action=action_np,
                                              effort_weight=config.effort_weight,
                                              wall_impulse=wall_imp)
                rewards.append(reward)
                # logged per episode so the penalty's effect is visible without
                # re-instrumenting: mean normalised impulse, and contact rate
                wall_imps.append(float(wall_imp) / T.MAX_WALL_IMPULSE)

                # Signed angle from the goal direction to the action taken --
                # the FEEDFORWARD adaptation measure, ported from the previous
                # iteration. Under rot +45 an agent that knows the context aims
                # ~-45 off so the resulting force points at the goal, while a
                # context-blind agent aims straight at it (~0). Unlike success,
                # this does not saturate once the task is solved, so it
                # separates configs even when every one of them succeeds.
                gd = goal - pos
                cross = gd[0] * action_np[1] - gd[1] * action_np[0]
                dot = action_np[0] * gd[0] + action_np[1] * gd[1]
                alpha_deg = float(np.degrees(np.arctan2(cross, dot)))
                angle_seq.append(alpha_deg)
                if first_angle is None:
                    first_angle = alpha_deg

                s_pred_new, cb_h2_next = cb_predict(cb, s_t, action_t_full, z_t, device)
                s_next_np = circuit_state(pos_new, vel_obs_new)
                s_next = torch.tensor(s_next_np, dtype=torch.float32, device=device).unsqueeze(0)
                e_next = s_next - s_pred_new

                # One-step lookahead so the replay buffer gets a genuine
                # "next observation" for SAC's Q-target -- deliberately
                # recomputed (not memoized) at the top of the next iteration;
                # see the module docstring.
                z_next, _h_next = ec_step_or_zero(
                    ec, s_t, action_t_full, s_next, e_next, h, config.arm, device)
                next_base_obs = build_base_obs(pos_new, vel_obs_new, goal)
                next_z_cb = None
                if config.use_cb:
                    with torch.no_grad():
                        next_z_cb = (cb_h2_next @ cb_proj).squeeze(0).cpu().numpy()
                    next_z_cb = cb_channel(next_z_cb, config, cb_stats, rng)

                # t -> t+1 for the oracle's noise, ONCE per env step: the
                # lookahead reads the advanced sample and the next iteration
                # reads the same one back, so (obs, next_obs) is a genuine
                # transition of this channel. See OracleNoise.advance.
                if oracle_noise is not None:
                    oracle_noise.advance()
                next_obs_np = augment_obs(
                    next_base_obs, config,
                    z_ec=(ec_channel(z_next, config, ec_offset, ec_std) if config.use_ec else None),
                    z_cb=next_z_cb,
                    oracle_vec=(oracle_noise.apply(oracle_vec)
                                if oracle_noise is not None else oracle_vec),
                )

                transition = dict(
                    obs=obs_np, action=action_np, reward=reward, next_obs=next_obs_np, done=done,
                    s_tm1=s_tm1.squeeze(0).cpu().numpy(),
                    a_tm1=a_tm1.squeeze(0).cpu().numpy(),
                    s_t=s_t_np, a_t=action_np, s_next=s_next_np,
                    e_t=e_t.squeeze(0).detach().cpu().numpy(),
                    h_ec_prev=(h_before.squeeze(0).detach().cpu().numpy() if h_before is not None
                               else np.zeros(EC_HIDDEN, dtype=np.float32)),
                    t_step=global_step, episode=global_ep,
                )
                if learn and config.relabel_z and config.use_ec:
                    pending.append(transition)          # stored at reach end, relabelled
                elif learn:
                    replay_buffer.add(
                        transition.pop("obs"), transition.pop("action"), transition.pop("reward"),
                        transition.pop("next_obs"), transition.pop("done"), **transition)
                z_last = z_next

                if (learn and len(replay_buffer) >= batch_size and global_step >= learning_starts
                        and global_step % UPDATE_EVERY == 0):
                    sac_batch, eccb_batch, idxs = replay_buffer.sample(
                        batch_size, device, t_now=global_step, ep_now=global_ep)

                    # |TD error| for the sampled transitions, measured BEFORE
                    # the update that would change it. Inert while
                    # _sampling_probs ignores it -- kept as the previous
                    # iteration had it; see PRIORITY_GAMMA_TD.
                    with torch.no_grad():
                        q_target = trainer.compute_q_target(sac_batch)
                        q1 = trainer.qf1(sac_batch.observations, sac_batch.actions).view(-1)
                        q2 = trainer.qf2(sac_batch.observations, sac_batch.actions).view(-1)
                        abs_tde = 0.5 * (torch.abs(q1 - q_target) + torch.abs(q2 - q_target))

                    latest = trainer.update(sac_batch)
                    if eccb_trainer is not None:
                        eccb_update = eccb_trainer.update(eccb_batch)
                        latest_ec_cb = eccb_update["ec_cb_loss"]

                    replay_buffer.update_abs_tde(idxs, abs_tde.detach().cpu().numpy())

                pos, vel_sim, vel_obs = pos_new, vel_sim_new, vel_obs_new
                # a_tm1 is EC's action history, deliberately NOT the actor's:
                # see build_base_obs
                s_tm1, a_tm1 = s_t, action_t_full
                s_t, s_t_np = s_next, s_next_np
                e_t = e_next
                cb_h2 = cb_h2_next
                global_step += 1
                steps += 1

                if done:
                    success = True
                    break

            ep_return = float(sum(rewards))
            dist_norm = (float(np.mean(dist_seq)) / init_dist) if init_dist > 0 else 0.0
            mean_angle = float(np.mean(angle_seq)) if angle_seq else 0.0
            first_angle = first_angle if first_angle is not None else 0.0

            # end of reach: the experience bank takes the settled state of any
            # reach that carried evidence (not a channel reach); the observer
            # counts the visit; relabelled transitions are stored with the
            # end-of-reach z in their ec block
            if steps > 0 and not channel:
                if bank is not None:
                    bank.add(h, global_ep, context_id)
                # the belief this reach settled on becomes the next reach's
                # start. NOT after a channel reach: its end state is about the
                # clamp, not the context, so the carry stays where it was and
                # only a leak moves it -- see EC_PRIORS.
                if config.ec_prior == "carry":
                    carry_h, carry_src = h.clone(), (PRIOR_SRC_NULL if context_id is None
                                                     else int(context_id))
                if observer is not None and log_block and observer.count_prior > 0.0:
                    observer.note_visit()
            if pending and z_last is not None:
                z_end = ec_channel(z_last, config, ec_offset, ec_std)
                for tr in pending:
                    tr["obs"] = tr["obs"].copy(); tr["next_obs"] = tr["next_obs"].copy()
                    tr["obs"][ec_slice] = z_end; tr["next_obs"][ec_slice] = z_end
                    replay_buffer.add(tr.pop("obs"), tr.pop("action"), tr.pop("reward"),
                                      tr.pop("next_obs"), tr.pop("done"), **tr)
                pending = []

            if not log_block:
                # warmup episodes train the agent but are not part of any
                # phase, so they stay out of the curve the analysis reads
                if local_ep % 50 == 0:
                    print(f"  warmup ep {local_ep:4d} | succ={float(success):.0f} | "
                          f"ret={ep_return:+.1f}", flush=True)
                continue

            curve["ep"].append(local_ep)
            curve["global_ep"].append(global_ep)
            curve["phase"].append(phase_idx)
            curve["occurrence"].append(occurrence)
            curve["label"].append(label)
            curve["success"].append(float(success))
            curve["steps"].append(float(steps))
            curve["init_dist"].append(float(init_dist))
            curve["dist_norm"].append(dist_norm)
            curve["angle"].append(mean_angle)
            curve["first_angle"].append(float(first_angle))
            curve["return"].append(ep_return)
            curve["q_loss"].append(float(latest["q_loss"]))
            curve["actor_loss"].append(float(latest["actor_loss"]))
            curve["alpha"].append(float(latest["alpha"]))
            curve["q_target_mean"].append(float(latest["q_target_mean"]))
            curve["log_pi_mean"].append(float(latest["log_pi_mean"]))
            curve["ec_cb_loss"].append(float(latest_ec_cb))
            curve["q_gnorm"].append(float(latest.get("q_gnorm", float("nan"))))
            curve["actor_gnorm"].append(float(latest.get("actor_gnorm", float("nan"))))
            curve["wall_impulse"].append(float(np.mean(wall_imps)) if wall_imps else 0.0)
            curve["wall_rate"].append(
                float(np.mean([w > 0 for w in wall_imps])) if wall_imps else 0.0)
            # the COIN schedules: was this a channel reach, and which context's
            # end state did EC start from (PRIOR_SRC_* codes for none / null)
            curve["channel"].append(float(channel))
            curve["prior_src"].append(int(prior_src))
            # the schedule's roles (schedule_contexts): P+, P- and the
            # baseline -- the null hypothesis under P0 schedules, the base
            # context under savings. nan for a role the run cannot report.
            roles = schedule_contexts(config.schedule)
            bp = bm = bn = float("nan")
            if observer is not None:
                post = observer.posterior
                for role, slot in (("pplus", "bp"), ("pminus", "bm"), ("base", "bn")):
                    cid = roles[role]
                    if role != "base" and cid is None:
                        continue
                    if cid in observer.ids:
                        val = float(post[observer.ids.index(cid)])
                        if slot == "bp": bp = val
                        elif slot == "bm": bm = val
                        else: bn = val
            curve["belief_pplus"].append(bp)
            curve["belief_pminus"].append(bm)
            curve["belief_base"].append(bn)
            # the per-reach readout probe (READOUT_PROBE_STEPS): the mean
            # first-action angle over the 8 directions, and that mean over the
            # targets whose EC start state was dealt from P+ / P- (nan when no
            # target drew that source -- and always without a bank)
            pfa = pfa_pp = pfa_pm = pfa_null = float("nan")
            if config.schedule != "cycle":
                ro = eval_center_out(trainer, ec, cb, config, context_id, family_idx,
                                     max_steps=READOUT_PROBE_STEPS, device=device,
                                     cb_proj=cb_proj, oracle_embed=oracle_embed,
                                     observer=observer, bank=bank,
                                     bank_ep_now=global_ep + 1,   # the next reach's clock
                                     ec_offset=ec_offset, cb_stats=cb_stats, ec_std=ec_std,
                                     channel=channel, observer_start="carry",
                                     probe_seed=EVAL_BATTERY_SEED + global_ep,
                                     carry_h=carry_h, carry_src=carry_src,
                                     oracle_noise=None)
                fa = np.asarray(ro["first_angle"], dtype=float)
                pfa = float(fa.mean())
                src = np.asarray(ro["prior_src"])
                # the bank codes a null-field state PRIOR_SRC_NULL, a context's
                # state by its id -- so the base role maps to either
                codes = {"pplus": roles["pplus"], "pminus": roles["pminus"],
                         "base": PRIOR_SRC_NULL if roles["base"] is None else roles["base"]}
                means = {}
                for role, code in codes.items():
                    if code is not None and (src == code).any():
                        means[role] = float(fa[src == code].mean())
                pfa_pp = means.get("pplus", pfa_pp); pfa_pm = means.get("pminus", pfa_pm)
                pfa_null = means.get("base", pfa_null)
            curve["probe_first_angle"].append(pfa)
            curve["probe_fa_pplus"].append(pfa_pp)
            curve["probe_fa_pminus"].append(pfa_pm)
            curve["probe_fa_base"].append(pfa_null)

            if local_ep % 50 == 0:
                w = min(50, len(curve["success"]))
                print(
                    f"  ep {local_ep:4d} | global_ep {global_ep:5d} | "
                    f"succ={np.mean(curve['success'][-w:]):.2f} | "
                    f"ret={np.mean(curve['return'][-w:]):.1f} | "
                    f"q_loss={np.mean(curve['q_loss'][-w:]):.4f} | "
                    f"actor={np.mean(curve['actor_loss'][-w:]):.4f} | "
                    f"alpha={np.mean(curve['alpha'][-w:]):.3f}",
                    flush=True,
                )

        # End-of-block behavioral probes, on EVERY context rather than only the
        # one just trained. Same start and targets every time, so everything is
        # comparable across phases, occurrences, seeds and configs.
        #
        # The block's own context (is_current) is what the center-out
        # trajectory plot draws. The others are the cross-context matrix, and
        # they answer two questions the learning curves cannot:
        #   INTERFERENCE -- how much does the policy just learned here cost
        #     performance in the contexts it is not in?
        #   MEMORY TRACES -- does a context trained in an EARLIER phase still
        #     perform above its naive level after training moved elsewhere?
        # The warmup block supplies the naive row those traces are measured
        # against. Side-effect free; see eval_center_out.
        eval_specs = []
        if not log_block:
            eval_specs.append(("null field", NULL_FIELD, None, True))
        for eval_fi, eval_cid in enumerate(PHASE_CONTEXT_IDS):
            eval_specs.append((PHASE_LABELS[eval_fi], eval_cid, eval_cid,
                               bool(log_block and eval_cid == block["context_id"])))

        summary = []
        for eval_label, eval_ctx, eval_cid, is_current in eval_specs:
            eval_fi = PHASE_CONTEXT_IDS.index(eval_cid) if eval_cid is not None else 0
            # The oracle must be told the context it is ACTUALLY being evaluated
            # in, not the phase's -- being told the truth is the entire premise
            # of that config, and a stale one-hot would measure something else.
            if config.use_oracle:
                ev_oracle = oracle_code(eval_fi if eval_cid is not None else None,
                                        oracle_embed)
            else:
                ev_oracle = None

            probe = eval_center_out(trainer, ec, cb, config, eval_ctx, eval_fi,
                                     max_steps=max_steps, oracle_vec=ev_oracle, device=device, cb_proj=cb_proj,
                                     oracle_embed=oracle_embed, observer=observer, bank=bank,
                                     bank_ep_now=global_ep,
                                     # BOTH of these were missing. The probe
                                     # builds its observations through the same
                                     # augmented_obs_step the rollout does, so
                                     # any channel transform the config asks for
                                     # has to be handed to it too -- otherwise
                                     # the probe feeds an actor trained on one
                                     # input distribution a different one, and
                                     # reports that as a policy failure.
                                     ec_offset=ec_offset, cb_stats=cb_stats,
                                     ec_std=ec_std,
                                     # A FRESH BATTERY PER BLOCK, SHARED BY THE
                                     # CONTEXTS EVALUATED IN IT. Both halves
                                     # matter. Varying it across blocks is what
                                     # makes the probe estimate performance over
                                     # the task's own start distribution instead
                                     # of over eight fixed pairs -- with 13
                                     # blocks that is ~520 (start, goal) pairs
                                     # per seed rather than 8. Holding it fixed
                                     # WITHIN a block is what keeps the
                                     # cross-context transfer matrix matched:
                                     # the contexts in a row differ only in the
                                     # field, not in which reaches they were
                                     # asked for.
                                     #
                                     # Derived from the block index rather than
                                     # drawn from the run's rng, so the probe
                                     # stays side-effect free and the run stays
                                     # reproducible.
                                     battery_seed=EVAL_BATTERY_SEED + block_idx,
                                     # NOT the run's object: the probe must not
                                     # advance the training noise stream, and
                                     # eval_center_out builds its own on a
                                     # fixed seed for exactly that reason
                                     oracle_noise=None)
            probe.update({"battery_seed": EVAL_BATTERY_SEED + block_idx,
                          "phase": phase_idx, "occurrence": occurrence, "label": label,
                          "context_id": block["context_id"], "eval_label": eval_label,
                          "eval_context_id": eval_cid, "is_current": is_current})
            probes.append(probe)
            mark = "*" if is_current else " "
            summary.append(f"{mark}{eval_label}: {int(probe['success'].sum())}/"
                           f"{len(probe['success'])} @{np.mean(probe['first_angle']):+.0f}deg")

        print("  probes | " + " | ".join(summary), flush=True)

    if agent_path is not None:
        save_agent(agent_path, trainer, config, spec, cb_proj, oracle_embed, seed,
                   ec_offset=ec_offset)

    return curve, probes


def run_one_seed(config: AblationConfig, seed: int, *, make_figures=True, **run_kwargs):
    """Train (or load from cache) a single seed and write its artifacts.

    Split out of main() so one (config, seed) pair is a self-contained unit of
    work -- that is what sweep_ablations parallelizes over. Results are
    independent of how it is scheduled: run_sac seeds torch, numpy and the
    replay buffer's own rng from `seed` alone, so a parallel sweep reproduces
    a serial one exactly.
    """
    seed_dir = run_dir(config.name, seed)
    curve_path = os.path.join(seed_dir, "curve.npz")
    probe_path = os.path.join(seed_dir, "probes.npz")

    if os.path.exists(curve_path):
        print(f"Cached run found for {config.name} seed {seed}, loading.")
        curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        probes = (np.load(probe_path, allow_pickle=True)["probes"].tolist()
                  if os.path.exists(probe_path) else [])
    else:
        curve, probes = run_sac(seed, config,
                                agent_path=os.path.join(seed_dir, AGENT_FILE),
                                **run_kwargs)
        np.savez(curve_path, curve=np.array(curve, dtype=object))
        np.savez(probe_path, probes=np.array(probes, dtype=object))

    if make_figures:
        import sac_figures
        seed_fig_dir = fig_dir(config.name, seed)
        sac_figures.plot_seed(seed, curve, seed_fig_dir,
                               key_params={"config": config.name, "arm": config.arm,
                                           "freeze": config.freeze})
        if probes:
            sac_figures.plot_center_out(seed, probes, seed_fig_dir, config_name=config.name)
            sac_figures.plot_cross_context(seed, probes, seed_fig_dir, config_name=config.name)
            sac_figures.plot_aim_by_step(seed, probes, seed_fig_dir, config_name=config.name)
            sac_figures.plot_transfer_trajectories(seed, probes, seed_fig_dir,
                                                   config_name=config.name)

    return curve


def _run_task(config: AblationConfig, seed: int, run_kwargs: dict, log_to_file: bool,
               out_dirs=None):
    """One (config, seed) pair, for execution in its own process.

    Module-level and picklable on purpose: macOS spawns rather than forks, so
    a closure or lambda could not be sent to a worker.

    out_dirs carries the parent's (RUN_DIR, FIG_DIR) explicitly. A spawned
    worker re-imports this module from scratch and so would otherwise use the
    DEFAULTS, silently writing to the repo's sac_runs/ even when the parent had
    been pointed elsewhere -- dropping runs into the real results directory
    where a later sweep would load them from cache as if they were real.
    """
    import torch as _torch
    # One thread per worker. Torch otherwise grabs every core in EVERY process,
    # and N processes each oversubscribing the machine is slower than serial.
    _torch.set_num_threads(1)

    if out_dirs is not None:
        global RUN_DIR, FIG_DIR
        RUN_DIR, FIG_DIR = out_dirs

    t0 = time.perf_counter()
    if log_to_file:
        # parallel workers would otherwise interleave per-episode output
        log_path = os.path.join(run_dir(config.name, seed), "train.log")
        with open(log_path, "w") as fh, redirect_stdout(fh), redirect_stderr(fh):
            run_one_seed(config, seed, **run_kwargs)
    else:
        run_one_seed(config, seed, **run_kwargs)
    return config.name, seed, time.perf_counter() - t0


def run_tasks(tasks, *, workers=None, run_kwargs=None, log_to_file=None):
    """Run (config, seed) pairs, in parallel when more than one worker.

    Shared by sac.main (parallel over seeds of one config) and
    sweep_ablations.run_named_sweep (parallel over every config x seed pair),
    so there is a single implementation of the pooling.

    Parallelism over seeds is the speedup that matters here: the networks are
    far too small for a GPU to help -- measured on an M-series machine, an MPS
    update ran ~2x SLOWER than CPU (17.5 vs 8.3 ms at batch 256) and each
    host<->device round trip cost 0.35 ms against 0.004 ms, which the per-step
    env loop pays several times over. Seeds are independent and identically
    seeded either way, so a parallel run reproduces a serial one exactly.
    """
    run_kwargs = dict(run_kwargs or {})
    tasks = list(tasks)
    if not tasks:
        return []

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, len(tasks)))
    if log_to_file is None:
        log_to_file = workers > 1

    if workers == 1:
        for config, seed in tasks:
            print(f"\n{'=' * 60}\n{config.name} seed {seed}\n{'=' * 60}", flush=True)
            _run_task(config, seed, run_kwargs, log_to_file=log_to_file)
        return []

    shown = os.path.relpath(RUN_DIR)
    if shown.startswith(".."):        # outside cwd: an absolute path reads better
        shown = RUN_DIR
    print(f"{len(tasks)} run(s) on {workers} workers; per-run output goes to "
          f"{shown}/<config>/seed_<n>/train.log", flush=True)

    out_dirs = (RUN_DIR, FIG_DIR)      # see _run_task
    failures, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_task, config, seed, run_kwargs, True, out_dirs):
                   (config, seed) for config, seed in tasks}
        for i, fut in enumerate(as_completed(futures), start=1):
            config, seed = futures[fut]
            try:
                name, s, dt = fut.result()
                print(f"  [{i}/{len(tasks)}] {name} seed {s} done in {dt / 60:.1f} min",
                      flush=True)
            except Exception as exc:              # keep the rest of the sweep going
                failures.append((config.name, seed, exc))
                print(f"  [{i}/{len(tasks)}] {config.name} seed {seed} FAILED: {exc!r}",
                      flush=True)

    print(f"\nall runs finished in {(time.perf_counter() - t0) / 60:.1f} min")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for name, seed, exc in failures:
            print(f"  {name} seed {seed}: {exc!r}")
    return failures


def aggregate_config(config: AblationConfig, seeds, save_agg=True):
    """Concatenate whichever seeds have finished into one frame per config.

    Reads the cached per-seed curves rather than taking them as arguments, so
    it works the same whether the seeds were run serially in-process or by
    separate worker processes.
    """
    frames = []
    for seed in seeds:
        curve_path = os.path.join(run_dir(config.name, seed), "curve.npz")
        if not os.path.exists(curve_path):
            print(f"  missing curve for {config.name} seed {seed}, skipping")
            continue
        curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        df = pd.DataFrame(curve)
        df["config"] = config.name
        df["seed"] = int(seed)
        frames.append(df)

    if not frames:
        return None

    agg_df = pd.concat(frames, ignore_index=True)
    if save_agg:
        base_dir = os.path.join(RUN_DIR, config.name)
        os.makedirs(base_dir, exist_ok=True)
        agg_df.to_csv(os.path.join(base_dir, "aggregated_results.csv"), index=False)
        agg_df.to_pickle(os.path.join(base_dir, "aggregated_results.pkl"))
        print(f"Aggregated {len(frames)} seed(s) of {config.name} -> {base_dir}")
    return agg_df


def main(config: AblationConfig, seeds=5, save_agg=True, workers=None, **run_kwargs):
    print(f"Device: {DEVICE}")
    print(f"Config | name={config.name}, seeds={seeds}, "
          f"use_ec={config.use_ec}, use_cb={config.use_cb}, freeze={config.freeze}, "
          f"use_oracle={config.use_oracle}, arm={config.arm}, arm_seed={config.arm_seed}")
    run_tasks([(config, seed) for seed in range(seeds)],
               workers=workers, run_kwargs=run_kwargs)
    aggregate_config(config, range(seeds), save_agg=save_agg)
    print(f"\nDone. Figures saved to {FIG_DIR}/{config.name}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train SAC with optional EC/CB-augmented observations.")
    # derived from the presets rather than restated, so adding a config in one
    # place cannot leave the CLI rejecting it
    import sweep_ablations
    parser.add_argument("--config", required=True,
                         choices=list(sweep_ablations.CONFIG_BY_NAME))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--arm-seed", type=int, default=None,
                         help="checkpoint seed to load for the frozen configs")
    parser.add_argument("--phase-episodes", type=int, default=N_PHASE_EP)
    parser.add_argument("--phase-reps", type=int, default=PHASE_REPS)
    parser.add_argument("--workers", type=int, default=None,
                         help="parallel processes over seeds (default: cpu_count-1; 1 = serial)")
    parser.add_argument("--clear-buffer", action="store_true",
                         help="wipe the replay buffer at every phase switch -- the "
                              "SAVINGS version of the experiment (see AblationConfig."
                              "clear_buffer_on_switch). Suffixes the run name with "
                              "_clearbuf so it can never overwrite the replay version; "
                              "'--config X --clear-buffer' is the same run as "
                              "'--config X_clearbuf'.")
    parser.add_argument("--episode-buffer", action="store_true",
                         help="replay buffer counts EPISODES, not transitions "
                              "(AblationConfig.replay_units='episodes'): same memory "
                              "span and recency for every config regardless of how "
                              "fast it succeeds. Suffixes the run name with _epbuf. The "
                              "DEFAULT under a COIN paradigm schedule (no suffix there).")
    parser.add_argument("--transition-buffer", action="store_true",
                         help="opt a paradigm-schedule run OUT of the episode buffer "
                              "(replay_units='transitions'); suffixes _transbuf")
    # which frozen circuit drives MC -- only meaningful for freeze=True presets
    parser.add_argument("--action-mode", default=None,
                         choices=list(M.ACTION_MODES) + list(M.LEGACY_ACTION_MODES),
                         help="the supervised circuit's action process (default: the "
                              "preset's, i.e. model.DEFAULT_ACTION_MODE). Suffixes the "
                              "run name with _<mode>.")
    parser.add_argument("--ec-readout", default=None, choices=list(M.READOUTS),
                         help="the circuit's EC readout; 'noisy' selects the "
                              "fixed-noise rate-cost circuit at --rate-beta and "
                              "suffixes the run name with _noisy<beta>.")
    parser.add_argument("--rate-beta", type=float, default=None,
                         help="beta of the noisy-readout circuit to load (implies "
                              "--ec-readout noisy)")
    parser.add_argument("--noise-sigma", type=float, default=None,
                         help="channel-noise sd of the noisy-readout circuit to load "
                              "(implies --ec-readout noisy); suffixes the run name "
                              "with _sigma<sigma> when it is not model.NOISE_SIGMA")
    parser.add_argument("--train-contexts", default=None,
                         choices=list(M.TRAIN_CONTEXT_SETS),
                         help="which contexts the frozen circuit was fitted on "
                              "(model.TRAIN_CONTEXT_SETS); suffixes the run name "
                              "with the circuit's _train<fam> tag")
    parser.add_argument("--contexts", default=None, choices=list(PHASE_SETS),
                         help="which contexts the task uses (AblationConfig.phase_set): "
                              "mixed = two per family (default), <family>s = all four "
                              "of one family. Suffixes the run name with the family.")
    # COIN's paradigms (see SCHEDULES); every one of these is inert by default
    parser.add_argument("--schedule", default=None, choices=list(SCHEDULES),
                         help="block schedule: cycle (default) or one of COIN's paradigms; "
                              "the latter need --contexts rotations. Suffix _sr/_er/_savw.")
    parser.add_argument("--ec-prior", default=None, choices=list(EC_PRIORS),
                         help="EC's start-of-reach state: zero, or experience (drawn from the "
                              "bank of end-of-reach states; suffix _prior). Default: experience "
                              "for a config with an EC channel, none otherwise; --ec-prior zero "
                              "recovers the old run under its old name.")
    parser.add_argument("--relabel-z", action="store_true",
                         help="store each transition with the END-of-reach z "
                              "(AblationConfig.relabel_z). Suffix _relabel. Off by default: the "
                              "actor then trains on a z it rarely sees while acting (the dealt "
                              "state's z at the first step is never the end-of-reach one).")
    parser.add_argument("--count-prior", type=float, default=None,
                         help="weight of the visit-count term in an inferred oracle's "
                              "transition prior (AblationConfig.observer_count_prior). "
                              "Suffix _cprior.")
    return parser.parse_args(argv)


CLEARBUF_SUFFIX = "_clearbuf"
EPBUF_SUFFIX = "_epbuf"
TRANSBUF_SUFFIX = "_transbuf"      # a paradigm run opted OUT of the episode buffer


def _with_infix(name: str, tag: str) -> str:
    """`name` with `tag` inserted BEFORE any trailing buffer suffixes, so a
    knob applied to a hand-written twin names the same run as the knob
    applied to the parent plus --clear-buffer: main_clearbuf + _prior ->
    main_prior_clearbuf, not main_clearbuf_prior (which the plot registry
    would not parse)."""
    trailing = ""
    for suffix in (CLEARBUF_SUFFIX, EPBUF_SUFFIX):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            trailing = suffix + trailing
    return name + tag + trailing


def apply_cli_variants(cfg: dict, *, clear_buffer: bool = False,
                       episode_buffer: bool = False, transition_buffer: bool = False,
                       action_mode=None,
                       ec_readout=None, rate_beta=None, noise_sigma=None,
                       phase_set=None, train_contexts=None, schedule=None,
                       ec_prior=None, relabel_z: bool = False,
                       count_prior=None) -> dict:
    """Turn a preset dict into the variant the CLI asked for, renaming it.

    Every knob renames the run because the run directory is config.name:
    setting a knob alone on `main` would write the variant into the parent's
    directory and the two would be indistinguishable on disk. Suffix order is
    circuit first, buffer second -- `main --action-mode iid --ec-readout noisy
    --rate-beta 0.001 --clear-buffer` is `main_iid_noisy0.001_clearbuf`, which
    reads as "the noisy-readout iid circuit, savings version". A knob a preset
    already has set is a no-op: no double suffix.

    The circuit knobs (action_mode, ec_readout, rate_beta, noise_sigma) select
    WHICH FROZEN CHECKPOINT drives MC, so they are refused on a config that
    does not freeze one: on `baseline` they would change fields nothing reads
    and mislabel the run. The readout suffix is model.readout_tag, the same
    function that names the checkpoint, so run name and circuit agree.

    Shared by sac.py's and sweep_ablations.py's CLIs so the two cannot drift
    in how they name the same run.
    """
    cfg = dict(cfg)
    circuit_knob = (action_mode is not None or ec_readout is not None
                    or rate_beta is not None or noise_sigma is not None
                    or train_contexts is not None)
    if circuit_knob and not cfg.get("freeze", False):
        raise ValueError(f"--action-mode/--ec-readout/--rate-beta/--noise-sigma/"
                         f"--train-contexts select a frozen circuit; {cfg['name']!r} "
                         f"does not freeze one")
    if action_mode is not None:
        mode = M.normalize_action_mode(action_mode)
        if mode != M.normalize_action_mode(cfg.get("action_mode", M.DEFAULT_ACTION_MODE)):
            cfg["action_mode"] = mode
            if f"_{mode}" not in cfg["name"]:
                cfg["name"] = _with_infix(cfg["name"], f"_{mode}")
    if train_contexts is not None and train_contexts != cfg.get("train_contexts", M.DEFAULT_TRAIN_CONTEXTS):
        cfg["train_contexts"] = train_contexts
        tag = M.train_contexts_tag(train_contexts)        # validates; "" for all
        if tag and tag not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], tag)
    if ec_readout is not None or rate_beta is not None or noise_sigma is not None:
        # --rate-beta / --noise-sigma alone imply the noisy readout: neither
        # means anything else
        readout = M.normalize_readout(
            ec_readout or ("noisy" if (rate_beta is not None or noise_sigma is not None)
                           else cfg.get("ec_readout", M.DEFAULT_READOUT)))
        beta = float(rate_beta if rate_beta is not None else cfg.get("rate_beta", M.RATE_BETA))
        sigma = float(noise_sigma if noise_sigma is not None
                      else cfg.get("noise_sigma", M.NOISE_SIGMA))
        have = (cfg.get("ec_readout", M.DEFAULT_READOUT),
                float(cfg.get("rate_beta", M.RATE_BETA)),
                float(cfg.get("noise_sigma", M.NOISE_SIGMA)))
        if have != (readout, beta, sigma):
            cfg["ec_readout"] = readout
            cfg["rate_beta"] = beta
            cfg["noise_sigma"] = sigma
            # the same suffix model.arm_tag gives the checkpoint: "" for the
            # default channel, "_tanh", or "_noisy<beta>[_sigma<sigma>]"
            tag = M.readout_tag(readout, beta, sigma)
            if tag and tag not in cfg["name"]:
                cfg["name"] = _with_infix(cfg["name"], tag)
    # a COIN paradigm needs P+/P- (the +-30 rotations), so it IMPLIES the
    # rotation set unless a set was named; the name then carries both:
    # baseline_rot_sr. Checked here, at config time, not in a worker.
    if schedule is not None and schedule != DEFAULT_SCHEDULE:
        if schedule not in SCHEDULES:
            raise KeyError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
        if phase_set is None and cfg.get("phase_set", DEFAULT_PHASE_SET) == DEFAULT_PHASE_SET:
            phase_set = "rotations"
        chosen = phase_set or cfg.get("phase_set", DEFAULT_PHASE_SET)
        ids = phase_ids_for(chosen)
        if any(c not in ids for c in schedule_required_ids(schedule)):
            need = [T.C_LABEL[c] for c in schedule_required_ids(schedule)]
            raise ValueError(f"schedule {schedule!r} needs the rotations {need}; the context "
                             f"set {chosen!r} lacks them -- use --contexts rotations")
        # ... and the fixed rehearsal budget, for the same reason: the
        # paradigms ask short blocks (P- 50, washout 75) to be learned beside
        # a 300-reach P+ block. See PARADIGM_REHEARSAL.
        if cfg.get("rehearsal_fraction", 0.0) == 0.0:
            cfg["rehearsal_fraction"] = PARADIGM_REHEARSAL
        # ... and the effort cost, which is what makes the +-60 pair pay
        if cfg.get("effort_weight", 0.0) == 0.0:
            cfg["effort_weight"] = EFFORT_WEIGHT
        # A PARADIGM ALSO IMPLIES THE EPISODE BUFFER (2026-09-18, user's
        # call): in transition units the 7900 P+ transitions were 88% of the
        # sampled batch after 10 P- reaches and still 55% after 50, and main's
        # P- memory, learned in the first 15 reaches, eroded back toward P+
        # under that replay; with equal mass per episode (gamma 40 episodes)
        # it held and kept building. No suffix -- it is the paradigms'
        # default; --transition-buffer opts out and suffixes _transbuf.
        if not transition_buffer and cfg.get("replay_units", "transitions") != "episodes":
            cfg["replay_units"] = "episodes"
    # the TASK knob: which contexts. Allowed on every preset (baseline
    # included -- it changes the task, not the circuit); after the circuit
    # suffixes, before the buffer ones: main_iid_rot_clearbuf
    if phase_set is not None and phase_set != cfg.get("phase_set", DEFAULT_PHASE_SET):
        if phase_set not in PHASE_SETS:
            raise KeyError(f"unknown phase_set {phase_set!r}; expected one of {PHASE_SETS}")
        cfg["phase_set"] = phase_set
        tag = PHASE_SET_SUFFIX[phase_set]
        if tag and tag not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], tag)
    # the COIN-paradigm knobs, after the task knob and before the buffer ones:
    # main_rot_sr_prior_relabel_clearbuf
    if schedule is not None and schedule != cfg.get("schedule", DEFAULT_SCHEDULE):
        if schedule not in SCHEDULES:
            raise KeyError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
        cfg["schedule"] = schedule
        tag = SCHEDULE_SUFFIX[schedule]
        if tag and tag not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], tag)
    if ec_prior is not None and ec_prior != cfg.get("ec_prior", "zero"):
        if ec_prior not in EC_PRIORS:
            raise KeyError(f"unknown ec_prior {ec_prior!r}; expected one of {EC_PRIORS}")
        if not cfg.get("use_ec", False):
            raise ValueError(f"--ec-prior needs an EC channel; {cfg['name']!r} has none")
        cfg["ec_prior"] = ec_prior
        tag = {"experience": "_prior", "carry": "_carry"}.get(ec_prior)
        if tag and tag not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], tag)
    if relabel_z and not cfg.get("relabel_z", False):
        if not cfg.get("use_ec", False):
            raise ValueError(f"--relabel-z relabels the EC block; {cfg['name']!r} has none")
        cfg["relabel_z"] = True
        if "_relabel" not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], "_relabel")
    if count_prior is not None and float(count_prior) != float(cfg.get("observer_count_prior", 0.0)):
        if not cfg.get("use_oracle", False) or cfg.get("oracle_source", "given") == "given":
            raise ValueError(f"--count-prior needs an inferred oracle; {cfg['name']!r} has none")
        cfg["observer_count_prior"] = float(count_prior)
        if float(count_prior) > 0 and "_cprior" not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], "_cprior")
    if episode_buffer and transition_buffer:
        raise ValueError("--episode-buffer and --transition-buffer exclude each other")
    if episode_buffer and cfg.get("replay_units", "transitions") != "episodes":
        cfg["replay_units"] = "episodes"
        if EPBUF_SUFFIX not in cfg["name"]:
            cfg["name"] = cfg["name"] + EPBUF_SUFFIX
    if transition_buffer:
        # only meaningful under a paradigm schedule, where episodes are the
        # default; under "cycle" transitions already are and nothing changes
        if cfg.get("schedule", DEFAULT_SCHEDULE) == DEFAULT_SCHEDULE:
            raise ValueError("--transition-buffer only opts a paradigm schedule out of the "
                             "episode buffer; the cycle schedule already counts transitions")
        cfg["replay_units"] = "transitions"
        if TRANSBUF_SUFFIX not in cfg["name"]:
            cfg["name"] = _with_infix(cfg["name"], TRANSBUF_SUFFIX)
    # REPLAY IS THE PARADIGMS' DEFAULT (settled 2026-09-18 after going both
    # ways): MC is a shared-weight controller with no gate, so without
    # rehearsal 50 P- reaches overwrite P+ (probe +0.05..+0.33) and the
    # channel rebounds toward a memory that is mostly gone. Rehearsal is how
    # this architecture implements the memory protection COIN builds into
    # responsibility-gated learning. The *_clearbuf twin (block-local buffer)
    # is the ablation that shows protection is needed; baseline with replay
    # (flat channel) shows a context signal is needed too. Held fixed:
    # episode-indexed replay, the null-field block never replayed.
    if clear_buffer and not cfg.get("clear_buffer_on_switch", False):
        cfg["clear_buffer_on_switch"] = True
        if not cfg["name"].endswith(CLEARBUF_SUFFIX):
            cfg["name"] = cfg["name"] + CLEARBUF_SUFFIX
    return cfg


def default_ec_prior(cfg: dict, ec_prior=None):
    """The prior a run gets when none is named: "carry" for a config with an
    EC channel (2026-09-19; it was "experience" for a day), none otherwise.
    Resolved at the ENTRY POINTS (config_from_cli, sweep presets_for), not
    inside apply_cli_variants, so twins derived from an already-resolved
    parent inherit its prior instead of re-defaulting -- and --ec-prior zero
    recovers the old run under its old name."""
    if ec_prior is not None:
        return ec_prior
    return "carry" if cfg.get("use_ec", False) else None


def config_from_cli(args) -> AblationConfig:
    """Resolve a parsed CLI into the config that will actually run.

    Overrides apply ONLY when the flag was actually given: a preset that pins
    its own arm_seed (because its checkpoint exists for one seed only) must not
    be silently overwritten by a CLI default nobody typed. The variant flags
    go through apply_cli_variants, which also renames the run -- see there.
    """
    import sweep_ablations
    cfg = dict(sweep_ablations.CONFIG_BY_NAME[args.config])
    if args.arm_seed is not None:
        cfg["arm_seed"] = args.arm_seed
    cfg.setdefault("arm_seed", 0)
    cfg = apply_cli_variants(cfg, clear_buffer=getattr(args, "clear_buffer", False),
                             episode_buffer=getattr(args, "episode_buffer", False),
                             transition_buffer=getattr(args, "transition_buffer", False),
                             action_mode=getattr(args, "action_mode", None),
                             ec_readout=getattr(args, "ec_readout", None),
                             rate_beta=getattr(args, "rate_beta", None),
                             noise_sigma=getattr(args, "noise_sigma", None),
                             phase_set=getattr(args, "contexts", None),
                             train_contexts=getattr(args, "train_contexts", None),
                             schedule=getattr(args, "schedule", None),
                             ec_prior=default_ec_prior(cfg, getattr(args, "ec_prior", None)),
                             relabel_z=getattr(args, "relabel_z", False),
                             count_prior=getattr(args, "count_prior", None))
    return AblationConfig(**cfg)


if __name__ == "__main__":
    args = _parse_args()
    config = config_from_cli(args)
    main(config, seeds=args.seeds, workers=args.workers,
         n_phase_ep=args.phase_episodes, phase_reps=args.phase_reps)
