"""
Tests for sac.py's new SAC/EC/CB wiring: confirms configs get the observation
dimensions they should, frozen configs really freeze (no gradient can reach
model.py's loaded checkpoint), end-to-end configs really learn, and the
smaller pure-function pieces (obs construction, goal-reaching reward, replay
buffer) do what they claim.
"""
import inspect
import os

import numpy as np
import pytest
import torch

import model as M
import task as T
import sac
import sweep_ablations as SW
from sac import (
    AblationConfig, Actor, SoftQNetwork, SACTrainer, ECCBTrainer, ReplayBuffer,
    obs_dim, oracle_one_hot, build_ec_cb, freeze_module,
)


def test_actor_action_bounded_and_log_prob_shape():
    torch.manual_seed(0)
    actor = Actor(8)
    obs = torch.randn(5, 8)
    action, log_prob, mean = actor.get_action(obs)
    assert action.shape == (5, 2)
    assert torch.all(action > -1) and torch.all(action < 1)
    assert log_prob.shape == (5, 1)
    assert torch.all(mean.abs() <= 1)


def _make_trainer(d, tau=0.005, target_entropy=-2.0):
    actor = Actor(d)
    qf1, qf2 = SoftQNetwork(d), SoftQNetwork(d)
    qf1_t, qf2_t = SoftQNetwork(d), SoftQNetwork(d)
    qf1_t.load_state_dict(qf1.state_dict())
    qf2_t.load_state_dict(qf2.state_dict())
    trainer = SACTrainer(
        actor, qf1, qf2, qf1_t, qf2_t,
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters())),
        tau=tau, target_entropy=target_entropy,
    )
    return trainer


def test_soft_update_targets_exact_polyak():
    trainer = _make_trainer(d=8, tau=0.1)
    for p in trainer.qf1.parameters():
        p.data.fill_(1.0)
    for p in trainer.qf1_target.parameters():
        p.data.fill_(3.0)
    for p in trainer.qf2.parameters():
        p.data.fill_(2.0)
    for p in trainer.qf2_target.parameters():
        p.data.fill_(4.0)

    trainer.soft_update_targets()

    for p in trainer.qf1_target.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 0.1 * 1.0 + 0.9 * 3.0))
    for p in trainer.qf2_target.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 0.1 * 2.0 + 0.9 * 4.0))


def test_obs_dim_matches_expected_table():
    """Spelled out from the composition rule rather than calling obs_dim, so
    this still catches drift -- but derived from the module constants, since
    PHASE_CONTEXT_IDS (and so the oracle one-hot width) is meant to be edited."""
    base = sac.SDIM + 2                     # [pos, vel, goal]
    # Ported from the reference: EC reaches the actor through z_t -- the same
    # tensor CB is conditioned on -- and CB through a FIXED RANDOM projection
    # of its last hidden layer, never through s_pred or e_t.
    expected = {
        "baseline": base,
        "oracle": base + sac.ORACLE_EMBED_DIM,
        "main_frozen": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        "control_frozen": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        "end_to_end": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        "cb_only_end_to_end": base + sac.CB_PROJ_DIM,
        # muting zeroes the cb block's CONTENTS, never its width
        "main_frozen_no_cb": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        # centering shifts the ec block's VALUES, never its width
        "main_frozen_no_cb_centered": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        # a different recurrent CELL inside EC changes neither z's width nor
        # CB's, so the observation is identical to main_frozen's
        "main_frozen_rnn": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        # cb_gain scales the cb block's VALUES, never its width
        "main_frozen_cb_scaled": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        "main_frozen_rnn_no_cb": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        # cb_noise SUBSTITUTES the cb block, matched width for width -- the
        # whole point is that the network is architecturally identical and
        # only the block's contents differ
        "main_frozen_cb_noise": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        # centering shifts both blocks' VALUES, never their widths
        "main_frozen_centered": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
        "main_frozen_normed": base + sac.LATENT_DIM + sac.CB_PROJ_DIM,
    }
    for name, cfg in SW.CONFIG_BY_NAME.items():
        config = AblationConfig(**cfg)
        assert obs_dim(config) == expected[name], name
        # obs_dim is the RAW width: the bottleneck lives inside the networks,
        # so it must not change what an observation is.
        assert sac.obs_spec(config).raw_dim == expected[name], name


def test_base_obs_excludes_action_history():
    """The actor must be reactive: no memory of its own actions in the
    observation. EC keeps its a_tm1 input (see test below) -- the actor does
    not, which is what leaves rotation/reflection reachable only through the
    cerebellar error channel."""
    pos = np.array([1.0, 2.0], dtype=np.float32)
    vel = np.array([0.5, -0.5], dtype=np.float32)
    goal = np.array([3.0, 4.0], dtype=np.float32)

    obs = sac.build_base_obs(pos, vel, goal)

    assert obs.shape == (sac.SDIM + 2,)
    assert np.allclose(obs, [1.0, 2.0, 0.5, -0.5, 3.0, 4.0])


def test_ec_still_receives_action_history_even_though_the_actor_does_not():
    """The asymmetry is the point: withholding a_tm1 from the actor while EC
    keeps it is what makes the augmented configs able to infer context."""
    for arm, width in (("main", 10), ("control", 14)):
        ec = M.EC(M.ARMS[arm])
        assert ec.cell.weight_ih.shape[1] == width
        # width accounts for a_tm1's 2 dims: 4 (s_tm1) + 2 (a_tm1) + 4 (e_t)
        # for main, plus s_t's 4 for control
        assert width == (1 if arm == "main" else 2) * sac.SDIM + 2 + sac.SDIM


def test_oracle_one_hot():
    v = oracle_one_hot(2, 4)
    assert v.shape == (4,)
    assert v[2] == 1.0
    assert v.sum() == 1.0


def test_freeze_module_sets_requires_grad_false():
    cb = M.CB()
    assert all(p.requires_grad for p in cb.parameters())
    freeze_module(cb)
    assert not cb.training
    assert all(not p.requires_grad for p in cb.parameters())


@pytest.mark.parametrize("name", ["main_frozen", "control_frozen"])
def test_frozen_configs_load_checkpoint_and_freeze(name):
    config = AblationConfig(**SW.CONFIG_BY_NAME[name], arm_seed=0)
    ec, cb = build_ec_cb(config, "cpu")
    for p in list(ec.parameters()) + list(cb.parameters()):
        assert not p.requires_grad
    assert not ec.training and not cb.training


def test_sac_update_never_touches_frozen_ec_cb_params():
    config = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0)
    ec, cb = build_ec_cb(config, "cpu")
    before = {n: p.clone() for n, p in list(ec.named_parameters()) + list(cb.named_parameters())}

    d = obs_dim(config)
    trainer = _make_trainer(d, target_entropy=-2.0)
    before_actor = trainer.actor.fc_mean.weight.clone()

    batch = sac.SACBatch(
        observations=torch.randn(16, d),
        actions=torch.empty(16, 2).uniform_(-1, 1),
        rewards=torch.randn(16, 1),
        next_observations=torch.randn(16, d),
        dones=torch.zeros(16, 1),
    )
    trainer.update(batch)

    assert not torch.allclose(before_actor, trainer.actor.fc_mean.weight)
    for n, p in list(ec.named_parameters()) + list(cb.named_parameters()):
        assert torch.allclose(before[n], p), n


@pytest.mark.parametrize("name", ["end_to_end", "cb_only_end_to_end"])
def test_ec_cb_params_change_under_end_to_end_training(name):
    config = AblationConfig(**SW.CONFIG_BY_NAME[name])
    ec, cb = build_ec_cb(config, "cpu")
    if name == "cb_only_end_to_end":
        assert ec is None

    params = ([p for p in ec.parameters()] if ec is not None else []) + list(cb.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    trainer = ECCBTrainer(ec, cb, optimizer, arm=config.arm, device="cpu")

    before = [p.clone() for p in params]
    B = 8
    batch = (
        torch.randn(B, sac.SDIM),                 # s_tm1
        torch.empty(B, 2).uniform_(-1, 1),        # a_tm1
        torch.randn(B, sac.SDIM),                 # s_t
        torch.empty(B, 2).uniform_(-1, 1),        # a_t
        torch.randn(B, sac.SDIM),                 # s_next
        torch.randn(B, sac.SDIM),                 # e_t
        torch.zeros(B, sac.EC_HIDDEN),            # h_ec_prev
    )
    result = trainer.update(batch)
    assert np.isfinite(result["ec_cb_loss"])
    assert any(not torch.allclose(b, a) for b, a in zip(before, params))


def test_replay_buffer_round_trip():
    buf = ReplayBuffer(capacity=10)
    obs = np.arange(8, dtype=np.float32)
    buf.add(
        obs, np.array([0.1, 0.2], dtype=np.float32), 1.0, obs + 1, False,
        s_tm1=np.zeros(4, dtype=np.float32), a_tm1=np.zeros(2, dtype=np.float32),
        s_t=np.ones(4, dtype=np.float32), a_t=np.array([0.1, 0.2], dtype=np.float32),
        s_next=np.ones(4, dtype=np.float32) * 2, e_t=np.zeros(4, dtype=np.float32),
        h_ec_prev=np.zeros(sac.EC_HIDDEN, dtype=np.float32), t_step=0,
    )
    assert len(buf) == 1

    sac_batch, eccb_batch, idxs = buf.sample(1, "cpu", t_now=0)
    assert torch.allclose(sac_batch.observations[0], torch.tensor(obs))
    assert torch.allclose(sac_batch.rewards[0], torch.tensor([1.0]))
    assert len(idxs) == 1

    s_tm1, a_tm1, s_t, a_t, s_next, e_t, h_ec_prev = eccb_batch
    assert torch.allclose(s_t[0], torch.ones(4))
    assert h_ec_prev.shape == (1, sac.EC_HIDDEN)


def _add_dummy(buf, value=1.0, t_step=0):
    buf.add(
        np.full(8, value, dtype=np.float32), np.zeros(2, dtype=np.float32), value,
        np.full(8, value, dtype=np.float32), False,
        s_tm1=np.zeros(4, dtype=np.float32), a_tm1=np.zeros(2, dtype=np.float32),
        s_t=np.zeros(4, dtype=np.float32), a_t=np.zeros(2, dtype=np.float32),
        s_next=np.zeros(4, dtype=np.float32), e_t=np.zeros(4, dtype=np.float32),
        h_ec_prev=np.zeros(sac.EC_HIDDEN, dtype=np.float32), t_step=t_step,
    )


def test_sampling_probs_decay_with_age_and_keep_a_uniform_floor():
    """p(e) proportional to exp(-dt/gamma_t), then mixed with uniform -- the
    scheme ported from the previous iteration's _ablations_sweep.py."""
    buf = ReplayBuffer(capacity=10, gamma_t=100.0, mix=0.02, seed=0)
    for t in (0, 50, 100):
        _add_dummy(buf, t_step=t)

    probs = buf._sampling_probs(t_now=100)

    assert probs == pytest.approx(probs.sum() * probs / probs.sum())   # normalized
    assert probs.sum() == pytest.approx(1.0)
    # newest transition is the most likely, oldest the least
    assert probs[2] > probs[1] > probs[0]
    # decay follows exp(-dt/gamma_t) up to the uniform mixture
    raw = np.exp(-np.array([100.0, 50.0, 0.0]) / 100.0)
    raw = raw / raw.sum()
    expected = 0.98 * raw + 0.02 * (1 / 3)
    assert probs == pytest.approx(expected / expected.sum(), rel=1e-6)


def test_sampling_probs_uniform_when_all_transitions_are_the_same_age():
    buf = ReplayBuffer(capacity=10, gamma_t=100.0, mix=0.0, seed=0)
    for _ in range(4):
        _add_dummy(buf, t_step=7)

    probs = buf._sampling_probs(t_now=7)
    assert probs == pytest.approx(np.full(4, 0.25))


def test_sampling_is_biased_toward_recent_transitions():
    """The weighting must actually reach sampling, not just _sampling_probs."""
    n, gamma_t = 50, 10.0
    buf = ReplayBuffer(capacity=100, gamma_t=gamma_t, mix=0.0, seed=0)
    for t in range(n):
        _add_dummy(buf, t_step=t)

    _, _, idxs = buf.sample(4000, "cpu", t_now=n - 1)

    weights = np.exp(-(n - 1 - np.arange(n)) / gamma_t)
    weights /= weights.sum()
    expected_mean = float((np.arange(n) * weights).sum())

    assert np.mean(idxs) == pytest.approx(expected_mean, abs=1.0)
    # and clearly recency-biased rather than uniform, whose mean is (n-1)/2
    assert np.mean(idxs) > (n - 1) / 2 + 5


def test_update_abs_tde_writes_back_to_the_sampled_indices():
    """Inert for sampling while gamma_td is unused, but the bookkeeping must
    stay correct so the term can be switched on."""
    buf = ReplayBuffer(capacity=10, seed=0)
    for t in range(3):
        _add_dummy(buf, t_step=t)
    assert list(buf.abs_tdes) == [0.0, 0.0, 0.0]

    buf.update_abs_tde(np.array([0, 2]), np.array([1.5, 2.5]))

    assert list(buf.abs_tdes) == [1.5, 0.0, 2.5]


def test_sampling_probs_ignore_abs_tde_as_in_the_ported_version():
    """Guards the documented divergence: the sibling reference file folded
    |TD error| into the energy, this one does not. If that changes, it should
    be a deliberate edit rather than a silent drift."""
    buf = ReplayBuffer(capacity=10, gamma_t=100.0, gamma_td=1.0, mix=0.0, seed=0)
    for t in (0, 50):
        _add_dummy(buf, t_step=t)

    before = buf._sampling_probs(t_now=50)
    buf.update_abs_tde(np.array([0, 1]), np.array([0.0, 500.0]))
    after = buf._sampling_probs(t_now=50)

    assert after == pytest.approx(before)


def test_replay_buffer_clear_empties_every_field():
    buf = ReplayBuffer(capacity=10)
    _add_dummy(buf)
    assert len(buf) == 1

    buf.clear()

    assert len(buf) == 0
    for field_name in ReplayBuffer._FIELDS:
        assert len(getattr(buf, field_name)) == 0, field_name


def test_replay_buffer_is_cleared_at_every_phase_switch_when_enabled(monkeypatch):
    """Opt-in path (config.clear_buffer_on_switch=True): no context's
    transitions survive into the next phase. Asserts on buffer contents at
    each boundary rather than trusting capacity eviction to be large enough.

    Set on the CONFIG, not by patching sac.CLEAR_BUFFER_ON_PHASE_SWITCH: the
    run loop reads the config field, and a spawned worker would not see a
    module patch anyway."""
    lengths_before_clear, lengths_after_clear = [], []
    real_clear = ReplayBuffer.clear

    def clear_spy(self):
        lengths_before_clear.append(len(self))
        result = real_clear(self)
        lengths_after_clear.append(len(self))
        return result

    monkeypatch.setattr(ReplayBuffer, "clear", clear_spy)

    config = AblationConfig(name="baseline", clear_buffer_on_switch=True)
    sac.run_sac(
        0, config, n_phase_ep=2, phase_reps=2, max_steps=5,
        buffer_capacity=1000, batch_size=4, learning_starts=0,
        warmup_episodes=0,
    )

    # one clear per phase; every phase after the first began holding data its
    # predecessor accumulated, and left the boundary with an empty buffer
    assert len(lengths_before_clear) == sac.N_PHASE_FAMILIES * 2
    assert lengths_before_clear[0] == 0            # nothing before phase 0
    assert all(n > 0 for n in lengths_before_clear[1:])
    assert all(n == 0 for n in lengths_after_clear)


def test_buffer_is_not_cleared_across_phases_by_default(monkeypatch):
    """Default path, matching the previous iteration: transitions persist
    across a phase switch and are down-weighted by recency rather than
    removed. The flag must actually gate this, or it reads as a documented
    guarantee the code ignores."""
    assert sac.CLEAR_BUFFER_ON_PHASE_SWITCH is False
    assert AblationConfig(name="baseline").clear_buffer_on_switch is False

    cleared = []
    real_clear = ReplayBuffer.clear
    monkeypatch.setattr(ReplayBuffer, "clear",
                         lambda self: (cleared.append(1), real_clear(self))[1])

    config = AblationConfig(name="baseline")
    sac.run_sac(
        0, config, n_phase_ep=2, phase_reps=2, max_steps=5,
        buffer_capacity=1000, batch_size=4, learning_starts=0,
        warmup_episodes=0,
    )

    assert cleared == []


def test_prediction_error_timing_matches_model_rollout_convention(monkeypatch):
    """sac.py reimplements model.rollout's e_t timing inside its RL loop, and
    the frozen ablations depend on that reimplementation being faithful:

      - e_t is forced to zero on an episode's first step (no CB prediction to
        compare against yet), NOT computed as s_t - 0, which would leak the
        start position in as a bogus error.
      - thereafter e_t = s_t - s_pred, where s_pred is the prediction CB made
        on the PREVIOUS step (read-before-write).
    """
    config = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0)

    added = []
    original_add = ReplayBuffer.add

    def capture_add(self, obs, action, reward, next_obs, done, **kw):
        added.append({k: np.array(v, copy=True) for k, v in kw.items()})
        return original_add(self, obs, action, reward, next_obs, done, **kw)

    preds = []
    real_cb_predict = sac.cb_predict

    def capture_cb_predict(cb, s_t, a_t, z_t, device):
        s_pred, h2 = real_cb_predict(cb, s_t, a_t, z_t, device)
        preds.append(s_pred.detach().cpu().numpy().reshape(-1).copy())
        return s_pred, h2

    monkeypatch.setattr(ReplayBuffer, "add", capture_add)
    monkeypatch.setattr(sac, "cb_predict", capture_cb_predict)

    max_steps = 4
    # learning_starts high enough that no SAC update runs: this isolates the
    # augmentation wiring from the policy changing underneath it
    sac.run_sac(0, config, n_phase_ep=1, phase_reps=1, max_steps=max_steps,
                 buffer_capacity=64, batch_size=8, learning_starts=10 ** 9,
                 warmup_episodes=0)

    assert len(added) >= max_steps, "first episode ended early; nothing to check"
    episode = added[:max_steps]          # first episode only, no boundary logic

    assert np.allclose(episode[0]["e_t"], 0.0), "e_t must be zero on step 0"
    assert np.allclose(episode[0]["s_tm1"], 0.0), "s_tm1 must be zero on step 0"
    assert np.allclose(episode[0]["a_tm1"], 0.0), "a_tm1 must be zero on step 0"

    for i in range(1, max_steps):
        assert np.allclose(episode[i]["e_t"], episode[i]["s_t"] - preds[i - 1], atol=1e-5), (
            f"step {i}: e_t must read the CB prediction made at step {i - 1}")
        assert np.allclose(episode[i]["s_t"], episode[i - 1]["s_next"], atol=1e-6), (
            f"step {i}: s_t must continue from the previous step's s_next")
        assert np.allclose(episode[i]["s_tm1"], episode[i - 1]["s_t"], atol=1e-6)


def test_velocity_is_reconstructed_from_consecutive_positions(monkeypatch):
    """sac.py deliberately derives velocity as (pos_t - pos_{t-1})/DT instead
    of reusing task.env_step's own velocity state, because the finite
    difference is what model.py's frozen checkpoints were trained on -- the two
    diverge exactly where position clips at a grid boundary."""
    added = []
    original_add = ReplayBuffer.add

    def capture_add(self, obs, action, reward, next_obs, done, **kw):
        added.append({k: np.array(v, copy=True) for k, v in kw.items()})
        return original_add(self, obs, action, reward, next_obs, done, **kw)

    monkeypatch.setattr(ReplayBuffer, "add", capture_add)

    config = AblationConfig(name="baseline")
    sac.run_sac(0, config, n_phase_ep=1, phase_reps=1, max_steps=4,
                 buffer_capacity=64, batch_size=8, learning_starts=10 ** 9,
                 warmup_episodes=0)

    for step in added:
        pos_t, vel_t = step["s_t"][:2], step["s_t"][2:]
        pos_next, vel_next = step["s_next"][:2], step["s_next"][2:]
        assert np.allclose(vel_next, (pos_next - pos_t) / sac.DT, atol=1e-4)
        del vel_t


def test_zero_latent_and_zero_prediction_match_the_batch_size():
    """Regression: both fallbacks once hardcoded a batch dimension of 1, which
    only shows up when a real replay batch (not a single env step) flows
    through cb_only_end_to_end."""
    B = 7
    s_t = torch.randn(B, sac.SDIM)
    a_t = torch.randn(B, sac.ACTION_DIM)

    z, h = sac.ec_step_or_zero(None, s_t, a_t, s_t, s_t, None, "main", "cpu")
    assert z.shape == (B, sac.LATENT_DIM)
    assert torch.count_nonzero(z) == 0
    assert h is None

    pred, h2 = sac.cb_predict(None, s_t, a_t, z, "cpu")
    assert pred.shape == (B, sac.SDIM)
    assert h2.shape == (B, sac.CB_HIDDEN)
    assert torch.count_nonzero(h2) == 0


def _probe_trainer(config):
    d = obs_dim(config)
    return _make_trainer(d), *build_ec_cb(config, "cpu")


def test_center_out_probe_geometry():
    """Fixed start and equally spaced targets are the whole point -- if the
    probe set varied, trajectories would not be comparable across seeds,
    configs, contexts or occurrences."""
    config = AblationConfig(name="baseline")
    trainer, ec, cb = _probe_trainer(config)

    probe = sac.eval_center_out(trainer, ec, cb, config, sac.PHASE_CONTEXT_IDS[0], 0,
                                 max_steps=6, device="cpu")

    center, goals = probe["center"], probe["goals"]
    # the origin, not GRID_SIZE / 2: position lives on [-POS_LIM, POS_LIM]
    assert np.allclose(center, 0.0)
    assert goals.shape == (sac.EVAL_N_TARGETS, 2)

    d = np.linalg.norm(goals - center, axis=1)
    assert np.allclose(d, sac.EVAL_RADIUS, atol=1e-5)     # exact radius, no jitter

    angles = np.sort(np.arctan2(goals[:, 1] - center[1], goals[:, 0] - center[0]))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
    assert np.allclose(gaps, 2 * np.pi / sac.EVAL_N_TARGETS, atol=1e-5)

    assert probe["trajectories"].shape == (sac.EVAL_N_TARGETS, 7, 2)
    # the start of every trajectory is the shared center
    assert np.allclose(probe["trajectories"][:, 0], center)


def test_probe_goals_lie_inside_the_workspace():
    """The probe's goals must be REACHABLE. env_step clips position to
    [-POS_LIM, POS_LIM], so a goal outside the box can never be arrived at:
    every probe scores success 0.00 with a NaN path ratio, in every context
    and every block, and the cross_context matrix and the center_out panels
    stop agreeing with the training curves beside them. That is what
    EVAL_RADIUS = 2.5 against POS_LIM = 1.0 did (found 2026-09-20), a value
    left over from the pre-rewrite plant."""
    starts, goals = sac.eval_battery()

    assert np.all(np.abs(goals) + sac.GOAL_RADIUS <= T.POS_LIM + 1e-6)
    assert np.all(np.abs(starts) <= T.POS_LIM)
    # and the step budget has to cover the reach at the speed cap
    reach = np.max(np.linalg.norm(goals - starts, axis=1))
    assert reach <= sac.MAX_STEPS * T.V_MAX * T.DT


def test_path_ratio_is_one_for_a_straight_reach_and_nan_for_failures():
    """Normalized against (radius - GOAL_RADIUS) because a reach ends on
    ENTERING the goal circle -- dividing by the full radius would score an
    ideal straight reach below 1.0. Failures must be NaN, not small: a
    truncated trajectory is short and would otherwise look efficient."""
    config = AblationConfig(name="baseline")
    trainer, ec, cb = _probe_trainer(config)

    # a policy that always drives straight at the goal, in the null field
    class StraightActor:
        training = True

        def eval(self): pass

        def train(self): pass

        def get_action(self, obs):
            pos, goal = obs[:, :2], obs[:, 4:6]
            u = goal - pos
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            return u, torch.zeros(obs.shape[0], 1), u

    trainer.actor = StraightActor()
    probe = sac.eval_center_out(trainer, ec, cb, config, sac.NEUTRAL_CONTEXT_PARAMS, 0,
                                 max_steps=200, device="cpu")

    reached = probe["success"] > 0
    assert reached.any(), "straight-line policy should reach in the null field"
    assert np.all(probe["path_ratio"][reached] >= 0.99)
    assert np.all(probe["path_ratio"][reached] < 1.15)
    assert np.all(np.isnan(probe["path_ratio"][~reached]))


def test_center_out_probe_is_deterministic_and_side_effect_free():
    """Uses the actor's mean rather than a sample, so repeated probes agree;
    and it must not perturb the run it is called from."""
    config = AblationConfig(name="baseline")
    trainer, ec, cb = _probe_trainer(config)
    before = {n: p.clone() for n, p in trainer.actor.named_parameters()}

    a = sac.eval_center_out(trainer, ec, cb, config, sac.PHASE_CONTEXT_IDS[0], 0,
                             max_steps=8, device="cpu")
    b = sac.eval_center_out(trainer, ec, cb, config, sac.PHASE_CONTEXT_IDS[0], 0,
                             max_steps=8, device="cpu")

    np.testing.assert_allclose(np.nan_to_num(a["trajectories"]),
                                np.nan_to_num(b["trajectories"]))
    for n, p in trainer.actor.named_parameters():
        assert torch.allclose(before[n], p), n
    assert trainer.actor.training, "probe must restore train mode"


def test_run_sac_returns_one_probe_per_phase(monkeypatch, tmp_path):
    config = AblationConfig(name="baseline")
    curve, probes = sac.run_sac(0, config, n_phase_ep=1, phase_reps=2, max_steps=4,
                                 buffer_capacity=32, batch_size=4, learning_starts=10 ** 9,
                                 warmup_episodes=0)

    n_phases = sac.N_PHASE_FAMILIES * 2              # warmup disabled above
    # every context is evaluated at the end of every phase
    assert len(probes) == n_phases * sac.N_PHASE_FAMILIES
    assert [p["occurrence"] for p in probes] == sorted(p["occurrence"] for p in probes)
    for p in probes:
        assert p["trajectories"].shape[0] == sac.EVAL_N_TARGETS
        assert set(p) >= {"label", "phase", "occurrence", "context_id", "eval_label",
                          "eval_context_id", "is_current", "success", "steps", "first_angle"}

    # exactly one probe per phase is the context that phase trained
    assert sum(p["is_current"] for p in probes) == n_phases
    for p in probes:
        if p["is_current"]:
            assert p["eval_context_id"] == p["context_id"]


def test_center_out_probe_does_not_write_to_the_replay_buffer(monkeypatch):
    """A probe that fed its own trajectories back into training would
    contaminate the very run it is measuring."""
    config = AblationConfig(name="baseline")
    trainer, ec, cb = _probe_trainer(config)

    calls = []
    original_add = ReplayBuffer.add
    monkeypatch.setattr(ReplayBuffer, "add",
                         lambda self, *a, **k: (calls.append(1), original_add(self, *a, **k))[1])

    sac.eval_center_out(trainer, ec, cb, config, sac.PHASE_CONTEXT_IDS[0], 0,
                         max_steps=6, device="cpu")
    assert calls == []


def test_neutral_context_is_the_identity_transformation():
    """The warmup field must favour no context in the phase set, so it has to
    be exactly F = a -- kind 2 (drift) with a zero gain."""
    pos = np.array([2.0, 2.0], dtype=np.float32)
    vel = np.array([0.3, -0.2], dtype=np.float32)
    action = np.array([0.7, -0.4], dtype=np.float32)

    _, vel_neutral = sac.env_step_params(pos, vel, action, sac.NEUTRAL_CONTEXT_PARAMS)
    # task.env_step is V <- LAMBDA*V + F, so F = vel_new - LAMBDA*vel
    force = vel_neutral - T.LAMBDA * vel
    assert np.allclose(force, action, atol=1e-6)


def test_warmup_block_is_not_logged_but_does_train(monkeypatch):
    """The confound this fixes: LEARNING_STARTS used to consume the first
    ~100 episodes of phase 1 with no gradient updates, and only the FIRST
    context paid it. Warmup moves that window into an unlogged null-field
    block, so every logged phase starts with learning already active."""
    config = AblationConfig(name="baseline")

    updates = []
    real_update = SACTrainer.update
    monkeypatch.setattr(SACTrainer, "update",
                         lambda self, b: (updates.append(1), real_update(self, b))[1])

    curve, probes = sac.run_sac(0, config, n_phase_ep=2, phase_reps=1, max_steps=5,
                                 buffer_capacity=256, batch_size=4, learning_starts=4,
                                 warmup_episodes=6)

    # warmup episodes stay out of the curve the analysis reads
    assert len(curve["ep"]) == sac.N_PHASE_FAMILIES * 2
    assert all(p >= 0 for p in curve["phase"])
    assert all(o >= 0 for o in curve["occurrence"])
    assert "null field" not in curve["label"]

    # the warmup block probes the null field AND every context, giving the
    # naive row that memory traces are later measured against
    warmup_probes = [p for p in probes if p["context_id"] is None]
    assert len(warmup_probes) == sac.N_PHASE_FAMILIES + 1
    assert warmup_probes[0]["eval_label"] == "null field"
    assert sum(p["is_current"] for p in warmup_probes) == 1
    assert len(updates) > 0


def test_cross_context_probes_tell_the_oracle_the_context_it_is_evaluated_in():
    """A stale one-hot would measure the oracle being lied to, not the oracle.
    Captures the observation each probe was actually given."""
    config = AblationConfig(**SW.CONFIG_BY_NAME["oracle"])
    seen = []
    real = sac.augmented_obs_step

    def spy(*args, **kw):
        # oracle_vec is the 12th positional argument; take it by name when the
        # caller passes it that way. Written variadically on purpose: this spy
        # broke twice when augmented_obs_step grew a parameter.
        oracle_vec = kw.get("oracle_vec", args[12] if len(args) > 12 else None)
        seen.append(None if oracle_vec is None else np.array(oracle_vec, copy=True))
        return real(*args, **kw)

    embed = sac.make_oracle_embed(0)
    import unittest.mock as mock
    with mock.patch.object(sac, "augmented_obs_step", spy):
        trainer, ec, cb = _probe_trainer(config)
        for fi, cid in enumerate(sac.PHASE_CONTEXT_IDS):
            seen.clear()
            sac.eval_center_out(trainer, ec, cb, config, T.CONTEXTS[cid][:4], fi,
                                 max_steps=4, oracle_embed=embed, device="cpu")
            assert seen, "probe made no observations"
            # dense code now, so identity is "equals this context's row",
            # not "argmax is this context"
            for vec in seen:
                assert vec is not None and vec.shape == (sac.ORACLE_EMBED_DIM,)
                assert np.allclose(vec, embed[fi]), (fi, vec[:4], embed[fi][:4])


def test_warmup_leaves_learning_active_from_the_first_logged_episode():
    """With warmup on, phase 1 must not begin with a random-action window --
    the buffer and step count are already past the learning gate."""
    config = AblationConfig(name="baseline")
    warmup_eps, max_steps = 8, 5

    curve, _ = sac.run_sac(0, config, n_phase_ep=1, phase_reps=1, max_steps=max_steps,
                            buffer_capacity=256, batch_size=4, learning_starts=4,
                            warmup_episodes=warmup_eps)

    # q_loss is 0.0 only when no update has run yet for that episode
    assert curve["q_loss"][0] != 0.0, "phase 1 episode 1 should already be learning"


def test_free_sampling_spans_the_grid_and_stays_in_bounds():
    """Goals sit ANYWHERE in the grid now, not on a ring: the distance must
    actually span the range (up to near the diagonal), both points must stay
    in [-POS_LIM, POS_LIM], and the one surviving constraint -- no goal inside
    GOAL_RADIUS of the start, which would succeed at step 0 -- must hold."""
    rng = np.random.default_rng(0)
    pairs = [sac.sample_pos_goal(rng) for _ in range(4000)]
    d = np.array([np.linalg.norm(p - g) for p, g in pairs])

    for p, g in pairs:
        assert np.all(g >= -T.POS_LIM - 1e-4) and np.all(g <= T.POS_LIM + 1e-4)
        assert np.all(p >= -T.POS_LIM - 1e-4) and np.all(p <= T.POS_LIM + 1e-4)

    # every episode starts outside GOAL_RADIUS, or it would succeed at step 0
    assert d.min() > sac.GOAL_RADIUS

    # the point of the change: distance is free, not pinned near 0.75
    diag = 2.0 * T.POS_LIM * np.sqrt(2.0)
    assert d.max() <= diag + 1e-4
    assert d.max() > 0.8 * diag, d.max()
    assert d.min() < 0.2, d.min()
    # mean distance between two uniform points in a square of side L is
    # ~0.5214*L; the rejected disc is tiny so it barely moves this
    assert abs(d.mean() - 0.5214 * 2.0 * T.POS_LIM) < 0.05, d.mean()

    # both marginals cover the grid, i.e. the start is not confined to an
    # inner box any more
    all_pts = np.array([q for pair in pairs for q in pair])
    assert all_pts.min() < -0.95 * T.POS_LIM and all_pts.max() > 0.95 * T.POS_LIM


def test_goal_directions_are_covered_uniformly():
    """Direction must stay uniform: it interacts with the context (rotation
    needs the desired direction rotated by -theta; reflection's offset depends
    on action direction), so a biased sample would make some contexts
    systematically easier and confound the ablation."""
    rng = np.random.default_rng(0)
    angles = []
    for _ in range(4000):
        pos, goal = sac.sample_pos_goal(rng)
        d = goal - pos
        angles.append(np.arctan2(d[1], d[0]))

    counts, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    expected = len(angles) / 8
    # no octant should deviate more than ~15% from uniform
    assert np.all(np.abs(counts - expected) / expected < 0.15), counts


def test_goal_env_step_clips_to_grid():
    pos = np.array([0.01, T.GRID_SIZE - 0.01], dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    action = np.array([-10.0, 10.0], dtype=np.float32)
    pos_new, _ = sac.goal_env_step(pos, vel, action, sac.PHASE_CONTEXT_IDS[0])
    assert np.all(pos_new >= 0) and np.all(pos_new <= T.GRID_SIZE)


"""Reward tests below assert PROPERTIES, not the formula.

The shaping function is a design knob under active tuning (linear distance,
sigmoid, its length scale), so re-deriving the arithmetic here just means the
tests break on every retune while catching nothing. What must hold regardless
of the shape: the success bonus is additive rather than a replacement, reward
decreases with distance, and reaching the goal costs no shaping penalty.
"""


def _at(d, goal_xy=(2.0, 2.0)):
    """Position at distance d from the goal, along +x."""
    return np.array([goal_xy[0] + d, goal_xy[1]], dtype=np.float32)


def _reward_at(d, goal_xy=(2.0, 2.0), d_prev=None):
    """Reward for arriving at distance d. d_prev defaults to d, i.e. no
    progress, which isolates the absolute-distance term."""
    goal = np.array(goal_xy, dtype=np.float32)
    d_prev = d if d_prev is None else d_prev
    return sac.step_reward(_at(d, goal_xy), goal, _at(d_prev, goal_xy))[0]


def test_step_reward_bonus_is_additive_across_the_goal_boundary():
    """Shaping is continuous in d, so the jump across GOAL_RADIUS isolates the
    bonus: if it replaced the shaping term instead of adding to it, the step
    would not equal SUCCESS_BONUS."""
    eps = 1e-3
    inside = _reward_at(sac.GOAL_RADIUS - eps)
    outside = _reward_at(sac.GOAL_RADIUS + eps)

    assert inside - outside == pytest.approx(sac.SUCCESS_BONUS, abs=1e-2)


def test_step_reward_flags_success_only_inside_goal_radius():
    goal = np.array([2.0, 2.0], dtype=np.float32)
    _, done_in, _ = sac.step_reward(_at(sac.GOAL_RADIUS / 2), goal, _at(1.0))
    _, done_out, _ = sac.step_reward(_at(sac.GOAL_RADIUS * 2), goal, _at(1.0))
    assert done_in and not done_out


def test_step_reward_does_not_penalize_arriving_at_the_goal():
    """Reaching the goal must never pay less than the bonus alone: the shaping
    term at d=0 is the best it ever gets, whatever its amplitude or offset."""
    goal = np.array([2.0, 2.0], dtype=np.float32)
    at_goal, _, _ = sac.step_reward(goal.copy(), goal, goal.copy())
    assert at_goal >= sac.SUCCESS_BONUS


def test_progress_term_rewards_closing_and_penalizes_opening():
    closed = _reward_at(1.0, d_prev=1.2)     # moved toward the goal
    still = _reward_at(1.0, d_prev=1.0)      # no progress
    opened = _reward_at(1.0, d_prev=0.8)     # moved away

    assert closed > still > opened


def _progress_contribution(d, d_prev):
    """The progress term alone, with the absolute-distance term subtracted off
    by comparing against a zero-progress step ending at the same d."""
    return _reward_at(d, d_prev=d_prev) - _reward_at(d)


def test_progress_is_relative_so_equal_fractions_pay_equally():
    """Closing 10% of the gap pays the same near and far -- that scale-freedom
    is what compresses start-distance variance in the episode return."""
    near = _progress_contribution(0.9, 1.0)     # 10% of a 1.0 gap
    far = _progress_contribution(3.6, 4.0)      # 10% of a 4.0 gap

    assert near == pytest.approx(far, rel=1e-4)
    assert near == pytest.approx(sac.PROGRESS_WEIGHT * 0.1, rel=1e-4)


def test_moving_away_is_punished_more_than_moving_closer_is_rewarded():
    """Isolates PROGRESS_PENALTY_SCALE by holding the DENOMINATOR fixed at
    d_prev=1.0, so both steps are the same fraction of the same gap. Comparing
    unequal denominators instead would conflate the explicit penalty with the
    relative form's own asymmetry (a retreat divides by the smaller d_prev),
    giving 2.22 rather than 2.0."""
    forward = _progress_contribution(0.9, 1.0)    # progress = +0.1
    backward = _progress_contribution(1.1, 1.0)   # progress = -0.1

    assert backward < 0 < forward
    assert abs(backward) / forward == pytest.approx(sac.PROGRESS_PENALTY_SCALE, rel=1e-4)


def test_progress_term_cannot_be_farmed_by_oscillating():
    """A round trip must net negative, so approach-retreat cycles earn nothing.
    Holds for the relative form even without the asymmetry (the retreat leg
    divides by the smaller d_prev), and the asymmetry deepens it. This is the
    property max(0, d_prev - d) would break."""
    out = _reward_at(1.2, d_prev=1.0)
    back = _reward_at(1.0, d_prev=1.2)
    assert out + back < 0

    # progress alone, with the distance term subtracted off, is still negative
    progress_only = (out + back) - (_reward_at(1.2) + _reward_at(1.0))
    assert progress_only < 0


def test_step_reward_decreases_with_distance():
    """Monotonic over distances that matter -- a saturating shape flattens far
    out, so this stays inside the band where a gradient must exist."""
    rewards = [_reward_at(d) for d in (sac.GOAL_RADIUS * 1.5, 0.75, 1.0, 1.5)]
    assert all(a > b for a, b in zip(rewards, rewards[1:]))


def test_step_reward_is_finite_and_bounded_across_the_whole_grid():
    """Loose sanity guard only -- catches NaN/inf and runaway magnitudes
    without pinning down the shaping function's amplitude, which is a knob."""
    diag = float(np.hypot(T.GRID_SIZE, T.GRID_SIZE))
    values = [_reward_at(d, goal_xy=(0.0, 0.0)) for d in np.linspace(0.0, diag, 40)]
    assert all(np.isfinite(v) for v in values)
    assert all(abs(v) <= sac.SUCCESS_BONUS + 50 for v in values)


# ── The controller's readout of the circuit ──────────────────────────────

def test_observation_carries_the_shared_z_and_cb_hidden_layer():
    """EC must reach the actor through the SAME z_t that CB is conditioned on
    -- not a second readout of h_t, and not h_t itself -- while CB reaches it
    through h2 rather than s_pred. Checks the actual values, since a slice of
    the right size taken from the wrong tensor would pass a shape-only test."""
    config = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0)
    ec, cb = build_ec_cb(config, "cpu")
    base = sac.SDIM + 2

    h = torch.zeros(1, sac.EC_HIDDEN)
    s_tm1, a_tm1 = torch.zeros(1, sac.SDIM), torch.zeros(1, sac.ACTION_DIM)
    s_t = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    e_t = torch.zeros(1, sac.SDIM)
    cb_h2 = torch.zeros(1, sac.CB_HIDDEN)
    cb_proj = sac.make_cb_proj(0)
    pos, vel = np.float32([1.0, 1.0]), np.zeros(2, np.float32)
    goal = np.float32([2.5, 1.0])

    for _ in range(3):
        obs, z_t, h_new = sac.augmented_obs_step(
            ec, config, h, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
            pos, vel, goal, None, "cpu")
        ec_block = obs[base:base + sac.LATENT_DIM]
        cb_block = obs[base + sac.LATENT_DIM:]

        assert ec_block.shape == (sac.LATENT_DIM,)
        assert cb_block.shape == (sac.CB_PROJ_DIM,)
        assert np.allclose(ec_block, z_t.detach().numpy()[0]), (
            "EC block must be z_t, the same tensor handed to CB")
        assert not np.allclose(ec_block, h_new.detach().numpy()[0][:sac.LATENT_DIM]), (
            "z_t must be the readout of h_t, not a slice of it")
        assert np.allclose(cb_block, (cb_h2 @ cb_proj).numpy()[0], atol=1e-5), (
            "CB block must be the PREVIOUS step's h2 through cb_proj -- a CB "
            "pass needs a_t, which is chosen from this very observation")

        # same tensor, both destinations -- that is the design commitment
        assert torch.equal(ec.readout(h_new), z_t)

        a_t = torch.tensor([[0.3, -0.2]])
        s_pred, cb_h2 = sac.cb_predict(cb, s_t, a_t, z_t, "cpu")
        s_next = s_t + torch.tensor([[0.04, 0.01, 0.5, 0.2]])
        h, s_tm1, a_tm1, s_t, e_t = h_new, s_t, a_t, s_next, s_next - s_pred
        pos = pos + np.float32([0.04, 0.01])


def test_prediction_error_is_linearly_recoverable_from_the_cb_block():
    """Why e_t is not in the observation any more. cb.head is linear, so
    e_t = s_t - (W @ h2 + b) is an exactly linear function of quantities the
    observation already carries -- the learned mapping can form it. If CB ever
    grew a nonlinear head this stops holding and e_t has to go back in."""
    _ec, cb = build_ec_cb(AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0), "cpu")
    B = 5
    s_t = torch.randn(B, sac.SDIM)
    a_t = torch.randn(B, sac.ACTION_DIM)
    z_t = torch.tanh(torch.randn(B, sac.LATENT_DIM))
    s_next = torch.randn(B, sac.SDIM)

    s_pred, h2 = sac.cb_predict(cb, s_t, a_t, z_t, "cpu")
    head = cb.net[4]
    assert isinstance(head, torch.nn.Linear), "recoverability needs a linear head"
    assert torch.allclose(s_next - s_pred,
                          s_next - (h2 @ head.weight.T + head.bias), atol=1e-5)


def test_cb_projection_is_fixed_random_and_untrained():
    """Ported from the reference: a FIXED random CB -> MC readout with
    unit-norm columns, drawn per run and never trained. It is applied while
    the observation is built, so one projection is shared by the actor and
    both critics and the projected feature is what the buffer stores."""
    P = sac.make_cb_proj(0)
    assert P.shape == (sac.CB_HIDDEN, sac.CB_PROJ_DIM)
    assert torch.allclose(P.norm(dim=0), torch.ones(sac.CB_PROJ_DIM), atol=1e-5)
    assert not P.requires_grad
    assert torch.equal(sac.make_cb_proj(0), sac.make_cb_proj(0)), "must be deterministic"
    assert not torch.equal(sac.make_cb_proj(0), sac.make_cb_proj(1)), (
        "a per-run draw keeps one unlucky projection from becoming a result")


def test_networks_read_the_observation_directly():
    """No readout module inside the networks any more -- the circuit channels
    are already projected by the time the trunk sees them, which is what makes
    the projection shared rather than per-network."""
    spec = sac.obs_spec(AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0))
    actor, qf = Actor(spec), SoftQNetwork(spec)
    assert not hasattr(actor, "read") and not hasattr(qf, "read")
    assert spec.feature_dim == spec.raw_dim
    assert actor.backbone[0].weight.shape[1] == spec.raw_dim
    assert qf.net[0].weight.shape[1] == spec.raw_dim + sac.ACTION_DIM


def test_rl_gradient_cannot_reach_a_frozen_circuit():
    """The projection is applied under no_grad while the observation is built,
    and the observation reaches the buffer as numpy -- so "frozen" stays
    frozen however the actor learns to read it."""
    config = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0)
    ec, cb = build_ec_cb(config, "cpu")
    freeze_module(ec); freeze_module(cb)
    before = [p.detach().clone() for p in list(ec.parameters()) + list(cb.parameters())]

    spec = sac.obs_spec(config)
    trainer = _make_trainer(spec, target_entropy=-2.0)
    actor_before = trainer.actor.backbone[0].weight.detach().clone()

    torch.manual_seed(0)
    B = 16
    batch = sac.SACBatch(
        observations=torch.randn(B, spec.raw_dim),
        actions=torch.rand(B, sac.ACTION_DIM) * 2 - 1,
        rewards=torch.randn(B, 1),
        next_observations=torch.randn(B, spec.raw_dim),
        dones=torch.zeros(B, 1),
    )
    trainer.update(batch)

    assert not torch.allclose(trainer.actor.backbone[0].weight, actor_before)
    for b, p in zip(before, list(ec.parameters()) + list(cb.parameters())):
        assert torch.equal(b, p), "no RL gradient may reach a frozen circuit"


def test_channel_widths_match_the_ported_design():
    """EC contributes LATENT_DIM (z passed through) and CB contributes
    CB_PROJ_DIM (h2 through cb_proj). The widths are unequal here, unlike the
    reference's 32/32, because this repo's EC latent is 64 -- a property of
    the checkpoints, not a choice."""
    for name, cfg in SW.CONFIG_BY_NAME.items():
        config = AblationConfig(**cfg)
        spec = sac.obs_spec(config)
        want = spec.base + (sac.LATENT_DIM if config.use_ec else 0) \
                         + (sac.CB_PROJ_DIM if config.use_cb else 0)
        assert spec.feature_dim == want, name


@pytest.mark.parametrize("variant", list(M.INPUT_VARIANTS))
def test_circuit_state_matches_model_build_exactly(variant):
    """The frozen checkpoints were fitted on model.build's units, which scale
    position by task.S_SCALE and derive velocity from the SCALED positions.
    Tied to model.build itself rather than to a hardcoded 15, so the two
    cannot drift apart.

    Regression: sac.py originally fed raw units, running the checkpoints 15x
    below their input scale. Nothing errored -- z simply stopped carrying
    context (6-way decode 0.28 vs 1.00, chance 0.17), and both frozen arms
    looked identical to the baseline.
    """
    torch.manual_seed(0)
    B, L = 2, 4
    # trans[b, t] = [x, y, ax, ay, x', y']  (see data.generate_sequences)
    pos = torch.rand(B, L, 2) * T.GRID_SIZE
    trans = torch.cat([pos, torch.rand(B, L, 2) * 2 - 1,
                       torch.rand(B, L, 2) * T.GRID_SIZE], dim=-1)
    obs, _act, _tgt = M.build(trans, variant)

    for b in range(B):
        for t in range(L):
            # raw finite-difference velocity, exactly as the rollout forms it;
            # circuit_state is responsible for matching build's units
            vel = (np.zeros(2, np.float32) if t == 0
                   else ((pos[b, t] - pos[b, t - 1]).numpy() / sac.DT))
            assert np.allclose(sac.circuit_state(pos[b, t].numpy(), vel, variant),
                               obs[b, t].numpy(), atol=1e-3), (variant, b, t)


def test_rollout_feeds_the_circuit_scaled_state():
    """End to end: what actually reaches the replay buffer as the circuit's
    state must be in model.build's units, with the finite-difference velocity
    relation still exact under the scaling."""
    added = []
    original_add = ReplayBuffer.add

    def capture_add(self, obs, action, reward, next_obs, done, **kw):
        added.append({k: np.array(v, copy=True) for k, v in kw.items()})
        return original_add(self, obs, action, reward, next_obs, done, **kw)

    import unittest.mock as mock
    config = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0)
    with mock.patch.object(ReplayBuffer, "add", capture_add):
        sac.run_sac(0, config, n_phase_ep=1, phase_reps=1, max_steps=5,
                     buffer_capacity=64, batch_size=8, learning_starts=10 ** 9,
                     warmup_episodes=0)

    assert added
    for step in added:
        # the actor's observation keeps raw position; the circuit's does not
        raw_pos = step["s_t"][:2] / sac.S_SCALE
        assert np.all(raw_pos >= -1e-4) and np.all(raw_pos <= T.GRID_SIZE + 1e-4)
        assert np.allclose(step["s_next"][2:],
                           (step["s_next"][:2] - step["s_t"][:2]) / sac.DT, atol=1e-3)
    # a raw-unit state would sit inside the grid; a scaled one must not, or
    # the scaling is not actually being applied
    assert max(np.abs(s["s_t"][:2]).max() for s in added) > T.GRID_SIZE


# ── Replayed EC/CB training ──────────────────────────────────────────────

def _e2e_trainer(name="end_to_end"):
    config = AblationConfig(**SW.CONFIG_BY_NAME[name])
    ec, cb = build_ec_cb(config, "cpu")
    params = ([p for p in ec.parameters()] if ec is not None else []) + list(cb.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    return config, ec, cb, ECCBTrainer(ec, cb, opt, arm=config.arm, device="cpu")


def _e2e_batch(B=6, first_rows=0):
    """A replayed batch. first_rows leading rows imitate an episode's first
    step, where the rollout stores s_tm1/a_tm1/e_t as exact zeros."""
    g = torch.Generator().manual_seed(0)
    s_tm1 = torch.rand(B, sac.SDIM, generator=g) * 30 + 5      # scaled units
    a_tm1 = torch.rand(B, sac.ACTION_DIM, generator=g) * 2 - 1
    s_t = torch.rand(B, sac.SDIM, generator=g) * 30 + 5
    a_t = torch.rand(B, sac.ACTION_DIM, generator=g) * 2 - 1
    s_next = torch.rand(B, sac.SDIM, generator=g) * 30 + 5
    e_t = torch.randn(B, sac.SDIM, generator=g)
    h_ec_prev = torch.randn(B, sac.EC_HIDDEN, generator=g) * 0.3
    if first_rows:
        s_tm1[:first_rows] = 0.0
        a_tm1[:first_rows] = 0.0
        e_t[:first_rows] = 0.0
        h_ec_prev[:first_rows] = 0.0
    return s_tm1, a_tm1, s_t, a_t, s_next, e_t, h_ec_prev


def test_replayed_error_is_recomputed_from_the_stored_hidden():
    """e_t must be rebuilt as s_t - cb(s_tm1, a_tm1, ec.readout(h_ec_prev)),
    since ec.readout(h_ec_prev) is exactly z_{t-1} -- the latent CB was
    conditioned on for the prediction e_t was measured against."""
    _config, ec, cb, trainer = _e2e_trainer()
    s_tm1, a_tm1, s_t, _a_t, _s_next, e_stored, h_ec_prev = _e2e_batch()

    got = trainer.recompute_error(s_tm1, a_tm1, s_t, h_ec_prev, e_stored)
    want = s_t - cb(s_tm1, a_tm1, ec.readout(h_ec_prev))
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, e_stored), "the stored error was passed through"


def test_replayed_error_matches_model_rollout_one_step_slice():
    """The invariant that makes this the right reconstruction: replaying one
    transition must reproduce exactly the e_t that model.rollout computes at
    the same step during pretraining."""
    _config, ec, cb, trainer = _e2e_trainer()
    torch.manual_seed(0)
    B, L = 3, 2
    pos = torch.rand(B, L, 2) * T.GRID_SIZE
    trans = torch.cat([pos, torch.rand(B, L, 2) * 2 - 1,
                       torch.rand(B, L, 2) * T.GRID_SIZE], dim=-1)
    obs, act, _tgt = M.build(trans)

    with torch.no_grad():
        _Z, P, _target, sites = M.rollout(ec, cb, trans, "main", sites=True)
        # rollout's t=0 convention: e_0 = 0, so the first real error is e_1
        want = obs[:, 1] - P[:, 0]
        got = trainer.recompute_error(obs[:, 0], act[:, 0], obs[:, 1],
                                      sites["mtl_hidden"][:, 0],
                                      torch.zeros(B, sac.SDIM))
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_replayed_error_carries_gradient_into_cb_and_ec_readout():
    """The whole point of the fix: the error EC consumes must be a live tensor,
    so CB is shaped by the signal it supplies EC and not only by its own
    prediction accuracy."""
    _config, ec, cb, trainer = _e2e_trainer()
    s_tm1, a_tm1, s_t, _a_t, _s_next, e_stored, h_ec_prev = _e2e_batch()

    e_t = trainer.recompute_error(s_tm1, a_tm1, s_t, h_ec_prev, e_stored)
    assert e_t.requires_grad and e_t.grad_fn is not None

    for tensor, label in ((cb.net[4].weight, "cb head"),
                          (ec.readout[0].weight, "ec readout")):
        g, = torch.autograd.grad(e_t.sum(), tensor, retain_graph=True,
                                  allow_unused=True)
        assert g is not None and torch.any(g != 0), f"no gradient reaches {label}"


def test_first_step_of_an_episode_keeps_its_zero_error():
    """model.rollout forces e_0 = 0 because there is no previous prediction.
    Recomputing it there would fabricate an error from ec.readout's bias."""
    _config, _ec, _cb, trainer = _e2e_trainer()
    s_tm1, a_tm1, s_t, _a_t, _s_next, e_stored, h_ec_prev = _e2e_batch(B=6, first_rows=2)

    got = trainer.recompute_error(s_tm1, a_tm1, s_t, h_ec_prev, e_stored)
    assert torch.count_nonzero(got[:2]) == 0, "t=0 rows must stay at zero"
    assert torch.any(got[2:] != 0), "t>0 rows must be recomputed"


def test_cb_only_config_defers_to_the_stored_error():
    """With no EC, ec_step_or_zero ignores e_t entirely -- there is nothing for
    a recomputed value to feed, so it passes through untouched."""
    config, ec, _cb, trainer = _e2e_trainer("cb_only_end_to_end")
    assert ec is None and not config.use_ec
    batch = _e2e_batch()
    got = trainer.recompute_error(batch[0], batch[1], batch[2], batch[6], batch[5])
    assert torch.equal(got, batch[5])


def test_replayed_update_still_trains_both_modules():
    """The fix must not break what already worked: one update moves EC and CB."""
    _config, ec, cb, trainer = _e2e_trainer()
    before = [p.detach().clone() for p in list(ec.parameters()) + list(cb.parameters())]
    out = trainer.update(_e2e_batch())

    assert np.isfinite(out["ec_cb_loss"])
    after = list(ec.parameters()) + list(cb.parameters())
    assert any(not torch.equal(b, p) for b, p in zip(before, after))


# ── Per-source input scaling ─────────────────────────────────────────────

def test_init_by_source_gives_each_source_its_own_fan_in():
    """std 1/sqrt(n_sources * n) per source, not one bound from the total
    fan-in. Without the per-source part, a source's influence at init grows
    with how many dimensions it spans: 6 kinematic dims beside 64 circuit dims
    left the kinematics at 2% of the trunk's pre-activation variance."""
    torch.manual_seed(0)
    widths = [6, 0, 32, 256]
    k = sum(1 for w in widths if w)
    lin = torch.nn.Linear(sum(widths), 4096)
    sac.init_by_source(lin, widths)

    start = 0
    for w in widths:
        if w:
            want = 1.0 / np.sqrt(k * w)
            got = float(lin.weight[:, start:start + w].std())
            assert abs(got - want) < 0.1 * want, (w, got, want)
        start += w


def test_init_by_source_preserves_the_overall_activation_scale():
    """The 1/sqrt(n_sources) factor. Per-source scaling alone gives each
    source a FULL unit of pre-activation variance rather than a share of one,
    so the total inflates as sqrt(n_sources) -- which breaks the criterion the
    default init exists to satisfy and shows up downstream as inflated Q
    values. The pre-activation std must stay ~1 for unit-variance input,
    matching what a single-source fan-in init would give."""
    torch.manual_seed(0)
    widths = [8, 0, 32, 64]
    lin = torch.nn.Linear(sum(widths), 512)
    sac.init_by_source(lin, widths)
    with torch.no_grad():
        lin.bias.zero_()
        pre = lin(torch.randn(4096, sum(widths)))
    assert 0.7 < float(pre.std()) < 1.4, float(pre.std())

    # and every source still contributes equally, which is the actual point
    shares, start = [], 0
    x = torch.randn(4096, sum(widths))
    with torch.no_grad():
        for w in widths:
            if w:
                shares.append(float((x[:, start:start+w] @ lin.weight[:, start:start+w].T).std()))
            start += w
    assert max(shares) / min(shares) < 1.25, shares


def test_init_by_source_rejects_a_mismatched_decomposition():
    lin = torch.nn.Linear(10, 3)
    with pytest.raises(ValueError):
        sac.init_by_source(lin, [6, 2])


def test_action_is_its_own_source_in_the_critic():
    """The critic reads [features, action]; the action must not inherit the
    feature block's scaling."""
    spec = sac.obs_spec(AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0))
    qf = SoftQNetwork(spec)
    assert qf.net[0].weight.shape[1] == spec.feature_dim + sac.ACTION_DIM
    # four sources here: kin, ec->32, cb->32, action
    k = sum(1 for w in spec.feature_widths + [sac.ACTION_DIM] if w)
    act_cols = qf.net[0].weight[:, spec.feature_dim:]
    want = 1.0 / np.sqrt(k * sac.ACTION_DIM)
    assert abs(float(act_cols.std()) - want) < 0.2 * want, (float(act_cols.std()), want)


def test_input_scales_are_still_calibrated():
    """Drift guard on sac's fixed input scales. They are literals measured
    from a real rollout, so anything that moves the observation statistics --
    S_SCALE, the goal sampling, MAX_STEPS, the phase set, new checkpoints --
    invalidates them silently. Tolerance is loose (40%) because the constants
    are a single set shared across arms and seeds that vary by ~30%; it is
    there to catch a factor-of-several move, not to pin a digit."""
    import circuit_probe as CP
    got = CP.source_rms("main", 0, n_episodes=12, max_steps=25)
    for key, stored in (("kin", sac.KIN_RMS), ("ec", sac.EC_STATE_RMS),
                        ("cb", sac.CB_STATE_RMS)):
        assert abs(got[key] - stored) / stored < 0.40, (key, got[key], stored)




def test_only_the_circuit_sees_scaled_units():
    """S_SCALE is input conditioning for EC/CB, not physics: the actor's own
    observation, and so the reward and goal test, stay in raw units. Survives
    the port back to the reference's channel construction, which changed what
    is READ off the circuit but not what is fed INTO it."""
    pos = np.array([1.0, 2.0], dtype=np.float32)
    vel = np.array([0.5, -0.5], dtype=np.float32)
    goal = np.array([3.0, 4.0], dtype=np.float32)

    assert np.allclose(sac.build_base_obs(pos, vel, goal),
                       [1.0, 2.0, 0.5, -0.5, 3.0, 4.0])
    assert np.allclose(sac.circuit_state(pos, vel),
                       sac.S_SCALE * np.concatenate([pos, vel]))
    assert sac.S_SCALE != 1.0, "a unit S_SCALE would make this test vacuous"


# ── The oracle's dense context code ──────────────────────────────────────

def test_oracle_code_is_dense_unit_norm_and_near_orthogonal():
    """The oracle now lives in EC's latent space so it differs from z in WHAT
    it knows, not in how that knowledge is shaped. A one-hot differed on three
    axes at once -- width, density, and exact orthogonality -- and any of them
    could explain the oracle's retention."""
    E = sac.make_oracle_embed(0)
    assert E.shape == (sac.N_PHASE_FAMILIES, sac.ORACLE_EMBED_DIM)
    assert sac.ORACLE_EMBED_DIM == sac.LATENT_DIM, "must match EC's latent space"
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)

    # dense: every code uses essentially all of its dimensions
    assert np.mean(np.abs(E) > 1e-6) > 0.99
    assert np.mean(np.abs(sac.oracle_one_hot(0)) > 1e-6) < 0.3, "the control is sparse"

    # near-orthogonal, not exactly: sd of a random cosine in d dims is 1/sqrt(d)
    C = E @ E.T
    off = C[~np.eye(len(C), dtype=bool)]
    assert np.abs(off).max() < 5.0 / np.sqrt(sac.ORACLE_EMBED_DIM), off
    assert np.abs(off).max() > 1e-6, "exact orthogonality would defeat the point"


def test_oracle_embed_is_deterministic_and_per_run():
    assert np.allclose(sac.make_oracle_embed(3), sac.make_oracle_embed(3))
    assert not np.allclose(sac.make_oracle_embed(0), sac.make_oracle_embed(1))
    # a distinct stream from cb_proj, so the two cannot be accidentally coupled
    assert sac.ORACLE_EMBED_SEED_BASE != sac.CB_PROJ_SEED_BASE


def test_oracle_code_is_zero_when_there_is_no_context():
    """The null-field warmup has no context to encode."""
    E = sac.make_oracle_embed(0)
    assert np.all(sac.oracle_code(None, E) == 0.0)
    assert np.allclose(sac.oracle_code(2, E), E[2])


def test_oracle_observation_carries_the_dense_code():
    """End to end: what reaches the buffer for the oracle config is the dense
    row, at LATENT_DIM width, in the slot right after the kinematics."""
    added = []
    original_add = ReplayBuffer.add

    def capture_add(self, obs, action, reward, next_obs, done, **kw):
        added.append(np.array(obs, copy=True))
        return original_add(self, obs, action, reward, next_obs, done, **kw)

    import unittest.mock as mock
    config = AblationConfig(**SW.CONFIG_BY_NAME["oracle"])
    with mock.patch.object(ReplayBuffer, "add", capture_add):
        sac.run_sac(0, config, n_phase_ep=1, phase_reps=1, max_steps=4,
                     buffer_capacity=64, batch_size=8, learning_starts=10 ** 9,
                     warmup_episodes=0)

    embed = sac.make_oracle_embed(0)
    base = sac.SDIM + 2
    assert added and added[0].shape == (base + sac.ORACLE_EMBED_DIM,)
    block = added[0][base:]
    assert np.allclose(block, embed[0], atol=1e-6), "first phase is family 0"


# ── Persisting a trained run ─────────────────────────────────────────────

def _tiny_run(tmp_path, config_name, seed=0):
    cfg = AblationConfig(**SW.CONFIG_BY_NAME[config_name])
    path = str(tmp_path / "agent.pt")
    curve, _probes = sac.run_sac(seed, cfg, n_phase_ep=2, phase_reps=1, max_steps=5,
                                  buffer_capacity=64, batch_size=8, learning_starts=8,
                                  warmup_episodes=2, agent_path=path)
    return cfg, path


@pytest.mark.parametrize("config_name", ["main_frozen", "oracle"])
def test_saved_agent_round_trips_to_identical_outputs(tmp_path, config_name):
    """A reloaded actor and both critics must reproduce the trained ones
    exactly -- otherwise a sensitivity analysis is measuring reload error."""
    cfg, path = _tiny_run(tmp_path, config_name)
    agent = sac.load_agent(path)

    assert agent.config.name == cfg.name
    assert agent.spec.raw_dim == sac.obs_spec(cfg).raw_dim

    torch.manual_seed(0)
    obs = torch.randn(7, agent.spec.raw_dim)
    act = torch.rand(7, sac.ACTION_DIM) * 2 - 1
    with torch.no_grad():
        # deterministic mean, so this compares policies not exploration noise
        _, _, mean_a = agent.actor.get_action(obs)
        reloaded = sac.load_agent(path)
        _, _, mean_b = reloaded.actor.get_action(obs)
        assert torch.equal(mean_a, mean_b)
        for x, y in ((agent.qf1, reloaded.qf1), (agent.qf2, reloaded.qf2),
                     (agent.qf1_target, reloaded.qf1_target),
                     (agent.qf2_target, reloaded.qf2_target)):
            assert torch.equal(x(obs, act), y(obs, act))


def test_saved_agent_carries_its_own_input_pipeline(tmp_path):
    """cb_proj and oracle_embed are drawn from the RUN seed, so they have to
    travel with the weights. Regenerating them at load time would only work
    when the seeds happened to match, and would otherwise feed a reloaded
    actor channels it never saw."""
    cfg, path = _tiny_run(tmp_path, "main_frozen", seed=3)
    agent = sac.load_agent(path)

    assert agent.seed == 3
    assert torch.allclose(agent.cb_proj, sac.make_cb_proj(3, n_out=agent.spec.cb))
    assert not torch.allclose(agent.cb_proj, sac.make_cb_proj(0, n_out=agent.spec.cb)), \
        "a different seed must give a different projection, or this test is vacuous"
    assert np.allclose(agent.oracle_embed, sac.make_oracle_embed(3))
    assert agent.cb_proj.shape == (sac.CB_HIDDEN, agent.spec.cb)


def test_run_one_seed_writes_the_agent_beside_the_curve(tmp_path, monkeypatch):
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cfg = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"])
    sac.run_one_seed(cfg, 0, make_figures=False, n_phase_ep=2, phase_reps=1,
                      max_steps=5, buffer_capacity=64, batch_size=8,
                      learning_starts=8, warmup_episodes=2)

    agent_path = os.path.join(sac.run_dir(cfg.name, 0), sac.AGENT_FILE)
    assert os.path.exists(agent_path)
    assert sac.load_agent(agent_path).config.name == "main_frozen"


# ── main_frozen_no_cb: lesioning the readout, not the structure ──────────

def _capture_obs(config, **kw):
    """Every observation and raw transition field a short run stores."""
    added = []
    original_add = ReplayBuffer.add

    def capture(self, obs, action, reward, next_obs, done, **fields):
        added.append({"obs": np.array(obs, copy=True),
                      "next_obs": np.array(next_obs, copy=True),
                      **{k: np.array(v, copy=True) for k, v in fields.items()}})
        return original_add(self, obs, action, reward, next_obs, done, **fields)

    import unittest.mock as mock
    with mock.patch.object(ReplayBuffer, "add", capture):
        sac.run_sac(0, config, n_phase_ep=1, phase_reps=1, max_steps=6,
                     buffer_capacity=64, batch_size=8, learning_starts=10 ** 9,
                     warmup_episodes=0, **kw)
    return added


def test_muting_cb_keeps_the_architecture_identical_to_main_frozen():
    """The control has to differ from main_frozen in the CONTENTS of an input,
    not in the shape of the network -- otherwise a difference in results could
    be capacity rather than information."""
    a = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"])
    b = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb"])
    sa, sb = sac.obs_spec(a), sac.obs_spec(b)
    assert sa == sb, (sa, sb)
    assert sb.cb == sac.CB_PROJ_DIM, "the cb block must still be allocated"

    torch.manual_seed(0); pa = sum(p.numel() for p in Actor(sa).parameters())
    torch.manual_seed(0); pb = sum(p.numel() for p in Actor(sb).parameters())
    assert pa == pb
    assert b.mute_cb_to_mc and not a.mute_cb_to_mc


def test_muted_cb_block_is_zero_in_the_observation():
    config = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb"], arm_seed=0)
    spec = sac.obs_spec(config)
    added = _capture_obs(config)
    assert added

    cb_slice = slice(spec.base + spec.ec, spec.raw_dim)
    ec_slice = slice(spec.base, spec.base + spec.ec)
    for step in added:
        for key in ("obs", "next_obs"):
            assert np.all(step[key][cb_slice] == 0.0), f"{key} cb block not muted"
    # and the lesion is confined: EC's block is still live
    assert any(np.any(step["obs"][ec_slice] != 0.0) for step in added)


def test_muting_cb_does_not_silence_cb_itself():
    """The point of the control: CB keeps predicting, so e_t still feeds EC and
    z is unchanged. If muting had disabled CB, this would be use_cb=False in
    disguise and would confound the readout with the circuit."""
    muted = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb"], arm_seed=0)
    plain = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"], arm_seed=0)

    a = _capture_obs(muted)
    b = _capture_obs(plain)
    assert len(a) == len(b)

    # e_t is CB's prediction error: non-zero after step 0, and IDENTICAL to the
    # un-muted run, since muting touches only what MC reads
    assert any(np.any(s["e_t"] != 0.0) for s in a), "CB stopped producing an error"
    for sa, sb in zip(a, b):
        assert np.allclose(sa["e_t"], sb["e_t"], atol=1e-6)

    # and so is EC's latent, which the observation's ec block carries
    spec = sac.obs_spec(muted)
    ec_slice = slice(spec.base, spec.base + spec.ec)
    for sa, sb in zip(a, b):
        assert np.allclose(sa["obs"][ec_slice], sb["obs"][ec_slice], atol=1e-6)


# ── center_ec: removing z's context-independent component ────────────────

def test_centering_is_representationally_free_but_changes_the_input():
    """W(z - m) + b == Wz + (b - Wm): a first layer can absorb any constant
    offset into its bias, so centering changes no expressible policy. What it
    changes is the INPUT, and therefore the gradient geometry -- which is the
    whole point of the control."""
    torch.manual_seed(0)
    spec = sac.obs_spec(AblationConfig(**SW.CONFIG_BY_NAME["main_frozen"]))
    lin = torch.nn.Linear(spec.ec, 8)
    z = torch.randn(5, spec.ec)
    m = torch.randn(spec.ec)

    shifted = torch.nn.Linear(spec.ec, 8)
    with torch.no_grad():
        shifted.weight.copy_(lin.weight)
        shifted.bias.copy_(lin.bias - lin.weight @ m)
    assert torch.allclose(lin(z - m), shifted(z), atol=1e-5)


def test_ec_channel_subtracts_the_offset_only_when_asked():
    plain = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb"], arm_seed=0)
    centered = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb_centered"],
                              arm_seed=0)
    z = torch.randn(1, sac.LATENT_DIM)
    off = np.arange(sac.LATENT_DIM, dtype=np.float32)

    assert np.allclose(sac.ec_channel(z, plain, off), z.numpy()[0]), \
        "an un-centered config must ignore the offset"
    assert np.allclose(sac.ec_channel(z, centered, off), z.numpy()[0] - off)
    # a missing offset must not silently corrupt the channel
    assert np.allclose(sac.ec_channel(z, centered, None), z.numpy()[0])


def test_calibrated_offset_actually_centers_z():
    """The offset has to be z's mean under the rollout, or centering removes
    the wrong vector and the control tests nothing."""
    import circuit_probe as CP
    cfg = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb_centered"], arm_seed=0)
    off = sac.calibrate_ec_offset(cfg, n_episodes=8, max_steps=20)
    assert off.shape == (sac.LATENT_DIM,)

    data = CP.rollout_z(cfg.arm, cfg.arm_seed, n_episodes=8, max_steps=20)
    z = data["z"][data["t"] > 0]
    assert np.linalg.norm(z.mean(0) - off) < 1e-4
    # and the centered version really is near zero-mean
    assert np.linalg.norm((z - off).mean(0)) < 1e-4 * max(np.linalg.norm(off), 1.0)
    assert np.linalg.norm(off) > 0.5, "a ~zero offset would make the control vacuous"


def test_centering_is_refused_for_an_end_to_end_circuit():
    """A fixed offset is only meaningful for a frozen circuit; an end-to-end
    EC moves under it."""
    cfg = AblationConfig(name="x", use_ec=True, use_cb=True, freeze=False, arm="main",
                          center_ec=True)
    with pytest.raises(ValueError, match="frozen"):
        sac.calibrate_ec_offset(cfg)


def test_centered_run_stores_a_centered_ec_block_and_saves_the_offset(tmp_path):
    """End to end: the observation reaching the buffer is centered, and the
    offset travels with the agent so an analysis reproduces the same inputs."""
    cfg = AblationConfig(**SW.CONFIG_BY_NAME["main_frozen_no_cb_centered"], arm_seed=0)
    spec = sac.obs_spec(cfg)
    path = str(tmp_path / "agent.pt")

    added = []
    original_add = ReplayBuffer.add

    def capture(self, obs, action, reward, next_obs, done, **fields):
        added.append(np.array(obs, copy=True))
        return original_add(self, obs, action, reward, next_obs, done, **fields)

    import unittest.mock as mock
    with mock.patch.object(ReplayBuffer, "add", capture):
        sac.run_sac(0, cfg, n_phase_ep=2, phase_reps=1, max_steps=6,
                     buffer_capacity=64, batch_size=8, learning_starts=10 ** 9,
                     warmup_episodes=0, agent_path=path)

    agent = sac.load_agent(path)
    assert agent.ec_offset is not None and agent.ec_offset.shape == (spec.ec,)

    ec_block = np.stack(added)[:, spec.base:spec.base + spec.ec]
    uncentered = ec_block + agent.ec_offset
    # the stored block is shifted DOWN by the offset relative to raw z
    assert np.linalg.norm(ec_block.mean(0)) < np.linalg.norm(uncentered.mean(0))
    # and the cb block is still muted, since this config mutes it too
    assert np.all(np.stack(added)[:, spec.base + spec.ec:] == 0.0)


def test_update_reports_preclip_gradient_norms():
    """clip_grad_norm_ rescales by the TOTAL norm across all parameters, so a
    single input block with large gradients shrinks the update to every other
    block as well. Whether that is happening is invisible from the losses, so
    the pre-clip norm is recorded -- a value above grad_clip_norm means the
    clip was actually binding on that step."""
    d = 16
    trainer = _make_trainer(d)
    trainer.grad_clip_norm = 1.0
    batch = sac.SACBatch(
        observations=torch.randn(16, d),
        actions=torch.empty(16, 2).uniform_(-1, 1),
        rewards=torch.randn(16, 1),
        next_observations=torch.randn(16, d),
        dones=torch.zeros(16, 1),
    )
    out = trainer.update(batch)
    for k in ("q_gnorm", "actor_gnorm"):
        assert k in out, k
        assert np.isfinite(out[k]) and out[k] >= 0.0, (k, out[k])


def test_curve_records_gradient_norms():
    """The norms have to reach curve.npz, or they cannot be read back from a
    finished run -- which is exactly the gap that made this unanswerable for
    the runs already on disk."""
    import inspect
    src = inspect.getsource(sac)
    assert '"q_gnorm", "actor_gnorm",' in src, "not declared in the curve keys"
    assert 'curve["q_gnorm"].append' in src, "declared but never appended"
    assert 'curve["actor_gnorm"].append' in src


def test_cb_gain_scales_only_the_cb_block():
    """cb_gain exists to separate MAGNITUDE from CONTENT, which
    mute_cb_to_mc confounds. It must touch the cb block and nothing else, and
    it must compose with muting rather than fight it."""
    base = np.arange(6, dtype=np.float32)
    z_ec = np.full(sac.LATENT_DIM, 3.0, dtype=np.float32)
    z_cb = np.full(sac.CB_PROJ_DIM, 2.0, dtype=np.float32)

    plain = sac.augment_obs(base, sac.AblationConfig(
        name="x", use_ec=True, use_cb=True), z_ec=z_ec, z_cb=z_cb)
    scaled = sac.augment_obs(base, sac.AblationConfig(
        name="x", use_ec=True, use_cb=True, cb_gain=0.31), z_ec=z_ec, z_cb=z_cb)

    assert plain.shape == scaled.shape
    n_cb = sac.CB_PROJ_DIM
    assert np.allclose(plain[:-n_cb], scaled[:-n_cb])          # kin + ec intact
    assert np.allclose(scaled[-n_cb:], plain[-n_cb:] * 0.31)   # cb scaled

    muted = sac.augment_obs(base, sac.AblationConfig(
        name="x", use_ec=True, use_cb=True, mute_cb_to_mc=True, cb_gain=0.31),
        z_ec=z_ec, z_cb=z_cb)
    assert np.allclose(muted[-n_cb:], 0.0)


def test_cb_noise_matches_moments_and_destroys_content():
    """The noise control's whole claim is "same scale, no information". Both
    halves have to hold: the substituted block must reproduce CB's per-unit
    mean and std, and it must not depend on the real block it replaced."""
    cfg = sac.AblationConfig(name="x", use_ec=True, use_cb=True, freeze=True,
                             cb_noise=True)
    n = sac.CB_PROJ_DIM
    mean = np.linspace(-4.0, 4.0, n).astype(np.float32)
    std = np.linspace(0.5, 2.5, n).astype(np.float32)
    rng = np.random.default_rng(0)

    draws = np.stack([sac.cb_channel(np.zeros(n, np.float32), cfg,
                                      (mean, std), rng) for _ in range(8000)])
    assert draws.dtype == np.float32
    assert np.abs(draws.mean(0) - mean).max() < 0.15
    assert np.abs(draws.std(0) - std).max() < 0.15

    # INDEPENDENT OF THE REAL BLOCK. Same rng stream, wildly different input:
    # if any of the real signal leaked through, these would differ.
    a = sac.cb_channel(np.zeros(n, np.float32), cfg, (mean, std),
                       np.random.default_rng(7))
    b = sac.cb_channel(np.full(n, 99.0, np.float32), cfg, (mean, std),
                       np.random.default_rng(7))
    assert np.array_equal(a, b)

    # and each call is a fresh draw -- a channel resampled per step, not a
    # fixed random vector the actor could memorize
    rng2 = np.random.default_rng(3)
    assert not np.array_equal(
        sac.cb_channel(np.zeros(n, np.float32), cfg, (mean, std), rng2),
        sac.cb_channel(np.zeros(n, np.float32), cfg, (mean, std), rng2))


def test_cb_noise_is_off_by_default_and_passes_the_real_block_through():
    off = sac.AblationConfig(name="x", use_ec=True, use_cb=True)
    assert off.cb_noise is False
    z = np.arange(5, dtype=np.float32)
    assert sac.cb_channel(z, off, None, np.random.default_rng(0)) is z

    import sweep_ablations as SW
    for name, cfg in SW.CONFIG_BY_NAME.items():
        assert sac.AblationConfig(**cfg).cb_noise == (name == "main_frozen_cb_noise"), name


def test_eval_probe_receives_every_channel_transform():
    """REGRESSION. The center-out probe builds its observations through the
    same augmented_obs_step the rollout does, and retention_table's lag scores
    are read off that probe -- so a transform the config applies during
    training and not during the probe is reported as a policy failure.

    ec_offset was exactly this bug: main_frozen_no_cb_centered trained on
    centered z and was scored on uncentered z. Assert on the call site, since
    the failure mode is a MISSING argument falling back to a default.
    """
    src = inspect.getsource(sac.run_sac)
    call = src[src.index("eval_center_out("):]
    call = call[:call.index(")\n")]
    for kw in ("ec_offset=", "cb_stats=", "cb_proj=", "oracle_embed="):
        assert kw in call, f"eval_center_out call is missing {kw}"

    sig = inspect.signature(sac.eval_center_out).parameters
    for kw in ("ec_offset", "cb_stats", "cb_proj", "oracle_embed"):
        assert kw in sig


def test_cb_gain_defaults_to_one_everywhere_else():
    import sweep_ablations as SW
    for name, cfg in SW.CONFIG_BY_NAME.items():
        expected = 0.31 if name == "main_frozen_cb_scaled" else 1.0
        assert sac.AblationConfig(**cfg).cb_gain == expected, name
