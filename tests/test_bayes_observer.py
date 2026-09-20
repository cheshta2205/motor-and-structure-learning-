"""The inferred oracle: an exact posterior over the phase contexts.

What has to hold for it to be the COIN baseline it claims to be:

  the force inversion is EXACT on a clean step, so the true context wins
  with command noise it still converges, for every context, in a few steps
  steps that carry no evidence (wall, speed cap, null action) leave the belief
    alone instead of corrupting it
  the sticky transition lets it notice a context switch it was not told about
  its code is the one-hot oracle's code at certainty and the mean of the codes
    at ignorance -- same channel, same geometry
  the preset differs from `oracle` in exactly that one field
"""
from dataclasses import asdict

import numpy as np
import pytest

import sac
import sweep_ablations as SW
import task as T
from sac import BayesObserver

IDS = list(sac.PHASE_CONTEXT_IDS)
K = len(IDS)


@pytest.fixture(scope="module")
def embed():
    return sac.make_oracle_embed(0)


def _step(cid, pos, vel, action, rng):
    """One plant step under context `cid`, returning what the observer sees."""
    pos_new, vel_new, _F, wall = T.env_step(pos, vel, action, cid, rng=rng,
                                            return_impulse=True)
    return pos_new, vel_new, wall


def _run(obs, cid, n_steps, rng, action_rng, start_vel=None):
    pos = np.zeros(2, dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32) if start_vel is None else start_vel
    for _ in range(n_steps):
        a = action_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        pos_new, vel_new, wall = _step(cid, pos, vel, a, rng)
        obs.update(a, vel, vel_new, wall)
        pos, vel = pos_new, vel_new
    return obs


def test_starts_uniform_and_normalised(embed):
    obs = BayesObserver(embed)
    assert obs.K == K
    assert np.allclose(obs.posterior, 1.0 / K)
    assert obs.posterior.sum() == pytest.approx(1.0)


def test_one_clean_step_from_rest_cannot_separate_the_curls(embed):
    """Curl acts on velocity, which is zero at the first step, so the two curl
    contexts in the set produce IDENTICAL first steps and the observer must
    leave its mass split between them -- while a rotation, whose evidence is
    in the action itself, is identified at once."""
    curls = [k for k, cid in enumerate(IDS) if T.FAMILY[cid] == 1]
    rots = [k for k, cid in enumerate(IDS) if T.FAMILY[cid] == 0]
    assert len(curls) == 2 and len(rots) == 2
    a = np.array([0.6, -0.3], dtype=np.float32)
    # from rest in a curl context: the two curls tie, everything else is out
    obs = BayesObserver(embed)
    vel = np.zeros(2, dtype=np.float32)
    _, vel_new, wall = _step(IDS[curls[0]], np.zeros(2, dtype=np.float32), vel, a, rng=None)
    obs.update(a, vel, vel_new, wall)
    p = obs.posterior
    assert p[curls[0]] == pytest.approx(p[curls[1]], rel=1e-6)
    assert set(np.argsort(-p)[:2]) == set(curls), p        # the tie is at the top
    assert p[curls].sum() > 0.8, p                          # a mild rotation keeps a little mass
    # from rest in a rotation context: identified in one step
    obs = BayesObserver(embed)
    _, vel_new, wall = _step(IDS[rots[0]], np.zeros(2, dtype=np.float32), vel, a, rng=None)
    obs.update(a, vel, vel_new, wall)
    # a drift whose increment happens to point the way this rotation turns
    # the force keeps some mass after ONE step; the rotation still wins clearly
    assert obs.posterior.argmax() == rots[0] and obs.posterior[rots[0]] > 0.7


def test_a_few_clean_steps_identify_every_context(embed):
    """Once the velocity has built, the curls separate too: y == A_c a + B_c v
    + w_c MASS/DT exactly, so a handful of noiseless steps in varied
    directions gives near-certainty for every context in the set."""
    acts = [np.array(x, dtype=np.float32) for x in
            ((0.6, -0.3), (-0.2, 0.7), (0.5, 0.5), (-0.6, -0.1), (0.1, -0.7), (0.7, 0.2))]
    for k, cid in enumerate(IDS):
        obs = BayesObserver(embed)
        pos = np.zeros(2, dtype=np.float32); vel = np.zeros(2, dtype=np.float32)
        for a in acts:
            pos, vel_new, wall = _step(cid, pos, vel, a, rng=None)
            obs.update(a, vel, vel_new, wall)
            vel = vel_new
        assert obs.posterior.argmax() == k, (T.C_LABEL[cid], obs.posterior)
        assert obs.posterior[k] > 0.9, obs.posterior


@pytest.mark.parametrize("k", range(K))
def test_converges_under_command_noise(embed, k):
    rng = np.random.default_rng(10 + k)
    obs = _run(BayesObserver(embed), IDS[k], n_steps=8, rng=rng,
               action_rng=np.random.default_rng(100 + k))
    assert obs.posterior[k] > 0.95, obs.posterior


def test_wall_and_speed_cap_and_null_action_carry_no_evidence(embed):
    obs = BayesObserver(embed)
    before = obs.posterior.copy()
    a = np.array([0.5, 0.5], dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    # a wall contact
    obs.update(a, vel, np.array([0.02, 0.0], dtype=np.float32), wall_impulse=1.0)
    assert np.allclose(obs.posterior, before)
    # at the speed cap the plant rescaled the velocity: not the force law
    capped = np.array([T.V_MAX, 0.0], dtype=np.float32)
    obs.update(a, vel, capped, wall_impulse=0.0)
    assert np.allclose(obs.posterior, before)
    # a null action has zero noise scale and no direction to read
    obs.update(np.zeros(2, dtype=np.float32), vel, vel, 0.0)
    assert np.allclose(obs.posterior, before)


def test_notices_a_switch_it_was_not_told_about(embed):
    """20 steps in context 0, then context 1: the hazard must let the belief
    move within a few steps -- 3.7 nats/step at 45 degrees against the 6.9
    nats the 1e-3 hazard demands."""
    rng = np.random.default_rng(3)
    obs = _run(BayesObserver(embed), IDS[0], 20, rng, np.random.default_rng(30))
    assert obs.posterior[0] > 0.99
    obs = _run(obs, IDS[1], 6, rng, np.random.default_rng(31))
    assert obs.posterior.argmax() == 1
    assert obs.posterior[1] > 0.9, obs.posterior


def test_zero_hazard_can_never_switch(embed):
    """The control for the test above: without the transition, a certain
    belief is frozen -- so it is the hazard doing the work, not the evidence
    alone overpowering a numerically finite prior."""
    rng = np.random.default_rng(4)
    obs = BayesObserver(embed, hazard=0.0)
    obs = _run(obs, IDS[0], 40, rng, np.random.default_rng(40))
    p0 = obs.posterior[0]
    obs = _run(obs, IDS[1], 6, rng, np.random.default_rng(41))
    # posterior on 0 was driven to (numerically) 1; with no hazard the other
    # contexts have no mass to grow from
    assert obs.posterior[0] == pytest.approx(p0, abs=1e-6) or obs.posterior[0] > 0.99


def test_code_is_the_one_hot_code_at_certainty_and_the_mean_at_ignorance(embed):
    obs = BayesObserver(embed)
    assert np.allclose(obs.code(), embed.mean(axis=0), atol=1e-6)
    obs.log_p = np.log(np.eye(K)[2] + 1e-300)
    assert np.allclose(obs.code(), sac.oracle_code(2, embed), atol=1e-6)
    assert obs.code().dtype == np.float32 and obs.code().shape == (embed.shape[1],)


def test_null_field_hypothesis_is_off_by_default_and_appended_last(embed):
    obs = BayesObserver(embed)
    assert obs.K == K and None not in obs.ids and obs.embed.shape[0] == K
    nul = BayesObserver(embed, null_context=True)
    assert nul.K == K + 1 and nul.ids[:K] == IDS and nul.ids[-1] is None
    assert np.allclose(nul.embed[:K], embed) and not nul.embed[-1].any()
    # a null-certain code is the given oracle's code outside a context: zeros
    nul.log_p = np.log(np.eye(K + 1)[K] + 1e-300)
    assert np.allclose(nul.code(), sac.oracle_code(None, embed))
    # the phase contexts keep their codes and indices
    nul.log_p = np.log(np.eye(K + 1)[2] + 1e-300)
    assert np.allclose(nul.code(), sac.oracle_code(2, embed), atol=1e-6)


def test_null_field_is_identified_and_leaves_the_rotations_alone(embed):
    """Under the null field the null hypothesis wins outright -- +-30 no longer
    split the belief 50/50 -- and under a rotation the rotation still wins."""
    rng, arng = np.random.default_rng(7), np.random.default_rng(70)
    nul = _run(BayesObserver(embed, null_context=True), None, n_steps=8, rng=rng, action_rng=arng)
    assert nul.posterior[-1] > 0.95, nul.posterior
    for k in range(K):
        nul = _run(BayesObserver(embed, null_context=True), IDS[k], n_steps=8,
                   rng=np.random.default_rng(10 + k), action_rng=np.random.default_rng(100 + k))
        assert nul.posterior.argmax() == k and nul.posterior[k] > 0.95, nul.posterior
    # without the hypothesis, on the paradigms' rotation set, the same
    # null-field evidence lands on +-30 -- and on WHICHEVER of the two the
    # command noise favours (their likelihoods differ by a zero-mean noise
    # term of sd 2 sin15 / CMD_NOISE nats per step, and the sticky prior then
    # holds it), so P0 reaches were tagged P+ or P- at random. The failure the
    # hypothesis exists to remove; on the mixed set a curl absorbs it instead.
    rot_ids = [c for c in range(len(T.CONTEXTS)) if T.FAMILY[c] == 0]
    rot_embed = sac.make_oracle_embed(0, n_contexts=len(rot_ids))
    near = [rot_ids.index(c) for c in rot_ids if abs(T.C_PARAM[c]) == min(abs(T.C_PARAM[c]) for c in rot_ids)]
    assert len(near) == 2
    winners = set()
    for s in range(6):
        old = _run(BayesObserver(rot_embed, context_ids=rot_ids), None, n_steps=8,
                   rng=np.random.default_rng(7 + s), action_rng=np.random.default_rng(70 + s))
        assert old.posterior[near].sum() > 0.95, old.posterior
        winners.add(int(old.posterior.argmax()))
    assert winners == set(near), winners
    fixed = _run(BayesObserver(rot_embed, context_ids=rot_ids, null_context=True), None, n_steps=8,
                 rng=np.random.default_rng(7), action_rng=np.random.default_rng(70))
    assert fixed.posterior[-1] > 0.95, fixed.posterior


def test_observer_null_resolves_from_the_schedule_unless_set():
    coin = dict(SW.CONFIG_BY_NAME["coin"])
    assert sac.observer_has_null(sac.AblationConfig(**coin)) is False           # cycle: unchanged
    sr = sac.AblationConfig(**sac.apply_cli_variants(coin, phase_set="rotations",
                                                     schedule="spontaneous_recovery"))
    assert sac.observer_has_null(sr) is True
    # savings-washout runs on a trained baseline: no null blocks, no hypothesis
    savw = sac.AblationConfig(**sac.apply_cli_variants(coin, phase_set="rotations",
                                                       schedule="savings_washout"))
    assert sac.observer_has_null(savw) is False
    assert sac.observer_has_null(sac.AblationConfig(**{**coin, "observer_null_context": True})) is True
    assert sac.observer_has_null(sac.AblationConfig(**{**asdict(sr), "observer_null_context": False})) is False
    assert sac.make_observer(sr, sac.make_oracle_embed(0)).null_context is True
    assert sac.make_observer(sac.AblationConfig(**coin), sac.make_oracle_embed(0)).null_context is False
    with pytest.raises(ValueError, match="null"):
        sac.CBObserver(sac.make_oracle_embed(0), None, [], [], null_context=True)
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_reset_returns_to_uniform(embed):
    obs = _run(BayesObserver(embed), IDS[3], 10, np.random.default_rng(5),
               np.random.default_rng(50))
    obs.reset()
    assert np.allclose(obs.posterior, 1.0 / K)


def test_make_bayes_observer_only_for_the_inferred_oracle(embed):
    given = sac.AblationConfig(name="oracle", use_oracle=True)
    inferred = sac.AblationConfig(name="coin", use_oracle=True, oracle_source="bayes")
    blind = sac.AblationConfig(name="baseline")
    assert sac.make_bayes_observer(given, embed) is None
    assert sac.make_bayes_observer(blind, embed) is None
    assert isinstance(sac.make_bayes_observer(inferred, embed), BayesObserver)


def test_coin_preset_differs_from_oracle_in_exactly_the_source():
    a = asdict(sac.AblationConfig(**SW.CONFIG_BY_NAME["oracle"]))
    b = asdict(sac.AblationConfig(**SW.CONFIG_BY_NAME["coin"]))
    assert {k for k in a if a[k] != b[k]} == {"name", "oracle_source"}
    assert b["oracle_source"] == "bayes"
    assert "coin_clearbuf" in SW.CONFIG_BY_NAME


def test_coin_runs_end_to_end_on_cut_down_constants(tmp_path, monkeypatch):
    """The observer is threaded through the run loop AND the probe; both paths
    must execute, in both versions."""
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figures_sac"))
    for name in ("coin", "coin_clearbuf"):
        config = sac.AblationConfig(**SW.CONFIG_BY_NAME[name])
        sac.main(config, seeds=1, save_agg=True, workers=1,
                 n_phase_ep=3, phase_reps=1, max_steps=5,
                 buffer_capacity=32, batch_size=4, learning_starts=0,
                 warmup_episodes=2)
        assert (tmp_path / "sac_runs" / name / "aggregated_results.csv").exists()
        # the cycle schedule's observer has no null hypothesis: the column is nan
        import pandas as pd
        df = pd.read_csv(tmp_path / "sac_runs" / name / "aggregated_results.csv")
        assert df.belief_base.isna().all()
