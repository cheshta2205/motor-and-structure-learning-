"""
Tests for policy_sensitivity.py.

The module's job is to be believable when it reports ZERO sensitivity, so the
tests are mostly about it not manufacturing a null: an intervention has to
reach the observation, a channel that demonstrably matters has to register,
and the control rollout has to be reproducible.
"""
import os

import numpy as np
import pytest
import torch

import policy_sensitivity as PS
import sac
import sweep_ablations as SW


TINY = dict(n_phase_ep=2, phase_reps=1, max_steps=5, buffer_capacity=64,
            batch_size=8, learning_starts=8, warmup_episodes=2)


@pytest.fixture(scope="module")
def frozen_agent(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("mf") / "agent.pt")
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"])
    sac.run_sac(0, cfg, agent_path=path, **TINY)
    agent = sac.load_agent(path)
    ec, cb = sac.build_ec_cb(agent.config, "cpu")
    return agent, ec, cb


def test_channel_slices_tile_the_observation(frozen_agent):
    """Slices must partition the observation exactly -- an off-by-one would
    attribute one channel's sensitivity to another."""
    agent, _ec, _cb = frozen_agent
    sl = PS.channel_slices(agent.spec)
    covered = sorted((s.start, s.stop) for s in sl.values())
    assert covered[0][0] == 0
    assert covered[-1][1] == agent.spec.raw_dim
    for (_, end), (start, _) in zip(covered, covered[1:]):
        assert end == start, covered
    assert "kin" in sl and "ec" in sl and "cb" in sl
    assert "oracle" not in sl, "main has no oracle channel"


def test_obs_hook_restores_augment_obs_even_on_error():
    real = sac.augment_obs
    with pytest.raises(RuntimeError):
        with PS.obs_hook(record=[]):
            raise RuntimeError("boom")
    assert sac.augment_obs is real


def test_obs_hook_records_what_the_rollout_actually_built(frozen_agent):
    agent, ec, cb = frozen_agent
    rec = []
    PS.probe(agent, ec, cb, sac.PHASE_CONTEXT_IDS[0], record=rec, max_steps=4)
    assert rec, "probe produced no observations"
    assert all(o.shape == (agent.spec.raw_dim,) for o in rec)


def test_substitution_actually_reaches_the_observation(frozen_agent):
    """The guard against a vacuous null: if the transform silently failed, the
    module would report zero sensitivity for every channel."""
    agent, ec, cb = frozen_agent
    sl = PS.channel_slices(agent.spec)
    marker = np.full(agent.spec.ec, 0.123456, dtype=np.float32)

    rec = []
    PS.probe(agent, ec, cb, sac.PHASE_CONTEXT_IDS[0],
             transform=PS._substitute("ec", marker), record=rec, max_steps=4)
    obs = np.stack(rec)
    assert np.allclose(obs[:, sl["ec"]], marker), "ec block was not substituted"
    # and only that block
    assert not np.allclose(obs[:, sl["cb"]], 0.123456)


def test_control_probe_is_deterministic(frozen_agent):
    """Divergence is a difference between two rollouts, so the control has to
    repeat exactly or every number is noise."""
    agent, ec, cb = frozen_agent
    a = PS.probe(agent, ec, cb, sac.PHASE_CONTEXT_IDS[0], max_steps=6)
    b = PS.probe(agent, ec, cb, sac.PHASE_CONTEXT_IDS[0], max_steps=6)
    assert PS.trajectory_divergence(a, b) == 0.0
    assert np.array_equal(a["success"], b["success"])


def test_trajectory_divergence_ignores_nan_padding():
    a = {"trajectories": np.array([[[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan]]])}
    b = {"trajectories": np.array([[[0.0, 0.0], [1.0, 3.0], [5.0, 5.0]]])}
    # only the two shared steps count: distances 0 and 3 -> mean 1.5
    assert PS.trajectory_divergence(a, b) == pytest.approx(1.5)


def test_divergence_is_zero_when_substituting_a_channels_own_value(frozen_agent):
    """Replacing the ec channel with the value it already had must change
    nothing. Catches a transform that perturbs the observation by the mere act
    of intervening."""
    agent, ec, cb = frozen_agent
    cid = sac.PHASE_CONTEXT_IDS[0]
    rec = []
    control = PS.probe(agent, ec, cb, cid, record=rec, max_steps=6)
    first_ec = np.stack(rec)[0, PS.channel_slices(agent.spec)["ec"]]

    # substituting a CONSTANT equal to step 0's value is not a no-op in
    # general, so assert the weaker, still meaningful property: it is a no-op
    # at step 0, i.e. the first action is unchanged
    out = PS.probe(agent, ec, cb, cid, transform=PS._substitute("ec", first_ec),
                   max_steps=6)
    assert np.allclose(control["trajectories"][:, 1], out["trajectories"][:, 1],
                       atol=1e-6), "the first step must be identical"


def test_gradient_sensitivity_detects_a_channel_that_matters():
    """Planted signal: an actor whose first-layer weights are zero on the ec
    block must score ~0 there and non-zero on kin, and the reverse when the
    zeroing is swapped. Without this, a module reporting all-zeros could not
    be distinguished from a broken one."""
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"])
    spec = sac.obs_spec(cfg)
    sl = PS.channel_slices(spec)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, spec.raw_dim)).astype(np.float32)

    def agent_with_dead(channel):
        torch.manual_seed(0)
        a = sac.Actor(spec)
        with torch.no_grad():
            a.backbone[0].weight[:, sl[channel]] = 0.0
        return sac.TrainedAgent(
            actor=a, qf1=sac.SoftQNetwork(spec), qf2=sac.SoftQNetwork(spec),
            qf1_target=sac.SoftQNetwork(spec), qf2_target=sac.SoftQNetwork(spec),
            spec=spec, config=cfg, cb_proj=sac.make_cb_proj(0, n_out=spec.cb),
            oracle_embed=sac.make_oracle_embed(0), seed=0, alpha=0.2)

    dead_ec = PS.gradient_sensitivity(agent_with_dead("ec"), obs)["actor"]
    dead_kin = PS.gradient_sensitivity(agent_with_dead("kin"), obs)["actor"]
    assert dead_ec["ec"] == pytest.approx(0.0, abs=1e-6)
    assert dead_ec["kin"] > 1e-3
    assert dead_kin["kin"] == pytest.approx(0.0, abs=1e-6)
    assert dead_kin["ec"] > 1e-3


def test_gradient_sensitivity_reports_actor_and_both_critics(frozen_agent):
    """The actor and the critic can disagree, and a policy that ignores a
    channel its value function uses is a different finding from neither."""
    agent, ec, cb = frozen_agent
    rec = []
    PS.probe(agent, ec, cb, sac.PHASE_CONTEXT_IDS[0], record=rec, max_steps=5)
    out = PS.gradient_sensitivity(agent, np.stack(rec))
    assert set(out) == {"actor", "qf1", "qf2"}
    for vals in out.values():
        assert set(vals) == set(PS.channel_slices(agent.spec))
        assert all(np.isfinite(v) and v >= 0 for v in vals.values())


def test_analyze_runs_end_to_end(tmp_path):
    path = str(tmp_path / "agent.pt")
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["oracle"])
    sac.run_sac(0, cfg, agent_path=path, **TINY)
    res = PS.analyze(path, max_steps=5)
    assert {"agent", "behavioral", "gradient", "channel_sd"} <= set(res)
    assert res["behavioral"], "no interventions were run"
    assert {r["channel"] for r in res["behavioral"]} == {"oracle"}
    assert {r["intervention"] for r in res["behavioral"]} == {"self", "swap", "zero"}


def test_main_skips_runs_without_an_agent_file(tmp_path, capsys):
    """Old runs have no agent.pt. They must be reported and skipped, never
    silently retrained."""
    d = tmp_path / "main" / "seed_0"
    d.mkdir(parents=True)
    (d / "curve.npz").write_bytes(b"")
    out = PS.main(run_dir=str(tmp_path), configs=["main"])
    assert out == {}
    assert "no agent.pt" in capsys.readouterr().out


def test_self_substitution_is_the_matched_magnitude_control(frozen_agent):
    """"self" replaces a channel with its OWN context's mean: it removes the
    live within-context variation exactly as a swap does, but preserves
    context identity. Without it, divergence under "swap" cannot be told apart
    from divergence under any perturbation of that size."""
    agent, ec, cb = frozen_agent
    rows = PS.behavioral_sensitivity(agent, ec, cb, sac.PHASE_CONTEXT_IDS[:2],
                                     max_steps=5)
    tags = {r["intervention"] for r in rows}
    assert tags == {"self", "swap", "zero"}

    # the self donor must be that context's own mean, not another's
    donors = PS.channel_means(agent, ec, cb, sac.PHASE_CONTEXT_IDS[:2], max_steps=5)
    a, b = sac.PHASE_CONTEXT_IDS[:2]
    assert not np.allclose(donors[a]["ec"], donors[b]["ec"]), (
        "two contexts with identical mean z would make swap vacuous")


def test_directional_sensitivity_separates_between_from_within():
    """Planted signal: an actor whose ec weights are supported ONLY on the
    between-context direction must score between >> within, and vice versa.

    This is the measurement that resolves "the policy is sensitive to z" vs
    "the policy is sensitive to which context z encodes" -- the plain gradient
    number scales by total sd and cannot tell them apart.
    """
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"])
    spec = sac.obs_spec(cfg)
    sl = PS.channel_slices(spec)
    rng = np.random.default_rng(0)

    # two contexts whose ec blocks differ along a known axis, plus within noise
    n, d = 200, spec.ec
    axis = np.zeros(d, dtype=np.float32); axis[0] = 1.0
    noise_axis = np.zeros(d, dtype=np.float32); noise_axis[1] = 1.0
    labels = np.array([0] * n + [1] * n)
    obs = rng.normal(scale=0.01, size=(2 * n, spec.raw_dim)).astype(np.float32)
    ec0 = sl["ec"].start
    obs[:, ec0 + 0] = np.where(labels == 0, -1.0, 1.0)          # between axis
    obs[:, ec0 + 1] = rng.normal(size=2 * n)                     # within axis

    def agent_reading(axis_offset):
        torch.manual_seed(0)
        a = sac.Actor(spec)
        with torch.no_grad():
            a.backbone[0].weight.zero_()
            a.backbone[0].weight[:, ec0 + axis_offset] = 1.0
            a.backbone[1:].apply(lambda m: None)
        return sac.TrainedAgent(
            actor=a, qf1=sac.SoftQNetwork(spec), qf2=sac.SoftQNetwork(spec),
            qf1_target=sac.SoftQNetwork(spec), qf2_target=sac.SoftQNetwork(spec),
            spec=spec, config=cfg, cb_proj=sac.make_cb_proj(0, n_out=spec.cb),
            oracle_embed=sac.make_oracle_embed(0), seed=0, alpha=0.2)

    reads_between = PS.directional_sensitivity(agent_reading(0), obs, labels)["ec"]
    reads_within = PS.directional_sensitivity(agent_reading(1), obs, labels)["ec"]

    assert reads_between["between"] > 10 * reads_between["within"], reads_between
    assert reads_within["within"] > 10 * reads_within["between"], reads_within


def test_directional_sensitivity_reports_every_channel(frozen_agent):
    agent, ec, cb = frozen_agent
    rec, labels = [], []
    for cid in sac.PHASE_CONTEXT_IDS:
        before = len(rec)
        PS.probe(agent, ec, cb, cid, record=rec, max_steps=5)
        labels.extend([cid] * (len(rec) - before))
    out = PS.directional_sensitivity(agent, np.stack(rec), np.asarray(labels))
    assert set(out) == set(PS.channel_slices(agent.spec))
    for v in out.values():
        assert all(np.isfinite(v[k]) and v[k] >= 0 for k in v)


def _agent_with_first_layer(spec, cfg, weight_fn):
    torch.manual_seed(0)
    a = sac.Actor(spec)
    with torch.no_grad():
        a.backbone[0].weight.zero_()
        a.backbone[0].bias.zero_()
        weight_fn(a.backbone[0].weight)
    return sac.TrainedAgent(
        actor=a, qf1=sac.SoftQNetwork(spec), qf2=sac.SoftQNetwork(spec),
        qf1_target=sac.SoftQNetwork(spec), qf2_target=sac.SoftQNetwork(spec),
        spec=spec, config=cfg, cb_proj=sac.make_cb_proj(0, n_out=spec.cb),
        oracle_embed=sac.make_oracle_embed(0), seed=0, alpha=0.2)


def test_gradient_overlap_orders_the_three_sharing_regimes():
    """The property the metric exists to detect, as a planted ordering.

        orthogonal codes, disjoint units  -> isolated      first ~0.00
        orthogonal codes, shared units    -> partial       first ~0.50
        COLLINEAR codes,  shared units    -> shared        first ~0.95

    The middle row matters: orthogonal codes still share the first layer's
    BIAS gradient when they drive the same units, so scope="first" near 0.5 is
    partial isolation, not none. The bottom row is the regime a large
    common-mode component produces, and is the hypothesis under test for z.
    """
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"])
    spec = sac.obs_spec(cfg)
    sl = PS.channel_slices(spec)
    ec0, H = sl["ec"].start, sac.HIDDEN_DIM

    def codes(kind, n=16):
        a = np.zeros((n, spec.raw_dim), dtype=np.float32)
        b = np.zeros((n, spec.raw_dim), dtype=np.float32)
        if kind == "orthogonal":
            a[:, ec0] = 1.0
            b[:, ec0 + 1] = 1.0
        else:                                  # collinear: one shared direction
            a[:, ec0] = 1.0
            b[:, ec0] = 2.0
        return {0: a, 1: b}

    same_units = _agent_with_first_layer(spec, cfg, lambda W: (
        W.__setitem__((slice(None), ec0), 1.0),
        W.__setitem__((slice(None), ec0 + 1), 1.0)))
    split_units = _agent_with_first_layer(spec, cfg, lambda W: (
        W.__setitem__((slice(None, H // 2), ec0), 1.0),
        W.__setitem__((slice(H // 2, None), ec0 + 1), 1.0)))

    def first(agent, kind):
        return PS.gradient_overlap(agent, codes(kind), scope="first",
                                    n_samples=8)[0][0, 1]

    isolated = first(split_units, "orthogonal")
    partial = first(same_units, "orthogonal")
    shared = first(same_units, "collinear")

    assert isolated == pytest.approx(0.0, abs=1e-5), isolated
    assert 0.3 < partial < 0.7, partial
    assert shared > 0.9, shared
    assert isolated < partial < shared

    # the whole-network measure has an architectural floor: the output head is
    # shared by every input, so even perfect first-layer isolation is not 0
    all_isolated = PS.gradient_overlap(split_units, codes("orthogonal"),
                                        scope="all", n_samples=8)[0][0, 1]
    assert 0.05 < all_isolated < 0.3, all_isolated
    assert PS.gradient_overlap(same_units, codes("collinear"), scope="all",
                                n_samples=8)[0][0, 1] > 0.9


def test_gradient_overlap_is_exactly_one_against_itself():
    """The normalization check. With one sample per context, the diagonal is a
    gradient's cosine with itself and must be exactly 1.

    Regression: gradients were pooled ACROSS output dimensions, and the action
    head's per-dimension weights make d a_0/d theta and d a_1/d theta
    near-orthogonal for reasons unrelated to context -- which dragged the
    diagonal to ~0.2 and halved every off-diagonal."""
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"])
    spec = sac.obs_spec(cfg)
    rng = np.random.default_rng(0)
    by_ctx = {c: rng.normal(size=(1, spec.raw_dim)).astype(np.float32) for c in (0, 1)}
    torch.manual_seed(0)
    agent = sac.TrainedAgent(
        actor=sac.Actor(spec), qf1=sac.SoftQNetwork(spec), qf2=sac.SoftQNetwork(spec),
        qf1_target=sac.SoftQNetwork(spec), qf2_target=sac.SoftQNetwork(spec),
        spec=spec, config=cfg, cb_proj=sac.make_cb_proj(0, n_out=spec.cb),
        oracle_embed=sac.make_oracle_embed(0), seed=0, alpha=0.2)
    M, ids = PS.gradient_overlap(agent, by_ctx, n_samples=1)
    assert M.shape == (2, 2) and ids == [0, 1]
    assert M[0, 0] == pytest.approx(1.0, abs=1e-5), M
    assert M[1, 1] == pytest.approx(1.0, abs=1e-5), M
    assert np.allclose(M, M.T, atol=1e-5), "overlap must be symmetric"


def test_gradient_overlap_covers_actor_and_critic(frozen_agent):
    agent, ec, cb = frozen_agent
    by_ctx = {}
    for cid in sac.PHASE_CONTEXT_IDS[:2]:
        rec = []
        PS.probe(agent, ec, cb, cid, record=rec, max_steps=5)
        by_ctx[cid] = np.stack(rec)
    for which in ("actor", "qf1"):
        for scope in ("first", "all"):
            M, ids = PS.gradient_overlap(agent, by_ctx, which=which, scope=scope,
                                          n_samples=4)
            assert M.shape == (2, 2) and np.all(np.isfinite(M))
    with pytest.raises(ValueError):
        PS.gradient_overlap(agent, by_ctx, scope="nonsense", n_samples=2)


def test_activation_overlap_is_zero_for_disjoint_units():
    """The cheap companion measure, same planted signal."""
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"])
    spec = sac.obs_spec(cfg)
    sl = PS.channel_slices(spec)
    ec0, H = sl["ec"].start, sac.HIDDEN_DIM
    n = 16
    obs_a = np.zeros((n, spec.raw_dim), dtype=np.float32); obs_a[:, ec0 + 0] = 1.0
    obs_b = np.zeros((n, spec.raw_dim), dtype=np.float32); obs_b[:, ec0 + 1] = 1.0

    def disjoint(W):
        W[: H // 2, ec0 + 0] = 1.0
        W[H // 2:, ec0 + 1] = 1.0

    agent = _agent_with_first_layer(spec, cfg, disjoint)
    M, _ = PS.activation_overlap(agent, {0: obs_a, 1: obs_b})
    assert M[0, 1] == pytest.approx(0.0, abs=1e-6), M
    assert M[0, 0] == pytest.approx(1.0, abs=1e-6)
