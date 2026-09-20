"""The matched-SNR oracle: a perfect context label degraded to the circuit's SNR.

The point of the config is that ONE property moves -- between-context variance
as a fraction of the total -- so most of these tests are about what must NOT
move with it: the code's geometry, the channel's magnitude, the observation
width, and the reproducibility of every finished oracle run.

A separate file rather than an addition to tests/test_sac.py, which does not
collect (it still parametrizes over model.INPUT_VARIANTS, removed in the
factorial rewrite).
"""
import numpy as np
import pytest

import sac
import sweep_ablations as SW
from sac import AblationConfig, OracleNoise, make_oracle_embed

# The matched-SNR ladder rungs, INLINE. They were sweep_ablations presets until
# 2026-09-16, when the ladder was closed and the preset table cut to the core
# set; the mechanism they exercise (AblationConfig.oracle_snr / OracleNoise)
# is still live, so the tests keep the exact rung definitions here.
SNR_CONFIGS = {
    "oracle_snr_fitted": dict(name="oracle_snr_fitted", use_oracle=True,
                              oracle_kernel="z", arm="main",
                              oracle_snr=0.61, oracle_noise_rho=0.60),
    "oracle_snr_mid": dict(name="oracle_snr_mid", use_oracle=True,
                           oracle_kernel="z", arm="main",
                           oracle_snr=0.40, oracle_noise_rho=0.72),
    "oracle_snr_deployed": dict(name="oracle_snr_deployed", use_oracle=True,
                                oracle_kernel="z", arm="main",
                                oracle_snr=0.19, oracle_noise_rho=0.84),
}


def _cfg(name):
    """An AblationConfig for a ladder rung defined above or a real preset."""
    src = SNR_CONFIGS.get(name) or SW.CONFIG_BY_NAME[name]
    return AblationConfig(**src)


@pytest.fixture(scope="module")
def embed():
    return make_oracle_embed(0, kernel="identity")


def _channel_samples(embed, snr, rho, n_per_ctx=8000, seed=1):
    """Stack the channel over every context, advancing once per step."""
    noise = OracleNoise(embed, snr, rho, rng=np.random.default_rng(seed))
    noise.reset()
    X, y = [], []
    for c in range(embed.shape[0]):
        code = sac.oracle_code(c, embed)
        for _ in range(n_per_ctx):
            X.append(noise.apply(code))
            y.append(c)
            noise.advance()
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def _snr(X, y):
    gm = X.mean(0)
    tot = ((X - gm) ** 2).sum(1).mean()
    means = np.stack([X[y == c].mean(0) for c in np.unique(y)])
    w = np.array([(y == c).mean() for c in np.unique(y)])
    return float((w[:, None] * (means - gm) ** 2).sum() / tot)


# ── the identity path must stay bit-identical ────────────────────────────

def test_snr_one_is_a_literal_no_op(embed):
    """Every finished oracle run in sac_runs/ was trained on the noiseless
    code. If snr=1.0 were merely CLOSE to a no-op, those runs would stop being
    reproducible and the sweep would silently compare against a moved anchor."""
    noise = OracleNoise(embed, 1.0, 0.0)
    code = sac.oracle_code(2, embed)
    assert noise.apply(code) is code
    assert not noise.active
    assert noise.scale == 0.0 and noise.divisor == 1.0


def test_default_config_has_no_noise():
    # oracle_geom_centered was removed from the preset table on 2026-09-16
    for name in ("oracle", "oracle_geom"):
        cfg = _cfg(name)
        assert cfg.oracle_snr == 1.0, name
        assert cfg.oracle_noise_rho == 0.0, name


def test_non_oracle_configs_build_no_noise():
    for name in ("baseline", "main", "end_to_end"):
        cfg = _cfg(name)
        assert sac.make_oracle_noise(cfg, make_oracle_embed(0)) is None, name


# ── the manipulation itself ──────────────────────────────────────────────

@pytest.mark.parametrize("target", [0.61, 0.40, 0.19, 0.05])
def test_realised_snr_matches_the_target(embed, target):
    X, y = _channel_samples(embed, target, rho=0.6)
    assert _snr(X, y) == pytest.approx(target, abs=0.02)


def test_magnitude_is_held_fixed_across_the_sweep(embed):
    """Without the renormaliser the block would get sqrt(1/snr) louder as SNR
    fell, so the sweep would confound SNR with the actor's input drive -- the
    exact confound cb_gain exists to separate for the CB block."""
    norms = []
    for snr in (1.0, 0.61, 0.40, 0.19, 0.05):
        X, _ = _channel_samples(embed, snr, rho=0.6, n_per_ctx=3000)
        norms.append(np.sqrt((X ** 2).sum(1).mean()))
    assert np.allclose(norms, norms[0], atol=0.02), norms


@pytest.mark.parametrize("rho", [0.0, 0.6, 0.84])
def test_temporal_correlation_is_realised(embed, rho):
    noise = OracleNoise(embed, 0.19, rho, rng=np.random.default_rng(3))
    noise.reset()
    seq = []
    for _ in range(20000):
        seq.append(noise.current.copy())
        noise.advance()
    seq = np.asarray(seq)
    got = np.corrcoef(seq[:-1].ravel(), seq[1:].ravel())[0, 1]
    assert got == pytest.approx(rho, abs=0.03)


def test_noise_lives_in_the_codes_own_subspace(embed):
    """Noise orthogonal to the centroid differences is removable by a linear
    projection, so it would lower the reported SNR without lowering what MC
    can actually read -- the sweep would then understate the circuit's
    handicap rather than match it."""
    noise = OracleNoise(embed, 0.19, 0.0, rng=np.random.default_rng(4))
    noise.reset()
    basis = sac.oracle_code_subspace(embed)
    assert basis.shape[0] == embed.shape[0] - 1      # K contexts -> K-1 dims
    for _ in range(50):
        v = noise.current
        recon = (v @ basis.T) @ basis
        assert np.allclose(v, recon, atol=1e-5)
        noise.advance()


def test_the_warmup_code_stays_exactly_zero(embed):
    """The null-field block has no context to encode and passes a zero code.
    Noising it would invent a context for the one condition defined not to
    have one -- and that block is the naive floor every memory trace is
    measured against."""
    noise = OracleNoise(embed, 0.19, 0.6, rng=np.random.default_rng(5))
    noise.reset()
    zero = np.zeros(embed.shape[1], dtype=np.float32)
    for _ in range(20):
        assert not np.any(noise.apply(zero))
        noise.advance()


def test_the_code_itself_is_untouched(embed):
    """The exact invariant: the channel is (code + zero-mean noise) / const.

    Subtract the noise actually drawn and the code must come back exactly, so
    the SNR leg cannot perturb the code table the geometry leg (oracle_kernel)
    is about. Asserted per draw rather than on a sample mean, which only
    recovers this to its own standard error -- see the statistical version
    below for why that distinction matters here.
    """
    noise = OracleNoise(embed, 0.19, 0.6, rng=np.random.default_rng(7))
    noise.reset()
    for c in range(embed.shape[0]):
        code = sac.oracle_code(c, embed)
        for _ in range(20):
            got = noise.apply(code) - noise.current / noise.divisor
            assert np.allclose(got, code / noise.divisor, atol=1e-6)
            noise.advance()


def test_geometry_is_preserved_in_the_realised_samples(embed):
    """The centroid cosines a run actually realises, not just in expectation.

    TOLERANCE IS SET BY THE CENTROID'S OWN SAMPLING ERROR, which is not small:
    the noise sits entirely in the discriminative subspace, which is exactly
    where cosine differences live, so a mean over n draws misses the true
    centroid by ~sqrt(P/n_eff) in the directions being measured. At snr=0.19
    that is ~3% of the code's length even at n=20000, i.e. cosines good to
    ~0.04. rho=0 here for the same reason -- correlated draws cut n_eff by
    (1-rho)/(1+rho) and would quadruple the error at rho=0.6.
    """
    X, y = _channel_samples(embed, 0.19, rho=0.0, n_per_ctx=20000, seed=8)
    cents = np.stack([X[y == c].mean(0) for c in np.unique(y)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    ref = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    assert np.allclose(cents @ cents.T, ref @ ref.T, atol=0.08)


def test_rejects_an_out_of_range_snr(embed):
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            sac.oracle_noise_scale(embed, bad)


# ── wiring ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SNR_CONFIGS)
def test_observation_width_is_unchanged(name):
    """Same width as the noiseless oracle, so the actor is architecturally
    identical and only the CONTENTS of the block differ."""
    assert sac.obs_dim(_cfg(name)) == \
        sac.obs_dim(AblationConfig(**SW.CONFIG_BY_NAME["oracle_geom"]))


@pytest.mark.parametrize("name", SNR_CONFIGS)
def test_the_ladder_keeps_the_circuit_geometry(name):
    cfg = _cfg(name)
    assert cfg.oracle_kernel == "z", "sweeping SNR on an identity code would move two things"
    assert 0.0 < cfg.oracle_snr < 1.0


def test_stored_transitions_are_consecutive_samples_not_independent_draws():
    """obs and next_obs must be timesteps t and t+1 of ONE AR(1) process.

    Drawing next_obs independently (which is right for cb_channel, whose
    premise is that the block carries nothing across steps) would destroy the
    correlation this config exists to impose: the critic would learn that the
    channel resets every step, and rho would do nothing however it was set.
    """
    embed = make_oracle_embed(0)
    noise = OracleNoise(embed, 0.19, 0.95, rng=np.random.default_rng(6))
    noise.reset()
    code = sac.oracle_code(1, embed)
    pairs = []
    for _ in range(4000):
        obs = noise.apply(code)
        noise.advance()                      # once per env step, as run_sac does
        nxt = noise.apply(code)
        pairs.append((obs, nxt))
        # the NEXT iteration must read the same sample back
        assert np.allclose(noise.apply(code), nxt)
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    r = np.corrcoef((a - a.mean(0)).ravel(), (b - b.mean(0)).ravel())[0, 1]
    assert r > 0.85, f"consecutive samples should stay correlated at rho=0.95, got {r:.3f}"


@pytest.mark.parametrize("name", SNR_CONFIGS)
def test_runs_end_to_end_and_is_reproducible(tmp_path, name):
    cfg = _cfg(name)
    # the "z" kernel measures the circuit's centroid geometry, so the rung
    # needs the circuit's checkpoint; before it is trained the run cannot start
    if cfg.oracle_kernel != "identity" and sac.M.load(cfg.arm, cfg.arm_seed, sac.circuit_of(cfg)) is None:
        pytest.skip(f"{name}: no {sac.circuit_of(cfg)} checkpoint for arm {cfg.arm!r} yet")
    kw = dict(n_phase_ep=4, phase_reps=1, max_steps=10, warmup_episodes=3,
              learning_starts=6, batch_size=6, buffer_capacity=300)
    c1, p1 = sac.run_sac(0, cfg, **kw)
    c2, p2 = sac.run_sac(0, cfg, **kw)
    assert np.allclose(c1["return"], c2["return"])
    assert np.allclose(np.concatenate([q["path_ratio"] for q in p1]),
                       np.concatenate([q["path_ratio"] for q in p2]),
                       equal_nan=True)


def test_the_probe_does_not_advance_the_training_noise_stream():
    """eval_center_out is side-effect free by contract; it runs mid-training,
    so an object shared with the rollout would make the learning curve depend
    on how often the probe happened to fire."""
    embed = make_oracle_embed(0)
    cfg = _cfg("oracle_snr_deployed")
    run_noise = sac.make_oracle_noise(cfg, embed, seed=0)
    run_noise.reset()
    before = run_noise.u.copy()

    trainer = _tiny_trainer(cfg)
    sac.eval_center_out(trainer, None, None, cfg, sac.PHASE_CONTEXT_IDS[0], 0,
                        max_steps=6, oracle_vec=sac.oracle_code(0, embed),
                        oracle_embed=embed, device="cpu")
    assert np.array_equal(run_noise.u, before)


def _tiny_trainer(cfg):
    import torch
    spec = sac.obs_spec(cfg)
    actor = sac.Actor(spec)
    qf1, qf2 = sac.SoftQNetwork(spec), sac.SoftQNetwork(spec)
    return sac.SACTrainer(
        actor, qf1, qf2, sac.SoftQNetwork(spec), sac.SoftQNetwork(spec),
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters())),
        device="cpu")
