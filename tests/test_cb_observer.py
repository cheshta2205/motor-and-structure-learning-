"""The inferred oracle scored through the LEARNED forward model.

`coin` inverts the true plant; `coin_cb` scores each context hypothesis with
the frozen CB conditioned on that context's z-centroid, using CB's own
calibrated residual std as the likelihood. What has to hold:

  the calibration is well-formed: K centroids, positive per-dimension std
  on the supervised transitions the circuit was fitted on, the posterior
    concentrates on the true context -- CB is a good enough model for that
  steps without evidence leave the belief alone, as for the plant observer
  the probe copy starts uniform but shares the expensive static parts
  the preset differs from `coin` in exactly the evidence source

Needs the default (gp) main seed-0 checkpoint; skipped otherwise.
"""
from dataclasses import asdict

import numpy as np
import pytest
import torch

import model as M
import sac
import sweep_ablations as SW
from data import generate_sequences


def _have_checkpoint():
    try:
        return M.load("main", 0) is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_checkpoint(),
                                reason="needs the main seed 0 checkpoint")

IDS = list(sac.PHASE_CONTEXT_IDS)
K = len(IDS)
CFG = sac.AblationConfig(**SW.CONFIG_BY_NAME["coin_cb"])


@pytest.fixture(scope="module")
def parts():
    return sac.cb_observer_parts(CFG, device="cpu")


@pytest.fixture(scope="module")
def embed():
    return sac.make_oracle_embed(0)


def test_calibration_is_well_formed(parts):
    cb, cen, resid_std = parts
    assert cen.shape == (K, M.LATENT_DIM)
    assert resid_std.shape == (sac.SDIM,)
    assert np.all(resid_std > 0)
    # centroids differ across contexts -- otherwise every hypothesis predicts
    # the same thing and the observer could never move
    assert np.linalg.norm(cen - cen.mean(0), axis=1).min() > 1e-3


def _supervised_walk(obs, cid, n_steps, seed):
    """Drive the observer along a supervised sequence of ONE context, feeding
    it the same (s, a, s') triples CB was fitted on."""
    # generate_sequences needs >= 2 contexts to draw from; stay=1.0 stops the
    # switching, so each sequence sits in one context -- pick one in `cid`
    trans, labels, _ = generate_sequences(16, seq_len=n_steps, seed=seed, stay=1.0,
                                          contexts=list(IDS))
    b = int(np.flatnonzero(labels[:, 0, 0].numpy() == cid)[0])
    assert bool((labels[b, :, 0] == cid).all())
    o, a, t = M.build(trans)
    o, a, t = o[b].numpy(), a[b].numpy(), t[b].numpy()
    raw = trans[b].numpy()
    for i in range(n_steps):
        vel_prev, vel_new = raw[i, 2:4], raw[i, 8:10]
        obs.update(a[i], vel_prev, vel_new, 0.0, s_prev=o[i], s_next=t[i])
    return obs


@pytest.mark.parametrize("k", range(K))
def test_posterior_concentrates_on_the_true_context(parts, embed, k):
    cb, cen, resid_std = parts
    obs = sac.CBObserver(embed, cb, cen, resid_std)
    obs = _supervised_walk(obs, IDS[k], n_steps=30, seed=7 + k)
    assert obs.posterior.argmax() == k, obs.posterior
    assert obs.posterior[k] > 0.8, obs.posterior


def test_no_evidence_steps_leave_the_belief_alone(parts, embed):
    cb, cen, resid_std = parts
    obs = sac.CBObserver(embed, cb, cen, resid_std)
    before = obs.posterior.copy()
    a = np.array([0.5, 0.2], dtype=np.float32)
    s = np.zeros(sac.SDIM, dtype=np.float32)
    obs.update(a, s[2:], s[2:], wall_impulse=1.0, s_prev=s, s_next=s)     # wall
    obs.update(a, s[2:], s[2:], 0.0)                                       # no states
    assert np.allclose(obs.posterior, before)


def test_fresh_copy_is_uniform_and_shares_static_parts(parts, embed):
    cb, cen, resid_std = parts
    obs = _supervised_walk(sac.CBObserver(embed, cb, cen, resid_std), IDS[0], 10, 3)
    f = obs.fresh()
    assert np.allclose(f.posterior, 1.0 / K)
    assert f.cb is obs.cb and f.centroids is obs.centroids
    assert not np.allclose(obs.posterior, 1.0 / K)        # the original untouched


def test_make_observer_dispatch(embed):
    assert sac.make_observer(sac.AblationConfig(name="o", use_oracle=True), embed) is None
    assert isinstance(sac.make_observer(
        sac.AblationConfig(name="c", use_oracle=True, oracle_source="bayes"), embed),
        sac.BayesObserver)
    assert isinstance(sac.make_observer(CFG, embed), sac.CBObserver)
    with pytest.raises(KeyError):
        sac.make_observer(sac.AblationConfig(name="x", use_oracle=True,
                                             oracle_source="minutes"), embed)


def test_coin_cb_differs_from_coin_in_exactly_the_source():
    a = asdict(sac.AblationConfig(**SW.CONFIG_BY_NAME["coin"]))
    b = asdict(CFG)
    assert {k for k in a if a[k] != b[k]} == {"name", "oracle_source"}
    assert "coin_cb_clearbuf" in SW.CONFIG_BY_NAME


def test_coin_cb_runs_end_to_end_on_cut_down_constants(tmp_path, monkeypatch):
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figures_sac"))
    sac.main(CFG, seeds=1, save_agg=True, workers=1,
             n_phase_ep=3, phase_reps=1, max_steps=5,
             buffer_capacity=32, batch_size=4, learning_starts=0,
             warmup_episodes=2)
    assert (tmp_path / "sac_runs" / "coin_cb" / "aggregated_results.csv").exists()
