"""
Tests for circuit_probe.py -- the decodability probe itself, not the circuit.

The probe's job is to be trustworthy when it reports a low number, so what
matters is that it cannot manufacture a high one: the split must be by
episode, and the rollout must go through sac.py's own functions rather than a
private copy that could stay correct while training drifts.
"""
import numpy as np
import pytest
import torch

import circuit_probe as CP
import sac
import task as T


SMALL = dict(n_episodes=3, max_steps=8)


def test_rollout_uses_sac_circuit_state(monkeypatch):
    """The probe must be driven by sac.py's own conditioning. If sac.py ever
    stops applying S_SCALE, the probe has to break the same way -- a private
    copy of the loop would keep reporting a healthy number while training
    quietly ran on mis-scaled inputs, which is the exact failure this probe
    exists to catch."""
    calls = []
    real = sac.circuit_state

    def spy(pos, vel, variant=None):
        out = real(pos, vel, variant)
        calls.append(out.copy())
        return out

    monkeypatch.setattr(sac, "circuit_state", spy)
    CP.rollout_z("main", 0, contexts=[1, 4], **SMALL)

    assert calls, "probe bypassed sac.circuit_state"
    raw = np.stack(calls)[:, :2] / sac.S_SCALE
    assert np.all(raw >= -1e-4) and np.all(raw <= T.GRID_SIZE + 1e-4)


def test_rollout_shapes_and_labels_line_up():
    contexts = [1, 4, 12]
    out = CP.rollout_z("main", 0, contexts=contexts, **SMALL)
    n = len(contexts) * SMALL["n_episodes"] * SMALL["max_steps"]

    assert out["z"].shape == (n, sac.LATENT_DIM)
    assert out["base_obs"].shape == (n, sac.SDIM + 2)
    for key in ("y", "episode", "t"):
        assert out[key].shape == (n,), key
    assert sorted(np.unique(out["y"])) == list(range(len(contexts)))
    # every episode carries exactly one context, or the split leaks labels
    for ep in np.unique(out["episode"]):
        assert len(np.unique(out["y"][out["episode"] == ep])) == 1
    assert set(np.unique(out["t"])) == set(range(SMALL["max_steps"]))


def test_episode_is_never_split_across_train_and_test():
    """The leakage guard. Steps inside an episode share a context AND a
    trajectory, so a timestep-wise split puts near-duplicates on both sides
    and inflates every number -- a probe of ours once read 0.60 that way when
    the truth was chance."""
    from decode import seq_split
    episode = np.repeat(np.arange(40), 7)
    for seed in range(5):
        test = seq_split(episode, seed=seed)
        assert not set(np.unique(episode[test])) & set(np.unique(episode[~test]))


def test_decode_accuracy_is_chance_on_shuffled_labels():
    """A label with no relationship to the features must come back at chance.
    Guards against the probe scoring on training rows."""
    rng = np.random.default_rng(0)
    n_ep, per_ep, k = 60, 8, 4
    episode = np.repeat(np.arange(n_ep), per_ep)
    X = rng.normal(size=(n_ep * per_ep, 12)).astype(np.float32)
    y = np.repeat(rng.integers(0, k, n_ep), per_ep)      # one label per episode

    mean, _ = CP.decode_accuracy(X, y, episode)
    assert abs(mean - 1.0 / k) < 0.18, mean


def test_decode_accuracy_recovers_a_planted_signal():
    """The other direction: a feature that does carry the label must be found,
    so a low number means absence of signal rather than a broken decoder."""
    rng = np.random.default_rng(0)
    n_ep, per_ep, k = 60, 8, 4
    episode = np.repeat(np.arange(n_ep), per_ep)
    labels = rng.integers(0, k, n_ep)
    y = np.repeat(labels, per_ep)
    X = rng.normal(scale=0.25, size=(n_ep * per_ep, k)).astype(np.float32)
    X[np.arange(len(y)), y] += 3.0

    mean, _ = CP.decode_accuracy(X, y, episode)
    assert mean > 0.9, mean


def test_degenerate_split_reports_nan_not_a_number():
    episode = np.zeros(20, dtype=int)          # a single episode cannot be split
    X = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    y = np.zeros(20, dtype=int)
    mean, sd = CP.decode_accuracy(X, y, episode)
    assert np.isnan(mean) and np.isnan(sd)


@pytest.mark.parametrize("arm", ["main", "control"])
def test_frozen_checkpoint_carries_context_under_the_sac_rollout(arm):
    """The finding this module was written to establish, as a regression test.

    Both arms decode ~1.00 over the six phase contexts once the circuit is fed
    model.build's units. Mis-scaling drops this to ~0.28 against a 0.167
    chance level, so the 0.6 threshold sits far from both and catches that
    class of wiring bug without being brittle about the exact value.
    """
    result = CP.probe_arm(arm, 0, n_episodes=25, max_steps=25)
    assert result["z"] > 0.6, result
    assert abs(result["raw_obs"] - result["chance"]) < 0.06, (
        "raw [pos, vel, goal] must be at chance -- otherwise the probe is "
        "leaking episode identity and the z number cannot be trusted")
