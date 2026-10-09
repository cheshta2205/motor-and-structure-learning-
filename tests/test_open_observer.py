"""COIN with its novel-context process (sac.OpenObserver, preset coin_open).

What has to hold for it to be the open-set COIN it claims to be:

  it starts with no contexts and instantiates one from the first evidence
  a different plant spawns a second context; the first, revisited, is
    re-selected rather than spawned again, so its code comes back
  the null field is a context of its own, as is a reflection
  codes are unit-norm rows from one pool, the novel slot's the origin
  copies for the battery (fresh / snapshot) share no plant state with the run
  the context count is capped
  the preset differs from `coin` in exactly the oracle source
  it runs end to end, with the probes, and logs its state per reach
"""
import numpy as np
import pytest

import sac
import sweep_ablations as SW
import task as T
from sac import OpenObserver


@pytest.fixture(scope="module")
def embed():
    return sac.make_oracle_embed(0)


def _run(obs, cid, n_steps, rng, action_rng):
    pos = np.zeros(2, dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    for _ in range(n_steps):
        a = action_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        pos_new, vel_new, _F, wall = T.env_step(pos, vel, a, cid, rng=rng, return_impulse=True)
        obs.update(a, vel, vel_new, wall)
        pos, vel = pos_new, vel_new
    obs.note_visit()
    return obs


def test_starts_empty_and_spawns_from_the_first_evidence(embed):
    obs = OpenObserver(embed)
    assert obs.ids == [sac.OPEN_NOVEL] and obs.K == 1 and obs.n_contexts == 0
    assert not obs.code().any()                                   # the origin: no context
    _run(obs, T.rotation_id(-30), 30, np.random.default_rng(0), np.random.default_rng(1))
    assert obs.n_contexts == 1 and obs.ids == ["open0", sac.OPEN_NOVEL]
    assert obs.posterior[0] > 0.9
    A = obs.contexts[0].theta[:, :2]
    assert np.allclose(A, T.context_affine(T.rotation_id(-30))[0], atol=0.25)   # the plant, learned


def test_new_plant_spawns_and_the_old_one_is_reselected(embed):
    obs = OpenObserver(embed)
    rng, arng = np.random.default_rng(0), np.random.default_rng(1)
    _run(obs, T.rotation_id(-30), 60, rng, arng)
    code_first = obs.code().copy()
    _run(obs, T.rotation_id(-90), 60, rng, arng)
    assert obs.n_contexts == 2 and obs.map_slot == 1
    _run(obs, T.rotation_id(-30), 60, rng, arng)
    assert obs.n_contexts == 2 and obs.map_slot == 0               # re-selected, not spawned
    assert np.dot(obs.code(), code_first) > 0.9                    # its code came back


def test_null_field_and_reflection_are_contexts_of_their_own(embed):
    obs = OpenObserver(embed)
    rng, arng = np.random.default_rng(2), np.random.default_rng(3)
    _run(obs, T.rotation_id(-30), 60, rng, arng)
    _run(obs, None, 60, rng, arng)
    assert obs.n_contexts == 2 and obs.map_slot == 1
    assert np.allclose(obs.contexts[1].theta[:, :2], np.eye(2), atol=0.25)
    _run(obs, T.reflection_id(45), 60, rng, arng)
    assert obs.n_contexts == 3 and obs.map_slot == 2


def test_codes_are_unit_rows_from_one_pool_and_the_novel_slot_is_the_origin(embed):
    obs = OpenObserver(embed)
    rng, arng = np.random.default_rng(0), np.random.default_rng(1)
    for cid in (T.rotation_id(-30), T.rotation_id(-90), None):
        _run(obs, cid, 40, rng, arng)
    assert obs.embed.shape == (obs.K, embed.shape[1])
    assert np.allclose(np.linalg.norm(obs.embed[:-1], axis=1), 1.0, atol=1e-5)
    assert not obs.embed[-1].any()
    assert np.allclose(obs.embed[:-1], obs.pool[:obs.n_contexts])
    assert abs(float(obs.pool[0] @ obs.pool[1])) < 0.5             # near-orthogonal, as the oracle's


def test_battery_copies_share_no_plant_state(embed):
    obs = OpenObserver(embed)
    rng, arng = np.random.default_rng(0), np.random.default_rng(1)
    _run(obs, T.rotation_id(-30), 40, rng, arng)
    snap, fr = obs.snapshot(), obs.fresh()
    assert np.allclose(fr.posterior, 1.0 / fr.K) and fr.n_contexts == obs.n_contexts
    assert np.allclose(snap.posterior, obs.posterior)
    _run(snap, T.reflection_id(135), 40, rng, arng)
    _run(fr, None, 40, rng, arng)
    assert snap.n_contexts == 2 and fr.n_contexts == 2 and obs.n_contexts == 1
    assert obs.ids == ["open0", sac.OPEN_NOVEL]


def test_context_count_is_capped(embed):
    obs = OpenObserver(embed, max_contexts=2)
    rng, arng = np.random.default_rng(0), np.random.default_rng(1)
    for cid in (T.rotation_id(-30), T.rotation_id(-90), None, T.reflection_id(45)):
        _run(obs, cid, 40, rng, arng)
    assert obs.n_contexts == 2 and obs.K == 3


def test_evidence_free_steps_only_relax(embed):
    obs = OpenObserver(embed)
    _run(obs, T.rotation_id(-30), 30, np.random.default_rng(0), np.random.default_rng(1))
    p = obs.posterior.copy()
    obs.update(np.zeros(2), np.zeros(2), np.zeros(2))              # a null action
    obs.update(np.ones(2) * 0.3, np.zeros(2), np.zeros(2), wall_impulse=1.0)
    assert obs.n_contexts == 1 and np.allclose(obs.posterior, p, atol=2.5 * sac.BAYES_SWITCH_HAZARD)


def test_observer_angle_ignores_its_labels(embed):
    obs = OpenObserver(embed)
    _run(obs, T.rotation_id(-30), 20, np.random.default_rng(0), np.random.default_rng(1))
    assert np.isnan(sac.observer_angle(obs))


def test_preset_differs_from_coin_in_exactly_the_source():
    coin, opened = SW.CONFIG_BY_NAME["coin"], SW.CONFIG_BY_NAME["coin_open"]
    assert {k for k in set(coin) | set(opened) if coin.get(k) != opened.get(k)} == {"name", "oracle_source"}
    assert opened["oracle_source"] == "open" and "open" in sac.ORACLE_SOURCES
    cfg = sac.AblationConfig(**opened)
    assert isinstance(sac.make_observer(cfg, sac.make_oracle_embed(0)), OpenObserver)
    assert sac.observer_has_null(cfg) is False                     # no fixed null: the process covers it


def test_runs_end_to_end_with_the_probes(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figures_sac"))
    cfg = sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin_open"], phase_set="pair_wide", probe=True)
    sac.apply_phase_set("pair_wide")
    try:
        sac.main(sac.AblationConfig(**cfg), seeds=1, save_agg=True, workers=1,
                 n_phase_ep=3, phase_reps=1, max_steps=6,
                 buffer_capacity=32, batch_size=4, learning_starts=0, warmup_episodes=2)
    finally:
        sac.apply_phase_set(sac.DEFAULT_PHASE_SET)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg["name"] / "aggregated_results.csv")
    assert cfg["name"] == "coin_open_n2wide_pr"
    assert df.belief_base.isna().all()                             # no task-id hypotheses to report
    logged = df[df.open_n_contexts.notna()]
    assert len(logged) and logged.open_n_contexts.max() >= 1 and logged.open_map.notna().all()
    assert set(df.branch.dropna()) >= {"washout", "revisit", "interior", "novel_family"}
