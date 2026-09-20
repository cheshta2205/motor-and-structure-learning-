"""Episode-indexed replay, and the CLIs that select the two versions.

The transition-indexed buffer couples memory span to competence: a fixed
10000 transitions is ~190 episodes for an agent failing at 53 steps each and
~400 for one succeeding at 25, so the agent that does best replays the most
old-context data. In episode units the span, the recency constant and each
episode's total sampling mass are independent of episode length. What has to
hold:

  transition units are BIT-IDENTICAL to before (every existing run)
  in episode units an episode's mass depends on its age and nothing else
  capacity evicts whole episodes, never a partial one
  the CLI variants rename the run so no two versions share a directory
  a derived twin equals the hand-written one, field for field
"""
from dataclasses import asdict

import numpy as np
import pytest

import sac
import sweep_ablations as SW
from sac import ReplayBuffer

Z = np.zeros(2, dtype=np.float32)


def _add(buf, *, episode, t_step, n=1):
    for _ in range(n):
        buf.add(Z, Z, 0.0, Z, False, s_tm1=Z, a_tm1=Z, s_t=Z, a_t=Z, s_next=Z,
                e_t=Z, h_ec_prev=Z, t_step=t_step, episode=episode)


def _old_transition_probs(timesteps, t_now, gamma_t, mix):
    """The formula as it was before episode units existed, verbatim."""
    dt = np.maximum(0.0, float(t_now) - np.asarray(timesteps, dtype=np.float64))
    logits = -(dt / max(gamma_t, 1e-8))
    logits -= np.max(logits)
    p = np.exp(logits)
    p = p / p.sum()
    if mix > 0.0:
        p = (1.0 - mix) * p + mix * (1.0 / len(p))
        p = p / p.sum()
    return p


# ── transition units: nothing moved ──────────────────────────────────────

def test_transition_units_are_bit_identical_to_the_old_formula():
    buf = ReplayBuffer(capacity=50, gamma_t=7.0, mix=0.02)
    ts = [0, 3, 3, 10, 21, 40]
    for i, t in enumerate(ts):
        _add(buf, episode=i, t_step=t)
    probs = buf._sampling_probs(t_now=41)
    assert np.array_equal(probs, _old_transition_probs(ts, 41, 7.0, 0.02))


def test_transition_units_ignore_ep_now_and_evict_by_transition():
    buf = ReplayBuffer(capacity=5)
    for i in range(8):
        _add(buf, episode=0, t_step=i)
    assert len(buf) == 5                       # deque maxlen, as before
    assert np.allclose(buf._sampling_probs(t_now=8, ep_now=99),
                       buf._sampling_probs(t_now=8))


def test_default_units_are_transitions():
    assert ReplayBuffer().units == "transitions"
    assert sac.AblationConfig(name="x").replay_units == "transitions"


# ── episode units ────────────────────────────────────────────────────────

def test_episode_mass_is_independent_of_episode_length():
    """A 60-step failure and a 20-step success at the same age must carry the
    same TOTAL sampling mass -- the whole point of the unit change."""
    # flat recency (huge gamma_ep), so the only thing that COULD differ is length
    buf = ReplayBuffer(units="episodes", gamma_ep=1e12, mix=0.0)
    _add(buf, episode=0, t_step=0, n=60)
    _add(buf, episode=1, t_step=60, n=20)
    p = buf._sampling_probs(t_now=80, ep_now=2)
    eps = np.asarray(buf.episodes)
    assert p[eps == 0].sum() == pytest.approx(p[eps == 1].sum())
    assert p.sum() == pytest.approx(1.0)
    # and with real recency the split is exactly the age ratio, length still absent
    buf2 = ReplayBuffer(units="episodes", gamma_ep=40.0, mix=0.0)
    _add(buf2, episode=0, t_step=0, n=60)
    _add(buf2, episode=1, t_step=60, n=20)
    p2 = buf2._sampling_probs(t_now=80, ep_now=2)
    eps2 = np.asarray(buf2.episodes)
    assert p2[eps2 == 0].sum() / p2[eps2 == 1].sum() == pytest.approx(np.exp(-1.0 / 40.0))


def test_episode_recency_decays_by_episode_age_not_steps():
    """Two single-transition episodes: the older one's mass is exp(-d/gamma_ep)
    of the newer one's, whatever their step stamps say."""
    buf = ReplayBuffer(units="episodes", gamma_ep=10.0, mix=0.0)
    _add(buf, episode=0, t_step=0)
    _add(buf, episode=5, t_step=100_000)       # step gap is irrelevant
    p = buf._sampling_probs(t_now=100_001, ep_now=5)
    assert p[0] / p[1] == pytest.approx(np.exp(-5.0 / 10.0))


def test_episode_uniform_floor_is_uniform_over_episodes():
    """With mix=1 every episode gets 1/n_episodes regardless of length."""
    buf = ReplayBuffer(units="episodes", mix=1.0)
    _add(buf, episode=0, t_step=0, n=30)
    _add(buf, episode=1, t_step=30, n=3)
    p = buf._sampling_probs(t_now=33, ep_now=2)
    eps = np.asarray(buf.episodes)
    assert p[eps == 0].sum() == pytest.approx(0.5)
    assert p[eps == 1].sum() == pytest.approx(0.5)


def test_episode_capacity_evicts_whole_oldest_episodes():
    buf = ReplayBuffer(units="episodes", capacity_episodes=2)
    _add(buf, episode=0, t_step=0, n=7)
    _add(buf, episode=1, t_step=7, n=3)
    assert buf.n_episodes == 2 and len(buf) == 10
    _add(buf, episode=2, t_step=10, n=4)       # third episode: evict episode 0 entirely
    assert buf.n_episodes == 2
    assert len(buf) == 7
    assert set(buf.episodes) == {1, 2}
    assert len(buf.timesteps) == len(buf.observations) == len(buf.episodes)


def test_episode_units_require_the_episode_index():
    buf = ReplayBuffer(units="episodes")
    with pytest.raises(ValueError):
        buf.add(Z, Z, 0.0, Z, False, s_tm1=Z, a_tm1=Z, s_t=Z, a_t=Z, s_next=Z,
                e_t=Z, h_ec_prev=Z, t_step=0)
    _add(buf, episode=0, t_step=0)
    with pytest.raises(ValueError):
        buf._sampling_probs(t_now=1)


def test_clear_resets_episode_bookkeeping():
    buf = ReplayBuffer(units="episodes", capacity_episodes=3)
    _add(buf, episode=0, t_step=0, n=4)
    _add(buf, episode=1, t_step=4, n=4)
    buf.clear()
    assert len(buf) == 0 and buf.n_episodes == 0
    _add(buf, episode=7, t_step=8, n=2)          # usable again, counts from empty
    assert buf.n_episodes == 1 and len(buf) == 2


def test_bad_units_rejected():
    with pytest.raises(ValueError):
        ReplayBuffer(units="minutes")


# ── the fixed rehearsal budget ───────────────────────────────────────────

def test_rehearsal_budget_is_inert_until_asked_for():
    """Every run before it: no budget, no marked block, identical probs."""
    buf = ReplayBuffer(units="episodes", gamma_ep=40.0)
    for e in range(10):
        _add(buf, episode=e, t_step=e, n=3)
    before = buf._sampling_probs(t_now=30, ep_now=10).copy()
    assert buf.rehearsal == 0.0 and buf.block_start_ep is None
    buf.mark_block_start(5)                       # marked, but no budget
    assert np.array_equal(buf._sampling_probs(t_now=30, ep_now=10), before)
    buf.rehearsal = 0.15                          # budget, but nothing older
    buf.mark_block_start(0)
    assert np.array_equal(buf._sampling_probs(t_now=30, ep_now=10), before)
    buf.mark_block_start(99)                      # ... or nothing current
    assert np.array_equal(buf._sampling_probs(t_now=30, ep_now=10), before)


def test_rehearsal_budget_fixes_both_shares_whatever_the_block_length():
    """The current block holds 1 - rehearsal of the mass from its first update
    on, instead of climbing with its length -- and recency still orders each
    pool. 300 old reaches then a new block, as P- follows P+."""
    r = 0.15
    for n_new in (2, 10, 50):
        buf = ReplayBuffer(units="episodes", gamma_ep=40.0, capacity_episodes=1000)
        for e in range(300):
            _add(buf, episode=e, t_step=e, n=2)
        buf.rehearsal = r; buf.mark_block_start(300)
        for e in range(300, 300 + n_new):
            _add(buf, episode=e, t_step=e, n=2)
        p = buf._sampling_probs(t_now=10_000, ep_now=300 + n_new)
        cur = np.asarray(buf.episodes) >= 300
        assert p[cur].sum() == pytest.approx(1.0 - r), (n_new, p[cur].sum())
        assert p.sum() == pytest.approx(1.0)
        # recency survives inside each pool
        eps = np.asarray(buf.episodes)
        assert p[eps == 300 + n_new - 1].sum() > p[eps == 300].sum()
        assert p[eps == 299].sum() > p[eps == 0].sum()


def test_rehearsal_budget_beats_recency_dilution_on_the_real_block_lengths():
    """Under recency alone a 50-reach P- block is a fifth of the batch ten
    reaches in; with the budget it is 85% from the start."""
    def share(n_new, rehearsal):
        buf = ReplayBuffer(units="episodes", gamma_ep=sac.PRIORITY_GAMMA_EP,
                           capacity_episodes=sac.BUFFER_CAPACITY_EPISODES, mix=sac.PRIORITY_MIX)
        for e in range(300):
            _add(buf, episode=e, t_step=e)
        buf.rehearsal = rehearsal; buf.mark_block_start(300)
        for e in range(300, 300 + n_new):
            _add(buf, episode=e, t_step=e)
        p = buf._sampling_probs(t_now=10_000, ep_now=300 + n_new)
        return p[np.asarray(buf.episodes) >= 300].sum()
    assert share(10, 0.0) < 0.3                      # recency alone: diluted
    assert share(10, sac.PARADIGM_REHEARSAL) == pytest.approx(1 - sac.PARADIGM_REHEARSAL)
    assert share(50, sac.PARADIGM_REHEARSAL) == pytest.approx(1 - sac.PARADIGM_REHEARSAL)


def test_clear_forgets_the_block_boundary():
    buf = ReplayBuffer(units="episodes")
    _add(buf, episode=0, t_step=0, n=2); buf.mark_block_start(0)
    buf.clear()
    assert buf.block_start_ep is None


def test_paradigms_set_the_budget_and_cycle_does_not():
    base = dict(SW.CONFIG_BY_NAME["main"])
    assert sac.AblationConfig(**sac.apply_cli_variants(base)).rehearsal_fraction == 0.0
    for sch in ("spontaneous_recovery", "evoked_recovery", "savings_washout"):
        c = sac.AblationConfig(**sac.apply_cli_variants(base, schedule=sch))
        assert c.rehearsal_fraction == sac.PARADIGM_REHEARSAL, sch
    assert sac.AblationConfig(name="x").rehearsal_fraction == 0.0


# ── CLI variants and the two versions ────────────────────────────────────

def test_episode_buffer_flag_sets_units_and_suffixes_the_name():
    cfg = sac.config_from_cli(sac._parse_args(["--config", "main", "--episode-buffer"]))
    assert cfg.replay_units == "episodes"
    assert cfg.name == "main_carry_epbuf"          # main runs default to the carry prior


def test_both_flags_name_the_savings_twin_of_the_epbuf_run():
    cfg = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--episode-buffer", "--clear-buffer"]))
    assert cfg.name == "main_carry_epbuf_clearbuf"
    assert cfg.replay_units == "episodes" and cfg.clear_buffer_on_switch is True


def test_variant_flags_are_idempotent():
    once = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"],
                                  clear_buffer=True, episode_buffer=True)
    twice = sac.apply_cli_variants(once, clear_buffer=True, episode_buffer=True)
    assert once == twice
    assert once["name"] == "main_epbuf_clearbuf"        # apply_cli_variants itself adds no default prior


def test_sweep_savings_version_derives_the_hand_written_twins():
    """`--version savings` on the parents must produce exactly the *_clearbuf
    presets, field for field -- otherwise two ways of asking for the same run
    could quietly differ."""
    # ec_prior="zero": the hand-written twins are the old runs; the default
    # prior for EC presets would otherwise name main_carry_clearbuf
    derived = SW.presets_for(list(SW.CORE_PRESETS), version="savings", ec_prior="zero")
    for cfg in derived:
        assert cfg.name in SW.CONFIG_BY_NAME, cfg.name
        assert asdict(cfg) == asdict(sac.AblationConfig(**SW.CONFIG_BY_NAME[cfg.name]))


def test_sweep_retention_version_is_the_presets_as_named():
    got = SW.presets_for(["baseline", "oracle"], version="retention")
    assert [c.name for c in got] == ["baseline", "oracle"]
    assert all(c.clear_buffer_on_switch is False for c in got)


def test_sweep_both_lists_parents_then_twins_without_duplicates():
    got = [c.name for c in SW.presets_for(["main", "main_clearbuf"], version="both")]
    assert got == ["main_carry", "main_carry_clearbuf"]


def test_sweep_episode_buffer_applies_to_every_selected_preset():
    got = SW.presets_for(["baseline", "main"], version="both", episode_buffer=True)
    assert [c.name for c in got] == ["baseline_epbuf", "main_carry_epbuf",
                                     "baseline_epbuf_clearbuf", "main_carry_epbuf_clearbuf"]
    assert all(c.replay_units == "episodes" for c in got)


def test_sweep_cli_parses_version_and_episode_buffer():
    args = SW._parse_args(["--configs", "baseline", "--version", "both", "--episode-buffer"])
    assert args.version == "both" and args.episode_buffer is True
    assert SW._parse_args(["--configs", "baseline"]).version == "retention"


def test_episode_units_run_end_to_end_on_cut_down_constants(tmp_path, monkeypatch):
    """The run loop must thread the episode index into add() and sample();
    the unit tests above never touch it. Both versions, so the clear-buffer
    path is exercised in episode units too."""
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figures_sac"))
    for name, clear in (("baseline_epbuf", False), ("baseline_epbuf_clearbuf", True)):
        config = sac.AblationConfig(name=name, replay_units="episodes",
                                    clear_buffer_on_switch=clear)
        sac.main(config, seeds=1, save_agg=True, workers=1,
                 n_phase_ep=3, phase_reps=1, max_steps=5,
                 buffer_capacity=32, buffer_capacity_episodes=4, batch_size=4,
                 learning_starts=0, warmup_episodes=2)
        assert (tmp_path / "sac_runs" / name / "aggregated_results.csv").exists()


def test_circuit_knobs_select_the_frozen_checkpoint_and_rename():
    """--action-mode / --ec-readout / --rate-beta pick WHICH circuit drives MC;
    the name carries all of it, circuit first, buffer second."""
    cfg = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--action-mode", "iid", "--ec-readout", "noisy",
         "--rate-beta", "0.001", "--clear-buffer"]))
    # the default channel (noisy, beta 1e-3, sigma 0.3) adds no readout suffix
    assert cfg.name == "main_iid_carry_clearbuf"
    assert cfg.action_mode == "iid" and cfg.ec_readout == "noisy"
    assert cfg.rate_beta == pytest.approx(1e-3) and cfg.clear_buffer_on_switch is True
    # --rate-beta alone implies the noisy readout
    cfg2 = sac.config_from_cli(sac._parse_args(["--config", "main", "--rate-beta", "0.01"]))
    assert cfg2.name == "main_noisy0.01_carry" and cfg2.ec_readout == "noisy"
    assert cfg2.noise_sigma == sac.M.NOISE_SIGMA
    # --noise-sigma joins the tag only when it is not the default, and alone
    # also implies the noisy readout (at the default beta)
    cfg5 = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--action-mode", "iid", "--rate-beta", "0.0001",
         "--noise-sigma", "0.1", "--clear-buffer"]))
    assert cfg5.name == "main_iid_noisy0.0001_sigma0.1_carry_clearbuf"
    assert cfg5.noise_sigma == pytest.approx(0.1) and cfg5.rate_beta == pytest.approx(1e-4)
    cfg6 = sac.config_from_cli(sac._parse_args(["--config", "main", "--noise-sigma", "1"]))
    assert cfg6.name == f"main_noisy{sac.M.RATE_BETA:g}_sigma1_carry" and cfg6.ec_readout == "noisy"
    # the deterministic readout is the variant now
    cfg7 = sac.config_from_cli(sac._parse_args(["--config", "main", "--ec-readout", "tanh"]))
    assert cfg7.name == "main_tanh_carry" and cfg7.ec_readout == "tanh"
    # the default mode adds no suffix; the legacy spelling normalises
    cfg3 = sac.config_from_cli(sac._parse_args(["--config", "main", "--action-mode", "gp"]))
    assert cfg3.name == "main_carry"
    cfg4 = sac.config_from_cli(sac._parse_args(["--config", "main", "--action-mode", "fitted"]))
    assert cfg4.name == "main_ar1_carry" and cfg4.action_mode == "ar1"


def test_circuit_knobs_are_refused_on_a_config_without_a_frozen_circuit():
    with pytest.raises(ValueError):
        sac.config_from_cli(sac._parse_args(["--config", "baseline", "--action-mode", "iid"]))
    with pytest.raises(ValueError):
        sac.apply_cli_variants(SW.CONFIG_BY_NAME["oracle"], ec_readout="noisy")


def test_sweep_applies_circuit_knobs_only_to_frozen_presets():
    got = SW.presets_for(["baseline", "main"], version="both", action_mode="iid",
                         ec_readout="noisy", rate_beta=1e-3)
    assert [c.name for c in got] == ["baseline", "main_iid_carry",
                                     "baseline_clearbuf", "main_iid_carry_clearbuf"]
    # the knobs reached the frozen preset and left the blind one untouched
    assert got[1].action_mode == "iid" and got[0].action_mode == SW.CONFIG_BY_NAME["baseline"].get(
        "action_mode", sac.M.DEFAULT_ACTION_MODE)


def test_epbuf_twin_still_differs_from_its_parent_in_one_knob():
    """replay_units is set on BOTH sides by the CLI, so the twin contrast stays
    a single knob in the episode-buffer version too."""
    parent, twin = SW.presets_for(["main"], version="both", episode_buffer=True)
    a, b = asdict(parent), asdict(twin)
    assert {k for k in a if a[k] != b[k]} == {"name", "clear_buffer_on_switch"}
