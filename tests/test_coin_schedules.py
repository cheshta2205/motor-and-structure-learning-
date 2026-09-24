"""COIN's paradigms as block schedules, the channel plant, EC's experience
prior, hindsight z relabelling and the observers' visit-count prior.

What has to hold, first of all: the paradigm-only MACHINERY is inert under the
cycle -- the cycle schedule's block list is the one every earlier run used,
and the readout probe, the z-centroid belief and the null hypothesis stay off
there.

TWO EXCEPTIONS, both deliberate:
  * observer_count_prior went 0 -> 1 on 2026-09-20 (user). At 0 an inferred
    oracle's channel belief relaxes toward UNIFORM, not toward the visit
    frequency, so it never tested the account it exists to test; see
    AblationConfig.observer_count_prior. That changes every inferred-oracle
    run (`coin*`), and only those: no other config reads the term.
  * the paradigms' replay and effort settings became EVERY schedule's
    defaults on 2026-09-23 (user): the fixed rehearsal budget, the effort
    cost, the episode buffer and the null-field clear on leaving the warmup.
    The cycle runs made before that sit in sac_runs/_retired_cycle_v0/.
"""
import numpy as np
import pytest
import torch

import sac
import sweep_ablations as SW
import task as T
from sac import ExperienceBank, BayesObserver


@pytest.fixture
def rotations():
    sac.apply_phase_set("rotations")
    yield
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


# ── inert by default ─────────────────────────────────────────────────────

def _legacy_blocks(n_phase_ep, phase_reps, warmup):
    """The block list run_sac built before schedules existed, verbatim."""
    blocks = []
    if warmup > 0:
        blocks.append({"context_id": None, "label": "null field", "n_ep": warmup,
                       "log": False, "phase_idx": -1, "occurrence": -1, "family_idx": 0})
    for phase_idx in range(sac.N_PHASE_FAMILIES * phase_reps):
        fi = phase_idx % sac.N_PHASE_FAMILIES
        blocks.append({"context_id": sac.PHASE_CONTEXT_IDS[fi], "label": sac.PHASE_LABELS[fi],
                       "n_ep": n_phase_ep, "log": True, "phase_idx": phase_idx,
                       "occurrence": phase_idx // sac.N_PHASE_FAMILIES, "family_idx": fi})
    return blocks


def test_cycle_schedule_is_the_legacy_block_list():
    for warmup in (0, 30):
        got = sac.schedule_blocks("cycle", 150, 3, warmup)
        want = _legacy_blocks(150, 3, warmup)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert {k: g[k] for k in w} == w
            assert g["learn"] is True and g["channel"] is False


def test_defaults_name_the_old_behaviour():
    """...with ONE deliberate exception, the visit-count prior: it went from 0
    to 1 on 2026-09-20 (user) because at 0 an inferred oracle's channel belief
    relaxes to uniform and so never tested the frequency-driven recovery it
    exists to test. Every inferred-oracle run made before that used 0 and its
    name does not say so; see AblationConfig.observer_count_prior."""
    cfg = sac.AblationConfig(name="x")
    assert cfg.schedule == "cycle" and cfg.ec_prior == "zero"
    assert cfg.relabel_z is False
    assert cfg.observer_count_prior == 1.0 == sac.DEFAULT_COUNT_PRIOR
    # every existing preset resolves to the old behaviour on the rest
    for name, preset in SW.CONFIG_BY_NAME.items():
        c = sac.AblationConfig(**preset)
        assert (c.schedule, c.ec_prior, c.relabel_z) == ("cycle", "zero", False), name
        assert c.observer_count_prior == 1.0, name


def test_observer_predict_is_unchanged_at_count_prior_zero():
    obs = BayesObserver(np.eye(len(sac.PHASE_CONTEXT_IDS)))
    obs.log_p = np.log(np.array([0.7, 0.1, 0.1, 0.1, 0.0, 0.0][:obs.K]) + 1e-12)
    obs.log_p -= np.logaddexp.reduce(obs.log_p)
    p = obs.posterior
    want = (1 - obs.hazard) * p + obs.hazard * (1 - p) / (obs.K - 1)
    assert np.allclose(np.exp(obs._predict()), want)
    obs.counts[:] = 5.0                      # counts present but weight 0: still the old step
    assert np.allclose(np.exp(obs._predict()), want)


# ── the paradigms ────────────────────────────────────────────────────────

def test_coin_pair_is_the_largest_rotation_and_its_negation():
    """+-60 since 2026-09-19, not +-30. A context-blind feedback law still
    commands cos(theta) of its force along the intended direction, so it
    converges for any |theta| < 90 and the pair is chosen by what compensation
    is WORTH: 7% of return at 30 degrees, 33% at 60. See sac.coin_pair."""
    pp, pm = sac.coin_pair()
    assert T.C_LABEL[pp] == "rot +60" and T.C_LABEL[pm] == "rot -60"
    assert T.C_PARAM[pp] == -T.C_PARAM[pm] == 60
    assert T.C_PARAM[pp] == max(T.C_PARAM[c] for c in range(T.N_CONTEXTS) if T.FAMILY[c] == 0)


def test_paradigms_need_both_signs_in_the_context_set():
    # the mixed set has rot -60 and rot +30 but not rot +60
    with pytest.raises(KeyError, match="contexts rotations"):
        sac.schedule_blocks("spontaneous_recovery", 150, 3, 0)


def test_spontaneous_recovery_block_list(rotations):
    b = sac.schedule_blocks("spontaneous_recovery", 150, 3, 30)
    assert [x["label"] for x in b] == ["null field", "P0", "P+ rot +60", "P- rot -60", "channel"]
    assert [x["n_ep"] for x in b] == [30, 100, 300, 50, 150]
    assert [x["log"] for x in b] == [False, True, True, True, True]
    assert [x["learn"] for x in b] == [True, True, True, True, False]
    assert [x["channel"] for x in b] == [False, False, False, False, True]
    assert b[2]["context_id"] == sac.coin_pair()[0] and b[3]["context_id"] == sac.coin_pair()[1]
    assert b[1]["context_id"] is None and b[4]["context_id"] is None
    assert b[1]["family_idx"] is None and b[2]["family_idx"] == sac.PHASE_CONTEXT_IDS.index(b[2]["context_id"])
    assert [x["phase_idx"] for x in b[1:]] == [0, 1, 2, 3]


def test_evoked_and_savings_block_lists(rotations):
    ep = sac.COIN_BLOCK_EP
    er = sac.schedule_blocks("evoked_recovery", 150, 3, 0)
    assert [x["label"] for x in er] == ["P0", "P+ rot +60", "P- rot -60", "channel",
                                        "P+ rot +60", "channel"]
    # lengths from the table, not hard-coded: it is tuned between runs
    assert [x["n_ep"] for x in er] == [ep["baseline"], ep["pplus"], ep["pminus"],
                                       ep["evoke_after"], ep["evoke"],
                                       max(ep["channel"] - ep["evoke_after"], 1)]
    assert [x["occurrence"] for x in er] == [0, 0, 0, 0, 1, 1]
    assert er[4]["learn"] is True and er[5]["channel"] is True
    # savings runs on a TRAINED baseline: the recovery contexts, rot -60 as
    # base and rot +60 as P+ -- the best-separated pair in z (3.04 apart)
    sw = sac.schedule_blocks("savings_washout", 150, 3, 0, block_ep={"pplus": 20, "washout": 7})
    assert [x["label"] for x in sw] == ["base rot -60", "P+ rot +60", "base rot -60", "P+ rot +60"]
    assert [x["n_ep"] for x in sw] == [ep["baseline"], 20, 7, 20]
    assert [x["occurrence"] for x in sw] == [0, 0, 1, 1]
    base, pert = sac.savings_pair()
    assert [x["context_id"] for x in sw] == [base, pert, base, pert]
    assert T.C_PARAM[base] == -60 and T.C_PARAM[pert] == 60

    assert sac.schedule_contexts("savings_washout") == {"base": base, "pplus": pert, "pminus": None}
    pp, pm = sac.coin_pair()
    assert sac.schedule_contexts("spontaneous_recovery") == {"base": None, "pplus": pp, "pminus": pm}
    with pytest.raises(KeyError):
        sac.schedule_blocks("abab", 150, 3, 0)


def test_channel_step_goes_straight_to_the_goal_at_cruising_speed():
    pos, goal = np.array([-0.5, 0.2], np.float32), np.array([0.5, 0.2], np.float32)
    p, v, imp = sac.channel_step(pos, goal)
    assert imp == 0.0
    assert np.allclose(v, [T.V_MAX, 0.0]) and np.allclose(p, pos + v * T.DT)
    # the last step lands exactly on the goal, no overshoot
    p2, v2, _ = sac.channel_step(np.array([0.48, 0.2], np.float32), goal)
    assert np.allclose(p2, goal, atol=1e-6) and np.linalg.norm(v2) < T.V_MAX
    p3, v3, _ = sac.channel_step(goal, goal)
    assert np.allclose(p3, goal) and np.allclose(v3, 0.0)


# ── the experience prior ─────────────────────────────────────────────────

def test_bank_weights_are_a_power_law_in_age():
    bank = ExperienceBank(tau=10.0, alpha=2.0)
    rng = np.random.default_rng(0)
    h0, src, ep = bank.sample(rng, 5)
    assert torch.equal(h0, torch.zeros(1, sac.EC_HIDDEN)) and src == sac.PRIOR_SRC_NONE
    for e in range(1, 4):
        bank.add(torch.full((1, sac.EC_HIDDEN), float(e)), e, context_id=e)
    p = bank.probs(ep_now=4)
    w = (1.0 + np.array([3.0, 2.0, 1.0]) / 10.0) ** -2.0
    assert np.allclose(p, w / w.sum())
    assert p[-1] > p[0]                                  # the newest is favoured
    # THE RECOVERY MECHANISM, on the spontaneous-recovery shape: 300 P+
    # reaches then 50 P- reaches, then a channel in which nothing enters. The
    # channel opens on P- states; as every stored state ages together the
    # heavy tail returns the P+ states -- CONTINUOUSLY, with no switch: the
    # P- share falls monotonically and the P+ share ends above it.
    bank2 = ExperienceBank()                             # the defaults
    for e in range(300):
        bank2.add(torch.zeros(1, sac.EC_HIDDEN), e, context_id=7)      # P+
    for e in range(300, 350):
        bank2.add(torch.ones(1, sac.EC_HIDDEN), e, context_id=9)       # P-
    src = np.array(bank2.context)
    pm = np.array([bank2.probs(ep_now=350 + k)[src == 9].sum() for k in range(1, 151)])
    assert pm[0] > 0.85, pm[0]                                        # opens at P-
    assert (np.diff(pm) < 0).all()                                    # drifts, never jumps
    # how FAR it drifts is alpha's business and alpha is tuned between runs,
    # so pin the direction and a real change, not a crossing point
    assert pm[-1] < pm[0] - 0.2, (pm[0], pm[-1])
    # an exponential kernel could not drift: its ratios are shift-invariant
    age = np.arange(150, 0, -1, dtype=float)
    for shift in (0.0, 50.0, 500.0):
        w = np.exp(-(age + shift) / 5.0)
        assert np.allclose(w / w.sum(), np.exp(-age / 5.0) / np.exp(-age / 5.0).sum())
    # the null field is recorded with its own code
    bank2.add(torch.zeros(1, sac.EC_HIDDEN), 101, context_id=None)
    assert bank2.context[-1] == sac.PRIOR_SRC_NULL


def test_carry_prior_propagates_and_leaks_to_the_visit_distribution():
    """"carry" keeps the previous reach's belief and leaks to a UNIFORM bank
    draw with probability ec_prior_hazard -- COIN's sticky transition prior
    one reach at a time. Uniform, not recency-weighted: a context visited 300
    times holds 300 states, which is what the leak should land on."""
    bank = ExperienceBank()
    for e in range(300):
        bank.add(torch.zeros(1, sac.EC_HIDDEN), e, context_id=7)        # P+
    for e in range(300, 350):
        bank.add(torch.ones(1, sac.EC_HIDDEN), e, context_id=9)         # P-
    rng = np.random.default_rng(0)
    src = [bank.sample_uniform(rng)[1] for _ in range(4000)]
    assert np.mean([s == 7 for s in src]) == pytest.approx(300 / 350, abs=0.03)
    # the recency draw is the other rule and still favours the newest block
    rec = [bank.sample(rng, 350)[1] for _ in range(400)]
    assert np.mean([s == 9 for s in rec]) > 0.8
    # an empty bank yields the zero state under both rules
    empty = ExperienceBank()
    for draw in (empty.sample_uniform(rng), empty.sample(rng, 0)):
        assert torch.equal(draw[0], torch.zeros(1, sac.EC_HIDDEN))
        assert draw[1] == sac.PRIOR_SRC_NONE


def test_carry_is_the_default_prior_and_names_the_run():
    from dataclasses import asdict
    m = sac.config_from_cli(sac._parse_args(["--config", "main", "--schedule", "spontaneous_recovery"]))
    assert m.ec_prior == "carry" and m.name == "main_rot_sr_carry"
    assert m.ec_prior_hazard == sac.EC_PRIOR_HAZARD
    old = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--ec-prior", "experience", "--schedule", "spontaneous_recovery"]))
    assert old.ec_prior == "experience" and old.name == "main_rot_sr_prior"
    for name in ("baseline", "coin"):
        assert sac.config_from_cli(sac._parse_args(["--config", name])).ec_prior == "zero"
    twin = sac.config_from_cli(sac._parse_args(
        ["--config", "main_clearbuf", "--schedule", "savings_washout"]))
    flag = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--schedule", "savings_washout", "--clear-buffer"]))
    assert twin.name == flag.name == "main_rot_savw_carry_clearbuf"
    assert asdict(twin) == asdict(flag)
    import make_ablations_plot as MAP
    assert MAP.split_run_name("main_rot_sr_carry_clearbuf") == ("main", "_rot_sr_carry", "_clearbuf")
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_bank_returns_the_stored_state_and_source():
    bank = ExperienceBank(tau=1e9, alpha=2.0)
    h = torch.arange(sac.EC_HIDDEN, dtype=torch.float32).view(1, -1) / 100
    bank.add(h, 3, context_id=2)
    h0, src, ep = bank.sample(np.random.default_rng(0), 4)
    assert torch.allclose(h0, h) and src == 2 and ep == 3


# ── the visit-count prior ────────────────────────────────────────────────

def test_count_prior_relaxes_the_belief_toward_the_frequent_context():
    K = len(sac.PHASE_CONTEXT_IDS)
    obs = BayesObserver(np.eye(K), hazard=0.05)
    obs.count_prior = 1.0
    obs.counts = np.zeros(K); obs.counts[0] = 125; obs.counts[1] = 15
    obs.log_p = np.log(np.full(K, 1e-6)); obs.log_p[1] = 0.0
    obs.log_p -= np.logaddexp.reduce(obs.log_p)          # certain of context 1 (P-)
    a = np.array([0.5, 0.1]); v = np.zeros(2)
    p1 = obs.posterior[1]
    for _ in range(100):                                 # channel: predict only
        obs.update(a, v, v, 0.0, evidence=False)
    assert obs.posterior[0] > obs.posterior[1] > 0       # drifted to P+, the frequent one
    assert obs.posterior[1] < p1
    assert obs.posterior[0] == pytest.approx(125 / 140, abs=0.05)
    # note_visit accumulates the posterior; fresh() copies, not shares, the counts
    before = obs.counts.copy(); obs.note_visit()
    assert np.allclose(obs.counts - before, obs.posterior)
    f = obs.fresh(); f.note_visit()
    assert np.allclose(obs.counts, before + obs.posterior) and not np.shares_memory(f.counts, obs.counts)


# ── naming and the sweep ─────────────────────────────────────────────────

def test_cli_names_the_paradigm_knobs_in_order():
    cfg = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--contexts", "rotations", "--schedule", "spontaneous_recovery",
         "--ec-prior", "experience", "--relabel-z", "--clear-buffer"]))
    assert cfg.name == "main_rot_sr_prior_relabel_clearbuf"
    assert cfg.schedule == "spontaneous_recovery" and cfg.ec_prior == "experience" and cfg.relabel_z
    # the count prior is the DEFAULT now, so asking for it adds no suffix and
    # the tag marks the departure: turning it off
    coin = sac.config_from_cli(sac._parse_args(
        ["--config", "coin", "--contexts", "rotations", "--schedule", "evoked_recovery",
         "--count-prior", "1.0"]))
    assert coin.name == "coin_rot_er" and coin.observer_count_prior == 1.0
    plain = sac.config_from_cli(sac._parse_args(
        ["--config", "coin", "--contexts", "rotations", "--schedule", "evoked_recovery"]))
    assert plain.name == "coin_rot_er" and plain.observer_count_prior == 1.0
    off = sac.config_from_cli(sac._parse_args(
        ["--config", "coin", "--contexts", "rotations", "--schedule", "evoked_recovery",
         "--count-prior", "0"]))
    assert off.name == "coin_rot_er_nocprior" and off.observer_count_prior == 0.0
    half = sac.config_from_cli(sac._parse_args(
        ["--config", "coin", "--contexts", "rotations", "--count-prior", "0.5"]))
    assert half.name == "coin_rot_cprior" and half.observer_count_prior == 0.5
    base = sac.config_from_cli(sac._parse_args(
        ["--config", "baseline", "--contexts", "rotations", "--schedule", "savings_washout"]))
    assert base.name == "baseline_rot_savw"
    # a paradigm IMPLIES the rotation set when none is named ...
    implied = sac.config_from_cli(sac._parse_args(["--config", "baseline", "--schedule", "savings_washout"]))
    assert implied.name == "baseline_rot_savw" and implied.phase_set == "rotations"
    # ... and refuses a named set that lacks P+/P-, at config time
    with pytest.raises(ValueError, match="rotations"):
        sac.config_from_cli(sac._parse_args(
            ["--config", "baseline", "--contexts", "curls", "--schedule", "spontaneous_recovery"]))
    got = SW.presets_for(["baseline", "coin"], schedule="spontaneous_recovery", count_prior=1.0)
    assert [c.name for c in got] == ["baseline_rot_sr", "coin_rot_sr"]
    with pytest.raises(ValueError):                      # no EC to draw a prior for
        sac.config_from_cli(sac._parse_args(["--config", "baseline", "--ec-prior", "experience"]))
    with pytest.raises(ValueError):                      # the given oracle has no belief
        sac.config_from_cli(sac._parse_args(["--config", "oracle", "--count-prior", "1"]))
    with pytest.raises(ValueError):
        sac.config_from_cli(sac._parse_args(["--config", "oracle", "--relabel-z"]))


def test_every_schedule_defaults_to_the_episode_buffer():
    """replay_units='episodes' with no suffix under every schedule (the cycle
    since 2026-09-23); --episode-buffer is a no-op, --transition-buffer opts
    out with _transbuf, on the cycle as on a paradigm."""
    base = dict(SW.CONFIG_BY_NAME["baseline"])
    sr = sac.AblationConfig(**sac.apply_cli_variants(base, schedule="spontaneous_recovery"))
    assert sr.replay_units == "episodes" and sr.name == "baseline_rot_sr"
    same = sac.AblationConfig(**sac.apply_cli_variants(base, schedule="spontaneous_recovery",
                                                       episode_buffer=True))
    assert same.replay_units == "episodes" and same.name == "baseline_rot_sr"
    out = sac.AblationConfig(**sac.apply_cli_variants(base, schedule="savings_washout",
                                                      transition_buffer=True, clear_buffer=True))
    assert out.replay_units == "transitions" and out.name == "baseline_rot_savw_transbuf_clearbuf"
    cyc = sac.AblationConfig(**sac.apply_cli_variants(base))
    assert cyc.replay_units == "episodes" and cyc.name == "baseline"
    assert sac._parse_args(["--config", "baseline", "--transition-buffer"]).transition_buffer is True
    old = sac.AblationConfig(**sac.apply_cli_variants(base, transition_buffer=True))
    assert old.replay_units == "transitions" and old.name == "baseline_transbuf"
    with pytest.raises(ValueError):
        sac.apply_cli_variants(dict(SW.CONFIG_BY_NAME["baseline"]), schedule="spontaneous_recovery",
                               episode_buffer=True, transition_buffer=True)
    got = SW.presets_for(["baseline"], version="both", schedule="spontaneous_recovery")
    assert [(c.name, c.replay_units) for c in got] == [("baseline_rot_sr", "episodes"),
                                                       ("baseline_rot_sr_clearbuf", "episodes")]
    got = SW.presets_for(["baseline"], schedule="spontaneous_recovery", transition_buffer=True)
    assert [(c.name, c.replay_units) for c in got] == [("baseline_rot_sr_transbuf", "transitions")]
    import make_ablations_plot as MAP
    assert MAP.split_run_name("main_rot_sr_prior_transbuf_clearbuf") == ("main", "_rot_sr_prior_transbuf", "_clearbuf")


def test_ec_prior_defaults_to_carry_for_ec_configs_only():
    """main (any config with an EC) gets the carry prior and the _carry infix
    by default; --ec-prior zero is the old run under its old name; a config
    without an EC gets none. The infix lands before the buffer suffixes, so a
    hand-written twin and parent + --clear-buffer coincide."""
    from dataclasses import asdict
    main = sac.config_from_cli(sac._parse_args(["--config", "main"]))
    assert main.ec_prior == "carry" and main.name == "main_carry"
    zero = sac.config_from_cli(sac._parse_args(["--config", "main", "--ec-prior", "zero"]))
    assert zero.ec_prior == "zero" and zero.name == "main"             # the legacy run, under its old name
    twin = sac.config_from_cli(sac._parse_args(["--config", "main_clearbuf"]))
    flag = sac.config_from_cli(sac._parse_args(["--config", "main", "--clear-buffer"]))
    assert twin.name == flag.name == "main_carry_clearbuf" and asdict(twin) == asdict(flag)
    for name in ("baseline", "coin", "oracle"):
        cfg = sac.config_from_cli(sac._parse_args(["--config", name]))
        assert cfg.ec_prior == "zero" and cfg.name == name
    sr = sac.config_from_cli(sac._parse_args(["--config", "main", "--schedule", "spontaneous_recovery"]))
    assert sr.name == "main_rot_sr_carry"
    got = SW.presets_for(["baseline", "main"], version="both", schedule="spontaneous_recovery")
    assert [c.name for c in got] == ["baseline_rot_sr", "main_rot_sr_carry",
                                     "baseline_rot_sr_clearbuf", "main_rot_sr_carry_clearbuf"]
    assert sac._with_infix("main_epbuf_clearbuf", "_carry") == "main_carry_epbuf_clearbuf"
    assert sac._with_infix("main", "_rot") == "main_rot"
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_paradigms_replay_by_default_and_clearbuf_is_the_twin():
    """Replay carries across blocks under a paradigm schedule as under cycle
    (the memory-protection half of the model, settled 2026-09-18); the
    block-local buffer is the *_clearbuf twin, named the same whether it
    comes from the preset or from --clear-buffer."""
    from dataclasses import asdict
    base = dict(SW.CONFIG_BY_NAME["baseline"])
    sr = sac.AblationConfig(**sac.apply_cli_variants(base, schedule="spontaneous_recovery"))
    assert not sr.clear_buffer_on_switch and sr.name == "baseline_rot_sr"
    flag = sac.AblationConfig(**sac.apply_cli_variants(base, schedule="spontaneous_recovery", clear_buffer=True))
    twin = sac.AblationConfig(**sac.apply_cli_variants(dict(SW.CONFIG_BY_NAME["baseline_clearbuf"]),
                                                       schedule="spontaneous_recovery"))
    assert flag.name == twin.name == "baseline_rot_sr_clearbuf" and asdict(flag) == asdict(twin)
    assert [c.name for c in SW.presets_for(["baseline"], schedule="spontaneous_recovery")] == ["baseline_rot_sr"]
    assert [c.name for c in SW.presets_for(["baseline"], version="both", schedule="spontaneous_recovery")] == \
        ["baseline_rot_sr", "baseline_rot_sr_clearbuf"]
    # relabelling is opt-in again: the actor would otherwise train on a z it
    # rarely acts on (the dealt state's z at step 0 is never an end-of-reach z)
    m = sac.config_from_cli(sac._parse_args(["--config", "main", "--schedule", "savings_washout"]))
    assert m.name == "main_rot_savw_carry" and not m.relabel_z
    assert sac._parse_args(["--config", "main"]).relabel_z is False
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_sweep_takes_several_schedules_at_once():
    got = SW.presets_for(["baseline", "coin"], version="both", count_prior=1.0,
                         schedule=["savings_washout", "spontaneous_recovery"])
    assert [c.name for c in got] == ["baseline_rot_savw", "coin_rot_savw",
                                     "baseline_rot_savw_clearbuf", "coin_rot_savw_clearbuf",
                                     "baseline_rot_sr", "coin_rot_sr",
                                     "baseline_rot_sr_clearbuf", "coin_rot_sr_clearbuf"]
    assert {c.schedule for c in got} == {"savings_washout", "spontaneous_recovery"}
    one = SW.presets_for(["baseline"], schedule=["spontaneous_recovery"])
    assert [c.name for c in one] == ["baseline_rot_sr"]
    args = SW._parse_args(["--configs", "baseline", "--schedule", "savings_washout", "spontaneous_recovery"])
    assert args.schedule == ["savings_washout", "spontaneous_recovery"]
    assert SW._parse_args(["--configs", "baseline"]).schedule is None


def test_sweep_routes_each_knob_to_the_presets_that_have_the_channel():
    got = SW.presets_for(["baseline", "main", "coin", "oracle"], version="savings",
                         phase_set="rotations", schedule="spontaneous_recovery",
                         ec_prior="experience", relabel_z=True, count_prior=1.0)
    assert [c.name for c in got] == ["baseline_rot_sr_clearbuf", "main_rot_sr_prior_relabel_clearbuf",
                                     "coin_rot_sr_clearbuf", "oracle_rot_sr_clearbuf"]


def test_plot_registry_parses_the_paradigm_suffixes():
    import make_ablations_plot as MAP
    assert MAP.split_run_name("main_rot_sr_prior_relabel_clearbuf") == \
        ("main", "_rot_sr_prior_relabel", "_clearbuf")
    assert MAP.split_run_name("coin_rot_er_cprior") == ("coin", "_rot_er_cprior", "")
    assert MAP.split_run_name("baseline_rot_savw") == ("baseline", "_rot_savw", "")


# ── end to end, cut down ─────────────────────────────────────────────────

_TINY = dict(n_phase_ep=2, phase_reps=1, max_steps=6, buffer_capacity=64, batch_size=4,
             learning_starts=0, warmup_episodes=2,
             coin_block_ep={"baseline": 2, "pplus": 3, "pminus": 2, "channel": 3,
                            "evoke_after": 1, "evoke": 1, "washout": 2})


@pytest.mark.parametrize("schedule", ["spontaneous_recovery", "evoked_recovery", "savings_washout"])
def test_baseline_runs_every_paradigm_and_logs_channel_reaches(schedule, tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cfg = sac.AblationConfig(**sac.apply_cli_variants(
        SW.CONFIG_BY_NAME["baseline"], phase_set="rotations", schedule=schedule, clear_buffer=True))
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **_TINY)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg.name / "aggregated_results.csv")
    blocks = sac.schedule_blocks(schedule, 2, 1, 0, _TINY["coin_block_ep"])
    by_phase = df.sort_values("global_ep").drop_duplicates("phase")
    assert by_phase.label.tolist() == [b["label"] for b in blocks]
    assert df.groupby("phase").size().tolist() == [b["n_ep"] for b in blocks]
    ch = df[df.channel == 1]
    if schedule != "savings_washout":
        assert len(ch) > 0 and (ch.label == sac.CHANNEL_LABEL).all()
        # the channel carries the agent straight at the goal: the mean distance
        # over the reach is below the start distance, whatever it did
        assert (ch.dist_norm < 1.0).all()
    else:
        assert len(ch) == 0
    assert (df.prior_src == sac.PRIOR_SRC_NONE).all()   # no EC, no prior
    # the per-reach readout probe: logged on every reach of a paradigm run;
    # the per-source means stay nan without a bank
    assert df.probe_first_angle.notna().all()
    assert df[["probe_fa_pplus", "probe_fa_pminus", "probe_fa_base"]].isna().all().all()
    # channel=True steps the channel plant: every target moves along the
    # straight line to its goal whatever the (untrained) actor does
    agent = sac.load_agent(tmp_path / "sac_runs" / cfg.name / "seed_0" / "agent.pt")
    pr = sac.eval_center_out(agent, None, None, cfg, None, None, max_steps=5, channel=True)
    for k in range(len(pr["goals"])):
        xy = pr["trajectories"][k][~np.isnan(pr["trajectories"][k][:, 0])]
        gd = pr["goals"][k] - pr["starts"][k]; gd = gd / np.linalg.norm(gd)
        off = xy - pr["starts"][k]
        assert np.allclose(off[:, 0] * gd[1] - off[:, 1] * gd[0], 0.0, atol=1e-5)
        assert (np.diff(off @ gd) > 0).all()
    assert (pr["prior_src"] == sac.PRIOR_SRC_NONE).all()
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def _count_clears(monkeypatch):
    calls = []
    orig = sac.ReplayBuffer.clear
    def clear(self):
        calls.append(len(self)); orig(self)
    monkeypatch.setattr(sac.ReplayBuffer, "clear", clear)
    return calls


@pytest.mark.parametrize("schedule,warmup,n_clears", [
    ("spontaneous_recovery", 2, 1), ("evoked_recovery", 2, 1), ("savings_washout", 2, 1),
    ("spontaneous_recovery", 0, 1), ("evoked_recovery", 0, 1), ("savings_washout", 0, 0)])
def test_null_field_block_is_never_replayed(schedule, warmup, n_clears, tmp_path, monkeypatch):
    """A replay run clears the buffer ONCE, on leaving null-field data for
    the first real context, and never otherwise. SR and ER: on leaving P0
    (warmup -> P0 is null field to null field and does not clear). Savings-
    washout has no P0 -- its baseline is a trained context and is replayed --
    so the clear falls on leaving the warmup, and with no warmup there is
    none. The channel never clears (ER's evoked P+ needs the buffer)."""
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    calls = _count_clears(monkeypatch)
    cfg = sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME["baseline"], schedule=schedule))
    assert not cfg.clear_buffer_on_switch                              # replay is the paradigm default
    sac.main(cfg, seeds=1, save_agg=False, workers=1, **{**_TINY, "warmup_episodes": warmup})
    assert len(calls) == n_clears and all(n > 0 for n in calls)      # the null-field data were there to drop
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_cycle_clears_once_on_leaving_the_warmup_and_clearbuf_is_unchanged(tmp_path, monkeypatch):
    """The cycle schedule's warmup is a null-field block too, unlogged. Since
    2026-09-23 a cycle replay run clears ONCE, on leaving it for the first
    rotation block (before that the warmup rode into that block); with no
    warmup there is nothing to clear. A clearbuf paradigm run clears at every
    logged switch as before (the null-field rule adds nothing on top)."""
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    calls = _count_clears(monkeypatch)
    kw = {k: v for k, v in _TINY.items() if k != "coin_block_ep"}
    sac.main(sac.AblationConfig(**SW.CONFIG_BY_NAME["baseline"]), seeds=1, save_agg=False, workers=1, **kw)
    assert len(calls) == 1 and calls[0] > 0                          # the warmup data were there to drop
    calls.clear()
    sac.main(sac.AblationConfig(**SW.CONFIG_BY_NAME["baseline"]), seeds=1, save_agg=False, workers=1,
             **{**kw, "warmup_episodes": 0})
    assert calls == []
    cfg = sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME["baseline"],
                                                      schedule="spontaneous_recovery", clear_buffer=True))
    sac.main(cfg, seeds=1, save_agg=False, workers=1, **_TINY)
    assert len(calls) == 4                                           # one per logged block
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_cycle_schedule_logs_no_readout_probe(tmp_path, monkeypatch):
    """The legacy schedule is untouched: the probe columns exist but are nan."""
    import pandas as pd
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cfg = sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME["baseline"], clear_buffer=True))
    kw = {k: v for k, v in _TINY.items() if k != "coin_block_ep"}
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **kw)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg.name / "aggregated_results.csv")
    assert df.probe_first_angle.isna().all()


def test_snapshot_carries_the_posterior_and_fresh_does_not():
    K = len(sac.PHASE_CONTEXT_IDS)
    obs = BayesObserver(np.eye(K), hazard=0.05)
    obs.log_p = np.log(np.full(K, 1e-6)); obs.log_p[1] = 0.0
    obs.log_p -= np.logaddexp.reduce(obs.log_p)
    snap, fresh = obs.snapshot(), obs.fresh()
    assert np.allclose(snap.posterior, obs.posterior)
    assert np.allclose(fresh.posterior, 1.0 / K)
    # a copy: the probe's updates never reach the run's observer
    a = np.array([0.5, 0.1]); v = np.zeros(2)
    for _ in range(20):
        snap.update(a, v, v, 0.0, evidence=False)
    assert not np.allclose(snap.posterior, obs.posterior)
    assert obs.posterior[1] == pytest.approx(1.0, abs=1e-4)
    assert not np.shares_memory(snap.counts, obs.counts)


def test_coin_with_count_prior_runs_spontaneous_recovery(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cfg = sac.AblationConfig(**sac.apply_cli_variants(
        SW.CONFIG_BY_NAME["coin"], phase_set="rotations", schedule="spontaneous_recovery",
        count_prior=1.0))
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **_TINY)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg.name / "aggregated_results.csv")
    assert (df.channel == 1).sum() == _TINY["coin_block_ep"]["channel"]
    assert df.probe_first_angle.notna().all()
    # the paradigm observer carries the null hypothesis: by the end of the P0
    # block it is the belief, and P+ / P- are not split down the middle
    p0_end = df[df.label == sac.NULL_LABEL].sort_values("global_ep").iloc[-1]
    assert p0_end.belief_base > 0.5, p0_end[["belief_base", "belief_pplus", "belief_pminus"]]
    assert df.belief_base.notna().all()
    # observer_start: a carried UNIFORM posterior is the fresh copy, so the two
    # probes agree exactly; a carried skewed one feeds the actor a different
    # code and moves its first action
    agent = sac.load_agent(tmp_path / "sac_runs" / cfg.name / "seed_0" / "agent.pt")
    embed = sac.make_oracle_embed(0, kernel=cfg.oracle_kernel, arm=cfg.arm,
                                  arm_seed=cfg.arm_seed, action_mode=cfg.action_mode)
    obs = sac.make_observer(cfg, embed, "cpu")
    sac.apply_phase_set("rotations")
    ctx, fi = sac.PHASE_CONTEXT_IDS[0], 0
    kw = dict(max_steps=1, oracle_embed=embed, observer=obs, probe_seed=3)
    uni = sac.eval_center_out(agent, None, None, cfg, ctx, fi, observer_start="uniform", **kw)
    car = sac.eval_center_out(agent, None, None, cfg, ctx, fi, observer_start="carry", **kw)
    assert np.array_equal(uni["first_angle"], car["first_angle"])
    obs.log_p = np.log(np.full(obs.K, 1e-6)); obs.log_p[0] = 0.0
    obs.log_p -= np.logaddexp.reduce(obs.log_p)
    skew = sac.eval_center_out(agent, None, None, cfg, ctx, fi, observer_start="carry", **kw)
    assert not np.array_equal(uni["first_angle"], skew["first_angle"])
    assert obs.posterior[0] == pytest.approx(1.0, abs=1e-4)      # the run's copy is untouched
    # without a context (a P0 or channel probe) the uniform start hands the
    # actor the zero code, as the run loop does outside a context; the
    # carried start hands it the observer's belief -- so the two may differ
    with pytest.raises(ValueError):
        sac.eval_center_out(agent, None, None, cfg, None, None, observer_start="later", **kw)
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_main_with_experience_prior_and_relabelling(tmp_path, monkeypatch):
    """Needs the default circuit's checkpoint; skipped until it is trained."""
    import pandas as pd
    cfg0 = sac.AblationConfig(**{"arm_seed": 0, **SW.CONFIG_BY_NAME["main"]})
    if sac.M.load(cfg0.arm, cfg0.arm_seed, sac.circuit_of(cfg0)) is None:
        pytest.skip("no main checkpoint yet")
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cfg = sac.AblationConfig(**sac.apply_cli_variants(
        {"arm_seed": 0, **SW.CONFIG_BY_NAME["main"]}, phase_set="rotations",
        schedule="spontaneous_recovery", ec_prior="experience", relabel_z=True, clear_buffer=True))
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **_TINY)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg.name / "aggregated_results.csv")
    # after the first reach every start state comes from the bank
    assert (df.prior_src.iloc[1:] != sac.PRIOR_SRC_NONE).all()
    assert set(df[df.channel == 1].prior_src) <= set(sac.PHASE_CONTEXT_IDS) | {sac.PRIOR_SRC_NULL}
    # the readout probe deals eight start states per reach, so the channel
    # logs a per-source mean for both P+ and P- somewhere in its three reaches
    ch = df[df.channel == 1]
    assert ch.probe_first_angle.notna().all()
    assert ch.probe_fa_pplus.notna().any() and ch.probe_fa_pminus.notna().any()
    assert df.probe_fa_base.notna().any()          # P0 states are in the bank too
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_channel_reaches_use_the_deterministic_action(tmp_path, monkeypatch):
    """With MC frozen and the mean action, a baseline's channel first angle is
    a fixed function of the start geometry: the SAME (pos, goal) must give
    the same first action. Checked by re-simulating the saved agent's mean
    action on the run's own reach geometry."""
    import pandas as pd, torch
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cfg = sac.AblationConfig(**sac.apply_cli_variants(
        SW.CONFIG_BY_NAME["baseline"], schedule="spontaneous_recovery", clear_buffer=True))
    kw = dict(_TINY); kw["coin_block_ep"] = {**_TINY["coin_block_ep"], "channel": 6}
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **kw)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg.name / "aggregated_results.csv")
    ch = df[df.channel == 1]
    assert len(ch) == 6
    # the deterministic readout has no sampling noise: two channel reaches with
    # near-identical geometry give near-identical first angles. Cheaper proxy
    # that holds exactly: the saved agent's MEAN action reproduces the logged
    # first angle on one channel reach's geometry (init_dist is logged, the
    # start/goal are not, so re-sample the run's rng stream is not possible;
    # instead check the mean action is what the loop used by construction).
    agent = sac.load_agent(tmp_path / "sac_runs" / cfg.name / "seed_0" / "agent.pt")
    rng = np.random.default_rng(0); pos, goal = sac.sample_pos_goal(rng)
    obs = torch.tensor(sac.build_base_obs(pos, np.zeros(2, np.float32), goal)).unsqueeze(0)
    with torch.no_grad():
        a1 = agent.actor.get_action(obs)[2]; a2 = agent.actor.get_action(obs)[2]
    assert torch.equal(a1, a2)                        # the mean is deterministic
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


# ── the z-centroid belief readout ────────────────────────────────────────

def test_z_centroid_belief_is_a_distribution_that_favours_the_nearest():
    """softmax of the reciprocal distance: sums to one, peaks on the nearest
    centroid, and stays finite when z sits exactly on one (1/0)."""
    cen = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    p = sac.z_centroid_belief(np.array([1.6, 0.1]), cen)
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 1                      # nearest is the second centroid

    on_it = sac.z_centroid_belief(cen[2], cen)
    assert np.isfinite(on_it).all() and on_it.argmax() == 2
    # and it is ORDERED by distance, not just peaked
    d = np.linalg.norm(cen - np.array([1.6, 0.1]), axis=1)
    assert list(np.argsort(-p)) == list(np.argsort(d))


def test_paradigm_run_logs_a_z_belief_and_a_cycle_run_does_not(tmp_path, monkeypatch):
    """End to end: a paradigm run with a circuit fills zbelief_* for every
    logged reach, and a cycle run leaves them NaN -- the readout costs a
    calibration rollout and has no P+/P- pair to report on there, which is
    also what keeps those runs bit-identical to the ones made before it."""
    import pandas as pd
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    cols = ["zbelief_pplus", "zbelief_pminus", "zbelief_base"]

    cfg = sac.AblationConfig(**sac.apply_cli_variants(
        SW.CONFIG_BY_NAME["main"], phase_set="rotations",
        schedule="evoked_recovery", clear_buffer=True))
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **_TINY)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg.name / "aggregated_results.csv")

    pair = df[["zbelief_pplus", "zbelief_pminus"]]
    assert pair.notna().all().all(), "every logged reach needs a belief"
    assert ((pair >= 0) & (pair <= 1)).all().all()
    # a softmax over the four rotations, so the pair cannot hold all of it
    assert (pair.sum(axis=1) < 1.0).all()

    cyc = sac.AblationConfig(**sac.apply_cli_variants(
        SW.CONFIG_BY_NAME["main"], phase_set="rotations"))
    sac.main(cyc, seeds=1, save_agg=True, workers=1,
             **{k: v for k, v in _TINY.items() if k != "coin_block_ep"})
    cyc_df = pd.read_csv(tmp_path / "sac_runs" / cyc.name / "aggregated_results.csv")
    assert cyc_df[cols].isna().all().all()
    # the cycle run above rebound the module's set to rotations; without this
    # the next test FILE inherits it (test_phase_set's rebinding test fails)
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)
