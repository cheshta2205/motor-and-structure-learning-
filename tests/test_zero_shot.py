"""The zero-shot schedule (sac.zero_shot_blocks): off-table rotations as
contexts, the block list, the run naming, and a cut-down end-to-end run.

The standing constraint, as for every schedule added since 2026-09-18: the
cycle schedule and the three paradigms are untouched. The frozen sweep is a
measurement -- no learning, no replay writes, nothing into long-term memory
-- and the few-shot block is ordinary training in a context the agent has
never met.
"""
import numpy as np
import pandas as pd
import pytest

import sac
import sweep_ablations as SW
import task as T


@pytest.fixture
def rotations():
    sac.apply_phase_set("rotations")
    yield
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


@pytest.fixture(autouse=True)
def _restore_phase_set_after_each_test():
    """A run inside a test rebinds the module's context set (run_sac applies
    the config's); a test that FAILS mid-way would otherwise leave it bound
    and break the next test file's rebinding test. Restored unconditionally."""
    yield
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


# ── off-table rotations ──────────────────────────────────────────────────

def test_rotation_id_returns_the_table_id_for_a_table_angle():
    for c in range(T.N_CONTEXTS):
        if T.FAMILY[c] == 0:
            assert T.rotation_id(T.C_PARAM[c]) == c
            assert T.is_table_context(c)


def test_rotation_id_registers_a_novel_angle_once_and_deterministically():
    n_before = T.N_CONTEXTS
    c = T.rotation_id(45.0)
    assert c == T.rotation_id(45.0) == T.rotation_id(45)
    assert c >= T.EXTRA_ID_BASE and not T.is_table_context(c)
    assert T.N_CONTEXTS == n_before                    # the table is not resized
    assert T.FAMILY[c] == 0 and T.C_PARAM[c] == 45.0 and T.C_LABEL[c] == "rot +45"
    # the id is a function of the angle alone, so -45 and 315 are one context
    assert T.rotation_id(-45.0) == T.rotation_id(315.0) != c
    A, B, w = T.context_affine(c)
    assert np.allclose(A, T.rot_mat(np.deg2rad(45.0))) and not B.any() and not w.any()
    # the plant runs it like any context
    pos, vel, F = T.env_step(np.zeros(2, np.float32), np.zeros(2, np.float32),
                             np.array([1.0, 0.0], np.float32), c, np.random.default_rng(0))
    assert np.isfinite(pos).all() and np.isfinite(vel).all()


def test_zero_rotation_is_the_identity_plant_with_an_id():
    c = T.rotation_id(0.0)
    A, B, w = T.context_affine(c)
    I, Z, z = T.context_affine(None)
    assert np.allclose(A, I) and np.allclose(B, Z) and np.allclose(w, z)
    assert c is not None and not T.is_table_context(c)


# ── the block list ───────────────────────────────────────────────────────

def test_zero_shot_blocks_are_the_cycle_then_the_sweep_then_one_learn_block(rotations):
    cycle = sac.schedule_blocks("cycle", 150, 3, 0)
    zs = sac.schedule_blocks("zero_shot", 150, 3, 0, seed=3, zs_learn_angle=45.0)
    assert zs[:len(cycle)] == cycle                          # the cycle, verbatim
    sweep, learn = zs[len(cycle):-1], zs[-1]
    assert len(sweep) == len(sac.ZS_TEST_ANGLES)
    assert sorted(b["param"] for b in sweep) == sorted(sac.ZS_TEST_ANGLES)
    for b in sweep:
        assert b["kind"] == sac.ZS_KIND and b["learn"] is False and b["channel"] is False
        assert b["readout"] is True and b["write"] is False and b["probe"] is False
        assert b["centre_out"] is sac.ZS_CENTRE_OUT and b["test_horizon"] is True
        assert b["n_ep"] == sac.ZS_PROBE_EP and b["log"] is True
        assert b["context_id"] == T.rotation_id(b["param"])
        assert b["label"] == f"zs rot {b['param']:+g}"
        # a trained anchor keeps its oracle row; a novel angle has none
        if b["context_id"] in sac.PHASE_CONTEXT_IDS:
            assert b["family_idx"] == sac.PHASE_CONTEXT_IDS.index(b["context_id"])
        else:
            assert b["family_idx"] is None
    assert learn["kind"] == sac.FS_KIND and learn["learn"] is True and learn["write"] is True
    assert learn["probe"] is True and learn["readout"] is False
    assert learn["centre_out"] is sac.ZS_CENTRE_OUT and learn["test_horizon"] is True
    assert all("centre_out" not in b for b in cycle)
    assert learn["param"] == 45.0 and learn["n_ep"] == 150 and learn["label"] == "fs rot +45"
    # phase indices continue the cycle's and are distinct
    idx = [b["phase_idx"] for b in zs if b["log"]]
    assert idx == list(range(len(idx)))


def test_sweep_order_is_a_per_seed_permutation(rotations):
    a = [b["param"] for b in sac.schedule_blocks("zero_shot", 2, 1, 0, seed=0)[4:-1]]
    b = [b["param"] for b in sac.schedule_blocks("zero_shot", 2, 1, 0, seed=1)[4:-1]]
    assert sorted(a) == sorted(b) and a != b
    assert a == [x["param"] for x in sac.schedule_blocks("zero_shot", 2, 1, 0, seed=0)[4:-1]]


def test_zs_probe_length_is_overridable_and_learn_angle_is_a_knob(rotations):
    zs = sac.schedule_blocks("zero_shot", 5, 1, 0, {"zs_probe": 2}, zs_learn_angle=90.0)
    assert all(b["n_ep"] == 2 for b in zs if b.get("kind") == sac.ZS_KIND)
    assert zs[-1]["param"] == 90.0 and zs[-1]["n_ep"] == 5 and zs[-1]["label"] == "fs rot +90"


def test_zero_shot_has_no_paradigm_roles_and_no_null_hypothesis():
    assert sac.schedule_contexts("zero_shot") == {"base": None, "pplus": None, "pminus": None}
    assert sac.schedule_required_ids("zero_shot") == []
    cfg = sac.AblationConfig(name="x", schedule="zero_shot")
    assert sac.observer_has_null(cfg) is False
    for s in sac.PARADIGM_SCHEDULES:
        assert sac.observer_has_null(sac.AblationConfig(name="x", schedule=s)) is \
            (sac.schedule_contexts(s)["base"] is None)


# ── the posterior-mean angle ─────────────────────────────────────────────

def test_observer_angle_is_the_posterior_mean_over_rotation_hypotheses(rotations):
    embed = sac.make_oracle_embed(0)
    obs = sac.BayesObserver(embed)
    ids = obs.ids
    p = np.zeros(len(ids)); p[ids.index(T.rotation_id(30.0))] = 1.0
    obs.log_p = np.log(np.maximum(p, 1e-300))
    assert sac.observer_angle(obs) == pytest.approx(30.0)
    p[:] = 0; p[ids.index(T.rotation_id(30.0))] = 0.5; p[ids.index(T.rotation_id(60.0))] = 0.5
    obs.log_p = np.log(np.maximum(p, 1e-300))
    assert sac.observer_angle(obs) == pytest.approx(45.0)
    # with a null hypothesis: it counts as 0 degrees
    obs_n = sac.BayesObserver(embed, null_context=True)
    q = np.zeros(obs_n.K); q[obs_n.ids.index(T.rotation_id(60.0))] = 0.5; q[-1] = 0.5
    obs_n.log_p = np.log(np.maximum(q, 1e-300))
    assert sac.observer_angle(obs_n) == pytest.approx(30.0)


# ── naming ───────────────────────────────────────────────────────────────

def test_cli_names_the_zero_shot_run_and_keeps_the_cycle_defaults():
    # the schedule implies the wide context set and, on a frozen preset, the
    # wide circuit; the few-shot angle defaults to the wide midpoint, 60
    cfg = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot")
    assert cfg["name"] == "main_trainrotw_rotw_zs60" and cfg["phase_set"] == sac.WIDE_PHASE_SET
    assert cfg["train_contexts"] == "rotations_wide"
    assert cfg["schedule"] == "zero_shot"
    assert sac.AblationConfig(**cfg).zs_learn_angle == sac.ZS_LEARN_ANGLE == 60.0
    # a named circuit still wins (the unstructured control), and a preset
    # with no circuit gets no circuit tag
    lin = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot",
                                 train_contexts="linear_random")
    assert lin["name"] == "main_trainlin_rotw_zs60" and lin["train_contexts"] == "linear_random"
    assert sac.apply_cli_variants(SW.CONFIG_BY_NAME["baseline"], schedule="zero_shot")["name"] \
        == "baseline_rotw_zs60"
    # the replay and effort settings are every schedule's defaults since
    # 2026-09-23 (the effort cost reached zero-shot on 2026-09-21, the rest
    # with the cycle), so a zero-shot run and a plain cycle run agree on all
    # of them and neither carries a suffix for it
    zs, plain = (sac.AblationConfig(**cfg),
                 sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"],
                                                             phase_set="rotations")))
    for c in (zs, plain):
        assert c.effort_weight == sac.EFFORT_WEIGHT
        assert c.rehearsal_fraction == sac.REHEARSAL_FRACTION
        assert c.replay_units == "episodes"
    assert plain.name == "main_rot"                 # the context set's tag only
    cfg90 = sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin"], schedule="zero_shot",
                                   zs_learn_angle=90)
    assert cfg90["name"] == "coin_rotw_zs90" and cfg90["zs_learn_angle"] == 90.0
    twin = sac.apply_cli_variants(cfg90, clear_buffer=True)
    assert twin["name"] == "coin_rotw_zs90_clearbuf"
    with pytest.raises(ValueError):
        sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], zs_learn_angle=45)
    # the sweep routes the angle to every preset in a zero_shot selection
    presets = SW.presets_for(["baseline", "main"], schedule=["zero_shot"], zs_learn_angle=90)
    # (an EC config takes the default carry prior and its suffix, as every sweep does)
    assert [p.name for p in presets] == ["baseline_rotw_zs90", "main_trainrotw_rotw_zs90_carry"]
    assert all(p.zs_learn_angle == 90.0 for p in presets)
    # `--schedule zero_shot` alone names the runs on disk
    presets = SW.presets_for(["baseline", "main", "coin"], schedule=["zero_shot"])
    assert [p.name for p in presets] == ["baseline_rotw_zs60", "main_trainrotw_rotw_zs60_carry",
                                         "coin_rotw_zs60"]
    # the test horizon is in the name, after the angle and before the prior
    h = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot", zs_max_steps=12,
                               ec_prior="carry")
    assert h["name"] == "main_trainrotw_rotw_zs60_h12_carry" and h["zs_max_steps"] == 12
    assert sac.AblationConfig(**presets[0].__dict__).zs_max_steps is None
    with pytest.raises(ValueError):
        sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], zs_max_steps=12)
    with pytest.raises(ValueError):
        sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot", zs_max_steps=0)
    pp = sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin"], schedule="zero_shot", zs_max_steps=12,
                                zs_probe_ep=32)
    assert pp["name"] == "coin_rotw_zs60_h12_p32" and pp["zs_probe_ep"] == 32
    import make_ablations_plot as MAP
    assert MAP.split_run_name("main_trainrotw_rotw_zs60_h12_p32_carry") == \
        ("main", "_trainrotw_rotw_zs60_h12_p32_carry", "")
    with pytest.raises(ValueError):
        sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], zs_probe_ep=32)
    ph = SW.presets_for(["baseline", "main"], schedule=["zero_shot"], zs_max_steps=12)
    assert [p.name for p in ph] == ["baseline_rotw_zs60_h12", "main_trainrotw_rotw_zs60_h12_carry"]
    import make_ablations_plot as MAP
    import zero_shot_analysis as ZA
    assert MAP.split_run_name("main_trainrotw_rotw_zs60_h12_carry_clearbuf") == \
        ("main", "_trainrotw_rotw_zs60_h12_carry", "_clearbuf")
    assert ZA.horizon_of("main_trainrotw_rotw_zs60_h12_carry") == 12
    assert ZA.horizon_of("coin_rotw_zs60") == sac.MAX_STEPS
    assert ZA.variant_of("main_trainrotw_rotw_zs60_h12_carry") == "main [trainrotw]"


def test_plot_registry_places_a_zero_shot_run():
    import make_ablations_plot as MAP
    assert MAP.split_run_name("main_rot_zs45_clearbuf") == ("main", "_rot_zs45", "_clearbuf")
    assert MAP.split_run_name("coin_rot_zs90") == ("coin", "_rot_zs90", "")
    assert MAP._schedule_of("main_rot_zs45") == "zero_shot"
    assert MAP._schedule_of("main_rot_sr") == "spontaneous_recovery"
    assert MAP._schedule_of("main_rot") == "cycle"


# ── the wide, sparse rotation set ────────────────────────────────────────

def test_wide_rotation_set_is_table_pm30_and_registered_pm90():
    ids = T.wide_rotation_ids()
    assert [T.C_PARAM[c] for c in ids] == T.WIDE_ROT_ANGLES == [-90.0, -30.0, 30.0, 90.0]
    assert T.is_table_context(ids[1]) and T.is_table_context(ids[2])
    assert not T.is_table_context(ids[0]) and not T.is_table_context(ids[3])
    assert all(T.FAMILY[c] == 0 for c in ids)


def test_wide_circuit_training_set_replaces_the_rotations_only():
    import model as M
    ids = M.train_context_ids(M.WIDE_TRAIN_SET)
    assert ids[:4] == T.wide_rotation_ids()
    assert sorted(ids[4:]) == [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] != 0]
    assert M.train_contexts_tag(M.WIDE_TRAIN_SET) == "_trainrotw"
    stamp = M.train_contexts_stamp(M.WIDE_TRAIN_SET)
    assert stamp == "+train-rotations_wide[-90,-30,30,90]"
    spec = M.CircuitSpec(train_contexts=M.WIDE_TRAIN_SET)
    assert spec.context_ids == ids
    assert M.ckpt_path("main", 0, spec).endswith("main_trainrotw_seed0.pt")
    assert M.convention_for(spec) != M.convention_for()
    assert M.convention_for(spec).endswith(stamp + M.readout_stamp(M.DEFAULT_READOUT,
                                                                  M.RATE_BETA, M.NOISE_SIGMA))


def test_wide_phase_set_and_its_zero_shot_run_name():
    import make_ablations_plot as MAP
    import zero_shot_analysis as ZA
    assert sac.phase_ids_for(sac.WIDE_PHASE_SET) == T.wide_rotation_ids()
    assert sac.PHASE_SET_SUFFIX[sac.WIDE_PHASE_SET] == "_rotw"
    cfg = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot",
                                 phase_set=sac.WIDE_PHASE_SET, train_contexts="rotations_wide",
                                 zs_learn_angle=60, ec_prior="carry")
    assert cfg["name"] == "main_trainrotw_rotw_zs60_carry"
    assert cfg["train_contexts"] == "rotations_wide" and cfg["phase_set"] == sac.WIDE_PHASE_SET
    assert MAP.split_run_name(cfg["name"]) == ("main", "_trainrotw_rotw_zs60_carry", "")
    assert MAP._schedule_of(cfg["name"]) == "zero_shot"
    assert ZA.phase_set_of(cfg["name"]) == sac.WIDE_PHASE_SET
    assert ZA.phase_set_of("coin_rot_zs45") == "rotations"
    # the sweep routes the circuit set to the frozen presets only
    presets = SW.presets_for(["baseline", "coin", "main"], schedule=["zero_shot"],
                             phase_set=sac.WIDE_PHASE_SET, train_contexts="rotations_wide",
                             zs_learn_angle=60)
    assert [p.name for p in presets] == ["baseline_rotw_zs60", "coin_rotw_zs60",
                                         "main_trainrotw_rotw_zs60_carry"]


def test_zero_shot_blocks_under_the_wide_set_mark_its_anchors():
    sac.apply_phase_set(sac.WIDE_PHASE_SET)
    try:
        zs = sac.schedule_blocks("zero_shot", 150, 3, 0, seed=0, zs_learn_angle=60.0)
        sweep = [b for b in zs if b.get("kind") == sac.ZS_KIND]
        with_row = sorted(b["param"] for b in sweep if b["family_idx"] is not None)
        assert with_row == T.WIDE_ROT_ANGLES
        assert zs[-1]["param"] == 60.0 and zs[-1]["family_idx"] is None
        assert zs[:4] == sac.schedule_blocks("cycle", 150, 3, 0)[:4]
        assert [b["context_id"] for b in zs[:4]] == T.wide_rotation_ids()
    finally:
        sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


# ── the unstructured control: random linear transforms ───────────────────

def test_random_linear_set_matches_the_wide_effect_sizes_without_the_axis():
    import model as M
    ids = T.random_linear_ids()
    assert ids == T.random_linear_ids()                         # deterministic, idempotent
    assert all(c >= T.LINEAR_ID_BASE and T.FAMILY[c] == T.LINEAR_KIND for c in ids)
    assert all(not T.is_table_context(c) for c in ids)
    want = sorted(2 * abs(np.sin(np.deg2rad(a) / 2)) for a in T.WIDE_ROT_ANGLES)
    got = sorted(T.effect_size(T.LINEAR_MATS[c]) for c in ids)
    assert np.allclose(got, want, atol=1e-6)
    # none is a rotation: a rotation's A - I has the antisymmetric form
    for c in ids:
        E = T.LINEAR_MATS[c] - np.eye(2)
        assert not (np.isclose(E[0, 0], E[1, 1]) and np.isclose(E[0, 1], -E[1, 0]))
    # the plant and the affine lookup accept them
    A, B, w = T.context_affine(ids[0])
    assert np.allclose(A, T.LINEAR_MATS[ids[0]]) and not B.any() and not w.any()
    pos, vel, _F = T.env_step(np.zeros(2, np.float32), np.zeros(2, np.float32),
                              np.array([1.0, 0.0], np.float32), ids[0], np.random.default_rng(0))
    assert np.isfinite(pos).all()
    # the circuit training set: the four draws plus the eight non-rotations
    tr = M.train_context_ids(M.LINEAR_TRAIN_SET)
    assert tr[:4] == ids and sorted(tr[4:]) == [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] != 0]
    assert M.train_contexts_tag(M.LINEAR_TRAIN_SET) == "_trainlin"
    assert M.train_contexts_stamp(M.LINEAR_TRAIN_SET).startswith("+train-linear_random[seed")
    spec = M.CircuitSpec(train_contexts=M.LINEAR_TRAIN_SET)
    assert M.ckpt_path("main", 0, spec).endswith("main_trainlin_seed0.pt")
    assert M.convention_for(spec) != M.convention_for(M.CircuitSpec(train_contexts=M.WIDE_TRAIN_SET))
    # the table is untouched
    assert T.N_CONTEXTS == 12 and set(T.FAMILY[c] for c in range(T.N_CONTEXTS)) == {0, 1, 2}


def test_analysis_tells_the_two_circuits_apart_by_variant():
    import make_ablations_plot as MAP
    import zero_shot_analysis as ZA
    lin = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot",
                                 phase_set=sac.WIDE_PHASE_SET, train_contexts="linear_random",
                                 zs_learn_angle=60, ec_prior="carry")
    assert lin["name"] == "main_trainlin_rotw_zs60_carry"
    assert MAP.split_run_name(lin["name"]) == ("main", "_trainlin_rotw_zs60_carry", "")
    assert ZA.variant_of(lin["name"]) == "main [trainlin]"
    assert ZA.variant_of("main_trainrotw_rotw_zs60_carry") == "main [trainrotw]"
    assert ZA.variant_of("coin_rotw_zs60") == "coin"
    # the wide circuit is the experiment's own: solid, the preset's hue. The
    # unstructured control is dashed and darkened. The default circuit is
    # solid when alone and dotted, lightened, beside another circuit.
    import make_ablations_plot as MAP2
    sibs = ["main", "main [trainlin]", "main [trainrotw]", "coin"]
    assert ZA._style("main [trainrotw]", sibs) == "-"
    assert ZA._colour("main [trainrotw]", sibs) == MAP2.CONFIG_COLORS["main"]
    assert ZA._style("main [trainlin]", sibs) == "--"
    assert ZA._colour("main [trainlin]", sibs) == MAP2._shade(MAP2.CONFIG_COLORS["main"], 0.45)
    assert ZA._style("main", sibs) == ":" and ZA._colour("main", sibs) != MAP2.CONFIG_COLORS["main"]
    assert ZA._style("main", ["main", "coin"]) == "-"
    assert ZA._colour("main", ["main", "coin"]) == MAP2.CONFIG_COLORS["main"]
    assert ZA._style("coin", sibs) == "-" and ZA._colour("coin", sibs) == MAP2.CONFIG_COLORS["coin"]
    assert ZA._order(["main [trainlin]", "coin", "main [trainrotw]", "baseline"]) == \
        ["baseline", "main [trainlin]", "main [trainrotw]", "coin"]


# ── end to end, cut down ─────────────────────────────────────────────────

_TINY = dict(n_phase_ep=2, phase_reps=1, max_steps=6, buffer_capacity=64, batch_size=4,
             learning_starts=0, warmup_episodes=2, coin_block_ep={"zs_probe": 2})


@pytest.mark.parametrize("preset,horizon", [("baseline", None), ("coin", 4), ("main", None)])
def test_zero_shot_run_logs_the_sweep_and_the_few_shot_block(preset, horizon, tmp_path, monkeypatch):
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    knobs = {"ec_prior": "carry"} if SW.CONFIG_BY_NAME[preset].get("use_ec") else {}
    if horizon is not None:
        knobs["zs_max_steps"] = horizon
    cfg = sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME[preset],
                                                      schedule="zero_shot", **knobs))
    sac.main(cfg, seeds=1, save_agg=True, workers=1, **_TINY)
    curve = np.load(tmp_path / "sac_runs" / cfg.name / "seed_0" / "curve.npz",
                    allow_pickle=True)["curve"].item()
    df = pd.DataFrame({k: v for k, v in curve.items() if k != "z_mean"})
    blocks = sac.schedule_blocks("zero_shot", 2, 1, 2, _TINY["coin_block_ep"], seed=0,
                                 zs_learn_angle=cfg.zs_learn_angle)
    logged = [b for b in blocks if b["log"]]
    by_phase = df.sort_values("global_ep").drop_duplicates("phase")
    assert by_phase.label.tolist() == [b["label"] for b in logged]
    assert df.groupby("phase").size().tolist() == [b["n_ep"] for b in logged]
    assert by_phase.kind.tolist() == [b.get("kind", sac.TRAIN_KIND) for b in logged]
    train, zs, fs = (df[df.kind == k] for k in (sac.TRAIN_KIND, sac.ZS_KIND, sac.FS_KIND))
    assert len(train) == 8 and len(zs) == 2 * len(sac.ZS_TEST_ANGLES) and len(fs) == 2
    assert (train.learn == 1).all() and (zs.learn == 0).all() and (fs.learn == 1).all()
    assert sorted(zs.param.unique()) == sorted(sac.ZS_TEST_ANGLES)
    assert (fs.param == cfg.zs_learn_angle).all()
    # the frozen sweep never updated MC: `latest` stays at its per-reach reset
    assert (zs.q_loss == 0).all() and zs.alpha.nunique() == 1
    # ... and the paradigm-only columns stay nan, as under the cycle
    assert df.probe_first_angle.isna().all() and df.zbelief_pplus.isna().all()
    assert df.channel.eq(0).all()
    # path ratio: finite for every reach that arrived, nan otherwise
    ok = df.success == 1
    assert df.path_ratio[ok].notna().all() and (df.path_ratio[ok] >= 0.99).all()
    assert df.path_ratio[~ok].isna().all()
    # the test blocks are centre-out under the test horizon; the cycle's are
    # free reaches under the run's
    test = df[df.kind != sac.TRAIN_KIND]
    assert np.allclose(test.init_dist, sac.EVAL_RADIUS) and (test.target >= 0).all()
    assert (test.budget == (cfg.zs_max_steps or _TINY["max_steps"])).all()
    assert (train.target == -1).all() and (train.budget == _TINY["max_steps"]).all()
    assert (df.steps <= df.budget).all()
    # every target of the ring is visited in each 16-reach... here 2-reach block:
    # two reaches draw two distinct targets of the permutation
    for _, blk in zs.groupby("phase"):
        assert blk.target.nunique() == len(blk)
    if preset == "coin":
        assert df.belief_angle.notna().all()
        assert df.belief_angle.between(-90, 90).all()      # a mixture of the anchors
    else:
        assert df.belief_angle.isna().all()
    z = curve["z_mean"]
    if cfg.use_ec:
        arr = [v for v in z if isinstance(v, np.ndarray)]
        assert arr and all(v.shape == (sac.LATENT_DIM,) for v in arr)
    else:
        assert all(not isinstance(v, np.ndarray) for v in z)
    # the few-shot block is probed in its own context; the frozen ones are not
    probes = np.load(tmp_path / "sac_runs" / cfg.name / "seed_0" / "probes.npz",
                     allow_pickle=True)["probes"].tolist()
    labels_by_phase = {}
    for p in probes:
        labels_by_phase.setdefault(p["phase"], set()).add(p["eval_label"])
    zs_phases = set(b["phase_idx"] for b in logged if b.get("kind") == sac.ZS_KIND)
    assert not (zs_phases & set(labels_by_phase))
    fs_phase = logged[-1]["phase_idx"]
    fs_label = f"rot {cfg.zs_learn_angle:+g}"
    assert fs_label in labels_by_phase[fs_phase]
    assert any(p["phase"] == fs_phase and p["is_current"] and p["eval_label"] == fs_label
               for p in probes)
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_analysis_builds_the_figure_from_tiny_runs(tmp_path, monkeypatch):
    """zero_shot_analysis pools runs that differ only in the learn angle,
    decodes an angle for coin (posterior) and main (z), and writes the
    figure through plotting_utils.save."""
    import zero_shot_analysis as ZA
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    for preset, angle in (("baseline", 45), ("baseline", 90), ("coin", 45), ("main", 45)):
        knobs = {"ec_prior": "carry"} if SW.CONFIG_BY_NAME[preset].get("use_ec") else {}
        cfg = sac.AblationConfig(**sac.apply_cli_variants(
            SW.CONFIG_BY_NAME[preset], schedule="zero_shot", zs_learn_angle=angle, **knobs))
        sac.main(cfg, seeds=2, save_agg=False, workers=1, **_TINY)
    runs = ZA.find_runs(str(tmp_path / "sac_runs"))
    assert set(runs) == {"baseline_rotw_zs45", "baseline_rotw_zs90", "coin_rotw_zs45",
                         "main_trainrotw_rotw_zs45_carry"}
    df = ZA.load_curves(runs)
    assert set(df.base) == {"baseline", "coin", "main"}
    assert set(df.variant) == {"baseline", "coin", "main [trainrotw]"}
    assert set(df.phase_set) == {sac.WIDE_PHASE_SET}
    assert ZA.anchors_of(df) == T.WIDE_ROT_ANGLES
    tbl = ZA.sweep_table(df)
    # pooled over the two baseline runs: one row per (seed, angle), 13 angles
    b = tbl[tbl.variant == "baseline"]
    assert len(b) == 2 * len(sac.ZS_TEST_ANGLES) and b.seed.nunique() == 2
    dec = ZA.decoded_angles(df)
    assert set(dec[dec.variant == "coin"].source) == {"posterior"}
    assert set(dec[dec.variant == "main [trainrotw]"].source) == {"z decoder"}
    assert dec[dec.variant == "baseline"].empty
    assert dec[dec.variant == "coin"].decoded.between(-90, 90).all()
    fs = ZA.few_shot_curves(df)
    assert set(fs) == {("baseline", 45.0), ("baseline", 90.0), ("coin", 45.0),
                       ("main [trainrotw]", 45.0)}
    s = ZA.summary(df)
    assert list(s.index) == ["baseline", "coin", "main [trainrotw]"] and (s.seeds == 2).all()
    out = tmp_path / "figs" / "zero_shot"
    ZA.make_figure(df, "return", out)
    assert out.with_suffix(".pdf").exists() and out.with_suffix(".png").exists()
    assert (tmp_path / "figs" / "zero_shot_caption.txt").exists()
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_frozen_sweep_writes_nothing_to_long_term_memory(tmp_path, monkeypatch):
    """No bank entries and no observer visit counts from the frozen blocks:
    the bank is written only by training reaches, the counts only by them."""
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    bank_ctx = []
    orig_add = sac.ExperienceBank.add
    def add(self, h, episode, context_id):
        bank_ctx.append(context_id); orig_add(self, h, episode, context_id)
    monkeypatch.setattr(sac.ExperienceBank, "add", add)
    visits = []
    orig_visit = sac._ContextPosterior.note_visit
    def note_visit(self):
        visits.append(int(np.argmax(self.posterior))); orig_visit(self)
    monkeypatch.setattr(sac._ContextPosterior, "note_visit", note_visit)

    cfg = sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin"], schedule="zero_shot"))
    sac.main(cfg, seeds=1, save_agg=False, workers=1, **_TINY)
    # training reaches (8) + few-shot reaches (2) count visits; the 26 frozen ones do not
    assert len(visits) == 10
    cfg = sac.AblationConfig(**sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot",
                                                      ec_prior="carry"))
    sac.main(cfg, seeds=1, save_agg=False, workers=1, **_TINY)
    # "novel" = outside the TRAINED set (which, under the wide set, itself
    # holds the off-table +-90 ids), i.e. the few-shot block's context alone
    trained = set(sac.phase_ids_for(cfg.phase_set))
    novel = [c for c in bank_ctx if c is not None and c not in trained]
    assert len(bank_ctx) == 2 + 8 + 2                      # warmup + training + few-shot
    assert novel and all(c == T.rotation_id(cfg.zs_learn_angle) for c in novel)
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


# ── the Rao-Blackwellised carry ──────────────────────────────────────────

def test_carry_mean_is_named_and_runs(tmp_path, monkeypatch):
    """carry_mean carries the settled-step MEAN of EC's hidden state; it
    names the run _carrymean, and a tiny zero-shot run completes with the
    same logged structure as carry."""
    import make_ablations_plot as MAP
    assert "carry_mean" in sac.EC_PRIORS
    cfg = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], schedule="zero_shot",
                                 zs_max_steps=12, ec_prior="carry_mean")
    assert cfg["name"] == "main_trainrotw_rotw_zs60_h12_carrymean"
    assert MAP.split_run_name(cfg["name"]) == ("main", "_trainrotw_rotw_zs60_h12_carrymean", "")
    assert MAP.base_config("main_trainrotw_rotw_zs60_h12_carry") == "main"
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figs"))
    sac.main(sac.AblationConfig(**cfg), seeds=1, save_agg=True, workers=1, **_TINY)
    df = pd.read_csv(tmp_path / "sac_runs" / cfg["name"] / "aggregated_results.csv")
    assert (df.kind == sac.FS_KIND).sum() == 2 and (df.kind == sac.ZS_KIND).sum() == 2 * len(sac.ZS_TEST_ANGLES)
    # with a bank and a carry, a reach's EC start is dealt from a context
    assert (df.prior_src != sac.PRIOR_SRC_NONE).any()
