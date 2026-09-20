"""
CONFIG_PRESETS sanity checks, plus a smoke test that runs every preset
end-to-end on drastically cut-down constants -- this is the fast, always-on
replacement for a manual "does it even run" check, covering all six configs.
"""
import pytest

import sac
import sweep_ablations as SW


CORE = ["baseline", "oracle", "coin", "coin_cb", "oracle_geom", "main", "control",
        "end_to_end", "cb_only_end_to_end"]


def test_config_presets_are_the_core_and_their_clearbuf_twins():
    """The table was cut back to the core presets on 2026-09-16 -- the
    oracle_snr ladder, the gp circuit and the no_cb lesion were removed -- and
    every core preset has exactly one *_clearbuf twin. Nothing else. `coin`
    (the inferred oracle) joined the core on 2026-09-17."""
    names = list(SW.CONFIG_BY_NAME)
    assert len(names) == len(set(names)) == 18
    assert set(names) == set(CORE) | {f"{c}_clearbuf" for c in CORE}
    assert list(SW.CORE_PRESETS) == CORE
    # every preset's dict must name itself: the run directory is config.name
    for name, cfg in SW.CONFIG_BY_NAME.items():
        assert cfg["name"] == name


CLEARBUF_TWINS = {f"{c}_clearbuf": c for c in CORE}


@pytest.mark.parametrize("twin,parent", list(CLEARBUF_TWINS.items()))
def test_clearbuf_twin_differs_from_parent_in_exactly_one_knob(twin, parent):
    """The retention-vs-savings contrast is parent against twin, and it is
    only interpretable if NOTHING else differs. Compare the fully-resolved
    configs, defaults included, so a default that later changes for one but
    not the other is caught here rather than in a figure."""
    from dataclasses import asdict
    a = asdict(sac.AblationConfig(**SW.CONFIG_BY_NAME[parent]))
    b = asdict(sac.AblationConfig(**SW.CONFIG_BY_NAME[twin]))
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"name", "clear_buffer_on_switch"}, diff
    assert a["clear_buffer_on_switch"] is False
    assert b["clear_buffer_on_switch"] is True


def test_cli_clear_buffer_sets_the_knob_and_renames_the_run():
    """--clear-buffer on a parent must produce the twin, byte for byte: same
    knob, same name, hence the same sac_runs/ directory. Without the rename
    the savings version would overwrite the retention version's data."""
    from dataclasses import asdict
    via_flag = sac.config_from_cli(sac._parse_args(["--config", "main", "--clear-buffer"]))
    via_preset = sac.config_from_cli(sac._parse_args(["--config", "main_clearbuf"]))
    assert via_flag.clear_buffer_on_switch is True
    assert via_flag.name == "main_carry_clearbuf"     # the default prior, inserted before the suffix
    assert asdict(via_flag) == asdict(via_preset)


def test_cli_clear_buffer_is_a_no_op_on_a_twin():
    """No double suffix, no change: the twin already has the knob."""
    cfg = sac.config_from_cli(sac._parse_args(["--config", "oracle_clearbuf", "--clear-buffer"]))
    assert cfg.name == "oracle_clearbuf"
    assert cfg.clear_buffer_on_switch is True


def test_cli_without_clear_buffer_leaves_the_parent_alone():
    cfg = sac.config_from_cli(sac._parse_args(["--config", "main"]))
    assert cfg.name == "main_carry"
    assert cfg.clear_buffer_on_switch is False


def test_clear_buffer_default_is_the_module_constant():
    """Every preset that does not pin the knob follows sac's default, so the
    constant is still the single place to change the default for all."""
    assert sac.AblationConfig(name="x").clear_buffer_on_switch is sac.CLEAR_BUFFER_ON_PHASE_SWITCH
    assert sac.CLEAR_BUFFER_ON_PHASE_SWITCH is False


def test_config_flag_combinations():
    expected_flags = {
        "baseline": dict(use_ec=False, use_cb=False, freeze=False, use_oracle=False),
        "oracle": dict(use_ec=False, use_cb=False, freeze=False, use_oracle=True),
        "main": dict(use_ec=True, use_cb=True, freeze=True, use_oracle=False),
        "control": dict(use_ec=True, use_cb=True, freeze=True, use_oracle=False),
        "end_to_end": dict(use_ec=True, use_cb=True, freeze=False, use_oracle=False),
        "cb_only_end_to_end": dict(use_ec=False, use_cb=True, freeze=False, use_oracle=False),
        # the two versions of the experiment: parent flags plus the one knob
        "main_clearbuf": dict(use_ec=True, use_cb=True, freeze=True,
                              use_oracle=False, clear_buffer_on_switch=True),
        "oracle_clearbuf": dict(use_ec=False, use_cb=False, freeze=False,
                                use_oracle=True, clear_buffer_on_switch=True),
    }
    for name, flags in expected_flags.items():
        cfg = SW.CONFIG_BY_NAME[name]
        for k, v in flags.items():
            assert cfg.get(k, False) == v, (name, k)

    assert SW.CONFIG_BY_NAME["main"]["arm"] == "main"
    assert SW.CONFIG_BY_NAME["control"]["arm"] == "control"
    # frozen is the default and unsuffixed; the GP circuit is the default mode
    for name in ("main", "control"):
        cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME[name])
        assert cfg.freeze is True and cfg.action_mode == "gp"


def test_run_tasks_clamps_workers_and_handles_no_tasks(monkeypatch):
    """Fewer tasks than workers must not spawn idle processes, and a single
    task must stay in-process (spawning for one seed is pure overhead)."""
    assert sac.run_tasks([], workers=8) == []

    ran = []
    monkeypatch.setattr(sac, "run_one_seed",
                         lambda cfg, seed, **kw: ran.append((cfg.name, seed)))

    config = sac.AblationConfig(name="baseline")
    # workers is clamped to len(tasks)==1, so this runs serially in-process --
    # which the monkeypatch above can only observe if no subprocess was used
    sac.run_tasks([(config, 0)], workers=8, run_kwargs={})

    assert ran == [("baseline", 0)]


@pytest.mark.parametrize("name", list(SW.CONFIG_BY_NAME))
def test_every_preset_runs_end_to_end_on_cut_down_constants(name, tmp_path, monkeypatch):
    monkeypatch.setattr(sac, "RUN_DIR", str(tmp_path / "sac_runs"))
    monkeypatch.setattr(sac, "FIG_DIR", str(tmp_path / "figures_sac"))

    # arm_seed 0 is only a DEFAULT here: a preset that pins its own (because
    # its checkpoint exists for one seed only) must win, not collide
    config = sac.AblationConfig(**{"arm_seed": 0, **SW.CONFIG_BY_NAME[name]})
    # a preset that drives a FROZEN circuit (or measures its geometry) needs
    # that circuit's checkpoint; before it is trained the run cannot start
    needs_ckpt = config.freeze or config.oracle_kernel != "identity" or config.oracle_source == "cb"
    if needs_ckpt and sac.M.load(config.arm, config.arm_seed, sac.circuit_of(config)) is None:
        pytest.skip(f"{name}: no {sac.circuit_of(config)} checkpoint for arm {config.arm!r} yet")
    sac.main(
        config, seeds=1, save_agg=True,
        n_phase_ep=2, phase_reps=1, max_steps=5,
        buffer_capacity=32, batch_size=4, learning_starts=0,
        warmup_episodes=2,   # exercise the warmup path, but keep it cheap
    )

    agg_path = tmp_path / "sac_runs" / name / "aggregated_results.csv"
    assert agg_path.exists()


def test_cli_accepts_every_preset():
    """--config's choices are derived from CONFIG_BY_NAME, not restated.

    Regression: they were a hardcoded list, so main_frozen_no_cb existed as a
    preset and ran fine in-process while the CLI rejected it."""
    import sys

    for name in SW.CONFIG_BY_NAME:
        argv = sys.argv
        try:
            sys.argv = ["sac.py", "--config", name, "--seeds", "1"]
            args = sac._parse_args()
        finally:
            sys.argv = argv
        assert args.config == name
        assert sac.AblationConfig(**SW.CONFIG_BY_NAME[name]).name == name


def test_cli_arm_seed_does_not_clobber_a_preset_that_pins_its_own(monkeypatch):
    """Regression: --arm-seed defaulted to 0 and overwrote a preset's pinned
    arm_seed=1 unconditionally, so the run asked for a checkpoint that does
    not exist. It failed loudly there, but the same pattern on a variant whose
    seed 0 DID exist would have silently frozen the wrong network.

    No current preset pins its own arm_seed (main_frozen_rnn, which did, is
    gone), so the scenario is synthesised: _parse_args derives --config's
    choices from the LIVE CONFIG_BY_NAME, so an entry added here is accepted."""
    import sys

    monkeypatch.setitem(SW.CONFIG_BY_NAME, "pinned_probe",
                        dict(SW.CONFIG_BY_NAME["main"],
                             name="pinned_probe", arm_seed=1))

    def resolve(argv):
        old = sys.argv
        try:
            sys.argv = argv
            args = sac._parse_args()
        finally:
            sys.argv = old
        cfg = dict(SW.CONFIG_BY_NAME[args.config])
        if args.arm_seed is not None:
            cfg["arm_seed"] = args.arm_seed
        cfg.setdefault("arm_seed", 0)
        return cfg["arm_seed"]

    assert resolve(["sac.py", "--config", "pinned_probe"]) == 1
    assert resolve(["sac.py", "--config", "pinned_probe", "--arm-seed", "5"]) == 5
    assert resolve(["sac.py", "--config", "main"]) == 0
