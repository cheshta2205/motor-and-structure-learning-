"""AblationConfig.phase_set: which contexts the RL task uses.

What has to hold:

  "mixed" is the derived default (two per family); "<family>s" is all four
    of one family; apply_phase_set rebinds the module's ids/labels/count and
    is idempotent
  the knob names the run (the family's short suffix) after circuit suffixes
    and before buffer ones, and is allowed on presets without a frozen circuit
  the sweep applies it to every selected preset
  make_ablations_plot derives the variant from its parent
"""
import pytest

import make_ablations_plot as MAP
import sac
import sweep_ablations as SW
import task as T


@pytest.fixture
def restore_phase_set():
    yield
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_sets_follow_the_family_table():
    assert sac.PHASE_SETS == ("mixed", "rotations", "curls", "drifts")
    mixed = sac.phase_ids_for("mixed")
    assert mixed == sac._phase_ids() and len(mixed) == 2 * len(T.FAM_NAMES)
    fams = {}
    for s, name in (("rotations", "rotation"), ("curls", "curl"), ("drifts", "drift")):
        ids = sac.phase_ids_for(s)
        assert len(ids) == 4 and all(T.FAM_NAMES[T.FAMILY[c]] == name for c in ids)
        fams[s] = set(ids)
    assert set.union(*fams.values()) == set(range(T.N_CONTEXTS))
    with pytest.raises(KeyError):
        sac.phase_ids_for("reflections")


def test_apply_phase_set_rebinds_module_state(restore_phase_set):
    ids0 = list(sac.PHASE_CONTEXT_IDS)
    rot = sac.apply_phase_set("rotations")
    assert list(sac.PHASE_CONTEXT_IDS) == rot != ids0
    assert sac.PHASE_LABELS == [T.C_LABEL[c] for c in rot]
    assert sac.N_PHASE_FAMILIES == 4
    # functions read the set at call time: the observer and the embed follow
    assert list(sac.BayesObserver(sac.make_oracle_embed(0)).ids) == rot
    assert sac.apply_phase_set("rotations") == rot                  # idempotent
    assert sac.apply_phase_set("mixed") == ids0
    assert list(sac.PHASE_CONTEXT_IDS) == ids0 and sac.N_PHASE_FAMILIES == len(ids0)


def test_phase_set_names_the_run_in_the_right_place():
    cfg = sac.config_from_cli(sac._parse_args(
        ["--config", "main", "--action-mode", "iid", "--contexts", "rotations", "--clear-buffer"]))
    assert cfg.name == "main_iid_rot_carry_clearbuf" and cfg.phase_set == "rotations"
    base = sac.config_from_cli(sac._parse_args(["--config", "baseline", "--contexts", "drifts"]))
    assert base.name == "baseline_drift" and base.phase_set == "drifts"   # allowed on baseline
    same = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], phase_set="mixed")
    assert same["name"] == "main"                                          # default set: no suffix (no CLI default here)
    with pytest.raises(KeyError):
        sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], phase_set="reflections")


def test_sweep_applies_the_task_knob_to_every_preset():
    got = SW.presets_for(["baseline", "main", "coin"], version="savings", phase_set="curls")
    assert [c.name for c in got] == ["baseline_curl_clearbuf", "main_curl_carry_clearbuf", "coin_curl_clearbuf"]
    assert all(c.phase_set == "curls" for c in got)


def test_plot_registry_derives_task_variants(monkeypatch):
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    assert MAP.split_run_name("main_rot_clearbuf") == ("main", "_rot", "_clearbuf")
    assert MAP.split_run_name("main_iid_noisy0.01_sigma0.1_rot") == ("main", "_iid_noisy0.01_sigma0.1_rot", "")
    assert MAP.split_run_name("coin_drift") == ("coin", "_drift", "")
    assert MAP.split_run_name("main_trainrot_curl") == ("main", "_trainrot_curl", "")
    assert MAP.register_circuit_variant("baseline_rot_clearbuf")
    assert MAP.CONFIG_ORDERS["baseline"] < MAP.CONFIG_ORDERS["baseline_rot"] < MAP.CONFIG_ORDERS["baseline_clearbuf"]
