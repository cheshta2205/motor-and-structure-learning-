"""The cycle-length experiment's knobs (2026-10-07): phase subsets of the
mixed set and a named block length. See sac.CYCLE_SUBSETS."""
import pytest
import numpy as np

import sac
import sweep_ablations as SW
import task as T


@pytest.fixture(autouse=True)
def _restore_phase_set():
    yield
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_subsets_are_drawn_from_the_mixed_set_family_major():
    mixed = sac.phase_ids_for("mixed")
    assert sac.phase_ids_for("pair_rotations") == mixed[0:2]
    assert sac.phase_ids_for("pair_mixed") == [mixed[0], mixed[2]]
    assert sac.phase_ids_for("triple") == [mixed[0], mixed[2], mixed[4]]
    # the two pairs differ only in whether the second context shares a family
    pr, pm = sac.phase_ids_for("pair_rotations"), sac.phase_ids_for("pair_mixed")
    assert pr[0] == pm[0]
    assert T.FAMILY[pr[1]] == T.FAMILY[pr[0]] and T.FAMILY[pm[1]] != T.FAMILY[pm[0]]
    # one per family in the triple
    assert sorted(T.FAMILY[c] for c in sac.phase_ids_for("triple")) == list(range(len(T.FAM_NAMES)))


@pytest.mark.parametrize("phase_set, n", [("pair_rotations", 2), ("pair_mixed", 2),
                                          ("triple", 3), ("mixed", 6)])
def test_cycle_length_follows_the_subset(phase_set, n):
    sac.apply_phase_set(phase_set)
    blocks = sac.schedule_blocks("cycle", 150, 3, 0)
    assert len(blocks) == n * 3
    assert [b["occurrence"] for b in blocks] == [i // n for i in range(n * 3)]
    assert len({b["context_id"] for b in blocks}) == n


def test_run_names_carry_the_subset_and_the_block_length():
    base = SW.CONFIG_BY_NAME["baseline"]
    assert sac.apply_cli_variants(base, phase_set="pair_rotations")["name"] == "baseline_n2rot"
    assert sac.apply_cli_variants(base, phase_set="pair_mixed")["name"] == "baseline_n2mix"
    assert sac.apply_cli_variants(base, phase_set="triple")["name"] == "baseline_n3"
    cfg = sac.apply_cli_variants(base, phase_set="pair_rotations", phase_episodes=300)
    assert cfg["name"] == "baseline_n2rot_ep300" and cfg["phase_episodes"] == 300
    # the default block length is a no-op: every existing name is unchanged
    assert sac.apply_cli_variants(base, phase_episodes=sac.N_PHASE_EP)["name"] == "baseline"
    assert "phase_episodes" not in sac.apply_cli_variants(base, phase_episodes=sac.N_PHASE_EP)
    # the block length sits before the EC-prior and buffer suffixes
    main = SW.CONFIG_BY_NAME["main"]
    cfg = sac.apply_cli_variants(main, phase_set="triple", phase_episodes=300, ec_prior="carry",
                                 clear_buffer=True)
    assert cfg["name"] == "main_n3_ep300_carry_clearbuf"
    with pytest.raises(ValueError):
        sac.apply_cli_variants(base, phase_episodes=0)


def test_sweep_threads_the_block_length():
    presets = SW.presets_for(["baseline"], phase_set="pair_rotations", phase_episodes=300)
    assert [p.name for p in presets] == ["baseline_n2rot_ep300"]
    assert presets[0].phase_episodes == 300


# ── the savings probes (sac.PROBE_ROLES) ─────────────────────────────────

def test_probe_blocks_are_a_washout_then_parallel_branches():
    for phase_set in ("mixed", "pair_rotations", "pair_mixed", "triple", "pair_wide"):
        sac.apply_phase_set(phase_set)
        roles = sac.probe_roles_for_cell()
        blocks = sac.schedule_blocks("cycle", 150, 3, 0, probe="savings")
        cycle = [b for b in blocks if not b.get("washout") and b.get("branch") is None]
        assert cycle == sac.schedule_blocks("cycle", 150, 3, 0)      # the cycle is untouched
        wash = [b for b in blocks if b.get("washout")]
        probes = [b for b in blocks if b.get("branch") is not None]
        assert len(wash) == 1 and wash[0]["context_id"] is None and wash[0]["log"]
        assert wash[0]["phase_idx"] == 1 + max(b["phase_idx"] for b in cycle)
        assert blocks.index(wash[0]) == len(cycle)                    # right after the cycle
        assert [b["branch"] for b in probes] == list(roles) and len(roles) == 3   # three roles in every cell
        assert {b["phase_idx"] for b in probes} == {wash[0]["phase_idx"] + 1}   # parallel
        by_role = {b["branch"]: b for b in probes}
        first = sac.PHASE_CONTEXT_IDS[0]
        assert by_role["revisit"]["context_id"] == first
        assert by_role["revisit"]["occurrence"] == 3                  # its fourth visit
        novel_rot = [r for r in ("novel_family_member", "interior") if r in by_role]
        assert len(novel_rot) == 1                                    # one novel rotation per cell
        rot = by_role[novel_rot[0]]["context_id"]
        assert T.FAMILY[rot] == 0
        if novel_rot[0] == "novel_family_member":
            assert T.C_PARAM[rot] == -T.C_PARAM[first]
        for role in (novel_rot[0], "novel_family"):
            assert by_role[role]["context_id"] not in sac.PHASE_CONTEXT_IDS
            assert by_role[role]["occurrence"] == 0 and by_role[role]["family_idx"] is None
        assert T.FAMILY[by_role["novel_family"]["context_id"]] == T.REFLECTION_KIND
        assert all(b["learn"] and b["log"] and b["kind"] == sac.PROBE_KIND for b in probes)
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_wide_pair_probes_the_midpoint():
    """The transfer cell (user, 2026-10-08): rot -30 then rot -90 trained, the
    midpoint -60 probed as `interior` (interpolation); no mirror probe; the
    other cells have no interior (their midpoint is not a table rotation)."""
    sac.apply_phase_set("pair_wide")
    try:
        assert [T.C_LABEL[c] for c in sac.PHASE_CONTEXT_IDS] == ["rot -30", "rot -90"]
        assert sac.probe_roles_for_cell() == ("revisit", "interior", "novel_family")
        assert sac.probe_context("interior") == T.rotation_id(-60) == 0   # the table's own -60
        assert sac.probe_context("novel_family_member") is None
    finally:
        sac.apply_phase_set(sac.DEFAULT_PHASE_SET)
    for phase_set in ("pair_rotations", "pair_mixed", "triple", "mixed"):
        sac.apply_phase_set(phase_set)
        assert sac.probe_context("interior") is None
        assert sac.probe_roles_for_cell() == ("revisit", "novel_family_member", "novel_family")
    sac.apply_phase_set(sac.DEFAULT_PHASE_SET)


def test_probe_washout_is_optional():
    sac.apply_phase_set("pair_wide")
    try:
        blocks = sac.schedule_blocks("cycle", 150, 3, 0, probe="savings", probe_washout=False)
        cycle = [b for b in blocks if b.get("branch") is None]
        assert not any(b.get("washout") for b in blocks)
        probes = [b for b in blocks if b.get("branch") is not None]
        assert {b["phase_idx"] for b in probes} == {1 + max(b["phase_idx"] for b in cycle)}
    finally:
        sac.apply_phase_set(sac.DEFAULT_PHASE_SET)
    base = SW.CONFIG_BY_NAME["baseline"]
    cfg = sac.apply_cli_variants(base, phase_set="pair_wide", probe=True, probe_washout=False)
    assert cfg["name"] == "baseline_n2wide_pr_nowash" and cfg["probe_washout"] is False
    assert sac.apply_cli_variants(base, phase_set="pair_wide", probe=True)["name"] == "baseline_n2wide_pr"
    with pytest.raises(ValueError):
        sac.apply_cli_variants(base, probe_washout=False)            # needs --probe


def test_coin_washout_elicits_the_null_hypothesis():
    """The probes' washout is a context of its own for COIN (user,
    2026-10-08): the plant observer carries the null hypothesis under a probe
    run with a washout, and only then; the CB observer (Full Circuit) cannot."""
    coin = sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin"], phase_set="pair_wide", probe=True)
    assert sac.observer_has_null(sac.AblationConfig(**coin)) is True
    sac.apply_phase_set("pair_wide")
    obs = sac.make_observer(sac.AblationConfig(**coin), np.eye(2, dtype=np.float32))
    assert obs.ids == [T.rotation_id(-30), T.rotation_id(-90), None] and obs.K == 3
    plain = sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin"], phase_set="pair_wide")
    assert sac.observer_has_null(sac.AblationConfig(**plain)) is False
    nowash = sac.apply_cli_variants(SW.CONFIG_BY_NAME["coin"], phase_set="pair_wide", probe=True,
                                    probe_washout=False)
    assert sac.observer_has_null(sac.AblationConfig(**nowash)) is False
    main = sac.apply_cli_variants(SW.CONFIG_BY_NAME["main"], phase_set="pair_wide", probe=True,
                                  ec_prior="carry")
    assert sac.observer_has_null(sac.AblationConfig(**main)) is False


def test_probe_refuses_a_set_that_already_holds_a_novel_context():
    sac.apply_phase_set("rotations")                 # all four rotations: the novel same-family +60 is trained
    with pytest.raises(ValueError):
        sac.schedule_blocks("cycle", 150, 3, 0, probe="savings")
    with pytest.raises(ValueError):
        sac.schedule_blocks("spontaneous_recovery", 150, 3, 0, probe="savings")


def test_probe_names_and_refusals():
    base = SW.CONFIG_BY_NAME["baseline"]
    assert sac.apply_cli_variants(base, probe=True)["name"] == "baseline_pr"
    cfg = sac.apply_cli_variants(base, phase_set="triple", phase_episodes=300, probe=True)
    assert cfg["name"] == "baseline_n3_ep300_pr" and cfg["probe"] == "savings"
    assert sac.apply_cli_variants(SW.CONFIG_BY_NAME["main_clearbuf"], probe=True,
                                  ec_prior="carry")["name"] == "main_pr_carry_clearbuf"
    assert "probe" not in sac.apply_cli_variants(base)                # off by default
    with pytest.raises(ValueError):
        sac.apply_cli_variants(base, probe=True, schedule="savings_washout")
    assert [p.name for p in SW.presets_for(["coin"], phase_set="pair_mixed", probe=True)] == ["coin_n2mix_pr"]


def test_reflection_is_a_registered_norm_preserving_context_outside_the_table():
    import numpy as np
    c = T.reflection_id(45)
    assert c == T.reflection_id(225) and not T.is_table_context(c)   # axis modulo 180
    A, B, w = T.context_affine(c)
    assert np.allclose(A @ A.T, np.eye(2), atol=1e-6) and np.isclose(np.linalg.det(A), -1)
    assert not np.any(B) and not np.any(w)
    assert T.C_LABEL[c] == "refl 45"


def test_fork_restores_networks_buffer_and_rng():
    import copy
    import numpy as np
    import torch
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["baseline"])
    spec = sac.obs_spec(cfg)
    nets = [sac.Actor(spec), sac.SoftQNetwork(spec), sac.SoftQNetwork(spec),
            sac.SoftQNetwork(spec), sac.SoftQNetwork(spec)]
    actor, qf1, qf2, qt1, qt2 = nets
    trainer = sac.SACTrainer(actor, qf1, qf2, qt1, qt2,
                             torch.optim.Adam(actor.parameters(), lr=1e-3),
                             torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-3),
                             target_entropy=-2.0, device="cpu")
    buf = sac.ReplayBuffer(64, seed=0, units=cfg.replay_units, capacity_episodes=8)
    rng = np.random.default_rng(3)
    before = {k: copy.deepcopy(getattr(trainer, k).state_dict()) for k in ("actor", "qf1")}
    fork = sac.fork_state(trainer, None, None, None, buf, None, None, None, None,
                          sac.PRIOR_SRC_NONE, rng, 10, 5)
    draw_before = rng.random()
    # mutate everything a branch would: a gradient step, the temperature, the rng
    for p in list(actor.parameters()) + list(qf1.parameters()):
        p.data.add_(1.0)
    with torch.no_grad():
        trainer.log_alpha.add_(0.7)
    rng.random(5)
    buf2, *_rest, gstep, gep = sac.restore_fork(fork, trainer, None, None, None, rng)
    for k in before:
        for a, b in zip(getattr(trainer, k).state_dict().values(), before[k].values()):
            assert torch.equal(a, b)
    assert float(trainer.log_alpha) == 0.0 and (gstep, gep) == (10, 5)
    assert rng.random() == draw_before                               # the rng stream rewound
    assert buf2 is not buf and buf2 is not fork["objs"][0]           # a fresh copy per branch
