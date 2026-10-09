"""
Tests for the numeric core of make_ablations_plot.py -- not the plotting
calls themselves (rendering isn't meaningfully unit-testable), but the pure
functions underneath: criterion-crossing interpolation, Hill-curve fitting,
phase-occurrence counting, and config loading/ordering.
"""
import numpy as np
import pandas as pd
import pytest

import make_ablations_plot as MAP


def test_criterion_crossing_interpolates_linear_ramp():
    episodes = np.arange(10, dtype=float)
    values = episodes.copy()   # y == x, so crossing target=4.5 is exactly at x=4.5
    result = MAP.estimate_criterion_crossing_episode(episodes, values, target=4.5, smooth_window=1)
    assert result == 4.5


def test_criterion_crossing_never_reached_returns_last_episode():
    episodes = np.arange(10, dtype=float)
    values = episodes.copy()
    result = MAP.estimate_criterion_crossing_episode(episodes, values, target=100.0, smooth_window=1)
    assert result == episodes.max()


def test_criterion_crossing_too_few_points_returns_nan():
    result = MAP.estimate_criterion_crossing_episode(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), target=0.5)
    assert np.isnan(result)


def test_fit_learning_curve_recovers_sensible_hill_fit():
    x = np.linspace(0, 200, 40)
    y = MAP.hill(x, a=0.1, L=0.9, n=2.0, K=10.0)
    result = MAP.fit_learning_curve(x, y, "success")
    assert result is not None
    assert result["model"] == "hill"
    assert 0.0 <= result["a"] <= 1.0
    assert 0.0 <= result["L"] <= 1.0
    # K is preserved (up to the normalization residual) since min-max scaling
    # doesn't move the half-rise point much when y_max is close to the true L.
    assert abs(result["K"] - 10.0) < 10.0


def test_fit_learning_curve_too_few_points_returns_none():
    result = MAP.fit_learning_curve(np.array([0.0, 1.0]), np.array([0.0, 1.0]), "success")
    assert result is None


def test_add_phase_occurrence_counts_non_consecutive_repeats():
    """Phase sequence A, A, B, B, A, A, C, C -- the second A block (eps 5-6)
    must be counted as occurrence 1, not conflated with the first (eps 1-2)."""
    df = pd.DataFrame({
        "config": ["c"] * 8, "seed": [0] * 8,
        "ep": [1, 2, 3, 4, 5, 6, 7, 8],
        "phase": [0, 0, 1, 1, 0, 0, 2, 2],
        "label": ["A", "A", "B", "B", "A", "A", "C", "C"],
    })

    result = MAP.add_phase_occurrence(df)

    assert result["phase_occurrence"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    assert result["phase_label"].iloc[0] == "A (x1)"
    assert result["phase_label"].iloc[4] == "A (x2)"
    assert result["occurrence_label"].iloc[4] == "x2"


def test_add_phase_occurrence_uses_global_ep_when_present():
    """sac.py's `ep` is local per phase block (1..N_PHASE_EP), so ordering by
    it interleaves blocks and collapses the counter to all-zeros. global_ep is
    the monotonic column and must win."""
    df = pd.DataFrame({
        "config": ["c"] * 4, "seed": [0] * 4,
        "ep": [1, 1, 2, 2],                 # local: not chronological
        "global_ep": [1, 3, 2, 4],          # chronological order is rows 0,2,1,3
        "phase": [0, 1, 0, 1],
        "label": ["A", "A", "A", "A"],
    })

    result = MAP.add_phase_occurrence(df)

    # phase alternates 0,1,0,1 chronologically, so label A is revisited and the
    # counter must advance past 0 rather than staying flat.
    assert result["phase_occurrence"].max() > 0


def test_build_phase_occurrence_label_keys_each_phase_visit():
    """A phase visit is one context's Nth block, so the key must split on label
    AND occurrence -- otherwise different force fields get merged into one
    learning curve."""
    df = pd.DataFrame({
        "label": ["C1: rot +45", "C4: refl 0", "C1: rot +45", "C4: refl 0"],
        "occurrence": [0, 0, 1, 1],
    })

    result = MAP.build_phase_occurrence_label(df)

    assert result["phase_occurrence"].tolist() == [
        "C1: rot +45 (x1)", "C4: refl 0 (x1)",
        "C1: rot +45 (x2)", "C4: refl 0 (x2)",
    ]
    assert result["phase_occurrence"].nunique() == 4


def test_build_phase_occurrence_label_requires_an_occurrence_column():
    with pytest.raises(KeyError):
        MAP.build_phase_occurrence_label(pd.DataFrame({"label": ["A"], "ep": [1]}))


def test_load_all_configs_concatenates_and_tags(tmp_path):
    for name in ("baseline", "oracle"):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"seed": [0, 1], "ep": [1, 1], "success": [0.0, 1.0]}).to_csv(
            d / "aggregated_results.csv", index=False)

    data = MAP.load_all_configs(str(tmp_path))
    assert data is not None
    assert set(data["config"].unique()) == {"baseline", "oracle"}
    assert len(data) == 4


def test_load_all_configs_ignores_unknown_directory_names(tmp_path):
    d = tmp_path / "not_a_real_config"
    d.mkdir()
    pd.DataFrame({"seed": [0], "ep": [1], "success": [1.0]}).to_csv(
        d / "aggregated_results.csv", index=False)

    data = MAP.load_all_configs(str(tmp_path))
    assert data is None


def test_testable_pairs_drops_untestable_comparisons():
    """A paired Wilcoxon needs >=2 seeds whose values differ: n=1 per config
    (a single-seed sweep) and two configs that saturate identically both leave
    scipy with nothing to test."""
    facet = pd.DataFrame({
        "occurrence": [0] * 8,
        "config": (["baseline"] * 4 + ["oracle"] * 2 + ["end_to_end"] * 2),
        "seed": [0, 1, 2, 3, 0, 1, 0, 1],
        # oracle differs from baseline on both shared seeds -> testable
        # end_to_end is identical to baseline on both shared seeds -> not
        "criterion_episode": [10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 10.0, 20.0],
    })
    pairs = [
        ((0, "baseline"), (0, "oracle")),
        ((0, "baseline"), (0, "end_to_end")),
    ]

    keep = MAP.testable_pairs(facet, pairs)

    assert keep == [((0, "baseline"), (0, "oracle"))]


def test_testable_pairs_drops_single_seed_comparisons():
    facet = pd.DataFrame({
        "occurrence": [0, 0],
        "config": ["baseline", "oracle"],
        "seed": [0, 0],
        "criterion_episode": [11.8, 6.8],
    })
    pairs = [((0, "baseline"), (0, "oracle"))]

    assert MAP.testable_pairs(facet, pairs) == []


def test_smooth_trace_matches_sac_figures_alignment():
    """The offset must place each smoothed point at the LAST episode in its
    window, the way sac_figures.plot_seed draws it (ep[-len(y):]). Getting this
    wrong shifts every trace left by the window width and makes the criterion
    markers look closer to the crossing than they are."""
    values = np.arange(10, dtype=float)
    offset, smoothed = MAP.smooth_trace(values, window=4)

    assert offset == 3
    assert len(smoothed) == 7
    assert smoothed[0] == pytest.approx(1.5)   # mean of episodes 0..3


def test_smooth_trace_passes_short_traces_through():
    values = np.arange(3, dtype=float)
    offset, smoothed = MAP.smooth_trace(values, window=25)
    assert offset == 0
    assert np.array_equal(smoothed, values)


def test_median_seed_band_smooths_per_seed_before_aggregating():
    """Order of operations: two seeds that are each constant but differ from
    each other must produce a band whose width is the seed difference, not the
    episode noise. Aggregating before smoothing would blur them together."""
    n = 100
    df = pd.DataFrame({
        "seed": [0] * n + [1] * n,
        "ep": list(range(1, n + 1)) * 2,
        "global_ep": list(range(1, n + 1)) * 2,
        "return": [0.0] * n + [10.0] * n,
    })

    band = MAP.median_seed_band(df, "return", "global_ep", smooth_window=5)

    assert band["median"].iloc[0] == pytest.approx(5.0)
    assert band["std"].iloc[0] == pytest.approx(np.std([0.0, 10.0], ddof=1))


def test_median_seed_band_is_robust_to_one_straggler_seed():
    """The reason this is a median: two seeds that adapt and one that never
    does must leave the band centered on the adapters, where a mean would sit
    a third of the way down to the straggler."""
    n = 100
    df = pd.DataFrame({
        "seed": [0] * n + [1] * n + [2] * n,
        "ep": list(range(1, n + 1)) * 3,
        "global_ep": list(range(1, n + 1)) * 3,
        "return": [10.0] * n + [10.0] * n + [-20.0] * n,
    })

    band = MAP.median_seed_band(df, "return", "global_ep", smooth_window=5)

    assert band["median"].iloc[0] == pytest.approx(10.0)


def test_median_seed_band_gives_a_single_seed_zero_width():
    n = 50
    df = pd.DataFrame({
        "seed": [0] * n, "ep": range(1, n + 1), "global_ep": range(1, n + 1),
        "return": np.linspace(0.0, 1.0, n),
    })

    band = MAP.median_seed_band(df, "return", "global_ep", smooth_window=5)

    assert (band["std"] == 0.0).all()      # ddof=1 would be NaN and draw nothing
    assert not band["median"].isna().any()


def test_median_seed_band_aligns_seeds_of_unequal_length():
    """A short run must contribute NaN past its end, not shift the others left."""
    df = pd.DataFrame({
        "seed": [0] * 60 + [1] * 30,
        "ep": list(range(1, 61)) + list(range(1, 31)),
        "global_ep": list(range(1, 61)) + list(range(1, 31)),
        "return": [4.0] * 60 + [6.0] * 30,
    })

    band = MAP.median_seed_band(df, "return", "global_ep", smooth_window=5)

    assert band.index.max() == 60
    assert band.loc[20, "median"] == pytest.approx(5.0)   # both seeds present
    assert band.loc[60, "median"] == pytest.approx(4.0)   # only seed 0 remains
    assert band.loc[60, "std"] == pytest.approx(0.0)


def test_phase_blocks_splits_repeat_visits_to_same_context():
    df = pd.DataFrame({
        "phase": [0, 0, 1, 1],
        "global_ep": [1, 2, 3, 4],
        "label": ["rot +45", "rot +45", "rot +45", "rot +45"],
        "occurrence": [0, 0, 1, 1],
    })

    blocks = MAP.phase_blocks(df)

    assert blocks["phase_occurrence"].tolist() == ["rot +45 (x1)", "rot +45 (x2)"]
    assert blocks["start"].tolist() == [1, 3]
    assert blocks["end"].tolist() == [2, 4]


def _bimodal_ramp_block(n=400, planted=slice(20, 25), seed=0):
    """A block shaped like `return`: two modes (+10 on success, -2 on failure)
    with the hit rate ramping 0 -> 1, plus a planted early run of 5 straight
    successes. The 5-episode median used to clear 0.8 * 10 = 8 on the planted
    run; the 25-episode mean does not cross until late."""
    rng = np.random.default_rng(seed)
    values = np.where(rng.random(n) < np.linspace(0.0, 1.0, n), 10.0, -2.0)
    values[planted] = 10.0
    return pd.DataFrame({
        "config": ["oracle"] * n, "seed": [0] * n,
        "ep": np.arange(1, n + 1), "global_ep": np.arange(1, n + 1),
        "phase": [0] * n, "occurrence": [0] * n, "label": ["rot +45"] * n,
        "phase_occurrence": ["rot +45 (x1)"] * n,
        "return": values,
    })


def test_criterion_crossing_ignores_a_short_lucky_run():
    """The regression this file exists to pin: a 5-in-a-row run of successes at
    episode 20 must NOT count as reaching criterion when the policy is still
    failing most episodes. The old 5-episode rolling median reported ~20 here."""
    block = _bimodal_ramp_block()
    values = block["return"].to_numpy()

    crossing = MAP.estimate_criterion_crossing_episode(
        np.arange(len(values), dtype=float), values, target=8.0)

    assert crossing > 200


def test_criterion_crossing_not_reported_before_the_smoothing_window():
    """A 25-episode mean does not exist until episode 25, so a block that is at
    criterion from its first episode reports the window edge, not episode 0."""
    values = np.full(400, 10.0)

    crossing = MAP.estimate_criterion_crossing_episode(
        np.arange(len(values), dtype=float), values, target=8.0)

    assert crossing == MAP.TRACE_SMOOTH_WINDOW - 1


def test_criterion_crossing_at_criterion_from_the_start_but_drifting_down_is_not_censored():
    """The regression of 2026-10-07: a block above criterion from its first
    episode that ends a little lower than it began (the savings probes'
    revisit, 1.0 -> 0.9) must report the window edge like any other block that
    starts above criterion -- not the block end. The old endpoint heuristic
    took it to be approaching the criterion from above."""
    values = np.concatenate([np.full(200, 1.0), np.full(200, 0.9)])
    crossing = MAP.estimate_criterion_crossing_episode(
        np.arange(len(values), dtype=float), values, target=0.8)
    assert crossing == MAP.TRACE_SMOOTH_WINDOW - 1


def test_criterion_trace_consistency_is_tight_on_a_bimodal_block():
    """Same data as the lucky-run test, checked from the other side: the
    criterion episode and the plotted-trace crossing must now be the same
    episode, up to the estimator's sub-episode interpolation."""
    table = MAP.build_criterion_trace_consistency_table(_bimodal_ramp_block(), ["return"])
    row = table.iloc[0]

    assert row["criterion_target"] == pytest.approx(8.0)
    assert 0.0 <= row["gap"] <= 1.0


def test_criterion_trace_consistency_agrees_on_a_clean_step():
    """Control: when the metric is not noisy the two estimates must agree to
    within the smoothing window, or the table would flag gaps everywhere."""
    n = 400
    values = np.where(np.arange(n) < 200, 0.0, 10.0)

    df = pd.DataFrame({
        "config": ["oracle"] * n, "seed": [0] * n,
        "ep": np.arange(1, n + 1), "global_ep": np.arange(1, n + 1),
        "phase": [0] * n, "occurrence": [0] * n, "label": ["rot +45"] * n,
        "phase_occurrence": ["rot +45 (x1)"] * n,
        "return": values,
    })

    table = MAP.build_criterion_trace_consistency_table(df, ["return"])
    row = table.iloc[0]

    assert 0.0 <= row["gap"] <= 1.0
    assert row["frac_above"] > 0.4


def _clean_step_block(config, high, n=400):
    """A block that steps from 0 to `high` half way through, under one run."""
    return pd.DataFrame({
        "config": [config] * n, "seed": [0] * n,
        "ep": np.arange(1, n + 1), "global_ep": np.arange(1, n + 1),
        "phase": [0] * n, "occurrence": [0] * n, "label": ["rot +45"] * n,
        "phase_occurrence": ["rot +45 (x1)"] * n,
        "return": np.where(np.arange(n) < 200, 0.0, float(high)),
    })


def test_cycle_only_keeps_variants_and_drops_the_paradigms():
    df = pd.DataFrame({"config": ["main", "main_clearbuf", "main_iid_noisy0.001_sigma0.1",
                                  "baseline_rot_sr", "main_er", "main_rot_savw_carry"]})

    assert MAP.cycle_only(df)["config"].tolist() == [
        "main", "main_clearbuf", "main_iid_noisy0.001_sigma0.1"]


def test_criterion_analysis_ignores_paradigm_runs(tmp_path):
    """A COIN-paradigm run must not reach the criterion figures at all, and --
    the quieter failure -- must not drag the global target, which is a
    fraction of the best value SEEN across the frame."""
    df = pd.concat([_clean_step_block("main", 10.0),
                    _clean_step_block("main_savw", 100.0)], ignore_index=True)

    table = MAP.report_criterion_trace_consistency(df, ["return"], str(tmp_path))

    assert table["config"].unique().tolist() == ["main"]
    assert table["criterion_target"].unique() == pytest.approx([8.0])


def test_get_config_order_filters_and_orders():
    df = pd.DataFrame({"config": ["oracle", "baseline", "oracle"]})
    order = MAP.get_config_order(df)
    assert order == ["baseline", "oracle"]


def test_get_shared_seeds_intersects_across_configs():
    df = pd.DataFrame({
        "config": ["baseline", "baseline", "oracle", "oracle"],
        "seed": [0, 1, 0, 2],
    })
    shared = MAP.get_shared_seeds(df, ["baseline", "oracle"])
    assert shared == {0}


# ── circuit variants: derived names registered when their runs are found ──────

def test_display_name_follows_the_base_preset_and_marks_the_savings_twin():
    """The figures name presets by anatomy (user, 2026-09-24); derived runs
    inherit the parent's name, only the cleared-buffer twin is marked, and
    an unlisted preset keeps its run name."""
    assert MAP.display_name("main") == "Full Circuit"
    assert MAP.display_name("main_carry") == "Full Circuit"
    assert MAP.display_name("main_iid_noisy0.001_carry") == "Full Circuit"
    assert MAP.display_name("main_carry_clearbuf") == "Full Circuit (buffer cleared)"
    assert MAP.display_name("baseline") == "Motor Cortex"
    assert MAP.display_name("cb_only_end_to_end") == "Motor Cortex + Cerebellum"
    assert MAP.display_name("cb_only_end_to_end", wrap=True) == "Motor Cortex\n+ Cerebellum"
    assert MAP.display_name("main", wrap=True) == "Full Circuit"          # nothing to break
    assert MAP.display_name("coin") == "COIN"
    assert MAP.display_name("oracle_geom") == "oracle_geom"
    assert set(MAP.DISPLAY_NAME) <= set(MAP.CONFIG_COLORS)


def test_split_run_name_separates_base_circuit_and_buffer():
    assert MAP.split_run_name("main") == ("main", "", "")
    assert MAP.split_run_name("main_clearbuf") == ("main", "", "_clearbuf")
    assert MAP.split_run_name("main_iid") == ("main", "_iid", "")
    assert MAP.split_run_name("main_ar1_epbuf_clearbuf") == ("main", "_ar1", "_epbuf_clearbuf")
    assert MAP.split_run_name("main_iid_noisy0.001_sigma0.1_clearbuf") == (
        "main", "_iid_noisy0.001_sigma0.1", "_clearbuf")
    assert MAP.split_run_name("main_noisy1e-05") == ("main", "_noisy1e-05", "")
    # a preset whose own name contains an underscore is not mistaken for a variant
    assert MAP.split_run_name("cb_only_end_to_end_clearbuf") == ("cb_only_end_to_end", "", "_clearbuf")
    assert MAP.base_config("main_iid_noisy0.001_sigma0.1_clearbuf") == "main"
    # replay-off is the *_clearbuf twin, under every schedule
    assert MAP.split_run_name("main_rot_sr_carry_clearbuf") == ("main", "_rot_sr_carry", "_clearbuf")
    assert MAP.base_config("coin_rot_savw_cprior_clearbuf") == "coin"
    assert MAP._replay_off("main_clearbuf") and not MAP._replay_off("main")
    assert MAP._replay_off("coin_rot_sr_cprior_clearbuf") and not MAP._replay_off("coin_rot_sr_cprior")
    assert MAP._model_label("coin_rot_sr_cprior") == "COIN · replay"      # display name since 2026-10-06
    assert MAP._model_label("coin_rot_sr_cprior_clearbuf") == "COIN · no replay"
    # the cycle-length experiment's task infixes (2026-10-07) parse like the others
    assert MAP.split_run_name("main_n2rot_ep300_carry") == ("main", "_n2rot_ep300_carry", "")
    assert MAP.base_config("baseline_n3") == "baseline"
    assert MAP.display_name("coin_n2mix_ep300") == "COIN"


def test_cycle_cell_reads_n_composition_and_block_length_off_the_name():
    import sac
    assert MAP.cycle_cell("main_carry") == (6, "mixed", sac.N_PHASE_EP)
    assert MAP.cycle_cell("baseline_n2rot_pr") == (2, "pair_rotations", sac.N_PHASE_EP)
    assert MAP.cycle_cell("coin_n2mix") == (2, "pair_mixed", sac.N_PHASE_EP)
    assert MAP.cycle_cell("main_n3_ep300_pr_carry_clearbuf") == (3, "triple", 300)
    assert MAP.cell_tag("main_carry") == "n6"
    assert MAP.cell_tag("baseline_n2rot_pr") == "n2rot"
    assert MAP.cell_tag("main_n2rot_ep300_pr_carry") == "n2rot_ep300"
    # _n2rot must not be mistaken for the rotations-family set (_rot)
    assert MAP.cycle_cell("baseline_rot_sr")[1] == "rotations"


def test_register_circuit_variant_derives_slot_shade_and_twins(monkeypatch):
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    name = "main_iid_noisy0.001_sigma0.1_clearbuf"
    assert name not in MAP.CONFIG_ORDERS
    assert MAP.register_circuit_variant(name) is True
    level = "main_iid_noisy0.001_sigma0.1"
    for n in (level, level + "_clearbuf", level + "_epbuf", level + "_epbuf_clearbuf"):
        assert n in MAP.CONFIG_ORDERS and n in MAP.CONFIG_COLORS
    # between the parent and the parent's buffer twins, and before the next base
    assert MAP.CONFIG_ORDERS["main"] < MAP.CONFIG_ORDERS[level] < MAP.CONFIG_ORDERS["main_clearbuf"]
    assert MAP.CONFIG_ORDERS[level + "_clearbuf"] < MAP.CONFIG_ORDERS["control"]
    # its own shade: darker than the parent, not the washed-out twin colour
    assert MAP.CONFIG_COLORS[level] not in (MAP.CONFIG_COLORS["main"], MAP.CONFIG_COLORS["main_clearbuf"])
    assert MAP.CONFIG_COLORS[level + "_clearbuf"] == MAP._lighten(MAP.CONFIG_COLORS[level], 0.55)
    # a second variant of the same base steps further out and stays distinct
    assert MAP.register_circuit_variant("main_iid")
    assert MAP.CONFIG_ORDERS[level] < MAP.CONFIG_ORDERS["main_iid"] < MAP.CONFIG_ORDERS["main_clearbuf"]
    assert MAP.CONFIG_COLORS["main_iid"] != MAP.CONFIG_COLORS[level]
    # idempotent, and visible to the dynamic order
    assert MAP.register_circuit_variant(name) is True
    order = MAP.config_order()
    assert order.index("main") < order.index(level) < order.index("main_clearbuf")


def test_register_circuit_variant_refuses_unknown_bases_and_plain_names():
    before = dict(MAP.CONFIG_ORDERS)
    assert MAP.register_circuit_variant("nonsense_iid_noisy0.001") is False
    assert MAP.register_circuit_variant("not_a_real_config") is False
    assert MAP.register_circuit_variant("main") is True          # already registered
    assert MAP.CONFIG_ORDERS == before


def test_load_all_configs_admits_circuit_variants_of_known_presets(tmp_path, monkeypatch):
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for name in ("main", "main_iid_noisy0.001_sigma0.1_clearbuf", "bogus_iid"):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"seed": [0], "ep": [1], "success": [1.0]}).to_csv(
            d / "aggregated_results.csv", index=False)
    data = MAP.load_all_configs(str(tmp_path))
    assert set(data["config"]) == {"main", "main_iid_noisy0.001_sigma0.1_clearbuf"}
    assert MAP.get_config_order(data) == ["main", "main_iid_noisy0.001_sigma0.1_clearbuf"]
    assert "main_iid_noisy0.001_sigma0.1_clearbuf" in MAP.CONFIG_COLORS


# ── the COIN-paradigm figure ─────────────────────────────────────────────

def test_adaptation_index_sign_and_scale():
    import sac, task as T
    P = T.C_PARAM[sac.coin_pair()[0]]
    assert MAP.adaptation_index([-P, 0.0, P]).tolist() == [1.0, 0.0, -1.0]


def test_schedule_figure_renders_from_curve_columns_alone(tmp_path, monkeypatch):
    """No probes on disk -> panel D is empty but the figure still renders,
    with the belief trace where the column exists and without it where not."""
    import sac
    rng = np.random.default_rng(0)
    blocks = [("P0", 6), ("P+ rot +30", 8), ("P- rot -30", 5), (sac.CHANNEL_LABEL, 12)]
    rows = []
    for cfg, has_belief in (("baseline_rot_sr", False), ("coin_rot_sr_cprior", True),
                            ("coin_rot_sr_cprior_clearbuf", True)):
        for seed in (0, 1):
            for ph, (lab, n) in enumerate(blocks):
                for ep in range(1, n + 1):
                    rows.append(dict(config=cfg, seed=seed, phase=ph, ep=ep, label=lab,
                                     channel=float(lab == sac.CHANNEL_LABEL),
                                     first_angle=rng.normal(0, 10), prior_src=-2,
                                     belief_pplus=(0.6 if has_belief else np.nan),
                                     belief_pminus=(0.3 if has_belief else np.nan)))
    data = pd.DataFrame(rows)
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for c in data.config.unique():
        MAP.register_circuit_variant(c)
    MAP.plot_schedule_timecourse(data, str(tmp_path), run_dir=str(tmp_path / "no_runs"))
    assert (tmp_path / "schedule_timecourse_sr.png").exists()
    # a frame without the paradigm labels is a no-op, not an error
    MAP.plot_schedule_timecourse(pd.DataFrame({"config": ["main"], "label": ["rot -60"], "channel": [0.0]}),
                                 str(tmp_path / "none"))
    assert not (tmp_path / "none" / "schedule_timecourse_sr.png").exists()


def test_schedule_figure_reads_the_probe_where_a_run_logged_it(tmp_path, monkeypatch):
    """A run with the per-reach readout probe is read from probe_first_angle
    and its panel-D bars from the probe's per-source means; a run without it
    falls back to the training reach's first action and its own draws."""
    import sac
    rng = np.random.default_rng(1)
    pp, pm = sac.coin_pair()
    blocks = [("P0", 4), ("P+ rot +30", 6), ("P- rot -30", 4), (sac.CHANNEL_LABEL, 10)]
    rows = []
    for cfg, probed in (("main_rot_sr_carry", True), ("main_rot_sr_carry_clearbuf", False)):
        for seed in (0, 1):
            for ph, (lab, n) in enumerate(blocks):
                for ep in range(1, n + 1):
                    ch = lab == sac.CHANNEL_LABEL
                    rows.append(dict(
                        config=cfg, seed=seed, phase=ph, ep=ep, label=lab, channel=float(ch),
                        first_angle=rng.normal(0, 10), prior_src=int(rng.choice([pp, pm])),
                        # index = -fa / 60 on the P0 schedules, so -60 deg is
                        # full P+ compensation and +60 is full P-
                        probe_first_angle=(-60.0 if probed else np.nan),
                        probe_fa_pplus=(-60.0 if probed and ch else np.nan),
                        probe_fa_pminus=(60.0 if probed and ch else np.nan)))
    data = pd.DataFrame(rows)
    d1 = data[data.config == "main_rot_sr_carry"]; d2 = data[data.config == "main_rot_sr_carry_clearbuf"]
    assert MAP._readout_column(d1) == "probe_first_angle"
    assert MAP._readout_column(d2) == "first_angle"
    src = MAP._channel_by_prior_source(d1)
    assert src["P+"] == [1.0, 1.0] and src["P-"] == [-1.0, -1.0]
    assert MAP._channel_by_prior_source(d2) is not None      # from the reaches' own draws
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for c in data.config.unique():
        MAP.register_circuit_variant(c)
    MAP.plot_schedule_timecourse(data, str(tmp_path), run_dir=str(tmp_path / "no_runs"))
    assert (tmp_path / "schedule_timecourse_sr.png").exists()


def _savings_rows(cfg, probed, rng, pp):
    """A synthetic savings-washout run: base | P+ | base | P+ (rot -60 / +60),
    two seeds, with the readout-probe columns filled (P+2 above P+1) when
    `probed`."""
    blocks = [("base rot -60", 0, 4), ("P+ rot +60", 0, 8), ("base rot -60", 1, 5), ("P+ rot +60", 1, 8)]
    rows = []
    for seed in (0, 1):
        for ph, (lab, occ, n) in enumerate(blocks):
            for ep in range(1, n + 1):
                second = lab.startswith("P+") and occ == 1
                rows.append(dict(
                    config=cfg, seed=seed, phase=ph, occurrence=occ, ep=ep, label=lab, channel=0.0,
                    first_angle=rng.normal(-30 if second else 30, 10),
                    prior_src=int(rng.choice([pp, -1])),
                    # index = (-fa - base) / (pert - base) = (-fa + 60) / 120,
                    # so -36 deg is 0.8 and +36 deg is 0.2
                    probe_first_angle=((-36.0 if second else 36.0) if probed else np.nan),
                    probe_fa_pplus=(-60.0 if probed and second else np.nan),
                    probe_fa_pminus=np.nan,
                    probe_fa_base=(60.0 if probed and second else np.nan)))
    return rows


def test_schedules_get_separate_figures(tmp_path, monkeypatch):
    """A savings-washout run and a spontaneous-recovery run in one frame land
    in different figures, keyed on the run name's schedule suffix -- both
    have P0 blocks, so the label alone cannot separate them."""
    import sac
    rng = np.random.default_rng(2)
    pp, pm = sac.coin_pair()
    rows = _savings_rows("baseline_rot_savw", False, rng, pp) + _savings_rows("main_rot_savw_carry", True, rng, pp)
    sr_blocks = [("P0", 4), ("P+ rot +30", 6), ("P- rot -30", 4), (sac.CHANNEL_LABEL, 10)]
    for seed in (0, 1):
        for ph, (lab, n) in enumerate(sr_blocks):
            for ep in range(1, n + 1):
                rows.append(dict(config="baseline_rot_sr", seed=seed, phase=ph, occurrence=0, ep=ep,
                                 label=lab, channel=float(lab == sac.CHANNEL_LABEL),
                                 first_angle=rng.normal(0, 10), prior_src=-2,
                                 probe_first_angle=np.nan, probe_fa_pplus=np.nan,
                                 probe_fa_pminus=np.nan, probe_fa_base=np.nan))
    data = pd.DataFrame(rows)
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for c in data.config.unique():
        MAP.register_circuit_variant(c)
    assert MAP._schedule_of("main_rot_savw_carry") == "savings_washout"
    assert MAP._schedule_of("baseline_rot_sr") == "spontaneous_recovery"
    assert MAP._schedule_of("main_clearbuf") == "cycle"
    assert MAP._paradigm_runs(data, "savings_washout") == ["baseline_rot_savw", "main_rot_savw_carry"]
    assert MAP._paradigm_runs(data, "spontaneous_recovery") == ["baseline_rot_sr"]
    MAP.plot_schedule_timecourse(data, str(tmp_path), run_dir=str(tmp_path / "no_runs"))
    MAP.plot_savings_washout(data, str(tmp_path))
    assert (tmp_path / "schedule_timecourse_sr.png").exists()
    assert (tmp_path / "schedule_timecourse_savw.png").exists()
    # the savings frame alone renders no channel figure, and vice versa
    MAP.plot_schedule_timecourse(data[data.config != "baseline_rot_sr"], str(tmp_path / "only_savw"))
    assert not (tmp_path / "only_savw").exists()
    MAP.plot_savings_washout(data[data.config == "baseline_rot_sr"], str(tmp_path / "only_sr"))
    assert not (tmp_path / "only_sr").exists()
    # the savings index is relative to the TRAINED baseline: base rot -60,
    # P+ rot +60, so compensating -60 (first angle +60) is 0, compensating
    # +60 (first angle -60) is 1, and aiming straight is 0.5
    assert MAP.adaptation_index([60.0, -60.0, 0.0], "savings_washout").tolist() == [0.0, 1.0, 0.5]
    assert MAP._reference_labels("savings_washout")[0.5] == "straight aim"
    assert MAP.adaptation_index([-60.0, 0.0]).tolist() == [1.0, 0.0]          # the P0 schedules
    # the savings metric is the JUMP at re-exposure: first 5 P+2 reaches minus
    # the last 5 washout reaches. Planted: everything before P+2 at +36 deg
    # (index 0.2) and P+2 at -36 (0.8), so jump2 is +0.6 and jump1 is 0
    d = data[data.config == "main_rot_savw_carry"].copy()
    d["index"] = MAP.adaptation_index(d.probe_first_angle, "savings_washout")
    label, occs = MAP._pplus_visits(d)
    assert label == "P+ rot +60" and occs == [0, 1] and MAP._base_label(d) == "base rot -60"
    jump2, jump1 = MAP._reexposure_jump(d, 5)
    assert np.allclose(jump2.to_numpy(), 0.6) and np.allclose(jump1.to_numpy(), 0.0)
    late = MAP._late_block(d, "base rot -60", 1, 5)
    assert (late.groupby("seed").size() == 5).all() and (late.ep >= 1).all()
    around = MAP._around_switch(d, "base rot -60", label, 1, 5, 8)
    ks = sorted(around[around.seed == 0].k.tolist())
    assert ks == list(range(-4, 9))                              # 5 before (k <= 0), 8 after
    assert np.allclose(around[around.k <= 0]["index"], 0.2) and np.allclose(around[around.k > 0]["index"], 0.8)
    assert MAP._reexposure_jump(d[d.occurrence == 0], 5) == (None, None)


def _evoked_rows(cfg, pre_index, post_index):
    """A synthetic evoked-recovery run: P0 | P+ | P- | channel | P+ (one
    reach) | channel, two seeds, with the channel index planted -- `pre_index`
    over the pre-evoke block and `post_index` (a per-reach sequence) over the
    post-evoke one."""
    import sac
    blocks = [("P0", 0, 4), ("P+ rot +60", 0, 6), ("P- rot -60", 0, 4),
              (sac.CHANNEL_LABEL, 0, 5), ("P+ rot +60", 1, 1),
              (sac.CHANNEL_LABEL, 1, len(post_index))]
    rows = []
    for seed in (0, 1):
        for ph, (lab, occ, n) in enumerate(blocks):
            for ep in range(1, n + 1):
                ch = lab == sac.CHANNEL_LABEL
                index = (pre_index if occ == 0 else post_index[ep - 1]) if ch else 0.0
                rows.append(dict(config=cfg, seed=seed, phase=ph, occurrence=occ, ep=ep,
                                 label=lab, channel=float(ch), index=index,
                                 # index = -first_angle / 60 on the P0 schedules
                                 first_angle=-60.0 * index, prior_src=-2,
                                 probe_first_angle=np.nan, probe_fa_pplus=np.nan,
                                 probe_fa_pminus=np.nan, probe_fa_base=np.nan))
    return pd.DataFrame(rows)


def test_evoke_segments_split_the_two_channel_blocks():
    """The two channel blocks share a label and are told apart by occurrence;
    k is counted from the evoke, so the pre-evoke block ends at 0 and the
    post-evoke one starts at 1."""
    d = _evoked_rows("main_rot_er_carry", 0.0, [1.0] * 8)

    pre, post = MAP._evoke_segments(d)

    assert pre.k[pre.seed == 0].tolist() == [-4, -3, -2, -1, 0]
    assert post.k[post.seed == 0].tolist() == list(range(1, 9))
    # a run with one channel block (spontaneous recovery) is not evoked
    assert MAP._evoke_segments(d[d.occurrence == 0]) is None


def test_evoked_stats_measure_the_jump_and_the_decay():
    """The two signatures of evoked recovery (user, 2026-09-20): the index
    over the first post-evoke channel reaches is HIGHER than over the
    pre-evoke ones, and it then decays. Planted: pre-evoke at -0.5, the first
    4 post-evoke reaches at +0.5, the last 4 at 0.0."""
    d = _evoked_rows("main_rot_er_carry", -0.5, [0.5] * 4 + [0.0] * 4)

    stats = MAP.evoked_recovery_stats(d, evoke_window=4, close_reaches=4)

    assert np.allclose(stats["jump"], 1.0)      # +0.5 against -0.5
    assert np.allclose(stats["decay"], -0.5)    # 0.0 against +0.5
    assert len(stats["jump"]) == 2              # one value per seed


def test_evoked_stats_do_not_average_across_the_evoke():
    """The regression: pooling the two channel blocks into one series and
    taking an opening window long enough to span the evoke reports the step
    as part of its own baseline. Here the whole pre-evoke block sits at -1
    and the whole post-evoke block at +1, so the jump is exactly 2 -- an
    opening window over the pooled reaches would report less."""
    d = _evoked_rows("main_rot_er_carry", -1.0, [1.0] * 12)

    stats = MAP.evoked_recovery_stats(d, evoke_window=10, close_reaches=6)

    assert np.allclose(stats["jump"], 2.0)
    assert np.allclose(stats["decay"], 0.0)     # flat: nothing fades
    assert MAP.evoked_recovery_stats(d[d.occurrence == 0]) is None


def _spontaneous_rows(cfg, channel_index, n_channel):
    """The spontaneous twin of _evoked_rows: the same P0 | P+ | P- prefix and
    a channel block of the same TOTAL length, with no evoke in it."""
    import sac
    blocks = [("P0", 0, 4), ("P+ rot +60", 0, 6), ("P- rot -60", 0, 4),
              (sac.CHANNEL_LABEL, 0, n_channel)]
    rows = []
    for seed in (0, 1):
        for ph, (lab, occ, n) in enumerate(blocks):
            for ep in range(1, n + 1):
                ch = lab == sac.CHANNEL_LABEL
                index = channel_index if ch else 0.0
                rows.append(dict(config=cfg, seed=seed, phase=ph, occurrence=occ, ep=ep,
                                 label=lab, channel=float(ch), index=index,
                                 first_angle=-60.0 * index, prior_src=-2,
                                 probe_first_angle=np.nan, probe_fa_pplus=np.nan,
                                 probe_fa_pminus=np.nan, probe_fa_base=np.nan))
    return pd.DataFrame(rows)


def test_sr_er_pairs_match_the_same_model_across_schedules():
    """The pair key is the run name with the schedule token dropped, so a
    model's spontaneous and evoked runs pair up while its buffer twin and a
    model that ran only one schedule do not get mixed in."""
    df = pd.DataFrame({"config": ["main_rot_sr_carry", "main_rot_er_carry",
                                  "baseline_rot_sr", "coin_rot_er",
                                  "main_rot_savw_carry", "main"]})

    assert MAP._pair_key("main_rot_sr_carry") == MAP._pair_key("main_rot_er_carry")
    assert MAP._pair_key("main_rot_sr_carry") == "main_rot_carry"
    # only the model with BOTH schedules pairs; sr-only, er-only, savings and
    # cycle runs are left out
    assert MAP.sr_er_pairs(df) == [("main_rot_carry", "main_rot_sr_carry",
                                    "main_rot_er_carry")]


def test_recovery_divergence_is_zero_before_the_evoke():
    """The figure's own control. Both schedules are the same run up to the
    evoke, so the paired difference there must be exactly zero; planted, the
    evoked run then sits 0.8 above the spontaneous one early and 0.2 late."""
    n_pre, n_post = 5, 12
    sr = MAP._channel_trace(_spontaneous_rows("main_rot_sr_carry", 0.0, n_pre + n_post),
                            "main_rot_sr_carry", 3)
    er = MAP._channel_trace(_evoked_rows("main_rot_er_carry", 0.0,
                                         [0.8] * 4 + [0.2] * (n_post - 4)),
                            "main_rot_er_carry", 3)
    # the two schedules must be comparable reach for reach
    assert sr.k.max() == er.k.max() == n_pre + n_post

    both, windows = MAP.recovery_divergence(sr, er, n_pre, evoke_window=4, close_reaches=8)

    assert np.allclose(windows["pre"], 0.0)
    assert np.allclose(windows["early"], 0.8)
    assert np.allclose(windows["late"], 0.2)
    assert len(windows["early"]) == 2                      # one value per seed


def test_recovery_comparison_figure_renders(tmp_path, monkeypatch):
    data = pd.concat([
        _spontaneous_rows("main_rot_sr_carry", -0.5, 17),
        _evoked_rows("main_rot_er_carry", -0.5, [0.9] * 6 + [0.3] * 6),
        _spontaneous_rows("baseline_rot_sr", -0.9, 17),
        _evoked_rows("baseline_rot_er", -0.9, [0.1] * 12),
    ], ignore_index=True)
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for c in data.config.unique():
        MAP.register_circuit_variant(c)

    MAP.plot_recovery_comparison(data, str(tmp_path))
    assert (tmp_path / "recovery_sr_vs_er.png").exists()
    # a frame with only one of the two schedules renders nothing
    MAP.plot_recovery_comparison(data[data.config.str.contains("_sr")], str(tmp_path / "one"))
    assert not (tmp_path / "one").exists()


def _belief_rows(cfg, n=24, **columns):
    """A two-seed run with a channel block and whatever belief columns the
    test wants to plant."""
    import sac
    rows = []
    for seed in (0, 1):
        for ep in range(n):
            ch = ep >= n // 2
            row = dict(config=cfg, seed=seed, phase=int(ch), occurrence=0, ep=ep + 1,
                       label=sac.CHANNEL_LABEL if ch else "P+ rot +60",
                       channel=float(ch), prior_src=-2)
            row.update({k: (v[ep] if isinstance(v, list) else v)
                        for k, v in columns.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def test_belief_frame_prefers_the_observer_then_the_bank_then_z():
    """Three different measurements answer the same question, so one order is
    fixed and the figure is told which it got. The bank draw beats the
    z-centroid softmax ON PURPOSE (2026-09-20, user): the softmax does not
    resolve the structure the bank draw shows, so a run that logged both must
    not switch to it just because it is the more direct measurement."""
    import sac
    pp, pm = sac.coin_pair()

    both = _belief_rows("coin_rot_er", belief_pplus=0.6, belief_pminus=0.2,
                        belief_base=0.1, zbelief_pplus=0.9, zbelief_pminus=0.1,
                        zbelief_base=0.0)
    d, source = MAP.belief_frame(both, "coin_rot_er")
    assert source == "observer posterior"
    # the two traces the figure draws, then the derived pair quantities
    assert np.allclose(d["p_pplus"], 0.6) and np.allclose(d["p_pminus"], 0.2)
    assert np.allclose(d["rel_pplus"], 0.75)          # 0.6 / (0.6 + 0.2)
    assert np.allclose(d["pair_mass"], 0.8)

    # no observer but a circuit: the z softmax, the only source left
    z_only = _belief_rows("main_rot_er_carry", belief_pplus=np.nan,
                          belief_pminus=np.nan, zbelief_pplus=0.3,
                          zbelief_pminus=0.1, zbelief_base=0.2)
    d, source = MAP.belief_frame(z_only, "main_rot_er_carry")
    assert source == "z-centroid softmax"
    assert np.allclose(d["rel_pplus"], 0.75)          # 0.3 / (0.3 + 0.1)

    # a bank draw, planted 3 P+ then 1 P- repeating, so a wide window sits at
    # 3/4. It wins over the z column when a re-run has logged BOTH.
    draws = [pp, pp, pp, pm] * 6
    bank = _belief_rows("main_rot_er_carry", belief_pplus=np.nan, belief_pminus=np.nan,
                        zbelief_pplus=0.3, zbelief_pminus=0.1, zbelief_base=0.2)
    bank["prior_src"] = list(draws) * 2
    d, source = MAP.belief_frame(bank, "main_rot_er_carry", bank_window=12)
    assert source.startswith("bank draw")
    mid = d[(d.ep_in_run > 3) & (d.ep_in_run < 8)]    # away from the block edges
    # a windowed estimate of a Bernoulli rate, so it sits AROUND 3/4 rather
    # than on it: the window is not a whole number of 4-draw cycles
    assert mid["rel_pplus"].mean() == pytest.approx(0.774, abs=0.01)
    assert np.allclose(mid["rel_pplus"], 0.75, atol=0.08)


def test_belief_frame_is_none_without_any_source():
    """baseline has no observer, no circuit and no bank -- nothing to read."""
    plain = _belief_rows("baseline_rot_er", belief_pplus=np.nan, belief_pminus=np.nan)
    assert MAP.belief_frame(plain, "baseline_rot_er") is None


def test_context_belief_figure_renders(tmp_path, monkeypatch):
    data = pd.concat([
        _belief_rows("coin_rot_sr", belief_pplus=0.6, belief_pminus=0.2),
        _belief_rows("coin_rot_er", belief_pplus=0.8, belief_pminus=0.1),
    ], ignore_index=True)
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for c in data.config.unique():
        MAP.register_circuit_variant(c)

    MAP.plot_context_belief(data, str(tmp_path))
    assert (tmp_path / "context_belief.png").exists()


def test_evoked_figure_renders(tmp_path, monkeypatch):
    data = pd.concat([_evoked_rows("baseline_rot_er", -0.9, [0.1] * 12),
                      _evoked_rows("main_rot_er_carry", -0.5, [0.9] * 6 + [0.3] * 6)],
                     ignore_index=True)
    monkeypatch.setattr(MAP, "CONFIG_ORDERS", dict(MAP.CONFIG_ORDERS))
    monkeypatch.setattr(MAP, "CONFIG_COLORS", dict(MAP.CONFIG_COLORS))
    for c in data.config.unique():
        MAP.register_circuit_variant(c)

    assert MAP._paradigm_runs(data, "evoked_recovery") == ["baseline_rot_er", "main_rot_er_carry"]
    MAP.plot_schedule_timecourse(data, str(tmp_path), schedule="evoked_recovery",
                                 run_dir=str(tmp_path / "no_runs"))
    assert (tmp_path / "schedule_timecourse_er.png").exists()
