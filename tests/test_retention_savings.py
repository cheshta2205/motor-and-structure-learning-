"""Retention (a level) against savings (a rate) must come apart.

The two are added together by any "revisit minus first visit" metric on a
block's early performance, and they came apart empirically: with the replay
buffer cleared at each switch the frozen circuit kept its savings and lost all
retention, the one-hot oracle the reverse. So the numeric core has to score a
block that STARTS higher as retention only, and a block that CLIMBS faster
from the same start as savings only -- exactly, on constructed ramps, not
approximately on real data.
"""
import numpy as np
import pandas as pd
import pytest

import make_ablations_plot as MAP

N_EP = 100


def _run(config, seed, onset_first, onset_revisit, slope_first, slope_revisit,
         n_ep=N_EP, n_ctx=2):
    """Two visits to each of n_ctx contexts. Each block is a linear ramp from
    its onset level; the metric is continuous so the arithmetic is exact."""
    rows = []
    phase = 0
    for occ, (onset, slope) in enumerate(((onset_first, slope_first),
                                          (onset_revisit, slope_revisit))):
        for _ in range(n_ctx):
            t = np.arange(n_ep)
            y = onset + slope * t / n_ep
            rows.append(pd.DataFrame({
                "config": config, "seed": seed, "phase": phase,
                "occurrence": occ, "ep": t + 1, "success": y,
                "global_ep": phase * n_ep + t + 1, "label": f"c{phase % n_ctx}"}))
            phase += 1
    return pd.concat(rows, ignore_index=True)


def _ramp_gain(onset, slope, n_ep=N_EP, frac=MAP.ONSET_FRACTION):
    """gain of a linear ramp exactly as retention_savings_table defines it."""
    y = onset + slope * np.arange(n_ep) / n_ep
    k = int(round(frac * n_ep))
    return y[k:].mean() - y[:k].mean()


def test_higher_start_same_slope_is_retention_not_savings():
    d = _run("c", 0, onset_first=0.1, onset_revisit=0.5,
             slope_first=0.4, slope_revisit=0.4)
    t = MAP.retention_savings_table(d)
    assert len(t) == 1
    assert t.retention.iloc[0] == pytest.approx(0.4)
    assert t.savings.iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_same_start_steeper_slope_is_savings_not_retention():
    d = _run("c", 0, onset_first=0.1, onset_revisit=0.1,
             slope_first=0.4, slope_revisit=0.8)
    t = MAP.retention_savings_table(d)
    assert t.retention.iloc[0] == pytest.approx(0.0, abs=1e-12)
    expected = _ramp_gain(0.1, 0.8) - _ramp_gain(0.1, 0.4)
    assert expected > 0
    assert t.savings.iloc[0] == pytest.approx(expected)


def test_gain_is_relative_to_the_blocks_own_onset():
    """A block that starts higher and climbs at the same rate must NOT be
    credited with savings -- that is the conflation the split exists to
    remove. Same numbers as the first test, asserted on the gain columns."""
    d = _run("c", 0, 0.1, 0.5, 0.4, 0.4)
    t = MAP.retention_savings_table(d)
    assert t.gain_first.iloc[0] == pytest.approx(t.gain_revisit.iloc[0])


def test_seed_is_the_unit_of_independence():
    """Blocks are averaged within a seed; the table has one row per (config,
    seed), never one per block."""
    d = pd.concat([_run("a", s, 0.1, 0.3, 0.4, 0.4) for s in range(3)]
                  + [_run("b", s, 0.1, 0.1, 0.4, 0.6) for s in range(2)])
    t = MAP.retention_savings_table(d)
    assert sorted(t.config.tolist()) == ["a", "a", "a", "b", "b"]
    assert set(t.columns) >= {"onset_first", "onset_revisit", "gain_first",
                              "gain_revisit", "retention", "savings"}


def test_run_without_a_revisit_is_skipped_not_scored():
    d = _run("c", 0, 0.1, 0.5, 0.4, 0.4)
    d = d[d.occurrence == 0]
    assert MAP.retention_savings_table(d).empty


def test_drop_first_block_removes_phase_zero_only():
    """For schedules without a warmup, phase 0 mixes initial task acquisition
    into the first visit. Dropping it must leave the other first visit(s)."""
    d = _run("c", 0, 0.1, 0.1, 0.4, 0.4, n_ctx=2)
    # make phase 0 an outlier: if it stays in, onset_first moves
    d.loc[d.phase == 0, "success"] += 5.0
    kept = MAP.retention_savings_table(d, drop_first_block=False)
    dropped = MAP.retention_savings_table(d, drop_first_block=True)
    assert kept.level_first.iloc[0] > 1.0
    assert dropped.level_first.iloc[0] == pytest.approx(0.1)


def test_level_is_the_value_at_episode_zero_not_the_window_mean():
    """The window MEAN of a ramp sits above its start by slope * (k-1)/2n, so
    a faster learner would read as retaining more. The intercept does not."""
    d = _run("c", 0, onset_first=0.1, onset_revisit=0.1,
             slope_first=0.4, slope_revisit=0.8)
    t = MAP.retention_savings_table(d)
    assert t.level_first.iloc[0] == pytest.approx(0.1)
    assert t.level_revisit.iloc[0] == pytest.approx(0.1)
    assert t.onset_revisit.iloc[0] > t.onset_first.iloc[0]   # the mean does leak


def test_linear_ramp_savings_is_split_invariant():
    """For a LINEAR ramp mean(late) - mean(early) is slope * n/2 wherever the
    block is split, so onset_fraction cannot move the number. Pinned so the
    next test is read correctly: curvature, not the split, is what makes the
    fraction matter."""
    d = _run("c", 0, 0.1, 0.1, 0.4, 0.8)
    narrow = MAP.retention_savings_table(d, onset_fraction=0.1).savings.iloc[0]
    wide = MAP.retention_savings_table(d, onset_fraction=0.5).savings.iloc[0]
    assert narrow == pytest.approx(wide)


def test_onset_fraction_matters_for_a_saturating_curve():
    """Real blocks saturate. Make the revisit saturate FASTER (same start,
    same asymptote): a narrow onset window sees the steep early part as gain;
    a wide one swallows the whole climb into the onset, and the fast learner
    then has LESS left to gain than the slow one -- savings goes negative.
    That sign flip is the ceiling compression the module comment warns about,
    and it is why ONSET_FRACTION is small."""
    d = _run("c", 0, 0.0, 0.0, 0.0, 0.0)
    t = d.ep.to_numpy() - 1
    tau = np.where(d.occurrence > 0, 10.0, 40.0)
    d["success"] = 1.0 - np.exp(-t / tau)
    narrow = MAP.retention_savings_table(d, onset_fraction=0.05).savings.iloc[0]
    wide = MAP.retention_savings_table(d, onset_fraction=0.5).savings.iloc[0]
    assert narrow > 0
    assert narrow > wide
    assert wide < 0


def test_visit_band_averages_blocks_within_seed_before_smoothing():
    """Two blocks per seed at levels 0.2 and 0.6 must give a flat 0.4 trace,
    not a 0.2-then-0.6 trace of double length (which is what feeding the
    seed's rows straight to median_seed_band would produce)."""
    d = _run("c", 0, 0.2, 0.2, 0.0, 0.0, n_ctx=2)
    d.loc[d.phase == 1, "success"] = 0.6
    band = MAP._visit_band(d, "success", revisit=False, smooth_window=1)
    assert len(band) == N_EP
    assert np.allclose(band["median"], 0.4)


def test_plot_writes_one_file_per_metric(tmp_path):
    d = pd.concat([_run("baseline", s, 0.1, 0.1, 0.2, 0.2) for s in range(2)]
                  + [_run("oracle", s, 0.1, 0.5, 0.2, 0.2) for s in range(2)]
                  + [_run("main", s, 0.1, 0.1, 0.2, 0.6) for s in range(2)])
    MAP.plot_retention_vs_savings(d, ["success", "return"], str(tmp_path))
    written = [p.name for p in tmp_path.iterdir()]
    assert any(n.startswith("retention_vs_savings_success") for n in written)
    # `return` is not a column of this frame: skipped, not raised
    assert not any(n.startswith("retention_vs_savings_return") for n in written)


def test_block_length_is_per_block_not_per_phase_across_seeds():
    """Three seeds of 100-episode blocks must give 100, not 300 -- the onset
    shading in the trace panels is drawn from this number."""
    d = pd.concat([_run("c", s, 0.1, 0.1, 0.2, 0.2) for s in range(3)])
    assert MAP._block_length(d) == N_EP


# ── variants derive their plot identity from the parent ──────────────────

def test_every_variant_of_every_base_is_a_real_key():
    """seaborn checks palette key MEMBERSHIP, so a variant that only resolved
    through a fallback would raise inside every hue-mapped plot."""
    for base in ["baseline", "oracle", "main", "end_to_end"]:
        for suffix in ("_epbuf", "_clearbuf", "_epbuf_clearbuf"):
            assert base + suffix in MAP.CONFIG_COLORS
            assert base + suffix in MAP.CONFIG_ORDERS


def test_variants_sit_right_after_their_parent_in_order():
    o = MAP.CONFIG_ORDERS
    assert o["baseline"] < o["baseline_epbuf"] < o["baseline_clearbuf"] \
        < o["baseline_epbuf_clearbuf"] < o["cb_only_end_to_end"]
    df = pd.DataFrame({"config": ["oracle_clearbuf", "baseline", "oracle", "baseline_clearbuf"]})
    assert MAP.get_config_order(df) == ["baseline", "baseline_clearbuf", "oracle", "oracle_clearbuf"]


def test_variant_colour_is_the_parent_washed_out():
    """Same hue family, strictly lighter, and the two-knob variant lighter
    still -- a shadow of the parent, not a new colour."""
    def lum(h):
        h = h.lstrip("#")
        return sum(int(h[i:i + 2], 16) for i in (0, 2, 4))
    p, e, c, ec = (MAP.CONFIG_COLORS[n] for n in
                   ("main", "main_epbuf", "main_clearbuf", "main_epbuf_clearbuf"))
    assert lum(p) < lum(e) < lum(c) < lum(ec) <= 3 * 255
    assert MAP._lighten("#000000", 1.0) == "#ffffff"
    assert MAP._lighten("#d8897b", 0.0) == "#d8897b"


def test_base_config_strips_variant_suffixes_only():
    assert MAP.base_config("main_epbuf_clearbuf") == "main"
    assert MAP.base_config("oracle_clearbuf") == "oracle"
    assert MAP.base_config("baseline") == "baseline"
    assert MAP.base_config("not_a_real_config") == "not_a_real_config"


def test_loader_accepts_variant_directories_and_still_ignores_unknown(tmp_path):
    for name in ("main_clearbuf", "totally_unknown"):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"seed": [0], "ep": [1], "success": [1.0]}).to_csv(
            d / "aggregated_results.csv", index=False)
    data = MAP.load_all_configs(str(tmp_path))
    assert data is not None
    assert set(data["config"]) == {"main_clearbuf"}
