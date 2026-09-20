"""
Tests for the per-step aim diagnostics.

The motivating bug was interpretive, not a crash: the probe summary's
first_angle is identical across contexts for any circuit config BY
CONSTRUCTION, and was being read as evidence that the policy ignores context.
These tests pin that invariant down and check that the later steps -- the only
place context dependence can appear -- are actually recorded and plotted.
"""
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import sac
import sac_figures
import sweep_ablations as SW
import task as T


def _probe_all_contexts(max_steps=12):
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["main"], arm_seed=0)
    ec, cb = sac.build_ec_cb(cfg, "cpu")
    trainer = SimpleNamespace(actor=sac.Actor(sac.obs_spec(cfg)))
    out = []
    for fi, cid in enumerate(sac.PHASE_CONTEXT_IDS):
        p = sac.eval_center_out(trainer, ec, cb, cfg, T.CONTEXTS[cid][:4], fi,
                                 max_steps=max_steps)
        p.update({"eval_label": T.C_LABEL[cid], "eval_context_id": cid,
                  "label": T.C_LABEL[sac.PHASE_CONTEXT_IDS[0]], "occurrence": 0,
                  "phase": 0, "context_id": sac.PHASE_CONTEXT_IDS[0],
                  "is_current": cid == sac.PHASE_CONTEXT_IDS[0]})
        out.append(p)
    return out


def test_first_action_is_context_blind_for_a_circuit_config():
    """The invariant behind the confound. At t=0 the probe has vel=0, e_t=0,
    cb_h2=0 and a freshly reset EC state, so z is context-independent and the
    whole observation is identical -- the first action CANNOT differ across
    contexts. Any reading of first_angle as context sensitivity is reading
    this artifact."""
    probes = _probe_all_contexts()
    firsts = [p["first_angle"] for p in probes]
    for f in firsts[1:]:
        assert np.allclose(firsts[0], f), "first action differed across contexts"
    for p in probes:
        assert np.allclose(p["angle_by_step"][:, 0], p["first_angle"], atol=1e-4)


def test_later_steps_do_differ_across_contexts():
    """The other half: if the probe could never distinguish contexts at any
    step, the per-step plot would be pointless. It can, a few steps in -- true
    even for an untrained actor, since the DYNAMICS differ."""
    probes = _probe_all_contexts()
    late = [np.nanmean(p["angle_by_step"][:, 8]) for p in probes]
    assert max(late) - min(late) > 1.0, late


def test_angle_by_step_is_nan_padded_after_a_reach_ends():
    probes = _probe_all_contexts(max_steps=20)
    p = probes[0]
    assert p["angle_by_step"].shape == (len(p["goals"]), 20)
    for k in range(len(p["goals"])):
        row = p["angle_by_step"][k]
        recorded = ~np.isnan(row)
        # recorded steps form a prefix: no gaps
        assert np.all(recorded[:recorded.sum()]) and not np.any(recorded[recorded.sum():])


def test_circular_mean_handles_the_wrap():
    """An arithmetic mean of +179 and -179 gives 0 -- the opposite direction.
    Aim angles live on a circle and must be averaged there."""
    got = sac_figures._circular_mean_deg(np.array([[179.0], [-179.0]]), axis=0)
    assert abs(abs(float(got[0])) - 180.0) < 1.0, got
    got2 = sac_figures._circular_mean_deg(np.array([[10.0], [-10.0]]), axis=0)
    assert abs(float(got2[0])) < 1e-6


def test_aim_and_trajectory_plots_write_files(tmp_path):
    probes = _probe_all_contexts()
    sac_figures.plot_aim_by_step(0, probes, str(tmp_path), config_name="main")
    sac_figures.plot_transfer_trajectories(0, probes, str(tmp_path),
                                           config_name="main")
    written = {f.split(".")[0] for f in os.listdir(tmp_path)}
    assert {"aim_by_step", "transfer_trajectories"} <= written


def test_plots_tolerate_probe_files_without_per_step_angles(tmp_path):
    """Older runs have no angle_by_step; the plot must skip, not raise."""
    probes = _probe_all_contexts()
    for p in probes:
        p.pop("angle_by_step")
    sac_figures.plot_aim_by_step(0, probes, str(tmp_path))
    assert "aim_by_step.png" not in os.listdir(tmp_path)
