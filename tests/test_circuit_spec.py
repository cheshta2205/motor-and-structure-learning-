"""model.CircuitSpec: one object naming a supervised circuit, accepted in the
`action_mode` slot everywhere.

What has to hold:

  a bare mode string still means that mode with the deterministic readout
  a spec in the slot selects tag, path, stamp, rollout distribution and
    figure directory -- the same ones the loose knobs select
  a spec beside conflicting loose knobs is refused, not merged
  every CLI (run, figures, representation_analysis, conjunctivity) parses the
    same four flags into the same spec, with beta/sigma implying noisy
  data.generate_sequences follows the spec's mode and rejects garbage
"""
import argparse

import pytest
import torch

import model as M
import data
import run


def test_bare_string_is_the_default_readout_circuit():
    s = M.circuit("iid")
    assert s == M.CircuitSpec("iid", M.DEFAULT_READOUT, M.RATE_BETA, M.NOISE_SIGMA)
    assert s.readout == "noisy" and s.train_contexts == "all" and s.context_ids is None
    assert M.circuit("fitted").action_mode == "ar1"
    assert M.arm_tag("main", "iid") == "main_iid" == M.arm_tag("main", M.circuit("iid"))
    assert M.normalize_action_mode(M.CircuitSpec("ar1", "noisy", 1e-4, 0.1)) == "ar1"


def test_spec_selects_what_the_loose_knobs_select():
    spec = M.CircuitSpec("iid", "noisy", 1e-4, 0.1)
    loose = dict(action_mode="iid", readout="noisy", beta=1e-4, sigma=0.1)
    assert M.arm_tag("main", spec) == M.arm_tag("main", **loose) == "main_iid_noisy0.0001_sigma0.1"
    assert M.ckpt_path("main", 2, spec) == M.ckpt_path("main", 2, **loose)
    assert M.convention_for(spec) == M.convention_for(**loose)
    assert str(spec) == "iid_noisy0.0001_sigma0.1"
    assert str(M.CircuitSpec()) == M.DEFAULT_ACTION_MODE
    assert M.CircuitSpec("gp", "noisy", 1e-3, 1.0) == M.CircuitSpec("gp", "noisy", 0.001, 1)


def test_spec_beside_conflicting_knobs_is_refused():
    spec = M.CircuitSpec("iid", "noisy", 1e-4, 0.1)
    assert M.circuit(spec) is spec
    with pytest.raises(ValueError):
        M.circuit(spec, readout="tanh")
    with pytest.raises(ValueError):
        M.arm_tag("main", spec, beta=5e-3)          # 1e-3 is the default and would pass
    with pytest.raises(KeyError):
        M.CircuitSpec("gp", "vib")


def test_fig_dir_and_load_follow_the_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "FIG_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(M, "CKPT_DIR", str(tmp_path / "ckpt"))
    spec = M.CircuitSpec("iid", "noisy", 2e-3, 0.3)       # default sigma: no sigma in the tag
    assert M.fig_dir("main", 0, action_mode=spec).endswith("figures/main_iid_noisy0.002/seed0")
    torch.manual_seed(0)
    ec, cb = M.EC(True, M.CELL_TYPE, readout="noisy", beta=2e-3, sigma=0.3), M.CB(M.CB_ACTIVATION)
    import os
    p = M.ckpt_path("main", 0, spec)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": [],
                "convention": M.convention_for(spec)}, p)
    ec2, _ = M.load("main", 0, spec)
    assert ec2.readout_kind == "noisy" and ec2.sigma == 0.3 and ec2.beta == 2e-3
    assert M.load("main", 0, "iid") is None                 # the default channel's path


def test_generate_sequences_follows_the_spec_and_rejects_garbage():
    a_iid, _, _ = data.generate_sequences(4, seed=1, action_mode=M.CircuitSpec("iid", "noisy", 1e-4, 0.1))
    b_iid, _, _ = data.generate_sequences(4, seed=1, action_mode="iid")
    assert torch.equal(a_iid, b_iid)
    a_gp, _, _ = data.generate_sequences(4, seed=1, action_mode=M.CircuitSpec("gp"))
    assert not torch.equal(a_gp, b_iid)
    with pytest.raises(KeyError):
        data.generate_sequences(4, seed=1, action_mode="mystery")


def _parse(argv):
    ap = argparse.ArgumentParser()
    M.add_circuit_args(ap)
    return M.circuit_from_args(ap.parse_args(argv))


def test_every_cli_parses_the_same_flags_into_the_same_spec():
    assert _parse([]) == M.CircuitSpec()
    assert _parse(["--action-mode", "fitted"]) == M.CircuitSpec("ar1")
    # beta or sigma alone imply the noisy readout
    assert _parse(["--noise-sigma", "0.1"]) == M.CircuitSpec("gp", "noisy", M.RATE_BETA, 0.1)
    assert _parse(["--rate-beta", "0.001"]) == M.CircuitSpec("gp", "noisy", 1e-3, M.NOISE_SIGMA)
    assert _parse(["--action-mode", "iid", "--rate-beta", "0.001", "--noise-sigma", "0.1"]) == \
        M.CircuitSpec("iid", "noisy", 1e-3, 0.1)
    assert _parse(["--readout", "tanh", "--action-mode", "iid"]) == M.CircuitSpec("iid", "tanh")
    assert _parse(["--train-contexts", "rotations"]) == M.CircuitSpec(train_contexts="rotations")
    assert run.circuit_from_args is M.circuit_from_args


def test_run_train_accepts_a_spec_in_the_action_mode_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "CKPT_DIR", str(tmp_path))
    seen = []
    monkeypatch.setattr(M, "train", lambda arm, seed, **kw: seen.append((arm, seed, kw)))
    spec = M.CircuitSpec("iid", "noisy", 1e-3, 0.1)
    run.train(["main"], [0], workers=1, action_mode=spec)
    assert seen and seen[0][2]["action_mode"] == spec
    assert M.circuit(**seen[0][2]) == spec           # defaults beside the spec: no conflict
