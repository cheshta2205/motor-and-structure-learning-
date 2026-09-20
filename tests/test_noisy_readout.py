"""EC's fixed-noise readout with a rate cost.

What has to hold:

  the deterministic readout is UNTOUCHED -- same parameters, same names, same
    output, zero penalty -- so every existing checkpoint still loads
  the noisy readout samples in training and is deterministic at inference,
    stays in [-1, 1], and its noise sd is the fixed constant, not learned
  the rate accumulates ||mu||^2 per step across a rollout, resets on demand,
    carries gradient, and enters the loss scaled by beta
  the circuit is its own axis: tag, checkpoint path, figure dir and stamp all
    carry beta (and the stamp sigma), so a beta mismatch is refused by load
  train() writes track.npz into the circuit's OWN figure dir
"""
import numpy as np
import pytest
import torch

import model as M
from model import EC


def _x(B):
    return torch.randn(B, SDIM_IN := (1 * M.SDIM + M.ADIM + M.SDIM))


def test_tanh_readout_is_unchanged_and_now_the_variant():
    """The deterministic readout still exists with its parameter names; since
    2026-09-18 it is the "_tanh" variant and the noisy channel the default."""
    ec = EC(no_current=True, readout="tanh")
    assert ec.readout_kind == "tanh"
    keys = set(ec.state_dict())
    assert {"readout.0.weight", "readout.0.bias"} <= keys and "mu.weight" not in keys
    x, h = _x(3), ec.init_hidden(3)
    ec.train()
    z1, _ = ec.step(x, h)
    z2, _ = ec.step(x, h)
    assert torch.equal(z1, z2)                         # no noise
    assert float(ec.rate_penalty()) == 0.0
    assert M.readout_stamp("tanh") == "+tanh-readout"
    assert M.convention_for("gp") == M.convention_for("gp", "noisy", M.RATE_BETA, M.NOISE_SIGMA)
    assert M.convention_for("gp") != M.convention_for("gp", "tanh")
    assert M.arm_tag("main", "ar1") == "main_ar1" and M.arm_tag("main", "ar1", "tanh") == "main_ar1_tanh"
    assert EC(no_current=True).readout_kind == "noisy"          # the default


def test_noisy_readout_samples_in_training_and_not_in_eval():
    torch.manual_seed(0)
    ec = EC(no_current=True, readout="noisy", beta=1e-3)
    x, h = _x(4), ec.init_hidden(4)
    ec.train()
    z1, _ = ec.step(x, h)
    z2, _ = ec.step(x, h)
    assert not torch.allclose(z1, z2)
    assert z1.abs().max() <= 1.0
    ec.eval()
    z3, _ = ec.step(x, h)
    z4, _ = ec.step(x, h)
    assert torch.equal(z3, z4)
    assert torch.allclose(z3, torch.tanh(ec.mu(ec.cell(x, h))))


def test_noise_sd_is_the_fixed_constant():
    """Many draws of the pre-tanh channel at the same input: sd == NOISE_SIGMA
    per dimension, and there is no learnable parameter for it."""
    torch.manual_seed(1)
    ec = EC(no_current=True, readout="noisy")
    assert not any("sigma" in n or "logvar" in n for n, _ in ec.named_parameters())
    x, h = _x(1), ec.init_hidden(1)
    ec.train()
    with torch.no_grad():
        hh = ec.cell(x, h)
        mu = ec.mu(hh)
        draws = torch.stack([torch.atanh(ec.step(x, h)[0].clamp(-0.999999, 0.999999))
                             for _ in range(4000)])
    # the tanh clamp biases far tails slightly; test on a well-inside dimension
    d = int(mu.abs().argmin())
    assert draws[:, 0, d].std() == pytest.approx(M.NOISE_SIGMA, rel=0.08)
    assert draws[:, 0, d].mean() == pytest.approx(float(mu[0, d]), abs=0.08)


def test_rate_accumulates_per_step_resets_and_has_gradient():
    torch.manual_seed(2)
    ec = EC(no_current=True, readout="noisy", beta=0.5)
    ec.train()
    x, h = _x(2), ec.init_hidden(2)
    assert float(ec.rate()) == 0.0
    mus = []
    for _ in range(3):
        mus.append(ec.mu(ec.cell(x, h)))
        _, h = ec.step(x, h)
    expected = torch.stack([(m ** 2).sum(-1).mean() for m in mus]).mean()
    assert float(ec.rate()) == pytest.approx(float(expected), rel=1e-5)
    assert float(ec.rate_penalty()) == pytest.approx(0.5 * float(expected), rel=1e-5)
    ec.rate_penalty().backward()
    assert ec.mu.weight.grad is not None and ec.mu.weight.grad.abs().sum() > 0
    ec.reset_rate()
    assert float(ec.rate()) == 0.0
    ec.eval()
    ec.step(x, ec.init_hidden(2))
    assert float(ec.rate()) == 0.0                     # no accumulation at inference


def test_noisy_circuit_is_its_own_axis():
    # the DEFAULT channel (beta 1e-3, sigma 0.3) adds nothing to the tag
    assert M.arm_tag("main", "iid", "noisy", 1e-3) == "main_iid"
    assert M.arm_tag("main", "gp", "noisy", 0.01) == "main_noisy0.01"
    assert M.ckpt_path("main", 2, "iid", "noisy", 0.01).endswith("main_iid_noisy0.01_seed2.pt")
    s = M.convention_for("iid", "noisy", 1e-3)
    assert s.startswith(M.CONVENTION) and "iid-actions" in s and "sigma0.3" in s and "beta0.001" in s
    assert M.convention_for("iid", "noisy", 1e-3) != M.convention_for("iid", "noisy", 3e-3)
    with pytest.raises(KeyError):
        M.arm_tag("main", "gp", "vib")


def test_sigma_is_a_stamped_parameter_with_the_default_tag_unchanged():
    """sigma reaches the stamp always and the tag only when it is not the
    default (0.3 since 2026-09-18)."""
    assert M.NOISE_SIGMA == 0.3 and M.RATE_BETA == 1e-3 and M.DEFAULT_READOUT == "noisy"
    assert M.arm_tag("main", "iid", "noisy", 1e-4, 0.3) == "main_iid_noisy0.0001"
    assert M.arm_tag("main", "iid", "noisy", 1e-4, 0.1) == "main_iid_noisy0.0001_sigma0.1"
    assert M.arm_tag("main", "iid", "noisy", 1e-3, 1.0) == "main_iid_noisy0.001_sigma1"
    assert M.readout_tag("tanh", 1e-4, 0.1) == "_tanh"
    assert M.ckpt_path("main", 0, "iid", "noisy", 1e-4, 0.3).endswith(
        "main_iid_noisy0.0001_seed0.pt")
    s1, s01 = (M.convention_for("iid", "noisy", 1e-4, s) for s in (1.0, 0.1))
    assert "sigma1-" in s1 and "sigma0.1-" in s01 and s1 != s01
    assert M.convention_for("iid", "noisy", 1e-4) == M.convention_for("iid", "noisy", 1e-4, 0.3)
    # the module reads its own sigma: two circuits at different sigma differ
    # in nothing but the noise sd they draw with
    torch.manual_seed(5)
    ec = EC(no_current=True, readout="noisy", beta=1e-4, sigma=0.1)
    assert ec.sigma == 0.1 and not any("sigma" in n for n, _ in ec.named_parameters())
    x, h = _x(1), ec.init_hidden(1)
    ec.train()
    with torch.no_grad():
        mu = ec.mu(ec.cell(x, h))
        draws = torch.stack([torch.atanh(ec.step(x, h)[0].clamp(-0.999999, 0.999999))
                             for _ in range(4000)])
    d = int(mu.abs().argmin())
    assert draws[:, 0, d].std() == pytest.approx(0.1, rel=0.08)


def test_load_round_trips_and_refuses_a_beta_mismatch(tmp_path, monkeypatch):
    """Write a noisy circuit's checkpoint the way train() does, load it back
    under the same (mode, readout, beta), and see load refuse a different
    beta -- the stamp is what protects a result from the wrong circuit."""
    monkeypatch.setattr(M, "CKPT_DIR", str(tmp_path / "ckpt"))
    torch.manual_seed(3)
    ec, cb = EC(True, M.CELL_TYPE, readout="noisy", beta=2e-3), M.CB(M.CB_ACTIVATION)
    p = M.ckpt_path("main", 0, "iid", "noisy", 2e-3)
    import os
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": [],
                "convention": M.convention_for("iid", "noisy", 2e-3),
                "action_mode": "iid", "readout": "noisy", "beta": 2e-3}, p)
    loaded = M.load("main", 0, "iid", readout="noisy", beta=2e-3)
    assert loaded is not None
    ec2, _ = loaded
    assert ec2.readout_kind == "noisy" and not ec2.training
    x, h = _x(2), ec.init_hidden(2)
    ec.eval()
    assert torch.allclose(ec.step(x, h)[0], ec2.step(x, h)[0])
    # a different beta is a different path -> absent, not silently loaded
    assert M.load("main", 0, "iid", readout="noisy", beta=3e-3) is None
    # same path, wrong stamp -> refused
    sd = torch.load(p); sd["convention"] = M.convention_for("iid", "noisy", 3e-3); torch.save(sd, p)
    with pytest.raises(ValueError):
        M.load("main", 0, "iid", readout="noisy", beta=2e-3)
    # a non-default sigma is its own path, and its stamp is checked the same way
    p2 = M.ckpt_path("main", 0, "iid", "noisy", 2e-3, 0.1)
    assert p2 != p
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "hist": [],
                "convention": M.convention_for("iid", "noisy", 2e-3, 0.1),
                "action_mode": "iid", "readout": "noisy", "beta": 2e-3, "sigma": 0.1}, p2)
    ec3, _ = M.load("main", 0, "iid", readout="noisy", beta=2e-3, sigma=0.1)
    assert ec3.sigma == 0.1
    with pytest.raises(ValueError):                    # default-sigma stamp at the sigma-0.1 path
        sd = torch.load(p2); sd["convention"] = M.convention_for("iid", "noisy", 2e-3); torch.save(sd, p2)
        M.load("main", 0, "iid", readout="noisy", beta=2e-3, sigma=0.1)


def test_fig_dir_follows_the_readout(tmp_path, monkeypatch):
    # fig_dir makedirs its result: point it at a temp root so the test leaves
    # nothing behind in the real figures/ tree
    monkeypatch.setattr(M, "FIG_DIR", str(tmp_path / "figures"))
    d_default = M.fig_dir("main", 0)
    d_noisy = M.fig_dir("main", 0, action_mode="iid", readout="noisy", beta=0.01)
    d_tanh = M.fig_dir("main", 0, readout="tanh")
    assert d_default.endswith("figures/main/seed0")
    assert d_noisy.endswith("figures/main_iid_noisy0.01/seed0")
    assert d_tanh.endswith("figures/main_tanh/seed0")
