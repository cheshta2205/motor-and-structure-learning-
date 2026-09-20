"""
Locks down model.py's rollout timing and input wiring.

PRUNED 2026-09-20. This file used to be dominated by the input-convention
VARIANT REGISTRY -- model.INPUT_VARIANTS, variant_spec, the v1/v2/v3 ladder,
per-variant alpha, bias_spread, scale_action and the whitened per-source init.
That registry went with the old plant (one convention now, guarded by
model.CONVENTION), so 44 of the 59 tests here were asserting against an API
that no longer exists and had been failing for some time. What survives is
everything that tests LIVE behaviour: the e_t timing convention and its
ablations, MTL noise, the weighted loss, CB's activation (relu vs tanh -- the
control that actually matters for context learning), the source slices and
the recurrent cell type. The deleted tests are in git history.
"""
import os

import numpy as np
import pytest
import torch

import model as M
import task as T
from data import generate_sequences


class _RecordingEC:
    """Wraps a real EC, recording the raw input `x` passed to every step()
    call so the test can inspect it without reimplementing model.rollout."""

    def __init__(self, ec):
        self._ec = ec
        self.no_current = ec.no_current
        self.calls = []

    def init_hidden(self, B):
        return self._ec.init_hidden(B)

    def step(self, x, h):
        self.calls.append(x.clone())
        return self._ec.step(x, h)


def _small_batch(seq_len=3, n=4, seed=0):
    trans, _, _ = generate_sequences(n, seq_len=seq_len, seed=seed)
    return trans


def test_mtl_input_width_matches_ec_cell():
    assert M.mtl_input_width("main") == 10
    assert M.mtl_input_width("control") == 14
    for arm in ("main", "control"):
        ec = M.EC(M.ARMS[arm])
        assert ec.cell.weight_ih.shape[1] == M.mtl_input_width(arm)


def test_e_t_forced_zero_at_first_step_then_read_before_write():
    trans = _small_batch(seq_len=3, n=4)
    ec, cb = M.EC(M.ARMS["main"]), M.CB()
    ec.eval(); cb.eval()
    rec = _RecordingEC(ec)

    with torch.no_grad():
        Z, P, target = M.rollout(rec, cb, trans, "main")

    obs, act, _ = M.build(trans)
    B = obs.shape[0]

    x0 = rec.calls[0]
    assert torch.allclose(x0[:, :4], torch.zeros(B, 4))     # s_tm1
    assert torch.allclose(x0[:, 4:6], torch.zeros(B, 2))    # a_tm1
    assert torch.allclose(x0[:, 6:10], torch.zeros(B, 4))   # e_t forced zero

    x1 = rec.calls[1]
    e1 = x1[:, 6:10]
    expected_e1 = obs[:, 1] - P[:, 0]   # read-before-write: e_1 reads CB's t=0 prediction
    assert torch.allclose(e1, expected_e1, atol=1e-6)


def test_zero_ablation_forces_e_t_to_zero():
    trans = _small_batch(seq_len=3, n=4)
    ec, cb = M.EC(M.ARMS["main"]), M.CB()
    ec.eval(); cb.eval()
    rec = _RecordingEC(ec)

    with torch.no_grad():
        M.rollout(rec, cb, trans, "main", e_mode="zero")

    for x_t in rec.calls:
        assert torch.allclose(x_t[:, 6:10], torch.zeros_like(x_t[:, 6:10]))


def test_shuffle_ablation_permutes_e_t_across_batch():
    """At t=1 specifically, e_1's pre-ablation value is identical whether the
    run is intact or shuffled (ablation has no effect at t=0, since e_0 is
    forced to zero either way, so h_0/s_pred_0 -- and hence the baseline e_1
    -- don't depend on the mode). So shuffle-mode's e_1 must be exactly a
    batch permutation of intact-mode's e_1."""
    trans = _small_batch(seq_len=3, n=6)
    ec, cb = M.EC(M.ARMS["main"]), M.CB()
    ec.eval(); cb.eval()

    rec_intact = _RecordingEC(ec)
    with torch.no_grad():
        M.rollout(rec_intact, cb, trans, "main", e_mode="intact")
    e1_intact = rec_intact.calls[1][:, 6:10]

    gen = torch.Generator().manual_seed(0)
    rec_shuffle = _RecordingEC(ec)
    with torch.no_grad():
        M.rollout(rec_shuffle, cb, trans, "main", e_mode="shuffle", gen=gen)
    e1_shuffle = rec_shuffle.calls[1][:, 6:10]

    sorted_intact = torch.sort(e1_intact.sum(dim=-1)).values
    sorted_shuffle = torch.sort(e1_shuffle.sum(dim=-1)).values
    assert torch.allclose(sorted_intact, sorted_shuffle, atol=1e-6)
    assert not torch.allclose(e1_intact, e1_shuffle, atol=1e-6)


def test_mtl_noise_perturbs_hidden_not_e_t_input():
    """mtl_noise adds to h AFTER ec.step returns, so it never changes what's
    fed INTO ec.step -- the perturbation shows up in z at the NEXT step, not
    in the current step's input."""
    trans = _small_batch(seq_len=3, n=4)
    ec, cb = M.EC(M.ARMS["main"]), M.CB()
    ec.eval(); cb.eval()

    rec_intact = _RecordingEC(ec)
    gen1 = torch.Generator().manual_seed(0)
    with torch.no_grad():
        Z_intact, _, _ = M.rollout(rec_intact, cb, trans, "main", e_mode="intact", gen=gen1)

    rec_noise = _RecordingEC(ec)
    gen2 = torch.Generator().manual_seed(0)
    with torch.no_grad():
        Z_noise, _, _ = M.rollout(rec_noise, cb, trans, "main", e_mode="mtl_noise", gen=gen2)

    # inputs at t=0 and t=1 are bit-identical between the two runs...
    assert torch.allclose(rec_intact.calls[0], rec_noise.calls[0])
    assert torch.allclose(rec_intact.calls[1], rec_noise.calls[1])
    # ...but the perturbation still shows up downstream, in z at t=1.
    assert not torch.allclose(Z_intact[:, 1], Z_noise[:, 1])


def test_weighted_loss_alpha_weights_only_velocity():
    pos_err, vel_err = 1.0, 2.0
    pred = torch.zeros(1, 2, 4)
    tgt = torch.zeros(1, 2, 4)
    tgt[:, 0, :2] = pos_err   # only the kept timestep (index 0) matters; index 1 is dropped
    tgt[:, 0, 2:] = vel_err

    loss = M.weighted_loss(pred, tgt)
    expected = ((pos_err ** 2) / T.DT ** 2) + M.ALPHA * (vel_err ** 2)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_cb_activation_follows_the_variant():
    _ec, cb1 = M.EC(M.ARMS["main"]), M.CB("relu")
    assert isinstance(cb1.net[1], torch.nn.ReLU)
    cb2 = M.CB("tanh")
    assert isinstance(cb2.net[1], torch.nn.Tanh) and isinstance(cb2.net[3], torch.nn.Tanh)
    with pytest.raises(KeyError):
        M.CB("sigmoid")


def test_tanh_cb_hidden_is_bounded_and_zero_centred():
    """Why the user asked for tanh: h2 is what MC reads, and under ReLU it is
    unbounded and positive-mean (measured RMS 5.85 against the kinematic
    block's 1.78), which is what made the channels unbalanceable."""
    torch.manual_seed(0)
    s, a = torch.randn(256, M.SDIM) * 3, torch.randn(256, 2)
    z = torch.tanh(torch.randn(256, M.LATENT_DIM))
    _p, _h1, h2_relu = M.CB("relu")(s, a, z, return_hidden=True)
    _p, _h1, h2_tanh = M.CB("tanh")(s, a, z, return_hidden=True)

    assert float(h2_tanh.abs().max()) <= 1.0
    assert abs(float(h2_tanh.mean())) < abs(float(h2_relu.mean()))
    assert float((h2_relu < 0).float().mean()) == 0.0, "ReLU output is non-negative"
    assert float((h2_tanh < 0).float().mean()) > 0.2, "tanh output must be signed"


def test_activation_change_is_invisible_to_state_dict():
    """The hazard the variant guard exists for. ReLU and Tanh carry no
    parameters and sit at the same Sequential indices, so a checkpoint trained
    under one loads into the other WITHOUT ERROR and predicts nonsense."""
    assert list(M.CB("relu").state_dict()) == list(M.CB("tanh").state_dict())
    src = M.CB("relu")
    dst = M.CB("tanh")
    dst.load_state_dict(src.state_dict())      # no error -- that is the point
    s, a = torch.randn(8, M.SDIM), torch.randn(8, 2)
    z = torch.tanh(torch.randn(8, M.LATENT_DIM))
    assert not torch.allclose(src(s, a, z), dst(s, a, z)), \
        "identical predictions would make the hazard moot"


def test_scaling_weight_and_bias_scales_the_pre_activation_exactly():
    """The property one-pass calibration rests on: scaling BOTH by k scales
    the pre-activation by exactly k. Scaling the weight alone would not."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(7, 5)
    x = torch.randn(32, 7)
    before = lin(x).detach().clone()
    k = 3.7
    with torch.no_grad():
        lin.weight.mul_(k); lin.bias.mul_(k)
    assert torch.allclose(lin(x), before * k, atol=1e-5)


def test_source_slices_tile_the_input_exactly():
    """A gap or an overlap would leave some columns un-whitened (or whitened
    twice) with no error -- the silent-miscalibration class again."""
    for arm in ("main", "control"):
        covered = []
        for _, sl in M.ec_source_slices(arm):
            covered.extend(range(sl.start, sl.stop))
        assert covered == list(range(M.mtl_input_width(arm))), arm
    covered = []
    for _, sl in M.cb_source_slices():
        covered.extend(range(sl.start, sl.stop))
    assert covered == list(range(M.SDIM + M.ADIM + T.LATENT_DIM))


def test_source_slices_name_the_columns_rollout_actually_builds():
    """Pins the slice names to rollout's concatenation order. If the arms'
    input order ever changes, whitening would silently scale the wrong
    columns by the wrong source's std."""
    names = [n for n, _ in M.ec_source_slices("main")]
    assert names == ["pos", "vel", "act", "e_pos", "e_vel"]
    names = [n for n, _ in M.ec_source_slices("control")]
    assert names == ["pos", "vel", "act", "cur_pos", "cur_vel", "e_pos", "e_vel"]


def test_alpha_scales_only_the_velocity_term():
    torch.manual_seed(0)
    pred, tgt = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
    e = (pred[:, :-1] - tgt[:, :-1]) ** 2
    pos = float((e[..., :2] / T.DT ** 2).mean())
    vel = float(e[..., 2:].mean())
    for a in (1.0, 25.0, 625.0):
        want = pos + a * vel
        # RELATIVE, not absolute: at alpha=625 the loss is ~2.5e3 and float32
        # accumulation in mean() is good to ~1e-7 of that, so an absolute 1e-4
        # bound fails for reasons unrelated to the weighting under test.
        assert abs(float(M.weighted_loss(pred, tgt, a)) - want) < 1e-6 * want


def test_cell_type_selects_the_recurrent_cell():
    assert isinstance(M.EC(True, "gru").cell, torch.nn.GRUCell)
    assert isinstance(M.EC(True, "rnn").cell, torch.nn.RNNCell)
    assert M.EC(True).cell_type == "gru"          # default is unchanged
    with pytest.raises(KeyError):
        M.EC(True, "lstm")


def test_rnn_has_no_gates():
    """A GRUCell stacks three gate blocks into weight_ih; an RNNCell has one.
    That IS the manipulation -- the gru's u*h product has no counterpart."""
    g, r = M.EC(True, "gru"), M.EC(True, "rnn")
    assert g.cell.weight_ih.shape[0] == 3 * T.EC_HIDDEN
    assert r.cell.weight_ih.shape[0] == 1 * T.EC_HIDDEN
    assert sum(p.numel() for p in r.cell.parameters()) * 3 == \
           sum(p.numel() for p in g.cell.parameters())
