"""
Locks down model.py's rollout timing and input wiring, since sac.py's
main_frozen/control_frozen ablations load these exact checkpoints as a
frozen feature extractor -- a silent change here would invisibly break what
those ablations are actually testing.
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


# ── Input-convention variants ────────────────────────────────────────────

def test_default_variant_leaves_existing_checkpoints_where_they_are():
    """The 27 committed checkpoints must not move: every existing analysis
    reads them at the flat path, and the point of variants is to ADD a
    convention beside the old data, not to relocate it."""
    assert M.DEFAULT_VARIANT in M.INPUT_VARIANTS
    flat = M.ckpt_path("main", 0)
    assert flat == os.path.join(M.CKPT_DIR, "main_seed0.pt")
    assert M.ckpt_path("main", 0, M.DEFAULT_VARIANT) == flat
    assert M.load("main", 0) is not None, "legacy checkpoint no longer loads"


def test_a_new_variant_gets_its_own_directory():
    M.INPUT_VARIANTS["_probe_variant"] = {"center_position": True, "s_scale": 3.0}
    try:
        p = M.ckpt_path("main", 0, "_probe_variant")
        assert p == os.path.join(M.CKPT_DIR, "_probe_variant", "main_seed0.pt")
        assert M.ckpt_path("main", 0) != p
        d = M.fig_dir("main", 0, "_probe_variant")
        assert "_probe_variant" in d
    finally:
        M.INPUT_VARIANTS.pop("_probe_variant")


def test_unknown_variant_is_rejected_everywhere():
    for call in (lambda: M.ckpt_path("main", 0, "nope"),
                 lambda: M.fig_dir("main", 0, "nope") if False else M.variant_spec("nope"),
                 lambda: M.load("main", 0, "nope"),
                 lambda: M.build(torch.rand(1, 2, 6), "nope")):
        with pytest.raises(KeyError):
            call()


def test_build_is_unchanged_under_the_default_variant():
    """Adding variants must not perturb v1 by a single bit, or every existing
    checkpoint is silently being driven differently."""
    torch.manual_seed(0)
    trans = torch.rand(3, 4, 6) * T.GRID_SIZE
    a = M.build(trans)
    b = M.build(trans, M.DEFAULT_VARIANT)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    # and v1 really is the uncentred convention
    assert float(a[0][..., :2].mean()) > 1.0


def test_centering_variant_zero_means_position_but_not_velocity():
    """What a centred variant is for: position becomes zero-mean while
    velocity -- a difference, already zero-mean -- is untouched."""
    M.INPUT_VARIANTS["_probe_centered"] = {"center_position": True,
                                           "s_scale": T.S_SCALE}
    try:
        torch.manual_seed(0)
        trans = torch.rand(64, 8, 6) * T.GRID_SIZE
        plain, _, tgt_plain = M.build(trans)
        cent, _, tgt_cent = M.build(trans, "_probe_centered")

        assert abs(float(cent[..., :2].mean())) < abs(float(plain[..., :2].mean()))
        assert abs(float(cent[..., :2].mean())) < 1.0
        # velocity identical: centring cancels in a difference
        assert torch.allclose(cent[..., 2:], plain[..., 2:], atol=1e-4)
        # the target's velocity half is likewise unaffected
        assert torch.allclose(tgt_cent[..., 2:], tgt_plain[..., 2:], atol=1e-4)
    finally:
        M.INPUT_VARIANTS.pop("_probe_centered")


def test_load_refuses_a_checkpoint_from_a_different_variant(tmp_path):
    """The guard that matters. Driving a checkpoint with the wrong input
    convention is silent -- it once cost z its context (decodability 1.00 ->
    0.28) and looked like a result about the circuit."""
    M.INPUT_VARIANTS["_probe_other"] = {"center_position": True, "s_scale": 1.0}
    try:
        ec, cb = M.EC(M.ARMS["main"]), M.CB()
        p = tmp_path / "main_seed0.pt"
        torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(),
                    "variant": "_probe_other"}, p)

        orig = M.ckpt_path
        M.ckpt_path = lambda arm, seed, variant=None: str(p)
        try:
            with pytest.raises(ValueError, match="trained under input variant"):
                M.load("main", 0, M.DEFAULT_VARIANT)
            assert M.load("main", 0, "_probe_other") is not None
        finally:
            M.ckpt_path = orig
    finally:
        M.INPUT_VARIANTS.pop("_probe_other")


def test_variantless_checkpoints_are_treated_as_the_default(tmp_path):
    """Files written before variants existed carry no record, and are v1."""
    ec, cb = M.EC(M.ARMS["main"]), M.CB()
    p = tmp_path / "main_seed0.pt"
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict()}, p)
    orig = M.ckpt_path
    M.ckpt_path = lambda arm, seed, variant=None: str(p)
    try:
        assert M.load("main", 0) is not None
    finally:
        M.ckpt_path = orig


def test_v2_normalizes_and_centers_position_without_s_scale():
    """The spec: divide by GRID_SIZE, then centre, and drop S_SCALE."""
    spec = M.variant_spec("v2_normpos_tanh")
    assert spec["pos_divisor"] == M.POS_STD
    assert spec["center_position"] is True
    assert spec["cb_activation"] == "tanh"
    # s_scale is NOT 1: centring removes the DC, but the GRU still needs drive
    assert spec["s_scale"] > 5.0

    trans, _ctx, _tss = generate_sequences(100, seed=0)
    obs, _act, tgt = M.build(trans, "v2_normpos_tanh")
    pos = obs[..., :2]
    # centred on the GRID, not on the data's empirical mean -- the transform
    # must not depend on which sequences happen to be drawn
    assert abs(float(pos.mean())) < 0.15, float(pos.mean())
    # the TARGET's position half must live in the same units, or the forward
    # model is asked to predict in a different coordinate system than it sees
    assert abs(float(tgt[..., :2].mean()) - float(pos.mean())) < 0.05


def test_v2_removes_the_constant_offset_that_v1_carries():
    """The defect this variant exists to fix: v1's state vector is dominated
    by a context-independent DC component, which makes two contexts nearly
    collinear at MC's input and puts them on the same weights."""
    trans, _ctx, _tss = generate_sequences(200, seed=0)

    def dc_share(variant):
        o = M.build(trans, variant)[0].reshape(-1, M.SDIM).numpy()
        return float(np.linalg.norm(o.mean(0)) ** 2 / (o ** 2).sum(1).mean())

    assert dc_share("v1_abspos") > 0.5
    assert dc_share("v2_normpos_tanh") < 0.01


def test_v2_balances_position_against_velocity():
    """Dividing by GRID_SIZE also rescales velocity, since build derives it
    from the divided positions. The two halves of the state end up comparable
    rather than differing by an order of magnitude."""
    trans, _ctx, _tss = generate_sequences(200, seed=0)
    o = M.build(trans, "v2_normpos_tanh")[0].reshape(-1, M.SDIM).numpy()
    pos_rms = float(np.sqrt((o[:, :2] ** 2).mean()))
    vel_rms = float(np.sqrt((o[:, 2:] ** 2).mean()))
    assert 0.3 < pos_rms / vel_rms < 3.0, (pos_rms, vel_rms)


def test_centering_and_division_leave_velocity_mean_untouched():
    """Velocity is a difference: the offset cancels exactly, and only the
    scale changes. If centring ever leaked into velocity it would inject a
    spurious constant drift."""
    trans, _ctx, _tss = generate_sequences(120, seed=1)
    v1 = M.build(trans, "v1_abspos")[0][..., 2:]
    v2 = M.build(trans, "v2_normpos_tanh")[0][..., 2:]
    spec = M.variant_spec("v2_normpos_tanh")
    factor = (spec["s_scale"] / spec["pos_divisor"]) / T.S_SCALE
    # relative tolerance: velocities are O(10) under both variants
    assert torch.allclose(v2, v1 * factor, rtol=1e-4, atol=1e-3)


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


def test_variant_entries_only_state_what_they_change():
    """Defaults are merged, so adding a schema key later cannot break an
    existing entry -- but an unknown key is rejected, since an option that
    silently did nothing is the same class of failure as a mis-scaled input."""
    M.INPUT_VARIANTS["_probe_partial"] = {"center_position": True}
    try:
        spec = M.variant_spec("_probe_partial")
        assert spec["center_position"] is True
        assert spec["pos_divisor"] == 1.0 and spec["cb_activation"] == "relu"
    finally:
        M.INPUT_VARIANTS.pop("_probe_partial")

    M.INPUT_VARIANTS["_probe_typo"] = {"centre_position": True}
    try:
        with pytest.raises(KeyError, match="unknown option"):
            M.variant_spec("_probe_typo")
    finally:
        M.INPUT_VARIANTS.pop("_probe_typo")


def test_train_can_run_without_writing_a_checkpoint(tmp_path, monkeypatch):
    """A truncated diagnostic run must not leave a checkpoint behind: run.py
    skips any (arm, seed, variant) whose file exists, so a ten-epoch check
    would silently prevent the real training from ever happening."""
    monkeypatch.setattr(M, "CKPT_DIR", str(tmp_path))
    monkeypatch.setattr(M, "N_TRAIN", 8)
    monkeypatch.setattr(M, "BATCH", 8)

    out = M.train("main", 0, epochs=1, decode_every=10 ** 6,
                  variant="v2_normpos_tanh", save=False)
    assert out is not None
    assert not os.path.exists(M.ckpt_path("main", 0, "v2_normpos_tanh"))
    assert os.listdir(tmp_path) == [] or "main_seed0.pt" not in os.listdir(tmp_path)


# ── CB's operating point under tanh ──────────────────────────────────────

def _cb_inputs(variant, n_seq=None):
    """Real (s, a, z) as CB sees them under a variant."""
    from data import generate_sequences
    n_seq = M.BATCH if n_seq is None else n_seq
    trans, _c, _t = generate_sequences(n_seq, seed=0)
    torch.manual_seed(0)
    ec = M.EC(M.ARMS["main"])
    cb = M.CB(M.variant_spec(variant)["cb_activation"])
    obs, act, _ = M.build(trans, variant)
    with torch.no_grad():
        Z, _p, _t2 = M.rollout(ec, cb, trans, "main", variant=variant)
    return (cb,
            obs[:, 1:].reshape(-1, M.SDIM),
            act[:, 1:].reshape(-1, M.ADIM),
            Z[:, 1:].reshape(-1, M.LATENT_DIM))


def _preacts(cb, s, a, z):
    with torch.no_grad():
        x = torch.cat([s, a, z], -1)
        p1 = cb.net[0](x)
        p2 = cb.net[2](cb.net[1](p1))
    return p1, p2


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


def test_relu_is_scale_free_but_tanh_is_not():
    """Why removing S_SCALE and switching to tanh only broke things together.
    ReLU is positively homogeneous, so shrinking its input never linearizes
    it; tanh has a characteristic scale of 1, so it does."""
    cb, s, a, z = _cb_inputs("v1_abspos")
    p1, _p2 = _preacts(cb, s, a, z)
    assert float((p1.abs() > 1).float().mean()) > 0.3

    x = torch.randn(64)
    for c in (0.01, 100.0):
        assert torch.allclose(torch.relu(c * x), c * torch.relu(x), atol=1e-4)


def _gate_and_readout_std(ec, cb, variant, arm="main", n_seq=64):
    from data import generate_sequences
    trans, _c, _t = generate_sequences(n_seq, seed=0)
    g, r, _S, _A, _Z = M.circuit_operating_point(ec, cb, trans, arm, variant)
    return g, r


def _fresh(variant, arm="main"):
    torch.manual_seed(0)
    spec = M.variant_spec(variant)
    return M.EC(M.ARMS[arm]), M.CB(spec["cb_activation"])




def test_pos_std_literal_still_matches_the_reference_data():
    """POS_STD fixes v2's coordinate system, so it is a literal rather than a
    per-batch measurement -- otherwise train and test would see different
    scalings. This re-measures it and fails if the data generator has moved."""
    from data import generate_sequences
    trans, _c, _t = generate_sequences(2000, seed=0)
    measured = float((trans[..., :2] - T.GRID_SIZE / 2.0).std())
    assert abs(measured - M.POS_STD) / M.POS_STD < 0.02, (measured, M.POS_STD)


def test_v2_state_has_unit_variance_and_zero_mean():
    """The whole point of POS_STD over GRID_SIZE: nn.Linear's default init
    assumes unit VARIANCE, and /GRID_SIZE gives unit RANGE but std 0.15 --
    which left every pre-activation inside tanh's linear region."""
    from data import generate_sequences
    trans, _c, _t = generate_sequences(400, seed=0)
    sc = M.variant_spec("v2_normpos_tanh")["s_scale"]
    o = M.build(trans, "v2_normpos_tanh")[0].reshape(-1, M.SDIM) / sc
    # divided back by s_scale, the state is unit-variance and zero-mean; the
    # scale on top of that is for DRIVE, not for normalisation
    assert abs(float(o[:, :2].mean())) < 0.15
    assert abs(float(o[:, :2].std()) - 1.0) < 0.15, float(o[:, :2].std())
    assert 0.5 < float(o[:, 2:].std()) < 2.0, float(o[:, 2:].std())


def test_v2_operating_point_is_usable_without_any_weight_rescaling():
    """With the input at unit variance the default init already lands EC's
    gates and CB's pre-activations in range -- no weight surgery, which was
    measured harmful (Adam's step is ~lr, so inflating weights shrinks the
    relative step by the same factor)."""
    from data import generate_sequences
    trans, _c, _t = generate_sequences(64, seed=0)
    torch.manual_seed(0)
    ec = M.EC(M.ARMS["main"])
    cb = M.CB(M.variant_spec("v2_normpos_tanh")["cb_activation"])
    g, r, S, A, Z = M.circuit_operating_point(ec, cb, trans, "main", "v2_normpos_tanh")
    # the target is v1's operating point (gate ~2.2, readout ~0.37), which is
    # where the configuration that actually learns sits
    assert g > 1.0, f"EC gates under-driven: {g}"
    assert r > 0.25, f"readout confines z to a sliver: {r}"
    with torch.no_grad():
        pre = cb.net[0](torch.cat([S, A, Z], -1))
    assert float(pre.std()) > 0.3, float(pre.std())


def test_no_calibration_hook_remains():
    """The weight-rescaling path was removed, not merely disabled: leaving a
    switch that was measured harmful invites re-enabling it."""
    assert "cb_init_calibrate" not in M._VARIANT_DEFAULTS
    for name, entry in M.INPUT_VARIANTS.items():
        assert "cb_init_calibrate" not in entry, name
    assert not hasattr(M.CB, "calibrate_init")
    assert not hasattr(M.EC, "rescale_input")
    assert not hasattr(M, "calibrate_circuit")


# ── the whitened per-source init (v3) ──────────────────────────────────────
# The claim is that DRIVE and BALANCE are separable: whitening by each
# source's own std fixes the relative contributions, and `drive` alone sets
# the total. Both halves are asserted, because getting balance right while
# landing at the wrong operating point is exactly what the `per_source` mode
# did (gate std 0.58 against a working 2.25) and it was indistinguishable
# from a bug by prediction error alone.

def _whitened(arm, variant="v3_whitened", n=400, seed=0):
    trans, _, _ = generate_sequences(n, seed=seed)
    spec = M.variant_spec(variant)
    torch.manual_seed(0)
    ec, cb = M.EC(M.ARMS[arm]), M.CB(spec["cb_activation"])
    ec, cb, src = M.apply_source_whitened_init(ec, cb, arm, trans, variant)
    return ec, cb, src, trans


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


def test_every_source_contributes_equal_drive():
    """The point of whitening: equal say per SOURCE, not per column. The
    action is the case that matters -- it is 16x quieter than position and
    two columns wide against z's sixty-four."""
    ec, cb, src, trans = _whitened("main")
    obs, act, _ = M.build(trans, "v3_whitened")
    cols = {"pos": obs[..., :2], "vel": obs[..., 2:], "act": act}
    contrib = {}
    with torch.no_grad():
        for name, sl in M.ec_source_slices("main"):
            if name not in cols:
                continue
            W = ec.cell.weight_ih[:, sl]
            contrib[name] = float((cols[name].reshape(-1, 2) @ W.T).std())
    lo, hi = min(contrib.values()), max(contrib.values())
    assert hi / lo < 1.25, contrib


def test_drive_sets_the_gate_operating_point_one_to_one():
    """`drive` is calibrated in units of gate std, so the mapping must stay
    1:1 -- that is the property that makes it a meaningful number rather than
    another opaque factor like s_scale."""
    trans, _, _ = generate_sequences(400, seed=0)
    for g in (1.0, 2.0, 3.0):
        M.INPUT_VARIANTS["_t"] = {**M.INPUT_VARIANTS["v3_whitened"], "drive": g}
        try:
            torch.manual_seed(0)
            ec, cb = M.EC(True), M.CB("tanh")
            M.apply_source_whitened_init(ec, cb, "main", trans, "_t")
            gate, *_ = M.circuit_operating_point(ec, cb, trans[:200], "main", "_t")
        finally:
            del M.INPUT_VARIANTS["_t"]
        assert abs(gate - g) / g < 0.12, (g, gate)


def test_whitening_lands_on_the_working_operating_point():
    """v1 TRAINED runs at gate std 2.25; WHITENED_DRIVE is calibrated to it."""
    assert M.WHITENED_DRIVE == 2.25
    ec, cb, _, trans = _whitened("main")
    gate, readout, *_ = M.circuit_operating_point(
        ec, cb, trans[:200], "main", "v3_whitened")
    assert 1.9 < gate < 2.6, gate
    assert readout > 0.25, readout


def test_whitening_scale_matches_the_two_arms():
    """The arm contrast is meant to isolate ONE input. Under the default init
    control's 14-wide input also drove 1.26x harder than main's 10-wide one,
    purely for being wider -- a confound in the headline comparison."""
    trans, _, _ = generate_sequences(400, seed=0)
    gates = {}
    for arm in ("main", "control"):
        ec, cb, _, _ = _whitened(arm)
        gates[arm], *_ = M.circuit_operating_point(
            ec, cb, trans[:200], arm, "v3_whitened")
    assert abs(gates["control"] / gates["main"] - 1.0) < 0.10, gates

    default = {}
    for arm in ("main", "control"):
        torch.manual_seed(0)
        ec, cb = M.EC(M.ARMS[arm]), M.CB("relu")
        default[arm], *_ = M.circuit_operating_point(
            ec, cb, trans[:200], arm, "v1_abspos")
    assert default["control"] / default["main"] > 1.15, default


def test_whitening_makes_the_input_scale_a_no_op():
    """w ~ 1/(c * std_raw) against an input of c * x_raw: the c cancels, which
    is WHY drive cannot live in the input under this init. If this ever fails,
    s_scale has crept back into being a second, hidden drive knob."""
    trans, _, _ = generate_sequences(400, seed=0)
    gates = []
    for c in (1.0, 15.0):
        M.INPUT_VARIANTS["_t"] = {**M.INPUT_VARIANTS["v3_whitened"], "s_scale": c}
        try:
            torch.manual_seed(0)
            ec, cb = M.EC(True), M.CB("tanh")
            M.apply_source_whitened_init(ec, cb, "main", trans, "_t")
            g, *_ = M.circuit_operating_point(ec, cb, trans[:200], "main", "_t")
            gates.append(g)
        finally:
            del M.INPUT_VARIANTS["_t"]
    assert abs(gates[0] - gates[1]) / gates[0] < 0.02, gates


def test_error_sources_take_the_state_scale():
    """Deliberate, not an oversight: at init CB predicts near zero so e ~ s.
    Documented as an INITIALIZATION -- e_pos falls 46x over training."""
    _, _, src, _ = _whitened("main")
    assert src["e_pos"] == src["pos"]
    assert src["e_vel"] == src["vel"]
    assert src["act"] < 0.65 * src["pos"], src   # the starved channel


def test_init_mode_is_validated():
    M.INPUT_VARIANTS["_t"] = {"init_mode": "whitend"}     # typo
    try:
        with pytest.raises(KeyError):
            M.variant_spec("_t")
    finally:
        del M.INPUT_VARIANTS["_t"]
    assert set(M.INIT_MODES) == {"default", "per_source", "whitened"}


def test_existing_variants_keep_the_default_init():
    """v1's 27 committed checkpoints were trained under torch's own init;
    changing it for them would invalidate every checkpoint on disk."""
    for name in ("v1_abspos", "v2_normpos_tanh"):
        assert M.variant_spec(name)["init_mode"] == "default", name


# ── per-variant alpha ──────────────────────────────────────────────────────

def test_alpha_defaults_to_the_module_constant_everywhere_it_matters():
    """v1's 27 checkpoints were fitted at ALPHA=25. A variant that silently
    changed it would make old checkpoints and new ones incomparable with
    nothing raising."""
    for name in ("v1_abspos", "v2_normpos_tanh", "v3_whitened"):
        assert M.variant_spec(name)["alpha"] == M.ALPHA, name
    torch.manual_seed(0)
    pred, tgt = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
    assert torch.allclose(M.weighted_loss(pred, tgt),
                          M.weighted_loss(pred, tgt, M.ALPHA))


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


def test_a_variant_can_reweight_the_two_halves():
    """alpha = 1/DT^2 would make both residuals carry coefficient 625 in their
    own raw units. Exercised through a temporary entry rather than a standing
    preset, so the knob stays covered without pinning a variant that may come
    and go."""
    M.INPUT_VARIANTS["_t"] = {**M.INPUT_VARIANTS["v3_scaled"],
                              "alpha": 1.0 / T.DT ** 2}
    try:
        spec = M.variant_spec("_t")
        assert spec["alpha"] == pytest.approx(625.0)
        base = M.variant_spec("v3_scaled")
        assert {k for k in base if base[k] != spec[k]} == {"alpha"}
    finally:
        del M.INPUT_VARIANTS["_t"]


def test_train_uses_the_variant_alpha_not_the_module_constant():
    """Regression guard: weighted_loss's default would silently win if train
    stopped passing the spec through, and the run would look fine."""
    import inspect
    src = inspect.getsource(M.train)
    assert 'weighted_loss(pred, tgt, _spec["alpha"])' in src


def test_sac_ec_cb_loss_follows_the_variant_too():
    """sac.py keeps its own flat-batch copy of the loss. If it stayed pinned
    to M.ALPHA, an end-to-end config under a reweighted variant would train
    its circuit under a different objective than the frozen checkpoints were
    fitted with, and nothing would raise."""
    import sac
    torch.manual_seed(0)
    pred, tgt = torch.randn(8, 4), torch.randn(8, 4)
    e = (pred - tgt) ** 2
    pos = float((e[..., :2] / T.DT ** 2).mean())
    vel = float(e[..., 2:].mean())
    M.INPUT_VARIANTS["_t"] = {**M.INPUT_VARIANTS["v3_scaled"],
                              "alpha": 1.0 / T.DT ** 2}
    try:
        got = float(sac.ec_cb_loss(pred, tgt, "_t"))
    finally:
        del M.INPUT_VARIANTS["_t"]
    assert abs(got - (pos + 625.0 * vel)) < 1e-3
    assert abs(float(sac.ec_cb_loss(pred, tgt)) - (pos + M.ALPHA * vel)) < 1e-4


# ── bias spread ────────────────────────────────────────────────────────────
# The hypothesis: what v1's uncentred position supplies is not DRIVE but a
# per-unit bias heterogeneity, 45x what torch initializes. These lock down the
# decomposition the claim rests on, since a pooled gate std cannot see it.

def _gate_decomp(ec, cb, arm, variant, trans):
    """Split EC's gate pre-activation into a per-unit constant (spread ACROSS
    units) and the part that varies WITHIN a unit. Only the second can gate."""
    no_current = M._check(arm)
    obs, act, _ = M.build(trans, variant)
    B, L, _ = obs.shape
    h = ec.init_hidden(B)
    s_pred = torch.zeros(B, M.ODIM)
    G = []
    with torch.no_grad():
        for t in range(L):
            s_t, a_t = obs[:, t], act[:, t]
            if t == 0:
                s_tm1 = torch.zeros(B, M.SDIM); a_tm1 = torch.zeros(B, M.ADIM)
                e_t = torch.zeros(B, M.ODIM)
            else:
                s_tm1, a_tm1, e_t = obs[:, t-1], act[:, t-1], s_t - s_pred
            x = torch.cat([s_tm1, a_tm1, e_t] if no_current
                          else [s_tm1, a_tm1, s_t, e_t], -1)
            gi = x @ ec.cell.weight_ih.T + ec.cell.bias_ih
            gh = h @ ec.cell.weight_hh.T + ec.cell.bias_hh
            i_r, i_u, i_n = gi.chunk(3, 1); h_r, h_u, h_n = gh.chunk(3, 1)
            r = torch.sigmoid(i_r + h_r)
            G.append(torch.cat([i_r + h_r, i_u + h_u, i_n + r * h_n], -1))
            z, h = ec.step(x, h)
            s_pred = cb(s_t, a_t, z)
    flat = torch.stack(G[1:]).reshape(-1, G[0].shape[-1])
    per_unit = flat.mean(0)
    return float(per_unit.std()), float((flat - per_unit).std())


def _init_ec_cb(variant, arm="main", seed=1):
    spec = M.variant_spec(variant)
    trans, _, _ = generate_sequences(250, seed=0)
    torch.manual_seed(seed)
    ec, cb = M.EC(M.ARMS[arm]), M.CB(spec["cb_activation"])
    if spec["init_mode"] == "whitened":
        M.apply_source_whitened_init(ec, cb, arm, trans, variant)
    if spec["bias_spread"]:
        M.apply_bias_spread(ec, spec["bias_spread"])
    return ec, cb, trans


def test_uncentred_position_acts_as_a_bias_not_as_drive():
    """The measurement the whole bias_spread hypothesis rests on: v1's gate
    drive is mostly a fixed per-unit offset, and its RESPONSIVE component is
    no larger than a centred variant's. A pooled std cannot distinguish these,
    which is how drive=2.25 got calibrated against the wrong quantity."""
    ec, cb, trans = _init_ec_cb("v1_abspos")
    v1_static, v1_dynamic = _gate_decomp(ec, cb, "main", "v1_abspos", trans)
    ec, cb, trans = _init_ec_cb("v3_scaled")
    c_static, c_dynamic = _gate_decomp(ec, cb, "main", "v3_scaled", trans)

    assert v1_static > 15 * c_static, (v1_static, c_static)
    # and v1 is NOT better driven on the part that gates
    assert v1_dynamic < 1.3 * c_dynamic, (v1_dynamic, c_dynamic)


def test_bias_spread_reproduces_v1s_operating_point_while_centred():
    """v3_scaled already matches v1's dynamic drive, so bias_spread changes
    exactly one thing. Both components must land on v1's."""
    ec, cb, trans = _init_ec_cb("v1_abspos")
    v1_static, v1_dynamic = _gate_decomp(ec, cb, "main", "v1_abspos", trans)
    ec, cb, trans = _init_ec_cb("v3_scaled_bias")
    b_static, b_dynamic = _gate_decomp(ec, cb, "main", "v3_scaled_bias", trans)

    assert abs(b_static - v1_static) / v1_static < 0.10, (b_static, v1_static)
    assert abs(b_dynamic - v1_dynamic) / v1_dynamic < 0.10, (b_dynamic, v1_dynamic)


def test_bias_spread_variants_differ_in_exactly_one_key():
    a, b = M.variant_spec("v3_scaled"), M.variant_spec("v3_scaled_bias")
    assert {k for k in a if a[k] != b[k]} == {"bias_spread"}
    assert b["bias_spread"] == M.BIAS_SPREAD


def test_bias_spread_touches_only_b_ih():
    """b_ih and b_hh add in the gate pre-activation, so spreading one is
    enough -- and leaving b_hh alone keeps the recurrent path untouched."""
    torch.manual_seed(0)
    ec = M.EC(True)
    hh_before = ec.cell.bias_hh.detach().clone()
    w_before = ec.cell.weight_ih.detach().clone()
    M.apply_bias_spread(ec, 2.28)
    assert torch.equal(ec.cell.bias_hh, hh_before)
    assert torch.equal(ec.cell.weight_ih, w_before)
    assert float(ec.cell.bias_ih.std()) > 1.5


def test_existing_variants_keep_torch_bias_init():
    for name in ("v1_abspos", "v2_normpos_tanh", "v3_whitened", "v3_scaled"):
        assert M.variant_spec(name)["bias_spread"] == 0.0, name


def test_train_applies_bias_spread():
    import inspect
    src = inspect.getsource(M.train)
    assert 'apply_bias_spread(ec, _spec["bias_spread"])' in src


# ── cell_type: the gating control ──────────────────────────────────────────

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


def test_cell_type_is_validated_in_the_variant():
    M.INPUT_VARIANTS["_t"] = {"cell_type": "grux"}
    try:
        with pytest.raises(KeyError):
            M.variant_spec("_t")
    finally:
        del M.INPUT_VARIANTS["_t"]


def test_existing_variants_all_use_the_gru():
    """Every checkpoint on disk was trained with a GRUCell. A variant that
    silently switched cells would fail to load with a SHAPE error rather than
    a variant error, which is louder but still worth pinning."""
    rnn = {n for n in M.INPUT_VARIANTS
           if M.variant_spec(n)["cell_type"] == "rnn"}
    # enumerated rather than skipped by name, so adding a gateless variant is
    # a deliberate edit here and never a silent one
    assert rnn == {"v3_scaled_rnn", "v3_rnn_whitened", "v3_rnn_allscaled"}, rnn


def test_operating_point_handles_a_gateless_cell():
    """circuit_operating_point chunked weight_ih into three gate blocks, which
    an RNNCell does not have. Regression: that indexing silently produced
    garbage rather than raising, since chunk() does not require divisibility."""
    trans, _, _ = generate_sequences(60, seed=0)
    for variant in ("v3_scaled_relu", "v3_scaled_rnn"):
        spec = M.variant_spec(variant)
        torch.manual_seed(1)
        ec = M.EC(True, spec["cell_type"])
        cb = M.CB(spec["cb_activation"])
        pre, readout, S, A, Z = M.circuit_operating_point(
            ec, cb, trans, "main", variant)
        assert np.isfinite(pre) and pre > 0, (variant, pre)
        assert np.isfinite(readout) and readout > 0
        assert Z.shape[-1] == T.LATENT_DIM


def test_rnn_and_gru_are_driven_equally_at_init():
    """Otherwise the rnn run tests under-driving all over again instead of
    testing gating."""
    trans, _, _ = generate_sequences(200, seed=0)
    pre = {}
    for variant in ("v3_scaled_relu", "v3_scaled_rnn"):
        spec = M.variant_spec(variant)
        torch.manual_seed(1)
        ec, cb = M.EC(True, spec["cell_type"]), M.CB(spec["cb_activation"])
        pre[variant], *_ = M.circuit_operating_point(ec, cb, trans, "main", variant)
    lo, hi = sorted(pre.values())
    assert hi / lo < 1.15, pre


def test_rnn_variant_differs_in_exactly_one_key():
    a = M.variant_spec("v3_scaled_relu")
    b = M.variant_spec("v3_scaled_rnn")
    assert {k for k in a if a[k] != b[k]} == {"cell_type"}


def test_scale_action_balances_the_sources_without_a_measured_constant():
    """The 16x action starvation was manufactured by s_scale skipping columns
    2:4, not by anything about the data: in raw grid units the three sources
    are already within 20% of each other. Scaling the action too fixes the
    balance with no per-source std to measure and drift out of date."""
    trans, _, _ = generate_sequences(300, seed=0)

    M.INPUT_VARIANTS["_raw"] = {**M.INPUT_VARIANTS["v3_scaled_rnn"], "s_scale": 1.0}
    try:
        obs, act, _ = M.build(trans, "_raw")
        raw = [float(obs[..., :2].std()), float(obs[..., 2:].std()), float(act.std())]
    finally:
        del M.INPUT_VARIANTS["_raw"]
    assert max(raw) / min(raw) < 1.25, raw          # already balanced

    obs, act, _ = M.build(trans, "v3_scaled_rnn")   # s_scale on position only
    assert float(obs[..., :2].std()) / float(act.std()) > 10

    obs, act, _ = M.build(trans, "v3_rnn_allscaled")
    got = [float(obs[..., :2].std()), float(obs[..., 2:].std()), float(act.std())]
    assert max(got) / min(got) < 1.25, got


def test_scale_action_leaves_the_target_untouched():
    """tgt carries no action term, so this changes the INPUT only -- the loss
    scale is identical and the comparison against v3_scaled_rnn is clean."""
    trans, _, _ = generate_sequences(120, seed=0)
    _, _, t_off = M.build(trans, "v3_scaled_rnn")
    _, _, t_on = M.build(trans, "v3_rnn_allscaled")
    assert torch.allclose(t_off, t_on)


def test_scale_action_defaults_off_for_every_existing_variant():
    for name in M.INPUT_VARIANTS:
        if name == "v3_rnn_allscaled":
            continue
        assert M.variant_spec(name)["scale_action"] is False, name
