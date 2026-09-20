"""The GP family of action processes.

What has to hold for this to be usable:

  it CONTAINS the current process, so "train on a range" is a strict
    generalisation of "train on RHO" rather than a different experiment
  the marginal sd is ACTION_SD for EVERY draw, so no parameter silently
    changes the effect sizes the plant is calibrated against
  it SPANS the two measured endpoints -- the fitted distribution and the
    trained policy's -- because covering a range that misses the deployment
    point would be pointless
  the existing path is untouched, bit for bit
"""
import numpy as np
import pytest

import data as D
import task as T

SEQ = 400          # long, so per-sequence statistics are estimable


def _lag1(a):
    x = a[:-1].ravel(); y = a[1:].ravel()
    return float(np.corrcoef(x, y)[0, 1])


# ── the family contains what we already train on ─────────────────────────

def test_nu_one_half_is_exactly_ar1():
    """Matern nu=1/2 is the OU kernel, so ell = -1/ln(rho) must reproduce
    next_action's autocorrelation. Distributionally, not sample-wise: a joint
    draw consumes the rng differently from a recursion."""
    rho = T.RHO
    ell = -1.0 / np.log(rho)
    lags = np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
    K = T.matern_kernel(lags, 0.5, ell)
    assert np.allclose(K[0], rho ** np.arange(6), atol=1e-12)


def test_the_gaussian_limit_matches_the_fitted_marginal():
    rng = np.random.default_rng(0)
    a = T.action_sequence(20000, rng, dict(nu=0.5, ell=1e-6, gain=0.0, scale=1.0))
    assert a.std() == pytest.approx(T.ACTION_SD, rel=0.03)
    # near-zero ell means near-white, like an AR(1) at rho ~ 0
    assert abs(_lag1(a)) < 0.05


def test_ar1_parameters_reproduce_the_fitted_correlation():
    rng = np.random.default_rng(1)
    a = T.action_sequence(SEQ, rng, dict(nu=0.5, ell=-1.0 / np.log(T.RHO),
                                          gain=0.0, scale=1.0))
    # one 400-step draw; tolerance is sampling error, not sloppiness
    assert _lag1(a) == pytest.approx(T.RHO, abs=0.12)


# ── the invariant the plant calibration depends on ───────────────────────

@pytest.mark.parametrize("gain", [0.0, 0.5, 2.2, 3.4, 5.0])
@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, np.inf])
def test_marginal_sd_is_action_sd_at_unit_scale(nu, gain):
    """At scale=1 the marginal must be exactly ACTION_SD whatever (nu, gain)
    are -- the warp's renormaliser is what makes `scale` the ONLY thing that
    moves magnitude, so a shape knob cannot change it as a side effect."""
    rng = np.random.default_rng(2)
    a = T.action_sequence(4000, rng, dict(nu=nu, ell=2.0, gain=gain, scale=1.0))
    assert a.std() == pytest.approx(T.ACTION_SD, rel=0.08)


def test_tanh_sd_is_deterministic():
    """Quadrature, not Monte Carlo -- a sampled normaliser would make the
    marginal depend on an rng draw."""
    assert T._tanh_sd(2.2) == T._tanh_sd(2.2)
    assert T._tanh_sd(0.001) == pytest.approx(0.001, rel=1e-3)  # linear limit


# ── coverage: does it actually span the two endpoints? ───────────────────

def test_lengthscale_range_covers_the_fitted_regime_and_stops_short_of_deployment():
    """DELIBERATELY NOT the deployed point, and this test exists to keep that
    decision from being quietly undone.

    The range was truncated from (0.3, 30) after the wide version measurably
    hurt: above ell ~ 3 a sequence has n_eff <= 2.2 independent action probes
    against the ~5 CMD_NOISE is calibrated to require, so the context is close
    to unidentifiable in principle -- AND those sequences carry ~1.9x the
    prediction loss, so they dominated the gradient while teaching least. A
    circuit trained on the wide range came out more conjunctive than the fitted
    one (31.2x conditioning gain against 8.15x).

    So covering the deployed rho = 0.92 is not the goal; every draw staying
    identifiable is. If someone widens this again to chase deployment coverage,
    this test should stop them long enough to read why.
    """
    lo, hi = T.ACTION_ELL_RANGE
    fitted = -1.0 / np.log(0.35)                  # task.RHO
    deployed = -1.0 / np.log(0.92)                # the trained policy
    assert lo < fitted < hi, (lo, fitted, hi)
    assert hi < deployed, (hi, deployed)
    # every draw stays above the ideal observer's ~5-probe requirement
    rho_worst = np.exp(-1.0 / hi)
    n_eff = T.SEQ_LEN * (1 - rho_worst) / (1 + rho_worst)
    assert n_eff > 5.0, (hi, rho_worst, n_eff)


def test_the_convention_stamp_tracks_the_parameter_ranges():
    """Two gp circuits trained on different ranges are identical in shape and
    architecture, so only the stamp can tell them apart -- and the ranges have
    already moved once. Derived, not hand-maintained, so a future change
    invalidates old checkpoints instead of leaving them loadable and wrong."""
    import model as M
    stamp = M.convention_for("gp")
    for lo, hi in (T.ACTION_ELL_RANGE, T.ACTION_GAIN_RANGE, T.ACTION_SCALE_RANGE):
        assert f"{lo:g}-{hi:g}" in stamp, (stamp, lo, hi)
    # ar1 adds no action stamp of its own: base convention + the readout's
    assert M.convention_for("ar1") == M.CONVENTION + M.readout_stamp(), "ar1 stamp must not move"
    # the legacy spelling resolves to the same stamp
    assert M.convention_for("fitted") == M.convention_for("ar1")
    assert M.convention_for() == stamp, "gp is the default"


def test_iid_mode_is_the_ar1_path_at_rho_zero():
    """The AR(0) control: independent actions every step, own stamp, own
    checkpoint path. Same marginal as ar1 (ACTION_SD), so only the correlation
    differs -- which is the manipulation."""
    import model as M
    a = D.generate_sequences(60, seq_len=40, seed=5, action_mode="iid")[0][:, :, 4:6].numpy()
    x = a[:, :-1].reshape(-1); y = a[:, 1:].reshape(-1)
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.03
    assert a.std() == pytest.approx(T.ACTION_SD, rel=0.05)
    b = D.generate_sequences(60, seq_len=40, seed=5, action_mode="ar1")[0][:, :, 4:6].numpy()
    assert np.corrcoef(b[:, :-1].reshape(-1), b[:, 1:].reshape(-1))[0, 1] == pytest.approx(T.RHO, abs=0.05)
    assert M.convention_for("iid") not in (M.CONVENTION, M.convention_for("gp"))
    assert M.arm_tag("main", "iid") == "main_iid"
    assert "iid" in M.ACTION_MODES


def test_gp_is_the_default_and_ar1_is_the_suffixed_variant():
    """Since 2026-09-16 the GP circuit is the unsuffixed checkpoint and the
    AR(1) one carries _ar1; "fitted" is only a legacy name for the latter."""
    import model as M
    assert M.DEFAULT_ACTION_MODE == "gp"
    assert M.arm_tag("main") == "main"
    assert M.arm_tag("main", "gp") == "main"
    assert M.arm_tag("main", "ar1") == "main_ar1"
    assert M.arm_tag("main", "fitted") == "main_ar1"
    assert M.ckpt_path("control", 2, "ar1").endswith("control_ar1_seed2.pt")
    assert M.ckpt_path("control", 2).endswith("control_seed2.pt")
    with pytest.raises(KeyError):
        M.arm_tag("main", "minutes")


def _kurtosis(a):
    x = a.ravel().astype(np.float64)
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def test_gain_spans_gaussian_to_saturated_in_SHAPE():
    """Scale-free, deliberately. Corner mass counts |a| past a FIXED 0.9, so it
    conflates shape with magnitude -- which is how the first version of this
    test failed against a renormalised warp. Kurtosis is 3 for a Gaussian and
    falls toward 1 as the marginal becomes two-point, whatever the sd."""
    rng = np.random.default_rng(3)
    def kurt(gain):
        return _kurtosis(T.action_sequence(
            30000, rng, dict(nu=0.5, ell=0.5, gain=gain, scale=1.0)))
    assert kurt(T.ACTION_GAIN_RANGE[0]) == pytest.approx(3.0, abs=0.15)
    assert kurt(T.ACTION_GAIN_RANGE[1]) < 1.6


def test_scale_range_brackets_both_measured_magnitudes():
    """Fitted actions sit at per-axis sd ACTION_SD; the trained policy at 1.53x
    that. Covering shape without covering magnitude would miss a third of the
    measured mismatch."""
    lo, hi = T.ACTION_SCALE_RANGE
    assert lo < 1.0 < hi, (lo, hi)
    assert lo < 1.53 < hi, (lo, hi)


def test_the_family_reaches_the_policys_corner_mass():
    """The end-to-end check the kurtosis test cannot make: with gain AND scale
    at their upper range, the family produces the joint saturation actually
    measured on the trained policy (corner mass 0.447)."""
    rng = np.random.default_rng(31)
    a = T.action_sequence(30000, rng, dict(nu=0.5, ell=0.5,
                                           gain=T.ACTION_GAIN_RANGE[1],
                                           scale=T.ACTION_SCALE_RANGE[1]))
    corner = float(((np.abs(a[:, 0]) > 0.9) & (np.abs(a[:, 1]) > 0.9)).mean())
    assert corner > 0.447, corner


def test_sampled_family_spans_a_wide_correlation_range():
    rng = np.random.default_rng(4)
    rhos = [_lag1(T.action_sequence(200, rng)) for _ in range(120)]
    assert min(rhos) < 0.2, min(rhos)
    assert max(rhos) > 0.85, max(rhos)


def test_smooth_kernels_at_long_lengthscale_do_not_blow_up():
    """nu=inf at ell=30 over 40 steps is numerically near-singular -- the case
    _chol's jitter ladder exists for, and one this family really produces."""
    rng = np.random.default_rng(5)
    a = T.action_sequence(40, rng, dict(nu=np.inf, ell=30.0, gain=1.0, scale=1.0))
    assert np.all(np.isfinite(a)) and a.shape == (40, 2)


def test_sequences_are_reproducible_from_the_seed():
    p = dict(nu=1.5, ell=3.0, gain=2.0, scale=1.2)
    a = T.action_sequence(50, np.random.default_rng(7), p)
    b = T.action_sequence(50, np.random.default_rng(7), p)
    assert np.array_equal(a, b)


# ── the existing path must not move ──────────────────────────────────────

def test_ar1_mode_is_bit_identical_under_both_spellings():
    """The AR(1) checkpoints were trained on this; if it moves they are all
    quietly invalid -- task.next_action's docstring documents a 5e-07 drift
    doing exactly that. "fitted" is the legacy spelling and must be the same
    bytes."""
    a = D.generate_sequences(6, seq_len=12, seed=11, action_mode="ar1")[0]
    b = D.generate_sequences(6, seq_len=12, seed=11, action_mode="fitted")[0]
    assert np.array_equal(a.numpy(), b.numpy())


def test_default_sequences_are_gp():
    d = D.generate_sequences(8, seq_len=20, seed=12)
    g = D.generate_sequences(8, seq_len=20, seed=12, action_mode="gp")
    assert all(np.array_equal(x.numpy(), y.numpy()) for x, y in zip(d, g))


def test_gp_and_ar1_differ_and_keep_the_shapes():
    f = D.generate_sequences(8, seq_len=20, seed=12, action_mode="ar1")
    g = D.generate_sequences(8, seq_len=20, seed=12, action_mode="gp")
    assert all(x.shape == y.shape for x, y in zip(f, g))
    assert not np.allclose(f[0].numpy(), g[0].numpy())


def test_gp_mode_marginal_sits_inside_the_scale_range():
    """Per-sequence sd now varies with the sampled scale, so the DATASET sd
    lands between ACTION_SD * (lo, hi) rather than on ACTION_SD."""
    g = D.generate_sequences(200, seq_len=40, seed=13, action_mode="gp")[0]
    sd = g[:, :, 4:6].numpy().std()
    lo, hi = T.ACTION_SCALE_RANGE
    assert T.ACTION_SD * lo < sd < T.ACTION_SD * hi, sd


def test_gp_actions_are_independent_of_the_context_stream():
    """Drawn jointly before the loop, so a context switch cannot show up in
    the actions -- the property 'fitted' mode maintains only by convention."""
    tr, ctx, tss = D.generate_sequences(200, seq_len=40, seed=14,
                                        action_mode="gp")
    acts = tr[:, :, 4:6].numpy()
    switch = (tss.numpy() == 0)
    switch[:, 0] = False                      # t=0 is not a switch
    if switch.sum() < 50:
        pytest.skip("too few switches to test")
    at_switch = np.linalg.norm(acts[switch], axis=-1)
    elsewhere = np.linalg.norm(acts[~switch], axis=-1)
    assert abs(at_switch.mean() - elsewhere.mean()) < 0.05 * elsewhere.mean()
