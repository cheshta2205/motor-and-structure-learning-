"""task.py's contexts and its double-integrator plant.

Every other module is built against this file, so a silent change here would
propagate into sixteen retrained checkpoints before anything complained.

Three kinds of test, deliberately separated:

  STRUCTURAL   the context table and its lookups -- things that must hold by
               construction and would be bugs if they did not.
  LOAD-BEARING properties other parts of the design ARGUE FROM, above all that
               every context matrix is ORTHOGONAL.  The claim that command
               noise cannot leak the context rests entirely on |A eta| = |eta|;
               if a context ever stopped being orthogonal that argument would
               silently become false.
  CALIBRATION  measured effect sizes and the ideal-observer curve.  These are
               measurements of the plant, not settings, so they are checked
               loosely -- they exist to catch drift, not to pin values.
"""
import ast
import inspect

import numpy as np
import pytest

import task as T


def _spread(per_family=2):
    """A few context ids spanning both families, derived from the table.

    Hardcoded ids are what broke when the table was resized -- and the failure
    is only loud when they run off the end.  Ids that stay in range simply
    test different contexts than intended, silently.
    """
    out = []
    for f in range(len(T.FAM_NAMES)):
        mem = [c for c in range(T.N_CONTEXTS) if T.FAMILY[c] == f]
        step = max(1, len(mem) // per_family)
        out += mem[::step][:per_family]
    return out


# ── structural ───────────────────────────────────────────────────────────

def test_contexts_are_three_flat_families_of_four():
    assert T.FAM_NAMES == ["rotation", "curl", "drift"]
    assert T.N_CONTEXTS == len(T.ROT_ANGLES) + len(T.CURL_GAINS) + len(T.DRIFT_ANGLES) == 12
    # over the TABLE: registered off-table contexts (task.rotation_id,
    # task.register_linear) share the dict under ids >= EXTRA_ID_BASE
    assert set(T.FAMILY[c] for c in range(T.N_CONTEXTS)) == {0, 1, 2}
    per = [sum(1 for c in range(T.N_CONTEXTS) if T.FAMILY[c] == f)
           for f in range(len(T.FAM_NAMES))]
    assert per == [4, 4, 4]
    # labels unique over the TABLE; registered off-table rotations
    # (task.rotation_id) share these dicts under ids >= EXTRA_ID_BASE
    table_labels = [T.C_LABEL[c] for c in range(T.N_CONTEXTS)]
    assert len(set(table_labels)) == T.N_CONTEXTS
    assert all(T.is_table_context(c) for c in range(T.N_CONTEXTS))
    assert T.ROT_ANGLES == [-60.0, -30.0, 30.0, 60.0]


def test_each_family_switches_on_one_term_of_the_law():
    """rotation moves A alone, curl moves B alone (an antisymmetric matrix:
    a force perpendicular to the velocity), drift moves the velocity
    increment alone. That is what makes family a DIRECTION in the space of
    force laws -- the property the reflection family lacked."""
    I = np.eye(2)
    for c in range(T.N_CONTEXTS):
        A, B, w = T.context_affine(c)
        kind = T.FAMILY[c]
        assert (not np.allclose(A, I)) == (kind == 0), T.C_LABEL[c]
        assert (not np.allclose(B, 0)) == (kind == 1), T.C_LABEL[c]
        assert (not np.allclose(w, 0)) == (kind == 2), T.C_LABEL[c]
        if kind == 1:
            assert np.allclose(B + B.T, 0), T.C_LABEL[c]          # perpendicular
        if kind == 2:
            assert np.linalg.norm(w) == pytest.approx(T.DRIFT_VEL, rel=1e-5)


def test_context_table_and_lookups_agree():
    """FAMILY, C_PARAM and C_LABEL are three views of one table and must not
    drift -- rebuild each from CONTEXTS and compare."""
    for kind, param, cid, label in T.CONTEXTS:
        assert T.FAMILY[cid] == kind
        assert T.C_PARAM[cid] == param
        assert T.C_LABEL[cid] == label


def test_no_context_is_the_identity():
    """An identity context would sit at effect size zero and be
    indistinguishable from 'no perturbation', which is not a member of any
    sweep.  rot 0, curl 0 and a zero drift are excluded for exactly that
    reason. Judged on the WHOLE law: curl and drift leave A alone, so the
    action transform is not where their effect is."""
    a = np.array([0.7, -0.3], dtype=np.float32)
    v = np.array([0.03, 0.02], dtype=np.float32)
    for c in range(T.N_CONTEXTS):
        A, B, w = T.context_affine(c)
        same_force = np.allclose(A @ a + B @ v, a, atol=1e-3)
        assert not (same_force and np.allclose(w, 0)), T.C_LABEL[c]


def test_the_null_field_sentinel_is_the_identity():
    """c=None must mean F = a, and it must work through EVERY entry point.

    REGRESSION.  This guard has been lost once already -- an edit to the
    surrounding block dropped it, and the failure surfaced only in sac.py's
    WARMUP phase as a TypeError from int(None), inside a ProcessPoolExecutor,
    three modules away from the cause.  The warmup needs a no-perturbation
    condition to fill a replay buffer without teaching a context, and no
    context in the table is the identity, so the sentinel is load-bearing.
    """
    a = np.array([0.7, -0.3], dtype=np.float32)
    v = np.array([0.02, 0.01], dtype=np.float32)
    assert np.allclose(T.context_matrix(None), np.eye(2))
    A, B = T.context_matrices(None)
    assert np.allclose(A, np.eye(2)) and np.allclose(B, 0.0)
    assert np.allclose(T.context_force(a, v, None), a)
    # and end to end, the path sac.py actually takes.  env_step returns the
    # force ACTUALLY APPLIED, which includes the plant's damping -- so it is
    # a - DAMPING*v, not a.  context_force is the plant-free half.
    p1, v1, F = T.env_step(np.zeros(2, np.float32), v, a, None)
    assert np.allclose(F, a - T.DAMPING * v, atol=1e-5)
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(v1))
    # with an rng too, since the warmup runs with noise on
    assert np.all(np.isfinite(
        T.env_step(np.zeros(2, np.float32), v, a, None,
                   np.random.default_rng(0))[1]))


# ── load-bearing ─────────────────────────────────────────────────────────

def test_every_context_matrix_is_orthogonal():
    """THE LOAD-BEARING PROPERTY.  Command noise is eta ~ N(0, (sigma|a|)^2)
    injected BEFORE the context transform, so what reaches the plant is A eta.
    Only orthogonality makes |A eta| = |eta|, and that is the entire reason the
    noise amplitude cannot itself identify the context."""
    for c in range(T.N_CONTEXTS):
        A = T.context_matrix(c)
        assert np.allclose(A @ A.T, np.eye(2), atol=1e-6), T.C_LABEL[c]
        assert abs(abs(np.linalg.det(A)) - 1.0) < 1e-6, T.C_LABEL[c]


def test_every_action_transform_preserves_the_determinant_sign():
    """No context reverses orientation any more (the reflection family is
    retired): every A is a proper rotation or the identity, det +1."""
    for c in range(T.N_CONTEXTS):
        assert float(np.linalg.det(T.context_matrix(c))) == pytest.approx(1.0, abs=1e-6), T.C_LABEL[c]


def test_damping_is_plant_not_context():
    """DAMPING must be applied by the PLANT and appear in no context.

    It WAS a context dimension and is not one any more, so a context whose B
    carried damping would silently reintroduce the drag factor that was
    removed.  Curl's B is the one velocity-dependent term allowed and it is
    ANTISYMMETRIC, so it never damps (v . B v = 0); every other B is zero.
    And env_step damps anyway, in every context.
    """
    for c in range(T.N_CONTEXTS):
        _A, B = T.context_matrices(c)
        if T.FAMILY[c] == 1:
            assert np.allclose(B + B.T, 0.0), T.C_LABEL[c]
        else:
            assert np.allclose(B, 0.0), T.C_LABEL[c]

    v = np.array([0.05, 0.0], dtype=np.float32)
    _p, v2, _F = T.env_step(np.zeros(2, np.float32), v,
                            np.zeros(2, np.float32), 0)     # rotation, no action
    assert np.allclose(v2, (1.0 - T.DAMPING / T.MASS) * v, atol=1e-6)
    assert float(np.linalg.norm(v2)) < float(np.linalg.norm(v))


def test_force_is_exactly_A_action_plus_B_velocity():
    rng = np.random.default_rng(0)
    for c in range(T.N_CONTEXTS):
        A, B = T.context_matrices(c)
        for _ in range(20):
            a = rng.standard_normal(2).astype(np.float32)
            v = rng.uniform(-T.V_MAX, T.V_MAX, 2).astype(np.float32)
            assert np.allclose(T.context_force(a, v, c), A @ a + B @ v, atol=1e-5)


def test_drift_is_a_velocity_increment_not_a_force():
    """The drift adds to the velocity AFTER the force has acted, so the
    returned F does not contain it and the next velocity does -- by exactly w,
    off the cap and the walls."""
    c = next(c for c in range(T.N_CONTEXTS) if T.C_LABEL[c] == "drift 90")
    _A, _B, w = T.context_affine(c)
    p = np.zeros(2, np.float32); v = np.array([0.01, 0.0], np.float32); a = np.array([0.2, 0.0], np.float32)
    pn, vn, F = T.env_step(p, v, a, c, rng=None)
    assert np.allclose(F, a - T.DAMPING * v, atol=1e-6)                    # F is action + damping only
    assert np.allclose(vn, v + (F / T.MASS) * T.DT + w, atol=1e-6)         # w rides on top
    assert w[1] > 0 and abs(w[0]) < 1e-7                                    # drift 90 pushes +y


def test_force_never_depends_on_position():
    """Position is a nuisance coordinate -- context_force does not even take
    one, so this is a signature test that fails loudly if a positional field
    is ever added."""
    assert "pos" not in inspect.signature(T.context_force).parameters


def test_force_law_is_plant_free():
    """A context is a statement about FORCES; the plant alone says what forces
    do.  No plant constant may appear in the force law, DAMPING included --
    otherwise a context's definition would move when the plant changed."""
    src = ""
    for fn in (T.context_force, T.context_matrices, T.context_matrix):
        # strip docstrings first: they NAME these constants in order to say the
        # code must not use them, so a raw text search matches its own comment
        tree = ast.parse(inspect.getsource(fn).lstrip())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Module)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:]
        src += ast.unparse(tree)
    for const in ("MASS", "DT", "V_MAX", "POS_LIM", "DAMPING"):
        assert const not in src, f"{const} leaked into the force law"


# ── the plant ────────────────────────────────────────────────────────────

def test_rng_none_is_deterministic_and_seeds_reproduce():
    p = np.zeros(2, np.float32)
    v = np.zeros(2, np.float32)
    a = np.array([0.6, 0.2], np.float32)
    d1, d2 = T.env_step(p, v, a, 5), T.env_step(p, v, a, 5)
    assert all(np.array_equal(x, y) for x, y in zip(d1, d2))

    n1 = T.env_step(p, v, a, 5, np.random.default_rng(7))
    n2 = T.env_step(p, v, a, 5, np.random.default_rng(7))
    assert all(np.array_equal(x, y) for x, y in zip(n1, n2))
    assert not np.array_equal(d1[1], n1[1])      # noise actually does something


def test_speed_cap_and_position_bounds_always_hold():
    rng = np.random.default_rng(0)
    # one from each family, derived so a resized table cannot break it
    for c in _spread():
        p = rng.uniform(-1, 1, 2).astype(np.float32)
        v = np.zeros(2, np.float32)
        for _ in range(400):
            a = rng.standard_normal(2).astype(np.float32)
            p, v, _F = T.env_step(p, v, a, c, rng)
            assert np.linalg.norm(v) <= T.V_MAX + 1e-5
            assert np.all(np.abs(p) <= T.POS_LIM + 1e-6)


def test_env_step_is_shape_polymorphic():
    """One plant, single vectors and batches alike.  blind_ceiling.py and
    structure.py each carried their own batched copy and both went stale --
    keeping one implementation is what stops that recurring."""
    A, B = T.context_matrices(_spread()[0])
    p1 = np.array([.1, .2], np.float32)
    v1 = np.array([.03, .01], np.float32)
    a1 = np.array([.6, .2], np.float32)
    one = T.env_step_ab(p1, v1, a1, A, B)
    many = T.env_step_ab(np.stack([p1] * 3), np.stack([v1] * 3),
                         np.stack([a1] * 3), A, B)
    assert all(t.shape == (3, 2) for t in many)
    assert all(np.allclose(x, y[0], atol=1e-7) for x, y in zip(one, many))
    # the cap is PER ROW, not Frobenius over the batch
    V = np.array([[0.09, 0.0], [0.5, 0.5], [0.0, 0.02]], np.float32)
    assert np.allclose(np.linalg.norm(T.clip_speed(V), axis=-1),
                       [0.09, T.V_MAX, 0.02], atol=1e-6)


def test_velocity_is_zeroed_on_the_axis_that_clips():
    """Inelastic collision.  Without it an agent would keep integrating force
    into the wall and need as many steps to escape as it spent arriving."""
    # `drift 0` leaves the action alone and pushes along x, so an x-only action
    # stays x-only and the y axis is genuinely untouched.  A rotation context
    # would turn the action and put force on y, which is the plant working,
    # not the clip leaking.
    c = next(c for c in range(T.N_CONTEXTS) if T.C_LABEL[c] == "drift 0")
    p = np.array([T.POS_LIM - 1e-4, 0.0], np.float32)
    v = np.array([T.V_MAX, 0.0], np.float32)
    a = np.array([1.0, 0.0], np.float32)
    pn, vn, _ = T.env_step(p, v, a, c)
    assert pn[0] == pytest.approx(T.POS_LIM)
    assert vn[0] == 0.0                                   # the clipping axis
    assert vn[1] == pytest.approx(0.0, abs=1e-6)          # the other one intact

    # and the y axis is NOT zeroed when only x clips
    v2 = np.array([T.V_MAX * 0.5, T.V_MAX * 0.5], np.float32)
    _pn2, vn2, _ = T.env_step(p, v2, np.zeros(2, np.float32), c)
    assert vn2[0] == 0.0 and abs(float(vn2[1])) > 0.0


def test_position_is_exactly_pos_plus_vel_in_the_interior():
    """pos' = pos + vel'*DT is what makes the next-state distribution a rigid
    translate in position, which is what justifies model.ALPHA."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.uniform(-0.5, 0.5, 2).astype(np.float32)   # far from any wall
        v = rng.uniform(-0.05, 0.05, 2).astype(np.float32)
        a = rng.standard_normal(2).astype(np.float32)
        pn, vn, _ = T.env_step(p, v, a, int(rng.integers(T.N_CONTEXTS)))
        assert np.allclose(pn, p + vn * T.DT, atol=1e-6)


def test_command_noise_magnitude_does_not_depend_on_the_context():
    """The consequence of orthogonality, measured end to end rather than
    argued: the injected disturbance must be the same size in every context,
    or the noise itself would identify it."""
    rms = []
    for c in _spread():
        rng = np.random.default_rng(0)
        res = []
        for _ in range(250):
            p = rng.uniform(-0.5, 0.5, 2).astype(np.float32)
            v = np.zeros(2, np.float32)
            a = T.initial_action(rng)
            _A, _B, w = T.context_affine(c)
            for _t in range(30):
                a = T.next_action(a, rng)
                clean = T.context_force(a, v, c) - T.DAMPING * v
                pn, vn, _ = T.env_step(p, v, a, c, rng)
                # only steps the cap and the walls left alone carry the noise
                # unclipped; the drift increment is known and subtracted
                free = (np.linalg.norm(vn) < T.V_MAX * 0.999
                        and np.all(np.abs(pn) < T.POS_LIM - 1e-6))
                if free:
                    res.append(vn - v - (clean / T.MASS) * T.DT - w)
                p, v = pn, vn
        rms.append(float(np.sqrt((np.array(res) ** 2).sum(-1).mean())))
    # the bound is looser than the 1.05 it was at MASS 100 / DAMPING 17.7:
    # at MASS 32 / DAMPING 1 the cap binds on ~25% of steps and dropping the
    # clipped ones selects on the noise itself (outward kicks get clipped and
    # removed), by a context-dependent amount. The property -- the injected
    # noise has the same size in every context, |A eta| = |eta| -- is exact
    # by construction; this test guards the plant from breaking it grossly.
    assert max(rms) / min(rms) < 1.15, rms


# ── calibration guards ───────────────────────────────────────────────────

def _one_step_effect(c, n=90, seed=0):
    """rms|dvel_c - dvel_null| / rms|dvel_null| at matched state and action,
    where `null` is the untransformed action under the same plant."""
    rng = np.random.default_rng(seed)
    I = np.eye(2, dtype=np.float32)
    Z = np.zeros((2, 2), dtype=np.float32)
    A, B, w = T.context_affine(c)
    num, den = [], []
    for _ in range(n):
        p = rng.uniform(-0.9, 0.9, 2).astype(np.float32)
        v = np.zeros(2, np.float32)
        a = T.initial_action(rng)
        for _t in range(T.SEQ_LEN):
            a = T.next_action(a, rng)
            # compare UNCLIPPED one-step velocity changes, so the cap does not
            # flatten the contrast between families at this mass
            fc = A @ a + B @ v - T.DAMPING * v
            fn = a - T.DAMPING * v
            vc = v + (fc / T.MASS) * T.DT + w
            vn = v + (fn / T.MASS) * T.DT
            num.append(((vc - vn) ** 2).sum())
            den.append((vn ** 2).sum())
            p, v, _ = T.env_step(p, v, a, c)
    return float(np.sqrt(np.mean(num)) / np.sqrt(np.mean(den)))


def test_family_effect_sizes_are_matched_across_families():
    """THE THREE FAMILIES ARE CALIBRATED TO ONE ANOTHER (task.py's family
    section): the low level of each -- rot +-30, curl +-7, one drift -- is
    the same effect size, the high level of rotation and curl is ~2x that,
    and drift is flat across its four directions. Loose bounds: this catches
    a table or plant constant moving without the others being re-matched,
    not a third decimal.
    """
    by_fam = {f: [_one_step_effect(c) for c in range(T.N_CONTEXTS) if T.FAMILY[c] == f]
              for f in range(3)}
    rot, curl, drift = by_fam[0], by_fam[1], by_fam[2]
    assert max(drift) / min(drift) < 1.10, drift                 # drift: flat
    assert max(rot) / min(rot) > 1.6, rot                        # graded: 2 sin30 / 2 sin15 = 1.93
    assert max(curl) / min(curl) > 1.5, curl                     # graded ~1.9 (velocity-dependent)
    # low levels matched: exactly in force units (task.py's table), within
    # ~1.5x in this one-step velocity measure, where curl's and drift's
    # effects ride on the velocity they themselves build up
    lows = [min(rot), min(curl), np.mean(drift)]
    assert max(lows) / min(lows) < 1.5, lows
    assert abs(max(rot) - max(curl)) < 0.3 * max(rot), (max(rot), max(curl))


def test_the_plant_is_not_identifiable_from_one_step():
    """WITHOUT NOISE AN IDEAL OBSERVER SOLVES IT ALMOST AT ONCE, so the task
    would never require history.  With CMD_NOISE it should be well below
    ceiling at k=2 and near it by k=10, the mean block length."""
    def acc(k, noisy, n=320, seed=0):
        rng = np.random.default_rng(seed)
        cor = tot = 0
        laws = [T.context_affine(cc) for cc in range(T.N_CONTEXTS)]
        for c in range(T.N_CONTEXTS):
            m = n // T.N_CONTEXTS + 1
            A_, V_, DV, MK = [], [], [], []
            for _ in range(m):
                p = rng.uniform(-0.9, 0.9, 2).astype(np.float32)
                v = np.zeros(2, np.float32)
                a = T.initial_action(rng)
                for _t in range(k):
                    a = T.next_action(a, rng)
                    pn, vn, _ = T.env_step(p, v, a, c, rng if noisy else None)
                    hit = (np.any(np.abs(pn) >= T.POS_LIM - 1e-6)
                           or np.linalg.norm(vn) >= T.V_MAX * 0.999)
                    # the damping is known, so subtract it: what identifies the
                    # context is the law's contribution alone
                    A_.append(a.copy()); V_.append(v.copy())
                    DV.append(vn - v + (T.DAMPING / T.MASS) * v * T.DT)
                    MK.append(0.0 if hit else 1.0)
                    p, v = pn, vn
            s = (m, k, 2)
            A_ = np.array(A_).reshape(s); V_ = np.array(V_).reshape(s)
            DV = np.array(DV).reshape(s)
            MK = np.array(MK).reshape(m, k, 1)
            ll = np.zeros((m, T.N_CONTEXTS))
            for cc, (A, B, w) in enumerate(laws):
                pred = (A_ @ A.T + V_ @ B.T) / T.MASS * T.DT + w
                ll[:, cc] = -((MK * (DV - pred) ** 2).sum(axis=(1, 2)))
            cor += (ll.argmax(1) == c).sum()
            tot += m
        return cor / tot

    # PIN THE PROPERTY, NOT THE LEVEL.  Absolute thresholds here encode one
    # particular CMD_NOISE and one particular context count, and both have
    # moved.  What must stay true is the SHAPE -- deterministic is solved
    # within a few steps, noise costs something, and history recovers it.
    # ONE clean step no longer suffices by design: curl acts on velocity,
    # which starts at zero, so the four curls are identical on the first step
    # from rest (chance among them: 9/12 = 0.75 is the one-step ceiling).
    clean1, clean3 = acc(1, noisy=False), acc(3, noisy=False)
    noisy1, noisy10 = acc(1, noisy=True), acc(10, noisy=True)
    assert 0.65 < clean1 < 0.9, clean1            # the rotations and drifts, not the curls
    assert clean3 > 0.9, clean3                   # a little velocity separates the curls
    assert noisy1 < clean1 - 0.1, (noisy1, clean1)     # noise costs real accuracy
    assert noisy10 > noisy1 + 0.15, (noisy1, noisy10)  # history recovers it
    assert noisy10 > 0.75, noisy10                # and gets most of the way within a block
