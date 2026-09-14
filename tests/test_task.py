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

def test_contexts_are_two_flat_families():
    assert T.N_CONTEXTS == len(T.ROT_ANGLES) + len(T.REFL_ANGLES)
    assert set(T.FAMILY.values()) == {0, 1}
    per = [sum(1 for c in range(T.N_CONTEXTS) if T.FAMILY[c] == f)
           for f in range(len(T.FAM_NAMES))]
    assert per == [len(T.ROT_ANGLES), len(T.REFL_ANGLES)]
    assert len(set(T.C_LABEL.values())) == T.N_CONTEXTS       # labels unique


def test_context_table_and_lookups_agree():
    """FAMILY, C_PARAM and C_LABEL are three views of one table and must not
    drift -- rebuild each from CONTEXTS and compare."""
    for kind, param, cid, label in T.CONTEXTS:
        assert T.FAMILY[cid] == kind
        assert T.C_PARAM[cid] == param
        assert T.C_LABEL[cid] == label


def test_no_context_is_the_identity():
    """An identity context would sit at effect size zero and be
    indistinguishable from 'no perturbation', which is not a member of either
    sweep.  rot 0 is excluded from ROT_ANGLES for exactly that reason; refl 0
    is a genuine reflection, diag(1, -1), not the identity."""
    a = np.array([0.7, -0.3], dtype=np.float32)
    for c in range(T.N_CONTEXTS):
        assert not np.allclose(T.context_matrix(c) @ a, a, atol=1e-3), T.C_LABEL[c]


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


def test_families_split_on_determinant():
    """rotation (det +1) against reflection (det -1).  The contrast is KIND,
    not magnitude: |M(phi)a - a| = sqrt(2)|a| for every axis, and rotation
    matches that only at theta = +-90, so the families overlap in effect size
    while never overlapping in determinant."""
    for c in range(T.N_CONTEXTS):
        d = float(np.linalg.det(T.context_matrix(c)))
        assert (d > 0) == (T.FAMILY[c] == 0), T.C_LABEL[c]


def test_damping_is_plant_not_context():
    """DAMPING must be applied by the PLANT and appear in no context.

    It WAS a context dimension and is not one any more, so a context whose B
    carried damping would silently reintroduce the velocity-dependent factor
    that was removed.  Two halves: every context's B is zero, and env_step
    damps anyway.
    """
    for c in range(T.N_CONTEXTS):
        _A, B = T.context_matrices(c)
        assert np.allclose(B, 0.0), T.C_LABEL[c]

    v = np.array([0.05, 0.0], dtype=np.float32)
    _p, v2, _F = T.env_step(np.zeros(2, np.float32), v,
                            np.zeros(2, np.float32), 0)     # no action
    assert np.allclose(v2, (1.0 - T.DAMPING / T.MASS) * v, atol=1e-6)
    assert float(np.linalg.norm(v2)) < float(np.linalg.norm(v))


def test_force_is_exactly_A_action():
    rng = np.random.default_rng(0)
    for c in range(T.N_CONTEXTS):
        A = T.context_matrix(c)
        for _ in range(20):
            a = rng.standard_normal(2).astype(np.float32)
            v = rng.uniform(-T.V_MAX, T.V_MAX, 2).astype(np.float32)
            assert np.allclose(T.context_force(a, v, c), A @ a, atol=1e-5)


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
    # `refl 0` is diag(1, -1), so an x-only action stays x-only and the y axis
    # is genuinely untouched.  A rotation context would turn the action and put
    # force on y, which is the plant working, not the clip leaking.
    c = next(c for c in range(T.N_CONTEXTS) if T.C_LABEL[c] == "refl 0")
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
            for _t in range(30):
                a = T.next_action(a, rng)
                clean = T.context_force(a, v, c) - T.DAMPING * v
                pn, vn, _ = T.env_step(p, v, a, c, rng)
                res.append(vn - v - (clean / T.MASS) * T.DT)
                p, v = pn, vn
        rms.append(float(np.sqrt((np.array(res) ** 2).sum(-1).mean())))
    assert max(rms) / min(rms) < 1.05, rms


# ── calibration guards ───────────────────────────────────────────────────

def _one_step_effect(c, n=90, seed=0):
    """rms|dvel_c - dvel_null| / rms|dvel_null| at matched state and action,
    where `null` is the untransformed action under the same plant."""
    rng = np.random.default_rng(seed)
    I = np.eye(2, dtype=np.float32)
    Z = np.zeros((2, 2), dtype=np.float32)
    A, B = T.context_matrices(c)
    num, den = [], []
    for _ in range(n):
        p = rng.uniform(-0.9, 0.9, 2).astype(np.float32)
        v = np.zeros(2, np.float32)
        a = T.initial_action(rng)
        for _t in range(T.SEQ_LEN):
            a = T.next_action(a, rng)
            _, vc, _ = T.env_step_ab(p, v, a, A, B)
            _, vn, _ = T.env_step_ab(p, v, a, I, Z)
            num.append(((vc - vn) ** 2).sum())
            den.append((vn ** 2).sum())
            p, v, _ = T.env_step(p, v, a, c)
    return float(np.sqrt(np.mean(num)) / np.sqrt(np.mean(den)))


def test_family_effect_sizes_are_where_geometry_says():
    """THE TWO FAMILIES ARE NOT MATCHED AND CANNOT BE.  Reflections sit at
    sqrt(2)|a| whatever the axis, so their effect size is constant; rotations
    are graded as 2|sin(theta/2)||a| and cross the reflections only at +-90.
    So pin reflection's flatness and rotation's grading, rather than assert a
    balance the geometry forbids.  Loose bounds: this catches an angle table
    drifting, not a third decimal.
    """
    rot = [_one_step_effect(c) for c in range(T.N_CONTEXTS) if T.FAMILY[c] == 0]
    ref = [_one_step_effect(c) for c in range(T.N_CONTEXTS) if T.FAMILY[c] == 1]
    assert max(ref) / min(ref) < 1.10, ref          # reflections: flat
    # rotations are graded, but the SIZE of the grading is set by which angles
    # are in the table: +-45/+-90 gives 2 sin(45)/2 sin(22.5) = 1.85, so the
    # bound has to sit below that rather than at a round 2.0
    assert max(rot) / min(rot) > 1.6, rot
    # THE FAMILIES MEET AT +-90, and that is an identity rather than a
    # coincidence: 2 sin(45) = sqrt(2), so the strongest rotation in the table
    # has exactly the reflection effect size.  (They would only EXCEED it if
    # the table carried angles past +-90, which it no longer does.)
    assert abs(max(rot) - np.mean(ref)) < 0.05 * np.mean(ref), (max(rot), ref)
    assert min(rot) < 0.7 * min(ref), (min(rot), min(ref))


def test_the_plant_is_not_identifiable_from_one_step():
    """WITHOUT NOISE AN IDEAL OBSERVER SOLVES IT ALMOST AT ONCE, so the task
    would never require history.  With CMD_NOISE it should be well below
    ceiling at k=2 and near it by k=10, the mean block length."""
    def acc(k, noisy, n=320, seed=0):
        rng = np.random.default_rng(seed)
        cor = tot = 0
        for c in range(T.N_CONTEXTS):
            m = n // T.N_CONTEXTS + 1
            A_, DV, MK = [], [], []
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
                    # context is the ACTION's contribution alone
                    A_.append(a.copy())
                    DV.append(vn - v + (T.DAMPING / T.MASS) * v * T.DT)
                    MK.append(0.0 if hit else 1.0)
                    p, v = pn, vn
            s = (m, k, 2)
            A_ = np.array(A_).reshape(s)
            DV = np.array(DV).reshape(s)
            MK = np.array(MK).reshape(m, k, 1)
            ll = np.zeros((m, T.N_CONTEXTS))
            for cc in range(T.N_CONTEXTS):
                pred = (A_ @ T.context_matrix(cc).T) / T.MASS * T.DT
                ll[:, cc] = -((MK * (DV - pred) ** 2).sum(axis=(1, 2)))
            cor += (ll.argmax(1) == c).sum()
            tot += m
        return cor / tot

    # PIN THE PROPERTY, NOT THE LEVEL.  Absolute thresholds here encode one
    # particular CMD_NOISE and one particular context count, and both have
    # moved: halving the table from 16 contexts to 8 took the noisy two-step
    # accuracy from below 0.80 to 0.90 without anything being wrong.  What must
    # stay true is the SHAPE -- deterministic is solved at once, noise costs
    # something, and history recovers it.
    clean1 = acc(1, noisy=False)
    noisy1, noisy10 = acc(1, noisy=True), acc(10, noisy=True)
    assert clean1 > 0.95, clean1                  # no noise: one step suffices
    assert noisy1 < clean1 - 0.15, (noisy1, clean1)   # noise costs real accuracy
    assert noisy10 > noisy1 + 0.15, (noisy1, noisy10)  # history recovers it
    assert noisy10 > 0.80, noisy10                # and gets there within a block
