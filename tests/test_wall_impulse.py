"""The impulse-scaled wall penalty.

Two things are being protected. First that the impulse is the plant's own
quantity, computed where the clipping happens rather than reconstructed
downstream -- structure.py and blind_ceiling.py each carried a private copy of
the integration once, and both kept a damping term and a [0, GRID] clip long
after those were removed from task.py. Second that adding it changed nothing
for the callers that do not ask for it: every existing call site unpacks
exactly three values from env_step.
"""
import numpy as np
import pytest

import sac
import task as T


# ── the plant keeps its old shape unless asked ───────────────────────────

def test_env_step_still_returns_three_values_by_default():
    p = np.zeros(2, np.float32)
    v = np.array([0.05, 0.0], np.float32)
    a = np.array([0.5, 0.2], np.float32)
    assert len(T.env_step(p, v, a, 0)) == 3
    assert len(T.env_step_ab(p, v, a, np.eye(2, dtype=np.float32),
                             np.zeros((2, 2), np.float32))) == 3


def test_asking_for_the_impulse_does_not_change_the_step():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    p = np.array([0.9, 0.1], np.float32)
    v = np.array([0.08, 0.0], np.float32)
    a = np.array([1.0, 0.3], np.float32)
    three = T.env_step(p, v, a, 2, rng_a)
    four = T.env_step(p, v, a, 2, rng_b, return_impulse=True)
    for x, y in zip(three, four[:3]):
        assert np.allclose(x, y)


# ── the impulse itself ───────────────────────────────────────────────────

def test_no_contact_means_no_impulse():
    """Mid-grid, slow, pushing inward: nothing to hit."""
    p = np.zeros(2, np.float32)
    v = np.array([0.01, 0.0], np.float32)
    a = np.array([0.1, 0.0], np.float32)
    *_, imp = T.env_step(p, v, a, 0, return_impulse=True)
    assert imp == 0.0


def test_impulse_is_mass_times_the_velocity_the_wall_removed():
    """Driven into +x at the boundary: |J| = MASS * |v_x| that got zeroed."""
    p = np.array([T.POS_LIM - 1e-4, 0.0], np.float32)
    v = np.array([0.09, 0.0], np.float32)
    a = np.array([1.0, 0.0], np.float32)
    A, B = T.context_matrices(None)
    F = a @ A.T + v @ B.T - T.DAMPING * v
    v_pre = T.clip_speed(v + (F / T.MASS) * T.DT)
    pos_new, vel_new, _F, imp = T.env_step(p, v, a, None, return_impulse=True)
    assert vel_new[0] == 0.0, "the x component should have been zeroed"
    assert imp == pytest.approx(T.MASS * abs(v_pre[0]), rel=1e-5)


def test_impulse_scales_with_impact_speed():
    """The whole point: faster contact costs more."""
    p = np.array([T.POS_LIM - 1e-4, 0.0], np.float32)
    a = np.array([1.0, 0.0], np.float32)
    imps = [T.env_step(p, np.array([s, 0.0], np.float32), a, None,
                       return_impulse=True)[3]
            for s in (0.01, 0.03, 0.06, 0.09)]
    assert all(np.diff(imps) > 0), imps


def test_a_corner_contact_counts_both_axes():
    p = np.array([T.POS_LIM - 1e-4, T.POS_LIM - 1e-4], np.float32)
    v = np.array([0.06, 0.06], np.float32)
    a = np.array([1.0, 1.0], np.float32)
    *_, both = T.env_step(p, v, a, None, return_impulse=True)
    p1 = np.array([T.POS_LIM - 1e-4, 0.0], np.float32)
    *_, one = T.env_step(p1, np.array([0.06, 0.0], np.float32),
                         np.array([1.0, 0.0], np.float32), None,
                         return_impulse=True)
    assert both > one


def test_impulse_never_exceeds_the_normaliser():
    """MAX_WALL_IMPULSE must actually bound it, or the reward term leaves
    [0, 1] and the weight stops meaning what its docstring says."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3000):
        p = rng.uniform(-T.POS_LIM, T.POS_LIM, 2).astype(np.float32)
        v = rng.uniform(-T.V_MAX, T.V_MAX, 2).astype(np.float32)
        a = rng.uniform(-1.5, 1.5, 2).astype(np.float32)
        c = int(rng.integers(T.N_CONTEXTS))
        *_, imp = T.env_step(p, v, a, c, rng, return_impulse=True)
        worst = max(worst, float(imp))
    assert worst <= T.MAX_WALL_IMPULSE * (1 + 1e-5), (worst, T.MAX_WALL_IMPULSE)


# ── the reward term ──────────────────────────────────────────────────────

def _r(imp, d=0.5, d_prev=0.5):
    goal = np.zeros(2, np.float32)
    return sac.step_reward(np.array([d, 0.0], np.float32), goal,
                           np.array([d_prev, 0.0], np.float32),
                           wall_impulse=imp)[0]


def test_zero_impulse_reproduces_the_old_reward():
    """Every run in sac_runs/ was trained without this term; the no-contact
    path has to stay bit-identical or their curves stop being comparable."""
    goal = np.zeros(2, np.float32)
    p, pp = np.array([0.5, 0.0], np.float32), np.array([0.6, 0.0], np.float32)
    assert sac.step_reward(p, goal, pp)[0] == sac.step_reward(
        p, goal, pp, wall_impulse=0.0)[0]


def test_penalty_is_monotonic_in_impulse():
    # probes as FRACTIONS of the largest impulse the plant can deliver, so the
    # test does not silently invert when MASS (and with it MAX_WALL_IMPULSE)
    # moves -- fixed values of 2 and 5 did exactly that at MASS 32
    rs = [_r(f * T.MAX_WALL_IMPULSE) for f in (0.0, 0.25, 0.6, 1.0)]
    assert all(np.diff(rs) < 0), rs


def test_a_full_speed_impact_costs_exactly_the_weight():
    assert _r(0.0) - _r(T.MAX_WALL_IMPULSE) == pytest.approx(
        sac.WALL_IMPULSE_WEIGHT, rel=1e-5)


def test_the_penalty_does_not_touch_the_success_flag():
    goal = np.zeros(2, np.float32)
    inside = np.array([sac.GOAL_RADIUS / 2, 0.0], np.float32)
    _, done, _ = sac.step_reward(inside, goal, np.array([1.0, 0.0], np.float32),
                                 wall_impulse=T.MAX_WALL_IMPULSE)
    assert done


def test_weight_zero_disables_it(monkeypatch):
    monkeypatch.setattr(sac, "WALL_IMPULSE_WEIGHT", 0.0)
    assert _r(T.MAX_WALL_IMPULSE) == _r(0.0)


# ── wiring ───────────────────────────────────────────────────────────────

def test_run_sac_logs_wall_contact():
    import sweep_ablations as SW
    cfg = sac.AblationConfig(**SW.CONFIG_BY_NAME["baseline"])
    curve, _ = sac.run_sac(0, cfg, n_phase_ep=4, phase_reps=1, max_steps=12,
                           warmup_episodes=3, learning_starts=6, batch_size=6,
                           buffer_capacity=300)
    assert "wall_impulse" in curve and "wall_rate" in curve
    assert len(curve["wall_impulse"]) == len(curve["return"])
    assert all(0.0 <= w <= 1.0 for w in curve["wall_impulse"])
    assert all(0.0 <= w <= 1.0 for w in curve["wall_rate"])
