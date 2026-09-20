"""
Tests for blind_ceiling.py.

The module's job is to be believable when it says "a context-blind policy
already solves this", so the two things worth pinning down are that its
dynamics really are task.py's, and that its policy class really can express
each family's exact compensation -- otherwise "blind failed" could just mean
"the controller was too weak" or "the physics was subtly different".
"""
import numpy as np
import pytest

import blind_ceiling as BC
import sac
import task as T


def test_vectorized_step_matches_task_env_step():
    """The one that matters. A private copy of the dynamics drifting from the
    real ones is the exact failure that cost this project a sweep
    (sac.circuit_state), so this is pinned to task.env_step rather than to a
    hand-derived expectation."""
    rng = np.random.default_rng(0)
    for cid in range(T.N_CONTEXTS):
        params = tuple(T.CONTEXTS[cid][:4])
        pos = rng.uniform(0, T.GRID_SIZE, (7, 2))
        vel = rng.normal(0, 2.0, (7, 2))
        act = rng.uniform(-1, 1, (7, 2))

        pos_b, vel_b = BC.step_batch(pos, vel, act, params)
        for i in range(len(pos)):
            p, v, _F = T.env_step(pos[i].astype(np.float32), vel[i].astype(np.float32),
                                  act[i].astype(np.float32), *params)
            assert np.allclose(pos_b[i], p, atol=1e-5), (cid, i)
            assert np.allclose(vel_b[i], v, atol=1e-5), (cid, i)


def test_step_batch_clips_to_the_grid():
    params = tuple(T.CONTEXTS[0][:4])
    pos = np.array([[0.0, T.GRID_SIZE], [T.GRID_SIZE, 0.0]])
    vel = np.array([[-500.0, 500.0], [500.0, -500.0]])
    pos_new, _ = BC.step_batch(pos, vel, np.zeros((2, 2)), params)
    assert np.all(pos_new >= 0.0) and np.all(pos_new <= T.GRID_SIZE)


def test_naive_policy_aims_straight_at_the_goal():
    W = BC.naive_W()
    center, goals = BC.center_out()
    u = (goals - center) / np.linalg.norm(goals - center, axis=1)[:, None]
    feats = np.concatenate([u, np.zeros((len(goals), 2)), np.ones((len(goals), 1))], 1)
    assert np.allclose(np.clip(feats @ W.T, -1, 1), u)


def test_probe_geometry_matches_sac_eval_center_out():
    """These ceilings are only comparable to the transfer-matrix panels if the
    probe is the same one."""
    center, goals = BC.center_out()
    assert np.allclose(center, T.GRID_SIZE / 2)
    assert len(goals) == sac.EVAL_N_TARGETS
    assert np.allclose(np.linalg.norm(goals - center, axis=1), sac.EVAL_RADIUS)


def test_shortfall_is_zero_exactly_when_reached():
    """The optimization objective's defining property: it must stop caring
    once a target is inside GOAL_RADIUS, or the fit chases precision."""
    center, goals = BC.center_out()
    out = BC.rollout(BC.naive_W(), tuple(T.CONTEXTS[9][:4]), center, goals)  # drift +
    assert out["reached"].any() and np.allclose(out["shortfall"][out["reached"]], 0.0)
    assert np.all(out["shortfall"][~out["reached"]] > 0.0)


def test_naive_policy_solves_the_families_feedback_absorbs():
    """The finding the module exists to report: rotation at 45 degrees, curl
    and drift are all reachable with no context knowledge and no training,
    because the closed loop corrects each step."""
    center, goals = BC.center_out()
    for cid in (1, 3, 12, 14, 8, 9, 13, 15):
        out = BC.rollout(BC.naive_W(), tuple(T.CONTEXTS[cid][:4]), center, goals)
        assert out["reached"].all(), (cid, T.C_LABEL[cid], out["reached"])


def test_naive_policy_fails_on_rotation_at_90_and_on_reflection():
    """Deliberately NOT called "where context is required": an optimized blind
    policy still reaches every target under rot +-90 (bang-bang with velocity
    feedback), so these are hard for the NAIVE policy specifically. Only the
    reflections resist an optimized blind policy on success.

    rot +-90 puts the force perpendicular to the aim, so aiming at the goal
    makes no radial progress. Reflection is the identity only along its own
    mirror line, so exactly 2 of 8 equally spaced targets survive -- an
    arithmetic floor, not a tuning artifact."""
    center, goals = BC.center_out()
    for cid in (0, 2):
        out = BC.rollout(BC.naive_W(), tuple(T.CONTEXTS[cid][:4]), center, goals)
        assert not out["reached"].any(), (cid, T.C_LABEL[cid])
    for cid in (4, 5, 6, 7):
        out = BC.rollout(BC.naive_W(), tuple(T.CONTEXTS[cid][:4]), center, goals)
        assert out["reached"].sum() == 2, (cid, T.C_LABEL[cid], out["reached"].sum())


@pytest.mark.parametrize("cid", [0, 2, 4, 6])
def test_policy_class_can_express_each_familys_exact_compensation(cid):
    """Guards the module's central claim from being an artifact of a weak
    controller class: for a context the naive policy fails, the ANALYTIC
    inverse must be expressible in this policy family and must succeed.

      rotation    a = R(-theta) u
      reflection  a = M(phi) u      (M is its own inverse)
    """
    kind, param = T.CONTEXTS[cid][0], T.CONTEXTS[cid][1]
    W = np.zeros((2, BC.FEATURE_DIM))
    W[:, :2] = T.rot_mat(-param) if kind == 0 else T.refl_mat(param)

    center, goals = BC.center_out()
    out = BC.rollout(W, tuple(T.CONTEXTS[cid][:4]), center, goals)
    assert out["reached"].all(), (T.C_LABEL[cid], out["reached"])
    # and it is a straight shot, unlike the naive policy's curve
    assert np.nanmax(out["path_ratio"]) < 1.1


def test_fit_recovers_a_specialist_where_naive_fails():
    """End to end on the optimizer, not just the policy class: fitting a
    single hard context must beat the naive policy there."""
    center, goals = BC.center_out()
    W, _ = BC.fit([0], center=center, goals=goals, n_restarts=4)   # rot +90
    assert BC.evaluate(W, [0], center, goals)[0]["success"] > 0.5
    assert BC.evaluate(BC.naive_W(), [0], center, goals)[0]["success"] == 0.0


def test_headroom_is_zero_on_a_blind_solvable_set():
    """The decision the module supports: a set the blind policy already solves
    must report no headroom, so it is visibly not worth a sweep."""
    rows, _ = BC.ceilings([1, 3, 12, 14], n_restarts=3)
    assert max(r["headroom"] for r in rows) < 0.15, rows
    assert min(r["generalist"] for r in rows) > 0.85, rows


def test_success_and_path_headroom_can_disagree():
    """The module's main caveat, as a test. On rot +-90 an optimized blind
    policy reaches everything, so success headroom is ~0, while it needs a
    far longer path than a specialist -- so path headroom is large. A version
    of this module that reported only success would call that set
    uninformative."""
    rows, _ = BC.ceilings([0, 2], n_restarts=4)
    assert all(r["headroom"] < 0.15 for r in rows), rows
    assert all(r["headroom_path"] > 0.3 for r in rows), rows
    assert all(r["generalist_saturation"] > 0.4 for r in rows), (
        "the blind solution to rot +-90 is expected to be mostly saturated")


def test_a_reflection_pair_alone_is_blind_solvable():
    """Corrects an assumption this module was written with: reflections do NOT
    intrinsically resist a blind policy. One fitted policy serves refl 0 and
    refl 90 together at 0.875 and 1.00 -- against 0.25 and 0.00 for the same
    two contexts once rot +-90 is added to the set (see the next test). So
    what discriminates is the SET, not the family.

    The fit is deterministic (default_rng(0) starts, Powell), so these values
    are reproducible; thresholds sit just below them rather than at a round
    number, to make drift visible."""
    rows, _ = BC.ceilings([4, 6], n_restarts=4)
    assert all(r["generalist"] > 0.85 for r in rows), rows
    assert all(r["headroom"] < 0.2 for r in rows), rows


def test_mixing_rotation_and_reflection_creates_headroom():
    """rot +-90 and reflections cannot be served by one blind policy: the fit
    takes the rotations and gives up the reflections. This is the set that
    discriminates."""
    rows, _ = BC.ceilings([0, 2, 4, 6], n_restarts=4)
    by_id = {r["context_id"]: r for r in rows}
    assert all(by_id[c]["specialist"] > 0.9 for c in (0, 2, 4, 6)), rows
    assert max(by_id[c]["headroom"] for c in (4, 6)) > 0.5, rows


def test_specialist_transfer_diagonal_beats_off_diagonal():
    """The null model's defining shape. The diagonal is the specialist
    ceiling; the off-diagonal is what a memoryless policy keeps when the
    context changes under it, and is what an augmented config has to beat."""
    M, ids = BC.specialist_transfer([0, 2, 4, 6], n_restarts=4)
    assert M.shape == (4, 4)
    diag = np.diag(M)
    off = M[~np.eye(4, dtype=bool)]
    assert np.nanmean(diag) > 0.9, diag
    assert np.nanmean(diag) - np.nanmean(off) > 0.3, (diag, off)
