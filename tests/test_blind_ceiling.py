"""
Tests for blind_ceiling.py.

The module's job is to be believable when it says "a context-blind policy
already solves this", so the thing worth pinning down is that its dynamics
really are task.py's -- otherwise "blind failed" could just mean "the physics
was subtly different".

PRUNED 2026-09-20. This file was written against a SIXTEEN-context table with
a REFLECTION family, and against a grid whose centre was GRID_SIZE/2. None of
that survives: the table is twelve contexts over rotation/curl/drift, position
lives in [-POS_LIM, +POS_LIM] so the centre is the origin, and context ids
12-15 do not exist. Nine tests asserted things about reflections, about rot
+-90, and about specific context ids -- claims with no referent now -- and had
been failing for some time. They are in git history.

What survives is what is generic over whatever the current table happens to
be. The calling convention was also stale throughout: step_batch/rollout take
a scalar context id, not the unpacked `tuple(T.CONTEXTS[cid][:4])` that an
older task.env_step signature wanted.
"""
import numpy as np

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
        pos = rng.uniform(-T.POS_LIM, T.POS_LIM, (7, 2))
        vel = rng.normal(0, 2.0, (7, 2))
        act = rng.uniform(-1, 1, (7, 2))

        pos_b, vel_b = BC.step_batch(pos, vel, act, cid)
        for i in range(len(pos)):
            p, v, _F = T.env_step(pos[i].astype(np.float32),
                                  vel[i].astype(np.float32),
                                  act[i].astype(np.float32), cid)
            assert np.allclose(pos_b[i], p, atol=1e-5), (cid, i)
            assert np.allclose(vel_b[i], v, atol=1e-5), (cid, i)


def test_step_batch_clips_to_the_grid():
    """Position is clipped to [-POS_LIM, +POS_LIM] on both axes, as in
    task.env_step -- a wall, not a wrap."""
    pos = np.array([[-T.POS_LIM, T.POS_LIM], [T.POS_LIM, -T.POS_LIM]])
    vel = np.array([[-500.0, 500.0], [500.0, -500.0]])
    pos_new, _ = BC.step_batch(pos, vel, np.zeros((2, 2)), 0)
    assert np.all(pos_new >= -T.POS_LIM) and np.all(pos_new <= T.POS_LIM)


def test_naive_policy_aims_straight_at_the_goal():
    W = BC.naive_W()
    center, goals = BC.center_out()
    u = (goals - center) / np.linalg.norm(goals - center, axis=1)[:, None]
    feats = np.concatenate([u, np.zeros((len(goals), 2)), np.ones((len(goals), 1))], 1)
    assert np.allclose(np.clip(feats @ W.T, -1, 1), u)


def test_probe_geometry_matches_sac_eval_center_out():
    """These ceilings are only comparable to the transfer-matrix panels if the
    probe is the same one. The centre is the ORIGIN: position lives in
    [-POS_LIM, +POS_LIM], so GRID_SIZE is an extent, not an offset."""
    center, goals = BC.center_out()
    assert np.allclose(center, 0.0)
    assert len(goals) == sac.EVAL_N_TARGETS
    assert np.allclose(np.linalg.norm(goals - center, axis=1), sac.EVAL_RADIUS)


def test_shortfall_is_zero_exactly_when_reached():
    """The optimization objective's defining property: it must stop caring
    once a target is inside GOAL_RADIUS, or the fit chases precision."""
    center, goals = BC.center_out()
    drift = next(i for i, c in enumerate(T.CONTEXTS) if c[0] == 2)
    out = BC.rollout(BC.naive_W(), drift, center, goals)
    assert out["reached"].any() and np.allclose(out["shortfall"][out["reached"]], 0.0)
    assert np.all(out["shortfall"][~out["reached"]] > 0.0)
