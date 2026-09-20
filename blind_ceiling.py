"""
blind_ceiling.py -- how far can a CONTEXT-BLIND controller get on a phase set?

Run this BEFORE spending a sweep. It answers, without training anything,
whether the contexts you picked can even distinguish the ablations.

THE PROBLEM IT CAUGHT. A sweep on rot +-45 and curl +-.5 showed `baseline`
reaching 1.00 success in every context at once, which reads as impossible for
an agent with no way to infer the context. It is not impossible: closed-loop
feedback absorbs those perturbations. Aiming straight at the goal under a 45
degree rotation still yields cos(45) = 0.71 of the progress each step, the
agent re-observes position and re-aims, and MAX_STEPS is ample. A blind policy
scores 8/8 there with ZERO training. Nothing about the circuit could have
shown up, because there was no headroom to show up in.

THE THREE NUMBERS, all for a MEMORYLESS REACTIVE policy (the class `baseline`
belongs to), computed by direct optimization of the policy's parameters:

  naive       aim straight at the goal. The zero-effort reference.
  specialist  best policy fitted to ONE context. This is what any config can
              reach on the transfer matrix's DIAGONAL, since within a block a
              memoryless policy only has to represent the current context's
              compensation -- which is a fixed function of the observation for
              every family here (rotation a = R(-theta)F*, reflection
              a = M(phi)F*, curl a = F* - k R(90)v, drift a = F* - g b).
  generalist  best SINGLE policy fitted to the whole set at once. This is the
              ceiling for a context-blind agent OFF the diagonal.

  headroom = specialist - generalist, on BOTH success and path ratio

HEADROOM IS THE WHOLE POINT. It is the most that knowing the context can buy
on this phase set. Where it is zero, no augmentation can beat `baseline` and
the comparison is uninformative no matter how good the circuit is. Where it is
large, `baseline` must trade contexts off against each other and a config that
identifies the context can win.

MEASURE IT ON BOTH METRICS, because they disagree and success saturates first.
On rot +-90 the naive policy scores 0/8 -- the force is perpendicular to the
aim, so aiming at the goal makes no radial progress -- yet an OPTIMIZED blind
policy still reaches every target, by running saturated bang-bang with strong
velocity feedback (gains in the hundreds, 76% of steps on the clip boundary).
Success headroom is therefore 0.00 there, while path headroom is 0.73: the
blind solution needs an 80% longer path than the specialist's straight shot.
Judging that set by success alone would call it uninformative when efficiency
still separates the configs widely. Hence the path panel here, and in the
transfer matrices.

SO WHAT MAKES A CONTEXT DISCRIMINATING is narrower than it looks. Not "the
naive policy fails" -- an optimized blind policy is much stronger than the
naive one. Only the REFLECTIONS resist it on success (0.25 and 0.00 against a
specialist's 1.00), which fits task.py's own note that reflection's required
angular offset 2*(phi - alpha) DEPENDS on target direction, so no single
fixed map serves two of them. rot +-45, curl and drift resist it on neither
metric.

    python blind_ceiling.py                  # the current PHASE_CONTEXT_IDS
    python blind_ceiling.py --all            # all 16 contexts, one by one
    python blind_ceiling.py --contexts 0 2 4 6
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize

import sac
import task as T

# The policy class: a = clip(W @ [u, DT*v, 1], -1, 1), W of shape (2, 5).
#
# u is the unit vector toward the goal and v the velocity. This family is
# deliberately just rich enough to express every family's exact compensation,
# so "the blind policy failed" can never be an artifact of too weak a
# controller:
#   aim rotation    a = R(-theta) u    the 2x2 block on u, det +1
#   aim reflection  a = M(phi) u       the same block, det -1
#   field drag      a = u + b v        the 2x2 block on v, diagonal
#   field curl      a = u - k R(90) v  the same block, antisymmetric
# The constant column is vestigial now -- it existed to cancel DRIFT, and drift
# is gone.  Kept because it costs two parameters and its presence makes the
# "too weak a controller" objection unavailable.
# It is also exactly the map an actor whose input is [pos, vel, goal] could
# learn if its first layer were linear, which is the point of comparison.
N_POLICY_PARAMS = 10
FEATURE_DIM = 5


def naive_W() -> np.ndarray:
    """Aim straight at the goal, full magnitude: a = u."""
    W = np.zeros((2, FEATURE_DIM), dtype=np.float64)
    W[0, 0] = W[1, 1] = 1.0
    return W


def center_out(n_targets=None, radius=None):
    """sac.eval_center_out's geometry, so these ceilings are comparable to the
    probe panels rather than to a differently-posed task."""
    n_targets = sac.EVAL_N_TARGETS if n_targets is None else n_targets
    radius = sac.EVAL_RADIUS if radius is None else radius
    center = np.zeros(2, dtype=np.float64)   # the grid centre is the origin
    angles = np.arange(n_targets) * 2 * np.pi / n_targets
    goals = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return center, goals


def step_batch(pos, vel, action, context_id, rng=None):
    """A batch of targets stepped at once.

    THIS USED TO BE A SECOND COPY OF THE DYNAMICS, kept for speed and, per its
    own docstring, "pinned to task.env_step by an equality test rather than by
    inspection -- a second copy that silently drifts from the real ones is
    precisely the failure mode that cost this project a sweep".

    It drifted anyway.  When the plant lost its damping and moved to [-1, 1],
    this copy kept `LAMBDA * vel` and `clip(..., 0, GRID_SIZE)`, and nothing
    complained because the equality test that was supposed to pin it went stale
    at the same moment.  A test cannot pin a duplicate it is not run against.

    task.env_step_ab is shape-polymorphic now, so there is no duplicate left to
    pin: this is a one-line delegation and the speed argument is moot.
    """
    A, B, w = T.context_affine(context_id)        # drift is a velocity increment
    pos_new, vel_new, _F = T.env_step_ab(pos, vel, action, A, B, rng, drift=w)
    return pos_new, vel_new


def rollout(W, context_id, center, goals, max_steps=None, rng=None):
    """One center-out probe under one context. Returns per-target arrays.

    `shortfall` is max(closest approach - GOAL_RADIUS, 0): zero exactly when
    the target was reached, and a graded distance when it was not. That makes
    it the natural optimization objective -- it stops rewarding precision once
    the goal is inside the radius, so the fit does not chase overshoot.
    """
    max_steps = sac.MAX_STEPS if max_steps is None else max_steps
    n = len(goals)
    pos = np.repeat(center[None, :], n, axis=0)
    vel = np.zeros((n, 2))
    closest = np.linalg.norm(goals - pos, axis=1)
    reached = np.zeros(n, dtype=bool)
    steps = np.full(n, max_steps, dtype=float)
    walked = np.zeros(n)
    saturated = []           # how often the action sits on the clip boundary

    for t in range(max_steps):
        d = goals - pos
        dist = np.linalg.norm(d, axis=1)
        closest = np.minimum(closest, dist)
        newly = (~reached) & (dist < sac.GOAL_RADIUS)
        if newly.any():
            steps[newly] = t
            reached |= newly
        if reached.all():
            break

        u = d / np.maximum(dist, 1e-9)[:, None]
        feats = np.concatenate([u, T.DT * vel, np.ones((n, 1))], axis=1)
        raw = feats @ W.T
        action = np.clip(raw, -1.0, 1.0)
        saturated.append(float(np.mean(np.abs(raw) >= 1.0)))

        pos_new, vel = step_batch(pos, vel, action, context_id, rng)
        walked = np.where(reached, walked, walked + np.linalg.norm(pos_new - pos, axis=1))
        pos = pos_new

    straight = np.linalg.norm(goals - center, axis=1) - sac.GOAL_RADIUS
    path_ratio = np.where(reached, walked / np.maximum(straight, 1e-9), np.nan)
    return {"reached": reached, "shortfall": np.maximum(closest - sac.GOAL_RADIUS, 0.0),
            "steps": steps, "path_ratio": path_ratio,
            "saturation": float(np.mean(saturated)) if saturated else float("nan")}


def cost(W, context_ids, center, goals, max_steps=None, step_weight=0.02):
    """Mean shortfall over contexts, with a light preference for reaching fast.

    The step term only breaks ties among policies that already succeed; it is
    small enough not to trade a reached target for a quicker miss.
    """
    max_steps = sac.MAX_STEPS if max_steps is None else max_steps
    total = 0.0
    for cid in context_ids:
        out = rollout(W, cid, center, goals, max_steps)
        total += out["shortfall"].mean() + step_weight * (out["steps"] / max_steps).mean()
    return total / len(context_ids)


def evaluate(W, context_ids, center=None, goals=None, max_steps=None):
    """Success and path ratio per context for a given policy."""
    if center is None or goals is None:
        center, goals = center_out()
    rows = {}
    for cid in context_ids:
        out = rollout(W, cid, center, goals, max_steps)
        rows[cid] = {"success": float(out["reached"].mean()),
                     "path_ratio": float(np.nanmean(out["path_ratio"]))
                                   if out["reached"].any() else float("nan"),
                     "saturation": out["saturation"]}
    return rows


def fit(context_ids, *, center=None, goals=None, max_steps=None, n_restarts=6,
        seed=0):
    """Best SINGLE policy over `context_ids` -- one context for a specialist
    ceiling, the whole set for the generalist one.

    Powell rather than a gradient method: the objective is piecewise smooth
    (clipping, the reached/not boundary), so derivatives are unreliable.
    Restarts include the naive policy, which is a strong start for the
    families feedback already handles, plus random draws to escape the local
    optimum that a rotation's wrong-signed block sits in.
    """
    if center is None or goals is None:
        center, goals = center_out()
    rng = np.random.default_rng(seed)
    starts = [naive_W()] + [naive_W() + rng.normal(0, 0.8, (2, FEATURE_DIM))
                            for _ in range(n_restarts - 1)]

    best_W, best_c = None, np.inf
    for W0 in starts:
        res = minimize(lambda w: cost(w.reshape(2, FEATURE_DIM), context_ids,
                                       center, goals, max_steps),
                       W0.ravel(), method="Powell",
                       options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-4})
        if res.fun < best_c:
            best_W, best_c = res.x.reshape(2, FEATURE_DIM), float(res.fun)
    return best_W, best_c


def ceilings(context_ids=None, **kw):
    """naive / specialist / generalist success per context, and the headroom.

    Returns (rows, generalist_W). `headroom` is specialist - generalist: the
    most that knowing the context can buy on this set.
    """
    context_ids = list(sac.PHASE_CONTEXT_IDS if context_ids is None else context_ids)
    center, goals = center_out()
    naive = evaluate(naive_W(), context_ids, center, goals)

    gen_W, _ = fit(context_ids, center=center, goals=goals, **kw)
    gen = evaluate(gen_W, context_ids, center, goals)

    rows = []
    for cid in context_ids:
        spec_W, _ = fit([cid], center=center, goals=goals, **kw)
        spec = evaluate(spec_W, [cid], center, goals)[cid]
        rows.append({"context_id": cid, "label": T.C_LABEL[cid],
                     "naive": naive[cid]["success"],
                     "naive_path": naive[cid]["path_ratio"],
                     "specialist": spec["success"],
                     "specialist_path": spec["path_ratio"],
                     "generalist": gen[cid]["success"],
                     "generalist_path": gen[cid]["path_ratio"],
                     "generalist_saturation": gen[cid]["saturation"],
                     "headroom": spec["success"] - gen[cid]["success"],
                     "headroom_path": gen[cid]["path_ratio"] - spec["path_ratio"]})
    return rows, gen_W


def specialist_transfer(context_ids=None, *, metric="success", **kw):
    """The transfer matrix a CONTEXT-BLIND agent must produce: fit a
    specialist to each context, then evaluate it in every context.

    This is the null model for transfer_analysis.py. Row i is the policy
    tuned for context i, column j is where it was evaluated, so the diagonal
    is the specialist ceiling and the off-diagonal is what a memoryless
    policy retains when the context changes underneath it -- computed without
    training anything. A sweep's `baseline` cannot do better than this, and a
    config that reads the context has to beat the OFF-DIAGONAL to have shown
    anything.

    It is also the honest version of the generalist number. `ceilings`
    optimizes the MEAN over contexts, which lets the fit abandon one context
    entirely to serve the others -- on [4, 5, 6, 7] it sacrifices refl 0 and
    takes 1.00 on the rest. That is a property of the objective, not of the
    task. Per-context specialists have no such degree of freedom.
    """
    context_ids = list(sac.PHASE_CONTEXT_IDS if context_ids is None else context_ids)
    center, goals = center_out()
    M = np.full((len(context_ids), len(context_ids)), np.nan)
    for i, train_id in enumerate(context_ids):
        W, _ = fit([train_id], center=center, goals=goals, **kw)
        scores = evaluate(W, context_ids, center, goals)
        for j, test_id in enumerate(context_ids):
            M[i, j] = scores[test_id][metric]
    return M, context_ids


def _short(label):
    """Context label for a table column.

    Labels were "C0: rot +90" and are now "rot -90" -- no colon -- so this
    takes the part after a colon IF there is one rather than indexing [1] and
    raising on the new form.
    """
    return label.split(":")[-1].strip()


def print_transfer(M, context_ids, metric="success"):
    labels = [T.C_LABEL[c] for c in context_ids]
    print(f"\nSpecialist transfer ({metric}) -- the context-blind null model")
    print(f"  {'trained on':18s} " + " ".join(f"{_short(l):>8s}"
                                               for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:18s} " + " ".join(f"{M[i, j]:8.2f}" for j in range(M.shape[1])))
    off = M[~np.eye(len(M), dtype=bool)]
    diag = np.diag(M)
    print(f"  {'':18s} diagonal {np.nanmean(diag):.2f}   "
          f"off-diagonal {np.nanmean(off):.2f}   "
          f"interference {np.nanmean(diag) - np.nanmean(off):+.2f}")
    print("  A config that reads the context must beat the OFF-DIAGONAL, not "
          "the diagonal.")


def print_table(rows, title):
    """Two headroom columns, because they can disagree and the success one
    saturates first.

    On rot +-90 a saturated bang-bang generalist reaches every target, so
    success headroom is 0 -- but it does so on a path 80% longer than the
    specialist's straight shot. Reading success alone would call that set
    uninformative when path efficiency still separates the configs by a wide
    margin. This is the same reason the transfer matrices carry a path panel.
    """
    print(f"\n{title}")
    print(f"  {'context':18s} {'naive':>6s} {'spec':>6s} {'gen':>6s} {'d.succ':>7s}"
          f"  | {'spec':>5s} {'gen':>5s} {'d.path':>7s} {'sat':>5s}")
    for r in rows:
        print(f"  {r['label']:18s} {r['naive']:6.2f} {r['specialist']:6.2f} "
              f"{r['generalist']:6.2f} {r['headroom']:7.2f}"
              f"  | {r['specialist_path']:5.2f} {r['generalist_path']:5.2f} "
              f"{r['headroom_path']:7.2f} {r['generalist_saturation']:5.2f}")
    h = np.array([r["headroom"] for r in rows])
    hp = np.array([r["headroom_path"] for r in rows], dtype=float)
    print(f"  {'MEAN':18s} {np.mean([r['naive'] for r in rows]):6.2f} "
          f"{np.mean([r['specialist'] for r in rows]):6.2f} "
          f"{np.mean([r['generalist'] for r in rows]):6.2f} {h.mean():7.2f}"
          f"  | {np.nanmean([r['specialist_path'] for r in rows]):5.2f} "
          f"{np.nanmean([r['generalist_path'] for r in rows]):5.2f} "
          f"{np.nanmean(hp):7.2f}")

    n_succ = int(np.sum(h > 0.25))
    n_path = int(np.sum(np.nan_to_num(hp, nan=np.inf) > 0.15))
    print(f"\n  discriminating on SUCCESS: {n_succ}/{len(rows)}   "
          f"on PATH EFFICIENCY: {n_path}/{len(rows)}")
    if n_succ == 0 and n_path == 0:
        print("  A context-blind policy handles this whole set as well as a "
              "specialist does.\n  A sweep on it cannot show anything about the "
              "circuit -- pick harder\n  contexts (--all shows every context's "
              "numbers).")
    elif n_succ == 0:
        print("  Success is at ceiling for a blind policy: report PATH RATIO as "
              "the primary\n  metric on this set, not success rate.")
    if np.nanmean([r["generalist_saturation"] for r in rows]) > 0.5:
        print("  NOTE: the generalist is mostly saturated (bang-bang). It wins "
              "success by\n  brute force, which is why its path cost is high.")


def main(context_ids=None, per_context=False, **kw):
    if per_context:
        rows = []
        for cid in range(T.N_CONTEXTS):
            rows += ceilings([cid], **kw)[0]
        print_table(rows, "Each context alone -- generalist == specialist here BY "
                          "CONSTRUCTION,\nso only the naive and path columns "
                          "carry information")
        return rows
    context_ids = list(sac.PHASE_CONTEXT_IDS if context_ids is None else context_ids)
    rows, _ = ceilings(context_ids, **kw)
    print_table(rows, f"Phase set {context_ids}")
    M, _ = specialist_transfer(context_ids, **kw)
    print_transfer(M, context_ids)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexts", nargs="+", type=int, default=None,
                    help="context ids to treat as the phase set "
                         "(default: sac.PHASE_CONTEXT_IDS)")
    ap.add_argument("--all", action="store_true", dest="per_context",
                    help="score all 16 contexts individually instead")
    ap.add_argument("--restarts", type=int, default=6)
    args = ap.parse_args()
    main(context_ids=args.contexts, per_context=args.per_context,
         n_restarts=args.restarts)
