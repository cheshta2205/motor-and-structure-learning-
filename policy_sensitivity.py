"""
policy_sensitivity.py -- does the trained policy (and critic) actually USE
each context channel?

The transfer matrix cannot answer this. A low off-diagonal there is equally
consistent with "the circuit's z was wrong in that context" and with "z was
right and the policy ignored it" -- and we know it is the second, because
eval_center_out re-infers z live in whichever context it is evaluating (see
circuit_probe: 6-way decodability ~1.00 under the SAC rollout). What is
missing is a measurement of whether the policy's OUTPUT depends on z at all.

So this module intervenes on one channel at a time, holding the dynamics
fixed, and asks what changes.

THREE MEASUREMENTS, weakest assumptions last:

  behavioral   Run the center-out probe in context C, but feed the actor a
               z_ec taken from a DIFFERENT context. If the policy is
               sensitive, the trajectories must move. This is the one that
               answers the question directly, in the units the experiment
               cares about.
  ablation     Replace a channel with zeros, or shuffle it across contexts,
               and read off the change in success and path ratio.
  gradient     ||d output / d channel|| scaled by the channel's own measured
               sd -- "how far does the action move per realistic unit of
               variation in this channel". Cheap, and it separates the ACTOR
               from the CRITIC, which the behavioral probe cannot: a policy
               that ignores a channel its value function does use is a
               different finding from neither using it.

WHY TRAJECTORY DIVERGENCE, NOT SUCCESS, IS THE PRIMARY READOUT. Success is
bounded and, at the current GOAL_RADIUS, sits near the floor for most configs,
so it hides moderate behavioral change. The distance between the intervened
and control trajectories is unbounded below and above and moves continuously,
so it registers an effect that success would round to zero.

WHY THE INTERVENTION PATCHES sac.augment_obs. That is the single point every
observation passes through -- the rollout's and the one-step lookahead's -- so
patching it keeps ONE copy of the loop. A private rollout that reimplemented
the stepping is exactly the failure that cost this project a sweep (see
sac.circuit_state), and it would be worse here, since the whole measurement is
a difference between two rollouts that must be identical apart from the
channel under test.

    python policy_sensitivity.py --run-dir sac_runs --configs main_frozen oracle
"""
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

import sac
import task as T

N_DONOR_EPISODES = 1        # the probe is deterministic, so one pass suffices
CHANNELS = ("kin", "oracle", "ec", "cb")


# ── channel bookkeeping ──────────────────────────────────────────────────

def channel_slices(spec: sac.ObsSpec) -> dict:
    """Where each input source lives in the observation vector."""
    out, start = {}, 0
    for name, width in zip(CHANNELS, [spec.kin, spec.oracle, spec.ec, spec.cb]):
        if width:
            out[name] = slice(start, start + width)
        start += width
    return out


@contextmanager
def obs_hook(transform=None, record=None):
    """Intervene on and/or record every observation the rollout builds.

    `transform(z_ec, z_cb, oracle_vec) -> (z_ec, z_cb, oracle_vec)` runs before
    the observation is assembled, so a substituted channel is what the actor,
    the critics and the replay lookahead all see -- no path can slip past it.
    """
    real = sac.augment_obs

    def patched(base_obs, config, z_ec=None, z_cb=None, oracle_vec=None):
        if transform is not None:
            z_ec, z_cb, oracle_vec = transform(z_ec, z_cb, oracle_vec)
        obs = real(base_obs, config, z_ec=z_ec, z_cb=z_cb, oracle_vec=oracle_vec)
        if record is not None:
            record.append(np.array(obs, copy=True))
        return obs

    sac.augment_obs = patched
    try:
        yield
    finally:
        sac.augment_obs = real


def _substitute(channel, value):
    """A transform that replaces one channel with a fixed vector."""
    def transform(z_ec, z_cb, oracle_vec):
        if channel == "ec" and z_ec is not None:
            z_ec = np.asarray(value, dtype=np.float32)
        elif channel == "cb" and z_cb is not None:
            z_cb = np.asarray(value, dtype=np.float32)
        elif channel == "oracle" and oracle_vec is not None:
            oracle_vec = np.asarray(value, dtype=np.float32)
        return z_ec, z_cb, oracle_vec
    return transform


# ── running the probe under an intervention ──────────────────────────────

def probe(agent, ec, cb, context_id, *, transform=None, record=None,
          max_steps=None, device="cpu"):
    """One center-out probe in `context_id`, optionally intervened on.

    Uses sac.eval_center_out unchanged, so the geometry, the deterministic
    tanh-mean action and the path-ratio normalization are the same ones the
    training probes use and the numbers are directly comparable to them.
    """
    family_idx = sac.PHASE_CONTEXT_IDS.index(context_id)
    trainer = SimpleNamespace(actor=agent.actor)
    with obs_hook(transform, record):
        return sac.eval_center_out(
            trainer, ec, cb, agent.config, T.CONTEXTS[context_id][:4], family_idx,
            max_steps=max_steps, cb_proj=agent.cb_proj,
            oracle_embed=agent.oracle_embed, device=device)


def channel_means(agent, ec, cb, context_ids, *, max_steps=None, device="cpu"):
    """Mean value of each channel per context, from control rollouts.

    These are the donors for the substitution test: substituting a context's
    MEAN rather than a sampled value isolates the context-identifying part of
    the channel from its within-context (trajectory-driven) variation, which
    for z_ec is 31% of its total variance.
    """
    slices = channel_slices(agent.spec)
    means = {}
    for cid in context_ids:
        rec = []
        probe(agent, ec, cb, cid, record=rec, max_steps=max_steps, device=device)
        obs = np.stack(rec)
        means[cid] = {name: obs[:, sl].mean(0) for name, sl in slices.items()}
    return means


def _nanmean(x):
    """np.nanmean without the warning when everything is NaN -- which happens
    legitimately: path_ratio is NaN for every target when none was reached."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")


def trajectory_divergence(a, b):
    """Mean distance between two probes' trajectories, over shared steps.

    NaN-padded tails are dropped per target, so a pair that diverges in LENGTH
    (one reaches, the other does not) is compared only where both exist; the
    success and path-ratio columns carry that difference instead.
    """
    ta, tb = a["trajectories"], b["trajectories"]
    per_target = []
    for k in range(ta.shape[0]):
        xa, xb = ta[k], tb[k]
        ok = ~np.isnan(xa[:, 0]) & ~np.isnan(xb[:, 0])
        if ok.sum() > 1:
            per_target.append(float(np.mean(np.linalg.norm(xa[ok] - xb[ok], axis=1))))
    return float(np.mean(per_target)) if per_target else float("nan")


def behavioral_sensitivity(agent, ec, cb, context_ids, *, max_steps=None,
                           device="cpu"):
    """Substitute each context channel with another context's mean, and with
    zeros, and measure how far the behavior moves.

    Reported per (evaluated context, channel, intervention):
      divergence  mean trajectory distance from the control rollout, in grid
                  units. GOAL_RADIUS is the natural yardstick -- a divergence
                  far below it is a policy that barely noticed.
      d_success / d_path   the same intervention in the units of the transfer
                  matrix, for comparability with it.
    """
    slices = channel_slices(agent.spec)
    donors = channel_means(agent, ec, cb, context_ids, max_steps=max_steps,
                           device=device)
    rows = []
    for cid in context_ids:
        control = probe(agent, ec, cb, cid, max_steps=max_steps, device=device)
        others = [c for c in context_ids if c != cid]
        for name in slices:
            if name == "kin":
                continue              # the task's own input, not a context channel
            # "self" is the control that makes "swap" interpretable: it
            # removes the channel's live within-context variation exactly as a
            # swap does, but preserves context identity. divergence(swap) -
            # divergence(self) is therefore the part attributable to being
            # told a DIFFERENT context, rather than to losing the live signal.
            #
            # "zero" is deliberately last and least trusted: z_ec carries a
            # large context-INDEPENDENT component (71% of the energy of its
            # context means is the grand mean) that the trunk's bias has
            # adapted to, so zeroing perturbs far more than context identity.
            for tag, value in (
                    ("self", donors[cid][name]),
                    ("swap", np.mean([donors[o][name] for o in others], axis=0)),
                    ("zero", np.zeros_like(donors[cid][name]))):
                out = probe(agent, ec, cb, cid, transform=_substitute(name, value),
                            max_steps=max_steps, device=device)
                rows.append({
                    "context": T.C_LABEL[cid], "channel": name, "intervention": tag,
                    "divergence": trajectory_divergence(control, out),
                    "d_success": float(out["success"].mean() - control["success"].mean()),
                    "d_path": _nanmean(out["path_ratio"])
                              - _nanmean(control["path_ratio"]),
                    "control_success": float(control["success"].mean()),
                })
    return rows


# ── gradient sensitivity ─────────────────────────────────────────────────

def gradient_sensitivity(agent, obs: np.ndarray, device="cpu"):
    """Expected output change per 1-sd perturbation of each channel.

    ||J_block|| * sd_block, where J is the Jacobian of the output with respect
    to that block of the observation and sd is the channel's own measured
    spread. Scaling by sd is what makes channels comparable: a large raw
    gradient on a channel that barely varies moves the output no more than a
    small gradient on one that varies a lot.

    Actor output is the deterministic tanh-mean action; critic output is
    Q(s, a) at that action. Both are reported, because they can disagree.
    """
    slices = channel_slices(agent.spec)
    X = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)
    sd = X.std(dim=0)

    def jac_norms(fn, n_out):
        """Per-block ||dfn/dx|| * sd, averaged over samples."""
        x = X.clone().requires_grad_(True)
        y = fn(x)
        grads = []
        for j in range(n_out):
            g, = torch.autograd.grad(y[:, j].sum(), x, retain_graph=(j < n_out - 1))
            grads.append(g)
        G = torch.stack(grads, dim=1)             # [B, n_out, obs_dim]
        scaled = G * sd.view(1, 1, -1)
        return {name: float(scaled[:, :, sl].norm(dim=(1, 2)).mean())
                for name, sl in slices.items()}

    def actor_mean(x):
        mean, _log_std = agent.actor(x)
        return torch.tanh(mean)

    with torch.no_grad():
        action = actor_mean(X)

    out = {"actor": jac_norms(actor_mean, sac.ACTION_DIM)}
    for tag, net in (("qf1", agent.qf1), ("qf2", agent.qf2)):
        out[tag] = jac_norms(lambda x, net=net: net(x, action), 1)
    return out


def directional_sensitivity(agent, obs, labels, device="cpu"):
    """Does the policy respond to the BETWEEN-context direction, or only to
    within-context variation?

    The plain gradient number scales by a channel's TOTAL sd, which mixes the
    two -- so a policy reading only trajectory-driven structure in z scores
    just as high as one reading context. This separates them by moving along
    each subspace explicitly:

      between  unit step from one context's mean toward another's, averaged
               over ordered pairs -- the direction that IS the context
      within   unit step along an actual within-context deviation, same length

    Reported as the resulting change in the actor's action. If `between` is
    far below `within`, the channel is being read for state, not for context,
    and swapping the context can leave behavior untouched while freezing the
    channel changes it a great deal -- the pattern that otherwise looks
    contradictory.

    `scale` columns give the typical real length of each kind of step, so
    between_effect * between_scale is the action change an actual context
    switch produces.
    """
    slices = channel_slices(agent.spec)
    X = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)
    y = np.asarray(labels)
    ctx = np.unique(y)

    x = X.clone().requires_grad_(True)
    mean, _ = agent.actor(x)
    action = torch.tanh(mean)
    grads = []
    for j in range(sac.ACTION_DIM):
        g, = torch.autograd.grad(action[:, j].sum(), x,
                                 retain_graph=(j < sac.ACTION_DIM - 1))
        grads.append(g)
    J = torch.stack(grads, dim=1).detach()            # [B, action_dim, obs_dim]

    rng = np.random.default_rng(0)
    out = {}
    for name, sl in slices.items():
        block = X[:, sl].cpu().numpy()
        mus = np.stack([block[y == c].mean(0) for c in ctx])

        between = [mus[b] - mus[a] for a in range(len(ctx)) for b in range(len(ctx))
                   if a != b]
        idx = rng.integers(0, len(block), size=min(256, len(block)))
        within = [block[i] - mus[list(ctx).index(y[i])] for i in idx]

        def effect(dirs):
            """Mean ||J . d_hat|| over unit directions and over samples."""
            D = np.stack([d / max(np.linalg.norm(d), 1e-12) for d in dirs])
            Dt = torch.tensor(D, dtype=torch.float32, device=device)
            # [B, action_dim, n_dirs] -> norm over action dims
            resp = torch.einsum("baj,dj->bad", J[:, :, sl], Dt)
            return float(resp.norm(dim=1).mean())

        out[name] = {
            "between": effect(between), "within": effect(within),
            "between_scale": float(np.mean([np.linalg.norm(d) for d in between])),
            "within_scale": float(np.mean([np.linalg.norm(d) for d in within])),
        }
    return out


# ── parameter overlap between contexts ───────────────────────────────────

def _flat_grad(output, params):
    g = torch.autograd.grad(output, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                      for gi, p in zip(g, params)])


def gradient_overlap(agent, obs_by_context, *, which="actor", scope="all",
                     n_samples=16, device="cpu", seed=0):
    """How much do two contexts share PARAMETERS, not just outputs?

    A policy can be strongly context-dependent and still forget catastrophically:
    sensitivity says the FUNCTION differs at z_A and z_B, protection requires
    the PARAMETERS producing those two outputs to be largely disjoint. Nothing
    forces them to coincide, and this measures the difference.

    To first order, a gradient step taken in context B changes context A's loss
    by -eta * (grad L_A . grad L_B). So the normalized overlap

        cos( d f(x_A)/d theta ,  d f(x_B)/d theta )

    reads as: >0 training B moves A (they share weights -- with conflicting
    targets, that is interference), ~0 training B is neutral for A (isolation,
    what protection requires), <0 training B actively opposes A.

    Gradients are taken on each context's OWN observations, as they actually
    arise during that block, rather than on synthetic matched inputs -- the
    quantity that governs forgetting is the overlap of the gradients the
    optimizer really receives.

    SCOPE MATTERS, because a standard MLP cannot reach zero. The output head
    is shared by every input -- its bias gradient is input-independent, so that
    cosine is exactly 1.0 whatever the context -- which puts an architectural
    FLOOR on the whole-network number. Measured on a synthetic actor whose
    first layer was planted to be perfectly disjoint across contexts, the
    scopes read:

        scope="first"   0.000   the layer the context channel enters
        scope="all"     0.170   the same actor, through the shared head

    So compare configs at scope="first" to ask whether the CONTEXT CODE buys
    isolation, and at scope="all" for the total exposure to interference. An
    "all" value near 0.17 here means as isolated as this architecture permits,
    not weak isolation.

    CALIBRATION, measured on planted arrangements (see the tests):

        codes / units                        first    all    activation
        orthogonal codes, disjoint units     0.000   0.170      0.000
        orthogonal codes, shared units       0.500   0.999      1.000
        COLLINEAR codes,  shared units       0.947   0.997      1.000

    The middle row is why scope="first" is not itself a sufficient summary:
    orthogonal codes still share the first layer's BIAS gradient when they
    drive the same units. The bottom row is the regime a large common-mode
    component produces -- z's context means carry 71% of their energy in one
    shared vector, which makes two contexts nearly collinear at the input and
    puts them on the same weights however distinguishable they are downstream.

    Returns (matrix, context_ids) with the mean normalized overlap per pair.
    """
    rng = np.random.default_rng(seed)
    ctx = list(obs_by_context)
    net = {"actor": agent.actor, "qf1": agent.qf1, "qf2": agent.qf2}[which]
    if scope == "first":
        first = agent.actor.backbone[0] if which == "actor" else net.net[0]
        params = [p for p in first.parameters() if p.requires_grad]
    elif scope == "all":
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        raise ValueError(f"scope must be 'all' or 'first', got {scope!r}")

    def outputs(x):
        if which == "actor":
            mean, _ = agent.actor(x)
            return torch.tanh(mean)
        with torch.no_grad():
            mean, _ = agent.actor(x)
            action = torch.tanh(mean)
        return net(x, action)

    # Gradients are kept SEPARATE PER OUTPUT DIMENSION and only ever compared
    # within a dimension. The action head has its own weights per dimension, so
    # d a_0/d theta and d a_1/d theta are near-orthogonal (~-0.02 measured) for
    # reasons that have nothing to do with context; pooling them across
    # dimensions halves every overlap and drags the diagonal from 1.0 to ~0.2.
    per_ctx = []
    for c in ctx:
        obs = np.asarray(obs_by_context[c], dtype=np.float32)
        idx = rng.choice(len(obs), size=min(n_samples, len(obs)), replace=False)
        X = torch.tensor(obs[idx], device=device)
        by_dim = None
        for i in range(len(X)):
            y = outputs(X[i:i + 1])
            if by_dim is None:
                by_dim = [[] for _ in range(y.shape[1])]
            for j in range(y.shape[1]):
                g = _flat_grad(y[0, j], params)
                n = g.norm()
                by_dim[j].append(g / n if n > 0 else g)
        per_ctx.append([torch.stack(d) for d in by_dim])

    k, n_out = len(ctx), len(per_ctx[0])
    M = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(k):
            M[a, b] = float(np.mean([float((per_ctx[a][j] @ per_ctx[b][j].T).mean())
                                     for j in range(n_out)]))
    return M, ctx


def activation_overlap(agent, obs_by_context, device="cpu"):
    """Fraction of first-hidden-layer units co-active across contexts.

    The cheap, interpretable companion to gradient_overlap: a ReLU unit that is
    off contributes no gradient, so two contexts that activate disjoint units
    cannot interfere through them. High co-activation is the mechanism by which
    a large shared input component -- z's context means carry 71% of their
    energy in a single common vector -- forces both contexts onto the same
    weights.
    """
    ctx = list(obs_by_context)
    acts = {}
    with torch.no_grad():
        for c in ctx:
            X = torch.tensor(np.asarray(obs_by_context[c], dtype=np.float32),
                             device=device)
            h = agent.actor.backbone[1](agent.actor.backbone[0](X))
            acts[c] = (h > 0).float().mean(0)          # activation rate per unit

    k = len(ctx)
    M = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(k):
            pa, pb = acts[ctx[a]], acts[ctx[b]]
            both = torch.minimum(pa, pb).sum()
            either = torch.maximum(pa, pb).sum()
            M[a, b] = float(both / torch.clamp(either, min=1e-12))
    return M, ctx


# ── driver ───────────────────────────────────────────────────────────────

def analyze(agent_path, *, max_steps=None, device="cpu"):
    agent = sac.load_agent(agent_path, device=device)
    ec, cb = sac.build_ec_cb(agent.config, device)
    context_ids = list(sac.PHASE_CONTEXT_IDS)

    rec, labels = [], []
    for cid in context_ids:
        before = len(rec)
        probe(agent, ec, cb, cid, record=rec, max_steps=max_steps, device=device)
        labels.extend([cid] * (len(rec) - before))
    obs, labels = np.stack(rec), np.asarray(labels)
    obs_by_context = {c: obs[labels == c] for c in context_ids}

    return {"agent": agent,
            "behavioral": behavioral_sensitivity(agent, ec, cb, context_ids,
                                                 max_steps=max_steps, device=device),
            "gradient": gradient_sensitivity(agent, obs, device=device),
            "directional": directional_sensitivity(agent, obs, labels, device=device),
            "grad_overlap": {f"{w}/{sc}": gradient_overlap(
                                 agent, obs_by_context, which=w, scope=sc,
                                 device=device)
                             for w in ("actor", "qf1") for sc in ("first", "all")},
            "act_overlap": activation_overlap(agent, obs_by_context, device=device),
            "channel_sd": {name: float(obs[:, sl].std())
                           for name, sl in channel_slices(agent.spec).items()}}


def print_report(name, seed, res):
    agent = res["agent"]
    print(f"\n=== {name} seed {seed} "
          f"(obs {agent.spec.raw_dim}: " +
          ", ".join(f"{k} {v.stop - v.start}"
                    for k, v in channel_slices(agent.spec).items()) + ") ===")

    print("\ngradient sensitivity -- output change per 1-sd of each channel")
    print(f"  {'output':8s} " + "".join(f"{c:>12s}" for c in res['gradient']['actor']))
    for tag, vals in res["gradient"].items():
        print(f"  {tag:8s} " + "".join(f"{vals[c]:12.4f}" for c in vals))
    print("  channel sd " + "".join(f"{res['channel_sd'][c]:12.4f}"
                                     for c in res["gradient"]["actor"]))

    def _matrix(M, ids, title, note):
        # labels carry no "Cn:" prefix any more; [-1] works for both forms
        labels = [T.C_LABEL[c].split(":")[-1].strip() for c in ids]
        print(f"\n{title}")
        print(f"  {'':14s}" + "".join(f"{l:>10s}" for l in labels))
        for i, l in enumerate(labels):
            print(f"  {l:14s}" + "".join(f"{M[i, j]:10.3f}" for j in range(len(ids))))
        off = M[~np.eye(len(ids), dtype=bool)]
        print(f"  mean off-diagonal {np.nanmean(off):+.3f}   {note}")

    for which, (M, ids) in res["grad_overlap"].items():
        note = ("0 = isolated (protects), >0 = shared"
                if which.endswith("/first")
                else "floor is ~0.17 here: the output head is shared by every input")
        _matrix(M, ids, f"gradient overlap ({which}) -- parameter sharing "
                        f"between contexts", note)
    _matrix(*res["act_overlap"],
            "first-layer activation overlap", "fraction of units co-active")

    print("\ndirectional -- actor response to a UNIT step along each subspace")
    print(f"  {'channel':8s} {'between':>9s} {'within':>9s} {'ratio':>7s} "
          f"{'|between|':>10s} {'|within|':>9s} {'real between':>13s}")
    for name, v in res["directional"].items():
        ratio = v["between"] / max(v["within"], 1e-12)
        print(f"  {name:8s} {v['between']:9.4f} {v['within']:9.4f} {ratio:7.2f} "
              f"{v['between_scale']:10.3f} {v['within_scale']:9.3f} "
              f"{v['between'] * v['between_scale']:13.4f}")
    print("  ratio << 1 means the channel is read for STATE, not for context:")
    print("  freezing it would change behavior while switching context would not.")

    print(f"\nbehavioral -- GOAL_RADIUS = {sac.GOAL_RADIUS} is the yardstick "
          f"for divergence")
    print(f"  {'context':16s} {'ctrl_succ':>9s} {'channel':8s} {'interv':7s} "
          f"{'divergence':>11s} {'d_success':>10s} {'d_path':>8s}")
    for r in res["behavioral"]:
        print(f"  {r['context']:16s} {r['control_success']:9.2f} {r['channel']:8s} "
              f"{r['intervention']:7s} {r['divergence']:11.4f} "
              f"{r['d_success']:10.2f} {r['d_path']:8.3f}")
    if all(r["control_success"] == 0.0 for r in res["behavioral"]):
        print("  WARNING: the control policy never succeeds anywhere, so "
              "divergence here\n  measures an unstable trajectory being "
              "perturbed, not context being used.")

    by = {}
    for r in res["behavioral"]:
        by.setdefault((r["channel"], r["intervention"]), []).append(r["divergence"])
    print("\n  mean divergence by channel:")
    for k, v in sorted(by.items()):
        d = _nanmean(v)
        print(f"    {k[0]:8s} {k[1]:7s} {d:8.4f}   "
              f"({d / sac.GOAL_RADIUS:5.2f} x GOAL_RADIUS)")

    print("\n  context-identity effect = divergence(swap) - divergence(self),")
    print("  i.e. the part of the change attributable to being told a "
          "DIFFERENT context:")
    for channel in sorted({k[0] for k in by}):
        if (channel, "swap") in by and (channel, "self") in by:
            eff = _nanmean(by[(channel, "swap")]) - _nanmean(by[(channel, "self")])
            print(f"    {channel:8s} {eff:+8.4f}   "
                  f"({eff / sac.GOAL_RADIUS:+6.2f} x GOAL_RADIUS)")


def main(run_dir=None, configs=("main", "oracle"), seeds=None,
         max_steps=None, device="cpu"):
    run_dir = sac.RUN_DIR if run_dir is None else run_dir
    results = {}
    for name in configs:
        cfg_dir = os.path.join(run_dir, name)
        if not os.path.isdir(cfg_dir):
            print(f"no runs for {name} under {run_dir}")
            continue
        for entry in sorted(os.listdir(cfg_dir)):
            if not entry.startswith("seed_"):
                continue
            seed = int(entry.split("_")[1])
            if seeds is not None and seed not in seeds:
                continue
            path = os.path.join(cfg_dir, entry, sac.AGENT_FILE)
            if not os.path.exists(path):
                print(f"{name} {entry}: no {sac.AGENT_FILE} -- this run predates "
                      f"agent checkpointing and must be re-run")
                continue
            res = analyze(path, max_steps=max_steps, device=device)
            results[(name, seed)] = res
            print_report(name, seed, res)
    if not results:
        print("\nNothing analyzed. Agent files are written by run_one_seed; a "
              "cached run\nwithout one is skipped rather than retrained.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--configs", nargs="+", default=["main", "oracle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    main(run_dir=args.run_dir, configs=args.configs, seeds=args.seeds,
         max_steps=args.max_steps)
