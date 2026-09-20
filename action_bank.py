"""action_bank.py -- record a trained oracle agent's actions for the circuit
to be fitted on (data.generate_sequences action_mode="policy").

WHY. The circuit is fitted on an action process and deployed under a policy
whose actions look nothing like it: mean |a| ~0.85 and lag-1 autocorrelation
~0.9 against the GP family's ~0.5 and ~0.4. Under that shift z's context code
collapses -- the between-context fraction of z's variance goes from ~0.4
(random actions) to ~0.1 (a trained policy's) for every circuit trained so far,
noisy readout or not -- and the ideal observer does not suffer it, so the
information is there and it is the circuit's training distribution that is
wrong. The fix is to fit the circuit on the actions it will meet.

WHAT. The `oracle` preset's agents (one per RL seed) are competent in the four
RL contexts because they are TOLD the context. Each is rolled out in each of
its contexts with the stochastic actor -- what the circuit sees during RL --
and the action sequences are recorded, episode by episode, into one stream.
generate_sequences then slices SEQ_LEN windows from that stream as the action
process, exactly as it uses a GP draw: chosen before the loop, independent of
the context stream, so the actions cannot leak the supervised context. The
closed-loop dependence of a on the state is deliberately NOT reproduced (the
bank is replayed open loop): what the earlier SNR ladder showed is that the
action process ALONE -- its saturation and autocorrelation -- accounts for most
of the collapse, and open-loop replay is the only way to keep the action
stream independent of the supervised context.

The bank's identity (source run, seeds, episodes per context, git-free
content hash) is stamped into the file and into the convention of every
circuit trained on it, so rebuilding the bank invalidates those checkpoints
instead of silently changing what they mean.

usage: python action_bank.py [--run oracle] [--seeds 0 1 ...] [--episodes 100]
"""
import argparse
import hashlib
import os
import time

import numpy as np
import torch

import data as D
import sac

DEFAULT_RUN = "oracle"
DEFAULT_EPISODES = 100          # per (agent seed, context)


def rollout_actions(agent, context_id, phase_idx, n_episodes, rng, device="cpu"):
    """Action sequences of `n_episodes` reaches of `agent` in `context_id`,
    with the oracle channel set to that context. Stochastic actor, as in
    training. Returns a list of (steps, 2) arrays and the success count."""
    config, cb_proj = agent.config, agent.cb_proj
    oracle_vec = sac.oracle_code(phase_idx, agent.oracle_embed)
    episodes, successes = [], 0
    for _ in range(n_episodes):
        pos, goal = sac.sample_pos_goal(rng)
        vel_sim = np.zeros(2, dtype=np.float32); vel_obs = vel_sim.copy()
        s_tm1 = torch.zeros(1, sac.SDIM, device=device)
        a_tm1 = torch.zeros(1, sac.ACTION_DIM, device=device)
        s_t = torch.tensor(sac.circuit_state(pos, vel_obs), device=device).unsqueeze(0)
        e_t = torch.zeros(1, sac.SDIM, device=device)
        cb_h2 = torch.zeros(1, sac.CB_HIDDEN, device=device)
        acts = []
        for _t in range(sac.MAX_STEPS):
            if np.linalg.norm(pos - goal) < sac.GOAL_RADIUS:
                successes += 1
                break
            obs, _z, _h = sac.augmented_obs_step(
                None, config, None, s_tm1, a_tm1, s_t, e_t, cb_h2, cb_proj,
                pos, vel_obs, goal, oracle_vec, device)
            with torch.no_grad():
                a, _, _ = agent.actor.get_action(torch.tensor(obs, device=device).unsqueeze(0))
            action = a.squeeze(0).numpy().astype(np.float32)
            acts.append(action)
            pos_new, vel_new = sac.env_step_params(pos, vel_sim, action, context_id, rng)
            vel_obs_new = vel_new.astype(np.float32)
            s_next = torch.tensor(sac.circuit_state(pos_new, vel_obs_new), device=device).unsqueeze(0)
            s_tm1, a_tm1, s_t = s_t, torch.tensor(action, device=device).unsqueeze(0), s_next
            pos, vel_sim, vel_obs = pos_new, vel_new, vel_obs_new
        if acts:
            episodes.append(np.stack(acts))
    return episodes, successes


def build(run=DEFAULT_RUN, seeds=None, n_episodes=DEFAULT_EPISODES, seed=0,
          path=D.ACTION_BANK_PATH, run_dir=None):
    """Record the bank and write it to `path`. Returns the metadata dict."""
    run_dir = run_dir or sac.RUN_DIR
    if seeds is None:
        seeds = sorted(int(d.split("_")[1]) for d in os.listdir(os.path.join(run_dir, run))
                       if d.startswith("seed_"))
    rng = np.random.default_rng(seed)
    episodes, ep_seed, ep_ctx, n_succ, n_ep = [], [], [], 0, 0
    t0 = time.time()
    for s in seeds:
        agent = sac.load_agent(os.path.join(run_dir, run, f"seed_{s}", "agent.pt"))
        for phase_idx, context_id in enumerate(sac.PHASE_CONTEXT_IDS):
            eps, succ = rollout_actions(agent, int(context_id), phase_idx, n_episodes, rng)
            episodes += eps; n_succ += succ; n_ep += len(eps)
            ep_seed += [s] * len(eps); ep_ctx += [int(context_id)] * len(eps)
        print(f"  seed {s}: {sum(len(e) for e in episodes)} steps so far, "
              f"success {n_succ / max(n_ep, 1):.2f}, {time.time() - t0:.0f}s", flush=True)
    # a RANDOM episode order, so consecutive windows mix agents and contexts
    # and an episode boundary is just a change of reach, not of anything else
    order = rng.permutation(len(episodes))
    stream = np.concatenate([episodes[i] for i in order]).astype(np.float32)
    ep_start = np.cumsum([0] + [len(episodes[i]) for i in order[:-1]])
    meta = dict(run=run, seeds=list(map(int, seeds)), n_episodes=int(n_episodes),
                seed=int(seed), n_steps=int(len(stream)), n_reaches=int(len(episodes)),
                success=float(n_succ / max(n_ep, 1)),
                abs_a=float(np.abs(stream).mean()),
                lag1=float(np.corrcoef(stream[:-1].ravel(), stream[1:].ravel())[0, 1]))
    stamp = hashlib.sha1(stream.tobytes()).hexdigest()[:8]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, actions=stream, episode_start=ep_start,
             episode_seed=np.array(ep_seed)[order], episode_context=np.array(ep_ctx)[order],
             stamp=np.array(f"{run}-s{len(seeds)}-e{n_episodes}-{stamp}"),
             **{f"meta_{k}": np.array(v) for k, v in meta.items()})
    D.load_action_bank.cache_clear()
    print(f"bank -> {path}: {meta['n_steps']} steps from {meta['n_reaches']} reaches, "
          f"success {meta['success']:.2f}, |a| {meta['abs_a']:.2f}, lag1 {meta['lag1']:.2f}, "
          f"stamp {D.action_bank_stamp()}", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="sac_runs/<run>/seed_*/agent.pt to record")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="agent seeds to record (default: every seed_* in the run)")
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                    help="reaches per (agent seed, context)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.run, a.seeds, a.episodes, a.seed)
