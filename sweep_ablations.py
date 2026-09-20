#!/usr/bin/env python
"""
sweep_ablations.py -- run the named SAC ablation sweep.

    python sweep_ablations.py --seeds 10                    # parallel by default
    python sweep_ablations.py --seeds 2 --configs baseline main
    python sweep_ablations.py --seeds 10 --workers 1        # serial, logs to console
    python sweep_ablations.py --version savings             # the *_clearbuf twins
    python sweep_ablations.py --version both --episode-buffer

The core configs (see sac.AblationConfig and CONFIG_BY_NAME below): baseline
(no augmentation), oracle (the true context, as a dense near-orthogonal code),
oracle_geom (the true context in the CIRCUIT'S measured geometry instead --
see sac.ORACLE_KERNELS), main/control (this repo's main/control checkpoints,
frozen -- frozen is the default and so unsuffixed; which checkpoint follows
model.DEFAULT_ACTION_MODE), end_to_end (fresh EC+CB, jointly trained with
SAC), cb_only_end_to_end (fresh CB alone -- no EC, no recurrent context latent
-- jointly trained with SAC). Each has a *_clearbuf twin: the same config with
the replay buffer wiped at every phase switch, which is the SAVINGS version of
the experiment against the parent's RETENTION version (see the twin block
below). See sac.py's module docstring for why this asks a genuinely new
question with the existing circuit.

oracle_geom and the frozen configs need a trained main checkpoint (oracle_geom
measures its code's geometry from one); the rest run without.

PARALLELISM. Each (config, seed) pair is an independent process. This is the
only speedup that matters for this workload: the networks are far too small
for a GPU to help -- measured on an M-series machine, an MPS update was ~2x
SLOWER than CPU (17.5 vs 8.3 ms at batch 256) and each host<->device round
trip cost 0.35 ms against 0.004 ms on CPU, which the per-step env loop pays
several times over. Results do not depend on scheduling: run_sac seeds torch,
numpy and the replay buffer's rng from the seed alone, so a parallel sweep
reproduces a serial one exactly.
"""
import argparse

import sac
from sac import AblationConfig

CONFIG_BY_NAME = {
    "baseline": dict(name="baseline", use_ec=False, use_cb=False, use_oracle=False),
    "oracle": dict(name="oracle", use_ec=False, use_cb=False, use_oracle=True),
    # The COIN baseline: the context INFERRED, exactly, from the realised
    # force -- the finite-context ideal observer (sac.BayesObserver). Same
    # channel and codes as `oracle`, but it has to work out which context it
    # is in, is not told when that changes, and starts every probe context
    # uniform. Against `oracle` it prices inference; against `main` it prices
    # EC's readout, since it is action-free by construction where z is not.
    "coin": dict(name="coin", use_ec=False, use_cb=False, use_oracle=True,
                 oracle_source="bayes"),
    # The same observer scored through the LEARNED forward model: each context
    # is the frozen CB conditioned on that context's z-centroid, and its
    # responsibility comes from its own prediction error -- the literal COIN
    # structure, with CB as the per-context predictor (sac.CBObserver). Splits
    # coin's gap to main into CB's model error (coin - coin_cb) and EC's
    # readout (coin_cb - main). NEEDS the (arm, arm_seed) checkpoint.
    "coin_cb": dict(name="coin_cb", use_ec=False, use_cb=False, use_oracle=True,
                    oracle_source="cb", arm="main"),
    # The same perfect context label in the CIRCUIT'S OWN geometry rather than
    # a near-orthogonal one: same width, same density, same GP construction,
    # pairwise cosines drawn from z's measured ones. Against "oracle" it asks
    # whether knowing the context is worth less when the code SHAPE is the
    # circuit's, which is the part of main_frozen's disadvantage that has
    # nothing to do with inference. See sac.ORACLE_KERNELS.
    #
    # NEEDS the (arm, arm_seed) checkpoint -- it measures the geometry from it
    # -- so unlike the other oracle configs arm is load-bearing here.
    "oracle_geom": dict(name="oracle_geom", use_ec=False, use_cb=False,
                        use_oracle=True, oracle_kernel="z", arm="main"),
    # The frozen supervised circuit feeding MC. FROZEN IS THE DEFAULT and so
    # unsuffixed; the end-to-end configs below spell out that they are not.
    # Which supervised checkpoint is frozen follows model.DEFAULT_ACTION_MODE
    # (the GP-trained one); the AR(1)-fitted circuit is action_mode="ar1".
    "main": dict(name="main", use_ec=True, use_cb=True, freeze=True, arm="main"),
    "control": dict(name="control", use_ec=True, use_cb=True, freeze=True, arm="control"),
    "end_to_end": dict(name="end_to_end", use_ec=True, use_cb=True, freeze=False, arm="main"),
    "cb_only_end_to_end": dict(name="cb_only_end_to_end", use_ec=False, use_cb=True, freeze=False, arm="main"),
    # EVERY OTHER PRESET HAS BEEN REMOVED, and the deletions are the point.
    #
    # 2026-09-16 -- cut back to the core set above (plus, below, one *_clearbuf
    # twin each), on the user's call. main_frozen and control_frozen were
    # renamed main and control at the same time. Removed and why:
    #
    # oracle_snr_fitted/_mid/_deployed, oracle_snr_r3/r9/r27/r81, oracle_snr_
    # r3_rho, oracle_snr_iso_read3/9/27
    #     the matched-SNR ladder and its cloud-geometry controls. CLOSED: at
    #     r >= 9 a single step decodes the context at 1.000 and retention was
    #     flat (0.091 -> 0.098 from r=9 to r=27), so information stopped being
    #     the limit long before the oracle's ceiling -- and the target itself
    #     then moved from retention to savings, on which SNR is not the lever.
    # oracle_geom_centered
    #     the paired control for oracle_geom on the "one shared DC vector"
    #     reading of z, which the centering presets below had already killed.
    # main_frozen_gp, main_frozen_gp_no_cb
    #     the GP-trained circuit is now the DEFAULT arm checkpoint (model.
    #     DEFAULT_ACTION_MODE), so main_frozen IS that run; the AR(1) circuit
    #     is the _ar1 variant. The gp-vs-AR(1) question is answered: retention
    #     0.413 vs 0.413 (paired 0.000), savings +0.27 vs +0.30 -- the action
    #     distribution the circuit was fitted on does not matter downstream.
    # main_frozen_no_cb
    #     CB-readout lesion. Savings -43.6 vs main_frozen's -32.3 episodes and
    #     retention indistinguishable: CB's channel to MC is not load-bearing
    #     for either effect. Re-add if a config makes it so.
    #
    # Older deletions (input-variant and DC-hypothesis presets):
    #
    # main_frozen_rnn, main_frozen_rnn_no_cb   distinguished only by an input
    #     VARIANT, which no longer exists -- there is one convention now, and
    #     the GRU/RNN choice is model.CELL_TYPE rather than a checkpoint family.
    # main_frozen_centered, main_frozen_normed, main_frozen_no_cb_centered,
    # main_frozen_cb_scaled, main_frozen_cb_noise
    #     all probes of the hypothesis that the circuit blocks' large
    #     condition-independent component was what blocked downstream learning.
    #     THAT HYPOTHESIS IS FALSIFIED.  Removing the DC made things much
    #     WORSE, not better -- lag 0 fell from 0.397 (raw) to 0.093 (centered)
    #     and 0.071 (standardized), against 0.798 with the block deleted
    #     outright.  Centering does not change the block's variance at all, so
    #     the 4x drop cannot be about its varying content: the DC was
    #     PROTECTIVE, most likely by forcing the actor's CB-column weights down
    #     and suppressing the varying part with them.
    #
    # Those numbers were measured on the OLD plant and do not carry over, but
    # the hypothesis they killed does not need re-killing, and keeping presets
    # whose comments cite dead numbers is worse than having fewer presets.
    # Re-add deliberately if a retrained circuit raises the question again.
}

# The core set, before any twin is added. presets_for() and the tests read it.
CORE_PRESETS = tuple(CONFIG_BY_NAME)

# ── the two versions of the experiment ───────────────────────────────────
#
# Every core preset gets ONE *_clearbuf twin: itself with the replay buffer
# wiped at every phase switch (AblationConfig.clear_buffer_on_switch) and
# nothing else changed. DERIVED through sac.apply_cli_variants rather than
# written out, so a twin can never drift from its parent, and so
# `--config X --clear-buffer`, `--config X_clearbuf` and sweep `--version
# savings` are provably the same run. tests/test_sweep_ablations.py pins the
# one-knob invariant on the resolved configs.
#
#   parent   replay carries the just-left context into the next block --
#            0.35-0.42 of every minibatch there. Shows RETENTION: a previous
#            context's policy is kept alive in MC's weights by replaying its
#            tagged transitions; probe success on earlier contexts holds at
#            ~0.4 (circuit) / ~0.9 (oracle).
#   twin     no cross-context replay. Retention collapses to baseline for the
#            circuit -- and SAVINGS remains: revisits start where first visits
#            do and climb ~0.3 further within the block, every seed, every
#            context. Faster readaptation after complete washout, with no
#            replay support. The oracle keeps ~60% of its retention.
#
# baseline_clearbuf is the null for both versions: no context signal to tag
# replay with, neither effect in either version.
for _name in CORE_PRESETS:
    CONFIG_BY_NAME[f"{_name}_clearbuf"] = sac.apply_cli_variants(
        CONFIG_BY_NAME[_name], clear_buffer=True)

CONFIG_PRESETS = [AblationConfig(**cfg) for cfg in CONFIG_BY_NAME.values()]


def run_named_sweep(presets, seeds, arm_seed=0, save_agg=True, workers=None,
                     make_plots=True, **run_kwargs):
    """Run every (config, seed) pair, then aggregate and plot.

    Pooling lives in sac.run_tasks, shared with sac.main -- which parallelizes
    over the seeds of a single config the same way.
    """
    # arm_seed matters to any config that READS a checkpoint, which is the
    # frozen ones AND the geometry oracles -- those measure their code's
    # kernel from it, so leaving them pinned to seed 0 while the frozen configs
    # moved would compare an oracle against a circuit it was not built from
    presets = [AblationConfig(**{**p.__dict__, "arm_seed": arm_seed})
               if (p.freeze or p.oracle_kernel != "identity") else p
               for p in presets]
    tasks = [(cfg, seed) for cfg in presets for seed in range(seeds)]

    print(f"{len(tasks)} runs ({len(presets)} configs x {seeds} seeds)")
    sac.run_tasks(tasks, workers=workers, run_kwargs=run_kwargs)

    for cfg in presets:
        sac.aggregate_config(cfg, range(seeds), save_agg=save_agg)

    if make_plots:
        import make_ablations_plot
        make_ablations_plot.main()


VERSIONS = ("retention", "savings", "both")


def presets_for(names, version="retention", episode_buffer=False,
                action_mode=None, ec_readout=None, rate_beta=None, noise_sigma=None,
                phase_set=None, train_contexts=None, schedule=None, ec_prior=None,
                relabel_z=False, count_prior=None, transition_buffer=False):
    """The configs a sweep over `names` runs, for one version of the
    experiment or both.

      retention  the presets as named: replay carries the just-left context
                 into the next block.
      savings    each preset's *_clearbuf twin: buffer wiped at every switch.
      both       parents and twins together, parents first.

    Twins are DERIVED from the parents through sac.apply_cli_variants rather
    than looked up, so `--version savings` works for a preset that has no
    hand-written twin, and a hand-written twin is asdict-identical to the
    derived one (tests pin that). Asking for the savings version of a preset
    that already is one is a no-op, not a double suffix.
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    # several schedules at once (`--schedule savings_washout spontaneous_recovery`):
    # the whole selection is built once per schedule, in the order given
    if isinstance(schedule, (list, tuple)):
        if len(schedule) != 1:
            out = []
            for s in schedule:
                out.extend(presets_for(names, version, episode_buffer, action_mode, ec_readout,
                                       rate_beta, noise_sigma, phase_set, train_contexts, s,
                                       ec_prior, relabel_z, count_prior, transition_buffer))
            return out
        schedule = schedule[0]
    circuit = dict(action_mode=action_mode, ec_readout=ec_readout, rate_beta=rate_beta,
                   noise_sigma=noise_sigma, train_contexts=train_contexts)
    base = []
    for n in names:
        cfg = CONFIG_BY_NAME[n]
        # circuit knobs only reach presets that freeze a circuit; the rest of a
        # mixed selection is left alone rather than refused
        knobs = circuit if cfg.get("freeze", False) else {}
        # the task knobs (which contexts, which schedule) apply to EVERY
        # preset; the EC-prior / relabel knobs only where there is an EC, and
        # the count prior only where there is an inferred oracle -- the rest
        # of a mixed selection is left alone rather than refused
        if cfg.get("use_ec", False):
            knobs = dict(knobs, ec_prior=sac.default_ec_prior(cfg, ec_prior),
                         relabel_z=relabel_z)
        if cfg.get("use_oracle", False) and cfg.get("oracle_source", "given") != "given":
            knobs = dict(knobs, count_prior=count_prior)
        base.append(sac.apply_cli_variants(cfg, episode_buffer=episode_buffer,
                                           transition_buffer=transition_buffer,
                                           phase_set=phase_set, schedule=schedule, **knobs))
    twins = [sac.apply_cli_variants(c, clear_buffer=True) for c in base]
    chosen = {"retention": base, "savings": twins, "both": base + twins}[version]
    seen, out = set(), []
    for cfg in chosen:                      # `both` on a twin would list it twice
        if cfg["name"] not in seen:
            seen.add(cfg["name"])
            out.append(AblationConfig(**cfg))
    return out


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the SAC ablation sweep.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--arm-seed", type=int, default=0,
                         help="checkpoint seed to load for the frozen configs")
    parser.add_argument("--configs", nargs="+", default=list(CONFIG_BY_NAME),
                         choices=list(CONFIG_BY_NAME), help="subset of presets to run")
    parser.add_argument("--version", choices=VERSIONS, default="retention",
                         help="retention: presets as named (replay carries across "
                              "contexts); savings: their *_clearbuf twins (buffer wiped "
                              "at every switch); both: parents then twins")
    parser.add_argument("--episode-buffer", action="store_true",
                         help="episode-indexed replay for every selected preset "
                              "(sac.AblationConfig.replay_units='episodes'); run names "
                              "get _epbuf. Already the default under --schedule.")
    parser.add_argument("--transition-buffer", action="store_true",
                         help="opt a --schedule sweep out of the episode buffer; run names "
                              "get _transbuf")
    parser.add_argument("--action-mode", default=None,
                         choices=list(sac.M.ACTION_MODES) + list(sac.M.LEGACY_ACTION_MODES),
                         help="frozen circuit's action process, applied to the "
                              "freeze=True presets in the selection (name gets _<mode>)")
    parser.add_argument("--ec-readout", default=None, choices=list(sac.M.READOUTS),
                         help="frozen circuit's EC readout; 'noisy' with --rate-beta")
    parser.add_argument("--rate-beta", type=float, default=None,
                         help="beta of the noisy-readout circuit to load")
    parser.add_argument("--noise-sigma", type=float, default=None,
                         help="channel-noise sd of the noisy-readout circuit to load")
    parser.add_argument("--train-contexts", default=None,
                         choices=list(sac.M.TRAIN_CONTEXT_SETS),
                         help="which contexts the frozen circuit was fitted on")
    parser.add_argument("--schedule", default=None, nargs="+", choices=list(sac.SCHEDULES),
                         help="block schedule(s) for every selected preset (sac.SCHEDULES); "
                              "several run the whole selection once per schedule")
    parser.add_argument("--ec-prior", default=None, choices=list(sac.EC_PRIORS),
                         help="EC start-state prior, applied to presets with an EC channel")
    parser.add_argument("--relabel-z", action="store_true",
                         help="hindsight z relabelling, applied to presets with an EC channel")
    parser.add_argument("--count-prior", type=float, default=None,
                         help="visit-count prior weight, applied to the inferred oracles")
    parser.add_argument("--contexts", default=None, choices=list(sac.PHASE_SETS),
                         help="which four contexts the task uses, for every selected "
                              "preset (names get _rot / _refl)")
    parser.add_argument("--workers", type=int, default=None,
                         help="parallel processes (default: cpu_count-1; 1 = serial)")
    parser.add_argument("--no-save-agg", dest="save_agg", action="store_false", default=True)
    parser.add_argument("--no-plots", dest="make_plots", action="store_false", default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    presets = presets_for(args.configs, version=args.version,
                          episode_buffer=args.episode_buffer,
                          action_mode=args.action_mode, ec_readout=args.ec_readout,
                          rate_beta=args.rate_beta, noise_sigma=args.noise_sigma,
                          phase_set=args.contexts, train_contexts=args.train_contexts,
                          schedule=args.schedule, ec_prior=args.ec_prior,
                          relabel_z=args.relabel_z, count_prior=args.count_prior,
                          transition_buffer=args.transition_buffer)
    run_named_sweep(presets, seeds=args.seeds, arm_seed=args.arm_seed,
                     save_agg=args.save_agg, workers=args.workers,
                     make_plots=args.make_plots)
