# Motor task and structure learning

Code and trained networks for two experiments on the same circuit: an MTL
recurrent network that infers context from cerebellar prediction error, and a
cerebellar forward model that predicts the next state.

    e_t      = s_t - s_hat_t
    z_t, h_t = MTL([s_{t-1}, a_{t-1}, e_t], h_{t-1})
    s_hat_t  = s_in + CB([s_in, a_t, z_t])

The MTL network never observes s_t directly; the prediction error is its only
window on the present.

## Arms

  main      MTL is denied s_t and must infer context from e_t alone
  control   MTL additionally receives s_t (augmented control)

## Part 1 — motor task (12 discrete contexts)

Three families, four levels each: rotation (±30, ±60 deg), curl (±7, ±13) and
drift (0, 90, 180, 270 deg).  `task.CONTEXTS` is the table; no context is the
identity, deliberately, since one would sit at effect size zero.

    run.py                      entry point (--train / --figures / --oracle / --pool)
    model.py                    MTL (EC) and CB modules, training loop, load()
    task.py                     12-context grid task, and the plant
    data.py                     trajectory generation
    decode.py                   decoding analyses
    oracle.py                   oracle / ceiling baselines
    figures.py                  ablation, decoding_curves, decoding_global,
                                epoch_curves, forces, linear_vs_mlp, mtl_control,
                                switch_aligned, tss_curves
    representation_analysis.py  clusters_*, mds_z, confusion_z, pca_*, weightcos,
                                pc_variance, topk

Checkpoints shipped: `main_seed0..3` (4 seeds).  **The `control` arm has no
checkpoint** — the ones that existed predate the factorial rewrite and will not
load against `model.CONVENTION`, so any `--arms control` command retrains from
scratch.  The pre-rewrite checkpoints are in git history (see the cleanup
commits) if you want them for reference; they are not valid under this task.

    python run.py --figures --arms main --seeds 0
    python run.py --figures --pool --arms main --seeds 0 1 2 3
    python run.py --train   --arms main control --seeds 0     # retrain

## Part 2 — structure learning (continuous parameter family)

    structure.py                does the circuit learn the FAMILY as a
                                continuous axis rather than a set of contexts?

Trained on a few anchor values of a parameter (e.g. rotation 0/45/90 deg) and
tested across a continuum, including outside the trained range.  Two panels:
CB prediction error against a one-hot floor (which cannot express an unseen
value) and a told-the-parameter ceiling, and a LINEAR decoder fitted on the
anchors only, predicting held-out values from z.

    python structure.py --family rotation --sweep --arms main control --seeds 0
    python structure.py --family rotation --overlay 0 90 --seeds 0

Checkpoints: none currently on disk.  The shipped set (rotation anchors
0-45-90 and 0-90, drift anchors 0.4-0.8-1.2, main/control/one-hot) predates the
factorial rewrite and will not load; it is in git history.  `--train` rebuilds
whichever anchor set you ask for.

### Known gap in the shipped baselines

The told-theta ceiling was later split into two oracles, one per arm -- a
single oracle cannot be a fair ceiling for both, since `main` is denied s_t
while `control` receives it.  The checkpoints here predate that split, so
`oracle_main` and `oracle_control` are absent.  The sweep handles this
gracefully: it prints `no checkpoint: ...` and draws the arms and the one-hot
floor without the dashed ceiling lines.  To restore them:

    python structure.py --family rotation --train --arms main control --seeds 0

(about 11 min per anchor set per baseline).  The `--skip-baselines` flag trains
only the arms.

## Part 3 — savings (reinforcement learning on the frozen circuit)

Parts 1 and 2 train the circuit by supervision and ask what `z` contains.  Part
3 puts a SAC agent on top of it and asks what the circuit is WORTH: exposed to
a context, then to others, then to the first one again, does the agent RELEARN
IT FASTER?  That relearning rate on re-exposure is the measure — savings, in
the COIN sense — not how much performance survived the interruption.

    sac.py                  the agent: SAC, the ablation configs, the COIN
                            schedules, the center-out probes, and the frozen
                            circuit's calibration (rollout_z and friends)
    sweep_ablations.py      run one or more configs across seeds
    make_ablations_plot.py  THE MAIN FIGURE: savings and relearning rate,
                            across configs, from the aggregated CSVs
    sac_figures.py          per-seed diagnostics for a single run
    transfer_analysis.py    seed-averaged cross-context transfer, from probes
    blind_ceiling.py        run this BEFORE a sweep (see below)

### The configs

    baseline             no circuit at all -- the floor
    main / control       the frozen circuit, arms as in Part 1
    oracle               told the context outright -- the ceiling
    oracle_geom          told it, but in the circuit's own measured geometry
    coin / coin_cb       a COIN-style Bayesian observer in place of the circuit
    end_to_end           circuit trained jointly with SAC, not frozen
    cb_only_end_to_end   a fresh CB alone: no EC, no context latent

Each has a `_clearbuf` twin that clears the replay buffer on a context switch.
`--version savings` selects the twins, `retention` the originals, `both` runs
each.

    python sweep_ablations.py --seeds 10                       # parallel default
    python sweep_ablations.py --seeds 5 --configs baseline main
    python sweep_ablations.py --seeds 10 --workers 1           # serial
    python sweep_ablations.py --seeds 5 --schedule spontaneous_recovery
    python make_ablations_plot.py                              # then the figure

A run writes `sac_runs/<config>/`: `aggregated_results.csv` (the learning
curves every cross-config figure reads) plus `seed_N/` holding `agent.pt`,
`curve.npz`, `probes.npz` and `train.log`.  Figures land in `figures/sac/`.
Neither tree is in git — both are large and regenerable.

### Check the ceiling before spending a sweep

`blind_ceiling.py` asks whether a context-blind controller already solves the
phase set, with no training and no circuit.  It exists because a sweep once
showed `baseline` reaching 1.00 success in every context at once, which looks
impossible and is not: closed-loop feedback absorbs a rotation, since the agent
re-aims every step and `MAX_STEPS` is ample.  A set the blind policy already
solves cannot separate the ablations, whatever the circuit does.

    python blind_ceiling.py                  # the current PHASE_CONTEXT_IDS
    python blind_ceiling.py --contexts 0 2 4 6

It reports success headroom and PATH headroom separately, and they disagree on
purpose: a blind policy can reach every target while needing a far longer path
than a specialist, and a version of this that reported only success would call
that set uninformative.

## Requirements

python 3.10+, torch, numpy, matplotlib, scikit-learn, pandas, seaborn, scipy,
statannotations (`pip install -r requirements.txt`).  `pytest` for the suite.
