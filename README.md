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

Every schedule, the default cycle included, runs with the same replay and
effort settings (`sac.AblationConfig` defaults): an episode-indexed buffer
(equal sampling mass per reach, so a competent agent does not buy itself a
longer memory), a fixed rehearsal split (85% of each batch from the block
being learned, 15% from earlier blocks), a per-step effort cost `-0.1 |a|²`
(what makes knowing the field pay), and the null-field warmup dropped from the
buffer when the first real block starts.  These were paradigm-only until
2026-09-23; cycle runs made before that sit in `sac_runs/_retired_cycle_v0/`.
`--transition-buffer` opts a run out of the episode buffer (`_transbuf`).

    python sweep_ablations.py --seeds 10                       # parallel default
    python sweep_ablations.py --seeds 5 --configs baseline main
    python sweep_ablations.py --seeds 10 --workers 1           # serial
    python sweep_ablations.py --seeds 5 --schedule spontaneous_recovery
    python make_ablations_plot.py                              # then the figure

A run writes `sac_runs/<config>/`: `aggregated_results.csv` (the learning
curves every cross-config figure reads) plus `seed_N/` holding `agent.pt`,
`curve.npz`, `probes.npz` and `train.log`.  Figures land in `figures/sac/`.
Neither tree is in git — both are large and regenerable.

### Zero-shot generalization to novel rotations

Trained on four rotations, what does each agent do in a rotation it has never
met — and does the circuit, which carries the family as an axis, generalize
better than COIN, whose only route to a novel context is a posterior-weighted
mixture of the ones it knows?  The `zero_shot` schedule is the cycle with a
test appended: a FROZEN sweep (16 deterministic reaches at each of 37 angles,
every 5° from −90 to +90, MC frozen, nothing written to long-term memory, block order
permuted per seed) and then one FEW-SHOT block of ordinary training at a novel
angle.

The anchors are the WIDE set, ±30 and ±90 (`task.WIDE_ROT_ANGLES`), with the
midpoints at ±60 as the interpolation test.  The table's ±30/±60 set was the
first version and was dropped: with anchors 30° apart a 50/50 mixture of two
neighbouring policies is nearly the policy for the angle between them, so
interpolation could not tell a mixture code from an axis code (those runs sit
in `sac_runs/_retired_narrow/`).  The wide set is a named set, not a change to
the table — `rotations_wide` in `model.TRAIN_CONTEXT_SETS` (the table with its
rotations replaced; circuit tag `_trainrotw`) and in `sac.PHASE_SETS` (suffix
`_rotw`) — and `--schedule zero_shot` implies both, so the circuit is held out
too: it never saw ±15, ±45, ±60 or ±75.  Off-table angles are real context ids
(`task.rotation_id`), so the plant, the bank and the curve carry them like any
context.

    python run.py --train --arms main --seeds 0 1 --train-contexts rotations_wide
    python circuit_axis_check.py                  # does z carry the angle across the 60° gaps?
    python sweep_ablations.py --seeds 10 --configs baseline main coin --schedule zero_shot --zs-max-steps 12
    python zero_shot_analysis.py                  # figures/sac/zero_shot_h12.pdf (one figure per horizon)

Runs are named `<config>_rotw_zs<angle>[_h<N>]` (`main_trainrotw_rotw_zs60_h12_carry`);
runs that differ only in the few-shot angle have identical sweeps and are
pooled.  The test blocks are CENTRE-OUT (the origin to eight targets at
`EVAL_RADIUS`, so every test reach is the same question) and run under a
shorter horizon, `--zs-max-steps` (`_h<N>`): under the full 30 steps feedback
absorbs a 15° aiming error at no cost — the hand accelerates at 1/MASS per
step and rides the speed cap, so a wrong first push is undone during the slow
first steps — and every trained config sat at the return ceiling, even with
the paradigms' effort cost, which the schedule also switches on.  A centre-out
reach needs 8 steps at full push and the trained policies take 10, so at 12 a
straight reach arrives and a curved one does not.  (The effort cost is now
every schedule's default, so `zero_shot` no longer differs from the cycle in
it.)  The headline measure stays
the FIRST action's angle, the feedforward compensation carried into a reach;
success and return under the short horizon are what it costs.  The `oracle`
config has no code for a novel context and is left out; the `control` arm was
flat against main and is not run.  Earlier runs with free-sampled test reaches
sit in `sac_runs/_retired_free/`.

**Is structure learning useful, or just present?**  The axis alone says the
circuit's code is continuous; it does not say that learning the family bought
anything.  Braun et al. (2009) established structural learning with a
contrast — exposure to a structured family against unstructured
transformations — and `--train-contexts linear_random` is that control: the
same table with the rotations replaced by four random linear transforms at
the wide rotations' effect sizes (`task.random_linear_ids`, tag `_trainlin`).
A novel rotation lies inside that family too, so the unstructured circuit can
represent it, but it has to identify four numbers rather than one.  Whatever
the rotation-trained circuit does better at novel rotations — faster
identification, larger first-action compensation, faster few-shot learning —
is what the structure bought.  COIN has no circuit, so the pretraining family
cannot reach it: the prediction is an interaction, family by model.

    python run.py --train --arms main --seeds 0 1 --train-contexts linear_random
    python circuit_axis_check.py --train-contexts rotations_wide linear_random
    python sweep_ablations.py --seeds 10 --configs main --schedule zero_shot --train-contexts linear_random --zs-max-steps 12
    python zero_shot_analysis.py            # main [trainrotw] solid against main [trainlin] dashed

`circuit_axis_check.py` is the gate between training a circuit and spending a
sweep on it: a ridge decoder from z to the angle, fitted on the anchors alone
and identically for every circuit, read at the held-out angles, plus the
decoder's error by step within an episode (how fast the angle is identified).

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
