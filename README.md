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

## Part 1 — motor task (16 discrete contexts)

    run.py                      entry point (--train / --figures / --oracle / --pool)
    model.py                    MTL (EC) and CB modules, training loop, load()
    task.py                     16-context grid task
    data.py                     trajectory generation
    decode.py                   decoding analyses
    oracle.py                   oracle / ceiling baselines
    figures.py                  ablation, decoding_curves, decoding_global,
                                epoch_curves, forces, linear_vs_mlp, mtl_control,
                                switch_aligned, tss_curves
    representation_analysis.py  clusters_*, mds_z, confusion_z, pca_*, weightcos,
                                pc_variance, topk

Checkpoints: main_seed0..9 (10 seeds), control_seed0..5 (6 seeds).

    python run.py --figures --arms main control --seeds 0
    python run.py --figures --pool --arms main control --seeds 0 1 2 3 4 5
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

Checkpoints included: rotation anchors 0-45-90 (main, control, one-hot; main
also at seed 1), rotation anchors 0-90 (main, one-hot; leave-one-out control
that omits 45), drift anchors 0.4-0.8-1.2 (main, control).

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

## Requirements

python 3.10+, torch, numpy, matplotlib, scikit-learn
