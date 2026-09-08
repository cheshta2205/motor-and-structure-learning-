#!/usr/bin/env python
"""
run.py -- one entry point for everything.

    python run.py --oracle                      build the MSE reference ladder (once, ~15 min)
    python run.py --train  --arms main --seeds 0 1 2 3 4 5
    python run.py --train  --arms control --seeds 0 1
    python run.py --figures --arms main control --seeds 0 1 2 3 4 5 --pool
    python run.py --all    --arms main control --seeds 0 1        everything, in order

TWO ARMS, ONE TASK.  `main` denies MTL the state its own action produced;
`control` hands it over.  Nothing else differs -- see model.py.

TRAINING IS SKIPPED if a checkpoint already exists, so re-running is safe and is
the normal way to regenerate figures.  If you see it start training when you
expected a skip, the checkpoint is missing: check before letting it overwrite a
result the manuscript depends on.
"""
import argparse
import os
import time

import model as M


def train(arms, seeds):
    for arm in arms:
        for s in seeds:
            p = M.ckpt_path(arm, s)
            if os.path.exists(p):
                print(f"== {arm} seed {s}: checkpoint exists, skipping -> {p}",
                      flush=True)
                continue
            t0 = time.time()
            print(f"\n{'='*64}\n== {arm} seed {s}\n{'='*64}", flush=True)
            M.train(arm, seed=s)
            print(f"== done in {(time.time()-t0)/60:.1f} min", flush=True)


def figures(arms, seeds, pool, mlp=False):
    import figures as F
    for arm in arms:
        for s in seeds:
            F.per_seed(arm, s, with_mlp=mlp)
        if pool:
            F.across_seeds(arm, seeds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cerebellar prediction error and context inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--oracle", action="store_true",
                    help="build the MSE reference ladder (needed for tss_curves)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="oracle (if absent) + train + figures + pool")
    ap.add_argument("--arms", nargs="+", default=list(M.ARMS),
                    choices=list(M.ARMS),
                    help="main = MTL denied s_t; control = MTL given s_t")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mlp", action="store_true",
                    help="also make decoding_curves.png and linear_vs_mlp.png "
                         "(slow: ~10 min per seed)")
    ap.add_argument("--pool", action="store_true",
                    help="also write the across-seed figures")
    a = ap.parse_args()
    if not (a.oracle or a.train or a.figures or a.all):
        ap.error("nothing to do: pass --oracle, --train, --figures or --all")
    print(f"arms {a.arms} | seeds {a.seeds}", flush=True)
    if a.oracle or a.all:
        import oracle as O
        if os.path.exists(O.LADDER) and not a.oracle:
            print(f"ladder already present -> {O.LADDER}", flush=True)
        else:
            O.main()
    if a.train or a.all:
        train(a.arms, a.seeds)
    if a.figures or a.all:
        figures(a.arms, a.seeds, a.pool or a.all, mlp=a.mlp)
