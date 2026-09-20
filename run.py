#!/usr/bin/env python
"""
run.py -- one entry point for everything.

    python run.py --oracle                      build the MSE reference ladder (once, ~15 min)
    python run.py --train  --arms main --seeds 0 1 2 3 4 5
    python run.py --train  --arms main control --seeds 0 1 2 3 4 5 6 7 8 9
                                                parallel by default
    python run.py --train  --arms main --seeds 0 1 2 3 --workers 1
                                                force serial
    python run.py --train  --arms control --seeds 0 1
    python run.py --figures --arms main control --seeds 0 1 2 3 4 5 --pool
    python run.py --all    --arms main control --seeds 0 1        everything, in order

TWO ARMS, ONE TASK.  `main` denies MTL the state its own action produced;
`control` hands it over.  Nothing else differs -- see model.py.

ONE INPUT CONVENTION.  The variant registry is gone -- its keys were all
workarounds for the old plant, which put position in [0, 60] and made the
velocity channel 97x smaller.  What survives is the GUARD: model.CONVENTION is
stamped into every checkpoint and checked on load, because a checkpoint from
before the factorial rewrite has the same parameter shapes and would otherwise
load in silence.

SEEDS RUN IN PARALLEL unless --workers says otherwise.  Each worker pins itself
to ONE BLAS thread, because N processes each spawning N threads oversubscribes
and runs slower than serial -- so a single job under --workers is slower than
the same job serial, and the win only appears when workers ~ jobs.

TRAINING IS SKIPPED if a checkpoint already exists, so re-running is safe and is
the normal way to regenerate figures.  If you see it start training when you
expected a skip, the checkpoint is missing: check before letting it overwrite a
result the manuscript depends on.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from plotting_utils import set_plt_params, make_axis_nice
import model as M


def _train_one(job):
    """One (arm, seed) pair. Module-level and picklable on purpose: macOS
    spawns rather than forks, so a closure could not be sent to a worker.

    A spawned worker RE-IMPORTS model.py, so anything the parent monkeypatched
    -- CKPT_DIR above all -- never reaches it. sac.py's parallel sweep hit
    exactly this and wrote toy runs into the real result directories. Nothing
    is patched here in normal use, but it is why the tests cover job selection
    and picklability rather than redirecting CKPT_DIR through a worker.
    """
    arm, seed, action_mode, readout, beta, sigma = job
    import torch
    # each worker gets one thread: 16 processes each spawning N BLAS threads
    # oversubscribes the machine and runs slower than serial
    torch.set_num_threads(1)
    t0 = time.time()
    M.train(arm, seed=seed, action_mode=action_mode, readout=readout, beta=beta,
            sigma=sigma)
    return arm, seed, (time.time() - t0) / 60.0


def resolve_workers(n_jobs, workers=None):
    """How many processes to use for `n_jobs` (arm, seed) pairs.

    PARALLEL BY DEFAULT when there is more than one job.  Seeds are
    independent -- M.train seeds torch from `seed` alone and writes only its
    own checkpoint -- so serialising them just left the machine idle: 16 seeds
    at ~27 min each is ~7 hours serial against under an hour spread out.

    An explicit `workers` OVERRIDES the default, including workers=1 for
    serial, which is what to use when the machine is needed for something else.
    A full-width run pins every core for its duration, and that is not free:
    running several heavy jobs at once here once turned a 27-minute training
    run into 242 minutes.

    Separated from train() so the rule is testable without training anything.
    """
    if n_jobs <= 0:
        return 1
    if workers is None:
        workers = min(n_jobs, os.cpu_count() or 1)
    return max(1, min(workers, n_jobs))


def train(arms, seeds, workers=None, action_mode=M.DEFAULT_ACTION_MODE,
          readout=M.DEFAULT_READOUT, beta=M.RATE_BETA, sigma=M.NOISE_SIGMA):
    jobs = []
    for arm in arms:
        for s in seeds:
            p = M.ckpt_path(arm, s, action_mode, readout, beta, sigma)
            if os.path.exists(p):
                # skip only a checkpoint of the CURRENT convention. A stale one
                # at this path (an older context table, another readout
                # default) would otherwise be skipped as "done" and every
                # downstream load would then refuse it -- or, worse, be
                # overwritten here. Neither silently.
                import torch
                stored = torch.load(p, map_location="cpu").get("convention")
                expected = M.convention_for(action_mode, readout, beta, sigma)
                if stored == expected:
                    print(f"== {arm} seed {s}: checkpoint exists, skipping -> {p}",
                          flush=True)
                    continue
                raise FileExistsError(
                    f"{p} holds a checkpoint of convention {stored!r}, not the "
                    f"current {expected!r}. Move it aside (e.g. into "
                    f"checkpoints/<old convention>/) before training this circuit.")
            jobs.append((arm, s, action_mode, readout, beta, sigma))
    if not jobs:
        return

    workers = resolve_workers(len(jobs), workers)
    if workers == 1:
        for arm, s, *_knobs in jobs:
            t0 = time.time()
            print(f"\n{'='*64}\n== {arm} seed {s}\n{'='*64}", flush=True)
            M.train(arm, seed=s, action_mode=action_mode, readout=readout, beta=beta,
                    sigma=sigma)
            print(f"== done in {(time.time()-t0)/60:.1f} min", flush=True)
        return

    # Seeds are independent -- M.train seeds torch from `seed` alone and writes
    # only its own checkpoint -- so a parallel run reproduces a serial one.
    print(f"\n{len(jobs)} runs across {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_train_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            arm, s, mins = f.result()
            print(f"== [{i}/{len(jobs)}] {arm} seed {s} done in {mins:.1f} min",
                  flush=True)


circuit_from_args = M.circuit_from_args      # one rule for every CLI


def figures(arms, seeds, pool, action_mode=M.DEFAULT_ACTION_MODE):
    import figures as F
    for arm in arms:
        for s in seeds:
            F.per_seed(arm, s, action_mode=action_mode)
        if pool:
            F.across_seeds(arm, seeds, action_mode=action_mode)


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
    ap.add_argument("--pool", action="store_true",
                    help="also write the across-seed figures")
    # which circuit: --action-mode / --readout / --rate-beta / --noise-sigma,
    # the same four flags on every CLI (model.add_circuit_args). For --train
    # they say what to fit; for --figures which checkpoint to render, on its
    # own distribution, into its own figures/<arm_tag>/ directory.
    M.add_circuit_args(ap)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes over (arm, seed) pairs. Default "
                         "is one per job, capped at the core count, so more "
                         "than one seed runs in parallel automatically; pass "
                         "1 for serial. Training only.")
    a = ap.parse_args()
    if not (a.oracle or a.train or a.figures or a.all):
        ap.error("nothing to do: pass --oracle, --train, --figures or --all")
    spec = circuit_from_args(a)
    print(f"arms {a.arms} | seeds {a.seeds} | circuit {spec}", flush=True)
    if a.oracle or a.all:
        import oracle as O
        if os.path.exists(O.LADDER) and not a.oracle:
            print(f"ladder already present -> {O.LADDER}", flush=True)
        else:
            O.main()
    if a.train or a.all:
        train(a.arms, a.seeds, workers=a.workers, action_mode=spec)
    if a.figures or a.all:
        # the SAME spec selects the checkpoint, the rollout distribution and
        # the figure directory -- see model's CircuitSpec section
        figures(a.arms, a.seeds, a.pool or a.all, action_mode=spec)
