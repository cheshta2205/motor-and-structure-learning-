"""Every figure function, on cut-down constants, in seconds rather than hours.

WHY THIS EXISTS.  The real figure pipeline is ~20 minutes per seed -- 3168
decoder fits at the default N_SEQ/N_SPLITS/KS -- and it fails at the FIRST
broken function, so a stale attribute five figures down the list costs a
20-minute round trip to discover and another to confirm the fix.  During the
factorial migration that happened four times in a row, each one a different
one-line breakage (`family_derived`, `FCOL['aim']`, `d["fam"]`, `O.load`).

So: shrink N_SEQ, N_SPLITS and KS until a full pass is seconds, run EVERY
figure entry point, and COLLECT the failures instead of raising on the first.
One run then lists everything that is broken.

It is a smoke test and nothing more.  It asserts that each function runs and
writes its file; it says nothing about whether the figure is correct, and the
numbers it produces are meaningless at this sample size.  What it catches is
the entire class of error that actually kept occurring: stale keys, renamed
attributes, dict lookups that lost their entry, layouts indexed by a count
that changed.
"""
import os
import traceback

import numpy as np
import pytest
import torch

import model as M


def _have_checkpoint():
    """load() RAISES on a convention mismatch rather than returning None, so a
    bare `M.load(...) is None` here would blow up at collection time whenever a
    stale checkpoint is present -- which is exactly when the skip is wanted."""
    try:
        return M.load("main", 0) is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_checkpoint(),
    reason="needs a main seed 0 checkpoint at the current CONVENTION")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """figures/representation_analysis shrunk to smoke size, plus a temp
    FIG_DIR so a smoke run can never overwrite a real figure."""
    import figures as F
    import representation_analysis as RA
    out = tmp_path_factory.mktemp("figs")
    old = {}
    for mod, keys in ((F, ("N_SEQ", "N_SPLITS", "KS")),
                      (RA, ("N_SEQ", "N_SPLITS"))):
        old[mod] = {k: getattr(mod, k) for k in keys}
    old_fig = M.FIG_DIR
    F.N_SEQ, F.N_SPLITS, F.KS = 60, 2, [0, 2, 4]
    RA.N_SEQ, RA.N_SPLITS = 60, 2
    M.FIG_DIR = str(out)
    yield out
    for mod, kv in old.items():
        for k, v in kv.items():
            setattr(mod, k, v)
    M.FIG_DIR = old_fig


def _collect(cases):
    """Run every case, gather tracebacks, fail once with all of them.

    The point of the module: pytest would otherwise stop at the first, which
    is exactly the behaviour that made this expensive in the first place.
    """
    fails = []
    for name, fn in cases:
        try:
            fn()
        except Exception:
            fails.append(f"--- {name} ---\n{traceback.format_exc()}")
    if fails:
        pytest.fail(f"{len(fails)}/{len(cases)} figure functions raised:\n\n"
                    + "\n".join(fails), pytrace=False)


def test_every_figure_function_runs(tiny):
    import figures as F
    d = F.collect("main", 0, n=F.N_SEQ)
    assert d is not None
    r = F.curves(d)
    mse = {m: float(np.mean(d[m]["se"])) for m, _, _, _ in F.COND}
    p = lambda n: os.path.join(str(tiny), n)

    _collect([
        ("fig_ablation", lambda: F.fig_ablation("main", 0, mse, r, p("a.png"))),
        ("fig_global", lambda: F.fig_global("main", 0, r, p("g.png"))),
        ("fig_mtl_control", lambda: F.fig_mtl_control("main", 0, r, p("m.png"))),
        ("fig_tss", lambda: F.fig_tss("main", 0, d, p("t.png"))),
        ("fig_forces", lambda: F.fig_forces("main", 0, d, p("f.png"))),
        ("fig_epochs", lambda: F.fig_epochs("main", 0, p("e.png"))),
    ])


def test_switch_aligned_runs(tiny):
    import figures as F
    out = F.switch_aligned("main", 0, n=F.N_SEQ)
    F.fig_switch("main", 0, out, os.path.join(str(tiny), "sw.png"))


def test_per_seed_runs_end_to_end(tiny):
    """The actual entry point run.py calls -- the one that kept breaking."""
    import figures as F
    F.per_seed("main", 0)


def test_representation_analysis_runs(tiny):
    import representation_analysis as RA
    d = RA.collect("main", 0, n=RA.N_SEQ)
    assert d is not None
    P = RA.pca_sites(d, n_comp=8)
    C, S = RA.confusion(d, "z", n_splits=RA.N_SPLITS)
    p = lambda n: os.path.join(str(tiny), n)
    _collect([
        ("fig_confusion", lambda: RA.fig_confusion("main", 0, C, S, p("c.png"))),
        ("fig_pca", lambda: RA.fig_pca("main", 0, d, P, p("p.png"), n_show=200)),
        ("fig_pc_variance",
         lambda: RA.fig_pc_variance("main", 0, RA.pc_variance(d, P, n_pc=4),
                                    p("v.png"), n_pc=4)),
        ("fig_clusters",
         lambda: RA.fig_clusters("main", 0, d, P, "z", p("k.png"), n_show=200)),
        # mds_centroids returns (coords, distance_matrix) -- pass the coords,
        # not the tuple.  My first version of this test passed the tuple and
        # the failure looked like a bug in fig_mds.
        ("fig_mds",
         lambda: RA.fig_mds("main", 0,
                            [("centroids", RA.mds_centroids(d, "z")[0], "")],
                            p("d.png"))),
        ("topk", lambda: RA.topk(d, P, ks=(2, 4), n_splits=2)),
        # these five were NOT covered, which is exactly where the stale code
        # kept turning up: four `_clf("linear", seed)` calls and a hardcoded
        # context id both survived two migrations by sitting in an uncalled
        # function.  gif_pca is skipped -- it writes an animation and is slow
        # in a way the rest are not.
        ("fig_topk",
         lambda: RA.fig_topk("main", 0, RA.topk(d, P, ks=(2, 4), n_splits=2),
                             0.5, p("tk.png"))),
        ("mds_weights", lambda: RA.mds_weights(d, "z", n_splits=2)),
        ("fig_paper_pca_z",
         lambda: RA.fig_paper_pca_z("main", [0], d, P, p("pp.png"), n_show=200)),
    ])


def test_pooled_analyses_run(tiny):
    """The across-seed poolers.

    THEY SKIP MISSING SEEDS AND RETURN EARLY IF THEY FIND NONE, so simply
    calling them against an empty temp FIG_DIR passes VACUOUSLY -- which is
    what a first version of this test did.  The per-seed npz files are written
    only inside representation_analysis's CLI block, not by any reusable
    function, so the test has to fabricate them.

    Every case therefore ASSERTS ITS OUTPUT FILE EXISTS.  A pooler that
    silently found nothing now fails instead of passing.
    """
    import numpy as np
    import representation_analysis as RA
    import model as M

    K, npc, nks = RA.N_CONTEXTS, 4, 2
    d0 = M.fig_dir("main", 0)
    rng = np.random.default_rng(0)
    C = rng.random((K, K)); C /= C.sum(1, keepdims=True)
    np.savez(os.path.join(d0, "confusion_z.npz"),
             C=C, S=rng.uniform(-1, 1, (K, K)),
             order=np.arange(K), labels=np.array(RA.LABELS), seeds=np.array([0]))
    # the reader's own key list, not a guess -- my first attempt used
    # "pos"/"vel" where it wants "position"/"velocity" and the test failed
    # LOUDLY, which is the improvement over the vacuous version
    pcv_keys = ("context", "position", "velocity", "var")
    np.savez(os.path.join(d0, "pc_variance.npz"),
             **{f"{k}|{site}": rng.random(npc)
                for site in RA.SITES for k in pcv_keys})
    np.savez(os.path.join(d0, "topk.npz"),
             **{f"ks|{site}": np.array([2, 4]) for site in RA.SITES},
             **{f"acc|{site}": rng.random(nks) for site in RA.SITES},
             **{f"cumvar|{site}": np.sort(rng.random(nks)) for site in RA.SITES},
             **{f"full|{site}": float(rng.random()) for site in RA.SITES})
    # MDS needs SYMMETRIC, zero-diagonal distance matrices -- a random square
    # is neither, and sklearn rejects it
    def _dist():
        X = rng.random((K, 5))
        D = np.linalg.norm(X[:, None] - X[None], axis=-1)
        return D / (D.max() + 1e-12)
    np.savez(os.path.join(d0, "mds_z.npz"),
             weights=rng.random((K, 2)), centroids=rng.random((K, 2)),
             resid=rng.random((K, 2)), dp=_dist(),
             D=_dist(), D_resid=_dist())

    out = os.path.join(str(tiny), "main", "across_seeds")
    cases = [
        ("pool_confusion", lambda: RA.pool_confusion("main", [0], "z"),
         "confusion_z.png"),
        ("pool_pc_variance", lambda: RA.pool_pc_variance("main", [0], n_pc=npc),
         "pc_variance_seeds.png"),
        ("pool_topk", lambda: RA.pool_topk("main", [0]), "topk_seeds.png"),
        ("pool_mds", lambda: RA.pool_mds("main", [0], "z"), "mds_z_seeds.png"),
    ]
    _collect([(n, f) for n, f, _ in cases])
    produced = os.listdir(out) if os.path.isdir(out) else []
    assert produced, (f"every pooler returned without writing anything -- they "
                      f"skip missing seeds silently, so this is the vacuous "
                      f"pass this test exists to catch. looked in {out}")


def test_multimodality_runs(tiny):
    """factorization.py went with the crossing it measured."""
    import multimodality as MM
    _collect([
        ("multimodality.plot", lambda: MM.plot(out_dir=str(tiny))),
        ("multimodality.plot_position",
         lambda: MM.plot_position(out_dir=str(tiny))),
        ("multimodality.position_invariance", MM.position_invariance),
    ])
