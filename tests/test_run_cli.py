"""
Tests for run.py's training driver.

M.train is monkeypatched throughout -- these check job selection, the worker
resolution rule and the parallel path, never that training works (model.py's
own tests cover that).  A real call is ~27 minutes per job.
"""
import os
import sys

import pytest

import model as M
import run


@pytest.fixture
def fake_train(monkeypatch, tmp_path):
    """Record calls and write a stub checkpoint, so skip logic is exercised."""
    calls = []

    def stub(arm, seed, **kw):
        calls.append((arm, seed))
        p = M.ckpt_path(arm, seed)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # a stub CARRYING THE CURRENT STAMP: the skip reads it (see below)
        import torch
        torch.save({"convention": M.convention_for()}, p)

    monkeypatch.setattr(M, "CKPT_DIR", str(tmp_path))
    monkeypatch.setattr(M, "train", stub)
    return calls


def test_train_covers_the_full_arm_seed_grid(fake_train):
    run.train(["main", "control"], [0, 1], workers=1)
    assert sorted(fake_train) == [("control", 0), ("control", 1),
                                  ("main", 0), ("main", 1)]


def test_train_skips_existing_checkpoints(fake_train, capsys):
    run.train(["main"], [0], workers=1)
    fake_train.clear()
    run.train(["main"], [0, 1], workers=1)
    assert fake_train == [("main", 1)]
    assert "checkpoint exists, skipping" in capsys.readouterr().out


def test_the_skip_checks_the_convention_stamp(fake_train, tmp_path):
    """The skip reads the stamp (2026-09-18; it used to go by path alone, and
    a convention bump then SILENTLY skipped every seed with a stale file).
    A stale checkpoint at the target path is refused with a message, never
    skipped as done and never overwritten."""
    import torch
    run.train(["main"], [0], workers=1)
    fake_train.clear()
    run.train(["main"], [0], workers=1)
    assert fake_train == []                                   # current stamp: skipped
    p = M.ckpt_path("main", 0)
    torch.save({"convention": "flat-families-v1+tanh-readout"}, p)
    with pytest.raises(FileExistsError, match="Move it aside"):
        run.train(["main"], [0], workers=1)
    assert fake_train == []                                   # and NOT retrained over it


def test_nothing_to_do_is_a_no_op(fake_train):
    run.train(["main"], [], workers=1)
    assert fake_train == []


def test_workers_default_to_parallel_when_there_is_more_than_one_job():
    """PARALLEL BY DEFAULT is a behavioural change and deserves pinning.

    The rule lives in run.resolve_workers rather than inline in train() so it
    can be checked without training anything -- which is the whole point, since
    the alternative is a multi-hour run.
    """
    import os
    import run

    cores = os.cpu_count() or 1
    # one job stays serial: there is nothing to overlap
    assert run.resolve_workers(1) == 1
    # more than one job goes parallel, up to the core count
    assert run.resolve_workers(4) == min(4, cores)
    assert run.resolve_workers(64) == cores
    # an explicit value overrides, in both directions
    assert run.resolve_workers(8, 2) == 2
    assert run.resolve_workers(8, 1) == 1
    # and is never allowed to exceed the jobs available -- asking for more
    # workers than jobs would spawn processes with nothing to do
    assert run.resolve_workers(3, 99) == 3
    # degenerate input does not produce a zero-worker pool
    assert run.resolve_workers(0) == 1


def test_cli_workers_defaults_to_none_so_the_rule_applies():
    """--workers must default to None, not 1.  A default of 1 would make every
    run serial and resolve_workers would never see the None it branches on."""
    import argparse
    import run

    src = open(run.__file__).read()
    assert '"--workers", type=int, default=None' in src


def test_worker_entry_point_is_picklable():
    """macOS spawns rather than forks, so the function sent to a worker must
    be importable by name -- a closure or lambda could not be sent."""
    import pickle
    assert pickle.loads(pickle.dumps(run._train_one)) is run._train_one
