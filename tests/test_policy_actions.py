"""The "policy" action process: windows of a recorded oracle-agent action
stream (action_bank.py) as the circuit's training actions.

What has to hold:

  the bank is sliced before the loop, so the action stream is independent of
    the supervised context stream (the convention every mode keeps)
  a window is a verbatim slice of the bank, may span a reach boundary, and
    the sequence's plant is driven by exactly those actions
  a missing bank is a clear error naming the builder, never a fallback
  the bank's stamp is in the convention, so a rebuilt bank invalidates the
    circuits fitted on the old one
  the mode is a first-class circuit: tag, spec, CLI
"""
import numpy as np
import pytest
import torch

import data as D
import model as M


@pytest.fixture
def bank(tmp_path, monkeypatch):
    """A tiny synthetic bank with a recognisable, non-stationary stream."""
    rng = np.random.default_rng(0)
    reaches = [np.tile(rng.uniform(-1, 1, 2).astype(np.float32), (L, 1))
               for L in (7, 12, 5, 30, 9, 41)]              # constant action per reach
    stream = np.concatenate(reaches)
    p = tmp_path / "bank.npz"
    np.savez(p, actions=stream, episode_start=np.cumsum([0] + [len(r) for r in reaches[:-1]]),
             stamp=np.array("synthetic-abc12345"))
    monkeypatch.setattr(D, "ACTION_BANK_PATH", str(p))
    D.load_action_bank.cache_clear()
    yield stream
    D.load_action_bank.cache_clear()


def test_missing_bank_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ACTION_BANK_PATH", str(tmp_path / "absent.npz"))
    D.load_action_bank.cache_clear()
    with pytest.raises(FileNotFoundError, match="action_bank.py"):
        D.generate_sequences(2, seed=0, action_mode="policy")
    with pytest.raises(FileNotFoundError):
        M.convention_for("policy")
    D.load_action_bank.cache_clear()


def test_windows_are_verbatim_slices_and_drive_the_plant(bank):
    trans, labels, tss = D.generate_sequences(6, seq_len=20, seed=3, action_mode="policy")
    assert trans.shape == (6, 20, 10)
    acts = trans[..., 4:6].numpy()
    stream = bank
    for b in range(6):
        # each sequence's actions are a contiguous window of the stream
        hits = [s for s in range(len(stream) - 20 + 1)
                if np.array_equal(stream[s:s + 20], acts[b])]
        assert hits, "sequence actions are not a window of the bank"
    # constant-per-reach stream: a window spanning a boundary changes action
    # exactly once; over six draws some do -- the boundary is not avoided
    n_changes = [(np.abs(np.diff(acts[b], axis=0)).sum(1) > 0).sum() for b in range(6)]
    assert max(n_changes) >= 1
    # the plant saw those actions: the next velocity moves with the action
    # (context-dependent direction, but never zero response off the walls)
    dv = trans[..., 8:10] - trans[..., 2:4]
    assert float(dv.abs().mean()) > 0


def test_action_stream_is_independent_of_the_context_stream(bank):
    """Same seed, the same pool in the opposite order -> identical actions but
    different context labels: the window is drawn before the context loop, as
    GP draws are. (Pools of DIFFERENT size would not do: numpy's bounded
    integers consume a range-dependent number of bits.)"""
    a1, c1, _ = D.generate_sequences(8, seq_len=16, seed=5, action_mode="policy", contexts=[0, 1])
    a2, c2, _ = D.generate_sequences(8, seq_len=16, seed=5, action_mode="policy", contexts=[1, 0])
    assert torch.equal(a1[..., 4:6], a2[..., 4:6])
    assert not torch.equal(c1, c2)


def test_stamp_enters_the_convention_and_the_tag(bank):
    conv = M.convention_for("policy")
    assert conv.startswith(M.CONVENTION) and "policy-actions" in conv and "synthetic-abc12345" in conv
    assert conv != M.convention_for("gp") and conv != M.convention_for("iid")
    assert M.arm_tag("main", "policy") == "main_policy"
    assert M.arm_tag("main", M.CircuitSpec("policy", "noisy", 1e-3, 0.1)) == "main_policy_noisy0.001_sigma0.1"
    assert "policy" in M.ACTION_MODES and "policy" in D.ACTION_MODES


def test_bank_window_refuses_a_bank_shorter_than_the_window(bank):
    b = D.load_action_bank()
    with pytest.raises(ValueError):
        D.bank_window(b, len(bank) + 1, np.random.default_rng(0))
    w = D.bank_window(b, len(bank), np.random.default_rng(0))     # exactly the whole stream
    assert np.array_equal(w, bank)
