"""Grounding: the distill plumbing TURNS end-to-end on local
shakespeare (zero network) -- a tiny spiking student trained by
soft-xent against the trigram teacher REDUCES loss. If this regresses,
STOP (systematic-debugging) -- do not run the scaled/corpus gate on a
broken pipeline. Goes GREEN after Task 3 (it IS the Task-3 gate)."""
import os
import pytest


def test_distill_plumbing_reduces_loss_local_shakespeare(tmp_path):
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    from research.runners.distill_subword_lm_train import (
        train_distill_subword_lm)
    r = train_distill_subword_lm(
        seed=42, corpus_path="data/tinyshakespeare.txt",
        vocab_size=64, hidden_layers=[32], T=12, epochs=3,
        batch_size=8, n_train_samples=40,
        ckpt_path=str(tmp_path / "d.ckpt.npz"),
        bpe_path=str(tmp_path / "d.bpe.json"),
        backend="cpu", verbose=False)
    assert r["final_loss"] is not None
    assert r["final_loss"] <= r["initial_loss"], (
        "distill plumbing does not reduce loss -- STOP, root-cause "
        "before any scaled/corpus run")
    assert r["vocab_size"] > 1 and r["n_layers"] == 2
