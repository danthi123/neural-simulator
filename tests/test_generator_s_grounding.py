"""Grounding smoke: the VALIDATED BPTT spiking core trains end-to-end
on the local 1.1MB data/tinyshakespeare.txt (zero network). If this
regresses, STOP -- the validated core is broken, nothing downstream
is interpretable."""
import os
import pytest


def test_validated_bptt_core_trains_on_local_shakespeare():
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    from research.runners.cortex_pretraining import train_shakespeare
    r = train_shakespeare(seed=42, T=24, hidden_layers=[48],
                          epochs=6, batch_size=16, n_train_samples=120,
                          corpus_path="data/tinyshakespeare.txt",
                          backend="cpu", verbose=False)
    assert r["final_loss"] < r["initial_loss"], (
        "validated BPTT core no longer reduces loss -- STOP, fix the "
        "core before any Generator-S work")
    assert r["vocab_size"] > 10 and r["n_layers"] == 2
