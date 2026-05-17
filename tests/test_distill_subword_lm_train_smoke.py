import inspect
from research.runners.distill_subword_lm_train import (
    train_distill_subword_lm)

def test_signature():
    p = inspect.signature(train_distill_subword_lm).parameters
    for k in ("seed","corpus_path","vocab_size","hidden_layers","T",
              "epochs","batch_size","lr","n_train_samples",
              "ckpt_path","bpe_path","backend"):
        assert k in p

def test_tiny_cpu_distill_reduces_loss_and_is_resumable(tmp_path):
    ck = str(tmp_path/"d.ckpt.npz"); bp = str(tmp_path/"d.bpe.json")
    r = train_distill_subword_lm(
        seed=42, corpus_path="data/tinyshakespeare.txt",
        vocab_size=64, hidden_layers=[32], T=12, epochs=3,
        batch_size=8, n_train_samples=32, ckpt_path=ck, bpe_path=bp,
        backend="cpu", verbose=False)
    assert r["final_loss"] is not None
    assert r["final_loss"] <= r["initial_loss"]
    assert r["n_layers"] == 2 and "_teacher" in r
    import os
    from sim.train_checkpoint import load_checkpoint, resume_epoch
    assert os.path.exists(ck)
    assert resume_epoch(load_checkpoint(ck)) == 3
    r2 = train_distill_subword_lm(
        seed=42, corpus_path="data/tinyshakespeare.txt",
        vocab_size=64, hidden_layers=[32], T=12, epochs=5,
        batch_size=8, n_train_samples=32, ckpt_path=ck, bpe_path=bp,
        backend="cpu", verbose=False)
    assert len(r2["loss_history"]) == 5
