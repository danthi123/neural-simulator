import inspect, numpy as np
from research.runners.scaled_subword_lm_train import train_subword_lm

def test_signature_present():
    p = inspect.signature(train_subword_lm).parameters
    for k in ("seed","corpus_path","vocab_size","hidden_layers","T",
              "epochs","batch_size","lr","n_train_samples",
              "ckpt_path","bpe_path","backend"):
        assert k in p

def test_tiny_cpu_train_reduces_loss_and_is_resumable(tmp_path):
    ck = str(tmp_path / "s.ckpt.npz"); bp = str(tmp_path / "s.bpe.json")
    r = train_subword_lm(seed=42, corpus_path="data/tinyshakespeare.txt",
                          vocab_size=64, hidden_layers=[32], T=12,
                          epochs=3, batch_size=8, n_train_samples=32,
                          ckpt_path=ck, bpe_path=bp, backend="cpu",
                          verbose=False)
    assert r["final_loss"] is not None
    assert r["final_loss"] <= r["initial_loss"]      # learns (>= not strict)
    assert r["vocab_size"] > 1 and r["n_layers"] == 2
    import os
    assert os.path.exists(ck) and os.path.exists(bp)
    # resume: re-run same command for MORE epochs -> starts from ckpt,
    # does not crash, ends with >= the prior number of loss entries.
    from sim.train_checkpoint import load_checkpoint, resume_epoch
    c = load_checkpoint(ck)
    assert resume_epoch(c) == 3                       # 3 epochs completed
    r2 = train_subword_lm(seed=42, corpus_path="data/tinyshakespeare.txt",
                          vocab_size=64, hidden_layers=[32], T=12,
                          epochs=5, batch_size=8, n_train_samples=32,
                          ckpt_path=ck, bpe_path=bp, backend="cpu",
                          verbose=False)
    assert len(r2["loss_history"]) == 5              # resumed 3 -> ran 4,5
