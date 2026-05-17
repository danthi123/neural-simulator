import inspect
from research.runners.tiny_transformer_train import train_tiny_gpt


def test_signature():
    p = inspect.signature(train_tiny_gpt).parameters
    for k in ("seed", "corpus_path", "vocab_size", "d_model",
              "n_layer", "n_head", "block_size", "steps",
              "batch_size", "lr", "ckpt_path", "bpe_path", "device"):
        assert k in p


def test_tiny_cpu_train_reduces_loss_and_resumes(tmp_path):
    ck = str(tmp_path / "t.ckpt")
    bp = str(tmp_path / "t.bpe.json")
    r = train_tiny_gpt(seed=42,
                        corpus_path="data/tinyshakespeare.txt",
                        vocab_size=64, d_model=32, n_layer=1,
                        n_head=2, block_size=16, steps=5,
                        batch_size=8, ckpt_path=ck, bpe_path=bp,
                        device="cpu", verbose=False)
    assert r["final_loss"] is not None
    assert min(r["loss_history"]) < r["initial_loss"]
    assert r["vocab_size"] > 1 and "_model" in r and "_tok" in r
    import os
    assert os.path.exists(ck + ".pt")
    r2 = train_tiny_gpt(seed=42,
                        corpus_path="data/tinyshakespeare.txt",
                        vocab_size=64, d_model=32, n_layer=1,
                        n_head=2, block_size=16, steps=8,
                        batch_size=8, ckpt_path=ck, bpe_path=bp,
                        device="cpu", verbose=False)
    assert len(r2["loss_history"]) >= len(r["loss_history"])
