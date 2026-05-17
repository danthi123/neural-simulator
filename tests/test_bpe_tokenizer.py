import json, numpy as np
from sim.bpe_tokenizer import BPETokenizer

def test_train_encode_decode_roundtrip():
    corpus = ("the cat sat on the mat . the cat ran . " * 50)
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=60)
    assert tok.vocab_size >= 1 and tok.vocab_size <= 60
    s = "the cat sat"
    ids = tok.encode(s)
    assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == s

def test_training_is_deterministic():
    corpus = "aa bb ab ba aa bb ab ba " * 40
    a = BPETokenizer(); a.train(corpus, vocab_size=40)
    b = BPETokenizer(); b.train(corpus, vocab_size=40)
    assert a.merges == b.merges and a.vocab == b.vocab

def test_save_load_roundtrip_is_byte_stable(tmp_path):
    corpus = "hello world hello there world . " * 30
    tok = BPETokenizer(); tok.train(corpus, vocab_size=50)
    p = tmp_path / "bpe.json"; tok.save(str(p))
    tok2 = BPETokenizer.load(str(p))
    assert tok2.vocab == tok.vocab and tok2.merges == tok.merges
    assert tok2.encode("hello world") == tok.encode("hello world")
    json.loads(p.read_text())

def test_make_seq_dataset_dropin_compat():
    from sim.char_tokenizer import make_seq_dataset
    corpus = "the quick brown fox jumps over the lazy dog . " * 60
    tok = BPETokenizer(); tok.train(corpus, vocab_size=80)
    rng = np.random.default_rng(0)
    X, y = make_seq_dataset(corpus, tok, seq_len=8, n_samples=5, rng=rng)
    assert X.shape == (5, 8, tok.vocab_size) and y.shape == (5, 8)

def test_unknown_char_does_not_crash_encode():
    tok = BPETokenizer(); tok.train("abc abc abc " * 20, vocab_size=20)
    tok.encode("abc zzz")
