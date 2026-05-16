"""Phase 2.2 char_tokenizer unit tests.

ONLY ON path-f-hybrid BRANCH.
"""
import numpy as np
import pytest


def test_vocab_size_includes_pad():
    """Vocab size = unique chars + 1 (PAD token)."""
    from sim.char_tokenizer import CharTokenizer

    corpus = "abcdef"  # 6 unique chars
    tok = CharTokenizer(corpus)
    assert tok.vocab_size == 7  # 6 + PAD
    assert tok.vocab[0] == "<PAD>"


def test_encode_decode_roundtrip():
    """encode -> decode should reproduce the original text."""
    from sim.char_tokenizer import CharTokenizer

    corpus = "ABCabc 123."
    tok = CharTokenizer(corpus)
    text = "ABC abc"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_encode_unknown_chars_skipped():
    """Unknown chars are skipped (don't crash)."""
    from sim.char_tokenizer import CharTokenizer

    corpus = "abc"
    tok = CharTokenizer(corpus)
    text = "abc DEF"  # space + DEF unknown (space not in corpus)
    ids = tok.encode(text)
    # Only 'abc' chars survive
    assert len(ids) == 3


def test_encode_one_hot_format():
    """encode_one_hot returns (T, V) float32 with exactly one 1 per row."""
    from sim.char_tokenizer import CharTokenizer

    corpus = "abc"
    tok = CharTokenizer(corpus)
    oh = tok.encode_one_hot("abc")
    assert oh.shape == (3, 4)  # T=3, V=4 (PAD + a,b,c)
    assert oh.dtype == np.float32
    # Each row sums to 1
    assert (oh.sum(axis=-1) == 1.0).all()


def test_make_seq_dataset_shapes():
    """make_seq_dataset returns expected shapes."""
    from sim.char_tokenizer import CharTokenizer, make_seq_dataset

    corpus = "ABCABCABCabc abc " * 50
    tok = CharTokenizer(corpus)
    rng = np.random.default_rng(42)
    inputs, targets = make_seq_dataset(
        corpus, tok, seq_len=16, n_samples=5, rng=rng,
    )
    assert inputs.shape == (5, 16, tok.vocab_size)
    assert targets.shape == (5, 16)
    # Inputs are one-hot
    assert (inputs.sum(axis=-1) == 1.0).all()


def test_make_seq_dataset_targets_correct():
    """For sample s, target[s, t] is the char following input[s, t]."""
    from sim.char_tokenizer import CharTokenizer, make_seq_dataset

    corpus = "ABCDEF" * 20
    tok = CharTokenizer(corpus)
    rng = np.random.default_rng(42)
    inputs, targets = make_seq_dataset(
        corpus, tok, seq_len=10, n_samples=5, rng=rng,
    )
    # For each sample s and time t < 9: target[s,t] should be the
    # class of input[s, t+1]
    for s in range(5):
        for t in range(9):
            input_class_t = int(np.argmax(inputs[s, t]))
            input_class_tp1 = int(np.argmax(inputs[s, t + 1]))
            # target[s, t] must equal input_class[s, t+1]
            assert targets[s, t] == input_class_tp1, (
                f"Sample {s}, t={t}: input[t]={input_class_t}, "
                f"input[t+1]={input_class_tp1}, target[t]={targets[s, t]}"
            )


def test_make_seq_dataset_too_short():
    """Raises if corpus shorter than seq_len + 1."""
    from sim.char_tokenizer import CharTokenizer, make_seq_dataset

    corpus = "abc"  # too short
    tok = CharTokenizer(corpus)
    with pytest.raises(ValueError):
        make_seq_dataset(corpus, tok, seq_len=10, n_samples=1,
                          rng=np.random.default_rng(42))


def test_load_tiny_shakespeare_missing_file_raises():
    """load_tiny_shakespeare raises if file missing."""
    from sim.char_tokenizer import load_tiny_shakespeare

    with pytest.raises(FileNotFoundError):
        load_tiny_shakespeare(path="/does/not/exist.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
