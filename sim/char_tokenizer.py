"""Phase 2.2 char-level tokenizer for Tiny Shakespeare.

ONLY ON path-f-hybrid BRANCH.

Simple character-level tokenizer:
- Vocab built from training corpus (unique chars, sorted)
- One-hot encoding for input to SNN
- Inverse map for decoding output predictions

Char-level chosen over byte-pair / LLaMA tokenizer because:
- Vocab size ~80 chars (vs 128K LLaMA), fits SNN architecture
- No external tokenizer dependency
- Easier to interpret outputs (literal characters)
- Matches Tiny Shakespeare's small scale

Tradeoff: longer sequences for same content vs subword tokens.
But with T=64-128 unroll, we cover ~10-20 word phrases, which
is sufficient for the toy/proof-of-concept stage.

Per Phase 2.2 design (master plan).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class CharTokenizer:
    """Build a char-level vocabulary from text and tokenize.

    Vocab: sorted list of unique chars, plus special <PAD>=index 0.
    """

    PAD_TOKEN = "<PAD>"

    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.vocab: List[str] = [self.PAD_TOKEN] + chars
        self.char_to_idx: Dict[str, int] = {c: i
                                              for i, c in enumerate(self.vocab)}
        self.vocab_size: int = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Convert string to list of token indices.

        Unknown chars are skipped (per Tiny Shakespeare's clean
        ASCII-only corpus, this shouldn't trigger).
        """
        return [self.char_to_idx[c] for c in text
                if c in self.char_to_idx]

    def decode(self, ids: List[int]) -> str:
        """Convert token indices to string. <PAD> -> empty char."""
        out = []
        for i in ids:
            if i == 0:
                continue  # skip PAD
            if 0 <= i < self.vocab_size:
                out.append(self.vocab[i])
        return "".join(out)

    def encode_one_hot(self, text: str) -> np.ndarray:
        """Encode string as (T, V) one-hot float32."""
        ids = self.encode(text)
        T = len(ids)
        oh = np.zeros((T, self.vocab_size), dtype=np.float32)
        for t, i in enumerate(ids):
            oh[t, i] = 1.0
        return oh


def make_seq_dataset(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
):
    """Build a dataset of (input, target) pairs for next-char
    prediction.

    Each sample:
    - input:  one-hot of seq_len chars from random position
    - target: integer class of next char (the char just after
              the input window)

    Returns:
        inputs: (n_samples, seq_len, vocab_size) float32 one-hot
        targets: (n_samples, seq_len) int64. target[t] = id of
                 char at position (start + t + 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    n_chars = len(ids)
    if n_chars < seq_len + 1:
        raise ValueError(f"Corpus too short: {n_chars} chars < "
                         f"seq_len+1 ({seq_len + 1})")

    V = tokenizer.vocab_size
    inputs = np.zeros((n_samples, seq_len, V), dtype=np.float32)
    targets = np.zeros((n_samples, seq_len), dtype=np.int64)

    for s in range(n_samples):
        start = int(rng.integers(0, n_chars - seq_len - 1))
        window = ids[start:start + seq_len + 1]   # seq_len+1 chars
        for t in range(seq_len):
            inputs[s, t, window[t]] = 1.0
            targets[s, t] = window[t + 1]
    return inputs, targets


def load_tiny_shakespeare(
    path: Optional[str] = None,
    download: bool = False,
) -> str:
    """Load Tiny Shakespeare corpus.

    Args:
        path: local path to tinyshakespeare.txt. Defaults to
              data/tinyshakespeare.txt
        download: not yet supported (would fetch from
                  https://raw.githubusercontent.com/karpathy/
                  char-rnn/master/data/tinyshakespeare/input.txt
                  but that requires network). For now, raise
                  if path missing.

    Returns:
        Corpus text.
    """
    if path is None:
        path = "data/tinyshakespeare.txt"
    p = Path(path)
    if not p.exists():
        if download:
            raise NotImplementedError(
                "Auto-download from URL not implemented. Manually "
                "fetch from https://raw.githubusercontent.com/karpathy/"
                "char-rnn/master/data/tinyshakespeare/input.txt and "
                "save to data/tinyshakespeare.txt"
            )
        raise FileNotFoundError(
            f"Tiny Shakespeare corpus not found at {path}. "
            f"Download from https://raw.githubusercontent.com/karpathy/"
            f"char-rnn/master/data/tinyshakespeare/input.txt"
        )
    return p.read_text(encoding="utf-8")


if __name__ == "__main__":
    # Smoke test on synthetic corpus
    corpus = "ABCABCABC the quick brown fox jumps over the lazy dog. "
    tok = CharTokenizer(corpus)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Vocab: {tok.vocab[:10]}...")

    test_str = "the quick"
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)
    assert decoded == test_str, f"Roundtrip failed: '{test_str}' -> '{decoded}'"
    print(f"Roundtrip OK: '{test_str}' -> {ids[:8]}... -> '{decoded}'")

    rng = np.random.default_rng(42)
    inputs, targets = make_seq_dataset(corpus, tok, seq_len=16, n_samples=3, rng=rng)
    print(f"Dataset: inputs {inputs.shape}, targets {targets.shape}")
    print(f"Sample 0 first chars: '{tok.decode(targets[0].tolist())}'")
