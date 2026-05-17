"""Pure deterministic Sennrich-2016 word-frequency BPE. No external
tokenizer dependency -> self-contained at runtime (the artifact is a
JSON merge table). Interface-compatible with sim.char_tokenizer
(.encode/.decode/.vocab_size/.encode_one_hot) so make_seq_dataset and
the validated trainer are drop-in DRY-reusable."""
from __future__ import annotations
import json
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

_EOW = "</w>"


class BPETokenizer:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: List[str] = []
        self._sym_to_id: Dict[str, int] = {}

    def train(self, corpus: str, vocab_size: int) -> None:
        words = [w for w in corpus.split() if w]
        wfreq = Counter(words)
        splits = {w: list(w) + [_EOW] for w in wfreq}
        merges: List[Tuple[str, str]] = []
        base = sorted({c for w in wfreq for c in w}) + [_EOW]
        while len(set(base) | {"".join(m) for m in merges}) < vocab_size:
            pair_freq: Counter = Counter()
            for w, f in wfreq.items():
                s = splits[w]
                for i in range(len(s) - 1):
                    pair_freq[(s[i], s[i + 1])] += f
            if not pair_freq:
                break
            best_f = max(pair_freq.values())
            best = min(p for p, c in pair_freq.items() if c == best_f)
            merges.append(best)
            a, b = best
            ab = a + b
            for w in splits:
                s = splits[w]
                i = 0
                out = []
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        out.append(ab)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                splits[w] = out
        self.merges = merges
        symbols = ["<UNK>"] + sorted(
            set(base) | {"".join(m) for m in merges})
        self.vocab = symbols
        self._sym_to_id = {s: i for i, s in enumerate(symbols)}

    def _encode_word(self, word: str) -> List[str]:
        s = list(word) + [_EOW]
        for a, b in self.merges:
            i = 0
            out = []
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            s = out
        return s

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in text.split():
            for sym in self._encode_word(w):
                ids.append(self._sym_to_id.get(sym, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        words, cur = [], []
        for i in ids:
            if not (0 <= i < len(self.vocab)):
                continue
            sym = self.vocab[i]
            if sym == "<UNK>":
                continue
            if sym.endswith(_EOW):
                cur.append(sym[: -len(_EOW)])
                words.append("".join(cur))
                cur = []
            else:
                cur.append(sym)
        if cur:
            words.append("".join(cur))
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_one_hot(self, text: str) -> np.ndarray:
        ids = self.encode(text)
        oh = np.zeros((len(ids), self.vocab_size), dtype=np.float32)
        for t, i in enumerate(ids):
            oh[t, i] = 1.0
        return oh

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(m) for m in self.merges],
                       "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = cls()
        t.merges = [tuple(m) for m in d["merges"]]
        t.vocab = list(d["vocab"])
        t._sym_to_id = {s: i for i, s in enumerate(t.vocab)}
        return t
