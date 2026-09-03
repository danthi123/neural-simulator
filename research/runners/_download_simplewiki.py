#!/usr/bin/env python
"""Download the full Simple-English-Wikipedia -> data/corpus/simplewiki.txt
(lowercased plain text) for the stream-cortex BREADTH corpus.

Owner-approved 2026-06-26 (decision #3). Simple-Wiki is the #1 breadth corpus
(more clusterable concepts past TinyStories' ~680 cap); corpus_stream.py's
multi-file --corpus-paths path already consumes it. Lowercase because the
tokenizer is re.findall(r"[a-z]+") (uppercase chars would mis-tokenize).

Re-runnable; tries a small set of public Simple-Wiki sources in order.

--- As downloaded 2026-09-03 (source rahular/simple-wikipedia), for the own-voice fluency retrain ---
The output data/corpus/simplewiki.txt stays GITIGNORED (regenerable ~142 MB cache); this docstring is
the tracked provenance/README pointer (owner-approved initial-education scaffold, decision #3).
  rows (articles) . . . . . . . . . . . 769,764
  size on disk  . . . . . . . . . . . . 142.4 MB
  word-tokens ([a-z']+) . . . . . . . . ~23,367,858
  BPE tokens (bridges/wkv_ckpt/wkv_bpe8k.json, V=8001) . . ~31.6M (measured BPE/word ratio 1.354)
  hard-OOV under wkv_bpe8k.json . . . . 0.0000% (0 <UNK> in a 42,812-token sample -> byte/char fallback covers it)
  mean/max BPE tokens per sentence (load_sentences 3-16 words) . . 14.3 / 45
Consumed by: research/runners/_emerge_wkv_lm_derisk.py --tokenizer bpe --corpus data/corpus/simplewiki.txt
"""
import os
import re
import sys

from datasets import load_dataset

OUT = "data/corpus/simplewiki.txt"
# public, no-auth Simple-English-Wikipedia plain-text sources (try in order)
CANDIDATES = [
    ("rahular/simple-wikipedia", None, "train"),
    ("wikimedia/wikipedia", "20231101.simple", "train"),
]


def load_first_available():
    for name, config, split in CANDIDATES:
        try:
            print(f"[download] trying {name} (config={config}) ...", flush=True)
            ds = (load_dataset(name, config, split=split) if config
                  else load_dataset(name, split=split))
            print(f"[download] loaded {name}: {len(ds)} rows, fields={ds.column_names}",
                  flush=True)
            return ds
        except Exception as e:  # noqa: BLE001
            print(f"[download] {name} FAILED: {repr(e)[:200]}", flush=True)
    return None


def main():
    ds = load_first_available()
    if ds is None:
        print("[download] NO dataset loaded -- ABORT", flush=True)
        return 2

    text_field = next((c for c in ("text", "content", "article", "body")
                       if c in ds.column_names), ds.column_names[-1])
    print(f"[download] text field = {text_field}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_rows = 0
    n_chars = 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for ex in ds:
            t = ex.get(text_field) or ""
            if not t:
                continue
            tl = t.lower()
            fh.write(tl + "\n")
            n_rows += 1
            n_chars += len(tl)
            if n_rows % 20000 == 0:
                print(f"[download]   {n_rows} rows, {n_chars/1e6:.1f}M chars", flush=True)
    print(f"[download] wrote {OUT}: {n_rows} rows, {n_chars/1e6:.1f}M chars", flush=True)

    # token sanity (same tokenizer as corpus_stream.py)
    toks = 0
    with open(OUT, encoding="utf-8") as fh:
        for line in fh:
            toks += len(re.findall(r"[a-z]+", line))
    print(f"[download] DONE: ~{toks} tokens, {os.path.getsize(OUT)/1e6:.1f} MB on disk",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
