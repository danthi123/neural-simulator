"""Subword-tokenizer coverage de-risk for the self-taught WKV mouth (board #99/#112/#199,
e-mouth-fluency lane). Companion CPU proof for the DESIGN finding
`research/findings/2026-09-02-subword-self-taught-mouth-design-*.md`.

THE QUESTION this de-risk answers, precisely. `research/findings/2026-09-02-self-taught-mouth-vocab-
coverage-for-chat-INSUFFICIENT.md` measured the production V=1000 WORD-level, CLOSED-vocabulary WKV mouth
against a 124-utterance chat-topic probe and found 39.9% content-word OOV / 5.65% fully-in-vocab / 10.48%
`in_vocab_scope` gate-pass -- a closed word vocab CANNOT partially express an OOV word, full stop. Its named
surpass #2 was "a genuinely wider-vocabulary or SUBWORD-capable checkpoint -- an OOV word becomes spellable
from subword pieces." This runner is the concrete, pure-CPU de-risk of exactly that claim: does subword
(BPE) tokenization close the hard-OOV gap, and at WHAT pieces-per-word COST to the spiking WTA read-out?

METHOD (apples-to-apples with the finding, nothing re-implemented):
 - The SAME 124-utterance probe corpus, imported VERBATIM via `_wkv_mouth_chat_topic_vocab_coverage_derisk.
   _build_probe_corpus` (its three groups: conversational_register / everyday_real_world_topics /
   wikidata_known_agents from the live store bundle).
 - The SAME content-word definition the finding + the production gate use: `WKV._WORD_RE.findall(text)`
   lowercased, minus `WKV._FUNCTION_WORDS`.
 - The WORD-level baseline is the ACTUAL production vocab set (`WKV._get_readout(seed)[1]`), so the
   `word_vocab` column here reproduces the finding's numbers as a sanity anchor.
 - The SUBWORD candidate is the PROJECT's own `sim.bpe_tokenizer.BPETokenizer` (Sennrich-2016 word-frequency
   BPE, self-contained, no external dep -- the same tokenizer the constrained-decode / generative-cortex
   work already ships). REUSE-BY-IMPORT: no `sim/` edit. Trained here at several vocab sizes on several
   corpora, purely on CPU (merge-table learning + coverage counting -- NO model training, NO GPU, NO torch).

METRICS per (corpus, vocab_size):
 - hard_oov_word_rate: fraction of content words whose BPE encoding contains the `<UNK>` sentinel (id 0) --
   i.e. a character absent from the training alphabet. For all-ASCII-[a-z'] content words (the probe is
   ASCII; `_WORD_RE` excludes digits/punctuation) this is ~0 by construction, and that IS the point: subword
   representability is (near-)total where word-level is 39.9% OOV.
 - fully_representable_pct: fraction of utterances whose every content word is representable (no `<UNK>`).
 - mean_pieces_per_content_word: THE honest residual -- each subword piece is one spiking-WTA emission, so
   this is the read-out COST multiplier vs word-level (which is 1 read/word). Reported overall AND
   specifically for the content words the production V=1000 word vocab could NOT represent (the proper-noun
   hard cases), and per probe group.

CLEANLINESS. BPE training is bounded (top-K frequent words, frequency-capped) so the pure-Python merge loop
is tractable on CPU within the memory budget; the alphabet is the lowercased `[a-z']` set the content words
are drawn from, so hard-OOV is a genuine (not an accounting-artifact) measurement. The word-vocab baseline
column is the real production vocab, verified to reproduce the finding.

Run: SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._subword_mouth_tokenizer_coverage_derisk \\
    --out research/findings/raw/_subword_mouth_tokenizer_coverage.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webapp import wkv_mouth_generator as WKV  # noqa: E402
from sim.bpe_tokenizer import BPETokenizer  # noqa: E402  (reuse-by-import; no sim/ edit)
from research.runners._wkv_mouth_chat_topic_vocab_coverage_derisk import _build_probe_corpus  # noqa: E402

WORD_RE = WKV._WORD_RE
FUNC = WKV._FUNCTION_WORDS

# The corpus data lake (data/corpus/*) is gitignored and lives only in the primary checkout; a git worktree
# does not carry it. Resolve a corpus path against the worktree first, then this fallback, so the runner is
# runnable from either checkout without editing paths.
_SHARED_CHECKOUT = Path("/home/dant123/Projects/sim")


def _resolve_corpus(path: str) -> Path | None:
    p = Path(path)
    if p.exists():
        return p
    alt = _SHARED_CHECKOUT / path
    if alt.exists():
        return alt
    return None


# ── corpus loading (bounded, lowercased [a-z'] word stream) ──────────────────────────────────────────────
def _load_corpus_words(path: Path, max_words: int) -> list:
    """Read a BOUNDED prefix of `path` and return a lowercased `[a-z']+` word list (the SAME token shape the
    production word Vocab is built from -- `load_stories`'s `re.findall(r"[a-z']+", ...)`). Bounded read keeps
    RSS small; a few M words is far more than a BPE merge table needs."""
    # ~10 bytes/word is a safe over-read; cap the raw read so we never slurp a 500 MB corpus.
    raw = open(path, encoding="utf-8", errors="ignore").read(max_words * 10)
    words = WORD_RE.findall(raw.lower())
    return words[:max_words]


def _train_bpe_bounded(words: list, vocab_size: int, top_k_words: int, freq_cap: int) -> tuple:
    """Train `sim.bpe_tokenizer.BPETokenizer` on the top-`top_k_words` most frequent words (each written at
    most `freq_cap` times to preserve merge-ranking while bounding the pure-Python merge loop). Returns
    (tokenizer, n_unique_words_used, train_seconds)."""
    c = Counter(words)
    kept = c.most_common(top_k_words)
    parts = []
    for w, f in kept:
        parts.extend([w] * min(int(f), freq_cap))
    corpus = " ".join(parts)
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(corpus, vocab_size)
    return tok, len(kept), round(time.time() - t0, 1)


# ── coverage analysis ────────────────────────────────────────────────────────────────────────────────────
def _content_words(text: str) -> list:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    return [t for t in toks if t not in FUNC]


def _word_vocab_row(text: str, vocab_set: set) -> dict:
    cw = _content_words(text)
    oov = [t for t in cw if t not in vocab_set]
    return {
        "n_content": len(cw),
        "n_oov": len(oov),
        "oov_words": sorted(set(oov)),
        "fully_representable": (len(oov) == 0 and len(cw) > 0),
    }


def _bpe_row(text: str, tok: BPETokenizer) -> dict:
    cw = _content_words(text)
    n_pieces = 0
    n_hard_oov = 0
    per_word = {}
    for w in cw:
        enc = tok.encode(w)
        pieces = len(enc)
        per_word[w] = pieces
        n_pieces += pieces
        if any(i == 0 for i in enc):  # id 0 == <UNK> sentinel (unseen character)
            n_hard_oov += 1
    return {
        "n_content": len(cw),
        "n_pieces": n_pieces,
        "n_hard_oov_words": n_hard_oov,
        "fully_representable": (n_hard_oov == 0 and len(cw) > 0),
        "per_word_pieces": per_word,
    }


def _aggregate_word(rows: list) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_utterances": 0, "note": "UNDEFINED (empty group)"}
    tot_c = sum(r["n_content"] for r in rows)
    tot_oov = sum(r["n_oov"] for r in rows)
    n_full = sum(1 for r in rows if r["fully_representable"])
    return {
        "n_utterances": n,
        "content_word_oov_rate": round(tot_oov / tot_c, 5) if tot_c else None,
        "fully_representable_pct": round(100.0 * n_full / n, 2),
        "fully_representable_frac": f"{n_full}/{n}",
    }


def _aggregate_bpe(rows: list, word_oov_set: set) -> dict:
    """`word_oov_set` = the set of content words the production WORD vocab could NOT represent (the hard
    proper-noun cases), so we can report pieces-per-word specifically for THOSE."""
    n = len(rows)
    if n == 0:
        return {"n_utterances": 0, "note": "UNDEFINED (empty group)"}
    tot_c = sum(r["n_content"] for r in rows)
    tot_pieces = sum(r["n_pieces"] for r in rows)
    tot_hard = sum(r["n_hard_oov_words"] for r in rows)
    n_full = sum(1 for r in rows if r["fully_representable"])
    # pieces specifically for the words the word-vocab could not represent
    hard_pieces = []
    all_pieces = []
    for r in rows:
        for w, p in r["per_word_pieces"].items():
            all_pieces.append(p)
            if w in word_oov_set:
                hard_pieces.append(p)
    return {
        "n_utterances": n,
        "hard_oov_word_rate": round(tot_hard / tot_c, 5) if tot_c else None,
        "fully_representable_pct": round(100.0 * n_full / n, 2),
        "fully_representable_frac": f"{n_full}/{n}",
        "mean_pieces_per_content_word": round(tot_pieces / tot_c, 3) if tot_c else None,
        "max_pieces_per_content_word": max(all_pieces) if all_pieces else None,
        "mean_pieces_per_WORDVOCAB_OOV_word": round(sum(hard_pieces) / len(hard_pieces), 3) if hard_pieces else None,
        "n_wordvocab_oov_words_seen": len(hard_pieces),
    }


def main(seed: int = 42, n_wikidata: int = 100, out: str | None = None,
         corpora=None, vocab_sizes=None, corpus_max_words: int = 3_000_000,
         top_k_words: int = 12_000, freq_cap: int = 40) -> dict:
    corpora = corpora or {
        "tinystories": "data/corpus/tinystories_train.txt",
        "wikitext": "data/corpus/wikitext103.txt",
    }
    vocab_sizes = vocab_sizes or [4000, 8000, 16000]

    out_data: dict = {
        "runner": "_subword_mouth_tokenizer_coverage_derisk",
        "seed": seed,
        "config": {
            "corpora": corpora, "vocab_sizes": vocab_sizes,
            "corpus_max_words": corpus_max_words, "top_k_words": top_k_words, "freq_cap": freq_cap,
        },
    }

    # ── the probe corpus (SAME 124 utterances as the finding) ────────────────────────────────────────────
    groups = _build_probe_corpus(seed, n_wikidata)
    all_items = [(g, it) for g, items in groups.items() for it in items]
    out_data["n_probe_utterances"] = len(all_items)

    # ── WORD-level baseline (the real production vocab) ──────────────────────────────────────────────────
    _, vocab_set, _ = WKV._get_readout(seed)
    out_data["word_vocab_size"] = len(vocab_set)
    word_rows_by_group = {g: [] for g in groups}
    word_oov_words = set()
    for g, items in groups.items():
        for it in items:
            r = _word_vocab_row(it["text"], vocab_set)
            word_rows_by_group[g].append(r)
            word_oov_words.update(r["oov_words"])
    out_data["word_vocab"] = {
        "per_group": {g: _aggregate_word(rows) for g, rows in word_rows_by_group.items()},
        "overall": _aggregate_word([r for rows in word_rows_by_group.values() for r in rows]),
        "n_distinct_oov_content_words": len(word_oov_words),
    }

    # ── SUBWORD (BPE) candidates ─────────────────────────────────────────────────────────────────────────
    out_data["subword"] = {}
    for cname, cpath in corpora.items():
        rp = _resolve_corpus(cpath)
        if rp is None:
            out_data["subword"][cname] = {"error": f"corpus_missing:{cpath}"}
            print(f"[{cname}] corpus_missing:{cpath}", flush=True)
            continue
        words = _load_corpus_words(rp, corpus_max_words)
        out_data["subword"][cname] = {"n_corpus_words_read": len(words),
                                      "resolved_path": str(rp), "by_vocab_size": {}}
        for V in vocab_sizes:
            tok, n_uniq, train_s = _train_bpe_bounded(words, V, top_k_words, freq_cap)
            bpe_rows_by_group = {g: [] for g in groups}
            for g, items in groups.items():
                for it in items:
                    bpe_rows_by_group[g].append(_bpe_row(it["text"], tok))
            entry = {
                "actual_vocab_size": tok.vocab_size,
                "n_merges": len(tok.merges),
                "n_unique_train_words": n_uniq,
                "train_seconds": train_s,
                "per_group": {g: _aggregate_bpe(rows, word_oov_words) for g, rows in bpe_rows_by_group.items()},
                "overall": _aggregate_bpe([r for rows in bpe_rows_by_group.values() for r in rows], word_oov_words),
            }
            out_data["subword"][cname]["by_vocab_size"][str(V)] = entry
            ov = entry["overall"]
            print(f"[{cname} V={V}] hard_oov={ov['hard_oov_word_rate']} "
                  f"fully_repr={ov['fully_representable_pct']}% "
                  f"mean_pieces/word={ov['mean_pieces_per_content_word']} "
                  f"mean_pieces/OOVword={ov['mean_pieces_per_WORDVOCAB_OOV_word']} "
                  f"(train {train_s}s)", flush=True)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(json.dumps(out_data["word_vocab"], indent=2))
    return out_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-wikidata", type=int, default=100)
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[4000, 8000, 16000])
    ap.add_argument("--corpus-max-words", type=int, default=3_000_000)
    ap.add_argument("--top-k-words", type=int, default=12_000)
    ap.add_argument("--freq-cap", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(seed=args.seed, n_wikidata=args.n_wikidata, out=args.out,
         vocab_sizes=args.vocab_sizes, corpus_max_words=args.corpus_max_words,
         top_k_words=args.top_k_words, freq_cap=args.freq_cap)
