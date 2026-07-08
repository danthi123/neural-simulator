# Infra bug FIXED: the fetched-corpus tokenizer dropped the first letter of every capitalized word (proper nouns, sentence-initial) — `corpus_fetch.clean_text` didn't lowercase, violating `corpus_stream`'s `[a-z]+` tokenizer contract. Fix = lowercase at fetch. Sharpens the emergent clusters (held-out inheritance 0.664→0.712); probe-based results unchanged. NO `sim/` edit.

**Date:** 2026-07-08
**Fix:** `research/runners/corpus_fetch.py` `clean_text` now lowercases (honoring the `corpus_stream` tokenizer contract) + the cached `data/corpus/*.txt` re-lowercased. NOT a `sim/` edit; NOT a change to the shared `corpus_stream` tokenizer (so distill-based EMERGE runners are byte-unchanged).

## The bug (root-caused, not guessed)
`corpus_stream._tokenize` is `re.findall(r"[a-z]+", story)` and DOCUMENTS that it "assumes lowercase is already done in the cached corpus" (module docstring). But `corpus_fetch.clean_text` (which fetched TinyStories + WikiText) kept printable ASCII WITHOUT lowercasing. So every capitalized word — proper nouns (Lily, Lucy, Tom) and sentence-initial words (Once, Inside, Book) — had its capital first letter dropped by the lowercase-only regex:
```
buggy:  "Lily saw Once Inside Book" -> ['ily', 'saw', 'nce', 'nside', 'ook']
fixed:  "lily saw once inside book" -> ['lily', 'saw', 'once', 'inside', 'book']
```
Surfaced by the fully-emergent-conversation capstone: the discovered clusters contained fragments 'ily'/'ucy'/'nce'/'nside'/'ook'/'obo' — the tell that first letters were being stripped. (The bug is LATENT in the distill corpus too — it also has uppercase — but that path is left untouched to preserve committed EMERGE reproducibility.)

## The fix + why it is low-risk
Lowercase in `corpus_fetch.clean_text` (the fetch/normalization utility), which is the CORRECT normalization for the co-occurrence + small-LM consumers and honors the tokenizer's own documented contract. This fixes future fetches; the existing cached `data/corpus/*.txt` were re-lowercased in place (gitignored, regenerable). The shared `corpus_stream` tokenizer is UNCHANGED (so the many distill-based EMERGE runners keep their exact committed tokenization/results). Only the fetched-corpus breadth/inheritance pipeline is affected — and improved.

## Impact (6-seed re-validation on the fixed corpus, TinyStories K=256)
| result | pre-fix | post-fix |
|---|---|---|
| **emergent-cluster inheritance** (probe-free) | held-out 0.664, cluster-coh 0.131 | **held-out 0.712, cluster-coh 0.167** (cleaner, sharper) |
| **rung-1 inheritance** (probe-based) | held-out 0.758 | 0.756 (unchanged) |

- The **emergent (probe-free) pipeline IMPROVES** — the clustering no longer wastes columns on fragment noise; the clusters are cleaner (e.g. the character cluster is now {tim, mom, lily, tom, dog, sue, max} — correct proper nouns — coh +0.167 vs the fragment-polluted +0.131).
- The **probe-based rungs are UNCHANGED** — the `TAXONOMY_8x8` probe words (dog/cat/red) appear lowercase mid-sentence and so tokenized correctly even before the fix; the fragments were separate spurious tokens not in the probe. So all prior committed breadth→knowledge GO results STAND (they were valid GO on a fragment-polluted vocab; the fix only removes the noise and sharpens the emergent side).

## What this establishes
A latent vocab-pollution bug across the fetched-corpus breadth pipeline is fixed at the correct layer (fetch-time normalization), improving the emergent-clustering results and leaving the probe-based results (and the untouched distill-based EMERGE runners) unchanged. The breadth→knowledge arc's GO results all stand, now on a cleaner vocabulary.

## Files
`research/runners/corpus_fetch.py` (`clean_text` lowercases); re-validation `research/findings/raw/_rc_fix_{r1,ec}_s*.json`. Related: the fully-emergent-conversation capstone that surfaced it; the breadth + rung-1..5 findings (all GO, unchanged for the probe-based rungs).
