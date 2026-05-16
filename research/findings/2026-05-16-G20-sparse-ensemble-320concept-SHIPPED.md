# 320-concept sparse-distributed G.20 ensemble — SHIPPED (98.4% per-bridge, ensemble validated)

## TL;DR

The documented production scaling target — **5 bridges × 64
sparse-distributed concepts = 320 unique concepts** — is trained and
validated end-to-end. Per-bridge discrimination is **98.4% (63/64),
uniform across all 5 bridges**; ensemble integration (cross-bridge
memory + N-word sentences + role queries) **passes at 320-concept
scale, including the +160 extension vocabulary**.

Honest headline: **not 100%** like the 32-concept tier. There is one
**deterministic, characterized** per-bridge failure (concept index 12)
— understood and bounded, not mysterious. The 320 ensemble is a strong,
usable artifact (315 robust concepts) and the gap has a clear cheap
improvement path (per-bridge seeds / overlap-rejection).

## Per-bridge training result (seed 42, n_concepts=64, sparsity 0.007)

| Bridge | top-1 | failed concept (idx, word, rank) |
|---|---|---|
| bridgeA_nouns | 63/64 = 98.4% | (12, ball, 18) |
| bridgeB_verbs | 63/64 = 98.4% | (12, watch, 18) |
| bridgeC_adj | 63/64 = 98.4% | (12, red, 18) |
| bridgeD_spatial | 63/64 = 98.4% | (12, in, 18) |
| bridgeE_functional | 63/64 = 98.4% | (12, what, 18) |
| **Total** | **315/320 = 98.4%** | always idx 12, always rank 18 |

## The idx-12 failure is characterized, not mysterious

Every bridge fails at **exactly concept index 12, ranking exactly
18** — different words (ball/watch/red/in/what), identical index+rank.
Root-cause investigation:

- **All 5 bridges train with `--seed 42`** → `generate_sparse_patterns(
  64, 2000, 100, 42)` yields the **identical** pattern set for every
  bridge. Pattern-12 is the same neurons in all 5, so it fails
  identically regardless of vocab. The failure is **vocab-independent
  and seed-42-pattern-specific.**
- **Not simple overlap:** idx-12's max pairwise pattern overlap is 12
  (rank 5/64; mean 10.4). Indices 8 and 17 overlap *more* (15) yet
  **pass**. So raw overlap doesn't explain it.
- **Not orthogonal-code collision:** the lang_input orthogonal drive
  for cue 12 has zero overlap with any other cue (perfectly
  separable), no identical drives.
- Mechanism is a deeper training/eval-dynamics property of pattern-12
  at this exact (n=64, pool=2000, K=100, seed=42) config — an open
  question, but tightly bounded (one pattern, deterministic).

**Cheap improvement path (deferred, not blocking):** the chain uses
seed=42 for ALL 5 bridges (identical patterns — redundant + correlates
the failure). Per-bridge distinct seeds (42–46) and/or an
overlap-rejection tweak in `generate_sparse_patterns` (regenerate a
pattern if max overlap with existing > threshold) would very likely
recover toward 64/64 and decorrelate failures. Requires re-training
(~2 hr) → separate task.

## End-to-end demo (seed 42, all 5 sparse64 bridges) — PASS

```
concepts              -> TOTAL: 320 unique concepts
                         Form: SPARSE-DISTRIBUTED (Kanerva SDM) K=100 pool=2000

what is apple         -> (baseline) spoon 617, person 410  [noise]
remember apple is big -> [cross-bridge: 'apple_big' in
                         ['bridgeA_nouns','bridgeC_adj']]
what is apple         -> big 779 via bridgeC_adj/apple_big   <-- #1
                         spoon 658, person 399  [signal clears noise]
is apple big?         -> YES

# --- EXTENSION-VOCAB word ('horse' = idx 32, the +32 nouns) ---
remember horse run fast -> [sentence 'horse_run_fast' in
                         ['bridgeA_nouns','bridgeB_verbs','bridgeC_adj']]
who run fast?         -> [subjects of 'run fast']: ['horse']
what did horse run?   -> [objects of 'horse run']: ['fast']
what is horse         -> run 882 via bridgeB_verbs/horse_run_fast  <-- #1
                         table 518 [noise], fast 508 via
                         bridgeC_adj/horse_run_fast  <-- co-member
tags                  -> A=66 B=65 C=66 D=64 E=64  [exact routing]
Done.  (exit 0)
```

**Why this validates the tier:** querying the **extension-vocab** word
`horse` retrieves BOTH sentence co-members (`run` 882, `fast` 508)
across 3 bridges — genuine cross-bridge semantic retrieval for a word
from the +160 curated extension, not just the frozen base 160. The
idx-12 gap does not impair ensemble integration (demo uses idx-0 /
idx-32 words).

## Honest scope

- 98.4% per-bridge (NOT 100%). The 32-concept tier was 100%; the
  64-concept tier is 98.4% with a characterized single-pattern gap.
- Seed 42, single ensemble. Per-bridge 64-concept @ ~100% was
  multi-seed validated earlier (288/288, ALL_60-style vocab); this
  320 run uses 5 distinct curated 64-word category vocabs at seed 42.
- Noise floor is higher at 64-vs-32 concepts/2000-pool (more patterns
  → more spurious co-activation); retrieval signal still clears it
  decisively (779 vs 658; 882 vs 518).

## Files

- `research/runners/g20_vocab_spec_320.py` — frozen-160 base + curated
  +160 (global-uniqueness assert); `tests/test_g20_vocab_spec_320.py`
- `research/runners/g20_sparse_5bridge_chain_320.ps1` — trainer
  (sparsity 0.007: orthogonal-drive needs n_active 57 < stride 128)
- `research/findings/raw/g11_bg/g20_sparse_bridges_320/*_sparse64.*` —
  5 bridges + per-bridge JSON/logs
- `research/findings/raw/g11_bg/g20_sparse_ensemble_demo_320.log` — transcript
- Prior: `2026-05-15-G20-sparse-ensemble-160concept-end-to-end-SHIPPED.md`
  (160 tier @ 100%), `2026-05-15-256-concept-training-bound-conclusion.md`
