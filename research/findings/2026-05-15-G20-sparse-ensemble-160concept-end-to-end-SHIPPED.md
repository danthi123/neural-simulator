---
type: finding
status: superseded
superseded_by: research/findings/2026-05-16-G20-sparse-160-multiseed-VALIDATED.md
date: 2026-05-15
mechanism: g20-sparse-ensemble
---

# 160-concept sparse-distributed G.20 ensemble — end-to-end SHIPPED

## TL;DR

The production conversational artifact is built and validated end-to-end:
**5 sparse-distributed (Kanerva SDM) bridges × 32 concepts = 160 unique
concepts**, every bridge trained at **100% discrimination**, loaded
through a new `g20_multibridge --sparse` mode, exercising cross-bridge
associative memory + N-word sentence role queries in a single scripted
run with **zero failures**.

This closes the path the 256-concept conclusion doc identified:
multi-bridge (N × sparse @ 100%) is the production scaling route, and
it now works through the ensemble loader, not just per-bridge.

## What shipped

| Piece | Status |
|---|---|
| 5 sparse bridges (A nouns, B verbs, C adj, D spatial, E functional) | all 32/32 top-1 = **100%** |
| `g20_multibridge --sparse` mode | shipped (commit 18f5398) |
| Sparse recall/encode helpers in `shared_pool_chat.py` | shipped + 16 CPU tests |
| Pattern-regen reproducibility invariant | pinned by tests; verified vocab order == training order (all 5) |
| End-to-end demo | **PASS** (commit 9f17454 launcher) |

## The architecture gap that was closed

`g20_multibridge.py` only spoke the **contiguous-slice** G.20 form
(concept i → neurons `[i·slice : (i+1)·slice]`). The production
ensemble is **sparse-distributed**: each concept is a scattered K-of-N
random pattern (K=100, N=2000-pool), regenerated deterministically from
the training seed. Loading the sparse bridges through the slice loader
would have read the wrong neurons and silently produced garbage.

Fix: `--sparse` mode that (a) builds via `build_sparse_pool_bridge`,
(b) regenerates per-bridge patterns from `--seed` (verified
byte-identical to training), (c) routes recall/encode through sparse
analogues, with the sparse-vs-contiguous branch centralized in
`SharedPoolMember` methods so the sentence/tokenizer/hierarchy dispatch
is reused unchanged (DRY). The contiguous path is fully preserved
(96 multibridge tests still green).

## End-to-end demo transcript (seed 42, all 5 sparse bridges)

```
concepts              -> TOTAL: 160 unique concepts across 5 bridges
                         Form: SPARSE-DISTRIBUTED (Kanerva SDM), K=100, pool=2000

what is apple         -> (pre-assoc baseline) child 453, sun 338,
                         person 331  [noise floor, no association yet]

remember apple is big -> [cross-bridge: 'apple_big' encoded in
                         ['bridgeA_nouns', 'bridgeC_adj']]

what is apple         -> big 662 via bridgeC_adj/apple_big   <-- #1
                         child 486, person 393, garden 336
                         [association retrieved cross-bridge; signal
                          662 >> noise floor ~400]

is apple big?         -> YES (tag 'apple_big' in
                         ['bridgeA_nouns','bridgeC_adj'])

remember dog run fast -> [sentence 'dog_run_fast' encoded in
                         ['bridgeA_nouns','bridgeB_verbs','bridgeC_adj']]

who run fast?         -> [subjects of 'run fast']: ['dog']
what did dog run?     -> [objects of 'dog run']: ['fast']

tags                  -> A=34 (32+apple_big+dog_run_fast)
                         B=33 (32+dog_run_fast)
                         C=34 (32+apple_big+dog_run_fast)
                         D=32  E=32   [exactly correct routing]
Done.  (exit 0, no Traceback)
```

## Why this is a genuine validation (not just plumbing)

The decisive line is `big 662 via bridgeC_adj/apple_big`. "apple"
lives in bridgeA (nouns); "big" lives in bridgeC (adjectives).
`remember apple is big` cross-bridge-encoded a shared `apple_big`
tag — in bridgeA over apple's sparse pattern, in bridgeC over big's
sparse pattern. Querying "apple" later, the **sparse recall**
(`stim_recall_sparse_rates`) stimulated that tag in bridgeC and the
scattered K-of-N pattern for "big" fired most strongly (662), clearing
the bridgeA noise floor (~330–490) decisively. This only works if the
regenerated patterns match training byte-for-byte — confirming the
reproducibility invariant the 16 CPU tests pin.

## Capability surface (all validated this run)

- 160-concept ensemble load (5 sparse bridges, 100% each)
- Cross-bridge associative memory (noun↔adjective)
- Exact cross-bridge tag match (`is X Y?`)
- N-word sentence spanning 3 bridges (`remember dog run fast`)
- Tag-name role queries (`who run fast?` → dog; `what did dog run?`
  → fast) — the v16-validated 100% multi-seed mechanism,
  architecture-independent, works identically on sparse bridges
- Path-3 hierarchy queries enabled (string-level, architecture-independent)

## Honest scope

- This is **seed 42, single ensemble**. Per-bridge 100% is multi-seed
  validated (288/288 in the capacity-curve work); the *ensemble
  integration* (cross-bridge + sentences through `--sparse`) is
  demonstrated at seed 42. Multi-seed ensemble integration is a
  cheap follow-up (the 5-bridge chain just needs re-running at other
  seeds; ~17 min/bridge).
- 160 concepts here = 5 × 32. The 5 × 64 = 320 scale-up remains the
  documented next tier (sparse-distributed validated to 64/bridge at
  100% in the capacity curve; training cost ~40 min/bridge at 64).
- Associative *quality* beyond the trained pair (transitive,
  multi-hop) is not claimed here — this validates the substrate +
  loader, not semantic reasoning depth.

## Files

- `research/runners/g20_multibridge.py` — `--sparse` mode
- `research/runners/shared_pool_chat.py` — sparse recall/encode helpers
- `research/runners/concept_pool_sparse_distributed.py` — trainer
  (unchanged; `generate_sparse_patterns` is the regen source of truth)
- `research/runners/g20_sparse_ensemble_demo.ps1` — reproducible demo
- `tests/test_g20_sparse_multibridge.py` — 16 reproducibility tests
- `research/findings/raw/g11_bg/g20_sparse_ensemble_demo.log` — transcript
- `research/findings/raw/g11_bg/g20_sparse_bridges/*.simstate.h5` — the 5 bridges
- Prior: `2026-05-15-256-concept-training-bound-conclusion.md`
  (why multi-bridge is the path), `2026-05-15-sparse-distributed-capacity-curve.md`
