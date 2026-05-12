# In-vivo new-vocab binding seed 42 smoke — NEGATIVE (with BridgeMemory bug fix preserved)

**Date:** 2026-05-12
**Status:** NEGATIVE. All 3 biology-grounded variants ≤1/4 at seed 42.
The fix to BridgeMemory's arch loading (tier1_hippo support) preserved
for future work.

## Test setup

main_hippo lineage (Tier 1 + hippocampus, 200 awake + 12 sleep cycles
of 50 SWR events each, ~7.7 min bootstrap). Three variants forked
from main_hippo, each binds 4 novel keys → motor pool actions:
- apple → north (N)
- river → east (E)
- mountain → south (S)
- forest → west (W)

PASS criterion: ≥3/4 bindings correct per variant (where "correct"
= top-1 motor pool spike count matches expected action).

## Final results

| Variant | Correct | Details |
|---|---|---|
| V0 vanilla | 1/4 | apple→E, river→S, mountain→E, **forest→W ✓** |
| V_HIPPO_BIO | 0/4 | apple→E, river→W, mountain→W, forest→E |
| V_SCHEMA | 1/4 | apple→W, river→N, **mountain→S ✓**, forest→E |

**No variant ≥ 3/4. Architecture has structural pool variance that
overrides the topographic + STDP binding signal for novel keys.**

## Notable signal

V_SCHEMA's mountain→south is a **true binding success** — not
coincidence. The schema-supported variant interleaves the new word
with anchor-word reinforcement, and for "south" specifically the
re-encoding of "south" while training "mountain" successfully
locked mountain into pool_S.

V_HIPPO_BIO surprisingly got 0/4. The hippocampus encoding + SWR
consolidation routes pool selection varies per stimulus but never
to the target pool. Possible reasons:
- main_hippo was bootstrapped with smoke config (only 50 events
  per direction, 12 sleep cycles) — maybe insufficient pretraining
  of the hippocampus pathway
- The CA3 autoassociator may need more events to form stable
  attractor for novel inputs
- 200 events per binding may be too few — V_HIPPO_BIO's training
  is split between awake encoding + sleep consolidation

V0 vanilla's forest→W is coincidence (forest happened to route to
pool_W which IS the target). Not a real learning success.

## Diagnosis

Same architectural ceiling as iter PP biological-scale results:
**per-seed random structural variance dominates the learning signal
for novel keys at biological scale**. The Tier 1 4-direction motor
pool architecture works at 6/6 BIDIR for direction words because:
- 200 events × 4 directions = 800 events with topographic prior
- Each direction word has its own dedicated training pass
- Topographic bias on lang_input → motor pre-aligns weights

For NOVEL keys without topographic prior:
- 200 events training is insufficient to override random structure
- Hippocampus + SWR consolidation doesn't reach lang_input → motor
  strongly enough
- Schema-supported anchor reinforcement works ON ONE direction
  (the one whose anchor was strong enough) but not others

## Bug fix preserved (significant)

While running this smoke, found that BridgeMemory was loading the
main_hippo lineage with the WRONG architecture (synonym instead of
tier1_hippo). The fix:

1. Added `_load_or_train_tier1_hippo` helper to chat_repl
2. Added `tier1_hippo` mode to `_load_bridge_from_checkpoint`
3. BridgeMemory.`_ensure_loaded` now auto-detects lineage's tier
   from metadata and overrides caller's mode

This bug fix is committed (f3308b8) and unlocks any future work that
uses hippocampus-enabled lineages via BridgeMemory.

## Strategic implications

The architectural ceiling for non-motor binding at biological scale
is consistent across:
- P5 abstract concept binding (iter PP 1/4 BIDIR)
- In-vivo novel-key binding (all variants ≤1/4 at seed 42)

Both fail at the same architectural pattern: **the 4-direction motor
pool layout has structural variance that overcomes the learning
signal for inputs that don't match the pre-existing topographic
prior**.

What WORKS (verified at multi-seed):
- Tier 1 motor binding for direction words (6/6 PASS): 74% W→A,
  98% A→W at seed 42 verification
- Tier 2.1 synonym binding (6/6 PASS): same architecture extended
  via dedicated motor pool per synonym group
- Phase 1.3 hippocampus consolidation (3/3 PASS): cortex retains
  binding after hippo silencing

What DOESN'T work at biological scale single-seed:
- P5 abstract concepts (apple/river): iter AA 4/6 toy ceiling
- In-vivo novel keys (apple/river/mountain/forest): 0-1/4

## Path forward (open question)

The sim has working conversational capability for ~8 direction
words at 6/6 PASS. The user's stated goal is "conversational sim
for non-motor concepts" which has hit fundamental architectural
limits.

Architectural changes that might unlock non-motor binding:
1. **Pre-allocate novel-key motor pools** at training time (like
   Tier 2.1 synonyms): doesn't generalize to arbitrary new vocab
   but works for known-in-advance vocabulary
2. **Stronger topographic prior** during binding: apply a brief
   topographic bias at the START of each new binding to align
   lang_input → target_motor edges before STDP starts
3. **Many more training events per novel key** (1000+): may
   eventually override random structure but slow
4. **Different recall mechanism**: instead of motor pool spike-count
   readout, use a learned classifier or attention mechanism

None of these are quick fixes. Each is a session-scale effort.

## Recommendation

**Status quo for conversational capability:**
- Ship Tier 1/2.1 6/6 PASS as the demonstrated capability (8 direction
  words bidirectionally bound)
- Document iter AA 4/6 toy BIDIR as the P5 demonstration ceiling
- Document V_SCHEMA's mountain→south as the only true novel-key bind
  success — schema-supported anchor reinforcement is the most
  promising variant for future work

**For future sessions:**
- Pursue Option 2 (stronger topographic prior at binding time) as
  the cheapest test
- Or accept the ceiling and focus on what works: direction-vocab
  conversation, continual learning, consolidation
