# Direction A v1 multi-seed STRICT TOP-1 = 0.333 (BELOW 0.80 bar; engram load-bearing, ec_context cue NOT load-bearing)

**Date:** 2026-05-24
**Status:** BOUNDARY (above chance, below bar; precise diagnostic)
**Frozen bar:** 0.80 multi-seed STRICT TOP-1 (NEVER tuned)
**No-confab moat:** 7/7 green throughout
**Protected set:** byte-empty diff e8a99a2..HEAD

## Headline

Direction A ec_context-based sequence storage on the v16 substrate
shows PARTIAL positional binding: strict top-1 multi-seed mean
**0.333** (5x chance ~0.0625, but well below the frozen 0.80 bar).
The original top-3 metric (0.875 multi-seed) was DEGENERATE per
adversarial reviewer (the engram captures all 3 slot-word concept
pools, stim drives all 3, top-3 is automatic). The honest mechanism
strength is the strict top-1 result.

## Background

The (c) generative-replay arc (2026-05-23 night to 2026-05-24 morning)
identified the substrate's REPLAY_DOESNT_REACTIVATE bound: SIMULTANEOUS
engrams work (multitag 91.7% multi-seed at n=100/n=101) but SEQUENTIAL
slot-position structure does not. Direction A tested the catalog-
grounded fix (D.01+D.02+D.11; ec_context spatial positional binding):
per-slot encoding via SIMULTANEOUS lang_input(word) + ec_context(pos)
drive + teacher current on target pool.

## Critical bug caught in flight (commit 219ff2a)

Original launch (commit 7330dd7) had region_filter=["ca3"] for
commit_engram_tag, but the v16+ec_context substrate has NO ca3 region.
Same exact bug as the 2026-05-14 multitag NEGATIVE (corrected at
cbcabf2). Killed run before wasted GPU; fixed both bugs (region_filter
= 16 concept-pool regions matching validated multitag; TEACHER_PA=500
on per-slot target pool). Encoding smoke verified 8/8 sequences engram
100 neurons each (commit 219ff2a + 95306ce).

## Reviewer caught critical methodology defect BEFORE multi-seed completion

Adversarial reviewer (fresh-agent subagent a9c7a4475ca26c33a; VERDICT
BLOCK): the top-3 readout was DEGENERATE -- the engram captures all 3
slot-word concept pools (each fired by 60 steps of TEACHER_PA=500),
stim drives all 3, lang_output cosines all 3 slot words near-
equivalently; including true slot-3 word in top-3 is AUTOMATIC for any
sequence where multitag set-membership works. The 0.875 top-3 PASS
tests multitag set-membership, NOT sequence-position retrieval.

Three STRENGTHEN-only fixes (no bar tuning):
1. Strict top-1 post-processor (commit 95306ce)
2. Smell test top-1 metrics + verdict logic (commit 72397a1)
3. Capacity sweep clarification (commit 9315b8e)

## Multi-seed v1 results

| Metric | Multi-seed mean | Per-seed |
|--------|-----------------|----------|
| top-3 (degenerate) | 0.875 | [0.875, 0.750, 1.000] |
| **strict top-1 (load-bearing)** | **0.333** | [0.250, 0.375, 0.375] |
| chance top-1 | 0.0625 | -- |

The 0.333 is 5x chance; mechanism is doing SOMETHING above chance, but
well below the 0.80 bar.

## Smell test top-1 results (multi-seed)

| Control | top-1 | Margin (main - control) | Interpretation |
|---------|-------|--------------------------|----------------|
| Main | 0.333 | -- | -- |
| (A) wrong-position cue | 0.375 | **-0.042** | Wrong-position cue does AS WELL as right cue; ec_context positional cue not adding selectivity |
| (B) no-stim (cue only) | 0.000 | **+0.333** | Cue alone produces NO correct answers; engram tag IS load-bearing |
| (C) no-cue (stim only) | 0.292 | **+0.042** | Cue adds ~4% over engram alone; ec_context positional cue marginally load-bearing |

**Diagnosis:** the engram tag is the ONLY load-bearing piece. The
ec_context positional cue is NOT providing positional selectivity --
the mechanism collapses to multitag set-membership with random
positional cue effects.

## Weight inspection (commit 6d2b9f3 runner)

Inspected ec_context → pool weights in cached seed-42 substrate. ALL
ec_context → pool pathway weights are UNIFORM at ~3.0 mean (std 0.6):

| Target region | n_edges | mean | abs_mean | std |
|---------------|---------|------|----------|-----|
| noun_pool_APPLE | 12086 | 2.9955 | 2.9955 | 0.6018 |
| noun_pool_RIVER | 11938 | 3.0081 | 3.0081 | 0.5990 |
| ... | ... | ... | ... | ... |
| motor_W | 11914 | 2.9934 | 2.9934 | 0.5977 |

All 16 target regions have indistinguishable weight distributions.
This is INITIALIZATION BASELINE; STDP did not differentially modify
the weights during v1's encoding (because plasticity gates were
FROZEN during encoding per the v1 design).

## Root cause + v2 hypothesis

The ec_context cue at retrieval drives ALL pools equally (all
weights uniform at 3.0). The cue's only "positional" information
comes from the engram tag's co-occurrence (which ec_context neurons
fired during which slot's encoding window are tagged together). This
mechanism is WEAK because:
- Engram top-K captures the strongest-firing 100 neurons across the
  full encoding window
- ec_context neurons for all 3 slots are roughly equally represented
  in the tag
- Stim of the tag activates ec_context neurons from all slots
- Cue at retrieval adds ec_context(slot 2) externally, but those
  neurons drive all pools equally
- Net result: all 3 slot-words activate roughly equally; top-1 is
  ~chance-among-the-3 = 1/3 = 0.333

The 0.333 strict top-1 matches the "pick random among 3 in-sequence
words" baseline. This IS what the data shows.

## v2 fix (LAUNCHED as background task bec7fnkby)

v2 opens ec_context_to_pool plasticity gates DURING the encoding
window for each sequence; closes after. With plasticity OPEN, STDP
strengthens connections from firing ec_context(slot_i) neurons to
firing slot_i_word's pool neurons. After encoding, the cue at retrieval
should selectively drive the right pool MORE than other pools.

Estimated v2 wall: ~30 min GPU (reuses cached trained bridges; only
re-runs encoding+retrieval portion). Pre-registered FROZEN bar
UNCHANGED at 0.80 multi-seed STRICT TOP-1.

## Honest scope

- v1 strict top-1 0.333 is the HONEST mechanism strength on the
  ec_context substrate with v1's frozen-plasticity recipe.
- The 0.875 top-3 PASS was DEGENERATE; do NOT claim Direction A v1
  as a substrate capability.
- v2 may or may not push past 0.80; if it does, pillar n=104 candidate;
  if not, the substrate's ec_context-based positional binding is
  bounded and Direction E theta-gamma substrate (temporal phase code;
  ALGEBRA validated as pillar n=103) is the next-decisive direction.

## Discipline preserved

- Bar FROZEN at 0.80 multi-seed STRICT TOP-1 throughout (the bar was
  the same in v1 design and v2; STRENGTHEN-only fixes applied to the
  measurement methodology, not the bar)
- No protected/frozen/moat module modified (e8a99a2..HEAD byte-empty
  diff across the full protected set)
- No autograd
- No-confab moat 7/7 green
- Reuse-by-import only
- Both remotes (origin + gitea) propagated throughout

## Next chain step

v2 in flight (~30 min). When v2 completes, post_v2_chain auto-runs
weight inspection + final next-direction recommendation. If v2 PASS:
pillar n=104 + capacity sweep + Direction E substrate complementary.
If v2 PARTIAL/NEGATIVE: pivot to Direction E substrate Task 0+1
(theta-gamma temporal phase code; ALGEBRA pillar n=103 already
validated).
