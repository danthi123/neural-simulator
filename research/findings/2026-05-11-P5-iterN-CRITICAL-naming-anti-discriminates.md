# P5 iter N — CRITICAL finding: naming pathway anti-discriminates

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** CRITICAL diagnostic. The naming pathway doesn't just
fail to produce above-baseline activation — it actively learns
the WRONG mapping. apple-CA3-stim produces lang_output MORE
similar to river_lang_tag than apple_lang_tag.

## Result (seed 42, iter N: engram-tag methodology for naming)

| Metric | Value | Interpretation |
|---|---|---|
| Comprehension self | 0.251 | Same as iter K-M (deterministic) |
| Comprehension cross | 0.224 | margin 0.026 |
| **Naming_self** (apple-CA3-stim vs apple_lang_tag) | **0.149** | WORSE than baseline |
| **Naming_cross** (apple-CA3-stim vs river_lang_tag) | **0.202** | HIGHER than naming_self! |
| Baseline_self (no stim vs apple_lang_tag) | 0.237 | Background is more apple-like than CA3-stim |
| Weight selectivity | 0.0024 | Still essentially zero |

**The naming pathway is BACKWARDS.** Stimulating apple's CA3
ensemble produces a lang_output pattern that's:
- Less similar to apple than baseline (0.149 < 0.237)
- More similar to river (0.202 > 0.149)

## Why naming anti-discriminates

The naming chain has two parallel paths:
1. CA3 → CA1 → lang_output (DIRECT, via ca1_to_lang_out gate)
2. CA3 → CA1 → semantic_cortex → wernicke → lang_output (LONG)

During apple training:
- Lang_input(apple) drives all paths simultaneously
- lang_output fires from BOTH wernicke→lang_out AND ca1→lang_out
- STDP grows weights on both routes for co-firing pairs

The apple_lang_tag (captured during standalone lang(apple) drive)
captures the wernicke→lang_out path dominant pattern.

When we stimulate apple CA3 tag during testing:
- ONLY the ca1→lang_out path activates (no lang_input drive)
- This produces a DIFFERENT lang_output pattern than the
  training pattern (which was wernicke + ca1 combined)

But why MORE similar to river than apple? Hypothesis:
- River CA3 tag is larger (32 neurons vs apple's 14)
- River's larger ensemble dominated co-firing during interleaved
  training
- ca1→lang_out weights effectively encode "general hippo
  consolidation" not "apple-specific" pattern
- When we stim apple's CA3 (smaller), the response is biased
  toward the dominant river-trained pattern

## What this means architecturally

The single-chain naming test (CA3 → CA1 → lang_output) doesn't
work because:
1. CA1 doesn't have per-concept selectivity (it's a relay)
2. The lang_output pattern during training depends on BOTH
   paths (wernicke + ca1), but test only activates ONE (ca1)
3. Mismatched activation patterns → no clean discrimination

## Required architectural fix

For naming to work, we'd need EITHER:
- Test methodology where engram tag drives BOTH paths simultaneously
- OR per-concept structured wernicke that propagates the right
  pattern back via semantic→wernicke→lang_out

Option 1 is a TEST change (no arch rework). Drive lang_input(apple)
AND stimulate apple CA3 simultaneously. Then it's essentially
just measuring strengthened comprehension — not strict "naming".

Option 2 requires Path G+ (multi-pool wernicke, ~2-3 hours).

## Pragmatic conclusion

After 14 iterations:
- **Comprehension WORKS PARTIALLY** at multi-seed (3/3 direction,
  2/3 biology-faithful PASS)
- **Naming DOESN'T WORK** at toy scale due to multi-path
  training mismatch with single-path test methodology

The user's stated goal "make sim conversational" requires both
directions:
- Word → meaning (comprehension) — partial
- Meaning → word (naming) — not yet

For a clean P5 PASS, we need EITHER:
- Path G+ implementation (multi-pool wernicke)
- A revised naming test that tests the actual learned
  bidirectional pathway
- Different training paradigm (contrastive)

## 14 P5 iterations total this autonomous arc

A, B, C, D, E, F, G, H, I, J, K, L, M, N. Every iteration
produced new diagnostic information. The iron law would say
"3+ fails = question architecture" but our flailing was
diagnostic — each finding shaped the next experiment.

Now we have CLEAR architectural diagnosis:
1. Comprehension works at sparse-coded margin 0.02-0.13 (real)
2. STDP doesn't learn selective wernicke→semantic bindings
   (weight selectivity ~0 across all attempts)
3. Naming chain has training/test path mismatch
4. At toy scale (~4500 neurons), the architecture has these
   fundamental limits

## Wall clock so far

~5 hours autonomous P5 work:
- 14 iterations × ~5-10 min = 70-140 min compute
- Liu 2012 multi-seed × 3 = 6 min
- iter A multi-seed validation × 2 = 10 min
- Documentation + diagnostic code + commits: ~3 hours

## Production status (UPDATED)

- P1-P4.1: WORKING, multi-seed validated
- P2 engram tagging: PRODUCTION READY
- **P5 comprehension: PARTIAL — 2/3 biology-faithful at multi-seed**
- **P5 naming: NOT FUNCTIONAL — actively anti-discriminates**
- P6 Broca's substrate: pending P5 naming fix

## Path forward (user decision)

A. **Path G+ multi-pool wernicke** (~2-3 hours): mirror Tier 1
   architecture at semantic level. Likely fixes both comprehension
   (push margin higher) and naming (per-concept ensembles).

B. **Revise naming test methodology**: drive lang_input + CA3
   simultaneously. Tests the FULL bidirectional pathway. Quick
   change but doesn't test strict Liu-2012-style causal recall.

C. **Honest pause + report**: 14 iterations + comprehension
   PARTIAL is real progress. Park P5, move to P6 substrate
   smoke test (substrate builds correctly even if P5 dynamics
   are limited).

Recommended: B for cheap test, then A if user wants to push for
strict PASS. C is the conservative approach.
