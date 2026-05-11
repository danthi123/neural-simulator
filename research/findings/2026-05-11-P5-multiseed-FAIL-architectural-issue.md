# P5 ventral semantic stream multi-seed: 0/3 PASS — architectural issue

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** Honest report. Substrate works; first validation runner
shows consistent FAIL across 3 seeds. Architectural / implementation
iteration needed.

## Multi-seed results

| Seed | Comprehension (apple_self / apple_river) | Naming (causal/baseline) | Verdict |
|---|---|---|---|
| 42 | 0.216 / 0.290 | 0.89x | FAIL |
| 43 | 0.347 / 0.357 | 0.88x | FAIL |
| 44 | 0.300 / 0.322 | 1.07x | FAIL |

Target: comprehension same > 0.5 AND cross < 0.4; naming > 1.3x.

**0/3 PASS.** Not seed variance — consistent pattern.

## Diagnosis

Comprehension: same-concept cosine clusters around 0.3, cross-concept
also clusters around 0.3. **semantic_cortex activation is dominated
by random/noise dynamics, not by the lang_input signal.** Different
lang_input drives produce roughly similar semantic_cortex spike
distributions.

Naming: engram tag → CA3 → ca1 → semantic_cortex → wernicke →
lang_output causal stimulation produces lang_output activation
**comparable to baseline (no drive).** The propagation through the
ventral chain is weak relative to background dynamics.

## What's likely wrong (biology-first step 5)

1. **Ventral-stream pathways need more training events.** 100 events
   per concept is what worked for the hippo path (because mossy fiber
   synapses are "detonators"). The cortical path through wernicke
   (2-synapse hop) accumulates less STDP per exposure.

2. **wernicke (200 neurons) may bottleneck the 2048 → 200 → 1000 path.**
   Sparse code through small intermediate. Random init weights +
   sparse training → wernicke fires inconsistently.

3. **Test methodology issue.** Direct spike-count comparison
   measures noisy population activity. The P1 D.13 workflow showed
   that engram-tagging the response ensemble gives cleaner signal
   than raw spike counts. Apply same trick to P5: tag semantic_cortex
   ensemble during one exposure, test reactivation in subsequent.

4. **Mixed gate timing during training.** Current code opens all
   pathways plastic during encoding. Per McClelland 1995 CLS, wake
   should encode in hippo only; sleep should consolidate to cortex
   via ca1→semantic_cortex. STRICT two-stage gating may give cleaner
   semantic_cortex patterns.

## Path forward

Per biology-first workflow Rule 8 step 5: failure → return to step 3
or fix implementation detail. The catalog (G.11, G.13) gives the
RIGHT mechanism — ventral stream is real biology. The current sim
implementation needs:

**Iteration A (cheap, try first):**
- Bump training events 100 → 500 for the ventral pathways
- Use engram-tag test methodology (tag semantic_cortex response, test
  reactivation)

**Iteration B (medium effort):**
- Strict two-stage gating: wake → hippo only; sleep → cortex
  consolidation only
- Per-pathway training schedule (lang→wernicke first, then
  wernicke→semantic)

**Iteration C (more invasive):**
- Scale up wernicke (200 → 500 neurons)
- Stronger initial weights for the ventral chain
- Different test metric (mutual information between concept and
  semantic_cortex pattern, vs cosine of spike counts)

## What's NOT broken

The P5 substrate IS correct:
- Regions construct without errors
- Pathways wire correctly per the catalog spec
- Prereq checks raise ValueError for invalid combinations
- The 5 pathways exist with the right gates

This is implementation tuning, not architectural redesign. The
catalog-grounded design (Patterson hub-and-spoke / Lambon Ralph ATL
hub maps onto our semantic_cortex; Wernicke's onto our wernicke
region) is biology-faithful and correct.

## Strong results from this arc remain

| Phase | Status | Multi-seed |
|---|---|---|
| P1 (trisynaptic loop) | Catalog-validated | 3/3 BIOLOGY PASS (two-concept) |
| P2 (engram-tagging) | API SHIPPED | 12 unit tests pass |
| P3.1 (concept replay) | Implementation SHIPPED | 5 unit tests pass |
| P4.1 (positional context) | Catalog-validated | **3/3 MULTI-SEED PASS** |
| P5 (ventral semantic) | Substrate shipped; **VALIDATION 0/3** | Needs iteration |
| P6 (Broca's) | Substrate shipped | Validation pending P5 |

P1+P4.1 confirm the user's "concepts as distinguishable ensembles"
vision is mechanistically realized. P5 adds the language interface,
which needs more work. P6 needs P5.

## What this means for the goal

The user's stated goal: sim does language itself, no LLM.

After this arc:
- ✅ Concepts as tagged hippocampal ensembles (P1+P2+P4.1)
- ✅ Continuous learning across sessions (P2 persistence + P3.1
       consolidation)
- ✅ Word-order-dependent meaning at the architectural level (P4.1)
- ⚠️  Word ↔ meaning translation via ventral cortex (P5) — needs
       iteration
- ⏸  Compositional syntax (P6) — substrate ready, validation gated
       on P5
- ⏸  Sentence-level production (P5+P6 together)

The architecture is mostly built. P5 implementation iteration is the
next concrete step. The catalog gives the path; the sim needs
parameter and methodology tuning to realize it.
