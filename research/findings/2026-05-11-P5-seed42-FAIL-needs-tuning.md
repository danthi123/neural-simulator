# P5 ventral semantic stream seed 42: FAIL on first validation

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Catalog:** G.11 + G.13
**Status:** Substrate works (regions + pathways build correctly), but
dynamics on first validation FAIL both criteria. Multi-seed (43, 44)
running — may show seed variance.

## Result (seed 42)

**Test 1 — Comprehension** (drive lang_input, measure semantic_cortex):
- apple trial 1 vs trial 2 cosine: **0.216** (target > 0.6 — same
  concept should reactivate the same pattern)
- apple vs river cosine: **0.290** (target < 0.3 — different concepts
  should be distinguishable)
- **FAIL**

Critically, `apple_self=0.216 < apple_river=0.290` — same-concept
reactivation is WEAKER than cross-concept. The semantic_cortex output
isn't tracking the lang_input identity stably.

**Test 2 — Naming** (stimulate apple CA3 tag, measure lang_output):
- Baseline lang_output spikes: 407
- Engram-driven lang_output spikes: 361
- Ratio: **0.89x** (target > 1.3x)
- **FAIL**

Causal stimulation of the CA3 engram tag produces LESS lang_output
than baseline. The CA3 → semantic_cortex → wernicke → lang_output
chain isn't propagating the engram signal.

## Diagnosis (biology-first workflow step 5)

Per Rule 8: if the biology copy doesn't work, return to step 3 or
check implementation detail.

Likely implementation issues:

1. **Training events insufficient.** With only 100 events per concept,
   lang→wernicke→semantic_cortex (two synapses) doesn't accumulate
   enough STDP. Direct hippo path (lang→ec→dg→ca3) is faster because
   the mossy-fiber synapse is strong. The cortical path needs more
   exposures.

2. **Mixed gate timing during training.** My encoding code opens ALL
   gates simultaneously (hippo + ventral). Per McClelland 1995 CLS:
   wake should encode in hippo first, sleep should consolidate to
   cortex via ca1→semantic_cortex. Mixing them dilutes STDP signal.

3. **Wernicke too small (200 neurons).** Two-layer chain
   (lang_input(2048) → wernicke(200) → semantic_cortex(1000))
   bottlenecks at wernicke. Random init weights + sparse pretraining
   may make wernicke fire too randomly.

4. **Test methodology may need engram-tag approach.** Currently I'm
   comparing semantic_cortex spike counts across trials directly.
   A cleaner test: TAG the semantic_cortex ensemble during one trial,
   then check whether subsequent trials reactivate the same ensemble.
   This is the same upgrade that turned P1 D.13 from "FAIL strict"
   to "PASS biology-faithful."

## Substrate IS correct (smoke test passes)

The regions + pathways construct correctly:
- semantic_cortex (1000 neurons) + wernicke (200 neurons) regions
  present
- 5 plastic pathways wired (lang_to_wernicke, wernicke_to_semantic,
  semantic_to_wernicke, wernicke_to_lang_out, ca1_to_semantic)
- Prereq check raises ValueError if used without
  enable_hippocampus_consolidation

The architecture is in place. The dynamics need tuning.

## Path forward

Wait for seeds 43, 44 to see if variance is seed-dependent. If all
3 seeds FAIL similarly, do biology-first workflow step 5 again:

**Tune option A — More training**: 100 → 500 events per concept.
**Tune option B — Two-stage gating**: wake = hippo only;
sleep = ca1→semantic_cortex only. STRICT McClelland 1995 CLS.
**Tune option C — Engram-tag test methodology**: tag semantic_cortex
ensemble explicitly, then test reactivation (P1 D.13's pattern).

Multi-seed completion will tell us which tuning to try.

This is the first P-phase to FAIL out of the gate (P1, P2, P3.1, P4.1
all worked). Expected: detailed designs from memory often need
implementation iteration. The catalog-grounded approach gives us the
RIGHT mechanism (G.11 + G.13); finding parameters that make it work
in the sim is the next step.

## What's still strong

- P1 trisynaptic loop: 3/3 multi-seed biology-faithful PASS
- P2 engram-tagging: 12 tests pass, persistence works
- P3.1 concept replay: 5 tests pass, used in P5 (works mechanically)
- P4.1 positional context: 3/3 multi-seed PASS
- P6 Broca's substrate: builds correctly, prereq checks raise

The P1+P2+P3.1+P4.1 chain works. P5's INTEGRATION just needs more
iteration.
