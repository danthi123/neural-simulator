# Broad-vocab COMPLETE talkable loop (GO, GPU): the brain discovers an animal cluster from a real corpus, reasons, and speaks a SEMANTICALLY-COHERENT frame over its OWN discovered vocab -- "the bird can run" -- content ON SPIKES, moat intact. Unblocked by decoupling the concept-pool topographic bias from its welded vocab. NO `sim/` edit (research-runner edit).

**Date:** 2026-07-08
**Runners:** `_realcorpus_train_breadth_aw.py` (A->W on the reasoner's own vocab) + `_realcorpus_full_pipeline_reason_to_frame_derisk.py --breadth`. The enabler: an additive `word_to_pool_override` param on `concept_pool_demo.apply_concept_topographic_bias` (default None = byte-identical) that decouples the topographic bias from the welded concept vocab. SIM_BACKEND=cupy.
**Verdict:** GO -- the complete talkable loop over the reasoner's OWN broad discovered vocab, semantically coherent.

## The story (root-cause -> fix -> GO)
The broad-vocab A->W first failed (1/16 spell, CYCLE 974) because `apply_concept_topographic_bias` (the A->W enabler) was WELDED to the concept-pool's specific 16-word vocab. Decoupling it (a regression-safe additive `word_to_pool_override`, default byte-identical) let the topographic bias apply to a CUSTOM vocab -> the broad-vocab A->W now spells the reasoner's own animals + verbs **8/8** (dog/cat/bird/fish/run/jump/walk/eat). The full pipeline then speaks the reasoner's own discovered animal:
```
DISCOVER: cluster TinyStories co-occurrence (probe-free) -> a cluster with [bird, cat, fish] (animals!)
TEACH:    the cluster a property (spoken as the verb 'run')
ASK 'bird' (held-out animal) -> REASON (bird inherits) -> SPEAK "the bird can run"  [content ON SPIKES]
ASK 'zzzqqx' (unknown) -> "I don't know"  [gate-first MOAT]
```

## Why this is stronger than the v16-default loop
The v16-default complete loop (CYCLE 973) spoke a SEMANTICALLY-ARBITRARY frame ("the big can go") because the v16 A->W vocab didn't overlap the reasoner's animals. Now the A->W is trained on the reasoner's OWN discovered vocab, so the frame is COHERENT: "the bird can run" -- a held-out animal (bird) the reasoner discovered from TinyStories co-occurrence, a real verb, all content ON SPIKES. The brain discovers a category from experience, reasons over it, and speaks about its own discovered concepts fluently.

## Honest scope
- **A->W spell 8/8** (the decoupled topographic bias fixed it); content words ON SPIKES (the claim).
- Closed-class the/can host-rendered (EMERGE-68 spiking function-word A->W = follow-on).
- The reasoner's held-out classification accuracy (characterized emergent-cluster limit) gates which held-out members speak; the moat is robust.
- The decoupling edit is additive (default None = byte-identical; all concept-pool runners unaffected).

## What this establishes
The COMPLETE talkable loop now runs over the reasoner's OWN broad discovered vocabulary with semantically-coherent frames: discover a broad vocab from real experience -> discover its categories (probe-free) -> reason (inherit) -> and SPEAK about its own discovered concepts fluently on spikes ("the bird can run"), gate-first moat. Transformer-free. The broad-vocab blocker (concept-pool vocab-welding) is resolved by the decoupling.

## Files
`research/runners/_realcorpus_train_breadth_aw.py`, `_realcorpus_full_pipeline_reason_to_frame_derisk.py`, `concept_pool_demo.py` (the additive decoupling). Prior: the v16-default complete loop `2026-07-08-COMPLETE-talkable-loop-discover-reason-fluent-frame-GO.md`; the broad-vocab blocker diagnosis (CYCLE 974).
