---
type: biology
id: cls-interleaved-consolidation
mechanism: Complementary Learning Systems -- a FAST hippocampal store buffers new episodes, and repeated SWR-replay INTERLEAVES them with ongoing experience into a SLOW neocortical store, so consolidating new memories does not overwrite old ones (catastrophic interference avoided)
status: established
last_verified: 2026-08-26
current_finding: research/findings/2026-05-21-catastrophic-forgetting-FULL-3x3-matrix-COMPLETE-substrate-resistance-is-seed-dependent-not-regime-specific-CLS-regime-prediction-NOT-robust-multi-seed-at-any-intensity.md
current_status: "PRIOR ARC (2026-05-21) tested the WRONG instrument: whether a UNIFIED substrate's compositional-vs-direct REGIME predicts interference resistance -- that was seed-dependent, no robust regime effect. This binding grounds a DIFFERENT, cleaner claim: the CLS ARCHITECTURE itself (a separate fast store whose content is REPLAYED interleaved into the slow store) is what protects old memories, and the load-bearing variable is the replay, not a regime."
sources:
  - path: ~/Projects/sim-catalog/references/feature-catalog.md
    anchor: "gradually transfers memory from HC-dependent"
    note: "N.14 hippocampal-neocortical dialogue -- repeated coordinated SWR reactivation gradually transfers a memory from a fast HC-dependent state to a durable neocortex-dependent state; the slow cortical store is the long-term repository"
  - path: ~/Projects/sim-catalog/references/feature-catalog.md
    anchor: "two-stage memory model"
    note: "Bz Cycle 12: (1) waking theta-sequenced encoding writes CA3 recurrent weights; (2) sleep SWR-replay drives the SAME sequences into neocortex where late-LTP/synaptic-tag converts them to durable cortical traces -- the fast store SEEDS the slow store by replay"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "could be replayed multiple times, assisting with the consolidation"
    note: "the transport: a single episode is replayed MANY times, which is exactly what lets a slow (small-step) cortical learner absorb it gradually without a single write overwriting prior structure"
constraints_config:
  - key: cortex_lr
    value: 0.02
    why: "The neocortical store must be SLOW (small per-step synaptic change) -- this is the defining CLS property (McClelland, McNaughton & O'Reilly 1995): a fast cortical learner catastrophically overwrites. The slow rate is why interleaving old replay with new experience protects the old mapping; a fast rate would forget within a phase regardless of replay."
implemented_by:
  - research/runners/_cls_interleaved_consolidation_derisk.py
findings:
  - research/findings/2026-05-21-catastrophic-forgetting-FULL-3x3-matrix-COMPLETE-substrate-resistance-is-seed-dependent-not-regime-specific-CLS-regime-prediction-NOT-robust-multi-seed-at-any-intensity.md
---

# Two stores, and replay is what glues them

**The claim the code must respect.** Catalog N.14: repeated coordinated hippocampal reactivation "gradually
transfers memory from HC-dependent (recent) to neocortex-dependent (remote) state." Buzsáki's two-stage model
makes the transport explicit: waking theta-sequenced experience writes fast CA3 recurrent weights, then sleep
SWR-replay "could be replayed multiple times, assisting with the consolidation process," driving the *same*
sequences into neocortex where slow late-LTP builds a durable trace. The neocortex is the long-term repository
(see `systems-consolidation`); the hippocampus is the fast buffer that *feeds* it by replay.

**Why this prevents catastrophic interference (the McClelland-McNaughton-O'Reilly 1995 argument).** A single
slow network trained on set A, then trained sequentially on set B, overwrites A -- classic catastrophic
interference (McCloskey & Cohen 1989). The biological escape is NOT a cleverer cortical rule; it is the
*architecture*: because A lives in a fast hippocampal store, that store can **replay A interleaved with the
ongoing B experience**, so the slow cortical learner only ever sees A and B *mixed*, which is the regime in
which Hebbian/gradient learning does not forget. Two things are therefore load-bearing and are the anti-cheats
the runner must fire:
1. **Replay must be present.** Remove it (train B sequentially, cortex-only) and A is forgotten. This is the
   lesion that makes the faculty load-bearing.
2. **Replay must carry the CORRECT content.** Shuffle the replayed A associations (right inputs, wrong targets)
   and the protection vanishes -- it is not generic "extra activity" or regularization, it is the *specific*
   re-instatement of A's mapping.

**The confound this binding forces the runner to close.** Interleaving adds training steps, and *more total
training* or a *lower effective learning rate* could each look protective on their own. So the runner MUST
exposure-match: an equal-total-updates, equal-B-updates no-replay control (the extra steps spent on *more B*
instead of replayed A). If A survives only in the replay arm, replay content -- not step count or rate -- is
what protects it.

## Scope / honesty

- The cortical learner here is a **rate-coded three-factor Hebbian associator** (error-gated plasticity: the
  synapse changes with pre-activity times a post-side teaching signal). This is a documented idealization of the
  slow neocortical store, not a spiking cortex; the *biological replay generator* it consumes is the GO
  `swr-sequence-replay` organ (`_gap5_ecker_recurrent_replay.py`, ordered compressed weight-borne replay). This
  runner de-risks the CLS *function* (does replaying the fast store's content into a slow store defeat
  catastrophic interference?), reusing that replay as the source of the interleaved samples.
- McClelland, McNaughton & O'Reilly 1995 (*Psych. Review* 102:419) is the computational theory and is **not in
  the local corpus** -- it is cited from knowledge; the *biological* mechanism (fast HC store, SWR transport,
  slow cortical repository) is grounded in the catalog N.14 + Buzsáki anchors above, which resolve.
- The prior 2026-05-21 arc is NOT contradicted: it tested whether a unified substrate's compositional-vs-direct
  regime predicts resistance (it does not, robustly). This tests the architecture-level CLS claim it never
  isolated -- separate store + interleaved replay vs a single store.
