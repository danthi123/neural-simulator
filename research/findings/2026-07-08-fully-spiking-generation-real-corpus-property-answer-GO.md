# Fully-spiking GENERATION of the real-corpus property answer (GO, 3-seed): the sentence STRUCTURE ("the X can Y") is produced ON SPIKES by the EMERGE-65 self-organized spiking-Broca producer (slot order via competitive queuing + wash-out) and every WORD by the breadth A→W — replacing the host f-string template. "Simulate Broca": the order is a spiking read-out, not a host literal. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_spiking_broca_property_answer_derisk.py` (reuse-by-import: EMERGE-65 `SelfOrganizedProducer` + `BrocaProducer` + the breadth concept-pool A→W `ConceptFrameSpeaker`). numpy. NO `sim/` edit.
**Verdict:** GO (3-seed) — the property answer's structure + words are both produced on spikes, moat intact.

## Why this ran (the fluency fork, completed)
The real-corpus talkable brain speaks its answer WORDS on spikes (A→W) but the sentence STRUCTURE ("the X can Y") was a host f-string. The fluency fork's fully-spiking generation was scoped (CYCLE 1019 PARTIAL): a naive `FrameSlotCQ` has the EMERGE-59 order tail (renders the slots scrambled) + multi-bridge co-execution interferes. The fix (identified from EMERGE-66): use the EMERGE-65 `SelfOrganizedProducer`, whose `MinedInventoryFrameSlotCQ` wraps the EMERGE-61 wash-out → render-exact + position-independent by construction.

## The result — 3-seed (42/43/44)
```
reason(bird, sleep) -> SPIKING-BROCA -> "the bird can sleep"   [exact, order+words ON SPIKES]
reason(frog, run)   -> SPIKING-BROCA -> "the frog can run"     [exact]
reason(dog,  eat)   -> SPIKING-BROCA -> "the dog can eat"      [exact]
ABSTAIN             -> producer NOT invoked (0 productions)    [gate-first moat]
VERDICT: GO (3/3 exact, all seeds)
```
- The **SLOT ORDER** ("the" → subject → "can" → verb) is produced by the self-organized spiking-Broca producer (rate-coded competitive queuing on a slot bridge, with the EMERGE-61 inter-utterance wash-out for position-independence) — NOT a host f-string.
- Every **WORD** (the/can/subject/verb) is spelled by the breadth concept-pool A→W read-out (`language_output` firing).
- The slot bridge + the A→W bridge co-execute in ONE numpy process (the SelfOrganizedProducer's wash-out + exact-order CQ handle the co-execution that the naive base CQ could not).
- The **gate-first moat** holds: on ABSTAIN the producer is NEVER invoked (0 productions — the load-bearing property).

## What this establishes
The real-corpus talkable brain's property answer is now GENERATED FULLY ON SPIKES — sentence structure (slot order via the spiking-Broca producer) AND words (via the A→W) — replacing the host template, transformer-free, moat intact. "If Broca drives articulation, we simulate Broca" — realized for the real-corpus property answer. The fluency fork's fully-spiking-generation is achieved (CYCLE-1019 PARTIAL → GO). Follow-on: the relational (SVO transitive) answer via the EMERGE-72/74 extended producer; wire into the console (the `_word_frame` host template → the spiking-Broca producer); the ditransitive/negated frames.

## Files
`research/runners/_realcorpus_spiking_broca_property_answer_derisk.py`. Reuses: EMERGE-65 `SelfOrganizedProducer` (self-organized grammar, renders exact 6-seed), EMERGE-61 wash-out, the breadth A→W (`_realcorpus_train_breadth_aw`). Prior: the scoped PARTIAL (CYCLE 1019, AUTONOMOUS_STATE).
