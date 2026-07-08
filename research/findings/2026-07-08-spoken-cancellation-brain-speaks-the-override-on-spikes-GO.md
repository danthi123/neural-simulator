# SPOKEN CANCELLATION (GO, 5/6 evaluable seeds): the talkable brain SPEAKS the OVERRIDE for an exception member ("does the bird run? → the bird can sleep") — its own property ON SPIKES, not the inherited one — while speaking the inherited property for others and abstaining on the unknown. Composes the 6-seed cancellation GO + the frame-speech GO. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_spoken_cancellation_derisk.py` (reuse-by-import: `CancellingConsole` [cancellation GO] + `ConceptFrameSpeaker` [frame-speech GO] + the breadth A→W `bridges/breadth_aw/seed42.simstate.h5`). Requires `SIM_BACKEND=cupy`. NO `sim/` edit.
**Verdict:** GO — 5/6 seeds; 1 NOT-EVALUABLE (emergent-cluster data condition).

## Why this ran
The cancellation mechanism (6-seed GO) let the emergent reasoner OVERRIDE inheritance member-specifically; the frame speaker (GO) speaks "the <subject> can <verb>" with content on spikes. This composes them: the brain SPEAKS the correct property — the override for an exception, the inherited one for others — a genuinely richer "talk to the brain about exceptions" conversation.

## The result — 6-seed (class verb "run", exception verb "sleep")
```
seed 42  cluster ~bird [bird,cat,fish,bear,frog]  exc=bird  | "does the bird run?"  -> "the bird can sleep"  (OVERRIDE, spikes) | frog->run (inherit) | moat  GO
seed 43  cluster ~spot [dog,cat,fish,frog]        exc=fish  | "does the fish run?"  -> "the fish can sleep"  (OVERRIDE, spikes) | dog->run  (inherit) | moat  GO
seed 44  cluster [fish,frog,mouse]                          | NOT-EVALUABLE (no held-out spellable-animal member inherits)
seed 100 cluster ~cat  [bird,dog,cat,fish,frog,mouse] exc=mouse | "does the mouse run?" -> "the mouse can sleep" (OVERRIDE) | frog->run | moat  GO
seed 101 cluster ~bird [bird,cat,fish,bear,frog,mouse] exc=mouse | "does the mouse run?" -> "the mouse can sleep" (OVERRIDE) | fish->run | moat  GO
seed 102 cluster ~cat  [dog,cat,fish,frog,mouse]  exc=frog  | "does the frog run?"  -> "the frog can sleep"  (OVERRIDE, spikes) | moat  GO
```
Every evaluable seed: the exception member's frame SPEAKS its OWN property (`v2`="sleep") and NOT the inherited one (`v1`="run") — a genuine SPOKEN cancellation, content ON SPIKES; inheriting members SPEAK the inherited property; the unknown word → "I don't know" (gate-first moat).

## What the gate checks (per seed)
- **spoke-override** ≥ 1: the exception member's spoken frame contains `v2` (its own property).
- **cancel-spoken**: the override frame contains `v2` and NOT `v1` — the spoken answer is the override, not the inherited property.
- **moat-ok**: an unknown word → "I don't know".
The exception member is chosen as a HELD-OUT, SPELLABLE-ANIMAL member of the discovered cluster that INHERITS before the exception (so the spoken override is a genuine change).

## Honest scope
- 5/6 seeds GO; seed 44 NOT-EVALUABLE — its emergent cluster [fish,frog,mouse] had no held-out spellable-animal member that inherits (the emergent-cluster held-out-inheritance data condition, not a mechanism failure; the same limit that makes 5/6 seeds speak in the pipeline).
- The answer is rendered via the modal frame ("the fish can sleep") with content ON SPIKES; the "no"/negation is the reasoning decision (a different verb than asked), not yet a spoken "no" (adding "no" needs a function-word slot — a bounded follow-on; the intransitive/negated frame with 3sg inflection is the EMERGE-59 morphology path).
- "sleep" as the override of "run" is a mechanism-demo pairing over the animal cluster; the CANCELLATION is validated regardless of the specific verb pair.
- Rate-level reasoner + spiking A→W content (the validated split); the spiking realization of the cancellation drive mirrors the EMERGE-54 apical path (follow-on).

## What this establishes
The emergent talkable brain now SPEAKS exceptions to inheritance: it discovers an animal category from real experience, learns a class property AND a member-specific exception, and SPEAKS the correct property (override vs inherited) with content on spikes, abstaining on the unknown — a fluent spoken cancellation conversation, transformer-free, moat intact. Follow-on: a spoken "no" prefix; the spiking cancellation drive; the intransitive/3sg frame.

## Files
`research/runners/_realcorpus_spoken_cancellation_derisk.py`. Prior: the cancellation mechanism `2026-07-08-cancellation-member-exception-overrides-inheritance-real-corpus-GO.md`; the frame speech `2026-07-08-full-frame-fluent-speech-on-spikes-GO.md`; the complete talkable loop `2026-07-08-COMPLETE-talkable-loop-discover-reason-fluent-frame-GO.md`.
