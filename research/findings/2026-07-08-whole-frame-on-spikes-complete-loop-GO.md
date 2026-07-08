# Whole-frame-on-spikes COMPLETE talkable loop (GO, GPU): the brain speaks a grounded frame with ALL FOUR words -- the/subject/can/verb -- produced ON SPIKES (from `language_output` firing), content AND closed-class. "the bird can run" is now 100% spiking render, no host tokens. Enabled by the topographic-bias decoupling. NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `_realcorpus_train_breadth_aw.py` (10-word A->W: 4 animals + 4 verbs + the/can) + `_realcorpus_full_frame_speech_derisk.py` (speak_frame spells the/can on spikes when present) + `_realcorpus_full_pipeline_reason_to_frame_derisk.py --breadth`. SIM_BACKEND=cupy.
**Verdict:** GO -- the whole grammatical frame produced on spikes (content + function words), in the complete discover->reason->speak->abstain loop.

## The result
A->W spell **10/10** (dog/cat/bird/fish + run/jump/walk/eat + the/can, all decoded exactly from `language_output`). The frame speaker now spells the closed-class the/can on spikes too:
```
speak_frame(bird, run) -> "the bird can run"   [ALL 4 words ON SPIKES, exact]
speak_frame(fish, eat) -> "the fish can eat"   [exact]
```
And the full pipeline: DISCOVER an animal cluster [dog,cat,fish] from TinyStories (probe-free) -> TEACH a property (verb 'run') -> ask held-out 'fish'/'dog' -> REASON (they inherit) -> SPEAK "the fish can run" + "the dog can run" (WHOLE frame on spikes) -> unknown -> "I don't know" [moat].

## What this establishes (the biology-purity completion)
The closed-class frame words (the/can) were the last host-rendered scaffold in the fluent-speech path (EMERGE-67's named residual, EMERGE-68's job). The topographic-bias DECOUPLING (which unblocked broad-vocab training) also made this trivial: train the/can as two more A->W words. Now the WHOLE grammatical frame -- "the bird can run" -- is produced ON SPIKES (every word decoded from `language_output`), content AND function, transformer-free. Combined with the arc: discover a broad vocab from real experience -> discover its categories (probe-free) -> reason (inherit) -> SPEAK the whole grounded frame on spikes -> abstain on the unknown (gate-first moat). The complete talkable loop, 100% spiking render.

## Honest scope
- Whole frame on spikes (10/10 A->W); the reasoner's held-out classification accuracy (characterized emergent-cluster limit) gates which held-out members speak; the moat is robust.
- The decoupling edit is additive (byte-identical default); all concept-pool runners unaffected.
- The frame inventory is the fixed "the <subj> can <verb>" construction (the EMERGE-59 frame family); the reasoner + A->W vocab is 8 content words + the/can.

## Files
`_realcorpus_train_breadth_aw.py`, `_realcorpus_full_frame_speech_derisk.py`, `_realcorpus_full_pipeline_reason_to_frame_derisk.py`, `concept_pool_demo.py` (the additive decoupling). Prior: the broad-vocab complete loop `2026-07-08-broad-vocab-complete-loop-coherent-frames-GO.md`; EMERGE-67/68 (the spiking A->W + function-word residual).
