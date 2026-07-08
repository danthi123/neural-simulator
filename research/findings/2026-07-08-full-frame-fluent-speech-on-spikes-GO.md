# FULL-FRAME fluent speech (GO, GPU): the brain speaks a grounded grammatical FRAME ("the dog can go") with the CONTENT words produced ON SPIKES (concept-pool A→W from `language_output`), not just yes/no — and abstains on an unknown word (gate-first moat). Upgrades the SPEAK rung to full frames. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_full_frame_speech_derisk.py` (reuse-by-import: `concept_pool_demo.build_concept_bridge` + the cached v16 A→W bridge `bridges/v16/seed42.simstate.h5` + `concept_speak_demo`'s pool-drive/decode). Requires `SIM_BACKEND=cupy`. NO `sim/` edit.
**Verdict:** GO — full-frame fluent grounded speech on spikes, moat intact.

## Why this ran (fluent speech, driven per owner "no deferrals")
The SPEAK rung spoke a yes/no proxy word. This speaks a full grammatical frame: the reasoner's (subject, verb) → "the <subject> can <verb>", with the CONTENT words produced ON SPIKES via the validated concept-pool A→W read-out (drive the word's pool → decode the spoken word from `cp_firing_states[language_output]`), reusing the cached v16 bridge (no retrain).

## The result — GPU
```
(dog,   go)   -> "the dog can go"      [content ON SPIKES, exact]
(cat,   come) -> "the cat can come"    [exact]
(apple, stop) -> "the apple can stop"  [exact]
(river, look) -> "the river can look"  [exact]
(zzzqqx, go)  -> "I don't know"        [MOAT: unknown word, no frame]
```
**Content-spell accuracy 4/4 exact** (each content word decoded correctly from `language_output` firing), and the gate-first **moat** holds (an unknown content word → "I don't know", no frame). The brain speaks a fluent grounded frame, not just a yes/no token.

## Honest scope
- **Content words (subject, verb) are ON SPIKES** (concept-pool A→W, 4/4 exact) — the claim.
- **Closed-class frame words (the, can) are host-rendered** — a documented scaffold; EMERGE-68's spiking function-word A→W (BRIDGE-F) is the follow-on that puts them on spikes too.
- **Vocab = the cached v16 concept-pool A→W** (16 words: nouns dog/cat/apple/river + verbs go/come/stop/look + adjectives + motor); the overlap with the breadth reasoner's TinyStories-discovered vocab is dog/cat/apple/river. Broad-vocab full-frame speech (the reasoner's full discovered vocab) = an A→W retrain on that vocab (the GPU follow-on).
- The demo speaks direct (subject, verb) frames; wiring the breadth reasoner's (held-out subject, inherited-property verb) decision into the frame is the reasoner-integration step (the decision path is the rung-4/SPEAK arc; this is the FRAME renderer).

## What this establishes
The talkable brain now speaks GROUNDED GRAMMATICAL FRAMES with content on spikes ("the dog can go") + gate-first moat — a fluent upgrade over the yes/no SPEAK rung, transformer-free, NO `sim/` edit. Combined with the arc: discover a broad vocab → reason (inherit) → and SPEAK a grounded frame (content on spikes) or abstain. Follow-ons: the spiking function-word A→W (the/can on spikes, EMERGE-68); the broad-vocab A→W retrain (speak the reasoner's full discovered vocab); wire the breadth reasoner's decision into the frame.

## Files
`research/runners/_realcorpus_full_frame_speech_derisk.py`. Prior: the yes/no SPEAK rung `2026-07-08-knowledge-half-SPEAK-grounded-answer-on-spikes-GO.md`; the cached v16 concept-pool A→W; EMERGE-59..68 (the spiking-Broca frame + function-word A→W).
