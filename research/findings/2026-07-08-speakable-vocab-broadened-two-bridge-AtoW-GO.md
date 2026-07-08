# Speakable vocab BROADENED to 31 words via a 2nd A->W bridge (GO): the talkable brain now SPEAKS about objects + people ("the girl likes ball", "the boy sees car"), not just animals -- the EMERGE-68 two-bridge dispatch, spell 31/31, moat intact. NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `research/runners/_realcorpus_train_breadth_aw2.py` (trains BRIDGE-2) + `_realcorpus_two_bridge_speaker.py` (`TwoBridgeFrameSpeaker`, dispatch) + the unified console `--two-bridge` flag. `SIM_BACKEND=cupy` to train BRIDGE-2; numpy to speak. NO `sim/` edit.
**Verdict:** GO — 31-word spoken vocab, spell 31/31, the broadened conversation works in the console.

## Why this ran (the biggest remaining lever)
The talkable brain REASONS over 256 discovered words but only SPEAKS 16 — a single concept-pool A->W bridge caps at 16 words (4 pool-kinds x 4). The proven path to more spoken words is the EMERGE-68 two-bridge dispatch (one bridge per 16 words, routed by word). This broadens the SPOKEN conversation from 8 animals to objects + characters.

## What was built
- **BRIDGE-2 (`_realcorpus_train_breadth_aw2`):** trained over 15 object/character nouns VERIFIED present in the TinyStories top-256 — ball, tree, box, sun, cake, toy, car, house, door, rock, boat, girl, boy, mom, dad (objects + people, disjoint from BRIDGE-1's 8 animals + 6 verbs + the/can). Same recipe as BRIDGE-1 (decoupled topographic bias + per-word train, ~18 min on GPU).
- **`TwoBridgeFrameSpeaker`:** dispatches `spell`/`speak_frame` across BRIDGE-1 + BRIDGE-2 by word membership, exposing the SAME interface as `ConceptFrameSpeaker` so the console uses it transparently.
- **Console `--two-bridge` flag:** additive, default-off (the default path is byte-unchanged).

## The result
**Spell 31/31** — the two-bridge speaker spells every word of the combined 31-word vocab correctly (BRIDGE-2's 15 nouns all spell; BRIDGE-1's 16 unaffected — separate bridges, no cross-interference).

**The broadened conversation (console `--two-bridge`, seed 42):**
```
"the girl likes ball"  (teach) -> "what does the girl like?" -> "the girl likes ball"  [girl/ball ON SPIKES via BRIDGE-2]
"the boy sees car"     (teach) -> "what does the boy see?"   -> "the boy sees car"     [boy/car ON SPIKES]
"what does the mom eat?"                                      -> "I don't know"         [moat]
```
The brain now teaches/answers relations over OBJECTS and PEOPLE (girl/boy/ball/car/...), spoken on spikes, moat intact — not just animals.

## Honest scope
- The spoken content vocab is 31 words (8 animals + 6 verbs + 15 objects/people + the/can); the relational reasoning already works over the full 256-word discovered vocab (non-spellable words render as text). More bridges → more spoken words (linear, EMERGE-68).
- The dispatch is by word-membership (each word lives on exactly one bridge); BRIDGE-1 wins on overlap (none by construction).
- Rate-level relational reasoning + spiking A->W content (the validated split); the/-s host scaffolds.

## What this establishes
The talkable brain's SPOKEN vocabulary broadened from 16 to 31 words via the EMERGE-68 two-bridge A->W dispatch, so it converses about objects and people, not just animals — spell 31/31, moat intact, the default path byte-unchanged. The biggest speech-breadth lever, delivered; the path scales linearly with more bridges.

## Files
`research/runners/_realcorpus_train_breadth_aw2.py`, `_realcorpus_two_bridge_speaker.py`, the console `--two-bridge` flag; `bridges/breadth_aw2/seed42.simstate.h5` (regenerable). Prior: the breadth A->W `_realcorpus_train_breadth_aw.py`; EMERGE-68 (the two-bridge function-word A->W); the unified console `2026-07-08-unified-talkable-console-property-and-relational-one-brain-GO.md`.
