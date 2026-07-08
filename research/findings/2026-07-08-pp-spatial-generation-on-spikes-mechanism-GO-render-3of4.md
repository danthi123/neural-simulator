# PP (spatial) answer GENERATION on spikes — MECHANISM demonstrated (order + PRODUCTIVE 3sg + prepositions + moat), render 3/4: "the owl runs TO the pond" / "the cat walks TO the hill" / "on the rock/nest" produced ON SPIKES via the EMERGE-72 registry producer + the productive multi-bridge A→W (the 3sg verb composed by neural affixation, CYCLE 1024). The one miss is the known EMERGE-75b A→W read-path state-carryover (moat-safe). NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `research/runners/_realcorpus_spiking_broca_pp_answer_derisk.py` + `_realcorpus_productive_multi_speaker.py` (`ProductiveMultiSpeaker`) + `_realcorpus_train_breadth_aw4.py` (BRIDGE-4 A→W). numpy. NO `sim/` edit.
**Verdict:** PARTIAL (mechanism demonstrated) — PP spatial generation on spikes works (order + productive morphology + prepositions + moat); render-fidelity 3/4, the 1/4 miss = the characterized EMERGE-75b A→W read-path residual.

## Why this ran (broadening spiking generation to spatial relations)
Spiking generation covered property (F_MODAL) + transitive (C_TRANS). This extends it to the PP (spatial/directional) constructions: "the owl runs TO the pond" (C_PPGOAL, goal) / "the owl runs ON the rock" (C_PPLOC, location). It composes THREE prior pieces: the EMERGE-72 registry producer (the 6-slot PP construction order on spikes), a new **productive multi-bridge A→W** (BRIDGE-1/2/3/4 + the affix bridge), and the CYCLE-1024 productive inflection (the 3sg verb "runs"/"walks" = spell(stem)+spell("-s"), on spikes).

## What was built
- **BRIDGE-4 A→W** (`_realcorpus_train_breadth_aw4.py`, GPU): the ditransitive/PP vocab (give/show/bring/send + bone/gift/... + to/on + pond/rock/nest/hill), 16 words.
- **`ProductiveMultiSpeaker`**: dispatches whole-word spelling across BRIDGE-1/2/3/4 AND composes productive inflections (a 3sg form not stored as a lexeme → spell(stem)+spell(affix), reusing the CYCLE-1024 affix bridge). So "runs" = spell("run")+spell("s") on spikes.
- The PP de-risk: the registry producer (C_PPGOAL/C_PPLOC) + the productive speaker.

## The result — seed 42
```
C_PPGOAL(owl, run, pond)  -> "the holds runs to the pond"   [MISS: "owl" -> "holds" on the deep-history 1st scored render]
C_PPLOC (owl, run, rock)  -> "the owl runs on the rock"     [exact -- same "owl", correct on the 2nd render]
C_PPGOAL(cat, walk, hill) -> "the cat walks to the hill"    [exact]
C_PPLOC (dog, walk, nest) -> "the dog walks on the nest"    [exact]
ABSTAIN                   -> producer NOT invoked            [gate-first moat]
render-fidelity = 3/4
```
- **The MECHANISM works**: the C_PPGOAL/C_PPLOC slot ORDER is produced on spikes; the 3sg verb ("runs"/"walks") is composed by PRODUCTIVE inflection on spikes (spell(stem)+spell("s")); the prepositions to/on are spelled on spikes; the gate-first moat holds.
- **The single miss** is "owl" misread as "holds" (both BRIDGE-3 words) in the FIRST scored render. "owl" spells correctly STANDALONE (the 46/46 multi-bridge check) AND in the SECOND render (C_PPLOC(owl,...) → "the owl runs on the rock") — so it is NOT a decode failure but the characterized **EMERGE-75b A→W read-path substrate-state carryover** (the multi-bridge sequential spelling accumulates state; a word misreads on the deepest render history). A warm-up render did not fix it (it changed the misread word kicks→holds, confirming state-sensitivity, not a first-render transient). Moat-safe (render-polish, not a moat breach).

## Honest scope
This is a MECHANISM demonstration, not a clean 4/4 GO. The capability — PP spatial-relation answers produced on spikes (order + productive morphology + prepositions) — is demonstrated; the render-fidelity is capped by the KNOWN, characterized EMERGE-75b boundary (A→W read-path state carryover on deep render history), which is already deferred below the frontier (named next hypothesis: a SETTLED-state snapshot per-spell, not the post-build reset that EMERGE-75b found made it worse). The productive-inflection integration (CYCLE-1024 affixation wired into generation) is itself a genuine advance demonstrated here (runs/walks composed on spikes in the renders).

## Files
`research/runners/_realcorpus_spiking_broca_pp_answer_derisk.py`, `_realcorpus_productive_multi_speaker.py`, `_realcorpus_train_breadth_aw4.py`. Reuses EMERGE-72 (registry producer), CYCLE-1024 (affix bridge / productive inflection), the multi-bridge A→W. Prior: the PP store `2026-07-08-pp-spatial-relation-store-GO.md`; the C_TRANS generation `2026-07-08-fully-spiking-relational-transitive-answer-generation-GO.md`; the A→W read-path residual (EMERGE-75b, AUTONOMOUS_STATE).
