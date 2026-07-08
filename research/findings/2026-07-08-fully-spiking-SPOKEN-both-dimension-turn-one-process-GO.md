# Fully-spiking SPOKEN both-dimension turn (GO, seed 42): the property HTM reasoner + the relational RF-FHRR composer + the A→W speaker ALL co-execute on cupy in ONE process — BOTH dimensions REASONED ON SPIKES and SPOKEN ON SPIKES, over spellable animals. The directive-central capstone of the fully-spiking arc. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_one_brain_spoken_smoke.py` (reuse-by-import: `CancellingPoolerProbe` [property HTM] + `RFPhasorComposer` [relational RF-FHRR] + `ConceptFrameSpeaker` [A→W], all on cupy). `SIM_BACKEND=cupy`. NO `sim/` edit.
**Verdict:** GO (seed 42) — both dimensions reasoned on spikes AND spoken on spikes, one process.

## Why this ran (the capstone I kept deferring — the skill says don't defer the hard thing)
CYCLE-1003 co-executed both spiking REASONERS in one process but with arbitrary emergent words (not spoken). The console speaks but reasons at rate-level. This composes ALL THREE spiking pieces — property HTM + relational RF-FHRR + A→W speech — on cupy in one process, over SPELLABLE ANIMALS, so the whole both-dimension turn is fully spiking AND spoken.

## The result — seed 42 (K=1024 emergent property clusters, D=64 relational, cupy)
```
=== ONE BRAIN, BOTH DIMENSIONS, FULLY SPIKING + SPOKEN (one cupy process) ===
property cluster 5: exception 'bear' + inheriting 'frog'; relational 'bird like dog'
  Q: does the bear run?      -> reason(EXC, HTM spikes)     -> "no, the bear can sleep"   [SPOKEN on A->W spikes]
  Q: does the frog run?      -> reason(inherit, HTM spikes) -> "yes, the frog can run"    [SPOKEN]
  Q: what does the bird like? -> reason(RF-FHRR spikes)     -> "the bird likes dog"       [SPOKEN]
VERDICT: GO
```
- **PROPERTY (HTM):** a spellable-animal exception (`bear`) overrides its class ON SPIKES (apical competition on the committed HTM coincidence kernel, `cp_v_apical`), and an inheriting spellable animal (`frog`) inherits — both spoken via the A→W read-out (`language_output` firing).
- **RELATIONAL (RF-FHRR):** the SVO object recovered ON SPIKES (RF resonate-and-fire + complex-synapse store) — spoken.
- **ALL THREE spiking bridges co-execute in ONE cupy process** (one-backend-per-process satisfied); every content word is decoded from `language_output` spikes.

## Honest scope (the alignment boundary)
- A 1-seed demonstration. The property HTM needs K=1024 emergent clusters (where the spiking read is strong); the spellable animals must co-cluster there AND a weakly-inherited spellable exception must exist (the CYCLE-987 alignment tension). Seed 42's cluster 5 aligned (bear/frog); seed 43 was NOT-EVALUABLE (confirmed -- no cluster with a weakly-inherited spellable-animal exception). So the alignment is ~1/2 seeds; other seeds may be NOT-EVALUABLE (an honest, characterized boundary — the runner searches clusters + picks a low-pass-budget exception to avoid the CYCLE-985 saturation, and reports NOT-EVALUABLE if none aligns).
- The words are whatever aligns ("the bird likes dog" is semantically odd) — the MECHANISM (both dimensions, one process, fully spiking + spoken) is what's demonstrated. The individual mechanisms are each multi-seed-validated (cancellation 6/6, relational SVO 3-seed substrate, A→W 31/31).

## What this establishes
The TRUE one brain, END TO END: the whole both-dimension conversational turn — property (inheritance + cancellation) AND relational (SVO) — REASONED FULLY ON SPIKES and SPOKEN FULLY ON SPIKES, co-executing in ONE cupy process, over the brain's own real-corpus codes, with the no-confab moat (by construction). The directive-central "if Broca drives articulation, we simulate Broca" milestone for the two-dimension talkable brain. Follow-on: robust multi-seed alignment (a spellable-animal-seeded cluster); fold into the interactive console.

## Files
`research/runners/_realcorpus_one_brain_spoken_smoke.py`; `research/findings/raw/_one_brain_spoken2.log`. Composes: the fully-spiking cancellation, the spiking relational (RFPhasorComposer), the A→W frame speech, and the CYCLE-1003 both-dimension reasoning.
