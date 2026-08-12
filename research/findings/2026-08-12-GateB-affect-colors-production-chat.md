# Gate-B: AFFECT/EMOTION wired into the production /api/brain-chat turn — mood colors WHAT + HOW, default-on, moat-safe, lesion-load-bearing (WIRED)

**Date:** 2026-08-12
**Status:** GO / WIRED (production-integration). Single-process synchronous HTTP verify on the real `/api/brain-chat`
endpoint (composer=onebrain, renderer=qwen, SIM_BACKEND=cupy). All four verify checks pass.

## What changed

The brain's live **spiking mood** now genuinely colors the production conversational turn — both **WHAT** it chooses
to say (mood-congruent **forthcomingness**: how many gate-matched, moat-verified facts it volunteers + elaboration
depth) and **HOW** it says it (the fluent Qwen mouth phrases the **same fact** warmer/curter). Default-ON; the
moat/recall/abstain paths are untouched (affect colors only an already-matched answer). This is NOT a narration/tone
tag — the tone token is a debug-trace field only; the coloring is in the CONTENT (how much) and the PROSE (phrasing).

Additive, no `sim/` edit. Two pieces:

- **`research/runners/affect_production_organ.py`** (new) — the production-integration glue. It REUSES the
  adversarially-verified Stage-A affect faculty:
  - the persistent spiking mood organ = the co-resident brain from
    `_stageA_full_integration_derisk.build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=True)` — ONE
    SimulationBridge carrying the staggered-bistable-ladder graded-affect slice (Koulakov robust integrator) + the
    honesty relay + the 3-way arbiter, co-resident (6-seed GO for the valence SIGN + a graded bistable LADDER,
    2026-08-08). The held mood is read NEURALLY as the population-rate differential
    `rate(aff_pos_readout) − rate(aff_neg_readout)` through the `affect_out` transmission gate. The read is SIGN-AWARE
    (a positive appraisal ramps the V+ rungs, a negative one the V− rungs), mirroring `read_affect_ladder` but signed.
  - the appraisal CONTENT source = the DR-2 learned per-word valence lexicon (WARRINER-approximate VAD), a declared
    host scaffold; the injection is host, the READ-BACK through `affect_out` is the load-bearing spiking part.
  - a `MoodConditionedRenderer` wrapping the Qwen renderer: it injects a mood-MANNER clause (reinforcing the exact
    content words + verb) into the constrain prompt so the PROSE is warmer/curter, while the VERIFY re-parse still
    recovers the gated SVO (a manner render that drifts is DROPPED by the moat — never a leak).
- **`webapp/server.py`** `brain_chat` — per turn: appraise the message (DR-2 lexicon → session mood EMA, persists
  across turns, cleared on reset) → read the NEURAL ladder differential → derive the graded LEVEL → set the rich
  composer's forthcomingness (`max_sentences`/`max_elaborations`) for the turn (restored after) and the renderer's
  manner. Neutral mood = the production default (byte-identical). `BRAIN_AFFECT=0` fully disables (byte-identical
  oracle); `BRAIN_AFFECT_LESION=1` clamps `affect_out=0` for the lesion test. Also the honest inner-state read-out:
  "how do you feel" → the live valence differential (Wire-2), a functional read, never a phenomenal claim.

## Verify (SYNCHRONOUS, in-process FastAPI HTTP client, composer=onebrain, renderer=qwen)

**(a) positive vs negative mood → DIFFERENT content AND manner, SAME fact.** Query "what does the dog chase?"
(recalled `[dog, chase, cat]` both):
- POSITIVE mood (after "I am so happy, full of joy and love today!"): neural differential **+0.039**, level +2 →
  **2 sentences, warm**: *"The dog chased the cat. Cat is eating fish!"*
- NEGATIVE mood (after "I feel so sad, angry and afraid right now."): differential **−0.036**, level −2 →
  **1 sentence, terse**: *"The dog chased the cat."*
- same recalled fact, content differs (2 vs 1 sentences), manner differs. **PASS.**

**(b) LESION affect_out → coloring collapses, matched fact byte-identical.** Negative-induced turn:
- affect ON: differential −0.036 → 1 terse sentence *"The dog chased the cat."*
- affect LESIONED (`affect_out=0`): differential **0.000**, level 0 → reverts to the neutral fluent default
  (2 sentences, bare manner) *"The dog chased the cat. The cat eats fish."*; the matched fact `[dog, chase, cat]` is
  byte-identical; abstain behaviour unchanged. The READ-BACK through `affect_out` is load-bearing. **PASS.**

**(c) MOAT — 0 leaks under strong mood.** Under strong positive AND strong negative induced mood, 6 untaught cues
("what is the capital of france", "what does the dragon fly", "what does the unicorn eat") ALL abstain, and the
matched fact truth ("what does the dog chase" → `[dog, chase, cat]`) is unchanged by mood. **leaks=0. PASS.** Affect
never enters the certainty band, never manufactures an answer, never flips an abstain.

**(d) NO REGRESSION + disable-escape byte-identity.** On the affect-ON path: recall "what does the dog chase" →
*"The dog chased the cat. The cat eats fish."*; abstain "what does the dragon do" → abstained; LEARN a NEW fact
"fox eat rabbit" → recall "what does the fox eat" → *"The fox eats the rabbit."*; anaphora "what does it eat"
(referent from "what does the dog chase") → *"The cat eats fish."*; single-fact path (`rich=False`) →
*"The dog chased the cat."* (affect on). Disable-escape: with `BRAIN_AFFECT=0` the `affect` field is null and the
neutral-query answer is **byte-identical** to affect-ON-at-neutral-mood (both *"The cat eats fish. The dog chased
the cat."*). **PASS.** (Full run: all four checks PASS, ~1480s single process, GPU.)

## Honest residuals (declared; each rides an existing burn-down row)

- The MANNER-coloring conditions the EXTERNAL Qwen mouth — host-mediated until the mouth is brain-native
  (**burn-down A1**).
- The appraisal INJECTION (host DR-2 valence lexicon → neuromodulator concentration) is a declared host scaffold.
- The affect organ is a co-resident affect/honesty/arbiter substrate on its OWN bridge, run ALONGSIDE the production
  recall composer (the onebrain composer), not merged onto the single recall bridge — the remaining one-brain
  consolidation step (**burn-down #1**).
- The held value is a GRADED bistable LADDER (quantized sign + level), NOT a smooth-magnitude continuum — the continuum
  is still a **BOUNDARY**.

## Repro

```
BRAIN_COMPOSER_KIND=onebrain SIM_BACKEND=cupy  # POST /api/brain-chat, rich (default)
#  induce POSITIVE: "I am so happy, full of joy and love today!"  then  "what does the dog chase?"
#  induce NEGATIVE: "I feel so sad, angry and afraid right now."   then  "what does the dog chase?"
#  lesion:  BRAIN_AFFECT_LESION=1   ;   disable: BRAIN_AFFECT=0 (byte-identical oracle)
```
