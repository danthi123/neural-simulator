# Grounded language faculty — P2 (Claude-teacher → brain-structured knowledge) = GO 3/3: the taught knowledge IS brain-structured (recall 1.000, moat 0-FA, multi-hop 6/6) (2026-06-22)

**Scope:** the #1 cheap-first de-risk of the grounded-language-faculty architecture — does a CLAUDE-authored OFFLINE
curriculum, learned by the brain ITS OWN WAY (Hebbian `BridgeParser` → `RFPhasorComposer` + the no-confab moat),
become genuinely BRAIN-STRUCTURED knowledge? This directly answers the owner's question — *"would the transferred
information be structurally significant and properly compartmentalized as true knowledge?"*
`research/runners/_p2_teacher_to_brain_derisk.py`, numpy/CPU. **NO `sim/` edit, NO GPU, NO downloads, NO cloud.** On
`main`.

## Result — GO (3/3 seeds, identical 42/43/44)
A Claude subagent (the TEACHER) authored 40 flat-SVO facts across 4 mini-topics (animals-eat-food, animals-chase-
animals, agents-go-places, agents-make-objects; a self-consistent 65-word vocab) + 6 two-hop chains + 10 held-out
never-taught in-vocabulary cues. The brain HEARD them through the existing conversational machinery, then:

| metric | result |
|---|---|
| structured RECALL (who_does + what_does over the 40 taught facts) | **1.000** (80/80) |
| no-confab MOAT (HARD — 10 held-out never-taught cues) | **0/10 false-accepts** (all abstained → `None`) |
| MULTI-HOP reasoning (6 chains) | **6/6 correct** + 2/2 broken chains correctly abstained |

Examples (seed 42, representative): taught `what_does('dog','eat') → 'meat'` ✓; held-out `what_does('dog','swim') →
None` ✓; chained `reason_chain('dog', ['chase','chase']) → 'mouse'` (dog→cat, cat→mouse) ✓.

The composer's DEFAULT abstention was SUFFICIENT — the moat did NOT break at this scale, so P3's explicit gate is
NOT required just to prevent confabulation on held-out cues (it remains the design for the harder *fluent-faculty*
grounding, where the generator could fabricate).

## What this settles
⇒ **The owner's original concern is answered YES for the teacher→brain path.** Claude-taught knowledge becomes
genuinely BRAIN-STRUCTURED — recallable by structured query, **abstaining on the unknown** (the no-confab moat),
and **chainable by multi-hop reasoning** — NOT transformer-form. It is "properly compartmentalized as true
knowledge." Claude is a legitimate OFFLINE teacher (a textbook author); the brain ends standalone (≠ the deprecated
runtime-external-LLM Path 3). The whole **P2 knowledge half + the P3 abstention gate are de-risked in one cheap,
multi-seed, no-GPU shot**, on the project's EXISTING validated machinery (no new mechanism, no `sim/` edit).

## Next
- **Faculty (P1):** the Gen-F spiking-convert de-risk (in flight) + convert a small fluent model (Qwen2.5-0.5B,
  **verifying the 2026 LLaMA-stack spiking-operator code-release** flagged in the scoping).
- **Scale the teaching:** graded-difficulty curriculum + the topical co-occurrence STREAM (the PPMI stream-cortex
  generalization channel), larger vocab; the cloud-trigger is a large teaching corpus.
- **P3 grounding smoke** for the fluent faculty (gate→constrain→verify), where the generator — unlike the composer —
  *can* fabricate, so the explicit gate earns its keep.
