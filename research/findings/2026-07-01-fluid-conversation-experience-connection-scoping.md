# Fluid conversation — the EXPERIENCE-connection: scoping (converse about LIVED percepts, not just taught facts)

**2026-07-01 (autonomous night; the standing practice — scope a new-direction integration before building).** The
owner's priority: responses "grounded in the brain's OWN knowledge **AND EXPERIENCES**." The Phase 0–7 fluid stack
converses about TAUGHT facts (`hear`/growth). This scopes the connection to the brain's LIVED **experiences** — the
objects it PERCEIVED — so the console can be asked *"what did you see?" / "tell me about the apple you found"*.

## What's reusable (validated pieces — this is largely a WIRING integration, not a new mechanism)
- **Perception → code grounding** (Tier-3 `nav_conv_merged_bridge.MergedNavConvAgent.perceive_and_ground`, 2026-06-30
  live-and-remember GO; step-3 `_step3_live_cortex_grounded_compose_probe` / `navigate_to_compose_then_answer`,
  6-seed GO): drive an object's percept into `cortex_it`, read the spiking firing-RATE, project it via a fixed
  cortico-cortical projection `M` → a unit **phasor** code, set `composer.concepts[obj] = angle(M @ rate)`. The
  perceived object's code then enters the SAME validated bind/store/recall algebra as any concept.
- **The fluid console** (Phase 0–7): `FluidChat` over `MultiTurnAgent`/`BrainConversationalAgent` + the RA-fine-tuned
  21M — grounded Q&A (what/who/yes-no/describe/elaborate) + multi-turn + growth + moat, over the composer.
- **Growth** (Phase 5): a new fact whose subject/object codes are non-default is answerable + the generator
  GENERALIZES to novel entities from the provided fact. The composer accepts `grounded_codes` (validated drop-in,
  `_step3_grounded_codes_production_composer_derisk`).

## The genuine integration point + the cheap-first de-risk
The NEW thing: the object's code comes from **perception** (the grounding projection over a percept), not from the
composer's default code-generation, and the fact is "experiential" (the brain SAW it). The de-risk
(`_fluidconv_phase8_experience_derisk.py`): ground a PERCEIVED object's code (the fixed `_projection` over a percept
rate-vector — a lightweight stand-in for the live `cortex_it` forward, which the heavier merged-bridge loop supplies),
inject it into the fluid console's composer, store a "saw" fact about it, then the console ANSWERS about the perceived
object (RA-render) + the moat holds on UN-perceived objects.
- **Anti-cheats:** (a) GROUNDING-LESION — corrupt the perceived code AFTER storing → recall collapses (the recall is
  load-bearing on the percept, not a taught label); (b) MOAT — an un-perceived object → abstain (0-FA); (c) the
  perceived-object answer is RA-rendered grounded (VERIFY-clean).
- **GO bar:** the console converses about a perceived (not taught) object, grounding load-bearing (lesion collapses),
  moat 0-FA, ≥3 seeds.

## Honest scope
- The cheap-first de-risk uses a **lightweight fixed-projection grounding** (a per-object percept vector → the
  validated `_projection` → a phasor code) as the perception stand-in — this isolates the CONVERSATIONAL handling of
  a perception-grounded code. The FULL live loop (a real `cortex_it` spiking forward on the merged nav+conv bridge,
  perceived DURING behaviour, then conversed about via the RA console) is the heavier follow-on integration — it
  composes the Tier-3 live-and-remember loop with the RA console, both validated, on one brain.
- This CLOSES the "experiences" clause of the fluid-conversation priority at the conversational layer; the full
  embodied loop (perceive-while-acting → converse) is the capstone follow-on.

**Next:** run `_fluidconv_phase8_experience_derisk.py` (the cheap-first, this cycle), then the full live-loop
integration (a follow-on). Reuse-by-import; NO `sim/` edit.

## RESULT — Phase-8 cheap-first GO (3 seeds)
`_fluidconv_phase8_experience_derisk.py`: perceived subjects (wolf/owl/frog) get codes grounded from percepts (the
fixed projection `angle(proj @ percept)` — the composer's phase-code format; a bug where I first passed complex
phasors → cast to float dropping the imaginary part, fixed to pass phases), stored as lived facts, conversed via the
RA console.
- **CONVERSE 3/3** all seeds — *"the wolf eats rabbit.", "the owl eats mouse.", "the frog eats worm, yes."* (the
  perceived objects answered grounded + RA-rendered).
- **GROUNDING-LESION collapses 3/3** — corrupting the percept (→ a different grounded code) collapses the recall
  (the answer is load-bearing on the PERCEPT, not a taught label).
- **MOAT 0-FA 3/3** — an un-perceived object abstains.
⇒ the fluid console converses about the brain's PERCEIVED experiences (not just taught facts), grounding load-bearing,
moat intact — the **"experiences" clause closed at the conversational layer**. The FULL embodied loop (a live
`cortex_it` spiking forward on the merged nav+conv bridge, perceived DURING behaviour, conversed about via the RA
console) is the follow-on integration (composes Tier-3 live-and-remember + the RA console on one brain).
Artifacts: `_fluidconv_phase8_experience_derisk.py`; `research/findings/raw/_fluidconv_phase8_experience.json`.
