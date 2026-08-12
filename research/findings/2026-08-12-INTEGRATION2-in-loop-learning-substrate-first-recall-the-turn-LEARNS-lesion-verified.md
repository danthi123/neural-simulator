---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — in-loop learning (the default /api/brain-chat turn LEARNS a fact heard this conversation)
lane: integration-first (WIRING BACKLOG #2)
integration_faculty: in-loop-learning
verdict: FIRST PRODUCTION-INTEGRATION WIN. The default /api/brain-chat turn now LEARNS: a fact heard this conversation is recalled from the SPIKING SUBSTRATE (inner.what_does), lesion-verified. The production lesion probe flipped LEARN False->True, and lesioning the substrate recall flips it back off (the taught fact disappears -> load-bearing, not a host-list effect). The change is a substrate-first recall in ChatBrain.gate: for a well-formed "what does AGENT ACTION?" question with known AGENT+ACTION, recall the patient from inner.what_does (role-aware, sees a fact heard this turn) BEFORE the host QuestionRouter (role-blind keyword overlap, snapshot-bound) fallback. No regression (pre-baked facts + multi-turn anaphora unchanged); no new confabulation (what_does returns nothing unless the binding is genuinely stored). Scope: KNOWN-word facts; a genuinely new word still needs on-the-fly code allocation (next step). wired=YES on_by_default=YES; scaffold NOT retired (host router still the fallback until #1 lands).
artifacts:
  - research/findings/raw/_production_lesion/probe.json
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
verification: production lesion probe (research/runners/_production_lesion_probe.py) — builds the IDENTICAL default /api/brain-chat brain; LEARN in_loop=True, recall_lesion_load_bearing=True.
---

# INTEGRATION #2 — the production turn LEARNS: substrate-first recall makes a fact heard this conversation answerable, lesion-verified

## Why (the integration-first goal)

Owner goal (2026-08-11): a working all-spiking one-substrate brain with ALL faculties ON BY DEFAULT in production. The
production baseline (`2026-08-11-PRODUCTION-chat-pipeline-is-largely-HOST-...`, lesion probe) measured CHOOSE/GENERATE/
LEARN all False — the default turn was read-only: `gate()` matched only a build-time `stored_facts` snapshot via the host
`QuestionRouter` (role-blind keyword overlap), so a fact heard mid-conversation was invisible even though the spiking
substrate held it. This is the first WIRING-BACKLOG item landed: LEARN.

## The change (additive, on the default path)

`ChatBrain.gate` is now **substrate-first**: it resolves (AGENT, ACTION) from the question and recalls the patient from
**`inner.what_does(agent, action)` — the spiking substrate — BEFORE** falling back to the host `QuestionRouter`.
`what_does` is ROLE-AWARE (it queries the specific (agent, action) binding, not keyword overlap) AND reflects a fact
heard this turn (`inner.hear(...)` stores to the same substrate). It returns the stored patient only if the binding is
genuinely present, so it cannot confabulate — the no-confab moat holds. The host router remains the fallback for
self/identity questions and anything not in this form. NO `sim/` edit; runner-side, default-on.

## Result — production lesion probe (`research/findings/raw/_production_lesion/probe.json`)

<!--derived-->
- **LEARN False -> True.** Teach "cat chase bird" (known words; "cat chase ?" is NOT pre-baked — the cat fact is "cat eat
  fish"). Before: "what does cat chase?" -> the host fallback "dog chase cat"; after teaching -> **"cat chase bird"**
  (recalled from the substrate). `in_loop=True`.
- **Lesion-verified load-bearing.** With `_substrate_recall` lesioned, the taught fact DISAPPEARS (answer reverts to "dog
  chase cat"). `recall_lesion_load_bearing=True` — LEARN is due to the substrate recall path, not a host-list effect.
- **No regression.** Pre-baked facts ("what does dog chase?" -> "dog chase cat"; "cat eat" -> "fish"; "brain use" ->
  "spikes") and multi-turn anaphora ("what does it eat?" -> "cat eat fish" after a dog-turn) unchanged.
- **No new confabulation.** substrate-first only returns genuinely-stored bindings. (A pre-existing host-router confab on
  malformed questions e.g. "what does fish fly?" is NOT introduced here — it is the CHOOSE gap, WIRING-BACKLOG #1.)

## Honest scope (what this is NOT yet)

- **KNOWN words only.** A genuinely new word ("otter") needs on-the-fly code allocation in the composer (a
  vocabulary-growth mechanism) — the next in-loop-learning step. This win covers facts composed of words the brain's
  vocabulary already carries.
- **Scaffold NOT retired.** The host `QuestionRouter` is still the fallback recall (role-blind) — it is retired only when
  #1 (a spiking content-selector) lands. So in-loop-learning is now `wired=YES, on_by_default=YES` but not `integrated`.
- **"Writes synapses" is the deeper target.** Recall here is from the resonate-and-fire composer substrate; a
  BTSP/plasticity write per turn (a lasting episodic trace) is the fuller LEARN. The composer-internal spiking-vs-numpy
  lesion (level-3 confirmation) is a follow-up (the probe's coarse composer-lesion currently reads composer_type=None on
  the multiturn agent — a probe refinement).

## Ledger + gate

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` in-loop-learning row -> wired/on_by_default YES; headline owner_visible_function
learn -> YES. This is the first faculty moved from de-risk to production under the CLASS PI discipline; the lesion probe
is the truth-check, the PI gate the consistency-check.
