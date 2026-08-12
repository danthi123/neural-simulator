---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — the #3E open-ended generator is wired onto the DEFAULT /api/brain-chat turn; on an explicit open-ended prompt the brain VOLUNTEERS a novel grounded proposition (generative replay over its own fact-association graph), moat-verified + flagged as a hypothesis
lane: integration-first (the GENERATE faculty on the default chat, beyond recall)
integration_faculty: open-ended-generation
verdict: LANDED — additive + guarded, no regression. The validated 6-seed-GO #3E generative-replay faculty (research/findings/2026-08-12-brain-owns-open-ended-generation-6seed-GO-novel-propositions-via-replay-moat-intact.md) is now reachable on the DEFAULT single-fact `/api/brain-chat` turn (not only behind rich=True). ChatBrain.gate() gained a guarded branch that fires ONLY on an explicit open-ended prompt ("what might X ...", "tell me something new about X", "what else about X", "guess ...", "make something up about X"); for EVERY other question `_parse_open_ended` returns a sentinel and gate() is byte-identical (recall / abstain / learn / anaphora untouched — proven pristine==modified on the GPU-free smoke). On a match the brain resamples a candidate SVO with the REUSED b2 GenerativeReplayProposer (its `_sample_weighted`/`_weight_partner` weighted draw), gates it with the REUSED b2 `_plausible` (selectional-preference plausibility over the brain's own concept co-occurrence graph) + `_contradicts` (non-contradiction vs the composer store), MOAT-VERIFIES it (not a self-loop; matches the requested topic/action; and — the no-confab guarantee — `what_does` != patient AND `is_it_true` == 'unknown'), and returns the FIRST passing proposal as a FLAGGED HypothesisSVO that render() prints as "perhaps a v p  [a guess ... not something I was taught]". No plausible grounded proposal / an unknown topic → abstain (None), never confabulate. Verified in-process on the production onebrain path AND over the real HTTP endpoint (composer=onebrain).
artifacts:
  - research/runners/brain_chat_tui.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
  - research/findings/raw/_gen_wirein/verify_gen_result.json
  - research/findings/raw/_gen_wirein/verify_gen_rf_result.json
  - research/findings/raw/_gen_wirein/http_test_result.json
verification: onebrain in-process gate() (the exact entry point the endpoint calls) — (a) "what might dog eat?" → "perhaps dog eat cat" (novel, plausible, moat-verified, flagged); MOAT 0-confab 0/30 (no hypothesis is a stored fact or passes known-fact retrieval across 30 open-ended prompts); unknown topic ("dragon"/"unicorn") ABSTAINS; (c) NO REGRESSION all-True (recall dog→cat, anaphora it→cat eat fish, abstain fish/fly, teach wolf-eat-sheep→recall), and the GPU-free smoke VERDICT is byte-identical pristine vs modified. RF composer (negation-capable) — non-contradiction gate: is_it_true("dog","eat","fish")=="no" + `_contradicts` True + the negated fact is NEVER re-proposed in 60 draws; (b) plausibility gate LOAD-BEARING: gated true-plausible 1.000 vs the #3E uniform-random floor 0.116 → 8.63x, and lesioning the gate admits 13 implausible triples it rejects ("fox chase meat", "brain learn memory", "cat chase fish", "wolf eat deer", ...). HTTP POST /api/brain-chat (default tiny-demo, composer=onebrain): an open-ended prompt returns the flagged "perhaps ..." guess; recall turns unaffected.
---

# Open-ended generation on the DEFAULT chat — the brain volunteers novel grounded propositions, moat intact

## The faculty this wires

#3E proved (6-seed GO) that the brain can GENERATE novel, grounded, plausible propositions it was never taught, by
generative replay over its learned association structure, with the no-confab moat intact. That was a runner-level
de-risk. This wires it onto the PRODUCTION default turn so the live chat can VOLUNTEER a novel thought — the step from
"a fact lookup that recalls / abstains / learns" toward "a mind you can talk to that offers an idea".

## What changed (additive + guarded, in `research/runners/brain_chat_tui.py` only — NO `sim/` edit)

- **`gate()` gained ONE guarded prefix branch.** `_parse_open_ended(question)` matches a small fixed set of explicit
  open-ended lead-ins and returns `(topic, action)`; for anything else it returns the `_NOT_OPEN_ENDED` sentinel and
  gate() falls through to the unchanged pipeline. The trigger surface is deliberately narrow — a normal recall
  ("what does dog chase"), teach ("dog eat bone"), yes/no, or anaphora turn matches none of it.
- **`_generate_hypothesis(topic, action)`** draws with the reused b2 `GenerativeReplayProposer` sampler, gates with its
  reused `_plausible` + `_contradicts`, then moat-verifies (`what_does`/`is_it_true`) and EARLY-STOPS at the first
  passing proposal (so a turn runs only a few spiking moat queries, not a full replay). It returns a `HypothesisSVO`
  (a `list` subclass so it flows unchanged through the endpoint JSON `recalled_svo` and the smoke transcript).
- **`render()`** recognises a `HypothesisSVO` and prints an EXPLICIT guess ("perhaps a v p  [a guess ... not something I
  was taught]") — never asserted as knowledge. The honesty boundary is a deliverable.
- **The plausibility graph** is the brain's CLEAN concept co-occurrence over its stored facts (the agent's association
  structure — what the dlPFC `_assoc_graph` learned graph approximates). This was chosen over reading the
  substrate-learned `_learned_assoc.graph()` for two measured reasons: that graph is dense/noisy (its `__free` reserve
  slots add spurious edges that flood implausible recombinations like "dog use worm"), and it is fixed-vocab so it
  never sees runtime-taught facts. The clean fact co-occurrence is robust, includes everything the brain has heard, and
  is the same host-computed selectional-preference plausibility signal #3E used (there a corpus PPMI; here the brain's
  own heard facts).

## Verification (all four, honest)

- **(a) Generates + moat.** onebrain gate(): "what might dog eat?" → "perhaps dog eat cat" — novel (not stored),
  plausible (b2 gate), moat-verified (not a known fact), flagged. Across 30 open-ended prompts: 0 hypotheses that are a
  stored fact or pass known-fact retrieval (moat 0-confab). An unknown topic abstains.
- **(b) Plausibility gate load-bearing.** On the rf composer: gated true-plausible 1.000 vs the #3E uniform-random floor
  0.116 (8.63x advantage); lesioning the gate admits clearly-implausible cross-category triples it otherwise rejects.
- **(c) No regression.** onebrain: recall / anaphora / abstain / teach+recall all correct. The GPU-free smoke VERDICT
  is byte-identical with the branch present vs a pristine file — the guarded prefix changes nothing off-trigger.
- **(d) HTTP.** POST /api/brain-chat with the default brain (composer=onebrain) returns the flagged "perhaps ..." guess
  on an open-ended prompt; recall turns are unaffected.

## Residual (honest — the row's `scaffold_retired: NO`)

The generative DRAW is the b2 HOST oracle: the validated spiking soft-WTA sampler hardcodes the 8x8-taxonomy role pools
and cannot encode an arbitrary conversational vocab (it `KeyError`s), so the draw is numpy weighted sampling; the
LOAD-BEARING plausibility SIGNAL (the brain's own fact-association graph) is the brain's. The non-contradiction gate is
reused and fires on a negation-storing composer (rf, verified), but is inert on the onebrain default because onebrain
does not store a negation as a retrievable 'no' (an onebrain scope limit; the primary hypothesis-not-known moat still
holds). Broader novel-sentence generation on any topic (the emerge stream-cortex) remains unwired, and Qwen renders the
fluent surface. Next depth: a vocab-agnostic spiking generative draw, and onebrain negation storage so the
non-contradiction gate is live on the default path.
