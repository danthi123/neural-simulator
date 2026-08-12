---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — #1 CHOOSE: the question→(agent,action) comprehension is now NEURAL (on-brain BridgeParser), host router retired for factual questions
lane: integration-first (WIRING BACKLOG #1 CHOOSE)
integration_faculty: content-selection
verdict: LANDED + lesion-load-bearing. The default `/api/brain-chat` recall now COMPREHENDS a factual question's (agent, action) NEURALLY: `ChatBrain._neural_question_parse` strips the wh/aux/stopwords and presents the content words (position-padded to SVO, the queried patient a placeholder) to the ON-BRAIN `BridgeParser`, whose (position, voice)→role conjunction FIRES the role assignment on Izhikevich neurons — the SAME parser `hear()` uses to comprehend a stored sentence, on the shared onebrain bridge (`role_of` verified [agent, action, patient] on that bridge). This replaces the host first-known-token / positional heuristic. The parser is AUTHORITATIVE for a factual-SVO-shaped question (≥2 content words, none a self-alias): when it DECLINES on such a question, the gate ABSTAINS ("__ABSTAIN__") instead of falling to the host `QuestionRouter`'s role-blind keyword confab — so the router is RETIRED for factual questions (still handles self/identity + the anaphora fallback). LESION-LOAD-BEARING: lesioning `parser.role_of` (return a junk role) makes BOTH a taught fact and a build-time fact ABSTAIN (was "dog chase cat" → IDK), so the comprehension is genuinely the parser's, not the router's. No regression: healthy recall works for build-time (brain→spikes, dog→cat), taught (wolf→deer), anaphora (it→cat eat fish), and abstain (fish/fly→IDK). Only the onebrain default gets the neural parse; the rf escape (no parser) keeps the host heuristic + router.
artifacts:
  - research/runners/brain_chat_tui.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
verification: onebrain in-process gate() (the production entry point the endpoint calls) — HEALTHY: "what does brain use?"→"brain use spikes", "what does dog chase?"→"dog chase cat", teach "wolf hunt deer"→recall "wolf hunt deer", anaphora "what does it eat?"→"cat eat fish", "what does fish fly?"→IDK; role_of on the shared bridge = [agent, action, patient]. LESIONED (role_of→junk): "what does wolf hunt?"→IDK AND "what does dog chase?"→IDK (the answer CHANGES → load-bearing).
---

# #1 CHOOSE — the question→(agent,action) comprehension is now NEURAL; the host router is retired for factual questions

## The host gap this closes

After #0 (the recall/store is genuinely spiking by default) and #2 (the SELECTION is substrate-first via `what_does`),
the residual host scaffold in the default CHOOSE path was the QUESTION PARSE: `_substrate_recall` extracted (agent,
action) with a Python heuristic (`next(t in agents_set)` then positional `content[0]/content[1]`) — a host vocabulary
lookup, not comprehension — and any question it could not resolve fell to `QuestionRouter.match_fact`, a role-blind
keyword overlap that CONFABULATED (the "what does fish fly?" → "cat eat fish" failure).

## The change

<!--derived-->
`ChatBrain._neural_question_parse(content)` presents the stripped content words (position-padded to SVO, the queried
patient a placeholder) to the ON-BRAIN `BridgeParser` — the composer's own parser on the shared onebrain bridge, whose
(position, voice)→role conjunction fires the role assignment on Izhikevich neurons (the same parser `hear()` uses to
comprehend a stored sentence). `role_of` on that shared bridge reads `[agent, action, patient]` for positions 0/1/2, so
`padded=[w0, w1, "__q__"]` → agent=w0, action=w1. In `_substrate_recall`, when a parser is present (the onebrain default)
and the question is factual-SVO-shaped (≥2 content words, none a self-alias), the parser is AUTHORITATIVE: a comprehended
parse drives the substrate recall; a DECLINED parse returns "__ABSTAIN__" (the gate abstains) rather than falling to the
router's keyword confab. A self/identity/short question, or the rf escape (no parser), keeps the host heuristic + the
router fallback. NO `sim/` edit.

## Result — lesion-load-bearing on the production entry point

Verified in-process through `gate()` (exactly what the `/api/brain-chat` endpoint calls):
- **Healthy** — factual recall is comprehended by the parser: "what does brain use?"→"brain use spikes", "what does dog
  chase?"→"dog chase cat", teach "wolf hunt deer" then recall→"wolf hunt deer", anaphora "what does it eat?"→"cat eat
  fish", "what does fish fly?"→IDK (honest abstain). `role_of` on the shared bridge = [agent, action, patient].
- **Lesioned** (`parser.role_of` → a junk role): "what does wolf hunt?"→IDK AND "what does dog chase?"→IDK — the answer
  CHANGES for BOTH a taught and a build-time fact, so the neural parse is doing the comprehension (not the router).

## Honest scope

Only the onebrain default gets the neural parse (the rf escape keeps the host heuristic). The router is retired for
factual-SVO questions but still owns self/identity + the anaphora-fallback (so `scaffold_retired=PARTIAL`, not full). The
parser is a 3-word SVO position→role map, so a 3+-content question (adjectives, oblique arguments) declines → abstains
(honest, not a host guess) rather than being parsed — richer comprehension (typed frames, the multicue competition
parser) is a later wire-in. Next: retire the router's self/identity path (neural self-model) + the deeper LEARN (BTSP
per-turn write) + the rich/GENERATE default.
