---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — content selection (CHOOSE): direct factual questions decided by the substrate, host keyword-confab retired
lane: integration-first (WIRING BACKLOG #1)
integration_faculty: content-selection
verdict: SECOND owner-visible act moved to the substrate. On the default /api/brain-chat turn, a direct factual question is now DECIDED by the spiking substrate — recall the fact (inner.what_does) OR abstain honestly — and the host QuestionRouter's role-blind keyword CONFABULATION is RETIRED for such questions. The production lesion probe reads CHOOSE False->True: it abstains on a well-formed unanswerable question ("what does fish fly?" -> "I don't know", previously "cat eat fish"), and lesioning the substrate recall brings the confab back (load-bearing). No regression: valid facts, multi-turn anaphora, self/identity, and new-word learning all intact; the host router remains only the fallback for self/identity + (noisy) anaphora. Trio now CHOOSE=True, LEARN=True, GENERATE=False.
artifacts:
  - research/findings/raw/_production_lesion/probe.json
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
verification: production lesion probe — CHOOSE neural=True, abstains_on_unanswerable=True, lesion_load_bearing=True.
---

# INTEGRATION #1 (CHOOSE) — the substrate decides a direct factual question (recall or honest abstain); the host keyword-confab is retired, lesion-verified

## The change

`ChatBrain.gate` extracts (agent, action) from a well-formed question and recalls from the spiking substrate first
(`_substrate_recall`). New this integration: when the substrate has a valid query but NO stored fact, it returns an
`__ABSTAIN__` sentinel and the turn **abstains honestly** instead of falling through to the host `QuestionRouter`'s
role-blind keyword match — which used to CONFABULATE (e.g. "what does fish fly?" keyword-matched "fish" -> "cat eat
fish"). A self-alias query (self/identity) and an anaphora-resolved query (the discourse WM referent is a noisy spiking
attractor, so its pick is not trusted for a hard abstain) still fall to the host router. NO `sim/` edit.

## Result — production lesion probe (`research/findings/raw/_production_lesion/probe.json`)

<!--derived-->
- **CHOOSE False -> True.** "what does fish fly?" -> "I don't know about that." (was a keyword-confab "cat eat fish").
- **Lesion-verified load-bearing.** With `_substrate_recall` lesioned, the same question CONFABULATES again (falls to the
  host keyword guess) -> the substrate decision is what produces the honest abstention.
- **No regression.** Valid facts ("what does dog chase?" -> "dog chase cat"), multi-turn anaphora ("what does it eat?" ->
  "cat eat fish", via the host fallback since the WM referent is noisy), self/identity ("what are you?"), and new-word
  learning ("wolf hunt deer" -> "Got it"; recall -> "wolf hunt deer") all unchanged.

## Honest scope

This is PARTIAL CHOOSE: the substrate now decides direct factual questions (answer or honest abstain), retiring the host
confab for them. The broader CHOOSE — neural QUESTION COMPREHENSION (parsing an arbitrary question to intent; today the
wh/aux strip + SVO extraction is host tokenization) and OPEN-ENDED topic choice — is further, and the noisy anaphora WM
(a spiking attractor that mis-resolves "it") is its own de-risk (the host fallback currently masks it). Scaffold NOT
retired (QuestionRouter still the self/identity + anaphora fallback). But the owner-visible act — *the brain decides
whether it can answer, honestly* — is now substrate-driven and lesion-verified. Two of the three defining acts (CHOOSE,
LEARN) are on; GENERATE is next.
