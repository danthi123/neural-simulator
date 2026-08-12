---
type: finding
status: contributing
date: 2026-08-11
mechanism: production conversation pipeline — code-traced honest inventory (what actually runs when the owner talks to the sim TODAY)
lane: one-brain / conversation integration (honest-negative)
verdict: CODE-TRACED HONEST NEGATIVE on the owner's mental model ("everything except the speech generator is already spiking/one-brain; the mouth-burn-down just biologizes an already-working open-ended speech faculty"). The DEFAULT production path (/api/brain-chat, brain=tiny-demo, composer_kind=rf, renderer=qwen) is verified single-SVO-fact recall over a FROZEN hand-seeded KB, bracketed by HOST comprehension + HOST planning + an off-bridge transformer mouth. Between sensation and speech the HOST does: content/topic SELECTION (QuestionRouter.match_fact keyword-overlap argmax, brain_chat_tui.py:120-145), honesty/abstention (host equality/None, :236-241), anaphora (string sub, :253-267), discourse planning (RichAnswerComposer neural_planner=False, server.py:3009), word-decode + fact-scan (numpy argmax/masking, rf_phasor_composer.py:684-705), and concept/role codes (host rng.uniform phasors, :173/187). GENUINELY spiking in the live turn = ONE op: the RF bind/bundle/unbind VSA algebra on resonate-and-fire neurons — and it only VERIFIES/vetoes the fact the host router already chose. NOT open-ended (5 hand-seeded facts, abstains otherwise; the open-endedness is the transformer's phrasing). NO in-loop learning (stored_facts snapshotted at construction; the turn is read-only gate()+render(); plasticity exists only on a DIFFERENT endpoint /api/llm-chat's narrow N/E/S/W direction memory). NOT one-brain (all 42+ bundles hardcode composer_kind=rf, 0 onebrain; per-op separate bridges + host numpy glue; hear_synaptic is an opt-in de-risk that never reaches the endpoint). The many GO spiking faculties (WM, binding, episodic, affect, familiarity gate, neural planner, one-brain composer) are VALIDATED DE-RISKS that are DEFAULT-OFF / not wired into the production answer path.
seeds: []
verification: workflow whbgq2k92 (5 code-mappers + 4 adversarial refuters + synthesis); load-bearing claims spot-verified against the code by the coordinator (match_fact host-selection; read-only gate()+render() turn; stored_facts snapshot).
artifacts:
  - research/findings/raw/_production_pipeline_map/pipeline_honest_map.json
  - research/runners/brain_chat_tui.py
  - research/runners/rf_phasor_composer.py
  - webapp/server.py
---
<!--derived--> Full traced result (the 5 area-maps + 4 adversarial verdicts + synthesis) is saved at
`research/findings/raw/_production_pipeline_map/pipeline_honest_map.json`; every file:line below is quoted from that
code trace and spot-verified against the source.

# The CURRENT production conversation pipeline is largely HOST, not one spiking brain — a code-traced honest inventory (read before claiming the production sim "converses on one substrate")

## Why this exists

The owner asked, precisely: *in the current/production state, can I get FLUENT, OPEN-ENDED, LEARNING conversation with
EVERYTHING EXCEPT the speech generator running biologized/spiking on a single shared substrate?* Their model: only the
mouth is non-biologized. This maps the ACTUAL live pipeline from the code (not from findings' claims — findings overclaim,
as this same session proved for gap#4) and answers honestly. **Answer: NO, on all three counts.**

## Evidence (the traced sources)

Code traced: `research/runners/brain_chat_tui.py` (the ChatBrain/QuestionRouter/render turn),
`research/runners/rf_phasor_composer.py` (the RF VSA composer + host word-decode), `webapp/server.py` (the
`/api/brain-chat` endpoint + renderer selection), `sim/bridge_memory.py` (the separate direction-memory endpoint), `webapp/static/app.js` (the chat surface).
Full agent trace: workflow whbgq2k92 (journal under the session subagents dir), load-bearing claims spot-verified against
the code by the coordinator.

## The default live turn (what runs when the owner types)

`/api/brain-chat` (webapp/server.py:3213) with the default `brain="tiny-demo"` → a turn is exactly
`gate_svo = chat.gate(msg)` then `answer = chat.render(gate_svo)` (server.py:3345-3349) — **both read-only**. The app's
only chat surface (webapp/static/app.js:2366) POSTs here.

## Host vs spiking in the live turn (verified file:line)

**HOST between sensation and speech (NOT just the mouth):**
- **Comprehension / topic SELECTION** — `QuestionRouter.match_fact` (brain_chat_tui.py:120-145): pure-Python
  keyword-overlap argmax (`content_hits*10 + brain_hit`) over a Python fact list, hand-coded stopwords + synonym dict
  (:79-96). **This decides what the reply is about** — the load-bearing comprehension step, and it is host.
- **Honesty / abstention** — host `None`/equality (brain_chat_tui.py:236-241); the spiking v320 familiarity gate is NOT
  on this path.
- **Anaphora** — host string substitution (:253-267).
- **Discourse planning (rich mode)** — `RichAnswerComposer(neural_planner=False)` (server.py:3009); the neural dlPFC
  planner is explicitly dropped for latency.
- **Word-decode + fact-scan** — numpy argmax matched-filter + masking (rf_phasor_composer.py:684-705); spiking cleanup
  is default-OFF (:656).
- **Concept/role codes** — host `rng.uniform` random phasors (:173/187), not learned/self-organized.
- **The mouth** — off-bridge Qwen2.5-0.5B transformer, pretrained weights untouched, full softmax attention (NOT reduced
  to WKV, NOT retrained) (`_grounded_lang_integration_derisk.py:193-257`); on a GPU-less host it degrades to host
  templates.

**GENUINELY spiking in the live turn — ONE op:** the RF bind/bundle/unbind VSA algebra on resonate-and-fire neurons +
complex synapses (rf_phasor_composer.py:210). In a Q&A turn it runs as `query_patient` and only VERIFIES/vetoes the fact
the host router already chose — it selects nothing, plans nothing, decides no honesty.

## Per requirement (the direct answer)

- **Fluent? YES — but it is the TRANSFORMER SCAFFOLD's fluency, not the brain's**, and clamped (render() re-parses the
  prose and requires the stored content words survive, else speaks the raw triple, :295-306).
- **Open-ended? NO.** Answers only by matching a pre-loaded SVO triple; the default brain is 5 hand-seeded facts
  (:447-453) and abstains on everything else. The apparent open-endedness is the transformer phrasing a pre-selected
  3-word fact.
- **Learning-in-loop? NO.** `stored_facts` is snapshotted once at construction (`_refresh_facts`, :216) and never
  refreshed in a turn; no `store()`/plasticity is reachable from the answer path. In-loop synaptic change exists only on
  a DIFFERENT endpoint (`/api/llm-chat` → `BridgeMemory.store`, sim/bridge_memory.py:188-237) — a narrow host-LLM-driven
  key→N/E/S/W direction memory on a separate bridge, not the open-ended agent.
- **One-brain? NO.** All 42+ bundles + both built-ins hardcode `composer_kind="rf"` (0 "onebrain"); `_build_chat_brain`
  has no onebrain branch. Even the RF algebra runs on a `_bridge_cache` of separate per-neuron-count bridges with host
  numpy carrying phasors between ops; parser→composer is a host string dict. The synaptic cross-region handoff
  (`hear_synaptic`) is an opt-in de-risk flag that never reaches the endpoint.

## The honest reconciliation (this is NOT "the spiking work is fake")

The project has a large body of GENUINELY-spiking, 6-seed-GO de-risk faculties — variable-binding WM, conjunctive
binding, episodic recall on a closed CA3, affect/curiosity/self-schema, the spiking familiarity gate, the neural
discourse planner, the one-brain composer. **They are real and validated.** The honest gap is that they are
**DEFAULT-OFF / not wired into the production answer path** — the default `/api/brain-chat` turn routes through the host
keyword-router + numpy composer + transformer mouth, using the spiking substrate as a verified associative-recall memory.
So the accurate framing is NOT "biologize the last non-spiking piece": **open-ended, learning, one-brain conversation
does not yet exist in the production path** — it exists as validated pieces awaiting integration.

## What it would actually take (ranked — this is the real remaining work to fluent one-brain conversation)

1. **A spiking comprehension / content SELECTOR** to replace `QuestionRouter` — the substrate must decide *what to say*,
   not just verify a host pick. (Biggest gap; this is the "what to say" cognition.)
2. **In-loop plasticity on the answer path** — a reachable `store()`/BTSP/Hebbian write so talking changes synapses
   (today the turn is read-only).
3. **Open-ended content generation from the brain**, so answers aren't confined to pre-seeded SVO triples (the emergence
   engine / stream cortex path).
4. **Genuine one-brain wiring** — route the built `OneBrainComposer` / `hear_synaptic` into `_build_chat_brain` so
   faculties hand off across synapses on one persistent substrate, not host glue over per-op bridges.
5. **Wire the already-built spiking pieces into the default path** — the v320 familiarity/abstention gate and
   `neural_planner=True` (both built, both default-off).
6. **THEN** biologize the mouth.

This maps directly onto the owner-approved crux re-point (mouth-burn-down + Gate-B) but shows it is BROADER than the
mouth: the comprehension, the content selection, the planning, and the learning are all still host on the production
path today.
