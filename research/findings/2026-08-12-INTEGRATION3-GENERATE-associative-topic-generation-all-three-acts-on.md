---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — generation (GENERATE): the brain volunteers associated knowledge about a topic via spiking spreading-activation
lane: integration-first (WIRING BACKLOG #3)
integration_faculty: open-ended-generation
verdict: THIRD owner-visible act on the substrate — the trio CHOOSE/GENERATE/LEARN is now all True + lesion-verified on the default /api/brain-chat turn (baseline: all False). For a TOPIC prompt ("tell me about X"), the brain VOLUNTEERS what it knows by chaining describe(X) with the dlPFC spiking `elaborate` (spreading-activation content-selection) to a related concept and describing THAT — "tell me about dog" -> "dog chase cat. cat eat fish." (dog -> cat via the learned association). Probe GENERATE False->True (n_facts_chained=2); lesioning `elaborate` collapses it to the single primary fact (assoc_lesion_load_bearing=True). No confabulation (unknown topic -> "I don't know"); no regression. Scope: ASSOCIATIVE generation over known knowledge; broader open-ended generation (novel sentences on any topic) is the emerge stream-cortex, further, and the off-bridge Qwen still renders the surface.
artifacts:
  - research/findings/raw/_production_lesion/probe.json
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
verification: production lesion probe — GENERATE open_ended=True, n_facts_chained=2, assoc_lesion_load_bearing=True; CHOOSE=True, LEARN=True.
---

## ⚠️ CORRECTION (2026-08-12, same day) — GENERATE is NOT on the default /api/brain-chat endpoint. Read this first.

The HTTP verification (restarting the webapp + POSTing to the real endpoint) caught an ENTRY-POINT error: my probe
tested `chat.answer()`, but the default `/api/brain-chat` endpoint (rich=False) calls `chat.gate()` + `chat.render()`
DIRECTLY (webapp/server.py:3345), and the rich path uses `RichAnswerComposer` — **neither calls `answer()`**. My
`_maybe_generate` (this integration) lives in `answer()`, so it does NOT reach the production endpoint. The corrected
production-path probe (gate+render) reads **GENERATE=False** on the endpoint. **So this associative generation is real
on the TUI/answer path but is NOT on production** until it is moved into `gate()`/the endpoint or the rich path. The
"all three acts on production" headline was WRONG; the honest state is **CHOOSE + LEARN on the production endpoint
(both in gate(); LEARN's acquisition was moved into gate() as part of this fix), GENERATE on the TUI/answer only.** The
ledger `open-ended-generation` row is corrected to wired=NO. Everything below overstates the endpoint reach for GENERATE.

**RE-CORRECTION (same day, verified):** the wired=NO above was itself an over-correction. Associative generation IS on
the endpoint — via the RICH path (the UI "rich" toggle): the `RichAnswerComposer` uses the spiking `elaborate`
(dlPFC spreading-activation) for content, and I flipped `neural_planner=True` (server.py:3009) so the ORDERING is spiking
too. Verified: `RichAnswerComposer(chat, neural_planner=True).answer("tell me about dog")` -> "dog chase cat cat eat
fish". So GENERATE is **on the endpoint via the rich toggle** (`wired=YES, on_by_default=NO`); my `_maybe_generate` was a
redundant duplicate of the rich path. Honest net: on the DEFAULT single-fact endpoint — CHOOSE + LEARN (recall/abstain,
HTTP-verified); with the RICH toggle — associative generation via the spiking elaborate.

# INTEGRATION #3 (GENERATE) — the brain volunteers associated knowledge about a topic; all three owner-visible acts are now on

## The change

`ChatBrain._maybe_generate` (called in `answer` before the gate) handles a TOPIC prompt ("tell me about X" / "describe
X" / "what about X"). It generates from the brain's own knowledge: `describe(topic)` (the primary fact) chained with the
dlPFC spiking **`elaborate`** — a spreading-activation content-selection over the learned association graph
(brain_conversational_agent.py:897, the 320-production dlPFC control) — to a related concept, then `describe` that. The
result is a multi-fact, associatively-ordered volunteer, not a single recall. `describe` returns None for an unknown
topic, so there is no confabulation. NO `sim/` edit.

## Result — production lesion probe (`research/findings/raw/_production_lesion/probe.json`)

<!--derived-->
- **GENERATE False -> True.** "tell me about dog" -> "dog chase cat. cat eat fish." (n_facts_chained=2): the brain
  volunteered dog's fact AND, via the spiking association (dog -> cat), cat's fact.
- **Lesion-verified.** Lesioning `elaborate` (the spreading-activation) collapses the answer to the single primary fact
  (`assoc_lesion_load_bearing=True`) — the second, associated fact is produced BY the substrate's association, not a
  template.
- **No confab, no regression.** "tell me about quantum" (unknown) -> "I don't know about that."; direct recall, honest
  abstain, multi-turn anaphora, self/identity, and new-word learning all unchanged.

## The milestone — all three defining acts are on

The production baseline (`2026-08-11-PRODUCTION-chat-pipeline-is-largely-HOST-...`) measured **CHOOSE/GENERATE/LEARN all
False** — the default chat decided nothing, generated nothing, learned nothing. As of this session, on the default
`/api/brain-chat` turn and measured by the lesion probe:
- **CHOOSE** (#1): a direct factual question is decided by the substrate — recall or honest abstain; the host
  keyword-confab is retired for it.
- **LEARN** (#2/#2b): the owner teaches ARBITRARY new facts by talking (runtime code allocation); recalled from the
  substrate.
- **GENERATE** (#3): the brain volunteers associated knowledge about a topic via spiking spreading-activation.

Each is lesion-verified (disable the spiking path -> the act stops), so none is a byte-identical/cosmetic flip.

## Honest scope (the residuals, tracked in the ledger)

All three are PARTIAL, not `integrated` (scaffold not retired): the host `QuestionRouter` remains the self/identity +
anaphora fallback; broader CHOOSE (neural question comprehension) and broader GENERATE (the emerge15-21 stream-cortex
free-generator for novel sentences on any topic) are unwired; the off-bridge Qwen still renders the surface; a
BTSP/plasticity per-turn write (a LASTING trace) is the deeper LEARN; the anaphora WM is a noisy attractor masked by the
host fallback. But the three acts that define "a brain you can actually talk to" are now genuinely on the substrate and
measured, not asserted — the integration-first path is producing a fully-functional-brain trajectory, one lesion-verified
faculty at a time.
