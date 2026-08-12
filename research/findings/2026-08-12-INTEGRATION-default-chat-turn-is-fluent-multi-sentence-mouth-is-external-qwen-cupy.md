---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — the DEFAULT /api/brain-chat turn now produces a FLUENT MULTI-SENTENCE grounded reply (the RichAnswerComposer path) instead of a single template-rendered SVO; the spiking onebrain supplies the content, the external Qwen-0.5B mouth supplies the prose surface (requires SIM_BACKEND=cupy)
lane: integration-first (the fluent surface of the default conversational turn)
integration_faculty: discourse-planner
verdict: LANDED — additive + guarded, no regression. The multi-sentence RichAnswerComposer path was already 6-seed GO but OPT-IN (rich=True). This flips the PRODUCTION default: BrainChatRequest.rich is now TRI-STATE (bool|None, default None); when a caller OMITS rich the turn takes _brain_rich_default() (env BRAIN_RICH, default ON) -> the fluent multi-sentence path with the neural dlPFC planner (neural_planner=True). ESCAPES preserved byte-identically — rich=False in the body, or BRAIN_RICH=0 in the environment, gives the OLD single-SVO turn; renderer=stub/raw still select the mouth. Content recall stays the genuinely-spiking onebrain composer (BRAIN_COMPOSER_KIND=onebrain default). The fluent PROSE surface is the off-bridge Qwen-0.5B scaffold (a temporary articulation mouth per the owner) and needs SIM_BACKEND=cupy; a GPU-free host falls back to the multi-sentence TEMPLATE stub (still multi-sentence + moat-gated, not prose). NO sim/ edit.
artifacts:
  - webapp/server.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
  - tests/test_webapp_server.py
  - research/findings/raw/_fluent_default/live_cupy_default_verify.txt
  - research/findings/raw/_fluent_default/live_cupy_anaphora_verify.txt
  - research/findings/raw/_fluent_default/verify_chat.py
verification: LIVE over the real HTTP endpoint, server launched SIM_BACKEND=cupy on the CUDA box (Qwen2.5-0.5B cached, no download). (a) DEFAULT body (no rich field) 'what are you' -> 'The brain uses spikes for communication. The brain learns words.' — FLUENT, 2 sentences, renderer='off-bridge Qwen-0.5B (spiking forward)', rich=True, n_sentences=2, composer=onebrain, verified=True. (b) MOAT: 'what is the capital of france' ABSTAINS ("I don't know about that.") — the Qwen mouth knows Paris but the brain does NOT leak it (firewall holds); each rendered sentence is a gate-sourced VERIFY-checked fact. Teach 'wolf hunt deer' (declarative SVO) -> recall 'what does the wolf hunt' -> 'A wolf hunts deer.' (in-loop learning, composer=onebrain). (c) NO REGRESSION: rich=False escape 'what does the dog chase' -> single-SVO 'The dog chased the cat.' (rich=False, no n_sentences/supporting_facts keys); 'what does the dragon breathe' abstains. ANAPHORA correct on BOTH paths (warm 'default' session): escape rich=False 'what does the dog chase'->'The dog chased the cat.' then 'what does it eat'->'The cat eats fish.' (it->cat->fish); default rich 'what does the dog chase'->'The dog chased the cat. The cat eats fish.' then 'what does it eat'->'The cat eats fish.'. CPU wiring (SIM_BACKEND=numpy stub renderer): the modified default-turn test + the new rich=False-escape test + the _brain_rich_default env unit test all pass.
---

# The default /api/brain-chat turn is now a fluent multi-sentence reply

## What this changes

The production conversational turn used to answer with ONE template-rendered SVO sentence: the default was
`rich=False`, which routed the single-fact gate -> one `StubRenderer`/Qwen sentence. The FLUENT multi-sentence path
(the `RichAnswerComposer`: multi-fact recall + multi-hop chain + per-topic elaboration, each sentence re-parsed and
moat-gated) already existed and was 6-seed GO, but it was OPT-IN behind `rich=True`. This flips the DEFAULT so every
`/api/brain-chat` turn that does not opt out produces the fluent multi-sentence reply from the spiking onebrain's own
content — the step from "a one-line fact lookup" toward "a mind that answers in prose".

## What changed (additive + guarded, `webapp/server.py` + tests + ledger only — NO `sim/` edit)

- **`BrainChatRequest.rich` is now TRI-STATE** (`bool | None`, default `None`). `None` = the caller omitted the field.
- **`_brain_rich_default()`** resolves the omit-default from the environment: `BRAIN_RICH` unset -> `True` (fluent);
  `BRAIN_RICH=0/false/no/off` -> `False` (the single-SVO production kill-switch).
- **`brain_chat()`** computes `use_rich = req.rich if req.rich is not None else _brain_rich_default()` and branches on
  `use_rich`. So: omit `rich` -> fluent; `rich=False` -> the OLD single-SVO path (byte-identical); `rich=True` -> fluent.
- **The `_get_rich_composer` docstring** was corrected: it claimed the webapp used `neural_planner=False` for latency,
  but the code already uses `neural_planner=True` (speed is secondary) — the stale claim (and its stray literal) is gone.
- **Tests**: the existing default-turn test now asserts `rich=True`/`n_sentences>=1`; a new `rich=False` escape test
  asserts the single-SVO shape survives; a `_brain_rich_default` unit test pins the env kill-switch.

## The mouth is an external scaffold — fluent prose needs SIM_BACKEND=cupy

The fluent PROSE surface is the off-bridge **Qwen2.5-0.5B** renderer, a temporary articulation mouth (Broca-like
scaffold, per the owner's "conditioned-articulation-crutch-if-faculties-load-bearing" allowance — burning it down is a
later step, not this one). `_default_brain_renderer()` returns `qwen` ONLY when `SIM_BACKEND==cupy` AND CUDA torch is
present; otherwise it returns the GPU-free `stub` template renderer. **Production must launch with `SIM_BACKEND=cupy`
for fluent prose.** A GPU-free host still gets a MULTI-SENTENCE reply — but rendered by the template stub, not prose.
The CONTENT (recall, multi-hop chain, elaboration) is the genuinely-spiking onebrain composer
(`BRAIN_COMPOSER_KIND=onebrain`) either way; only the surface form differs.

## Honest boundary — an observed brain characteristic (not a regression, not a moat break)

On the tiny 5-fact demo brain, some self-referential "how do you X" questions ABSTAIN even in a FRESH session — e.g.
`how do you learn` returns "I don't know about that." although `what are you` surfaces `brain learn words` among its
facts. This is a question-comprehension / recall property of this small brain's parser+recall on its 5-fact vocab, and
it is ORTHOGONAL to the rich-vs-single default flip (the underlying gate/recall for that question is the same on both
paths; the flip only chooses which path is default). The flip is verified on the questions the brain does answer
(`what are you` -> a 2-sentence fluent reply; direct recall; teach->recall; anaphora on both paths). Not attributed to
this change; noted for a later look at the tiny-demo's self-question recall (a bigger developed bundle carries more
facts + a denser graph, so the fluent default has more to say).

## Verification

See the `verification:` frontmatter and the live cupy HTTP transcripts under
`research/findings/raw/_fluent_default/`. The Qwen fluent path was confirmed LIVE (not simulated): the qwen renderer
name string is `off-bridge Qwen-0.5B (spiking forward)` and `composer='onebrain'` on every turn.
