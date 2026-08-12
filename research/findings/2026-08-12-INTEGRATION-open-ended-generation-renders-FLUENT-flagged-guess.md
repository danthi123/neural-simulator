---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — the brain's GENERATED open-ended content (a #3E HypothesisSVO) now renders as FLUENT PROSE via the Qwen mouth on the DEFAULT /api/brain-chat turn, framed as an explicit guess and SVO-VERIFIED so the mouth cannot swap the content; the raw "perhaps a v p" template is the fallback
lane: integration-first (the GENERATE faculty spoken FLUENTLY on the default turn, not as a template)
integration_faculty: open-ended-generation
verdict: LANDED — additive + guarded, no regression. Builds on the #3E wire-in (the generator on the default gate/render) and e2c62656d (the default turn flipped to the fluent RichAnswerComposer path). Two changes. (1) ChatBrain.render_hypothesis[_verified] renders a HypothesisSVO via the mouth's render_svo, VERIFIES the fluent sentence re-parses to the SAME (a,v,p) (the recall path's re-parse), frames it "Maybe <fluent> -- that's a guess from what I've learned, not something I was taught", and FALLS BACK to the raw flagged template on a mouth-verify miss / no renderer / raw mode. render() dispatches a HypothesisSVO here. (2) The RICH (default) path previously LEAKED the generated hypothesis: RichAnswerComposer treated the HypothesisSVO as a normal fact and CHAINED/ELABORATED stored recall around it, and on "guess about dog" spoke the novel guess ("The dog eats cat.") as an ASSERTED fact with no flag. gather() now intercepts the HypothesisSVO and returns it ALONE; answer() renders it as a SINGLE flagged guess (hypothesis=True, hypothesis_svo, fluent_hypothesis), never chained with asserted recall. The endpoint surfaces hypothesis/hypothesis_svo/fluent_hypothesis and reports recalled_svo=null for a guess. Recall/abstain/learn/anaphora + the rich RECALL path are byte-identical; the rich=False escape + the stub renderer still work; an unknown/ungrounded hypothesis still ABSTAINS.
artifacts:
  - research/runners/brain_chat_tui.py
  - research/runners/rich_answer_composer.py
  - webapp/server.py
  - tests/test_open_ended_generation_fluent.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
  - research/findings/raw/_fluent_guess/gpu_qwen_result.json
verification: CPU (stub, deterministic, tests/test_open_ended_generation_fluent.py all pass) — a HypothesisSVO renders as "Maybe the dog likes fish -- that's a guess ..." (fluent, flagged, SVO-verified), NOT the raw template; the RICH default path returns a SINGLE flagged guess with hypothesis=True + hypothesis_svo matching the prose + fluent_hypothesis=True over an 8-prompt battery (0 leaks — no unflagged non-abstain), the MOAT abstains on unknown subjects (dragon/unicorn/wizard/xyzzy), and recall/abstain are unregressed; render_hypothesis falls back to the raw "perhaps a v p" template with no mouth (raw mode). GPU (main process, real Qwen-0.5B spiking faculty, exact code path render_hypothesis_verified -> render_svo -> _verify): model on cuda:0; 6/6 open-ended prompts rendered a FLUENT flagged guess whose re-parse matched the hypothesis SVO (fluent=True), e.g. [dog,eat,bird] -> "Maybe the dog eats the bird -- that's a guess ..." (0.6-2.8s/render); the MOAT abstained 3/3 on unknown subjects; the content-swap guard is LOAD-BEARING on the real model — an ADVERSARIAL render forced to a wrong patient ("The dog chased the bird." for true (dog,chase,cat)) is REJECTED by _verify (would fall back to the template). The existing webapp brain-chat endpoint tests (recall/abstain, rich=False single-SVO escape, _brain_rich_default env) pass.
---

# The brain's GENERATED open-ended content now speaks as FLUENT PROSE — a flagged, moat-safe guess

## What this closes

The #3E generator was wired onto the default `/api/brain-chat` turn earlier today (the brain VOLUNTEERS a novel
grounded proposition on an open-ended prompt), but `render()` only printed the raw template `perhaps a v p  [a
guess ...]`. Meanwhile the production DEFAULT turn is the fluent `RichAnswerComposer` path (e2c62656d) — and that
path did NOT render the hypothesis as a guess at all: it treated the `HypothesisSVO` as an ordinary fact, chained
+ elaborated stored recall around it, and on some prompts spoke the novel guess itself as an ASSERTED fluent fact
with no flag (a moat leak — a generated, not-taught proposition presented as knowledge). This makes the generated
content render as FLUENT PROSE that is unmistakably a guess, on the production default path, without leaking.

## What changed (additive + guarded — NO `sim/` edit)

- **`ChatBrain.render_hypothesis` / `render_hypothesis_verified`** (`research/runners/brain_chat_tui.py`): render a
  `HypothesisSVO` via the mouth's `render_svo`, then VERIFY the fluent sentence re-parses to the hypothesis's exact
  `(a, v, p)` — the SAME re-parse the recall path uses, so the mouth cannot swap the content. On success the fluent
  sentence is framed `Maybe <fluent> -- that's a guess from what I've learned, not something I was taught`. On a
  mouth-verify miss (or no renderer / `--raw`) it falls back to the raw flagged template `perhaps a v p  [a guess
  ...]` (byte-identical to before). `render()` dispatches a `HypothesisSVO` here.
- **`RichAnswerComposer` gather()/answer()** (`research/runners/rich_answer_composer.py`): `gather()` intercepts a
  `HypothesisSVO` from the direct gate and returns it ALONE (no chain, no elaboration, topic `None` so a guess does
  not pollute the discourse thread). `answer()` renders that single guess via `render_hypothesis_verified` and
  returns `hypothesis=True`, `hypothesis_svo`, `fluent_hypothesis`. A guess is never mixed with asserted recall.
- **The endpoint** (`webapp/server.py`): on a hypothesis turn it adds `hypothesis`/`hypothesis_svo`/
  `fluent_hypothesis` and reports `recalled_svo=null` (a guess is not a recalled fact). A non-hypothesis rich turn
  is byte-identical (no extra keys).

## Why the RICH path had to intercept (the pre-fix leak, measured)

On the pre-fix rich path (richer in-process brain, stub): `guess about dog` returned `"The dog eats cat. ..."` with
`facts=[['dog','eat','cat'], ...]` — but `dog eat cat` is NOT stored; it is the generated hypothesis, spoken as an
asserted fact. Other open-ended prompts dropped the hypothesis and asserted a chain of stored facts. Both are
wrong: a guess must be flagged, and it must not be dressed up with asserted recall. Intercepting the
`HypothesisSVO` and speaking it as ONE flagged guess fixes both (0 leaks across the battery, post-fix).

## Verification

- **(a) CPU, deterministic (stub renderer), `tests/test_open_ended_generation_fluent.py` — all pass.** A
  `HypothesisSVO` renders "Maybe the dog likes fish -- that's a guess from what I've learned, not something I was
  taught" (fluent, flagged, SVO-verified), not the raw template. The RICH default path over an 8-prompt open-ended
  battery returns a SINGLE flagged guess each time (`hypothesis=True`, `hypothesis_svo` re-parses to the prose,
  `fluent_hypothesis=True`, `n_sentences==1`) with **0 leaks** (no unflagged non-abstain). The `render_hypothesis`
  fallback returns the raw `perhaps a v p` template with no mouth (raw mode).
- **(b) MOAT.** Open-ended prompts about unknown/ungrounded subjects (dragon, unicorn, wizard, xyzzy) ABSTAIN — the
  brain does not invent about what it never heard of. 0 confab leaks across the open-ended battery.
- **(c) No regression.** Recall (multi-sentence, taught cue) + abstain (untaught cue) are unchanged on the rich
  path; the rich=False single-SVO escape and the `_brain_rich_default` env behaviour are unchanged (existing
  webapp brain-chat endpoint tests pass). Only the `HypothesisSVO` branch of `render()` changed.
- **(d) GPU (main process, real Qwen-0.5B spiking faculty, model on `cuda:0`).** The exact production render path
  (`render_hypothesis_verified` -> the real `render_svo` -> `_verify` re-parse of the PROSE) rendered a FLUENT
  flagged guess for all 6 open-ended prompts, each SVO-verified against the hypothesis (`fluent=True`, 0.6-2.8s per
  render): e.g. `[dog,eat,bird]` -> "Maybe the dog eats the bird -- that's a guess from what I've learned, not
  something I was taught", `[bird,chase,cat]` -> "Maybe a bird chased a cat through the park -- ...". The moat
  abstained 3/3 on unknown subjects. **The content-swap guard is LOAD-BEARING on the real generative model**: an
  ADVERSARIAL render forced toward a WRONG patient (`render_svo_adversarial(dog, chase, "bird")` for the true fact
  `(dog, chase, cat)`) produced "The dog chased the bird." and `_verify` rejected it as `[dog,chase,cat]` (=False)
  — so a drifted/hallucinated fluent sentence never passes as the hypothesis; it falls back to the raw template.

## Residual (honest)

The fluent SURFACE is the external Qwen-0.5B mouth (the temporary articulation scaffold; `scaffold_retired: NO`).
The generative DRAW + plausibility signal are unchanged from the #3E wire-in (the b2 host oracle draw; the clean
co-occurrence plausibility graph over the brain's own facts; onebrain does not store a negation as a retrievable
'no' so the non-contradiction gate is inert on the default composer — the primary hypothesis-not-known moat still
holds). This finding is about the SURFACE (fluent, flagged, verified, leak-free), not those residuals.

## Note on the live server launch

The `uvicorn` server's STARTUP-WARM daemon thread built the Qwen renderer on CPU (`torch.cuda.is_available()`
returned differently in that thread context), which made the first turn pathologically slow. This is a pre-existing
INFRA issue in the warm path, not in this change — the exact fluent-mouth code path was confirmed on GPU by
building the renderer on the MAIN process. A follow-up should build the warm Qwen singleton on the main thread (or
guard the warm thread's device selection) so the server's first turn is fast on GPU.
