---
type: finding
status: contributing
date: 2026-08-28
mechanism: open-ended-generation-time-consensus-veto-skip-continue
lane: E-language-open-ended-honesty
seeds: [42]
seed-waiver: The evidence landed this session is a deterministic wiring-sanity battery over a fixed
  fake-tokenizer/fake-model script (no seed dependence at all -- the same reasoning the 2026-08-27 PARTIAL
  finding's controlled unit battery and the 2026-08-28 token-id-continuation finding's own wiring battery
  already used). The DECISIVE live-mouth multi-seed GPU measurement (does skip-and-continue actually reach
  genuine, unfabricated later content on the real off-bridge Qwen mouth, and does it regress the safety net) is
  explicitly NOT run this session -- CPU/numpy-only, no GPU touch, per this session's own scope -- and this
  finding does not claim its result.
instrument: research/runners/_open_ended_gen_time_skip_continue_wiring_verify.py (fake tokenizer/model, no
  GPU/Qwen/organs) exercising the REAL, unmodified `_generate_tokenid_continuation_skip` /
  `generate_with_generation_time_veto` / `clause_filter_sentence`, plus re-runs of the two pre-existing
  gen-time-honesty wiring/mechanism verifies (unmodified) and the string-filter safety net verify (unmodified)
  to confirm no-regression.
runner: research/runners/_open_ended_gen_time_skip_continue_wiring_verify.py
external: NO NEW external search logged this session -- the DR gate (`tools/gates/deep_research_at_wall.py`,
  lane `e-language-open-ended-honesty` now carries 4 findings within 3 days) is satisfied by the EXISTING
  in-window, lane-tagged record (the queue's `.external_searches.jsonl` log, entry timestamped 2026-08-28T13:47:13Z:
  "SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching" <!--derived--> arXiv:2504.00970
  <!--derived-->, logged by the 2026-08-28 token-id-continuation finding for the SAME token-space-continuation
  property this file's `context_ids` stream reuses verbatim). This file's own addition -- DROP a clause yet
  keep the raw generation context advancing so the mouth is never stuck re-emitting the identical unrepairable
  sentence forever under greedy decode, while excluding that span from the user-visible stream -- is an
  ORCHESTRATION change over that already-cited, already-GO token-id mechanism (no new biology, no new filter
  decision), not a fresh mechanism proposal requiring its own external grounding.
artifacts:
  - research/findings/raw/_open_ended_gen_time_skip_continue_wiring_verify.json
---
# Generation-time honesty: skip-and-continue past a dropped sentence reaches later, verifiable residuals instead of truncating the whole reply (PARTIAL -- wiring GO, live-mouth GPU measurement staged)

Artifact: research/findings/raw/_open_ended_gen_time_skip_continue_wiring_verify.json (28/28 checks GO on a
deterministic fake-tokenizer/fake-model battery exercising the REAL new orchestration code against the REAL,
unmodified `clause_filter_sentence` and `generate_with_generation_time_veto` dispatcher).

**One line.** Both the 2026-08-27 PARTIAL finding and the 2026-08-28 token-id-continuation finding named the
same NEXT rung and neither attempted it: "skip-and-continue past a dropped sentence to reach later same-reply
residuals." This file builds it -- `_generate_tokenid_continuation_skip` (new), selected via
`generate_with_generation_time_veto(..., continuation="token_id", skip_continue=True)` -- and proves the
orchestration correct with a CPU-only wiring battery, exactly the honesty-of-scope discipline the two parent
findings modeled: mechanism proven on a bounded, deterministic harness; the decisive, GPU-costly live-mouth
measurement staged, not claimed.

## Verify-first (before building)

`.venv-rag/bin/python tools/rag/rag_search.py "generation-time honesty skip and continue dropped sentence" 6
--corpus finding` and `bash tools/before_you_build.sh "skip-and-continue past a vetoed sentence in
generation-time honesty"` both surfaced only the two parent findings, each of which names this rung under its
own "NEXT" / "Honest scope" and explicitly disclaims having attempted it ("v1 also conservatively STOPS
generation on an unrepairable sentence ... rather than skip-and-continue past it" -- 2026-08-27 PARTIAL; "Remaining
NEXT items the parent finding also named and this session did not attempt: skip-and-continue past a dropped
sentence" -- 2026-08-28 token-id-continuation finding). No duplicate implementation exists in
`research/runners/` or `webapp/` (grepped for `skip.and.continue` / `skip_continue` / `SKIP_CONTINUE` before
writing any code).

## The mechanism

`research/runners/_open_ended_gen_time_consensus_veto_derisk.py` gains `_generate_tokenid_continuation_skip`,
selected by a new `skip_continue: bool = False` parameter on `generate_with_generation_time_veto` (default
False -- unchanged behavior; `continuation="token_id"` only, as directed -- the text-continuation path's
every-step retokenization makes hiding a dropped span from the model's own future context substantially harder
and is out of scope here). It carries TWO separate token-id streams between sentence-generation steps, instead
of the ONE `_generate_tokenid_continuation` already used:

  - `context_ids` -- the RAW generation context (every sentence the mouth generated, kept OR dropped) -- what
    the NEXT `model.generate()` call continues from, so the model's own autoregressive state always advances
    (a dropped sentence's own generated tokens still move the model's context forward, exactly as if it had
    said them, just never surfaced to the caller -- this is what prevents a greedy-decode infinite loop on the
    identical unrepairable sentence without needing sampling/temperature).
  - `accepted_ids` -- the FINAL, user-visible stream -- receives a KEPT sentence's own generated ids (zero
    retokenization, same property `_generate_tokenid_continuation` already has) or a REPAIRED sentence's
    re-encoded span, but a DROPPED sentence (`clause_filter_sentence` returns `None`) is skipped entirely:
    `context_ids` moves past it, `accepted_ids` does not.

Every candidate sentence -- kept, repaired, or dropped -- still runs through the exact same, unmodified
`clause_filter_sentence(candidate, topic, facts)` call the non-skip path uses; nothing here changes what counts
as a veto, only what happens next when one fires. The loop is still bounded by `max_sentences` (total
generation ATTEMPTS -- kept+repaired+dropped combined, the same accounting the non-skip path already uses) and
`max_new_tokens`, so a persistently-hallucinating mouth cannot loop unboundedly even though a drop no longer
stops the loop early.

Wired into `webapp/open_ended_chat.py` behind a THIRD, independent, default-OFF flag
(`BRAIN_HONESTY_SKIP_CONTINUE`, `skip_continue_enabled()`) stacked on top of `BRAIN_OPEN_ENDED` +
`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` -- all three must be truthy for anything to change. `answer_turn` passes
`skip_continue=skip_continue_enabled()` to `generate_with_generation_time_veto`; with the flag unset (default),
this evaluates to `False`, the SAME default the parameter itself carries, so the call is unchanged from before
this parameter existed. NO `sim/` edit; no change to `clause_filter_sentence` / `sentence_contradicts` /
`consensus_facts_for_topic` / the string safety net, which stays layered on top of whatever text this produces,
unconditionally, exactly as before.

## The adversarial fixture, and why it had to be a NEW one

The existing controlled unit battery's adversarial sentences (`ADVERSARIAL_SENTENCES`, e.g. "Canada is bordered
by the United States to the south and Mexico to the west.") are all **repairable** -- `clause_filter_sentence`
drops only the wrong list item and keeps the rest, so `repaired is not None` and the `dropped_stop` /
`dropped_skip` branches this file adds are never reached by them. Demonstrating skip-and-continue requires a
sentence `clause_filter_sentence` cannot repair at all (`candidate == original` after both repair attempts ->
returns `None`). This file uses "The capital of Canada is Toronto." with `facts=[("capital", "ottawa")]`:
`sentence_contradicts` flags "wrong capital: toronto", but `_bad_relation_tokens` only locates
border/continent spans (not capital), so no span is removed and the sentence is a FULL DROP -- verified by
assertion at module load (`assert clause_filter_sentence(S1, "canada", FACTS_ON) is None`) before any check is
built on top of it, so a fixture that silently repaired instead of dropping could not make every check below
vacuously trivial.

## Results

**Wiring sanity (decisive for what it claims, CPU-only, deterministic, fake tokenizer/model, no GPU/Qwen/
organs).** `_open_ended_gen_time_skip_continue_wiring_verify.py` (new): **28/28 checks GO**.

  - (A) `skip_continue=True`, veto firing (`facts=[("capital","ottawa")]`): the unrepairable first sentence is
    `dropped_skip` (not `dropped_stop`); the model is called a SECOND time (`model.calls == 2`); the later
    sentence ("It has ten provinces and three territories.") is generated, checked, and KEPT. **Accepted text:
    `"It has ten provinces and three territories."`** -- the vetoed content ("toronto") is absent, the later
    supported content is present.
  - (B) `skip_continue=False`, the SAME veto firing: the model is called ONCE; generation stops at the drop.
    **Accepted text: `""`** (empty) -- the later sentence is never even generated, let alone reached.
  - (C) THE LEVER: holding the veto firing constant, `skip_continue` is the ONLY varying term between (A) and
    (B); `tools.lab.lever` confirms it MOVES the final text (`"It has ten provinces and three territories."` ->
    `""`) -- attributed to `skip_continue` alone, not to anything else that happened to differ. (A) is vary ->
    differ; (B) is the lesion (`skip_continue` back to its pre-existing default) -> vanish -- the full
    vary/differ/lesion/vanish chain this project's anti-hollow discipline requires.
  - (D) BYTE-IDENTICAL WHEN OFF, proven by executed comparison, not by comment: `skip_continue=False` through
    the new dispatcher branch produces text and trace IDENTICAL (Python `==`, not "should be") to calling the
    pre-existing, completely untouched `_generate_tokenid_continuation` directly on the same script; and NOT
    passing `skip_continue` at all produces the identical result to passing `skip_continue=False` explicitly --
    the parameter's own default reproduces the pre-existing call signature's behavior exactly.
  - (E) sanity cross-check (not this file's own lever -- the underlying consensus-coupling lesion is already
    proven by the 2026-08-28 token-id-continuation finding): with `facts=[]` (nothing to suppress at all),
    `skip_continue` has ZERO effect either way -- the wrong capital survives regardless, confirming this
    extension only changes behavior WHEN the veto actually fires.
  - (F) a token budget smaller than the full script truncates cleanly under `skip_continue=True`, never raises,
    never exceeds `max_sentences`.
  - (G)/(H) `BRAIN_HONESTY_SKIP_CONTINUE` env-var reading, AND the full env-var -> `answer_turn` ->
    `skip_continue_enabled()` -> `generate_with_generation_time_veto(skip_continue=...)` routing, checked with
    the mechanism function stubbed (isolating the wiring from the mechanism (A)-(F) already proved): the exact
    `skip_continue` kwarg the mechanism receives is captured per turn and matches the env var precisely --
    unset -> `False`, `"0"` -> `False`, `"1"` -> `True` -- and the unset case is Python-`==` identical to the
    explicit-`False` case (`H_default_matches_explicit_false`).

**No-regression on the existing, unmodified gates (re-run this session, numpy backend).**
`_open_ended_gen_time_tokenid_continuation_wiring_verify.py` -- still **20/20 GO** (this file's addition does
not alter `_generate_tokenid_continuation`, `_continue_chunk_ids`, or `_find_sentence_boundary_ids`, all reused
verbatim). `_open_ended_gen_time_consensus_veto_wiring_verify.py` (the `answer_turn`/flag-routing check that
predates this session, using a `**kw`-accepting stub) -- still **6/6 GO**, confirming the new `skip_continue=`
kwarg addition to the `answer_turn` call site does not break the pre-existing flag-routing checks.
`_open_ended_clause_contradiction_filter_verify.py` (the string safety-net battery, unmodified) -- still **GO**
(10/10 catch rate, 0 leaks), confirming the safety net this mechanism layers on top of is untouched.

`git diff --stat` against `main`: 2 files touched (`research/runners/_open_ended_gen_time_consensus_veto_derisk.py`
+123/-6 lines, `webapp/open_ended_chat.py` +32/-1 lines), both additive (no existing line removed except two
one-line docstring insertions splitting a sentence to attach a cross-reference); one new file
(`research/runners/_open_ended_gen_time_skip_continue_wiring_verify.py`).

## Honest scope

**Not run this session, by design (CPU/numpy-only, no GPU touch -- a cupy run was already holding the GPU when
this session started).** The decisive live-mouth GPU comparison: does `skip_continue=True` on the REAL
off-bridge Qwen actually reach genuine, unfabricated later content across more topics/seeds than the
truncate-only path, and does it hold the safety net's 0-leak property under a live, non-scripted mouth (the
fake-model script here is deterministic and adversarial-by-construction; a real Qwen's own spontaneous
fabrications are the harder case the 2026-08-27 PARTIAL finding's live-mouth section already disclosed as
opportunistic, not decisive)? This needs a real Qwen load + generation across topics x lesion-states x seeds on
the GPU, exactly the same shape of staging the 2026-08-28 token-id-continuation finding used for its own
decisive measurement (`research/queue/gpu.queue`, 6 seeds 42/43/44/100/101/102) -- **this finding does not
claim that result.**

**`wired` but NOT `production-default`** (per `docs/TERMS.md`'s code-condition definitions): `skip_continue`'s
call path IS reachable from `webapp/server.py`'s `/api/brain-chat` (through the same, already-verified
`answer_turn(chat=chat)` call site the 2026-08-28 consensus-veto wiring verify's check (6) already covers,
`git diff` shows no change to that call site's own gating), but `BRAIN_HONESTY_SKIP_CONTINUE` is default-OFF
and stacked behind two OTHER default-OFF flags -- the default turn a real user gets is completely unaffected.
Calling this "closed" or "on-by-default" would be a `docs/TERMS.md` misuse; it is `wired (default-off)`.

**Scope limit shared with the parent mechanism, unchanged by this file:** relations checked remain
capital/continent/borders (matching the string filter's own structural scope); a bare unsupported number/date
is still caught by `sentence_contradicts`'s own facts-independent branch regardless of skip-vs-truncate,
unaffected by (and not attributed to) this file. `max_sentences` still bounds total ATTEMPTS (kept + repaired +
dropped combined) -- a reply that drops several sentences in a row consumes that budget faster than one that
never drops; this file does not add a separate "how many drops tolerated" budget, reusing the existing
parameter's semantics rather than adding a new tunable this session has no data to size.

NEXT: run the staged 6-seed cupy GPU verify (build it on the pattern the token-id-continuation finding
established) -- confirm skip-and-continue reaches genuinely MORE verified content per reply than the
truncate-only path on the real mouth, with no regression on the safety net's 0-leak property. If GO, this
extension is a candidate for promotion toward `on-by-default` for `BRAIN_HONESTY_SKIP_CONTINUE` specifically
(the other two stacked flags remain their own, separately-gated promotion decisions). Remaining item the parent
findings also named and this file did not attempt: broaden past capital/continent/borders toward the store's
fuller relation set. NO `sim/` edit.
