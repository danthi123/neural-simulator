---
type: finding
status: complete
claim_check: measured
date: 2026-09-04
mechanism: additive fix to `webapp/wkv_mouth_generator.py::generate()` (a new `trace: dict | None = None`
  out-parameter records which of its OWN two internal branches -- `render_fact_sentence`
  (SpikingClauseProducer) vs the genuine WKV/linattn free-gen spiking decode -- actually produced the returned
  text) plus `webapp/open_ended_chat.py::answer_turn` (reads that trace back and sets `generator`/
  `fact_clause_used`/`wkv_mouth_used` from the ACTUAL producer, gating the separate fact-clause-fallback block
  on a new `wkv_attempted` flag instead of the old `wkv_used` to avoid a second-order double-render bug).
seed-waiver: this is a DETERMINISTIC ROUTING/LABELLING fix (does the trace dict read back what a pure function
  of its own inputs just did), not a stochastic generalization claim -- the project's 6-seed policy targets
  ruling out a favorable-seed artifact in a LEARNED/statistical effect. The primary bug-reproduction case
  (`TestGeneratorLabelMatchesProducerViaAnswerTurn::test_broad_scope_covered_relation_traces_spiking_clause`,
  tests/test_generator_trace_matches_producer.py) IS run across all 6 non-negotiable seeds (42/43/44/100/101/
  102) and passes on every one; the remaining sub-cases (uncovered-relation routing, qwen fallback, the `trace`
  parameter's own mechanics) are seed-independent control-flow branches spot-checked at seed=42, mirroring
  `2026-09-03-affect-wiring-into-wkv-mouth-GO.md`'s own precedent for this class of fix.
lane: language (own-voice mouth / linattn production wiring) + one-brain (per-touchpoint Qwen-vs-substrate
  provenance instrumentation, de-risk #2 of research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md)
seeds: [42, 43, 44, 100, 101, 102]
verdict: fixed and verified. The `generator` trace field (plus `wkv_mouth_used`/`fact_clause_used`) now follows
  the ACTUAL producer of each open-ended reply, independent of `BRAIN_WKV_MOUTH_SCOPE` and independent of which
  of `answer_turn`'s two internal try-blocks reached it. Confirmed byte-identical reply CONTENT (`raw`/`answer`
  text unchanged, exact-string compared, not inferred) -- only the trace metadata changes. A 19-case regression
  test (tests/test_generator_trace_matches_producer.py) passes with the fix and, checked directly by reverting
  the two source files to HEAD and re-running the identical suite, fails on 12 of the 19 cases without it.
artifacts:
  - research/findings/raw/_generator_trace_mislabel_fix_verify.json
  - research/runners/_generator_trace_mislabel_fix_verify.py
  - webapp/wkv_mouth_generator.py
  - webapp/open_ended_chat.py
  - tests/test_generator_trace_matches_producer.py
  - research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md
  - research/runners/_wkv_fact_to_sentence_lexicon_lever.py
---

# Generator-trace mislabel fix — `answer_turn`'s `generator` field now follows the actual producer

## Summary

The 2026-09-03 linattn live verification found that under `BRAIN_WKV_MOUTH_SCOPE=broad`, open-ended replies
genuinely produced by the fact-clause-first routing (`webapp.wkv_mouth_generator.render_fact_sentence`, the
already-6-seed-GO `SpikingClauseProducer`) were mislabeled in the `generator` trace field as `"wkv_mouth"`
instead of `"spiking_clause"`. This corrupts per-touchpoint provenance/audits -- specifically it blocked the
one-brain roadmap's planned instrumentation of the per-touchpoint Qwen-vs-substrate call share (de-risk #2,
`research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md` SS3), which depends on this label being
correct. This finding fixes the mislabel, additively and without changing reply content, and adds a regression
test that pins the corrected routing across both `BRAIN_WKV_MOUTH_SCOPE` settings and all three generator paths
(`spiking_clause` / `wkv_mouth` / `qwen`).

## Root cause

`webapp/open_ended_chat.py::answer_turn` tries the WKV mouth first, inside its own `try` block (pre-fix,
~L595-632):

```python
if wkv_mouth_enabled():
    try:
        if _WKV.in_vocab_scope(msg, seed=seed):
            sentence_facts = facts if (known and wkv_fact_sentence_enabled()) else None
            raw, secs = _WKV.generate(msg, ..., sentence_facts=sentence_facts, ...)
            wkv_used = True
            generator_name = "wkv_mouth"          # <- set UNCONDITIONALLY once generate() returns
    except Exception:
        wkv_used = False
```

But `webapp/wkv_mouth_generator.py::generate()` has TWO internal ways to produce `raw` (pre-fix, ~L973-989):

```python
def _run():
    if sentence_facts:
        sentence = render_fact_sentence(sentence_facts, seed=seed)   # SpikingClauseProducer, lexicon-covered
        if sentence is not None:
            return sentence                        # <- can ALREADY be the fact-clause producer's output
    ro, _vocab, word_to_id = _get_readout(seed)      # genuine WKV/linattn free-gen spiking decode
    ...
    return text
```

`answer_turn` labelled `raw` `"wkv_mouth"` purely because it was returned from inside its own WKV try-block --
never checking WHICH of `generate()`'s two branches actually wrote it. When `sentence_facts` names a known
topic whose relation `RELATION_LEXICON` covers (`research/runners/_wkv_fact_to_sentence_lexicon_lever.py`,
34/34 live-relation coverage on the shipped store), `render_fact_sentence` -- the SAME mechanism the SEPARATE
"fact-clause fallback" block further down in `answer_turn` also wires in -- silently becomes the producer, but
the label stays `"wkv_mouth"`.

**Why `BRAIN_WKV_MOUTH_SCOPE=broad` is where this was found, but not the whole story.** `in_vocab_scope`'s
default (`scope='vocab'`) is a narrow TinyStories word-overlap gate; most real known-topic messages (Wikidata
slugs) fail it, so the WKV try-block is usually skipped and the SEPARATE, already-correct fact-clause-fallback
block below (gated `if not wkv_used and known and fact_clause_fallback_enabled()`) is what actually renders --
correctly labelled `"spiking_clause"`. `scope='broad'` makes `in_vocab_scope` return `True` unconditionally
(`wkv_mouth_generator.py`'s own documented, disclosed honest gap -- a real coverage threshold is
not-yet-measured, see `scope_mode()`'s docstring), so the WKV try-block now ALWAYS fires first when the WKV
mouth is enabled (default-ON under the open-ended channel), starving the outer fallback block of ever running
and exposing the mislabel on nearly every known-topic reply. The identical bug also fires under the DEFAULT
`scope='vocab'` whenever a message happens to pass the narrow word-overlap gate AND names a covered relation --
the ~3% real-traffic case `wkv_fact_sentence_enabled()`'s own docstring already named as a residual (measured
2026-09-01) but had not connected to the trace label specifically. Both cases are covered by the fix and the
test below.

## The fix

**1. `webapp/wkv_mouth_generator.py::generate()`** gains an additive `trace: dict | None = None` out-parameter
(default `None`, every pre-existing call site unaffected). `_run()` now records which branch fired:

```python
def _run():
    if sentence_facts:
        sentence = render_fact_sentence(sentence_facts, seed=seed)
        if sentence is not None:
            if trace is not None:
                trace["sentence_fact_used"] = True
            return sentence
    if trace is not None:
        trace["sentence_fact_used"] = False
    ...   # genuine free-gen, unchanged
```

**2. `webapp/open_ended_chat.py::answer_turn`** passes a fresh dict and reads it back to set the label from the
actual producer, instead of from which try-block reached it:

```python
wkv_trace: dict = {}
raw, secs = _WKV.generate(msg, ..., sentence_facts=sentence_facts, ..., trace=wkv_trace)
wkv_attempted = True
if wkv_trace.get("sentence_fact_used"):
    fact_clause_used = True
    generator_name = "spiking_clause"
else:
    wkv_used = True
    generator_name = "wkv_mouth"
```

**3. Second-order bug avoided.** Simply changing the label above (without anything else) would leave the outer
fact-clause-fallback block's guard as `if not wkv_used and known and ...` -- and since the corrected `wkv_used`
is now `False` in the mislabeled case, that guard would fire a SECOND TIME and re-render the SAME fact through
`render_fact_sentence` again. The fix introduces a separate `wkv_attempted` flag (True whenever `_WKV.generate()`
returned `raw` by EITHER of its own mechanisms) and gates the fallback on `not wkv_attempted` instead -- so a
`sentence_facts` hit inside the WKV try-block is never re-rendered by the outer block.

Both changes are additive/guarded: `trace=None` (every pre-existing call site) is an exact no-op inside
`generate()` (both `if trace is not None:` guards never fire), and `answer_turn`'s own reply-producing calls
(`_WKV.generate(...)`, `render_fact_sentence(...)`, `gen.generate(...)`) are byte-for-byte unchanged -- only
which local variables get set from the (new) `wkv_trace` dict differs.

## Verification

**End-to-end through the real `answer_turn`/`generate()` entry points**, no mocked mechanism:
`research/runners/_generator_trace_mislabel_fix_verify.py` (run via
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._generator_trace_mislabel_fix_verify`) exercises all 7
scenarios below and writes `research/findings/raw/_generator_trace_mislabel_fix_verify.json` -- `verdict: GO`,
every one of the 7 `Verdict.require(...)` preconditions holds, including the 6/6-seed check on the primary bug
scenario.

**Byte-identical reply content**, confirmed by exact string comparison (not inferred from reading the code, per
`docs/TERMS.md`'s "byte-identical" condition):
  - `wkv_mouth_generator.generate()` called with `trace=None` / `trace={}` / `trace` omitted, same seed/prompt/
    `sentence_facts`, returns the IDENTICAL text in all three cases
    (`TestGenerateTraceParameterDirect::test_trace_parameter_never_changes_returned_text`).
  - `answer_turn()` called on the SAME known+covered-relation turn under `scope='vocab'` (reaches the outer
    fallback) vs `scope='broad'` (reaches the SAME mechanism from inside the WKV try-block) returns IDENTICAL
    `raw`/`answer` text either way -- only the internal route and (now, correctly) the shared `"spiking_clause"`
    label differ (`TestByteIdenticalReplyContent`).
  - The rendered clause matches the independently-reconstructed `expected_surface(...)` from
    `research/runners/_wkv_fact_to_sentence_lexicon_lever.py` exactly, on all 6 seeds.

**The regression test is a genuine pin, not vacuous.** `tests/test_generator_trace_matches_producer.py` (19
cases) passes in full with the fix applied. Reverting ONLY the two source files to HEAD (`git checkout --
webapp/open_ended_chat.py webapp/wkv_mouth_generator.py`, test file left in place) and re-running the identical
`pytest` invocation fails 12 of the 19 cases -- every case that exercises the bug scenario or the new `trace`
parameter -- while the 7 cases that test pre-existing-already-correct behavior (the vocab-scope out-of-vocab
fallback, genuine free-gen on an uncovered relation, the qwen fallback, `trace=None` backward-compatibility)
continue to pass either way, exactly as expected for a correctly-targeted fix.

**No regressions in the surrounding suite.** The broader existing WKV-mouth / open-ended test files (93 tests
across `test_wkv_mouth_learned_head_path.py`, `test_wkv_mouth_bpe_decode_wiring.py`,
`test_wkv_invocab_scope_leadin_fix.py`, `test_wkv_fact_svo_clause_first_lever.py`, `test_wkv_onebridge_merged.py`,
`test_wkv_readout_multilayer.py`, `test_wkv_spiking_forward.py`, `test_grounded_wkv_renderer.py`,
`test_open_ended_generation_fluent.py`) were run before and after this fix. The SAME 6 tests fail identically
in both cases (`test_wkv_invocab_scope_leadin_fix.py`'s two multiword-content tests,
`test_wkv_readout_multilayer.py::TestMultiLayerNumericalCorrectness::test_state_dict_key_layout_matches_
documented_contract`, `test_grounded_wkv_renderer.py::test_fluidchat_wkv_grounded_and_gatefirst_moat`, and
`test_open_ended_generation_fluent.py`'s two `render_hypothesis_*` tests) -- confirmed by reverting the two
source files to HEAD and re-running the EXACT SAME 9-file command, which reproduced the identical 6-failed/
87-passed/1-skipped result. These are pre-existing, order/state-dependent failures in code this fix never
touches (`in_vocab_scope`, the multilayer readout, the grounded renderer, `brain_chat_tui.py`'s
`render_hypothesis_verified`) -- unrelated to this change, not investigated further here.

Commands run (`CUDA_VISIBLE_DEVICES=""` to force CPU -- the GPU was at 100% utilization/16GB used for an
unrelated job at verification time):

```bash
CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_generator_trace_matches_producer.py -q
# 19 passed

CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_wkv_mouth_learned_head_path.py \
  tests/test_wkv_mouth_bpe_decode_wiring.py tests/test_wkv_invocab_scope_leadin_fix.py \
  tests/test_wkv_fact_svo_clause_first_lever.py tests/test_wkv_onebridge_merged.py \
  tests/test_wkv_readout_multilayer.py tests/test_wkv_spiking_forward.py tests/test_grounded_wkv_renderer.py \
  tests/test_open_ended_generation_fluent.py -q
# 6 failed, 87 passed, 1 skipped -- IDENTICAL with and without this fix (pre-existing, unrelated)
```

## What this does NOT do (honest scope)

- Does not distinguish `"wkv_mouth"` produced by the `ssm` vs `linattn` recurrence family -- both still share the
  one `"wkv_mouth"` label (the `trace["sentence_fact_used"]` check happens before `recurrence_mode()` is even
  read, so the fix is recurrence-family-agnostic by construction; a real `--recurrence linattn --save-ssm`
  checkpoint is not present in this worktree's `bridges/wkv_ckpt/` to exercise end-to-end, so this fix is
  verified against the shipped `ssm` checkpoint only -- the mechanism itself has no recurrence-family dependency).
- Does not itself execute the roadmap's de-risk #2 (instrumenting the LIVE per-touchpoint Qwen-call share over
  real traffic) -- it removes the blocker (the label was wrong) so that instrumentation can now be built on top
  of a trustworthy `generator` field.
- Does not touch reply CONTENT, the moat/post-filter, retrieval, or any of the other trace fields (`topic`,
  `known`, `facts`, `state`, `gen_seconds`, `gen_time_honesty_used`, `gen_time_trace`) -- confirmed unchanged by
  the byte-identical checks above.

## Provenance

Shipped code read this session (2026-09-04): `webapp/open_ended_chat.py::answer_turn` (the full WKV-mouth +
fact-clause-fallback dispatch, pre-fix ~L593-683), `webapp/wkv_mouth_generator.py::generate`/`_run` (pre-fix
~L896-992), `research/runners/_wkv_fact_to_sentence_lexicon_lever.py` (`RELATION_LEXICON`, `expected_surface`,
`pick_covered_fact`, `_dctx_and_slots`). Bug first reported by the 2026-09-03 linattn live verification and
named in `research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md` SS3 de-risk #2. Reproduced
directly (pre-fix: a known topic with a `RELATION_LEXICON`-covered relation under `BRAIN_WKV_MOUTH_SCOPE=broad`
traced `generator="wkv_mouth"`/`wkv_mouth_used=True`/`fact_clause_used=False` while `raw` exactly equalled
`render_fact_sentence`'s own output) before writing the fix, and re-verified after.
