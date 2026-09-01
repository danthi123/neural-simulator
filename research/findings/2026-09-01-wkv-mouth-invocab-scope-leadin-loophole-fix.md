---
type: finding
status: fixed
date: 2026-09-01
mechanism: closes the `in_vocab_scope` lead-in-phrase loophole named as board #112's next rung 1 (a
  content-hit exclusion set, `_LEADIN_WORDS`, added to `webapp/wkv_mouth_generator.py`)
verdict: GO (this runner's own verdict, per `tools.verdict.Verdict`) — direct adversarial catch 21/21 (every
  reproduced pre-fix false positive is now caught, 0 leaking), no regression on genuine multi-content-word
  messages (0/5 regressed, with and without a lead-in prefix), and a real-store stratified measurement (n=600,
  same sample Part 2 of the fact-grounding finding used) shows the loophole's own signature collapsing exactly
  where predicted: the "zero genuine topic content" bucket (n=487, 81.2% of the sample) drops from 63.45%
  wrongly passing to 0.0%, while the "genuinely >=2 content words in the topic itself" bucket (n=24) stays at
  100% passing before and after — the fix removes ONLY the loophole's contribution, not genuine content's.
lane: e-mouth-fluency / A1 (crutch-burndown), board #112, rung 1 of the two named next steps
seeds: [42]
seed-waiver: single-seed is sufficient here — `in_vocab_scope` is a DETERMINISTIC pure function of `text` and
  the checkpoint's fixed vocabulary (no RNG draw of its own); the `seed` parameter only selects WHICH
  checkpoint's `.npz` vocabulary to test against, so seed 42 exercises the identical code path every other
  seed's checkpoint would (a different vocabulary, same logic). No claim in this finding varies by generation
  seed (nothing here calls `generate()`); the multi-seed discipline in CLAUDE.md targets stochastic-outcome
  claims, which this is not. The 6-seed fact-grounding *lever* this rung sits downstream of already carries its
  own 6-seed validation (`2026-09-01-wkv-mouth-fact-grounding-lever.md`).
instrument: research/runners/_wkv_invocab_scope_leadin_fix_verify.py — direct measurement against the real
  shipped `wikidata_core_15k` facts.json (600-agent stratified sample, seed 42) plus a synthetic adversarial +
  no-regression battery; tests/test_wkv_invocab_scope_leadin_fix.py — the cheap, always-run regression pin
  (19 tests, all pass in 0.48s).
runner: research/runners/_wkv_invocab_scope_leadin_fix_verify.py (--seed 42, the only seed needed per the
  seed-waiver above)
external: NO-EXTERNAL-NEEDED — this is a pure bug fix to an existing, already-cited decode-scope gate
  (`in_vocab_scope`, whose own design already cites the 2026-08-28 adversarial verify-go pass that found the
  ORIGINAL function-word-only loophole); this rung closes a second instance of the SAME class of bug the same
  gate was built to prevent, using the same technique (widen the excluded-word set), not a new external method.
artifacts:
  - research/findings/raw/_wkv_invocab_scope_leadin_fix_verify.json (the full stratified + adversarial + no-
    regression measurement this finding cites)
  - webapp/wkv_mouth_generator.py (the fix: `_LEADIN_WORDS` + `in_vocab_scope`'s content_hits exclusion)
  - tests/test_wkv_invocab_scope_leadin_fix.py (the committed regression pin)
  - research/FAILURE_LOG.md (the 2026-09-01 entry this rung closes, updated with a FIXED note)
  - research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md (the finding that found and logged the
    loophole; this rung is its Next-steps item 1)
---

# Closing the `in_vocab_scope` lead-in-phrase loophole (board #112, rung 1)

## 0. What this is

`research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md` Part 2 found and logged a real bug (not fixed
there, scoped out): `webapp.wkv_mouth_generator.in_vocab_scope("tell me about " + <anything>)` returned `True`
even for total nonsense, because the lead-in words `"tell"`/`"me"`/`"about"` sit in this checkpoint's V=1000
vocabulary AND are not in `_FUNCTION_WORDS`, so the fixed lead-in phrase alone satisfied `min_content_hits=2`
regardless of what the actual topic was. Measured on a real 600-agent sample of `wikidata_core_15k` topics,
this made the gate pass 68.17% of the time — far above what genuine TinyStories-domain overlap predicts. This
finding is that fix, plus the measurement proving it closes the loophole without silently narrowing genuine
in-vocab traffic.

## 1. The fix

`webapp/wkv_mouth_generator.py` gains one new module-level constant, `_LEADIN_WORDS` — every word appearing
across `webapp.open_ended_chat._LEADINS`'s lead-in phrases ("tell me about", "what do you know about", "who
is", "describe", "explain", ...), duplicated locally as a literal so this module stays import-independent of
`open_ended_chat` (the same discipline `_FUNCTION_WORDS` already follows — no cross-module coupling). `in_vocab_
scope`'s `content_hits` line changes from

```python
content_hits = [w for w in hits if w not in _FUNCTION_WORDS]
```

to

```python
content_hits = [w for w in hits if w not in _FUNCTION_WORDS and w not in _LEADIN_WORDS]
```

`min_hits` and `min_frac` are UNCHANGED — still scored over the full original message, not a stripped
remainder — so a genuinely content-bearing message is never penalized just because it also happens to start
with a recognized lead-in phrase. Only which hits are credited as CONTENT changes. `fact_grounding_ids` (a
different function, used to pull content words OUT of retrieved facts rather than to gate a user message) is
untouched — it still excludes only `_FUNCTION_WORDS`, verified by `test_fact_grounding_ids_unaffected_by_the_
leadin_exclusion`.

Every number in SS2-SS4 below is read directly from
`research/findings/raw/_wkv_invocab_scope_leadin_fix_verify.json`, produced by
`research/runners/_wkv_invocab_scope_leadin_fix_verify.py`.

## 2. Verification — direct adversarial catch

`research/runners/_wkv_invocab_scope_leadin_fix_verify.py` Part A re-tests 12 `_LEADINS` phrases x 3 nonsense
tails (36 cases). 21 of the 36 were pre-fix false positives (the rest already failed the gate's other
conditions, e.g. shorter phrases falling below `min_frac`); **all 21 are now caught, 0 leaking.** The exact
case named in the original finding (`"tell me about zzznonsenseword qqqgibberish"`) now reads `False`, was
`True`.

## 3. Verification — no regression on genuine content

Part B re-tests 5 real TinyStories-register sentences (adapted from `_wkv_learned_vs_native_head_ab.py`'s own
independently-verified 8-prompt battery), both with and without a recognized lead-in prefix. **0/5 regressed**
— every genuinely content-bearing message, lead-in or not, still passes.

## 4. Verification — stratified real-store measurement (the load-bearing number)

Part C reproduces the fact-grounding finding's exact Part 2 sample (600 real `wikidata_core_15k` agents, seed
42, `"tell me about " + agent`), then buckets each agent by how many genuine in-vocab content words its OWN
slug carries (independent of any lead-in — the ground-truth signal this fix is supposed to track):

| Bucket | n | old pass frac | new pass frac | reading |
|---|---|---|---|---|
| 0 genuine content words | 487 (81.2%) | 63.45% | **0.00%** | the pure loophole population — CATCH RATE |
| 1 genuine content word | 89 (14.8%) | 85.39% | 0.00% | honest residual, see SS5 |
| 2+ genuine content words | 24 (4.0%) | 100.00% | **100.00%** | genuinely in-scope — NO REGRESSION |
| **Overall** | **600** | **68.17%** | **4.00%** | matches Part 1's own ~4-6% conditional ceiling |

(The 68.17% old-overall figure reproduces the fact-grounding finding's Part 2 number exactly, from an
independent reimplementation of the pre-fix code — confirming this measurement is isolating the same
population, not a different one.) The new overall pass rate (4.00%, 24/600) equals the bucket-2+ count exactly
— after the fix, `in_vocab_scope` passes if and only if the topic itself carries genuine content, with zero
contribution from the lead-in. This is the intended, designed behavior.

## 5. Honest residual — the 1-content-word bucket, named not hidden

Bucket 1 (89/600 agents whose topic carries exactly ONE genuine in-vocab content word) also drops to 0% passing
post-fix. This is **not a new regression against correct behavior** — a single content word can never reach
`min_content_hits=2` on its own, with or without a lead-in; these cases were ALSO only ever passing via the
loophole (the lead-in propping the count up to 2), never via genuine 2-word support. `test_single_content_word_
topic_is_an_honest_residual_not_silently_hidden` pins this explicitly. Whether the gate's `min_content_hits=2`
floor is the right bar for a bare single-word topic (vs., say, `min_content_hits=1` when the message IS just a
topic phrase with no other content) is a separate design question, out of THIS rung's scope — named here as a
candidate follow-up, not built.

## 6. Downstream effect on the (separately, already-merged) fact-grounding lever

`in_vocab_scope` is also the gate `answer_turn` uses to decide whether the WKV mouth engages AT ALL (independent
of the `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND` flag), and it is reused inside `_wkv_mouth_fact_grounding_
derisk.py`'s OWN Part 2/3 demo-selection. Because that gate now genuinely reflects in-vocab content, re-running
`_wkv_mouth_fact_grounding_derisk.py --seed 42` after this fix draws its demo agents from a much smaller,
differently-composed pool (topics whose names are themselves ordinary words, e.g. `tommy_boy_music`, `football_
world_cup_1954`, `leaf_cutter_bee`) than the original 6-seed run did. A spot re-run at seed 42 post-fix reads a
`control` tie (`treatment=5 control=5`, `UNDEFINED`) rather than the original run's clean 7/7 boost — because
this NEW population's topic slugs already overlap heavily with ordinary high-frequency TinyStories words the
BASELINE (unboosted) generation surfaces on its own, raising baseline's own chance-hit rate on exactly this
subpopulation (the SAME "demo-selection favors words with more grounding, which also raises baseline's own
chance-overlap odds" dynamic the original finding's own SS3 already named for its seed-102 tie). **This does not
retract the fact-grounding finding** — that finding's 6-seed GO is a frozen, valid record of what was measured
against the code as it existed then, and its own artifacts are unchanged. It DOES mean a fresh re-validation of
the fact-grounding lever's real-traffic surfacing rate, sampled through THIS fixed gate, is a genuine next step
(not attempted here — out of this rung's scope, named as a candidate follow-up alongside SS5's residual).

## 7. What this is, and is not

**Is:** a real, verified, additive fix to an already-default-ON (since 2026-08-30) gate function, closing a
logged `research/FAILURE_LOG.md` failure mode with a mechanical regression test (`tests/test_wkv_invocab_scope_
leadin_fix.py`, 19/19 pass) plus a real-data stratified measurement proving both the catch and the no-regression
claim on the actual shipped store, not a synthetic proxy. `BRAIN_OPEN_ENDED` and `BRAIN_OPEN_ENDED_WKV_MOUTH`
both stay default-OFF, so this fix has zero effect on current production defaults — it only changes behavior
for the experimental opt-in open-ended WKV-mouth channel, making that channel's own documented scope claim
("in-vocab-only") actually true.

**Is not:** a closure of board #112 (rung 2, the word-vs-sentence residual, is untouched by this rung) or a
re-validation of the fact-grounding lever's already-merged 6-seed GO (SS6 names that as a follow-up, not done
here).
