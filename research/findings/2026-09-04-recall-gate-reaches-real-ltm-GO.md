---
type: finding
status: go
date: 2026-09-04
mechanism: (1) a direct-recall topic-extraction fallback onto the routed LTM shard
  (`RichAnswerComposer._direct_fact`, research/runners/rich_answer_composer.py:369-411, gated
  `BRAIN_DIRECT_LTM_TOPIC_FALLBACK`, default ON), reusing Surface B's own `webapp.open_ended_chat.extract_topic`
  lead-in-stripping plus the already-6-seed-GO buffer+LTM read `_facts_about`/`_with_ltm`/`_ltm_facts_about`
  the elaboration/chain paths already use (2026-08-28); (2) an underscore-preserving fix to the claim-level
  VERIFY pipeline's own prose tokenizer (`ClaimEntailmentVerifier._clauses`/`_noun_is_before`,
  research/runners/_moat_claim_entailment_derisk.py:226-260,436-450; `_extract_svo_from_prose`,
  research/runners/_grounded_lang_integration_derisk.py:120-141), found DURING this session's own verification
  of (1) -- together closing the root cause research/findings/2026-09-04-per-touchpoint-qwen-call-share.md
  (commit 64fc4d5f) diagnosed -- the production-default recall surface never reached real LTM content on
  natural phrasing.
lane: one-brain (substrate consolidation) + language (own-voice mouth / retire the Qwen scaffold)
seeds: [42]
seed-waiver: a real-traffic measurement soak through the REAL `/api/brain-chat` entry point
  (`webapp.server.brain_chat`, in-process), reusing the root-cause finding's own instrument VERBATIM (commit
  64fc4d5f), not a stochastic training run. `seed=42` draws a reproducible sample of known topics from the
  live store's agent pool -- confirmed IDENTICAL to the root-cause finding's own sample (`angora_turkey`,
  `college_for_interdisciplinary_studies`, `imperial_roman`, `l_quipe_de_france`) -- and is the fixed internal
  generation seed every real turn already uses; the STUB renderer's `render_svo` and the substrate's
  `BridgeParser.parse` used for VERIFY are deterministic given a fixed seed, so a 6-seed repeat of THIS
  before/after comparison would reproduce the identical pass/fail pattern for the identical topics, not
  additional evidence -- exactly the precedent the root-cause finding's own seed-waiver already established for
  this instrument. The SEPARATE claim-entailment-verifier regression check below IS run at the project's
  standard 6 seeds (42, 43, 44, 100, 101, 102), because that check is the module's own pre-existing adversarial
  self-test suite, not a real-traffic sample.
instrument: research/runners/_per_touchpoint_qwen_share_measure.py --phase shipped_default (Surface A, the
  production default, `BRAIN_OPEN_ENDED` unset), REUSED VERBATIM from commit 64fc4d5f (byte-identical, see
  Provenance), CPU-forced (`CUDA_VISIBLE_DEVICES=""`, `SIM_BACKEND=numpy`); plus
  research/runners/_moat_claim_entailment_derisk.py's own pre-existing 6-seed adversarial leak/false-reject
  regression suite, run before and after the tokenizer fix.
runner: research/runners._per_touchpoint_qwen_share_measure ; research/runners._moat_claim_entailment_derisk
external: NO-EXTERNAL-NEEDED -- a repo-internal fix to this repo's own already-shipped mechanisms, re-measured
  through this repo's own real-traffic instrument.
verdict: GO. All 4/4 known-topic natural-phrasing probes that abstained in the root-cause finding's BEFORE
  baseline (0/4 reached any answer) now return a genuine, multi-fact, VERIFIED, grounded answer sourced from
  the routed LTM shard (4/4 AFTER) -- a real recall-reachability fix, not a proxy improvement. Every other
  probe class (unknown/dangerous/open-ended/greeting, 10/16 rows) is answer-text BYTE-IDENTICAL before vs
  after -- the no-confab moat is unaffected, confirmed both by this real-traffic diff and by the claim-entailment
  verifier's own adversarial 6-seed self-test remaining GO (leak_rate=0.0, false_reject_rate=0.0) across both
  fixes. Two disclosed residuals (SS6): a topic-extraction phrasing-coverage gap (unrelated to this fix, present
  before and after) and a follow-up topic-carry-forward interaction the fix newly makes visible.
artifacts:
  - research/findings/raw/_per_touchpoint_qwen_share_shipped_default_BEFORE_recall_gate_fix.json (pre-fix,
    verbatim copy of commit 64fc4d5f's own like-named upstream artifact (same basename, no
    `_BEFORE_recall_gate_fix` suffix, at that commit only -- see Provenance);
    complete: true, 16/16 rows, peak RSS 1067.3 MB)
  - research/findings/raw/_per_touchpoint_qwen_share_shipped_default_AFTER_recall_gate_fix.json (post-fix, this
    session; complete: true, 16/16 rows, peak RSS 1485.1 MB, wall 369.8 s)
  - research/findings/raw/_moat_claim_entailment_underscore_fix_regression.json (this session's re-run of the
    claim-entailment verifier's own 6-seed adversarial self-test AFTER the tokenizer fix: GO=true,
    leak_rate=0.0, false_reject_rate=0.0, core=10/10 and hyp=3/3 on every seed)
---

# The direct-recall gate now reaches the real LTM shard on natural phrasing -- GO, 4/4 known-topic probes fixed

**One-brain-wiring de-risk #2, closed.** research/findings/2026-09-04-per-touchpoint-qwen-call-share.md (commit
64fc4d5f) measured that on the production-default reply surface, a natural "Tell me about X" / "What is X?"
question about a REAL entity the brain's own routed LTM shard holds abstained on every phrasing tried, 4/4,
because the gate's host-router fallback searches a construction-time snapshot that structurally never contains
the LTM shard's content. This finding closes that gap and reports the real-traffic before/after.

## 1. The exact snapshot-before-attach site (root cause, confirmed by reading the code)

`ChatBrain.gate()`'s host-router fallback (`_gate_router_combine`, research/runners/brain_chat_tui.py:702-724)
matches a question against `self.stored_facts` -- a plain Python list comprehension built ONCE at
`ChatBrain.__init__` -> `_refresh_facts()` (research/runners/brain_chat_tui.py:655-663):

```python
def _refresh_facts(self):
    comp = self.inner.composer
    self.stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in comp.kb
                         if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
```

By the time this runs, `webapp/server.py::_build_chat_brain` has ALREADY wrapped `comp` in a `TieredFactStore`
pointed at the routed LTM shard for the tiny-demo path (`_tiny_inner.composer = TieredFactStore(_tiny_inner.
composer, _tiny_ltm)`, webapp/server.py:3857, executed before `ChatBrain(agent, ...)` is constructed at
:3897) -- so the ordering intuition in the root-cause finding's own title ("snapshot ... before the shard
attaches") is not strictly a TIMING bug. The real mechanism is structural: `comp.kb` on a `TieredFactStore`
does not resolve to a merged view. `TieredFactStore` defines no `kb` of its own, so Python's attribute lookup
falls through to `__getattr__` (research/runners/tiered_fact_store.py:273-275):

```python
def __getattr__(self, name):
    # only reached for attributes NOT defined on the class/instance; delegate to the buffer composer.
    return getattr(object.__getattribute__(self, "buffer"), name)
```

`comp.kb` therefore ALWAYS resolves to `self.buffer.kb` -- the small conversational-buffer tier only, regardless
of when `_refresh_facts()` runs relative to the LTM attach. The cortical LTM shard (`ShardedPhasorStore`) is
addressed by HASHING a concept string to ONE shard (`ltm.shard_for(concept)`); it has no flat `.kb` a snapshot
could ever have captured. `self.stored_facts`/`self.agents_set` therefore never contain an LTM entity,
structurally, by construction -- not because of a race with the attach.

The only OTHER path into `gate()`, `_substrate_recall` (research/runners/brain_chat_tui.py:1240), requires the
NEURAL BridgeParser to extract a literal `(agent, action)` pair from the question -- "Tell me about X" / "What
is X?" (no verb) cannot produce one, so it returns `None` and falls through to the equally-blind host router.
Confirmed directly by the root-cause finding's real-traffic measurement: 4/4 known_factual probes abstained
across 4 different natural phrasings, every one via a curiosity-flavored non-answer, never reaching the
render fork at all (`research/findings/raw/_per_touchpoint_qwen_share_shipped_default_BEFORE_recall_gate_fix.json`).

## 2. The fix, part 1 -- direct recall reaches the LTM shard by topic (the requested fix)

`RichAnswerComposer._direct_fact()` (research/runners/rich_answer_composer.py:369-411) now tries ONE more
thing after `chat.gate(question)` itself abstains: extract the bare topic entity the SAME way Surface B does
(`webapp.open_ended_chat.extract_topic`, lead-in stripping -- `"Tell me about angora_turkey."` ->
`"angora_turkey"`), then look that CONCEPT up via `self._facts_about(topic)` -- the ALREADY-SHIPPED,
already-default-ON-since-2026-09-01 buffer+LTM read the elaboration/chain paths already use (`_with_ltm` ->
`_ltm_facts_about`, which routes the concept to its ONE shard and scans that shard's own `kb`, with the
store's alias-hop fallback) -- NOT the frozen `stored_facts` snapshot:

```python
direct = self.chat.gate(question)             # [a, v, p] or None
if direct is not None:
    return direct
if _direct_ltm_topic_fallback_enabled():
    try:
        from webapp.open_ended_chat import extract_topic
        topic = extract_topic(question)
        if topic:
            facts = self._facts_about(topic)
            if facts:
                return list(facts[0])          # a genuine stored fact -- render_paragraph VERIFIES it too
    except Exception:
        pass                                   # never let this fallback crash a turn -- degrade to abstain
return None
```

A hit is a genuine stored fact (moat-safe by construction -- drawn straight from a shard the brain genuinely
holds), returned as an ordinary `[a, v, p]` so it flows through the EXACT SAME chain/elaboration/render/VERIFY
pipeline every other gathered fact does -- it changes nothing about how a fact is spoken, only how a fact is
FOUND. A miss returns exactly the pre-fix `None` -> the honest abstain, unchanged. This only ever ADDS a way to
succeed where `gate()` already gave up; it never overrides an open-ended hypothesis, in-loop teaching, anaphora
resolution, or a host-router match, all of which still take precedence unchanged (they are checked first,
inside `gate()` itself). Guarded by `BRAIN_DIRECT_LTM_TOPIC_FALLBACK` (default ON); `=0` reverts to
byte-identical pre-fix behavior.

## 3. The fix, part 2 -- found DURING verification: the render/verify pipeline mangled underscored slugs

Turning the gate on exposed a SECOND, previously-invisible gap. The first after-fix run showed the direct fact
was now genuinely GATHERED for every known_factual topic (`spiking_miss` counts of 3-5, where BEFORE they were
0 -- the render fork was never even reached) -- but 3 of 4 topics still ended up abstaining, because every
gathered sentence FAILED the per-sentence VERIFY and was silently dropped
(`RichAnswerComposer.render_paragraph`'s `if not kept: ... abstain`).

**Root cause, confirmed with a direct probe of the live pipeline** (not inferred from reading alone -- see
Provenance): the CLAIM-LEVEL entailment verifier's own prose tokenizer (`ClaimEntailmentVerifier._clauses`,
research/runners/_moat_claim_entailment_derisk.py:226, plus `_noun_is_before`:436) stripped ALL non-alphabetic
characters from each rendered word -- `re.sub(r"[^a-z]", "", w)` -- including the underscore that separates a
Wikidata-style multi-word slug: `'angora_turkey'` -> `'angoraturkey'`, the inflected relation
`'located_in_time_zones'` -> `'locatedintimezones'`. `self.nouns`/`self.verbs` (built from the SAME gathered
facts' own literal strings, `_build_claim_verifier`, research/runners/brain_chat_tui.py:1379-1408) still hold
the UNDERSCORED form, so the stripped token never matched -- every clause about a real LTM entity was
classified as containing an "unrepresentable content word" and REJECTED outright (`reject:unknown_content`),
regardless of whether the substrate's own role-parse (`self.inner.parse`, confirmed working correctly on the
identical triple by a direct probe -- see below) would have recovered the fact perfectly. The identical bug
exists in the single-triple Qwen/spiking-mouth re-parse path (`_extract_svo_from_prose`,
research/runners/_grounded_lang_integration_derisk.py:120, `re.findall(r"[a-z]+", ...)`), which is also why
`spiking_recall_surface` MISSED on every known_factual probe (`spiking_hit_count=0` throughout both the before
and after runs) -- it fell through to the renderer only to be dropped there too, for the 3 affected topics.

A direct probe of the live, warm chat brain (script not committed -- see Provenance) confirms both halves of
this diagnosis on the `angora_turkey` fact `(angora_turkey, located_in_time_zone, kaliningrad_time)`:

```
renderer.render_svo(...) -> surface='The angora_turkey located_in_time_zones kaliningrad_time.'
  _verify_claim_set -> accepted=False  trace: clause=['the','angoraturkey','locatedintimezones','kaliningradtime']
                        nouns=[] verb=None n_unknown=3 verdict='reject:unknown_content'   # BEFORE the tokenizer fix
  chat._verify(single-triple) -> True                                                     # the vocab-independent
  chat.inner.parse(['angora_turkey','located_in_time_zone','kaliningrad_time']) ->         # path already worked
      {'agent': 'angora_turkey', 'action': 'located_in_time_zone', 'patient': 'kaliningrad_time'}
```

This also explains why ONE of the 4 known_factual topics (`college_for_interdisciplinary_studies`) already
"worked" in the FIRST after-fix run (before the tokenizer fix): its gathered set happened to include a genuine
pair of mirror-image facts (`canada_portal shares_border_with u_s_of_a` / `u_s_of_a shares_border_with
canada_portal`), which trips `ClaimEntailmentVerifier.__init__`'s own role-permutation-collision guard
(research/runners/_moat_claim_entailment_derisk.py:191-195: `raise AssertionError` when a gated triple is a
role-permutation of another). `_build_claim_verifier` catches that `AssertionError` and returns `None`
(research/runners/brain_chat_tui.py:1403-1406), so `_verify_rendered` falls back to the vocabulary-independent
single-triple `_verify` -- which trivially succeeds for the STUB renderer regardless of underscores, because
its own `asserted` is the literal `[agent, action, patient]` list, never a re-tokenized prose string. Every
other topic, lacking that lucky collision, hit the tokenizer bug directly and lost all 3-5 sentences.

**Fix:** widen the tokenizer's character class from `[a-z]` to `[a-z_]` in all three sites (`_clauses`,
`_noun_is_before`, `_extract_svo_from_prose`) so an underscore survives inside a token instead of being
stripped. This is NOT gated behind a new flag: it is a plain character-class widening inside an
already-shipped, already-default-on verifier, a strict superset of what it recognized before (adds, never
removes), and inert for every underscore-free token -- which is 100% of all production traffic before this
session, since underscored content only exists in the LTM shard this session's part-1 fix is what first made
reachable from the direct-recall gate at all. Confirmed BYTE-IDENTICAL for underscore-free content via the
module's OWN pre-existing 6-seed adversarial regression self-test, unaffected by this change:
`research/findings/raw/_moat_claim_entailment_underscore_fix_regression.json` -- `GO=true`, `leak_rate=0.0`,
`false_reject_rate=0.0`, `core=10/10` and `hyp=3/3` on every one of seeds 42/43/44/100/101/102, identical to
the pre-fix numbers quoted in this repo's own 2026-08-12 landing finding for this module.

`old_single_triple_moat_accepts` (research/runners/_moat_claim_entailment_derisk.py:147) -- a hand-rolled
CONTROL that deliberately replicates the PRE-FIX `_extract_svo_from_prose` behavior for this file's own
internal leak-detection demo ("used as a CONTROL... proving the leak-detector can SEE a leak") -- is
deliberately left untouched: it is not reachable from production, and fixing it would defeat its documented
purpose as a fixed historical baseline for the module's own self-test.

## 4. Before/after, the real-traffic re-measurement

Same instrument, same probes, same seed=42 (same 4 sampled known topics:
`angora_turkey`/`college_for_interdisciplinary_studies`/`imperial_roman`/`l_quipe_de_france`), run through the
real `webapp.server.brain_chat` entry point, CPU-forced, Surface A only (`BRAIN_OPEN_ENDED` unset -- the
production default):

| idx | class | prompt | BEFORE abstained | AFTER abstained | BEFORE n_sent | AFTER n_sent |
|---|---|---|---|---|---|---|
| 0 | known_factual | Tell me about angora_turkey. | True | **False** | 0 | 3 |
| 1 | known_factual | What do you know about college_for_interdisciplinary_studies? | True | **False** | 0 | 5 |
| 2 | known_factual | Can you tell me about imperial_roman? | True | **False** | 0 | 4 |
| 3 | known_factual | What is l_quipe_de_france? | True | **False** | 0 | 4 |
| 4 | known_multi_sentence | Tell me everything you know about angora_turkey and explain why it matters. | True | True | 0 | 0 |
| 5 | known_followup | tell me more | True | **False** | 0 | 1 |
| 6-9 | unknown/dangerous | (4 probes) | False | False | 1 each | 1 each (BYTE-IDENTICAL answer text) |
| 10-11 | open_ended_opinion | (2 probes) | True | True | 0 | 0 |
| 12 | open_ended_opinion | Why do you think memory matters? | False | False | 3 | 3 |
| 13-15 | greeting_social | (3 probes) | (1 abstain, 2 answer) | (identical pattern) | 0/1/None | 0/1/None |

**Every known_factual probe that abstained BEFORE now returns a real, grounded, multi-fact answer sourced
directly from the LTM shard, AFTER** (4/4, up from 0/4). Sampled verbatim (both from
`research/findings/raw/_per_touchpoint_qwen_share_shipped_default_AFTER_recall_gate_fix.json`):

- `angora_turkey`: *"Sure -- The angora_turkey located_in_time_zones kaliningrad_time. The angora_turkey
  instance_ofs city_work. The angora_turkey countrys the_republic_of_turkey. -- worth going further here."*
- `imperial_roman`: *"Sure -- The imperial_roman followses res_publica_romana. The res_publica_romana
  countrys italian_republic. The italian_republic located_in_time_zones rome_time. The imperial_roman
  followed_bys byzantine_empire. -- worth going further here."* -- a genuine 2-hop CHAIN (imperial_roman ->
  its predecessor state -> that state's own country -> that country's own time zone), plus a second
  independent fact (followed_by), all four VERIFIED against the LTM shard.

versus BEFORE, every one of the same 4 topics produced the SAME curiosity-flavored non-answer template with
the topic's slug truncated to its first underscore-delimited token (a SEPARATE, already-disclosed bug the
root-cause finding named and left out of scope): *"Sure -- I don't know about that. My curiosity is piqued --
I haven't learned about angora yet: what can you tell me about angora? -- worth going further here."*

**The moat check: every unknown/dangerous probe's answer text is BYTE-IDENTICAL before vs after** (verified by
direct string comparison of the two artifacts' `answer` fields, not just the `abstained`/`n_sentences` counts
-- see Provenance for the comparison method). The pre-existing in-loop-acquisition mis-teach the root-cause
finding disclosed for bare-word "Tell me about X" probes (zorplaxian/flibberwock/paris/python) is completely
unaffected by either fix -- it happens earlier in `gate()`, before this fallback is ever reached, and this
fallback never runs for a question `gate()` already resolved (see SS2's precedence note). open_ended_opinion
and greeting_social are likewise unaffected (no topic to extract; `extract_topic` returns nothing storeable).

RSS stayed comfortably inside this task's RSS<4GB budget: BEFORE peak 1067.3 MB, AFTER peak 1485.1 MB (both
`ru_maxrss` high-water marks over the full 16-probe run, template-stub renderer, `wikidata_core_15k`).

## 5. Honest residuals

- **`known_multi_sentence` still abstains, unchanged.** "Tell me everything you know about X and explain why
  it matters." does not start with any pattern in `webapp.open_ended_chat._LEADINS`, so `extract_topic` cannot
  isolate the bare entity from this specific phrasing and returns the whole (unmatchable) sentence instead.
  This is a genuine, disclosed coverage LIMIT of the lead-in list this fix reuses, present identically before
  and after (BEFORE also abstained on this exact probe) -- not a regression, and not claimed as closed here.
  Extending `_LEADINS` to cover more natural phrasings is a natural, separate next rung.
- **`known_followup` ("tell me more") now succeeds, but on a DIFFERENT topic than the probe intended.** Because
  probe 4 (`l_quipe_de_france`) now succeeds and probe 5 (the `known_multi_sentence` elaboration attempt on
  `angora_turkey`) still abstains without updating the discourse thread (`RichAnswerComposer.answer()`'s
  `if not facts` / `if not kept` abstain paths return BEFORE `self._topic` is ever (re)set), `self._topic`
  still holds `l_quipe_de_france` from probe 4 when probe 5's "tell me more" is reached (a genuine followup
  phrase). The reply -- *"Sure -- The l_quipe_de_france participant_ofs world_championship_football_2010. --
  worth going further here."* -- is itself a real, grounded, VERIFIED fact (the moat holds), just not about the
  topic the probe sequence intended. This exact interaction could not have been OBSERVED before this fix
  (probe 4 always abstained previously, so `self._topic` was never set to begin with, and probe 6 also
  abstained, matching BEFORE row 5) -- it is a newly-VISIBLE consequence of two already-independently-disclosed
  behaviors (the `known_multi_sentence` coverage gap above, and the followup mechanism's design of carrying the
  LAST successful topic forward), not a new defect this fix introduces. Disclosed rather than hidden.
- **Two pre-existing, unrelated test failures noticed, not caused by, and not fixed by this work**:
  `tests/test_open_ended_generation_fluent.py::test_render_hypothesis_fluent_flagged_guess_stub` and
  `::test_render_hypothesis_template_fallback_without_mouth` fail identically on `main` at
  `aa55a2f63ff3e8278e4999d3eca5c04af846289c` with ZERO changes applied (confirmed via `git stash`) -- a
  separate `ChatBrain.render_hypothesis_verified` regression unrelated to `RichAnswerComposer`/the recall gate.
  Flagged as a separate follow-up task, out of this fix's scope.
- **The `known_factual`/`known_multi_sentence` sample is 4 topics, one seed's draw** -- the same N the
  root-cause finding used, for direct before/after comparability (see seed-waiver). A larger/independently
  sampled battery is the natural next rung, not a 6-seed repeat of this deterministic comparison.
- **Renderer substitution unchanged from the root-cause finding**: Surface A's renderer resolves to the
  GPU-free `template-stub` under `SIM_BACKEND=numpy` (confirmed live in both artifacts:
  `"renderer": "template-stub (GPU-free)"`), not the literal Qwen2.5-0.5B a CUDA host would pick. The
  tokenizer fix (part 2) also applies to `_extract_svo_from_prose`, the Qwen-path re-parse, so it should
  extend the SAME benefit there, but that specific renderer path was not exercised in this CPU-forced
  measurement.

## Provenance

Shipped code read/edited this session (2026-09-04): `research/runners/rich_answer_composer.py` (`_direct_fact`
:369-411, the new `_direct_ltm_topic_fallback_enabled` :90-112, `_facts_about`/`_with_ltm`/`_ltm_facts_about`
:243-299 read, unmodified), `research/runners/brain_chat_tui.py` (`ChatBrain.__init__`/`_refresh_facts`
:604-663, `gate`/`_gate_router_combine`/`_substrate_recall` :666-724,1240-1330, `_verify`/`_verify_claim_set`/
`_build_claim_verifier` :1358-1423, read not modified), `research/runners/tiered_fact_store.py`
(`TieredFactStore.__getattr__` :273-275, read not modified), `webapp/server.py` (`_build_chat_brain`
:3794-3897, read not modified), `research/runners/_moat_claim_entailment_derisk.py` (`_clauses` :226-260,
`_noun_is_before` :436-450, `_build_claim_verifier`'s role-permutation guard :191-195, `main`/`build_suite`/
`run_seed` :581+ read to run the self-test), `research/runners/_grounded_lang_integration_derisk.py`
(`_extract_svo_from_prose` :120-141), `research/runners/_grounded_lang_p3_derisk.py`
(`TemplateStubFaculty.render_svo`/`_inflect`/`_determiner` :75-113, read to confirm the STUB's `asserted` is
the literal triple, not a re-tokenized string).

Instrument reuse: `research/runners/_per_touchpoint_qwen_share_measure.py` copied VERBATIM from commit
`64fc4d5fc2ddf97fca8f70522e51767045b5c744` (`git show 64fc4d5f:research/runners/_per_touchpoint_qwen_share_measure.py`,
diffed byte-identical against the working copy before use). The BEFORE artifact
(`_per_touchpoint_qwen_share_shipped_default_BEFORE_recall_gate_fix.json`) is likewise a verbatim copy of that
commit's own like-named artifact (same basename, no `_BEFORE_recall_gate_fix` suffix, present only at commit
64fc4d5f) -- fetched via `git show` and diffed byte-identical against the committed copy in this branch before
use.

The claim-verifier tokenizer diagnosis (SS3) was confirmed with a direct, throwaway probe script (not
committed -- pure diagnostic, no new claim beyond what SS3/SS4 quote verbatim from its stdout) that built the
same tiny-demo+LTM chat brain via `webapp.server._build_chat_brain("tiny-demo", None)` and called
`RichAnswerComposer._facts_about`/`_direct_fact`/`render_paragraph`, `ChatBrain._verify_claim_set`/`_verify`,
and `ChatBrain.inner.parse` directly on the `angora_turkey` and `college_for_interdisciplinary_studies` fact
sets, once before and once after the tokenizer fix, comparing the `_verify_claim_set` trace dict in both runs.

The unknown/dangerous byte-identical-answer-text claim (SS4) was verified by a direct Python comparison of the
two JSON artifacts' `rows[i]["answer"]` fields for every `class in ("unknown","dangerous")` row (this session's
own shell, not separately artifacted -- both source artifacts are cited above and the comparison is a pure
string equality check reproducible from them).

Regression checks run this session, both CPU-forced (`CUDA_VISIBLE_DEVICES=""`, `SIM_BACKEND=numpy`):
`tests/test_rich_answer_batch_render.py` (5/5 pass, unchanged) and `tests/test_open_ended_generation_fluent.py`
(3/5 pass, 2 pre-existing failures confirmed via `git stash` to reproduce identically on unmodified `main`) --
run once before this session's edits and once after both fixes, identical pass/fail pattern both times.
`research/runners/_moat_claim_entailment_derisk.py --seeds 42 43 44 100 101 102` run once before the tokenizer
fix (GO=true, matching the module's own 2026-08-12 landing numbers, not separately artifacted since it
predates this session's only new artifact for this check) and once after
(`research/findings/raw/_moat_claim_entailment_underscore_fix_regression.json`, GO=true, identical
leak/false-reject/core/hyp numbers).

Builds on: research/findings/2026-09-04-per-touchpoint-qwen-call-share.md (commit 64fc4d5f), the roadmap it
built on (research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md), the LTM-shard-elaboration GO
this fix's part-1 reuses (2026-08-28-ltm-shard-elaboration-cupy-6seed-GO-unblocks-confidence-forthcomingness.md),
and the claim-entailment verifier's own landing GO this fix's part-2 re-verifies
(2026-08-12-moat-claim-entailment-derisk-multiclause-fluent-prose-passes-iff-grounded-GO.md).
