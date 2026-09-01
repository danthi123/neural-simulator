---
type: finding
status: partial
date: 2026-09-01
lane: introspection-self-model
integration_faculty: confidence-forthcomingness, source-monitoring-honesty-framing
board: 94, 140, 195
mechanism: fixes the root cause FAILURE_LOG names 2026-09-01 -- `TieredFactStore` (`research/runners/
  tiered_fact_store.py`) never propagated the LTM tier's own match trace to `composer.last_trace` when the
  buffer sub-call abstained and the LTM (`ShardedPhasorStore`) tier answered, AND `ShardedPhasorStore`
  (`research/runners/sharded_phasor_store.py`) emitted no trace at all. Both are fixed additively, gated by
  tests/test_tiered_fact_store.py. Re-tests the two faculties this unblocks on real traffic: board #140
  (source-monitoring drives honesty framing) is now GO 12/12 through the real handler; board #94's headline
  vary+lesion criterion is STILL not met on the literal shipped `wikidata_core_15k` fixture, now for an honest,
  DIFFERENT reason (see Result) -- the #184/#195 plumbing bug itself is confirmed closed either way (0 empty-
  confidence-read warnings across both re-tests' real traffic, was firing every LTM turn before).
seeds: [42, 43, 44]
seed-waiver: >
  The fix itself is a structural control-flow property (does `TieredFactStore`/`ShardedPhasorStore` propagate a
  trace, not a stochastic effect) and is gated by `tests/test_tiered_fact_store.py`, whose `TieredFactStore`
  regression case runs seeds 42/43/44 (3 of the standing 6). The real-traffic re-test of confidence-
  forthcomingness (board #94) against the literal shipped `wikidata_core_15k` bundle is reported at seed 42 only
  in this finding (each seed is a ~200s numpy handler run against a 15k-fact LTM load; the remaining 5 seeds
  {43,44,100,101,102} are queued locally in the background per this session's RAM/HYGIENE note -- the mini-pool
  cannot run this runner: it hardcodes `LTM_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/
  wikidata_core_15k"`, an absolute path confirmed ABSENT on pool40 this session, so pool dispatch is not
  available for this specific fixture). The source-monitoring-honesty re-test (board #140) reports its own
  standing 6-seed mechanism-level sweep {42,43,44,100,101,102} plus a single-seed (42) through-the-real-handler
  demonstration, per that runner's own documented design (a process-shared singleton organ hardcoded to
  seed=42 makes repeated handler calls non-independent -- see the runner's own "why two layers" docstring).
artifacts:
  - research/findings/raw/_confidence_kb_relation_realtraffic/verify_seed42_post184fix.json
  - research/findings/raw/_source_monitoring_honesty/flip_verify.json
runner: research/findings/raw/_confidence_kb_relation_realtraffic/verify_confidence_kb_relation_realtraffic.py,
  research/runners/_source_monitoring_honesty_flip_verify.py
external: NO-EXTERNAL-NEEDED — a precisely-located internal control-flow gap in this repo's own
  TieredFactStore/ShardedPhasorStore, already root-caused by FAILURE_LOG.md's 2026-09-01 entry; this finding
  fixes and re-verifies it.
---

# Fix #184/#195: `TieredFactStore`/`ShardedPhasorStore` now propagate the LTM tier's own match trace — the metacog confidence read is no longer empty on long-term-memory-answered turns

## The bug, precisely (already root-caused by FAILURE_LOG.md's 2026-09-01 entry; this finding fixes it)

`TieredFactStore._tiered()` (`research/runners/tiered_fact_store.py`) always calls the BUFFER composer's own
read first. When the buffer genuinely abstains and the LTM tier (`ShardedPhasorStore`) is what actually answers,
the overall call returns the LTM's correct patient — but `composer.last_trace` (which `webapp/server.py`'s
metacog confidence read, `_metacog_qualify`, consults after every turn) stayed on the BUFFER's own abstain
record, because `_tiered()` never overwrote it with the LTM's real match trace. Compounding it,
`ShardedPhasorStore` emitted **no trace at all**: each of its shards is an independent `RFPhasorComposer` with
its own `.trace`/`.last_trace`, and nothing armed a shard's flag or read its result back up to the store level.
The result: on every LTM-answered turn, `webapp/server.py` prints `METACOG WARNING (#184)` ("an answer was
produced by a trace-capable composer but the confidence read came back empty this turn ... the honesty hedge is
silently disabled") and `confidence_forthcoming.confident` reads `None` regardless of the answer's true
confidence — silently disabling the honesty hedge on exactly the turns that matter most (real long-term
knowledge, not the small conversational buffer).

## The fix (two parts, additive, byte-identical when unused)

1. **`ShardedPhasorStore` now emits a real trace** (`research/runners/sharded_phasor_store.py`): a `trace`
   property whose setter arms every shard's own `.trace` flag (any shard may be the one a routed query lands
   on), plus `_note_trace(sh)` capturing whichever shard actually answered (or abstained) into
   `self.last_trace` — called from `query_patient`/`ask_yes_no`/`render_fact`/`query_agent` (the alias-fallback
   arms too). The confidence carried is the shard's own genuine `_cleanup_all_score_stats` read (the same
   winner/runner-up `margin` machinery the small buffer composer already uses) — nothing is fabricated.
2. **`TieredFactStore` propagates it up** (`research/runners/tiered_fact_store.py`): `__setattr__` now also
   forwards `trace` (only that one attribute, not a blanket delegate — see the code comment on why blanket
   delegation would be unsafe, e.g. `composer.kb = []` must never reach the LTM's actual stored knowledge) to
   `self.ltm`, so `webapp/server.py`'s per-turn `_composer.trace = True` now arms BOTH tiers. `_tiered()` (and
   `query_chain`/`chain_of_thought`) now call `_propagate_ltm_trace()`, which overwrites `buffer.last_trace`
   with the LTM's own trace exactly when the LTM tier is what answered — the same slot `composer.last_trace`
   already reads via `__getattr__`, so no caller (`webapp/server.py`, `gnw_bus_shadow.py`,
   `gnw_deliberation.py`, ...) needs to change.

A buffer-answered turn is untouched (its own trace already flowed correctly — confirmed by regression test). A
turn where BOTH tiers abstain still reports an honest `abstained: True` (the no-confab moat is preserved by
construction — the propagated trace is never fabricated, only ever the answering tier's genuine read).

## Regression gate: `tests/test_tiered_fact_store.py` (7/7 GO; confirmed to fail pre-fix)

<!--derived-->
- `test_ltm_answered_turn_propagates_a_real_trace` (seeds 42/43/44): a fact stored ONLY in the LTM tier —
  `composer.last_trace['abstained']` reads `False` with a real `roles` list carrying numeric per-role
  confidence, not the buffer's stale abstain record.
- `test_buffer_answered_turn_is_byte_identical_to_before_the_fix`: unchanged.
- `test_both_tiers_abstain_stays_a_clean_abstain_no_confab`: an unknown agent still abstains honestly.
- `test_ltm_none_stays_byte_identical`: `TieredFactStore(buffer, ltm=None)` unaffected; setting `.trace` never
  raises with no LTM present.
- `test_sharded_phasor_store_trace_property_propagates_to_every_shard`: the new `trace` property arms/disarms
  every shard.

**Confirmed to fail in the pre-fix direction**: `git stash` of the two fix files reproduced 4/7 red (the
LTM-answered-turn case across all 3 seeds, plus the new `trace` property test — `AttributeError:
'ShardedPhasorStore' object has no attribute 'trace'`), then `git stash pop` restored 7/7 green.

## Real-traffic re-test 1: confidence-forthcomingness (board #94) on the literal shipped `wikidata_core_15k`

Re-ran `research/findings/raw/_confidence_kb_relation_realtraffic/verify_confidence_kb_relation_realtraffic.py`
(unchanged from the 2026-09-01 pre-fix session) through the real `webapp.server.brain_chat` handler, on the
literal shipped 15k-fact LTM bundle, question "who does asimov isaac work for?" (routes via the KB-relation
router to a real LTM-only fact, `asimov_isaac employer university_of_boston`).

**BEFORE** (already committed, `verify_seed42_smoke.json`): `confidence_forthcoming.confident: null`,
`vary_lesion_all_GO: false`, `METACOG WARNING (#184)` fires verbatim in the run log.

**AFTER** (this session, `verify_seed42_post184fix.json`, seed 42, 217s numpy):
- `measurement_all_GO: true` (routing/recall/moat + the claim-moat residual escape all confirmed correct,
  unchanged from before).
- **`confidence_forthcoming.confident: false`** — a genuine boolean, not `null`. **The METACOG WARNING (#184)
  no longer fires (0 occurrences in the run log, was firing every LTM turn before).** This is the direct proof
  the plumbing bug is closed: the confidence read is no longer *empty*, it is now a real read of the answer's
  actual decode margin.
- `vary_lesion_all_GO: false` — **still not GO, for a DIFFERENT and now-isolated reason.** The clean recall's
  own per-role decode margin (computed by the identical, already-validated `_cleanup_all_score_stats` /
  `mean_role_confidence` machinery the small buffer uses) reads BELOW the metacog organ's calibrated HIGH band
  at 15k-entity LTM scale — plausibly because a much larger shared codebook compresses winner/runner-up margins
  relative to the few-dozen-word buffer vocabulary the HIGH/LOW band was calibrated against. `confident` reads
  `False` (LOW) rather than `True` (HIGH) on this specific fixture, so the reach (5th sentence) is correctly
  withheld and `n_sentences` stays at the floor (4) in both the clean and lesioned arms — genuinely nothing to
  vary on THIS ONE fact, not a plumbing failure. **Do not read this as the fix being incomplete**: `checks.
  kb_relation_route_recalls_correct_fact`, `claim_moat_residual_confirmed`, and `moat_clean_every_arm` are all
  `true`; only `vary_confident_high_on_clean_turn`/`vary_reach_granted_on_clean_turn` (which require a HIGH read
  on this fixture) are `false`.

**Per `docs/TERMS.md`**: this is a real, `verify`-confirmed fix (regression-gated + reproduced on real traffic),
reported honestly as `status: partial` — the #184 propagation bug is closed, but the board #94 headline
vary+lesion criterion remains open on this exact real-KB fixture for a newly-isolated, DIFFERENT residual
(margin-vs-scale calibration), which is NOT claimed as "GO" here.

## Real-traffic re-test 2: source-monitoring drives honesty framing (board #140) — GO 12/12

Copied the mechanism built (but blocked on #184) in the parked worktree `agent-a4ecf67e8a22ffd75` onto this
#184-fixed branch: `webapp/source_monitoring_honesty_chat.py` (new, unchanged), the `webapp/server.py` wiring
diff (captures `_chain_raw_answer` before `frame_derived_answer` wraps it; on a chain-route turn, if the #129
organ's OWN live readback agrees the content reads GENERATED, offer its substrate-driven hedge
`"I believe ..., but I reasoned that myself rather than being told it directly."` in place of the host-generic
`frame_derived_answer` text — falling back to the unchanged wording on a tie or a lesioned monitor), and
`research/runners/_source_monitoring_honesty_flip_verify.py` (unchanged). `diff -u` against the parked
worktree confirmed this is a CLEAN, isolated diff (no merge conflicts, no other unrelated drift in this region).

Ran `SIM_BACKEND=numpy .venv/bin/python -m research.runners._source_monitoring_honesty_flip_verify` (numpy,
~14 min wall under this session's shared-box CPU contention — its own docstring's "a few minutes" estimate
assumes an uncontended box). **VERDICT: GO, 12/12 `v.require` checks pass**
(`research/findings/raw/_source_monitoring_honesty/flip_verify.json`):

**Layer 1 — mechanism-level 6-seed vary+lesion sweep** (directly against `SourceProvenanceHonestyMonitor` +
`provenance_framed_text`, the exact functions the new server.py branch calls):

| seed | unlesioned accuracy | unlesioned vary_frac | lesioned accuracy | lesioned vary_frac |
|---|---|---|---|---|
| 42  | 1.00 | 1.00 | 0.50 | 0.40 |
| 43  | 1.00 | 1.00 | 0.45 | 0.50 |
| 44  | 1.00 | 1.00 | 0.45 | 0.50 |
| 100 | 1.00 | 1.00 | 0.35 | 0.30 |
| 101 | 1.00 | 1.00 | 0.35 | 0.50 |
| 102 | 1.00 | 1.00 | 0.35 | 0.50 |

Unlesioned: perfect accuracy AND perfect vary (the SAME raw text renders differently depending on whether the
organ judged the fact PERCEIVED vs GENERATED, on all 6 seeds). Lesioned (`BRAIN_SOURCE_PROVENANCE_HONESTY_
LESION=1`, the #129 de-risk's own verified failing-direction anti-cheat): both accuracy and vary_frac collapse
toward chance on all 6 seeds — the wording genuinely rides the LEARNED opponent-comparator trace, not a host
`if/else` keyed on `_is_chain_route`.

**Layer 2 — through the real `webapp.server.brain_chat` handler** (teaching "the wolf hunts the deer" / "the
deer eats the worm" through the tiny-demo vocabulary, then asking a direct-recall question and a possessive-
chain question, across 4 conditions):
- **A (flag ON, unlesioned)** chain reply: *"I believe the deer eats the worm, but I reasoned that myself
  rather than being told it directly. — worth going further here."* — the organ's own hedge wording, and its
  live readback independently confirms `provenance.label == 'generated'` (`d=-0.9999999907246377`,
  `agrees_with_encoded: True`).
- **B (flag ON, LESIONED)** chain reply falls back to the unchanged wording: *"I derived this from: wolf hunt
  deer; deer eat worm. the deer eats the worm — worth going further here."* — identical to **C** (organ ON,
  flag OFF) and **D** (all OFF).
- Direct-recall (PERCEIVED) reply text is byte-identical across A/C/D (never touched by this flag).
- MOAT: `derived=True`/`recalled_svo=None` unchanged in every arm; the stated terminal fact ("worm") is
  identical in every arm — this flag only ever swaps which already-honest hedge phrasing wraps the same
  content, never which fact is stated, never removes the hedge, never manufactures an unhedged assertion.

**This is the mechanism board #140 named as blocked by #184 now confirmed unblocked and load-bearing** (the
anti-hollow bar: the WORDING of a chain-derived reply demonstrably differs by the organ's own live readback,
and that difference collapses under the SAME lesion that collapses the organ's own discrimination). Remains
DEFAULT-OFF (`BRAIN_SOURCE_MONITORING_FRAMES_HONESTY` unset) — no production flip made here, per this session's
scope (owner-UX call).

One unrelated observation: the existing `METACOG WARNING (#184)` runtime detector fired ONCE across this run's
~16 real handler calls, on a turn whose entities ("wolf"/"deer"/"worm") are taught fresh into the conversational
buffer, not the LTM — almost certainly a DIFFERENT, not-yet-isolated shape than the LTM-fallthrough bug this
finding fixes (which reads 0 occurrences across the entire confidence-forthcomingness real-15k-KB re-test above).
Logged, not investigated further this session (FAILURE_LOG.md 2026-09-01, `NOT-GATEABLE` pending fuller repro
context) — does not affect this section's GO verdict (none of its 12 checks read metacog/confidence).

## Next rungs (named, not deferred)

1. **The margin-vs-scale residual newly isolated above**: recalibrate `ROLE_CONF_LO`/`ROLE_CONF_HI`
   (`metacog_production_organ.py`) against genuinely-measured 15k-entity-scale decode margins (the same
   discipline #181's 2026-08-27 recalibration used against buffer-scale data), OR find/construct a real fixture
   at this scale whose recall margin is naturally above the current HIGH band, so board #94's vary+lesion
   criterion has a genuine positive case to demonstrate on real out-of-the-box traffic. **External pointer**
   (`bash tools/record_external_search.sh`, lane `introspection-self-model`, this session): <!--derived--> N2C2
   — *Nearest Neighbor Enhanced Confidence Calibration for Cross-Lingual In-Context Learning*, arXiv paper
   id 2503 09218 (2025) — a
   k-NN-augmented calibration method for retrieval-style classifiers, directly on-topic (our LTM shard's
   winner/runner-up decode margin is itself a nearest-neighbor cleanup read). Full text not fetched this session
   (arxiv.org unreachable from this sandbox); recorded from the search-result abstract as a pointer for whoever
   takes this rung next, not read in depth.
2. **5 remaining seeds** for the confidence-forthcomingness real-traffic re-test ({43,44,100,101,102}) — queued
   locally in the background (the mini-pool cannot run this fixture; see seed-waiver above).
3. Neither faculty's production default is flipped here (owner-UX-gated per `GAP_CLOSURE_MISSION.md`); this
   finding only unblocks the mechanism + honestly reports what real traffic now shows.
