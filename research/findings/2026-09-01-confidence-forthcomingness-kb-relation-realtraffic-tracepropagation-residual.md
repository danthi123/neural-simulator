---
type: finding
status: partial
date: 2026-09-01
lane: introspection-self-model
integration_faculty: confidence-forthcomingness
board: 94
mechanism: confidence-caps-forthcomingness re-tested through the real webapp.server.brain_chat handler on the
  literal shipped wikidata_core_15k LTM via the newly-closed KB-relation router; the coupling's own mechanics
  (routing, recall, moat, reach/cap arithmetic) are confirmed correct, but the metacog confidence READ itself
  reads `None` on this real-traffic construction due to a newly-located `TieredFactStore.last_trace`
  propagation gap -- a DIFFERENT, more upstream residual than the one 2026-09-01's loadbearing finding closed.
seeds: [42]
seed-waiver: >
  The residual this finding characterizes (TieredFactStore.last_trace never being updated to the LTM tier's own
  match trace when the buffer's own sub-call abstains) is a STRUCTURAL control-flow property of
  `research/runners/tiered_fact_store.py::TieredFactStore._tiered()`, not a stochastic effect -- `seed` only
  varies the substrate build (`_build_tiny_demo(seed, ...)`'s heterogeneity), which does not touch this code
  path's control flow. The KB-relation routing itself is separately confirmed seed-independent (deterministic
  regex routing) in 2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md's own seed-waiver.
  5 additional seeds (43, 44, 100, 101, 102) are QUEUED to the mini-PC pool as separate single-seed jobs
  (`research/queue/pool.queue`, added this session) to confirm cross-seed stability of the `confident: None`
  read; not yet returned at the time of this finding -- this finding's claims rest on the seed-42 measurement
  plus the code-level root-cause trace (read directly from `tiered_fact_store.py` and `gnw_bus_shadow.py`),
  not on a seed-count argument.
artifacts:
  - research/findings/raw/_confidence_kb_relation_realtraffic/verify_seed42_smoke.json
runner: research/findings/raw/_confidence_kb_relation_realtraffic/verify_confidence_kb_relation_realtraffic.py
external: NO-EXTERNAL-NEEDED — a precisely-located internal control-flow gap in this repo's own TieredFactStore,
  not a capability wall or biological question.
---

# Confidence-forthcomingness re-tested on the literal shipped wikidata_core_15k, through the real KB-relation router: mechanics confirmed correct, but a newly-located `TieredFactStore.last_trace` gap keeps the metacog confidence read at `None` on this real-traffic path

## The precise question this closes (or doesn't)

[`2026-09-01-confidence-forthcomingness-ltm-elaboration-load-bearing-GO.md`](2026-09-01-confidence-forthcomingness-ltm-elaboration-load-bearing-GO.md)
proved confidence-forthcomingness genuinely load-bearing (5-vs-4 sentences, lesion-attributable, 6-seed GO)
through the real `webapp.server.brain_chat` handler — but on a **controlled fixture** (a small buffer + a
routed LTM shard), explicitly **not** on "the literal shipped `wikidata_core_15k` out-of-box traffic (its vocab
doesn't route through the live NL parser)". That NL-parser vocab gap was closed the same day
([`2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md`](2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md),
commit `8047b73a`), making the deferred real-traffic re-test possible for the first time. This finding runs
that re-test.

**Result: the routing, recall, and moat mechanics are all confirmed correct on the literal shipped bundle — but
the vary+lesion target criterion (does a high-confidence turn produce more grounded sentences than a
low-confidence turn) does NOT hold on this real-traffic construction**, because of a newly-located, precisely-
characterized wiring gap upstream of confidence-forthcomingness itself: `TieredFactStore.last_trace` (the
signal the metacog confidence read depends on) is never updated to reflect the LTM tier's own successful match
when the buffer tier's own sub-attempt abstains — which is the *normal* case for any concept that lives only in
the LTM shard, i.e. essentially every shipped-KB entity. This is reported honestly as a **partial** result, not
recharacterized as a pass: the confidence-forthcomingness *mechanism* (proven load-bearing on the controlled
fixture) is not itself refuted here; a *different, upstream* signal-plumbing gap is what blocks it on real
traffic.

## Fixture: a real shipped fact, real KB-relation-routed question

Entity `asimov_isaac` (10 real facts in the shipped `wikidata_core_15k` bundle — enough own-agent facts plus a
real 2-hop chain-turn for the rich-answer gather to genuinely exceed the `NEUTRAL_SENTENCES=4` floor), relation
`employer`. Question: **"who does asimov isaac work for?"** — one of the 29 idiom-routed KB-relation questions
independently confirmed to route + recall the exact real stored patient (`university_of_boston`) in the
NL-parser-routing finding's own 29/29 sweep. Built via `_build_tiny_demo(seed, composer_kind="onebrain")` +
`ShardedPhasorStore.load(wikidata_core_15k)` attached through `TieredFactStore` — the SAME attach
`webapp.server._build_chat_brain` uses for the out-of-the-box "tiny-demo +LTM" default brain — then driven
through the REAL `webapp.server.brain_chat` handler (`S._BRAIN_CHATS` pre-populated, `rich=True`), mirroring
`verify_confidence_ltm_loadbearing.py`'s own construction pattern.

## Residual 1 (a workaround, not fixed): the claim-entailment verifier's vocab/grammar gap

Before the confidence-forthcomingness question could even be asked, every gathered sentence was DROPPED by the
claim-level moat generalization (`ClaimEntailmentVerifier`, `BRAIN_CLAIM_MOAT`, default ON): the StubRenderer's
template surface form for underscored multi-word Wikidata tokens ("The asimov_isaac employers
university_of_boston.") is not correctly clause-parsed by the verifier, even though the single-triple
`chat._verify` independently confirms every one of the 4 gathered facts is grounded and correct. Measured
per-seed 42, artifact `claim_moat_residual`: `BRAIN_CLAIM_MOAT=1` (the production default) → `abstained: true,
n_sentences: 0` (every fact dropped); `BRAIN_CLAIM_MOAT=0` (the existing escape flag documented in
`brain_chat_tui.py`) → `abstained: false, n_sentences: 4` (all facts restored). This finding uses the existing
escape flag to reach the confidence-forthcomingness measurement itself — not a new mechanism, not a `sim/`
edit, an already-documented flag. The residual (why the claim-level verifier's clause-parse fails on this
vocabulary) is banked in `research/FAILURE_LOG.md`, not fixed here.

## Residual 2 (the actual blocker): `TieredFactStore.last_trace` does not propagate the LTM tier's answer

With residual 1 escaped, the CLEAN turn (seed 42, no lesion) correctly recalls the real fact
(`recalled_svo: [asimov_isaac, employer, university_of_boston]`, `recall_correct: true`, bus-committed with all
3 ignition organs unanimous — confirmed via a direct scratchpad probe of `webapp.gnw_bus_shadow._organ_reads`/
`bus_combine`, not committed) — an objectively unambiguous, high-confidence recall by construction. Yet
`confidence_forthcoming.confident` reads `null` (artifact: `clean.confident: null`, `clean.reason:
"low_confidence_capped"`), not `True`, so the reach's bonus fact is never granted (`granted: false,
kept_sentences: 4` of the `requested_sentences: 5`, `elaborations_dropped: 1`) — `n_sentences` stays at the
floor (4) regardless of the turn's true confidence. `webapp/server.py`'s OWN pre-existing runtime detector
fires verbatim during this measurement: `[webapp] METACOG WARNING (#184): an answer was produced by a
trace-capable composer but the confidence read came back empty this turn ... the honesty hedge is silently
disabled. This is the plumbing-bug signature (TieredFactStore.__setattr__ ate last_trace for a day the same
way); check the activity-trace plumbing.`

Root cause, located by direct code read (`research/runners/tiered_fact_store.py`,
`webapp/gnw_bus_shadow.py`): `TieredFactStore._tiered()` always calls the BUFFER composer's own read FIRST;
when the concept lives only in the LTM shard (the normal case for a shipped-KB entity), the buffer's own call
ABSTAINS and sets `buffer.last_trace` to ITS OWN abstain record — `_tiered()` then falls through to the LTM
tier, returns the LTM's correct answer, but never overwrites `buffer.last_trace` with the LTM's own match
trace. Since `composer.last_trace` (== `buffer.last_trace` via `TieredFactStore.__getattr__`) is what
`webapp/server.py`'s `_read_activity()` / the GNW ignition bus's `surface_forward_trace` both read, and what
`RichAnswerComposer._chain_facts`'s existing 2026-08-27 "TRACE PRESERVATION" fix assumes correctly reports
whether the JUST-MADE `query_patient` call matched — every one of these consumers instead sees the buffer's
spurious abstain, for an answer the LTM tier actually and correctly supplied. This is a DIFFERENT, more
upstream residual than either the NL-parser vocab gap (closed) or the buffer-tier-only elaboration gap the
2026-09-01 loadbearing finding closed on its controlled fixture (where the topic word happened to already be a
buffer concept, so the chain never needed to query the LTM tier and never hit this gap).

Because `confident` reads `None` (not a real high/low value) on every turn observed, the LESION check
(`BRAIN_METACOG_LESION=1`) is consequently MOOT on this construction — there is no positive vary-difference for
the lesion to collapse (clean and lesioned both read `confident: null`, `n_sentences: 4`; artifact
`lesion.confident: null`, `lesion.n_sentences: 4`). This is reported as such, not recharacterized as a
lesion-attributable GO.

## Verification (seed 42; artifact: `research/findings/raw/_confidence_kb_relation_realtraffic/verify_seed42_smoke.json`)

| check | result |
|---|---|
| KB-relation route recalls the exact real stored fact | PASS (`recall_correct: true`) |
| claim-moat residual confirmed (ON drops everything, OFF restores) | PASS (`on_abstained: true, on_n_sentences: 0` → `off_abstained: false, off_n_sentences: 4`) |
| moat clean in every arm (no invented fact) | PASS (`moat_ok: true` clean + lesion + claim-moat-off) |
| **target: `confident` reads `True` on the clean turn** | **FAIL** (`confident: null`) |
| **target: the reach is granted on the clean turn (n_sentences > 4)** | **FAIL** (`n_sentences: 4`) |

`measurement_all_GO` (routing + recall + moat + both residuals precisely confirmed): **True**.
`vary_lesion_all_GO` (the ORIGINAL board #94 target — confidence discriminates real-traffic forthcomingness):
**False.**

Runner: `research/findings/raw/_confidence_kb_relation_realtraffic/verify_confidence_kb_relation_realtraffic.py`
(numpy backend, CPU; 273.1s / seed — one `TieredFactStore`-wrapped onebrain build reused for the 4 turns a seed
needs, avoiding the per-seed RSS accumulation that OOM-exhausted the NL-parser routing finding's own attempted
6-seed rebuild).

## What this is (and is not)

This is **not** a retraction of the 2026-09-01 loadbearing finding — that finding's controlled-fixture GO
stands unchanged (it never needed a `TieredFactStore` LTM-fallthrough on its topic word, so it never hit this
gap). This **is** the true-production-floor re-test the loadbearing finding explicitly deferred, and it
surfaces the actual remaining blocker precisely: not the NL-parser vocabulary (closed), not a hollow
reach-cap (closed on the controlled fixture), but a signal-plumbing gap in `TieredFactStore` that the
codebase's own runtime instrumentation (#184) already anticipated as a class but had not yet been traced to
its exact code location. Per `research/FAILURE_LOG.md`'s standing rule, this is banked precisely (not a "wall",
not an "honest negative" stopping point) with a concrete next step: `TieredFactStore._tiered()` needs to
propagate whichever tier's `last_trace` actually answered (or expose a per-tier trace) — deliberately NOT
attempted in this bounded measurement session because that class is depended on by the GNW bus corroboration,
self-schema honesty, and source-provenance monitors (a change there needs its own dedicated regression pass,
not a same-session patch layered onto a measurement task).

## Next rungs

- Fix `TieredFactStore`'s tier-trace propagation (see `research/FAILURE_LOG.md`'s 2026-09-01 entry for the
  precise location and a candidate regression test), then re-run this exact verify — if `confident` now reads
  `True` on the clean turn and `False` under noise/lesion with `n_sentences` genuinely discriminating, board #94
  reaches its first real-out-of-the-box-traffic GO.
- Separately, the claim-entailment verifier's vocab/grammar gap on underscored multi-word tokens (residual 1)
  is worth its own small fix (a candidate regression test is named in the FAILURE_LOG entry) so
  `BRAIN_CLAIM_MOAT=0` stops being a required escape for this vocabulary class.
- 5-seed pool confirmation (43/44/100/101/102, queued this session to `research/queue/pool.queue`) will report
  whether the `confident: None` read is uniform across seeds as the structural root-cause predicts.
- No production-default change made or implied here (owner's UX call on the coupling itself is unaffected by
  this finding — it characterizes a plumbing gap, not a mechanism verdict).
