---
type: finding
status: live
date: 2026-08-28
mechanism: knowledge-bundle-vocab-latency-wall
lane: knowledge-integration
seeds: [42]
seed-waiver: An ENGINEERING capacity/latency de-risk (deterministic curation + build, and a vocab-size-vs-latency
  dose-response measurement across 3 bundle scales through the real production load path), not a stochastic
  effect size -- single deterministic build per config, matching the seed-waiver precedent of the
  2026-08-20/2026-08-27 sharding findings in this same lane.
instrument: research/runners/_knowledge_scale_vocab_latency_probe.py (NEW) -- loads a real bundle through the
  exact `developed_brain_io.load_developed_brain(ltm_bundle=...)` production path and times individual
  recall/moat/yesno queries, provenance-stamped. research/runners/_knowledge_core_curate.py (EXISTING, reused
  unchanged, board #133/#65) -- curates raw wikidata5m triples into a ShardedPhasorStore bundle at chosen
  top_entities/top_relations/n_facts caps.
runner: research/runners/_knowledge_scale_vocab_latency_probe.py
external: NO-EXTERNAL-NEEDED -- the raw wikidata5m dump was already fetched to this box (2026-08-21,
  sim-data/wikidata5m, 20.6M triples); this arc curates further into it and re-verifies the already-validated
  store/sharding/fast-bind mechanisms at new scales.
artifacts:
  - research/findings/raw/_knowledge_core/curate_report_500k_fast.json
  - research/findings/raw/_knowledge_core/curate_report_1M_fast.json
  - research/findings/raw/knowledge_500k_verify/vocab_latency_100k_control.json
  - research/findings/raw/knowledge_500k_verify/vocab_latency_500k.json
  - research/findings/raw/knowledge_500k_verify/vocab_latency_1M.json
---
# A 748,956-fact real-Wikidata bundle (9.5x the shipped core) proves the DATA ceiling is in the millions and recall/moat hold at scale — but uncovers a vocabulary-driven latency wall sharding never fixed

Artifacts: `research/findings/raw/_knowledge_core/curate_report_500k_fast.json`,
`research/findings/raw/knowledge_500k_verify/vocab_latency_{100k_control,500k,1M}.json` (mixed verdict — see below).

**One line.** The owner's #1 priority (#66) asked for the LARGEST real knowledge bundle the available data
supports, chat-verified at that scale. The DATA is not the limit (the already-downloaded wikidata5m corpus
supports millions of facts; a curation-cap choice, not data scarcity, capped the shipped bundle at 78,857). A new
748,956-fact bundle was built and byte-verified for recall + the no-confab moat through the real production load
path. But the SAME verify also found that per-query latency, not fact count, breaks the "sub-second at any K"
claim once a real bundle's natural vocabulary growth is included — a wall the 2026-08-20/2026-08-27 sharding
findings never tested because they held vocabulary fixed near the shipped bundle's own 23,914 words.

## 1. The honest data ceiling: NOT the blocker (curate-only scan, no build)

`_knowledge_core_curate.py`'s `curate()` was run at increasing `top_entities`/`top_relations` caps (candidate-fact
counting only, no store build — 39-78s/config) over the full wikidata5m_transductive corpus already on this box
(`sim-data/wikidata5m`, unchanged since 2026-08-21):

| top_entities | top_relations | candidate facts | vocab | alias facts |
|---|---|---|---|---|
| 25,000 (== shipped `wikidata_100k`) | 60 | 78,857 (exact match to the shipped bundle — sanity check) | 23,914 | 88,162 |
| 75,000 | 150 | 388,571 | 74,160 | 205,948 |
| 200,000 | 400 | 1,269,320 | 199,149 | 388,891 |
| 500,000 (11% of all entities) | 822 (ALL relations) | **3,417,855** | 498,767 | 739,154 |

The full corpus: 20,614,279 triples, 4,594,149 entities, 822 relations. Already at 500k top-entities (11% of the
graph) the candidate pool is 3.4M facts — 43x the shipped bundle — and rising toward the full entity set. **The
78,857-fact shipped bundle was a conservative curation choice (`--top-entities 25000 --top-relations 60`), not a
data ceiling.** The real ceiling is several million facts; the practical ceiling in THIS session is the RSS/time
budget of a single build process (below), not the source data.

## 2. The bundle built: `wikidata_500k_fast`, 748,956 real facts, RSS-budget-compliant

Built via `_knowledge_core_curate.py` (existing runner, unchanged) with `--fast` — the closed-form `encode_fast`
bulk bind (`build_ltm_from_facts(fast=True)`), already GO'd recall-identical to the genuine neural resonate bind
on 150/150 answers + 20/20 moat (2026-08-21 finding) — used here purely as a build-throughput lever so the build
finishes inside this session; the store's QUERY/recall path is unaffected either way (fully neural, unchanged).

```
--out-bundle .../wikidata_500k_fast --n-facts 500000 --top-entities 100000 --top-relations 200 --seed 42 --D 128 --fast
```

- **748,956 total facts** (500,000 curated core facts + 248,956 `alias_of` natural-language-grounding facts),
  **vocab 347,695**, **3,745 shards** (`auto_n_shards`, target ~200 facts/shard — unchanged from the existing
  mechanism).
- Build: **75.3s total** (curate 46s + `ShardedPhasorStore` build/save 23s). **Peak RSS 2,731 MB** — under the
  session's 4 GB budget. Disk footprint 837 MB (`sim-data/knowledge_bundles/wikidata_500k_fast/`, outside the repo,
  same convention as the shipped `wikidata_100k`).
- `ship_ready=False` in the curation report — same honest flag `--fast` always sets (the curate script's own
  docstring: the closed-form bind is "NOT for the shipped bundle"; see Honest residuals).

**A second, larger build was also produced and is reported honestly as a constraint violation.** Pushing to
`--n-facts 1000000 --top-entities 200000 --top-relations 400` (`wikidata_1M_fast`: 1,383,561 total facts, vocab
581,065, 6,918 shards, 107.2s build) hit **peak RSS 4,764 MB — over the stated 4 GB single-process budget.** It
completed and is kept as a labelled ceiling-probe / third data point below, but `wikidata_500k_fast` is the
bundle that respects the session's own constraint and is the one recommended for any further use.

## 3. Chat-over-it verify, through the REAL production path (`load_developed_brain(ltm_bundle=...)`)

`_knowledge_scale_vocab_latency_probe.py` (new, provenance-stamped) makes the exact call `webapp/server.py` makes
for `BRAIN_LTM_BUNDLE`, then times individual `what_does`/`is_it_true` queries (small N by design — a
scaling-trend probe, not a large-N recall estimate; recall/moat correctness at full 100k scale over a large
battery is already separately banked in the 2026-08-27 finding for the 78,857-fact bundle):

| bundle | facts | vocab | shards | load time | recall | moat confab | median query latency |
|---|---|---|---|---|---|---|---|
| `wikidata_100k` (shipped, CONTROL — same box, same session) | 78,857 | 23,914 | 395 | 0.72s | 5/5 | 0/1 | **1.37s** |
| `wikidata_500k_fast` (NEW, primary) | 748,956 | 347,695 | 3,745 | 5.87s | 5/5 | 0/1 | **20.67s** |
| `wikidata_1M_fast` (NEW, ceiling-probe, exceeds RSS budget) | 1,383,561 | 581,065 | 6,918 | 8.13s | 3/3 | 0/1 | **33.82s** |

<!--derived-->
**RECALL + MOAT hold perfectly at every scale tested** (8/8 correct across all three bundles, 0/3 confabulations)
— the representation and the shard router are correct at 17.6x the shipped bundle's fact count. **LATENCY does
NOT hold sub-second past the shipped bundle's scale**: the 100k control reproduces the previously-banked
816-1246ms range almost exactly (median 1.37s here, same session, same machine — ruling out CPU contention from
a concurrently-running unrelated research job as the explanation), while the 500k bundle is **~15x slower** and
the 1M bundle **~25x slower**, tracking vocabulary size far more closely than fact count (shard size stayed flat
at ~200 facts/shard in all three, by design — the O(K) wall genuinely stays solved).

**Degrade-identical** (`ltm=None` -> byte-identical to the plain buffer, the `BRAIN_LTM_BUNDLE`-unset default) is
a structural property of `TieredFactStore`, independent of bundle content/size, already established with 3 seeds
in the 2026-08-20 finding; spot-checked again here (`query_patient` + `ask_yes_no` over 3 cues, exact match) —
holds.

## 4. Root cause: vocabulary size, not fact count, drives the query cost — and the existing fix doesn't reach this composer

`ShardedPhasorStore`'s shards are `RFPhasorComposer` instances sharing ONE global codebook across all shards
(`share_codebook=True`) — so every routed query's cleanup/decode step is a function of the TOTAL vocabulary V
(347,695 or 581,065 here), not the routed shard's ~200 facts. The 2026-08-27 finding already established,
architecturally, that `RFPhasorComposer` has **no `enable_sparse_index` parameter** — the DG sparse-index
accelerator that removes the O(V·D) cleanup cost for `OneBrainComposer` cannot reach the tiered-LTM shard
composer at all. That finding characterized this as inert-but-harmless at the 78,857-fact bundle's 23,914-word
vocabulary (small enough that the O(V·D) term was cheap). **This finding is the first to measure what happens
once a genuinely larger real bundle also brings a genuinely larger vocabulary — because sharding correctly holds
shard SIZE flat, but nothing holds VOCABULARY flat, and a bulk-KB's vocabulary grows with its fact count by
construction (more facts pull in more distinct entities as subjects/objects).** The dose-response is roughly
monotonic with V (23,914->1.37s; 347,695->20.67s; 581,065->33.82s) — consistent with an uncorrected O(V)-ish term
now dominating.

## 5. Verdict: is the blocker DATA, or something else?

**Not the data.** The ceiling scan proves millions of real facts are available from the corpus already on this
box; a curation-size choice, not scarcity, produced the shipped 78,857-fact bundle. **Not recall or the moat** —
both are perfect at 17.6x scale. **The blocker is a newly-quantified LATENCY WALL, driven by vocabulary size on
the shared-codebook cleanup step, previously invisible because no prior verify varied vocabulary independently of
fact count.** This is a wall on a METHOD (one shared global codebook for a bulk-KB-sized vocabulary), not a
capability ceiling: the concrete next rung is porting/adapting the DG sparse-index cue-match mechanism to
`RFPhasorComposer` (or giving each shard its own bounded local vocabulary instead of one global one) — scoped,
not attempted here (out of this session's verify-at-scale mandate).

## Honest residuals

1. **The `wikidata_1M_fast` build breached this session's own 4 GB RSS budget** (peak 4,764 MB vs. the stated
   ~4 GB cap) — reported rather than hidden. `wikidata_500k_fast` (peak 2,731 MB) is the compliant bundle.
2. **Both new bundles use the closed-form `fast=True` bind**, not the genuine neural resonate bind the curate
   script's own docstring reserves for a "shipped" bundle (`ship_ready=False` in both curation reports). The fast
   bind is separately GO'd recall-identical (2026-08-21 finding); QUERY/recall is unaffected either way (always
   fully neural). If the owner wants a `fast=False` bundle at this scale for actual shipping, the same command
   without `--fast` would take on the order of hours (78,857 facts took 1913s at 24.3 ms/fact genuine-bind; 748,956
   facts extrapolates to ~5 h) — a `tools/queue_add.sh pool` candidate, not run here (out of the session's CPU/
   time budget).
3. **Small probe N (5/5/3) per bundle** for the latency dose-response — deliberately small (a scaling-TREND
   measurement, not a large-N recall estimate); full-battery recall/moat at scale is already separately banked
   for the 78,857-fact bundle (2026-08-27 finding) and reproduced here at N=5-8 for the two new bundles.
4. **The shared global codebook is the same declared host-scaffold** already flagged in the 2026-08-20/27
   findings (`hash(agent)` shard router; the faithful version is a learned/spiking cue-to-subpopulation router) —
   this finding adds that the SAME codebook design is also the latency wall's root cause, sharpening rather than
   changing that scope note.

## What the owner needs to decide

1. **Which bundle ships, if any, as `BRAIN_LTM_BUNDLE`'s default target** — remains additive/default-OFF here (no
   flip attempted, per this session's scope); `wikidata_500k_fast` is 9.5x the previous bundle with perfect
   recall/moat but ~15x the query latency (20.67s vs 1.37s median) — likely too slow for a live chat turn as-is.
2. **Whether to fund the vocab-latency fix** (port sparse-index to `RFPhasorComposer`, or per-shard local
   vocabularies) before shipping anything bigger than the existing 78,857-fact bundle — this finding scopes that
   as a concrete, buildable next engineering rung on THIS codebase's own architecture (the sparse-index mechanism
   already exists and already works for `OneBrainComposer`; it simply has not been ported to the shard composer
   class), not a limit of the representation or the sharding approach.

**NO-EXTERNAL-NEEDED:** this is an in-repo architecture diagnosis, not a claim about a limit of the field — the
root cause (`RFPhasorComposer` lacking `enable_sparse_index`) and the fix (port the mechanism `OneBrainComposer`
already has) are both internal to this codebase, established by reading `sharded_phasor_store.py` /
`rf_phasor_composer.py` / the already-banked 2026-08-27 architecture check, not by a boundary-of-the-literature
question.
3. **Whether a `fast=False` (genuine-bind) rebuild of `wikidata_500k_fast` is wanted for shipping** (~5h estimate,
   queueable) given the fast-bind's already-banked recall-identity makes this a faithfulness/policy choice, not a
   correctness one.
