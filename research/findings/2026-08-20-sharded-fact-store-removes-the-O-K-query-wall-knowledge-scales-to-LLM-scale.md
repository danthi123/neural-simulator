---
type: finding
status: live
date: 2026-08-20
mechanism: semantic-fact-store-capacity
lane: integration
seeds: [0]
seed-waiver: An ENGINEERING capacity de-risk (a query-latency + recall + moat measurement of a sharded data
  structure at matched K), not a stochastic effect size — the load-bearing evidence is the measured speedup +
  byte-identical answers + the per-unit-cost scaling, single deterministic build per K.
instrument: research/runners/_knowledge_scale_sharding_verdict.py — measures a ShardedPhasorStore vs an unsharded
  RFPhasorComposer at matched K: routed query latency, recall, moat abstain, and answer-mismatches, with a
  tools.verdict.Verdict.
runner: research/runners/_knowledge_scale_sharding_verdict.py
external: NO-EXTERNAL-NEEDED — an in-repo capacity de-risk of the existing VSA fact-store; the routing motif
  (concept-centric = a hippocampal index / cortical concept map) is biologically grounded, the measurement is internal.
artifacts:
  - research/findings/raw/_knowledge_scale_sharding_verdict.json
---
# Sharding removes the O(K) query wall — the brain's knowledge can scale to LLM-scale at sub-second latency, recall + moat byte-identically preserved

Artifact: research/findings/raw/_knowledge_scale_sharding_verdict.json (GO).

**One line.** The owner's #1 priority: teach the sim-brain the fundamental knowledge an LLM has, then interact daily.
The VSA fact-store's REPRESENTATION was already fact-count-independent (recall stays perfect as facts pile up); the ONLY
binding limit was the O(K·D) linear query scan (~5 s at 2.4k facts, minutes at LLM-scale). **Sharding eliminates it: a
routed query's latency depends only on the SHARD size, not the total fact count — so it is sub-second at ANY K — with
recall and the no-confab moat byte-identically preserved.** GO.

## The build (`ShardedPhasorStore`, reuse-by-import, NO `sim/` edit)
S independent `RFPhasorComposer` shards, **routed by `hash(agent) mod S`** (concept-centric — all facts about a subject
co-located, the hippocampal-index / cortical-concept-map motif), over a **shared codebook** (one concept/role dict for
all shards → footprint is codebook + facts, not S×codebook). Same conversational API. An agent-cued query
(query_patient / ask_yes_no / render_fact / chain_of_thought) is **O(1) route + O(K/S) scan and BYTE-IDENTICAL to
unsharded** (a subject's facts never split, so first-match-in-shard == first-match-overall). The one exception, the
reverse lookup `query_agent(action,patient)`, fans out to all shards (S-way parallelizable) or needs a 2nd
patient-routed index — flagged honestly.

## Measured (numpy CPU, D=128 — measurements, not extrapolations)
<!--derived-->
- **Verdict run (K=2000, S=16):** unrouted 4149 ms/query → routed **291 ms (14.2×)**; **0 answer-mismatches** vs
  unsharded; recall 12/12 both; moat 8/8 abstain both → GO.
- **REAL Wikidata 2413 facts:** unsharded 5008 ms → S=32 **221 ms (22.6×)**; recall 40/40, moat 15/15; load-balance 1.35.
- **20k synthetic (S=128, V=40060):** footprint **61.5 MB** (41 MB shared codebook + 20.5 MB composites, ~1 KB/fact);
  routed **974 ms**; recall 25/25; moat 12/12; ~46× vs the derived unrouted 44.8 s.
- **Latency scaling** (fixed V): linear at 2.24 ms/fact UNROUTED — routing replaces total-K with shard-size m=K/S.
- **Vocab / cleanup accuracy: 1.000 to 100k concepts** at D=128 (and D=256) — vocab is NOT the wall at D=128; a
  V-independent cue-match optimisation removes the O(V·D) term from all but the single answer-decode (5.5 ms @ V=100k).

## The honest ceiling → LLM-scale (derived from the measured unit costs, labelled as such)
Routed latency depends only on shard size m=K/S; pick S so m≈125-250 → sub-second (measured 291 ms @125, 566 ms @250) at
ANY K; S is unbounded. **K=100k:** S≈400-800, footprint ~200 MB, one-time build ~33 min (embarrassingly parallel across
shards), routed query ~300-566 ms. **K=1M:** S≈4000-8000, footprint ~2 GB, query unchanged. **Vocab→~1M concepts:** raise
D 128→256 (measured clean at 100k for both), linear cost. Per-query floor = the RF resonate (208 steps × 2mD neurons);
further levers = more shards, GPU cudagraph megakernel (needs the fused path to beat per-op launch overhead; unmeasured,
flagged), parallel shard fan-out.

## Live-chat cap (k_max=32) — the plan (scoped, not wired)
`one_brain_composer.py:664` literally says "shard or raise k_max"; raising it re-hits the O(k_max) wall on one bridge.
Plan (the biological hippocampal-buffer / cortical-semantic split): keep the k_max=32 co-resident composer as the
active-conversation WORKING-SET buffer, add a `ShardedPhasorStore` as the LONG-TERM semantic store — `store` routes to
an LTM shard, `query` checks the buffer then the routed shard (~291 ms @ m=125). Additive (~1 field + call-routing on
`BrainConversationalAgent`, no `sim/` edit).

## Honest scope
The recall + no-confab moat inside each shard are the genuine RF reads (measured preserved, byte-identical). The
**`hash(agent)` router is a declared HOST scaffold** for the capacity de-risk — the faithful version is a learned/spiking
cue→sub-population router (in the Verdict's disabled scope). The 20k/verdict/latency facts are synthetic; the 2413-fact
run is genuine Wikidata. One representation-level recall miss at the V=80k probe (39/40, same fact both methods); all
other scales perfect. NEXT: build the sharded LTM store at 100k+ real facts + wire the buffer/LTM split into the live
chat (lifting the 32-fact cap) — the enabler for the daily-teachable, knowledge-rich, learning brain. (Agent-built;
parent verified the GO + 14.2× + 0 mismatches + recall/moat-preserved from the artifact.)
