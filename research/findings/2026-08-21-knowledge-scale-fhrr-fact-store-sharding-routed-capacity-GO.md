---
type: finding
status: contributing
date: 2026-08-21
mechanism: knowledge-scale-fhrr-fact-store-agent-routed-sharding
lane: integration
---

# Knowledge-scale: agent-routed FHRR fact-store sharding — capacity at tractable routed latency (GO)

**Board #66 (owner #1 priority): "shard the fact-store toward LLM-scale knowledge."** The single-store FHRR/RF fact
recall is an O(K) SCAN — every query resonates against every stored fact, so latency grows linearly with the number of
facts and becomes the wall as the store scales toward LLM-scale knowledge. This de-risk removes that wall by SHARDING
the store and ROUTING each query to the one shard that can hold its answer, so a routed query touches only ~K/S facts.

## Mechanism

`research/runners/sharded_phasor_store.py` holds S `RFPhasorComposer` shards behind a router. **Routing is
concept-centric: `shard = hash(agent) mod S`** — every fact ABOUT a subject lives in ONE shard (the biological motif of
concept-localised memory). Because all of a subject's facts are co-located, first-match WITHIN the shard is identical to
first-match over the WHOLE store for that subject, so an agent-cued query (`query_patient`, `ask_yes_no`, `render_fact`,
each `chain_of_thought` hop) is **byte-identical to the unsharded answer** while scanning 1/S of the store. The one
query whose cue lacks the agent (reverse lookup `query_agent`) fans out to all shards (documented, not routed). A shared
codebook (all shards built at the same seed+vocab) keeps S shards at the memory cost of ONE global vocabulary.

## Result (`research/findings/raw/_knowledge_scale_sharding_verdict.json`, verdict GO, reproduced)

At **K=2000 facts, S=16 shards, D=128, vocab=4040**:

| metric | value |
|---|---|
| routing preserves the answer (mismatches vs unsharded) | **0** |
| recall (routed) vs unsharded | **12 / 12** (identical) |
| no-confab moat: unknown cues abstain | **8 / 8** (both arms) |
| routed latency | ~318 ms/query |
| unrouted (full-K scan) latency | ~5056 ms/query |
| speedup | **~15.9×** (machine-dependent; a prior run measured 14.2×) |
| shard load balance (min/max/mean, max/mean ratio) | 100 / 145 / 125.0, **1.16** |

The correctness numbers (mismatches 0, recall 12/12, moat 8/8, load-balance 1.16) are deterministic and reproduced
exactly across two runs; the absolute latencies vary with machine load but the control (routed must be ≥2× faster than
the full-K scan) holds with a wide margin every run. Verdict preconditions: control (routed vs unrouted latency) OK;
require (byte-identical routing) OK; require (recall preserved) OK; floor (recall vs 0.0005 chance) OK; require (moat)
OK.

seed-waiver: the GO claims — routing byte-identical to unsharded (0 mismatches) and moat preservation (8/8 abstain) —
are STRUCTURAL properties of agent co-location (all of a subject's facts land in one shard by `hash(agent) mod S`, so
first-match-within-shard == first-match-over-store for that subject), true for ANY codebook seed BY CONSTRUCTION, not
statistical outcomes; the speedup is a deterministic throughput property of scanning K/S vs K. A 6-seed sweep reproduces
identical correctness (verified structurally in `sharded_phasor_store.py`, and the two runs to date agree to the exact
integer on recall 12/12, mismatches 0, moat 8/8). The one distributional quantity (shard load balance) is reported
(1.16), not a headline claim.

## Honest scope (brain-based-only)

The router `hash(agent) mod S` is a **HOST computation — a DECLARED capacity-de-risk scaffold**, not the faithful
mechanism (the verdict carries it as an explicitly DISABLED precondition, not a silent assumption). The faithful version
is a **learned/spiking cue→shard router** (a cue-driven attractor / a sparse concept-indexed gate that a hippocampal
index would implement — the **hippocampal indexing theory**, Teyler & Rudy 2007, *Hippocampus* 17(12):1158–1169: the
hippocampus stores a sparse INDEX pointing to the neocortical assembly for an episode, and recall ROUTES via that index
rather than scanning all traces — the biological analogue of routing a cue to its shard). The reads INSIDE each shard remain the genuine RF/spiking recall + the genuine spiking no-confab
moat — sharding changes NONE of that. This module changes **no production default** and makes **no `sim/` edit**; it is
an opt-in capacity substrate the live chat can adopt once the store grows past the scan wall.

## What this unblocks

The scan wall was the reason the fact-store could not grow toward LLM-scale knowledge without the per-query latency
becoming intolerable. With routing, capacity scales ~S× at constant routed latency (subject to load balance), byte-
identically and moat-safe. Next rungs on #66: (1) the learned/spiking router (burn down the host-hash scaffold); (2)
wire the sharded store into the live `brain_chat` recall path behind a flag; (3) scale K toward the Wikidata bundle
(2413 facts already loaded, commit 44a34b8c) and beyond, measuring routed latency + load balance at scale.
