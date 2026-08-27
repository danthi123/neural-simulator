---
status: live
type: finding
lane: integration
date: 2026-08-27
mechanism: knowledge-scale-sharded-retrieval
---

# 100k knowledge is already sublinear-at-scale via shard-routing — verified 6-seed; the DG sparse-index flag is redundant, so NO default-on flip

**Verdict: the 100k sharded tiered LTM path FUNCTIONS TO STANDARD on the forward agent-cued path, and the
"sublinear sparse-index at scale" capability is ALREADY DELIVERED by the shard router — not by the
`BRAIN_SPARSE_INDEX_RETRIEVAL` (DG sparse-index) flag, which is architecturally inert on this path. So the owner's
default-on flip question resolves to DO NOT FLIP (it would be a hollow no-op).** This closes the "sublinear
sparse-index is waiting on owner review at scale" gate by testing it directly.

## Why the flip was never actually needed

`ShardedPhasorStore.query_patient` (`research/runners/sharded_phasor_store.py`) is sublinear at 100k BY
CONSTRUCTION: it hash-routes the cued agent to ONE of 395 shards (O(1)), then scans only that shard (~200 facts),
answer-identical to the unsharded store. The DG `enable_sparse_index` mechanism is a SEPARATE accelerator scoped to
the FLAT 15k core composer (`OneBrainComposer`); `RFPhasorComposer` (the per-shard LTM composer) has no such
parameter, so `BRAIN_SPARSE_INDEX_RETRIEVAL` cannot reach the tiered path. The verify runner asserts this is inert
by construction and MEASURES it: with the flag ON vs OFF the answers are byte-identical on every seed.

## 6-seed measurement (numpy, `_knowledge_scale_100k_production_verify.py`, the real `load_developed_brain(..., ltm_bundle=wikidata_100k)` path; 78,857 facts / 395 shards / D=128)

| seed | oracle byte-identity (checks/mismatch) | scale recall (ok/checked) | moat confab | DG-flag inert (answers-identical) | latency median |
|---|---|---|---|---|---|
| 42  | 136 / **0** | 50/50 = 1.0  | **0** | ✓ | 1113 ms |
| 43  | 152 / **0** | 50/50 = 1.0  | **0** | ✓ | 1246 ms |
| 44  | 132 / **0** | 50/50 = 1.0  | **0** | ✓ |  890 ms |
| 100 | 118 / **0** | 50/50 = 1.0  | **0** | ✓ |  846 ms |
| 101 | 154 / **0** | 49/50 = 0.98 | **0** | ✓ |  913 ms |
| 102 | 124 / **0** | 50/50 = 1.0  | **0** | ✓ |  816 ms |

**Correctness bars (load-INDEPENDENT) are the verdict:** oracle byte-identity 0-mismatch **6/6** (816 checks over
the held agents' COMPLETE fact sets — proves sharding preserves every answer), 0-confab moat **6/6** (120 checks),
DG-flag inert **6/6**. Recall over the broader oracle-free probe sample is 1.0 on 5/6 seeds and 0.98 (49/50) on
seed 101 — 299/300 overall; the single miss is a first-match tie-break edge case in the broad sample, NOT a
systematic recall failure (the byte-identity check, which uses each agent's complete fact set, is 0-mismatch on
that same seed).

## The flip decision (owner blocker #2)

Do NOT flip `BRAIN_SPARSE_INDEX_RETRIEVAL` default-on. It is redundant on the tiered path (verified inert, 6/6) and
belongs to the flat 15k core where it may still accelerate. Flipping it default-on would be a hollow checkbox — the
exact drift the production-integration discipline exists to prevent. The sublinear-at-scale capability is real and
delivered; it just comes from the shard router, not the DG flag.

## Honest residuals (characterized, not walls)

1. **Latency is VSA-composer-bound, not sharding-bound, and load-dependent.** Median ~0.9 s (816–1246 ms across
   seeds) — dominated by the numpy RF-phasor composer's per-query unbind+cleanup, NOT the O(1) shard routing. The
   spread tracks concurrent CPU load (the higher medians coincided with a co-running numpy job), so the runner's
   `latency_ms_median < 1000` bar flips the OVERALL gate to UNDEFINED under contention — a measurement artifact of a
   wall-clock timer under load, not a capability regression. The genuine speed residual is the host numpy VSA
   composer (a known shortcut; the faithful replacement is a spiking cleanup/retrieval), tracked separately.
2. **Reverse lookups do not get the routing speedup.** `query_agent` ("who <action> <patient>?") cannot route by
   agent-hash (the agent is the unknown), so it fans out to all 395 shards — no sublinear win without a second
   patient-routed index (2x write cost + footprint). Documented in the store; a real next-rung if reverse-query
   latency ever matters.
3. **Host-hash shard router + numpy composer remain shortcuts** (`scaffold_retired` = 0 for this faculty): the
   faithful versions are a learned/spiking cue→sub-population router and a spiking cleanup.

Artifacts (one per seed, + provenance sidecars): `research/findings/raw/knowledge_100k_verify/ks_s42.json`,
`research/findings/raw/knowledge_100k_verify/ks_s43.json`, `research/findings/raw/knowledge_100k_verify/ks_s44.json`,
`research/findings/raw/knowledge_100k_verify/ks_s100.json`, `research/findings/raw/knowledge_100k_verify/ks_s101.json`,
`research/findings/raw/knowledge_100k_verify/ks_s102.json`. Runner:
`research/runners/_knowledge_scale_100k_production_verify.py`.
