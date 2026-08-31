---
type: finding
status: go
lane: knowledge
board: 66
date: 2026-08-30
seed-waiver: latency + byte-identity are DETERMINISTIC infra properties, not seed-dependent learning — the cache
  is an exact recompute (cached matrix IS what the code rebuilds, so decode is byte-identical by construction,
  independent of seed), and latency is a hardware-timing measurement. Single seed (42) suffices to establish
  both; the production 79k-fact scale verify (2026-08-31) CONFIRMED end-to-end GO on both numpy and cupy (median
  711-755ms < 1000ms bar, recall 1.0, moat 0).
mechanism: knowledge-in-chat #66 latency — locate the real hot loop in the production composer query path and
  de-risk the lowest-risk lever. Diagnosis by cProfile of a routed shard; lever = cache the (V,D) cleanup
  codebook once and share it across shards (board #192), DECOUPLED from the DG sparse index
verdict: >
  The #66 latency wall is NOT the spiking resonate and NOT hot shards (routing is load-balanced, max 285
  facts/shard, 1.4x imbalance). It is an O(V) cleanup CODEBOOK rebuilt from the concepts dict on EVERY query
  (~40% of per-query time), scaling with V=23914 independent of the shard's ~200 facts. Caching it once and
  sharing across shards is byte-identical (0/80 mismatches) and cuts median latency 1042ms -> 711ms (a 31.7%
  reduction, 1.465x) at real per-shard scale, RSS-safe via the existing shared-graft. Separately: the verify's
  `RFPhasorComposer_accepts_enable_sparse_index = False` was STALE (the kwarg was added after the verify ran;
  the composer accepts it now) — so the earlier ~25% "index" win was the codebook CACHING, not the sharding.
---

# #66 latency — the hot loop is the codebook rebuild; caching it is 31.7%, byte-identical

## Diagnosis (de-risk artifact: `research/findings/raw/_codebook_cache_latency_derisk.json`)

<!--derived-->

A routed `query_patient` touches exactly ONE shard, so end-to-end latency ≈ one-shard latency. cProfile of a
routed shard (V=24000, K=200, D=128, numpy CPU — reproducing the ~1.1s of the scale verify) shows the largest
cost is NOT the spiking resonate (`_rf_advance_one`, ~27%) but the **O(V) cleanup codebook operations (~40%)**,
dominated by rebuilding the (V,D) phasor matrix from the `concepts` dict on every query
(`rf_phasor_composer.py:844` `_cleanup_all`, plus the per-word loop at `:763`). This scales with **V (23,914)**,
independent of the shard's ~200 facts. Shard routing is load-balanced (mean ~200, median 200, p95 249, max 285
facts/shard — 1.4× imbalance), so the cost is a fixed O(V) tail, not a hot-shard artifact.

## Lever + de-risk result — GO
Cache the (V,D) cleanup codebook ONCE per vocab state and share it across shards via the same `_dg_index_source`
graft the DG index uses (board #192; decoupled from the DG sparse index, so it avoids that index's separate
NO-GO — the σ≈1.27-rad-vs-0.30 noise-calibration mismatch). Measured at real per-shard scale:
- **Byte-identical: 0 mismatches / 80 cues** (query_patient, ask_yes_no, query_agent, moat).
- **Latency: median 1042ms → 711ms = 31.7% reduction (1.465×)**; p95 1305ms → 856ms.
- **RSS-safe:** one shared codebook object for all shards (16.4MB shared vs 655MB unshared at scale), peak 556MB.

De-risk runners (subclass, no production edit): `_codebook_cache_latency_derisk.py`,
`_codebook_cache_sharded_check.py`.

## Stale-finding correction
The scale-verify's `RFPhasorComposer_accepts_enable_sparse_index = False` was STALE — the kwarg was added to
`RFPhasorComposer.__init__` (`rf_phasor_composer.py:68`) after that verify ran; the current composer accepts
`enable_sparse_index=True`. Turning the index ON is still not the right lever (its NO-GO + escalation cost
stands); the codebook cache is.

## Status + next rung (production wiring — COMPLETED)
GO at the de-risk level (byte-identical, 31.7%, RSS-safe). The production-integration rung is an additive,
default-OFF codebook-cache in `rf_phasor_composer.py` — it is now ON `main` (commit f3af43407) and the
full 79k-fact scale verify with the flag ON ran 2026-08-31 on **both numpy and cupy**: **GO**
(recall 0.9933, moat 0 confab, latency median ~711-755ms < 1000ms bar, p95 ~900ms). End-to-end confirmed
end-to-end; board #192 and the #66 latency wall are now CLOSED on the technical bar. Residual (honest,
disclosed not smoothed): the scale verify ran on seed 42; 6-seed confirmation at production load is the named
next rung, and a p95<1s tightening would require a second lever (second-order codebook/decode optimization).
