---
type: finding
status: partial
lane: knowledge
board: 66
date: 2026-08-29
mechanism: knowledge-in-chat at LLM-ish scale (#66) — production verify of the sharded phasor fact-store at
  ~79k facts, measuring recall correctness, exact-value oracle match, and per-query latency through the
  production composer path
verdict: >
  Recall is SOLVED at scale; latency is the characterized wall. At 78,857 facts / 395 shards the production
  path recalls at rate 1.0 (50/50 checked) and matches the oracle exactly (0 mismatches on exact-value
  compare), but per-query
  latency is ~1.1s median / ~2.2s p95 (inertness path up to ~3.5s p95). The sparse-index that would cut this
  is NOT wired into the production composer (RFPhasorComposer_accepts_enable_sparse_index = False). So #66's
  remaining frontier is not correctness — it is the seconds-scale latency, and the sparse-index/codebook path
  is where it must be closed.
---

# Knowledge-scale #66 — recall solved at 79k facts; latency is the wall

## What ran
A production verify of the sharded phasor fact-store at scale, exercised through the live composer path.
Artifact: `research/findings/raw/_knowledge_scale_100k_production_verify.json`.

## Result

<!--derived-->

Bundle: **78,857 facts across 395 shards**.
- **Recall correct at scale:** recall rate 1.0 (50/50 checked); the flag-set/flag-unset inertness paths both
  recall at rate 1.0 (20/20 each).
- **Oracle exact-value compare: 0 mismatches** (15-agent / 68-fact oracle) — recalled values equal the
  oracle exactly (n_mismatches=0).
- **Latency is the wall:** scale-battery latency ~1.1s median, ~2.2s p95 (70 samples); the inertness path is
  slower still (~1.8-1.9s median, up to ~3.5s p95).
- **Sparse-index arch-check read False** (`RFPhasorComposer_accepts_enable_sparse_index = False`) in this
  verify — but see the correction below.

> ⚠️ CORRECTION (2026-08-30): that arch-check was STALE — the `enable_sparse_index` kwarg was added to
> `RFPhasorComposer.__init__` (`rf_phasor_composer.py:68`) *after* this verify ran, so the composer accepts it
> now. And the sparse index is NOT the right latency lever anyway: profiling
> ([`2026-08-30-knowledge-66-latency-hot-loop-is-codebook-rebuild-cache-31pct-byte-identical.md`](2026-08-30-knowledge-66-latency-hot-loop-is-codebook-rebuild-cache-31pct-byte-identical.md))
> shows the hot loop is an O(V) codebook rebuilt every query; caching it (board #192) cuts median latency 31.7%
> byte-identically, decoupled from the DG index.

## Interpretation
Knowledge-in-chat is *correct* at LLM-ish scale — recall and exact oracle match hold at ~79k facts. The blocker for
a usable knowledge-rich chat is the seconds-scale per-query latency of the 395-shard scan. This matches the
prior #66 record: the DG sparse-index is default-off (a NO-GO from a noise-calibration mismatch, not geometry)
and the ~25% latency win seen earlier was a codebook-caching side-effect, not the sharding — i.e. the latency
lever lives in the codebook/scan path (board #192: decouple codebook caching from the DG index).

## Status
Banks the scale verify: correctness done, latency characterized as the residual. The latency lever is under
active investigation (which hot loop dominates the 395-shard scan, and the lowest-risk way to cut it).
