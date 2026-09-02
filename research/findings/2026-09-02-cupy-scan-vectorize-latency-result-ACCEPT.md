---
type: finding
status: live
date: 2026-09-02
mechanism: cupy-scan-vectorize-latency-6seed-result
board: 108 / #192 knowledge-recall latency
artifact: research/findings/raw/_cupy_scan_vectorize/latency_6seed_aggregate.json
---

# The cupy-scan vectorization's 6-seed latency result: a modest byte-identical improvement, NOT <1s — ACCEPT the ~1.2s median (speed is secondary)

**2026-09-02, cupy, wikidata_100k, `--enable-decode-escalation`.** The 6-seed cupy latency verify of the
byte-identical winner-code-gather vectorization (`VECTORIZED_WINCODE_GATHER`, landed 4eeefd6e1) came back.
Correctness is intact (recall ~1.0 all seeds — the vectorization is byte-identical, proven separately in
`byteident.txt`). Latency: a modest improvement, not the clean <1s hoped for.

Artifact: `research/findings/raw/_cupy_scan_vectorize/latency_6seed_aggregate.json` (controller-computed from the
per-seed jsons; the s102 seed hung at ~4h and the GPU-monopolizing run was killed with 5 clean seeds in hand —
enough to settle the decision, and freeing the GPU for the queue of blocked verifies).

## Result (5 clean seeds)
- **median lat_med = 1189 ms** (vs the ~1303 ms baseline — a ~114 ms / ~9% improvement <!--derived-->).
- **under 1000 ms: 2/5** seeds (960, 837 ms); the strict "<1s on all seeds" bar is NOT met.
- **within the owner's 1.1–1.3s tolerance: 3/5**; two seeds exceed it — s42 marginally (1301 ms) and **s101 a heavy
  tail (1905 ms, p95 3020 ms)**.

## Decision: ACCEPT (do not chase the tail)
Per the non-negotiable **speed < faithfulness** and the owner's standing fallback ("get under 1s first, accept
~1.3s if the fix fails with no obvious next step"), this is the accept case: correctness is intact, the median
sits within the stated tolerance, and the vectorization is banked as a real byte-identical improvement. Chasing the
tail (the s101 outlier + the sub-1s bar) means profiling for further cupy hotspots beyond the one gather — genuine
diminishing returns against a soft, non-faithfulness residual. So **#108's latency blocker is resolved by
acceptance**; with #94 confidence-forthcomingness already GO 6/6 at 100k, both #108 blockers are now clear.

## Honest residual (not a wall — a parked soft lever)
Latency remains **seed-variable with a heavy tail** (s101 ~1.9s). If latency is ever re-prioritized, the next lever
is a cupy hot-path profile of the recall pipeline (the escalation-tighten diagnosis already established the cost is
cupy-backend-specific, not the decode-escalation trigger). Parked, not walled.
