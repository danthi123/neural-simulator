# 160-concept sparse ensemble — multi-seed (42–46): integration robust, per-bridge seed-variant

## TL;DR

The 160-concept sparse-distributed G.20 ensemble (5 bridges × 32) was
hardened from seed-42 to **5 seeds (42–46)**:

- **Ensemble integration: 5/5 seeds PASS.** Cross-bridge memory
  (`apple is big` → `what is apple` returns `big` ranked **#1** at
  every seed) + exact-tag match + N-word sentence + role queries all
  pass at all 5 seeds. The integration layer is **robustly
  multi-seed validated.**
- **Per-bridge discrimination: seed-variant**, mean **98.1%**
  (785/800), range 93.8–100%. NOT a flat 100%.
- The per-seed failure count is **identical across all 5 bridges
  within a seed** (seed 43: every bridge 31/32; seed 46: every
  bridge 30/32) — hard multi-seed confirmation of the idx-12
  mechanism: all 5 bridges share that seed's `generate_sparse_patterns`
  set, so they fail on the same patterns. **Pattern-set quality, not
  architecture, is the lever.**

## Per-bridge top-1 by seed (5×32, sparsity 0.02)

| Seed | Total | % | Per-bridge | Note |
|---|---|---|---|---|
| 42 | 160/160 | **100.0%** | 32/32 ×5 | (validated 2026-05-15) |
| 43 | 155/160 | 96.9% | 31/32 ×5 | uniform 1 fail/bridge |
| 44 | 160/160 | **100.0%** | 32/32 ×5 | |
| 45 | 160/160 | **100.0%** | 32/32 ×5 | |
| 46 | 150/160 | 93.8% | 30/32 ×5 | uniform 2 fails/bridge |
| **Mean** | **785/800** | **98.1%** | — | 3/5 seeds @ 100% |

## Cross-bridge retrieval (`what is apple` → top associate)

| Seed | Result |
|---|---|
| 42 | big 779 via bridgeC_adj/apple_big (#1) |
| 43 | big **876** via bridgeC_adj/apple_big (#1) |
| 44 | big **649** via bridgeC_adj/apple_big (#1) |
| 45 | big **764** via bridgeC_adj/apple_big (#1) |
| 46 | big **840** via bridgeC_adj/apple_big (#1) |

`big` is the #1 associate at **every** seed — cross-bridge sparse
retrieval is seed-robust even when per-bridge discrimination dips
(seed 46 is the weakest per-bridge at 93.8% yet retrieves `big` at a
strong 840). The integration mechanism (engram tag-name aggregation
across bridges) is more robust than raw per-pattern discrimination.

## Interpretation (honest)

- **What is multi-seed validated:** the multi-bridge *integration* —
  cross-bridge memory, exact-tag match, N-word sentences, role
  queries. 5/5 seeds. This is the conversational-capability claim and
  it holds.
- **What is seed-variant:** raw per-bridge concept discrimination
  (93.8–100%, mean 98.1%). The variance is fully explained: all 5
  bridges in a run share `--seed`, so they share one
  `generate_sparse_patterns(32,2000,100,seed)` set; a seed with 1–2
  high-collision patterns fails those same 1–2 concepts in every
  bridge. Seeds 42/44/45 drew "clean" sets (100%); 43/46 drew sets
  with 1–2 unlucky patterns.
- This is the SAME mechanism as the 320-tier idx-12 gap
  (`2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md`), now
  confirmed across 5 seeds at the 160 tier. It is strong evidence
  that the flagged recovery (per-bridge distinct seeds +
  overlap-rejection in `generate_sparse_patterns`) targets the right
  lever — pattern-set quality, not the architecture.

## Not overclaiming

The headline is **"160 ensemble integration: 5/5 seeds PASS; per-bridge
98.1% mean (seed-variant 93.8–100%)"** — NOT "100% multi-seed". The
single-seed-42 100% was real but is not representative of per-bridge
discrimination across seeds. The integration robustness IS
representative and is the capability that matters for conversation.

## Files

- `research/runners/g20_sparse_160_multiseed.ps1` — the harness
- `research/findings/raw/g11_bg/g20_sparse_bridges_s{43,44,45,46}/` —
  per-seed bridges + JSON + per-seed demo logs
- Prior: `2026-05-15-G20-sparse-ensemble-160concept-end-to-end-SHIPPED.md`
  (seed 42), `2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md`
  (320 tier, same idx-pattern mechanism)
