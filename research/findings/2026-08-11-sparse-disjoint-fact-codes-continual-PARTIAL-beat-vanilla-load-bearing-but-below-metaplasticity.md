---
type: finding
status: contributing
date: 2026-08-11
mechanism: sparse / DISJOINT fact codes (each fact routed to a low-overlap sparse subset of hidden units) against continual acquisition-at-scale forgetting
lane: H-memory / continual-learning
verdict: PARTIAL — disjoint sparse codes BEAT vanilla and the disjointness is LOAD-BEARING (vs overlapping codes), validating disjoint-codes-under-pressure in the continual lane too; but they do NOT beat metaplasticity and do NOT solve the oldest-fact overwrite. Disjointness is a real lever, complementary to (not a replacement for) consolidation.
seeds: [42, 43, 100, 101, 102]
runner: research/runners/_teacher_loop_sparse_fact_codes_derisk.py
artifacts:
  - research/findings/raw/_sparse_fact_codes/sparse_s42.json
  - research/findings/raw/_sparse_fact_codes/sparse_s43.json
  - research/findings/raw/_sparse_fact_codes/sparse_s100.json
  - research/findings/raw/_sparse_fact_codes/sparse_s101.json
  - research/findings/raw/_sparse_fact_codes/sparse_s102.json
instrument: N-sweep {16,32} facts, hidden 256, sparse code-size 6. Arms: vanilla, metaplastic (the prior best single mechanism), disjoint (facts routed to disjoint sparse unit-subsets), kwta_trained (k-winner sparse codes), kwta_shared (OVERLAPPING codes — the disjointness control), dense_readout (the dense lesion). SIM_BACKEND=numpy.
---

# Sparse DISJOINT fact codes for continual acquisition — a PARTIAL: disjointness is a real, load-bearing lever (beats vanilla + overlapping) but is BELOW metaplasticity and does not solve the oldest-fact overwrite

The Benna-Fusi NEGATIVE (`2026-08-11-benna-fusi-multitimescale-chain-...`) re-diagnosed the continual acquisition-at-scale
residual: the oldest fact is overwritten because later facts REUSE its synapses — a fact-CODE interference problem, not
a consolidation-timescale problem. It named **sparse / orthogonal fact codes** (disjoint synapse subsets) as the next
mechanism — the SAME "disjoint-codes-under-pressure" biology that just won in the source-monitoring (competitive-encoding)
and emergence (hetero-allocation) lanes. This de-risk tests it in the continual lane.

## Result — 6-seed sweep, 5 seeds completed (`research/findings/raw/_sparse_fact_codes/sparse_s*.json`)

<!--derived-->
Cross-seed mean frac_recalled / oldest-fact-acc (hidden 256, N to 32):

| arm | frac_recalled | oldest_fact_acc |
|---|---|---|
| vanilla | 0.537 | 0.150 |
| **metaplastic** (prior best) | **0.812** | **0.275** |
| **disjoint** (sparse, disjoint subsets) | **0.656** | 0.175 |
| kwta_trained | 0.644 | 0.175 |
| kwta_shared (OVERLAPPING — the control) | 0.156 | 0.000 |
| dense_readout (dense lesion) | 0.537 | 0.150 |

- **Disjoint codes BEAT vanilla** (0.656 vs 0.537) and — the earned tooth — the **disjointness is LOAD-BEARING**: the
  OVERLAPPING-code control (`kwta_shared`) collapses to 0.156 (far below vanilla), and the dense lesion lands exactly on
  vanilla. So it is the DISJOINTNESS, not sparsity per se, that helps. The runner's own `disjoint_beats_vanilla` = True.
  **This validates disjoint-codes-under-pressure as a real lever in the continual lane — a THIRD confirmation of the
  cross-lane convergence** (with source competitive-encoding + emergence hetero-allocation).
- **But it is BELOW metaplasticity** (0.656 < 0.812; `disjoint_beats_metaplastic` = False) and does NOT better-protect
  the oldest fact (disjoint 0.175 vs metaplastic 0.275; barely above vanilla 0.150). At low capacity (H=128, tighter
  code) sparse coding over-restricts and HURTS (<vanilla) — the disjoint benefit needs capacity headroom.

## Scope / honesty + next (per THE LAW — the capability stays OPEN)

<!--derived-->
NO-EXTERNAL-NEEDED: a quantitative comparison of two mechanisms in one lane, not a fundamental-limit claim; the negative
half (disjoint < metaplastic) redirects, it does not close.

- **The honest verdict:** disjoint sparse codes are COMPLEMENTARY to consolidation, not a replacement. Disjointness is a
  genuine, load-bearing lever (the convergence holds), but consolidation (metaplasticity) is the stronger single
  mechanism here, and NEITHER cleanly solves the oldest-fact overwrite (both leave it low: 0.175 / 0.275).
- **Named next mechanism:** COMBINE them — disjoint sparse allocation PLUS per-synapse consolidation (protect the few
  units a fact owns) — the two levers attack different failure modes (code interference vs weight erosion) and should
  compound; and/or capacity GROWTH / neurogenesis as N scales (the disjoint benefit needs headroom, which growth supplies).
- Runner-side, reuse-by-import of the metaplastic/continual machinery. NO `sim/` edit. Provenance: the build agent
  DEFERRED mid-exploration (backgrounded its config sweep + stopped); the coordinator recovered the runner, ran the clean
  6-seed at the working H=256 config (one seed's artifact did not write — 5-seed aggregate), and authored this finding.
