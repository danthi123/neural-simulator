---
type: finding
status: contributing
date: 2026-08-11
mechanism: COMBINING disjoint sparse fact-codes + per-synapse metaplastic consolidation (the two load-bearing but individually-insufficient continual-learning levers) — does the combination COMPOUND?
lane: H-memory / continual-learning
verdict: NEGATIVE (no compounding) — combined ~= consolidation alone (0.854 vs 0.839); disjoint adds nothing on top of consolidation, which dominates. The two levers do not stack; the acquisition-at-scale residual is not solved by combining them.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_teacher_loop_combine_disjoint_consolidation_derisk.py
artifacts:
  - research/findings/raw/_combine_disj_consol/combine_s42.json
  - research/findings/raw/_combine_disj_consol/combine_s43.json
  - research/findings/raw/_combine_disj_consol/combine_s44.json
  - research/findings/raw/_combine_disj_consol/combine_s100.json
  - research/findings/raw/_combine_disj_consol/combine_s101.json
  - research/findings/raw/_combine_disj_consol/combine_s102.json
instrument: N-sweep {16,32} facts, hidden 256, sparse code-size 5. Arms: vanilla, consolidation (per-synapse metaplastic), disjoint (sparse disjoint fact-codes), combined (both), overlap_combined (OVERLAPPING codes + consolidation — the disjointness control). SIM_BACKEND=numpy.
---

# Combining disjoint fact-codes + metaplastic consolidation does NOT compound (6-seed) — combined ~= consolidation alone; the two continual levers do not stack

The sparse-codes finding (`2026-08-11-sparse-disjoint-fact-codes-continual-PARTIAL...`) named COMBINING the two
individually-load-bearing-but-insufficient continual levers — disjoint sparse fact-codes (attacks code interference) +
per-synapse metaplastic consolidation (attacks weight erosion) — as the next step, on the hypothesis they attack
DIFFERENT failure modes and should COMPOUND. This de-risk tests that. They do not compound.

## Result — 6 seeds (`research/findings/raw/_combine_disj_consol/combine_s*.json`)

<!--derived-->
Cross-seed mean frac_recalled / oldest-fact-acc (hidden 256, N to 32):

| arm | frac_recalled | oldest_fact_acc |
|---|---|---|
| vanilla | 0.542 | 0.208 |
| **consolidation** (metaplastic) | **0.839** | **0.533** |
| disjoint (sparse) | 0.667 | 0.142 |
| **combined** (both) | **0.854** | 0.500 |
| overlap_combined (OVERLAPPING + consolidation) | 0.219 | 0.083 |

- **Combined ≈ consolidation alone** (0.854 vs 0.839 frac — within noise; 0.500 vs 0.533 oldest). Adding disjoint sparse
  allocation on top of consolidation buys **nothing** — consolidation dominates, and it already does what the
  combination does. So the hypothesis that the two levers attack orthogonal failure modes and COMPOUND is **not**
  supported at this scale.
- **Disjointness is real but subordinate:** disjoint alone (0.667) beats vanilla (0.542) and the OVERLAPPING-codes
  control collapses (0.219, far below vanilla) — so disjoint sparse codes ARE a load-bearing property WHEN they carry
  the recall; but under consolidation they are not the bottleneck, so making them disjoint adds nothing.

## Scope / honesty + next

<!--derived-->
NO-EXTERNAL-NEEDED: a quantitative non-compounding result within one lane; it redirects, it does not close the capability.

- **The honest read:** consolidation is the dominant continual lever; disjoint sparse allocation does not stack usefully
  on top of it at this scale. The acquisition-at-scale residual (oldest-fact protection, frac→1/N as N grows) is NOT
  solved by combining the two mechanisms we have.
- **Where this points:** with metaplasticity/consolidation, chains, and disjoint codes all mapped (each load-bearing,
  none sufficient, no compounding), the continual residual looks less like a plasticity-rule problem and more like a
  CAPACITY / representational one — converging with the emergence-engine stream-language negative
  (`2026-08-11-emergence-engine-stream-language-...`): the substrate MEMORISES but does not ABSTRACT/generalise. The
  next mechanism is likely capacity growth (neurogenesis) and/or the **variable-binding working-memory** arc both lanes
  converged on (a persistent-activity slot that latches a variable), not another consolidation/coding variant.
- Runner-side, reuse-by-import; NO `sim/` edit. Coordinator-recovered + 6-seed-run from a deferred agent.
