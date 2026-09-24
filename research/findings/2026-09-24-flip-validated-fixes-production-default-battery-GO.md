---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: load-bearing
mechanism: production-default validation of three fixes switched ON by default (BRAIN_EPISODIC_STORE_VERIFY,
  BRAIN_PMEM_FACILITATION, BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE) on branch research/flip-validated-fixes @bd391aa31
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-23-flip-validated-fixes-production-default-validation-prereg.md
artifacts:
  - research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json
  - research/findings/raw/_load_bearing/_shards/flipdefaults-thin/aggregate.json
  - research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json
verdict: GO -- every pre-registered criterion passes (A1, A2, A3 on the adequate battery; T1, T2 on the thin battery). The three
  flips stand at the production default. Shipped thin headline 0.603 with robust core 15; adequate 0.949, robust core 24, identical to the all-fixes battery
  that also set the per-seed stabilizer.
---

# The three validated fixes hold at the production default: GO on both batteries

## Result

Two sharded batteries, 31 faculties x 6 seeds each (186 shards per battery), numpy backend, no fix flag set (each mechanism runs at
its new default), every job guarded by `tools/assert_flipped_defaults.py`. Aggregates:
`research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json` and
`research/findings/raw/_load_bearing/_shards/flipdefaults-thin/aggregate.json`; comparison
`research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json`. The thin baseline is the 2026-09-20 monolithic per-seed set
`research/findings/raw/_load_bearing/load_bearing_s42.json` and its five siblings.

<!--derived-->
| criterion | requirement | result |
|---|---|---|
| A1 | episodic-memory, prospective-memory, source-provenance-honesty each load-bearing 6/6, null clean, none UNRELIABLE | 6/6, 6/6, 6/6; no dirty or unreliable seed |
| A2 | no faculty's n_load_bearing below allfixes2 | no drop on any faculty |
| A3 | no incomplete faculty | none |
| T1 | thin robust core 14 all 6/6; no faculty below its 2026-09-20 count | all 6/6; no drop |
| T2 | source-provenance-honesty 6/6 under thin probes (baseline 4/6) | 6/6 |

- Adequate battery: robust core 24, union 25, mean fraction 0.9487, the same as allfixes2. allfixes2 also set
  BRAIN_PMEM_OP_STABILIZER (a per-seed lookup table over exactly these seeds); this battery did not, and prospective-memory still
  reads 6/6. The pre-registered open question (seed 44 without the stabilizer) resolves in favour of the fix holding alone.
- Thin battery (the shipped-headline probe set): robust core 15, union 16, mean fraction 0.6026. The 2026-09-20 thin baseline
  read robust core 14, mean 0.590 <!--derived-->; the only per-faculty change is source-provenance-honesty, 4/6 to 6/6.

## Scope and residuals

- The batteries measured revision bd391aa31. The merge into main carries 160 later main commits; the next battery (B2a, tonight)
  measures the merged production default itself.
- Latency is not measured by either battery: the episodic store adds at least one recall read per stored topic.
- Shards were run on AWS (r7i.4xlarge), the mini-PC pool and the local box; all are numpy and deterministic.

## Honesty

"Load-bearing" means lesioning the faculty's spiking contribution changes the chat reply, with a clean null control. Functional
read-outs only.
