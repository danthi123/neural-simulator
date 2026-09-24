---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: memory (knowledge scale; D6 fact store capacity)
mechanism: ca3-superposed-fact-attractor -- capacity law of a shared-synapse CA3 fact store (DG pattern separation,
  perforant + recurrent completion, k-WTA readout)
seeds: [42, 43, 44, 100, 101, 102]
instrument: the runner's own `--aggregate` combine mode (`research.runners.ca3_superposed_fact_attractor.aggregate`),
  which reproduces the pre-registered gate math exactly (G1-G9, per-seed pass/fail, MIN_SEEDS=5 except G6 at 6/6,
  an UNDEFINED seed counted as FAIL never a pass) -- no scoring logic outside the runner was used
prereg: research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md (e88f0a339;
  AMENDMENT 2026-09-24 at 3b43837fa/50e5f87db -- corrected the k_fit to a per-seed marginal -- both are ancestors
  of the commit the grid ran under, ab2adcf51)
artifacts:
  - research/findings/raw/_ca3_superposed_fact_attractor/grid/*.json (54 arm x seed result files, 9 arms x 6 seeds,
    plus the runner's own aggregate.json -- 55 files total; regenerable byte-for-byte from the 54 inputs via
    `python -m research.runners.ca3_superposed_fact_attractor --aggregate research/findings/raw/_ca3_superposed_fact_attractor/grid`)
verdict: GO 6/6 -- all 9 pre-registered gates (G1-G9) pass on 6 of 6 seeds (G6 integrity required 6/6, got 6/6; the
  rest required >=5/6, all got 6/6). Zero missing arm/seed cells, zero UNDEFINED gate outcomes. k_fit (the
  per-seed MARGINAL capacity-per-recurrent-synapse, corrected form) = 0.130, inside the pre-registered 0.1-0.3
  band. Provenance note: the 54 grid artifacts carry a top-level `runner`/`config` key (passes
  gates/artifact_provenance) but have NO `.prov.json` sidecar from the automatic door -- a real gap in
  research/runners/__init__.py for directory-style `--out` runners, flagged separately, not a defect in these
  results (pool1's own runs.jsonl independently confirms all 36 of its shards ran as
  `-m research.runners.ca3_superposed_fact_attractor --arm <arm> --seed <seed> --out .../grid` under revision
  ab2adcf51, and the other seeds' shards match by content/config/backend across every file checked).
---

# CA3 superposed fact-store capacity law: GO on all 9 gates, 6/6 seeds

Scored against `research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md` (as
amended 2026-09-24), using the runner's own `aggregate()` combine mode -- no criteria or thresholds outside the
prereg were applied.

## Completeness check (before any scoring)

54/54 arm x seed cells present under `research/findings/raw/_ca3_superposed_fact_attractor/grid/` (9 arms:
`sparse_dg`, `sparse_dg_c2`, `sparse_dg_recx2`, `dense_nodg`, `sparse_dg_hub`, `dense_nodg_hub`,
`sparse_nodg_hub`, `sparse_dg_c2_hub`, `sparse_dg_bounded_hub` x seeds 42/43/44/100/101/102), reproduced by
`research/findings/raw/_ca3_superposed_fact_attractor/grid/aggregate.json` (the runner's own `--aggregate`
output, re-generated in this worktree with a live `.prov.json` sidecar). `aggregate()`
reports `missing: []`. Every file parses and carries the expected `config`/`backend`/`runner`/`summary` shape;
`sha_facts` (the fact-set hash) is present in each, so this is not the disclosed non-evidence dev-seed-7 material
(that directory is untouched and separately marked NON_EVIDENCE). `bash tools/pool_sync.sh` was run once; it
pulled 0 new files for this family (the grid was already fully synced locally) and confirmed on pool1 directly
that the isolated revision `ab2adcf51...` -- the commit the pool dispatch log cites for every `ca3_superposed`
line -- is a descendant of both amendment commits (`3b43837fa`, `50e5f87db`) and is itself already merged into
`main`.

## Gate table (runner's own `aggregate()`, 6/6 seeds each)

| gate | criterion | result (median across seeds unless noted) | pass |
|---|---|---|---|
| G1 learns | recall(P=50)>=0.9 AND recall(P=200)>=0.9, `sparse_dg` | 1.0 / 1.0 on every seed | 6/6 |
| G2 cliff | recall at largest P <=0.2, all 8 unbounded arms | 0.0 on every arm/seed pair (instrument valid) | 6/6 |
| G3 companion | P50(`sparse_dg_hub`)/P50(`dense_nodg_hub`) >= 1.5 | 3.73 (range 3.48-4.04) | 6/6 |
| G4 capacity law | P50(`sparse_dg_c2`)/P50(`sparse_dg`) in [1.4,3.0] | 2.01 (range 1.79-2.10) | 6/6 |
| G5 recurrent load-bearing | P50(rec_zero) and P50(rec_shuffle) <= P50/1.2, `sparse_dg` | rec_zero ratio 0.64-0.72xP50_intact; rec_shuffle ratio 0.37-0.42xP50_intact -- both well under the 1/1.2=0.833 bar | 6/6 | <!--derived-->
| G6 cost/mem, INTEGRITY | latency ratio (max P / P=50) <=2.0 AND synapse bytes constant | ratio ~1.0-1.5, bytes identical at every checkpoint on every seed | 6/6 |
| G7 palimpsest | recent-100 recall @P=50000: bounded>=0.5, unbounded<=0.2 | bounded 0.75-0.88, unbounded 0.0-0.01 | 6/6 |
| G8 shared crosstalk | log-log slope of d' vs P in [-0.8,-0.2] | -0.483 (range -0.488 to -0.481) | 6/6 | <!--derived-->
| G9 hub limit not synaptic | P50(`sparse_dg_c2_hub`)/P50(`sparse_dg_hub`) < 1.4 | 1.12 (range 1.10-1.14) | 6/6 |

G9 shares its code path with G4 (both a P50 ratio against a doubled-fan-in arm) and the prereg says not to
headline it unless G4 also passes on the same seeds -- G4 passes 6/6 here, so G9 is reported without that
caveat.

## G5 side quantity (descriptive, not gated): the recurrent edge's linear share of capacity

`attributable_linear_frac = (P50_intact - P50_rec_zero) / P50_intact`, per seed: 0.361, 0.329, 0.276, 0.347, <!--derived-->
0.326, 0.335 (median 0.331). Consistent with the disclosed dev-seed-7 number (~25%, non-evidence) and confirming <!--derived-->
the prereg's own framing: most of this store's capacity is perforant -> CA3 -> readout heteroassociation: the
recurrent attractor is load-bearing (G5 passes: zeroing or shuffling it does pull P50 down by more than the
1.2x bar) but owns roughly a third of P50 linearly, not a majority.

## Capacity-per-recurrent-synapse fit (prediction, reported not gated -- corrected marginal form)

Per the 2026-09-24 amendment, `k_fit` is the median over seeds of the per-seed MARGINAL between `sparse_dg` and
`sparse_dg_recx2` (only `c_rec` differs between the two arms):

- Per-seed marginal `k_rec`: 0.133, 0.129, 0.131, 0.111, 0.131, 0.128 (seeds 42/43/44/100/101/102) -- all <!--derived-->
  positive, tightly clustered.
- `k_fit = 0.130`, inside the pre-registered 0.1-0.3 band (Rolls 2013's 0.2-0.3 asymptotic estimate, lowered by
  finite size, as predicted).
- Extrapolation (reported as EXTRAPOLATION, not a measurement, per the runner's own label) to one 3090-sized fast
  store (n_ca3=1e5, c_rec=1e4, a=0.005): ~49,000 facts -- inside the pre-registered 5e4-1e5 order-of-magnitude
  band, at its low end. The a-scaling from 0.01 to 0.005 remains untested (no arm varies a with the DG held
  fixed), as the prereg discloses.
- The per-arm all-fan-in numbers (the confounded quantity the amendment retired from the fit) are reported by
  `aggregate()`'s debug stream per seed and were NOT used here, matching the amendment.

## What this does and does not show

GO here means: the superposed CA3 store has a measurable capacity law (roughly linear in synapses per cell, in
the uniform regime), a working DG companion in the correlated-cue (hub) regime, a load-bearing (not decorative)
recurrent attractor, a working bounded-synapse palimpsest, and shared-synapse crosstalk that falls with load as
predicted -- all six-seed, all pre-registered, all passing by the runner's own gate math with zero UNDEFINED
seeds. It does **not** show LLM parity (the prereg's own arithmetic: order 1e8 facts for a tiny LLM's storable
knowledge vs ~5e4-1e5 for this fast store -- about 3 orders of magnitude, the "fast store alone is
rat-hippocampus scale" framing). It does not wire anything into the chat path, and the k-WTA/gamma-cycle
discretization (h1 in the prereg) is not cross-checked against a spiking bridge. Functional read-outs only; no
phenomenal claim.

## Provenance note (separate from this verdict)

The 54 grid artifacts carry inline `"runner": "research.runners.ca3_superposed_fact_attractor"` and a `"config"`
block at their top level, which is what satisfies `gates/artifact_provenance`. None of them has a `.prov.json`
sidecar from the automatic provenance door (`research/runners/__init__.py`), even though pool1's own provenance
log (its `runs.jsonl`, under the pool node's own `_provenance` directory) independently shows a start record for every one of its 36 shards under
`-m research.runners.ca3_superposed_fact_attractor ... --out .../grid`. Root cause (confirmed, not fixed here):
`_declared_output_paths()` requires the `--out` value to be an existing FILE; this runner's `--out` is a
DIRECTORY, so the explicit-path branch finds nothing, and because the flag was still *seen*, the code never
falls back to scanning for fresh files either -- the exit-time sidecar write is silently skipped for every
directory-style `--out` runner. Flagged as a background task rather than fixed inline here, to keep this scoring
pass narrow; it does not change the verdict above (the artifacts are independently corroborated by the pool's
own `runs.jsonl` and by matching `config`/`backend`/`sha_facts` fields across every file).

## Cost

Per-run wall time (build + per-checkpoint write/materialize + batched-query time, summed from each file's own <!--derived-->
`build_s` and `checkpoints[].{write_s_interval,materialize_s,*.per_query_ms_batched}` fields): median 254 s, range <!--derived-->
102-584 s across the 54 runs, one CPU core each (`SIM_BACKEND=numpy`, `OMP_NUM_THREADS=4`); the 54 ran in <!--derived-->
parallel across the mini-PC pool, not sequentially. No arm approached the `mem_gb` estimates in the prereg's <!--derived-->
Staging section closely enough to matter for this grid's cost.

## Honesty

Functional read-outs only. No phenomenal claim.
