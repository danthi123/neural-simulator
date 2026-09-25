---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: memory (knowledge scale; D6 fact store capacity)
mechanism: ca3-superposed-fact-attractor
prereg: research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md
seeds: [42, 43, 44, 100, 101, 102]
supersedes: commit 3e81251ee9 (branch research/score-ca3-grid, never merged to main) -- rejected per
  research/FAILURE_LOG.md's 2026-09-24 row on the grid's seeds-101/102 provenance gap
artifacts:
  - research/findings/raw/_ca3_superposed_fact_attractor/grid/*_s42.json
  - research/findings/raw/_ca3_superposed_fact_attractor/grid/*_s43.json
  - research/findings/raw/_ca3_superposed_fact_attractor/grid/*_s44.json
  - research/findings/raw/_ca3_superposed_fact_attractor/grid/*_s100.json
  - research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102/*_s101.json
  - research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102/*_s102.json
  - research/findings/raw/_ca3_superposed_fact_attractor/scored_54cell_valid_provenance/aggregate.json
  - research/findings/raw/_ca3_superposed_fact_attractor/PROVENANCE_NOTES_2026-09-24.md
verdict: GO 6/6 on all 9 pre-registered gates (G1-G9), 0 missing, 0 UNDEFINED, scored over the 54/54 registered
  cells whose execution provenance is verifiable (36 from the original grid + 18 from the seeds-101/102 reruns).
---

# CA3 superposed fact attractor: capacity-law GO 6/6, this time with valid provenance for every registered cell

Scores the full 9-arm x 6-seed grid registered in
`research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md` (as amended 2026-09-24),
using the runner's own `--aggregate` combine mode, over ONLY the cells whose execution provenance is verifiable
at the registered revision.

## Why this finding exists

A prior scoring of this same grid (commit `3e81251ee9`, branch `research/score-ca3-grid`) reported the identical
GO but was never merged to `main`, because 18 of the grid's 54 cells -- all 9 arms at seeds 101 and 102, under
`research/findings/raw/_ca3_superposed_fact_attractor/grid/` -- had NO discoverable execution provenance: absent
from `research/queue/dispatch.log`, absent from every reachable pool node's job log, and sharing one identical
mtime about 42 minutes before the earliest genuine dispatch. This is recorded in `research/FAILURE_LOG.md`'s
2026-09-24 row. 18 rerun jobs (all 9 arms x seeds 101/102) were queued to a fresh output directory,
`rerun_s101_s102/`, and the suspect files were left untouched under `grid/` for comparison rather than deleted.

This finding scores the SAME registered grid again, using ONLY provenance-clean cells: the original 36
(`grid/`, seeds 42/43/44/100) plus the 18 verified reruns (`rerun_s101_s102/`, seeds 101/102) -- and supersedes
the unmerged commit. The 18 original seed-101/seed-102 files under `grid/` remain excluded and uncited here.

## Correction after scoring (2026-09-25 00:40)

One rerun cell, `rerun_s101_s102/sparse_dg_c2_s101.json`, was still running on a pool node when this finding was
first scored: the copy scored then ended at the P=50000 checkpoint, and the job wrote its final P=100000 checkpoint at
00:05. The queue files showed no pending CA3 line, but a running job is not in the queue, so the completeness check
missed it. The complete file now replaces the partial one in both `rerun_s101_s102/` and
`scored_54cell_valid_provenance/`, and the registered `--aggregate` was re-run: the new `aggregate.json` is identical
to the first one key by key (every gate, every per-seed value, k_fit), so the verdict below is unchanged.

## Provenance check, before any scoring

Full detail (extracted log lines, exact counts, mtimes) is in
`research/findings/raw/_ca3_superposed_fact_attractor/PROVENANCE_NOTES_2026-09-24.md`. Summary:

- `research/queue/dispatch.log` (a live, un-tracked queue log) records exactly 54 CA3 dispatch lines, zero more
  and zero fewer, all pinned to revision `ab2adcf51bb452f322fabe94b5668335c7d687a7`: 36 to `.../grid` (seeds
  42/43/44/100, all 9 arms, 09:50-11:11 on 2026-09-24) and 18 to `.../rerun_s101_s102` (seeds 101/102, all 9
  arms, pool1 + pool42, 22:55-23:44 the same day). `research/queue/pool.queue`, `pool.queue.unchecked` and
  `gpu.queue` hold zero pending CA3 lines -- nothing registered is still in flight or unqueued.
- That revision is verified in THIS worktree to be an ancestor of `main`, to contain the amended prereg text and
  the amended `aggregate()` (the per-seed-marginal `k_fit` fix), and to be byte-identical to this worktree's own
  `research/runners/ca3_superposed_fact_attractor.py` at the current `main` tip -- so scoring with the checked-out
  runner code is scoring with the registered code, not a diverged copy.
- `research/queue/pool_sync.log` independently corroborates both windows: files pulled from `pool1:ab2adcf51.../
  research/findings/raw` and, from 23:23 onward, `pool42:ab2adcf51.../research/findings/raw`, landing in small
  increments across exactly these two time windows and nowhere else in the log.
- The 18 excluded seed-101/seed-102 files under `grid/` share one identical mtime and are absent from both
  logs, exactly as `research/FAILURE_LOG.md` describes; the 36 included `grid/` files and 18 included
  `rerun_s101_s102/` files each have distinct, staggered mtimes consistent with real per-job runtimes.
- Informational only, not provenance: the excluded files' recall curves match the included reruns' almost
  exactly (identical `recall`/`recall_recent`/`recall_rec_zero`/`recall_rec_shuffle` arrays, `dprime` agreeing to
  5-6 significant figures) -- the excluded runs were very likely genuine, just unprovable, which is why they stay
  excluded rather than being asserted false.

## Scoring: the registered grid, 54/54 cells, 0 missing

Merged the 36 verified `grid/` cells with the 18 verified `rerun_s101_s102/` cells into
`research/findings/raw/_ca3_superposed_fact_attractor/scored_54cell_valid_provenance/` (54 per-cell files) and
ran the prereg's own combine command, unmodified:

```
SIM_BACKEND=numpy python -m research.runners.ca3_superposed_fact_attractor \
  --aggregate research/findings/raw/_ca3_superposed_fact_attractor/scored_54cell_valid_provenance
```

`research/findings/raw/_ca3_superposed_fact_attractor/scored_54cell_valid_provenance/aggregate.json` (+ a
`.prov.json` sidecar, stamped by the automatic provenance
door since `--aggregate` writes a single file) reports `missing: []` and, per gate:

<!--derived-->
| gate | requires | result |
|---|---|---|
| G1 learns | >=5/6 | PASS 6/6 |
| G2 cliff | >=5/6 | PASS 6/6 |
| G3 companion (hub) | >=5/6 | PASS 6/6 |
| G4 capacity law | >=5/6 | PASS 6/6 |
| G5 recurrent load-bearing | >=5/6 | PASS 6/6 |
| G6 cost/integrity | 6/6 | PASS 6/6 |
| G7 palimpsest | >=5/6 | PASS 6/6 |
| G8 shared crosstalk | >=5/6 | PASS 6/6 |
| G9 hub limit not synaptic | >=5/6, only meaningful with G4 | PASS 6/6 (G4 also passes) |

Zero cells UNDEFINED on any gate. This meets the prereg's own GO rule verbatim: "G1-G5, G7-G9 on >=5/6 and G6 on
6/6."

Per-seed detail for the two gates with the most texture (G5's recurrent-edge lesion, G3/G4/G9's synapse-count
ratios):

<!--derived-->
| seed | P50 intact | P50 rec_zero | P50 rec_shuffle | attributable_linear_frac | G3 hub ratio | G4 c2/sparse | G9 c2hub/hub |
|---|---|---|---|---|---|---|---|
| 42 | 8262.4 | 5279.3 | 3386.3 | 0.361 | 3.87 | 2.05 | 1.10 |
| 43 | 8232.6 | 5520.4 | 3461.9 | 0.329 | 3.50 | 2.08 | 1.14 |
| 44 | 8118.6 | 5877.7 | 3396.7 | 0.276 | 4.04 | 2.10 | 1.10 |
| 100 | 8996.6 | 5877.7 | 3358.5 | 0.347 | 3.67 | 1.79 | 1.13 |
| 101 | 8355.2 | 5629.9 | 3458.0 | 0.326 | 3.79 | 1.96 | 1.14 |
| 102 | 8387.2 | 5576.4 | 3477.4 | 0.335 | 3.48 | 1.95 | 1.10 |

`attributable_linear_frac` (the linear fraction of P50 lost when only the recurrent edge is zeroed) lands at
0.28-0.36 across all 6 seeds, on both the seeds that had provenance already (42/43/44/100) and the two reruns
(101/102) -- consistent with the dev-seed-7 disclosure of ~25% and comfortably under the G5 threshold ratio (1.2x)
on both `rec_zero` and `rec_shuffle`. The recurrent attractor is a genuine, minority (roughly a third) contributor
to capacity, not the whole store and not decorative either -- exactly as pre-registered and disclosed before any
evaluation seed ran.

## k_fit and the extrapolation (reported, not gated)

The corrected per-seed marginal fit (AMENDMENT (a): `(P50_recx2 - P50_sparse_dg) * a * ln(1/a) /
(c_rec_recx2 - c_rec_sparse_dg)`, median over seeds) is k_fit = 0.13, inside the pre-registered 0.1-0.3 band
(Rolls 2013's asymptotic range is 0.2-0.3; finite size is expected to lower it). Extrapolated to one 3090-sized
fast store (n_ca3=1e5, c_rec=1e4, a=0.005): about 49,022 facts. **Correction to the unmerged commit's wording:**
that commit called this "inside the registered 5e4-1e5 band"; 49,022 is just under the 5e4 lower edge, not inside
it. This is a reported EXTRAPOLATION, not a gate, and the a-scaling from 0.01 (this runner's arms) to 0.005 (the
GPU point) is itself untested per the prereg's own disclosure -- order-of-magnitude only.

## What this does and does not show

Same boundary as the prereg. This is a capacity law for the FAST hippocampal store, at rat-hippocampus scale, not
LLM parity: the prereg's own arithmetic against Qwen2.5-0.5B (~2 bits/parameter, this runner's own ~11
bits/fact scoring cost) puts a tiny LLM's storable fact count at order 1e8, about three orders of magnitude above
the extrapolated fast-store ceiling. It does not wire into the chat path -- `sim/` and `webapp/` import nothing
from this runner, and nothing here claims otherwise. It cannot show LLM parity, says nothing about the slow
cortical store, and does not cross-check the k-WTA idealization against the full spiking bridge (h1 in the
prereg).

No production default exists to flip: this is a default-off, standalone research runner with no `sim/`/`webapp/`
import, so the owner's flip bar (6-seed GO + SOUND review + no-regression battery + production-default
validation) is not applicable here -- there is no default in play, on or off, for this GO to be a candidate
against.

## Artifacts

- `research/findings/raw/_ca3_superposed_fact_attractor/grid/*_s{42,43,44,100}.json` -- 36 files, the original
  provenance-clean dispatch (seeds 42/43/44/100, all 9 arms).
- `research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102/*_s{101,102}.json` -- 18 files, THE
  RERUNS this finding relies on for seeds 101/102 (all 9 arms). The excluded, no-provenance seed-101/seed-102
  originals remain under `grid/` in the primary checkout and are not cited or copied here.
- `research/findings/raw/_ca3_superposed_fact_attractor/scored_54cell_valid_provenance/` -- the 54-file merge of
  the two directories above, plus `aggregate.json` and `aggregate.json.prov.json` freshly computed in this
  worktree with the command shown above.
- `research/findings/raw/_ca3_superposed_fact_attractor/PROVENANCE_NOTES_2026-09-24.md` -- the extracted
  `dispatch.log` / `pool_sync.log` lines and mtime check this finding's provenance claim rests on.
