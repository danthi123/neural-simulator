---
title: "Interleaved generative replay (van de Ven) does NOT beat like-for-like self-replay at N=50 (0.72 vs 0.90) — NEGATIVE on retention; it FIXES the newest-fact acquisition collapse (0.65 vs post-hoc 0.05) but trades bulk retention for it (stability-plasticity)"
date: 2026-08-11
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42]
seed-waiver: "1-seed N=50 SMOKE (numpy, ~88 min); the exact 6-seed pool command is returned for the coordinator to bank the negative. The margins are large (interleaved −0.18 vs the self-replay baseline, −0.06 vs post-hoc) so a 6-seed sweep is expected to CONFIRM, not flip, the negative. Not framed as a GO."
claim_check: measured
---

# Interleaved generative replay is a NEGATIVE on continual retention — the interleaving is not the lever <!--derived-->

## Claim

<!--derived-->

The record's continual-retention arc had SATURATED replay-scheduling (fixed-k prioritized replay NEGATIVE at scale
`777fcb0d`; bounded two-store NEGATIVE `0c7531785`; weight-protect REFUTED `e50f5d45a`) and CAPACITY (resolves N=20,
SLIPS at N=100, `7ee36d66b`). The board's one un-run, externally-grounded lever was **interleaved generative replay**
(van de Ven, Siegelmann & Tolias 2020, Nat Commun doi:10.1038/s41467-020-17866-2; McClelland/McNaughton/O'Reilly 1995;
Kumaran/Hassabis/McClelland 2016): both the base sleep-replay AND `generative_v2` teach each new fact ALONE then replay
POST-HOC — never the canonical brain-inspired-replay move of INTERLEAVING the generator's regenerations of old facts
INTO new-fact acquisition (joint training). That specific interleaving uniquely targets the flagged
"acquisition-at-scale = upstream of all" wall (immediate-acq degrades at large N).

**Built + smoke-tested (1-seed N=50, exactly-matched A/B: same non-forgetting generator, same data, same total
gradient steps — ONLY the ordering differs). Verdict: NEGATIVE.** Interleaving does NOT beat the self-replay cap and
does NOT beat post-hoc generative replay for retention. It DOES fix the newest-fact acquisition collapse — an isolated,
real mechanistic win — but that does not translate to overall retention, because it trades bulk retention for it.

## Data — N=50, 1-seed (42), slow-cortex reservoir fixed at 120 units (matched across arms), chance = 0.02

<!--derived-->

| arm | retention N=25 | retention N=50 | mean immediate-acq | immediate-acq of newest fact @ N=50 |
|---|---|---|---|---|
| **interleaved_gr** (van de Ven, treatment) | 0.88 | **0.72** | 0.796 | **0.65** |
| **posthoc_gr** (established GR ordering, control) | 0.92 | **0.78** | 0.523 | **0.05** |
| **self_replay** (mean-prototype cap baseline) | 0.92 | **0.90** | 0.768 | 0.57 |
| **scramble** (interleaved, labels shuffled = content lesion) | 0.16 | 0.06 | 0.828 | 0.90 |

Deltas @ N=50: interleaved − posthoc = **−0.06**; interleaved − self_replay = **−0.18**; interleaved − scramble = **+0.66**.
Artifact: `research/findings/raw/teacher_loop_interleaved_gr_s42.json` (+ `.provenance.json`). Runner:
`research/runners/_teacher_loop_interleaved_generative_replay_derisk.py`. `attributable_to` is emitted in the verdict
for all three treatment/control contrasts.

## Read — honest, and it dissociates acquisition from retention

<!--derived-->

- **NEGATIVE on the task GO (beat the replay cap): NO.** Interleaved 0.72 < the self-replay baseline 0.90 (−0.18) at
  MATCHED capacity. The interleaving lever does not beat the record's replay cap; it underperforms it.
- **NEGATIVE on the novel lever (interleaving > post-hoc): NO.** Interleaved 0.72 < post-hoc generative replay 0.78
  (−0.06). Ordering the replay as van de Ven interleaving does not raise retention over the established post-hoc order.
- **The apparent "beats the historical ~55% cap" is CAPACITY, not the lever.** The ~55% cap was a SMALL-reservoir
  number (N=10 6-seed mean 0.55 / N=50 small-reservoir 0.71). At MATCHED capacity (reservoir=120) the self-replay
  control is 0.90 and interleaving underperforms it — the gain over 0.55 lives entirely in the control (capacity),
  which is exactly what the treatment/control attribution flags.
- **Generative replay UNDERPERFORMS simple mean-prototype self-replay here (0.72 / 0.78 vs 0.90).** In this unimodal
  Gaussian world the engram MEAN is a sufficient statistic, so the fixed non-forgetting generator (1344 params, 0
  stored raw) only loses fidelity vs storing the mean — reconfirming `8d2510d3a` (engram fidelity is not the
  bottleneck; the mean is sufficient). The generator's bounded-storage advantage is not purchased where the mean is
  the store you'd want anyway.
- **The one genuine, isolated win: interleaving FIXES the acquisition-at-scale collapse.** Post-hoc's immediate
  acquisition of the NEWEST fact collapses to 0.05 (= chance) at N=50 — teaching the 50th fact alone into a crowded
  shared readout barely registers. Interleaving keeps it at 0.65. This confirms the board's "acquisition-at-scale =
  upstream of all" wall is real, and that co-training new + regenerated-old is the mechanism that addresses it.
- **But it TRADES bulk retention for it (stability-plasticity).** Interleaved has HIGH new-fact acquisition (mean 0.796,
  newest 0.65) yet LOWER overall retention (0.72); post-hoc has LOW mean acquisition (0.523, newest 0.05) yet HIGHER
  retention (0.78). Interleaving redistributes shared-readout capacity toward the newest fact at the expense of the
  consolidated bulk — a clean plasticity-vs-stability dissociation, not a free win.
- **Content is load-bearing (anti-cheat holds):** scramble 0.06 ≪ interleaved 0.72 (+0.66) — the retention is from the
  stored regenerated CONTENT, not the interleaved compute. Anti-cheats pass: substrate byte-identical across two builds
  at seed 42; git diff main -- sim/ empty (runner-side only, NO sim/ edit); generator trained-param count CONSTANT in N
  (1344); 0 stored raw patterns; consolidation never read the true-engram ruler.

## The next lever (a negative that names the direction)

<!--derived-->

1. **Replay ORDERING is exhausted for retention** (prioritized selection NEGATIVE; interleaving NEGATIVE-for-retention).
   The productive residual is the acquisition/retention TRADE-OFF this run isolated: pair interleaving's newest-fact
   acquisition fix with a BULK-RETENTION protector — a consolidation-graded learning rate / metaplasticity on
   already-committed readout rows (so new-fact interleaving cannot overwrite the bulk), or a replay budget that GROWS
   with N so bulk coverage is not diluted as the store grows.
2. **Generative replay is only worth its bounded-storage cost where the mean prototype is INSUFFICIENT** (high-rank /
   multimodal facts). This unimodal-Gaussian world is the wrong test for it; the compressing/compositional generator
   (sub-linear O(√N) storage, `2026-08-09-compositional-generator-*`) should be tested in a HIGH-RANK world at large N —
   still the OPEN storage leg.
3. **Simple self-replay already gives 0.90 @ N=50 with adequate capacity**, so the true lifetime wall is at larger N
   where even O(N) self-replay's per-sleep compute/capacity breaks — the compressing generator (sub-linear storage AND
   compute) remains the named lifetime lever.

## Grounding + evidence

<!--derived-->

- van de Ven, Siegelmann & Tolias 2020 (brain-inspired replay: interleave generated old-task samples with new-task
  training); McClelland/McNaughton/O'Reilly 1995 + Kumaran/Hassabis/McClelland 2016 (CLS systems consolidation).
- Runner (reuse-by-import, NO sim/ edit): `research/runners/_teacher_loop_interleaved_generative_replay_derisk.py`
  reuses `GenerativeReplayNetV2` (non-forgetting generator), `_build_slow_cortex` (fixed de-clamped cortex),
  `Hippocampus`/`_self_replay_consolidate`, `_teach_fact`/`_fact_acc`/`_corrective_batch`, `ReferentEnv`.
- Artifact: `research/findings/raw/teacher_loop_interleaved_gr_s42.json` (+ `.provenance.json`). Backend numpy; ~88 min.
- Prior record NOT re-derived: prioritized-replay NEGATIVE (`777fcb0d`), non-forgetting generator GO@N=20
  (`0933fdb7a`), engram-mean-sufficient (`8d2510d3a`), N=100 capacity SLIP (`7ee36d66b`).

## Exact 6-seed pool command (to CONFIRM the negative; not decision-changing)

<!--derived-->

(`<raw>` = `research/findings/raw`; the exact runnable command is also in the runner docstring.)

```
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._teacher_loop_interleaved_generative_replay_derisk \
  --seeds 42 43 44 45 46 47 --n-max 50 --milestones 25 50 --slow-hidden 120 --gen-hidden 96 --gen-k 96 \
  --out <raw>/teacher_loop_interleaved_gr.json
```

Writes per-seed `_s{seed}.json` + a `_AGG.json` (GO gate: interleaved > posthoc + 0.05 AND > self_replay + 0.10, 6/6).
The 1-seed margins (−0.18 to the self-replay baseline, −0.06 to post-hoc) make a flip to GO very unlikely; the sweep
banks the NEGATIVE at the project's 6-seed standard.
