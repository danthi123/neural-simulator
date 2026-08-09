---
type: finding
status: negative
date: 2026-08-09
mechanism: sleep-replay-consolidation-self-generated-hippocampal-engram
lane: breadth / teacher-loop / memory
runner: research/runners/_teacher_loop_sleep_replay_budget_sweep_derisk.py
builds-on: research/runners/_teacher_loop_sleep_replay_consolidation_derisk.py
attacks-baseline: teacher-loop SLEEP-REPLAY consolidation (6-seed REPLAY frac_recalled mean ~0.55 range 0.20-0.90, no-replay 0.13, interleaved ceiling 0.8; finding 2026-08-08-teacher-loop-sleep-replay-consolidation-self-replay-beats-catastrophic-forgetting.md)
biological-pattern: hippocampal->cortical systems consolidation (McClelland/McNaughton/O'Reilly 1995; Wilson/McNaughton 1994) -- this tests replay QUANTITY (nightly budget) vs engram FIDELITY
artifacts:
  - research/findings/raw/sleep_replay_budget_s43.json
  - research/findings/raw/sleep_replay_budget_s45.json
  - research/findings/raw/sleep_replay_budget_scout_s42.json
  - research/findings/raw/sleep_replay_budget_scout_s43.json
  - research/findings/raw/sleep_replay_budget_scout_s44.json
  - research/findings/raw/sleep_replay_budget_scout_s45.json
  - research/findings/raw/sleep_replay_budget_scout_s46.json
  - research/findings/raw/sleep_replay_budget_scout_s47.json
  - research/findings/raw/sleep_replay_budget_noise43_nz0.05.json
  - research/findings/raw/sleep_replay_budget_noise43_nz0.20.json
  - research/findings/raw/sleep_replay_budget_noise43_nz0.35.json
---

# NEGATIVE (with teeth): more sleep-replay budget does NOT lift host-store retention toward the 0.8 ceiling -- store-fidelity (WS-1), not replay quantity, is the lever

## The question (WS-2: replay QUANTITY vs engram FIDELITY)

<!--derived-->
(this paragraph quotes the PRIOR sleep-replay consolidation finding's measured baseline, not this run's artifact)

The host-store sleep-replay consolidation de-risk is self-generated and robust (REPLAY > NO-REPLAY every seed;
SCRAMBLE content-lesion -> ~0) but UNDER-CONSOLIDATES: 6-seed REPLAY frac_recalled mean ~0.55 (range 0.20-0.90),
no-replay 0.13, against the interleaved ceiling of 0.8. The declared likely cause was engram FIDELITY -- the
replayed engram is a LOSSY host mean-vector (`X.mean(axis=0)` per fact), not a neural attractor. This de-risk
tests the OTHER hypothesis first, on the SAME host store: is the ~0.55 partly just TOO LITTLE REPLAY? The clean
result: NO. More/better replay does not move it. That maps the residual to store-fidelity (WS-1).

## The lever (reuse-by-import, NO sim/ edit, NO new store)

The sweep turns ONLY the offline SLEEP replay-budget knobs on the existing runner, holding the net / world /
per-fact WAKE teaching budget FIXED (identical to baseline). It reuses the sleep-replay runner's own sequential
arm `_run_arm` (net build + wake-teach + hippocampal engram capture + `_self_replay_consolidate`) and sweeps a
grid of (replay_epochs, replay_per_fact) ordered by total replay work = replay_epochs * replay_per_fact, spanning
64x (work 96 .. 6144). Two 1-D sweeps (epochs @ per_fact=16; per_fact @ epochs=24) cross at the (24,16) baseline
anchor; a replay_noise sweep {0.05,0.20,0.35} covers the third named budget axis. It adds NO new store, NO host
fact->slot table: it only replays the same lossy mean-vector engram more.

Brain-based / self-generated is inherited unchanged: the engram store is the brain's own captured trace, replay is
self-generated with a brain-owned RNG, and the consolidation path (`_self_replay_consolidate`,
`Hippocampus.generate_replay`) takes NO `env` parameter and calls NO `env.*` in its body (grep-verified) -- the
teacher and the world are ABSENT during consolidation.

## Seeds (why the low ones)

The ~0.55 is a 6-seed MEAN. A baseline-budget scout (this runner, --grid baseline, all six seeds) reproduces it:

<!--derived-->

| seed | 42 | 43 | 44 | 45 | 46 | 47 | mean |
|---|---|---|---|---|---|---|---|
| baseline-budget REPLAY frac@N=10 | 0.90 | 0.20 | 0.40 | 0.20 | 0.60 | 0.80 | 0.52 |

no-replay 0.10 and SCRAMBLE ~0.1 on every seed. Seed 42 is already saturated (0.90, no headroom) so a budget
sweep on it is flat and uninformative. The informative seeds are the LOW ones (43, 45 at 0.20) with the most
headroom to 0.8 -- if budget were the lever, THESE are where it would show. It does not.

## Result -- seed 43 (baseline 0.20), full budget grid

<!--derived-->

| replay work | (re_e, re_pf) | frac_recalled@N=10 | immediate acq | wall |
|---|---|---|---|---|
| 96 | (6, 16) | 0.20 | 1.000 | 136s |
| 192 | (24, 8) | 0.30 | 1.000 | 176s |
| 192 | (12, 16) | 0.20 | 1.000 | 172s |
| 384 | (24, 16) *baseline* | 0.20 | 1.000 | 244s |
| 768 | (48, 16) | 0.20 | 1.000 | 373s |
| 768 | (24, 32) | 0.40 | 1.000 | 375s |
| 1536 | (96, 16) | 0.10 | 1.000 | 587s |
| 1536 | (24, 64) | 0.20 | 1.000 | 588s |
| 6144 | (96, 64) **MAX (16x)** | **0.20** | 1.000 | 2069s |

- NO-REPLAY floor 0.10; SCRAMBLE@work-96 0.20, SCRAMBLE@work-6144 (max) 0.10.
- rise(max - min budget) = **+0.00**; Spearman(work, frac) = **+0.18** (flat/noise); best = **0.40** at (24,32),
  **gap to the 0.8 ceiling = +0.40**. The MAX budget (16x the baseline replay work, 2069s) returns **0.20 = the
  baseline value**. Adding replay epochs at fixed per_fact even *dropped* to 0.10 at (96,16) (over-consolidating
  the lossy prototype).
- replay_noise sweep (seed 43, baseline budget, the third named budget axis): REPLAY frac_recalled at
  replay_noise = {0.05, 0.10, 0.20, 0.35} is {0.20, 0.20, 0.20, 0.20} -- FLAT at the baseline value; immediate
  acq 1.000 at every noise; SCRAMBLE {0.20, 0.10, 0.10, 0.00} (higher generative noise makes the content-lesioned
  replay progressively useless -- the self-generation control tightens). Varying the replay variability does not
  help either.

## Result -- seed 45 (baseline 0.20), full budget grid

<!--derived-->

| replay work | (re_e, re_pf) | frac_recalled@N=10 | immediate acq | wall |
|---|---|---|---|---|
| 96 | (6, 16) | 0.10 | 0.988 | 124s |
| 192 | (24, 8) | 0.10 | 1.000 | 156s |
| 192 | (12, 16) | 0.20 | 1.000 | 155s |
| 384 | (24, 16) *baseline* | 0.20 | 0.993 | 220s |
| 768 | (48, 16) | 0.20 | 0.997 | 324s |
| 768 | (24, 32) | 0.30 | 0.990 | 290s |
| 1536 | (96, 16) | 0.10 | 1.000 | 534s |
| 1536 | (24, 64) | 0.10 | 1.000 | 530s |
| 6144 | (96, 64) **MAX (16x)** | **0.10** | 1.000 | 1792s |

Same shape, sharper: rise(max - min) = **+0.00**, Spearman +0.25 (flat); best **0.30** at (24,32), **gap +0.50**;
the MAX budget returns **0.10 -- BELOW the 0.20 baseline**. NO-REPLAY 0.10; SCRAMBLE@min 0.10, SCRAMBLE@max 0.10;
immediate acq >= 0.988 at every budget. Both low seeds (43, 45) read **NO-GO / NEGATIVE_FLAT**.

## Teeth

- **(a) retention does NOT rise with budget** -- flat across a 64x range; max-budget == baseline. The WS-2
  hypothesis ("too little replay") is FALSE on the seeds that had headroom.
- **(b) best budget does NOT reach the ceiling** -- best 0.40 (seed 43), gap +0.40 to 0.8. This is the mapped
  first-class result: **store-fidelity (WS-1) is the lever, not quantity.**
- **(c) [instrument valid] immediate acquisition stays perfect** -- 1.000 at EVERY budget on both low seeds, so
  the flatness is not a broken learner; the net can still learn each new fact, it just cannot RETAIN more from
  more replay of a lossy prototype.
- **(d) [instrument valid] self-generation holds under max compute** -- SCRAMBLE@max-budget (content-lesioned,
  identical 16x compute) forgets like no-replay (0.10 <= 0.10+0.10). The tiny replay-vs-scramble margin here
  (0.20 vs 0.20 at low budget on seed 43) is itself the tell: on a low seed the lossy engram barely carries
  content, and MORE of it does not add any.
- `attributable_to(replay-budget quantity, max vs min)` reads **0%** of the (nil) effect on the manipulation --
  the budget knob owns none of the retention; the number is set elsewhere (the engram content / the readout
  interference), exactly the store-fidelity residual.

Because the instrument preconditions (c,d) HOLD while the hypothesis (a,b) is FALSE, the verdict is a clean
**NO-GO / NEGATIVE_FLAT**, not UNDEFINED: the measurement is valid and the lever genuinely does not move the metric.

## What this maps (the point of a negative with teeth)

The consolidation bottleneck is NOT the amount of nightly replay. It is the CONTENT being replayed: a single
lossy host mean-vector per fact cannot re-separate facts that interfere on the shared readout, and replaying it
16x only reinforces the same insufficient prototype. This is the direct evidence for **WS-1 (store-fidelity)**:
replace the host mean-vector engram with the brain's OWN neural engram -- a spiking pattern-completing CA3
attractor (research/runners/_riii_ca3_cortical_episodic_wta_derisk.py) whose completed pattern is a
higher-fidelity, separable reactivation -- and re-run the SAME consolidation. That is the lever this negative
points to.

## Scope / honest boundary

- SMOKE on the two LOW seeds (43, 45) that carry headroom, plus a 6-seed baseline-budget scout. The full
  6-seed budget-vs-retention curve is the aggregate command below; the negative is already unambiguous on the
  seeds where a budget effect COULD appear (a flat curve on a saturated seed proves nothing; a flat curve on a
  0.20 seed with 0.60 of headroom is decisive).
- The store is a LOSSY host mean-vector engram -- the WS-1 residual this negative isolates. This is a documented
  host shortcut (a Python `X.mean(axis=0)` list), not the brain's own neural engram; converting it is WS-1.
- Two budget axes swept as pure QUANTITY (epochs, per_fact) over 64x; a third (replay_noise, a generative-
  variability knob) swept at baseline budget. The recency-weighted vs replay-all SCHEDULE is a further axis not
  swept here; but a schedule change redistributes a fixed budget and cannot exceed the replay-all upper bound
  already tested, so it cannot rescue a hypothesis that a 16x budget increase did not.
- N=10 world, OnBridgeEpropNet transport-free e-prop substrate (48-neuron net), numpy backend (cupy is launch-
  bound and slower at this size -- verified 2026-08-09). The mechanism is backend-independent.
- Consolidation compute grows ~quadratically in N (fact i replays i+1 engrams); the max-budget point is 16x the
  baseline replay work (2069s), a wall-clock (not fidelity) cost -- and it bought nothing.

## Reproduce

SCOUT (find the low seeds -- baseline budget, all 6 seeds parallel):
```
for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --seed $s --grid baseline \
  --pool 3 --out research/findings/raw/sleep_replay_budget_scout_s$s.json & done; wait
```

SMOKE (a low seed, full budget grid -- as run for 43 and 45):
```
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --seed 43 --grid full \
  --pool 12 --out research/findings/raw/sleep_replay_budget_s43.json
```

6-SEED (the deliverable curve; one seed per process, full grid, in parallel), then aggregate:
```
for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --seed $s --grid full \
  --pool 6 --out research/findings/raw/sleep_replay_budget_s$s.json & done; wait
.venv/bin/python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --aggregate \
  research/findings/raw/sleep_replay_budget_s{42,43,44,45,46,47}.json
```
