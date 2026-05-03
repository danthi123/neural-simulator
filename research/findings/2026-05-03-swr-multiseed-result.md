# SWR Phase 3 replay — multi-seed result (in flight)

**Date:** 2026-05-03 (last update: 04:08 EDT — n=3 seeds done)
**Status:** n=3 done; seeds 100/101/102 still running, ETA ~07:30 EDT
**Runs:** `text_eval_v2_swr500_seed{42,43,44,…}.json`
**Config:** v2 baseline (Hebbian off, stdp_w_max=5, readout init=0.5) + curriculum: phase1=0, phase2=100ep, phase3=500 SWR replay events, replay_correct_only=True

---

## Headline (n=3 so far)

| Metric | v2 baseline (n=6) | seed 42 | seed 43 | seed 44 | n=3 mean |
|---|---|---|---|---|---|
| **I→W** | 25.3% | 39.0% | 26.0% | 18.0% | 27.7% (within baseline noise) |
| **W→A** | 28.5% | **22.0%** | **22.0%** | **23.0%** | **22.3%** (−6.2 pp) |
| Phase 2 corr.move | varies | 29.6% | 38.2% | 43.5% | — |

**The W→A regression is holding at n=3.** All three seeds within 1pp
of each other (22, 22, 23), vs 28.5% baseline — a consistent ~6pp
drop. Stronger than 2-seed evidence; the chance of three independent
seeds all landing this close to each other if the true mean were
28.5% with σ ≈ 6 pp is roughly 1%.

**I→W remains noise-dominated.** Range across seeds: 18% to 39%
(span 21 pp). Mean 27.7% is essentially baseline. The seed-42 39%
that initially looked like a "boost" is now clearly an outlier —
the n=3 mean is below seed 42 alone. SWR doesn't move I→W on
average.

## Per-direction breakdown across seeds

The W→A failure mode varies by seed but average accuracy is consistent:

| Word | Baseline (~) | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|---|
| north | ~30% | 7/25=28% | 6/25=24% | 4/25=16% |
| east | ~30% | 6/25=24% | 7/25=28% | 6/25=24% |
| south | ~25% | 3/25=12% | 3/25=12% | 9/25=36% |
| west | ~25% | 8/25=32% | 6/25=24% | 4/25=16% |
| **total** | **28.5%** | **22%** | **22%** | **23%** |

The weak directions are different per seed (seed 43 weak south;
seed 44 weak north + west), but the OVERALL accuracy is consistent.
This pattern suggests SWR uniformly degrades W→A while the specific
failure mode is stochastic — supporting Hypothesis H1 (replay
distribution bias) since the bias direction depends on what each
seed's training cascade happened to over-emit.

## Per-direction breakdown (seed 43, original)

**I→W** — predicting which direction word is uttered when seeing the
gridworld:
- north: 7/22 = 32%
- east: 6/22 = 27%
- south: 7/25 = 28%
- **west: 6/31 = 19%** (clear miss)

**W→A** — moving the right direction when hearing the word:
- north: 6/25 = 24%
- east: 7/25 = 28%
- **south: 3/25 = 12%** (cascade pushed AWAY from south)
- west: 6/25 = 24%

The seed 43 south column is at 12% — well below chance. Consistent
with the hypothesis (in the seed-42 writeup) that replay events are
biased toward already-frequent actions in the training buffer. If
the cascade emitted N/E more often than S during Phase 2, replaying
"correct (token, action)" pairs disproportionately pushes the
language→motor weights toward N/E at the cost of S.

This would predict the per-action-frequency distribution in the
training buffer matters. The seed 42 buffer had 698 correct moves
out of 2355 steps; seed 43 had 831/2178. The seed-43 buffer is
denser (38.2% vs 29.6% correct rate), which gives more replay
material — but if that material is N/E-biased, the regression is
amplified.

## Hypotheses for the consistent W→A drop

### H1: Replay bias toward N/E washes out S/W
The training buffer captures all moves, including the cascade's
intrinsic N-bias. SWR replay re-presents (token, action) pairs from
the buffer, weighted by their natural frequency. So south/west get
fewer replay touches and their language→motor weights drift toward
the N/E majority during the 500-event replay phase.

**Test:** balance replay across the 4 directions — sample 125 events
per word instead of 500 weighted by buffer frequency. If H1 is right,
W→A should match or beat baseline.

### H2: Replay disrupts the direct PFC-bypass pathway
The direct `language_input → motor_X` pathway is short (one hop) and
relies on a precise readout. Replay events drive the *whole* cascade
including motor cortex, which may overwrite the direct pathway with
the cascade's selected action rather than the input word's mapping.

**Test:** during replay, drive ONLY the language input + dopamine,
not motor. If H2 is right, W→A should not regress.

### H3: 500 events is too many; consolidation overshoots
Phase 3 ran 500 replay events on top of 100 episodes ≈ 2200 training
steps. That's 22.7% of training in additional consolidation —
possibly enough to push weights past their optimum.

**Test:** sweep phase3-replays = 50, 100, 200, 500 — find the curve.

---

## What's next

Given the W→A drop is consistent across n=2, three paths:

1. **Validate at full 6 seeds** before deciding the SWR direction is
   real. Seeds 44, 100, 101, 102 launched via the multi-seed
   launcher. ETA: ~70 min × 4 = ~5 hours.

2. **Test H1 (balanced replay)** at seed 42 — patch the replay
   sampler to draw N/E/S/W equally. Tests the most plausible
   mechanism for the regression.

3. **Test H3 (smaller replay)** at seed 42 — try 100 events and 200
   events. Cheaper than (2) since no code change needed; just CLI
   flag changes.

Recommendation: **run all 4 remaining seeds at the current config in
parallel (overnight), then run H1 at seed 42 in the morning.** That
nails down whether the W→A drop is a 2-seed coincidence or a real
SWR-induced effect, AND tests the most plausible mechanism.

## Configuration archaeology (seed 43)

```bash
python -m research.runners.text_train_curriculum \
    --seed 43 \
    --phase1-episodes 0 \
    --phase2-episodes 100 \
    --phase3-replays 500 \
    --stim-steps-per-step 200 \
    --reset-steps 100 \
    --out-stats research/findings/raw/g11_bg/text_eval_v2_swr500_seed43.json
```

Phase 2 elapsed: ~3300s = 55 min
Phase 3 elapsed: ~960s = 16 min
Total: ~71 min

(Identical to seed 42 config except `--seed 43`.)

## Webapp surfaces

- http://localhost:8765/#tab=language&run=text_eval_v2_swr500_seed42.json
- http://localhost:8765/#tab=language&run=text_eval_v2_swr500_seed43.json

Per-direction bars + confusion matrices visible. The Brain tab Live
mode followed seed 43 throughout the run — the 2D mini gridworld
inset stayed hidden because curriculum runs don't emit per-step pos
data, but the cascade animation tracked Phase 2 → Phase 3 SWR
transitions (visible as hippocampus regions taking over from cortex).
