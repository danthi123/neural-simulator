# SWR W->A regression: buffer composition shows east-bias

**Date:** 2026-05-03 (autonomous overnight)
**Context:** v2+SWR (default, frequency-weighted replay) regresses W->A
from 28.5% baseline to 24.3% (paired-t = -6.37, p < 0.001 vs same-seed
baseline; 6/6 seeds regressed). H1 balanced replay queued to test the
"replay-distribution bias" hypothesis.

---

## What the buffer captured

For seeds 101 and 102 (where buffer-composition recording was added in
commit a6e349f), the Phase 2 experience buffer composition was:

| seed | north | east | south | west | east% | west% |
|---|---|---|---|---|---|---|
| 101 | 477 | 1127 | 516 | 271 | **47%** | 11% |
| 102 | 620 | 991 | 381 | 283 | **41%** | 12% |

East is over-represented 4x relative to west in both seeds. This
matches the known cascade N/E-bias (cortex_N and cortex_E spontaneous
firing dominate from cluster A/E feedback), filtered through the
"correct moves only" replay filter.

## What this should do to W->A

**Naive prediction:** if SWR replay reinforces east-aligned weights
proportionally to event count, the agent would over-predict east.
We'd see W->A confusion matrix dominated by east predictions.

**Actual data** (per-direction W->A across 6 seeds):

| direction | baseline | v2+SWR | delta |
|---|---|---|---|
| north | 26.7% | 22.0% | -4.7pp |
| east  | 31.3% | 26.0% | **-5.3pp** |
| south | 29.3% | 23.3% | **-6.0pp** |
| west  | 26.7% | 26.0% | -0.7pp |

The biggest losses are **east (-5.3pp) and south (-6.0pp)**. East gets
WORSE despite (or because of) being heavily over-replayed.

## What the buffer composition actually does

Re-read of `_run_swr_replay_phase` (in `text_train_curriculum.py`):
each replay event drives language_input + language_output with the
event's token, then drives motor_X with the event's correct action,
then applies +1 reward. STDP + R-STDP sculpt weights between language
and motor.

When east is replayed 47% of the time, language_input neurons activated
by "east" co-fire with motor_E neurons 47% of the replay events. STDP
strengthens those (token, motor) pairings ~4x more than west's pairings.

**But** east has 26 active neurons in language_input (sparsity=0.1),
and motor_E has 10 neurons. STDP doesn't strengthen JUST motor_E from
east-replays — it strengthens whatever motor neurons co-fire during the
replay window. Cascade dynamics during the replay produce broad cortical
firing, hitting non-target motor pools as well. So the over-replayed
"east" pattern produces:

1. Strong language_input(east) -> motor_E weights (intended)
2. Also strong language_input(east) -> motor_other weights (collateral)
3. Soft-bound STDP at stdp_w_max=5 caps the runaway

The collateral plasticity is why east_accuracy DECREASES — motor_E now
fires for many input patterns, becoming less discriminative.

## Why west barely regresses

West is under-replayed (12% of events). Its weights barely change
during SWR. So west_accuracy is approximately unchanged from baseline.
This is consistent with the "broad collateral" hypothesis: the
replayed token's plasticity hurts discrimination broadly, but
under-replayed tokens are spared.

## H1 prediction

If "buffer-distribution bias hurts discrimination" is the mechanism,
then H1 (balanced replay, 125 events per direction = 500 total) should:
- Reduce east-collateral interference (less over-replay)
- Boost west readout strength (more replay than 12%)
- Net W->A: should approach baseline (28%) or beyond if balanced replay
  is genuinely beneficial

If H1 fixes the regression to baseline, the mechanism is buffer bias,
no architectural change needed for SWR.

If H1 doesn't fix it, the mechanism is more fundamental — possibly:
- Soft-bound STDP saturating before discrimination emerges
- Cascade dominance during replay broadcasting plasticity to wrong pools
- Pre-existing weight imbalance amplification that doesn't depend on
  replay distribution

H1 6-seed batch runs after H4 finishes (~6 hours from this commit).

## Implication for autonomous tonight

The arch sweep (auto-launches after H1 completes) tests whether
**any** structural change breaks the v2 ceiling. If H1 succeeds, then
balanced replay can be the default and we don't need structural pivot.
If H1 fails, the arch sweep gives us the next direction.

Both data points are independent and additive — H1 tells us about SWR;
arch sweep tells us about structure. We get both tonight.

## Files

- `research/findings/raw/g11_bg/text_eval_v2_swr500_seed{42..102}.json`
  — n=6 v2+SWR final
- `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_seed{42..102}.json`
  — n=6 baseline
- `research/findings/2026-05-03-swr-multiseed-summary.md` — auto-aggregated
- buffer composition is in `training_stats[1].buffer_per_direction`
  (only seeds 101, 102 onward; the recording was added late)
