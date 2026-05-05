# 3-factor failure is NOT learning-rate-limited

**Date:** 2026-05-05 ~15:25 EDT
**Status:** Step 2a of post-verdict plan complete. Higher LR doesn't
help — actually slightly harms.

---

## The check

After the 18-day W→A verdict (global scalar feedback fails), the
post-verdict plan recommended trying alternatives BEFORE pivoting to
dendritic learning (1.5-2 mo investment). Step 2a tested whether
3-factor's 1/6 alignment was just under-tuned — would 5x or 10x
learning rate move it toward the dendritic decision gate (>=4/6)?

```bash
python -m research.experiment_runner experiments/bio_three_factor_high_lr.yaml
```

3 seeds × 2 conditions (LR 5e-3, LR 1e-2) at parallel=3, ~100 min
total.

## Result: 0/3 aligned at both 5x and 10x

| Condition | n | true mean | best mean | excess | aligned/n |
|---|---|---|---|---|---|
| Default LR 1e-3 (historical) | 6 | 24.3% | 32.0% | +7.7pp | **1/6** |
| LR 5x = 5e-3 | 3 | 23.3% | 32.0% | +8.7pp | **0/3** |
| LR 10x = 1e-2 | 3 | 23.7% | 32.0% | +8.3pp | **0/3** |

Per-seed breakdown:

| condition | seed | true | best | best perm | aligned? |
|---|---|---|---|---|---|
| LR 5x | 42 | 25% | 34% | EWNS | no |
| LR 5x | 43 | 20% | 31% | SWEN | no |
| LR 5x | 44 | 25% | 31% | SWNE | no |
| LR 10x | 42 | 25% | 35% | EWNS | no |
| LR 10x | 43 | 21% | 31% | SWEN | no |
| LR 10x | 44 | 25% | 30% | SWNE | no |

Best permutations are random across seeds (EWNS, SWEN, SWNE) — same
seed-dependent structural noise pattern as the default-LR case, just
with the 1-seed default-LR alignment (seed 101) absent at higher LR.

## Confusion matrices show pure noise

```
seed 42 LR 5x:
  north: N= 7 E= 8 S= 4 W= 6
  east:  N= 6 E= 7 S= 6 W= 6
  south: N=10 E= 6 S= 7 W= 2
  west:  N= 8 E= 3 S=10 W= 4
```

Every word predicts approximately uniformly across the 4 actions.
Note that "west" is predicted as everything BUT west (only 4/25
correct on a 25%-chance baseline). The signal is below noise.

## Interpretation

Higher LR doesn't help because:

1. **Topographic prior preservation requires DELICATE updates.** The
   prior pre-installs target weights at 4.5 (factor 1.5 × baseline 3)
   and off-target at 2.1 (factor 0.7 × baseline 3). A single +1
   reward × LR=5e-3 × eligibility=1 update is +0.005, ~0.1% of the
   prior gap. With 4000 events × ~0.5 motor coincidences per event,
   total update per weight is ~10 × LR. At 5e-3 that's 0.05 = ~2% of
   the prior gap. Noise easily dominates.

2. **At LR=1e-2, single events make ~0.1% direct changes to weights**
   that aren't aligned with the prior structure. Across 4000 events,
   this ERODES the prior faster than learning compensates.

3. **The fundamental issue is information-theoretic.** Scalar global
   DA × eligibility cannot disambiguate which of the ~25 active
   language_input neurons is responsible for which of 4 motor pool
   firing patterns. More iterations of the same ambiguous signal
   don't help — they just add noise to noise.

This is the textbook prediction for global scalar feedback (Frémaux
& Gerstner 2016 §6) and matches our empirical result.

## Implication for the post-verdict plan

✅ **Step 2a done: rule is not LR-limited.** Higher LR doesn't rescue
the rule. The bottleneck is structural, not parametric.

➡ **Step 2b: orthogonal-cues experiment launched.** If the W→A
failure is partly caused by overlapping input codes (random hash
gives 2-3 neuron overlap per cue pair), non-overlapping banded
codes might rescue 3-factor. ETA ~100 min from launch.

If orthogonal cues also gives 0-1/6 → input encoding isn't the
bottleneck. Rule is genuinely inadequate. Dendritic learning warranted.

If orthogonal cues gives ≥4/6 → input encoding ambiguity was a major
factor. Cheap fix; dendritic learning may not be needed for this task.

## Files

- This finding
- Sweep config: `experiments/bio_three_factor_high_lr.yaml`
- LR 5x JSONs: `research/findings/raw/g11_bg/text_eval_3factor_tf_lr_5x_seed*.json`
- LR 10x JSONs: `research/findings/raw/g11_bg/text_eval_3factor_tf_lr_10x_seed*.json`
- Original verdict: `research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`
- Step 1 validation: `research/findings/2026-05-05-gradient-passes-permuted-label-VALIDATED.md`
