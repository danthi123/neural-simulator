# 2026-05-02 — Reward shaping NEGATIVE: north reversal isn't reward-asymmetry

**TL;DR:** Setting `wrong_move_reward=0` (eliminating the -0.5 LTD penalty
for wrong moves) at seed=42 did NOT fix the north reversal. Result is
nearly identical to v2 with default reward (-0.5 wrong, +1.0 right):

| Run | I→W | W→A | North weight diff |
|---|---|---|---|
| v2 (default reward) | 33% | 27% | -0.079 REV |
| v2 + wrong_move_reward=0 | 33% | 25% | -0.076 REV |

The hypothesis was that asymmetric LTP/LTD pressure (70% LTD events at
-0.5 vs 30% LTP events at +1.0, aggregate LTD magnitude exceeds LTP)
caused the consistent north reversal across seeds. **This hypothesis
is falsified.** Same north reversal with or without negative reward.

## Implications

The cascade structural N-bias (cortex_N fires 2x baseline) is the
dominant factor. Even with purely positive reward, motor_N fires for
non-north targets too, so STDP can't grow "north_active → motor_N"
differential preference. The structural firing pattern of the cascade
overrides the differential learning signal.

**Architectural fix needed.** Candidates:
1. Reduce cluster A topographic bias toward cortex_N
2. Reduce cluster E weight toward cortex_N
3. Add per-direction lateral inhibition between cortex pools (motor-WTA-like)
4. Tune cortex_N initial firing threshold higher (homeostatic shift)

## Per-direction comparison (v2 vs NoLTD, seed=42)

```
I→W:
  north: v2 32%   NoLTD 36%   (slightly better)
  east:  v2 45%   NoLTD 36%   (worse)
  south: v2 20%   NoLTD 20%   (same)
  west:  v2 36%   NoLTD 39%   (slightly better)

W→A:
  north: v2 28%   NoLTD 36%   (better!)
  east:  v2 20%   NoLTD 12%   (much worse)
  south: v2 40%   NoLTD 24%   (worse)
  west:  v2 20%   NoLTD 28%   (better)
```

Removing negative reward seems to redistribute accuracy across
directions but doesn't increase total. East and south did worse,
north and west did better. Consistent total.

## Conclusion

Reward shaping is NOT the path to fixing the north reversal. The
cascade structural N-bias is the issue.

`wrong_move_reward=0` is kept as a CLI option (committed in d44b82c)
but should NOT be the default. Default `-0.5` keeps overall accuracy
similar with slightly more direction balance via standard learning.

## Files

- Result: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_NoLTD_seed42.json`
- Checkpoint: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_NoLTD_seed42.simstate.h5`
- Weight diag: `research/findings/raw/g11_bg/text_weight_diag_R3R6_HebOff_v2_NoLTD_seed42.json`
- v2 baseline: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_seed42.json`
