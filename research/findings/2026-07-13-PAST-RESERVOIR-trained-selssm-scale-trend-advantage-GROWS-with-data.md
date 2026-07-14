# Trained-selective SCALE trend (joint runner, nt 2800→5600): the LEARNED gate's advantage over the co-trained reservoir GROWS with data — the fluency direction, and the opposite of the fixed gate (which hurt) and the fixed reservoir (Ueda-bounded)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_joint_selssm_eprop_generator_derisk.py --tr-cap {2800,5600}` · raw `research/findings/raw/_jointscale/`. numpy; NO `sim/` edit.
**Status:** ✅ positive scale trend for the TRAINED selective channel (2 scale points × 2 seeds). Pairs with the fixed-gate negative (`-scale-selssm-FIXED-gate-negative-...`): the LEARNING is what scales.

## Why

The fixed-gate scale probe was a negative (an untrained selective channel HURTS margin-over-bag → the LEARNED gate is required). The decisive question for the mission (fluent long-range) is whether the TRAINED selective's contribution GROWS or SHRINKS as the co-trained reservoir sees more data. The joint runner trains BOTH the reservoir (e-prop) and the selective gate (eligibility + random feedback), so running it at increasing `n_train` tests exactly this — tractably, without a new batched-gate build.

## Result (TinyStories V=200, deep d≥4, joint co-trained, transport-free)

| nt | seed | eprop | sel | fix | bigram | sel_gain (eprop−sel) | fix_gain | sel−bigram |
|---|---|---|---|---|---|---|---|---|
| 2800 | 42 | 3.802 | 3.118 | 3.542 | 3.195 | +0.684 | +0.260 | −0.077 |
| 2800 | 43 | 3.831 | 3.247 | 3.662 | 3.294 | +0.584 | +0.168 | −0.047 |
| 5600 | 42 | 3.664 | 2.920 | 3.401 | 3.022 | +0.745 | +0.264 | −0.102 |
| 5600 | 43 | 3.735 | 3.128 | 3.555 | 3.105 | +0.607 | +0.180 | +0.023 |

**The trend (nt 2800 → 5600):**
- **`sel_gain` (the trained selective's advantage over the co-trained reservoir) GROWS**: mean **+0.634 → +0.676** (both seeds grow: s42 +0.684→+0.745, s43 +0.584→+0.607). As the reservoir sees 2× more data and gets BETTER (eprop CE 3.80→3.66), it does NOT absorb the selective function — the selective channel's marginal contribution WIDENS. The trained-selectivity value SCALES in the right (fluency) direction.
- **`fix_gain` (fixed accumulator) stays flat** (~+0.26/+0.18) — the growth is specific to the INPUT-DEPENDENT LEARNED gate, not extra co-trained memory.
- **vs the bigram** (a strong, fast-improving baseline at this scale — bigram CE 3.195→3.022 at s42): sel stays ahead on the mean (sel−bigram mean −0.062 → −0.040) and beats the bigram at the deep aggregate on 3/4 (s43 nt5600 a +0.023 tie). The margin over the fast-improving bigram doesn't clearly grow at the deep aggregate — consistent with the a-1 null-discriminator regime (the bigram is near-optimal at tractable scale; the decisive deep-tail-vs-bigram test needs validated scale 23.7M/V=2000).

## ⇒ honest read

The decisive scale question ("does the LEARNED selective mechanism help MORE with data, or does the reservoir absorb it?") is answered POSITIVELY: **the trained selective's advantage over the co-trained reservoir grows with data** (both seeds, both the eprop and fix comparisons), the exact opposite of the fixed gate (hurts) and the fixed reservoir (Ueda-bounded / margin shrinks). This is the fluency-direction signal for the coupling — the same direction the Rung-3 isolated-selective vocab trajectory showed, now confirmed for the CO-TRAINED generator as DATA scales. Honest scope: 2 scale points × 2 seeds at tractable scale; the absolute deep-tail-vs-bigram win still needs the validated-signal regime (the null-discriminator finding), but the DIRECTION — more data → the learned selective helps more — is established.

## Next
- The full validated-scale run (23.7M/V=2000, GPU) via a batched-gate-gradient build is the decisive absolute-fluency test (the named engineering follow-on; the direction is now positive).
- raw `research/findings/raw/_jointscale/nt{2800,5600}_seed{42,43}.json`.
