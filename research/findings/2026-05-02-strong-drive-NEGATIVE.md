# 2026-05-02 — Stronger drive NEGATIVE: training drive 400 doesn't beat 200

**TL;DR:** Increasing language drives (lang_input 200→400, lang_output_coactive
150→300) AND eval drive (200→500) at seed=42 produced essentially identical
results to v2 baseline:

| Run | I→W | W→A | North weight |
|---|---|---|---|
| v2 baseline (drives 200/150/eval=200) | 33.0% | 27.0% | -0.079 REV |
| StrongDrive (drives 400/300/eval=500) | 33.0% | 25.0% | -0.079 REV |

The weight diagnostic shows nearly identical token-targeted differentials:
- north: -0.079 (both runs, REV)
- east: +0.210 (both runs)
- south: +0.304 (both runs)
- west: +0.073 (both runs)

## Why this is interesting

This suggests the differential weight learning HAS saturated at the current
training duration (100 ep). Stronger drives don't push weights further apart
because:
1. STDP soft-bound (now stdp_w_max=5.0) provides ample headroom
2. Reward magnitude is already strong (+1.0 / -0.5)
3. Active synapses receive similar relative spike pairings regardless of
   drive amplitude

The 100-ep training extracts ~all the differentiable signal available with
current architecture. To push accuracy higher, need to change either:
- More training (longer episodes — more STDP events to accumulate)
- More capacity (bigger language regions, more motor neurons per pool)
- Different cascade structure (fix N-bias)
- Different decoding methodology

Stronger drives alone aren't a free lunch.

## Why eval-drive 500 didn't help here

The reeval sweep on the v2 checkpoint showed W→A=32% at drive=500 (vs 27%
at default). But that was on the SAME trained network, post-training, in
the bridge state with weights but cold-start dynamics.

This new run trained AND evaluated at strong drives. The eval is in-vivo
warm state. Result: 25% W→A, no improvement.

Possibility: cold-start reeval with stronger drive gets a "boost" from
the stronger drive overwhelming residual cascade noise. Warm-state eval
in vivo doesn't have the same noise pattern, so stronger drive doesn't
help similarly.

## Conclusion

Stronger drives alone don't unlock more text I/O capacity. The 28.5%
W→A from 6-seed v2 validation (p=0.027) is the realistic ceiling for
this architecture at 100-ep training.

To exceed: change architecture (region size, cascade structure) or
training duration. Reward shaping (NEGATIVE in earlier test) and
stronger drives (NEGATIVE here) are not the answer.

## Files

- Result: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_StrongDrive_seed42.json`
- Checkpoint: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_StrongDrive_seed42.simstate.h5`
- Weight diag: `research/findings/raw/g11_bg/text_weight_diag_R3R6_HebOff_v2_StrongDrive_seed42.json`
- v2 baseline: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_seed42.json`
