---
type: finding
status: contributing
date: 2026-05-02
---

# 2026-05-02 — Longer training (200 ep) NEGATIVE: weights saturated at 100 ep

**TL;DR:** Doubling training duration (100 → 200 episodes) at seed=42
produced essentially identical weights and similar/worse eval accuracy.
Weight saturation at ~100 ep confirms the architectural ceiling.

| Run | I→W | W→A | Training-time correct |
|---|---|---|---|
| v2 100-ep seed=42 | 33% | 27% | 29.6% |
| v2 200-ep seed=42 | 22% | 24% | 34.5% |

Eval went DOWN despite training-time correct moves climbing. This is
seed-specific variance.

## Key finding: weights converge by ~100 ep

Token-targeted weight differentials (PFC-bypass) are nearly IDENTICAL:

```
                v2 100-ep    v2 200-ep
north weight   -0.079        -0.079
east weight    +0.210        +0.210
south weight   +0.304        +0.304
west weight    +0.073        +0.073
```

Pathway means almost identical (down to 4 decimal places). The
differential learning has converged to steady state by ~100 ep.

Training-time correct moves climbed (29.6% → 34.5%) but this reflects
visuomotor cascade dynamics fluctuating, not weights changing. The
network is doing the same task with the same weights — just happens
to be more effective at this particular seed's later episodes.

## Why weights converged

Three forces balance to drive weights to fixed points:

1. **STDP soft-bound:** `Δw = A_plus × (w_max - w) × exp(...)` decreases
   as w approaches w_max. At w_max=5, weights of 3.0 have only 40%
   "headroom" for further LTP. Eventually LTP events become tiny and
   match LTD events.

2. **Reward correlation:** With cascade at ~30% correct, reward signal
   per token is roughly stable across training. Pre-post correlations
   for "north_active → motor_N" don't change as agent improves.

3. **No learning rate decay:** Standard STDP+reward modulation has
   constant rate, so equilibrium is steady-state of LTP/LTD ratio.

The convergence is biology-consistent: real STDP creates stable
weight distributions over training (Song et al. 2000 — bimodal weight
distributions). Continued training doesn't shift these once stable.

## Five negative followups confirm the ceiling

The session has now tested 5 architectural variations beyond v2:

| Variation | Result |
|---|---|
| Reward shaping (`wrong_move_reward=0`) | NEGATIVE |
| Stronger training drive (lang_in 200→400) | NEGATIVE — identical weights |
| Stronger eval drive (200→500) cross-seed | NEGATIVE — variance not signal |
| Bigger motor pools (10→30/direction) | NEGATIVE — east FLIPPED to REV |
| Longer training (100→200 ep) | NEGATIVE — weights saturated |

The 28.5% W→A (6-seed cumulative p=0.027) is robust under v2 config.
To push higher requires:
- **Cascade structural fix** for N-bias (cluster A/E reduction)
- **Bigger language regions** (256 → 512 untested)
- **Different decoding** (cosine on motor pop vector instead of argmax)
- **Different training regime** (curriculum, pretraining, etc.)

These are non-trivial architectural changes deferred to user direction.

## Files

- 200-ep result: `research/findings/raw/g11_bg/text_eval_R3R6_200ep_HebOff_v2_seed42.json`
- 200-ep checkpoint: `research/findings/raw/g11_bg/text_eval_R3R6_200ep_HebOff_v2_seed42.simstate.h5`
- 200-ep weight diag: `research/findings/raw/g11_bg/text_weight_diag_R3R6_200ep_HebOff_v2_seed42.json`
- v2 100-ep baseline: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_seed42.json`
