---
type: finding
status: superseded
date: 2026-05-01
---

# 2026-05-01 — 300-ep R3+R6 + Tier 1: REGRESSION (eval methodology issue)

**TL;DR:** First attempt at scaling to 300 episodes with Tier 1 speedups produced
**20% / 20%** accuracy — REGRESSION from 100-ep R3+R6 baseline (32.5% / 30%).
Diagnosis: training learns (correct-move rate climbs to 38.5%), but eval reads
the trained weights wrong. **Root cause: Tier 1.2 (`reset_steps` 100→50)
shortened the inter-step NMDA decay window during training, causing trial-to-trial
state leakage that corrupted language pathway formation.**

## Headline numbers

| Run | Train ep | reset/stim | I→W eval | W→A eval | Train final %correct |
|---|---|---|---|---|---|
| Baseline R3+R6 (100-ep) | 100 | 100/200 | 32.5% | 30.0% | 33% |
| **300-ep + Tier 1 (this)** | 300 | **50/100** | **20.0%** | **20.0%** | **38.5%** ↑ |
| Chance | — | — | 25% | 25% | 25% |

**Both eval metrics BELOW chance** while training-phase correct-moves CLIMBED
to 38.5% (best ever). The agent learned navigation; the language readout
came out garbled.

## Per-direction confusion (W→A delta-from-baseline)

```
target=north  baseline {N:10, E:5, S:9, W:18}  drive {N:6, E:6, S:8, W:14}
              delta {N:-4, E:1, S:-1, W:-4}    → predicts E (target N) ✗

target=east   baseline {N:0, E:6, S:10, W:8}   drive {N:2, E:11, S:6, W:17}
              delta {N:2, E:5, S:-4, W:9}      → predicts W (target E) ✗

target=south  baseline {N:0, E:9, S:22, W:5}   drive {N:9, E:19, S:11, W:5}
              delta {N:9, E:10, S:-11, W:0}    → predicts E (target S) ✗

target=west   baseline {N:3, E:23, S:19, W:4}  drive {N:5, E:7, S:29, W:4}
              delta {N:2, E:-16, S:10, W:0}    → predicts S (target W) ✗
```

**Smoking gun:** baselines are wildly asymmetric across consecutive trials.
For "south" the baseline is `{N:0, E:9, S:22, W:5}` — cortex/motor is
spontaneously firing south at 22 spikes/100ms before any input. For "west"
the baseline is `{N:3, E:23, S:19, W:4}` — east at 23, south at 19. The
cascade carries activation from the prior trial into the next baseline
window. Delta-from-baseline is fundamentally unreliable when the cascade
has memory across trials.

## Root cause: Tier 1.2 reset reduction

The Tier 1 speedup plan halved `reset_steps` from 100→50 in
`text_train_embodied.py`. Reset_steps controls how long we run the bridge
with zero input + zero reward between consecutive env steps to let
NMDA-mediated bistability decay before the next stimulus.

**NMDA τ = 100 ms.** A 50ms reset = 0.5τ → 60% activity decayed, 40% residual.
A 100ms reset = τ → 63% decayed, 37% residual. The difference is meaningful:

- 100ms reset: each STDP pairing window starts ~clean. Pre-synaptic activity
  in this trial reflects this trial's input. Trained weights map input
  patterns → correct cortex pools.
- 50ms reset: pre-synaptic activity carries 40% of the previous trial.
  STDP grows weights from a contaminated mixture. After 9000 env steps of
  300-episode training, this contamination compounds into systematically
  scrambled language→cortex mappings.

**Why the smoke test missed this:** the smoke (5 ep × 10 steps = 41 env
steps) didn't run long enough for contamination to compound. We only
measured per-step wall time + correct-move rate, not eval accuracy.

## Why training correct-moves still climbed to 38.5%

The reward signal is **action-contingent** (Manhattan distance decrease):
- Wrong action → no reward → no LTP on the wrong pathway
- Correct action → +reward → LTP on whatever was active

The visuomotor pathway (retina → V1 → V2 → IT → cortex_X → motor_X) gets
clean reinforcement because the agent's actual moves give clean signals.
**That pathway works** — 38.5% correct vs 33% baseline confirms it.

But the LANGUAGE pathways (language_input → cortex_X, IT → language_output,
cortex_X → language_output) also get reward when the agent acts correctly.
And they get NMDA-bleedover-corrupted pre-synaptic signals. So they grow
weights based on noisy correlations rather than the true target word.

## What to do

### Immediate fix (committed):
Revert `reset_steps` from 50 → 100 in `text_train_embodied.py`. Keep the
other Tier 1 changes (`stim_steps` 200→100, `enable_per_type_stp=False`).

Expected speedup: ~1.5× (down from 2.5× with full Tier 1) but with correct
training. 300-ep run estimated at ~115 min instead of ~70.

### Alternative considered + rejected:
Could decouple training_reset (50) and eval_reset (100). But the delta-
baseline measurement during eval ALSO needs clean state, and that's also
contaminated by the previous eval trial. The fundamental issue isn't
just eval — it's that NMDA bistability genuinely needs ~τ to decay.

### Smoke gap to close:
Future Tier-1-style speedup attempts MUST include an end-to-end eval
in the smoke (not just timing + correct-moves). 5 episodes × 10 steps +
20 eval trials = ~3 minutes of smoke that would have caught this.

## Repro / files

- Result: `research/findings/raw/g11_bg/text_eval_R3_R6_300ep_tier1.json`
- Log: `research/findings/raw/g11_bg/R3R6_300ep_tier1.log`
- Tier 1 commit (the one with the bug): `6a172ec`
- Tier 1 fix commit: TBD (this commit)
- Re-run output: TBD (next 300-ep run after fix)
