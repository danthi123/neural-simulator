# Population read (pool-k) is NOT the deep-credit-on-spikes lever — flat multi-seed, and the arm margin needs trained scale (2026-07-12)

**One-line verdict:** the K-normalized pool-k sweep (`_semantic_inheritance_onbridge_spiking_derisk`, K=1/8/16 × seeds 42/43/44, fanned out across cores) shows **NO population-read benefit multi-seed** — `read_snr_corr` is FLAT in pool-k (K1 **0.288** / K8 **0.292** / K16 **0.277**) and the biological-credit arm never beats plain feedback-alignment (`mc_beats_fa=False`, `bp_beats_fa=False` at every K/seed). The earlier "rising" signal (K8 seed-42 0.423 > K1 seed-42 0.322) was a **single-seed artifact** inside the large seed variance. Population read is ruled out as the credit-noise lever; the genuine next mechanism is the dendritic input-representation substrate (the standing priority) and/or a trained-scale run (the batched-forward infra) to resolve the arm margin the smoke is too under-trained to expose.

## The question this gate resolved

This session's convergence localized the deep-credit-on-spikes boundary to two halves; this sweep tests the **credit-noise half**: does a population (pool-k) read average down the finite-sample spike-count noise enough that (1) the credit read cleans up (`read_snr_corr` = corr(pooled-E, soma-rate) rises) AND (2) the D1-predicted biological-credit advantage appears (the microcircuit arm beats plain-FA)?

**A-1 reconciliation (corrects a prior AUTONOMOUS_STATE overstatement):** the 2026-06-22 popcode-NEGATIVE finding rules out population read ONLY for a *deterministic graded* readout (within-pop std = 0.00 → nothing to average). The kfair read is the *stochastic spike-count* read, which DOES have per-neuron noise to average — so an empirical sweep was warranted, not an a-priori ruling. This sweep IS that empirical test.

## Results

**read_snr_corr (mechanism: does the pooled read clean up?) — FLAT in pool-k, high seed variance:**

| pool-k | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| K=1  | 0.322 | 0.147 | 0.396 | **0.288** |
| K=8  | 0.423 | 0.242 | 0.210 | **0.292** |
| K=16 | 0.266 | 0.298 | 0.267 | **0.277** |

The means are statistically indistinguishable (0.277–0.292); the per-seed spread (0.15–0.42) dwarfs any K trend. **No population-read cleanup.**

**Arm margin (function: does biological credit beat plain-FA?) — absent, task under-trained:** at every K/seed, `microcircuit_inherit == plain_fa_inherit` (~0.22–0.26), `burstprop_inherit` lower (~0.22), all near the `single_layer_inherit` floor (0.222) and barely above `permuted_inherit` (0.148). `bio_beats_fa=False`, `mc_beats_fa=False`, `bp_beats_fa=False`. The controls are intact (`oracle_inherit=1.0`, `oracle_ok=True`, `apical_lesion_inherit=0.074`, `oracle_memctrl=0.0`) — the harness is valid; the on-bridge arms are simply under-trained at smoke scale (H40/ep30), so no credit rule can separate.

## Verdict + honest scope

- **Population read (pool-k) is NOT the credit-noise lever** — no `read_snr_corr` lift and no arm margin, multi-seed. Combined with the a-priori structural argument (a sparse point-neuron spike-count read lets the ~30 identity dims dominate the ~3 class dims — structural swamping, not averageable noise), the weight of evidence points AWAY from more neurons-per-feature.
- **The arm-margin-at-scale question is OPEN, not closed by this smoke.** All arms sit at the single-layer floor because H40/ep30 under-trains the depth-2 net (the K1 verdicts already flagged "scale before the contrast is readable"). Whether the biological-credit advantage appears once the net is actually TRAINED is unresolved — and the on-bridge per-example spiking forward is too slow to train it at smoke (each K16 run = 1472-neuron bridge, per-example, ~60–160 min on numpy-CPU; a demonstrated wall-clock wall).

## Next (two sharpened mechanisms, both grounded)

1. **The batched on-bridge forward (scale enabler).** The demonstrated wall-clock wall is the per-example spiking forward; a batched forward (Izhikevich-step CUDA-graph + on-device firing accumulation + example batching) makes a TRAINED-scale arm comparison feasible — which resolves whether biological credit beats plain-FA when the net is actually trained. This is the same infra the emergent-generation frontier needs for its data-scale run, so it unblocks two decision-critical runs. Research-gate + cheap-first de-risk before the full build.
2. **The dendritic input-representation substrate (the standing priority, `project_dendritic_cortex_for_emergence`).** The point-neuron spike-count read structurally cannot let W_in learning suppress a dominant confound before the read (this sweep adds the empirical "pooling doesn't fix it" datapoint). A substrate carrying a clean continuous dendritic error is the genuine path past both the read-variance and structural halves.

**Rigor:** 9 runs (K∈{1,8,16} × seeds 42/43/44), fanned out across cores (the serial `_kfair.sh` was single-threaded — drift #6 — killed + relaunched as 5 concurrent 1-thread procs). Controls intact every run (oracle 1.0, permuted ~0.15, apical-lesion ~0.07). NO `sim/` edit. A first-class honest negative: it maps what population read on the point-neuron substrate can't do, and points to the dendritic substrate + trained-scale validation.
