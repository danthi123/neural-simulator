# On-bridge Node Perturbation (small spiking net): the credit is PRESENT but NOT stably convergent — a variance/stability BOUNDARY set by the noisy spike-count readout `dL`; the lever is noise-reduction via averaging (scale). The feedforward NP result is OFF-BRIDGE (numpy) + robust; the on-SPIKE claim is honestly UNPROVEN at small scale.

**Date:** 2026-07-13
**Runners:** `_nodepert_onbridge_derisk.py` (trained readout + population sweep), `_np_onbridge_fixedpool_probe.py` (readout-free fixed-pooling + 2-layer NP + gradient-clip), `_np_onbridge_eligibility_probe.py` (STDP-eligibility route). All numpy small-net; NO permanent `sim/` edit.

## The claim being tested + why it matters
NP's FEEDFORWARD deep credit is robust OFF-BRIDGE (numpy: 6/6 standard + 6/6 fresh seeds, depth 2–6; `2026-07-13-fresh-deep-credit-class-NODE-PERTURBATION-*`). The mission's fully-spiking bar needs it ON the real spiking substrate — the KEY unproven claim (does NP's no-backward-channel property deliver on-spike deep credit where the burst family's SNR failed).

## Result — the on-bridge small-net NP does NOT stably train (NP ≈ frozen across EVERY readout variant), but the credit is PRESENT
| on-bridge variant | outcome |
|---|---|
| trained readout (host delta / ridge on the spiking output) | noise-limited ~chance (`2026-07-13-fresh-...` on-bridge boundary) |
| population-averaged readout (pool_out 20/40/80) | NP ≈ frozen ≈ chance (bigger pools don't unblock) |
| sim's-own three-factor via STDP eligibility | eligibility unreliable (STDP-timing; 2/6 seeds; `2026-07-13-onbridge-NP-STDP-eligibility-route-UNRELIABLE-*`) |
| fixed-pooling readout (NO trained readout) + 2-layer NP | NP=frozen (0.375=0.375) |
| + gradient norm-clipping | NP=frozen again (clipped step too small → random-walk) |
- **BUT the credit is genuinely present** (the load-bearing diagnostic): 5 epochs of NP moves 810/832 synapses AND flips a misclassified input to CORRECT (x1 pooled logits [14,9]→[15,16], argmax 0→1). So NP's on-spike credit WORKS directionally — it is NOT a bug and NOT a dead signal.
- **The boundary is STABILITY, set by the noisy spike-count `dL`:** the directional-derivative `0.5(L(+ξ)−L(−ξ))` is read from DISCRETE spike counts with Poisson-like noise; run-to-run the same input gives different counts, so the `dL` that rides the ξ perturbation is swamped. Un-clipped → huge chaotic weight swings (max|dW|=540/5-epochs) that don't settle; clipped → steps too small vs the noise → random-walk. Neither converges → NP ≈ frozen accuracy. This is NP's zeroth-order variance wall, amplified by spiking-readout noise at small scale.

## ⇒ The lever (a boundary = the next mechanism): reduce the readout-noise floor via AVERAGING (scale)
The Poisson readout noise averages down ∝ 1/√N with more output neurons AND more settle steps (time). Population (neurons) alone failed to pool_out=80; the untested lever is MUCH more time-averaging (long settle) and/or a genuinely larger net — where the spiking-readout `dL` becomes clean enough that NP's (present) credit converges stably. This is the genuine GPU-scale test of the on-spike claim (a bigger on-bridge net + long settle on cupy). Honest status: **the on-SPIKE deep-credit claim is UNPROVEN at small scale — the credit is present but variance-limited by readout noise; the scale/averaging lever is the next arc.** The OFF-bridge feedforward NP result (robust, the mission-critical headline) stands independently.
