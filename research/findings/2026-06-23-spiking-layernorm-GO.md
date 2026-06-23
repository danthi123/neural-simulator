# Fully-spiking-C1, op 1 of 3 — LayerNorm SPIKING = GO (0.962): routed through the SHIPPED `sim/` norm circuits, NO `sim/` edit (2026-06-23)

**The first of the generator's 3 parameter-free nonlinearities is now SPIKING on the bridge (not a host read).
LayerNorm via the shipped `enable_input_mean_adapt` (subtractive) + `enable_input_divisive_norm` (divisive) +
affine-on-read — full-block fidelity 0.962 spearman / 0.967 cosine vs the all-host-read C1 teacher (bar ≥0.90).
NO `sim/` edit (opt-in flags only).** `research/runners/_genseq_spiking_layernorm_derisk.py`, real Gen-F block-0,
live GPU bridge.

## Result
- **Block fidelity: spearman 0.9620, cosine 0.9668** (≥0.90 bar). Sanity: RF-weights + host-LN == C1 (spearman 1.0).
- Centre (μ) → `enable_input_mean_adapt` (`bridge.py:6238`), on-bridge read-back vs `x−μ` = **1.2e-6**.
- Scale (1/spread) → `enable_input_divisive_norm` (`bridge.py:6190`, L1 / mean-abs spread), read-back = **1.1e-3**.
- Affine (w,b) rides on the read (as in C1). Both arms run inside the real `_run_one_simulation_step` on a live GPU bridge.
- **ANTI-CHEATS PASS:** specificity 0.810 (matched 0.962 vs mismatched 0.152); both arms load-bearing
  (drop-centre −0.026, drop-scale −0.030, residual-floor-aware); no-norm residual floor 0.917 (the Gen-F block is
  residual, so raw `x` already ~0.92) << spiking-LN 0.962 (recovers 55% of the exact-RMS-LN lift over the floor);
  pool-noise honesty +0.0007.

## The approximation (honest)
mean-abs L1 vs exact RMS √var = **+0.0368** (the dominant, only meaningful residual). The L1 divisor differs from RMS
by a per-token scalar (per-token cosine 0.9998) → rescales `h` → shifts the softmax attention temperature → a small
downstream block effect. Exact-RMS divisive norm (reaches 0.9988) would close most of it but needs a square+sqrt
circuit (heavier than the shipped L1 op) — reachable if the +0.037 ever matters for generation. For LayerNorm-spiking
alone, comfortably within the bar.

## Scope + next (fully-spiking-C1 = 1/3)
LayerNorm-spiking DONE (the cheapest, machinery shipped). softmax + GELU remain host reads → the follow-ons:
**GELU next** (the next cheap — a graded/fitted-neuron read; its input is LN-bounded → accurate), then **softmax**
(the genuine boundary candidate — the content-dependent exponential / rate-code wall, per the scoping). NO `sim/`
edit anywhere in the LayerNorm op.
