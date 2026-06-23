# Fully-spiking-C1, op 3 of 3 — softmax SPIKING = GO (0.9998): the scoping's BOUNDARY candidate does NOT bite on the trained Gen-F (max-subtract bounds exp); fully-spiking-C1 COMPLETE, NO `sim/` edit (2026-06-23)

**The generator's LAST + hardest nonlinearity is now SPIKING on the bridge — the predicted rate-code wall does NOT
bite. Softmax via a calibrated graded `exp` read (21-knot, reusing the GELU mechanism) + the shipped divisive-norm
circuit (`enable_input_divisive_norm`, bridge.py:6190) for the sum — full-block fidelity 0.9998 spearman / 0.9999
cosine vs the all-host-read C1 teacher (noise-free == C1 exactly). ⇒ ALL 3 nonlinearities now spiking (LayerNorm
0.962 + GELU 0.991 + softmax 0.9998) + every learned matvec exact-on-RF = THE GENERATOR IS FULLY SPIKING ON THE
BRIDGE.** `research/runners/_genseq_spiking_softmax_derisk.py`, real Gen-F block-0, live GPU bridge. NO `sim/` edit.

## Why the boundary candidate didn't bite (the key result)
The scoping predicted softmax as the genuine rate-code wall (a content-dependent exponential normalization). The
MEASURED logit range explains why it isn't, on THIS model: the standard softmax **max-subtract** makes all logits ≤0,
so the post-subtract range is **[−3.96, 0.0]** → `exp` dynamic range **52×**, well inside the graded read's ~4-decade
band. The max-subtract bounds `exp`'s input EXACTLY like LayerNorm bounds GELU's input → the same calibrated graded
read tracks it.

**HONEST regime caveat (recorded precisely):** a LOW-TEMPERATURE softmax (logits ≪ [−9, 0], `exp` dynamic range
> 1e4) WOULD saturate the graded clip and lose the small-weight tail — THAT is the genuine rate-code wall, and it
would need a native spiking-`exp` / log-domain / expansive-f-I primitive to close. The trained Gen-F simply does not
reach that regime, so the wall does not bite here. (Raw scores `qkᵀ/√d` range [−1.78, 2.71], std 0.45.)

## Result + mechanism
- **spearman 0.9998, cosine 0.9999.** Noise-free on-bridge softmax == C1 teacher exactly (exp-fit+denominator gap
  +7e-6); graded-pool-noise cost only +0.0002.
- **exp (a):** 21-knot rectified-basis graded read (the GELU mechanism verbatim) — each knot-neuron driven with
  `(s−knot)`, Izhikevich read-back recovers the offset ~1e-6, the shipped `a_cont` computes the rectifier; on-bridge
  exp max-err 0.0027.
- **sum-norm (b):** the shipped divisive-norm circuit with per-row `gain = n_keys` → `D = sigma + sum(exp)` = the
  softmax denominator; membrane read-back vs exact = 1.0e-3. n_keys 1..90 (a structural causal-mask quantity) handled
  by the per-row gain.

## Anti-cheats (all pass)
- **Specificity:** matched 1.000 vs mismatched 0.122 → margin 0.878.
- **Load-bearing lesion (uniform attention):** at the attention-OUTPUT level (undiluted by the residual),
  spiking-softmax 0.9925 vs uniform 0.9046 (margin 0.088 — meaningful: a value-mix is a convex combination so it
  can't fall to chance). Full-block (residual-floor-dominated): uniform 0.9973 vs spiking 0.9998 → recovers 93% of
  the exact-softmax lift.
- **SUM-vs-MEAN control:** divisive-by-mean (linear-attention, no `gain=n_keys`) collapses to 0.378 → the
  gain-corrected sum-norm is the correct, exact one.
- **Pool-noise honesty:** reported with ~1/√64 SEM on both the exp basis reads and the divisive mean.

## ⇒ fully-spiking-C1 COMPLETE (the generative arc's brain-purity milestone)
LayerNorm + GELU + softmax all spiking; the matvec is exact-on-RF → the consolidated generator runs fully in spikes
on the bridge. Combined with the LOOP DEMONSTRATED (CYCLE 478 — C2 grow-no-forget GO) + C1 (the generator generating
byte-identical), the generative-sequence frontier's load-bearing gates — **C1 (one fully-spiking bridge) + C2
(no-catastrophic-forgetting)** — are BOTH MET at toy scale. NO `sim/` edit anywhere in the 3-op fully-spiking-C1 arc.
Optional future polish: the low-temperature-softmax spiking-`exp` primitive (only if a future model enters that
regime).
