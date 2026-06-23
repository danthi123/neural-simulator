# Fully-spiking-C1, op 2 of 3 — GELU SPIKING = GO (0.991): a 25-knot rectified-basis transfer through the SHIPPED graded read, NO `sim/` edit (2026-06-23)

**The generator's 2nd of 3 parameter-free nonlinearities is now SPIKING on the bridge. GELU via a calibrated
25-knot rectified-basis transfer realized through the shipped graded read (`a_cont`, `bridge.py:6144`) on a live
Izhikevich pool — full-block fidelity 0.9911 spearman / 0.9930 cosine vs the all-host-read C1 teacher (bar ≥0.90,
clears with headroom; > the LayerNorm op's 0.962). NO `sim/` edit.** `research/runners/_genseq_spiking_gelu_derisk.py`,
real Gen-F block-0, live GPU bridge.

## Result
- **Block fidelity: spearman 0.9911, cosine 0.9930** (≥0.90 bar). Noise-free arm == C1 EXACTLY (PWL-fit gap +0.0000).
- The GELU input (the MLP hidden pre-activations `LN2(x1)@W1+b1`) is tightly LN-bounded: range [-3.31, 4.28],
  std 0.68, ZERO mass beyond ±6 (no fat tails — the premise held exactly).
- Calibration: a 25-knot rectified basis `GELU(x) ≈ c0 + Σ a_k·relu((x−knot_k)/scale)`, knots concentrated where
  GELU bends (near 0), fit ONCE off-line on a fixed [-6,6] grid (NOT per-token, NOT on test data), fit max-err 0.0055.
- On-bridge realization: each knot-neuron is driven with `(x−knot_k)` on a live GPU Izhikevich pool; the
  Izhikevich-2007 read-back recovers the membrane offset (~1e-6, the same exact-linear-inverse the LayerNorm op uses),
  then the shipped graded transfer `a_cont = clip((v−rest)/scale, 0, 1)` gives the rectified-basis value (the live
  neurons' rectify+saturate membrane response). The host only lays out the K drives + combines them with the fixed
  coefficients.

## Anti-cheats (all pass)
- **Specificity:** matched 0.991 vs mismatched 0.127 → margin 0.864 (not a constant).
- **Load-bearing lesion:** identity-GELU 0.787, zero-GELU 0.703 — both drop below the full 0.991 (the MLP
  nonlinearity does work).
- **Residual-floor-aware:** no-GELU floor 0.787 (the block is residual `out = x1 + W2@GELU(...)`); spiking-GELU
  recovers 96% of the exact-GELU lift over that floor.
- **Pool-noise honesty:** the headline 0.9911 is WITH ~1/√64 graded-pool SEM noise on every basis read; it costs
  only −0.0089 vs the noise-free 1.0000.

## The residual (honest)
The only deterministic approximation is the rectified-basis fit (max-err 0.0056 over the range), negligible vs the
GELU output range [-0.17, 4.28] — the noise-free arm reproduces C1 exactly. The pool-noisy per-element outlier (1.16)
is rate-coded-read SEM noise (mean per-element ~0.14), NOT a fit failure — already absorbed in the −0.0089 cost.
No fat-tail miss (zero input mass beyond ±6; grid spans [-6,6]).

## Scope + next (fully-spiking-C1 = 2/3)
LayerNorm (0.962) + GELU (0.991) DONE. **Only softmax remains** — the content-dependent exponential normalization
`softmax(qkᵀ/√d)`, the scoping's genuine rate-code BOUNDARY candidate (the exponential's dynamic range + the
normalization over a content-dependent set, unlike the two fixed parameter-free transfers). The last fully-spiking-C1
op — it either completes the fully-spiking generator or precisely maps the wall. NO `sim/` edit anywhere in the GELU op.
