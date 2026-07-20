# gap#1 M0 — the wall is BIASED encode error, NOT raw fidelity. i.i.d. noise crosses zero at corr ~0.80.

The research gate's #1 recommendation (the M0 meta-de-risk: measure the fidelity->deep-NLL curve on the DEPLOYED
accumulated state before building any encode) — run, and it OVERTURNS the "need near-1.0 fidelity" framing.

## The measurement — inject calibrated per-channel i.i.d. Gaussian noise into the exact v_t, sweep the fidelity

Additive per-channel Gaussian noise (x per-channel std) on the exact host `v_t` BEFORE it charges `cp_ssm_state`,
on the validated harness (M1 re-confirmed at noise 0). NO new mechanism; a ~6-line addition to the M1 path.

| inject-noise | accumulated cp_ssm_state corr | deep d10-99 vs-trigram |
|---|---|---|
| 0.0 | 1.000 | **+0.542** (M1 control ✓) |
| 0.2 | 0.976 | +0.510 |
| 0.35 | 0.933 | +0.410 |
| 0.5 | 0.877 | +0.241 |
| **0.7** | **0.794** | **-0.080 (crosses ~zero)** |
| 1.0 | 0.678 | -0.754 |
| 1.4 | 0.559 | -1.943 |

**The i.i.d. curve is GRACEFUL and crosses zero at corr ~0.80.** An UNBIASED encode reaching corr ~0.80-0.85 would
be roughly break-even to GO — a FAR more forgiving target than "near-1.0."

## THE DECISIVE COMPARISON — the actual encodes are ~1.5-2 nats WORSE than i.i.d. at the same corr

| accumulated-state corr | i.i.d. noise (M0) | ACTUAL ENCODE |
|---|---|---|
| 0.616 | ~-1.1 (interp) | **NEF regression: -2.904** |
| 0.501 | ~-2.4 (extrap) | **token-SDR: -3.416** |

**At the SAME accumulated-state corr, the real encodes cost ~1.5-2 nats MORE than i.i.d. Gaussian noise.** So the
encode error is NOT i.i.d. — it is STRUCTURED / BIASED (a systematic component that accumulates coherently over the
recurrence and correlates across channels/time), which is disproportionately damaging to the deep-context read-out
vs zero-mean noise.

## ⇒ THE REFRAME (this redirects the whole gap#1 build)

- **The wall is NOT "raw fidelity too low to reach near-1.0" (seemingly impossible on the rate-code floor).** It is
  **"the encode error is BIASED."** An UNBIASED encode at corr ~0.80-0.85 would be GO.
- **The likely bias source:** the encodes deliver `v_t` via a NON-NEGATIVE conductance `g_e` with rectification /
  dead-zone / spontaneous-firing floor (the dual-nonneg relu split), producing a systematic (non-zero-mean) error
  that the M1 host path (exact signed `v_t`) does not have.
- **Two cheap new directions this opens (both cheaper than a new encode mechanism):**
  1. **DE-BIAS the existing encode** — estimate and subtract the encode's systematic per-channel offset (a fixed
     bias-correction, since `v_t` is a fixed dictionary of V vectors). If the residual becomes ~i.i.d., NEF's corr
     0.616 would move from -2.904 toward the i.i.d. ~-1.1, and a modestly better corr crosses zero.
  2. **RF PHASE code (the gate's #1)** — phase/latency error tends to be SYMMETRIC/UNBIASED (a timing jitter around
     the true phase), unlike the rate code's rectification bias. If its error is unbiased, corr ~0.85 suffices.

## Pre-flight for the next build (the day's lesson, baked in)

Before pre-registering ANY encode: (a) re-confirm M1 at noise 0 (+0.542); (b) measure the encode's error
DISTRIBUTION on the deployed accumulated state — specifically its MEAN (bias) per channel, not just corr; (c) the
GO target is now "unbiased error at corr >= ~0.82," derived from THIS curve, not the old "near-1.0."

## Status

M0 done, ~1 hour, no new mechanism. It converts an apparently-hard wall (need near-1.0 spiking fidelity) into a
DIFFERENT, more tractable problem (make the encode error UNBIASED; the i.i.d. tolerance is corr ~0.80). The cheapest
next de-risk is the DE-BIAS of the existing NEF/token-SDR encode; the RF phase code is the principled build if
de-bias confirms bias is the issue.
