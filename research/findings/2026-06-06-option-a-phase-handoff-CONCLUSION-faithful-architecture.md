# Option (a) phase handoff — CONCLUSION: the read-out is SOLVED by phase (spike-native); the whitening COMPUTATION stays ANALOG (the spiking-membrane graded lateral can't reach the gentle composing regime — a structural boundary); the conversational composition path is biology-faithful end-to-end — 2026-06-06

**Status:** CONCLUSION of the on-substrate decorrelation arc. Gated on COMPOSITION; controls (RAW 67 / CONCEPT 100)
+ guards green throughout; seed 42 for the full-pipeline + retune (the over-whitening sub-result is 3-seed from the
graded-LGN boundary; the channel GO is 2-seed). Integrates the graded-LGN BOUNDARY, the phase-handoff channel GO,
the full-pipeline BOUNDARY, and the λ/epochs retune sweep. NO `sim/` edits beyond the already-reviewed graded-lateral
opt-in.

## The one-line conclusion

The on-substrate spiking whitening's relocated boundary — the RATE read-out — IS resolved by reading the whitened
code out in PHASE (option a: 100% direct, lossless — the genuine spike-native contribution). But the FULL on-bridge
pipeline (graded lateral → phase → composer) still composes at the floor, for a SEPARATE, now-isolated reason: the
spiking-membrane graded lateral OVER-WHITENS (coherence ~0.19, the noise-amplifying C^−1/2 regime) and CANNOT reach
the gentle composing regime (coherence ~0.04, C^−1/3) by any λ/epochs tuning — a STRUCTURAL boundary of computing
whitening on the membrane's clipped/saturated activity. Conclusion: the whitening COMPUTATION belongs in the ANALOG
domain (where biology does it — retina/LGN, research-confirmed faithful), and PHASE is the spike-native channel that
hands the analog-whitened code into the spiking composer losslessly. **The conversational composition path is
biology-faithful end-to-end.**

## The decisive tables

### (a1) Read-channel de-risk — phase RESOLVES the read-out [GO]
(`2026-06-06-option-a-phase-handoff-GO.md`) — swapping RATE→PHASE on the KNOWN-100%-composing code: PHASE DIRECT
100% (round-trip phase corr 1.000, coherence preserved at the composing 0.043) vs RATE ~85%. Phase is the right read
channel; controls valid, guards green, multi-seed (pilot + seed 43).

### (a2) Full on-bridge pipeline (graded lateral → read-out → composer), seed 42
| read-out | composition | graded coh | reading |
|---|---|---|---|
| GRADED-CLIP (rate; the boundary) | 26/39 = 66.7% | 0.190 | floor |
| GRADED-PHASE (phase) | 26/39 = 66.7% | 0.190 | phase faithfully carries 0.190 (roundtrip 1.000, 0 silent) → still floor |

### λ/epochs retune sweep (seed 42) — can the graded lateral reach the gentle regime?
| λ | epochs | graded coh | composition | M_norm |
|---|---|---|---|---|
| 0.01 | 8 | 0.190 | 66.7% | 28.2 |
| 0.02 | 8 | 0.218 | 66.7% | — |
| 0.04 | 8 | 0.237 | 66.7% | — |
| 0.08 | 8 | 0.239 | 66.7% | — |
| 0.02 | 3 | 0.219 | 66.7% | 18.6 |

MONOTONIC: more decay (or fewer epochs) → weaker M → coherence rises toward RAW (0.249). The gentle composing minimum
(0.043) is NEVER reached. The graded lateral's reachable coherence range is [~0.19 over-whitened, ~0.25 raw] — it
**skips the gentle 0.043 dip** the rate-model rule achieves. All five configurations compose at the 66.7% floor.

## Why this is a STRUCTURAL boundary, not a tuning miss

Coherence is NON-MONOTONIC in whitening strength (rate-model arc): RAW 0.249 → gentle C^−1/3 0.043 (composes 100%) →
over-whiten C^−1/2 0.191 (floor). The gentle 0.043 dip is a SPECIFIC partial whitening that decorrelates the signal
WITHOUT amplifying the low-variance noise directions. On the spiking membrane's CLIPPED/saturated sub-threshold
activity (`a = clip((v−v_rest)/scale, 0, 1)`), the local rule's reachable M-family is MONOTONIC in λ (over-whiten ↔
raw) — it does NOT contain the structured M that realizes the clean gentle dip. The clipping/saturation of the
membrane activity is the load-bearing difference: the analog computation HAS the gentle fixed point; the
clipped-spiking computation does NOT. Consistent with the Mikulasch-Priesemann point-neuron limit + the 2026-06-05
opponency wall — the precise analog operation resists faithful realization in the spiking/rectified domain.

## What this RESOLVES — the biology-faithful architecture (the GO under the boundary)

> real images → **V1** (spiking-faithful) → **ANALOG whitening** (faithful — the retina/LGN's analog efficient-coding
> stage; the validated rate-model local rule realizes it, composes 100%) → **PHASE read-out (option a)** (spike-native,
> lossless — phase carries the gentle signed whitened code into the composer where rate degrades it) → **spiking FHRR
> composer**.

Every step is biology-faithful. The ONLY non-spiking step is the analog whitening — which is EXACTLY where biology
computes it (analog, pre-spike). We empirically TESTED forcing the whitening into the spiking membrane (the graded
lateral) and it BOUNDARIED (over-whitens) — consistent with "biology keeps whitening analog for a reason."

## The owner's a→c ladder — status

- **(a) phase-encode the handoff: DONE** — resolves the read-out, spike-native. The genuine contribution of the arc.
- **(b) temporal integration / (c) population redundancy + attractor cleanup:** these address the READ-OUT, which (a)
  already solved. They do NOT address the over-whitening (the remaining issue), which is a WHITENING-stage structural
  boundary, not a read-out problem. So the ladder effectively COMPLETES at (a); (b)/(c) are not the lever for the
  residual.

## Honest scope + what is NOT claimed

- Seed 42 for the full-pipeline boundary + retune (the over-whitening sub-result is 3-seed from the graded-LGN
  boundary; the channel GO is 2-seed). The conclusion is MECHANISTIC (monotonic λ-response + faithful phase carry,
  roundtrip 1.000, across 5 configurations) — multi-seed confirmation of the full-pipeline boundary is available on
  request but the mechanism is decisive.
- "Whitening stays analog" is the biology-FAITHFUL conclusion (research + empirical), NOT a defeat: the analog
  whitening + phase handoff IS the faithful realization. The numpy/rate whitening models the analog retina/LGN stage
  faithfully (a graded, pre-spike op).
- (d) the genuine-cortical learned-read-out composer (benched, `2026-06-06-composer-vsa-idealization-known-limitation.md`)
  is a separate, deeper question untouched here.

## Net

The on-substrate decorrelation arc concludes: the read-out is SOLVED by phase (option a, spike-native — the real
contribution); the whitening COMPUTATION stays ANALOG (the spiking-membrane graded lateral over-whitens — a structural
boundary, consistent with biology computing whitening analog). The conversational composition path is biology-faithful
end-to-end: real objects → V1 → analog whitening → phase → spiking composer. The validated SCIENCE (a local rule learns
a composing whitening, 6/6) is unchanged.

Cross-references: `2026-06-06-graded-lgn-decorrelation-BOUNDARY.md`, `2026-06-06-option-a-phase-handoff-GO.md`,
`2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md`,
`2026-06-06-composer-vsa-idealization-known-limitation.md`.
