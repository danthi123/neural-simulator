# Rate-coded spiking CAN compute whitening — the boundary is LOCAL LEARNING, not computation (verdict FLIPPED) — 2026-06-06

Follow-on to `2026-06-06-realobject-grounding-and-whitening-synthesis.md` (which had concluded the boundary was the
local spiking COMPUTATION of whitening). The decisive computation de-risk FLIPS that — and a methodology-honesty note
matters here because the flip came from catching my own bug.

## The construction (resolved)
The project's ZCA decorrelates CONCEPTS (the N×N gram) — needs all concepts simultaneously present → NOT
substrate-realizable (concepts are sequential). The realizable analogue is **DIMENSION-whitening**: for each concept's
code `x`, the IT-pool DIMS inhibit each other via fixed lateral inhibition `L = C^½ − I` (C = D×D dim covariance), so
`(I+L)⁻¹ = C^−½` and the settled rate `r = C^−½ x` = dim-whitened. This is the fixed Ω-wiring on the realizable axis.

## Two results (`_A_whitening_computation_derisk.py`, 3 seeds)
- **Q1 — DIM-whitening is a VALID realizable target.** It reduces concept coherence from RAW mean/max 0.45/0.96 to
  **0.032/0.037**, matching the (non-realizable) concept-whitening target 0.032. The substrate-realizable decorrelation
  EXISTS (down-weight the shared high-variance dims, up-weight the distinguishing dims → concepts decorrelate).
- **Q2 — rate-coded spiking COMPUTES it.** Stable leaky dynamics `dr/dt = Xc − r − L·r_hat`, with the lateral term
  `L·r` carried by RATE-CODED ON/OFF Poisson spikes (`r_hat`), the membrane integration AVERAGING the rate-noise,
  converges to the analytic whitened codes: concept coherence **0.043 at window=20 (noisiest) → 0.037 at window=2000**,
  matching the analytic 0.037 at every window, all seeds. A NOISELESS control returns 0.036 = analytic (validates the
  solver).

## Methodology honesty (the bug that almost shipped a false wall)
My FIRST computation attempt returned concept coherence **1.0 (re-correlated, worse than raw) at every window incl.
window=2000** — and I nearly recorded it as "the rate-coded spiking whitening wall, confirmed." It was a NUMERICAL
ARTIFACT: the naive fixed-point iteration `r ← Xc − r·Lᵀ` is Euler with dt=1, and `L = C^½−I` has eigenvalues >1 for
high-variance dims → the iteration DIVERGES regardless of rate-noise. Caught by scrutinizing the implausible
"1.0 even at window=2000 (near-zero noise)" — if noise were the cause, low noise would help; it didn't, so the solver
itself was broken. The stable leaky-dynamics solver + the noiseless control fix it. Lesson: a too-convenient "wall"
that confirms the prior deserves the same scrutiny as a too-convenient win.

## FLIPPED VERDICT
**Rate-coded spiking CAN do whitening** — it HOLDS a whitened code (representation de-risk) AND COMPUTES one (this, with
the analytic lateral inhibition + stable leaky dynamics; the membrane AVERAGES the rate-noise, which a single-shot
opponency subtraction cannot). **The "rate-coded spiking can't whiten" wall is REFUTED for the handed-in analytic
wiring.** The boundary is NARROWER than I had concluded: it is NOT the computation — it is **LOCAL LEARNING of the
lateral inhibition** (the analytic `L = C^½−I` is computed offline from C and handed in — "ZCA-as-wiring").

This realigns with the Track B research's deeper point: the project's 3 failed on-bridge attempts were WRONG-TARGET
SPARSIFICATION rules; the CORRECT rule is the **stable-fixed-point similarity-matching lateral rule** (Pehlevan-
Chklovskii 2015: lateral weights → C⁻¹ at a saddle, "not merely increased inhibition"; SAILnet `c_ij − p²`), which the
failed naive anti-Hebbian missed (it lacked the fixed point → the instability that drove it to silence).

## This RE-OPENS option 1 (it is not a dead-end boundary)
Earlier this session I concluded the worst-pair was a "confirmed point-neuron limit." That conclusion was built on the
WRONG-TARGET sparsification attempts and predated the research + these de-risks. The corrected picture:
1. Spikes HOLD whitened codes ✓ (representation de-risk).
2. Spikes COMPUTE whitening with the analytic lateral inhibition ✓ (this de-risk; membrane averages rate-noise).
3. Dim-whitening is a realizable target that decorrelates concepts ✓ (Q1).
→ The ONLY remaining gap is LEARNING the lateral inhibition locally, and the research named the correct rule.

## NEXT EXPERIMENT (the genuine path forward, not a confirm-the-boundary test)
Test whether the **Pehlevan-Chklovskii stable-fixed-point lateral rule** (or SAILnet `c_ij − p²`) can LEARN the
whitening lateral inhibition `L` on the bridge (vs handing it in). Since the computation WITH `L` already works in
spikes, learning `L` is the sole remaining gap — and there is now strong reason (the computation works; the correct
rule has a proven fixed point at ≈C⁻¹) to expect it can succeed where the naive anti-Hebbian failed. Cheap-first
numpy: does the stable rule converge `L → C^½−I`-equivalent (i.e. whiten) where `c_ij` (no `−p²`) diverged?
Honest caveat to carry: the analytic `L` is still a fixed wiring derived from the data covariance; "learned locally
from the stream" is the real biological bar, and the upstream-graded-stage framing (retina/LGN, biology-faithful)
remains a legitimate alternative if local learning proves seed-fragile at scale.

## Artifact
`research/findings/raw/_A_whitening_computation_derisk.py` (Q1 + Q2 + noiseless control + stable solver). NO sim/ edits.
