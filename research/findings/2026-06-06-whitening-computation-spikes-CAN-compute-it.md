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

## UPDATE — local-learning de-risk: a 2ND near-false-positive caught; the worst-pair-LEARNING gap STANDS
`_A_whitening_learn_lateral_derisk.py`. Tested whether a LOCAL stable rule LEARNS the whitening lateral inhibition from
raw correlated input. (A) analytic M=C^½−I control (coh 0.036); (B) naive ΔM∝⟨yyᵀ⟩ (= attempt #2, no target);
(C) whitening-target ΔM∝⟨yyᵀ⟩−I.
- **(C) reported coh 0.032 — looked like a WIN — but ‖M_learned−M_analytic‖/‖M_analytic‖ = 72–9047** (the learned M is
  up to 9000× the analytic). The M BLEW UP: the toy data is RANK-DEFICIENT (32 concepts in 128 dims), so driving the
  output covariance → I tries to amplify the empty null-space to unit variance → M diverges in magnitude → the output
  collapses toward noise → noise is decorrelated → low coherence that is NOT whitening. **A 2nd near-false-positive,
  caught by the M-ratio control** (the 1st was the unstable-solver bug). **COHERENCE ALONE IS INSUFFICIENT** — a
  noise-collapsed output passes it.
- (B) naive: coh 0.118/0.345 — partial only, does NOT reach the worst-pair (max 0.345 vs target 0.036).

**Honest result:** this de-risk does NOT demonstrate local learning of worst-pair whitening — the naive rules either
blow up (whitening-target, rank-deficiency) or reach only partial (naive). **The worst-pair-via-LOCAL-LEARNING gap
STANDS.** Handed-in analytic L works in spikes (the computation de-risk above); learning it locally does not, yet.

**Rigorous follow-on (clean context):** (1) the rule needs REGULARIZATION / rank-handling — whiten only the data
subspace, OR the proper Pehlevan `y=M⁻¹Wx` formulation where the feedforward `W` learns the subspace, OR a decay `−λM`;
(2) **gate on COMPOSITION (the agent benchmark), NOT coherence** — a noise-collapsed output has low coherence but won't
compose; the coherence proxy is what nearly shipped the false positive. The 320-concept full-rank production case may
behave differently from this 32-in-128 rank-deficient toy. After TWO near-false-positives here, the local-learning
question wants a fresh careful build with composition as the gate.

**Honest meta-note:** two convenient-but-wrong results in one thread (an unstable solver that confirmed the pessimistic
prior; a noise-collapse that confirmed the optimistic hope) — the controls (noiseless solver check; analytic-M match
check) caught both. The net scientific position is UNCHANGED by the learning de-risk: spikes HOLD and COMPUTE whitening
with handed-in/analytic L; local LEARNING of the worst-pair solution remains open; the upstream-graded-stage (retina/
LGN, biology-faithful) remains the robust alternative.

## Artifact
`research/findings/raw/_A_whitening_computation_derisk.py` (Q1 + Q2 + noiseless control + stable solver) +
`research/findings/raw/_A_whitening_learn_lateral_derisk.py` (learning de-risk + analytic/naive controls + M-ratio
guard). NO sim/ edits.
