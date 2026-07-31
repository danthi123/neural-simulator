---
type: finding
status: qualified
date: 2026-06-06
mechanism: whitening
---

# Option 1 RESOLVED (algorithm level): a regularized LOCAL rule learns a composing whitening — 6/6 seeds, 100% — 2026-06-06

The worst-pair decorrelation "boundary" I confidently declared a *confirmed point-neuron limit* earlier this same
session is **resolved** at the rate/algorithm level. A regularized LOCAL learning rule (synaptic weight-decay) learns a
whitening that composes END-TO-END at **100%, 6/6 seeds**, judged on COMPOSITION (the agent benchmark), with a BOUNDED
matrix (the guard that caught the false positives), and a coherent mechanism.

## The result — composition gate, CIFAR real-object grounding, K=300 subspace, 6 seeds (42/43/44/45/46/100)
`research/findings/raw/_A_whitening_compose_gate.py` (+ per-seed `_s{N}lam.json`).

| condition | composition (per-seed) | reading |
|---|---|---|
| RAW (no whitening) | 66.7–69.2% | floor control ✓ |
| CONCEPT-whiten (N×N gram; NOT substrate-realizable) | **100%** all 6 | the proven target control ✓ |
| DIM-analytic (full dim-whiten C^−1/2; realizable) | 66.7–69.2% all 6 | **OVER-WHITENS → does not compose** |
| **LEARNED (regularized local rule, −λM, λ=0.01)** | **100% (39/39) all 6, M-ratio 0.09 (bounded)** | **composes = target** |

## Mechanism (coherent — resolves the dim-analytic 66.7% vs learned 100% paradox)
**Over-whitening hurts; regularized whitening composes; the local rule finds the right amount.** The full dim-whitening
(C^−1/2) over-amplifies the low-variance (noise) directions → only 66.7%. The −λM weight-decay (with λ=η) moves the
fixed point to M = C^1/3 − I, i.e. a GENTLER partial whitening (C^−1/3) that decorrelates WITHOUT the over-amplification
→ 100%. (The M-ratio 0.09 vs the analytic C^1/2−I is small because the eigenvalues are mostly near 1; the small
difference at the extreme eigenvalues is exactly what separates over-whitening from composing.) The rule is local:
ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λ M_ij — co-firing (Hebbian/anti-Hebbian) + a target + synaptic weight-decay, all
biologically standard.

## The full arc this resolves (epistemic honesty — the path went THROUGH my errors)
Earlier this session I declared the worst-pair a "confirmed point-neuron boundary." The deep research + a chain of
de-risks walked that back, each step gated by a control that caught a convenient-but-wrong number:
1. **Citable limit, not a dead end** (Mikulasch-Priesemann: a point neuron + one global inhibitory pool can't whiten).
2. **Not a representation limit** — spikes HOLD a whitened code (the ON/OFF rate-code de-risk; the opponency wall
   doesn't apply because whitened codes are variance-equalized).
3. **Not a computation limit** — spikes COMPUTE whitening with the analytic lateral inhibition + stable leaky dynamics
   (the membrane averages the rate-noise). [A 1st near-FP here: an unstable-solver bug that "confirmed the wall" —
   caught by the noiseless-solver control.]
4. **Not a handed-in-only limit** — a LOCAL rule learns it. [A 2nd near-FP: an M that blew up on rank-deficient data
   gave low coherence that looked like a win — caught by the M-ratio control. And the COHERENCE PROXY misled THREE
   times: dim-whitening drops coherence but does NOT compose — caught only by gating on COMPOSITION.]
5. **Resolved (this):** a REGULARIZED local rule composes at 100%, 6/6, judged on composition, with a bounded matrix.

Five convenient-but-wrong results across the thread, all caught by controls (noiseless check, M-ratio guard,
composition-not-coherence). That is why the 6/6 is credible rather than a sixth convenient number.

## Honest SCOPE + the remaining follow-on (do NOT overclaim)
- This is validated at the **algorithm / rate level** on the **numpy VSA reference pipeline** (CIFAR real V1 grounding →
  the learned regularized whitening → `NestedCompositionAgent`) — the SAME pipeline the 2026-06-04 + Track A grounding
  results were measured on. The SCIENTIFIC question — *can a local rule learn a composing whitening from raw correlated
  grounded codes?* — is answered **YES**.
- The **on-bridge SPIKING realization of the LEARNING** is the engineering follow-on. It is now well-supported: prior
  de-risks showed spikes HOLD (representation) AND COMPUTE (with analytic L + leaky dynamics) whitening; this shows the
  weights are LOCALLY LEARNABLE with a stable, bounded, weight-decayed rule. Realizing ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λM_ij
  as plastic IT↔FS lateral inhibition + homeostatic decay on the bridge is the next build (the bridge has FS lateral +
  homeostasis; the −λM is synaptic weight-decay).
- The **production RF-on-bridge composer** (vs the numpy reference agent) is the other axis; the grounding+whitening
  feeds `grounded_codes` either way.
- **Caveat retained:** the learned weights are learned from the data covariance over the codebook, not yet from a
  truly streaming input; and the upstream graded-whitening stage (retina/LGN) remains a valid biology-faithful
  alternative. But "no local rule can do it" — my earlier claim — is **falsified**.

## Net for option 1
From "confirmed point-neuron boundary" → **a biologically-plausible local rule (Hebbian/anti-Hebbian + threshold
homeostasis + synaptic weight-decay) learns a whitening that composes end-to-end at 100%, 6/6 seeds.** The decorrelation
blocker is resolved at the algorithm level; the on-bridge spiking realization is a supported engineering follow-on, not
an open scientific question. NO sim/ edits in this whole arc.

## Artifacts
`research/findings/raw/_A_whitening_compose_gate.py` + `_s{42,43,44,45,46,100}lam.json` (6-seed),
`_A_whitening_compose_gate_lam01.json` (seed 42), the de-risk chain
(`_A_whitening_{ratecode,computation,learn_lateral}_derisk.py`), and the deep-research doc
(`2026-06-06-decorrelation-blocker-deep-research.md`).
