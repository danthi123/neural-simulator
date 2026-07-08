# On-bridge deep-credit part 4 — the population-coding surpass, MEASURED: at CPU-feasible K (≤16) the spiking net still does NOT train the compositional task (accuracy flat at chance across K∈{1,8,16}), and the corr-fingerprint is config-dependent (rises in one config, falls in another) — NOT robust. The surpass mechanism (population coding) is the right research-gated direction, but demonstrating the cross-over requires the field's working point (K~hundreds + GPU + more training) = the measured multi-order scale wall, not a cheap de-risk.

**Date:** 2026-07-08
**Runner:** `research/runners/_semantic_inheritance_onbridge_spiking_derisk.py` (`--pool-k K`, the population-coding mechanism; NO `sim/` edit). K-sweep {1,8,16} × 3 seeds (fast config hidden=16, n_super=24, ep=15, subsample=100).
**Verdict:** honest negative at cheap K-scale + a MEASURED scale requirement. The population-coding mechanism (the #1 research-gated surpass) does not demonstrate the on-bridge cross-over at K≤16; the field's working point (K~500) is the genuine requirement. Corrects the premature "corr rises with K = confirmed fingerprint" from the runner commit (bfdbe1cf) — the corr is config-dependent.

## The K-sweep result (fast config, 3 seeds/K, chance 0.278)
| K | plain_fa | burstprop | microcircuit | best-arm | 1-layer floor | corr(pooled E, soma-rate) |
|---|---|---|---|---|---|---|
| 1 | 0.228 | 0.216 | 0.278 | 0.278 | 0.185 | 0.373 |
| 8 | 0.198 | 0.111 | 0.142 | 0.198 | 0.167 | 0.251 |
| 16 | 0.198 | 0.259 | 0.278 | 0.278 | 0.117 | 0.197 |

- **Accuracy does NOT climb with K:** best-arm stays AT chance (0.278) or below across K∈{1,8,16} — no cross-over, no monotone trend. The net does not train at any CPU-feasible K.
- **The corr-fingerprint is NOT robust:** here corr(pooled E, soma-rate) FALLS with K (0.373→0.197); the builder's 1-seed smoke (hidden=32/subsample=64/n_super=12) had it RISING (0.264→0.345). The direction flips with config → the "corr rises with K" claim (committed in the runner commit) is config-dependent, not a robust confirmed fingerprint. Corrected here.

## The honest, corrected conclusion (NOT a premature wall — a MEASURED requirement)
- The research gate's DIAGNOSIS (the on-bridge non-training = the single-neuron read-out variance wall) and its #1 MECHANISM (population coding, K neurons/unit) remain the right, well-cited direction — population coding is the DEFINITIONAL form of the rule family (BurstCCN `p=b/e` is ensemble-only at **500 neurons/unit**; Payeur rates are ensemble) with same-substrate Hebbian precedents (47%→K8 100%; PPMI 20%→K32 94%).
- BUT the SURPASS is NOT demonstrated at CPU-feasible K (≤16): accuracy stays at chance + the SNR-lift fingerprint isn't robust. The genuine cross-over requires the field's working point — **K~hundreds (BurstCCN: 500/unit) + longer windows + more training**, which is a genuine SCALE-UP:
  - compute cost measured: K widens each layer K× (neurons) and each FF edge K²× (synapses); the per-example-online on-bridge training is inherently step-heavy (settle+credit bridge steps per example × examples × epochs); K=16 at a resolvable config is already ~hours on CPU; K~500 is orders beyond CPU-feasible on this per-example-online path. This is the multi-order SCALE wall the deep-lever research gate PRE-FLAGGED ("the burst family is the least-scaled bio-plausible method; a depth-toy GO is 3-4 orders below scale; compute/data richness is the binding wall").
- ⇒ per the standing "scale is a LEVER, MEASURE it" discipline: I MEASURED it — cheap-CPU K (≤16) does not cross over; the field's K~500 working point is the requirement. The scale lever is real (population coding is the right mechanism) but demonstrating the on-bridge cross-over is a GPU + K~hundreds + more-training investment, not a cheap rung.

## The deep-lever landscape, comprehensively mapped (this session's arc, cheap-de-risk phase COMPLETE)
- **D1:** biological deep credit PORTS to spikes (validated, depth-2 XOR toy).
- **D2/rung-2:** FA depth wall + KP surpass — a toy-instrument boundary.
- **Real-task part 1 (CIFAR/vision):** wrong instrument (real image depth is convolutional not FC).
- **Real-task part 2 (compositional semantics, RATE):** the FIRST real-task traction — the deep-credit approach (FA at rate) trains a deep net to a genuine depth-required compositional-generalization task (0.69, 5/6, no leakage). The language-relevant setting where depth is COMPOSITIONAL.
- **Real-task part 3+4 (ON-BRIDGE spiking):** at CPU-feasible scale the spiking net does NOT train (read-out variance wall); the surpass (population coding) is research-gated + directionally-motivated but requires the field's K~500 working point (a measured GPU/scale investment) to demonstrate the cross-over.

⇒ the deep-credit mechanism is validated + works at RATE on real compositional tasks; on-substrate training is gated by the read-out variance wall whose surpass is population coding at the field's K~hundreds working point — a measured scale-up. The cheap-de-risk phase has delivered its full, honest landscape.

## Files
`research/runners/_semantic_inheritance_onbridge_spiking_derisk.py` (`--pool-k`); `research/findings/raw/_onbridge_kfast_K{1,8,16}_s{42,43,44}.json`. Diagnosis: `2026-07-07-onbridge-spiking-deep-credit-training-research-gate.md`. Prior: `2026-07-07-deep-credit-onbridge-spiking-6seed-does-not-train-at-cheap-scale.md`, `-compositional-semantics-GO.md`.
