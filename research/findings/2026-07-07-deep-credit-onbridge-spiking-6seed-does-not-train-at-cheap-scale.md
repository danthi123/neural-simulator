# Deep-credit real-task de-risk, part 3 (ON-BRIDGE SPIKING) — 6-seed: the depth-2 spiking net does NOT train the compositional task at CPU-smoke scale (all arms BELOW chance, all 6 seeds; the oracle trains it 1.0), and the biological mechanism does NOT beat plain FA (microcircuit 0/6, actually worse). The builder's 1-seed "microcircuit > FA" signal was a fluke — refuted by 6-seed-blind. The point-neuron noise + scale wall on the substrate, reproduced concretely.

**Date:** 2026-07-07
**Runner:** `research/runners/_semantic_inheritance_onbridge_spiking_derisk.py` (a depth-2 two-compartment SPIKING net on ONE `SimulationBridge`, reusing the committed `enable_bdsp`/`enable_bdsp_microcircuit` mechanism by import + the compositional-semantic task/metric/controls; genuinely spiking forward via `_run_one_simulation_step` → `cp_bdsp_E`; three credit arms differ ONLY in apical-credit injection = clean like-for-like). NO `sim/` edit.
**Verdict:** honest negative at cheap scale — the thesis (does the biological burst/microcircuit mechanism beat plain FA on spikes, where at rate it's inert?) is NOT supported by the 6-seed run; the spiking net does not train the task at CPU-smoke scale.

## The 6-seed result (dev 42/43/44 + blind 100/101/102; H=40, ep=30, settle=25; held-out inheritance, chance 0.333)
| arm | per-seed | mean |
|---|---|---|
| plain-FA | 0.185/0.111/0.259/0.148/0.111/0.185 | 0.167 |
| Burstprop | 0.148/0.148/0.259/0.148/0.222/0.222 | 0.191 |
| **microcircuit** | 0.074/0.074/0.185/0.148/0.111/0.111 | **0.117** |
| 1-layer floor | — | 0.167 |
| oracle (backprop) | — | 0.895 (1.000 on 5/6) |

- **The spiking net does NOT train:** every deep-credit arm is BELOW chance (0.333) on ALL 6 seeds (`trains_at_all=False` every seed). Meanwhile the fenced-backprop ORACLE reaches 1.000 (the TASK is learnable) — so it is the SPIKING DEEP-CREDIT RULE that fails to train at CPU-smoke scale, not the task.
- **The biological mechanism does NOT beat plain FA:** microcircuit > FA on **0/6** seeds (delta −0.049 — microcircuit is *worse*); burstprop > FA on 3/6 (delta +0.025, within noise). The builder's 1-seed smoke (microcircuit 0.370 > FA 0.222, a tuned/lucky config) does NOT replicate — refuted by 6-seed-blind (the same lesson as XOR rung-2: a 1-seed ordering can be pure noise).
- **What still holds (the honest controls):** depth-genuineness passes (Stage-0 2-layer oracle 1.0), memorization control 0.000 (no leakage), permuted ≈ chance, no weight transport, feedforward weights genuinely move (`ff_weight_moved` True) — so the run is valid; the deep-credit rule simply cannot assign credit through 2 spiking layers well enough to clear chance at this noise/scale.

## What this establishes (honest, and NOT a premature wall)
The biological deep-credit rule, which works at the numpy RATE reference on this exact compositional task (0.69, part 2), does NOT train it on the SPIKING substrate at CPU-smoke scale (below chance, 6/6) — the point-neuron noise limit (D1: raw Burstprop is noise-limited/depth-fragile) + the multi-order SCALE wall the research gate pre-flagged ("the burst family is the least-scaled bio-plausible method; data/experience richness + scale is the binding wall"), reproduced concretely on a real task. And the distinctive biological value (microcircuit > FA) is not demonstrated: inert at rate (part 2, byte-identical to FA), noise at cheap-spiking (this).

This is a SCALE/instrument limit, per the owner's standing correction (a cheap-scale limitation is not a mechanism wall). The open question it defines: is the cheap-scale non-training FIXABLE (population coding for the noisy single-neuron read-out — the documented rate-code-wall lift; wider H; more epochs/settle; GPU scale), or a deeper limit of deep credit on point neurons? The cheapest surpass-check is population coding (single-neuron `cp_bdsp_E` read → a pooled population read), still CPU — the next disciplined step before accepting the scale wall.

## The deep-lever landscape, now mapped (this session's arc)
- **D1:** biological deep credit PORTS to spikes (mechanism validated, depth-2 XOR toy).
- **D2/rung-2:** the FA depth wall + the KP learned-feedback surpass — a toy-instrument boundary (the XOR toy is the wrong instrument).
- **Real-task part 1 (CIFAR/vision):** FC-vision is the wrong instrument (depth-required ⟂ rule-learnable; real image depth is convolutional).
- **Real-task part 2 (compositional semantics, rate):** the first real-task TRACTION — the deep-credit APPROACH (FA at rate) trains a deep net to a genuine depth-required compositional-generalization task (0.69, 5/6, no leakage).
- **Real-task part 3 (on-bridge spiking, this):** at cheap scale the spiking net does NOT train it + the biological mechanism does NOT beat FA — the scale/noise wall on the substrate.

⇒ the mechanism is validated + works at rate on real compositional tasks; its distinctive biological value + real-task success ON THE SPIKING SUBSTRATE is gated by the scale/noise wall — the honest frontier is the population-coding surpass-check, then (if needed) the expensive scale-up, or reconnecting the validated substrate to the conversational goal.

## Files
`research/runners/_semantic_inheritance_onbridge_spiking_derisk.py`; `research/findings/raw/_semantic_onbridge_seed{42,43,44,100,101,102}.json` + `_semantic_inheritance_onbridge_smoke.json`. Part 2: `2026-07-07-deep-credit-real-task-compositional-semantics-GO.md`.
