# D2 rung-1 (depth-3 rate reference) — raw Burstprop HITS the depth-3 wall (0.669, mid-layer credit-alignment collapses to 0.05); clean-error FA clears depth-3 ACCURACY (0.846) but its per-layer credit-alignment DEGRADES with depth (deepest 0.27) — so the D2 depth-wall metric is PER-LAYER ALIGNMENT (the toy is too easy for an accuracy wall at practical width), and the surpass target is: lift the deep-layer alignment

**Date:** 2026-07-07
**Runner:** `research/runners/_gnw_d1_spiking_bdsp_derisk.py` (`--depth {2,3}`, default 2 = byte-identical; `make_task_d3` + a `FANet` plain-FA arm + a per-layer credit-alignment metric). Rung-1 of the D2 spec (`docs/plans/2026-07-07-D2-depth-stability-spiking-build-spec.md`). Diagnostic (not a surpass GO). NO `sim/` edit (runner-only).
**Verdict:** the FA depth-wall SIGNATURE is present at depth-3 in the CREDIT-ALIGNMENT axis (deepest-layer alignment 0.27, degrading with depth), and raw Burstprop hits an ACCURACY wall (0.669); clean-error FA's accuracy survives depth-3 on this toy → the D2 metric is per-layer alignment + depth-4/harder tasks for the accuracy level. The surpass (rung-2) is well-motivated: learned apical feedback + per-layer homeostasis should LIFT the deep-layer alignment (Greedy-Costa 2026).

## The depth-3 result (3-seed 42/43/44, hidden=24, ep=800, lr=0.3, batch=32)
| Arm | held-out | deepest-layer credit-alignment |
|---|---|---|
| oracle (depth-3 backprop) | 0.970 | 1.0 (weight transport) |
| clean-error FA / microcircuit | 0.846 | 0.27 |
| plain-FA (`FANet`, no interneuron/burst) | 0.846 | 0.27 (== microcircuit → confirms D1: the interneuron loop is numerically inert at the rate level) |
| **raw Burstprop** | **0.669** (barely > chance) | 0.14 (mid-layer 0.05) |
| single-layer floor | 0.198 | — |

Per-layer alignment `[deepest … output]`: **clean-error FA `[0.27, 0.63, 0.82, 1.0]`** (monotone decay with distance from the output — the classic FA signature); **Burstprop `[0.14, 0.26, 0.05, 1.0]`** (collapses, the mid-layer credit is scrambled). Anti-cheats all collapse (lesion 0.50, wrong-sign 0.49, null 0.50 with hidden-drift 0.0, permuted 0.50); no weight transport; oracle 0.970 ≥ 0.80 (valid ceiling at lr=0.3/ep=800 — avoided the D1 lr=0.5-destabilizes-oracle trap).

## Two honest diagnostic findings (the load-bearing rung-1 outcome)
1. **Raw Burstprop is depth-fragile — confirmed.** 0.669 at depth-3 (vs 0.92 at depth-2 best batch); its middle-layer credit-alignment collapses to 0.05. This is the compounding burst-noise the research gate + the 2025-neuromorphic "depth HURTS" result predicted. Burstprop needs the depth-stability fix.
2. **Clean-error FA clears depth-3 ACCURACY but its credit-ALIGNMENT degrades with depth** (deepest 0.27). The FA depth wall is real in the credit-DIRECTION axis at depth-3, but the accuracy consequence is deeper than depth-3 on THIS toy. **Why the toy hides the accuracy wall:** a 10-bit Boolean function does not representationally separate depth-2 from depth-3 at practical MLP width — the depth-separation theorems (Eldan-Shamir 2016) require EXPONENTIAL width; at the FA-readable width (H24) even the depth-2 oracle fits (0.972), and only at a narrow width (H6) does the depth-2 oracle underfit (0.72 vs depth-3 0.87, margin +0.15, the depth-genuineness control). ⇒ **the D2 depth-wall must be read from per-layer credit-alignment degradation, not accuracy**, at the FA-arm width; depth-4/harder tasks are the accuracy-level testbed.

## What this sets up for the surpass (rung-2)
The D2 GO metric is now precise: **does the surpass LIFT the deep-layer credit-alignment** (clean-error FA baseline: deepest 0.27; oracle: 1.0), AND does it restore Burstprop's accuracy (0.669 → toward oracle)? Per the research gate (`2026-07-07-D2-feedback-alignment-depth-stability-research-gate.md`), the #1 surpass = **learn the apical feedback (Y-plasticity / Kolen-Pollack — fixes credit-DIRECTION decay, i.e. lifts the alignment) + per-layer homeostatic gain (fixes MAGNITUDE drift)**; Greedy-Costa 2026 holds credit through 8 layers with exactly this, transport-free, on the two-compartment substrate. rung-2 adds these to the depth-3 net (numpy first), gated on: alignment lifts AND the new no-weight-transport probe on the LEARNED feedback holds (Y update must never read Wᵀ). Then rung-3 (on-bridge) tests whether the interneuron cancellation becomes causally load-bearing at depth (the Greedy-Costa SST-rank-at-depth role, which the D1 finding predicted is NOT load-bearing at depth-2).

## Files
`research/runners/_gnw_d1_spiking_bdsp_derisk.py` (`--depth`, `make_task_d3`, `FANet`, per-layer alignment); `research/findings/raw/_gnw_d2_depth3_{microcircuit,burstprop}.json`. Spec: `docs/plans/2026-07-07-D2-depth-stability-spiking-build-spec.md`; research: `2026-07-07-D2-feedback-alignment-depth-stability-research-gate.md`.
