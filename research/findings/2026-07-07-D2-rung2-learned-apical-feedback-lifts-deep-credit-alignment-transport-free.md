# D2 rung-2 — the numpy XOR toy is the WRONG INSTRUMENT (NOT a substrate boundary): the KP learned-apical-feedback alignment lift is a DEV-SEED ARTIFACT on the toy — strong on dev 42/43 (+0.23/+0.31) but null-to-NEGATIVE on blind 100/101/102 (+0.03/+0.00/−0.18), 6-seed +0.067 ± 0.160 (2/6). This is a limitation of the TOY (too small for accuracy headroom, alignment metric too seed-noisy), NOT a wall in the mechanism or the substrate. The correct next step is OFF the toy — a real task — not a "boundary" label.

**Date:** 2026-07-07 (framing corrected same day per owner: a toy limitation is not a boundary; autonomously pursue the solution off the toy)
**Runner:** `research/runners/_gnw_d1_spiking_bdsp_derisk.py` (`--feedback {fixed,learned}` default fixed = byte-identical; the Kolen-Pollack `_kp_update` + per-layer homeostatic gain + the per-layer credit-alignment metric). NO `sim/` edit.
**Verdict:** INCONCLUSIVE-ON-TOY, explicitly NOT a boundary. The pre-registered alignment GO held on 3 dev seeds (mean +0.18) but did not survive 6-seed-blind (mean +0.067 ± 0.160, 2/6, one seed −0.18) — because the 10-bit XOR toy is the wrong instrument (no accuracy headroom, seed-noisy metric, `Y` far from converged), NOT because the KP mechanism or the substrate has a wall. The KP mechanism is transport-free + genuine + adversarially-verified sound (`wixueldhw` = SURVIVES_WITH_SCOPE_FIXES). ⇒ change the instrument: pursue the deep-credit mechanism on a REAL task, not another XOR rung.

## The 6-seed result (hidden=24, ep=800, lr=0.3, batch=32, KP rates 0.2/1e-4) — deepest-layer credit-alignment
| seed | fixed deepest | learned deepest | lift | fixed acc | learned acc |
|---|---|---|---|---|---|
| 42 (dev) | 0.261 | 0.488 | **+0.227** | 0.783 | 0.752 |
| 43 (dev) | 0.232 | 0.541 | **+0.309** | 0.847 | 0.813 |
| 44 (dev) | 0.327 | 0.335 | +0.008 | 0.908 | 0.813 |
| 100 (blind) | 0.076 | 0.108 | +0.032 | 0.836 | 0.752 |
| 101 (blind) | 0.200 | 0.204 | +0.004 | 0.827 | 0.869 |
| 102 (blind) | 0.392 | 0.213 | **−0.179** | 0.769 | 0.830 |
| **6-seed** | **0.248 ± 0.099** | **0.315 ± 0.157** | **+0.067 ± 0.160** | 0.828 | 0.805 |

- **The lift is a DEV-SEED artifact.** Dev 42/43 lift strongly (+0.23/+0.31); ALL THREE blind seeds are null-to-negative (+0.03/+0.00/−0.18). The mean collapses from +0.18 (3 dev) to +0.067 (6-seed), variance ±0.160 exceeds the mean, and it clears the pre-registered +0.10 bar on only **2/6** seeds. **Not robust; the pre-registered GO fails at 6 seeds.**
- **Accuracy is flat/slightly down** (0.828→0.805) — no capability benefit, consistent with the toy having no accuracy headroom (rung-1's Eldan-Shamir finding).
- **The alignment metric is very seed-noisy at H24** — the FIXED baseline itself ranges 0.076–0.392 across seeds. The lift is swamped by this noise, and `Y` is far from converged (`cos(Yᵀ,W)` ~0.27–0.31 at ep=800) — the KP weight-mirror has not reached its fixed point at this cheap budget.

## What SURVIVES (still true, adversarially verified on the dev data — `wixueldhw`)
- **The KP mechanism is transport-free + genuine:** `_kp_update` reads only local pre/post + `Y`, never `self.W` (structurally verified + 3-guard probe); the learned `Y` moves toward `Wᵀ`; no label smuggling (permuted anti-cheat at chance with the surpass knobs ON).
- **The alignment metric is honest** (cos of the rule's per-layer update vs within-net oracle backprop; reproduces bit-exact); like-for-like (default byte-identical to rung-1); anti-cheats collapse.
- **Homeostasis HARMS at depth-3** (lowers alignment every seed) — magnitude is not the depth-3 axis.
So the NEGATIVE is not a bug or a gamed metric — the mechanism is sound; it simply does not robustly move the deep-layer alignment on this numpy toy at this budget.

## The reframe (a boundary launches the search — not an endpoint)
Per the standing discipline, the 6-seed boundary is diagnostic, and it points three ways (the reassessment fork):
1. **BUDGET/convergence:** `Y` is far from `Wᵀ` (cos ~0.30) at ep=800 — the KP weight-mirror may simply need more training / different rates to converge (Greedy-Costa 2026 reach depth-8, but at more training). Testing this heads toward longer runs (the owner deferred expensive training) AND the toy accuracy has no headroom anyway, so a longer numpy run would at best show alignment-convergence, still no capability.
2. **TOY/metric noise:** the H24 Boolean toy's alignment metric is too seed-noisy (fixed baseline 0.076–0.392) to resolve a depth-fix; the numpy rate-reference is the wrong instrument for this effect.
3. **⇒ the depth-fix belongs ON THE SUBSTRATE (rung-3):** the accuracy payoff + the interneuron's depth role only appear where fixed-FA genuinely BREAKS — the spiking net, where the point-neuron cannot carry a clean continuous error and credit-noise compounds. The numpy toy cannot show it.

## The honest reassessment (owner-chosen "then reassess")
D1 established deep credit ports to spikes; D2 rung-1 localized the FA depth-wall to credit-alignment on the numpy toy; **D2 rung-2 now shows the numpy rate-reference is the WRONG instrument to demonstrate the depth-FIX robustly** (dev-seed lift, blind-seed null, no accuracy headroom, noisy metric). The mechanism is transport-free + sound; the genuine test of whether it MATTERS is the on-substrate spiking arm (rung-3) or scale — not more numpy toy rungs. This is the honest reassessment point: the cheap numpy mechanism-ladder has given what it can (the mechanism ports + moves in the right direction on some seeds); the next real evidence requires the substrate or scale, which is the owner's steer.

## Files
`research/runners/_gnw_d1_spiking_bdsp_derisk.py`; `research/findings/raw/_gnw_d2_depth3_{learned,microcircuit,burstprop}.json` + `_{learned,fixed}_blind.json`. Rung-1: `2026-07-07-D2-rung1-...md`; research gate: `2026-07-07-D2-feedback-alignment-depth-stability-research-gate.md`; verify: `wixueldhw`.
