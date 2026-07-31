---
type: finding
status: contributing
date: 2026-07-22
mechanism: gap4-credit
---

# gap#4 — the FAITHFUL on-bridge BDSP rule (coincidence-gated + sigmoid-baseline credit) BEATS a reservoir at spiking sparsity; the on-bridge negative is an OPERATING-POINT/LR-scale issue, NOT the rule's math

**2026-07-22, CPU/numpy, coexisting with the fluency training.** The sparse-hidden de-risk ruled out sparsity-per-se as
the gap#4 blocker but used FA's linear credit + a straight-through post-derivative -- NOT the actual on-bridge rule. Reading
the real kernel (`sim/kernels.py fused_bdsp_update`, Payeur-Naud 2021 M1.2) shows the on-bridge rule differs from the
working FA rate rule in TWO faithful ways: `dw = eta * Etilde_pre * E_post * (P_post - Pbar_post)` -- (1) a COINCIDENCE
gate (the pre AND post EVENT rates multiply; at firing 0.05 that product is ~0.0025, a ~400x starvation), and (2) a
SIGMOID+BASELINE credit `(sigmoid(beta*apical) - Pbar)` (bounded, EMA-baselined at p0=0.30) vs FA's unbounded linear
`e@B`. `_gap4_bdsp_faithful_credit_derisk.py` isolates each feature on MNIST with the SAME fixed-random DFA feedback B,
5 arms (reservoir / fa_linear / fa_coinc[+coincidence only] / bdsp_nocoinc[+sigmoid-credit only] / bdsp[the faithful rule]).

## Result (3-seed 42/43/44 at spiking sparsity; + a dense-LR probe) -- the faithful rule WORKS at spiking firing
| firing | RESERVOIR | fa_linear | fa_coinc | bdsp_nocoinc | **bdsp (faithful on-bridge)** |
|--------|-----------|-----------|----------|--------------|-------------------------------|
| dense (lr 0.3) | 0.750-0.771 | 0.932 | 0.83-0.86 | 0.10 (scale artifact) | 0.15 (scale artifact) |
| dense (lr 0.03) | 0.750 | 0.941 | 0.862 | **0.911** | **0.810** |
| 10% | 0.617 | 0.819 | 0.846 | 0.856 | **0.776** |
| **5% (spiking-like)** | **0.514** | 0.775 | 0.832 | 0.825 | **0.779** |

Three decisive findings:
1. **The COINCIDENCE GATE is not the blocker** -- `fa_coinc` (add ONLY the pre*post binary-spike gate) works great at
   every firing rate, and is actually BETTER than fa_linear at low firing (5%: 0.832 vs 0.775 -- a binary post-spike is a
   cleaner target than the sigmoid derivative). So "coincidences are too rare at low firing" is REFUTED.
2. **The FAITHFUL rule beats the reservoir at spiking sparsity, 3-seed** -- `bdsp` at 5% firing = 0.779 vs reservoir
   0.514 (+0.265, all 3 seeds). The full on-bridge rule's math (coincidence gate + sigmoid-baseline credit) is SOUND at
   the actual spiking firing rate.
3. **The only failure -- bdsp -> chance at DENSE firing under lr=0.3 -- was a pure LR-SCALE artifact:** the
   sigmoid-baseline credit `(sigmoid(beta*ap)-Pbar)` has a different magnitude than the linear `ap`, so lr=0.3 over-steps
   at dense activation. At lr=0.03 the SAME arm recovers to 0.810 (bdsp) / 0.911 (bdsp_nocoinc). A tuning mismatch, not a
   degeneracy.

## Implication -- the gap#4 on-bridge negative is narrowed to OPERATING POINT / LR-SCALE, not the rule
Combined with the MNIST 6-seed reframe (credit beats reservoir on a real task) and the sparse-hidden 3-seed result
(sparsity per se is fine), this closes the last "maybe the RULE is wrong" hypothesis: the FAITHFUL on-bridge BDSP rule --
its exact coincidence gate and sigmoid-baseline credit -- BEATS a reservoir on a proper deep task at true spiking
sparsity. So the on-bridge BDSP negative (degenerate at firing 0.04-0.07) is an OPERATING-POINT / LEARNING-RATE-SCALE /
implementation issue, NOT a fundamental rule-math or sparse-code wall. That is a precise, narrow, tractable on-bridge fix:
match the eta to the sigmoid-baseline credit magnitude at the bridge's firing rate + p0, and check the on-bridge operating
point (firing rate, Pbar EMA tracking, eligibility-trace scale) -- rather than searching for a new credit rule.

Honest scope: a rate numpy replica of the rule (faithful to the coincidence gate + sigmoid-baseline credit + EMA baseline),
NOT the live spiking bridge -- it isolates the RULE's soundness (which was the open question), and localizes the residual
to the bridge's op-point. The named next on-bridge test: an eta/op-point sweep of the live BDSP at its firing rate. NO
`sim/` edit. `research/findings/raw/gap4/bdsp_faithful_{seed42,2seed,denselr}.log`.
