---
type: finding
status: superseded
superseded_by:
  - research/findings/2026-07-18-gap4-onbridge-BDSP-coupling-REVIVED-depth-fragile-boundary-confirmed-BTSP-pivot.md
  - research/findings/2026-07-18-gap4-BTSP-plateau-gated-oneshot-credit-GO-the-keystone-is-the-enabler.md
date: 2026-07-18
mechanism: bdsp
---

# Gap #4 (local-credit keystone) research gate — the deep-credit block is FIXABLE, not a wall: BDSP with the apical error carried by the just-built BISTABLE HELD apical plateau

**2026-07-18.** Gap #4 (a local spiking credit rule, no weight transport — the engine for #2 & #5) had a wall of banked
negatives (feedforward e-prop NOT-GO, recurrent e-prop refuted, Node Perturbation retired, BDSP-on-classifier blocked,
graded-readout-doesn't-unlock). The deep-research gate (read-only, papers + the project's own kernels/findings read in
depth) reframes it: **the negatives are one family — supervised global-loss deep credit through a spiking classifier
readout — and the BDSP failure was already ROOT-CAUSED to a specific, fixable mechanism, not a fundamental wall.**

## The prior BDSP failure (root-caused, `2026-07-10-D1-onbridge-BDSP-apical-decoupled-from-soma-BOUNDARY-root-caused.md`)
The committed rule (`sim/kernels.py:462 fused_bdsp_update`): `dw_ij = η·Ẽ_j·(B_i − P̄_i·E_i) ≡ η·Ẽ_j·E_i·(P_i − P̄_i)`
— `Ẽ_j` presynaptic eligibility trace, `E_i` post event rate, `B_i` post BURST rate, `P̄_i` slow EMA baseline, apical
sets `P_i = σ(β·v_apical)`. The failure: driving `cp_bdsp_apical_drive` raised the burst-PROBABILITY read `P` (0.30→1.00)
but **NOT the measured burst rate `B` (0.000→0.000)**, because the `enable_bdsp` block wrote `v_apical` only to compute
`P` and NEVER applied the apical→soma electrotonic coupling. The rule reads `B`, not `P` → the apical delivered ZERO
directed credit (separation 1.33×) and the P₀ moat leaked. A 2026-07-10 `sim/` edit coupling apical→soma lifted `B`
0.117→0.49 and separation to 3.75× (→ 20× at a sparse regime) — **but the end-to-end learning-to-accuracy run was NEVER
completed, then the whole arc was confounded by the unseeded-substrate bug.**

## Why the JUST-BUILT bistable apical fixes it (the new capability the prior arc never had)
The prior apical was a ~50-100 ms TRANSIENT (`fused_coincidence_plateau`, self_regen=0) → even coupled, the credit was
a blip barely overlapping the slow presynaptic trace. **The new bistable apical (self_regen≥0.8 + KIR, single-cell
latch-and-hold + I-V validated, gap #5) LATCHES a stable UP state that HOLDS the credit signal across the whole
eligibility window, and the KIR DOWN state gives a silent rest (`P_i=P̄_i` → dev=0 → the P₀ no-spurious-learning moat
holds BY CONSTRUCTION).** With the asymmetric coupling ALSO just built (`apical_g_couple_to_soma ≫ apical_g_couple`), a
held UP-state is read STRONGLY into the soma → SUSTAINED real bursts `B` → a persistent, sign-correct directed-credit
signal `(B − P̄·E)` integrating cleanly against `Ẽ_j`. This is exactly the apical→burst coupling whose absence sank the
prior BDSP, now made SUSTAINED. (BDSP-with-a-held-plateau IS BTSP physics — Bittner-Magee 2017 — an apical Ca²⁺ plateau
gating a seconds-long eligibility trace; the bistable UP-state is the mechanistic realization of the seconds-long plateau.)

## Ranked recommendation
- **#1: BDSP with the apical error carried by the BISTABLE HELD plateau + the asymmetric strong apical→soma read.**
  Attacks the exact root-caused failure; minimal code on committed kernels (`fused_bdsp_update` + `fused_coincidence_plateau(self_regen>0)`
  + `apical_g_couple_to_soma`, all additive/default-off); a genuine DEEP-credit rule (the depth_helps gate applies).
- **Fallback: BTSP (Bittner-Magee)** — if the global-loss depth gate stalls (the rule-variance wall the record flagged
  for FA/NP), pivot to BTSP: local, one-shot, plateau-gated, eligibility-trace, NO global-loss backward pass, native to
  the CA3/CA1 machinery just closed (gap #5); the bistable UP-state is THE enabling mechanism. Best mission-fit (aligns
  with the unsupervised on-spike stream-cortex direction).
- **Do NOT lead with:** plain feedback-alignment / target-prop (memorizes at depth) or a fresh Node-Perturbation build (retired).

## The cheap-first de-risk (CPU, minutes, decisive) — the NEXT concrete step
`BurstpropMLP` (rate sibling of `DendriticMLP`, spec `2026-07-01-burst-multiplexed-dendritic-credit-assignment-spec.md`,
~30-50 lines, no sim/ edit for the rate version) on the EMERGE-1 depth-2 target (threshold-of-5-pair-XORs over 10 bits,
65/35 held-out; deep oracle 1.00, 1-hidden oracle ≈chance → depth-REQUIRED). **THREE arms, identical net/seed/init/task,
only the apical-signal dynamics differ:** (1) `frozen_hidden` (hidden fixed, only readout trains = the RESERVOIR control);
(2) `transient_apical` (self_regen=0, the old regime); (3) `held_apical` (self_regen≥0.8, the new bistable regime).
**GO (depth_helps gate): heldout(held) > heldout(frozen)+0.05 AND heldout(held) > heldout(transient)+0.05** (first =
anti-reservoir; second attributes the win to the BISTABILITY). Anti-cheats (all must hold): wrong-sign anti-learns;
no-teaching null (apical silent → weights ~unchanged = the moat); apical-lesion → frozen floor; oracle ceiling ≥0.80;
no-weight-transport assert (`B` never a function of `W`). **MANDATORY instrument checks (the two silent-failure bugs):
set `cfg.seed` [NOT actual_seed_used] + two-process threshold-hash verify the substrate is seeded; reset dendritic state
(`cp_v_apical`, `cp_conductance_g_coincidence`) between conditions.** 6 seeds (42/43/44 dev + 100/101/102 blind).
DECISIVE: held>transient>frozen → the bistability is load-bearing for credit → the never-completed on-bridge BDSP-at-
sparse-regime run is warranted; held≈transient≈frozen → an honest negative that further characterizes the wall.

Sources: Payeur 2021 (10.1038/s41593-021-00857-x), Urbanczik-Senn 2014 (PMID 24507189), Sacramento 2018 (arXiv
1810.11393), Bittner 2017 (10.1126/science.aan3846) + Milstein 2021 (eLife 73046) + BTSP-credit biorxiv 2025.06.12.659336.
Project: `sim/kernels.py:253,462`, `sim/dendritic_{neuron,plasticity,mlp}.py`, the 2026-07-10 root-cause + surpass findings.
