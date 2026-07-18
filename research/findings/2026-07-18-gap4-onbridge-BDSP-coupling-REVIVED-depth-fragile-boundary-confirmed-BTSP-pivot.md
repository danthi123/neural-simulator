# Gap #4 (local-credit keystone) — the on-bridge BDSP-FA path REVIVED (coupling fixed, moat clean), but the depth-fragile FA-credit boundary is CONFIRMED on-bridge (inherits the rate ceiling ~0.715 < 0.75). The prescribed pivot: BTSP, where the gap#5 bistable plateau IS the enabler.

**2026-07-18.** Built the gap#4 keystone `sim/` edit (BISTABLE BDSP apical, reusing gap#5's dendritic bistability) and
ran it to ground on the on-bridge `_d1_onbridge_learn_to_accuracy` harness. Honest, informative characterization —
one real advance, one honest negative, one confirmed boundary, one prescribed pivot.

## What was built (banked, byte-identical-when-off, CI-guarded)
`bdsp_apical_bistable` (`config.py`) + the guarded block at `bridge.py:7258`: when on, the BDSP top-down apical
integration adds the SAME gap#5 terms — a v-gated self-LIMITING self-regen SUSTAIN
(`self_regen*sigmoid(v_hold_k*(v-v_hold))*(E_e-v)`, == `fused_coincidence_plateau`'s form) + the KIR down-state — so a
brief top-down teaching error LATCHES a held apical plateau (silent at rest = P0 moat). Default off => byte-identical
(CI: `test_bdsp_apical_bistable_off_is_plain_leaky` + latch-and-hold + `test_plasticity_inertness` 15 pass). Runner
wired: `--apical-bistable/--apical-self-regen/--apical-kir-g`.

## The real ADVANCE — the coupling IS the fix (correcting a banked error)
The prior arc concluded "`bdsp_apical_couples_soma` is NOT the fix (soma_g=2.0 no effect)". **That was an
UNDER-POWERED test.** With a strong coupling the apical raises the MEASURED burst rate B:
`soma_g` 2→8→20→50 gives B_apical 0.000→0.000→0.006→**0.11** (`B_rises True`), moat intact (B_rest 0.000 => dev=0 at
rest). A SPARSE output-bias (240-280) then gives a CLEAN MOAT: apical-lesion hid>out dw << credit dw (8-10:1). ⇒ the
2026-07-10 "coupling lifts B but the learning run was NEVER completed" arc is now COMPLETED on the diagnostic side —
the on-bridge BDSP path is ALIVE (B rises, directed credit, clean moat). Silent-failure lesson: a single under-powered
lever value (soma_g=2) is NOT a mechanism verdict — sweep the lever before concluding "X is not the fix."

## The honest NEGATIVE — bistability does NOT help THIS protocol
`_d1_onbridge_learn_to_accuracy.train_epoch` HOLDS the apical drive ON for the entire teach+learn window, so the
bistability (which engages only AFTER a drive is removed) is INERT here — and when forced on, it HURTS (B_apical→0.000,
KIR fights the held coupling; P_rest drops below p0). The bistability's value is the BRIEF-error / gap#5 regime (hold a
transient across the eligibility window / a silent CA3 down-state), NOT a booster for a held-drive BDSP protocol.

## The CONFIRMED boundary — depth-fragile FA credit (~0.715 ceiling < 0.75)
Even with clean directed credit (moat 8-10:1) + 12 epochs at the clean-moat regime (soma_g=50, output_bias=280),
on-bridge BDSP held-out = **chance** (0.549) on the depth-2 threshold-of-5-pair-XORs task (oracle 0.989). This is NOT
a coupling/moat failure — it is the SAME depth-fragile FA-credit-QUALITY wall already multiply-documented:
`_emerge1b` (faithful rate Burstprop) caps at held-out **0.715** (< the 0.75 bar) even though the level-1 XOR latents
EMERGE (probe 0.87); `2026-07-07-D2-...-burstprop-depth-fragile-FA-alignment-degrades`. The rate ceiling (0.715)
UPPER-BOUNDS the noisier on-bridge spiking version, so more on-bridge epochs/width cannot clear 0.75. Confirmed
boundary, not a tuning miss.

## The prescribed PIVOT (per this arc's own research gate) — BTSP, where the keystone IS the enabler
The gap#4 research gate (`2026-07-18-gap4-research-gate-BDSP-on-bistable-apical...`) named the fallback for exactly
this stall: **"if the global-loss depth gate stalls (the FA rule-variance wall), pivot to BTSP (Bittner-Magee 2017) —
local, one-shot, plateau-gated, eligibility-trace, NO global-loss backward pass, native to the CA3/CA1 machinery just
closed (gap #5); the bistable UP-state is THE enabling mechanism."** BTSP sidesteps the FA-credit-quality wall entirely
(it does not backprop a global loss through depth), and the gap#5 bistable plateau I built is precisely its
seconds-long plateau-gated eligibility mechanism. This also aligns with the EMERGENCE BAR: the brain does not do deep
backprop — it uses local plateau-gated + recurrent + burst rules, and the EMERGE arcs already learn emergent structure
with local rules. ⇒ reframe gap#4 from "deep supervised backprop credit" (a confirmed wall) to "local plateau-gated
credit that lets the substrate LEARN" — biological, mission-aligned, reusing the keystone. NEXT: the BTSP cheap-first
de-risk (plateau-gated one-shot association on the bistable plateau; sources Bittner 2017 10.1126/science.aan3846,
Milstein 2021 eLife 73046, BTSP-credit biorxiv 2025.06.12.659336).

## Status
- CLOSED/banked: the bistable-apical `sim/` edit (byte-identical off, CI-guarded) — the gap#5 keystone + BTSP enabler.
- ADVANCE: on-bridge BDSP coupling REVIVED (B rises with strong soma_g; clean moat with sparse bias) — corrects the
  under-powered "coupling isn't the fix" conclusion.
- CONFIRMED BOUNDARY: BDSP-FA deep supervised credit is depth-fragile (~0.715 rate ceiling) — the on-bridge path
  inherits it; not a coupling/tuning miss.
- PIVOT (next mechanism, per THE LAW): BTSP plateau-gated local credit on the bistable plateau.
