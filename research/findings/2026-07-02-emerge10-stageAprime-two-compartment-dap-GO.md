# EMERGE-10 / rung-4 Stage A' — GO: the guarded two-compartment dAP NEURON works on the real substrate. A distally-primed cell fires FIRST at lower feedforward drive WITHOUT firing on the plateau alone (predictive != active), the somatic runaway is gone, byte-identical when off, tested. The rung-4 substrate port's riskiest piece is done. (First `sim/` edit of the arc — additive, guarded, default-off.)

**2026-07-02 (autonomous).** `sim/` edit: `sim/config.py` (+`enable_two_compartment_dap` + apical params) + `sim/bridge.py` (a `cp_v_apical` compartment + a coupled-ODE plateau block, guarded). Test `tests/test_two_compartment_dap.py` (2/2). Runner `research/runners/_emerge10_stageA_dap_fire_first_derisk.py` (`--two-compartment`). CPU numpy-backend `SimulationBridge`; multi-seed 42/43/44.

## The mechanism (the genuine two-compartment dAP)
Stage A confirmed a single-compartment coincidence plateau injected as somatic current is binary (risk-1), and a minimal current-routing shortcut ran away (the plateau senses the somatic voltage, closing a feedback loop). The genuine fix, now built:
- `cp_v_apical` is a real leaky membrane VOLTAGE (allocated at rest, guarded by `enable_two_compartment_dap`, default off).
- The coincidence-plateau kernel `fused_coincidence_plateau` is called with `cp_v_apical` (not the soma), so the Mg2+-unblock REGENERATION happens on the APICAL compartment.
- The apical ODE: `tau dv_apical/dt = -(v_apical - E_rest) + R * I_plateau + g_couple * (v_soma - v_apical)`.
- Only the electrotonic coupling `g_couple * (v_apical - v_soma)` reaches the soma — so a full apical plateau/spike stays SUB-THRESHOLD at the soma, and there is no somatic runaway (the regeneration never senses the soma).

## Results — GO, multi-seed, byte-inert-when-off
On the real bridge (ctx_weight 0.1, plateau_scale 1.0, two-compartment ON), across g_couple in {0.3, 1.0, 2.0} and seeds 42/43/44:
- **Fire-first:** a distally-primed cell fires at LOWER feedforward drive than the SAME cell with the plateau off (same context volley) — plateau-specific advantage **+40 pA**.
- **predictive != active:** the plateau ALONE (no feedforward) does NOT fire the cell — **noFF = 0.00** (the invariant that FAILED under the single-compartment injection now HOLDS).
- **Anti-cheats:** DESYNCHRONIZED context gives no advantage (~ baseline); the effect is the plateau's (coincidence-on vs off).
- **Byte-inert when off:** `tests/test_two_compartment_dap.py` — with the flag off, `cp_v_apical` is never allocated (the two-compartment code path is not taken -> byte-identical to before the edit); with the flag on, it is allocated. 2/2 pass.

So the ONE genuinely-new biophysical behavior of the HTM-TM port — a dendritic dAP that primes a cell to fire first without firing it — works faithfully on the real spiking substrate, guarded and additive.

## Next: rung-4 Stage B + C (compose the proven pieces)
1. **Stage B — per-column WTA + the two-compartment dAP + FROZEN numpy-learned permanences => EMERGE-9c parity.** Wire M columns (subpopulations of two-compartment cells + an FS-interneuron WTA, the nav lateral-inhibition recipe) with the distal `coincidence_detector` permanence pathways loaded from an EMERGE-9b-learned TM; drive an overlapping-sequence; check the branch-prediction reproduces the numpy spiking-inference GO (dAP-primed cells win the WTA -> sparse context-specific firing).
2. **Stage C — the three-term permanence kernel (the additive `fused_htm_permanence_update` + per-cell z EMA) => EMERGE-9d parity** (learning on the substrate).
Then the whole unsupervised sequence-learning mechanism runs on the real `SimulationBridge` (rung-4 complete) — the single-spiking-substrate realization.

## Honest scope
- The apical params (tau/R/g_couple/E_rest) are at illustrative values that give the GO across a coupling range; Stage B will tune them jointly with the WTA operating point.
- This is a real `sim/` edit (fair game for faithful biology): additive, guarded by `enable_two_compartment_dap` (default off -> byte-inert, tested), a coupled-ODE two-compartment realized on the Izhikevich path.

## Artifacts
`sim/config.py`, `sim/bridge.py` (guarded two-compartment dAP), `tests/test_two_compartment_dap.py`, `research/runners/_emerge10_stageA_dap_fire_first_derisk.py`, `research/findings/raw/_emerge10_stageAprime_2comp_GO.json`. Prior: `2026-07-02-emerge10-stageAprime-two-compartment-must-sense-apical-voltage.md`, `2026-07-02-emerge10-rung4-stageA-risk1-confirmed-need-two-compartment.md`.
