# R-iii on-substrate — DECISIVE: the point-neuron CA3 does NOT complete a partial cue even with a CLEAN, HAND-INSTALLED, strong attractor (all confounds ruled out), and the existing dendritic-coincidence plateau AS WIRED does not rescue it. The minimal-model surpass (CYCLE 1065) used an ABSTRACTED plateau (fires on clustered-input COUNT, no Mg voltage-gating, no point-soma dilution); the substrate's NMDA plateau is voltage-gated + additive to a point soma with no thin high-input-resistance dendritic compartment. The named next mechanism: a two-compartment dAP neuron with a high-local-resistance dendrite where a few CLUSTERED coincident recurrent inputs trigger a local regenerative spike that reliably fires the soma (Larkum thin-dendrite; Bouhadjar-Diesmann dAP) — research-gated. NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `_riii_ca3_coincidence_completion_derisk.py`, `_riii_ca3_attractor_diag.py`, `_riii_onsubstrate_readout_test.py` (+ inline direct-transmission probe). GPU. NO `sim/` edit (only public-array writes + guarded config flags).
**Verdict:** BOUNDARY, decisively characterized with confound-ruling controls — NOT a wall (names the next mechanism, research-gated).

## The diagnostic chain, run to convergence (no flip-flop — each confound tested + ruled out)
1. **Plateau correctly wired** (CYCLE 1066): routing ca3->ca3 through `fused_coincidence_plateau` gives a non-zero weighted coincident drive.
2. **No learned attractor** (CYCLE 1066): the ca3->ca3 rate-Hebbian does NOT potentiate within-ensemble specifically — within-ensemble weight 5.32 ~= member->truly-silent 5.22 (both ~init). Root cause included the `hebbian_max_weight=1.0` clip (found + fixed: it was clamping the recurrents DOWN, not potentiating up).
3. **So test the READ-OUT independent of learning** (`_riii_onsubstrate_readout_test.py`): INSTALL a clean attractor by hand (within-ensemble ca3->ca3 = W_HIGH, all others W_LOW) and test partial-cue completion.
4. **Result — the point-neuron does NOT complete even the installed attractor:** held-out completion = 0.014, IDENTICAL for PLATEAU vs LINEAR vs the FLAT control (no attractor), and INVARIANT across W_HIGH ∈ {15,40}, cue-drive ∈ {200,800} pA, and Mg ∈ {1.0,0.3,0.1} (the Mg-bootstrap hypothesis TESTED and REFUTED before it could become an overclaim). The installed within-ensemble structure has ZERO differential effect on held-out firing.
5. **Direct transmission probe** (rules out an installation bug): after install the recurrent weight read back = 40.0 (installation works); driving ONE pre (18 spikes, w=40) → its 89 ca3 posts reach only **max −46.5 mV** (sub-threshold), ~22 total post-spikes. The recurrents transmit but WEAKLY; a handful of clustered recurrent inputs on a partial cue is sub-threshold at the point soma.

## What this decisively establishes (the confounds it rules out)
- NOT a missing-attractor artifact: a PERFECT hand-installed attractor still fails (installed weight verified = 40.0).
- NOT an installation bug: the read-back confirms the weights, and the direct probe shows real (weak) transmission.
- NOT the Mg-bootstrap (tested Mg 1.0->0.1, no change).
- NOT cue-drive weakness alone (tested 200->800 pA).
⇒ The residual is the POINT-NEURON limit: a partial cue's sparse recurrent input, summed linearly (or via the additive-to-soma plateau), does not cross threshold — and the coincidence-plateau AS WIRED does not rescue it.

## The abstraction gap to CYCLE 1065 (honest)
CYCLE 1065's minimal numpy model surpassed this because its "plateau" was an ABSTRACTED branch-threshold that fires purely on the clustered-input COUNT — no Mg voltage-gating (no sub-threshold-membrane requirement) and no point-soma dilution. The substrate's `fused_coincidence_plateau` is (a) Mg voltage-gated (regenerates only once depolarized) and (b) additive to a POINT soma that dilutes the sparse recurrent input across the whole membrane. The minimal model's SUFFICIENCY claim stands for the mechanism (clustered supralinear integration completes); the substrate needs the STRUCTURE that makes a few clustered inputs produce a large LOCAL depolarization.

## The named next mechanism (research gate — do NOT hack configs further)
A genuine two-compartment neuron with a THIN, HIGH-LOCAL-INPUT-RESISTANCE dendritic compartment: a few CLUSTERED coincident recurrent synapses cause a large LOCAL depolarization there (high R -> large dV for small I), unblocking Mg locally -> a regenerative local dAP/plateau -> a strong, reliable soma drive (Larkum BAC thin-dendrite; Bouhadjar-Diesmann 2022 dAP; the project's EMERGE two-compartment dAP + `enable_two_compartment_dap` with the coupling RE-TUNED so the dAP is supra-threshold for the soma, unlike the current attenuating `apical_g_couple`/`apical_R` which SUPPRESSED completion in the CYCLE-1066 corner). Deep-research the local-compartment input-resistance + dAP->soma coupling regime FIRST, then a cheap-first de-risk on the installed attractor (does a properly-tuned dAP complete where the point soma fails?), then the emergent-attractor learning problem (item 2 above) once completion is demonstrable.

## Files
`research/runners/_riii_ca3_coincidence_completion_derisk.py`, `_riii_ca3_attractor_diag.py`, `_riii_onsubstrate_readout_test.py`. Prior: `2026-07-08-riii-onsubstrate-coincidence-wired-but-blocked-by-missing-attractor.md` (1066), `2026-07-08-riii-dendritic-completion-surpass-cheap-first-GO.md` (1065). NEXT: research-gate the high-input-resistance two-compartment dAP coupling regime, then the installed-attractor read-out de-risk with it.
