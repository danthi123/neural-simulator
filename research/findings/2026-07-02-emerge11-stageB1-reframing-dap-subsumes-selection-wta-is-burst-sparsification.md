# EMERGE-11 / rung-4 Stage B1 — reframing (build-informative): the two-compartment dAP ALREADY provides the sparse, context-specific SELECTION (Stage A' covers it — primed cells fire at a drive where non-primed do not); the per-column WTA's genuine role is BURST sparsification (pick k winners when an unpredicted column bursts), a separate delicate timing calibration. The genuine remaining rung-4 deliverable is Stage B2 — the full multi-column recurrent-distal TM on the real bridge -> EMERGE-9c branch-prediction parity.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge11_stageB_wta_dap_derisk.py` (Stage-B1 scaffold: columns + FS-WTA + two-compartment dAP on a real `SimulationBridge`). Reuse-by-import; NO new `sim/` edit; CPU numpy-backend.

## What Stage B1 attempted
Compose the two proven pieces — the Stage-A' two-compartment dAP + a within-column FS-interneuron WTA — to show a dAP-primed subset wins the spiking competition (sparse, context-specific firing). Built one column split into a primed subset + others, both sharing an FS-WTA, with a distal `coincidence_detector` pathway priming the subset; swept the feedforward drive.

## What it revealed (the reframing)
1. **The dAP already differentiates.** Stage A' established that a distally-primed cell fires at a LOWER feedforward drive than an unprimed one. So in the transition drive-zone, the primed subset fires and the non-primed do NOT — sparse, context-specific selection **without needing the WTA**. This is exactly the "predicted column -> sparse" behaviour of EMERGE-9c, and it is already GO (Stage A').
2. **The WTA's real role is BURST sparsification, not selection.** In HTM, the WTA matters when a column BURSTS (unpredicted -> all cells above threshold): the inhibitory neuron fires after the first spikers and caps the winners at k. In the PREDICTED case the dAP threshold-differentiation already yields sparsity, so the WTA is largely redundant there.
3. **The Stage-B1 WTA wiring over-suppresses** (FS -> whole column silenced across the sweep) — the classic WTA timing calibration (the FS must fire AFTER the first spikers and its inhibition catch only the rest; a 1-2 step window tuned by col->FS / FS->col weights vs the drive). This is the burst-sparsification tuning, deferred to where it is load-bearing (Stage B2's burst step).

## Implication — the genuine remaining rung-4 build is Stage B2
The sparse context-specific SELECTION (predicted -> primed fire) is already substrate-validated (Stage A'). The remaining rung-4 deliverable is the FULL TM assembly on the bridge:
- **M columns** (subpopulations of two-compartment cells), a **recurrent distal `coincidence_detector` permanence pathway** (cells -> cells) whose connectivity is the FROZEN EMERGE-9b-learned segments (mature synapses), and a per-column WTA (tuned for the burst step).
- Drive an overlapping-sequence; the prior-active SDR (via `cp_prev_firing_states`) primes the next column's context-specific cells (dAP) -> they fire sparse -> propagate distinct SDRs -> **branch-prediction parity with EMERGE-9c**.
- The hardest piece is loading the arbitrary learned distal connectivity into a bridge pathway with the coincidence mask (an explicit-wiring build). GO = branch-prediction reproduces EMERGE-9c + dAP-lesion + (burst) WTA-lesion collapse + multi-seed.

## Status of the rung-3 -> rung-4 arc (all committed)
Rung-3 fully validated + fully spiking (EMERGE-9b/c/d GOs, 32-context capacity). Rung-4: scoped, Stage-A de-risked (risk-1), **Stage A' GO** (the two-compartment dAP neuron in `sim/`, guarded + tested). Stage B1 (this) reframes the remaining work to Stage B2 (the full TM assembly, the large explicit-wiring build) + Stage C (the three-term permanence kernel -> EMERGE-9d parity). The sparse selection is already validated; the remaining work is assembly + connectivity-loading + the burst-WTA tuning.

## Artifacts
`research/runners/_emerge11_stageB_wta_dap_derisk.py`, `research/findings/raw/_emerge11_stageB_wta_dap.json`. Prior: `2026-07-02-emerge10-stageAprime-two-compartment-dap-GO.md`, `2026-07-02-emerge9c-spiking-tm-rung3b-GO.md`.
