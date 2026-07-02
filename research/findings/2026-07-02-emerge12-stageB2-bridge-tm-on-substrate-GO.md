# EMERGE-12 / rung-4 Stage B2 — GO (6/6 seeds UNANIMOUS): the FULL HTM Temporal Memory INFERENCE runs on the real `SimulationBridge`. The frozen EMERGE-9b-learned distal connectivity, LOADED via `inject_explicit_wiring` as a recurrent `coincidence_detector` pathway, reproduces context-specific branch prediction (== the EMERGE-9c numpy spiking reference) from the bridge's OWN coincidence recurrence + apical-dAP fire-first selection. The unsupervised sequence-memory mechanism is now on-substrate; only Stage C (learning) remains for rung-4.

**6-seed:** BRIDGE branch **1.000** on all of 42/43/44/100/101/102; dAP-lesion **0.000** / untrained **0.000** / EMERGE-9c-parity **1.000** / >> Markov 0.500 / >> chance 0.500 — every seed, every anti-cheat.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge12_stageB2_bridge_tm_derisk.py`. Reuse-by-import (EMERGE-9b `HTM`, EMERGE-9c `SpikingTM`); the two-compartment dAP is the already-committed guarded `sim/` edit (`enable_two_compartment_dap`, default-off byte-inert); the connectivity load is the EXISTING `inject_explicit_wiring` API. CPU numpy-backend; multi-seed.

## What Stage B2 is
Rung-3 validated the unsupervised HTM Temporal Memory fully in numpy (EMERGE-9b discrete GO, 9c spiking GO, 9d spiking-learning GO). Stage A' put the ONE genuinely-new biophysical piece — the two-compartment apical dAP — on the real bridge. Stage B2 is the FULL INFERENCE on the bridge: the learned connectivity lives ON the substrate (not in a numpy `HTM` object), and the branch prediction emerges from the bridge's own spiking recurrence.

## The on-substrate realization (faithful; host only for the world/body interface)
- **Prediction = native bridge machinery.** The recurrent distal `coincidence_detector` pathway computes, per post-cell, `c_drive = COUNT of coincidence-routed synapses whose presyn fired last step` (`cp_prev_firing_states`) — EXACTLY the HTM `_seg_conn_active`. With `coincidence_k_threshold = act_th`, a cell becomes PREDICTED (its apical dAP plateau fires, charging `cp_v_apical` = the two-compartment apical compartment) iff its active distal segment cleared `act_th`. The 1-step `prev_firing` gives the recurrence (winners at t prime cells at t+1).
- **Winner selection = spiking.** A dAP-primed cell (elevated `cp_v_apical` -> electrotonic soma coupling) reaches the Izhikevich threshold at a LOWER feedforward drive; a per-column global fast-spiking WTA interneuron caps the winners -> sparse, context-specific spiking selection.
- **The frozen connectivity export:** for each post-cell, the UNION over its segments of connected synapses (permanence >= `perm_conn`) -> one `(pre->post)` edge tagged `coincidence_detector=True`, injected via `inject_explicit_wiring`. In this task the allocated SDRs are DISJOINT (A-context vs E-context), so the per-cell flat coincidence count == the active segment's per-segment count (only one prior SDR active at a time) — the honest simplification the 9c-parity control verifies.
- **Host code = only the world/body interface:** presenting the input symbol sequence (driving each symbol's column = the sensory stream) and a per-symbol clean priming step (feed the captured spiking winners to the coincidence pathway == EMERGE-9c's per-symbol predictive-set computation). The COGNITION (winner selection = spiking WTA; prediction = coincidence recurrence) is all the bridge's.

## Results — GO (3-seed; 6-seed confirming)
Overlapping sequences `[cue]+[shared middle L=4]+[branch]`, `n_seq=2`, `n_cells=16`, `act_th=3`, sub-threshold dAP (`apical_g_couple=2`, `plateau_scale=1`), middle-drive ~340-400 pA (the window where dAP-primed cells fire but non-primed stay sub-rheobase), WTA `col_fs_weight=80`/`fs_col_weight=100`:
- **BRIDGE branch prediction 1.000** (seeds 42/43/44) — perfect context-specific high-order prediction (cue0 -> branch col6, cue1 -> branch col7) from the loaded connectivity.
- **== EMERGE-9c numpy spiking reference 1.000** (parity: the bridge reproduces the numpy spiking TM exactly, so the flat per-cell coincidence == per-segment in this disjoint-SDR task).
- **dAP-LESION 0.000** (coincidence off -> no apical priming -> the middle never fires context-specifically -> collapses; the loaded distal pathway is LOAD-BEARING).
- **untrained 0.000** (empty segments -> no distal connectivity -> collapses).
- **>> Markov floor 0.500, >> chance 0.500.**

## The debugging arc (systematic; the operating point is delicate but principled)
The winner-selection operating point on slow synchronized Izhikevich integrators was the delicate piece Stage B1 flagged. The arc, root-caused not guessed:
1. The IZH2007_RS cell ramps slowly (10-14 step latency at constant current) and fires synchronized all-or-none -> no within-window graded "primed-first" from drive alone.
2. The dAP DOES create a step-level lead (measured), but a strong dAP (`gc=5`) fires primed cells DURING priming (violating predictive!=active) -> they adapt (Izhikevich `u`) -> non-primed win. Fix: SUB-THRESHOLD dAP (`gc=2`, Stage-A'-validated) + middle-drive below the non-primed rheobase -> only dAP-boosted primed cells fire.
3. A per-column conductance WTA is knife-edge (burst vs over-suppress); the robust regime is sub-threshold-dAP drive-level sparsity + the WTA only to cap.
4. Propagation died after 2 hops because within-window winner firing is scattered in time -> the coincidence count drops below `act_th`. Fix: an explicit clean priming step (hold the captured winners in `prev_firing` for several no-drive steps) so the coincidence sees all winners synchronously and primes the next column to a clear apical level — == EMERGE-9c's symbol-by-symbol predictive-set computation.
5. The readout was contaminated by residual somatic firing backprop into the apical (the gc coupling raises `v_apical` when the soma spikes); fixed by resetting the soma inside the priming step.

## Status of the rung-3 -> rung-4 arc
Rung-3 fully validated + fully spiking (EMERGE-9b/c/d). Rung-4: Stage A (risk-1) -> Stage A' (two-compartment dAP GO, `sim/`) -> Stage B1 (reframing) -> **Stage B2 (this, INFERENCE on-substrate GO)**. Remaining: **Stage C** — the three-term permanence kernel (the additive/guarded `fused_htm_permanence_update` + per-cell dAP-rate homeostasis) so LEARNING runs on the substrate too -> EMERGE-9d parity -> **rung-4 complete** (the whole unsupervised sequence-learning mechanism on the real `SimulationBridge`).

## Artifacts
`research/runners/_emerge12_stageB2_bridge_tm_derisk.py`, `research/findings/raw/_emerge12_stageB2_bridge_tm.json` (3-seed), `_emerge12_stageB2_bridge_tm_6seed.json` (6-seed). Prior: `2026-07-02-emerge10-stageAprime-two-compartment-dap-GO.md`, `2026-07-02-emerge11-stageB1-reframing-dap-subsumes-selection-wta-is-burst-sparsification.md`, `2026-07-02-emerge9c-spiking-tm-rung3b-GO.md`.
