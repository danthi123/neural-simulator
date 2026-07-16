# EDGE 5 rung 2 (plan #1, 6-seed GO): the content-addressable STORE is realized ON SPIKES on a real SimulationBridge via Mongillo synaptic FACILITATION — the bind lives in the facilitated `cp_stp_u`, re-presenting a written barcode retrieves its value; the store half of the Gap-A discourse memory is now fully spiking (the FS-WTA binder stays banked)

**Date:** 2026-07-15 · **Runner:** `research/runners/_edge5_rung2_stp_store_onbridge_derisk.py` (reuse-by-import `_mint_codes`; the committed STP path `enable_short_term_plasticity` + `stp_tau_f` + `fused_stp_decay_recovery`/`cp_stp_u`, `sim/kernels.py:333`; numpy-CPU; NO `sim/` edit). Realizes the Edge-5-rung-1 delta-STORE on spikes; Mongillo 2008 source-read done.

## Mechanism (the scoped, non-banked path)
A barcode-input region --[dense plastic-OFF barcode→value synapses, STP ON, LONG `tau_f`=1500ms]--> K value pools on ONE bridge. **WRITE** (present barcode_i + a teacher on value_i's pool): the co-active barcode_i-input→value_i synapses FACILITATE (`cp_stp_u` rises — the Mongillo Ca buffer). **FILLERS** decay it within `tau_f`. **RETRIEVE** (present barcode_j ALONE): the facilitated barcode_j→value_j synapses RELEASE strongest (release ∝ u·x) → value_j's pool fires most → the read (argmax value-pool rate; a plain rate read, NO hand-tuned FS-WTA) = the retrieved value. A NOVEL barcode has no facilitation → no bound value.

## Result (6-seed 42/43/44/100/101/102; chance 0.25; T=8 fillers within `tau_f`)
| arm | acc (mean) | reading |
|---|---|---|
| **RETRIEVE via facilitation** | **0.97** (0.933–1.000) | re-presenting the written barcode retrieves its bound value — ON SPIKES |
| novel-barcode control | 0.21 (0.100–0.300) | a never-written barcode → no facilitation → ~chance (the content-addressing is GENUINE, not a fixed map) |
| STP-OFF lesion | 0.06 (0.000–0.167) | facilitation disabled → fixed weights can't store the bind → collapses (facilitation is LOAD-BEARING) |
- **6/6 GO, unanimous.** The Mongillo facilitation store content-addresses by barcode on the real substrate.

## ⇒ What this closes
Edge-5 rung-1 (the mechanism unification, numpy) showed the delta-STORE + the clean-barcode BINDER compose, and that **the store ALONE content-addresses by barcode** (the binder buys the role cue + novel slots, separately). This rung realizes that STORE half fully on spikes: the content-addressable discourse memory (barcode→value, retrieve-by-content, past-horizon within `tau_f`) is a spiking, activity-silent, Mongillo-facilitation store on a real `SimulationBridge` — the Gap-A "one-brain-that-LEARNS discourse memory" store, no `sim/` edit, no hand-tuned FS-WTA.

## Honest scope + next
- The horizon is `tau_f`-BOUNDED (biological: ~1.5s augmentation; T=8 fillers fit; longer discourse needs the augmentation/replay regime — a characterized biological property, not a wall).
- The BINDER's spiking realization (barcode→slot + a clean slot-WTA read) stays BANKED per the emergence bar (RUNG 6f; hand-tuning the FS-WTA is the drift). The store's rate read here sidesteps it (no WTA hand-tuning). A non-hand-tuned spiking binder (for the role cue + novel slots) is the deferred research rung.
- Cheap-first single-pair here; the multi-pair-with-interference regime (does DELTA-style error-correcting facilitation beat additive saturation on-bridge, as the numpy store showed) is the named follow-on.
Reuse-by-import; NO `sim/` edit. Runner: `_edge5_rung2_stp_store_onbridge_derisk.py`.
