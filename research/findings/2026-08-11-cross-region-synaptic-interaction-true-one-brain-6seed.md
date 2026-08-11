---
type: finding
status: go
date: 2026-08-11
mechanism: genuine cross-region SYNAPTIC pathway conv-cue -> eprop_in on the merged one-brain bridge — co-residency becomes cross-region INTERACTION
lane: E-language / INTEGRATION (the "one brain" non-negotiable — the interaction level)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/i7_cross_region/xreg_6seed.json
runner: research/runners/_i7_cross_region_synaptic_derisk.py
instrument: builds on the #7 one-brain merge (co_resident_eprop) and injects a real synaptic pathway from the conversational cue to the e-prop acquisition input slice on the SAME merged SimulationBridge; the acquisition input now arrives via SYNAPSES, not a host hand-off. SIM_BACKEND=numpy; cfg.seed-controlled.
---

# Cross-region synaptic INTERACTION on the one-brain bridge — co-residency becomes true one-brain (6/6 GO)

The #7 one-brain merge (burn-down #1) put the e-prop acquisition net onto the SINGLE conversational bridge as
disjoint slices (CO-RESIDENCY: one `SimulationBridge`, one `cp_connections`) — but with ZERO conv↔eprop synapses,
so it was co-location, NOT yet cross-region synaptic INTERACTION (per `project_one_brain_substrate_vs_functional`,
co-location without cross-synapses is not full one-brain). This arc crosses that line: it injects a GENUINE SYNAPTIC
PATHWAY from the conversational cue to the e-prop input slice on the merged bridge, so the acquisition input arrives
via SYNAPSES on the shared substrate, not a host hand-off.

## Result — 6/6 GO (`research/findings/raw/i7_cross_region/xreg_6seed.json`: `GO_all: true`, `n_smoke_go: 6`, `n_cross_region_load_bearing: 6`)

<!--derived-->

Across seeds 42/43/44/100/101/102:
- **The #7 chat still GO 6/6** with the cue arriving via synapses (`taught_recall=3` every seed; smoke_go 6/6; moat intact).
- **The cross-region pathway is LOAD-BEARING on all 6 seeds** (`cross_region_load_bearing=True` × 6): LESION the conv→eprop
  synapse and the acquisition input collapses — the pathway is a real functional interaction, not a decorative edge.
  This is the tooth that distinguishes cross-region INTERACTION from co-location.

## Scope / honesty

<!--derived-->

- Reaches **genuine cross-region synaptic INTERACTION** on the one shared bridge: the conversational substrate now drives
  the acquisition slice through real synapses in the same `cp_connections`, and lesioning that pathway is load-bearing —
  the level above the co-residency the one-brain merge reached.
- Additive on top of the `co_resident_eprop` merge (default-off; #7 byte-identical when the cross-region flag is off).
  Runner-side; NO `sim/` edit.
- Named residuals (per THE LAW): the OUTPUT side (`eprop_out spikes -> composer render`) may still be a host read for the
  patient word (the neural patient read-out is a separate merged burn-down); the familiarity gate is the spiking v320
  (burn-down #2); the argmax/leaky-readout + AI-teacher remain declared scaffolds. This closes the INPUT cross-region
  pathway; a symmetric output synaptic pathway is the named next step toward a fully-interacting one brain.
- Build agent deferred pre-commit; coordinator ran the 6-seed sweep + verified the load-bearing teeth from the raw
  artifact + merged.
