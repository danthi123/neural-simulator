# Conversion Phase 2 (cheat C, memory store) — substrate weight-store de-risk → GO — 2026-06-05

Second phase of the cheat-conversion plan (`docs/plans/2026-06-05-conversational-cheat-conversion-plan.md`). The RF
composer's fact memory is a Python list `self.kb = [(fact_dict, composite_phases), ...]` — the bound composite is a
numpy array, not in the substrate (cheat C). The biology-grounded replacement (per
`2026-06-05-cheat-BC-spiking-phasor-cleanup-memory-research.md`, Route C-A) is the Crawford-Eliasmith weight-store,
phasor version: hold the bound composite phasor in per-fact COMPLEX output weights (a 'trigger' neuron whose complex
synapses to D readout neurons carry the composite `c[k]`); fire the trigger → the readout neurons reconstruct the
composite IN PHASE (the magnitude-invariant RF phase readout). Biology: Hebb cell-assembly memory-in-synaptic-weights
(Kandel 6e p.1357 verbatim: facts "bound together by excitatory synaptic connections strengthened at the time the
memory was formed"; Tonegawa/Liu 2012) + Marr CA3. The trigger→readout complex weights ARE the bridge complex synapse
— the SAME `W` object as the Phase-1 cleanup.

**De-risk GATE: a role unbind+cleanup from the SUBSTRATE-retrieved composite == the same from the numpy-stored
composite (cleanup held constant), multi-seed; PLUS a trigger-silence control (don't fire the trigger → the readout
is not a real composite → its unbind+cleanup must NOT equal the true agent — a genuine read, not a passthrough).**
`research/findings/raw/_phase2_substrate_store_derisk.py` (3 facts × 3 roles, seeds 42/43/44):

| D | substrate-store == numpy | trigger-silence-genuine |
|---|---|---|
| 128 | 27/27 | 9/9 |
| 256 | 27/27 | 8/9 |

**Verdict: GO.** The bound composite can be held in the SUBSTRATE (complex output weights) and retrieved in spikes
(fire the trigger → phase readout), with the unbind+cleanup matching the numpy store EXACTLY, multi-seed. No
phase-offset issue (the trigger fires at phase 0 → no offset). The trigger-silence control is genuine (8-9/9; the
single D=256 coincidence is a silent readout's garbage unbind happening to clean up to the agent by chance, 1/9 —
not a passthrough leak).

## What this establishes + the integration scope
- ESTABLISHES: the fact memory's BOUND COMPOSITE is substrate-holdable (synaptic weights) + spike-retrievable at
  parity. Cheat C's composite is CONVERTIBLE.
- The integration (next): `RFPhasorComposer(enable_substrate_store=True)` holds each fact's composite in a substrate
  weight bank (trigger → readout) instead of the numpy `kb` array; the query path retrieves via firing. Default OFF =
  numpy fast path (the Phase-1 cleanup opt-in pattern). Re-validate the agent's full suite at parity.
- HONEST residual (documented, per the plan): the `fact_dict` LABELS (agent/action/patient words + clause-vs-flat
  structure routing) stay Python — the STRUCTURE metadata. Decoding the full structure from the bound vector is the
  harder residual (the composite holds the bound roles; the routing labels are convenience). Phase 2 converts the
  COMPOSITE (the bound content); the label-structure is a named residual. Retrieval via firing is slower than a numpy
  read (a resonate per fact per query) — acceptable for the opt-in no-shortcut path; numpy stays the fast default.

## Artifacts
`research/findings/raw/_phase2_substrate_store_derisk.py` (substrate weight-store retrieve + unbind+cleanup vs numpy,
trigger-silence control, multi-seed). NO sim/ edits — reuses `rf_set_complex_weights` + `rf_kick` + `rf_read_phases`
(the same RF complex-synapse machinery as bind/unbind/cleanup).
