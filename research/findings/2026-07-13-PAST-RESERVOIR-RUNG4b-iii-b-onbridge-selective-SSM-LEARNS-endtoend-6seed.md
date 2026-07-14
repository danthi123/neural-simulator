# Past the reservoir bound, Rung 4b-iii-b (END-TO-END ON-BRIDGE, 5/6 GO, 6/6 directional): the transport-free selective SSM LEARNS while its state lives on the spiking bridge — reproducing the numpy ladder EXACTLY

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung4b_iiib_onbridge_selective_ssm_task_derisk.py` (uses the `enable_selective_ssm_state` `sim/` mechanism; numpy backend; NO further `sim/` edit).
**Status:** ✅ 5/6 GO; 6/6 DIRECTIONAL — and the accuracies MATCH the numpy Rung-4a ladder exactly.

## What this closes — the on-bridge learning loop

Rung 4b-ii landed the additive on-bridge SSM-state mechanism; Rung 4b-iii-a proved the on-bridge state is byte-equivalent to the numpy selective SSM. This rung runs the FULL LEARNING LOOP through the on-bridge state: per token of the gated-conjunction task, the runner sets `cp_ssm_inject = Win·E[tok]` + `cp_ssm_shunt = softplus(w·u)` (the selective gate), STEPS the real `SimulationBridge`, READS `s = cp_ssm_state`, and at the read (query) step trains the read-out (delta rule) + the gate (the forward-mode eligibility trace × the local read-out error) — all reading the ON-BRIDGE state. The gate learns to hold the distal key + conjoin it with the recent query, entirely from the state on the spiking bridge.

## Result — 6-seed (task = [KEY, filler×12, QUERY]→rule[KEY,QUERY]; chance 1/6=0.167)

| seed | ON-BRIDGE selective | fixed_res |
|---|---|---|
| 42 | 0.541 | 0.270 |
| 43 | 0.422 | 0.319 |
| 44 | 0.656 | 0.478 |
| 100 | 0.544 | 0.444 |
| 101 | 0.674 | 0.452 |
| 102 | 0.437 | 0.411 |
| **mean** | **0.546** | 0.396 |

- **selective > fixed_res on 6/6 seeds** (mean +0.150, min +0.026); 5/6 by the +0.08 gate margin (seed 102 marginal — the same seed that was marginal in the numpy Rung 4a).
- **The accuracies MATCH the numpy Rung-4a ladder EXACTLY** — Rung 4a (numpy conductance-shunt) was "selective mean 0.546, 5/6 GO"; this on-bridge run is "selective mean 0.546, 5/6 GO", seed-for-seed (42: 0.541/0.270 identical). This is the Rung-4b-iii-a equivalence confirmed THROUGH the full learning loop: the on-bridge state IS the numpy SSM, so the learning reproduces exactly.

## ⇒ the claim

The transport-free selective diagonal SSM — the honest path PAST the reservoir bound (Rung 1 conjunction-dx → Rung 2 recurrent → Rung 3 real text → Rung 4a conductance-shunt) — now LEARNS end-to-end while its state lives on the spiking `SimulationBridge`, via a forward-mode eligibility trace (no BPTT, no weight transport), reproducing the validated numpy result exactly. The long-range learner is on the substrate.

## Honest scope / next (the full-autonomy refinements)

- The STATE is on-bridge (the `sim/` mechanism); the LEARNING loop reads that on-bridge state. Still host-computed (reading the bridge state): the gate's eligibility trace + the read-out. The remaining refinements to a fully-autonomous on-substrate learner:
  1. the ELIGIBILITY TRACE as an on-bridge local synaptic quantity (another additive per-neuron trace — the trace is `e = lam·e + (dlam/dw)(s_prev - inj)`, all local),
  2. a synaptic input→shunt PATHWAY (input → `cp_ssm_shunt` via a conductance, so the gate is synaptic),
  3. the spiking READ-OUT (the FS-WTA read-out, already validated).
- These are each a bounded rung; the core result — the mechanism + its learning run on the substrate, reproducing the validated ladder — stands.
- Uses ONLY the Rung-4b-ii `sim/` mechanism (no further edit). CI: covered by the equivalence + hold/release guards (`tests/test_selective_ssm_state.py`).

## Files
- `research/runners/_reslm_rung4b_iiib_onbridge_selective_ssm_task_derisk.py`; raw `research/findings/raw/_rung4biiib/`.
- Follows Rung 4b-ii (`-RUNG4b-ii-...`) + 4b-iii-a (`-RUNG4b-iii-a-...`) + Rungs 1–4a.
