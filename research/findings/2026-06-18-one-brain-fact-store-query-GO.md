# Roadmap phase 2, step 2 — a full FACT stores + queries on ONE persistent bridge (no host round-trip): GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** (3 seeds × 2 D, 6/6). A complete
2-role conversational fact — `composite = bind(agent_role, agent) + bind(action_role, action)` — is **stored in a
register and queried by BOTH roles on ONE persistent bridge**, the whole bind→bundle→unbind chain running
register→register through complex synapses with **no `rf_read_phases` between ops**. Both roles recover the correct
filler 100% and **== the host composer pipeline** (`_encode` + `_unbind_phases` + `_cleanup`); lesioned binds
collapse. Builds on step 1 (the register→register handoff, `2026-06-18-one-brain-register-handoff-GO.md`).
**Runner:** `research/runners/_phaseB_onebrain_fact_store_query_derisk.py` | **Raw:**
`research/findings/raw/_phaseB_onebrain_fact_store_query.json`

## Mechanism — the store-and-query core on one bridge

7 registers (D each): a_in, v_in, a_bound, v_bound, **C (the stored composite)**, Q_agent, Q_action. Synapse banks
installed in 3 windows, no read-out between: (1) **bind** agent + action (a_in→a_bound via agent_role, v_in→v_bound
via action_role); (2) **bundle** (a_bound + v_bound → C); (3) **query** both roles to SEPARATE registers (C→Q_agent
via conj(agent_role), C→Q_action via conj(action_role)). Kick the two fillers, settle each window, read Q_agent /
Q_action. The composite stays in C between queries, so one stored fact answers both "who" and "what".

## Result — 3 seeds × {D=64, D=128}

| metric | mean | reading |
|---|---|---|
| agent recall self / == host | **1.000 / 1.000** | "who" recovers, identical to host |
| action recall self / == host | **1.000 / 1.000** | "what" recovers, identical to host |
| both-roles ≥ 0.99 | **6/6** | the chained store+query holds at every seed/D |
| lesion (sever a bind → that role's recall) | 0.17 / 2 | collapses (the on-bridge binds are load-bearing) |

## Two design insights for the integrated pipeline (load-bearing)

1. **Separate output registers per query.** Reusing one Q register made the second query's unbind drive land on top
   of the first's residual phasor (lam=0, no decay) and corrupted it (action recall went 0.00 → 1.00 once each query
   got its own register). In the full pipeline each read-out op needs its own target register.
2. **Work registers must be RESET between facts; stored facts must live in SYNAPSES, not register state.** The
   persistent bridge retains register phasors between facts, so a previous fact's `a_bound` leaked into `C` (the
   lesion control didn't collapse). Clearing the work registers' RF state (re = `cp_membrane_potential_v`, im =
   `cp_recovery_variable_u`) before each fact fixed it. ⇒ the real store must put facts in plastic synapse weights
   (the spiking weight-store, already validated) so a register reset can't erase them — the operand-vs-store split.

## Next (phase 2 continues)

The store+query core works on one bridge. Remaining to close the full who/what turn on one persistent bridge: the
**cleanup** is still numpy here (`comp._cleanup`) — fold in the validated spiking cleanup-WTA as an on-bridge region
reading Q; add the **familiarity-gate moat**; and the **parser front-end** (drive the operand registers from the
parser's role firing) so comprehension→store→query is one spiking flow, host doing only text I/O. Top risk stays
phase coherence as the chain lengthens (the multi-window settle is the mitigation; a phase-latch on C the fallback).

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_fact_store_query_derisk --seeds 42,43,44 --dims 64,128
```
