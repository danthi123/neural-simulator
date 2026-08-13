---
type: finding
status: contributing
date: 2026-08-12
mechanism: reconsolidation
---

# Reconsolidation production organ — D2-spiking-surprise-gated in-place belief revision — WIREABLE (GO)

**Date:** 2026-08-12
**Status:** **WIREABLE GO.** Verify-first reproduced the prior 6/6 GO, then built the co-resident organ + a
standalone verify harness. numpy/CPU, reuse-by-import, **NO `sim/` edit**. Committed to the worktree branch
(not pushed); a later integrate agent wires it.
**Organ:** `research/runners/reconsolidation_production_organ.py`
**Verify harness:** `research/runners/_reconsolidation_production_organ_verify.py`
**Raw:** `research/findings/raw/_reconsolidation_production_organ_verify.json`
**Reuses (by import, no reimplementation):** the D2 `SurpriseProductionOrgan`
(`research/runners/surprise_production_organ.py`, 6/6-GO spiking mismatch unit); the composer's own de-risked store
rewrite — `RFPhasorComposer.update_on_mismatch` (Option A, `2026-06-17-reconsolidation-update-derisk-GO.md`, 6/6)
and the production-default `OneBrainComposer` substrate-slot rewrite (`_write_block` + `_compose_phases`).

## The gap it closes

The production KB is APPEND-ONLY: correcting a stored fact appends a contradictory duplicate and recall answers the
STALE first match. "the dog went north" then "actually, south" leaves two facts, north answered. No belief revision.

## What was verified FIRST (the E1 lesson: a claimed GO can be a mis-read / host-formula / seed-fragile)

- **Reproduced the prior GO** numpy-CPU: `_phaseB_reconsolidation_update_derisk` = **6/6** (reconsolidate corrects, C1 boundary, C2 moat, C3 lesion, arms separated), PE separation same 0.480 / diff 1.003 from `research/findings/raw/_phaseB_reconsolidation_update.json`. <!--derived-->
- The load-bearing numbers of THIS organ are in `research/findings/raw/_reconsolidation_production_organ_verify.json`.
- **Read the load-bearing code, not the prose.** The Option-A `update_on_mismatch` gate is a HOST cosine
  (`_patient_prediction_error` + auto-calibrated `_calibrate_pe_labile`) — a host formula, and the in-place update
  mechanism already lives on `RFPhasorComposer` **but is UNWIRED** (brain_chat never calls it; the correction path
  appends). The production-default `OneBrainComposer` does not expose `update_on_mismatch` at all (oracle-only) and
  its recall reads the DEVICE store, so a host-kb-dict rewrite would not change the recalled answer.
- **The honest fix = the mission's own design.** Drive the reconsolidation WINDOW from the D2 SPIKING surprise
  (`cp_firing_states[surprise]`), NOT the host cosine, and perform the in-place update via each composer's OWN
  de-risked store path (rf `update_on_mismatch`; onebrain SAME-slot `_write_block`+`_compose_phases`). The
  genuinely-spiking part — the PE-gated window — is exactly what the mission asked for.

## The mechanism (composition; reuse-by-import)

1. **Reactivation + moat (B3 non-contradiction):** the FIRST stored fact whose cue (agent+action) matches; a
   never-stored cue **abstains** — a reactivated trace is updated, a missing one is never fabricated.
2. **The window (D2 surprise, SPIKING):** `SurpriseProductionOrgan.judge()` reads `cp_firing_states[surprise]`.
   A contradiction FIRES (window OPENS); a confirmed re-statement CANCELS to ~0 Hz (window CLOSED). The gate is a
   threshold on the spiking rate — no host `stored==asserted` compare.
3. **The in-place update:** window open + trace present -> rewrite the reactivated fact's patient IN PLACE via the
   composer's own store (rf: `update_on_mismatch(pe_labile=0.0)`, the spike gate already decided; onebrain: rewrite
   the SAME persistent-store slot). No contradictory duplicate.

Default-ON; `BRAIN_RECONSOLIDATION=0` -> the byte-identical append-only oracle.
`BRAIN_RECONSOLIDATION_LESION=1` -> the window fires but the update is blocked -> append-only fallback -> STALE.

## Result — GO on BOTH composers

| check | rf (6 seeds) | onebrain (3 seeds, production default) |
|---|---|---|
| INTACT: contradiction opens the spiking window (5.613 Hz) -> in-place rewrite; recall **north->south**, ONE fact, untouched intact | **6/6** | **3/3** |
| C1 BOUNDARY: a re-statement does NOT open the window (0.0 Hz) -> restabilize, no write | **6/6** | — |
| GATE load-bearing: intact window separates contradict(open) from restate(closed); lesioning the D2 prediction edges makes the re-statement read surprised | **6/6** | — |
| MOAT: never-stored cue -> abstain, no fact fabricated | **6/6** | (rf-covered) |
| LESION load-bearing: block the update -> recall flips **south->north** (stale), duplicate coexists | **6/6** | **3/3** |
| FLAG-OFF byte-identical to a plain `store()` append (production-today) | **6/6** | (by construction) |

**Load-bearing numbers:** intact q=**south** n=1; lesion q=**north** n=2; window contradict **5.613 Hz** (open) vs
re-statement **0.0 Hz** (closed), gate-lesion makes the re-statement open (the separation is caused by the learned
spiking prediction, not a fixed input artifact). Harness elapsed 82.9s.

## Honest scope / residuals

- **Genuinely spiking = the WINDOW.** The decision to open reconsolidation is a `cp_firing_states[surprise]` read.
  The STORE rewritten is the composer fact KB — the documented composer-as-idealization layer ALL recall uses (the
  bind/unbind/composite/store run on the resonate substrate). The window is spiking; the store idealization is the
  same residual recall already carries, not a new shortcut.
- **Reactivation selection is a kb cue-match** (host), identical to the selection recall already performs; a fully
  spiking cue-addressed reactivation rides the one-brain merge (the same residual the surprise/comprehension organs
  carry: co-resident bridge, not merged onto the one recall bridge).
- **The synaptic-literal tier** (engram tag-and-capture reactivation + a PE-driven `plasticity_window_gate` +
  forget/consolidate) is the named next rung the block-major store already de-risks the gate logic for.
- **Superposed/lossy store** has an SNR knee below D/K ~64/18 (`2026-08-11-reconsolidation-superposed-lossy-...`);
  this organ uses the block-major KB (the production store), where isolation is structural.

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._reconsolidation_production_organ_verify \
    --seeds 42,43,44,100,101,102 --onebrain-seeds 42,43,44
```
