# EDGE 5 rung 3 (plan #1) — HONEST PARTIAL + a refuted sub-hypothesis: a WEIGHT-based store lifts the multi-bind capacity from ~1 (rung-2 STP) to a weak ~2 via VALUE-SPECIFIC potentiation, but the error-correcting DELTA write (depression) is NOT the load-bearing lever at this scale (delta ≈ additive, 5/6) — the full multi-bind spiking store past ~2 stays open

**Date:** 2026-07-15 · **Runner:** `research/runners/_edge5_rung3_delta_store_onbridge_derisk.py` (a PLASTIC barcode→value weight store on the rung-2 bridge; a two-phase write: read the current spiking prediction, potentiate the target value + depress the wrong prediction; numpy-CPU; NO `sim/` edit — the write is a neural-read-gated potentiate/depress on `cp_connections.data`). Attempts the rung-2 multi-bind boundary's named surpass.

## The hypothesis + the honest result (6-seed 42/43/44/100/101/102; chance 0.25)
Rung-2's raw STP facilitation is ADDITIVE (single-bind; below chance at P=2). The numpy store showed a DELTA (error-correcting) write holds multiple binds. Rung-3 realizes a delta-like write on the bridge weights. Result:
| P (#binds) | DELTA (depress) retrieve | additive (no-depress) | reading |
|---|---|---|---|
| 1 | 0.81 (0.62–1.00) | — | single-bind works (a systematic-debug fix: the drive had to drop 600→55 so the potentiated pool crosses threshold while the baseline doesn't; the weight store is NOT release-scaled like STP) |
| 2 | **0.45** (0.31–0.56) | **0.44** | above chance but WEAK (holds≥2 only 1/6 at a 0.55 gate); **delta ≈ additive** |
| 3 | 0.31 | 0.30 | ~chance |
| 4 | 0.19 | — | ~chance |
- **The error-correction (depression) is NOT load-bearing at this scale: DELTA ≈ additive on 5/6 seeds** (only s102 showed a small depression benefit). ⇒ the "delta write surpasses via error-correction" hypothesis is REFUTED here.
- **What actually lifts multi-bind over rung-2 is the weight store's VALUE-SPECIFICITY:** weight potentiation targets barcode_i→value_i's specific synapses (discriminative at low drive), where rung-2's STP facilitation was PRESYNAPTIC-UNIFORM (barcode_i→all-val, so it couldn't separate values). That specificity buys a weak ~2-bind capacity; it degrades to chance by P3.

## ⇒ Why the error-correction stays idle + the honest frontier
At this SMALL, EASY scale (KV=4 value pools, P≤4, DISTINCT values per trial), the specific potentiation already handles ~2 binds with mild interference — so there is little wrong-prediction to correct, and the depression is idle (delta ≈ additive). The error-correction would only ENGAGE at genuinely HIGH interference (P > KV, SHARED values, noisier/overlapping codes) — the untested regime where the numpy delta>additive gap opened. So the honest status:
- **The single-bind spiking content-addressable store is the GO deliverable (rung-2).** A weight-based store extends it to a WEAK ~2-bind capacity via value-specific potentiation (rung-3).
- **The robust multi-bind (P≥3) spiking store remains OPEN.** Two honest paths: (a) the error-correcting write tested at HIGH interference (where it should engage — the untested regime); (b) per-bind SLOT SEPARATION via the binder (the scoping's point — bounded slots so binds don't share value pools; the spiking binder's FS-WTA is BANKED per the emergence bar). Path (b) is the more aligned one (the binder's whole purpose is bounded slots).
- This is a clean, honest boundary: the Gap-A spiking discourse memory holds 1 (robust) to ~2 (weak) binds; robust multi-referent spiking memory is the named frontier.

## ⇒ Discipline note
Recorded as a PARTIAL with the error-correction sub-hypothesis explicitly REFUTED (delta ≈ additive) — NOT framed as a "delta-write surpass GO" (the RUNG-1/2 adversarial-verify lesson: gate on the load-bearing control, here the no-depress/additive arm, which showed the depression idle). NO `sim/` edit. Runner: `_edge5_rung3_delta_store_onbridge_derisk.py`.
