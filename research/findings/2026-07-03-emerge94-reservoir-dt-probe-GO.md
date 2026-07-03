# EMERGE-94 (probe) — RUNG A.3 dt-reconciliation: the on-bridge SPIKING reservoir parses at dt=1.0 — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge94_reservoir_dt_probe.py`
**Raw:** `research/findings/raw/_emerge94_reservoir_dt_probe.json`

## Why

RUNG A.2 (EMERGE-93) folded the composer + producer onto ONE bridge at dt=1.0 (the producer's dt). To fold the SPIKING
reservoir (`OnBridgeLSM`, tuned at dt=0.5) onto the SAME bridge — RUNG A.3, all three spiking components on one bridge —
it must parse at the shared dt=1.0 (a `SimulationBridge` has one global dt). This single-variable probe resolves that.

## The probe — **GO**

An additive default-`0.5` `dt=` param on `_build_reservoir_bridge`/`OnBridgeLSM` (default = byte-identical to the
shipped tuning; the OnBridgeLSM consumers EMERGE-82/87/89 pass unchanged). The reservoir comprehender is built at dt=0.5
(baseline) and dt=1.0, fit, and its transitive parse_acc measured on held-out content.

| dt | parse_acc (6-seed) |
|---|---|
| 0.5 (tuned baseline) | **1.000** |
| 1.0 (shared-bridge dt) | **1.000** |

**Result:** the reservoir parses at dt=1.0 **identically** to dt=0.5 (1.000 all 6 seeds; the per-seed spike counts are
byte-identical between the two dt values — the strong per-token input drive dominates the dt-scaled integration, so the
population role-code is robust to dt). RUNG A.3 (all three spiking components co-resident at dt=1.0) is **unblocked** —
no re-tuning or per-phase dt switch is needed.

## Files
- `research/runners/_emerge82_onbridge_lsm_derisk.py` — the additive `dt=` param (default 0.5 = byte-identical).
- `research/runners/_emerge94_reservoir_dt_probe.py` — the dt probe.
- `research/findings/raw/_emerge94_reservoir_dt_probe.json` — the 6-seed dt-invariance.
