---
type: finding
status: no-go
lane: consolidation
board: 64
date: 2026-08-29
mechanism: dendritic consolidation OPERATING-POINT sweep (point-plateau) — the design-gate's NEXT-(a), a
  comprehensive grid over self_regen / k_thresh / wta / kir_g / slot_drive to find a NARROW selective bistable
  plateau that separates per-slot consolidation targets and beats a linear read-out
verdict: >
  NO-GO on the point-plateau. Harvested 343 cells (0 errors) across the mini-pc pool. NO operating point
  robustly separates: the best cells reach dend_selective 2/3 but dend_separated 0/3 with dend_ratio ~1.00
  (i.e. they barely match, never beat, the linear control). The design doc's pre-registered branch point
  NEXT-(b) is a continuous-attractor consolidation GEOMETRY (a line/ring of laterally-coupled slots) — memory
  separation, categorically distinct from the dendritic deep-CREDIT rule refuted in
  2026-07-22-gap4-real-issue-NOT-dendrites (that negative is about hidden-layer credit on spikes, a different
  question). Per NO-DEFER this is a verdict on the point-plateau METHOD, not on the consolidation capability.
---

# Consolidation dendritic opsweep — point-plateau NO-GO; continuous-attractor geometry is next

## What ran
The operating-point sweep [`_consol_dendritic_opsweep`](../runners/_consol_dendritic_opsweep.py) —
the prescribed NEXT-(a) of
[`2026-07-25-consolidation-dendritic-surpass-DESIGN...`](2026-07-25-consolidation-dendritic-surpass-DESIGN-weighted-coincidence-bistable-apical-on-slots-reuse-no-sim-edit.md)
— grids the bistable-plateau operating point (self_regen ∈ {0,0.05,0.10,0.15,0.20}, k_thresh, wta, kir_g,
slot_drive) looking for a NARROW selective plateau: self_regen low enough that the plateau does not latch ALL
slots, ignition tuned so per-slot consolidation targets separate AND beat a linear read-out.

Cells were pool-dispatched to pool40/41/42 and collected with
[`pool_opsweep_collect.sh`](../../tools/pool_opsweep_collect.sh).

## Result — NO-GO

Harvest artifact: `research/findings/raw/consol_opsweep_harvest_2026-08-29.json`.

<!--derived-->

- 343 cells, 0 errors.
- 0 robust candidates (candidate = dend_separated AND dend_selective ≥ 2 AND beats linear).
- Best cells: dend_selective 2/3 but dend_separated 0/3, dend_ratio ~1.00 — they match, never beat, the
  linear control. The point-plateau does not carry separable per-slot consolidation.

## Why — and the named surpass
The isolated point-plateau saturates non-selectively (consistent with the INTERIM finding that the write is a
nonselective plateau). The design doc's NEXT-(b) names a **continuous-attractor consolidation geometry** — a
line/ring of laterally-coupled slots that holds a graded, separable state the isolated point plateau cannot.
This is a MEMORY-SEPARATION mechanism and is categorically distinct from the dendritic deep-CREDIT rule
refuted in
[`2026-07-22-gap4-real-issue-NOT-dendrites`](2026-07-22-gap4-real-issue-NOT-dendrites.md)
(that negative is about hidden-layer credit assignment on spikes — a different question, not re-proposed here).
The successor de-risk runner is being built and will be pool-swept with the SAME verdict structure so the two
are directly comparable.

## Status
This closes the point-plateau operating-point search (banks the negative) and hands off to the
continuous-attractor de-risk. It does not defer the consolidation capability (board #64 / sleep-replay + D5
learn-through-use) — the successor mechanism is specified and in build.
