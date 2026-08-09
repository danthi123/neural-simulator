---
title: "Capacity-scaling holds the ceiling to N=50 but SLIPS at N=100 (grown 0.97→0.91→0.73) — capacity still helps but is NOT sufficient for lifetime scale"
date: 2026-08-09
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42, 43, 44]
---

# Capacity-scaling: real at every N, but the ceiling slips at N=100 — capacity alone is not the lifetime answer

## Claim

<!--derived-->

The reservoir-capacity lever that resolved the N=20 crux (grown 0.967) and held at N=50 (0.913) **degrades at
N=100 to 0.727** (3-seed). Capacity STILL helps at N=100 — grown 0.727 vs the fixed-store self_replay 0.617
(+0.110, grown > self_replay every seed) — so the lever is real at all scales. But it **no longer holds the ~0.8
ceiling**, and the substrate's *immediate acquisition* itself degrades (0.95→0.82). **⇒ "size the reservoir to the
fact count" is a small-N patch (on top of the de-clamp bound-trap fix), NOT the lifetime-scaling answer** — which
is exactly why the consolidation path (generative replay + a non-forgetting generator) is the real lever at scale.

## The capacity-scaling curve (grown/capacity arm, de-clamped)

<!--derived-->

| N | grown (capacity) | self_replay (fixed) | capacity rise | grown immediate-acq |
|---|---|---|---|---|
| 20 (6-seed) | 0.967 | 0.742 | +0.225 | ~0.95 |
| 50 (3-seed) | 0.913 | 0.713 | +0.200 | ~0.90 |
| **100 (3-seed)** | **0.727** | 0.617 | +0.110 | **0.823** |

N=100 per-seed grown: 0.51 / 0.81 / 0.86 (seed 42 the worst; high variance). self_replay per-seed: 0.41/0.65/0.79.
Raws: `research/findings/raw/teacher_loop_neurogenesis_N100lean_s42.json`,
`research/findings/raw/teacher_loop_neurogenesis_N100lean_s43.json`,
`research/findings/raw/teacher_loop_neurogenesis_N100lean_s44.json`.

## Read (honest, and a correction)

<!--derived-->

- **Directionally, capacity holds:** grown beats self_replay at N=20, N=50, AND N=100. The lever is not falsified.
- **But the ceiling slips with N:** grown 0.967→0.913→0.727. The capacity rise itself shrinks (+0.225→+0.110). And
  the diminishing return is compounded by an ACQUISITION deficit at N=100 (immediate-acq 0.95→0.82) — at 100
  sequential facts the shared readout struggles to even acquire cleanly, before any retention question.
- **Correction to an earlier over-claim:** the N=50 result was extrapolated as "holds → N=100 should hold"; N=100
  shows a real slip (and seed-42 alone, 0.51, briefly read as a near-collapse — the 3-seed mean 0.727 is a moderate
  degradation, not a collapse). Both the optimism and the momentary pessimism are corrected here by the 3-seed.
- **Consistent with the whole arc:** capacity was (a) a bound-trap fix (de-clamp bdsp_wmax=6→1e9, the 75%) + (b) a
  small-N capacity patch (the 25%). Neither scales to lifetime. The lifetime answer is the consolidation
  architecture — generative replay (bounded storage, PARTIAL `443351967`) with a non-forgetting generator + sparse
  replay — not a bigger reservoir.

## Rigor

3-seed at N=100 (6-seed at N=20). Same de-clamped config + per-fact recipe across scales; only n0/n-max scale.
cfg.seed byte-identical; no `sim/` edit; backend numpy. N=100 on CPU is the correct backend (GPU measured 8× slower
— tiny launch-bound net). A first N=100 attempt (full 3-arm × 6-seed) was killed as timeout-bound; this lean 2-arm
run is the confirmation.

NO-EXTERNAL-NEEDED: scaling replication; the consolidation surpass is externally grounded elsewhere (`443351967`,
van de Ven 2020).
