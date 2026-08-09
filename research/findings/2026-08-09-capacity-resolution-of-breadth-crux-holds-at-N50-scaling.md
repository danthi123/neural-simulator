---
title: "The capacity resolution of the breadth crux HOLDS at N=50 — the reservoir-size lever is stable across 2.5× scale"
date: 2026-08-09
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42, 43, 44]
---

# The capacity resolution of the teacher-loop breadth crux holds at N=50 (2.5× scale)

## Claim

The N=20 breadth-crux resolution — **de-clamp the `bdsp_wmax` bound-trap + size the DG reservoir to the fact
count** (`efdbea210`, `8ca014ff2`) — is **stable across scale**. Teaching **N=50** facts sequentially, the grown
(capacity) arm holds retention near the ceiling while the fixed-reservoir self-replay baseline degrades — the
same effect, same size, as at N=20.

## Data (de-clamped `bdsp_wmax=1e9`; same per-fact recipe at both scales)

<!--derived-->

| N | self_replay | grown (capacity) | matched_fixed | grown − self_replay | grown − matched |
|---|---|---|---|---|---|
| **20** (6-seed, `efdbea210`) | 0.742 | 0.967 | 0.917 | +0.225 | +0.050 |
| **50** (3-seed, this) | 0.713 | 0.913 | 0.953 | +0.200 | −0.040 |

Per-seed N=50 grown: 0.86 / 0.90 / 0.98; self_replay: 0.44 / 0.82 / 0.88.

Read: the capacity rise is **flat across scale** (+0.225 → +0.200), grown/matched stay near the 0.8+ ceiling
(0.91–0.95) while the fixed-reservoir baseline sits ~0.71–0.74. And **matched_fixed ≈ grown at both scales**
(|Δ| ≤ 0.05) — reconfirming the lever is reservoir **capacity**, not neurogenesis grow-as-you-go **timing**: a
fixed-large reservoir works as well as a grown one.

## Artifacts

N=50 per-seed raws (each carries the `self_replay`/`grown`/`matched_fixed` retention curves at N=50):
`research/findings/raw/teacher_loop_neurogenesis_N50_s42.json`,
`research/findings/raw/teacher_loop_neurogenesis_N50_s43.json`,
`research/findings/raw/teacher_loop_neurogenesis_N50_s44.json`.
N=20 aggregate (banked `efdbea210`): `research/findings/raw/teacher_loop_neurogenesis_AGG.json`.

## Honest scope + rigor

- **N=50 is 3-seed** (the fast scaling signal), N=20 is 6-seed. Two consistent points establish the trend; not
  yet a 6-seed N=50 headline.
- Same de-clamped config + per-fact recipe at both scales (epochs 20, replay 12×8, grow-k 4); only `n0`/`n-max`
  scale. `cfg.seed` seeds the substrate; no `sim/` edit.
- **N=100 (5×) lean confirmation is RUNNING** (`bdgp2f09i`, self_replay + grown, 3-seed, faithful settle=20). A
  first N=100 attempt at the full 3-arm × 6-seed config was killed: the ~420-neuron sim at settle=20 is ~6×
  slower than N=50 on CPU and would have timed out with no output — a COMPUTE limit (speed is secondary /
  faithful-but-slow), NOT a mechanism limit. N=1000 (the real breadth target) is a further compute step, not a
  new mechanism question.
- **What this does NOT yet claim:** that capacity-scaling holds all the way to N=1000. Two points (N=20, N=50)
  show a flat, near-ceiling trend; N=100 tests a third. If the trend ever bends, CLS systems-consolidation
  (fast fixed store → slow store via interleaved generative replay) is the researched next lever.

NO-EXTERNAL-NEEDED: scaling replication of an already-banked, externally-grounded result (`a513f118a` continual-
learning round; DG-neurogenesis/DSD-SNN recorded lane-tagged).
