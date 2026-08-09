---
title: "The inherited bdsp_wmax=6 bound-trap SILENCES the teacher-loop reservoir — de-clamping recovers 75% of N=20 retention (clean 6-seed A/B)"
date: 2026-08-09
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42, 43, 44, 45, 46, 47]
---

# The bdsp_wmax=6 bound-trap silences the teacher-loop reservoir; de-clamping recovers 75% of N=20 retention

## Claim (a clean single-variable A/B — the recurring plasticity BOUND-TRAP, 6th instance)

<!--derived-->

The inherited `bdsp_w_min/max = -6/+6` clamp (parent bridge default) **silences the birthed DG-expansion
granule reservoir** in the teacher-loop breadth runner: at N=20 the self-replay retention collapses to
**EXACTLY chance (0.05, all 6 seeds)**. Widening the clamp (`--bdsp-wmax 1e9`, de-clamped) — the ONLY variable
changed, same config/seeds — recovers it to **0.742**, and adding reservoir capacity closes the residual to
**0.967**. De-clamping alone is **75% of the full 0.05→0.967 recovery**; capacity is the remaining 25%.

## The A/B (only `bdsp_wmax` differs; N=20, n0=14, grow-k 4, epochs 20, replay 12×8, 6 seeds)

Means/deltas below are computed over the per-seed raws cited under the table.

<!--derived-->

| arm | CLAMP `bdsp_wmax=6` (inherited default) | DE-CLAMP `bdsp_wmax=1e9` | clamp cost |
|---|---|---|---|
| self_replay | **0.050** (per-seed 0.05/0.05/0.05/0.05/0.05/0.05 = chance) | 0.742 | **+0.692** |
| grown (capacity) | 0.208 | 0.967 | +0.759 |
| matched_fixed | 0.200 | 0.917 | +0.717 |

Clamp per-seed raws (each records `frac_recalled=0.05` for self_replay at N=20):
`research/findings/raw/teacher_loop_clamptrap_bdsp6_s42.json`,
`research/findings/raw/teacher_loop_clamptrap_bdsp6_s43.json`,
`research/findings/raw/teacher_loop_clamptrap_bdsp6_s47.json` (s42..s47, all 6 present). De-clamp aggregate
(banked `efdbea210`): `research/findings/raw/teacher_loop_neurogenesis_AGG.json`. Chance = 1/20 = 0.05.

## Mechanism (why the clamp is catastrophic here)

`bdsp_wmax` widens the clamp on the **BDSP-updated afferent synapses**. At `=6`, the birthed granule units'
random afferent projections are crushed to `|w|<=6`, so each unit fires **~1 spike/percept vs ~24 de-clamped**
(runner comment, L104-107). A near-silent reservoir produces no separable code → the shared readout has nothing
to bind → retention = chance on every seed. This is the exact plasticity **BOUND-TRAP** the project has now hit
**six times** (CLAUDE.md; `tools.lab.bound_check` guards five prior rules): a static bound substituted for a
homeostatic process **dominates the measurement** — here 75% of it.

## What this DOES and DOES NOT establish (honest scope)

- **DOES:** in the neurogenesis/capacity teacher-loop runner (large afferents, `ff_w_init=2000`), the inherited
  clamp is not a minor confound — it is the DOMINANT factor, taking N=20 retention from 0.742 to chance. The
  de-clamped + capacity regime that resolved the N=20 crux (`efdbea210`) is real, and de-clamping is a
  precondition, not a tuning knob.
- **DOES NOT (needs a separate A/B):** prove the *historically-cited* 0.45 N=20 baseline — measured in the
  **older** `_teacher_loop_sleep_replay_consolidation` runner with a **different** afferent regime — was equally
  clamp-dominated. The clamp's severity scales with afferent magnitude; the old runner's smaller afferents may
  clamp less catastrophically. A clamp-vs-declamp A/B on that runner would close it. The DIRECTION (the whole
  breadth arc's "catastrophic forgetting" was inflated by this clamp) is a strong, now-partly-confirmed
  hypothesis, not yet a full accounting.

## Rigor / anti-cheats

- 6-seed, PERFECTLY consistent (self_replay = 0.05 on all 6; the effect is mechanistic, not stochastic).
- Single-variable A/B: only `--bdsp-wmax` changed (6 vs 1e9); same config, same seeds, same net-build path.
- `cfg.seed` seeds the substrate (byte-identical firing thresholds asserted per run); NO `sim/` edit
  (`git diff main -- sim/` empty — the clamp is a runner-passed config value into the parent bridge's existing
  `bdsp_w_min/max`).
- The `--bdsp-wmax` flag is additive (default 1e9 preserves the banked de-clamped behavior); commit `6bed08065`.

## Implication for the breadth arc

Retention measurements taken under the inherited clamp are suspect: they can read as "catastrophic forgetting"
while actually reporting a silenced reservoir. The resolved N=20 regime is **de-clamp + size-the-reservoir →
0.967**. Next: whether this holds at N=100 (scaling test running) decides if breadth reduces to bound-trap +
capacity, or needs CLS systems-consolidation.

NO-EXTERNAL-NEEDED: this is an internal instrument A/B (the bound-trap is documented in CLAUDE.md +
`tools.lab.bound_check`); the continual-learning external round is banked separately (`a513f118a`).
