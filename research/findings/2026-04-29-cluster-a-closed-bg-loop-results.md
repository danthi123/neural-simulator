---
type: finding
status: qualified
date: 2026-04-29
---

# Cluster A — Closed BG Loop Results

**Date:** 2026-04-29
**Status:** PENDING (eval running)
**Plan:** [`docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md`](../../docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md)
**Implementation commit:** `2d8be00`

## TL;DR

> _Filled in once 6-run cheat-5 eval completes._

## Hypothesis

Per the 2026-04-28 cheat-5 reframe, cross-projections need a closed BG loop to behaviorally pay off. The closed thalamo-cortical loop provides:
1. **Hyperdirect (cortex_X → STN):** fast global "stop" signal so action selection settles before BG output commits.
2. **Thal_X → cortex_X feedback:** the post-synaptic activity that lets STDP shape useful cross-action weights — the "teaching signal" missing for cross-projection learning.

Cluster A is the smallest possible addition to test this hypothesis: 8 new pathways (4 hyperdirect + 4 closed-loop), all static, action-specific.

## Implementation summary

Two pathway groups added (opt-in `--enable-cluster-a-closed-loop`):

| Pathway | Density | Weight | Plastic | Biology |
|---|---|---|---|---|
| cortex_X → stn (×4) | 0.10 | 3.0 | False | Nambu 2002, ~30% of cortex pyramids |
| thal_X → cortex_X (×4) | 0.50 | 5.0 | False | VA/VL → motor/premotor; topographic |
| thal_X → cortex_Y (Y≠X) | — | — | — | NOT added; biology is action-specific |

Synapse-count smoke (50 steps, full Cluster B + Cluster A): 32839 (was 32122 without Cluster A; +717).

## Cheat-5 multi-goal re-eval (n=3)

Compared post-remediation flagship config WITH and WITHOUT Cluster A. Both runs use:
- `--bg-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-fsis`
- `--goal-schedule multi`
- 1800 steps, seeds 42, 43, 44

### Baseline (no Cluster A)

| Seed | P0 | P1 | P2 | P3 | Sum |
|------|----|----|----|----|-----|
| 42 | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |
| 43 | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |
| 44 | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |
| **Mean ± std** | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |

### + Cluster A (`--enable-cluster-a-closed-loop`)

| Seed | P0 | P1 | P2 | P3 | Sum |
|------|----|----|----|----|-----|
| 42 | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |
| 43 | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |
| 44 | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |
| **Mean ± std** | _PEND_ | _PEND_ | _PEND_ | _PEND_ | _PEND_ |

### Direct comparison

| | No-A | +A | Δ |
|---|---|---|---|
| Mean | _PEND_ | _PEND_ | _PEND_ |
| Std | _PEND_ | _PEND_ | _PEND_ |

## Decision matrix

| Δ Mean | Δ Std | Verdict |
|---|---|---|
| ≥ −1.0 | ≤ baseline | **GO** — tier-3 (6-seed) validation next |
| −1.0 to 0.0 | any | **PARTIAL** — possibly add Cluster C (DA-system completeness) before tier-3 |
| > 0.0 | any | **NO-GO** — closed loop doesn't help in current parameters; tune weights or revert |

## Discussion

> _Filled in once data lands._

## Files

- Plan: `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md`
- Implementation: commit `2d8be00`
- 6-run eval outputs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_clusterB_postR.json` (no-A baseline) and `g11_seed{42,43,44}_clusterA.json` (+A)
- Logs: `research/findings/raw/g11_bg/clusterB3_eval_logs/seed{42,43,44}_{postR,clusterA}.log`
