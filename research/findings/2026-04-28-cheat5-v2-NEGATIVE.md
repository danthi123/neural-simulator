# Cheat #5 v2 — Zero-Init Cross-Projections — NEGATIVE

**Date:** 2026-04-28
**Status:** **NEGATIVE.** 3-seed smoke mean sum 7.89 vs baseline 5.88. Phase 0 structurally fixed (mean 2.49) but phase 1 destroyed by thaw-time STDP corruption (mean 5.40). Proceeding to v3 (add MSN lateral inhibition before re-attempting cross-projections).

## Per-seed results

| seed | P0 finalQ | P1 finalQ | sum |
|---|---:|---:|---:|
| 42 | 1.64 | 6.49 | 8.13 |
| 43 | 1.44 | 6.37 | 7.81 |
| 44 | 4.40 | 3.33 | 7.73 |
| **mean** | **2.49** | **5.40** | **7.89** |

For comparison:
- Flagship (no cross-projections): mean 4.08, P0 ~1.75, P1 ~1.95
- v1 (weight=5, frozen): mean 10.87, P0 ~5.92, P1 ~5.28
- **v2 (weight=0, thaw at 1200): mean 7.89, P0 ~2.49, P1 ~5.40**
- Baseline: mean 5.88

## What v2 fixed

**Phase 0 is now structurally fine.** Initializing cross-projections at `weight_mean=0.0` means they have zero forward effect during phases 1+2, so the BG cascade behaves identically to the flagship (no cross-projections) until the phase-3 thaw at step 1200. Seed 42 P0=1.64, seed 43 P0=1.44 — both within flagship range. v1's structural-damage failure mode is closed.

(Seed 44 P0=4.40 is an outlier; likely seed-specific variance, not a v2 issue.)

## What v2 didn't fix

**Phase 1 is destroyed by thaw-time STDP corruption.** Looking at seed 42's mid-run trace:
```
step 1100: pos=(1,5) goal=(1,6) recent_dist=1.42  ← agent has converged on goal
step 1200: PHASE 3 — bg_cross_projections gain=0.50
step 1300: recent_dist=3.36                       ← starting to drift
step 1400: pos=(7,0) goal=(1,6) recent_dist=8.02  ← agent now at the wrong corner
```

The agent had cleanly learned phase 1 by step 1100. Phase-3 thaw lets cross-projections start learning from zero. STDP+reward picks up cross-projection patterns (cortex_X firing → str_D1_Y firing pairs) that don't reflect the agent's actual policy — the cross-correlations are noisy in a converged BG cascade. The new cross-projection weights then START driving wrong actions, which feeds back into more spurious learning. Result: rapid divergence.

This is a **fundamental issue with the approach**, not a tuning problem:
- Smaller `--bg-cross-phase3-gain` would slow the corruption but not stop it
- Later thaw step would just delay the disruption

## Diagnosis: missing biology

Real BG handles cross-projections cleanly because of multiple winner-take-all mechanisms our simulator doesn't fully model:

| Mechanism | Real BG | Our simulator |
|---|---|---|
| MSN-MSN lateral inhibition (within pool) | Strong; collaterals suppress competing actions | Sparse, not biological strength |
| MSN cross-pool inhibition (between pools) | Strong; str_D1_N → str_D1_E inhibitory | **Absent** |
| FS interneurons in striatum | Mediate strong feed-forward inhibition | Not modeled |
| Center-surround in pallidum | Sharpens contrast | Partial |

When the simulator has cross-projections AND lacks the lateral inhibition that would normally suppress cross-talk, the cascade becomes too noisy for clean STDP learning.

This means cheat #5 is real and closeable, but the **prerequisite** is adding the missing winner-take-all biology. v3 plan addresses this.

## v3 plan

See [`docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md`](../../docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md).

Two-stage:
1. **v3 (independent of cheat #5):** add MSN cross-pool lateral inhibition. Validate it doesn't regress flagship (sum ≤ 4.08, P0+P1 stable). Worth doing on its own — biology improvement.
2. **v3.1:** re-attempt cross-projections on top of lateral inhibition. Validate cheat #5 closure with 6 seeds.

## Files

- `research/findings/raw/g11_bg/g11_seed{42,43,44}_cheat5v2.json` — 3-seed v2 data
- `research/findings/raw/g11_bg/cheat5v2_smoke.log` — runner stdout
- `docs/plans/2026-04-28-cheat5-v2-zero-init.md` — v2 plan (now superseded)

## Lessons

1. **Plasticity gate semantics matters** (v1 lesson, kept here for trail): the gate freezes weight UPDATES, not synaptic CURRENT. Zero-init synapses or runtime weight scale are required for safe staged introduction.
2. **STDP needs clean firing patterns to learn well.** A cascade with cross-talk produces noisy spike correlations; STDP learning from those produces noise-amplifying weight changes. Lateral inhibition is the prerequisite for clean STDP in any cross-pool learning task.
3. **Cheat #5 is decomposable.** It's not "add cross-projections" — it's "add the WTA biology that makes cross-projections safe". Decomposing into v3 (lateral inhibition) and v3.1 (cross-projections on top) lets us validate each piece independently.
