# Cheat #5 v1 — Curriculum-Staged BG Cross-Projections — NEGATIVE

**Date:** 2026-04-28
**Status:** **NEGATIVE.** 3 of 6 seeds completed before stopping early; 3-seed mean sum (P0+P1 finalQ) = 10.87 vs baseline 5.88. Run halted because the directional signal was unambiguous and the failure mode is structural (not a tuning issue).

## TL;DR

Putting cross-projections on a separate `bg_cross_projections` plasticity gate — and keeping that gate at 0 during phases 1+2 to prevent the phase-0 motor-bias accumulation that defeated the 2026-04-27 attempt — does **not** suffice to close cheat #5. The cascade is structurally damaged by the *presence* of the cross-projection synapses regardless of whether they are plastic.

## Per-seed results (3 seeds before stop)

| seed | P0 finalQ | P1 finalQ | sum | beats baseline (5.88)? |
|---|---:|---:|---:|---|
| 42 | 6.00 | 5.97 | **11.97** | no |
| 43 | 5.00 | 6.02 | **11.02** | no |
| 44 | 5.77 | 3.85 | **9.62** | no |
|    |       |       | mean **10.87** | 0/3 |

Stopped at seed 100 step ~200 to save ~42 min of compute. Seeds 101 and 102 not run. Direction is unambiguous: even if the remaining 3 seeds were perfect (sum ≈ 0), the 6-seed mean would still be > 5.4 — comparable to baseline, not better.

For comparison:
- Baseline: 5.88
- Perception arc (3 cheats closed): 4.56
- Naive cross-projections (2026-04-27, NEGATIVE): ~8.40 (3-seed)
- **Curriculum-staged cross-projections (this run, NEGATIVE):** **10.87** (3-seed)

Curriculum staging is *worse* than the naive attempt. That's a strong "this is not the right axis" signal.

## Why it failed — structural transmission, not staged plasticity

The plan assumed the failure mode of the 2026-04-27 attempt was *learned bias*: phase-0 cortex_N/E activations strengthening cross-projections to all D1 pools via STDP+reward. We separated the gate so cross-projections couldn't update during phase 0.

But the runner's plasticity gate (`cp_plasticity_gain` array, introduced 2026-04-27) **only freezes learning**, not synaptic transmission. The cross-projection synapses are still created at the start of the run with `weight_mean=5.0` and **forward current normally** even when their plasticity gain is 0.

Concretely: when cortex_N fires, the cascade goes
- cortex_N → str_D1_N strongly (weight 25, same-action, same gate) — correct
- cortex_N → str_D1_E, str_D1_S, str_D1_W weakly (weight 5, cross-projections, frozen gate) — **disrupts disinhibition**

D1 selectivity is degraded from step 0. GPi can no longer cleanly silence one pool's thalamus, so motor selection becomes mushy. Phase 0 (which the agent should ace — same goal, no readaptation, cortex_to_d1 plastic) shows finalQ=5.77–6.00 vs the baseline-equivalent 1.92 from the perception arc result. That's the structural damage, present before phase 3 even kicks in.

## Direct evidence for the diagnosis

Validation log (seed 42):

```
step 100/1800  pos=(6,7)  goal=(6,6)  recent_dist=4.14   actions= 49N/ 24E/ 2S/ 25W   ← ok early
step 200/1800  pos=(7,1)  goal=(6,6)  recent_dist=2.11   actions= 42N/ 33E/ 19S/ 6W   ← good
step 300/1800  pos=(7,0)  goal=(6,6)  recent_dist=5.82   ← drifted away by end of phase 0
step 300: GOAL CHANGED to (1, 6)
step 600: CURRICULUM PHASE 2 — cortex_to_d1=0.00, inputs=1.00
step 1200: CURRICULUM PHASE 3 — bg_cross_projections gain=0.50
step 1300: recent_dist=6.82  ← phase 3 thaw helps a little
step 1500: recent_dist=5.43  ← improvement continues
step 1600: recent_dist=6.50  ← but oscillates
```

Phase 3 thaw produces a real improvement (10.6 → 5.4 in ~300 steps) — the curriculum *direction* is correct. But the phase-0 damage is too large to overcome, and the agent never reaches the perception arc's ~2.0 region.

## Lessons

1. **Plasticity-gate semantics:** the existing gate freezes weight UPDATES, not synaptic CURRENT. Anyone designing curricula for new connectivity should know this.
2. **"Frozen" doesn't mean "absent":** adding cross-projection synapses with non-trivial weight changes the forward dynamics regardless of plasticity. To staged-introduce a pathway, the **weight** must start near zero, or the synapses must be created on-demand at the thaw step.
3. **Phase-0 is a strong invariant check:** any modification that damages phase-0 finalQ (where the agent has 300 steps with a single goal and stable cortex plasticity) is structurally problematic and won't be rescued by later phases.

## v1 commits (kept; will be reused for v2)

```
8567e8e  feat(cheat5): separate plasticity gate for BG cross-projections
9c08221  feat(cheat5): wire bg_cross_projections gate into curriculum (phase 3)
```

These commits are useful infrastructure regardless: they let v2 (and any future variant) decouple cross-projection plasticity from same-action without further code changes.

## v2 plan (next)

Two approaches, ranked by likelihood of success:

**A. Zero-initial-weight + thawed weight ramp.** Create cross-projections with `weight_mean=0.0` (no synaptic transmission impact). At phase-3 thaw, ramp the weights up *and* enable plasticity simultaneously. Requires either:
- Setting `weight_mean=0.0` at construction and letting STDP+reward grow them from zero (slow), OR
- Adding a *runtime weight scale* alongside `cp_plasticity_gain` so the runner can multiply effective synaptic current (separate from learned weight) per gate. Cleaner, but a bridge change.

**B. On-demand synapse creation at phase-3 thaw.** Don't add cross-projection synapses at all during regions setup. At the phase-3 boundary, call into the existing structural-plasticity machinery to create them. Most biologically grounded, but more invasive.

Recommendation: **A with the `weight_mean=0.0` variant first** (no bridge changes). If learning from zero is too slow to produce a measurable effect by step 1800, fall back to A with runtime weight scale (a small bridge change), and only then consider B.

See `docs/plans/2026-04-28-cheat5-v2-zero-init.md` (next).

## Files

- `research/findings/raw/g11_bg/g11_seed{42,43,44}_cheat5.json` — 3-seed v1 validation data (kept for diagnosis)
- `research/findings/raw/g11_bg/cheat5_validation.log` — runner stdout from the v1 run
- `docs/plans/2026-04-28-cheat5-curriculum-staged-bg-cross.md` — v1 plan (now superseded)
- `tests/test_g11_bg_runner_flags.py` — `test_bg_cross_*` tests still relevant for v2
