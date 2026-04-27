# Plastic-Input-Layer Arc — RESOLVED via per-pathway plasticity gating

**Date:** 2026-04-27
**Status:** **GO** — closes the 7-NEGATIVE plastic-input-layer arc that ran 2026-04-26. 6-seed validation: 6/6 seeds beat baseline (sum 4.72 vs 5.88, **19.8% improvement**, p=0.0212).
**Companion:** [Drive-gated curriculum fail](2026-04-26-curriculum-fail.md), [Cortex WTA partial](2026-04-26-cortex-wta.md), [Hippocampus additive fail](2026-04-26-hippocampus-additive-fail.md)

## TL;DR

After 7 consecutive NEGATIVE attempts to add a plastic input layer to
the BG cascade (cold-start perception, informed-init, hippocampus
replacement, hippocampus additive, cortex WTA + hippo, WTA + adaDA +
hippo, drive-gated curriculum), the architecture finally accepts a
plastic input layer:

| Variant | 6-seed avg sum | std | beats baseline |
|---|---:|---:|---|
| Baseline (heuristic only) | 5.88 | 1.32 | — |
| Hippo additive (cold-start) | 10.98 | — | 0/3 |
| Real curriculum + cortex WTA | 8.87 | 2.04 | 1/6 |
| **No-WTA + real curriculum + hippo** | **4.72** | **0.78** | **6/6** |

**Statistical significance: t=-3.31, p=0.0212** (one-sample t-test
against baseline 5.88).

P0 finalQ: 2.36 (vs baseline ~3.0)
P1 finalQ: 2.36 (vs baseline ~2.9)
n_at_goal_P1: 117.8 / 1500 steps (~7.9%)

## What worked

Three things had to come together:

### 1. Per-pathway plasticity gating infrastructure (Stage 1)

Real biological brains gate plasticity differentially via neuromodulators,
critical periods, and developmental staging. We added a `plasticity_gate`
field to `RegionPathway` and a `cp_plasticity_gain` per-synapse array on
the bridge. Runners can call `bridge.set_plasticity_gate(name, value)` at
any time to freeze/thaw specific pathways. ALL plasticity paths gate by
this gain: STDP weight delta, eligibility-trace accumulation, reward
modulation update, Hebbian potentiation, Hebbian decay, synaptic scaling.

8 unit tests verify: freeze blocks all weight changes, thaw restores
plasticity, freeze→thaw→freeze cycle works, unknown gate names raise.
77/77 broader tests still pass.

### 2. Real curriculum learning (Stage 3)

Pathways tagged in `g11_bg_runner.py`:
- `cortex→D1/D2`: tagged with `plasticity_gate="cortex_to_d1"`
- `hippo→cortex`: tagged with `plasticity_gate="hippo_to_cortex"`

Runner's curriculum logic:
- **Phase 1 (steps 0–599)**: cortex_to_d1=1, hippo_to_cortex=0, hippo
  drive=0. Cortex builds D1 mapping with the heuristic. Goal flips at
  step 300 (default schedule), so cortex sees BOTH goals during this
  phase.
- **Phase 2 (steps 600–1799)**: cortex_to_d1=0, hippo_to_cortex=1, hippo
  drive=full. Cortex weights are now LOCKED. Hippo learns place→action
  given that the cascade is mature.

This is the biological developmental analog: sensory cortex matures
before association cortex (rough hippocampus analog).

### 3. NO cortex-level WTA

This was the crucial last insight. Earlier attempts ran curriculum +
cortex WTA and got 8.87 (still 1.5× worse than baseline). When we
removed WTA, sum dropped to 4.72.

WTA was actively hurting: it added the same exploitation/readaptation
penalty observed at the motor level (Session G), and the heuristic
already provides clean cortex selectivity (one pool gets 800 pA, others
0). Adding WTA on top introduced cross-pool inhibition that fights the
cascade's natural readaptation.

**The heuristic is the WTA**, in effect.

## Per-seed details

```
Curriculum + Hippo + adaDA (asym ema_neg=0.7), no cortex WTA
warmup_steps=600, default goal schedule (flip at step 300)

seed  42: P0 finalQ=1.96 P1 finalQ=1.79 sum=3.75 n_at_goal_P1=131
seed  43: P0 finalQ=2.37 P1 finalQ=3.44 sum=5.81 n_at_goal_P1= 70
seed  44: P0 finalQ=2.52 P1 finalQ=2.23 sum=4.75 n_at_goal_P1=121
seed 100: P0 finalQ=1.91 P1 finalQ=2.29 sum=4.20 n_at_goal_P1=122
seed 101: P0 finalQ=1.80 P1 finalQ=2.32 sum=4.12 n_at_goal_P1=141
seed 102: P0 finalQ=3.61 P1 finalQ=2.07 sum=5.68 n_at_goal_P1=122
                                       avg sum=4.72 (std=0.78)
```

P1 actions (e.g. seed 42: [397, 351, 322, 430]) show **strong W
preference (430)** — exactly what's needed for goal (1,6) westward
navigation. The agent is genuinely learning goal-relative actions, not
random walking.

## Recommended configuration

```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --hippocampus \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```

This config is now the default recommended setup for moving-goal
learning with plastic input layer.

## What this enables

This isn't just "better numbers". It's the first time the architecture
**actually uses plastic input neurons** to learn task-relevant
associations. The hippocampus place + goal cells genuinely learn the
place→action mapping during phase 2, biased by the heuristic teacher
during early trials.

This unlocks:
- **Truly learned perception** (next experiment): replace the heuristic
  with a plastic sensory layer, using the same curriculum technique.
- **Spatial memory persistence**: hippo weights now survive across
  trials. The agent has actual learned spatial knowledge.
- **Multi-region developmental sequencing**: the gate infrastructure
  generalizes to any pathway, enabling complex maturation schedules.
- **Sleep/replay experiments**: with mature hippo weights, we can now
  test memory consolidation via offline replay.

## Files

- `sim/regions.py:128`: `RegionPathway.plasticity_gate` field
- `sim/regions.py:328`: wiring plan emits gate metadata
- `sim/bridge.py:248-260`: `cp_plasticity_gain` allocation in `__init__`
- `sim/bridge.py:1681-1763`: gate setup in `inject_explicit_wiring`
- `sim/bridge.py:1808-1860`: bridge methods (set/get/list/count)
- `sim/bridge.py:4100-4115`: Hebbian gating
- `sim/bridge.py:4185-4210`: STDP gating
- `sim/bridge.py:4248-4265`: reward modulation gating
- `sim/bridge.py:4500-4510`: synaptic scaling gating
- `tests/test_regions.py:454-700`: 8 unit tests for gating
- `research/runners/g11_bg_runner.py:300-340`: pathway tags
- `research/runners/g11_bg_runner.py:790-895`: curriculum gate logic
- `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_nowta.json`: 6-seed validation

## Context: the 7-attempt arc

| Attempt | Mechanism | Sum (3-seed) | Status |
|---|---|---:|---|
| 1 (2026-04-26) | Cold-start learned perception | — | NEGATIVE (cascade silenced) |
| 2 (2026-04-26) | Informed-init perception | 12.09 | NEGATIVE |
| 3 (2026-04-26) | Hippocampus replacement | — | NEGATIVE (cascade silenced) |
| 4 (2026-04-26) | Hippocampus additive | 10.98 | NEGATIVE |
| 5 (2026-04-26) | Cortex WTA + hippo | 9.26 | PARTIAL |
| 6 (2026-04-26) | WTA + adaDA + hippo | 8.01 | PARTIAL |
| 7 (2026-04-26) | Drive-gated curriculum | 10.25 | NEGATIVE |
| 8 (2026-04-27) | Real curriculum + WTA + adaDA + hippo | 8.87 (6s) | PARTIAL |
| 9 (2026-04-27) | **No-WTA + real curriculum + adaDA + hippo** | **4.72 (6s)** | **GO** |

Each negative result narrowed the search space. The breakthrough came
from realizing:
- (4-7) Plastic random weights destabilize the cascade — addressed by
  curriculum (freeze cortex, then add plastic inputs)
- (5-8) WTA isn't needed — the heuristic already provides selectivity
- (Drive gating in 7) doesn't actually freeze plasticity — needed
  per-pathway gain infrastructure

## Architectural lesson

The BG cascade has multiple sources of cortex selectivity:
1. **The heuristic** provides 1-of-4 input asymmetry directly
2. **WTA** would also provide it, but at cost of commitment

Once you have the heuristic, you don't need WTA. WTA was a fix for a
problem that didn't exist when the heuristic was on. Removing it freed
the cascade to readapt cleanly.

The plastic-input-layer cold-start was a real problem, but solvable with
proper staged plasticity (curriculum + per-pathway gates), not with
runner-side flags.

## Next step

The natural extension: truly **learned perception**. The heuristic in
this architecture is the part that's NOT plastic — we hand-coded the
direction-to-cortex-pool mapping. Now that curriculum+plastic-input
works, replace the heuristic with a plastic sensory→cortex layer using
the same technique:
- Phase 1: heuristic + sensory layer driven, sensory→cortex frozen,
  cortex→D1 plastic (cortex matures)
- Phase 2: heuristic OFF, sensory→cortex plastic, cortex→D1 frozen
  (sensory learns)

If this works, we have a fully learned perception-action loop with no
hand-coded shortcuts. That's the next major arc.
