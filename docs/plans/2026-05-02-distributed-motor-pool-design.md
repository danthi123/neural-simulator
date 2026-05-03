# 2026-05-02 — Distributed motor pool architecture (Pulvermüller G.20)

After 9 negative architectural followups, the 28.5% W→A ceiling is robust
under labeled-line motor architecture. This doc designs the full distributed-
motor-pool rebuild — the closest test of Pulvermüller's distributed
action-word coding (G.20) we can do within the BrainRegion framework.

## Biology source

Pulvermüller F. (1999, 2005): action-word neurons in motor cortex are
DISTRIBUTED across overlapping ensembles, not labeled-line pools.
Hauk-Johnsrude-Pulvermüller (2004) shows somatotopic mapping of action
verbs onto motor cortex with smooth tuning curves, not discrete pools.

Penfield 1937 homunculus: body-part representations have FUZZY boundaries.
Real M1 neurons have cosine tuning curves over movement direction
(Georgopoulos 1986).

## Current architecture (and why it ceiling at 28.5%)

```
4 motor_X pools (10 neurons each, 40 total)
- motor_N: 10 neurons, all "north" labeled
- motor_E: 10 neurons, all "east" labeled
- motor_S: 10 neurons, all "south" labeled
- motor_W: 10 neurons, all "west" labeled

Cluster A pathway: thal_X -> motor_X (per-action, separate channels)
Language pathway: lang_input -> motor_X (PFC bypass, separate per direction)

W->A eval: argmax over 4 pool firing rates
```

Empirical: W->A 28.5% (p=0.027) ceiling. 9 architectural variations all
NEGATIVE (reward, drives, sizes, durations, decoders, cross-coupling).

The fundamental issue: STDP+reward sees 4 OPTHOGONAL motor pools with no
representational redundancy. There's no gradient to climb beyond pool-level
selection — each direction either fires its labeled pool or doesn't.

## New architecture: 8 sub-pools at 45° intervals

```
8 motor_pop_θ sub-pools (5 neurons each, 40 total)
- motor_pop_E   (preferred θ = 0°)     5 neurons
- motor_pop_NE  (preferred θ = 45°)    5 neurons
- motor_pop_N   (preferred θ = 90°)    5 neurons
- motor_pop_NW  (preferred θ = 135°)   5 neurons
- motor_pop_W   (preferred θ = 180°)   5 neurons
- motor_pop_SW  (preferred θ = 225°)   5 neurons
- motor_pop_S   (preferred θ = 270°)   5 neurons
- motor_pop_SE  (preferred θ = 315°)   5 neurons
```

Same total neurons (40). 8 sub-pools approximate continuous tuning curves
(real M1 neurons have cosine tuning; 8 steps is reasonable discretization).

## Cosine-tuned pathways

For each input source X with preferred direction θ_X (cortex_X, thal_X,
language_input "north" pattern, etc.), the connection to motor_pop_θ has
weight scaled by:
```
weight(X -> motor_pop_θ) = max(0, cos(θ_X - θ)) * base_weight
```

**Example: thal_N (90°) → motor_pop sub-pools:**
| sub-pool | preferred θ | cos(90 - θ) | actual weight (if base=10) |
|---|---|---|---|
| motor_pop_E | 0° | 0.0 | 0 (no pathway) |
| motor_pop_NE | 45° | 0.707 | 7.07 |
| motor_pop_N | 90° | 1.0 | 10.0 |
| motor_pop_NW | 135° | 0.707 | 7.07 |
| motor_pop_W | 180° | -1.0 → 0 | 0 |
| motor_pop_SW | 225° | -0.707 → 0 | 0 |
| motor_pop_S | 270° | -1.0 → 0 | 0 |
| motor_pop_SE | 315° | 0.0 | 0 |

So thal_N drives 3 sub-pools (N + 2 adjacent at 45°). Smooth tuning across
the population.

## Action selection / readout

Population vector projection onto cardinal directions:

```
N_score = sum_θ firing(motor_pop_θ) * max(0, cos(θ - 90°))
        = motor_pop_N * 1.0 + (motor_pop_NE + motor_pop_NW) * 0.707
E_score = sum_θ firing(motor_pop_θ) * max(0, cos(θ - 0°))
        = motor_pop_E * 1.0 + (motor_pop_NE + motor_pop_SE) * 0.707
S_score = motor_pop_S * 1.0 + (motor_pop_SE + motor_pop_SW) * 0.707
W_score = motor_pop_W * 1.0 + (motor_pop_NW + motor_pop_SW) * 0.707
```

Action = argmax(N_score, E_score, S_score, W_score).

Mathematically equivalent to population vector decoding (Georgopoulos 1986).

## Why this might break the ceiling

The labeled-line architecture has a HARD CEILING because:
- 4 pools = 4 possible "states" of the motor output
- argmax over 4 = at most log2(4) = 2 bits of information
- Per-trial baseline noise (typical std ~5 spikes) is comparable to signal
  differences between pools (typical delta ~5 spikes)

The distributed architecture:
- 8 sub-pools = at most log2(8) = 3 bits of "raw" information
- BUT key benefit: smooth tuning means even noisy spike counts can be
  averaged via population vector to give continuous direction estimate
- Adjacent sub-pools have correlated activity (shared inputs from cortex
  via cosine tuning), so noise averages OUT
- STDP can sculpt FINER selectivity within sub-pools (some neurons in
  motor_pop_NE develop stronger N-affinity, others stronger E-affinity)

Real biology: motor cortex with population vectors achieves MUCH better
discrimination than sparse labeled pools (~degrees of accuracy in arm
reaching tasks vs our 4-direction discrete).

## Implementation plan

1. **Add `enable_distributed_motor_pop` flag** to `build_bg_brain_regions`
2. **When enabled, skip 4 motor_X regions; create 8 motor_pop_θ regions**
3. **Replace pathways:**
   - thal_X → motor_X (cluster A) → thal_X → motor_pop_θ with cosine-weighted
     density / weight per sub-pool
   - language_input → motor_X (PFC-bypass) → language_input → motor_pop_θ
     all-to-all (8 sub-pools), plastic; STDP+reward learns the tuning
4. **Skip motor_FS lateral inhibition** when distributed motor enabled
   (incompatible; would need separate FS per sub-pool — defer to later)
5. **Action selection (cascade)** still reads cortex_X — no change for training
6. **W→A eval methodology** updated:
   - Read 8 motor_pop_θ firing rates
   - Decode via population vector: 4 cardinal scores
   - argmax over scores

## Backwards compatibility

- Default `enable_distributed_motor_pop = False` — all existing experiments
  unchanged
- v2 baseline (28.5% W→A) preserved as comparison reference
- K v2 visual cortex flagship doesn't enable text I/O so unaffected
- Cluster A closed loop (cortex → stn → thal → cortex) only minimally
  affected (just the thal → motor portion changes)

## Test plan

1. Smoke test (5 ep, n=20 eval) — verify no crashes
2. Full test seed=42, 100 ep — compare to v2 baseline
3. If positive, multi-seed validation
4. If negative, document and decide if 16 sub-pools (22.5° intervals) is
   worth trying

## Estimated effort

~3-5 hours implementation:
- 30 min: BrainRegion creation logic for 8 sub-pools
- 90 min: cosine-tuned pathway construction
- 30 min: text_eval population vector decoding
- 30 min: smoke test + debugging
- 60 min: full validation run (~75 min wall clock)

## Risks

- Cluster A path goes through thal → motor; with new motor_pop sub-pools,
  thal_X drives only motor_pop_θ where cos>0. Net cluster A activity may
  differ.
- Action selection currently reads cortex_X — that path doesn't change.
  But IF eval mismatches (cortex_X argmax vs motor_pop population vector),
  we get inconsistent signal.
- Skipping motor_FS lateral inhibition: shouldn't matter (it's opt-in,
  off by default in text training).

## Files to modify

- `research/runners/g11_bg_runner.py` — region + pathway construction
- `research/runners/text_eval.py` — population vector decoding for W→A
- `research/runners/text_train_embodied.py` — pass flag through
- `research/runners/text_eval_embodied.py` — CLI flag
