# Asymmetric Adaptive DA — Closes Phase 1 Gap, New Best

**Date:** 2026-04-26
**Status:** **CORRIGENDUM**: 3-seed claim of 33% improvement was overstated. See [6-seed correction](2026-04-26-six-seed-correction.md) — true 6-seed result is 5.23 ± 1.90 vs baseline 5.88 (11% improvement, t=0.64, NOT significant). Mechanism is highly seed-dependent. Use `--surprise-lr-boost` instead for reliable improvement.
**Original status (kept for trail):** GO. Best total finalQ measured (3.53 vs baseline 5.24, -33%). Phase 1 gap nearly closed.
**Companion:** [Adaptive DA (symmetric)](2026-04-26-adaptive-da-targeting.md), [Phase B real-win](2026-04-25-phase-b-acid-test-real-win.md), [6-seed correction](2026-04-26-six-seed-correction.md)

## TL;DR

Asymmetric reward-EMA ramp: slow positive decay (tau~10), fast negative decay (tau~3). Models phasic dopamine biology — DA neurons dip faster on negative reward-prediction-error than they ramp up on positive (Schultz 1998).

3-seed acid test results across all sharpening variants:

| Variant | P0 finalQ | P1 finalQ | Sum |
|---|---:|---:|---:|
| Baseline (no sharpening) | 3.48 | **1.76** | 5.24 |
| WTA (lateral inhibition) | 2.40 | 2.46 | 4.86 |
| Hard per-action DA | 2.04 | 2.61 | 4.65 |
| Adaptive DA (sym, tau~10) | 1.85 | 2.14 | 3.99 |
| Adaptive DA (sym, tau~3) | 2.19 | 2.13 | 4.33 |
| WTA + adaptive DA (sym) | 2.23 | 2.18 | 4.41 |
| **Adaptive DA (asym, slow+/fast-)** | **1.61** | **1.92** | **3.53** |

Asymmetric DA wins on both:
- Phase 0: -54% vs baseline (best phase 0 measured)
- Phase 1: only +9% vs baseline (essentially closes the readaptation gap)
- Sum: -33% vs baseline, -11% vs symmetric tau~10

## Mechanism

```python
# Per-trial reward EMA update
if reward < 0:
    decay = adaptive_da_ema_decay_negative  # 0.7, tau~3 trials
else:
    decay = adaptive_da_ema_decay            # 0.9, tau~10 trials
reward_ema = decay * reward_ema + (1 - decay) * reward

# Linear map [-1, +1] → [0, 1]
gating_strength = max(0.0, min(1.0, (reward_ema + 1.0) / 2.0))
scale = 1.0 - gating_strength
eligibility[non_selected_d1_synapses] *= scale
```

The asymmetry creates a "ratchet" dynamic:
- Positive reward → reward_ema climbs slowly toward +1 (sustained good performance commits gradually)
- Negative reward → reward_ema crashes quickly toward -1 (one bad outcome triggers fast exploration)

This matches the phasic DA biology more closely than symmetric ramps: DA neurons fire transient bursts to unexpected positive reward (small ramp up) and a deeper, faster pause to unexpected negative reward (sharp drop).

## Per-seed results

| Seed | P0 finalQ | P1 finalQ |
|------|---:|---:|
| 42   | 1.68 | 1.95 |
| 43   | 1.43 | 2.00 |
| 44   | 1.72 | 1.80 |
| **avg** | **1.61** | **1.92** |

Lower variance than symmetric variants. All seeds reach phase 1 finalQ < 2.0 — agent stays at Manhattan distance ~1.92 from goal in steady state.

## Comparison with all variants

The full landscape now:

```
                    P0       P1       Sum
baseline         3.48     1.76     5.24
WTA              2.40     2.46     4.86
hard DA          2.04     2.61     4.65
adaDA tau~10     1.85     2.14     3.99
adaDA tau~3      2.19     2.13     4.33
WTA+adaDA        2.23     2.18     4.41
adaDA asym       1.61     1.92     3.53  ← new best
```

## Why this works

The fundamental tension: sharpening helps acquisition (commit faster) but hurts readaptation (locked in old policy when world changes).

Symmetric adaptive DA (tau~10) handles this OK but lags:
- Phase 1 onset: reward drops, but EMA still high (~0.7) for ~5 trials
- Those 5 trials use exploit-mode gating, missing exploration window

Symmetric tau~3 reacts faster but is too noisy:
- Single -1 reward in phase 0 drops EMA quickly → premature exploration
- Phase 0 acquisition slowed

Asymmetric: get both:
- During phase 0: positive reward dominant, EMA climbs slow, gating ramps up gradually (good acquisition)
- At phase change: -1 rewards crash EMA fast (~3 trials), gating relaxes immediately (good adaptation)
- During phase 1 stable state: EMA recovers slowly, gating returns to high (exploit found policy)

## Decision

Promote `--adaptive-da --adaptive-da-ema-decay-negative 0.7` as the **recommended** Phase B sharpening configuration. Document in CLAUDE.md and SCIENCE_ROADMAP.

Keep the legacy baseline (no sharpening) as default for backward compatibility, but mark adaptive DA asymmetric as the recommended setting for moving-goal scenarios.

## Files

- `research/runners/g11_bg_runner.py:540-595`: asymmetric ramp implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_adaDA_asym.json`: 3-seed acid test data

## Next experiments

1. **Try WTA + asymmetric adaptive DA** — earlier WTA+sym-adaDA was worse than sym-adaDA alone (4.41 vs 3.99). Maybe asymmetric DA's faster relaxation lets WTA contribute without the lock-in penalty.
2. **DA-gated WTA**: extend asymmetric DA to also modulate FS-motor inhibition strength. WTA strong when reward is high (commit), weak when reward drops (explore). Direct test of the user's "DA gate" question.
3. **Pivot to #3 real position encoding** if compositional gains plateau — the heuristic cortex drive remains the most obvious upstream cheat.

## Lesson

Bio-inspired asymmetry is a free win when the underlying signal naturally has different-valenced importance. The DA system handles positive and negative reward differently for evolutionary reasons; encoding that difference in the algorithm captured a 0.46-finalQ improvement (3.99 → 3.53) at the cost of one extra `if` statement.
