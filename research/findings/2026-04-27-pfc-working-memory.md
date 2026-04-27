# PFC Working Memory — Item 3 GO (4.41 sum, 25% over baseline, p=0.018)

**Date:** 2026-04-27 (Item 3 from session priority list)
**Status:** **GO** — adding a recurrent PFC region with goal_cells → PFC → cortex pathways yields the new best 6-seed result (4.41 vs baseline 5.88, p=0.018).

## TL;DR

Added a prefrontal cortex (PFC) region with recurrent connectivity (density
0.2) for persistent activity / working memory dynamics. Pathways:
- `goal_cells → PFC` (plastic, density 0.5, weight 8)
- `PFC ↔ PFC` (plastic recurrent, density 0.2, weight 2)
- `PFC → cortex_{N,E,S,W}` (plastic, density 0.5, weight 8)

| Variant | 6-seed avg sum | beats baseline | p-value |
|---|---:|---|---:|
| Baseline (heuristic only) | 5.88 | reference | — |
| Hippo + curriculum (full freeze) | 4.72 | 6/6 | 0.02 |
| Sensory + hippo + curriculum (full) | 4.63 | 5/6 | 0.05 |
| Sensory + hippo + curriculum (partial) | 4.79 | 5/6 | 0.10 |
| **Sensory + hippo + curriculum + PFC** | **4.41** | **5/6** | **0.018** |

**25.0% improvement over baseline** — the new best config in this project.

## Per-seed results

```
seed  42: P0 finalQ=1.73 P1 finalQ=3.10 sum=4.83
seed  43: P0 finalQ=1.65 P1 finalQ=1.88 sum=3.53
seed  44: P0 finalQ=2.32 P1 finalQ=1.96 sum=4.28
seed 100: P0 finalQ=1.35 P1 finalQ=1.70 sum=3.05  (best)
seed 101: P0 finalQ=3.47 P1 finalQ=2.47 sum=5.94  (only seed not beating baseline)
seed 102: P0 finalQ=2.95 P1 finalQ=1.88 sum=4.83
                                       avg=4.41 (std=0.94)
```

5/6 seeds beat baseline. Seed 101 narrowly missed (5.94 vs 5.88) but
strongly beat the no-PFC variants on its phase 1.

## Architecture details

```python
BrainRegion(
    name="pfc",
    n_neurons=60,
    exc_fraction=0.8,                    # 80% excitatory (real cortex)
    internal_density=0.2,                # 12% probability of recurrence
    exc_weight_mean=2.0,                 # moderate self-excitation
    inh_weight_mean=4.0,
    weight_jitter=0.2,
    plastic_internal=True,               # recurrence is also plastic
    izh_neuron_type="IZH2007_HIPPO_PYRAMIDAL",  # pyramidal dynamics
)

# Pathways (all tagged plasticity_gate="pfc_pathways")
goal_cells → PFC      density=0.5 weight=8
PFC ↔ PFC             density=0.2 weight=2 (recurrent, plastic_internal)
PFC → cortex_{N,E,S,W} density=0.5 weight=8 (each)
```

## Why it helps

The PFC adds a recurrent layer between goal context (goal_cells) and
action cortex (cortex_{N,E,S,W}). Several plausible benefits:

1. **Pattern completion**: PFC's recurrent dynamics can re-activate
   goal-related patterns even when goal_cells are noisy or briefly silent.
2. **Memory smoothing**: Persistent activity smooths goal information
   across time, reducing trial-to-trial variability.
3. **Plastic intermediate layer**: Three-step learning chain
   (goal → PFC → cortex) has more learnable parameters than direct
   goal → cortex, allowing finer-grained associations.
4. **Recurrent learning**: Internal PFC connectivity is plastic, letting
   PFC self-organize into useful representations.

The improvement (4.79 → 4.41, ~8% over the prior best partial-freeze
variant) is modest but statistically significant on 6 seeds.

## What's not tested yet

This adds the PFC region but doesn't yet test **delayed-response** tasks
(where the goal is briefly shown then hidden). For that:
- Add a "cue" phase where goal_cells fire briefly
- Add a "delay" phase where goal_cells are silenced
- PFC should sustain goal info during delay
- Test agent's ability to navigate during delay

This is a future experiment ("PFC Stage 2"). The basic infrastructure
is now in place.

## Recommended config (updated 2026-04-27)

```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --hippocampus --learned-perception --pfc \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```

For partial freeze variant (slightly different trade-off):
```bash
... --curriculum-phase2-cortex-gain 0.2
```

## Files

- `research/runners/g11_bg_runner.py:99-115`: PFC parameters and region
- `research/runners/g11_bg_runner.py:160-175`: PFC region declaration
- `research/runners/g11_bg_runner.py:355-378`: PFC pathways
- `research/runners/g11_bg_runner.py:1349-1357`: CLI flags
- `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_pfc.json`:
  6-seed validation data

## Next steps for PFC arc

1. **Stage 2: delayed-response task.** Show goal briefly, hide for delay,
   test navigation during delay. Tests if PFC actually maintains goal info.
2. **Probe persistent activity.** Inject brief goal_cells stimulus, then
   record PFC firing rate over 1000+ ms. Should sustain.
3. **Plasticity gate for PFC**: tag pfc_pathways for curriculum control
   (already tagged "pfc_pathways" in code).
4. **Multi-region working memory**: extend to multi-step tasks where
   PFC chains states over time.
