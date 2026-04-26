# Hippocampal Module (Additive) — Doesn't Help, Same Cold-Start Pattern

**Date:** 2026-04-26 (post-Pavlovian, after informed-init failure)
**Status:** NEGATIVE — adding plastic place + goal cells with random weights on top of working heuristic *degrades* performance ~2x. Same cold-start failure as the learned-perception arc.
**Companion:** [Informed-init perception fail](2026-04-26-informed-init-perception-fail.md), [Learned perception cold-start fail](2026-04-26-learned-perception-cold-start-fail.md), [Pavlovian demo](2026-04-26-pavlovian-demo.md)

## TL;DR

Hypothesis: add a hippocampal module (place cells + goal cells, sparse Gaussian σ=0.5) that projects plastically to cortex *on top of* the heuristic drive. Real biology has hippocampus augmenting cortex, not replacing it — so the additive form should keep the cascade working and let plasticity build a memory layer.

| Variant | Sum (3-seed avg) | vs. Baseline |
|---|---:|---|
| Baseline (heuristic only) | **5.88** | recommended default |
| Hippocampus alone (replacement, smoke-only) | — | BG-active=0% (cascade silent) |
| **Hippocampus additive (heuristic + plastic place/goal)** | **10.98** | **1.87× WORSE** |

Per-seed: 9.43 / 11.99 / 11.52. P1 action counts ~uniform (e.g. seed 44: 362/378/395/365), confirming cascade selectivity collapsed.

## Implementation (kept opt-in: `--hippocampus`)

Two new regions inside the brain-region framework:

```python
BrainRegion(name="place_cells", n_neurons=64, izh_neuron_type="IZH2007_HIPPO_PYRAMIDAL", ...)
BrainRegion(name="goal_cells",  n_neurons=64, izh_neuron_type="IZH2007_HIPPO_PYRAMIDAL", ...)
```

64 cells each, one per (x, y) position on the 8×8 grid. Per-step drive is a Gaussian over preferred position with σ=0.5 — only 1–3 cells fire per step (sparse encoding).

Eight plastic pathways: `{place_cells, goal_cells} × cortex_{N,E,S,W}`, density=1.0, weight_mean=10.0.

Drive logic (additive, not replacement):
```python
# Heuristic cortex drive runs as before (gx>x → cortex_E, etc.)
if enable_learned_perception: ...
else: heuristic 800 pA on aligned cortex pool

# Hippocampus drive ADDED on top:
if enable_hippocampus:
    place_drive  = max_pA * exp(-||(hippo_pref) - (x, y)||² / 2σ²)   # σ=0.5
    goal_drive   = max_pA * exp(-||(hippo_pref) - (gx, gy)||² / 2σ²)
    bridge.cp_external_input_current[place_indices] = place_drive
    bridge.cp_external_input_current[goal_indices]  = goal_drive
```

Smoke-tested 200 steps clean.

## Why it fails

The exact same cold-start pattern as learned perception:

1. **Random plastic weights have no asymmetry.** `place_cells → cortex_{N,E,S,W}` are 4 pathways with random weights (mean=10, jitter=0.2) that look approximately equal in the unsigned mean. When place_cells fire, all 4 cortex pools receive ~equal extra drive from the hippocampus.

2. **Equal drive across pools destroys cascade selectivity.** The heuristic puts 800 pA on ONE cortex pool and 0 on the others — the BG cascade amplifies that asymmetry into a clean motor decision. Hippocampus adds noise *to all four pools simultaneously*, and the cascade can't tell them apart.

3. **STDP can't bootstrap from no asymmetry.** With all 4 cortex pools firing at similar rates, all 4 hippocampus→cortex pathways get similar reward credit. There's no signal for plasticity to amplify.

This is the fourth time we've hit this exact failure mode:

| Attempt | Result |
|---|---|
| Cold-start learned perception (random sensory→cortex) | NEGATIVE |
| Informed-init perception (directional prior on sensory→cortex) | NEGATIVE — even slightly worse than random |
| Hippocampus alone (random place→cortex, replaces heuristic) | BG-active=0% (cascade silent) |
| **Hippocampus additive (random place→cortex on top of heuristic)** | **NEGATIVE — degrades heuristic by 1.87×** |

## Architectural insight

This is now a robust, repeated finding: **the BG cascade requires clean asymmetric cortex pool selectivity, and any plastic input layer with random initial weights breaks this.** Adding "more biology" without solving the asymmetry-injection problem makes things worse, not better.

To make a plastic input layer work, you need ONE of:
- **Curriculum**: lock cortex→D1 weights first under fixed-goal training, then thaw and expose the input layer
- **Cortex-level WTA / lateral inhibition**: enforce one-cortex-pool-fires-at-a-time at the architecture level, not at the input level
- **Sparse 1-of-N encoding at the input** with weights initialized so that exactly one input cell strongly drives one cortex pool (essentially a hand-built look-up table — at which point you're back to the heuristic)
- **A different gating signal** that suppresses hippocampus contribution until reward training has built selectivity (a "novelty" or "uncertainty" gate)

None of these were tested in this autonomous session — each is a multi-day arc.

## Per-seed details

```
seed 42: P0 finalQ=4.99 P1 finalQ=4.44 sum=9.43  P1 actions=[392,396,349,363] n_at_goal=14
seed 43: P0 finalQ=5.08 P1 finalQ=6.91 sum=11.99 P1 actions=[381,383,375,361] n_at_goal=17
seed 44: P0 finalQ=6.49 P1 finalQ=5.03 sum=11.52 P1 actions=[362,378,395,365] n_at_goal=18
                                                        avg sum=10.98
```

Action counts within ~10% of uniform across all seeds — cascade has been disabled by hippocampus noise. n_at_goal is similar to random walk (~15/1500 steps).

## Decision

- Keep `--hippocampus` flag opt-in for future curriculum / WTA experiments.
- Default remains heuristic only.
- Treat this as **architectural ceiling closure**, not just one failed variant: any plastic input layer with random weights fails on this BG cascade. Future learned-perception / hippocampus / sensory-cortex work needs to start with curriculum or cortex-WTA, not yet-another-init-scheme.

## Files

- `research/runners/g11_bg_runner.py:79-138, 263-281, 386-435, 715-790`: hippocampus module
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_hippo.json`: 3-seed acid test data
- `research/findings/raw/g11_bg/g11_seed42_hippo_smoke.json`: smoke verification

## Lesson

The pivot from "replacement" to "additive" felt like the right biological fix — and it cleaned up the smoke test (no more BG=0%) — but it didn't solve the underlying problem because the actual issue isn't the heuristic disappearing, it's that random weights to multiple cortex pools inject *noise that breaks the asymmetry* the cascade is built to amplify. Heuristic + noise < heuristic.

Pavlovian (this morning) and the BG cascade (Phase B) both work because their inputs have *clean asymmetry* — Pavlovian via the CS/US distinction, BG via the heuristic's hard-coded directional drive. Once we put a noisy plastic layer in front, the cascade loses the clean signal it needs.

Closing the autonomous session here. Four consecutive variations on "add a plastic input layer to the BG cascade" all hit the same architectural ceiling. Productive next moves require structural work (curriculum, WTA), not parameter tweaks. That's a fresh-conversation problem with the user, not a 2 AM autonomous decision.

## What's *not* the bottleneck (re-confirmed today)

- The plasticity stack (Pavlovian: 5.56 → 16.32 Hz, weights 0.10 → 0.9995). Fine.
- The BG cascade (Phase B: phase 1 finalQ 1.76 vs G9 baseline 6.74). Fine.
- The neuromodulator subsystem (E.1 framework GO; multiple production rules tested today). Fine.
- The brain-region framework (E.2 — adding two new regions worked first try, no edits required). Fine.

## What *is* the bottleneck

Cortex-pool selectivity under graded multi-pool inputs. That's the next major architectural problem — and it's not a runner-side fix.
