# Learned Perception (Sensory→Cortex) — Cold-Start Bootstrap Failure

**Date:** 2026-04-26
**Status:** NEGATIVE — does not learn within 1800 trials. Architectural finding, not a tuning issue.
**Companion:** [Phase B real-win](2026-04-25-phase-b-acid-test-real-win.md), [Asymmetric adaptive DA](2026-04-26-asymmetric-adaptive-da.md)

## TL;DR

Replaced heuristic cortex drive (`if gy > y: drive cortex_N at 800 pA`) with a learned mapping: 49-neuron sensory layer (7×7 grid tuned to (dx, dy) ∈ [-3, 3]²) projects via plastic weights to all 4 cortex pools. STDP+reward must shape the position→action mapping during training.

**Result: agent stays at random-walk level for 1800 trials.**

| Variant | Phase 0 finalQ | Phase 1 finalQ | Sum | BG-active |
|---|---:|---:|---:|---:|
| Random walk | ~5.5 | ~5.5 | ~11 | n/a |
| Heuristic baseline | 3.48 | 1.76 | 5.24 | 22-24% |
| **Learned perception (cold start)** | **5.58** | **5.27** | **10.85** | **3-4%** |

Per-seed: P0/P1 are 4.97/4.61, 5.27/6.60, 6.49/4.60. The variation between seeds is similar to between-phase variation — there's no learning signal.

## Why it doesn't bootstrap

The cold-start architecture has two compounding problems:

1. **Random sensory→cortex weights produce roughly uniform cortex drive.** With 49 sensory × 25 cortex × density=1.0 = 1225 synapses per cortex pool, and weight_mean=10 ± 20% jitter, every position pattern activates all 4 cortex pools roughly equally. Cortex_N, cortex_E, cortex_S, cortex_W all fire at similar rates → BG cascade has no asymmetric input to amplify into a clear winner.

2. **BG-active rate collapses.** Because cortex is firing uniformly (or barely firing), str_D1 doesn't get the strong selective drive it needs to silence GPi. GPi stays tonic, thalamus stays inhibited, motor stays silent. **Phase 1 BG-active rate drops from 22% (with heuristic) to 3-4%** — in 96-97% of trials, all motors are silent and the runner falls back to random action selection.

Without ANY initial selectivity, there's no signal for STDP+reward to amplify. The system has nothing to learn FROM.

## Mechanism check

When all 4 cortex pools fire equally:
- All 4 D1 pools fire equally
- All 4 GPi pools get equal D1 inhibition + STN excitation → all stay roughly tonic
- All 4 thal pools stay inhibited equally
- No motor pool fires → random action
- Random action might happen to reduce distance → +1 reward
- Reward is delivered but eligibility traces are spread across ALL cortex→D1 pathways equally (because all D1s had eligibility from firing equally)
- All 4 cortex→D1 pathways get the same boost
- Net effect: all weights drift similarly, no differentiation

This is the classic "credit assignment with random initial conditions" problem from RL literature. Normally you need either:
- Some initial selectivity to break the symmetry (informed initialization)
- Strong noise to randomly create asymmetries (exploration)
- Curriculum learning (easier task first)
- Pre-training on a related task

Our setup has none of these.

## Why this is consistent with prior findings

The original (heuristic) baseline had cortex driven directly at 800 pA for goal-relative directions. This is essentially a **hand-coded inductive bias** — the agent doesn't need to learn that "goal is up → drive cortex_N" because we set it that way. Plasticity then only had to learn the smaller task: "given my action choices, which cortex→D1 weights need adjustment?"

Removing that bias creates a much harder task: the agent must learn BOTH the perceptual mapping AND the action-conditional weights, simultaneously, from scratch. STDP+reward over 1800 trials isn't enough.

## Workarounds (not yet tried)

1. **Informed initialization**: pre-warm sensory→cortex weights with a heuristic prior (e.g., sensory neurons tuned to +dx connect more strongly to cortex_E). Plasticity then refines rather than discovers.
2. **Curriculum**: train on a fixed-goal scenario first (say 5000 steps to one goal), then test the moving-goal acid test.
3. **Reduce dimensionality**: smaller sensory layer (fewer neurons), each connecting more sparsely so weights matter more per-synapse.
4. **Hybrid drive**: keep heuristic but add learned modulation (best of both worlds).
5. **Pre-training**: warm sensory→cortex with random target-goal pairs before the moving-goal task.

Most of these treat learned perception as a refinement on top of a working baseline rather than a from-scratch architecture replacement.

## Decision

- Keep `--learned-perception` flag as opt-in. Default OFF (heuristic remains).
- Document this as the cold-start failure. Future work could revisit with one of the workarounds above.
- The bigger architectural insight: cortex pre-firing structure (whether hand-coded or curriculum-learned) is essentially a prerequisite for the BG cascade to function. The cascade amplifies asymmetric cortex inputs into decisive motor output, but it can't create asymmetry from nothing.

## Files

- `research/runners/g11_bg_runner.py:88-96, 197-208, 510-525, 591-606`: learned perception implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_learnedP.json`: 3-seed acid test data

## Next experiments

Not pursuing learned perception further this session. Pivoting to **DA-gated WTA** — the original "DA gate" concept from the user's question, now well-positioned to test on top of the asymmetric adaptive DA win.
