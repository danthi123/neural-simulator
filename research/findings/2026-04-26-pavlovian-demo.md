# Pavlovian Conditioning — Architecture Demonstrates Classical Learning

**Date:** 2026-04-26 (post-Phase-B, pre-hippocampus arc)
**Status:** **GO**. The simulator's existing experiment system + plasticity stack produces clean Pavlovian conditioning. CS firing rate triples after CS-US pairing.
**Companion:** [Phase B real-win](2026-04-25-phase-b-acid-test-real-win.md), [SCIENCE_ROADMAP §2.1 (STDP timing benchmark)](../../docs/SCIENCE_ROADMAP.md)

## TL;DR

After spending two days on BG-cascade RL refinements, validated that the same architecture handles canonical Pavlovian conditioning cleanly using the built-in `ASSOCIATIVE_PAIRING` training mode.

| Metric | Pre-training | Post-training | Δ |
|---|---:|---:|---|
| CS firing rate | 5.56 ± 1.78 Hz | **16.32 ± 2.36 Hz** | +10.76 Hz (t=13.95) |
| CS→US weights (mean) | 0.10 | 0.9995 | saturated potentiation |

**Learning detected: YES.** The architecture's STDP + reward-modulation stack reproduces the canonical Pavlovian result without any task-specific tuning.

## Setup

- 100 CS input neurons, 100 US output neurons
- Dense (80%) random CS→US connectivity, plastic with STDP enabled
- 100 training trials, each: CS pulse train (40 Hz, 200 ms) + US constant pulse 100 ms later, 500 ms total
- Pre-test: 5 CS-alone presentations (no plasticity, no US)
- Post-test: 10 CS-alone presentations (no plasticity, no US)

This used `ExperimentPresets.associative_conditioning()` with default parameters and the built-in `run_experiment_headless.py --experiment associative` runner.

## Why this matters

Three reasons:

1. **Sanity check on the plasticity stack.** All the recent BG-cascade work has been on a complex multi-region task. This is a 200-neuron two-population test that exercises only the STDP+reward+conductance machinery. If this didn't work, all the higher-level claims would be suspect. It works.

2. **Different problem class than RL.** Pavlovian is associative learning — a stimulus pattern that predicts another stimulus. No agent, no actions, no goal. The plasticity stack handles it independently of the BG architecture stuff we've been building.

3. **Concrete biology demonstration.** The Bi & Poo STDP timing curve was already validated kernel-level (`run_benchmarks.py --benchmark stdp-timing`). This shows the same mechanism produces emergent learning at the population level, which is the actual claim of biological plausibility.

## Per-trial dynamics

Saturation pattern is informative:
- **Trial 1**: CS→US weights = 0.10 (initial)
- **End of training**: CS→US weights = 0.9995 (essentially at `stdp_w_max=1.0`)

Soft-bound STDP, by design, asymptotes near `w_max`. The 100 trials × 200 ms CS × 40 Hz = 8000 CS spikes per neuron, each chance to do LTP with the US-coincident post-spikes. Saturation is expected and correct biological behavior.

If we wanted graduated learning (more like spaced trial training), we'd reduce the LTP rate (`A_plus`) or use fewer/sparser trials. Current setup is "rapid full-saturation" Pavlovian — fine for demonstrating the mechanism.

## Files

- `experiment/presets.py:70-149`: Pavlovian preset definition (existed before tonight)
- `run_experiment_headless.py --experiment associative`: runner (existed before tonight)
- `experiment/associative_demo_2026-04-26.json`: 100-trial run output

## Decision

- No code changes, no new flags. The existing infrastructure works as designed.
- This is a re-validation finding: the architecture handles canonical biology, not just RL.
- Pivoting to hippocampal place cells next (next major arc).

## Lesson

Sometimes the best way to break out of an iteration loop on a hard task is to do a quick cross-check on a different task class. Two days of BG-cascade refinements gave modest task-conditional gains; one half-hour Pavlovian run shows the underlying machinery is rock solid. The architecture isn't the bottleneck — the moving-goal task with heuristic perception just hit its ceiling.

## Next: hippocampal place cells

Building on this: add a hippocampal module that learns place→action associations via the same STDP+reward mechanism that just worked on Pavlovian. This gives the agent positional memory it currently lacks. Plan in next finding doc.
