# PFC Bistable Persistent Activity — NEGATIVE Result

**Date:** 2026-04-25
**Session:** F (PFC working-memory module attempt)
**Status:** PFC track abandoned, pivot to motor exploration noise

## Goal

Tune a PFC-only BrainRegion (200 Izhikevich RS neurons) to exhibit Wang-2002-style
bistable persistent activity: ~5 Hz quiescent baseline → brief input pulse →
sustained 20-40 Hz "upstate" for ≥1 sec post-pulse. Pass criteria:
- Pre-pulse 1-12 Hz
- Pulse-window ≥20 Hz
- Persistent (500-1500ms post-onset) 12-60 Hz
- CV < 50%

If achievable, plug PFC into G9 as goal-context holder so motor-selection
gets stable PFC drive across trials, hopefully fixing the silent-motor trap.

## Method

[`research/probe_pfc_persistent_activity.py`](research/probe_pfc_persistent_activity.py)
sweeps `(exc_weight_mean, inh_weight_mean, internal_density, exc_fraction)`
on a frozen-plasticity 200-neuron Izhikevich PFC region with a 250 pA / 20 ms
pulse to all neurons at t=200ms. Total 18 configs, ~3s each, 51s wall.

Two passes were run:
1. **Quick (4 configs)**: exc_weight ∈ {0.5, 1.0, 1.5}, density ∈ {0.15} — all collapse to 5.3 Hz persistent rate
2. **Wider (18 configs)**: exc_weight ∈ {1.5, 2.5, 3.5}, inh_weight ∈ {0.5, 1.0, 1.5}, density ∈ {0.20, 0.30}

## Result

Every config in the wider sweep produced essentially identical activity:

| Window | Range across 18 configs |
|--------|--------------------------|
| Pre (0-200ms) | 6.1-6.5 Hz |
| Pulse (200-300ms) | 24.1-25.9 Hz |
| Persistent (500-1500ms) | **5.3-5.4 Hz** |

4× variation in `exc_weight_mean` (1.5→3.5), 1.5× variation in `internal_density`
(0.20→0.30), and 3× variation in `inh_weight_mean` (0.5→1.5) **all produced the
same persistent rate to within 0.1 Hz**. This is not parameter sensitivity — it's
a homeostatic floor.

Best score across all 18 configs: 2/4 (pre + pulse pass; persistent + stability fail).

## Diagnosis

The simulator's default configuration includes:
- OU background noise (σ ≈ 100 pA, τ ≈ 15ms) — substantial drive jitter
- Homeostatic firing rate regulation (EMA α ≈ 0.0002, τ ≈ 5s) targeting ~5 Hz
- Inhibitory reversal at -75mV with 0.7× propagation scaling

These mechanisms are designed to keep the network in a stable balanced regime,
which is the *opposite* of what's needed for bistability. The transient response
to the pulse (24 Hz) is consistent with a balanced E/I network, but the network
returns to baseline within ~200ms — driven back by inhibition + homeostasis.

To get true bistability with this architecture would require either:
1. Disabling homeostasis (`enable_homeostasis = False`) and reducing OU σ to <30 pA
2. Switching to HH neurons with the `HH_PFC_PYRAMIDAL` preset (NaP-driven persistence)
3. Adding NMDA receptors with τ≈100ms (Wang 2002's actual mechanism)
4. Dramatically increasing pulse strength (>1000 pA) and pulse duration (>100ms)

Each of these is a substantial investment with uncertain return.

## Strategic Pivot: Motor Exploration Noise

The silent-motor trap is, mechanistically:
1. In phase 1, only some motor neurons fire (others silent due to weak afferent weights)
2. STDP-eligibility traces only form for synapses that experienced a postsynaptic spike
3. Weights TO silent motors stay frozen — no eligibility, no reward-mediated update
4. Agent gets stuck at whatever motor distribution emerged from initial random weights

The classical fix is **motor exploration**: stochastic activation of motor neurons
that ensures every motor fires occasionally, regardless of upstream input. This is:

- **Biologically real**: Tonic dopamine / cholinergic baseline drives spontaneous
  motor activity in real brains; basal ganglia + cortex have noisy spontaneous activity
- **CS-classical**: Equivalent to ε-greedy exploration in tabular RL, or entropy
  regularization in policy gradient methods
- **Direct**: Targets the failure mechanism (silent neurons) without trying to
  reshape upstream representations
- **Cheap**: Implementation is ~50 lines (Poisson spike injection at motor neurons)

This is what the next session attempts. PFC working memory remains a valuable
*future* feature for richer cognition, but it's the wrong tool for the silent-motor
trap.

## Files

- [probe](research/probe_pfc_persistent_activity.py) — tuning sweep harness
- [raw results](research/findings/raw/pfc_tuning/pfc_tuning_results.json)
- [Session F plan](docs/plans/2026-04-25-pfc-working-memory-module.md) — superseded
