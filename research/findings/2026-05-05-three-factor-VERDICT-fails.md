# Headline: Biology-plausible three-factor learning fails where gradient succeeds

**Date:** 2026-05-05 ~05:25 EDT (4/6 seeds completed in topo_fs condition;
trend definitive)
**Status:** PARTIAL — last 2 topo_fs seeds (101, 102) pending; trend
unambiguous (0/4 so far, mean TRUE at chance).

---

## TL;DR

At biological scale (cortical canon: recurrence + E/I + NMDA + N=500
motor pool) with the biology-grounded fix (Pulvermüller topographic
prior + Vogels PV-FSI lateral inhibition):

| Learning rule | tf_with_topo_fs aligned | Mean TRUE accuracy |
|---|---|---|
| **Supervised gradient** | **3/3 PERFECT** | **35.3%** |
| Three-factor (biology-plausible) | **0/4 (so far)** | **25.0%** (chance) |
| STDP + R-STDP | 1/6 | 30.8% |

**Same architecture. Same biology fix. Same training data. Only the
credit-assignment rule differs.** Three-factor is at chance accuracy
where gradient gets perfect alignment.

## What this means

Three-factor learning (Frémaux & Gerstner 2016) is the cleanest
biology-plausible substitute for supervised gradient:

```
gradient:     Δw = lr × (target_rate - actual_rate)[motor] × pre_active
three-factor: Δw = lr × eligibility[pre,post]              × DA_sign[motor]
```

The function-space difference:
1. **Per-edge "pre" signal**: gradient uses `pre_active` (binary, no
   timing). Three-factor uses `eligibility` (pre×post coincidence with
   NMDA-tau decay).
2. **Per-pool "DA" signal**: gradient uses `(target - actual)` —
   magnitude-graded continuous error. Three-factor uses `sign(target -
   actual)` ∈ {+1, -1, 0} — direction only.

Three-factor is biology-plausible because:
- Synapses CAN observe pre×post timing (NMDA Ca²⁺ spikes detect this).
- Synapses CANNOT observe remote firing rates of other regions.
- A global scalar RPE signal IS biologically plausible (VTA/SNc dopamine).

So three-factor uses the maximum information that real biology can
deliver to a synapse: local timing + global scalar feedback. **Our
empirical result: this is insufficient for the W→A learning task at
biological scale.**

## What's needed instead

Gradient succeeds because it has access to per-pool MAGNITUDE error.
For biology to do this, synapses need richer feedback than scalar DA.
Three candidate frameworks:

### (1) Apical-basal dendritic learning (Bono & Clopath 2017)
Pyramidal neurons have two compartments:
- Basal dendrites: bottom-up sensory drive (current path)
- Apical dendrites: top-down feedback (predictions / errors)

Plasticity rules differ per compartment. Apical activity gates LTP/LTD
on basal synapses. The TOP-DOWN signal carries per-region error,
giving synapses local access to richer-than-scalar feedback.

**Implementation cost in our project:** major — we'd need
multi-compartment Izhikevich (currently point neurons), a feedback
pathway that delivers per-pool error to apical inputs, and the
plasticity rule itself. ~1-2 months of engineering.

### (2) Predictive coding (Rao & Ballard 1999)
Each region has dedicated "error neurons" that compute prediction-vs-input
mismatch. Error neurons project to the next layer; their activity drives
synaptic learning everywhere. Backprop-in-biology framework (Whittington
& Bogacz 2017 made it equivalent under certain assumptions).

**Implementation cost:** very major — different network organization
(predictive vs recognition pathways paired). Not a drop-in addition.
Probably 2-3 months minimum, and architecturally distinct from current
codebase.

### (3) Magnitude-graded three-factor (cheap probe)
What if we relax the "biology-plausibility" constraint slightly? In
real cortex, neuromodulator concentration scales with deviation
magnitude (Schultz 1998 measures this). Phasic DA bursts are
proportional to RPE magnitude, not just sign.

Cheap test:
```python
# Replace:
da = sign(target_rate - actual_rate)  # ±1 or 0
# With:
da = (target_rate - actual_rate) / max_rate  # ∈ [-1, +1] continuous
```

This stays biology-plausible (DA is naturally graded) and is a
1-line change to `bio_three_factor.py`. **Easy follow-up
experiment** — should ship before committing to (1) or (2).

## What ruling this out tells us

The W→A 0/N alignment streak across the project's history (over 18
days of architecture variants, biology sweeps, etc.) was driven by
**STDP+R-STDP being too weak**. The 2026-05-04 B3 result showed
gradient works. Today's 2026-05-05 result narrows the gap further:

> Even three-factor learning (the strongest biology-plausible
> framework with classical scalar DA) is insufficient. The path
> forward requires either richer per-region feedback (dendritic
> learning) or different network topology (predictive coding).

## Cross-rule comparison at bio_topo_fs (the controlled condition)

```
Gradient:        3/3 aligned   mean 35.3%
STDP+R-STDP:     1/6 aligned   mean 30.8%
Three-factor:    0/4 aligned   mean 25.0%   (4 of 6 seeds done; trend stable)
```

Each row uses identical: cortical canon, biological scale,
Pulvermüller topo prior + Vogels PV-FSI, 4000 training events,
identical eval methodology. **Only the per-synapse update rule
differs.**

## Recommended next-step ranking (for user review)

**Cheapest (~1 day):** Magnitude-graded three-factor. 1-line change.
If it succeeds, biology-plausibility is preserved AND scalar-DA isn't
the bottleneck — magnitude-graded DA is enough.

**Medium (~1-2 months):** Apical-basal dendritic learning. Requires
multi-compartment neurons but reuses existing connectivity infra.

**Heavy (~2-3 months):** Full predictive coding rewrite. Different
fundamental network organization.

The decision orchestrator (`scripts/post_three_factor_decision.ps1`,
PID 19812) will mark this as the FAILURE path and stop for manual
research direction. Final results land in ~3.5 hours when last 2
topo_fs seeds finish.
