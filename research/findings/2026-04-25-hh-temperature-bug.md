# HH Model: Action Potentials Don't Fire at 37°C — Temperature Scaling Bug

**Date:** 2026-04-25
**Severity:** Major — affects every HH-based simulation in the project
**Discovered during:** Phase A brain-region library validation (Session "Phase A")
**Status:** Confirmed; needs follow-up to fix the simulator's HH integration at biological temperatures

## Summary

The simulator's HH neuron model **does not produce action potentials at the
default biological temperature (37°C)** for any of its registered presets,
including the base `HH_L5_CORTICAL_PYRAMIDAL_RS` and the legacy
`HH_EXCITATORY_DEFAULT_LEGACY` (= original Hodgkin-Huxley squid axon
parameters). At 37°C with Q10=3.0, the cell depolarizes to a tonic plateau
~+30 mV under strong current injection but never crosses `v_peak=40 mV`,
so spike detection never triggers.

At the original characterization temperature (6.3°C), spikes fire correctly
for both the original parameters and the L5 RS preset.

## Reproduction

`research/validate/validate_hh_original.py` and
`research/validate/validate_hh_l5_pyramidal_rs.py`. With:
- `cfg.hh_temperature_celsius = 37.0` (default), `cfg.hh_q10_factor = 3.0` (default)
- `dt_ms = 0.005` (small enough for stable Euler integration)
- Step current injections from 0 to 500e6 pA (= 0 to 500 µA/cm² density)
- 5 isolated neurons, no plasticity, no noise, no synapses

### At 37°C (default):

```
HH_EXCITATORY_DEFAULT_LEGACY (original HH params):
      I (pA)   max_Vm   mean_Vm  spikes
           0   -65.00    -65.00      0
       5e+06   -61.70    -61.74      0
     1.5e+07   -57.01    -57.94      0
       3e+07   -48.95    -54.47      0
       5e+07   -30.44    -51.39      0
       1e+08   -16.53    -46.53      0
       5e+08    13.27    -30.87      0   ← still no spikes
```

The cell depolarizes monotonically with current but max Vm tops out at
+13 mV. v_peak (40 mV) is never reached.

### At 6.3°C:

```
HH_EXCITATORY_DEFAULT_LEGACY at 6.3°C:
      I (pA)   max_Vm   mean_Vm  spikes
           0   -65.00    -65.00      0
       5e+06    39.25    -61.33      0   ← just below threshold
     1.5e+07    40.99    -54.16      5   ← crosses v_peak, spike detected!
       3e+07    41.84    -51.24      5
       5e+07    42.99    -48.56      5
       1e+08    45.17    -44.95      5
       5e+08    61.08    -30.34      5
```

Spikes fire from I=15e6 pA upward.

`HH_L5_CORTICAL_PYRAMIDAL_RS` at 6.3°C also fires (max_Vm reaches 47-72 mV
across current range), with rest Vm = -71.16 mV (matches preset's E_L=-70).

## Root cause

`sim/kernels.py:36 fused_hodgkin_huxley_dynamics_update` applies a single
Q10 temperature factor to all gating variables:

```python
phi = q10_factor**((temperature_celsius - 6.3) / 10.0)
alpha_m = alpha_m_orig * phi  # Same phi for all gates
beta_m  = beta_m_orig  * phi
alpha_h = alpha_h_orig * phi
...
```

At 37°C with Q10=3.0, phi ≈ 27. All gate kinetics speed up by 27×, which
shrinks AP duration from ~3 ms to ~0.1 ms. With dt=0.05ms (the simulator's
HH default), spike events span only 1-2 timesteps and the integration becomes
numerically unstable: the upstroke and downstroke overlap in a single step,
preventing V from reaching v_peak=40 mV.

Even at dt=0.005ms (10× smaller), the issue persists because the underlying
problem is not numerical but biophysical:
- h (Na inactivation) at high V has τ_h ≈ 0.04 ms (after Q10 scaling)
- n (K activation) at high V has τ_n ≈ 0.04 ms
- These are *faster* than typical AP upstroke (which the m-gate drives)
- Result: K activation and Na inactivation kick in fast enough to clamp V
  before it reaches v_peak

In real neurons, different gates have *different* Q10 values (e.g., Q10
for activation rates is typically 2-3, but Q10 for inactivation/recovery
is often 1.3-1.5). The biophysically-correct fix is per-gate Q10, not a
uniform value.

## Implications

1. Every test or experiment using HH neurons at 37°C in this codebase has
   been running with cells in **plateau depolarization mode**, not actual
   action potentials. This affects:
   - All HH-based brain region presets in CLAUDE.md (PFC, CA1, CA3, TRN,
     MSN, STN, GPe, Cerebellar Purkinje/Granule, Spinal Motor, Olfactory
     Mitral, DA SNc, FS interneuron, Inferior Olive)
   - Any plasticity/STDP testing on HH cells
   - Any HH network probes
2. Network-level behaviors that "work" with HH at 37°C are likely
   succeeding via the plateau Vm crossing some other threshold or via the
   legacy spike detection picking up the tonic depolarization noise — not
   actual spike-based dynamics.
3. The `HH_PFC_PYRAMIDAL` preset specifically claims "strong NaP for
   persistent activity" — this can't be validated until the AP firing
   bug is fixed.

## Workarounds (until fixed)

1. **Run HH at 6.3°C base temperature**: `cfg.hh_temperature_celsius = 6.3`.
   Loses the biological-temperature realism but produces real APs.
2. **Reduce v_peak**: lower from 40 mV to e.g. 20 mV so spike detection
   triggers at the plateau peak. Doesn't fix the underlying biophysics
   but makes "spikes" countable.
3. **Use Izhikevich or AdEx instead** — these are not affected by this
   issue and are widely used for cortical models.

## Recommended fix (deferred to a future session)

1. Implement per-gate Q10 in `fused_hodgkin_huxley_dynamics_update`:
   - `phi_m = q10_m^((T-6.3)/10)`
   - `phi_h = q10_h^((T-6.3)/10)`
   - `phi_n = q10_n^((T-6.3)/10)`
   - With Q10_m ≈ 3 (fast activation), Q10_h ≈ 1.5, Q10_n ≈ 1.5 (typical)
2. Re-validate all 17 presets at 37°C
3. Update temperature defaults in `CoreSimConfig` if needed

## Files

- [`research/validate/_isolated_neuron.py`](research/validate/_isolated_neuron.py) — isolated-cell validation harness
- [`research/validate/validate_hh_l5_pyramidal_rs.py`](research/validate/validate_hh_l5_pyramidal_rs.py)
- [`research/validate/validate_hh_original.py`](research/validate/validate_hh_original.py)
- Raw data: [`research/findings/raw/preset_validation/`](research/findings/raw/preset_validation/)

## Knock-on bug found in same investigation

The `build_hh_isolated_config` initial draft set `cfg.hh_*` fields directly,
expecting them to override the preset. They are IGNORED — the bridge
(`sim/bridge.py:1029`) reads HH params via
`DefaultHodgkinHuxleyParams.get_params(NeuronType[cfg.default_neuron_type_hh])`
and uses those, regardless of cfg.hh_C_m / cfg.hh_g_Na_max / etc. fields.

This means: setting `cfg.hh_g_Na_max = 200.0` does NOT change the simulation;
it only updates the dataclass field. To use a custom HH parameter set, you
must add it to `DefaultHodgkinHuxleyParams.PARAMS` under a NeuronType enum
entry.

This is documented in the `build_hh_isolated_config` docstring.
