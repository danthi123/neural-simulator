# Cluster C v1 — Tonic DA + Phasic Depression Coding Design

**Date:** 2026-04-29
**Goal:** Make dopamine a real neuromodulator with **tonic baseline** and proper phasic activation/depression dynamics, replacing the current signed-scalar `current_reward_signal`.
**Why now:**
- B.3 TANs is currently a no-op because there's no tonic DA-driven plasticity for ACh to gate (per `2026-04-28-cluster-b3-tans-results.md`).
- R2.4 added asymmetric magnitude scaling but kept the signed-scalar design. The catalog (Schultz98/16) wants tonic DA + phasic deviations as separate dynamics.
- Tonic DA is a foundational primitive — once it exists, B.3 + neuropeptide arms (R3.6) compose meaningfully.

## What changes

### 1. New `dopamine` neuromodulator (opt-in)

Register a `dopamine` `NeuromodulatorConfig` via `--enable-tonic-da`:

```python
def _default_dopamine_config():
    return NeuromodulatorConfig(
        name="dopamine",
        baseline=0.5,                # tonic DA, modest positive baseline
        decay_tau_ms=200.0,          # phasic responses decay over ~200 ms
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0),
            # plasticity_rate = 1 + sensitivity * (conc - baseline)
            # → at baseline (0.5), gain=1.0 (no change)
            # → above (e.g. 1.5), gain=2.0 (LTP ramp)
            # → below (e.g. 0.0), gain=0.5 (LTD: gentler than full block)
        ],
        production_rules=[
            ProductionRule(rule_type="from_reward", sensitivity=+1.0, threshold=0.0),
            # Positive reward → DA activation above baseline
            # Negative reward → DA depression below baseline
            # The signed-reward semantic is preserved at the neuromod level.
        ],
    )
```

### 2. Bridge: replace signed-scalar reward modulation with DA-modulated path

Currently:
```python
# bridge.py:4419
weight_updates = effective_reward_lr * reward_prediction_error * eligibility[:n]
```

After Cluster C (when subsystem on AND dopamine registered):
```python
# DA concentration encodes reward state directly (tonic + phasic).
da_signal = self.neuromodulator_manager.get_concentration("dopamine") - dopamine_baseline
# da_signal > 0: phasic activation (LTP)
# da_signal < 0: phasic depression (LTD)
weight_updates = effective_reward_lr * da_signal * eligibility[:n]
```

This naturally:
- Gives B.3's plasticity_window_gate something to gate (DA-driven plasticity is now non-zero between rewards as long as ACh permits).
- Couples seamlessly with R3.6 neuropeptide arms (dynorphin lowers plasticity_rate; combines with DA-driven plasticity_rate via multiplicative aggregation).

### 3. Compatibility: keep legacy path

When `--enable-tonic-da` is OFF (default), fall back to the existing `reward_prediction_error * eligibility` path. No regression.

## Implementation steps

1. **`sim/neuromodulators.py`:** add `_default_dopamine_config()` helper, mirroring `_default_acetylcholine_config()`. ~30 LOC.
2. **`sim/bridge.py`:** in the reward modulation block (line 4419), branch based on whether `dopamine` is registered:
   - If registered: use `da_signal * eligibility` (as above)
   - If not: keep existing `reward_prediction_error * eligibility`
3. **`research/runners/g11_bg_runner.py`:**
   - Add `enable_tonic_da: bool = False` kwarg
   - Add `--enable-tonic-da` CLI flag
   - When on, register `_default_dopamine_config()` and enable subsystem (cumulative with --enable-tans / --enable-bg-neuropeptides).
4. **`tests/test_tonic_da.py`** (new file): 4-5 tests
   - dopamine config registers
   - tonic baseline gives plasticity_rate=1.0
   - positive reward → DA above baseline → plasticity_rate > 1.0
   - negative reward → DA below baseline → plasticity_rate < 1.0
   - bridge uses DA path when subsystem on (regression: legacy path unchanged when off)
5. **Smoke test + cheat-5 multi-goal eval n=3**

## Validation criteria

### Smoke test
- 50-step run with `--enable-tonic-da` completes; DA concentration spans baseline ± phasic deviations as expected.

### Cheat-5 multi-goal re-eval

Compare 4 conditions (n=3 seeds 42/43/44 each):
- A) Baseline (no Cluster A, no tonic DA, no TANs)
- B) Cluster A only (tested in `2026-04-29-cluster-a-closed-bg-loop-results.md`)
- C) Cluster C only (`--enable-tonic-da`)
- D) Cluster A + C + B.3 (`--enable-cluster-a-closed-loop --enable-tonic-da --enable-tans`)

**Hypothesis:** D has the strongest effect because:
- A provides closed loop (post-synaptic teaching signal for STDP)
- C provides tonic DA (gives B.3's gate something to gate)
- B.3 + C unlocks the temporal plasticity window biology

If A alone doesn't close cheat-5 (likely partial), and C alone doesn't either, D might.

## Estimated effort

2-3 hours: implementation + tests + smoke. Cheat-5 eval: 12 sequential 1800-step runs ~110 min. Total: 4-5 hours including findings.

Will dispatch right after Cluster A eval results are in.
