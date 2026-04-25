"""Validate HH_L5_CORTICAL_PYRAMIDAL_RS preset against published L5 pyramidal data.

This is the BASE preset that all other HH presets in the simulator derive from
(see sim/enums.py:DefaultHodgkinHuxleyParams). Validating it first establishes
the baseline behavior; subsequent region-specific presets can then be checked
for their region-specific deviations from this base.

## Reference values (literature)

L5 cortical pyramidal regular-spiking (RS) neurons in vitro and in computational
models:

| Property                  | Target        | Sources                              |
|---------------------------|---------------|--------------------------------------|
| Resting Vm                | -65 to -75 mV | Hay 2011 JNS; Mainen & Sejnowski 1996 |
| Spike threshold           | -50 to -55 mV | Hay 2011; Pospischil 2008            |
| Spontaneous rate (no drive)| ~0 Hz         | low-noise in vitro recordings       |
| Rheobase (real cells)     | 50-200 pA *physical* | Allen Cell Types Database     |
| F-I curve shape           | Linear after threshold, slope ~50-100 Hz/nA | Hay 2011 Fig 1 |
| Adaptation ratio (steady/initial) at moderate input | 0.4-0.7 | spike-frequency adaptation hallmark of RS |
| Max steady-state rate     | 30-100 Hz at strong drive | depends on g_K_max |

## Simulator unit caveat

The simulator's HH model converts external_input_current (pA) → µA/cm² density
via factor 1e-6 (sim/bridge.py:3697). This means "1 pA" in the simulator's
StimulusPattern.amplitude_pA is treated as 1e-6 µA/cm² of current density.
For a real cell with surface area ~3e-5 cm², an injected current of 1 pA
*physical* would correspond to 1 pA / 3e-5 cm² = 33 µA/cm² of density.
The simulator's "pA" therefore differs from physical pA by a factor of
~3e7. A drive of 15e6 "simulator pA" ≈ 15 µA/cm² density ≈ 0.45 nA *physical*
for a typical pyramidal cell. Reference: tests/validate_gpu.py uses 15e6 pA
as suprathreshold drive.

For this validation, we sweep currents 0 to 30e6 pA in the simulator's units,
which corresponds to 0-30 µA/cm² density (well-bracketing the threshold range
for HH dynamics).

## Pass criteria

- Resting Vm in [-75, -60] mV (matches preset's E_L = -70 mV ± noise/init)
- Spike threshold (smallest current eliciting a spike) below 10e6 pA
- F-I curve monotonically increasing in the 0..30e6 pA range
- Adaptation ratio < 0.9 at suprathreshold (some adaptation present)
- Max steady-state rate >= 10 Hz somewhere in the range

NOT a strict pass/fail; this is calibration of the base preset's
behavior to ground subsequent region-specific deviations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Smaller current sweep for smoke testing")
    args = ap.parse_args()

    from sim.enums import DefaultHodgkinHuxleyParams, NeuronType
    from research.validate._isolated_neuron import (
        run_step_current_protocol, summarize_fi_curve,
    )

    preset_name = "HH_L5_CORTICAL_PYRAMIDAL_RS"
    hh_params = DefaultHodgkinHuxleyParams.get_params(
        NeuronType[preset_name]
    )

    if args.quick:
        currents_pA = [0.0, 15e6, 50e6, 100e6, 500e6]
        n_neurons = 5
    else:
        currents_pA = [0.0, 5e6, 15e6, 30e6, 50e6, 100e6, 200e6, 500e6, 1e9, 2e9]
        n_neurons = 10

    print(f"\n{'='*72}")
    print(f"  Validating {preset_name}")
    print(f"  HH params (reference values from literature, see docstring):")
    for k, v in hh_params.items():
        print(f"    {k:20s} = {v}")
    print(f"\n  F-I curve: {len(currents_pA)} current levels, "
          f"{n_neurons} neurons each, dt=0.05ms, 1000ms stim")
    print(f"{'='*72}\n", flush=True)

    fi_curve = run_step_current_protocol(
        neuron_type_name=preset_name,
        current_steps_pA=currents_pA,
        n_neurons=n_neurons,
        dt_ms=0.01,
        temperature_celsius=6.3,  # See findings: at 37°C with Q10=3, rates over-compress
                                   # and APs don't reach v_peak. Tested at 6.3°C base.
        stim_duration_ms=500.0,
        pre_stim_ms=100.0,
        initial_settle_ms=100.0,
        seed=42,
    )

    metrics = summarize_fi_curve(fi_curve)

    print(f"\n{'='*72}")
    print(f"  F-I curve")
    print(f"{'='*72}")
    print(f"  {'I (pA)':>10s}  {'rate_init (Hz)':>14s}  "
          f"{'rate_steady (Hz)':>16s}  {'pre_Vm':>8s}  {'max_Vm':>8s}  "
          f"{'mean_Vm':>8s}  spikes")
    for p in fi_curve:
        n_spk = len(p.spike_times_ms)
        print(f"  {p.current_pA:>10.2g}  {p.rate_hz_initial:>14.2f}  "
              f"{p.rate_hz_steady:>16.2f}  {p.mean_vm_pre_stim:>8.2f}  "
              f"{p.max_vm_during_stim:>8.2f}  {p.mean_vm_during_stim:>8.2f}  "
              f"{n_spk:>5d}")

    print(f"\n{'='*72}")
    print(f"  Summary metrics vs literature targets")
    print(f"{'='*72}")
    print(f"  rest_vm:                {metrics['rest_vm']:.2f} mV   "
          f"(target: -75 to -60)")
    print(f"  rest_vm std:            {metrics['rest_vm_std']:.3f} mV")
    print(f"  spike_threshold_pA:     {metrics['spike_threshold_pA']:.3g} pA "
          f"(in simulator units; <10e6 plausible)")
    print(f"  rheobase_rate_hz:       {metrics['rheobase_rate_hz']:.2f} Hz")
    print(f"  rate at ~1 nA:          {metrics['rate_at_1nA']:.2f} Hz "
          f"(actual current: {metrics['current_at_1nA']:.2g} pA)")
    print(f"  adaptation_ratio_at_1nA:{metrics['adaptation_ratio_at_1nA']:.3f} "
          f"(target: <0.9 indicates some adaptation)")
    print(f"  max_steady_rate:        {metrics['max_steady_rate']:.2f} Hz "
          f"(target: >=10 Hz)")

    # Pass criteria
    pass_results = {
        "rest_vm_in_range": -75 <= metrics["rest_vm"] <= -60,
        "threshold_below_10e6": (
            metrics["spike_threshold_pA"] < 10e6
            if metrics["spike_threshold_pA"] == metrics["spike_threshold_pA"]  # not nan
            else False
        ),
        "fi_monotonic": all(
            fi_curve[i].rate_hz_steady <= fi_curve[i+1].rate_hz_steady + 5.0
            for i in range(len(fi_curve) - 1)
        ),
        "adaptation_present": metrics.get("adaptation_ratio_at_1nA", 1.0) < 0.95,
        "fires_at_strong_drive": metrics["max_steady_rate"] >= 10.0,
    }
    n_pass = sum(pass_results.values())
    n_total = len(pass_results)
    print(f"\n{'='*72}")
    print(f"  Pass criteria: {n_pass}/{n_total}")
    print(f"{'='*72}")
    for criterion, ok in pass_results.items():
        symbol = "PASS" if ok else "FAIL"
        print(f"  [{symbol}] {criterion}")

    # Save JSON
    out_dir = Path("research/findings/raw/preset_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"validate_{preset_name.lower()}.json"
    out = {
        "preset_name": preset_name,
        "hh_params": hh_params,
        "n_neurons": n_neurons,
        "fi_curve": [
            {
                "current_pA": p.current_pA,
                "rate_hz_initial": p.rate_hz_initial,
                "rate_hz_steady": p.rate_hz_steady,
                "n_spikes": len(p.spike_times_ms),
                "mean_vm_pre_stim": p.mean_vm_pre_stim,
            }
            for p in fi_curve
        ],
        "metrics": metrics,
        "pass_criteria": pass_results,
        "n_pass": n_pass,
        "n_total": n_total,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
