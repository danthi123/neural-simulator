"""Q10 sweep at 37°C: find what value lets original HH params fire APs.

Diagnostic for the HH temperature bug. We know:
  - At 6.3°C with Q10=3.0, HH params fire correctly
  - At 37°C with Q10=3.0, HH params don't fire (over-compressed dynamics)
  - At 37°C with Q10=1.0 (no temperature scaling), kinetics same as 6.3°C
    case, should fire

Sweep Q10 ∈ {1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0} at temperature=37°C.
Find smallest Q10 that produces APs at moderate input.

Uses HH_EXCITATORY_DEFAULT_LEGACY (= ORIGINAL_HH_PARAMS) as the test cell.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    from sim.enums import NeuronType, DefaultHodgkinHuxleyParams
    from research.validate._isolated_neuron import (
        run_step_current_protocol,
    )

    preset_name = "HH_EXCITATORY_DEFAULT_LEGACY"
    test_currents = [0.0, 15e6, 30e6, 100e6]  # cover sub- to suprathreshold
    n_neurons = 5

    print(f"\n{'='*72}")
    print(f"  Q10 sweep at 37°C — find lowest Q10 that produces APs")
    print(f"  Preset: {preset_name} (original HH params)")
    print(f"{'='*72}\n")

    print(f"  {'Q10':>6s}  {'pre_Vm':>8s}  {'I=15e6':>10s}  {'I=30e6':>10s}  {'I=100e6':>10s}  {'fires?':>7s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}")

    # Patch the simulator's q10_factor for each test
    from sim.config import CoreSimConfig
    from research.validate._isolated_neuron import build_hh_isolated_config

    # We need a way to override hh_q10_factor. Build a config and check.
    cfg = build_hh_isolated_config(neuron_type_name=preset_name, temperature_celsius=37.0)
    print(f"  (default hh_q10_factor = {cfg.hh_q10_factor}, temperature = {cfg.hh_temperature_celsius})\n")

    import json
    results = []
    out_path = Path("research/findings/raw/preset_validation/hh_q10_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Monkey-patch hh_q10_factor by overriding the config field after building
    for q10 in [1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
        # We need the protocol to use a custom Q10. The cleanest way is to
        # override CoreSimConfig.hh_q10_factor after build. But our protocol
        # function takes neuron_type_name, temperature_celsius — it uses the
        # default Q10. Need to extend the protocol with q10_factor parameter.
        try:
            fi = run_step_current_protocol(
                neuron_type_name=preset_name,
                current_steps_pA=test_currents,
                n_neurons=n_neurons,
                dt_ms=0.005,
                temperature_celsius=37.0,
                stim_duration_ms=200.0,
                pre_stim_ms=50.0,
                initial_settle_ms=50.0,
                seed=42,
                q10_factor_override=q10,
            )
        except TypeError:
            print(f"  ERROR: protocol doesn't accept q10_factor_override yet (need to add)")
            return

        rest = fi[0].mean_vm_pre_stim
        rates = {p.current_pA: len(p.spike_times_ms) for p in fi}
        any_spike = any(n > 0 for n in rates.values())
        max_vm = max(p.max_vm_during_stim for p in fi)

        line = (f"  {q10:>6.1f}  {rest:>8.2f}  "
                f"{rates.get(15e6, 0):>10d}  "
                f"{rates.get(30e6, 0):>10d}  "
                f"{rates.get(100e6, 0):>10d}  "
                f"{('Y' if any_spike else 'n'):>7s} (max_Vm={max_vm:.1f})")
        print(line, flush=True)
        result_dict = {
            "q10": q10, "rest_vm": float(rest), "max_vm": float(max_vm),
            "spikes_at_15e6": int(rates.get(15e6, 0)),
            "spikes_at_30e6": int(rates.get(30e6, 0)),
            "spikes_at_100e6": int(rates.get(100e6, 0)),
            "any_spike": bool(any_spike),
        }
        results.append(result_dict)
        # Save incrementally so partial results survive interruption
        with open(out_path, "w") as f:
            json.dump({"temperature": 37.0, "preset": preset_name,
                       "currents_pA": test_currents, "n_neurons": n_neurons,
                       "results": results}, f, indent=2)

    print(f"\n{'='*72}\n  CONSOLIDATED RESULT TABLE (also saved to {out_path})\n{'='*72}")
    print(f"  {'Q10':>6s}  {'pre_Vm':>8s}  {'15e6':>6s}  {'30e6':>6s}  {'100e6':>6s}  {'fires?':>6s}  {'max_Vm':>8s}")
    for r in results:
        print(f"  {r['q10']:>6.1f}  {r['rest_vm']:>8.2f}  "
              f"{r['spikes_at_15e6']:>6d}  {r['spikes_at_30e6']:>6d}  "
              f"{r['spikes_at_100e6']:>6d}  "
              f"{('Y' if r['any_spike'] else 'n'):>6s}  "
              f"{r['max_vm']:>8.2f}")


if __name__ == "__main__":
    main()
