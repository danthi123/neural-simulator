"""Sanity-check: validate the HH ORIGINAL parameters (squid axon) which are
known-good textbook parameters. If THESE don't produce action potentials in
this simulator, the implementation has a bug. If they do, the L5 RS preset
parameters are misconfigured and the simulator's HH math is fine.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    from sim.enums import DefaultHodgkinHuxleyParams, NeuronType
    from research.validate._isolated_neuron import (
        run_step_current_protocol, summarize_fi_curve,
    )

    # HH_EXCITATORY_DEFAULT_LEGACY maps to ORIGINAL_HH_PARAMS in the enum's PARAMS dict
    preset_name = "HH_EXCITATORY_DEFAULT_LEGACY"
    hh_params = DefaultHodgkinHuxleyParams.get_params(NeuronType[preset_name])

    # Original HH at temp=6.3°C; the simulator default is 37°C with q10=3.0,
    # which would scale rates by 3^((37-6.3)/10) ≈ 25x. For original params to
    # work as published, we'd need temperature=6.3.
    # For now, run at 37°C and see what happens — this is the simulator's
    # default operating point.
    currents_pA = [0.0, 5e6, 15e6, 30e6, 50e6, 100e6, 500e6]
    n_neurons = 5

    print(f"\n{'='*72}")
    print(f"  Sanity check: {preset_name}")
    print(f"  HH params:")
    for k, v in hh_params.items():
        print(f"    {k:20s} = {v}")
    print(f"\n  F-I curve: {len(currents_pA)} levels, {n_neurons} neurons each")
    print(f"{'='*72}\n", flush=True)

    fi_curve = run_step_current_protocol(
        neuron_type_name=preset_name,
        current_steps_pA=currents_pA,
        n_neurons=n_neurons,
        dt_ms=0.01,
        temperature_celsius=6.3,  # Original HH was characterized at 6.3°C; running at
                                   # 37°C with Q10=3.0 over-compresses dynamics.
        stim_duration_ms=200.0,
        pre_stim_ms=50.0,
        initial_settle_ms=50.0,
        seed=42,
    )

    print(f"\n{'='*72}")
    print(f"  F-I curve")
    print(f"{'='*72}")
    print(f"  {'I (pA)':>10s}  {'rate_init':>10s}  {'rate_steady':>11s}  "
          f"{'pre_Vm':>8s}  {'max_Vm':>8s}  {'mean_Vm':>8s}  spikes")
    for p in fi_curve:
        n_spk = len(p.spike_times_ms)
        print(f"  {p.current_pA:>10.2g}  {p.rate_hz_initial:>10.2f}  "
              f"{p.rate_hz_steady:>11.2f}  {p.mean_vm_pre_stim:>8.2f}  "
              f"{p.max_vm_during_stim:>8.2f}  {p.mean_vm_during_stim:>8.2f}  "
              f"{n_spk:>5d}")


if __name__ == "__main__":
    main()
