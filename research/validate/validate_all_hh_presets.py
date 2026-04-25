"""Batch validation of all 17 HH presets at 6.3°C base temperature.

WHY 6.3°C: see research/findings/2026-04-25-hh-temperature-bug.md. At 37°C
with the simulator's uniform Q10=3.0, AP dynamics over-compress and spikes
don't fire. Until per-gate Q10 is implemented, validation must use the
base temperature.

Each preset gets:
  - F-I curve at currents [0, 1e7, 3e7, 1e8, 3e8] pA (covers sub- through
    suprathreshold density range)
  - Resting Vm
  - Spike threshold (smallest current eliciting >=1 spike)
  - Max Vm during stim (whether AP upstroke reaches v_peak)
  - Adaptation indicator (initial vs steady rate)

Outputs a single summary JSON + a markdown table for the findings doc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# All 17 HH presets in priority order:
# 1. Cortical (most-used) — pyramidals, FS interneuron
# 2. Subcortical control — striatum, GPe, STN, dopamine
# 3. Memory — hippocampus
# 4. Thalamus
# 5. Cerebellum, spinal, olfactory
PRESETS = [
    # name, expected_role
    "HH_L5_CORTICAL_PYRAMIDAL_RS",      # base, regular spiking
    "HH_PFC_PYRAMIDAL",                 # PFC with NaP
    "HH_CORTICAL_FS_INTERNEURON",       # PV+ fast-spiking, no adaptation
    "HH_CA1_PYRAMIDAL_BURST",           # hippocampus, mod CaT/Ih
    "HH_CA3_PYRAMIDAL_BURST",           # hippocampus, high CaT
    "HH_STRIATAL_MSN",                  # MSN, M-current dominant
    "HH_GPE_PACEMAKER",                 # tonic pacemaker
    "HH_STN_BURST",                     # bursty pacemaker
    "HH_DOPAMINE_SNC",                  # DA neuron
    "HH_THALAMIC_RELAY_TBURST",         # thalamic relay with strong CaT/Ih
    "HH_TRN_BURST_INHIB",               # reticular nucleus, very high CaT
    "HH_INFERIOR_OLIVE",                # IO with subthreshold osc
    "HH_CEREBELLAR_PURKINJE",           # complex spiking
    "HH_CEREBELLAR_GRANULE",            # compact, low Cm
    "HH_SPINAL_MOTOR",                  # motor neuron, plateau NaP
    "HH_SPINAL_INTERNEURON",            # spinal interneuron, no NaP
    "HH_OLFACTORY_MITRAL",              # mitral cell, high g_Na
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currents", nargs="+", type=float,
                    default=[0.0, 1e7, 3e7, 1e8, 3e8],
                    help="Step currents (pA) to test")
    ap.add_argument("--n-neurons", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=6.3)
    ap.add_argument("--single", type=str, default=None,
                    help="Validate just this preset (for debugging)")
    args = ap.parse_args()

    from research.validate._isolated_neuron import (
        run_step_current_protocol, summarize_fi_curve,
    )

    presets_to_run = [args.single] if args.single else PRESETS
    out_dir = Path("research/findings/raw/preset_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    t0 = time.time()
    for i, preset_name in enumerate(presets_to_run):
        print(f"\n{'='*76}")
        print(f"  [{i+1}/{len(presets_to_run)}] {preset_name} at {args.temperature}°C")
        print(f"{'='*76}", flush=True)

        try:
            fi_curve = run_step_current_protocol(
                neuron_type_name=preset_name,
                current_steps_pA=args.currents,
                n_neurons=args.n_neurons,
                dt_ms=0.01,
                temperature_celsius=args.temperature,
                stim_duration_ms=300.0,
                pre_stim_ms=50.0,
                initial_settle_ms=50.0,
                seed=42,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            summary.append({
                "preset": preset_name,
                "error": str(e),
            })
            continue

        metrics = summarize_fi_curve(fi_curve)
        # Per-current details
        max_vm_per_I = {p.current_pA: p.max_vm_during_stim for p in fi_curve}
        spikes_per_I = {p.current_pA: len(p.spike_times_ms) for p in fi_curve}
        rate_steady_per_I = {p.current_pA: p.rate_hz_steady for p in fi_curve}

        print(f"  rest_vm:       {metrics['rest_vm']:.2f} mV")
        print(f"  spike_thresh:  {metrics.get('spike_threshold_pA'):.2g} pA")
        print(f"  max steady rate: {metrics['max_steady_rate']:.1f} Hz")
        print(f"  per-I max_Vm: " + ", ".join(
            f"{I:.0e}={V:.1f}" for I, V in max_vm_per_I.items()))
        print(f"  per-I spikes: " + ", ".join(
            f"{I:.0e}={n}" for I, n in spikes_per_I.items()))

        any_spike = any(n > 0 for n in spikes_per_I.values())
        max_vm_seen = max(p.max_vm_during_stim for p in fi_curve)
        ap_emerges = max_vm_seen >= 40.0  # crosses default v_peak

        summary.append({
            "preset": preset_name,
            "rest_vm": metrics["rest_vm"],
            "spike_threshold_pA": metrics.get("spike_threshold_pA"),
            "max_steady_rate": metrics["max_steady_rate"],
            "max_vm_seen": float(max_vm_seen),
            "ap_emerges": bool(ap_emerges),
            "fires_at_least_once": bool(any_spike),
            "max_vm_per_I": {f"{I:.0e}": float(V) for I, V in max_vm_per_I.items()},
            "spikes_per_I": {f"{I:.0e}": int(n) for I, n in spikes_per_I.items()},
            "rate_steady_per_I": {f"{I:.0e}": float(r) for I, r in rate_steady_per_I.items()},
        })

    out_path = out_dir / "all_hh_presets_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "temperature_celsius": args.temperature,
            "currents_pA": args.currents,
            "n_neurons": args.n_neurons,
            "presets": summary,
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)

    print(f"\n{'='*76}")
    print(f"  SUMMARY ({time.time() - t0:.0f}s wall)")
    print(f"{'='*76}")
    print(f"\n  {'preset':<35s}  {'rest_Vm':>8s}  {'maxVm':>8s}  AP?  fires?")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  ---  ------")
    for r in summary:
        if "error" in r:
            print(f"  {r['preset']:<35s}  ERROR: {r['error'][:30]}")
            continue
        ap_mark = "Y" if r["ap_emerges"] else "n"
        fire_mark = "Y" if r["fires_at_least_once"] else "n"
        print(f"  {r['preset']:<35s}  {r['rest_vm']:>8.2f}  "
              f"{r['max_vm_seen']:>8.2f}   {ap_mark}     {fire_mark}")
    print(f"\nSummary JSON: {out_path}")


if __name__ == "__main__":
    main()
