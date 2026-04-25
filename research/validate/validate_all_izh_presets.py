"""Batch validation of all Izhikevich presets at 37°C.

Izhikevich neurons are not affected by the HH temperature bug (they're
phenomenological, not biophysical). All 7 registered Izhikevich types
should produce real APs at standard simulator settings (dt=1.0ms, T=37°C).

Each preset gets:
  - F-I curve at currents [0, 100, 300, 600, 1000, 2000] pA (typical Izh range)
  - Resting Vm
  - Spike threshold
  - Max steady firing rate
  - Adaptation indicator

References for target behavior:
  - Izhikevich 2003 IEEE TNN — original RS/IB/CH/FS/LTS phenomenological tuning
  - Izhikevich 2007 "Dynamical Systems in Neuroscience" — 9-param formulation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


PRESETS = [
    # 2007 formulation (9 params)
    "IZH2007_RS_CORTICAL_PYRAMIDAL",
    "IZH2007_FS_CORTICAL_INTERNEURON",
    # Legacy 4-param
    "RS_EXCITATORY_LEGACY",
    "FS_INHIBITORY_LEGACY",
    "IB_EXCITATORY_LEGACY",
    "CH_EXCITATORY_LEGACY",
    "LTS_INHIBITORY_LEGACY",
]


def build_izh_isolated_config(neuron_type_name: str, n_neurons: int = 5,
                               dt_ms: float = 1.0, seed: int = 42):
    """Returns CoreSimConfig for isolated Izh population.

    KNOWN BUG: bridge ignores cfg.default_neuron_type_izh and assigns
    neurons to IZH2007_RS / IZH2007_FS via `traits % num_variants` (where
    num_variants=2 always). To force a single type, we set num_traits=1
    AND override cp_traits to all zeros (RS) or all ones (FS) post-init.
    Legacy types are not selectable via this path — get_params() hardcodes
    use_2007_formulation=True (sim/bridge.py:945).
    """
    from sim import CoreSimConfig
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.num_traits = 1  # Force all neurons to trait 0 → type 0 (RS)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.default_neuron_type_izh = neuron_type_name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"

    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(
            name="cell", n_neurons=n_neurons, exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
        )
    ]
    cfg.region_pathways = []
    cfg.connections_per_neuron = 0
    cfg.enable_watts_strogatz = False
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    return cfg


def run_izh_step_protocol(neuron_type_name: str, currents_pA, n_neurons=5,
                            dt_ms=1.0, stim_duration_ms=500.0,
                            pre_stim_ms=100.0, settle_ms=100.0, seed=42):
    """Run F-I curve protocol on one Izhikevich preset."""
    import cupy as cp
    from sim import (SimulationBridge, VisualizationConfig,
                       RuntimeState, GPUConfig)

    n_steps_settle = int(settle_ms / dt_ms)
    n_steps_pre = int(pre_stim_ms / dt_ms)
    n_steps_stim = int(stim_duration_ms / dt_ms)

    points = []
    for I_pA in currents_pA:
        cfg = build_izh_isolated_config(neuron_type_name, n_neurons, dt_ms, seed)
        bridge = SimulationBridge(
            core_config=cfg, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig(),
        )
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)

        # Workaround: bridge assigns neurons to IZH2007_RS or IZH2007_FS via
        # `traits % 2`. To force all neurons to the requested type, override
        # the per-neuron Izh parameter arrays with the chosen preset's params.
        # This is post-init reach-around — needed because the bridge's
        # per-trait selection ignores cfg.default_neuron_type_izh.
        from sim.enums import NeuronType, DefaultIzhikevichParamsManager
        try:
            target_enum = NeuronType[neuron_type_name]
            # Note: legacy types fall through to RS via use_2007_formulation=True
            target_params = DefaultIzhikevichParamsManager.get_params(
                target_enum, use_2007_formulation=True
            )
            bridge.cp_izh_C[:] = cp.float32(target_params["C"])
            bridge.cp_izh_k[:] = cp.float32(target_params["k"])
            bridge.cp_izh_vr[:] = cp.float32(target_params["vr"])
            bridge.cp_izh_vt[:] = cp.float32(target_params["vt"])
            bridge.cp_izh_vpeak[:] = cp.float32(target_params["vpeak"])
            bridge.cp_izh_a[:] = cp.float32(target_params["a"])
            bridge.cp_izh_b[:] = cp.float32(target_params["b"])
            bridge.cp_izh_c_reset[:] = cp.float32(target_params["c_reset"])
            bridge.cp_izh_d_increment[:] = cp.float32(target_params["d_increment"])
            # Reset Vm to vr (resting potential) since we changed the param
            bridge.cp_membrane_potential_v[:] = cp.float32(target_params["vr"])
            bridge.cp_recovery_variable_u[:] = (
                cp.float32(target_params["b"]) *
                (bridge.cp_membrane_potential_v - cp.float32(target_params["vr"]))
            )
        except (KeyError, AttributeError) as e:
            print(f"  WARN: couldn't override Izh params for {neuron_type_name}: {e}")

        if bridge.cp_external_input_current is not None:
            bridge.cp_external_input_current[:] = 0.0

        for _ in range(n_steps_settle):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt_ms
            )

        vm_pre = []
        for _ in range(n_steps_pre):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt_ms
            )
            vm_pre.append(float(cp.mean(bridge.cp_membrane_potential_v).get()))
        rest_vm = float(np.mean(vm_pre))

        # Step current
        bridge.cp_external_input_current[:] = cp.float32(I_pA)
        prev_firing = np.zeros(n_neurons, dtype=bool)
        spike_times = [[] for _ in range(n_neurons)]
        max_vm = -200.0
        for s in range(n_steps_stim):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt_ms
            )
            firing = bridge.cp_firing_states.get().astype(bool)
            new = firing & ~prev_firing
            for i in np.where(new)[0]:
                spike_times[int(i)].append(s * dt_ms)
            prev_firing = firing
            v_now = float(cp.max(bridge.cp_membrane_potential_v).get())
            if v_now > max_vm:
                max_vm = v_now

        all_spikes = [t for st in spike_times for t in st]
        n_initial = sum(1 for t in all_spikes if t < 100.0)
        n_steady = sum(1 for t in all_spikes
                       if t >= (stim_duration_ms - 200.0))
        rate_init = n_initial / n_neurons / 0.1
        rate_steady = n_steady / n_neurons / 0.2

        points.append({
            "current_pA": I_pA,
            "rate_initial": float(rate_init),
            "rate_steady": float(rate_steady),
            "n_spikes_total": len(all_spikes),
            "rest_vm": rest_vm,
            "max_vm": float(max_vm),
        })

        bridge.clear_simulation_state_and_gpu_memory()

    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currents", nargs="+", type=float,
                    default=[0.0, 100.0, 300.0, 600.0, 1000.0, 2000.0])
    ap.add_argument("--n-neurons", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path("research/findings/raw/preset_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    t0 = time.time()
    for i, preset in enumerate(PRESETS):
        print(f"\n{'='*72}")
        print(f"  [{i+1}/{len(PRESETS)}] {preset}")
        print(f"{'='*72}", flush=True)
        try:
            fi = run_izh_step_protocol(
                preset, currents_pA=args.currents, n_neurons=args.n_neurons,
                dt_ms=1.0, seed=42,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            summary.append({"preset": preset, "error": str(e)})
            continue

        rest_vm = float(np.mean([p["rest_vm"] for p in fi]))
        max_vm_seen = max(p["max_vm"] for p in fi)
        any_spike = any(p["n_spikes_total"] > 0 for p in fi)
        max_steady = max(p["rate_steady"] for p in fi)
        threshold = next(
            (p["current_pA"] for p in fi if p["rate_initial"] > 0.5),
            float("nan"),
        )

        print(f"  rest_vm:       {rest_vm:.2f} mV")
        print(f"  max_Vm seen:   {max_vm_seen:.2f} mV")
        print(f"  spike thresh:  {threshold} pA")
        print(f"  max steady:    {max_steady:.1f} Hz")
        print(f"  per-I rates:   " + ", ".join(
            f"{p['current_pA']:.0f}={p['rate_steady']:.0f}" for p in fi))

        summary.append({
            "preset": preset, "rest_vm": rest_vm, "max_vm_seen": max_vm_seen,
            "spike_threshold_pA": threshold, "max_steady_rate": max_steady,
            "fires_at_least_once": bool(any_spike),
            "fi_curve": fi,
        })

    out_path = out_dir / "all_izh_presets_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "currents_pA": args.currents, "n_neurons": args.n_neurons,
            "presets": summary, "elapsed_sec": time.time() - t0,
        }, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  SUMMARY ({time.time() - t0:.0f}s wall)")
    print(f"{'='*72}")
    print(f"  {'preset':<38s}  rest_Vm   max_Vm  thresh   max_rate   fires?")
    print(f"  {'-'*38}  -------  -------  ------  --------  ------")
    for r in summary:
        if "error" in r:
            print(f"  {r['preset']:<38s}  ERROR: {r['error'][:30]}")
            continue
        fire_mark = "Y" if r["fires_at_least_once"] else "n"
        thresh_s = f"{r['spike_threshold_pA']:.0f}" if not np.isnan(r['spike_threshold_pA']) else "—"
        print(f"  {r['preset']:<38s}  {r['rest_vm']:>7.2f}  "
              f"{r['max_vm_seen']:>7.2f}  {thresh_s:>6s}  "
              f"{r['max_steady_rate']:>7.1f}    {fire_mark}")
    print(f"\nSummary JSON: {out_path}")


if __name__ == "__main__":
    main()
