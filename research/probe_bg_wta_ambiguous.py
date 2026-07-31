"""Test winner-take-all selection under ambiguous cortex drive.

Drives cortex_N and cortex_E EQUALLY at 800 pA. Without lateral inhibition,
both motor_N and motor_E should fire at similar rates → no clear winner.
With WTA, one should dominate via FS-mediated cross-pool inhibition.

Usage:
    python -m research.probe_bg_wta_ambiguous           # WTA off
    python -m research.probe_bg_wta_ambiguous --wta     # WTA on
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cupy as cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wta", action="store_true", help="Enable motor lateral inhibition")
    ap.add_argument("--asym", action="store_true", help="Drive cortex_N=850, cortex_E=750 (slight asymmetry)")
    ap.add_argument("--motor-to-fs", type=float, default=10.0)
    ap.add_argument("--fs-to-motor", type=float, default=5.0)
    args = ap.parse_args()

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES

    regions, pathways = build_bg_brain_regions(
        n_cortex=100,
        enable_motor_lateral_inhibition=args.wta,
        motor_to_fs_weight=getattr(args, "motor_to_fs", 10.0),
        fs_to_motor_weight=getattr(args, "fs_to_motor", 5.0),
    )
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.stdp_w_max = 30.0

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # 50-step warmup
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    region_idx = {r.name: list(bridge.region_manager.indices(r.name))
                  for r in regions if list(bridge.region_manager.indices(r.name))}
    region_idx_cp = {k: cp.asarray(v, dtype=cp.int64) for k, v in region_idx.items()}
    motor_arr = {a: np.array(region_idx[f"motor_{a}"]) for a in ACTION_NAMES}
    fs_arr = ({a: np.array(region_idx[f"motor_FS_{a}"]) for a in ACTION_NAMES}
              if args.wta else {})

    bridge.cp_external_input_current[:] = 0.0
    for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(150.0)
    for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(110.0)
    for rn in ["stn", "dopamine"]:
        bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(150.0)
    for rn in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(300.0)

    # AMBIGUOUS DRIVE: cortex_N and cortex_E both at given pA (default both 800).
    # Pass --asym to test slight imbalance (cortex_N 850, cortex_E 750).
    drive_n = 850.0 if getattr(args, "asym", False) else 800.0
    drive_e = 750.0 if getattr(args, "asym", False) else 800.0
    bridge.cp_external_input_current[region_idx_cp["cortex_N"]] = cp.float32(drive_n)
    bridge.cp_external_input_current[region_idx_cp["cortex_E"]] = cp.float32(drive_e)

    print(f"\nAmbiguous drive probe (WTA={'ON' if args.wta else 'OFF'})")
    print(f"  cortex_N={drive_n}pA, cortex_E={drive_e}pA, all others=0\n")

    n_steps = 500
    bin_size = 50
    n_bins = n_steps // bin_size

    if args.wta:
        print(f"  Per-50ms bin: [motor_N, motor_E, motor_S, motor_W] / [FS_N, FS_E, FS_S, FS_W]")
    else:
        print(f"  Per-50ms bin motor counts [N, E, S, W]:")
    total = {a: 0 for a in ACTION_NAMES}
    fs_total = {a: 0 for a in ACTION_NAMES}
    for bin_idx in range(n_bins):
        bin_counts = {a: 0 for a in ACTION_NAMES}
        bin_fs = {a: 0 for a in ACTION_NAMES}
        for s in range(bin_size):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states.get().astype(bool)
            for a in ACTION_NAMES:
                bin_counts[a] += int(firing[motor_arr[a]].sum())
                total[a] += int(firing[motor_arr[a]].sum())
                if args.wta:
                    bin_fs[a] += int(firing[fs_arr[a]].sum())
                    fs_total[a] += int(firing[fs_arr[a]].sum())
        if args.wta:
            print(f"  Bin {bin_idx:2d} (steps {bin_idx*bin_size:3d}-{(bin_idx+1)*bin_size:3d}): "
                  f"motor={[bin_counts[a] for a in ACTION_NAMES]}  "
                  f"FS={[bin_fs[a] for a in ACTION_NAMES]}")
        else:
            print(f"  Bin {bin_idx:2d} (steps {bin_idx*bin_size:3d}-{(bin_idx+1)*bin_size:3d}): "
                  f"{[bin_counts[a] for a in ACTION_NAMES]}")

    print(f"\n  Total motor: N={total['N']}, E={total['E']}, S={total['S']}, W={total['W']}")
    if args.wta:
        print(f"  Total FS:    N={fs_total['N']}, E={fs_total['E']}, S={fs_total['S']}, W={fs_total['W']}")
    if total['N'] + total['E'] > 0:
        ratio_max_to_other = max(total['N'], total['E']) / max(1, min(total['N'], total['E']))
        print(f"  N/E asymmetry (max/min of driven pools): {ratio_max_to_other:.2f}x")
    print(f"  Off-target (S+W) leak: {total['S'] + total['W']}")


if __name__ == "__main__":
    main()
