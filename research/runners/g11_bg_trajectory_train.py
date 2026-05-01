"""g11 trajectory training runner — imitation learning via STDP+reward.

Trains a g11 BG-cascade agent on pre-recorded expert trajectories instead
of running live RL. The trajectory is loaded from JSON; for each step,
the runner drives `cortex_X` corresponding to the recorded action and
sets the reward signal from the recorded reward, then runs the
simulation step. STDP + reward modulation update synaptic weights over
many epochs.

This is the imitation-learning analog of g11_bg_runner.py — bypasses
heuristic, bypasses BG action selection, imposes the expert trajectory.
The agent's BG cascade learns to associate (state → action) via plasticity
on imposed-action signals.

Usage:
    python -m research.runners.g11_bg_trajectory_train \\
        --trajectories research/datasets/expert_8x8_v1.json \\
        --n-epochs 10 \\
        --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry \\
        --enable-striatal-pv-fsi --enable-cluster-a-closed-loop \\
        --enable-cluster-e-topography \\
        --output-checkpoint research/checkpoints/imitation_8x8_v1.h5

Output: a runtime stats JSON next to --output-checkpoint, with per-epoch
mean reward and final firing rate distributions.

Eval: load the checkpoint into a live g11_bg_runner.py session
(--load-checkpoint) and run cheat-5 multi-goal det. Compare to
fresh-init agent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse builder + constants from g11_bg_runner
from research.runners.g11_bg_runner import (
    ACTION_NAMES,
    N_ACTIONS,
    build_bg_brain_regions,
)

# ACTION_DELTAS isn't exported by g11_bg_runner — define locally to match.
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def train_on_trajectories(
    trajectories_path: str,
    n_epochs: int = 10,
    output_checkpoint: str | None = None,
    output_stats: str | None = None,
    seed: int = 42,
    deterministic: bool = True,
    # Cluster flags (mirror g11_bg_runner)
    enable_msn_lateral_inhibition: bool = False,
    enable_d1_d2_asymmetry: bool = False,
    enable_striatal_pv_fsi: bool = False,
    enable_cluster_a_closed_loop: bool = False,
    enable_cluster_e_topography: bool = False,
    enable_cluster_f_cerebellum: bool = False,
    enable_cluster_f_v2: bool = False,
    enable_pfc: bool = False,
    enable_pfc_nmda: bool = False,
    reward_learning_rate: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Train the g11 BG-cascade agent on pre-recorded expert trajectories.

    Returns a dict with per-epoch stats."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import cupy as cp

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    from sim.regions import RegionManager

    # Load trajectories
    data = json.loads(Path(trajectories_path).read_text())
    trajectories = data["trajectories"]
    grid_size = data["grid_size"]
    if verbose:
        n_steps = sum(len(t["steps"]) for t in trajectories)
        print(f"[traj_train] loaded {len(trajectories)} trajectories "
              f"({n_steps} total steps) from {trajectories_path}")

    # Build regions + pathways via canonical builder
    regions, pathways = build_bg_brain_regions(
        n_cortex=100,
        enable_striatal_fsis=enable_striatal_pv_fsi,
        enable_cluster_a_closed_loop=enable_cluster_a_closed_loop,
        enable_cluster_e_topography=enable_cluster_e_topography,
        enable_cluster_f_cerebellum=enable_cluster_f_cerebellum,
        enable_bg_lateral_inhibition=enable_msn_lateral_inhibition,
    )
    rmgr = RegionManager(regions, pathways)
    rmgr.initialize(seed=seed)
    plan = rmgr.build_wiring_plan(seed=seed)
    n_neurons = rmgr.total_neurons()

    # Build CoreSimConfig (mirror g11_bg_runner.py)
    cfg = CoreSimConfig(num_neurons=n_neurons, enable_brain_region_framework=True)
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = reward_learning_rate
    cfg.reward_eligibility_tau_ms = 500.0
    _ctx_msn_density = 0.20
    _ctx_msn_weight = (25.0 / _ctx_msn_density) if _ctx_msn_density < 1.0 else 25.0
    cfg.stdp_w_max = max(30.0, _ctx_msn_weight * 1.2)
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_d1_d2_asymmetry = enable_d1_d2_asymmetry
    if enable_pfc_nmda:
        cfg.enable_nmda = True
        cfg.nmda_ratio = 0.5

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Pre-cache region indices
    region_indices_cp = {}
    for r in regions:
        idx = list(bridge.region_manager.indices(r.name))
        if idx:
            region_indices_cp[r.name] = cp.asarray(idx, dtype=cp.int64)

    # Stim window timing (mirrors g11_bg_runner)
    STIMULUS_MS = 100.0
    n_stim_steps = int(STIMULUS_MS / cfg.dt_ms)  # 100 sim steps per env step

    HEURISTIC_DRIVE_PA = cp.float32(800.0)

    # Training loop
    epoch_stats = []
    t_start = time.time()
    total_env_steps = 0

    for epoch in range(n_epochs):
        epoch_reward = 0.0
        epoch_steps = 0
        # Iterate trajectories in sequence
        for traj in trajectories:
            for step in traj["steps"]:
                action_idx = step["action"]
                reward = step["reward"]

                # Reset all cortex pool drives to 0, set the imposed action's pool
                # Plus baseline drives (gpe, gpi, stn, snc, thal) and cerebellum
                # — match g11_bg_runner.py's per-step setup
                bridge.cp_external_input_current[:] = 0.0
                for prefix in ["gpe_", "gpi_", "stn", "snc"]:
                    if prefix in ("stn", "snc"):
                        if prefix in region_indices_cp:
                            bridge.cp_external_input_current[region_indices_cp[prefix]] = cp.float32(150.0)
                    else:
                        for a in ACTION_NAMES:
                            rn = f"{prefix}{a}"
                            if rn in region_indices_cp:
                                bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(
                                    150.0 if prefix == "gpe_" else 110.0
                                )
                for a in ACTION_NAMES:
                    rn = f"thal_{a}"
                    if rn in region_indices_cp:
                        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)
                if enable_cluster_f_cerebellum:
                    if "inferior_olive" in region_indices_cp:
                        bridge.cp_external_input_current[region_indices_cp["inferior_olive"]] = cp.float32(80.0)
                    for a in ACTION_NAMES:
                        for rn_pre in ("dcn_aip_", "purkinje_"):
                            rn = f"{rn_pre}{a}"
                            if rn in region_indices_cp:
                                bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(
                                    180.0 if rn_pre == "dcn_aip_" else 120.0
                                )

                # Drive the imposed action's cortex pool
                cortex_letter = ACTION_NAMES[action_idx]
                cortex_key = f"cortex_{cortex_letter}"
                if cortex_key in region_indices_cp:
                    bridge.cp_external_input_current[region_indices_cp[cortex_key]] = HEURISTIC_DRIVE_PA

                # Set reward signal — STDP + reward modulation will fire
                # over the n_stim_steps window with this reward active.
                bridge.core_config.current_reward_signal = float(reward)

                # Run the stimulus window
                for _ in range(n_stim_steps):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    bridge.runtime_state.current_time_ms = (
                        bridge.runtime_state.current_time_step * cfg.dt_ms
                    )

                epoch_reward += reward
                epoch_steps += 1
                total_env_steps += 1

        avg_reward = epoch_reward / max(1, epoch_steps)
        epoch_stats.append({
            "epoch": epoch,
            "n_steps": epoch_steps,
            "total_reward": epoch_reward,
            "mean_reward_per_step": avg_reward,
        })
        if verbose:
            elapsed = time.time() - t_start
            print(f"[traj_train] epoch {epoch + 1}/{n_epochs}: "
                  f"{epoch_steps} steps, mean_reward={avg_reward:.3f}, "
                  f"elapsed={elapsed:.0f}s")

    # Save checkpoint if requested
    if output_checkpoint:
        try:
            out_path = Path(output_checkpoint)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bridge.save_checkpoint(str(out_path))
            if verbose:
                print(f"[traj_train] saved checkpoint: {out_path}")
        except Exception as e:
            if verbose:
                print(f"[traj_train] checkpoint save failed: {e}")

    # Save stats
    stats_out = {
        "trajectories_path": str(trajectories_path),
        "n_epochs": n_epochs,
        "seed": seed,
        "n_neurons": n_neurons,
        "n_synapses": int(bridge.cp_connections.nnz),
        "total_env_steps": total_env_steps,
        "epoch_stats": epoch_stats,
        "elapsed_seconds": time.time() - t_start,
    }
    if output_stats:
        Path(output_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(output_stats).write_text(json.dumps(stats_out, indent=2))
        if verbose:
            print(f"[traj_train] saved stats: {output_stats}")

    return stats_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", type=str, required=True,
                    help="Path to JSON trajectory file from generate_expert_trajectories.")
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-checkpoint", type=str, default=None,
                    help="Optional path to save final bridge state.")
    ap.add_argument("--output-stats", type=str, default=None,
                    help="Optional path to save per-epoch training stats.")
    ap.add_argument("--reward-lr", type=float, default=0.01)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("-q", "--quiet", action="store_true")
    # Cluster flags (subset of g11_bg_runner's, just the ones that work
    # without env interaction)
    ap.add_argument("--enable-msn-lateral-inhibition", action="store_true")
    ap.add_argument("--enable-d1-d2-asymmetry", action="store_true")
    ap.add_argument("--enable-striatal-pv-fsi", action="store_true")
    ap.add_argument("--enable-cluster-a-closed-loop", action="store_true")
    ap.add_argument("--enable-cluster-e-topography", action="store_true")
    ap.add_argument("--enable-cluster-f-cerebellum", action="store_true")
    ap.add_argument("--enable-cluster-f-v2", action="store_true")
    ap.add_argument("--enable-pfc", "--enable-dlpfc-wm", action="store_true",
                    dest="enable_pfc")
    ap.add_argument("--enable-pfc-nmda", action="store_true")
    args = ap.parse_args()

    train_on_trajectories(
        trajectories_path=args.trajectories,
        n_epochs=args.n_epochs,
        output_checkpoint=args.output_checkpoint,
        output_stats=args.output_stats,
        seed=args.seed,
        deterministic=args.deterministic,
        enable_msn_lateral_inhibition=args.enable_msn_lateral_inhibition,
        enable_d1_d2_asymmetry=args.enable_d1_d2_asymmetry,
        enable_striatal_pv_fsi=args.enable_striatal_pv_fsi,
        enable_cluster_a_closed_loop=args.enable_cluster_a_closed_loop,
        enable_cluster_e_topography=args.enable_cluster_e_topography,
        enable_cluster_f_cerebellum=args.enable_cluster_f_cerebellum,
        enable_cluster_f_v2=args.enable_cluster_f_v2,
        enable_pfc=args.enable_pfc,
        enable_pfc_nmda=args.enable_pfc_nmda,
        reward_learning_rate=args.reward_lr,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
