"""g11 batched-replica runner (E.3, 2026-04-30).

Runs N independent g11 BG-cascade agents in a single Python process via
the block-diagonal embedding from sim/replicas.py. Replaces the
subprocess-parallelism approach (N independent runners, CPU-bound on
per-step Python orchestration) with one process processing all N
replicas via shared CuPy kernel dispatch.

Scope (v1):
- Multi-goal evaluation only. Skips pretraining, perception arc
  (sensed-reward, beacon, cue-reflex, landmark sensors), sleep replay,
  curriculum learning, interactive control. The next experiments
  (Cluster F v2, D v2) don't need any of these.
- Cluster flags supported: --enable-msn-lateral-inhibition,
  --enable-d1-d2-asymmetry, --enable-striatal-pv-fsi,
  --enable-cluster-a-closed-loop, --enable-cluster-e-topography,
  --enable-cluster-f-cerebellum, --enable-cluster-d-hippocampus,
  --enable-tonic-da, --enable-tans, --enable-bg-neuropeptides.
- Per-replica reward via bridge.cp_per_synapse_reward_override (E.3
  bridge-side change). Each replica's synapse block gets its own
  per-step reward.
- Output: N separate JSON files (one per replica) in the same shape
  as g11_bg_runner.py's output, so the existing aggregator can consume.

Acceptance criteria (per docs/plans/2026-04-29-g11-batched-replica-retrofit.md):
- Smoke: --n-replicas 2 --n-steps 60 --enable-cluster-f-cerebellum
  runs to completion, returncode 0.
- Acid test: --n-replicas 6 --seed 42 produces results within +/-15%
  of the equivalent 6-subprocess run (different per-seed RNG so exact
  match isn't expected).
- Speedup: at least 3x faster than 6 subprocesses (target 6x).

Usage:
    python -m research.runners.g11_bg_replicated_runner \\
        --seeds 42 43 44 100 101 102 \\
        --out-template g11_seedSEED_AEF_replicated.json \\
        --enable-msn-lateral-inhibition \\
        --enable-d1-d2-asymmetry \\
        --enable-striatal-pv-fsi \\
        --enable-cluster-a-closed-loop \\
        --enable-cluster-e-topography \\
        --enable-cluster-f-cerebellum \\
        --moving-goal --deterministic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np


# Action geometry — must match g11_bg_runner.py (NESW)
ACTION_NAMES = ["N", "E", "S", "W"]
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
N_ACTIONS = 4


def manhattan(a, b, c, d):
    return abs(a - c) + abs(b - d)


def _heuristic_action(x: int, y: int, gx: int, gy: int, rng: np.random.Generator) -> int:
    """Greedy heuristic that picks an action shrinking Manhattan distance.
    Tie-broken by RNG. Mirrors g11_bg_runner.py heuristic logic."""
    candidates = []
    if y < gy:
        candidates.append(0)  # N
    if x < gx:
        candidates.append(1)  # E
    if y > gy:
        candidates.append(2)  # S
    if x > gx:
        candidates.append(3)  # W
    if not candidates:
        return int(rng.integers(0, N_ACTIONS))
    if len(candidates) == 1:
        return candidates[0]
    return int(rng.choice(candidates))


def run_replicated_multi_goal(
    *,
    seeds: Sequence[int],
    out_path_template: str,
    n_steps: int = 1800,
    grid_size: int = 8,
    start_pos: tuple = (1, 1),
    moving_goal: bool = True,
    goal_schedule_kind: str = "multi",  # "default" | "multi" | "single"
    deterministic: bool = False,
    # Cluster flags
    enable_msn_lateral_inhibition: bool = False,
    enable_d1_d2_asymmetry: bool = False,
    enable_striatal_pv_fsi: bool = False,
    enable_cluster_a_closed_loop: bool = False,
    enable_cluster_e_topography: bool = False,
    enable_cluster_f_cerebellum: bool = False,
    enable_cluster_f_v2: bool = False,  # CF-gated LTD per Albus 1971 §IV.C
    enable_cluster_d_hippocampus: bool = False,
    enable_tonic_da: bool = False,
    enable_tans: bool = False,
    enable_bg_neuropeptides: bool = False,
    # Plasticity
    reward_learning_rate: float = 0.01,
    weight_jitter: float = 0.05,  # per-replica weight initial jitter
    # Pause-on-demand control file. When the file at this path exists and
    # contains JSON {"paused": true}, the runner sleeps at env-step boundaries
    # until the flag flips to false (or the file is deleted). Lets the user
    # pause without killing — e.g. to free the GPU for other work — and
    # resume later without losing progress. Default None disables polling.
    pause_flag_path: str | None = None,
    pause_poll_interval: float = 2.0,
    verbose: bool = True,
) -> dict:
    """Run N independent g11 multi-goal eval replicas in a single process.

    Returns a results dict with per-replica trajectories + aggregate
    summary stats. Writes N output files (one per replica) at
    out_path_template.replace('SEED', str(seed)).
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import cupy as cp

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.regions import RegionManager
    from sim.replicas import (
        ReplicaConfig, ReplicaManager, replicate_wiring_plan_with_seeds,
    )
    from sim.neuromodulators import (
        NeuromodulatorConfig, ProductionRule, ModulatorTarget,
    )
    from research.runners.g11_bg_runner import build_bg_brain_regions

    n_replicas = len(seeds)
    base_seed = int(seeds[0])

    # ---- 1. Build single-replica template ----
    # build_bg_brain_regions takes the wiring kwargs (cluster flags + FSIs +
    # lateral inhibition). D1/D2 asymmetry is a CFG flag, not a builder
    # kwarg — applied at bridge time.
    regions, pathways = build_bg_brain_regions(
        n_cortex=100,
        enable_striatal_fsis=enable_striatal_pv_fsi,
        enable_cluster_a_closed_loop=enable_cluster_a_closed_loop,
        enable_cluster_e_topography=enable_cluster_e_topography,
        enable_cluster_f_cerebellum=enable_cluster_f_cerebellum,
        enable_cluster_d_hippocampus=enable_cluster_d_hippocampus,
        enable_bg_lateral_inhibition=enable_msn_lateral_inhibition,
    )
    rmgr_template = RegionManager(regions, pathways)
    rmgr_template.initialize(seed=base_seed)
    template_plan = rmgr_template.build_wiring_plan(seed=base_seed)
    n_per_replica = rmgr_template.total_neurons()
    if verbose:
        print(f"[replicated] template: {len(regions)} regions, {len(pathways)} pathways, {n_per_replica} neurons/replica")

    # ---- 2. Replicate ----
    replicas = [
        ReplicaConfig(replica_id=i, seed_offset=base_seed * 1000 + int(s))
        for i, s in enumerate(seeds)
    ]
    replicated_plan = replicate_wiring_plan_with_seeds(
        template_plan,
        replicas=replicas,
        neurons_per_replica=n_per_replica,
        weight_jitter=weight_jitter,
    )
    rmgr = ReplicaManager(replicas, neurons_per_replica=n_per_replica)
    rmgr.initialize()
    total_neurons = rmgr.total_neurons()

    # ---- 3. Build CoreSimConfig ----
    cfg = CoreSimConfig(
        num_neurons=total_neurons,
        enable_brain_region_framework=False,  # we inject manually
    )
    cfg.dt_ms = 0.5
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = True
    cfg.enable_d1_d2_asymmetry = enable_d1_d2_asymmetry
    cfg.reward_learning_rate = reward_learning_rate
    cfg.reward_baseline = 0.0
    # Match g11_bg_runner's stdp_w_max idiom: density 0.20, weight 125 -> 150
    cfg.stdp_w_max = 200.0  # extra headroom for jitter tail
    cfg.hebbian_max_weight = 200.0
    # Disable structural plasticity in v1 — it grows nnz during the run,
    # which would require dynamic resizing of cp_per_synapse_reward_override.
    # The g11 evals don't use structural plasticity anyway (it's pretraining-only).
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = False  # eval runs don't use Hebbian
    cfg.enable_homeostasis = False
    cfg.enable_synaptic_scaling = False
    cfg.enable_ou_process = True
    cfg.enable_parameter_heterogeneity = True

    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)

    # Stash region_manager so the bridge can use it for d1_d2_sign tagging
    # in inject_explicit_wiring. This is the per-replica region_manager
    # (not template) — it covers ALL replicas' regions via the original
    # template's index space repeated N times.
    # IMPORTANT: the bridge's d1_d2_sign tagging walks region_manager.regions()
    # and tags synapses whose post is in str_D2_*. Under replication, str_D2_X
    # is the union of all replicas' str_D2_X blocks. Build a shadow
    # RegionManager that has the union indices for the bridge.
    # For now, set region_manager=None on the bridge and tag d1_d2_sign manually.
    sb.region_manager = None
    sb._initialize_simulation_data(called_from_playback_init=False)

    # ---- 4. Inject the replicated wiring ----
    sb.inject_explicit_wiring(replicated_plan)
    nnz = int(sb.cp_connections.nnz)

    # Manually tag D1/D2 sign across all replica blocks if asymmetry is on.
    if enable_d1_d2_asymmetry and nnz > 0:
        sb.cp_d1_d2_sign = cp.ones(nnz, dtype=cp.float32)
        d2_post_indices = []
        for r_cfg in replicas:
            shift = r_cfg.replica_id * n_per_replica
            for region in regions:
                if region.name.startswith("str_D2_"):
                    base_indices = rmgr_template.indices(region.name)
                    d2_post_indices.extend(idx + shift for idx in base_indices)
        if d2_post_indices:
            d2_set_gpu = cp.asarray(np.asarray(d2_post_indices, dtype=np.int64))
            d2_mask = cp.isin(sb.cp_connections.indices, d2_set_gpu)
            sb.cp_d1_d2_sign[d2_mask] = -1.0

    if verbose:
        print(f"[replicated] booted: {total_neurons} neurons, {nnz} synapses, {n_replicas} replicas")

    # ---- 5. Build per-replica region-index lookup ----
    # region_indices_cp[name] = cp.array of GLOBAL indices (union across
    # replicas, in replica-major order so block r's slice is at indices
    # [r*group_size, (r+1)*group_size)).
    region_indices_per_replica: dict[str, list[cp.ndarray]] = {}
    region_indices_union: dict[str, cp.ndarray] = {}
    for region in regions:
        per_replica_indices = []
        for r_cfg in replicas:
            shift = r_cfg.replica_id * n_per_replica
            base = rmgr_template.indices(region.name)
            per_replica_indices.append(cp.asarray([i + shift for i in base], dtype=cp.int64))
        region_indices_per_replica[region.name] = per_replica_indices
        region_indices_union[region.name] = cp.concatenate(per_replica_indices) if per_replica_indices else cp.zeros(0, dtype=cp.int64)

    # ---- 6. Per-replica state ----
    xs = [start_pos[0]] * n_replicas
    ys = [start_pos[1]] * n_replicas
    # Goal schedule
    if goal_schedule_kind == "single":
        goal_schedule = [(0, (6, 6))]
    elif goal_schedule_kind == "default":
        goal_schedule = [(0, (6, 6)), (300, (1, 6))]
    else:  # multi
        goal_schedule = [(0, (6, 6)), (450, (6, 1)), (900, (1, 1)), (1350, (1, 6))]
    goal_change_steps = [s for s, _ in goal_schedule[1:]]

    gxs = [goal_schedule[0][1][0]] * n_replicas
    gys = [goal_schedule[0][1][1]] * n_replicas

    trajectories = [[] for _ in range(n_replicas)]
    goal_log = [[] for _ in range(n_replicas)]
    action_log = [[] for _ in range(n_replicas)]
    reward_log = [[] for _ in range(n_replicas)]
    distance_log = [[] for _ in range(n_replicas)]
    motor_counts_total = [{a: 0 for a in ACTION_NAMES} for _ in range(n_replicas)]
    n_steps_at_goal = [0] * n_replicas

    rngs = [np.random.default_rng(int(s) * 13_417) for s in seeds]

    # ---- 7. Bake region-index handles for the per-step loop ----
    # For drive-setting we need, per replica, the cortex_X / motor_X / etc
    # index arrays (block-local, accessible by "r"). We'll iterate over replicas
    # in Python — fine for N <= 32.
    cortex_idx_per_replica = {a: [region_indices_per_replica[f"cortex_{a}"][r] for r in range(n_replicas)] for a in ACTION_NAMES}
    motor_idx_per_replica = {a: [region_indices_per_replica[f"motor_{a}"][r] for r in range(n_replicas)] for a in ACTION_NAMES}
    # Vectorized motor readout: build a 3D index tensor [N_REPLICAS, N_ACTIONS, n_motor_per_pool]
    # so we can gather + sum in ONE GPU kernel call per stim step instead of
    # 4×N_REPLICAS .get() round-trips per stim step.
    n_motor_per_pool = len(motor_idx_per_replica[ACTION_NAMES[0]][0])
    motor_indices_3d = cp.zeros((n_replicas, N_ACTIONS, n_motor_per_pool), dtype=cp.int64)
    for r in range(n_replicas):
        for ai, a in enumerate(ACTION_NAMES):
            idx_r_a = motor_idx_per_replica[a][r]
            motor_indices_3d[r, ai, :len(idx_r_a)] = idx_r_a
    # Cortex drive: build a 2D [N_REPLICAS, N_ACTIONS] table of cortex
    # block index ranges, so we can set drives in one vectorized op.
    n_cortex_per_pool = len(cortex_idx_per_replica[ACTION_NAMES[0]][0])
    cortex_indices_3d = cp.zeros((n_replicas, N_ACTIONS, n_cortex_per_pool), dtype=cp.int64)
    for r in range(n_replicas):
        for ai, a in enumerate(ACTION_NAMES):
            idx_r_a = cortex_idx_per_replica[a][r]
            cortex_indices_3d[r, ai, :len(idx_r_a)] = idx_r_a

    # Cluster F IO indices per replica (only when F is on)
    io_idx_per_replica = None
    if enable_cluster_f_cerebellum:
        io_idx_per_replica = [region_indices_per_replica["inferior_olive"][r] for r in range(n_replicas)]

    # Stimulus + readout window
    STIMULUS_MS = 100.0
    READOUT_START_MS = 30.0
    READOUT_END_MS = 100.0
    n_stim_steps = int(STIMULUS_MS / cfg.dt_ms)
    readout_start = int(READOUT_START_MS / cfg.dt_ms)
    readout_end = int(READOUT_END_MS / cfg.dt_ms)

    HEURISTIC_DRIVE_PA = cp.float32(800.0)
    BASELINE_DRIVES = {
        # (region prefix or name, drive_value)
    }

    def set_baseline_drives():
        """Set all per-step baseline drives (gpe, gpi, stn, snc, thal, etc.)
        across all replicas. The union-region indices broadcast the scalar
        value to all replicas' blocks at once."""
        sb.cp_external_input_current[:] = 0.0
        for prefix in ("gpe_",):
            for a in ACTION_NAMES:
                rn = f"{prefix}{a}"
                if rn in region_indices_union:
                    sb.cp_external_input_current[region_indices_union[rn]] = cp.float32(150.0)
        for prefix in ("gpe_arky_",):
            for a in ACTION_NAMES:
                rn = f"{prefix}{a}"
                if rn in region_indices_union:
                    sb.cp_external_input_current[region_indices_union[rn]] = cp.float32(120.0)
        for a in ACTION_NAMES:
            sb.cp_external_input_current[region_indices_union[f"gpi_{a}"]] = cp.float32(110.0)
        for rn in ("stn", "snc"):
            if rn in region_indices_union:
                sb.cp_external_input_current[region_indices_union[rn]] = cp.float32(150.0)
        for a in ACTION_NAMES:
            sb.cp_external_input_current[region_indices_union[f"thal_{a}"]] = cp.float32(300.0)
        # Cluster F baselines
        if enable_cluster_f_cerebellum:
            sb.cp_external_input_current[region_indices_union["inferior_olive"]] = cp.float32(80.0)
            for a in ACTION_NAMES:
                sb.cp_external_input_current[region_indices_union[f"dcn_aip_{a}"]] = cp.float32(180.0)
                sb.cp_external_input_current[region_indices_union[f"purkinje_{a}"]] = cp.float32(120.0)

    # Per-synapse reward override array
    sb.cp_per_synapse_reward_override = cp.zeros(nnz, dtype=cp.float32)
    # F v2: cache cerebellum_pf_pc gate mask for CF-gated LTD signaling.
    # When v2 is on, these synapses see -1.0 (LTD) when their replica's
    # reward is negative, 0 otherwise — decoupled from the global reward.
    cerebellum_pf_pc_mask = None
    if enable_cluster_f_v2 and enable_cluster_f_cerebellum:
        gate_to_syns = getattr(sb, "_plasticity_gate_to_synapses", {})
        cere_idx_list = gate_to_syns.get("cerebellum_pf_pc")
        if cere_idx_list:
            cerebellum_pf_pc_mask = cp.zeros(nnz, dtype=cp.bool_)
            cerebellum_pf_pc_mask[cp.asarray(np.asarray(cere_idx_list, dtype=np.int64))] = True
            if verbose:
                print(f"[replicated] F v2 enabled: {len(cere_idx_list)} cerebellum_pf_pc synapses tagged for CF-gated LTD")
        elif verbose:
            print("[replicated] WARNING: --enable-cluster-f-v2 set but no cerebellum_pf_pc gate found")
    # Per-replica synapse-index masks for the reward override
    pre_global = sb.cp_connections.indptr  # CSR row pointer; pre-neuron i has row [indptr[i]:indptr[i+1]]
    # Each synapse's "owner" is its pre-neuron's replica. Build a per-synapse replica-id array.
    # Synapse index s -> pre-neuron index (use indptr binary search). For block-diagonal
    # wiring all synapses with pre in [r*N, (r+1)*N) are replica r.
    pre_replica = cp.zeros(nnz, dtype=cp.int32)
    for r in range(n_replicas):
        # CSR indptr: synapses with pre in [r*N, (r+1)*N) are at indices
        # [indptr[r*N], indptr[(r+1)*N])
        s_start = int(pre_global[r * n_per_replica].get())
        s_end = int(pre_global[(r + 1) * n_per_replica].get())
        pre_replica[s_start:s_end] = r

    # ---- 8. Main eval loop ----
    t_start = time.time()
    current_phase_idx = 0
    bridge_step_count = 0
    if verbose:
        print(f"[replicated] starting {n_steps}-step eval, {n_replicas} replicas (n_per_replica={n_per_replica})")

    def _check_paused() -> None:
        """Poll pause_flag_path; sleep here if {"paused": true}."""
        if pause_flag_path is None:
            return
        printed = False
        while True:
            try:
                with open(pause_flag_path) as f:
                    state = json.load(f)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return
            if not state.get("paused"):
                if printed and verbose:
                    print(f"[replicated] resumed at env-step {step}", flush=True)
                return
            if not printed and verbose:
                print(f"[replicated] PAUSED at env-step {step} "
                      f"(touch {pause_flag_path} with paused=false to resume)",
                      flush=True)
                printed = True
            time.sleep(pause_poll_interval)

    for step in range(n_steps):
        # Pause check (does nothing if pause_flag_path is None)
        _check_paused()

        # Goal change (scheduled, applies to all replicas)
        while (current_phase_idx + 1 < len(goal_schedule)
               and step >= goal_schedule[current_phase_idx + 1][0]):
            current_phase_idx += 1
            new_goal = goal_schedule[current_phase_idx][1]
            for r in range(n_replicas):
                gxs[r] = new_goal[0]
                gys[r] = new_goal[1]
            if verbose:
                print(f"[replicated] step {step}: GOAL CHANGED to {new_goal}", flush=True)

        # Set baseline drives for all replicas
        set_baseline_drives()

        # Per-replica heuristic cortex drive
        for r in range(n_replicas):
            action_dir = _heuristic_action(xs[r], ys[r], gxs[r], gys[r], rngs[r])
            cortex_letter = ACTION_NAMES[action_dir]
            sb.cp_external_input_current[cortex_idx_per_replica[cortex_letter][r]] = HEURISTIC_DRIVE_PA

        # Run stimulus window with reward signal = 0 (eligibility builds up)
        cfg.current_reward_signal = 0.0
        # Reset per-synapse reward to 0 during stim window so no weight updates
        sb.cp_per_synapse_reward_override[:] = cp.float32(0.0)

        # Vectorized motor readout: accumulate firing counts on GPU, ONE
        # .get() call per stim window. counts_gpu shape: [N_REPLICAS, N_ACTIONS]
        counts_gpu = cp.zeros((n_replicas, N_ACTIONS), dtype=cp.int32)
        for s in range(n_stim_steps):
            sb._run_one_simulation_step()
            bridge_step_count += 1
            if readout_start <= s < readout_end:
                firing = sb.cp_firing_states.astype(cp.int32)
                # Gather firing at all motor indices: shape [N_REPLICAS, N_ACTIONS, n_motor_per_pool]
                fired_at_motor = firing[motor_indices_3d]
                counts_gpu = counts_gpu + fired_at_motor.sum(axis=-1)
        # ONE sync to CPU
        counts_cpu = counts_gpu.get()  # shape [N_REPLICAS, N_ACTIONS], int32

        # Per-replica action selection (CPU-side, fast)
        actions_this_step = []
        for r in range(n_replicas):
            row = counts_cpu[r]
            if int(row.max()) > 0:
                action_idx = int(row.argmax())
            else:
                action_idx = int(rngs[r].integers(0, N_ACTIONS))
            actions_this_step.append(action_idx)
            for ai, a in enumerate(ACTION_NAMES):
                motor_counts_total[r][a] += int(row[ai])

        # Apply action + reward per replica
        rewards_this_step = []
        for r in range(n_replicas):
            dist_before = manhattan(xs[r], ys[r], gxs[r], gys[r])
            dxa, dya = ACTION_DELTAS[actions_this_step[r]]
            new_x = int(np.clip(xs[r] + dxa, 0, grid_size - 1))
            new_y = int(np.clip(ys[r] + dya, 0, grid_size - 1))
            xs[r], ys[r] = new_x, new_y
            dist_after = manhattan(xs[r], ys[r], gxs[r], gys[r])
            if dist_after < dist_before:
                reward = 1.0
            elif dist_after > dist_before:
                reward = -1.0
            else:
                reward = 0.0
            rewards_this_step.append(reward)

            trajectories[r].append((xs[r], ys[r]))
            goal_log[r].append((gxs[r], gys[r]))
            action_log[r].append(int(actions_this_step[r]))
            reward_log[r].append(float(reward))
            distance_log[r].append(int(dist_after))
            if dist_after == 0:
                n_steps_at_goal[r] += 1

        # Apply per-synapse reward signal: each replica's synapses get its reward
        rewards_gpu = cp.asarray(np.asarray(rewards_this_step, dtype=np.float32))
        sb.cp_per_synapse_reward_override = rewards_gpu[pre_replica]

        # Cluster F v2: CF-gated LTD per Albus 1971 §IV.C eq.4. Cerebellum
        # synapses see -1.0 when their replica's reward<0 (CF event proxy),
        # 0.0 otherwise — decoupled from the global per-replica reward signal.
        if cerebellum_pf_pc_mask is not None:
            cf_per_replica = cp.where(rewards_gpu < 0, -1.0, 0.0).astype(cp.float32)
            cf_per_synapse = cf_per_replica[pre_replica]
            sb.cp_per_synapse_reward_override = cp.where(
                cerebellum_pf_pc_mask,
                cf_per_synapse,
                sb.cp_per_synapse_reward_override,
            )

        # Cluster F: bump IO drive for replicas with negative reward
        if enable_cluster_f_cerebellum:
            for r in range(n_replicas):
                if rewards_this_step[r] < 0:
                    sb.cp_external_input_current[io_idx_per_replica[r]] = cp.float32(450.0)

        # One reward-modulation step (the stim window had reward=0; now apply
        # per-replica reward). Use reward_signal=1.0 so the bridge's
        # multiplicative reward path remains active; the per-synapse override
        # carries the actual per-replica signed reward.
        cfg.current_reward_signal = 1.0
        sb._run_one_simulation_step()
        bridge_step_count += 1

    elapsed = time.time() - t_start
    if verbose:
        print(f"[replicated] done {n_steps} steps in {elapsed:.1f}s ({bridge_step_count/elapsed:.1f} bridge-steps/sec)")

    # ---- 9. Compute per-replica phase_stats ----
    out_paths = []
    for r in range(n_replicas):
        # Phase stats
        phase_stats = []
        for p_idx, (start_step, goal) in enumerate(goal_schedule):
            end_step = goal_schedule[p_idx + 1][0] if p_idx + 1 < len(goal_schedule) else n_steps
            phase_distances = distance_log[r][start_step:end_step]
            mean_d = float(np.mean(phase_distances)) if phase_distances else 0.0
            # finalQ = mean over last 25%
            quarter_start = start_step + int(0.75 * (end_step - start_step))
            final_q_distances = distance_log[r][quarter_start:end_step]
            final_q_mean = float(np.mean(final_q_distances)) if final_q_distances else 0.0
            n_at_goal = sum(1 for d in phase_distances if d == 0)
            ac_counts = [0, 0, 0, 0]
            for a in action_log[r][start_step:end_step]:
                ac_counts[a] += 1
            phase_stats.append({
                "phase": p_idx,
                "step_start": start_step,
                "step_end": end_step,
                "goal": list(goal),
                "mean_distance": round(mean_d, 4),
                "final_quarter_mean_distance": round(final_q_mean, 4),
                "n_steps_at_goal": n_at_goal,
                "n_steps": end_step - start_step,
                "action_counts": ac_counts,
            })

        result = {
            "seed": int(seeds[r]),
            "n_steps": n_steps,
            "grid_size": grid_size,
            "start_pos": list(start_pos),
            "goal_pos": list(goal_schedule[-1][1]),
            "goal_schedule": [[s, list(g)] for s, g in goal_schedule],
            "goal_change_steps": list(goal_change_steps),
            "phase_stats": phase_stats,
            "reward_learning_rate": reward_learning_rate,
            "trajectory": [[int(x), int(y)] for x, y in trajectories[r]],
            "goal_log": [[int(g[0]), int(g[1])] for g in goal_log[r]],
            "motor_counts": motor_counts_total[r],
            "action_log": action_log[r],
            "reward_log": reward_log[r],
            "distance_log": distance_log[r],
            "mean_distance_overall": round(float(np.mean(distance_log[r])) if distance_log[r] else 0.0, 4),
            "mean_distance_quarters": [
                round(float(np.mean(distance_log[r][i*n_steps//4:(i+1)*n_steps//4])), 4)
                for i in range(4)
            ] if distance_log[r] else [0, 0, 0, 0],
            "n_steps_at_goal": n_steps_at_goal[r],
            "elapsed_seconds": elapsed,
            "config_flags": [],
            "replicated_runner": True,
            "n_replicas": n_replicas,
        }
        out_path = out_path_template.replace("SEED", str(int(seeds[r])))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f)
        out_paths.append(out_path)
        if verbose:
            print(f"[replicated] saved seed={seeds[r]} -> {out_path}")

    return {
        "n_replicas": n_replicas,
        "n_steps": n_steps,
        "elapsed_seconds": elapsed,
        "out_paths": out_paths,
    }


def _emit_webapp_sidecar(args) -> None:
    """Write a `.cmd.json` sidecar in the format the webapp's
    /api/runs orphan-scan expects, so a raw-spawned replicated runner
    is visible in the dashboard's Live picker.

    Sidecar fields (mirror webapp/server.py launch_run sidecar):
    - run_id: short hex generated here
    - cmd: sys.argv (resolved python + args)
    - pid: this process
    - log_file: None — replicated runner doesn't redirect stdout, so the
      webapp displays the entry but can't tail it. Acceptable: pause and
      kill still work; progress events just won't stream to the picker.
    - control_file: pause-flag-path if any (so webapp shows pause button)
    - out_path: the resolved out-path of the FIRST replica seed (the
      sidecar lives next to it so the webapp's run-listing finds it).
    - started_at: now.

    File location: next to the first seed's output, named
    `<first_seed_basename>.cmd.json` so the webapp's RAW_RUNS_DIR.glob
    picks it up.
    """
    import uuid
    run_id = uuid.uuid4().hex[:12]
    # Resolve first seed's out path — same format string as run_replicated
    first_seed = args.seeds[0]
    out_path = args.out_template.replace("SEED", str(first_seed))
    if not os.path.isabs(out_path):
        # Webapp expects absolute; resolve relative to CWD just like
        # the webapp launcher does.
        out_path = os.path.abspath(out_path)
    sidecar_path = Path(out_path).with_suffix(".cmd.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "run_id": run_id,
        "preset": "replicated",  # synthetic; not a real preset
        "seed": first_seed,
        "extra_args": [a for a in sys.argv[1:] if a != "--emit-webapp-sidecar"],
        "deterministic": bool(args.deterministic),
        "cmd": [sys.executable, "-m", "research.runners.g11_bg_replicated_runner", *sys.argv[1:]],
        "pid": os.getpid(),
        "log_file": None,
        "control_file": args.pause_flag_path,
        "out_path": out_path,
        "started_at": time.time(),
        # Distinguishing tag so the webapp can render this differently
        # if desired (e.g. show "REPLICATED" badge). Unknown keys are
        # ignored by the existing recovery code.
        "runner_kind": "replicated",
        "n_replicas": len(args.seeds),
        "all_seeds": list(args.seeds),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[replicated_runner] webapp sidecar: {sidecar_path} (run_id={run_id} pid={os.getpid()})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="Per-replica seeds, one per replica.")
    ap.add_argument("--out-template", type=str, required=True,
                    help='Output filename template with SEED placeholder, e.g. "g11_seedSEED_AEF_replicated.json".')
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--start-x", type=int, default=1)
    ap.add_argument("--start-y", type=int, default=1)
    ap.add_argument("--moving-goal", action="store_true", default=True)
    ap.add_argument("--goal-schedule", type=str, default="multi",
                    choices=["default", "multi", "single"])
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--enable-msn-lateral-inhibition", "--bg-lateral-inhibition",
                    action="store_true", dest="msn_lat_inh")
    ap.add_argument("--enable-d1-d2-asymmetry", action="store_true")
    ap.add_argument("--enable-striatal-pv-fsi", "--enable-striatal-fsis",
                    action="store_true", dest="striatal_pv_fsi")
    ap.add_argument("--enable-cluster-a-closed-loop", action="store_true")
    ap.add_argument("--enable-cluster-e-topography", action="store_true")
    ap.add_argument("--enable-cluster-f-cerebellum", action="store_true")
    ap.add_argument("--enable-cluster-f-v2", action="store_true",
                    help="CF-gated anti-Hebbian LTD per Albus 1971 §IV.C eq.4. "
                         "Decouples cerebellum_pf_pc plasticity from global reward.")
    ap.add_argument("--enable-cluster-d-hippocampus", action="store_true")
    ap.add_argument("--reward-lr", type=float, default=0.01)
    ap.add_argument("--weight-jitter", type=float, default=0.05)
    # Pause-on-demand control file. Accepts BOTH names:
    # - --interactive-control-file: the canonical name used by g11_bg_runner.py
    #   and the webapp's launcher. Allows pause + (in non-replicated runs)
    #   goal override / reward injection.
    # - --pause-flag-path: replicated-runner-only legacy alias.
    # Both route to the same dest. While {"paused": true} the runner
    # sleeps at env-step boundaries until flipped to false or deleted.
    ap.add_argument("--interactive-control-file", "--pause-flag-path",
                    type=str, default=None, dest="pause_flag_path",
                    help='Path to a JSON control file. While {"paused": true}, '
                         'runner sleeps at env-step boundaries until flipped '
                         'to false or deleted. Lets you pause without losing '
                         'progress.')
    ap.add_argument("--pause-poll-interval", type=float, default=2.0,
                    help="How often (sec) to re-check pause_flag_path while paused.")
    ap.add_argument("-q", "--quiet", action="store_true")
    # Webapp discovery: when this runner is launched directly via the
    # terminal (not via the webapp's /api/runs/launch endpoint), the
    # webapp's "Live mode" run picker can't see it. Writing a sidecar
    # `.cmd.json` next to the per-seed output file with our PID lets
    # the webapp's periodic orphan-scan pick us up and surface the
    # process in the picker (where it can be killed or paused like any
    # other run). Off by default to keep raw `python -m` invocations
    # silent; enable for any run you want visible in the dashboard.
    ap.add_argument("--emit-webapp-sidecar", action="store_true",
                    help="Write a webapp-compatible sidecar so /api/runs "
                         "orphan-scan picks up this raw-spawned runner.")
    args = ap.parse_args()

    if args.emit_webapp_sidecar:
        _emit_webapp_sidecar(args)

    return 0 if run_replicated_multi_goal(
        seeds=args.seeds,
        out_path_template=args.out_template,
        n_steps=args.n_steps,
        grid_size=args.grid_size,
        start_pos=(args.start_x, args.start_y),
        moving_goal=args.moving_goal,
        goal_schedule_kind=args.goal_schedule,
        deterministic=args.deterministic,
        enable_msn_lateral_inhibition=args.msn_lat_inh,
        enable_d1_d2_asymmetry=args.enable_d1_d2_asymmetry,
        enable_striatal_pv_fsi=args.striatal_pv_fsi,
        enable_cluster_a_closed_loop=args.enable_cluster_a_closed_loop,
        enable_cluster_e_topography=args.enable_cluster_e_topography,
        enable_cluster_f_cerebellum=args.enable_cluster_f_cerebellum,
        enable_cluster_f_v2=args.enable_cluster_f_v2,
        enable_cluster_d_hippocampus=args.enable_cluster_d_hippocampus,
        reward_learning_rate=args.reward_lr,
        weight_jitter=args.weight_jitter,
        pause_flag_path=args.pause_flag_path,
        pause_poll_interval=args.pause_poll_interval,
        verbose=not args.quiet,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
