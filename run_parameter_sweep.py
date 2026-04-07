#!/usr/bin/env python3
"""Parameter sweep framework for automated experiment analysis.

Runs an experiment across parameter ranges, collects results, and produces
a summary with statistical analysis.

Usage:
    python run_parameter_sweep.py --config sweep_config.json
    python run_parameter_sweep.py --experiment associative --sweep stdp_a_plus=0.005,0.01,0.02

Sweep configs are JSON files:
    {
        "experiment": "associative",
        "num_neurons": 10000,
        "num_trials": 50,
        "parameters": {
            "stdp_a_plus": [0.005, 0.01, 0.02, 0.04],
            "propagation_strength": [0.05, 0.10, 0.15]
        },
        "sweep_mode": "grid"  // "grid" (all combos) or "zip" (parallel)
    }
"""

import sys
import os
import json
import time
import csv
import argparse
import itertools
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp

from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from experiment import ExperimentEngine, ExperimentPresets


def create_sim_bridge(num_neurons, core_overrides=None):
    """Create and initialize a SimulationBridge with optional parameter overrides."""
    core_cfg = CoreSimConfig()
    core_cfg.num_neurons = num_neurons
    core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "CORTEX_L23_RS_FS"
    core_cfg.dt_ms = 1.0
    core_cfg.enable_hebbian_learning = True
    core_cfg.enable_stdp = True
    core_cfg.enable_short_term_plasticity = True
    core_cfg.enable_homeostasis = True
    core_cfg.enable_reward_modulation = True
    core_cfg.stdp_a_plus = 0.012
    core_cfg.stdp_a_minus = 0.01
    core_cfg.reward_learning_rate = 0.05

    # Apply parameter overrides to core_config
    if core_overrides:
        for key, val in core_overrides.items():
            if hasattr(core_cfg, key):
                setattr(core_cfg, key, val)

    sim_bridge = SimulationBridge(
        core_config=core_cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    dt = core_cfg.dt_ms
    sim_bridge.runtime_state.max_delay_steps = (
        int(core_cfg.max_synaptic_delay_ms / dt) if dt > 0 else 200
    )
    sim_bridge._initialize_simulation_data(called_from_playback_init=False)
    if not sim_bridge.is_initialized:
        return None, core_cfg, dt
    return sim_bridge, core_cfg, dt


def run_single_experiment(experiment_name, num_neurons, num_trials,
                          core_overrides=None):
    """Run one experiment and return structured results dict."""
    sim_bridge, core_cfg, dt = create_sim_bridge(num_neurons, core_overrides)
    if sim_bridge is None:
        return {"error": "Initialization failed"}

    # Load preset
    presets = {
        "associative": lambda: ExperimentPresets.associative_conditioning(
            num_trials=num_trials),
        "stimulus-response": lambda: ExperimentPresets.basic_stimulus_response(),
        "frequency-response": lambda: ExperimentPresets.frequency_response_characterization(
            num_frequencies=12, amplitude_pA=300.0),
        "reinforcement": lambda: ExperimentPresets.reinforcement_learning(
            num_trials=num_trials),
    }
    exp_config = presets[experiment_name]()

    # Setup engine
    engine = ExperimentEngine(core_cfg.num_neurons, dt)
    engine.load_experiment(exp_config)
    engine.initialize(cp_traits=sim_bridge.cp_traits, cp_module=cp)
    added = engine.ensure_inter_group_connectivity(sim_bridge, cp)
    sim_bridge.experiment_engine = engine

    total_exp_ms = sum(p.duration_ms * p.num_repetitions for p in exp_config.phases)
    total_steps = int(total_exp_ms / dt) + 2000

    # Run
    engine.start(current_time_ms=0.0, sim_bridge_ref=sim_bridge)
    t_start = time.time()

    for step in range(total_steps):
        sim_bridge._run_one_simulation_step()
        sim_bridge.runtime_state.current_time_step += 1
        sim_bridge.runtime_state.current_time_ms = (
            sim_bridge.runtime_state.current_time_step * dt
        )
        if engine.is_experiment_complete:
            break

    wall_time = time.time() - t_start

    # Extract metrics based on experiment type
    results = _extract_metrics(engine, experiment_name, wall_time)
    results["injected_connections"] = added
    results["total_synapses"] = sim_bridge.cp_connections.nnz

    # Cleanup GPU memory
    sim_bridge.clear_simulation_state_and_gpu_memory()
    cp.get_default_memory_pool().free_all_blocks()

    return results


def _extract_metrics(engine, experiment_name, wall_time):
    """Extract experiment-type-specific metrics from the engine log."""
    results = {"wall_time_s": round(wall_time, 1)}

    if experiment_name == "associative":
        pre_on, post_on = [], []
        weights = []
        for entry in engine.log:
            ev = entry.get("event", "")
            if ev == "readout":
                cs = entry.get("rates", {}).get("cs_input", 0)
                us = entry.get("rates", {}).get("us_output", 0)
                phase = entry.get("phase", "")
                if cs > 20:
                    if phase == "pre_test":
                        pre_on.append(us)
                    elif phase == "post_test":
                        post_on.append(us)
            elif ev == "intergroup_weights":
                weights.append(entry)

        pre_a = np.array(pre_on) if pre_on else np.array([0.0])
        post_a = np.array(post_on) if post_on else np.array([0.0])
        delta = float(post_a.mean() - pre_a.mean())
        se = np.sqrt(pre_a.var() / max(len(pre_a), 1)
                     + post_a.var() / max(len(post_a), 1))
        t_stat = delta / se if se > 0 else 0.0
        cohens_d = delta / np.sqrt((pre_a.var() + post_a.var()) / 2) if (pre_a.var() + post_a.var()) > 0 else 0.0

        w_pre = next((w for w in weights if "training" in w.get("label", "")), {})
        w_post = next((w for w in weights if "post" in w.get("label", "")), {})

        results.update({
            "pre_test_hz": round(float(pre_a.mean()), 3),
            "post_test_hz": round(float(post_a.mean()), 3),
            "delta_hz": round(delta, 3),
            "t_statistic": round(t_stat, 3),
            "cohens_d": round(float(cohens_d), 3),
            "p_significant": abs(t_stat) > 2.0,
            "n_pre": len(pre_on),
            "n_post": len(post_on),
            "weight_pre": w_pre.get("mean_weight", 0),
            "weight_post": w_post.get("mean_weight", 0),
            "learning_detected": delta > 2.0 and abs(t_stat) > 2.0,
        })

    elif experiment_name == "stimulus-response":
        bl_out, st_out = [], []
        for entry in engine.log:
            if entry.get("event") != "readout":
                continue
            out = entry.get("rates", {}).get("output", 0)
            phase = entry.get("phase", "")
            if phase == "baseline":
                bl_out.append(out)
            elif phase == "stimulus":
                st_out.append(out)

        bl_a = np.array(bl_out) if bl_out else np.array([0.0])
        st_a = np.array(st_out) if st_out else np.array([0.0])
        delta = float(st_a.mean() - bl_a.mean())
        se = np.sqrt(bl_a.var() / max(len(bl_a), 1)
                     + st_a.var() / max(len(st_a), 1))
        t_stat = delta / se if se > 0 else 0.0

        results.update({
            "baseline_hz": round(float(bl_a.mean()), 3),
            "stimulus_hz": round(float(st_a.mean()), 3),
            "delta_hz": round(delta, 3),
            "t_statistic": round(t_stat, 3),
            "p_significant": abs(t_stat) > 2.0,
        })

    elif experiment_name == "frequency-response":
        baseline_net, freq_data = [], {}
        for entry in engine.log:
            if entry.get("event") != "readout":
                continue
            net = entry.get("rates", {}).get("network", 0)
            phase = entry.get("phase", "")
            if phase in ("baseline", "post"):
                baseline_net.append(net)
            elif phase.startswith("freq_"):
                freq_data.setdefault(phase, []).append(net)

        bl_mean = float(np.mean(baseline_net)) if baseline_net else 0.0
        freq_results = []
        for phase_name in sorted(freq_data.keys(),
                                  key=lambda x: float(x.split('_')[1].replace('hz', ''))):
            freq_hz = float(phase_name.split('_')[1].replace('hz', ''))
            net_arr = np.array(freq_data[phase_name])
            freq_results.append({
                "freq_hz": freq_hz,
                "network_mean": round(float(net_arr.mean()), 3),
                "net_delta": round(float(net_arr.mean() - bl_mean), 3),
            })

        if freq_results:
            deltas = [f["net_delta"] for f in freq_results]
            peak_idx = int(np.argmax(deltas))
            trough = min(d for d in deltas if d > 0) if any(d > 0 for d in deltas) else 0.001
            peak = max(deltas)
            results.update({
                "baseline_hz": round(bl_mean, 3),
                "peak_freq_hz": freq_results[peak_idx]["freq_hz"],
                "peak_delta_hz": round(peak, 3),
                "trough_delta_hz": round(trough, 3),
                "peak_trough_ratio": round(peak / trough, 3) if trough > 0 else 0,
                "response_range_hz": round(max(deltas) - min(deltas), 3),
                "freq_profile": freq_results,
            })

    elif experiment_name == "reinforcement":
        trials_data = engine.training.trials_data if engine.training else []
        if trials_data:
            n = min(20, len(trials_data))
            early = trials_data[:n]
            late = trials_data[-n:]
            early_sr = sum(1 for t in early if t.get("success")) / len(early)
            late_sr = sum(1 for t in late if t.get("success")) / len(late)
            early_rate = np.mean([t.get("output_rate", 0) for t in early])
            late_rate = np.mean([t.get("output_rate", 0) for t in late])
            results.update({
                "total_trials": len(trials_data),
                "early_success_rate": round(early_sr, 3),
                "late_success_rate": round(late_sr, 3),
                "success_improvement": round(late_sr - early_sr, 3),
                "early_rate_hz": round(float(early_rate), 3),
                "late_rate_hz": round(float(late_rate), 3),
                "learning_detected": late_sr > early_sr + 0.1,
            })
        else:
            results.update({"total_trials": 0, "learning_detected": False})

    return results


def generate_sweep_combinations(parameters, mode="grid"):
    """Generate parameter combinations from a sweep config."""
    keys = list(parameters.keys())
    values = [parameters[k] for k in keys]

    if mode == "grid":
        combos = list(itertools.product(*values))
    elif mode == "zip":
        combos = list(zip(*values))
    else:
        raise ValueError(f"Unknown sweep mode: {mode}")

    return [dict(zip(keys, combo)) for combo in combos]


def run_sweep(config):
    """Execute a full parameter sweep and return results."""
    experiment = config["experiment"]
    num_neurons = config.get("num_neurons", 10000)
    num_trials = config.get("num_trials", 50)
    parameters = config.get("parameters", {})
    mode = config.get("sweep_mode", "grid")

    combos = generate_sweep_combinations(parameters, mode)
    total = len(combos)

    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP: {experiment.upper()}")
    print(f"{'='*70}")
    print(f"  Parameters: {list(parameters.keys())}")
    print(f"  Values: {parameters}")
    print(f"  Mode: {mode} ({total} combinations)")
    print(f"  Neurons: {num_neurons}, Trials: {num_trials}")
    print(f"{'='*70}")

    all_results = []
    t_sweep_start = time.time()

    for i, combo in enumerate(combos):
        print(f"\n--- Run {i+1}/{total}: {combo} ---")

        result = run_single_experiment(
            experiment, num_neurons, num_trials,
            core_overrides=combo,
        )
        result["params"] = combo
        result["run_index"] = i
        all_results.append(result)

        # Print key metric
        if "delta_hz" in result:
            sig = "***" if result.get("p_significant", False) else ""
            print(f"    -> delta={result['delta_hz']:+.2f} Hz "
                  f"t={result.get('t_statistic', 0):.2f} {sig}")
        elif "learning_detected" in result:
            print(f"    -> learning={'YES' if result['learning_detected'] else 'NO'}")

    sweep_time = time.time() - t_sweep_start

    # Summary
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY ({total} runs in {sweep_time:.0f}s)")
    print(f"{'='*70}")

    # Print table header
    param_keys = list(parameters.keys())
    header = "  " + "  ".join(f"{k:>12s}" for k in param_keys)
    if experiment == "associative":
        header += f"  {'delta_hz':>10s}  {'t_stat':>8s}  {'cohens_d':>9s}  {'sig':>5s}"
    elif experiment == "stimulus-response":
        header += f"  {'delta_hz':>10s}  {'t_stat':>8s}  {'sig':>5s}"
    elif experiment == "frequency-response":
        header += f"  {'peak_freq':>10s}  {'pk/tr':>8s}  {'range':>8s}"
    elif experiment == "reinforcement":
        header += f"  {'early_sr':>10s}  {'late_sr':>10s}  {'improve':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in all_results:
        row = "  " + "  ".join(f"{r['params'].get(k, ''):>12}" for k in param_keys)
        if experiment == "associative":
            sig = "YES" if r.get("p_significant") else "no"
            row += (f"  {r.get('delta_hz', 0):>+10.2f}"
                    f"  {r.get('t_statistic', 0):>8.2f}"
                    f"  {r.get('cohens_d', 0):>9.2f}"
                    f"  {sig:>5s}")
        elif experiment == "stimulus-response":
            sig = "YES" if r.get("p_significant") else "no"
            row += (f"  {r.get('delta_hz', 0):>+10.2f}"
                    f"  {r.get('t_statistic', 0):>8.2f}"
                    f"  {sig:>5s}")
        elif experiment == "frequency-response":
            row += (f"  {r.get('peak_freq_hz', 0):>10.1f}"
                    f"  {r.get('peak_trough_ratio', 0):>8.2f}"
                    f"  {r.get('response_range_hz', 0):>8.2f}")
        elif experiment == "reinforcement":
            row += (f"  {r.get('early_success_rate', 0):>10.1%}"
                    f"  {r.get('late_success_rate', 0):>10.1%}"
                    f"  {r.get('success_improvement', 0):>+10.1%}")
        print(row)

    return all_results, sweep_time


def main():
    parser = argparse.ArgumentParser(description="Parameter Sweep Runner")
    parser.add_argument("--config", type=str, help="Path to sweep config JSON")
    parser.add_argument("--experiment", "-e", type=str,
                        choices=["associative", "stimulus-response",
                                 "frequency-response", "reinforcement"],
                        help="Experiment type (if not using --config)")
    parser.add_argument("--sweep", type=str,
                        help="Inline sweep: 'param=v1,v2,v3' (repeatable)",
                        action="append", default=[])
    parser.add_argument("--num-neurons", type=int, default=10000)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Build config from args or file
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    elif args.experiment and args.sweep:
        parameters = {}
        for s in args.sweep:
            key, vals = s.split("=")
            parameters[key] = [float(v) if "." in v else int(v) for v in vals.split(",")]
        config = {
            "experiment": args.experiment,
            "num_neurons": args.num_neurons,
            "num_trials": args.num_trials,
            "parameters": parameters,
            "sweep_mode": "grid",
        }
    else:
        parser.error("Provide --config or (--experiment + --sweep)")
        return

    print("Loading simulator packages...")
    t0 = time.time()
    # Packages are already imported at module level; this just reports timing
    print(f"Loaded in {time.time() - t0:.1f}s")

    results, sweep_time = run_sweep(config)

    # Save results
    output_path = args.output or f"sweep_{config['experiment']}_{int(time.time())}.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "sweep_time_s": round(sweep_time, 1),
        "num_runs": len(results),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Also write CSV for easy spreadsheet analysis
    csv_path = output_path.replace(".json", ".csv")
    if results:
        flat_keys = set()
        for r in results:
            flat_keys.update(r.get("params", {}).keys())
            flat_keys.update(k for k in r.keys()
                             if k not in ("params", "freq_profile"))
        flat_keys = sorted(flat_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, r.get("params", {}).get(k, ""))
                       for k in flat_keys}
                writer.writerow(row)
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
