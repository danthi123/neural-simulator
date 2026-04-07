#!/usr/bin/env python3
"""Headless runner for all experiment presets.

Drives experiments without the GUI, following the same initialization path
as the queue-based UI handler.

Usage:
    python run_experiment_headless.py --experiment associative [--num-trials N]
    python run_experiment_headless.py --experiment stimulus-response
    python run_experiment_headless.py --experiment frequency-response
    python run_experiment_headless.py --experiment reinforcement [--num-trials N]
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp
import cupyx.scipy.sparse as csp


def load_simulator():
    """Load the neural-simulator module."""
    simulator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural-simulator.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("neural_simulator", simulator_path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [simulator_path]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


def create_sim_bridge(mod, num_neurons=10000):
    """Create and initialize a SimulationBridge."""
    core_cfg = mod.CoreSimConfig()
    core_cfg.num_neurons = num_neurons
    core_cfg.neuron_model_type = mod.NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "CORTEX_L23_RS_FS"
    core_cfg.dt_ms = 1.0
    core_cfg.enable_hebbian_learning = True
    core_cfg.enable_stdp = True
    core_cfg.enable_short_term_plasticity = True
    core_cfg.enable_homeostasis = True
    core_cfg.enable_reward_modulation = True
    core_cfg.stdp_a_plus = 0.012
    core_cfg.stdp_a_minus = 0.01
    core_cfg.reward_learning_rate = 0.05  # Faster RL convergence for experiment timescales

    viz_cfg = mod.VisualizationConfig()
    runtime_state = mod.RuntimeState()
    gpu_cfg = mod.GPUConfig()

    sim_bridge = mod.SimulationBridge(
        core_config=core_cfg, viz_config=viz_cfg,
        runtime_state=runtime_state, gpu_config=gpu_cfg,
    )

    dt = core_cfg.dt_ms
    runtime_state.max_delay_steps = int(core_cfg.max_synaptic_delay_ms / dt) if dt > 0 else 200

    sim_bridge._initialize_simulation_data(called_from_playback_init=False)
    if not sim_bridge.is_initialized:
        print("ERROR: Simulation initialization failed!")
        sys.exit(1)

    return sim_bridge, core_cfg, dt


def run_experiment(sim_bridge, engine, exp_config, dt, total_exp_ms):
    """Run an experiment to completion and return wall time + step count."""
    total_steps = int(total_exp_ms / dt) + 2000

    engine.start(current_time_ms=0.0, sim_bridge_ref=sim_bridge)

    t_start = time.time()
    last_report = t_start
    report_interval = 10.0
    step = 0

    for step in range(total_steps):
        sim_bridge._run_one_simulation_step()
        sim_bridge.runtime_state.current_time_step += 1
        sim_bridge.runtime_state.current_time_ms = sim_bridge.runtime_state.current_time_step * dt

        if engine.is_experiment_complete:
            print(f"\n    Experiment completed at step {step} "
                  f"({sim_bridge.runtime_state.current_time_ms:.0f} ms)")
            break

        now = time.time()
        if now - last_report >= report_interval:
            elapsed = now - t_start
            current_ms = sim_bridge.runtime_state.current_time_ms
            pct = min(100, (current_ms / total_exp_ms) * 100)
            phase_name = "?"
            if engine.phases and engine.current_phase_idx < len(engine.phases):
                phase_name = engine.phases[engine.current_phase_idx].name
            eta = (total_steps - step) * (elapsed / (step + 1))
            print(f"    [{pct:5.1f}%] t={current_ms:.0f}ms | phase={phase_name} | "
                  f"{(elapsed/(step+1))*1000:.2f}ms/step | ETA {eta:.0f}s")
            last_report = now

    wall_time = time.time() - t_start
    print(f"\n    Wall time: {wall_time:.1f}s | Avg: {(wall_time/max(1,step+1))*1000:.2f} ms/step")
    return wall_time, step


def setup_engine(mod, sim_bridge, exp_config, dt):
    """Create experiment engine, inject connectivity, attach to bridge."""
    n = sim_bridge.core_config.num_neurons
    engine = mod.ExperimentEngine(n, dt)
    engine.load_experiment(exp_config)
    engine.initialize(cp_traits=sim_bridge.cp_traits, cp_module=cp)

    added = engine.ensure_inter_group_connectivity(sim_bridge, cp)
    if added > 0:
        print(f"    Injected {added} inter-group connections")
    print(f"    Total synapses: {sim_bridge.cp_connections.nnz:,}")

    sim_bridge.experiment_engine = engine

    total_exp_ms = sum(p.duration_ms * p.num_repetitions for p in exp_config.phases)
    print(f"    Phases: {len(exp_config.phases)}")
    for p in exp_config.phases:
        chans = p.active_channels if p.active_channels else []
        print(f"      - {p.name}: {p.duration_ms:.0f}ms x {p.num_repetitions} reps, "
              f"channels={chans}")
    print(f"    Total experiment time: {total_exp_ms:.0f} ms ({total_exp_ms/1000:.1f}s)")

    return engine, total_exp_ms


# ============================================================
# Experiment-specific runners
# ============================================================

def run_stimulus_response(mod, args):
    """Basic Stimulus-Response: inject current, measure I/O transfer function."""
    print("\n[2/5] Creating SimulationBridge...")
    sim_bridge, core_cfg, dt = create_sim_bridge(mod, args.num_neurons)

    print("\n[3/5] Loading Basic Stimulus-Response preset...")
    exp_config = mod.ExperimentPresets.basic_stimulus_response(
        input_amplitude_pA=150.0,
        stimulus_duration_ms=500.0,
        num_trials=20,
        input_group_size=100,
        output_group_size=100,
    )
    engine, total_exp_ms = setup_engine(mod, sim_bridge, exp_config, dt)

    print("\n[4/5] Running experiment...")
    wall_time, _ = run_experiment(sim_bridge, engine, exp_config, dt, total_exp_ms)

    # --- Analysis ---
    print("\n[5/5] Analyzing results...")
    baseline_input, baseline_output = [], []
    stimulus_input, stimulus_output = [], []
    post_input, post_output = [], []

    for entry in engine.log:
        if entry.get("event") != "readout":
            continue
        rates = entry.get("rates", {})
        inp = rates.get("input", 0.0)
        out = rates.get("output", 0.0)
        phase = entry.get("phase", "")

        if phase == "baseline":
            baseline_input.append(inp)
            baseline_output.append(out)
        elif phase == "stimulus":
            stimulus_input.append(inp)
            stimulus_output.append(out)
        elif phase == "post":
            post_input.append(inp)
            post_output.append(out)

    def stats(arr):
        a = np.array(arr) if arr else np.array([0.0])
        return a.mean(), a.std(), len(a)

    bl_in_m, bl_in_s, bl_in_n = stats(baseline_input)
    bl_out_m, bl_out_s, bl_out_n = stats(baseline_output)
    st_in_m, st_in_s, st_in_n = stats(stimulus_input)
    st_out_m, st_out_s, st_out_n = stats(stimulus_output)
    po_in_m, po_in_s, po_in_n = stats(post_input)
    po_out_m, po_out_s, po_out_n = stats(post_output)

    # Welch's t-test: stimulus output vs baseline output
    delta_out = st_out_m - bl_out_m
    se = np.sqrt(bl_out_s**2/max(bl_out_n,1) + st_out_s**2/max(st_out_n,1)) if (bl_out_n > 0 and st_out_n > 0) else 1.0
    t_stat = delta_out / se if se > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"BASIC STIMULUS-RESPONSE RESULTS")
    print(f"{'='*60}")
    print(f"                    Input (Hz)          Output (Hz)")
    print(f"  Baseline:     {bl_in_m:6.2f} ± {bl_in_s:5.2f} (n={bl_in_n:3d})   "
          f"{bl_out_m:6.2f} ± {bl_out_s:5.2f} (n={bl_out_n:3d})")
    print(f"  Stimulus:     {st_in_m:6.2f} ± {st_in_s:5.2f} (n={st_in_n:3d})   "
          f"{st_out_m:6.2f} ± {st_out_s:5.2f} (n={st_out_n:3d})")
    print(f"  Post:         {po_in_m:6.2f} ± {po_in_s:5.2f} (n={po_in_n:3d})   "
          f"{po_out_m:6.2f} ± {po_out_s:5.2f} (n={po_out_n:3d})")
    print(f"\n  Input delta (stim - baseline):  {st_in_m - bl_in_m:+.2f} Hz")
    print(f"  Output delta (stim - baseline): {delta_out:+.2f} Hz  (t={t_stat:.2f})")
    print(f"  Post recovery (post ~ baseline): {po_out_m - bl_out_m:+.2f} Hz")

    success = (st_in_m - bl_in_m > 5.0) and (abs(t_stat) > 2.0 or delta_out > 1.0)
    print(f"\n  Input responds to stimulus: {'YES' if st_in_m - bl_in_m > 5.0 else 'NO'}")
    print(f"  Output response significant: {'YES' if abs(t_stat) > 2.0 else 'NO'} (|t|>2)")
    print(f"  Post returns to baseline: {'YES' if abs(po_out_m - bl_out_m) < 2.0 else 'NO'}")
    print(f"{'='*60}")

    output_path = args.output or f"experiment_stimulus_response_{int(time.time())}.json"
    engine.save_log(output_path)
    print(f"  Log saved to: {output_path}")
    return success


def run_associative_conditioning(mod, args):
    """Associative Conditioning (CS-US Pairing)."""
    print("\n[2/5] Creating SimulationBridge...")
    sim_bridge, core_cfg, dt = create_sim_bridge(mod, args.num_neurons)

    print("\n[3/5] Loading Associative Conditioning preset...")
    exp_config = mod.ExperimentPresets.associative_conditioning(
        cs_amplitude_pA=500.0, us_amplitude_pA=500.0,
        cs_us_delay_ms=100.0, num_trials=args.num_trials,
        input_group_size=100, output_group_size=100,
    )
    engine, total_exp_ms = setup_engine(mod, sim_bridge, exp_config, dt)

    print("\n[4/5] Running experiment...")
    wall_time, _ = run_experiment(sim_bridge, engine, exp_config, dt, total_exp_ms)

    # --- Analysis ---
    print("\n[5/5] Analyzing results...")
    pre_on, post_on = [], []
    weight_snapshots = []

    for entry in engine.log:
        ev = entry.get("event", "")
        if ev == "readout":
            cs = entry.get("rates", {}).get("cs_input", 0)
            us = entry.get("rates", {}).get("us_output", 0)
            phase = entry.get("phase", "")
            if cs > 20:
                if phase == "pre_test": pre_on.append(us)
                elif phase == "post_test": post_on.append(us)
        elif ev == "intergroup_weights":
            weight_snapshots.append(entry)

    pre_a, post_a = np.array(pre_on or [0.0]), np.array(post_on or [0.0])
    delta = post_a.mean() - pre_a.mean()
    se = np.sqrt(pre_a.var()/max(len(pre_a),1) + post_a.var()/max(len(post_a),1))
    t_stat = delta / se if se > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"ASSOCIATIVE CONDITIONING RESULTS")
    print(f"{'='*60}")
    print(f"  CS-ON Pre:  {pre_a.mean():.2f} ± {pre_a.std():.2f} Hz (n={len(pre_on)})")
    print(f"  CS-ON Post: {post_a.mean():.2f} ± {post_a.std():.2f} Hz (n={len(post_on)})")
    print(f"  Delta:      {delta:+.2f} Hz  (t={t_stat:.2f})")
    for ws in weight_snapshots:
        print(f"  Weights [{ws['label']}]: mean={ws['mean_weight']:.4f} n={ws['n_connections']}")
    success = delta > 2.0 and t_stat > 2.0
    print(f"\n  Learning detected: {'YES' if success else 'NO'}")
    print(f"{'='*60}")

    output_path = args.output or f"experiment_associative_{int(time.time())}.json"
    engine.save_log(output_path)
    print(f"  Log saved to: {output_path}")
    return success


def run_frequency_response(mod, args):
    """Frequency Response Characterization: sinusoidal sweep."""
    print("\n[2/5] Creating SimulationBridge...")
    sim_bridge, core_cfg, dt = create_sim_bridge(mod, args.num_neurons)

    print("\n[3/5] Loading Frequency Response preset...")
    exp_config = mod.ExperimentPresets.frequency_response_characterization(
        freq_start_hz=1.0, freq_end_hz=100.0,
        num_frequencies=12,  # Reduced for faster runs
        duration_per_freq_ms=2000.0,
        amplitude_pA=300.0,  # 300 pA: strong enough to modulate vs OU noise sigma=100 pA
        input_group_size=200,
    )
    engine, total_exp_ms = setup_engine(mod, sim_bridge, exp_config, dt)

    print("\n[4/5] Running experiment...")
    wall_time, _ = run_experiment(sim_bridge, engine, exp_config, dt, total_exp_ms)

    # --- Analysis ---
    print("\n[5/5] Analyzing results...")

    # Collect rates per frequency phase
    freq_data = {}  # phase_name -> {"input": [...], "network": [...]}
    baseline_net = []

    for entry in engine.log:
        if entry.get("event") != "readout":
            continue
        rates = entry.get("rates", {})
        phase = entry.get("phase", "")
        inp = rates.get("input", 0.0)
        net = rates.get("network", 0.0)

        if phase == "baseline" or phase == "post":
            baseline_net.append(net)
        elif phase.startswith("freq_"):
            if phase not in freq_data:
                freq_data[phase] = {"input": [], "network": []}
            freq_data[phase]["input"].append(inp)
            freq_data[phase]["network"].append(net)

    bl_mean = np.mean(baseline_net) if baseline_net else 0.0

    print(f"\n{'='*65}")
    print(f"FREQUENCY RESPONSE CHARACTERIZATION RESULTS")
    print(f"{'='*65}")
    print(f"  Baseline network rate: {bl_mean:.2f} Hz")
    print(f"\n  {'Frequency':>12s}  {'Input Hz':>10s}  {'Network Hz':>12s}  {'Net Delta':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}")

    freq_results = []
    for phase_name in sorted(freq_data.keys(), key=lambda x: float(x.split('_')[1].replace('hz',''))):
        freq_hz = float(phase_name.split('_')[1].replace('hz', ''))
        inp_arr = np.array(freq_data[phase_name]["input"])
        net_arr = np.array(freq_data[phase_name]["network"])
        net_delta = net_arr.mean() - bl_mean
        freq_results.append({"freq_hz": freq_hz, "input_mean": inp_arr.mean(),
                             "network_mean": net_arr.mean(), "net_delta": net_delta,
                             "network_std": net_arr.std()})
        print(f"  {freq_hz:10.1f}Hz  {inp_arr.mean():10.2f}  {net_arr.mean():12.2f}  {net_delta:+10.2f}")

    # Check for frequency-dependent response (not flat)
    if freq_results:
        net_means = [r["network_mean"] for r in freq_results]
        response_range = max(net_means) - min(net_means)
        peak_freq = freq_results[np.argmax(net_means)]["freq_hz"]
        trough = min(r["net_delta"] for r in freq_results)
        peak = max(r["net_delta"] for r in freq_results)
        ratio = peak / trough if trough > 0 else float('inf')
        print(f"\n  Response range: {response_range:.2f} Hz (max - min across frequencies)")
        print(f"  Peak/trough ratio: {ratio:.2f}x")
        print(f"  Peak response at: {peak_freq:.1f} Hz")
        # Bandpass detection: ratio > 1.5 means meaningful frequency selectivity
        is_bandpass = ratio > 1.5 and response_range > 0.3
        print(f"  Bandpass filter: {'YES' if is_bandpass else 'NO'} (ratio > 1.5x)")
    print(f"{'='*65}")

    output_path = args.output or f"experiment_freq_response_{int(time.time())}.json"
    engine.save_log(output_path)
    with open(output_path.replace('.json', '_freq_data.json'), 'w') as f:
        json.dump(freq_results, f, indent=2)
    print(f"  Log saved to: {output_path}")
    if freq_results:
        trough = min(r["net_delta"] for r in freq_results)
        peak = max(r["net_delta"] for r in freq_results)
        ratio = peak / trough if trough > 0 else float('inf')
        return ratio > 1.5 and response_range > 0.3
    return False


def run_reinforcement_learning(mod, args):
    """Reinforcement Learning (R-STDP): three-factor learning."""
    print("\n[2/5] Creating SimulationBridge...")
    sim_bridge, core_cfg, dt = create_sim_bridge(mod, args.num_neurons)

    print("\n[3/5] Loading Reinforcement Learning preset...")
    exp_config = mod.ExperimentPresets.reinforcement_learning(
        stimulus_amplitude_pA=120.0,
        num_trials=args.num_trials,
        input_group_size=100,
        output_group_size=50,
    )
    engine, total_exp_ms = setup_engine(mod, sim_bridge, exp_config, dt)

    print("\n[4/5] Running experiment...")
    wall_time, _ = run_experiment(sim_bridge, engine, exp_config, dt, total_exp_ms)

    # --- Analysis ---
    print("\n[5/5] Analyzing results...")

    baseline_rates = []
    training_trials = []
    post_test_rates = []

    for entry in engine.log:
        ev = entry.get("event", "")
        if ev == "readout":
            rates = entry.get("rates", {})
            phase = entry.get("phase", "")
            resp = rates.get("response", 0.0)
            stim = rates.get("stimulus", 0.0)
            if phase == "baseline":
                baseline_rates.append(resp)
            elif phase == "post_test":
                post_test_rates.append({"stim": stim, "resp": resp})

    # Extract per-trial data from training engine
    trials_data = engine.training.trials_data if engine.training else []

    print(f"\n{'='*60}")
    print(f"REINFORCEMENT LEARNING (R-STDP) RESULTS")
    print(f"{'='*60}")

    bl_mean = np.mean(baseline_rates) if baseline_rates else 0.0
    tc = engine.training.config if engine.training else None
    tmin = tc.target_min_rate_hz if tc else 0
    tmax = tc.target_max_rate_hz if tc else 0
    print(f"  Baseline response rate: {bl_mean:.2f} Hz")
    print(f"  Target window: {tmin}-{tmax} Hz")
    print(f"  Total trials: {len(trials_data)}")

    if trials_data:
        # Early vs late trial comparison
        n_early = min(20, len(trials_data))
        n_late = min(20, len(trials_data))
        early = trials_data[:n_early]
        late = trials_data[-n_late:]

        early_rates = [t.get("output_rate", 0) for t in early]
        late_rates = [t.get("output_rate", 0) for t in late]
        early_success = sum(1 for t in early if t.get("success", False)) / len(early)
        late_success = sum(1 for t in late if t.get("success", False)) / len(late)

        print(f"\n  Early trials (1-{n_early}):")
        print(f"    Response rate: {np.mean(early_rates):.2f} ± {np.std(early_rates):.2f} Hz")
        print(f"    Success rate:  {early_success*100:.0f}%")
        print(f"  Late trials ({len(trials_data)-n_late+1}-{len(trials_data)}):")
        print(f"    Response rate: {np.mean(late_rates):.2f} ± {np.std(late_rates):.2f} Hz")
        print(f"    Success rate:  {late_success*100:.0f}%")
        print(f"  Improvement:     {late_success*100 - early_success*100:+.0f}% success rate")

    if post_test_rates:
        post_resp = np.array([r["resp"] for r in post_test_rates])
        in_window = np.sum((post_resp >= 15) & (post_resp <= 40)) / len(post_resp) * 100
        print(f"\n  Post-test response: {post_resp.mean():.2f} ± {post_resp.std():.2f} Hz")
        print(f"  Post-test in target window: {in_window:.0f}%")

    success = False
    if trials_data:
        success = late_success > early_success + 0.1  # At least 10% improvement
    print(f"\n  Learning detected: {'YES' if success else 'NO'}")
    print(f"{'='*60}")

    output_path = args.output or f"experiment_rl_{int(time.time())}.json"
    engine.save_log(output_path)
    print(f"  Log saved to: {output_path}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Headless Experiment Runner")
    parser.add_argument("--experiment", "-e", required=True,
                        choices=["stimulus-response", "associative", "frequency-response",
                                 "reinforcement", "all"],
                        help="Which experiment to run")
    parser.add_argument("--num-trials", type=int, default=100, help="Training trials (learning exps)")
    parser.add_argument("--num-neurons", type=int, default=10000, help="Total neuron count")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print(f"HEADLESS EXPERIMENT RUNNER: {args.experiment.upper()}")
    print("=" * 70)

    print("\n[1/5] Loading simulator module...")
    t0 = time.time()
    mod = load_simulator()
    print(f"    Loaded in {time.time() - t0:.1f}s")

    experiments = {
        "stimulus-response": run_stimulus_response,
        "associative": run_associative_conditioning,
        "frequency-response": run_frequency_response,
        "reinforcement": run_reinforcement_learning,
    }

    if args.experiment == "all":
        results = {}
        for name, func in experiments.items():
            print(f"\n{'#'*70}")
            print(f"# RUNNING: {name.upper()}")
            print(f"{'#'*70}")
            results[name] = func(mod, args)
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS SUMMARY")
        for name, success in results.items():
            print(f"  {name:30s}: {'PASS' if success else 'NEEDS WORK'}")
        print(f"{'='*70}")
    else:
        experiments[args.experiment](mod, args)


if __name__ == "__main__":
    main()
