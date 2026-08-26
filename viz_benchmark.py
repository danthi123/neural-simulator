#!/usr/bin/env python
"""
Visualization Performance Benchmark for GPU-Accelerated Neural Network Simulator

Tests the maximum neuron and synapse count that can be simulated in realtime
with visualization enabled. Determines hardware limits for interactive simulation.

Usage:
    python viz_benchmark.py [--output results.json] [--quick] [--model MODEL]

Options:
    --output FILE     Save results to FILE (default: benchmarks/viz_performance_results.json)
    --quick           Run reduced sweep for faster testing
    --model MODEL     Test specific model (IZHIKEVICH, HODGKIN_HUXLEY, ADEX) or ALL (default: IZHIKEVICH)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
import threading

# Import simulator components
try:
    import cupy as cp
    import numpy as np
    
    # Import from neural-simulator.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "neural_simulator",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural-simulator.py")
    )
    neural_simulator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural_simulator)
    
    SimulationBridge = neural_simulator.SimulationBridge
    CoreSimConfig = neural_simulator.CoreSimConfig
    VisualizationConfig = neural_simulator.VisualizationConfig
    RuntimeState = neural_simulator.RuntimeState
    GPUConfig = neural_simulator.GPUConfig
    NeuronModel = neural_simulator.NeuronModel
except ImportError as e:
    print(f"Error importing simulator components: {e}")
    print("Make sure you're running from the simulator directory.")
    sys.exit(1)


class VizBenchmarkRunner:
    """Runs visualization performance benchmarks to determine realtime capacity."""
    
    def __init__(self, quick_mode=False, model_filter=None):
        self.quick_mode = quick_mode
        self.model_filter = model_filter  # None means all, or specific model name
        self.results = []
        
        # Benchmark configurations - scale neuron counts incrementally
        if quick_mode:
            self.neuron_counts = [1000, 5000, 10000]
            self.connections_per_neuron = [100, 500]
            self.models = [NeuronModel.IZHIKEVICH.name]
        else:
            # Full test: progressively scale up to find limits
            self.neuron_counts = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
            self.connections_per_neuron = [100, 500, 1000]
            
            if model_filter and model_filter != "ALL":
                self.models = [model_filter]
            else:
                self.models = [
                    NeuronModel.IZHIKEVICH.name,
                    NeuronModel.HODGKIN_HUXLEY.name,
                    NeuronModel.ADEX.name
                ]
        
        # Test duration per configuration
        self.num_steps = 500  # Run 500 steps per config (enough to measure steady-state)
        self.viz_update_interval = 1  # Update visualization every step (worst case)
        
        # Threshold: viz update time based on achieving at least 30 FPS
        # 30 FPS = 33.3ms per frame (acceptable minimum)
        # 60 FPS = 16.7ms per frame (good)
        # 90+ FPS = 11.1ms per frame (great)
        self.dt_ms = 1.0  # Standard simulation timestep  
        self.viz_update_threshold_ms = 33.3  # Min acceptable: 30 FPS
        self.viz_good_threshold_ms = 16.7  # Good: 60 FPS
        self.viz_great_threshold_ms = 11.1  # Great: 90+ FPS
        
    def get_system_info(self):
        """Collects system and GPU information."""
        try:
            dev_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = dev_props.get('name', b'Unknown').decode()
            total_mem = dev_props['totalGlobalMem']
            
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            
            return {
                "gpu_name": gpu_name,
                "gpu_memory_gb": total_mem / (1024**3),
                "cuda_version": f"{cuda_major}.{cuda_minor}",
                "cupy_version": cp.__version__,
                "numpy_version": np.__version__,
                "python_version": sys.version.split()[0]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def estimate_memory_requirement(self, n_neurons, conn_per_neuron):
        """Estimates GPU memory requirement including visualization overhead."""
        # Conservative estimates in bytes:
        # - Each neuron: ~300 bytes (state + viz buffers)
        # - Each synapse: ~12 bytes (weight + viz data)
        bytes_per_neuron = 300
        synapses_estimate = n_neurons * conn_per_neuron
        bytes_per_synapse = 12
        
        # Visualization overhead: positions, colors, activity timers
        viz_overhead_per_neuron = 50
        
        # OpenGL buffers
        opengl_buffer_estimate = n_neurons * 100  # VBOs for rendering
        
        overhead_factor = 1.5  # Safety margin
        
        estimated_bytes = (
            (n_neurons * (bytes_per_neuron + viz_overhead_per_neuron) + 
             synapses_estimate * bytes_per_synapse +
             opengl_buffer_estimate) 
            * overhead_factor
        )
        
        return estimated_bytes
    
    def check_memory_available(self, required_bytes):
        """Checks if enough GPU memory is available."""
        mem_info = cp.cuda.Device().mem_info
        free_mem, total_mem = mem_info
        
        # Use 75% of free memory as safe limit (more conservative for viz)
        safe_limit = free_mem * 0.75
        
        return required_bytes <= safe_limit, free_mem, total_mem
    
    def run_single_benchmark(self, config_dict):
        """Runs a single benchmark configuration with visualization simulation.
        
        Returns:
            dict: Benchmark metrics if successful
            'SKIPPED': If skipped due to insufficient memory
            'STOPPED': If stopped due to exceeding realtime threshold
            None: If failed due to error
        """
        print(f"\n{'='*70}")
        print(f"Testing: {config_dict['num_neurons']:,} neurons, "
              f"{config_dict['connections_per_neuron']} conn/neuron, "
              f"{config_dict['neuron_model_type']}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        # Check memory before attempting
        estimated_mem = self.estimate_memory_requirement(
            config_dict['num_neurons'],
            config_dict['connections_per_neuron']
        )
        
        can_run, free_mem, total_mem = self.check_memory_available(estimated_mem)
        
        print(f"Estimated memory: {estimated_mem / (1024**3):.2f}GB")
        print(f"Available memory: {free_mem / (1024**3):.2f}GB free / {total_mem / (1024**3):.2f}GB total")
        sys.stdout.flush()
        
        if not can_run:
            print(f"[SKIP] Insufficient GPU memory")
            print(f"       Need ~{estimated_mem / (1024**3):.2f}GB, have {free_mem / (1024**3):.2f}GB available")
            sys.stdout.flush()
            return 'SKIPPED'
        
        # Create config objects
        core_config = CoreSimConfig(
            num_neurons=config_dict['num_neurons'],
            connections_per_neuron=config_dict['connections_per_neuron'],
            neuron_model_type=config_dict['neuron_model_type'],
            dt_ms=self.dt_ms,
            seed=42,  # Fixed seed for reproducibility
            enable_hebbian_learning=False,  # Disable for consistent benchmarking
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        viz_config = VisualizationConfig(
            viz_update_interval_steps=self.viz_update_interval
        )
        
        gpu_config = GPUConfig(
            enable_profiling=True,
            profiling_window_size=self.num_steps,
            enable_gpu_buffered_recording=False  # Don't record during benchmarks
        )
        
        # Create simulator
        try:
            sim = SimulationBridge(
                core_config=core_config,
                viz_config=viz_config,
                runtime_state=RuntimeState(),
                gpu_config=gpu_config
            )
            
            # Initialize
            print("Initializing simulation...")
            sys.stdout.flush()
            init_start = time.time()
            sim._initialize_simulation_data()
            init_time = time.time() - init_start
            
            if not sim.is_initialized:
                print("[FAIL] Initialization error")
                sys.stdout.flush()
                return None
            
            print(f"Initialization: {init_time:.2f}s")
            sys.stdout.flush()
            
            # Measure GPU memory after initialization
            mem_info = cp.cuda.Device().mem_info
            free_mem_after, total_mem = mem_info
            used_mem_gb = (total_mem - free_mem_after) / (1024**3)
            
            print(f"GPU memory used: {used_mem_gb:.2f}GB")
            sys.stdout.flush()
            
            # Run simulation steps with visualization updates
            print(f"Running {self.num_steps} simulation steps with visualization updates...")
            sys.stdout.flush()
            
            step_times = []
            viz_update_times = []
            combined_times = []  # Step + viz update
            
            # First, measure step time WITHOUT viz updates to get baseline
            print(f"Warming up and measuring baseline step time (no viz updates)...")
            warmup_step_times = []
            for step in range(100):  # Warmup phase
                step_start = time.time()
                sim._run_one_simulation_step()
                step_time = time.time() - step_start
                warmup_step_times.append(step_time)
            
            baseline_step_time_ms = float(np.mean(warmup_step_times[-50:])) * 1000
            print(f"Baseline step time (no viz): {baseline_step_time_ms:.2f}ms")
            sys.stdout.flush()
            
            # Now measure with viz updates
            print(f"\nRunning {self.num_steps} steps WITH visualization updates...")
            for step in range(self.num_steps):
                # Time the simulation step
                step_start = time.time()
                sim._run_one_simulation_step()
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Time the visualization update
                viz_start = time.time()
                if step % self.viz_update_interval == 0:
                    gui_data = sim.get_latest_simulation_data_for_gui(force_fetch=True)
                    viz_time = time.time() - viz_start
                    viz_update_times.append(viz_time)
                else:
                    viz_time = 0.0
                
                combined_time = step_time + viz_time
                combined_times.append(combined_time)
                
                # Progress indicator
                if (step + 1) % 50 == 0:
                    avg_step = np.mean(step_times[-50:])
                    avg_viz = np.mean([t for t in viz_update_times[-50:] if t > 0]) if viz_update_times else 0
                    avg_viz_ms = avg_viz * 1000
                    print(f"  Step {step+1}/{self.num_steps} - "
                          f"Step: {avg_step*1000:.2f}ms, Viz: {avg_viz_ms:.2f}ms "
                          f"(threshold: {self.viz_update_threshold_ms:.1f}ms)")
                    sys.stdout.flush()
                
                # Check if viz time is too high
                if step > 50 and len(viz_update_times) > 10:
                    recent_viz_avg_ms = np.mean([t for t in viz_update_times[-10:] if t > 0]) * 1000
                    if recent_viz_avg_ms > self.viz_update_threshold_ms * 1.5:  # 50% over threshold
                        print(f"\n[STOP] Viz update time too high!")
                        print(f"       Viz time: {recent_viz_avg_ms:.2f}ms > threshold: {self.viz_update_threshold_ms:.1f}ms")
                        sys.stdout.flush()
                        break
            
            # Collect metrics
            step_times_arr = np.array(step_times)
            viz_times_arr = np.array([t for t in viz_update_times if t > 0])
            combined_times_arr = np.array(combined_times)
            
            profiling_stats = sim.get_profiling_stats()
            
            # Calculate throughput
            total_sim_time = sum(combined_times)
            steps_per_sec = len(combined_times) / total_sim_time if total_sim_time > 0 else 0
            neurons_per_sec = config_dict['num_neurons'] * steps_per_sec
            
            # Determine if this config has acceptable viz performance
            mean_step_ms = float(np.mean(step_times_arr)) * 1000
            mean_viz_ms = float(np.mean(viz_times_arr)) * 1000 if len(viz_times_arr) > 0 else 0.0
            
            # Check if viz update time is within acceptable range (30+ FPS)
            is_viz_acceptable = mean_viz_ms <= self.viz_update_threshold_ms
            
            # Determine performance tier
            if mean_viz_ms <= self.viz_great_threshold_ms:
                viz_performance_tier = "Great (90+ FPS)"
            elif mean_viz_ms <= self.viz_good_threshold_ms:
                viz_performance_tier = "Good (60 FPS)"
            elif mean_viz_ms <= self.viz_update_threshold_ms:
                viz_performance_tier = "Acceptable (30 FPS)"
            else:
                viz_performance_tier = "Poor (<30 FPS)"
            
            # Calculate actual FPS
            actual_fps = 1000.0 / mean_viz_ms if mean_viz_ms > 0 else 0
            
            metrics = {
                "init_time_s": init_time,
                "total_sim_time_s": total_sim_time,
                "steps_per_sec": steps_per_sec,
                "neurons_per_sec": neurons_per_sec,
                "num_steps_completed": len(combined_times),
                
                # Baseline timing (no viz)
                "baseline_step_time_ms": baseline_step_time_ms,
                
                # Step timing (with viz updates happening)
                "step_time_mean_ms": mean_step_ms,
                "step_time_std_ms": float(np.std(step_times_arr)) * 1000,
                "step_time_p50_ms": float(np.percentile(step_times_arr, 50)) * 1000,
                "step_time_p95_ms": float(np.percentile(step_times_arr, 95)) * 1000,
                "step_time_p99_ms": float(np.percentile(step_times_arr, 99)) * 1000,
                
                # Viz timing
                "viz_update_mean_ms": mean_viz_ms,
                "viz_update_std_ms": float(np.std(viz_times_arr)) * 1000 if len(viz_times_arr) > 0 else 0.0,
                "viz_update_p95_ms": float(np.percentile(viz_times_arr, 95)) * 1000 if len(viz_times_arr) > 0 else 0.0,
                
                # Viz performance analysis
                "viz_update_threshold_ms": self.viz_update_threshold_ms,
                "viz_actual_fps": actual_fps,
                "viz_performance_tier": viz_performance_tier,
                "is_viz_acceptable": is_viz_acceptable,
                
                "gpu_memory_used_gb": used_mem_gb,
                "profiling": profiling_stats
            }
            
            # Print results (may fail on encoding but don't let it prevent data return)
            try:
                print(f"\nResults:")
                print(f"  Baseline step time (no viz): {baseline_step_time_ms:.2f}ms")
                print(f"  Step time (with viz running): {mean_step_ms:.2f}ms")
                print(f"  Viz update time: {mean_viz_ms:.2f}ms (~{actual_fps:.1f} FPS)")
                print(f"  Performance tier: {viz_performance_tier} {'[OK]' if is_viz_acceptable else '[POOR]'}")
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Windows console encoding issue - just skip the cosmetic print
                print(f"\nResults: (encoding issue with status display)")
                print(f"  Viz: {mean_viz_ms:.2f}ms, FPS: {actual_fps:.1f}, Tier: {viz_performance_tier}")
            
            # Cleanup
            sim.clear_simulation_state_and_gpu_memory()
            del sim
            
            return metrics
            
        except Exception as e:
            print(f"[FAIL] Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return None
    
    def run_all_benchmarks(self):
        """Runs all benchmark configurations."""
        print("\n" + "="*70)
        print("VISUALIZATION PERFORMANCE BENCHMARK STARTING")
        print("="*70)
        
        if self.quick_mode:
            print("Running in QUICK mode (reduced configurations)")
        
        total_configs = (len(self.neuron_counts) * 
                        len(self.connections_per_neuron) * 
                        len(self.models))
        
        print(f"Total configurations to test: {total_configs}")
        print(f"Viz update time threshold: {self.viz_update_threshold_ms:.1f}ms")
        
        config_num = 0
        stopped_early = {}  # Track which (model, conn) combos hit limit
        
        for model in self.models:
            for conn_per_neuron in self.connections_per_neuron:
                combo_key = f"{model}_{conn_per_neuron}"
                
                for n_neurons in self.neuron_counts:
                    # Skip if we already found the limit for this combo
                    if combo_key in stopped_early:
                        print(f"\n[SKIP] Already exceeded realtime limit for {model} with {conn_per_neuron} conn/neuron")
                        continue
                    
                    config_num += 1
                    print(f"\n[{config_num}/{total_configs}]")
                    
                    config_dict = {
                        "num_neurons": n_neurons,
                        "connections_per_neuron": conn_per_neuron,
                        "neuron_model_type": model
                    }
                    
                    result = self.run_single_benchmark(config_dict)
                    
                    if result == 'SKIPPED':
                        # Memory limit - stop this combo
                        stopped_early[combo_key] = f"memory_limit_at_{n_neurons}"
                    elif result is not None:
                        self.results.append({
                            "config": config_dict,
                            "metrics": result
                        })
                        print("[ OK ] Benchmark completed successfully")
                        
                        # Check if viz overhead is unacceptable
                        if not result.get("is_viz_acceptable", False):
                            print(f"[INFO] Found viz overhead limit for this configuration")
                            stopped_early[combo_key] = f"viz_limit_at_{n_neurons}"
                        
                        sys.stdout.flush()
                    else:
                        # Actual failure
                        pass
                    
                    # Small delay between runs
                    time.sleep(1.0)
        
        print("\n" + "="*70)
        print("VISUALIZATION BENCHMARK COMPLETE")
        print("="*70)
        
        num_successful = len(self.results)
        num_total = config_num
        
        print(f"Results: {num_successful} successful / {num_total} tested")
        sys.stdout.flush()
    
    def analyze_results(self):
        """Analyzes results to determine maximum realtime capacity."""
        print("\n" + "="*70)
        print("REALTIME CAPACITY ANALYSIS")
        print("="*70)
        
        # Group by model and connection density
        by_model = {}
        for result in self.results:
            model = result['config']['neuron_model_type']
            conn = result['config']['connections_per_neuron']
            n = result['config']['num_neurons']
            is_acceptable = result['metrics']['is_viz_acceptable']
            
            key = (model, conn)
            if key not in by_model:
                by_model[key] = []
            by_model[key].append((n, is_acceptable, result['metrics']))
        
        # Find maximum viz-acceptable configuration for each combo
        capacity_summary = {}
        for (model, conn), configs in by_model.items():
            configs.sort(key=lambda x: x[0])  # Sort by neuron count
            
            max_acceptable_n = 0
            max_acceptable_metrics = None
            
            for n, is_acceptable, metrics in configs:
                if is_acceptable:
                    max_acceptable_n = max(max_acceptable_n, n)
                    max_acceptable_metrics = metrics
            
            capacity_summary[(model, conn)] = {
                "max_neurons": max_acceptable_n,
                "connections_per_neuron": conn,
                "total_synapses": max_acceptable_n * conn if max_acceptable_n > 0 else 0,
                "metrics": max_acceptable_metrics
            }
            
            print(f"\n{model} with {conn} connections/neuron:")
            if max_acceptable_n > 0:
                viz_fps = max_acceptable_metrics.get('viz_actual_fps', 0)
                viz_tier = max_acceptable_metrics.get('viz_performance_tier', 'Unknown')
                print(f"  Max with acceptable viz performance: {max_acceptable_n:,} neurons ({max_acceptable_n * conn:,} synapses)")
                print(f"  Viz update time: {max_acceptable_metrics['viz_update_mean_ms']:.2f}ms (~{viz_fps:.1f} FPS)")
                print(f"  Performance tier: {viz_tier}")
                print(f"  Baseline step time: {max_acceptable_metrics['baseline_step_time_ms']:.2f}ms")
            else:
                print(f"  No configurations with acceptable viz performance found")
        
        return capacity_summary
    
    def generate_hardware_note(self, capacity_summary):
        """Generates a hardware performance note for profiles."""
        sys_info = self.get_system_info()
        gpu_name = sys_info.get('gpu_name', 'Unknown GPU')
        
        # Find the best overall configuration
        best_config = None
        best_neurons = 0
        
        for (model, conn), data in capacity_summary.items():
            if data['max_neurons'] > best_neurons:
                best_neurons = data['max_neurons']
                best_config = (model, conn, data)
        
        if best_config:
            model, conn, data = best_config
            viz_time = data['metrics']['viz_update_mean_ms'] if data['metrics'] else 0
            viz_fps = data['metrics'].get('viz_actual_fps', 0) if data['metrics'] else 0
            note = (f"Hardware Performance ({gpu_name}): "
                   f"Max with real-time viz (30+ FPS): ~{best_neurons:,} neurons "
                   f"with {conn} conn/neuron ({data['total_synapses']:,} synapses) "
                   f"using {model} model. Going above this may reduce refresh rate below 30 FPS.")
        else:
            note = f"Hardware Performance ({gpu_name}): No configurations achieving 30+ FPS found. Viz bottleneck detected - consider optimizing GPU-CPU transfers or reducing viz update frequency."
        
        return note
    
    def save_results(self, filepath):
        """Saves benchmark results to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        capacity_summary = self.analyze_results()
        hardware_note = self.generate_hardware_note(capacity_summary)
        
        report = {
            "benchmark_version": "2.0",
            "benchmark_type": "visualization_performance",
            "timestamp": datetime.now().isoformat(),
            "quick_mode": self.quick_mode,
            "num_steps_per_config": self.num_steps,
            "viz_update_threshold_ms": self.viz_update_threshold_ms,
            "dt_ms": self.dt_ms,
            "viz_update_interval": self.viz_update_interval,
            "system_info": self.get_system_info(),
            "capacity_summary": {
                f"{model}_{conn}conn": data 
                for (model, conn), data in capacity_summary.items()
            },
            "hardware_performance_note": hardware_note,
            "results": self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        print(f"\nHardware Note: {hardware_note}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualization performance benchmark for neural network simulator"
    )
    parser.add_argument(
        '--output', 
        default='benchmarks/viz_performance_results.json',
        help='Output file for results (default: benchmarks/viz_performance_results.json)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run reduced sweep for faster testing'
    )
    parser.add_argument(
        '--model',
        default='IZHIKEVICH',
        choices=['IZHIKEVICH', 'HODGKIN_HUXLEY', 'ADEX', 'ALL'],
        help='Test specific model or ALL (default: IZHIKEVICH)'
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    runner = VizBenchmarkRunner(
        quick_mode=args.quick,
        model_filter=args.model
    )
    runner.run_all_benchmarks()
    
    # Save results
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
