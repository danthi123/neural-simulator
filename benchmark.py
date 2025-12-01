#!/usr/bin/env python
"""
Headless Benchmark Runner for GPU-Accelerated Neural Network Simulator

Usage:
    python benchmark.py [--output results.json] [--quick] [--compare baseline.json]

Options:
    --output FILE     Save results to FILE (default: benchmarks/benchmark_results.json)
    --quick           Run reduced sweep for faster testing
    --compare FILE    Compare results against baseline FILE
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

# Import simulator components
try:
    import cupy as cp
    import numpy as np
    
    # Import from neural-simulator.py (using importlib to handle hyphen)
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


class BenchmarkRunner:
    """Runs benchmark sweeps and collects performance metrics."""
    
    def __init__(self, quick_mode=False):
        self.quick_mode = quick_mode
        self.results = []
        
        # Benchmark configurations
        if quick_mode:
            self.neuron_counts = [1000, 10000]
            self.connections_per_neuron = [100, 500]
            self.models = [NeuronModel.IZHIKEVICH.name]
        else:
            self.neuron_counts = [1000, 10000, 50000]
            self.connections_per_neuron = [100, 500, 1000]
            self.models = [
                NeuronModel.IZHIKEVICH.name,
                NeuronModel.HODGKIN_HUXLEY.name,
                NeuronModel.ADEX.name
            ]
        
        self.num_steps = 1000  # Run 1000 simulation steps per config
        
    def get_system_info(self):
        """Collects system and GPU information."""
        try:
            dev_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = dev_props.get('name', b'Unknown').decode()
            total_mem = dev_props['totalGlobalMem']
            
            # Get CUDA version
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
        """Estimates GPU memory requirement for a configuration."""
        # Conservative estimates in bytes:
        # - Each neuron: ~200 bytes (state variables)
        # - Each synapse: ~8 bytes (weight in sparse matrix)
        bytes_per_neuron = 200
        synapses_estimate = n_neurons * conn_per_neuron
        bytes_per_synapse = 8
        
        # Add overhead for recording buffers, temporary arrays, etc.
        overhead_factor = 1.5
        
        estimated_bytes = (
            (n_neurons * bytes_per_neuron + synapses_estimate * bytes_per_synapse) 
            * overhead_factor
        )
        
        return estimated_bytes
    
    def check_memory_available(self, required_bytes):
        """Checks if enough GPU memory is available."""
        mem_info = cp.cuda.Device().mem_info
        free_mem, total_mem = mem_info
        
        # Use 80% of free memory as safe limit
        safe_limit = free_mem * 0.8
        
        return required_bytes <= safe_limit, free_mem, total_mem
    
    def run_single_benchmark(self, config_dict):
        """Runs a single benchmark configuration.
        
        Returns:
            dict: Benchmark metrics if successful
            'SKIPPED': If skipped due to insufficient memory
            None: If failed due to error
        """
        print(f"\n{'='*60}")
        print(f"Testing: {config_dict['num_neurons']} neurons, "
              f"{config_dict['connections_per_neuron']} conn/neuron, "
              f"{config_dict['neuron_model_type']}")
        print(f"{'='*60}")
        sys.stdout.flush()  # Force log output
        
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
            return 'SKIPPED'  # Special return value
        
        # Create config objects
        core_config = CoreSimConfig(
            num_neurons=config_dict['num_neurons'],
            connections_per_neuron=config_dict['connections_per_neuron'],
            neuron_model_type=config_dict['neuron_model_type'],
            dt_ms=1.0,
            seed=42,  # Fixed seed for reproducibility
            enable_hebbian_learning=False,  # Disable for consistent benchmarking
            enable_short_term_plasticity=False,
            enable_homeostasis=False
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
                viz_config=VisualizationConfig(),
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
            free_mem, total_mem = mem_info
            used_mem_gb = (total_mem - free_mem) / (1024**3)
            
            print(f"GPU memory used: {used_mem_gb:.2f}GB")
            sys.stdout.flush()
            
            # Run simulation steps
            print(f"Running {self.num_steps} simulation steps...")
            sys.stdout.flush()
            step_times = []
            
            for step in range(self.num_steps):
                step_start = time.time()
                sim._run_one_simulation_step()
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Progress indicator
                if (step + 1) % 100 == 0:
                    avg_step_time = np.mean(step_times[-100:])
                    print(f"  Step {step+1}/{self.num_steps} - "
                          f"Avg time: {avg_step_time*1000:.2f}ms")
                    sys.stdout.flush()
            
            # Collect metrics
            step_times_arr = np.array(step_times)
            profiling_stats = sim.get_profiling_stats()
            
            # Calculate throughput
            total_sim_time = sum(step_times)
            steps_per_sec = self.num_steps / total_sim_time if total_sim_time > 0 else 0
            neurons_per_sec = config_dict['num_neurons'] * steps_per_sec
            
            metrics = {
                "init_time_s": init_time,
                "total_sim_time_s": total_sim_time,
                "steps_per_sec": steps_per_sec,
                "neurons_per_sec": neurons_per_sec,
                "step_time_mean_ms": float(np.mean(step_times_arr)) * 1000,
                "step_time_std_ms": float(np.std(step_times_arr)) * 1000,
                "step_time_p50_ms": float(np.percentile(step_times_arr, 50)) * 1000,
                "step_time_p95_ms": float(np.percentile(step_times_arr, 95)) * 1000,
                "step_time_p99_ms": float(np.percentile(step_times_arr, 99)) * 1000,
                "gpu_memory_used_gb": used_mem_gb,
                "profiling": profiling_stats
            }
            
            print(f"\nResults:")
            print(f"  Total time: {total_sim_time:.2f}s")
            print(f"  Steps/sec: {steps_per_sec:.1f}")
            print(f"  Mean step time: {metrics['step_time_mean_ms']:.2f}ms")
            print(f"  P95 step time: {metrics['step_time_p95_ms']:.2f}ms")
            sys.stdout.flush()
            
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
        print("\n" + "="*60)
        print("BENCHMARK SUITE STARTING")
        print("="*60)
        
        if self.quick_mode:
            print("Running in QUICK mode (reduced configurations)")
        
        total_configs = (len(self.neuron_counts) * 
                        len(self.connections_per_neuron) * 
                        len(self.models))
        
        print(f"Total configurations to test: {total_configs}")
        
        config_num = 0
        for model in self.models:
            for n_neurons in self.neuron_counts:
                for conn_per_neuron in self.connections_per_neuron:
                    config_num += 1
                    print(f"\n[{config_num}/{total_configs}]")
                    
                    config_dict = {
                        "num_neurons": n_neurons,
                        "connections_per_neuron": conn_per_neuron,
                        "neuron_model_type": model
                    }
                    
                    result = self.run_single_benchmark(config_dict)
                    
                    if result == 'SKIPPED':
                        # Memory-skipped configs are noted but not counted as failures
                        pass  # Already printed skip message
                    elif result is not None:
                        self.results.append({
                            "config": config_dict,
                            "metrics": result
                        })
                        print("[ OK ] Benchmark completed successfully")
                        sys.stdout.flush()
                    else:
                        # Actual failure (error during execution)
                        pass  # Already printed failure message
                    
                    # Small delay between runs
                    time.sleep(1.0)
        
        print("\n" + "="*60)
        print("BENCHMARK SUITE COMPLETE")
        print("="*60)
        
        num_successful = len(self.results)
        num_total = total_configs
        num_failed_or_skipped = num_total - num_successful
        
        print(f"Results: {num_successful} successful / {num_total} total")
        if num_failed_or_skipped > 0:
            print(f"         ({num_failed_or_skipped} skipped due to insufficient memory or errors)")
        sys.stdout.flush()
    
    def save_results(self, filepath):
        """Saves benchmark results to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        report = {
            "benchmark_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "quick_mode": self.quick_mode,
            "num_steps_per_config": self.num_steps,
            "system_info": self.get_system_info(),
            "results": self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def compare_with_baseline(self, baseline_path):
        """Compares current results with a baseline."""
        if not os.path.exists(baseline_path):
            print(f"ERROR: Baseline file not found: {baseline_path}")
            return
        
        try:
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
            
            print("\n" + "="*60)
            print("COMPARISON WITH BASELINE")
            print("="*60)
            
            baseline_results = {
                (r['config']['num_neurons'], 
                 r['config']['connections_per_neuron'],
                 r['config']['neuron_model_type']): r['metrics']
                for r in baseline.get('results', [])
            }
            
            print(f"Baseline: {baseline.get('timestamp', 'unknown')}")
            print(f"Current:  {datetime.now().isoformat()}")
            print()
            
            improvements = []
            regressions = []
            
            for result in self.results:
                config = result['config']
                metrics = result['metrics']
                
                key = (config['num_neurons'], 
                       config['connections_per_neuron'],
                       config['neuron_model_type'])
                
                if key not in baseline_results:
                    continue
                
                baseline_metrics = baseline_results[key]
                
                # Compare mean step time
                current_time = metrics['step_time_mean_ms']
                baseline_time = baseline_metrics['step_time_mean_ms']
                
                if baseline_time > 0:
                    change_pct = ((current_time - baseline_time) / baseline_time) * 100
                    
                    print(f"{config['neuron_model_type']} - {config['num_neurons']}N, "
                          f"{config['connections_per_neuron']}C:")
                    print(f"  Baseline: {baseline_time:.2f}ms")
                    print(f"  Current:  {current_time:.2f}ms")
                    print(f"  Change:   {change_pct:+.1f}%")
                    
                    if change_pct < -5:  # 5% faster
                        improvements.append((key, change_pct))
                    elif change_pct > 5:  # 5% slower
                        regressions.append((key, change_pct))
            
            print()
            print(f"Improvements: {len(improvements)}")
            print(f"Regressions:  {len(regressions)}")
            
            if regressions:
                print("\nWARNING: Performance regressions detected!")
                for key, change in regressions:
                    print(f"  {key[2]} {key[0]}N: {change:+.1f}%")
            
        except Exception as e:
            print(f"ERROR comparing with baseline: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for neural network simulator"
    )
    parser.add_argument(
        '--output', 
        default='benchmarks/benchmark_results.json',
        help='Output file for results (default: benchmarks/benchmark_results.json)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run reduced sweep for faster testing'
    )
    parser.add_argument(
        '--compare',
        metavar='BASELINE',
        help='Compare results against baseline file'
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    runner = BenchmarkRunner(quick_mode=args.quick)
    runner.run_all_benchmarks()
    
    # Save results
    runner.save_results(args.output)
    
    # Compare if requested
    if args.compare:
        runner.compare_with_baseline(args.compare)


if __name__ == "__main__":
    main()
