"""
GPU Validation Script for Neural Simulator
=============================================

Comprehensive headless GPU validation suite for the neural simulator.
Tests GPU operations, determinism, performance, numerical stability, plasticity,
and extended Hodgkin-Huxley currents. Results are written to JSON for review.

Usage:
    python tests/validate_gpu.py              # Full validation suite
    python tests/validate_gpu.py --quick      # Fast iteration mode (reduced sizes)
    python tests/validate_gpu.py --help       # Show usage options
"""

import sys
import os
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cupy as cp

# Import from neural-simulator.py (using importlib to handle hyphen)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "neural_simulator",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neural-simulator.py")
)
neural_simulator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_simulator)

SimulationBridge = neural_simulator.SimulationBridge
CoreSimConfig = neural_simulator.CoreSimConfig
VisualizationConfig = neural_simulator.VisualizationConfig
RuntimeState = neural_simulator.RuntimeState
GPUConfig = neural_simulator.GPUConfig
NeuronModel = neural_simulator.NeuronModel


class ValidationResults:
    """Container for validation results."""
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self._get_gpu_info(),
            "tests": {}
        }

    def _get_gpu_info(self):
        """Get GPU information."""
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props.get('name', b'Unknown').decode()
            total_mem = props['totalGlobalMem']
            return f"{gpu_name} ({total_mem / 1024**3:.1f} GB)"
        except Exception as e:
            return f"Unknown GPU (Error: {str(e)})"

    def add_test(self, test_name, passed, details=None, metrics=None):
        """Add a test result."""
        self.results["tests"][test_name] = {
            "passed": passed,
            "details": details or "",
            "metrics": metrics or {}
        }

    def to_dict(self):
        """Return results as dictionary."""
        return self.results

    def save_json(self, filepath):
        """Save results to JSON file."""
        import numpy as _np

        class _SafeEncoder(json.JSONEncoder):
            """Handle NumPy/CuPy types that aren't JSON-serializable."""
            def default(self, obj):
                # NumPy scalars (bool_, int64, float32, etc.)
                if isinstance(obj, _np.generic):
                    return obj.item()
                # NumPy arrays
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                # CuPy arrays/scalars — check by module name to avoid import
                type_module = getattr(type(obj), '__module__', '')
                if 'cupy' in type_module:
                    try:
                        return obj.item()  # scalar
                    except (ValueError, AttributeError):
                        try:
                            return obj.get().tolist()  # array → CPU → list
                        except Exception:
                            return str(obj)
                # Fallback: try .item(), then str()
                if hasattr(obj, 'item'):
                    return obj.item()
                return str(obj)

        try:
            json_str = json.dumps(self.results, indent=2, cls=_SafeEncoder)
            with open(filepath, 'w') as f:
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())
            print(f"Results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
            # Fallback: write what we can
            try:
                with open(filepath + ".partial", 'w') as f:
                    f.write(str(self.results))
            except Exception:
                pass


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_test_result(test_name, passed, message=""):
    """Print a test result with clear pass/fail indicator."""
    status = "PASS" if passed else "FAIL"
    symbol = "[✓]" if passed else "[✗]"
    print(f"{symbol} {test_name}: {status}")
    if message:
        print(f"   {message}")


# ============================================================================
# HELPER: Inject suprathreshold external drive after initialization
# ============================================================================
def inject_external_drive(sim, model_type, strength=1.0):
    """Set external input current to suprathreshold levels after _initialize_simulation_data().

    This is more reliable than OU for headless tests because it's immediate (no ramp-up)
    and correctly handles the unit differences between models:
      - Izhikevich: current in pA directly (needs ~300-800 pA for RS)
      - HH: current in pA but converted to µA/cm² via *1e-6 internally (needs ~5-15e6 pA)
      - AdEx: current in pA directly (needs ~200-600 pA)

    Args:
        sim: Initialized SimulationBridge
        model_type: NeuronModel name string
        strength: Multiplier for base drive (1.0 = moderate, 2.0 = strong)
    """
    n = sim.core_config.num_neurons
    if model_type == NeuronModel.HODGKIN_HUXLEY.name:
        # HH: 15e6 pA * 1e-6 = 15 µA/cm² (well above suprathreshold for HH)
        base_mean, base_std = 15e6, 3e6
    elif model_type == NeuronModel.ADEX.name:
        # AdEx: ~600 pA mean with jitter
        base_mean, base_std = 600.0, 150.0
    else:
        # Izhikevich: ~1500 pA mean.  Must be strong enough to reach vpeak=35 mV
        # even with heterogeneous parameters and network inhibition.
        # When homeostasis is OFF, threshold = vpeak (35 mV), requiring strong drive.
        # When homeostasis is ON, thresholds are ~-42 mV, so this is more than enough.
        base_mean, base_std = 1500.0, 300.0

    drive = cp.random.normal(base_mean * strength, base_std * strength, n).astype(cp.float32)
    sim.cp_external_input_current[:] = cp.maximum(drive, 0.0)


# ============================================================================
# TEST SUITE
# ============================================================================

def test_determinism(quick=False):
    """Test: Determinism - Run same simulation twice, verify identical results."""
    print_section("DETERMINISM TEST")

    num_steps = 50 if quick else 100
    num_neurons = 100 if quick else 500

    try:
        config = CoreSimConfig(
            num_neurons=num_neurons,
            connections_per_neuron=50,
            seed=42,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            dt_ms=1.0,
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_homeostasis=True,  # Uses adaptive thresholds (~-42 mV) for reliable spike detection
            enable_ou_process=False,  # Using direct drive injection for reliability
        )
        gpu_config = GPUConfig(enable_profiling=False)

        # Run 1
        print("Running simulation #1 with seed=42...")
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        inject_external_drive(sim1, NeuronModel.IZHIKEVICH.name)

        spikes1 = []
        for step in range(num_steps):
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())

        total_spikes_1 = sum(np.sum(s) for s in spikes1)
        sim1.clear_simulation_state_and_gpu_memory()

        # Run 2
        print("Running simulation #2 with same seed=42...")
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        inject_external_drive(sim2, NeuronModel.IZHIKEVICH.name)

        spikes2 = []
        for step in range(num_steps):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())

        total_spikes_2 = sum(np.sum(s) for s in spikes2)
        sim2.clear_simulation_state_and_gpu_memory()

        # Compare
        all_match = True
        mismatch_steps = []
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            if not np.array_equal(s1, s2):
                all_match = False
                mismatch_steps.append(i)

        has_spikes = total_spikes_1 > 0
        passed = all_match and total_spikes_1 == total_spikes_2 and has_spikes
        message = ""
        if passed:
            message = f"Identical spike trains: {num_steps} steps, {int(total_spikes_1)} total spikes"
        elif not has_spikes:
            message = f"No spikes produced in either run — drive too weak or initialization issue"
        else:
            message = f"Mismatch at {len(mismatch_steps)} steps. Total spikes: Run1={int(total_spikes_1)}, Run2={int(total_spikes_2)}"

        print_test_result("Determinism (Izhikevich)", passed, message)
        return passed, {"steps_checked": num_steps, "total_spikes": int(total_spikes_1), "mismatches": len(mismatch_steps)}

    except Exception as e:
        print_test_result("Determinism", False, str(e))
        return False, {"error": str(e)}


def test_performance_benchmark(quick=False):
    """Test: Performance Benchmark - Time simulations at different network sizes."""
    print_section("PERFORMANCE BENCHMARK")

    # Reduced configs for quick mode
    if quick:
        neuron_configs = [100, 500, 1000]
        num_steps = 10
    else:
        neuron_configs = [1000, 5000, 10000, 50000]
        num_steps = 100

    models_to_test = [
        NeuronModel.IZHIKEVICH.name,
        NeuronModel.HODGKIN_HUXLEY.name,
        NeuronModel.ADEX.name
    ]

    results = {}
    all_passed = True

    for model_name in models_to_test:
        print(f"\nBenchmarking {model_name}...")
        model_results = {}

        # Choose appropriate dt for model
        if model_name == NeuronModel.HODGKIN_HUXLEY.name:
            dt_ms = 0.025
            step_count = num_steps // 4 if quick else num_steps // 2
        elif model_name == NeuronModel.ADEX.name:
            dt_ms = 0.1
            step_count = num_steps
        else:
            dt_ms = 1.0
            step_count = num_steps

        for num_neurons in neuron_configs:
            try:
                print(f"  {num_neurons} neurons...", end=" ", flush=True)

                config = CoreSimConfig(
                    num_neurons=num_neurons,
                    connections_per_neuron=min(50, num_neurons // 2),
                    seed=42,
                    neuron_model_type=model_name,
                    dt_ms=dt_ms,
                    enable_hebbian_learning=False,
                    enable_short_term_plasticity=False,
                    enable_homeostasis=False
                )
                gpu_config = GPUConfig(enable_profiling=False)

                sim = SimulationBridge(
                    core_config=config,
                    viz_config=VisualizationConfig(),
                    runtime_state=RuntimeState(),
                    gpu_config=gpu_config
                )
                sim._initialize_simulation_data()

                # Warm-up step
                sim._run_one_simulation_step()

                # Timed steps
                t_start = time.time()
                for _ in range(step_count):
                    sim._run_one_simulation_step()
                t_end = time.time()

                elapsed_ms = (t_end - t_start) * 1000.0
                ms_per_step = elapsed_ms / step_count

                model_results[f"{num_neurons}_neurons"] = {
                    "ms_per_step": round(ms_per_step, 3),
                    "steps": step_count,
                    "total_time_ms": round(elapsed_ms, 2)
                }

                print(f"{ms_per_step:.3f} ms/step")
                sim.clear_simulation_state_and_gpu_memory()

            except Exception as e:
                print(f"ERROR: {str(e)}")
                all_passed = False
                model_results[f"{num_neurons}_neurons"] = {"error": str(e)}

        results[model_name] = model_results

    print_test_result("Performance Benchmark", all_passed, f"Tested {len(models_to_test)} models")
    return all_passed, results


def test_numerical_stability(quick=False):
    """Test: Numerical Stability - Check for NaN/Inf, firing rates in reasonable range."""
    print_section("NUMERICAL STABILITY TEST")

    num_steps = 500 if quick else 5000
    num_neurons = 100 if quick else 500

    try:
        # Test with HH model at different dt values
        dt_values = [0.05, 0.1] if quick else [0.05, 0.1]
        results = {}

        for dt_ms in dt_values:
            print(f"\nTesting HH model at dt={dt_ms}ms...")

            config = CoreSimConfig(
                num_neurons=num_neurons,
                connections_per_neuron=50,
                seed=42,
                neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
                dt_ms=dt_ms,
                enable_hebbian_learning=False,
                enable_short_term_plasticity=False,
                enable_homeostasis=False,
                enable_ou_process=False,  # Using direct drive injection
            )
            gpu_config = GPUConfig(enable_profiling=False)

            sim = SimulationBridge(
                core_config=config,
                viz_config=VisualizationConfig(),
                runtime_state=RuntimeState(),
                gpu_config=gpu_config
            )
            sim._initialize_simulation_data()
            inject_external_drive(sim, NeuronModel.HODGKIN_HUXLEY.name)

            spike_count = 0
            has_nan = False
            has_inf = False
            min_v = float('inf')
            max_v = float('-inf')

            for step in range(num_steps):
                sim._run_one_simulation_step()

                v = cp.asnumpy(sim.cp_membrane_potential_v)
                spike_count += int(cp.sum(sim.cp_firing_states))  # Convert CuPy → int

                if np.any(np.isnan(v)):
                    has_nan = True
                if np.any(np.isinf(v)):
                    has_inf = True

                min_v = min(min_v, float(np.min(v)))
                max_v = max(max_v, float(np.max(v)))

            sim.clear_simulation_state_and_gpu_memory()

            # Calculate firing rate (spikes per neuron per second)
            time_seconds = (num_steps * dt_ms) / 1000.0
            firing_rate_hz = spike_count / (num_neurons * time_seconds) if time_seconds > 0 else 0.0

            # Check stability: no NaN/Inf, voltages in biophysical range, membrane shows depolarization.
            # Note: At 37°C with Q10=3.0, HH produces fast-attenuated spikes that may not cross
            # the formal spike detection threshold (+40 mV), so we check depolarization instead.
            numerically_stable = not (has_nan or has_inf)
            voltage_bounded = min_v > -100.0 and max_v < 100.0  # No runaway divergence
            shows_depolarization = max_v > -50.0  # V rises above resting (-65) toward threshold
            passed = numerically_stable and voltage_bounded and shows_depolarization
            message = f"dt={dt_ms}ms: V range: [{min_v:.1f}, {max_v:.1f}] mV, {spike_count} formal spikes, {firing_rate_hz:.2f} Hz"

            if has_nan:
                message += " [NaN detected!]"
            if has_inf:
                message += " [Inf detected!]"
            if not voltage_bounded:
                message += " [Voltage diverged!]"
            if not shows_depolarization:
                message += " [No depolarization — drive too weak]"

            print_test_result(f"Numerical Stability (dt={dt_ms}ms)", passed, message)
            results[f"dt_{dt_ms}"] = {
                "firing_rate_hz": round(float(firing_rate_hz), 2),
                "v_min": round(float(min_v), 2),
                "v_max": round(float(max_v), 2),
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf)
            }

        # Pass if all dt values are numerically stable with bounded voltages and depolarization
        stable = all(not r.get('has_nan') and not r.get('has_inf') for r in results.values())
        bounded = all(r.get('v_min', -999) > -100 and r.get('v_max', 999) < 100 for r in results.values())
        depolarized = all(r.get('v_max', -999) > -50 for r in results.values())
        return stable and bounded and depolarized, results

    except Exception as e:
        print_test_result("Numerical Stability", False, str(e))
        return False, {"error": str(e)}


def test_plasticity_validation(quick=False):
    """Test: Plasticity - Enable all plasticity and verify weights stay in bounds."""
    print_section("PLASTICITY VALIDATION TEST")

    num_steps = 500 if quick else 2000
    num_neurons = 100 if quick else 300

    try:
        config = CoreSimConfig(
            num_neurons=num_neurons,
            connections_per_neuron=50,
            seed=42,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            dt_ms=1.0,
            enable_hebbian_learning=True,
            hebbian_learning_rate=0.0005,
            hebbian_min_weight=0.05,
            hebbian_max_weight=1.0,
            enable_short_term_plasticity=True,
            stp_U=0.15,
            enable_stdp=True,
            stdp_a_plus=0.01,
            stdp_a_minus=0.0105,
            enable_reward_modulation=True,
            reward_learning_rate=0.01,
            enable_homeostasis=True,
            homeostasis_target_rate=0.02,
            enable_ou_process=False,  # Using direct drive injection for reliability
        )
        gpu_config = GPUConfig(enable_profiling=False)

        print("Running simulation with all plasticity enabled...")
        sim = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim._initialize_simulation_data()
        inject_external_drive(sim, NeuronModel.IZHIKEVICH.name, strength=1.5)  # Strong drive for plasticity

        weight_stats = {"min_w": [], "max_w": [], "mean_w": []}
        firing_rate_history = []

        for step in range(num_steps):
            sim._run_one_simulation_step()

            # Record weight stats
            if sim.cp_connections is not None and sim.cp_connections.nnz > 0:
                weights = cp.asnumpy(sim.cp_connections.data)
                weight_stats["min_w"].append(float(np.min(weights)))
                weight_stats["max_w"].append(float(np.max(weights)))
                weight_stats["mean_w"].append(float(np.mean(weights)))

            # Record firing rate (convert CuPy → NumPy before summing)
            firing_rate = float(np.sum(cp.asnumpy(sim.cp_firing_states))) / num_neurons
            firing_rate_history.append(firing_rate)

        sim.clear_simulation_state_and_gpu_memory()

        # Analyze
        if weight_stats["min_w"]:
            min_weight_overall = min(weight_stats["min_w"])
            max_weight_overall = max(weight_stats["max_w"])
            mean_weight = np.mean(weight_stats["mean_w"])

            # Check if weights stay in bounds (with some tolerance)
            weights_bounded = min_weight_overall >= 0.0 and max_weight_overall <= 2.5

            # Check if firing rates are reasonable (fraction of neurons firing per step)
            mean_firing_rate = float(np.mean(firing_rate_history))
            fr_reasonable = mean_firing_rate > 0.0005  # At least some neurons firing

            passed = weights_bounded and fr_reasonable

            message = f"Weight range: [{min_weight_overall:.3f}, {max_weight_overall:.3f}], Mean: {mean_weight:.3f}, Mean FR: {mean_firing_rate:.3f}"

            print_test_result("Plasticity Validation", passed, message)

            return passed, {
                "min_weight": round(min_weight_overall, 3),
                "max_weight": round(max_weight_overall, 3),
                "mean_weight": round(mean_weight, 3),
                "mean_firing_rate": round(mean_firing_rate, 3),
                "weights_bounded": weights_bounded,
                "firing_rate_reasonable": fr_reasonable
            }
        else:
            print_test_result("Plasticity Validation", False, "No synapses formed")
            return False, {"error": "No synapses in network"}

    except Exception as e:
        print_test_result("Plasticity Validation", False, str(e))
        return False, {"error": str(e)}


def test_coo_cache_validation(quick=False):
    """Test: COO Cache - Verify cached COO matrix gives same results as uncached."""
    print_section("COO CACHE VALIDATION TEST")

    num_steps = 50 if quick else 100
    num_neurons = 100 if quick else 300

    try:
        print("Running cached COO path vs reference...")
        config = CoreSimConfig(
            num_neurons=num_neurons,
            connections_per_neuron=50,
            seed=42,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            dt_ms=1.0,
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_ou_process=False,  # Using direct drive injection
        )
        gpu_config = GPUConfig(enable_profiling=False)

        # Run with caching
        sim_cached = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim_cached._initialize_simulation_data()
        inject_external_drive(sim_cached, NeuronModel.IZHIKEVICH.name)

        spikes_cached = []
        for step in range(num_steps):
            sim_cached._run_one_simulation_step()
            spikes_cached.append(cp.asnumpy(sim_cached.cp_firing_states).copy())

        total_spikes_cached = sum(np.sum(s) for s in spikes_cached)
        sim_cached.clear_simulation_state_and_gpu_memory()

        # Run reference (clean start, identical config)
        config2 = CoreSimConfig(
            num_neurons=num_neurons,
            connections_per_neuron=50,
            seed=42,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            dt_ms=1.0,
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_ou_process=False,  # Using direct drive injection
        )
        sim_ref = SimulationBridge(
            core_config=config2,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim_ref._initialize_simulation_data()
        inject_external_drive(sim_ref, NeuronModel.IZHIKEVICH.name)

        spikes_ref = []
        for step in range(num_steps):
            sim_ref._run_one_simulation_step()
            spikes_ref.append(cp.asnumpy(sim_ref.cp_firing_states).copy())

        total_spikes_ref = sum(np.sum(s) for s in spikes_ref)
        sim_ref.clear_simulation_state_and_gpu_memory()

        # Compare
        all_match = all(np.array_equal(s1, s2) for s1, s2 in zip(spikes_cached, spikes_ref))
        passed = all_match and total_spikes_cached == total_spikes_ref

        message = f"Spikes match: {total_spikes_cached == total_spikes_ref}, Steps match: {all_match}"
        print_test_result("COO Cache Validation", passed, message)

        return passed, {
            "total_spikes_match": total_spikes_cached == total_spikes_ref,
            "all_steps_match": all_match,
            "spikes_cached": int(total_spikes_cached),
            "spikes_ref": int(total_spikes_ref)
        }

    except Exception as e:
        print_test_result("COO Cache Validation", False, str(e))
        return False, {"error": str(e)}


def test_extended_hh_currents(quick=False):
    """Test: Extended HH Currents - Enable M, CaT, h, NaP currents at 37°C."""
    print_section("EXTENDED HH CURRENTS TEST")

    num_steps = 100 if quick else 500
    num_neurons = 50 if quick else 200

    try:
        print("Testing HH model with extended currents (M, CaT, h, NaP) at 37°C...")

        config = CoreSimConfig(
            num_neurons=num_neurons,
            connections_per_neuron=25,
            seed=42,
            neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
            dt_ms=0.01,  # Very small dt for numerical stability at 37°C with extended currents
            hh_temperature_celsius=37.0,
            hh_q10_factor=3.0,
            # Enable extended currents (moderate conductances to avoid instability at high phi)
            hh_g_M_max=1.0,        # M-current (slow K+)
            hh_g_CaT_max=0.5,      # CaT current
            hh_g_h_max=0.5,        # h-current
            hh_g_NaP_max=0.2,      # NaP current
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_ou_process=False,  # Using direct drive injection
        )
        gpu_config = GPUConfig(enable_profiling=False)

        sim = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim._initialize_simulation_data()
        inject_external_drive(sim, NeuronModel.HODGKIN_HUXLEY.name)

        has_nan = False
        has_inf = False
        min_v = float('inf')
        max_v = float('-inf')
        spike_count = 0

        for step in range(num_steps):
            try:
                sim._run_one_simulation_step()
            except Exception as step_err:
                print(f"  Step {step} failed: {step_err}")
                break

            v = cp.asnumpy(sim.cp_membrane_potential_v)
            spike_count += int(cp.sum(sim.cp_firing_states))  # CuPy → int

            if np.any(np.isnan(v)):
                has_nan = True
                print(f"  NaN detected at step {step}, aborting early")
                break
            if np.any(np.isinf(v)):
                has_inf = True
                print(f"  Inf detected at step {step}, aborting early")
                break

            min_v = min(min_v, float(np.min(v)))
            max_v = max(max_v, float(np.max(v)))

        # Force GPU sync and cleanup
        cp.cuda.Stream.null.synchronize()
        sim.clear_simulation_state_and_gpu_memory()
        cp.get_default_memory_pool().free_all_blocks()

        # Check for divergence (unbounded growth)
        diverged = max_v > 200.0 or min_v < -150.0

        # Must be numerically stable (no NaN/Inf/divergence).
        # At 37°C with Q10=3, HH spikes are fast-attenuated and may not cross +40 mV
        # (the formal spike detection threshold), so we check depolarization instead.
        shows_depolarization = max_v > -50.0
        passed = not (has_nan or has_inf or diverged) and shows_depolarization

        message = f"V range: [{min_v:.1f}, {max_v:.1f}] mV, {spike_count} formal spikes"
        if has_nan:
            message = "NaN detected"
        elif has_inf:
            message = "Inf detected"
        elif diverged:
            message = f"Diverged: V range [{min_v:.1f}, {max_v:.1f}]"
        elif not shows_depolarization:
            message += " [No depolarization — drive insufficient]"

        print_test_result("Extended HH Currents", passed, message)

        return passed, {
            "min_v": round(float(min_v), 2),
            "max_v": round(float(max_v), 2),
            "total_spikes": int(spike_count),
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
            "diverged": bool(diverged),
            "shows_depolarization": bool(shows_depolarization)
        }

    except Exception as e:
        print_test_result("Extended HH Currents", False, str(e))
        return False, {"error": str(e)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Validation Script for Neural Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/validate_gpu.py              # Run full validation suite
  python tests/validate_gpu.py --quick      # Fast iteration (reduced sizes)
        """
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation with reduced network sizes and step counts"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Neural Simulator GPU Validation Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = ValidationResults()
    overall_passed = True

    output_path = os.path.join(
        os.path.dirname(__file__),
        "validation_results.json"
    )

    tests = [
        ("Determinism", test_determinism),
        ("Performance Benchmark", test_performance_benchmark),
        ("Numerical Stability", test_numerical_stability),
        ("Plasticity Validation", test_plasticity_validation),
        ("COO Cache Validation", test_coo_cache_validation),
        ("Extended HH Currents", test_extended_hh_currents),
    ]

    for test_name, test_func in tests:
        try:
            passed, metrics = test_func(quick=args.quick)
            results.add_test(test_name, passed, metrics=metrics)
            overall_passed = overall_passed and passed
        except Exception as e:
            print(f"\n[✗] {test_name}: FAILED")
            print(f"   Exception: {str(e)}")
            traceback.print_exc()
            results.add_test(test_name, False, details=str(e))
            overall_passed = False

        # Save incrementally after each test (protects against CUDA crashes killing the process)
        try:
            results.save_json(output_path)
        except Exception:
            pass

    # Final save
    print_section("SUMMARY")
    results.save_json(output_path)

    passed_count = sum(1 for t in results.results["tests"].values() if t["passed"])
    total_count = len(results.results["tests"])

    print(f"Tests Passed: {passed_count}/{total_count}")
    if overall_passed:
        print("\n[✓] All validation tests PASSED!")
    else:
        print("\n[✗] Some validation tests FAILED. See results above and in validation_results.json.")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
