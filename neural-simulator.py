# main_app.py
import dearpygui.dearpygui as dpg
import os
import json
import time
import numpy as np
import random
from enum import Enum
from collections import deque
import threading
import sys
import h5py
import math
import queue
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict
from sim.enums import (NeuronModel, NeuronType, DefaultHodgkinHuxleyParams,
                        StimulusPatternType, NeuronGroupRole, ExperimentPhaseType,
                        TrainingMode,
                        DefaultIzhikevichParamsManager, NeuronTypeIDMapper, NEURON_TYPE_MAPPER)
from sim.config import (CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
                         ReadoutConfig, TrainingConfig, StimulusPattern, StimulusChannel,
                         NeuronGroup, ExperimentPhase, ExperimentConfig,
                         _create_config_from_dict, _get_full_config_dict)

# Optional: hdf5plugin for LZ4 compression (faster than gzip)
try:
    import hdf5plugin
    HAS_HDF5PLUGIN = True
except ImportError:
    HAS_HDF5PLUGIN = False
    # Fallback warning printed later when needed

# Optional: psutil for CPU memory monitoring during recording
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # CPU memory monitoring disabled without psutil

# === LOG CAPTURE SYSTEM ===
# Initialize IMMEDIATELY after imports to capture ALL print output
class LogCapture:
    """Thread-safe log capture system for displaying console output in the GUI."""
    def __init__(self, max_lines=5000):
        self.max_lines = max_lines
        self.log_buffer = []
        self.lock = threading.Lock()
        self.original_stdout = None
        self.original_stderr = None
        self.enabled = False
    
    def start_capture(self):
        """Begin capturing print statements and stderr."""
        if self.enabled:
            return
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self.enabled = True
    
    def stop_capture(self):
        """Restore original stdout/stderr."""
        if not self.enabled:
            return
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.enabled = False
    
    def write(self, text):
        """Called by print() to capture output."""
        # Write to original stdout as well
        if self.original_stdout:
            self.original_stdout.write(text)
        
        # Add to buffer
        if text and text.strip():
            with self.lock:
                self.log_buffer.append(text.rstrip())
                if len(self.log_buffer) > self.max_lines:
                    self.log_buffer = self.log_buffer[-self.max_lines:]
    
    def flush(self):
        """Required for file-like object interface."""
        if self.original_stdout:
            self.original_stdout.flush()
    
    def get_logs(self):
        """Get all captured log lines."""
        with self.lock:
            return self.log_buffer.copy()
    
    def clear(self):
        """Clear the log buffer."""
        with self.lock:
            self.log_buffer.clear()
    
    def search(self, query, case_sensitive=False):
        """Find all line indices containing the search query."""
        with self.lock:
            if not case_sensitive:
                query = query.lower()
            matches = []
            for i, line in enumerate(self.log_buffer):
                search_text = line if case_sensitive else line.lower()
                if query in search_text:
                    matches.append(i)
            return matches

# Initialize global log capture immediately
_global_log_capture = LogCapture(max_lines=5000)
_global_log_capture.start_capture()

# === END LOG CAPTURE SYSTEM ===

# Attempt to get screen resolution using tkinter
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 760 # Default values
try:
    import tkinter
    root = tkinter.Tk()
    root.withdraw() # Hide the main window
    SCREEN_WIDTH = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()
    root.destroy()
    print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
except Exception as e:
    print(f"Could not detect screen resolution using tkinter: {e}. Using defaults {SCREEN_WIDTH}x{SCREEN_HEIGHT}.")

# OpenGL and GLUT imports
try:
    from OpenGL.GL import *
    import OpenGL.GLUT as glut
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
    print("PyOpenGL found. OpenGL visualization will be used.")
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL or its dependencies not found. Visualization will be text-based if possible, or disabled.")
    print("Install with: pip install PyOpenGL PyOpenGL_accelerate")


import cupy as cp
try:
    import cupy.sparse as csp
except (ImportError, ModuleNotFoundError):
    import cupyx.scipy.sparse as csp
print("CuPy initialized for GPU acceleration.")

RECORDING_FORMAT_VERSION = "1.1.0-h5" # Version for .simrec.h5 files

# --- Performance Tuning Constants ---
# For 60fps: 1000ms / 60fps = ~16.67ms per frame
# With dt=1.0ms: 16.67ms / 1.0ms ≈ 17 steps
# Adjust based on your actual dt if different
SYNAPSE_SAMPLE_UPDATE_INTERVAL_STEPS = 17  # Update synapse samples for ~60fps visualization

# --- Threading Globals ---
simulation_thread = None
ui_to_sim_queue = queue.Queue()
sim_to_ui_queue = queue.Queue()
# shutdown_flag is already a threading.Event, will be initialized later in main
# gl_data_lock is already a threading.Lock, will be initialized later if OPENGL_AVAILABLE

# --- Configuration & Data Classes ---

# --- Configuration & Data Classes (extracted to sim/config.py) ---
# DefaultIzhikevichParamsManager, NeuronTypeIDMapper, NEURON_TYPE_MAPPER imported from sim.enums
# CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig imported from sim.config
# _create_config_from_dict, _get_full_config_dict imported from sim.config

# --- Auto-tuned override support ---
AUTO_TUNED_OVERRIDES_PATH = os.path.join("simulation_profiles", "auto_tuned_overrides.json")
AUTO_TUNED_OVERRIDES = None  # Lazy-loaded mapping from combo key -> overrides dict

# --- Performance test stop flag ---
performance_test_stop_flag = threading.Event()  # Global flag to signal stop for benchmarks/optimization
performance_test_running_type = None  # Track which test is running: "benchmark" or "optimization"


def _load_auto_tuned_overrides_if_needed():
    """Lazily loads auto-tuned overrides from JSON if present.

    File format:
        {
          "schema_version": 1,
          "created_at": "...",
          "tuned_combinations": {
             "MODEL|PROFILE|HH_TYPE_OR_NONE": {"core_overrides": {...}, "metrics": {...}, ...},
             ...
          }
        }
    """
    global AUTO_TUNED_OVERRIDES
    if AUTO_TUNED_OVERRIDES is not None:
        return

    if not os.path.exists(AUTO_TUNED_OVERRIDES_PATH):
        AUTO_TUNED_OVERRIDES = {}
        return

    try:
        with open(AUTO_TUNED_OVERRIDES_PATH, "r") as f:
            data = json.load(f)
        tuned_map = data.get("tuned_combinations", {})
        if isinstance(tuned_map, dict):
            AUTO_TUNED_OVERRIDES = tuned_map
        else:
            AUTO_TUNED_OVERRIDES = {}
        print(f"Loaded {len(AUTO_TUNED_OVERRIDES)} auto-tuned combinations from {AUTO_TUNED_OVERRIDES_PATH}.")
    except Exception as e:
        print(f"Warning: Failed to load auto-tuned overrides from {AUTO_TUNED_OVERRIDES_PATH}: {e}")
        AUTO_TUNED_OVERRIDES = {}


def get_auto_tuned_overrides_for_combo(neuron_model_type_str, profile_name_str, default_hh_type_str=None):
    """Returns auto-tuned overrides dict for a given (model, profile, HH preset) combo, if available.

    The key format is "MODEL|PROFILE|HH_TYPE_OR_NONE". For non-HH models we allow HH type to be "NONE".
    """
    _load_auto_tuned_overrides_if_needed()
    if not AUTO_TUNED_OVERRIDES:
        return None

    key_full = f"{neuron_model_type_str}|{profile_name_str}|{default_hh_type_str or 'NONE'}"
    entry = AUTO_TUNED_OVERRIDES.get(key_full)

    # For non-HH models, also allow a generic per-(model,profile) entry with HH type NONE
    if entry is None and neuron_model_type_str != NeuronModel.HODGKIN_HUXLEY.name:
        key_model_profile = f"{neuron_model_type_str}|{profile_name_str}|NONE"
        entry = AUTO_TUNED_OVERRIDES.get(key_model_profile)

    return entry


# --- Benchmark-derived hardware limits ---
BENCHMARK_RESULTS_PATH = os.path.join("benchmarks", "benchmark_results.json")
HARDWARE_LIMITS = None  # Lazy-loaded dict: model_name -> {max_neurons, max_conn, limits_table, hardware_note}


def _parse_benchmark_limits(results_data):
    """Parses benchmark_results.json and derives per-model hardware limits.

    Builds a table of tested configurations with their performance, and determines
    the maximum neuron/connection counts that succeeded for each model.

    Returns:
        dict: {
            "gpu_name": str,
            "per_model": {
                "IZHIKEVICH": {
                    "max_neurons_tested": 50000,
                    "max_conn_tested": 1000,
                    "realtime_max_neurons": 10000,   # Steps/s >= 1000/dt (real-time threshold)
                    "configs": [  # All tested configs for this model, sorted by size
                        {"neurons": 1000, "conn": 100, "steps_per_sec": 345.0, "mean_ms": 2.9, "gpu_gb": 1.2},
                        ...
                    ]
                },
                ...
            },
            "hardware_note": str  # Human-readable summary
        }
    """
    gpu_info = results_data.get("system_info", {})
    gpu_name = gpu_info.get("gpu_name", "Unknown GPU")
    gpu_mem_gb = gpu_info.get("gpu_memory_gb", 0)

    per_model = {}
    for entry in results_data.get("results", []):
        cfg = entry.get("config", {})
        metrics = entry.get("metrics", {})
        if not cfg or not metrics:
            continue

        model = cfg.get("neuron_model_type", "UNKNOWN")
        dt_ms = cfg.get("dt_ms", 1.0)
        neurons = cfg.get("num_neurons", 0)
        conn = cfg.get("connections_per_neuron", 0)
        steps_per_sec = metrics.get("steps_per_sec", 0)
        mean_ms = metrics.get("step_time_mean_ms", 0)
        gpu_gb = metrics.get("gpu_memory_used_gb", 0)

        if model not in per_model:
            per_model[model] = {
                "max_neurons_tested": 0,
                "max_conn_tested": 0,
                "realtime_max_neurons": 0,
                "dt_ms": dt_ms,
                "configs": []
            }

        info = per_model[model]
        info["configs"].append({
            "neurons": neurons, "conn": conn,
            "steps_per_sec": steps_per_sec, "mean_ms": mean_ms, "gpu_gb": gpu_gb
        })

        if neurons > info["max_neurons_tested"]:
            info["max_neurons_tested"] = neurons
        if conn > info["max_conn_tested"]:
            info["max_conn_tested"] = conn

        # Real-time threshold: steps_per_sec >= 1000/dt_ms (i.e., 1 second of bio time per wall second)
        realtime_threshold = 1000.0 / dt_ms if dt_ms > 0 else 1000.0
        if steps_per_sec >= realtime_threshold and neurons > info["realtime_max_neurons"]:
            info["realtime_max_neurons"] = neurons

    # Sort configs by neuron count then connection count
    for model, info in per_model.items():
        info["configs"].sort(key=lambda x: (x["neurons"], x["conn"]))

    # Build human-readable summary
    model_short = {"IZHIKEVICH": "Izh", "HODGKIN_HUXLEY": "HH", "ADEX": "AdEx"}
    lines = [f"{gpu_name} ({gpu_mem_gb:.0f}GB) — Benchmark Limits:"]
    for model_name in ["IZHIKEVICH", "HODGKIN_HUXLEY", "ADEX"]:
        if model_name not in per_model:
            continue
        info = per_model[model_name]
        short = model_short.get(model_name, model_name[:3])
        max_n = info["max_neurons_tested"]
        max_c = info["max_conn_tested"]
        dt = info["dt_ms"]

        # Find performance range for max neuron count
        max_n_configs = [c for c in info["configs"] if c["neurons"] == max_n]
        if max_n_configs:
            best_steps = max(c["steps_per_sec"] for c in max_n_configs)
            max_gpu = max(c["gpu_gb"] for c in max_n_configs)
            # Bio throughput: steps_per_sec * dt_ms = bio_ms per wall_second
            best_bio_ms_per_s = best_steps * dt
            worst_bio_ms_per_s = min(c["steps_per_sec"] for c in max_n_configs) * dt
            lines.append(f"  {short} (dt={dt}ms): up to {max_n//1000}K neurons, "
                         f"{worst_bio_ms_per_s:.0f}-{best_bio_ms_per_s:.0f} bio-ms/s, "
                         f"{max_gpu:.1f}GB VRAM")

    hardware_note = "\n".join(lines)

    return {
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "per_model": per_model,
        "hardware_note": hardware_note
    }


def _load_benchmark_limits():
    """Loads and parses benchmark results at startup."""
    global HARDWARE_LIMITS
    if not os.path.exists(BENCHMARK_RESULTS_PATH):
        HARDWARE_LIMITS = {}
        return

    try:
        with open(BENCHMARK_RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        HARDWARE_LIMITS = _parse_benchmark_limits(data)
        print(f"Loaded hardware limits from {BENCHMARK_RESULTS_PATH} "
              f"({len(HARDWARE_LIMITS.get('per_model', {}))} models)")
    except Exception as e:
        print(f"Warning: Could not load benchmark limits from {BENCHMARK_RESULTS_PATH}: {e}")
        HARDWARE_LIMITS = {}


def get_hardware_limits_for_model(model_name):
    """Returns hardware limits dict for a specific neuron model, or None if unavailable.

    Returns:
        dict with keys: max_neurons_tested, max_conn_tested, realtime_max_neurons, dt_ms, configs
        or None if no benchmark data exists for this model.
    """
    if HARDWARE_LIMITS is None:
        _load_benchmark_limits()
    if not HARDWARE_LIMITS:
        return None
    return HARDWARE_LIMITS.get("per_model", {}).get(model_name)


def get_hardware_note():
    """Returns the human-readable hardware note from benchmark results."""
    if HARDWARE_LIMITS is None:
        _load_benchmark_limits()
    return HARDWARE_LIMITS.get("hardware_note", "") if HARDWARE_LIMITS else ""


def check_config_against_limits(model_name, num_neurons, conn_per_neuron):
    """Checks a proposed config against benchmark-derived limits.

    Returns:
        tuple: (is_safe: bool, warning_message: str or None)
            is_safe = True means config is within tested limits
            warning_message = None if safe, otherwise a descriptive string
    """
    limits = get_hardware_limits_for_model(model_name)
    if limits is None:
        return True, None  # No benchmark data — can't warn

    max_tested_n = limits["max_neurons_tested"]
    max_tested_c = limits["max_conn_tested"]

    if num_neurons > max_tested_n:
        return False, (f"WARNING: {num_neurons} neurons exceeds benchmark-tested maximum "
                       f"({max_tested_n} for {model_name}). May cause OOM or severe slowdown.")
    if conn_per_neuron > max_tested_c:
        return False, (f"WARNING: {conn_per_neuron} conn/neuron exceeds benchmark-tested maximum "
                       f"({max_tested_c} for {model_name}). May cause OOM.")

    # Check if this specific combo was tested — find closest match
    configs = limits["configs"]
    matching = [c for c in configs if c["neurons"] == num_neurons and c["conn"] == conn_per_neuron]
    if matching:
        gpu_gb = matching[0]["gpu_gb"]
        steps_s = matching[0]["steps_per_sec"]
        return True, None

    # Interpolate: check if a similar-sized config was tested and had high VRAM
    larger_configs = [c for c in configs if c["neurons"] >= num_neurons and c["conn"] >= conn_per_neuron]
    if larger_configs:
        best_match = larger_configs[0]  # Smallest config >= requested
        return True, None

    return True, None  # Within max bounds but exact combo not tested — assume OK


# Compatibility class for old SimulationConfiguration usage
class SimulationConfiguration:
    """Legacy configuration class for backward compatibility. Wraps the new dataclass structure."""
    def __init__(self):
        # Core Simulation Timing & Structure
        self.total_simulation_time_ms = 60000.0 # Total duration of the simulation in milliseconds
        self.dt_ms = 1.000 # Simulation time step in milliseconds (e.g., 0.1 ms for Izh, 0.025 for HH)
        self.num_neurons = 1000 # Total number of neurons in the network
        self.connections_per_neuron = 100 # Average number of outgoing connections per neuron (for spatial/random)
        self.num_traits = 5 # Number of distinct neuron traits/types for coloring/behavioral differences
        self.seed = -1 # Random seed for reproducibility (-1 for random initialization)

        # Neuron Model Selection
        self.neuron_model_type = NeuronModel.IZHIKEVICH.name # Current neuron model ('IZHIKEVICH', 'HODGKIN_HUXLEY', or 'ADEX')
        self.default_neuron_type_izh = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name # Default Izhikevich type if trait mapping fails
        self.default_neuron_type_hh = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name # Default Hodgkin-Huxley type

        # High-level structural profile (brain region / mode)
        self.neural_profile_name = "GENERIC_UNSTRUCTURED"

        # Izhikevich Model Parameters (2007 Formulation - Global defaults, can be overridden per-neuron by trait)
        # These are initialized from a default Izhikevich neuron type (e.g., RS Cortical Pyramidal)
        rs_params_2007 = DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]
        self.izh_C_val = rs_params_2007["C"]       # Membrane capacitance (pF)
        self.izh_k_val = rs_params_2007["k"]       # Constant related to Na+ channel kinetics (nS/mV or similar)
        self.izh_vr_val = rs_params_2007["vr"]     # Resting membrane potential (mV)
        self.izh_vt_val = rs_params_2007["vt"]     # Instantaneous threshold potential (mV)
        self.izh_vpeak_val = rs_params_2007["vpeak"] # Spike cutoff/peak value (mV)
        self.izh_a_val = rs_params_2007["a"]       # Timescale of recovery variable u (1/ms)
        self.izh_b_val = rs_params_2007["b"]       # Sensitivity of u to subthreshold fluctuations (nS)
        self.izh_c_val = rs_params_2007["c_reset"] # After-spike reset value of v (mV)
        self.izh_d_val = rs_params_2007["d_increment"] # After-spike increment of u (pA)

        # Legacy Izhikevich Model Parameters (Not actively used if 2007 formulation is primary)
        self.lif_v_rest = -65.0 # Legacy resting potential (mV) - used if no Izhikevich params available
        self.initial_firing_threshold = -40.0 # Legacy firing threshold (mV)
        self.initial_threshold_variation = 1.0 # Variation for legacy threshold (mV)

        # Hodgkin-Huxley Model Parameters (Global defaults, can be overridden per-neuron by trait)
        # Initialized from a default HH neuron type (e.g., L5 Cortical Pyramidal RS)
        hh_defaults = DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]
        self.hh_C_m = hh_defaults["C_m"]             # Membrane capacitance (uF/cm^2)
        self.hh_g_Na_max = hh_defaults["g_Na_max"]   # Max Na+ conductance (mS/cm^2)
        self.hh_g_K_max = hh_defaults["g_K_max"]     # Max K+ conductance (mS/cm^2)
        self.hh_g_L = hh_defaults["g_L"]             # Leak conductance (mS/cm^2)
        self.hh_E_Na = hh_defaults["E_Na"]           # Na+ reversal potential (mV)
        self.hh_E_K = hh_defaults["E_K"]             # K+ reversal potential (mV)
        self.hh_E_L = hh_defaults["E_L"]             # Leak reversal potential (mV)
        self.hh_v_rest_init = hh_defaults["v_rest_hh"] # Initial resting Vm for HH model (mV)
        self.hh_v_peak = hh_defaults["v_peak_hh"]    # Spike peak for HH model (mV)
        self.hh_m_init = hh_defaults["m_init"]       # Initial m gating variable value
        self.hh_h_init = hh_defaults["h_init"]       # Initial h gating variable value
        self.hh_n_init = hh_defaults["n_init"]       # Initial n gating variable value
        self.hh_temperature_celsius = 37.0           # Temperature for HH kinetics (Celsius)
        self.hh_q10_factor = 3.0                     # Q10 temperature coefficient for HH rates
        # Optional extended HH currents (all disabled by default)
        self.hh_g_M_max = hh_defaults.get("g_M_max", 0.0)            # Max M-current conductance (mS/cm^2); 0 disables
        self.hh_m_current_tau_ms = 100.0 # Approximate activation time constant for M-current (ms)
        self.hh_g_CaT_max = hh_defaults.get("g_CaT_max", 0.0)
        self.hh_E_CaT = hh_defaults.get("E_CaT", 120.0)
        self.hh_g_h_max = hh_defaults.get("g_h_max", 0.0)
        self.hh_E_h = hh_defaults.get("E_h", -30.0)
        self.hh_g_NaP_max = hh_defaults.get("g_NaP_max", 0.0)

        # AdEx Model Parameters (Adaptive Exponential IF)
        self.adex_C = 281.0          # Membrane capacitance (pF)
        self.adex_g_L = 30.0         # Leak conductance (nS)
        self.adex_E_L = -70.6        # Leak reversal (mV)
        self.adex_V_T = -50.4        # Threshold (mV)
        self.adex_Delta_T = 2.0      # Slope factor (mV)
        self.adex_a = 4.0            # Subthreshold coupling (nS)
        self.adex_tau_w = 144.0      # Adaptation time constant (ms)
        self.adex_b = 80.5           # Spike-triggered increment (pA)
        self.adex_V_r = -70.6        # Reset voltage (mV)
        self.adex_V_peak = -40.0     # Spike detection threshold (mV)

        # External drive scaling (tuned per model/profile; 1.0 = baseline range)
        self.hh_external_drive_scale = 1.0
        self.adex_external_drive_scale = 1.0

        # Basic Neuron & Synapse Properties
        self.refractory_period_steps = 2 # Absolute refractory period in simulation steps (dt units)
        self.syn_reversal_potential_e = 0.0 # Reversal potential for excitatory synapses (mV)
        self.syn_reversal_potential_i = -75.0 # Reversal potential for inhibitory synapses (mV) — Cl- Nernst at 37C
        self.syn_tau_g_e = 5.0 # Time constant for excitatory synaptic conductance decay (ms)
        self.syn_tau_g_i = 10.0 # Time constant for inhibitory synaptic conductance decay (ms)
        self.propagation_strength = 0.05 # Scaling factor for excitatory synaptic conductance increase per spike
        self.inhibitory_propagation_strength = 0.105 # Scaled for E_inh=-75mV (was 0.15 at -70mV)
        self.max_synaptic_delay_ms = 20.0 # Maximum synaptic delay in ms (Not fully implemented for individual delays yet)

        # Inhibitory Neuron Configuration
        self.enable_inhibitory_neurons = True # Whether to model inhibitory neurons
        self.inhibitory_trait_index = 1 # Trait index designated as inhibitory (0-indexed)
        self.inhibitory_trait_indices = [] # Optional list of inhibitory trait indices (overrides inhibitory_trait_index if non-empty)

        # Hebbian Learning / Long-Term Potentiation (LTP)
        self.enable_hebbian_learning = True # Enable Hebbian-like weight potentiation
        self.hebbian_learning_rate = 0.0005 # Learning rate for LTP
        self.hebbian_weight_decay = 0.00001 # Multiplicative weight decay factor per step
        self.hebbian_min_weight = 0.05 # Minimum synaptic weight
        self.hebbian_max_weight = 1.0 # Maximum synaptic weight

        # Short-Term Plasticity (STP) - Tsodyks-Markram model
        self.enable_short_term_plasticity = True # Enable STP
        self.stp_U = 0.15 # STP U parameter (baseline utilization of synaptic resources)
        self.stp_tau_d = 200.0 # STP tau_d (depression time constant, ms)
        self.stp_tau_f = 50.0 # STP tau_f (facilitation time constant, ms)
        # Per-connection-type STP [E->E, E->I, I->E, I->I]
        self.enable_per_type_stp = True
        self.stp_U_per_type = [0.5, 0.5, 0.25, 0.25]
        self.stp_tau_d_per_type = [200.0, 200.0, 100.0, 100.0]
        self.stp_tau_f_per_type = [20.0, 20.0, 50.0, 50.0]

        # Homeostatic Plasticity (Adaptive Thresholds for Izhikevich model)
        self.enable_homeostasis = True # Enable homeostatic threshold adaptation
        self.homeostasis_target_rate = 0.02 # Target firing rate (spikes per dt step)
        self.homeostasis_threshold_adapt_rate = 0.0005 # Adaptation rate for firing thresholds (slower, biologically grounded)
        self.homeostasis_ema_alpha = 0.0002 # Alpha for EMA of neuron activity (tau ~5s at dt=1ms)
        self.homeostasis_threshold_min = -55.0 # Minimum firing threshold (mV)
        self.homeostasis_threshold_max = -30.0 # Maximum firing threshold (mV)

        # Synaptic Scaling (Turrigiano 2008) - multiplicative excitatory weight scaling
        self.enable_synaptic_scaling = False
        self.synaptic_scaling_rate = 0.001 # Slow scaling rate (operates on seconds timescale)

        # NMDA conductance with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)
        self.enable_nmda = False
        self.nmda_ratio = 0.4             # NMDA:AMPA conductance ratio
        self.nmda_tau_decay = 100.0       # NMDA decay time constant (ms)
        self.nmda_tau_rise = 3.0          # NMDA rise time constant (ms)
        self.nmda_mg_concentration = 1.0  # Extracellular [Mg²⁺] in mM

        # STDP (Spike-Timing Dependent Plasticity)
        self.enable_stdp = True
        self.stdp_a_plus = 0.012          # LTP amplitude (biased > A- for net potentiation)
        self.stdp_a_minus = 0.01          # LTD amplitude
        self.stdp_tau_plus_ms = 20.0      # LTP time constant (ms)
        self.stdp_tau_minus_ms = 20.0     # LTD time constant (ms)
        self.stdp_w_min = 0.0             # Minimum STDP weight
        self.stdp_w_max = 2.0             # Maximum STDP weight
        self.stdp_only_nearest_spike = True

        # Reward-Modulated Plasticity
        self.enable_reward_modulation = True
        self.reward_learning_rate = 0.01
        self.reward_eligibility_tau_ms = 1000.0
        self.reward_baseline = 0.0
        self.current_reward_signal = 0.0

        # Structural Plasticity
        self.enable_structural_plasticity = True
        self.struct_plast_formation_rate = 1e-6
        self.struct_plast_elimination_rate = 5e-7
        self.struct_plast_weight_threshold = 0.05
        self.struct_plast_target_density = 0.1
        self.struct_plast_distance_kernel = "exp_decay"
        self.struct_plast_distance_scale = 20.0
        self.struct_plast_update_interval_steps = 100
        self.struct_plast_activity_bias = 0.5  # Co-activity bias for synapse formation

        # Parameter Heterogeneity (Phase B2)
        self.enable_parameter_heterogeneity = False # Enable per-neuron parameter variability
        self.heterogeneity_seed = -1 # Seed for heterogeneity sampling (-1 = use main seed)
        self.heterogeneity_distributions = {} # Dict of parameter distributions (empty = use defaults)
        
        # Enhanced Channel Noise (Phase B4)
        self.enable_conductance_noise = False # Enable multiplicative conductance noise (HH only)
        self.conductance_noise_relative_std = 0.05 # Relative std for conductance noise (5%)
        self.enable_ou_process = False # Enable Ornstein-Uhlenbeck background current
        self.ou_mean_current_pA = 0.0 # OU process mean current (pA)
        self.ou_std_current_pA = 100.0 # OU process std current (pA)
        self.ou_tau_ms = 15.0 # OU process time constant (ms)
        self.ou_seed = -1 # Seed for OU process (-1 = use main seed)
        
        # Hardware Performance Note (populated by viz_benchmark.py)
        self.hardware_performance_note = "" # Note about hardware realtime capacity

        # Network Generation (Watts-Strogatz specific, if spatial fallback is not used)
        self.enable_watts_strogatz = True # Use Watts-Strogatz generator for connections
        self.connectivity_k = 10 # K for Watts-Strogatz (number of nearest neighbors in ring)
        self.connectivity_p_rewire = 0.1 # Rewiring probability for Watts-Strogatz

        # Runtime State (Managed by SimulationBridge, not typically saved in profiles)
        self.current_time_ms = 0.0 # Current simulation time in ms
        self.current_time_step = 0 # Current simulation step
        self.is_running = False # Simulation is actively running
        self.is_paused = False # Simulation is paused
        self.simulation_speed_factor = 1.0 # Multiplier for simulation speed relative to real-time

        # Visualization & Network Definition Data (Populated during initialization)
        self.network_definition_dict = {"neuron_groups": [], "connections": []} # For potential export/import
        self.neuron_positions_x = [] # List of X coordinates for 2D projection/GL
        self.neuron_positions_y = [] # List of Y coordinates for 2D projection/GL
        self.neuron_types_list_for_viz = [] # List of string types for each neuron for visualization filters
        self.max_delay_steps = int(self.max_synaptic_delay_ms / self.dt_ms) if self.dt_ms > 0 else 200 # Max delay in dt steps

        # 3D Visualization & Camera Parameters
        self.volume_min_x = -50.0; self.volume_max_x = 50.0 # X-axis bounds of the simulation volume
        self.volume_min_y = -50.0; self.volume_max_y = 50.0 # Y-axis bounds
        self.volume_min_z = -50.0; self.volume_max_z = 50.0 # Z-axis bounds

        # Camera spherical coordinates for orbiting
        self.camera_center_x = 0.0; self.camera_center_y = 0.0; self.camera_center_z = 0.0 # Point camera looks at
        self.camera_radius = 150.0 # Distance from center to camera
        self.camera_azimuth_angle = 0.0 # Horizontal angle (radians)
        self.camera_elevation_angle = 0.0 # Vertical angle (radians)
        
        # Camera view properties (derived from spherical for gluLookAt, but kept for potential direct use or DPG)
        self.camera_eye_x = 0.0; self.camera_eye_y = 0.0; self.camera_eye_z = self.camera_radius # Initial eye position
        self.camera_up_x = 0.0; self.camera_up_y = 1.0; self.camera_up_z = 0.0 # Up vector for camera

        self.camera_fov = 60.0 # Field of view in degrees
        self.camera_near_clip = 0.1 # Near clipping plane
        self.camera_far_clip = 1000.0 # Far clipping plane

        # Mouse interaction state for camera control
        self.mouse_last_x = 0; self.mouse_last_y = 0 # Last mouse position for calculating deltas
        self.mouse_left_button_down = False # Is left mouse button currently pressed
        self.mouse_right_button_down = False # Is right mouse button currently pressed


    def reset_simulation_time_and_counters(self):
        """Resets time-dependent simulation variables."""
        self.current_time_ms = 0.0
        self.current_time_step = 0

    def to_dict(self):
        """Serializes the configuration to a dictionary."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
        # Ensure Enum types are stored as their names for JSON compatibility
        if isinstance(data.get('neuron_model_type'), Enum):
            data['neuron_model_type'] = data['neuron_model_type'].name
        if isinstance(data.get('default_neuron_type_izh'), Enum):
            data['default_neuron_type_izh'] = data['default_neuron_type_izh'].name
        if isinstance(data.get('default_neuron_type_hh'), Enum):
            data['default_neuron_type_hh'] = data['default_neuron_type_hh'].name
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates a SimulationConfiguration instance from a dictionary with robust type casting."""
        config = cls() # Initialize with class defaults

        def _to_python_bool(value, default_val_if_none=False):
            if value is None:
                return default_val_if_none
            if isinstance(value, bool):
                return value
            if hasattr(value, 'item') and isinstance(value.item(), bool): # For numpy.bool_
                return value.item()
            if isinstance(value, (int, float)): # Treat 0 as False, non-zero as True
                return value != 0
            if isinstance(value, str):
                if value.lower() == 'true':
                    return True
                elif value.lower() == 'false':
                    return False
            # Fallback or warning if conversion is ambiguous for other types
            # print(f"Warning: Could not convert value '{value}' (type: {type(value)}) to bool. Using default: {default_val_if_none}")
            return default_val_if_none

        for key, value_from_data in data.items():
            if hasattr(config, key):
                default_value_for_key = getattr(config, key) # Get default type/value from cls instance

                # Handle Enums first
                if key == 'neuron_model_type' and isinstance(value_from_data, str):
                    try: setattr(config, key, NeuronModel[value_from_data].name)
                    except KeyError: setattr(config, key, cls().neuron_model_type)
                    continue
                elif key == 'default_neuron_type_izh' and isinstance(value_from_data, str):
                    try: setattr(config, key, NeuronType[value_from_data].name)
                    except KeyError: setattr(config, key, cls().default_neuron_type_izh)
                    continue
                elif key == 'default_neuron_type_hh' and isinstance(value_from_data, str):
                    try: setattr(config, key, NeuronType[value_from_data].name)
                    except KeyError: setattr(config, key, cls().default_neuron_type_hh)
                    continue

                # Handle Booleans (many config flags are bools)
                if isinstance(default_value_for_key, bool):
                    setattr(config, key, _to_python_bool(value_from_data, default_value_for_key))
                    continue

                # Handle Integers
                if isinstance(default_value_for_key, int) and not isinstance(default_value_for_key, bool): # Exclude bools here
                    if value_from_data is not None:
                        try:
                            setattr(config, key, int(value_from_data))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {key} value '{value_from_data}' to int. Using default: {default_value_for_key}.")
                            setattr(config, key, default_value_for_key)
                    else: # Value from data is None, use default
                        setattr(config, key, default_value_for_key)
                    continue
                
                # Handle Floats
                if isinstance(default_value_for_key, float):
                    if value_from_data is not None:
                        try:
                            setattr(config, key, float(value_from_data))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {key} value '{value_from_data}' to float. Using default: {default_value_for_key}.")
                            setattr(config, key, default_value_for_key)
                    else: # Value from data is None, use default
                        setattr(config, key, default_value_for_key)
                    continue
                
                # For other types (like lists, dicts, or strings not covered above), assign directly
                # This also handles cases where default_value_for_key is None, and value_from_data might be None or a valid value.
                setattr(config, key, value_from_data)

        # Ensure dt_ms is valid after loading and is float
        config.dt_ms = float(getattr(config, 'dt_ms', 0.1))
        if config.dt_ms <= 0: config.dt_ms = 0.1

        config.max_delay_steps = int(config.max_synaptic_delay_ms / config.dt_ms) if config.dt_ms > 0 else 200
        
        # Ensure camera FOV is float
        config.camera_fov = float(getattr(config, 'camera_fov', 60.0))

        # Re-check critical numeric default fallbacks for parameters that might be missing entirely from older files
        default_instance_for_fallback = cls()
        numeric_param_keys_to_check = [
            'izh_C_val', 'izh_k_val', 'izh_vr_val', 'izh_vt_val', 'izh_vpeak_val',
            'izh_a_val', 'izh_b_val', 'izh_c_val', 'izh_d_val',
            'hh_C_m', 'hh_g_Na_max', 'hh_g_K_max', 'hh_g_L', 'hh_E_Na', 'hh_E_K', 'hh_E_L',
            'hh_v_rest_init', 'hh_v_peak', 'hh_temperature_celsius', 'hh_q10_factor',
            'hh_g_M_max', 'hh_m_current_tau_ms',
            'hh_g_CaT_max', 'hh_E_CaT', 'hh_g_h_max', 'hh_E_h', 'hh_g_NaP_max',
            'adex_C', 'adex_g_L', 'adex_E_L', 'adex_V_T', 'adex_Delta_T', 'adex_a', 'adex_tau_w', 'adex_b', 'adex_V_r', 'adex_V_peak',
            'volume_min_x', 'volume_max_x', 'volume_min_y', 'volume_max_y', 'volume_min_z', 'volume_max_z',
            'camera_eye_x', 'camera_eye_y', 'camera_eye_z',
            'camera_center_x', 'camera_center_y', 'camera_center_z',
            'camera_up_x', 'camera_up_y', 'camera_up_z',
            'camera_near_clip', 'camera_far_clip', 'camera_radius',
            'connections_per_neuron', 'seed', 'num_traits', 'connectivity_k' # ints
        ]
        for param_key in numeric_param_keys_to_check:
            if not hasattr(config, param_key) or getattr(config, param_key) is None:
                fallback_val = getattr(default_instance_for_fallback, param_key)
                setattr(config, param_key, fallback_val) # Set to default
                print(f"Info: {param_key} was missing or None, set to default {fallback_val}")
            # Ensure correct type after potential None or load
            current_val = getattr(config, param_key)
            default_type_val = getattr(default_instance_for_fallback, param_key)
            if isinstance(default_type_val, bool): # Should have been handled by _to_python_bool logic primarily
                setattr(config, param_key, _to_python_bool(current_val, default_type_val))
            elif isinstance(default_type_val, int):
                setattr(config, param_key, int(current_val) if current_val is not None else default_type_val)
            elif isinstance(default_type_val, float):
                setattr(config, param_key, float(current_val) if current_val is not None else default_type_val)

        return config

# Neural structure profiles, connectivity motifs, and helpers imported from sim.profiles
from sim.profiles import (NEURAL_STRUCTURE_PROFILES, CONNECTIVITY_MOTIFS,
                          get_profile_default_hh_type_name,
                          get_compatible_hh_type_names_for_profile,
                          enforce_profile_neuron_type_compatibility)

# --- HDF5 Helper Functions ---
def save_dict_to_hdf5_attrs(h5_group_or_file, data_dict):
    """Saves dictionary items as attributes to an HDF5 group or file."""
    for key, value in data_dict.items():
        try:
            if value is None:
                h5_group_or_file.attrs[key] = "NoneType" # Special string for None
            elif isinstance(value, (list, tuple, dict)):
                 # For complex types, store as JSON string
                h5_group_or_file.attrs[key] = json.dumps(value)
            else:
                h5_group_or_file.attrs[key] = value
        except TypeError as e:
            print(f"Warning: Could not save attribute '{key}' (value: {value}, type: {type(value)}): {e}. Storing as string.")
            try:
                h5_group_or_file.attrs[key] = str(value)
            except Exception as e_str:
                 print(f"ERROR: Failed to store attribute '{key}' even as string: {e_str}")


def load_dict_from_hdf5_attrs(h5_group_or_file):
    """Loads attributes from an HDF5 group or file into a dictionary."""
    data_dict = {}
    for key, value in h5_group_or_file.attrs.items():
        if isinstance(value, str):
            if value == "NoneType":
                data_dict[key] = None
            else:
                try:
                    # Attempt to parse if it's a JSON string
                    data_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Not a JSON string, keep as string
                    data_dict[key] = value
        else:
            data_dict[key] = value
    return data_dict

from sim.connectivity import (generate_spatial_connections_gpu,
                              generate_spatial_connections_chunked,
                              generate_spatial_connections_binned,
                              generate_random_connections_large,
                              generate_spatial_connections_3d,
                              generate_watts_strogatz_3d,
                              generate_motif_connections_3d,
                              _calculate_distances_3d_gpu)

from sim.kernels import (fused_izhikevich_legacy_dynamics_update,
                         fused_izhikevich2007_dynamics_update,
                         fused_hodgkin_huxley_dynamics_update,
                         fused_hh_m_current_update,
                         fused_hh_CaT_current_update,
                         fused_hh_h_current_update,
                         fused_hh_NaP_current_update,
                         fused_adex_dynamics_update,
                         fused_conductance_decay_and_current,
                         fused_nmda_update_and_current,
                         fused_stp_decay_recovery,
                         fused_homeostasis_update,
                         fused_stdp_weight_update,
                         fused_eligibility_trace_decay)


# =============================================================================
# EXPERIMENT & STIMULUS SYSTEM (extracted to experiment/ package)
# =============================================================================
from experiment import (ExperimentEngine, ExperimentPresets, ReadoutEngine,
                         TrainingProtocolEngine, StimulusManager, NeuronGroupManager)
from experiment.engine import experiment_config_from_dict, experiment_config_to_dict

# --- Simulation Bridge (extracted to sim/bridge.py) ---
from sim.bridge import SimulationBridge
import sim.bridge as _bridge_module

# Set legacy SimulationConfiguration reference so bridge can use it in load_checkpoint
SimulationBridge._SimulationConfiguration = SimulationConfiguration

# Original class definition (~4500 lines) moved to sim/bridge.py

# --- Global Variables & GUI State (Must be defined before use in SimulationBridge or DPG) ---
# Note: global_simulation_bridge is initialized in main()
global_gui_state = { # Primarily managed by the UI thread
    "filters_changed": False, # Flag for OpenGL to know if its data needs refresh due to filter changes
    "current_profile_name": "default_profile.json", # Profiles remain .json
    "_was_running_last_frame": False, # Internal DPG loop state tracking
    "show_connections_gl": True, # Toggle for showing synapses in OpenGL
    "neuron_filter_mode_gl": 0, # Example: 0: All, 1: Spiking (actual filtering more complex)
    "_dt_warning_logged": False, # Flag to prevent spamming dt warnings
    "reset_sim_needed_from_ui_change": False, # If UI params changed that require sim reset

    # Recording & Playback State (Managed by UI thread based on commands and sim_thread feedback)
    "is_recording_active": False, # True if UI has commanded start_recording and sim_thread confirmed
    "is_playback_mode_active": False, # True if UI has commanded start_playback and sim_thread confirmed & setup
    "current_playback_frame_index": 0, # Current frame index for playback (UI perspective)
    "active_recording_data_source": None, # Holds dict from _prepare_loaded_recording_metadata (incl. H5 file handle for sim_thread)
    "playback_is_playing_ui": False, # UI's view of whether playback is auto-stepping
    "last_playback_autostep_time_ui": 0.0, # For UI-driven playback timing
    "playback_fps_ui": 30.0, # Target FPS for UI-driven playback
    "loaded_recording_filepath_for_ui": None # Path of the currently loaded recording file
}

# Data cache for visualization (primarily for OpenGL, updated by UI thread from sim_to_ui_queue)
# This data is prepared by the UI thread before being passed to OpenGL functions.
global_viz_data_cache = {
    "neuron_positions_x": [], # For 2D projection (if used)
    "neuron_positions_y": [], # For 2D projection (if used)
    "neuron_types": [], # List of type strings for each neuron (for GL filtering)
    "last_visible_neuron_indices": [], # Indices of neurons actually rendered in last GL frame
    "last_visible_synapse_indices": [], # Indices of synapses rendered
    # --- Data passed from Sim_Thread to UI_Thread for OpenGL update ---
    "gl_render_data_buffer": None, # Holds the latest full data snapshot from sim_thread for GL
    "gl_render_data_available": threading.Event(), # Event to signal new data for GL
    "gl_render_data_lock": threading.Lock() # Lock for accessing gl_render_data_buffer
}


# --- Shutdown Flag & Other Top-Level Globals ---
shutdown_flag = threading.Event() # Global shutdown signal for all threads
last_sim_update_time_dpg = 0.0 # Used in the DPG rendering loop (main thread) if it drives sim steps (not in threaded)

# --- OpenGL Specific Globals & Config ---
# These are primarily accessed by the OpenGL rendering functions, running in the main thread.
gl_neuron_pos_vbo = None # Vertex Buffer Object for neuron positions
gl_neuron_color_vbo = None # VBO for neuron colors
gl_synapse_vertices_vbo = None # VBO for synapse lines
gl_pulse_vertices_vbo = None # VBO for synaptic pulse points

gl_num_neurons_to_draw = 0 # Number of neurons to draw in current GL frame
gl_num_synapse_lines_to_draw = 0 # Number of synapse lines
gl_num_pulses_to_draw = 0 # Number of synaptic pulses

# Frame rate limiting for smooth 60 FPS
gl_last_render_time = 0.0
gl_target_frame_time = 1.0 / 60.0  # 60 FPS = 16.67ms per frame

# FPS counter tracking
gl_frame_times = []  # Rolling window of recent frame times
gl_fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
gl_last_fps_update_time = 0.0
gl_current_fps = 0.0  # Current FPS to display

# NumPy arrays holding data ready for VBO buffering (populated by UI thread before GL render)
gl_neuron_pos_cp = cp.array([], dtype=cp.float32).reshape(0,3) # Changed from _np
gl_neuron_colors_cp = cp.array([], dtype=cp.float32).reshape(0,4) # Changed from _np
gl_connection_vertices_cp = cp.array([], dtype=cp.float32).reshape(0,3) # Changed from _np
gl_pulse_vertices_cp = cp.array([], dtype=cp.float32).reshape(0,3) # Changed from _np

# CUDA-OpenGL interop flag (initialized in init_gl)
cuda_gl_interop_enabled = False


if OPENGL_AVAILABLE:
    # opengl_viz_config is primarily read by GL functions in main thread.
    # Changes from UI (e.g. point size slider) will update this dict in main thread.
    opengl_viz_config = {
        "WINDOW_WIDTH": 800, "WINDOW_HEIGHT": 600, # Initial, updated on reshape
        "POINT_SIZE": 3.0, # Default neuron point size
        "MAX_NEURONS_TO_RENDER": 1000000, # Max neurons GL will attempt to draw (performance cap)
        "MAX_CONNECTIONS_TO_RENDER": 10000000, # Max synapses GL will attempt to draw
        "INACTIVE_NEURON_OPACITY": 0.25, # Base opacity for non-firing neurons
        "FIRING_NEURON_COLOR": [1.0, 1.0, 0.0, 1.0], # RGBA for spiking neurons
        "ACTIVITY_HIGHLIGHT_FRAMES": 7, # Frames a neuron stays highlighted after firing
        "FOOTER_HEIGHT_PIXELS": 75, # Height of text overlay at bottom of GL window
        "SYNAPSE_ALPHA_MODIFIER": 0.50, # Multiplier for base synapse alpha
        "SYNAPSE_BASE_COLOR": [0.4, 0.4, 0.5], # Base RGB for synapses
        "VBO_UPDATE_SKIP": 2, # Update VBOs every Nth render frame (reduces GPU-CPU sync overhead)
        "CAMERA_PAN_SPEED_FACTOR": 0.1, # Mouse pan speed
        "CAMERA_ROTATE_SPEED_FACTOR": 0.005, # Mouse rotate speed
        "CAMERA_ZOOM_SPEED_FACTOR": 20.0, # Mouse scroll zoom speed
        "ENABLE_SYNAPTIC_PULSES": True, # Toggle for visualizing synaptic pulses
        "SYNAPTIC_PULSE_COLOR": [0.7, 0.9, 1.0, 0.9], # RGBA for pulses
        "SYNAPTIC_PULSE_SIZE": 3.0, # Point size for pulses
        "SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES": 5, # How many sim steps a pulse point lasts
    }
    # Color map for neuron traits (RGBA, A is base opacity)
    TRAIT_COLOR_MAP_RAW = [ # Keep raw Python list for DPG UI trait count reference
    [0.8, 0.2, 0.2, 0.85], [0.2, 0.8, 0.2, 0.85], [0.2, 0.2, 0.8, 0.85],
    [0.8, 0.8, 0.2, 0.85], [0.8, 0.2, 0.8, 0.85], [0.2, 0.8, 0.8, 0.85],
    [1.0, 0.5, 0.0, 0.85], [0.5, 0.2, 0.8, 0.85], [0.1, 0.5, 0.5, 0.85],
    [0.7, 0.7, 0.7, 0.85] 
    ] 
    TRAIT_COLOR_MAP_GPU = cp.array(TRAIT_COLOR_MAP_RAW, dtype=cp.float32) if TRAIT_COLOR_MAP_RAW else cp.array([[0.5,0.5,0.5,0.25]], dtype=cp.float32)
# Add more colors to TRAIT_COLOR_MAP_RAW if num_traits can exceed its length
# Ensure cfg_num_traits input in DPG is limited by len(TRAIT_COLOR_MAP_RAW)
    glut_window_id = None # Will store GLUT window ID if created
else: # OpenGL not available
    opengl_viz_config = {}
    TRAIT_COLOR_MAP_RAW = []
    TRAIT_COLOR_MAP_GPU = cp.array([], dtype=cp.float32).reshape(0,4) # Or None, if preferred
    # gl_data_lock is not needed if no GL thread access, but global_viz_data_cache.gl_render_data_lock is general
    glut_window_id = None

# Share opengl_viz_config with bridge module so SimulationBridge can reference it
_bridge_module.opengl_viz_config = opengl_viz_config

# Share opengl_viz_config with bridge module so SimulationBridge can reference it
_bridge_module.opengl_viz_config = opengl_viz_config

# --- OpenGL Visualization (extracted to viz/ package) ---
from viz.renderer import (
    init_gl, update_gl_data, render_scene_gl, fast_vbo_update,
    get_color_for_trait, apply_neuron_filters_to_indices,
    apply_synapse_filters_to_indices, trigger_filter_update_signal,
    get_current_filter_settings_from_gui,
)
from viz.camera import (
    mouse_button_func_gl, mouse_motion_func_gl, keyboard_func_gl,
    reshape_gl_window,
)
from viz.overlays import render_text_gl
import viz.overlays as _viz_overlays
import viz.renderer as _viz_renderer
import viz.camera as _viz_camera

# Set viz overlay config reference
_viz_overlays.set_viz_config(opengl_viz_config)

# --- DearPyGUI UI (extracted to ui/ package) ---
from ui.callbacks import (
    init_callbacks as _init_ui_callbacks,
    update_status_bar,
    _update_sim_config_from_ui,
    _populate_ui_from_config_dict,
    _toggle_model_specific_params_visibility,
    update_ui_for_simulation_run_state,
    update_ui_for_recording_state,
    update_ui_for_playback_mode_state,
    update_ui_for_playback_playing_state,
    update_ui_after_recording_loaded,
    update_monitoring_overlay_values,
    get_current_gui_configuration_dict,
    apply_gui_configuration_core,
    handle_start_simulation_event,
    handle_stop_simulation_event,
    handle_pause_simulation_event,
    handle_apply_config_changes_and_reset,
    handle_log_search_change,
    _handle_experiment_preset_change,
    _update_experiment_ui_from_status,
    _scan_profile_directory,
    _handle_full_profile_dropdown_change,
    _refresh_full_profile_dropdown,
    _execute_profile_load_on_ui_thread,
)
from ui.layout import create_gui_layout, add_parameter_table_row


def _init_shared_state_for_modules():
    """Initialize shared state references in viz/ and ui/ modules.
    Called once from main() after global_simulation_bridge is created."""
    # Initialize viz modules
    _viz_renderer.set_shared_state(
        sim_bridge=global_simulation_bridge,
        gui_state=global_gui_state,
        viz_data_cache=global_viz_data_cache,
        viz_config=opengl_viz_config,
        trait_color_map_raw=TRAIT_COLOR_MAP_RAW,
        trait_color_map_gpu=TRAIT_COLOR_MAP_GPU,
        shutdown_evt=shutdown_flag,
        ui_sim_queue=ui_to_sim_queue,
        neuron_type_mapper=NEURON_TYPE_MAPPER,
        update_run_state_fn=update_ui_for_simulation_run_state,
        update_status_fn=update_status_bar,
    )
    _viz_camera.set_shared_state(
        sim_bridge=global_simulation_bridge,
        gui_state=global_gui_state,
        viz_config=opengl_viz_config,
        shutdown_evt=shutdown_flag,
        ui_sim_queue=ui_to_sim_queue,
        filter_fn=trigger_filter_update_signal,
        update_run_state_fn=update_ui_for_simulation_run_state,
    )

    # Initialize ui callbacks module
    _init_ui_callbacks(
        global_simulation_bridge=global_simulation_bridge,
        global_gui_state=global_gui_state,
        global_viz_data_cache=global_viz_data_cache,
        opengl_viz_config=opengl_viz_config,
        OPENGL_AVAILABLE=OPENGL_AVAILABLE,
        TRAIT_COLOR_MAP_RAW=TRAIT_COLOR_MAP_RAW,
        shutdown_flag=shutdown_flag,
        ui_to_sim_queue=ui_to_sim_queue,
        sim_to_ui_queue=sim_to_ui_queue,
        SimulationConfiguration=SimulationConfiguration,
        NeuronModel=NeuronModel,
        NeuronType=NeuronType,
        NeuronGroupRole=NeuronGroupRole,
        ExperimentPhaseType=ExperimentPhaseType,
        DefaultHodgkinHuxleyParams=DefaultHodgkinHuxleyParams,
        DefaultIzhikevichParamsManager=DefaultIzhikevichParamsManager,
        NEURON_TYPE_MAPPER=NEURON_TYPE_MAPPER,
        CoreSimConfig=CoreSimConfig,
        VisualizationConfig=VisualizationConfig,
        RuntimeState=RuntimeState,
        GPUConfig=GPUConfig,
        _create_config_from_dict=_create_config_from_dict,
        _get_full_config_dict=_get_full_config_dict,
        NEURAL_STRUCTURE_PROFILES=NEURAL_STRUCTURE_PROFILES,
        get_compatible_hh_type_names_for_profile=get_compatible_hh_type_names_for_profile,
        get_auto_tuned_overrides_for_combo=get_auto_tuned_overrides_for_combo,
        check_config_against_limits=check_config_against_limits,
        get_hardware_limits_for_model=get_hardware_limits_for_model,
        ExperimentConfig=ExperimentConfig,
        ExperimentPresets=ExperimentPresets,
        StimulusPattern=StimulusPattern,
        StimulusChannel=StimulusChannel,
        NeuronGroup=NeuronGroup,
        ExperimentPhase=ExperimentPhase,
        ReadoutConfig=ReadoutConfig,
        TrainingConfig=TrainingConfig,
        StimulusPatternType=StimulusPatternType,
        experiment_config_to_dict=experiment_config_to_dict,
        experiment_config_from_dict=experiment_config_from_dict,
        performance_test_stop_flag=performance_test_stop_flag,
        BENCHMARK_RESULTS_PATH=BENCHMARK_RESULTS_PATH,
        _load_auto_tuned_overrides_if_needed=_load_auto_tuned_overrides_if_needed,
        _load_benchmark_limits=_load_benchmark_limits,
        get_hardware_note=get_hardware_note,
    )


def simulation_worker_loop(sim_bridge, local_shutdown_event, command_q, data_q):
    """
    Main loop for the dedicated simulation thread.
    Handles commands from the UI thread and runs simulation steps.
    """
    print("Simulation worker thread started.")
    # sim_bridge.ui_queue is already set to the global sim_to_ui_queue in its __init__
    
    last_sim_batch_time = time.perf_counter()
    # Max number of simulation steps to run before checking command queue / yielding
    # This helps keep the simulation thread responsive to commands.
    MAX_STEPS_PER_BATCH = 60 # e.g., if dt=0.1ms, this is 10ms of sim time
    # How often to send data updates to UI (in terms of simulation steps)
    # Lower = more responsive visualization at cost of more GPU→CPU transfers
    # For 60 FPS visualization: Update every 1-2 steps for real-time display
    DATA_UPDATE_INTERVAL_STEPS = 10 # GUI update every 10 steps (~100 FPS at dt=1ms, was 1 = every step)
    SYNAPSE_SAMPLE_UPDATE_INTERVAL_STEPS = 200 # Update synapse samples much less frequently

    try:
        while not local_shutdown_event.is_set():
            # --- 1. Process Commands from UI Thread ---
            try:
                # Use exception handling instead of empty() check to avoid TOCTOU race
                while True:
                    try:
                        command = command_q.get_nowait()
                    except queue.Empty:
                        break
                    cmd_type = command.get("type")

                    if cmd_type == "START_SIM":
                        sim_bridge.start_simulation()
                    elif cmd_type == "STOP_SIM":
                        sim_bridge.stop_simulation()
                    elif cmd_type == "PAUSE_SIM":
                        sim_bridge.pause_simulation()
                    elif cmd_type == "RESUME_SIM":
                        sim_bridge.resume_simulation()
                    elif cmd_type == "STEP_SIM_ONE_MS":
                        dt_ms_val = sim_bridge.core_config.dt_ms if sim_bridge.core_config.dt_ms > 0 else 0.1
                        steps_for_1ms = max(1, int(round(1.0 / dt_ms_val)))
                        sim_bridge.step_simulation(num_steps=steps_for_1ms)
                    elif cmd_type == "APPLY_CONFIG_AND_RESET":
                        sim_bridge.apply_simulation_configuration_core(command["config_dict"])
                        # After applying, send back the (potentially modified by from_dict) config and initial data
                        if sim_bridge.is_initialized:
                            data_q.put({
                                "type": "CONFIG_APPLIED_AND_RESET_DONE",
                                "new_config_dict": sim_bridge.get_current_simulation_configuration_dict(),
                                "initial_gui_data": sim_bridge.get_initial_sim_data_snapshot()
                            })
                        else:
                            data_q.put({"type": "CONFIG_APPLIED_ERROR", "reason": "Initialization failed after apply"})
                    elif cmd_type == "SET_SIM_SPEED":
                        sim_bridge.set_simulation_speed_factor(command["factor"])
                    elif cmd_type == "SAVE_CHECKPOINT":
                        sim_bridge.save_checkpoint(command["filepath"]) # Sim_bridge will send status to UI
                    elif cmd_type == "LOAD_CHECKPOINT":
                        sim_bridge.load_checkpoint(command["filepath"]) # Sim_bridge sends status/data
                    elif cmd_type == "SET_RECORDING_OPTIONS":
                        # Update gpu_config with recording options from UI
                        if "recording_mode" in command:
                            sim_bridge.gpu_config.recording_mode = command["recording_mode"]
                        if "recording_skip_synaptic_data" in command:
                            sim_bridge.gpu_config.recording_skip_synaptic_data = command["recording_skip_synaptic_data"]
                        if "recording_frame_skip" in command:
                            sim_bridge.gpu_config.recording_frame_skip = max(1, command["recording_frame_skip"])
                        sim_bridge._log_console(
                            f"Recording options set: mode={sim_bridge.gpu_config.recording_mode}, "
                            f"skip_synaptic={sim_bridge.gpu_config.recording_skip_synaptic_data}, "
                            f"frame_skip={sim_bridge.gpu_config.recording_frame_skip}"
                        )
                    elif cmd_type == "START_RECORDING":
                        sim_bridge.start_recording_to_file(command["filepath"])  # Sim_bridge sends status
                    elif cmd_type == "STOP_RECORDING":
                        sim_bridge.stop_recording() # Sim_bridge sends status
                    elif cmd_type == "LOAD_RECORDING": # UI requests sim_thread to prepare metadata
                        stream_only = command.get("stream_only", False)
                        sim_bridge.load_recording(command["filepath"], stream_only=stream_only) # Sim_bridge sends RECORDING_METADATA_PREPARED or _FAILED
                    elif cmd_type == "SETUP_PLAYBACK_FROM_RECORDING":
                        # This command implies UI has received RECORDING_METADATA_PREPARED
                        # and now tells sim_thread to use that data to set its state.
                        # The 'active_recording_data_source' is UI state. Sim thread needs the HDF5 group/handle.
                        # This flow needs refinement: SimThread should hold its own H5 handle for playback.
                        # When UI commands "LOAD_RECORDING", sim_thread opens file, prepares meta, keeps handle.
                        # When UI commands "ENTER_PLAYBACK_MODE", sim_thread uses its handle.
                        active_playback_handle = command.get("h5_file_handle_for_sim_thread") # UI must pass this
                        initial_state_group_name = command.get("initial_state_group_name", "initial_state")
                        
                        if active_playback_handle and hasattr(active_playback_handle, 'get'): # Check if it's a valid h5py group/file
                            initial_state_group = active_playback_handle.get(initial_state_group_name)
                            if initial_state_group:
                                sim_bridge._apply_config_and_initial_state_from_recording(
                                    command["config_snapshot"], 
                                    initial_state_group # Pass the HDF5 group object
                                )
                            else:
                                sim_bridge._log_to_ui(f"Playback setup error: initial_state group '{initial_state_group_name}' not found in HDF5.", "error")
                                data_q.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Initial state group missing"})
                        else:
                            sim_bridge._log_to_ui("Playback setup error: Invalid HDF5 handle provided to sim_thread.", "error")
                            data_q.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Invalid H5 handle"})

                    elif cmd_type == "SET_PLAYBACK_FRAME":
                        active_playback_handle_for_frame = command.get("h5_file_handle_for_sim_thread")
                        num_frames_for_prefetch = command.get("num_frames")
                        if active_playback_handle_for_frame:
                            sim_bridge.set_playback_frame(
                                command["frame_index"],
                                active_playback_handle_for_frame,
                                num_frames=num_frames_for_prefetch
                            )
                        else:
                             sim_bridge._log_to_ui("Playback error: No HDF5 handle for SET_PLAYBACK_FRAME.", "error")
                             data_q.put({"type": "PLAYBACK_ERROR", "reason": "Missing H5 handle for frame set"})
                    elif cmd_type == "EXIT_PLAYBACK_MODE":
                        # Sim_thread doesn't directly manage global_gui_state.is_playback_mode_active.
                        # It just needs to reset its internal state if it was in a playback-specific mode.
                        # For example, if it was holding an HDF5 file open for playback, it should close it.
                        # The main task is to re-apply the "live" simulation config.
                        sim_bridge._clear_prefetch_buffer()  # Clean up prefetch resources
                        sim_bridge.apply_simulation_configuration_core(sim_bridge.core_config.to_dict(), is_part_of_playback_setup=False)
                        data_q.put({
                            "type": "PLAYBACK_EXITED_SIM_SIDE",
                            "new_config_dict": sim_bridge.get_current_simulation_configuration_dict(),
                            "initial_gui_data": sim_bridge.get_initial_sim_data_snapshot()
                        })

                    # --- Experiment System Commands ---
                    elif cmd_type == "LOAD_EXPERIMENT_PRESET":
                        preset_name = command.get("preset_name", "")
                        try:
                            exp_config = ExperimentPresets.get_preset(preset_name)
                            if exp_config:
                                sim_bridge.experiment_config = exp_config
                                if sim_bridge.is_initialized:
                                    sim_bridge.experiment_engine = ExperimentEngine(
                                        sim_bridge.core_config.num_neurons,
                                        sim_bridge.core_config.dt_ms
                                    )
                                    sim_bridge.experiment_engine.load_experiment(exp_config)
                                    sim_bridge.experiment_engine.initialize(
                                        cp_traits=sim_bridge.cp_traits, cp_module=cp
                                    )
                                    # Ensure inter-group connectivity for STDP learning
                                    added = sim_bridge.experiment_engine.ensure_inter_group_connectivity(sim_bridge, cp)
                                    if added > 0:
                                        sim_bridge._log_to_ui(
                                            f"Injected {added} inter-group connections for experiment learning paths", "info")
                                data_q.put({
                                    "type": "EXPERIMENT_LOADED",
                                    "name": exp_config.name,
                                    "description": exp_config.description,
                                    "num_phases": len(exp_config.phases),
                                    "num_channels": len(exp_config.stimulus_channels),
                                    "num_groups": len(exp_config.neuron_groups),
                                })
                            else:
                                data_q.put({"type": "EXPERIMENT_ERROR", "reason": f"Unknown preset: {preset_name}"})
                        except Exception as e:
                            data_q.put({"type": "EXPERIMENT_ERROR", "reason": str(e)})

                    elif cmd_type == "LOAD_EXPERIMENT_CONFIG":
                        try:
                            config_dict = command.get("config_dict", {})
                            exp_config = experiment_config_from_dict(config_dict)
                            sim_bridge.experiment_config = exp_config
                            if sim_bridge.is_initialized:
                                sim_bridge.experiment_engine = ExperimentEngine(
                                    sim_bridge.core_config.num_neurons,
                                    sim_bridge.core_config.dt_ms
                                )
                                sim_bridge.experiment_engine.load_experiment(exp_config)
                                sim_bridge.experiment_engine.initialize(
                                    cp_traits=sim_bridge.cp_traits, cp_module=cp
                                )
                                # Ensure inter-group connectivity for STDP learning
                                added = sim_bridge.experiment_engine.ensure_inter_group_connectivity(sim_bridge, cp)
                                if added > 0:
                                    sim_bridge._log_to_ui(
                                        f"Injected {added} inter-group connections for experiment learning paths", "info")
                            data_q.put({
                                "type": "EXPERIMENT_LOADED",
                                "name": exp_config.name,
                                "description": exp_config.description,
                                "num_phases": len(exp_config.phases),
                                "num_channels": len(exp_config.stimulus_channels),
                                "num_groups": len(exp_config.neuron_groups),
                            })
                        except Exception as e:
                            data_q.put({"type": "EXPERIMENT_ERROR", "reason": str(e)})

                    elif cmd_type == "START_EXPERIMENT":
                        if sim_bridge.experiment_engine is not None:
                            sim_bridge.experiment_engine.start(
                                sim_bridge.runtime_state.current_time_ms,
                                sim_bridge_ref=sim_bridge
                            )
                            data_q.put({"type": "EXPERIMENT_STARTED"})
                        else:
                            data_q.put({"type": "EXPERIMENT_ERROR", "reason": "No experiment loaded"})

                    elif cmd_type == "STOP_EXPERIMENT":
                        if sim_bridge.experiment_engine is not None:
                            sim_bridge.experiment_engine.stop()
                            data_q.put({"type": "EXPERIMENT_STOPPED"})

                    elif cmd_type == "GET_EXPERIMENT_STATUS":
                        if sim_bridge.experiment_engine is not None:
                            status = sim_bridge.experiment_engine.get_experiment_status()
                            data_q.put({"type": "EXPERIMENT_STATUS", "status": status})
                        else:
                            data_q.put({"type": "EXPERIMENT_STATUS", "status": {"is_running": False}})

                    elif cmd_type == "SAVE_EXPERIMENT_LOG":
                        if sim_bridge.experiment_engine is not None:
                            filepath = command.get("filepath", "experiment_log.json")
                            try:
                                sim_bridge.experiment_engine.save_log(filepath)
                                data_q.put({"type": "EXPERIMENT_LOG_SAVED", "filepath": filepath})
                            except Exception as e:
                                data_q.put({"type": "EXPERIMENT_ERROR", "reason": f"Log save failed: {e}"})

                    command_q.task_done()
            except queue.Empty:
                pass # No commands from UI

            # --- 2. Run Simulation Step if Active ---
            if sim_bridge.is_initialized and sim_bridge.runtime_state.is_running and \
               not sim_bridge.runtime_state.is_paused:
                
                current_perf_time = time.perf_counter()
                elapsed_real_time_s = current_perf_time - last_sim_batch_time
                dt_ms_val = sim_bridge.core_config.dt_ms
                if dt_ms_val is None or dt_ms_val <= 0: dt_ms_val = 0.1 # Safety

                sim_time_to_advance_ms = elapsed_real_time_s * 1000.0 * sim_bridge.runtime_state.simulation_speed_factor
                
                num_steps_to_run_total = 0
                if dt_ms_val > 0:
                    num_steps_to_run_total = int(sim_time_to_advance_ms / dt_ms_val)

                if num_steps_to_run_total > 0:
                    steps_executed_in_batch = 0
                    for _ in range(min(num_steps_to_run_total, MAX_STEPS_PER_BATCH)):
                        if sim_bridge.runtime_state.current_time_ms < sim_bridge.core_config.total_simulation_time_ms:
                            sim_bridge._run_one_simulation_step() # Core simulation logic
                            sim_bridge.runtime_state.current_time_step += 1
                            # Compute time from step count to avoid floating point drift
                            sim_bridge.runtime_state.current_time_ms = sim_bridge.runtime_state.current_time_step * dt_ms_val
                            steps_executed_in_batch +=1

                            # Periodically send data to UI
                            if sim_bridge.runtime_state.current_time_step % DATA_UPDATE_INTERVAL_STEPS == 0:
                                latest_data = sim_bridge.get_latest_simulation_data_for_gui(force_fetch=True)
                                if data_q and latest_data:
                                    data_q.put({"type": "SIM_DATA_UPDATE", "data": latest_data})
                        else: # Total simulation time reached
                            sim_bridge.stop_simulation() # Sets flags
                            data_q.put({"type": "SIM_STOPPED_OR_ENDED", "reason": "Total time reached"})
                            break 
                    
                    # Adjust last_sim_batch_time based on simulated time processed
                    if steps_executed_in_batch > 0:
                        last_sim_batch_time += (steps_executed_in_batch * dt_ms_val) / \
                                               (sim_bridge.runtime_state.simulation_speed_factor * 1000.0 
                                                if sim_bridge.runtime_state.simulation_speed_factor > 0 else 1000.0)
                else: # No steps to run, but sim is active, so just update time to prevent large jump on resume
                    last_sim_batch_time = time.perf_counter()

            else: # Simulation not running or paused
                last_sim_batch_time = time.perf_counter() # Keep resetting to avoid large jump
                time.sleep(0.01) # Yield CPU if sim is idle or paused
    
    except Exception as e_worker:
        print(f"FATAL ERROR in simulation_worker_loop: {e_worker}")
        import traceback; traceback.print_exc()
        if data_q: data_q.put({"type": "SIM_FATAL_ERROR", "error": str(e_worker)})
    finally:
        print("Simulation worker thread finished.")


def main_dpg_loop_and_gl_idle():
    """
    Main loop for DPG rendering, processing messages from sim_thread, and driving OpenGL updates.
    If GLUT is used, this function is set as GLUT's idle function.
    """
    global global_simulation_bridge, global_gui_state, shutdown_flag, glut_window_id # Ensure glut_window_id is global if used here
    
    if shutdown_flag.is_set(): # Check for shutdown signal
        if OPENGL_AVAILABLE and glut.glutGetWindow() != 0 : # Check if a GLUT window exists
            try:
                current_glut_window = glut.glutGetWindow()
                if current_glut_window != 0: # Ensure we have a valid window ID
                    print(f"Shutdown signaled: Attempting to exit GLUT loop...")
                    # Try glutLeaveMainLoop first (freeglut), fallback to DestroyWindow
                    try:
                        glut.glutLeaveMainLoop()
                    except AttributeError:
                        # glutLeaveMainLoop not available, use DestroyWindow
                        glut.glutDestroyWindow(current_glut_window)
            except Exception as e_glut_shutdown:
                 print(f"Exception during GLUT shutdown: {e_glut_shutdown}")

        # Ensure DPG is also signaled to stop if it hasn't already by the on_close callback.
        if dpg.is_dearpygui_running():
            dpg.stop_dearpygui()

        # Don't call sys.exit() from within GLUT callback - just return and let main loop handle exit
        return

    if not dpg.is_dearpygui_running(): # If DPG window was closed by user (on_close already ran)
        # This block might be redundant if the above shutdown_flag block handles everything,
        # but it's a safeguard.
        if not shutdown_flag.is_set(): # If on_close didn't set it for some reason
            print("DPG not running, setting shutdown_flag from idle loop.")
            shutdown_flag.set() 
        # The shutdown_flag.is_set() block above will then handle GLUT termination.
        return

    # --- 1. Process Messages from Simulation Thread ---
    try:
        while not sim_to_ui_queue.empty():
            message = sim_to_ui_queue.get_nowait()
            msg_type = message.get("type")

            if msg_type == "STATUS_UPDATE":
                update_status_bar(message.get("text","Status N/A"), message.get("color"), message.get("level","info"))
            elif msg_type == "SIM_DATA_UPDATE":
                data_payload = message.get("data")
                if data_payload:
                    update_monitoring_overlay_values(data_payload)
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = data_payload
                    global_viz_data_cache["gl_render_data_available"].set()
                    # Update experiment UI if experiment status is present
                    exp_status = data_payload.get("experiment_status")
                    if exp_status is not None:
                        _update_experiment_ui_from_status(exp_status)
            elif msg_type == "SIM_STOPPED_OR_ENDED":
                global_gui_state["_sim_is_running_ui_view"] = False
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=False, is_paused=False)
                update_status_bar(f"Simulation stopped/ended: {message.get('reason', '')}", level="info")
                if global_simulation_bridge: 
                     initial_data = global_simulation_bridge.get_initial_sim_data_snapshot()
                     if initial_data: update_monitoring_overlay_values(initial_data)
            elif msg_type == "CONFIG_APPLIED_AND_RESET_DONE":
                # Repopulate UI from the configuration actually used by the sim thread.
                # This ensures any profile/model-specific defaults or auto-tuned overrides
                # are reflected in the visible parameters.
                new_cfg_full = message.get("new_config_dict")
                if new_cfg_full:
                    _populate_ui_from_config_dict(new_cfg_full)

                # Update monitoring values and GL snapshot
                update_monitoring_overlay_values(message["initial_gui_data"])
                
                initial_gl_data = message.get("initial_gui_data")
                if initial_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = initial_gl_data
                    global_viz_data_cache["gl_render_data_available"].set() 

                update_ui_for_simulation_run_state(is_running=False, is_paused=False) 
                global_gui_state["reset_sim_needed_from_ui_change"] = False
                update_status_bar("Configuration applied and simulation reset.", color=[0,200,0,255], level="success")
            elif msg_type == "CHECKPOINT_LOADED_SUCCESS":
                _populate_ui_from_config_dict(message["config_dict"])
                apply_gui_configuration_core(message.get("gui_settings_from_checkpoint",{}))
                update_monitoring_overlay_values(message["initial_gui_data"])
                # Also push this initial data to GL cache
                initial_gl_data_chkpt = message.get("initial_gui_data")
                if initial_gl_data_chkpt:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = initial_gl_data_chkpt
                    global_viz_data_cache["gl_render_data_available"].set()

                update_ui_for_simulation_run_state(is_running=False, is_paused=False)
                global_gui_state["reset_sim_needed_from_ui_change"] = False
                update_status_bar("Checkpoint loaded successfully.", color=[0,200,0,255], level="success")
            elif msg_type == "RECORDING_METADATA_PREPARED":
                update_ui_after_recording_loaded(message["data"]) 
            elif msg_type == "RECORDING_STARTED": # Sim thread confirms recording started
                update_ui_for_recording_state(is_recording_active_ui=True)
                update_status_bar(f"Recording started: {os.path.basename(message.get('filepath','N/A'))}", color=[0,150,200,255], level="info")
            elif msg_type == "RECORDING_FINALIZED":
                update_ui_for_recording_state(is_recording_active_ui=False)
                if message.get("success"):
                    update_status_bar(f"Recording saved: {os.path.basename(message.get('filepath','N/A'))}", color=[0,200,0,255], level="success")
                else:
                    update_status_bar("Recording finalization failed or was cancelled by sim.", color=[255,0,0,255], level="error")
            elif msg_type == "PLAYBACK_READY": 
                global_gui_state["is_playback_mode_active"] = True
                global_gui_state["current_playback_frame_index"] = 0 
                global_gui_state["playback_is_playing_ui"] = False 
                update_ui_for_playback_mode_state(is_playback_active_ui=True, num_frames_from_meta=global_gui_state.get("active_recording_data_source",{}).get("num_frames",0))
                update_monitoring_overlay_values(message.get("initial_gui_data")) 
                # Push this initial frame data to GL
                initial_pb_gl_data = message.get("initial_gui_data")
                if initial_pb_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = initial_pb_gl_data
                    global_viz_data_cache["gl_render_data_available"].set()
                update_status_bar("Playback mode ready. Use playback controls.", level="info")
            elif msg_type == "PLAYBACK_FRAME_APPLIED":
                global_gui_state["current_playback_frame_index"] = message["frame_index"]
                update_monitoring_overlay_values(message["gui_data"])
                if dpg.does_item_exist("playback_slider"): 
                    if dpg.get_value("playback_slider") != message["frame_index"]:
                        dpg.set_value("playback_slider", message["frame_index"])
                # Push new frame data to GL
                pb_frame_gl_data = message.get("gui_data")
                if pb_frame_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = pb_frame_gl_data
                    global_viz_data_cache["gl_render_data_available"].set()
            elif msg_type == "PLAYBACK_EXITED_SIM_SIDE":
                global_gui_state["is_playback_mode_active"] = False
                global_gui_state["playback_is_playing_ui"] = False
                if global_gui_state.get("active_recording_data_source") and \
                   global_gui_state["active_recording_data_source"].get("h5_file_obj_for_playback"):
                    try:
                        # The H5 file handle is owned by sim_thread; UI thread shouldn't close it.
                        # Sim thread should close it when it processes EXIT_PLAYBACK_MODE.
                        # We just clear the reference in UI state.
                        print("Playback exited on sim side. UI clearing its reference to HDF5 data source.")
                    except Exception as e_close_h5_ui:
                        print(f"Error clearing HDF5 ref on playback exit (UI): {e_close_h5_ui}")
                global_gui_state["active_recording_data_source"] = None 
                
                _populate_ui_from_config_dict(message["new_config_dict"]) 
                update_monitoring_overlay_values(message["initial_gui_data"]) 
                # Push this initial live data to GL
                live_initial_gl_data = message.get("initial_gui_data")
                if live_initial_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = live_initial_gl_data
                    global_viz_data_cache["gl_render_data_available"].set()

                update_ui_for_playback_mode_state(is_playback_active_ui=False) 
                update_status_bar("Exited playback mode. Live simulation mode restored.", level="info")

            # --- Experiment System Messages ---
            elif msg_type == "EXPERIMENT_LOADED":
                exp_name = message.get("name", "Unknown")
                n_phases = message.get("num_phases", 0)
                n_channels = message.get("num_channels", 0)
                n_groups = message.get("num_groups", 0)
                dpg.set_value("experiment_info_text",
                              f"Loaded: {exp_name}\n  {n_phases} phases, {n_channels} channels, {n_groups} groups")
                dpg.configure_item("experiment_info_text", color=[100, 255, 100])
                update_status_bar(f"Experiment loaded: {exp_name}", color=[100, 200, 255, 255])
            elif msg_type == "EXPERIMENT_STARTED":
                update_status_bar("Experiment started", color=[100, 255, 100, 255])
            elif msg_type == "EXPERIMENT_STOPPED":
                update_status_bar("Experiment stopped", color=[255, 200, 100, 255])
            elif msg_type == "EXPERIMENT_LOG_SAVED":
                update_status_bar(f"Experiment log saved: {message.get('filepath', '')}", color=[100, 255, 100, 255])
            elif msg_type == "EXPERIMENT_ERROR":
                update_status_bar(f"Experiment error: {message.get('reason', 'Unknown')}", color=[255, 100, 100, 255])

            elif msg_type in ["CONFIG_APPLIED_ERROR", "CHECKPOINT_LOAD_FAILED", "RECORDING_LOAD_FAILED",
                              "RECORDING_START_FAILED", "PLAYBACK_SETUP_FAILED", "PLAYBACK_ERROR", "SIM_FATAL_ERROR",
                              "CHECKPOINT_SAVE_FAILED"]:
                update_status_bar(f"Error: {message.get('reason', message.get('error', 'Unknown error'))}", color=[255,0,0,255], level="error")
                if msg_type == "SIM_FATAL_ERROR": shutdown_flag.set() 

            sim_to_ui_queue.task_done()
    except queue.Empty:
        pass 

    # --- 1.5. Update Log Display ---
    if hasattr(handle_log_search_change, "log_capture"):
        if dpg.is_dearpygui_running() and dpg.does_item_exist("system_logs_display"):
            log_capture = handle_log_search_change.log_capture
            logs = log_capture.get_logs()
            if logs:
                # Show ALL logs
                display_text = "\n".join(logs)
                current_value = dpg.get_value("system_logs_display")
                if current_value != display_text:
                    dpg.set_value("system_logs_display", display_text)
                    # Update input_text height based on text size for proper scrolling
                    FRAME_PADDING = 3
                    text_size = dpg.get_text_size(display_text)
                    if text_size is not None:
                        dpg.set_item_height("system_logs_display", text_size[1] + (2 * FRAME_PADDING))

    # --- 2. Handle UI-Driven Playback Stepping (if active and playing) ---
    if global_gui_state.get("is_playback_mode_active", False) and global_gui_state.get("playback_is_playing_ui", False):
        current_time_ui = time.perf_counter()
        time_since_last_step_ui = current_time_ui - global_gui_state.get("last_playback_autostep_time_ui", 0.0)
        playback_interval_ui = 1.0 / global_gui_state.get("playback_fps_ui", 30.0)

        if time_since_last_step_ui >= playback_interval_ui:
            active_rec_meta = global_gui_state.get("active_recording_data_source")
            if active_rec_meta and active_rec_meta.get("h5_file_obj_for_playback"): # Ensure handle is there
                num_frames = active_rec_meta.get("num_frames", 0)
                current_frame_idx_ui = global_gui_state.get("current_playback_frame_index", 0)
                next_frame_idx = current_frame_idx_ui + 1
                
                if num_frames > 0 and next_frame_idx < num_frames:
                    ui_to_sim_queue.put({
                        "type": "SET_PLAYBACK_FRAME", 
                        "frame_index": next_frame_idx,
                        "h5_file_handle_for_sim_thread": active_rec_meta["h5_file_obj_for_playback"] 
                    })
                else: 
                    global_gui_state["playback_is_playing_ui"] = False 
                    update_ui_for_playback_playing_state(is_playing=False)
            global_gui_state["last_playback_autostep_time_ui"] = current_time_ui

    # --- 3. DPG Rendering ---
    if dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    # --- 4. OpenGL Rendering with 60 FPS Frame Rate Limiting ---
    if OPENGL_AVAILABLE and glut.glutGetWindow() != 0:
        global gl_last_render_time, gl_target_frame_time
        
        current_time = time.perf_counter()
        time_since_last_frame = current_time - gl_last_render_time
        
        # Only render if enough time has passed for 60 FPS (16.67ms)
        # OR if filters changed (force immediate update)
        should_render = (time_since_last_frame >= gl_target_frame_time) or global_gui_state.get("filters_changed", False)
        
        if should_render:
            # Update GL data if new data available or filters changed
            if global_viz_data_cache["gl_render_data_available"].is_set() or global_gui_state.get("filters_changed", False):
                update_gl_data() 
                global_viz_data_cache["gl_render_data_available"].clear()
            
            try: 
                current_win_gl = glut.glutGetWindow()
                if glut_window_id is not None and current_win_gl != 0 and current_win_gl != glut_window_id : 
                    glut.glutSetWindow(glut_window_id) # Ensure correct GL context
                if current_win_gl != 0: # Only post redisplay if window exists
                    glut.glutPostRedisplay()
                    gl_last_render_time = current_time  # Update last render time
            except Exception: pass

    if not OPENGL_AVAILABLE and dpg.is_dearpygui_running(): # DPG only mode
        time.sleep(0.005) # Prevent DPG-only loop from busy-waiting excessively


# --- Headless auto-tuning runner -------------------------------------------------

def _evaluate_candidate_config(sim_bridge, core_cfg, viz_cfg, total_time_ms):
    """Initializes sim_bridge with the given config, runs a short headless simulation,
    and returns basic activity/connectivity metrics for auto-tuning.
    """
    # Reset any previous state and GPU memory
    sim_bridge.clear_simulation_state_and_gpu_memory()
    sim_bridge.core_config = core_cfg
    sim_bridge.viz_config = viz_cfg
    sim_bridge.runtime_state = RuntimeState()

    dt = core_cfg.dt_ms if core_cfg.dt_ms > 0 else 0.0
    sim_bridge.runtime_state.max_delay_steps = int(core_cfg.max_synaptic_delay_ms / dt) if dt > 0 else 200

    sim_bridge._initialize_simulation_data(called_from_playback_init=False)
    if not sim_bridge.is_initialized or core_cfg.num_neurons <= 0 or dt <= 0:
        sim_bridge._log_console("Auto-tune: initialization failed or invalid config.", "warning")
        return None

    n = core_cfg.num_neurons
    num_steps = int(total_time_ms / dt)
    if num_steps <= 0:
        return None

    ever_spiked = cp.zeros(n, dtype=bool)
    total_spikes = 0

    for _ in range(num_steps):
        sim_bridge._run_one_simulation_step()
        sim_bridge.runtime_state.current_time_step += 1
        # Compute time from step count to avoid floating point drift
        sim_bridge.runtime_state.current_time_ms = sim_bridge.runtime_state.current_time_step * dt

        fired = sim_bridge.cp_firing_states
        if fired is None:
            break
        ever_spiked = cp.logical_or(ever_spiked, fired)
        step_spikes = int(cp.sum(fired).get())
        total_spikes += step_spikes

    num_synapses = int(sim_bridge.cp_connections.nnz) if sim_bridge.cp_connections is not None else 0
    num_spiking_neurons = int(cp.sum(ever_spiked).get())
    avg_spikes_per_step = total_spikes / float(max(1, num_steps))
    total_time_s = (dt * num_steps) / 1000.0
    avg_spikes_per_neuron_hz = 0.0
    if n > 0 and total_time_s > 0.0:
        avg_spikes_per_neuron_hz = total_spikes / (n * total_time_s)
    spiking_fraction = num_spiking_neurons / float(n) if n > 0 else 0.0

    return {
        "num_neurons": n,
        "num_synapses": num_synapses,
        "num_steps": num_steps,
        "total_spikes": total_spikes,
        "avg_spikes_per_step": avg_spikes_per_step,
        "avg_spikes_per_neuron_hz": avg_spikes_per_neuron_hz,
        "spiking_neuron_fraction": spiking_fraction,
    }


def _score_auto_tune_metrics(metrics):
    """Scores a candidate based on firing activity and neuron participation."""
    n = metrics.get("num_neurons", 0)
    if n <= 0:
        return -1.0

    total_spikes = metrics.get("total_spikes", 0)
    num_synapses = metrics.get("num_synapses", 0)
    if total_spikes <= 0 or num_synapses <= 0:
        return -1.0

    avg_spikes_per_step = metrics.get("avg_spikes_per_step", 0.0)
    spiking_fraction = metrics.get("spiking_neuron_fraction", 0.0)

    frac_spikes_per_step = avg_spikes_per_step / float(n) if n > 0 else 0.0

    # Desired range (fraction of neurons spiking per step on average)
    target_frac = 0.10
    min_frac = 0.02
    max_frac = 0.30

    in_range = min_frac <= frac_spikes_per_step <= max_frac

    # Component 1: closeness to target firing fraction
    diff = abs(frac_spikes_per_step - target_frac)
    score_firing = max(0.0, 1.0 - diff / max(target_frac, 1e-6))

    # Component 2: fraction of neurons that ever spiked
    target_spiking_fraction = 0.3
    score_participation = min(1.0, spiking_fraction / max(target_spiking_fraction, 1e-6))

    base_score = 0.6 * score_firing + 0.4 * score_participation

    # Penalty if firing is outside desired window
    if not in_range:
        base_score -= 0.5

    return float(base_score)


def run_auto_tuning(quick=False):
    """Headless auto-tuning entry point.

    When quick=True, only a small subset of combinations is tuned for faster testing.
    """
    print(f"Starting auto-tuning workflow (quick={quick})...")
    sim_bridge = SimulationBridge(ui_queue=sim_to_ui_queue)

    # Profiles to sweep
    if "NEURAL_STRUCTURE_PROFILES" in globals():
        profile_names = sorted(NEURAL_STRUCTURE_PROFILES.keys())
    else:
        profile_names = ["GENERIC_UNSTRUCTURED"]

    if quick:
        profile_names = profile_names[:2]

    # Models to tune: HH + AdEx (Izhikevich already behaves well in most cases)
    models_to_tune = [NeuronModel.HODGKIN_HUXLEY, NeuronModel.ADEX]

    # Determine HH presets to tune per profile, respecting realism constraints.
    # For structured profiles, this will typically be a single region-appropriate
    # preset; generic/unstructured profiles fall back to all HH types.
    profile_to_hh_types = {}
    for profile_name in profile_names:
        allowed_names = get_compatible_hh_type_names_for_profile(profile_name)
        hh_list = []
        for name in allowed_names:
            if name in NeuronType.__members__ and name.startswith("HH_"):
                hh_list.append(NeuronType[name])
        if not hh_list:
            hh_list = [nt for nt in NeuronType if nt.name.startswith("HH_")]
        if quick and len(hh_list) > 3:
            hh_list = hh_list[:3]
        profile_to_hh_types[profile_name] = hh_list

    tuned_combos = {}
    num_hh_combos = sum(len(ts) for ts in profile_to_hh_types.values())
    num_adex_combos = len(profile_names)
    total_combos = num_hh_combos + num_adex_combos
    combo_index = 0

    hh_scales = [0.5, 1.0, 2.0, 4.0]
    adex_scales = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Hodgkin-Huxley tuning: per (profile, HH preset)
    for profile_name in profile_names:
        hh_types_for_profile = profile_to_hh_types.get(profile_name, [])
        for hh_type in hh_types_for_profile:
            combo_index += 1
            key = f"{NeuronModel.HODGKIN_HUXLEY.name}|{profile_name}|{hh_type.name}"
            print(f"[{combo_index}/{total_combos}] Tuning {key} ...")

            best_score = -1e9
            best_scale = None
            best_metrics = None

            for scale in hh_scales:
                core_cfg = CoreSimConfig()
                core_cfg.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
                core_cfg.neural_profile_name = profile_name
                core_cfg.default_neuron_type_hh = hh_type.name
                core_cfg.num_neurons = 400 if not quick else 250
                core_cfg.dt_ms = 1.0
                core_cfg.total_simulation_time_ms = 600.0 if not quick else 300.0
                core_cfg.hh_external_drive_scale = scale

                viz_cfg = VisualizationConfig()
                metrics = _evaluate_candidate_config(sim_bridge, core_cfg, viz_cfg, core_cfg.total_simulation_time_ms)
                if metrics is None:
                    continue

                score = _score_auto_tune_metrics(metrics)
                metrics["score"] = score
                metrics["selected_scale"] = scale

                if score > best_score:
                    best_score = score
                    best_scale = scale
                    best_metrics = metrics

            if best_scale is not None:
                tuned_combos[key] = {
                    "neuron_model_type": NeuronModel.HODGKIN_HUXLEY.name,
                    "neural_profile_name": profile_name,
                    "default_neuron_type_hh": hh_type.name,
                    "core_overrides": {"hh_external_drive_scale": float(best_scale)},
                    "metrics": best_metrics,
                }
            else:
                tuned_combos[key] = {
                    "neuron_model_type": NeuronModel.HODGKIN_HUXLEY.name,
                    "neural_profile_name": profile_name,
                    "default_neuron_type_hh": hh_type.name,
                    "core_overrides": {},
                    "metrics": {"note": "no viable candidate found"},
                }

    # AdEx tuning: per (profile, model) only
    for profile_name in profile_names:
        combo_index += 1
        key = f"{NeuronModel.ADEX.name}|{profile_name}|NONE"
        print(f"[{combo_index}/{total_combos}] Tuning {key} ...")

        best_score = -1e9
        best_scale = None
        best_metrics = None

        for scale in adex_scales:
            core_cfg = CoreSimConfig()
            core_cfg.neuron_model_type = NeuronModel.ADEX.name
            core_cfg.neural_profile_name = profile_name
            core_cfg.num_neurons = 400 if not quick else 250
            core_cfg.dt_ms = 1.0
            core_cfg.total_simulation_time_ms = 800.0 if not quick else 400.0
            core_cfg.adex_external_drive_scale = scale

            viz_cfg = VisualizationConfig()
            metrics = _evaluate_candidate_config(sim_bridge, core_cfg, viz_cfg, core_cfg.total_simulation_time_ms)
            if metrics is None:
                continue

            score = _score_auto_tune_metrics(metrics)
            metrics["score"] = score
            metrics["selected_scale"] = scale

            if score > best_score:
                best_score = score
                best_scale = scale
                best_metrics = metrics

        if best_scale is not None:
            tuned_combos[key] = {
                "neuron_model_type": NeuronModel.ADEX.name,
                "neural_profile_name": profile_name,
                "default_neuron_type_hh": None,
                "core_overrides": {"adex_external_drive_scale": float(best_scale)},
                "metrics": best_metrics,
            }
        else:
            tuned_combos[key] = {
                "neuron_model_type": NeuronModel.ADEX.name,
                "neural_profile_name": profile_name,
                "default_neuron_type_hh": None,
                "core_overrides": {},
                "metrics": {"note": "no viable candidate found"},
            }

    # Persist results
    os.makedirs(os.path.dirname(AUTO_TUNED_OVERRIDES_PATH), exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "tuned_combinations": tuned_combos,
    }
    with open(AUTO_TUNED_OVERRIDES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Auto-tuning complete. Wrote {len(tuned_combos)} combinations to {AUTO_TUNED_OVERRIDES_PATH}.")
    return 0


def load_viz_benchmark_hardware_note():
    """Loads the hardware performance note from viz benchmark results if available.
    
    Returns:
        str: Hardware note if found, empty string otherwise
    """
    viz_results_path = os.path.join("benchmarks", "viz_performance_results.json")
    
    if not os.path.exists(viz_results_path):
        return ""
    
    try:
        with open(viz_results_path, 'r') as f:
            results = json.load(f)
        
        hardware_note = results.get("hardware_performance_note", "")
        if hardware_note:
            print(f"Loaded hardware performance note from {viz_results_path}")
            return hardware_note
    except Exception as e:
        print(f"Warning: Could not load viz benchmark results from {viz_results_path}: {e}")
    
    return ""


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    print("\nCtrl+C detected. Shutting down gracefully...")
    shutdown_flag.set()


def main():
    global global_simulation_bridge, simulation_thread, shutdown_flag, glut_window_id
    # global_gui_state, global_viz_data_cache, opengl_viz_config are already defined globally.
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Use the global log capture instance that was started at module load
    global _global_log_capture
    # Store reference in handler function for access
    handle_log_search_change.log_capture = _global_log_capture

    dpg.create_context()
    dpg.configure_app(docking=False)

    global_simulation_bridge = SimulationBridge(ui_queue=sim_to_ui_queue) # Initialize the simulation core

    # Initialize shared state for extracted viz/ and ui/ modules
    _init_shared_state_for_modules()

    # Attempt to load default profile (JSON) - This is a UI-side operation before sim_thread starts.
    default_profile_filename = "default_profile.json"
    default_profile_path = os.path.join(global_simulation_bridge.PROFILE_DIR, default_profile_filename)
    loaded_default_sim_config_dict = None
    loaded_default_gui_config_dict = None

    if os.path.exists(default_profile_path):
        try:
            with open(default_profile_path, 'r', encoding='utf-8') as f: profile_content = json.load(f)
            loaded_default_sim_config_dict = profile_content.get("simulation_configuration")
            loaded_default_gui_config_dict = profile_content.get("gui_configuration")
            if loaded_default_sim_config_dict:
                print(f"Default profile '{default_profile_filename}' found. Will apply after UI setup.")
                global_gui_state["current_profile_name"] = default_profile_filename
            else: print(f"Default profile '{default_profile_filename}' is missing simulation_configuration.")
        except Exception as e:
            print(f"Error loading default profile '{default_profile_path}': {e}")
            loaded_default_sim_config_dict = None; loaded_default_gui_config_dict = None
    
    if loaded_default_sim_config_dict is None: # Fallback if no valid default profile
        print("Using basic default internal configuration for initial UI population.")
        loaded_default_sim_config_dict = SimulationConfiguration().to_dict() # Use fresh defaults
        global_gui_state["current_profile_name"] = "unsaved_internal_defaults.json"
    
    # Load hardware performance notes from benchmarks
    # Priority: benchmark_results.json (comprehensive) > viz_performance_results.json (viz-only)
    _load_benchmark_limits()  # Parse benchmark_results.json into HARDWARE_LIMITS
    hardware_note = get_hardware_note()  # From benchmark_results.json
    if not hardware_note:
        hardware_note = load_viz_benchmark_hardware_note()  # Fallback to viz benchmark
    if hardware_note and loaded_default_sim_config_dict:
        loaded_default_sim_config_dict["hardware_performance_note"] = hardware_note


    # DPG Viewport setup
    dpg_viewport_width = 700 
    dpg_viewport_height = int(SCREEN_HEIGHT * 0.90) if SCREEN_HEIGHT > 300 else 760 
    dpg.create_viewport(title="Neuron Simulator Controls (DPG)",
                        width=dpg_viewport_width, height=dpg_viewport_height,
                        x_pos=0, y_pos=20) # Position on the left

    create_gui_layout() # Create all DPG widgets
    dpg.set_primary_window("controls_monitor_window", True)

    # Populate UI with the (default or profile-loaded) simulation configuration
    _populate_ui_from_config_dict(loaded_default_sim_config_dict)
    if loaded_default_gui_config_dict: # Apply GUI settings from profile if they exist
        apply_gui_configuration_core(loaded_default_gui_config_dict)
    
    # Ensure hardware note is displayed (direct widget update after UI population)
    if hardware_note and dpg.does_item_exist("cfg_hardware_performance_note"):
        dpg.set_value("cfg_hardware_performance_note", hardware_note)
    
    if dpg.does_item_exist("profile_name_input"): # Show current profile name
        dpg.set_value("profile_name_input", global_gui_state["current_profile_name"].replace(".json", ""))

    # Populate neuron type filter based on the initial model type in UI
    if dpg.does_item_exist("filter_neuron_type_combo") and dpg.does_item_exist("cfg_neuron_model_type"):
        initial_model_name = dpg.get_value("cfg_neuron_model_type")
        _toggle_model_specific_params_visibility(None, initial_model_name) # Updates filter items

    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Initial UI state updates
    update_ui_for_simulation_run_state(is_running=False, is_paused=False)
    update_ui_for_recording_state(is_recording_active_ui=False)
    update_ui_for_playback_mode_state(is_playback_active_ui=False) # Hides playback controls initially
    update_monitoring_overlay_values(None) # Clear monitor

    global_gui_state["reset_sim_needed_from_ui_change"] = True # Force "Apply Changes" for initial config
    update_status_bar("Application started. Apply initial config or load a profile/state.", level="info")


    # --- Start the Simulation Worker Thread ---
    # Sim_thread will initialize sim_bridge with the config currently reflected in the UI.
    # So, send an "APPLY_CONFIG_AND_RESET" with current UI config as the first command.
    initial_config_from_ui = _update_sim_config_from_ui(update_model_specific=True)
    if initial_config_from_ui:
        ui_to_sim_queue.put({
            "type": "APPLY_CONFIG_AND_RESET",
            "config_dict": initial_config_from_ui
        })
    else: # Should not happen if UI is built correctly
        print("CRITICAL: Failed to get initial config from UI for sim_thread.")
        # Sim_thread will start with default SimulationConfiguration in sim_bridge.

    simulation_thread = threading.Thread(target=simulation_worker_loop, 
                                         args=(global_simulation_bridge, shutdown_flag, ui_to_sim_queue, sim_to_ui_queue),
                                         daemon=True) # Daemon so it exits if main thread crashes
    simulation_thread.start()


    # --- Main Loop (DPG + OpenGL if available) ---
    if OPENGL_AVAILABLE:
        glut.glutInit(sys.argv if hasattr(sys, "argv") and sys.argv else ["sim3d_threaded.py"]) # Init GLUT
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH) # Display modes

        gl_win_width = SCREEN_WIDTH - dpg_viewport_width - 30 if SCREEN_WIDTH > dpg_viewport_width + 30 else 600
        gl_win_height = dpg_viewport_height; gl_win_width = max(400, gl_win_width); gl_win_height = max(300, gl_win_height)
        gl_win_x_pos = dpg_viewport_width + 10 

        glut.glutInitWindowPosition(gl_win_x_pos, 20); glut.glutInitWindowSize(gl_win_width, gl_win_height)
        try: glut_window_id = glut.glutCreateWindow(b"3D Network Visualization (OpenGL - Threaded)")
        except TypeError: glut_window_id = glut.glutCreateWindow("3D Network Visualization (OpenGL - Threaded)")
        _viz_renderer.set_glut_window_id(glut_window_id)

        opengl_viz_config['WINDOW_WIDTH'] = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        opengl_viz_config['WINDOW_HEIGHT'] = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)

        init_gl(); # Initialize OpenGL state (VBOs, etc.)
        glut.glutDisplayFunc(render_scene_gl); 
        glut.glutReshapeFunc(reshape_gl_window) 
        glut.glutKeyboardFunc(keyboard_func_gl); 
        glut.glutMouseFunc(mouse_button_func_gl) 
        glut.glutMotionFunc(mouse_motion_func_gl); 
        glut.glutIdleFunc(main_dpg_loop_and_gl_idle) # Main loop function for GLUT

        # Initial GL data population (empty or from first sim_to_ui message)
        # update_gl_data() will be called by main_dpg_loop_and_gl_idle when data is available.
        print("Starting GLUT main loop (with DPG integration)...")
        try: glut.glutMainLoop()
        except Exception as e_glut: print(f"Exception during GLUT main loop: {e_glut}")
        finally:
            print("Exited GLUT main loop."); shutdown_flag.set()
            if dpg.is_dearpygui_running(): dpg.stop_dearpygui()
    else: # No OpenGL, run DPG only
        print("OpenGL not available. Running DPG controls only.")
        while dpg.is_dearpygui_running() and not shutdown_flag.is_set():
            main_dpg_loop_and_gl_idle(); # Call the DPG part of the loop
            # time.sleep(0.005) # Replaced by queue checks and DPG's own timing
        if dpg.is_dearpygui_running(): dpg.stop_dearpygui()

    # --- Cleanup ---
    shutdown_flag.set() # Ensure flag is set for sim_thread if not already
    if simulation_thread and simulation_thread.is_alive():
        print("Waiting for simulation worker thread to finish...")
        simulation_thread.join(timeout=5.0) # Wait for sim_thread
        if simulation_thread.is_alive():
            print("Warning: Simulation thread did not terminate gracefully.")

    # Clean up OpenGL VBOs to prevent GPU memory leaks
    if OPENGL_AVAILABLE:
        try:
            vbo_list = [gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo]
            valid_vbos = [v for v in vbo_list if v is not None and v > 0]
            if valid_vbos:
                from OpenGL.GL import glDeleteBuffers
                glDeleteBuffers(len(valid_vbos), valid_vbos)
                print(f"Cleaned up {len(valid_vbos)} OpenGL VBOs.")
        except Exception as e:
            print(f"Note: OpenGL VBO cleanup skipped ({e})")

    if dpg.is_dearpygui_running(): dpg.destroy_context()
    print("Neuron simulator application shutdown complete.")

if __name__ == '__main__':
    # If launched with --auto-tune, run the headless tuning workflow instead of the GUI.
    if '--auto-tune' in sys.argv:
        quick = '--quick' in sys.argv
        exit_code = run_auto_tuning(quick=quick)
        sys.exit(exit_code)
    else:
        main()
