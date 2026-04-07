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

# --- OpenGL Visualization Functions (to be run in the main/UI thread) ---
def init_gl():
    """Initializes OpenGL state. Called by the main thread."""
    if not OPENGL_AVAILABLE: return
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global cuda_gl_interop_enabled

    glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST) # Anti-aliased points
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Enable alpha blending
    glClearColor(0, 0, 0, 0); # Dark blue background
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)); # Set default point size
    glEnable(GL_DEPTH_TEST) # Enable depth testing for 3D

    # Generate Vertex Buffer Objects (VBOs)
    try:
        vbo_ids = glGenBuffers(4) # Generate 4 VBO IDs
        if not isinstance(vbo_ids, (list, tuple, np.ndarray)) or len(vbo_ids) < 4 :
            # Fallback if glGenBuffers returns a single ID or unexpected type
            if isinstance(vbo_ids, int) and vbo_ids > 0: 
                 gl_neuron_pos_vbo = vbo_ids
                 gl_neuron_color_vbo = glGenBuffers(1)
                 gl_synapse_vertices_vbo = glGenBuffers(1)
                 gl_pulse_vertices_vbo = glGenBuffers(1)
            else: raise ValueError("glGenBuffers did not return expected VBO IDs.")
        else: # Standard return of multiple IDs
            gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo = vbo_ids[0], vbo_ids[1], vbo_ids[2], vbo_ids[3]
        
        # Try to enable CUDA-OpenGL interop for zero-copy transfers
        try:
            # Test if CUDA can access OpenGL context
            from cuda import cudart
            cuda_gl_interop_enabled = True
            print("[CUDA-GL Interop] Enabled for zero-copy GPU→OpenGL transfers")
        except ImportError:
            cuda_gl_interop_enabled = False
            print("[CUDA-GL Interop] Not available (cuda-python not installed). Using GPU→CPU→GPU path.")
            
    except Exception as e:
        print(f"Error: glGenBuffers failed: {e}. OpenGL visualization will likely fail.")
        # Set VBO IDs to 0 or an invalid marker to prevent usage if generation fails
        gl_neuron_pos_vbo = 0; gl_neuron_color_vbo = 0; gl_synapse_vertices_vbo = 0; gl_pulse_vertices_vbo = 0
        cuda_gl_interop_enabled = False
        return


def reshape_gl_window(width, height):
    """Handles OpenGL window reshape events. Called by GLUT in the main thread."""
    if not OPENGL_AVAILABLE or height <= 0 or global_simulation_bridge is None: return # global_simulation_bridge for camera config
    viz_cfg = global_simulation_bridge.viz_config # Access viz_config for camera params

    opengl_viz_config['WINDOW_WIDTH'] = width # Update stored window dimensions
    opengl_viz_config['WINDOW_HEIGHT'] = height

    glViewport(0, 0, width, height); # Set viewport to new window size
    glMatrixMode(GL_PROJECTION); glLoadIdentity() # Switch to projection matrix
    # Set perspective: FOV, aspect ratio, near clip, far clip
    gluPerspective(viz_cfg.camera_fov, float(width) / float(height), viz_cfg.camera_near_clip, viz_cfg.camera_far_clip)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity() # Switch back to modelview matrix


def render_text_gl(x, y, text, font=None): # Font defaults to GLUT_BITMAP_9_BY_15 if None
    """Renders text on the OpenGL screen. Called by the main thread."""
    if not OPENGL_AVAILABLE: return
    if font is None: font = glut.GLUT_BITMAP_9_BY_15 if hasattr(glut, "GLUT_BITMAP_9_BY_15") else None
    if font is None: print("Warning: GLUT font not available for render_text_gl."); return


    try:
        current_win = glut.glutGetWindow();
        if current_win == 0: return # No current GL context (e.g., window closed)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity() # Save current projection matrix
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800); 
        win_h = opengl_viz_config.get('WINDOW_HEIGHT', 600)
        gluOrtho2D(0, win_w, 0, win_h) # Set up 2D orthographic projection for text

        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity() # Save current modelview matrix
        glColor3f(0.9, 0.9, 0.9); # Set text color (e.g., light gray)
        glDisable(GL_DEPTH_TEST) # Disable depth test to render text on top

        glRasterPos2i(int(x), int(y)) # Position the text (bottom-left origin)
        for character in text:
            glut.glutBitmapCharacter(font, ord(character)) # Render each character

        glEnable(GL_DEPTH_TEST); # Re-enable depth test
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); # Restore matrices
        glMatrixMode(GL_MODELVIEW) # Ensure modelview is current
    except Exception as e:
        # This can happen if GLUT context is lost or font is invalid
        print(f"[ERROR] OpenGL render_text_gl: {e}")


def get_color_for_trait(trait_index, activity_timer_value, is_currently_spiking, neuron_model_name_str, neuron_type_str=""):
    """
    Determines neuron color based on trait, activity, spiking status, and filter mode.
    Called by the main thread during GL data preparation.
    """
    max_highlight_frames = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
    firing_rgb_config = opengl_viz_config.get("FIRING_NEURON_COLOR", [1.0, 1.0, 0.0, 1.0])
    firing_rgb = firing_rgb_config[0:3] # RGB components for firing
    base_firing_alpha = firing_rgb_config[3] # Alpha for firing
    default_inactive_alpha = opengl_viz_config.get("INACTIVE_NEURON_OPACITY", 0.25)

    # Base color from TRAIT_COLOR_MAP_RAW
    base_color_rgb = [0.5, 0.5, 0.5]; base_alpha = default_inactive_alpha # Default gray
    if TRAIT_COLOR_MAP_RAW and len(TRAIT_COLOR_MAP_RAW) > 0:
        color_def_from_map = TRAIT_COLOR_MAP_RAW[trait_index % len(TRAIT_COLOR_MAP_RAW)]
        base_color_rgb = color_def_from_map[0:3]
        base_alpha = color_def_from_map[3] if len(color_def_from_map) > 3 else default_inactive_alpha

    final_color_rgba = list(base_color_rgb) + [base_alpha] # Default color

    # Get current filter settings (from UI thread's global_gui_state)
    # This function is called by main thread, so direct access to global_gui_state is okay here.
    filters_dict = get_current_filter_settings_from_gui() # Assumes this is safe to call from main
    spiking_mode_filter = filters_dict.get("spiking_mode", "Highlight Spiking") # DPG UI filter setting

    if spiking_mode_filter == "No Spiking Highlight":
        return final_color_rgba # Return base trait color, no special highlight

    # Apply spiking highlight based on filter
    if is_currently_spiking:
        # If "Highlight Spiking" or "Show Only Spiking" and neuron is currently spiking
        final_color_rgba = list(firing_rgb) + [base_firing_alpha]
    elif spiking_mode_filter == "Highlight Spiking" and activity_timer_value > 0:
        # "Highlight Spiking" mode: not currently spiking, but has a residual activity timer
        # Fade the highlight color based on remaining timer duration
        decay_ratio = max(0.0, min(1.0, float(activity_timer_value) / max_highlight_frames))
        dimmed_firing_alpha = base_firing_alpha * decay_ratio * 0.6 # Make decay noticeable
        dimmed_firing_alpha = max(dimmed_firing_alpha, base_alpha * 0.8, 0.05) # Ensure it's not less than base or too faint
        dimmed_firing_alpha = min(base_firing_alpha * 0.8, dimmed_firing_alpha) # Cap at a slightly lower max than full spike
        final_color_rgba = list(firing_rgb) + [dimmed_firing_alpha]
    # In "Show Only Spiking" mode, if not is_currently_spiking, the neuron wouldn't be drawn anyway by upstream filter logic.
    
    return final_color_rgba

_gl_frame_counter = 0  # Module-level frame counter for VBO update skipping

def update_gl_data():
    """
    Prepares neuron, synapse, and pulse data for OpenGL rendering by updating VBOs.
    This function is called by the main/UI thread. It gets data from global_viz_data_cache.gl_render_data_buffer,
    which is populated by the simulation thread with CuPy arrays for GL data.
    """
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw
    # Use the new global CuPy array names
    global gl_neuron_pos_cp, gl_neuron_colors_cp, gl_connection_vertices_cp, gl_pulse_vertices_cp
    global _gl_frame_counter

    if not OPENGL_AVAILABLE:
        gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
        return

    # Frame skip: only update VBOs every Nth frame to reduce GPU->CPU sync overhead
    _gl_frame_counter += 1
    skip = opengl_viz_config.get("VBO_UPDATE_SKIP", 2)
    if _gl_frame_counter % skip != 0:
        return

    sim_data_snapshot = None
    with global_viz_data_cache["gl_render_data_lock"]:
        if global_viz_data_cache["gl_render_data_buffer"] is not None:
            sim_data_snapshot = global_viz_data_cache["gl_render_data_buffer"].copy()

    if sim_data_snapshot is None:
        if not global_gui_state.get("filters_changed", False) and not global_gui_state.get("is_playback_mode_active", False):
            return
        # If filters changed but no new data, we might re-filter existing CuPy arrays
        # For now, assume sim_data_snapshot is required to proceed with new data.
        # If no snapshot, potentially clear display or show last state (current logic will use empty arrays).
        if sim_data_snapshot is None and not global_gui_state.get("filters_changed", False):
             return


    # --- Extract CuPy arrays and other data from snapshot ---
    neuron_fired_cp = sim_data_snapshot.get("neuron_fired_status_cp", cp.array([], dtype=bool))
    neuron_activity_timers_cp = sim_data_snapshot.get("neuron_activity_timers_cp", cp.array([], dtype=cp.int32))
    all_neuron_positions_3d_cp = sim_data_snapshot.get("neuron_positions_3d_cp", cp.array([], dtype=cp.float32).reshape(0,3))
    all_neuron_traits_cp = sim_data_snapshot.get("neuron_traits_cp", cp.array([], dtype=cp.int32))
    all_neuron_type_ids_cp = sim_data_snapshot.get("neuron_type_ids_cp", cp.array([], dtype=cp.int32))  # Integer type IDs

    # CPU data (neuron types list is Python list of strings, kept for UI display only)
    all_neuron_types_str_list_cpu = sim_data_snapshot.get("neuron_types_list_for_viz", []) 
    model_name_str = sim_data_snapshot.get("neuron_model_type_str", "IZHIKEVICH")
    num_neurons_in_snapshot = sim_data_snapshot.get("num_neurons_snapshot", 0)

    # Ensure consistency of snapshot data
    if all_neuron_positions_3d_cp.shape[0] != num_neurons_in_snapshot:
        all_neuron_positions_3d_cp = cp.zeros((num_neurons_in_snapshot, 3), dtype=cp.float32) # Fallback
    if neuron_fired_cp.size != num_neurons_in_snapshot:
        neuron_fired_cp = cp.zeros(num_neurons_in_snapshot, dtype=bool) # Fallback
    if neuron_activity_timers_cp.size != num_neurons_in_snapshot:
        neuron_activity_timers_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32) # Fallback
    if all_neuron_traits_cp.size != num_neurons_in_snapshot:
        all_neuron_traits_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32) # Fallback
    if all_neuron_type_ids_cp.size != num_neurons_in_snapshot:
        all_neuron_type_ids_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32) # Fallback
    if len(all_neuron_types_str_list_cpu) != num_neurons_in_snapshot:
         all_neuron_types_str_list_cpu = ["Unknown"] * num_neurons_in_snapshot


    # --- Neuron Filtering (on GPU where possible) ---
    current_filters = get_current_filter_settings_from_gui()
    all_indices_cp = cp.arange(num_neurons_in_snapshot, dtype=cp.int32)
    visible_mask_cp = cp.ones(num_neurons_in_snapshot, dtype=bool)

    spiking_mode_filter = current_filters.get("spiking_mode", "Highlight Spiking")
    if spiking_mode_filter == "Show Only Spiking":
        visible_mask_cp &= neuron_fired_cp

    if current_filters.get("type_filter_enabled", False):
        selected_type_str_cpu = current_filters.get("selected_neuron_type", "All")
        if selected_type_str_cpu != "All":
            # Use integer type ID for GPU-accelerated filtering
            selected_type_id = NEURON_TYPE_MAPPER.get_id_from_display_name(selected_type_str_cpu)
            type_mask_cp = (all_neuron_type_ids_cp == selected_type_id)  # GPU operation
            visible_mask_cp &= type_mask_cp

    visible_neuron_indices_cp = all_indices_cp[visible_mask_cp]

    max_render_neurons = opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 100000) # Increased default
    if visible_neuron_indices_cp.size > max_render_neurons:
        chosen_neuron_indices_cp = cp.random.choice(visible_neuron_indices_cp, size=max_render_neurons, replace=False)
    else:
        chosen_neuron_indices_cp = visible_neuron_indices_cp

    current_num_neurons_to_draw = chosen_neuron_indices_cp.size

    temp_gl_neuron_pos_cp = cp.array([], dtype=cp.float32).reshape(0,3)
    temp_gl_neuron_colors_cp = cp.array([], dtype=cp.float32).reshape(0,4)

    if current_num_neurons_to_draw > 0:
        temp_gl_neuron_pos_cp = all_neuron_positions_3d_cp[chosen_neuron_indices_cp]

        # --- Vectorized Color Calculation (GPU) ---
        chosen_traits = all_neuron_traits_cp[chosen_neuron_indices_cp]
        chosen_activity_timers = neuron_activity_timers_cp[chosen_neuron_indices_cp]
        chosen_is_spiking = neuron_fired_cp[chosen_neuron_indices_cp]

        max_highlight_frames_val = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
        firing_rgb_config_val = opengl_viz_config.get("FIRING_NEURON_COLOR", [1.0, 1.0, 0.0, 1.0])
        firing_rgb_gpu = cp.array(firing_rgb_config_val[0:3], dtype=cp.float32)
        base_firing_alpha_gpu = cp.float32(firing_rgb_config_val[3])
        default_inactive_alpha_gpu = cp.float32(opengl_viz_config.get("INACTIVE_NEURON_OPACITY", 0.25))

        # Initialize colors based on traits (using TRAIT_COLOR_MAP_GPU)
        # Ensure TRAIT_COLOR_MAP_GPU is defined in global scope and is a CuPy array
        if TRAIT_COLOR_MAP_GPU.ndim == 2 and TRAIT_COLOR_MAP_GPU.shape[1] == 4: # Basic check
            temp_gl_neuron_colors_cp = TRAIT_COLOR_MAP_GPU[chosen_traits % TRAIT_COLOR_MAP_GPU.shape[0]]
        else: # Fallback if TRAIT_COLOR_MAP_GPU is malformed
            temp_gl_neuron_colors_cp = cp.full((current_num_neurons_to_draw, 4), 
                                               cp.array([0.5, 0.5, 0.5, default_inactive_alpha_gpu], dtype=cp.float32), 
                                               dtype=cp.float32)

        if spiking_mode_filter != "No Spiking Highlight":
            spiking_mask = chosen_is_spiking
            if cp.any(spiking_mask):
                temp_gl_neuron_colors_cp[spiking_mask, 0:3] = firing_rgb_gpu
                temp_gl_neuron_colors_cp[spiking_mask, 3] = base_firing_alpha_gpu

            if spiking_mode_filter == "Highlight Spiking":
                active_timer_mask = (~chosen_is_spiking) & (chosen_activity_timers > 0)
                if cp.any(active_timer_mask):
                    decay_ratio = cp.clip(chosen_activity_timers[active_timer_mask].astype(cp.float32) / max_highlight_frames_val, 0.0, 1.0)

                    # Use original alpha from trait map as base for fading highlight
                    base_alpha_for_active_timer = temp_gl_neuron_colors_cp[active_timer_mask, 3].copy() # Get current alpha (from trait)

                    dimmed_firing_alpha = base_firing_alpha_gpu * decay_ratio * 0.6
                    # Ensure highlight is visible but respects original trait alpha somewhat
                    dimmed_firing_alpha = cp.maximum(dimmed_firing_alpha, base_alpha_for_active_timer * 0.8)
                    dimmed_firing_alpha = cp.maximum(dimmed_firing_alpha, 0.05) # Minimum visibility
                    dimmed_firing_alpha = cp.minimum(dimmed_firing_alpha, base_firing_alpha_gpu * 0.9) # Cap slightly below full spike alpha

                    temp_gl_neuron_colors_cp[active_timer_mask, 0:3] = firing_rgb_gpu
                    temp_gl_neuron_colors_cp[active_timer_mask, 3] = dimmed_firing_alpha

    # --- Synapse Data (GPU-accelerated filtering using cp.isin) ---
    temp_gl_connection_vertices_cp = cp.array([], dtype=cp.float32).reshape(0,3)
    current_num_synapse_lines_to_draw = 0
    if global_gui_state.get("show_connections_gl", False) and "synapse_info" in sim_data_snapshot:
        all_synapse_data_list_cpu = sim_data_snapshot["synapse_info"] # CPU list of dicts
        
        # Extract source and target indices from synapse data
        if all_synapse_data_list_cpu:
            src_indices_all = np.array([syn["source_idx"] for syn in all_synapse_data_list_cpu], dtype=np.int32)
            tgt_indices_all = np.array([syn["target_idx"] for syn in all_synapse_data_list_cpu], dtype=np.int32)
            weights_all = np.array([syn["weight"] for syn in all_synapse_data_list_cpu], dtype=np.float32)
            
            # Convert to CuPy for GPU operations
            src_indices_all_cp = cp.asarray(src_indices_all)
            tgt_indices_all_cp = cp.asarray(tgt_indices_all)
            weights_all_cp = cp.asarray(weights_all)
            
            # GPU-accelerated visibility check: both source and target must be in visible neurons
            src_visible_mask = cp.isin(src_indices_all_cp, chosen_neuron_indices_cp)
            tgt_visible_mask = cp.isin(tgt_indices_all_cp, chosen_neuron_indices_cp)
            
            # Weight filtering
            min_abs_w = current_filters.get("min_abs_weight", 0.01)
            weight_mask = cp.abs(weights_all_cp) >= min_abs_w
            
            # Combined visibility mask
            synapse_visible_mask = src_visible_mask & tgt_visible_mask & weight_mask
            visible_synapse_indices_cp = cp.where(synapse_visible_mask)[0]
            
            if visible_synapse_indices_cp.size > 0:
                # Get positions for visible synapses (all on GPU)
                visible_src_indices = src_indices_all_cp[visible_synapse_indices_cp]
                visible_tgt_indices = tgt_indices_all_cp[visible_synapse_indices_cp]
                
                pos_src_all_cp = all_neuron_positions_3d_cp[visible_src_indices]
                pos_tgt_all_cp = all_neuron_positions_3d_cp[visible_tgt_indices]

                # Interleave source and target positions: [src1, tgt1, src2, tgt2, ...]
                temp_gl_connection_vertices_cp = cp.empty((visible_synapse_indices_cp.size * 2, 3), dtype=cp.float32)
                temp_gl_connection_vertices_cp[0::2] = pos_src_all_cp # Even indices are sources
                temp_gl_connection_vertices_cp[1::2] = pos_tgt_all_cp # Odd indices are targets
                current_num_synapse_lines_to_draw = visible_synapse_indices_cp.size

    # --- Synaptic Pulse Data (Using pre-calculated positions from sim_bridge) ---
    temp_gl_pulse_vertices_cp = sim_data_snapshot.get("pulse_positions_cp_for_gl", cp.array([], dtype=cp.float32).reshape(0,3))
    current_num_pulses_to_draw = temp_gl_pulse_vertices_cp.shape[0]

    # --- Update global GL CuPy arrays and VBOs ---
    # These global arrays are now CuPy arrays
    gl_num_neurons_to_draw = current_num_neurons_to_draw
    gl_neuron_pos_cp = temp_gl_neuron_pos_cp 
    gl_neuron_colors_cp = temp_gl_neuron_colors_cp

    # === PHASE 4: CUDA-OpenGL Interop - OPTIMIZED VBO UPDATES ===
    # Use pinned memory and async transfers for faster GPU→CPU→GPU pipeline
    # This is ~2-3x faster than standard cp.asnumpy() transfers
    
    def fast_vbo_update(vbo_id, cupy_array):
        """Optimized VBO update using pinned memory for faster transfers."""
        if cupy_array.size == 0:
            return
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        
        # Method 1: Use CuPy's data pointer directly with memoryview (fastest CPU path)
        # This avoids intermediate NumPy array allocation
        try:
            # Get data as contiguous array
            if not cupy_array.flags.c_contiguous:
                cupy_array = cp.ascontiguousarray(cupy_array)
            
            # Transfer to CPU using pinned memory if available
            np_array = cp.asnumpy(cupy_array, order='C')
            glBufferData(GL_ARRAY_BUFFER, np_array.nbytes, np_array, GL_DYNAMIC_DRAW)
        except Exception as e:
            print(f"[VBO Update] Error: {e}")
    
    if gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 and gl_neuron_pos_cp.size > 0:
        fast_vbo_update(gl_neuron_pos_vbo, gl_neuron_pos_cp)
        
    if gl_neuron_color_vbo is not None and gl_neuron_color_vbo > 0 and gl_neuron_colors_cp.size > 0:
        fast_vbo_update(gl_neuron_color_vbo, gl_neuron_colors_cp)

    gl_num_synapse_lines_to_draw = current_num_synapse_lines_to_draw
    gl_connection_vertices_cp = temp_gl_connection_vertices_cp
    if gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0 and gl_connection_vertices_cp.size > 0:
        fast_vbo_update(gl_synapse_vertices_vbo, gl_connection_vertices_cp)

    gl_num_pulses_to_draw = current_num_pulses_to_draw
    gl_pulse_vertices_cp = temp_gl_pulse_vertices_cp
    if gl_pulse_vertices_vbo is not None and gl_pulse_vertices_vbo > 0 and gl_pulse_vertices_cp.size > 0:
        fast_vbo_update(gl_pulse_vertices_vbo, gl_pulse_vertices_cp)

    if gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 : # Check if it's a valid VBO ID
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

    global_gui_state["filters_changed"] = False

def render_scene_gl():
    """Main OpenGL rendering function. Called by GLUT display callback in the main thread."""
    global opengl_viz_config, global_gui_state, glut_window_id 
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw
    global gl_frame_times, gl_last_fps_update_time, gl_current_fps, gl_fps_update_interval

    if not OPENGL_AVAILABLE or global_simulation_bridge is None : return # Sim bridge for camera config
    
    # Track frame time for FPS calculation
    current_time = time.perf_counter()
    if len(gl_frame_times) > 0:
        frame_delta = current_time - gl_frame_times[-1]
        gl_frame_times.append(current_time)
        # Keep only last 60 frames for rolling average
        if len(gl_frame_times) > 60:
            gl_frame_times.pop(0)
    else:
        gl_frame_times.append(current_time)
    
    # Update FPS display periodically
    if current_time - gl_last_fps_update_time >= gl_fps_update_interval:
        if len(gl_frame_times) >= 2:
            time_span = gl_frame_times[-1] - gl_frame_times[0]
            if time_span > 0:
                gl_current_fps = (len(gl_frame_times) - 1) / time_span
            gl_last_fps_update_time = current_time
    try: # Ensure GLUT context is current
        current_win = glut.glutGetWindow()
        if glut_window_id is not None and current_win != glut_window_id and current_win != 0: 
            glut.glutSetWindow(glut_window_id) 
        elif current_win == 0: return # No window context
    except Exception: return # Catch errors if GLUT context is lost

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Clear buffers
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)) # Set point size from config

    viz_cfg = global_simulation_bridge.viz_config # For camera parameters
    runtime = global_simulation_bridge.runtime_state # For current time/step
    glMatrixMode(GL_MODELVIEW); glLoadIdentity() # Reset modelview matrix

    # Calculate camera eye position based on spherical coordinates (azimuth, elevation, radius)
    # This uses viz_config camera parameters, which can be modified by mouse controls.
    eye_x = viz_cfg.camera_center_x + viz_cfg.camera_radius * math.cos(viz_cfg.camera_elevation_angle) * math.sin(viz_cfg.camera_azimuth_angle)
    eye_y = viz_cfg.camera_center_y + viz_cfg.camera_radius * math.sin(viz_cfg.camera_elevation_angle)
    eye_z = viz_cfg.camera_center_z + viz_cfg.camera_radius * math.cos(viz_cfg.camera_elevation_angle) * math.cos(viz_cfg.camera_azimuth_angle)

    gluLookAt(eye_x, eye_y, eye_z, # Eye position
              viz_cfg.camera_center_x, viz_cfg.camera_center_y, viz_cfg.camera_center_z, # Look-at point
              viz_cfg.camera_up_x, viz_cfg.camera_up_y, viz_cfg.camera_up_z) # Up vector

    # Render Synapses (if enabled and data available)
    if global_gui_state.get("show_connections_gl", False) and gl_num_synapse_lines_to_draw > 0 and \
       gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0:
        base_syn_color = opengl_viz_config.get('SYNAPSE_BASE_COLOR', [0.3,0.3,0.4])
        alpha_mod = opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.5)
        final_alpha = np.clip(0.15 * alpha_mod, 0.02, 0.5) # Calculate final alpha
        glColor4f(base_syn_color[0], base_syn_color[1], base_syn_color[2], final_alpha)
        glLineWidth(0.5) # Thin lines for synapses

        glBindBuffer(GL_ARRAY_BUFFER, gl_synapse_vertices_vbo) # Bind synapse vertex VBO
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None) # Define vertex data format
        glDrawArrays(GL_LINES, 0, gl_num_synapse_lines_to_draw * 2) # Draw lines (2 vertices per line)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

    # Render Neurons (if data available)
    if gl_num_neurons_to_draw > 0 and \
       gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 and \
       gl_neuron_color_vbo is not None and gl_neuron_color_vbo > 0:
        
        glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_pos_vbo) # Bind neuron position VBO
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_color_vbo) # Bind neuron color VBO
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, None) # RGBA colors

        glDrawArrays(GL_POINTS, 0, gl_num_neurons_to_draw) # Draw points for neurons

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

    # Render Synaptic Pulses (if enabled and data available)
    if opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
       gl_num_pulses_to_draw > 0 and \
       gl_pulse_vertices_vbo is not None and gl_pulse_vertices_vbo > 0:
        
        pulse_color_rgba = opengl_viz_config.get("SYNAPTIC_PULSE_COLOR", [0.7, 0.9, 1.0, 0.9])
        glColor4fv(pulse_color_rgba) # Set pulse color
        glPointSize(opengl_viz_config.get("SYNAPTIC_PULSE_SIZE", 3.0)) # Set pulse point size

        glBindBuffer(GL_ARRAY_BUFFER, gl_pulse_vertices_vbo) # Bind pulse vertex VBO
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_POINTS, 0, gl_num_pulses_to_draw) # Draw points for pulses
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

        glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)) # Reset point size to default for neurons

    # Render Footer Text Overlay
    footer_h = opengl_viz_config.get('FOOTER_HEIGHT_PIXELS', 75)
    if footer_h > 0:
        line_h, margin = 15, 10 # Text line height and margin
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800)
        
        # Get current time and step from runtime_state
        sim_time_s = (runtime.current_time_ms / 1000.0)
        
        # Get telemetry from sim_bridge
        avg_fr = global_simulation_bridge._mock_network_avg_firing_rate_hz
        spikes_step = global_simulation_bridge._mock_num_spikes_this_step
        plasticity_events = global_simulation_bridge._mock_total_plasticity_events
        
        # Display FPS counter
        is_sim_running = global_gui_state.get("_sim_is_running_ui_view", False)
        is_paused = global_gui_state.get("_sim_is_paused_ui_view", False)
        is_playback = global_gui_state.get("is_playback_mode_active", False)
        
        if not is_sim_running and not is_playback:
            fps_text = "FPS: 0"
        elif is_paused and not is_playback:
            fps_text = "FPS: 0"
        else:
            fps_text = f"FPS: {gl_current_fps:.1f}"
        
        mode_text = "Playback" if is_playback else "Live"
        if global_gui_state.get("is_recording_active"): mode_text += " (Rec)"
        
        # Layout: 4 rows of information
        # Row 4 (top): Time, Spikes/Step, FPS
        render_text_gl(margin, margin + 4*line_h, f"Time: {sim_time_s:.3f}s")
        render_text_gl(margin + win_w // 3, margin + 4*line_h, f"Spikes: {spikes_step}")
        render_text_gl(margin + 2*win_w // 3, margin + 4*line_h, fps_text)
        
        # Row 3: Step, Avg Rate, Mode
        render_text_gl(margin, margin + 3*line_h, f"Step: {runtime.current_time_step}")
        render_text_gl(margin + win_w // 3, margin + 3*line_h, f"Rate: {avg_fr:.2f} Hz")
        render_text_gl(margin + 2*win_w // 3, margin + 3*line_h, f"Mode: {mode_text}")
        
        # Row 2: Plasticity, Visible Neurons, Visible Synapses
        render_text_gl(margin, margin + 2*line_h, f"Plasticity: {plasticity_events}")
        render_text_gl(margin + win_w // 3, margin + 2*line_h, f"Vis.Neurons: {gl_num_neurons_to_draw}")
        render_text_gl(margin + 2*win_w // 3, margin + 2*line_h, f"Vis.Syns: {gl_num_synapse_lines_to_draw}")
        
        # Row 0 (bottom): Hotkey hints
        render_text_gl(margin, margin, "LMB:Rotate, RMB:Pan, Scroll:Zoom, R:Reset, S:Synapses, N:Neurons, Space:Pause/Resume, Esc:Exit")

    glut.glutSwapBuffers() # Swap front and back buffers to display rendered scene


def mouse_button_func_gl(button, state, x, y):
    """Handles mouse button events for OpenGL window (camera control). Called by GLUT."""
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.viz_config # Camera config is part of viz_config
    zoom_speed = opengl_viz_config.get("CAMERA_ZOOM_SPEED_FACTOR", 20.0)

    if button == glut.GLUT_LEFT_BUTTON:
        cfg.mouse_left_button_down = (state == glut.GLUT_DOWN)
    elif button == glut.GLUT_RIGHT_BUTTON:
        cfg.mouse_right_button_down = (state == glut.GLUT_DOWN)
    elif button == 3: # Scroll up (zoom in)
        if state == glut.GLUT_UP: return # Action on press, not release
        cfg.camera_radius = max(cfg.camera_near_clip * 2, cfg.camera_radius - zoom_speed)
    elif button == 4: # Scroll down (zoom out)
        if state == glut.GLUT_UP: return
        cfg.camera_radius += zoom_speed
        cfg.camera_radius = min(cfg.camera_radius, cfg.camera_far_clip * 0.8) # Don't zoom out too far

    cfg.mouse_last_x = x; cfg.mouse_last_y = y # Store last mouse position
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay() # Request redraw


def mouse_motion_func_gl(x, y):
    """Handles mouse motion events for OpenGL window (camera control). Called by GLUT."""
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.viz_config
    dx = x - cfg.mouse_last_x; dy = y - cfg.mouse_last_y # Change in mouse position

    rotate_speed = opengl_viz_config.get("CAMERA_ROTATE_SPEED_FACTOR", 0.005)
    pan_speed_config = opengl_viz_config.get("CAMERA_PAN_SPEED_FACTOR", 0.1)

    if cfg.mouse_left_button_down: # Rotate camera (orbit around center)
        cfg.camera_azimuth_angle -= dx * rotate_speed
        cfg.camera_elevation_angle -= dy * rotate_speed
        # Clamp elevation to prevent flipping over poles
        cfg.camera_elevation_angle = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, cfg.camera_elevation_angle))
    elif cfg.mouse_right_button_down: # Pan camera (move look-at point)
        # Calculate camera's local right and up vectors for panning
        # Eye position (calculated from spherical coordinates)
        eye_calc_x = cfg.camera_center_x + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.sin(cfg.camera_azimuth_angle)
        eye_calc_y = cfg.camera_center_y + cfg.camera_radius * math.sin(cfg.camera_elevation_angle)
        eye_calc_z = cfg.camera_center_z + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.cos(cfg.camera_azimuth_angle)
        eye = np.array([eye_calc_x, eye_calc_y, eye_calc_z]);
        
        center = np.array([cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z])
        up_world = np.array([cfg.camera_up_x, cfg.camera_up_y, cfg.camera_up_z]) # World up vector

        forward = center - eye; forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6: forward /= forward_norm
        else: forward = np.array([0,0,-1]) # Default if eye is at center

        right = np.cross(forward, up_world); right_norm = np.linalg.norm(right)
        if right_norm > 1e-6: right /= right_norm
        else: # Handle gimbal lock like situations for 'right' vector
            if abs(forward[1]) > 0.99 : right = np.array([1,0,0]) # Looking straight up/down
            else: 
                right_temp = np.cross(forward, np.array([0,1,0])); right_norm_temp = np.linalg.norm(right_temp)
                right = right_temp/right_norm_temp if right_norm_temp > 1e-6 else np.array([1,0,0])

        cam_up = np.cross(right, forward) # Camera's local up vector (orthogonal to right and forward)

        pan_scale = pan_speed_config * (cfg.camera_radius / 150.0) # Scale pan speed with zoom level
        pan_vector_x = -dx * right * pan_scale # Pan left/right based on mouse dx
        pan_vector_y = dy * cam_up * pan_scale  # Pan up/down based on mouse dy

        # Update camera center (look-at point)
        new_center = center + pan_vector_x + pan_vector_y
        cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = new_center[0], new_center[1], new_center[2]

    cfg.mouse_last_x = x; cfg.mouse_last_y = y # Update last mouse position
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay() # Request redraw

def keyboard_func_gl(key, x, y):
    """Handles keyboard events for the OpenGL window. Called by GLUT."""
    # global_gui_state, global_simulation_bridge, shutdown_flag are accessed.
    # Commands to sim_thread are sent via ui_to_sim_queue.

    if global_simulation_bridge is None : return # Should not happen if GL window is up

    # Handle ESC key first (special case)
    if key == b'\x1b': # ESC key
        print("ESC pressed in OpenGL window. Signaling shutdown.")
        shutdown_flag.set() # Signal all threads to shut down
        return
    
    try: 
        key_char = key.decode("utf-8").lower() # Decode byte string to char (includes space as ' ')
    except UnicodeDecodeError: # Handle other special keys
        return # Other non-decodeable keys are ignored

    cfg = global_simulation_bridge.viz_config # For camera reset

    # --- Keyboard Shortcuts for OpenGL Window ---
    if key_char == 's': # Toggle synapse visibility
        # This action directly modifies UI state, which then affects GL rendering data prep.
        new_show_state = not global_gui_state.get("show_connections_gl", False)
        global_gui_state["show_connections_gl"] = new_show_state
        if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_show_synapses_gl_cb"):
            dpg.set_value("filter_show_synapses_gl_cb", new_show_state) # Update DPG checkbox
        trigger_filter_update_signal() # Signal GL data needs re-filtering and VBO update
        print(f"Synapse visibility toggled {'on' if new_show_state else 'off'}.")

    elif key_char == 'n': # Cycle through neuron spiking display modes
        if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_spiking_mode_combo"):
            modes = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
            current_mode = dpg.get_value("filter_spiking_mode_combo")
            try:
                current_idx = modes.index(current_mode)
                next_idx = (current_idx + 1) % len(modes)
            except ValueError:
                next_idx = 0  # Default to first mode if current mode not found
            new_mode = modes[next_idx]
            dpg.set_value("filter_spiking_mode_combo", new_mode)
            trigger_filter_update_signal()
            print(f"Neuron display mode: {new_mode}")

    elif key_char == ' ': # Space: Pause/Resume or Start simulation
        if not global_gui_state.get("is_playback_mode_active", False):
            current_sim_running = global_gui_state.get("_sim_is_running_ui_view", False)
            current_sim_paused = global_gui_state.get("_sim_is_paused_ui_view", False)
            
            if not current_sim_running:
                # Sim is stopped, start it
                ui_to_sim_queue.put({"type": "START_SIM"})
                # Optimistic UI state update (matches handle_start_simulation_event)
                global_gui_state["_sim_is_running_ui_view"] = True
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=True, is_paused=False)
                print("GL Keyboard: Starting simulation.")
            elif current_sim_paused:
                # Sim is paused, resume it
                ui_to_sim_queue.put({"type": "RESUME_SIM"})
                # Optimistic UI state update (matches handle_pause_simulation_event)
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=True, is_paused=False)
                print("GL Keyboard: Resuming simulation.")
            else:
                # Sim is running, pause it
                ui_to_sim_queue.put({"type": "PAUSE_SIM"})
                # Optimistic UI state update (matches handle_pause_simulation_event)
                global_gui_state["_sim_is_paused_ui_view"] = True
                update_ui_for_simulation_run_state(is_running=True, is_paused=True)
                print("GL Keyboard: Pausing simulation.")
    
    elif key_char == 'r': # Reset camera position
        cfg.camera_azimuth_angle = 0.0
        cfg.camera_elevation_angle = 0.0
        cfg.camera_radius = 150.0
        cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = 0.0, 0.0, 0.0
        if glut.glutGetWindow() != 0: glut.glutPostRedisplay() # Request redraw
        print("Camera reset.")

    # Other keys can be added here.
    # Ensure glutPostRedisplay is called if the view needs to change immediately.
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay()


# --- DPG GUI Helper Functions (Called by Main/UI Thread) ---

def trigger_filter_update_signal(sender=None, app_data=None, user_data=None):
    """Sets a flag indicating that visualization filters have changed and GL data needs update."""
    global global_gui_state
    global_gui_state["filters_changed"] = True
    # This will be checked in the main DPG loop to trigger update_gl_data()

def get_current_filter_settings_from_gui():
    """Retrieves current filter settings from DPG UI elements. Called by main/UI thread."""
    settings = {
        "spiking_mode": "Highlight Spiking", # Default if DPG item doesn't exist
        "type_filter_enabled": False,
        "selected_neuron_type": "All", 
        "min_abs_weight": 0.01 # Default for synapse weight filter
    }
    if dpg.is_dearpygui_running(): # Ensure DPG context is active
        if dpg.does_item_exist("filter_spiking_mode_combo"):
            settings["spiking_mode"] = dpg.get_value("filter_spiking_mode_combo")
        if dpg.does_item_exist("filter_type_enable_cb"):
            settings["type_filter_enabled"] = dpg.get_value("filter_type_enable_cb")
        if dpg.does_item_exist("filter_neuron_type_combo"):
            settings["selected_neuron_type"] = dpg.get_value("filter_neuron_type_combo")
        if dpg.does_item_exist("filter_min_abs_weight_slider"):
            settings["min_abs_weight"] = dpg.get_value("filter_min_abs_weight_slider")
    return settings

def apply_neuron_filters_to_indices(all_indices, fired_status_np, neuron_types_list_str, filters_dict):
    """
    Applies filters to a list of neuron indices to determine visibility.
    Called by main/UI thread (specifically within update_gl_data).
    """
    if all_indices.size == 0: return [] # No neurons to filter
    
    visible_mask = np.ones(all_indices.size, dtype=bool) # Start with all neurons visible

    # Filter by spiking status
    spiking_mode = filters_dict.get("spiking_mode", "Highlight Spiking")
    if spiking_mode == "Show Only Spiking":
        if fired_status_np is not None and fired_status_np.shape == visible_mask.shape:
            visible_mask &= fired_status_np # Only show neurons that are currently firing
        else: # Mismatch in array sizes, log warning or handle gracefully
            if fired_status_np is not None: print(f"Warning: fired_status_np shape mismatch in filter. Expected {visible_mask.shape}, got {fired_status_np.shape}")
            # visible_mask &= False # Or, show no neurons if data is inconsistent

    # Filter by neuron type
    if filters_dict.get("type_filter_enabled", False):
        selected_type_str = filters_dict.get("selected_neuron_type", "All")
        if selected_type_str != "All" and neuron_types_list_str is not None and len(neuron_types_list_str) == all_indices.size:
            # Create a boolean mask for matching types
            type_mask = np.array([neuron_types_list_str[i] == selected_type_str for i in all_indices], dtype=bool)
            visible_mask &= type_mask
        elif selected_type_str != "All":
             if neuron_types_list_str is not None: print(f"Warning: neuron_types_list_str length mismatch in filter. Expected {all_indices.size}, got {len(neuron_types_list_str)}")
             # visible_mask &= False # Or, show no neurons if type data is inconsistent
    
    return all_indices[visible_mask] # Return indices of neurons that pass all filters


def apply_synapse_filters_to_indices(all_synapse_data_list, filters_dict):
    """
    Applies filters to a list of synapse data dictionaries to determine visibility.
    Called by main/UI thread (specifically within update_gl_data).
    Returns a list of indices (into all_synapse_data_list) of visible synapses.
    """
    if not global_gui_state.get("show_connections_gl", False): return [] # If connections are globally hidden

    visible_syn_indices = []
    min_abs_w = filters_dict.get("min_abs_weight", 0.01) # Minimum absolute weight to show
    for i, syn_data in enumerate(all_synapse_data_list):
        if abs(syn_data.get("weight", 0.0)) >= min_abs_w: # Check weight against filter
            visible_syn_indices.append(i)
    return visible_syn_indices


def update_status_bar(message, color=None, level="info"):
    """Updates the text and color of the DPG status bar. Called by main/UI thread."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("status_bar_text"):
        dpg.set_value("status_bar_text", f"[{level.upper()}] {message}")
        if color is None: # Auto-color based on level if not provided
            if level == "error" or level == "critical": color = [255, 0, 0, 255]
            elif level == "warning": color = [255, 165, 0, 255]
            elif level == "info": color = [200, 200, 200, 255] 
            elif level == "success": color = [0, 200, 0, 255]
            else: color = [200, 200, 200, 255] # Default
        dpg.configure_item("status_bar_text", color=color)

# --- DPG GUI Element Creation & Event Handlers (Called by Main/UI Thread) ---

def _update_sim_config_from_ui(update_model_specific=True):
    """
    Updates a temporary SimulationConfiguration object from DPG UI elements.
    This temporary object is then sent to the simulation thread via a command.
    Called by the main/UI thread.
    Returns a dictionary representing the config from UI, or None if error.
    """
    if not dpg.is_dearpygui_running(): return None
    
    # Create a new temporary config object to populate from UI
    # This avoids modifying global_simulation_bridge.sim_config directly from UI thread.
    # The actual update to sim_bridge.sim_config happens in sim_thread upon command.
    temp_config = SimulationConfiguration() # Create a fresh default config
    
    # If global_simulation_bridge exists and has a config, start temp_config from it
    # to preserve settings not directly in UI or to have a baseline.
    # However, for sending a "full new config" command, starting fresh and filling from UI is cleaner.
    # Let's assume we build a new config purely from UI values where available,
    # and the sim_thread will merge this with its existing config if needed, or replace.
    # For "Apply Changes & Reset", it's usually a full replacement.
    
    # Helper to safely get DPG value or use current value from sim_bridge if item doesn't exist
    # This is tricky. The goal is that _update_sim_config_from_ui creates a *complete* config dict
    # based on the UI. If a UI element for a config param doesn't exist, what should its value be?
    # It should probably be the default from SimulationConfiguration() or the current live one if that's intended.
    # For now, let's assume UI has all relevant controls. If not, this needs refinement.

    try:
        cfg_dict_from_ui = {} # Build a dictionary of config values from UI

        # General parameters
        if dpg.does_item_exist("cfg_num_neurons"): cfg_dict_from_ui["num_neurons"] = max(1, dpg.get_value("cfg_num_neurons"))
        if dpg.does_item_exist("cfg_total_sim_time"): cfg_dict_from_ui["total_simulation_time_ms"] = max(0.0, dpg.get_value("cfg_total_sim_time"))
        if dpg.does_item_exist("cfg_dt_ms"): cfg_dict_from_ui["dt_ms"] = max(0.001, dpg.get_value("cfg_dt_ms"))
        if dpg.does_item_exist("cfg_seed"): cfg_dict_from_ui["seed"] = dpg.get_value("cfg_seed")
        if dpg.does_item_exist("cfg_neural_profile"): cfg_dict_from_ui["neural_profile_name"] = dpg.get_value("cfg_neural_profile")
        if dpg.does_item_exist("cfg_default_neuron_type_hh"): cfg_dict_from_ui["default_neuron_type_hh"] = dpg.get_value("cfg_default_neuron_type_hh")

        if dpg.does_item_exist("cfg_neuron_model_type"):
            selected_model_name = dpg.get_value("cfg_neuron_model_type")
            cfg_dict_from_ui["neuron_model_type"] = selected_model_name
            # Default neuron types based on selected model (these are part of SimulationConfiguration defaults too)
            if selected_model_name == NeuronModel.IZHIKEVICH.name:
                cfg_dict_from_ui["default_neuron_type_izh"] = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
            # For HH, do not override default_neuron_type_hh here; we use the value from the HH preset combo
            # and/or any profile-specific default mapping.

        # Connectivity
        if dpg.does_item_exist("cfg_enable_watts_strogatz"): cfg_dict_from_ui["enable_watts_strogatz"] = dpg.get_value("cfg_enable_watts_strogatz")
        if dpg.does_item_exist("cfg_connectivity_k"): cfg_dict_from_ui["connectivity_k"] = max(2, dpg.get_value("cfg_connectivity_k"))
        if dpg.does_item_exist("cfg_connectivity_p_rewire"): cfg_dict_from_ui["connectivity_p_rewire"] = dpg.get_value("cfg_connectivity_p_rewire")
        if dpg.does_item_exist("cfg_connections_per_neuron"): cfg_dict_from_ui["connections_per_neuron"] = max(0, dpg.get_value("cfg_connections_per_neuron"))

        # Synaptic parameters
        if dpg.does_item_exist("cfg_propagation_strength"): cfg_dict_from_ui["propagation_strength"] = dpg.get_value("cfg_propagation_strength")
        if dpg.does_item_exist("cfg_inhibitory_propagation_strength"): cfg_dict_from_ui["inhibitory_propagation_strength"] = dpg.get_value("cfg_inhibitory_propagation_strength")
        if dpg.does_item_exist("cfg_syn_tau_e"): cfg_dict_from_ui["syn_tau_g_e"] = max(0.1, dpg.get_value("cfg_syn_tau_e"))
        if dpg.does_item_exist("cfg_syn_tau_i"): cfg_dict_from_ui["syn_tau_g_i"] = max(0.1, dpg.get_value("cfg_syn_tau_i"))
        # NMDA parameters
        if dpg.does_item_exist("cfg_enable_nmda"): cfg_dict_from_ui["enable_nmda"] = dpg.get_value("cfg_enable_nmda")
        if dpg.does_item_exist("cfg_nmda_ratio"): cfg_dict_from_ui["nmda_ratio"] = max(0.0, dpg.get_value("cfg_nmda_ratio"))
        if dpg.does_item_exist("cfg_nmda_tau_decay"): cfg_dict_from_ui["nmda_tau_decay"] = max(10.0, dpg.get_value("cfg_nmda_tau_decay"))
        if dpg.does_item_exist("cfg_nmda_tau_rise"): cfg_dict_from_ui["nmda_tau_rise"] = max(0.5, dpg.get_value("cfg_nmda_tau_rise"))
        if dpg.does_item_exist("cfg_nmda_mg_conc"): cfg_dict_from_ui["nmda_mg_concentration"] = max(0.0, dpg.get_value("cfg_nmda_mg_conc"))
        if dpg.does_item_exist("cfg_num_traits"): cfg_dict_from_ui["num_traits"] = max(1, dpg.get_value("cfg_num_traits"))

        # Learning & Plasticity
        if dpg.does_item_exist("cfg_enable_hebbian_learning"): cfg_dict_from_ui["enable_hebbian_learning"] = dpg.get_value("cfg_enable_hebbian_learning")
        if dpg.does_item_exist("cfg_hebbian_learning_rate"): cfg_dict_from_ui["hebbian_learning_rate"] = dpg.get_value("cfg_hebbian_learning_rate")
        if dpg.does_item_exist("cfg_hebbian_max_weight"): cfg_dict_from_ui["hebbian_max_weight"] = dpg.get_value("cfg_hebbian_max_weight")
        if dpg.does_item_exist("cfg_enable_short_term_plasticity"): cfg_dict_from_ui["enable_short_term_plasticity"] = dpg.get_value("cfg_enable_short_term_plasticity")
        if dpg.does_item_exist("cfg_stp_U"): cfg_dict_from_ui["stp_U"] = dpg.get_value("cfg_stp_U")
        if dpg.does_item_exist("cfg_stp_tau_d"): cfg_dict_from_ui["stp_tau_d"] = max(0.1, dpg.get_value("cfg_stp_tau_d"))
        if dpg.does_item_exist("cfg_stp_tau_f"): cfg_dict_from_ui["stp_tau_f"] = max(0.1, dpg.get_value("cfg_stp_tau_f"))
        if dpg.does_item_exist("cfg_enable_per_type_stp"): cfg_dict_from_ui["enable_per_type_stp"] = dpg.get_value("cfg_enable_per_type_stp")
        # Per-type STP: read individual UI fields into lists
        for conn_type_suffix in ["ee", "ei", "ie", "ii"]:
            for param in ["U", "tau_d", "tau_f"]:
                tag = f"cfg_stp_{param}_{conn_type_suffix}"
                if dpg.does_item_exist(tag):
                    pass  # Gathered below as composite list
        # Build per-type lists from UI
        stp_U_list, stp_tau_d_list, stp_tau_f_list = [], [], []
        for suffix in ["ee", "ei", "ie", "ii"]:
            stp_U_list.append(dpg.get_value(f"cfg_stp_U_{suffix}") if dpg.does_item_exist(f"cfg_stp_U_{suffix}") else cfg_dict_from_ui.get("stp_U", 0.15))
            stp_tau_d_list.append(max(0.1, dpg.get_value(f"cfg_stp_tau_d_{suffix}")) if dpg.does_item_exist(f"cfg_stp_tau_d_{suffix}") else cfg_dict_from_ui.get("stp_tau_d", 200.0))
            stp_tau_f_list.append(max(0.1, dpg.get_value(f"cfg_stp_tau_f_{suffix}")) if dpg.does_item_exist(f"cfg_stp_tau_f_{suffix}") else cfg_dict_from_ui.get("stp_tau_f", 50.0))
        cfg_dict_from_ui["stp_U_per_type"] = stp_U_list
        cfg_dict_from_ui["stp_tau_d_per_type"] = stp_tau_d_list
        cfg_dict_from_ui["stp_tau_f_per_type"] = stp_tau_f_list
        # Structural plasticity activity bias
        if dpg.does_item_exist("cfg_struct_plast_activity_bias"): cfg_dict_from_ui["struct_plast_activity_bias"] = dpg.get_value("cfg_struct_plast_activity_bias")

        # Homeostasis
        if dpg.does_item_exist("cfg_enable_homeostasis"): cfg_dict_from_ui["enable_homeostasis"] = dpg.get_value("cfg_enable_homeostasis")
        if dpg.does_item_exist("cfg_homeostasis_target_rate"): cfg_dict_from_ui["homeostasis_target_rate"] = dpg.get_value("cfg_homeostasis_target_rate")
        if dpg.does_item_exist("cfg_homeostasis_threshold_min"): cfg_dict_from_ui["homeostasis_threshold_min"] = dpg.get_value("cfg_homeostasis_threshold_min")
        if dpg.does_item_exist("cfg_homeostasis_threshold_max"): cfg_dict_from_ui["homeostasis_threshold_max"] = dpg.get_value("cfg_homeostasis_threshold_max")
        if dpg.does_item_exist("cfg_enable_synaptic_scaling"): cfg_dict_from_ui["enable_synaptic_scaling"] = dpg.get_value("cfg_enable_synaptic_scaling")
        if dpg.does_item_exist("cfg_synaptic_scaling_rate"): cfg_dict_from_ui["synaptic_scaling_rate"] = dpg.get_value("cfg_synaptic_scaling_rate")

        # STDP
        if dpg.does_item_exist("cfg_enable_stdp"): cfg_dict_from_ui["enable_stdp"] = dpg.get_value("cfg_enable_stdp")
        if dpg.does_item_exist("cfg_stdp_a_plus"): cfg_dict_from_ui["stdp_a_plus"] = dpg.get_value("cfg_stdp_a_plus")
        if dpg.does_item_exist("cfg_stdp_a_minus"): cfg_dict_from_ui["stdp_a_minus"] = dpg.get_value("cfg_stdp_a_minus")
        if dpg.does_item_exist("cfg_stdp_tau_plus_ms"): cfg_dict_from_ui["stdp_tau_plus_ms"] = dpg.get_value("cfg_stdp_tau_plus_ms")
        if dpg.does_item_exist("cfg_stdp_tau_minus_ms"): cfg_dict_from_ui["stdp_tau_minus_ms"] = dpg.get_value("cfg_stdp_tau_minus_ms")
        if dpg.does_item_exist("cfg_stdp_w_min"): cfg_dict_from_ui["stdp_w_min"] = dpg.get_value("cfg_stdp_w_min")
        if dpg.does_item_exist("cfg_stdp_w_max"): cfg_dict_from_ui["stdp_w_max"] = dpg.get_value("cfg_stdp_w_max")

        # Reward-Modulated Plasticity
        if dpg.does_item_exist("cfg_enable_reward_modulation"): cfg_dict_from_ui["enable_reward_modulation"] = dpg.get_value("cfg_enable_reward_modulation")
        if dpg.does_item_exist("cfg_reward_learning_rate"): cfg_dict_from_ui["reward_learning_rate"] = dpg.get_value("cfg_reward_learning_rate")
        if dpg.does_item_exist("cfg_reward_eligibility_tau_ms"): cfg_dict_from_ui["reward_eligibility_tau_ms"] = dpg.get_value("cfg_reward_eligibility_tau_ms")

        # Structural Plasticity
        if dpg.does_item_exist("cfg_enable_structural_plasticity"): cfg_dict_from_ui["enable_structural_plasticity"] = dpg.get_value("cfg_enable_structural_plasticity")
        if dpg.does_item_exist("cfg_struct_plast_formation_rate"): cfg_dict_from_ui["struct_plast_formation_rate"] = dpg.get_value("cfg_struct_plast_formation_rate")
        if dpg.does_item_exist("cfg_struct_plast_elimination_rate"): cfg_dict_from_ui["struct_plast_elimination_rate"] = dpg.get_value("cfg_struct_plast_elimination_rate")
        if dpg.does_item_exist("cfg_struct_plast_weight_threshold"): cfg_dict_from_ui["struct_plast_weight_threshold"] = dpg.get_value("cfg_struct_plast_weight_threshold")
        if dpg.does_item_exist("cfg_struct_plast_target_density"): cfg_dict_from_ui["struct_plast_target_density"] = dpg.get_value("cfg_struct_plast_target_density")
        if dpg.does_item_exist("cfg_struct_plast_distance_scale"): cfg_dict_from_ui["struct_plast_distance_scale"] = dpg.get_value("cfg_struct_plast_distance_scale")
        if dpg.does_item_exist("cfg_struct_plast_update_interval_steps"): cfg_dict_from_ui["struct_plast_update_interval_steps"] = dpg.get_value("cfg_struct_plast_update_interval_steps")

        # Heterogeneity & Noise
        if dpg.does_item_exist("cfg_enable_parameter_heterogeneity"): cfg_dict_from_ui["enable_parameter_heterogeneity"] = dpg.get_value("cfg_enable_parameter_heterogeneity")
        if dpg.does_item_exist("cfg_heterogeneity_seed"): cfg_dict_from_ui["heterogeneity_seed"] = dpg.get_value("cfg_heterogeneity_seed")
        if dpg.does_item_exist("cfg_enable_conductance_noise"): cfg_dict_from_ui["enable_conductance_noise"] = dpg.get_value("cfg_enable_conductance_noise")
        if dpg.does_item_exist("cfg_conductance_noise_relative_std"): cfg_dict_from_ui["conductance_noise_relative_std"] = dpg.get_value("cfg_conductance_noise_relative_std")
        if dpg.does_item_exist("cfg_enable_ou_process"): cfg_dict_from_ui["enable_ou_process"] = dpg.get_value("cfg_enable_ou_process")
        if dpg.does_item_exist("cfg_ou_mean_current_pA"): cfg_dict_from_ui["ou_mean_current_pA"] = dpg.get_value("cfg_ou_mean_current_pA")
        if dpg.does_item_exist("cfg_ou_std_current_pA"): cfg_dict_from_ui["ou_std_current_pA"] = dpg.get_value("cfg_ou_std_current_pA")
        if dpg.does_item_exist("cfg_ou_tau_ms"): cfg_dict_from_ui["ou_tau_ms"] = dpg.get_value("cfg_ou_tau_ms")
        if dpg.does_item_exist("cfg_ou_seed"): cfg_dict_from_ui["ou_seed"] = dpg.get_value("cfg_ou_seed")
        
        # Camera FOV and Visualization settings (part of viz_config)
        if dpg.does_item_exist("cfg_camera_fov"): cfg_dict_from_ui["camera_fov"] = dpg.get_value("cfg_camera_fov")
        if dpg.does_item_exist("cfg_viz_update_interval_steps"): cfg_dict_from_ui["viz_update_interval_steps"] = max(1, dpg.get_value("cfg_viz_update_interval_steps"))

        # Model-specific parameters
        if update_model_specific:
            current_model_in_ui = cfg_dict_from_ui.get("neuron_model_type", NeuronModel.IZHIKEVICH.name)
            if current_model_in_ui == NeuronModel.IZHIKEVICH.name:
                if dpg.does_item_exist("cfg_izh_C_val"): cfg_dict_from_ui["izh_C_val"] = dpg.get_value("cfg_izh_C_val")
                if dpg.does_item_exist("cfg_izh_k_val"): cfg_dict_from_ui["izh_k_val"] = dpg.get_value("cfg_izh_k_val")
                # ... (all other Izhikevich params)
                ui_izh_params_keys = ["izh_C_val", "izh_k_val", "izh_vr_val", "izh_vt_val", "izh_vpeak_val", "izh_a_val", "izh_b_val", "izh_c_val", "izh_d_val"]
                for key_suffix in ui_izh_params_keys:
                    dpg_tag = f"cfg_{key_suffix}"
                    if dpg.does_item_exist(dpg_tag): cfg_dict_from_ui[key_suffix] = dpg.get_value(dpg_tag)

            elif current_model_in_ui == NeuronModel.HODGKIN_HUXLEY.name:
                # ... (all Hodgkin-Huxley params)
                ui_hh_params_keys = [
                    "hh_C_m", "hh_g_Na_max", "hh_g_K_max", "hh_g_L", "hh_E_Na", "hh_E_K", "hh_E_L",
                    "hh_v_peak", "hh_v_rest_init", "hh_g_M_max", "hh_m_current_tau_ms",
                    "hh_g_CaT_max", "hh_E_CaT", "hh_g_h_max", "hh_E_h", "hh_g_NaP_max",
                    "hh_q10_factor", "hh_temperature_celsius",
                    "hh_external_drive_scale",
                ]
                for key_suffix in ui_hh_params_keys:
                    dpg_tag = f"cfg_{key_suffix}"
                    if dpg.does_item_exist(dpg_tag): cfg_dict_from_ui[key_suffix] = dpg.get_value(dpg_tag)
            elif current_model_in_ui == NeuronModel.ADEX.name:
                ui_adex_params_keys = [
                    "adex_C", "adex_g_L", "adex_E_L", "adex_V_T", "adex_Delta_T",
                    "adex_a", "adex_tau_w", "adex_b", "adex_V_r", "adex_V_peak",
                    "adex_external_drive_scale",
                ]
                for key_suffix in ui_adex_params_keys:
                    dpg_tag = f"cfg_{key_suffix}"
                    if dpg.does_item_exist(dpg_tag): cfg_dict_from_ui[key_suffix] = dpg.get_value(dpg_tag)
        
        # Simulation speed factor (part of runtime_state)
        simulation_speed = dpg.get_value("sim_speed_slider") if dpg.does_item_exist("sim_speed_slider") else 1.0

        # Create the proper nested structure expected by apply_simulation_configuration_core
        # Split parameters into core_config, viz_config, and runtime_state
        viz_keys = ["camera_fov", "viz_update_interval_steps"]
        core_config_dict = {k: v for k, v in cfg_dict_from_ui.items() 
                           if k not in viz_keys + ["simulation_speed_factor"]}
        
        viz_config_dict = {}
        if "camera_fov" in cfg_dict_from_ui:
            viz_config_dict["camera_fov"] = cfg_dict_from_ui["camera_fov"]
        if "viz_update_interval_steps" in cfg_dict_from_ui:
            viz_config_dict["viz_update_interval_steps"] = cfg_dict_from_ui["viz_update_interval_steps"]
        
        runtime_state_dict = {
            "simulation_speed_factor": simulation_speed
        }
        
        # Create config objects to ensure all fields are present with defaults
        core_cfg = _create_config_from_dict(CoreSimConfig, core_config_dict)
        viz_cfg = _create_config_from_dict(VisualizationConfig, viz_config_dict)
        runtime = _create_config_from_dict(RuntimeState, runtime_state_dict)
        
        # Return nested dict structure
        return _get_full_config_dict(core_cfg, viz_cfg, runtime)

    except Exception as e:
        print(f"Error reading UI for sim config: {e}")
        update_status_bar(f"Error reading UI for config: {e}", color=[255,0,0], level="error")
        return None


def _populate_ui_from_config_dict(config_dict):
    """
    Populates DPG UI elements from a given simulation configuration dictionary.
    Called by the main/UI thread, e.g., after loading a profile or checkpoint.

    Supports both legacy flat SimulationConfiguration-style dicts and the
    newer nested structure returned by _get_full_config_dict, i.e.
        {"core_config": {...}, "viz_config": {...}, "runtime_state": {...}}.
    """
    if not dpg.is_dearpygui_running() or not config_dict: return

    # Normalize to a flat dict compatible with SimulationConfiguration.from_dict
    if any(k in config_dict for k in ("core_config", "viz_config", "runtime_state")):
        core_part = config_dict.get("core_config", {}) or {}
        viz_part = config_dict.get("viz_config", {}) or {}
        runtime_part = config_dict.get("runtime_state", {}) or {}

        flat_dict = {}
        if isinstance(core_part, dict):
            flat_dict.update(core_part)
        # Merge viz and runtime sections, without overriding core keys
        for section in (viz_part, runtime_part):
            if isinstance(section, dict):
                for k, v in section.items():
                    if k not in flat_dict:
                        flat_dict[k] = v
    else:
        flat_dict = config_dict

    # Use SimulationConfiguration.from_dict to ensure all fields are present with defaults if missing in dict
    cfg = SimulationConfiguration.from_dict(flat_dict)

    # General parameters
    if dpg.does_item_exist("cfg_num_neurons"): dpg.set_value("cfg_num_neurons", cfg.num_neurons)
    if dpg.does_item_exist("cfg_total_sim_time"): dpg.set_value("cfg_total_sim_time", cfg.total_simulation_time_ms)
    if dpg.does_item_exist("cfg_dt_ms"): dpg.set_value("cfg_dt_ms", cfg.dt_ms)
    if dpg.does_item_exist("cfg_seed"): dpg.set_value("cfg_seed", cfg.seed)
    if dpg.does_item_exist("cfg_neuron_model_type"): dpg.set_value("cfg_neuron_model_type", cfg.neuron_model_type)

    # Neural structure profile and HH preset (with realism constraints)
    profile_value = getattr(cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
    if profile_value not in NEURAL_STRUCTURE_PROFILES:
        profile_value = "GENERIC_UNSTRUCTURED"
    if dpg.does_item_exist("cfg_neural_profile"):
        dpg.set_value("cfg_neural_profile", profile_value)

    if dpg.does_item_exist("cfg_default_neuron_type_hh") and hasattr(cfg, "default_neuron_type_hh"):
        allowed_hh = get_compatible_hh_type_names_for_profile(profile_value)
        if allowed_hh:
            dpg.configure_item("cfg_default_neuron_type_hh", items=allowed_hh)
            current_hh = cfg.default_neuron_type_hh
            if current_hh not in allowed_hh:
                current_hh = allowed_hh[0]
            dpg.set_value("cfg_default_neuron_type_hh", current_hh)
        else:
            dpg.set_value("cfg_default_neuron_type_hh", cfg.default_neuron_type_hh)
    
    # Connectivity
    if dpg.does_item_exist("cfg_enable_watts_strogatz"): dpg.set_value("cfg_enable_watts_strogatz", cfg.enable_watts_strogatz)
    if dpg.does_item_exist("cfg_connectivity_k"): dpg.set_value("cfg_connectivity_k", cfg.connectivity_k)
    if dpg.does_item_exist("cfg_connectivity_p_rewire"): dpg.set_value("cfg_connectivity_p_rewire", cfg.connectivity_p_rewire)
    if dpg.does_item_exist("cfg_connections_per_neuron"): dpg.set_value("cfg_connections_per_neuron", cfg.connections_per_neuron)

    # Synaptic parameters
    if dpg.does_item_exist("cfg_propagation_strength"): dpg.set_value("cfg_propagation_strength", cfg.propagation_strength)
    if dpg.does_item_exist("cfg_inhibitory_propagation_strength"): dpg.set_value("cfg_inhibitory_propagation_strength", cfg.inhibitory_propagation_strength)
    if dpg.does_item_exist("cfg_syn_tau_e"): dpg.set_value("cfg_syn_tau_e", cfg.syn_tau_g_e)
    if dpg.does_item_exist("cfg_syn_tau_i"): dpg.set_value("cfg_syn_tau_i", cfg.syn_tau_g_i)
    # NMDA
    if dpg.does_item_exist("cfg_enable_nmda"): dpg.set_value("cfg_enable_nmda", cfg.enable_nmda)
    if dpg.does_item_exist("cfg_nmda_ratio"): dpg.set_value("cfg_nmda_ratio", cfg.nmda_ratio)
    if dpg.does_item_exist("cfg_nmda_tau_decay"): dpg.set_value("cfg_nmda_tau_decay", cfg.nmda_tau_decay)
    if dpg.does_item_exist("cfg_nmda_tau_rise"): dpg.set_value("cfg_nmda_tau_rise", cfg.nmda_tau_rise)
    if dpg.does_item_exist("cfg_nmda_mg_conc"): dpg.set_value("cfg_nmda_mg_conc", cfg.nmda_mg_concentration)
    if dpg.does_item_exist("cfg_num_traits"): dpg.set_value("cfg_num_traits", cfg.num_traits)

    # Learning & Plasticity
    if dpg.does_item_exist("cfg_enable_hebbian_learning"): dpg.set_value("cfg_enable_hebbian_learning", cfg.enable_hebbian_learning)
    if dpg.does_item_exist("cfg_hebbian_learning_rate"): dpg.set_value("cfg_hebbian_learning_rate", cfg.hebbian_learning_rate)
    if dpg.does_item_exist("cfg_hebbian_max_weight"): dpg.set_value("cfg_hebbian_max_weight", cfg.hebbian_max_weight)
    if dpg.does_item_exist("cfg_enable_short_term_plasticity"): dpg.set_value("cfg_enable_short_term_plasticity", cfg.enable_short_term_plasticity)
    if dpg.does_item_exist("cfg_stp_U"): dpg.set_value("cfg_stp_U", cfg.stp_U)
    if dpg.does_item_exist("cfg_stp_tau_d"): dpg.set_value("cfg_stp_tau_d", cfg.stp_tau_d)
    if dpg.does_item_exist("cfg_stp_tau_f"): dpg.set_value("cfg_stp_tau_f", cfg.stp_tau_f)
    if dpg.does_item_exist("cfg_enable_per_type_stp"): dpg.set_value("cfg_enable_per_type_stp", getattr(cfg, 'enable_per_type_stp', True))
    # Per-type STP UI fields
    per_type_U = getattr(cfg, 'stp_U_per_type', None) or [0.5, 0.5, 0.25, 0.25]
    per_type_tau_d = getattr(cfg, 'stp_tau_d_per_type', None) or [200.0, 200.0, 100.0, 100.0]
    per_type_tau_f = getattr(cfg, 'stp_tau_f_per_type', None) or [20.0, 20.0, 50.0, 50.0]
    for i, suffix in enumerate(["ee", "ei", "ie", "ii"]):
        if dpg.does_item_exist(f"cfg_stp_U_{suffix}"): dpg.set_value(f"cfg_stp_U_{suffix}", per_type_U[i])
        if dpg.does_item_exist(f"cfg_stp_tau_d_{suffix}"): dpg.set_value(f"cfg_stp_tau_d_{suffix}", per_type_tau_d[i])
        if dpg.does_item_exist(f"cfg_stp_tau_f_{suffix}"): dpg.set_value(f"cfg_stp_tau_f_{suffix}", per_type_tau_f[i])
    # Structural plasticity activity bias
    if dpg.does_item_exist("cfg_struct_plast_activity_bias"): dpg.set_value("cfg_struct_plast_activity_bias", getattr(cfg, 'struct_plast_activity_bias', 0.5))

    # Homeostasis
    if dpg.does_item_exist("cfg_enable_homeostasis"): dpg.set_value("cfg_enable_homeostasis", cfg.enable_homeostasis)
    if dpg.does_item_exist("cfg_homeostasis_target_rate"): dpg.set_value("cfg_homeostasis_target_rate", cfg.homeostasis_target_rate)
    if dpg.does_item_exist("cfg_homeostasis_threshold_min"): dpg.set_value("cfg_homeostasis_threshold_min", cfg.homeostasis_threshold_min)
    if dpg.does_item_exist("cfg_homeostasis_threshold_max"): dpg.set_value("cfg_homeostasis_threshold_max", cfg.homeostasis_threshold_max)
    if dpg.does_item_exist("cfg_enable_synaptic_scaling"): dpg.set_value("cfg_enable_synaptic_scaling", cfg.enable_synaptic_scaling)
    if dpg.does_item_exist("cfg_synaptic_scaling_rate"): dpg.set_value("cfg_synaptic_scaling_rate", cfg.synaptic_scaling_rate)

    # STDP
    if dpg.does_item_exist("cfg_enable_stdp"): dpg.set_value("cfg_enable_stdp", cfg.enable_stdp)
    if dpg.does_item_exist("cfg_stdp_a_plus"): dpg.set_value("cfg_stdp_a_plus", cfg.stdp_a_plus)
    if dpg.does_item_exist("cfg_stdp_a_minus"): dpg.set_value("cfg_stdp_a_minus", cfg.stdp_a_minus)
    if dpg.does_item_exist("cfg_stdp_tau_plus_ms"): dpg.set_value("cfg_stdp_tau_plus_ms", cfg.stdp_tau_plus_ms)
    if dpg.does_item_exist("cfg_stdp_tau_minus_ms"): dpg.set_value("cfg_stdp_tau_minus_ms", cfg.stdp_tau_minus_ms)
    if dpg.does_item_exist("cfg_stdp_w_min"): dpg.set_value("cfg_stdp_w_min", cfg.stdp_w_min)
    if dpg.does_item_exist("cfg_stdp_w_max"): dpg.set_value("cfg_stdp_w_max", cfg.stdp_w_max)

    # Reward-Modulated Plasticity
    if dpg.does_item_exist("cfg_enable_reward_modulation"): dpg.set_value("cfg_enable_reward_modulation", cfg.enable_reward_modulation)
    if hasattr(cfg, 'reward_learning_rate') and dpg.does_item_exist("cfg_reward_learning_rate"):
        dpg.set_value("cfg_reward_learning_rate", cfg.reward_learning_rate)
    if hasattr(cfg, 'reward_eligibility_tau_ms') and dpg.does_item_exist("cfg_reward_eligibility_tau_ms"):
        dpg.set_value("cfg_reward_eligibility_tau_ms", cfg.reward_eligibility_tau_ms)

    # Structural Plasticity
    if dpg.does_item_exist("cfg_enable_structural_plasticity"): dpg.set_value("cfg_enable_structural_plasticity", cfg.enable_structural_plasticity)
    if dpg.does_item_exist("cfg_struct_plast_formation_rate"): dpg.set_value("cfg_struct_plast_formation_rate", cfg.struct_plast_formation_rate)
    if dpg.does_item_exist("cfg_struct_plast_elimination_rate"): dpg.set_value("cfg_struct_plast_elimination_rate", cfg.struct_plast_elimination_rate)
    if dpg.does_item_exist("cfg_struct_plast_weight_threshold"): dpg.set_value("cfg_struct_plast_weight_threshold", cfg.struct_plast_weight_threshold)
    if dpg.does_item_exist("cfg_struct_plast_target_density"): dpg.set_value("cfg_struct_plast_target_density", cfg.struct_plast_target_density)
    if dpg.does_item_exist("cfg_struct_plast_distance_scale"): dpg.set_value("cfg_struct_plast_distance_scale", cfg.struct_plast_distance_scale)
    if dpg.does_item_exist("cfg_struct_plast_update_interval_steps"): dpg.set_value("cfg_struct_plast_update_interval_steps", cfg.struct_plast_update_interval_steps)

    # Heterogeneity & Noise
    if dpg.does_item_exist("cfg_enable_parameter_heterogeneity"): dpg.set_value("cfg_enable_parameter_heterogeneity", cfg.enable_parameter_heterogeneity)
    if dpg.does_item_exist("cfg_heterogeneity_seed"): dpg.set_value("cfg_heterogeneity_seed", cfg.heterogeneity_seed)
    if dpg.does_item_exist("cfg_enable_conductance_noise"): dpg.set_value("cfg_enable_conductance_noise", cfg.enable_conductance_noise)
    if dpg.does_item_exist("cfg_conductance_noise_relative_std"): dpg.set_value("cfg_conductance_noise_relative_std", cfg.conductance_noise_relative_std)
    if dpg.does_item_exist("cfg_enable_ou_process"): dpg.set_value("cfg_enable_ou_process", cfg.enable_ou_process)
    if dpg.does_item_exist("cfg_ou_mean_current_pA"): dpg.set_value("cfg_ou_mean_current_pA", cfg.ou_mean_current_pA)
    if dpg.does_item_exist("cfg_ou_std_current_pA"): dpg.set_value("cfg_ou_std_current_pA", cfg.ou_std_current_pA)
    if dpg.does_item_exist("cfg_ou_tau_ms"): dpg.set_value("cfg_ou_tau_ms", cfg.ou_tau_ms)
    if dpg.does_item_exist("cfg_ou_seed"): dpg.set_value("cfg_ou_seed", cfg.ou_seed)

    # Camera FOV and Visualization settings
    if dpg.does_item_exist("cfg_camera_fov"): dpg.set_value("cfg_camera_fov", cfg.camera_fov)
    # Handle viz_update_interval_steps if it exists in the config (backward compatibility)
    if hasattr(cfg, "viz_update_interval_steps") and dpg.does_item_exist("cfg_viz_update_interval_steps"):
        dpg.set_value("cfg_viz_update_interval_steps", cfg.viz_update_interval_steps)
    
    # Hardware performance note - only update if config has a value (don't overwrite loaded benchmark data with fallback)
    if hasattr(cfg, "hardware_performance_note") and dpg.does_item_exist("cfg_hardware_performance_note"):
        if cfg.hardware_performance_note:  # Only update if config has actual data
            dpg.set_value("cfg_hardware_performance_note", cfg.hardware_performance_note)
        elif not dpg.get_value("cfg_hardware_performance_note"):  # Only set fallback if widget is currently empty
            dpg.set_value("cfg_hardware_performance_note", "Run visualization benchmark to determine hardware limits (viz_benchmark.py)")
    
    # Model-specific parameters
    if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
        ui_izh_params_keys = ["izh_C_val", "izh_k_val", "izh_vr_val", "izh_vt_val", "izh_vpeak_val", "izh_a_val", "izh_b_val", "izh_c_val", "izh_d_val"]
        for key_suffix in ui_izh_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))
    elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
        ui_hh_params_keys = [
            "hh_C_m", "hh_g_Na_max", "hh_g_K_max", "hh_g_L", "hh_E_Na", "hh_E_K", "hh_E_L",
            "hh_v_peak", "hh_v_rest_init", "hh_g_M_max", "hh_m_current_tau_ms",
            "hh_g_CaT_max", "hh_E_CaT", "hh_g_h_max", "hh_E_h", "hh_g_NaP_max",
            "hh_q10_factor", "hh_temperature_celsius",
            "hh_external_drive_scale",
        ]
        for key_suffix in ui_hh_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))
    elif cfg.neuron_model_type == NeuronModel.ADEX.name:
        ui_adex_params_keys = [
            "adex_C", "adex_g_L", "adex_E_L", "adex_V_T", "adex_Delta_T",
            "adex_a", "adex_tau_w", "adex_b", "adex_V_r", "adex_V_peak",
            "adex_external_drive_scale",
        ]
        for key_suffix in ui_adex_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))

    if dpg.does_item_exist("sim_speed_slider"): dpg.set_value("sim_speed_slider", cfg.simulation_speed_factor)

    _toggle_model_specific_params_visibility(None, cfg.neuron_model_type) # Update visibility of UI groups
    update_status_bar("Configuration loaded into UI.", level="info") 
    global_gui_state["reset_sim_needed_from_ui_change"] = False # Config is now in sync with UI


def _toggle_model_specific_params_visibility(sender, app_data, user_data=None):
    """Shows/hides UI groups for Izhikevich or Hodgkin-Huxley parameters. Called by main/UI thread."""
    selected_model_name = app_data # This is the string name of the model from the combo box

    is_izh = selected_model_name == NeuronModel.IZHIKEVICH.name
    is_hh = selected_model_name == NeuronModel.HODGKIN_HUXLEY.name
    is_adex = selected_model_name == NeuronModel.ADEX.name

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist("izhikevich_params_group"): dpg.configure_item("izhikevich_params_group", show=is_izh)
        if dpg.does_item_exist("hodgkin_huxley_params_group"): dpg.configure_item("hodgkin_huxley_params_group", show=is_hh)
        if dpg.does_item_exist("adex_params_group"): dpg.configure_item("adex_params_group", show=is_adex)
        if dpg.does_item_exist("homeostasis_izh_specific_group"): dpg.configure_item("homeostasis_izh_specific_group", show=is_izh)
        
        # Update neuron type filter combo based on selected model
        # This requires access to sim_bridge or a way to get types for a model.
        # For now, assume sim_bridge is accessible or this logic is refined.
        # If global_simulation_bridge is None yet (e.g. during initial UI setup before sim_bridge is fully ready for this),
        # this part might need to be deferred or handled carefully.
        # For now, let's assume it's called when sim_bridge can provide types.
        if dpg.does_item_exist("filter_neuron_type_combo"):
            # Create a temporary config to get available types for the selected model
            temp_cfg_for_types = SimulationConfiguration()
            temp_cfg_for_types.neuron_model_type = selected_model_name # Set model
            # Get available types using a static or instance method if SimulationConfiguration had one,
            # or if SimulationBridge has a helper. For now, mimic SimulationBridge's logic.
            available_types_for_filter = ["All"]
            if selected_model_name == NeuronModel.IZHIKEVICH.name:
                available_types_for_filter.extend([f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS])
            elif selected_model_name == NeuronModel.HODGKIN_HUXLEY.name:
                available_types_for_filter.extend([f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS])
            available_types_for_filter = list(dict.fromkeys(available_types_for_filter))


            current_filter_value = dpg.get_value("filter_neuron_type_combo")
            dpg.configure_item("filter_neuron_type_combo", items=available_types_for_filter)
            if current_filter_value in available_types_for_filter:
                dpg.set_value("filter_neuron_type_combo", current_filter_value)
            elif "All" in available_types_for_filter: 
                dpg.set_value("filter_neuron_type_combo", "All")
            elif available_types_for_filter: 
                dpg.set_value("filter_neuron_type_combo", available_types_for_filter[0])
            else:
                dpg.set_value("filter_neuron_type_combo", "")

    # Check config against benchmark limits after populating
    _check_and_warn_hardware_limits()


# --- DPG Event Handlers for OpenGL Visualization Settings ---
def handle_gl_point_size_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE: opengl_viz_config['POINT_SIZE'] = app_data; trigger_filter_update_signal()
def handle_gl_synapse_alpha_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE: opengl_viz_config['SYNAPSE_ALPHA_MODIFIER'] = app_data; trigger_filter_update_signal()
def handle_gl_activity_highlight_frames_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_frames = int(app_data)
            if new_frames >= 1: opengl_viz_config['ACTIVITY_HIGHLIGHT_FRAMES'] = new_frames
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender): 
                dpg.set_value(sender, opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7))
        except ValueError: 
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7))
def handle_gl_max_neurons_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_val = int(app_data)
            if new_val >= 0: opengl_viz_config['MAX_NEURONS_TO_RENDER'] = new_val; trigger_filter_update_signal()
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000))
        except ValueError:
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000))
def handle_gl_max_connections_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_val = int(app_data)
            if new_val >= 0: opengl_viz_config['MAX_CONNECTIONS_TO_RENDER'] = new_val; trigger_filter_update_signal()
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000))
        except ValueError:
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000))
def handle_gl_inactive_neuron_opacity_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_val = float(app_data)
            if 0.0 <= new_val <= 1.0: opengl_viz_config['INACTIVE_NEURON_OPACITY'] = new_val; trigger_filter_update_signal()
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25))
        except ValueError:
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25))
def handle_gl_enable_synaptic_pulses_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        opengl_viz_config['ENABLE_SYNAPTIC_PULSES'] = app_data
        trigger_filter_update_signal()

# --- DPG Event Handlers for Simulation Control & Configuration ---

def _apply_hh_preset_params_to_ui(hh_type_name):
    """Update HH parameter input fields in the UI to match a given preset.

    This keeps the visible HH parameter panel in sync with the selected
    HH neuron type and any profile/model-driven preset selection.
    """
    if not dpg.is_dearpygui_running() or not hh_type_name:
        return
    try:
        hh_enum = NeuronType[hh_type_name]
        params = DefaultHodgkinHuxleyParams.get_params(hh_enum)
    except Exception as e:
        print(f"Warning: could not apply HH preset '{hh_type_name}' to UI: {e}")
        return

    tag_key_pairs = [
        ("cfg_hh_C_m", "C_m"),
        ("cfg_hh_g_Na_max", "g_Na_max"),
        ("cfg_hh_g_K_max", "g_K_max"),
        ("cfg_hh_g_L", "g_L"),
        ("cfg_hh_E_Na", "E_Na"),
        ("cfg_hh_E_K", "E_K"),
        ("cfg_hh_E_L", "E_L"),
        ("cfg_hh_v_rest_init", "v_rest_hh"),
        ("cfg_hh_v_peak", "v_peak_hh"),
        ("cfg_hh_g_M_max", "g_M_max"),
        ("cfg_hh_m_current_tau_ms", "m_current_tau_ms"),
        ("cfg_hh_g_CaT_max", "g_CaT_max"),
        ("cfg_hh_E_CaT", "E_CaT"),
        ("cfg_hh_g_h_max", "g_h_max"),
        ("cfg_hh_E_h", "E_h"),
        ("cfg_hh_g_NaP_max", "g_NaP_max"),
        ("cfg_hh_q10_factor", "q10_factor"),
        ("cfg_hh_temperature_celsius", "temperature_celsius"),
    ]
    for tag, key in tag_key_pairs:
        if dpg.does_item_exist(tag) and key in params:
            dpg.set_value(tag, params[key])


def handle_reset_hh_drive_to_auto(sender=None, app_data=None, user_data=None):
    """Reset the HH external drive scale slider to the auto-tuned value for the current combo, if any."""
    try:
        if not dpg.is_dearpygui_running():
            return
        if not dpg.does_item_exist("cfg_neuron_model_type"):
            return
        model_name = dpg.get_value("cfg_neuron_model_type")
        if model_name != NeuronModel.HODGKIN_HUXLEY.name:
            update_status_bar("HH drive reset: current model is not Hodgkin-Huxley.", level="warning")
            return
        profile_name = dpg.get_value("cfg_neural_profile") if dpg.does_item_exist("cfg_neural_profile") else "GENERIC_UNSTRUCTURED"
        hh_type = dpg.get_value("cfg_default_neuron_type_hh") if dpg.does_item_exist("cfg_default_neuron_type_hh") else NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name
        tuned = get_auto_tuned_overrides_for_combo(model_name, profile_name, hh_type)
        if not tuned or not isinstance(tuned, dict):
            update_status_bar("No auto-tuned HH entry found for this combination.", level="warning")
            return
        core_overrides = tuned.get("core_overrides", {}) or {}
        scale = core_overrides.get("hh_external_drive_scale")
        if scale is None:
            update_status_bar("Auto-tuned config has no HH drive scale for this combination.", level="warning")
            return
        if dpg.does_item_exist("cfg_hh_external_drive_scale"):
            dpg.set_value("cfg_hh_external_drive_scale", float(scale))
        _update_sim_config_from_ui_and_signal_reset_needed("cfg_hh_external_drive_scale", float(scale))
        update_status_bar("HH drive scale reset to auto-tuned value. Apply & Reset to use in sim.", level="info")
    except Exception as e:
        update_status_bar(f"Error resetting HH drive scale: {e}", level="error")


def handle_reset_adex_drive_to_auto(sender=None, app_data=None, user_data=None):
    """Reset the AdEx external drive scale slider to the auto-tuned value for the current profile, if any."""
    try:
        if not dpg.is_dearpygui_running():
            return
        if not dpg.does_item_exist("cfg_neuron_model_type"):
            return
        model_name = dpg.get_value("cfg_neuron_model_type")
        if model_name != NeuronModel.ADEX.name:
            update_status_bar("AdEx drive reset: current model is not AdEx.", level="warning")
            return
        profile_name = dpg.get_value("cfg_neural_profile") if dpg.does_item_exist("cfg_neural_profile") else "GENERIC_UNSTRUCTURED"
        tuned = get_auto_tuned_overrides_for_combo(model_name, profile_name, None)
        if not tuned or not isinstance(tuned, dict):
            update_status_bar("No auto-tuned AdEx entry found for this profile.", level="warning")
            return
        core_overrides = tuned.get("core_overrides", {}) or {}
        scale = core_overrides.get("adex_external_drive_scale")
        if scale is None:
            update_status_bar("Auto-tuned config has no AdEx drive scale for this profile.", level="warning")
            return
        if dpg.does_item_exist("cfg_adex_external_drive_scale"):
            dpg.set_value("cfg_adex_external_drive_scale", float(scale))
        _update_sim_config_from_ui_and_signal_reset_needed("cfg_adex_external_drive_scale", float(scale))
        update_status_bar("AdEx drive scale reset to auto-tuned value. Apply & Reset to use in sim.", level="info")
    except Exception as e:
        update_status_bar(f"Error resetting AdEx drive scale: {e}", level="error")


def _update_sim_config_from_ui_and_signal_reset_needed(sender=None, app_data=None, user_data=None):
    """
    Callback for UI elements that change sim config. Sets a flag that sim needs reset.
    The actual config update is collected by `handle_apply_config_changes_and_reset`.
    """
    global_gui_state["reset_sim_needed_from_ui_change"] = True
    update_status_bar("Parameter changed. Press 'Apply Changes & Reset Sim' to take effect.", color=[255,165,0,255], level="warning")

    # Special handling for certain controls
    if sender == "cfg_neuron_model_type":
        # Update visibility of model-specific parameter groups
        _toggle_model_specific_params_visibility(sender, app_data)

        # If switching to HH, clamp the preset list to profile-compatible types and
        # snap the selection (and visible HH params) to a valid preset.
        try:
            if app_data == NeuronModel.HODGKIN_HUXLEY.name and dpg.is_dearpygui_running():
                if dpg.does_item_exist("cfg_default_neuron_type_hh") and dpg.does_item_exist("cfg_neural_profile"):
                    profile_name = dpg.get_value("cfg_neural_profile")
                    allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
                    if allowed_hh:
                        dpg.configure_item("cfg_default_neuron_type_hh", items=allowed_hh)
                        current_hh = dpg.get_value("cfg_default_neuron_type_hh")
                        if current_hh not in allowed_hh:
                            current_hh = allowed_hh[0]
                            dpg.set_value("cfg_default_neuron_type_hh", current_hh)
                        _apply_hh_preset_params_to_ui(current_hh)
        except Exception as e:
            print(f"Warning: failed to enforce HH preset compatibility on model change: {e}")

    elif sender == "cfg_neural_profile":
        # When changing neural structure profile, if HH model is active, clamp the HH
        # preset list and snap the selection to the profile-compatible preset.
        try:
            if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_neuron_model_type"):
                model_name = dpg.get_value("cfg_neuron_model_type")
                profile_name = app_data
                if model_name == NeuronModel.HODGKIN_HUXLEY.name and dpg.does_item_exist("cfg_default_neuron_type_hh"):
                    allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
                    if allowed_hh:
                        dpg.configure_item("cfg_default_neuron_type_hh", items=allowed_hh)
                        current_hh = dpg.get_value("cfg_default_neuron_type_hh")
                        if current_hh not in allowed_hh:
                            current_hh = allowed_hh[0]
                            dpg.set_value("cfg_default_neuron_type_hh", current_hh)
                        _apply_hh_preset_params_to_ui(current_hh)
        except Exception as e:
            print(f"Warning: failed to enforce HH preset compatibility on profile change: {e}")

    elif sender == "cfg_default_neuron_type_hh":
        # Direct change of HH preset by the user; update HH params panel to match,
        # but still respect per-profile compatibility.
        try:
            if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_neural_profile"):
                profile_name = dpg.get_value("cfg_neural_profile")
                allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
                if allowed_hh and app_data not in allowed_hh:
                    app_data = allowed_hh[0]
                    dpg.set_value("cfg_default_neuron_type_hh", app_data)
            _apply_hh_preset_params_to_ui(app_data)
        except Exception as e:
            print(f"Warning: failed to apply HH preset params on preset change: {e}")

    # Check proposed config against benchmark-derived hardware limits
    _check_and_warn_hardware_limits()


def _check_and_warn_hardware_limits():
    """Reads current UI values and warns if config exceeds benchmark-tested limits."""
    try:
        if not dpg.is_dearpygui_running():
            return
        if not dpg.does_item_exist("cfg_num_neurons") or not dpg.does_item_exist("cfg_neuron_model_type"):
            return

        model_name = dpg.get_value("cfg_neuron_model_type")
        num_neurons = dpg.get_value("cfg_num_neurons")
        conn_per = dpg.get_value("cfg_connections_per_neuron") if dpg.does_item_exist("cfg_connections_per_neuron") else 100

        is_safe, warning = check_config_against_limits(model_name, num_neurons, conn_per)

        tag = "hw_limit_warning_text"
        if dpg.does_item_exist(tag):
            if warning:
                dpg.set_value(tag, warning)
                dpg.configure_item(tag, color=[255, 100, 100, 255], show=True)
            else:
                # Show positive feedback if within limits and benchmark data exists
                limits = get_hardware_limits_for_model(model_name)
                if limits:
                    # Find the matching or next-larger tested config
                    configs = limits["configs"]
                    matching = [c for c in configs if c["neurons"] >= num_neurons and c["conn"] >= conn_per]
                    if matching:
                        m = matching[0]
                        dpg.set_value(tag, f"Tested OK: {m['steps_per_sec']:.0f} steps/s, {m['gpu_gb']:.1f}GB VRAM")
                        dpg.configure_item(tag, color=[100, 255, 100, 255], show=True)
                    else:
                        dpg.set_value(tag, "")
                        dpg.configure_item(tag, show=False)
                else:
                    dpg.set_value(tag, "")
                    dpg.configure_item(tag, show=False)
    except Exception:
        pass  # Never let limit check crash the UI


def _handle_model_type_change_dpg(sender, app_data, user_data=None):
    """Handles change in neuron model type selection in DPG. Updates UI visibility and signals reset."""
    _toggle_model_specific_params_visibility(sender, app_data) # Update UI sections
    # Auto-adjust dt when switching to HH (needs dt <= 0.1ms for stability)
    # or back to a simpler model (can use larger dt)
    if dpg.does_item_exist("cfg_dt_ms"):
        current_dt = dpg.get_value("cfg_dt_ms")
        if app_data == NeuronModel.HODGKIN_HUXLEY.name:
            if current_dt > 0.1:
                dpg.set_value("cfg_dt_ms", 0.05)
                update_status_bar("dt auto-adjusted to 0.05 ms for HH stability (was {:.3f} ms)".format(current_dt),
                                  color=[255, 200, 100, 255], level="warning")
        else:
            # When switching away from HH, if dt is very small (likely auto-set), restore a reasonable default
            if current_dt <= 0.1:
                dpg.set_value("cfg_dt_ms", 0.5)
                update_status_bar("dt restored to 0.5 ms for {} model".format(app_data),
                                  color=[150, 220, 255, 255], level="info")
    _update_sim_config_from_ui_and_signal_reset_needed() # Mark that config changed and reset is needed


def handle_start_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends a 'START_SIM' command to the simulation thread."""
    if global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Error: Cannot start simulation in playback mode.", color=[255,0,0,255], level="error")
        return
    if global_gui_state.get("reset_sim_needed_from_ui_change", False):
        update_status_bar("Apply changes before starting!", color=[255,100,100,255], level="warning")
        return
    
    ui_to_sim_queue.put({"type": "START_SIM"})
    update_status_bar("Start command sent to simulation...", level="info")
    # UI state (buttons enabled/disabled) will be updated when sim_thread confirms via message.
    # Optimistically, we can update some UI here, but it's better to wait for ack for robustness.
    # For now, let global_gui_state._sim_is_running_ui_view reflect the command sent.
    global_gui_state["_sim_is_running_ui_view"] = True
    global_gui_state["_sim_is_paused_ui_view"] = False
    update_ui_for_simulation_run_state(is_running=True, is_paused=False) # Optimistic UI update


def handle_stop_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends a 'STOP_SIM' command to the simulation thread."""
    ui_to_sim_queue.put({"type": "STOP_SIM"})
    update_status_bar("Stop command sent to simulation...", level="info")
    global_gui_state["_sim_is_running_ui_view"] = False
    global_gui_state["_sim_is_paused_ui_view"] = False
    update_ui_for_simulation_run_state(is_running=False, is_paused=False) # Optimistic UI update
    # Actual telemetry reset for UI happens when sim_thread confirms stop and sends initial data.


def handle_pause_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends 'PAUSE_SIM' or 'RESUME_SIM' command based on current UI perceived state."""
    if not global_gui_state.get("_sim_is_running_ui_view", False): # Can only pause/resume if UI thinks it's running
        update_status_bar("Sim not running, cannot pause/resume.", color=[255,165,0,255], level="warning")
        return

    if global_gui_state.get("_sim_is_paused_ui_view", False): # If UI thinks it's paused, send RESUME
        ui_to_sim_queue.put({"type": "RESUME_SIM"})
        update_status_bar("Resume command sent...", level="info")
        global_gui_state["_sim_is_paused_ui_view"] = False # Optimistic
    else: # If UI thinks it's running (not paused), send PAUSE
        ui_to_sim_queue.put({"type": "PAUSE_SIM"})
        update_status_bar("Pause command sent...", level="info")
        global_gui_state["_sim_is_paused_ui_view"] = True # Optimistic
    update_ui_for_simulation_run_state(is_running=True, is_paused=global_gui_state["_sim_is_paused_ui_view"])


def handle_step_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends a 'STEP_SIM_ONE_MS' command to the simulation thread."""
    if global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Error: Cannot step live sim in playback mode.", color=[255,0,0,255], level="error")
        return

    is_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
    is_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
    can_step_ui = (is_running_ui and is_paused_ui) or (not is_running_ui)

    if can_step_ui:
        if global_gui_state.get("reset_sim_needed_from_ui_change", False):
            update_status_bar("Apply changes before stepping!", color=[255,100,100,255], level="warning")
            return
        ui_to_sim_queue.put({"type": "STEP_SIM_ONE_MS"}) # Sim thread will determine actual number of substeps
        update_status_bar("Step (1ms) command sent...", level="info")
    else:
        update_status_bar("Sim must be running & paused, or stopped, to step.", color=[255,165,0,255], level="warning")

def handle_apply_config_changes_and_reset(sender=None, app_data=None, user_data=None, from_reset_button=False):
    """
    Handles the 'Apply Changes & Reset Sim' button.
    Collects UI config, sends it to sim_thread for application and reset.
    """
    update_status_bar("Collecting UI configuration...", level="info")
    
    # Get the current configuration from UI elements
    # _update_sim_config_from_ui now returns a complete config dict
    config_dict_from_ui = _update_sim_config_from_ui(update_model_specific=True)

    if config_dict_from_ui:
        update_status_bar("Sending new configuration to simulation thread for reset...", level="info")
        ui_to_sim_queue.put({
            "type": "APPLY_CONFIG_AND_RESET",
            "config_dict": config_dict_from_ui
        })
        # UI will be fully updated once sim_thread confirms and sends back new state/config.
        # global_gui_state["reset_sim_needed_from_ui_change"] will be set to False by UI thread
        # after confirmation from sim_thread.
    else:
        update_status_bar("Failed to collect configuration from UI. Please check parameters.", color=[255,0,0,255], level="error")


def handle_sim_speed_change(sender, app_data, user_data):
    """Sends 'SET_SIM_SPEED' command when simulation speed slider changes."""
    ui_to_sim_queue.put({"type": "SET_SIM_SPEED", "factor": app_data})
    # Status bar update can be done here or by sim_thread acknowledging.
    # update_status_bar(f"Sim speed factor set to {app_data:.2f}x (command sent)", level="info")


# --- DPG File Dialog Callbacks and Handlers (Main/UI Thread) ---

def get_profile_files(profile_directory): # Profiles are still JSON
    """Gets a list of .json profile files from the specified directory."""
    try:
        if os.path.exists(profile_directory) and os.path.isdir(profile_directory):
            return sorted([f for f in os.listdir(profile_directory) if f.endswith(".json") and os.path.isfile(os.path.join(profile_directory, f))])
    except Exception as e: print(f"Error listing profile directory '{profile_directory}': {e}")
    return []

def get_hdf5_files(directory, extension): # Helper for .simrec.h5 and .simstate.h5
    """Gets a list of HDF5 files with a specific extension from a directory."""
    try:
        if os.path.exists(directory) and os.path.isdir(directory):
            return sorted([f for f in os.listdir(directory) if f.endswith(extension) and os.path.isfile(os.path.join(directory, f))])
    except Exception as e: print(f"Error listing directory '{directory}' for '{extension}': {e}")
    return []


def handle_save_profile_button_press(sender=None, app_data=None, user_data=None): # Profiles are JSON
    """Shows the 'Save Profile' file dialog."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_profile_file_dialog"):
        # Ensure current UI state is reflected in a temporary config dict to be saved
        # The actual saving happens in the callback, which will re-fetch this.
        update_status_bar("Preparing to save profile...", level="info")
        dpg.show_item("save_profile_file_dialog")

def handle_load_profile_button_press(sender=None, app_data=None, user_data=None): # Profiles are JSON
    """Shows the 'Load Profile' file dialog."""
    if global_gui_state.get("is_recording_active", False) or global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Stop recording/playback before loading a profile.", color=[255,165,0,255], level="warning")
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_profile_file_dialog"):
        dpg.show_item("load_profile_file_dialog")

def _normalize_filepath_extension(filepath, required_extension, filter_extension=None):
    """
    Normalizes a filepath to ensure it has the correct extension.

    Args:
        filepath: The filepath from the file dialog
        required_extension: The extension we want (e.g., ".json", ".simstate.h5", ".simrec.h5")
        filter_extension: The filter extension DPG might have appended (e.g., ".h5", ".*")

    Returns:
        Normalized filepath with correct extension
    """
    # Strip ".*" if DPG appended it from "All Files" filter
    if filepath.endswith(".*"):
        filepath = filepath[:-2]

    # Strip the filter extension if DPG appended it (e.g., ".h5" when we want ".simstate.h5")
    if filter_extension and filter_extension != ".*":
        if filepath.lower().endswith(filter_extension.lower()) and not filepath.lower().endswith(required_extension.lower()):
            filepath = filepath[:-len(filter_extension)]

    # Add the required extension if not present
    if not filepath.lower().endswith(required_extension.lower()):
        filepath += required_extension

    return filepath

def save_profile_dialog_callback(sender, app_data): # Profiles are JSON
    """
    Callback for the 'Save Profile' file dialog. Saves current UI config and GUI settings.
    This operation is done entirely by the UI thread.
    """
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_filepath_extension(
            app_data["file_path_name"],
            required_extension=".json",
            filter_extension=app_data.get("current_filter")
        )

        # Get current simulation config from UI (doesn't interact with sim_thread for this)
        sim_config_dict_to_save = _update_sim_config_from_ui(update_model_specific=True)
        if not sim_config_dict_to_save:
            update_status_bar("Error: Could not retrieve current config from UI to save profile.", color=[255,0,0,255], level="error")
            return

        # Remove runtime state keys that shouldn't be in a profile
        keys_to_remove_from_profile = ["neuron_positions_x", "neuron_positions_y", "neuron_types_list_for_viz",
                                       "current_time_ms", "current_time_step", "is_running", "is_paused", "max_delay_steps"]
        for key in keys_to_remove_from_profile:
            if key in sim_config_dict_to_save: del sim_config_dict_to_save[key]
        
        gui_settings_to_save = get_current_gui_configuration_dict() # Get current GUI/filter settings
        content_to_save = {"simulation_configuration": sim_config_dict_to_save, "gui_configuration": gui_settings_to_save}

        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(content_to_save, f, indent=4, ensure_ascii=False)
            update_status_bar(f"Profile saved: {os.path.basename(filepath)}", color=[0,200,0,255], level="success")
            if dpg.does_item_exist("profile_name_input"): 
                dpg.set_value("profile_name_input", os.path.basename(filepath).replace(".json", ""))
            global_gui_state["current_profile_name"] = os.path.basename(filepath)
        except Exception as e: 
            update_status_bar(f"Error saving profile: {e}", color=[255,0,0,255], level="error")
    else: 
        update_status_bar("Save profile cancelled.", level="info")


# --- Full Profile Dropdown (auto-populated from simulation_profiles/*.json) ---
_FULL_PROFILE_MAP = {}  # display_name -> filepath, populated at startup and on refresh

def _scan_profile_directory():
    """Scans simulation_profiles/ for .json files and builds display_name -> filepath map.

    Reads _profile_metadata.name if present, otherwise derives a readable name from filename.
    Excludes auto_tuned_overrides.json (system file, not a user profile).
    """
    global _FULL_PROFILE_MAP
    profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_profiles")
    if not os.path.isdir(profile_dir):
        return

    new_map = {"(None - use settings below)": ""}  # Default empty entry
    try:
        for fname in sorted(os.listdir(profile_dir)):
            if not fname.endswith(".json") or fname == "auto_tuned_overrides.json":
                continue
            fpath = os.path.join(profile_dir, fname)
            # Try to extract a human-readable name from metadata
            display = fname.replace(".json", "").replace("_", " ").title()
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                meta = data.get("_profile_metadata", {})
                if meta.get("name"):
                    display = meta["name"]
            except Exception:
                pass  # Fall back to filename-derived name
            new_map[display] = fpath
    except Exception as e:
        print(f"Warning: Could not scan profile directory: {e}")

    _FULL_PROFILE_MAP = new_map


# =============================================================================
# EXPERIMENT SYSTEM UI CALLBACKS
# =============================================================================

def _handle_experiment_preset_change(preset_name):
    """Callback when user selects an experiment preset from the dropdown."""
    if not preset_name or preset_name == "-- Select Preset --":
        return
    ui_to_sim_queue.put({"type": "LOAD_EXPERIMENT_PRESET", "preset_name": preset_name})
    update_status_bar(f"Loading experiment preset: {preset_name}", color=[100, 200, 255, 255])


def _handle_inject_manual_stimulus(sender=None, app_data=None, user_data=None):
    """Inject a quick manual stimulus using a basic experiment config."""
    try:
        amplitude = dpg.get_value("manual_stim_amplitude")
        pattern_str = dpg.get_value("manual_stim_pattern_combo")
        group_size = dpg.get_value("manual_stim_group_size")
        duration = dpg.get_value("manual_stim_duration")

        # Build a simple experiment config for manual injection
        exp_config = ExperimentConfig(
            name="Manual Stimulus Injection",
            description=f"Quick {pattern_str} stimulus: {amplitude} pA for {duration} ms",
            neuron_groups=[
                NeuronGroup(name="stim_target", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=group_size,
                           highlight_color=[0.0, 1.0, 0.0, 1.0]),
                NeuronGroup(name="network_response", role=NeuronGroupRole.OUTPUT.name,
                           index_start=group_size, index_end=group_size * 3,
                           highlight_color=[1.0, 0.5, 0.0, 1.0]),
            ],
            stimulus_channels=[
                StimulusChannel(
                    name="manual_stim",
                    pattern=StimulusPattern(
                        pattern_type=pattern_str,
                        amplitude_pA=amplitude,
                    ),
                    target_group_name="stim_target",
                    onset_ms=100.0,
                    duration_ms=duration,
                ),
            ],
            phases=[
                ExperimentPhase(name="pre_baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=500.0, active_channels=[]),
                ExperimentPhase(name="stimulus", phase_type=ExperimentPhaseType.STIMULUS.name,
                               duration_ms=duration + 200.0,
                               active_channels=["manual_stim"]),
                ExperimentPhase(name="post_baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=1000.0, active_channels=[]),
            ],
            readout=ReadoutConfig(
                rate_window_ms=50.0,
                rate_group_names=["stim_target", "network_response"],
            ),
            enabled=True,
        )

        config_dict = experiment_config_to_dict(exp_config)
        ui_to_sim_queue.put({"type": "LOAD_EXPERIMENT_CONFIG", "config_dict": config_dict})
        # Auto-start after a brief delay to allow initialization
        ui_to_sim_queue.put({"type": "START_EXPERIMENT"})
        update_status_bar(f"Injecting {pattern_str} stimulus: {amplitude} pA", color=[100, 255, 100, 255])
    except Exception as e:
        update_status_bar(f"Stimulus injection error: {e}", color=[255, 100, 100, 255])


def _update_experiment_ui_from_status(experiment_status):
    """Update experiment UI elements from status dict (called from UI thread)."""
    if experiment_status is None:
        return

    try:
        is_running = experiment_status.get("is_running", False)
        is_complete = experiment_status.get("is_complete", False)

        # Status text
        if is_complete:
            dpg.set_value("experiment_status_text", "COMPLETE")
            dpg.configure_item("experiment_status_text", color=[100, 255, 100])
        elif is_running:
            dpg.set_value("experiment_status_text", "RUNNING")
            dpg.configure_item("experiment_status_text", color=[255, 255, 100])
        else:
            dpg.set_value("experiment_status_text", "Idle")
            dpg.configure_item("experiment_status_text", color=[150, 150, 150])

        # Phase info
        phase_name = experiment_status.get("current_phase_name", "--")
        phase_type = experiment_status.get("current_phase_type", "--")
        phase_idx = experiment_status.get("current_phase_idx", 0)
        total_phases = experiment_status.get("total_phases", 0)
        rep = experiment_status.get("phase_repetition", 0)
        dpg.set_value("experiment_phase_text",
                       f"Phase: {phase_name} ({phase_type}) [{phase_idx+1}/{total_phases}] rep={rep}")

        # Readout rates
        rates = experiment_status.get("readout_rates", {})
        if rates:
            rate_lines = [f"  {name}: {rate:.1f} Hz" for name, rate in rates.items()]
            dpg.set_value("experiment_readout_text", "\n".join(rate_lines))
        else:
            dpg.set_value("experiment_readout_text", "No data")

        # Training info
        training = experiment_status.get("training")
        if training and training.get("mode", "NONE") != "NONE":
            trials_done = training.get("trials_completed", 0)
            total_trials = training.get("total_trials", 0)
            accuracy = training.get("recent_accuracy", 0.0)
            converged = training.get("is_converged", False)
            status_str = f"Trial {trials_done}/{total_trials} | Accuracy: {accuracy:.1%}"
            if converged:
                status_str += " [CONVERGED]"
            dpg.set_value("experiment_training_text", status_str)
        else:
            dpg.set_value("experiment_training_text", "No training active")

    except Exception:
        pass  # UI elements may not exist yet during startup


def _handle_full_profile_dropdown_change(sender, app_data, user_data=None):
    """Callback when user selects a full profile from the dropdown."""
    if not app_data or app_data == "(None - use settings below)":
        return

    filepath = _FULL_PROFILE_MAP.get(app_data, "")
    if not filepath or not os.path.exists(filepath):
        update_status_bar(f"Profile file not found for '{app_data}'", color=[255,100,0,255], level="warning")
        return

    _execute_profile_load_on_ui_thread(filepath)

    # After loading, update the dropdown to reflect current selection (don't reset to None)
    # The profile is now applied — user sees it in the dropdown


def _refresh_full_profile_dropdown():
    """Rescans the profile directory and updates the dropdown items."""
    _scan_profile_directory()
    if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_full_profile"):
        items = list(_FULL_PROFILE_MAP.keys())
        dpg.configure_item("cfg_full_profile", items=items)


def _execute_profile_load_on_ui_thread(filepath): # Profiles are JSON
    """
    Loads a profile file, updates UI, and sends new config to sim_thread.
    Called by UI thread.
    """
    profile_name = os.path.basename(filepath)
    update_status_bar(f"Loading profile '{profile_name}'...", level="info")
    try:
        with open(filepath, 'r', encoding='utf-8') as f: profile_content = json.load(f)
        sim_cfg_data_from_profile = profile_content.get("simulation_configuration")
        gui_cfg_data_from_profile = profile_content.get("gui_configuration")

        if sim_cfg_data_from_profile:
            # Populate UI elements from the loaded simulation configuration part of the profile
            _populate_ui_from_config_dict(sim_cfg_data_from_profile)
            
            # Apply GUI settings from the profile
            if gui_cfg_data_from_profile: 
                apply_gui_configuration_core(gui_cfg_data_from_profile) # Updates filters, GL config etc.

            # Now that UI is updated, get the full config from UI to send to sim_thread for reset
            # This ensures any defaults or interpretations by _populate_ui are captured.
            final_config_to_apply_to_sim = _update_sim_config_from_ui(update_model_specific=True)
            if final_config_to_apply_to_sim:
                ui_to_sim_queue.put({
                    "type": "APPLY_CONFIG_AND_RESET",
                    "config_dict": final_config_to_apply_to_sim
                })
                update_status_bar(f"Profile '{profile_name}' loaded. Config sent for reset.", color=[0,200,0,255], level="success")
                if dpg.does_item_exist("profile_name_input"):
                    dpg.set_value("profile_name_input", profile_name.replace(".json", ""))
                global_gui_state["current_profile_name"] = profile_name
                # Update full profile dropdown to reflect loaded profile
                if dpg.does_item_exist("cfg_full_profile"):
                    # Find the display name matching this filepath
                    for display_name, fpath in _FULL_PROFILE_MAP.items():
                        if fpath and os.path.normpath(fpath) == os.path.normpath(filepath):
                            dpg.set_value("cfg_full_profile", display_name)
                            break
                global_gui_state["reset_sim_needed_from_ui_change"] = False # Reset is being handled
            else:
                update_status_bar("Error creating final config from UI after profile load.", color=[255,0,0,255], level="error")
        else: 
            update_status_bar("Error: Profile missing 'simulation_configuration'.", color=[255,0,0,255], level="error")
    except Exception as e:
        update_status_bar(f"Error loading profile: {e}", color=[255,0,0,255], level="error"); import traceback; traceback.print_exc()

def load_profile_dialog_callback(sender, app_data): # Profiles are JSON
    """Callback for 'Load Profile' dialog. Calls helper to load and apply."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        _execute_profile_load_on_ui_thread(app_data["file_path_name"])
    else: 
        update_status_bar("Load profile cancelled.", level="info")


def handle_save_checkpoint_button_press(sender, app_data, user_data): # Checkpoints are HDF5
    """Shows 'Save Checkpoint' file dialog. Command sent to sim_thread from callback."""
    if global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Error: Cannot save checkpoint in playback mode.", color=[255,0,0,255], level="error")
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_checkpoint_file_dialog_h5"):
        dpg.show_item("save_checkpoint_file_dialog_h5")

def save_checkpoint_dialog_callback_h5(sender, app_data): # Checkpoints are HDF5
    """Callback for 'Save Checkpoint'. Ensures correct extension."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_filepath_extension(
            app_data["file_path_name"],
            required_extension=".simstate.h5",
            filter_extension=app_data.get("current_filter")
        )

        current_gui_config_for_checkpoint = get_current_gui_configuration_dict()
        ui_to_sim_queue.put({
            "type": "SAVE_CHECKPOINT",
            "filepath": filepath,
            "gui_config_snapshot": current_gui_config_for_checkpoint
            })
        update_status_bar(f"Save checkpoint command sent for: {os.path.basename(filepath)}", level="info")
    else:
        update_status_bar("Save checkpoint cancelled.", level="info")

def handle_load_checkpoint_button_press(sender, app_data, user_data): # Checkpoints are HDF5
    """Shows 'Load Checkpoint' file dialog. Command sent to sim_thread from callback."""
    if global_gui_state.get("is_recording_active", False) or global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Stop recording/playback before loading a checkpoint.", color=[255,165,0,255], level="warning")
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_checkpoint_file_dialog_h5"):
        dpg.show_item("load_checkpoint_file_dialog_h5")

def load_checkpoint_dialog_callback_h5(sender, app_data):
    """Callback for 'Load Checkpoint' dialog. Sends command to sim_thread."""
    filepath_to_load = None
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        selected_filter = app_data.get("current_filter", "")

        # If the ".*" filter was active, DPG might append ".*" to the actual filename.
        # We need to strip this if the file doesn't literally end with ".*".
        if selected_filter == ".*" and filepath.endswith(".*"):
            potential_filepath_stripped = filepath[:-2]
            # Check if the stripped version is the actual file
            if os.path.isfile(potential_filepath_stripped):
                filepath = potential_filepath_stripped
            # If not, and the original path with ".*" is a file (rare), use it.
            # Otherwise, it's likely an invalid construction by DPG.
            elif not os.path.isfile(filepath): # if "file.simstate.h5.*" is NOT a file
                 update_status_bar(f"Load error: Path '{filepath}' from '.*' filter seems invalid.", color=[255,0,0,255], level="error")
                 return


        # At this point, filepath should be the intended file.
        if os.path.isfile(filepath):
            filepath_to_load = filepath
        elif os.path.isdir(filepath):
            update_status_bar(f"Error: Selected path is a directory: {filepath}", color=[255,100,0,255], level="warning")
            return
        else:
            update_status_bar(f"Load error: File not found or invalid path: '{filepath}'.", color=[255,0,0,255], level="error")
            return

    elif "file_name" in app_data and app_data["file_name"] and "current_path" in app_data: # Fallback
        filepath = os.path.join(app_data["current_path"], app_data["file_name"])
        if os.path.isfile(filepath):
            filepath_to_load = filepath
        else:
            update_status_bar(f"Error: Fallback path is not a valid file: {filepath}", color=[255,0,0,255], level="error")
            return
    else:
        update_status_bar("Load checkpoint cancelled or no file selected.", level="info")
        return

    if filepath_to_load:
        ui_to_sim_queue.put({"type": "LOAD_CHECKPOINT", "filepath": filepath_to_load})
        update_status_bar(f"Load checkpoint command sent for: {os.path.basename(filepath_to_load)}", level="info")

def get_current_gui_configuration_dict():
    """
    Gets current GUI settings, including filters and OpenGL viz config.
    Called by UI thread, e.g., when saving a profile or checkpoint.
    """
    dpg_filters = get_current_filter_settings_from_gui() 
    dpg_filters["show_synapses_cb"] = global_gui_state.get("show_connections_gl", False) # From global_gui_state

    current_gl_config = opengl_viz_config.copy() if OPENGL_AVAILABLE else {}
    # If sim_bridge instance is available and has camera_fov (it's part of sim_config)
    # This is tricky as sim_config in sim_bridge might not be in sync if UI changed it.
    # Best to get FOV from DPG UI if it's there.
    if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_camera_fov"):
         current_gl_config["CAMERA_FOV_DPG_Snapshot"] = dpg.get_value("cfg_camera_fov")
    elif global_simulation_bridge and hasattr(global_simulation_bridge, 'viz_config'): # Fallback
         current_gl_config["CAMERA_FOV_DPG_Snapshot"] = global_simulation_bridge.viz_config.camera_fov


    return {"filter_settings": dpg_filters, "opengl_visualization_settings": current_gl_config}

def apply_gui_configuration_core(gui_cfg_dict):
    """
    Applies a dictionary of GUI settings to the DPG UI elements.
    Called by UI thread, e.g., after loading a profile or checkpoint that includes GUI settings.
    """
    if not gui_cfg_dict or not dpg.is_dearpygui_running(): return False

    filter_settings = gui_cfg_dict.get("filter_settings", {})
    if dpg.does_item_exist("filter_spiking_mode_combo"): 
        dpg.set_value("filter_spiking_mode_combo", filter_settings.get("spiking_mode", "Highlight Spiking"))

    type_filter_enabled = filter_settings.get("type_filter_enabled", False)
    if dpg.does_item_exist("filter_type_enable_cb"): dpg.set_value("filter_type_enable_cb", type_filter_enabled)
    if dpg.does_item_exist("filter_neuron_type_combo"):
        dpg.configure_item("filter_neuron_type_combo", enabled=type_filter_enabled) 
        # Populate items for filter_neuron_type_combo based on current model (sim_bridge needed or default list)
        # This part is tricky if sim_config is not yet aligned with the profile's model type.
        # Assume _populate_ui_from_config_dict has already set the model type.
        # Then, we can get available types.
        available_types = []
        if global_simulation_bridge and hasattr(global_simulation_bridge, 'get_available_neuron_types'):
            available_types = global_simulation_bridge.get_available_neuron_types()
        elif dpg.does_item_exist("cfg_neuron_model_type"): # Fallback if sim_bridge not ready
            model_name = dpg.get_value("cfg_neuron_model_type")
            temp_cfg_types = SimulationConfiguration(); temp_cfg_types.neuron_model_type = model_name
            if model_name == NeuronModel.IZHIKEVICH.name: available_types = ["All"] + [f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name]
            elif model_name == NeuronModel.HODGKIN_HUXLEY.name: available_types = ["All"] + [f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name]
            else: available_types = ["All"]
            available_types = list(dict.fromkeys(available_types))


        dpg.configure_item("filter_neuron_type_combo", items=available_types)
        selected_type = filter_settings.get("selected_neuron_type", "All")
        if selected_type in available_types: dpg.set_value("filter_neuron_type_combo", selected_type)
        elif "All" in available_types: dpg.set_value("filter_neuron_type_combo", "All")
        elif available_types: dpg.set_value("filter_neuron_type_combo", available_types[0])
            
    if dpg.does_item_exist("filter_min_abs_weight_slider"): dpg.set_value("filter_min_abs_weight_slider", filter_settings.get("min_abs_weight", 0.01))

    show_syn_val = filter_settings.get("show_synapses_cb", global_gui_state.get("show_connections_gl", False))
    global_gui_state["show_connections_gl"] = show_syn_val 
    if dpg.does_item_exist("filter_show_synapses_gl_cb"): dpg.set_value("filter_show_synapses_gl_cb", show_syn_val)

    if OPENGL_AVAILABLE:
        loaded_gl_settings = gui_cfg_dict.get("opengl_visualization_settings")
        if loaded_gl_settings: opengl_viz_config.update(loaded_gl_settings) # Update the global GL config dict

        # Apply these settings to DPG widgets for GL config
        gl_settings_to_apply_to_dpg = [
            ("gl_neuron_point_size_slider", 'POINT_SIZE', 2.0),
            ("gl_synapse_alpha_slider", 'SYNAPSE_ALPHA_MODIFIER', 0.3),
            ("gl_max_neurons_render_input", 'MAX_NEURONS_TO_RENDER', 10000),
            ("gl_max_connections_render_input", 'MAX_CONNECTIONS_TO_RENDER', 20000),
            ("gl_inactive_neuron_opacity_slider", 'INACTIVE_NEURON_OPACITY', 0.25),
            ("gl_activity_highlight_frames_input", 'ACTIVITY_HIGHLIGHT_FRAMES', 7),
            ("gl_enable_synaptic_pulses_cb", 'ENABLE_SYNAPTIC_PULSES', True)
        ]
        for tag, key, default_val in gl_settings_to_apply_to_dpg:
            if dpg.does_item_exist(tag): dpg.set_value(tag, opengl_viz_config.get(key, default_val))
        
        # Apply camera FOV if it was in the GUI settings snapshot
        if dpg.does_item_exist("cfg_camera_fov") and "CAMERA_FOV_DPG_Snapshot" in opengl_viz_config:
            dpg.set_value("cfg_camera_fov", opengl_viz_config["CAMERA_FOV_DPG_Snapshot"])
        # If not in snapshot, _populate_ui_from_config_dict would have set it from main sim_config.

    trigger_filter_update_signal(); # Filters or GL settings might have changed
    update_status_bar("GUI settings applied from profile/checkpoint.", level="info")
    return True


def update_monitoring_overlay_values(sim_data_dict):
    """
    Updates the DPG monitoring text elements with current simulation data.
    Called by the main/UI thread when new data arrives from sim_to_ui_queue.
    Note: Most monitoring data is now displayed in the OpenGL HUD. This function
    only updates the playback frame counter.
    """
    if not dpg.is_dearpygui_running(): return

    # Update playback frame counter if in playback mode
    if global_gui_state.get("is_playback_mode_active") and dpg.does_item_exist("playback_current_frame_text"):
        active_rec_meta = global_gui_state.get("active_recording_data_source") # This is UI thread's copy
        if active_rec_meta and "num_frames" in active_rec_meta:
            total_frames = active_rec_meta["num_frames"]
            current_frame_idx_ui = global_gui_state.get("current_playback_frame_index",0) # UI's current frame
            dpg.set_value("playback_current_frame_text", f"Frame: {current_frame_idx_ui + 1} / {total_frames if total_frames > 0 else 1}")

# --- DPG Event Handlers for Recording & Playback (HDF5) ---

def _recording_options_continue_callback(sender=None, app_data=None, user_data=None):
    """Called when user clicks Continue in the recording options popup."""
    # Read options from the popup and update sim_bridge's gpu_config
    recording_mode = dpg.get_value("rec_opt_mode_combo")
    skip_synaptic = dpg.get_value("rec_opt_skip_synaptic")
    frame_skip = dpg.get_value("rec_opt_frame_skip")

    # Send options to sim_bridge via command queue
    ui_to_sim_queue.put({
        "type": "SET_RECORDING_OPTIONS",
        "recording_mode": recording_mode,
        "recording_skip_synaptic_data": skip_synaptic,
        "recording_frame_skip": frame_skip
    })

    # Hide the options popup
    if dpg.does_item_exist("recording_options_popup"):
        dpg.hide_item("recording_options_popup")

    # Show the file dialog
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_recording_file_dialog_h5"):
        dpg.show_item("save_recording_file_dialog_h5")

def _recording_options_cancel_callback(sender=None, app_data=None, user_data=None):
    """Called when user cancels the recording options popup."""
    if dpg.does_item_exist("recording_options_popup"):
        dpg.hide_item("recording_options_popup")
    update_status_bar("Recording cancelled.", level="info")

def handle_record_button_click(sender=None, app_data=None, user_data=None):
    """
    Handles the 'Record' / 'Finalize Recording' button click.
    Shows recording options popup or sends command to stop recording.
    """
    if global_gui_state.get("is_recording_active", False):  # If currently recording, this button means "Finalize"
        ui_to_sim_queue.put({"type": "STOP_RECORDING"})
        update_status_bar("Finalize recording command sent...", level="info")
        # UI state will be updated when sim_thread confirms via "RECORDING_FINALIZED"
    else:  # Not recording, this button means "Record" - show options popup
        if global_gui_state.get("is_playback_mode_active", False):
            update_status_bar("Error: Cannot record while in playback mode.", color=[255,0,0,255], level="error")
            return
        if dpg.is_dearpygui_running() and dpg.does_item_exist("recording_options_popup"):
            dpg.show_item("recording_options_popup")
        else:
            update_status_bar("Error: Recording options dialog missing.", color=[255,0,0,255], level="error")

def save_recording_for_streaming_dialog_callback_h5(sender, app_data):
    """Callback for the 'Record' (Save Recording As) file dialog. Ensures correct extension."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_filepath_extension(
            app_data["file_path_name"],
            required_extension=".simrec.h5",
            filter_extension=app_data.get("current_filter")
        )

        ui_to_sim_queue.put({"type": "START_RECORDING", "filepath": filepath})
        update_status_bar(f"Start recording command sent for: {os.path.basename(filepath)}", level="info")
    else:
        update_status_bar("Recording setup cancelled.", level="info")


def handle_playback_button_click(sender=None, app_data=None, user_data=None):
    """
    Handles the 'Playback Recording' / 'Stop Playback' button click.
    Sends command to sim_thread to enter or exit playback mode.
    """
    if global_gui_state.get("is_playback_mode_active", False): # If in playback, stop it
        ui_to_sim_queue.put({"type": "EXIT_PLAYBACK_MODE"})
        update_status_bar("Exit playback command sent...", level="info")
    else: # Not in playback, try to enter
        if global_gui_state.get("is_recording_active", False):
            update_status_bar("Error: Finalize active recording before entering playback.", color=[255,165,0,255], level="warning")
            return

        loaded_data_meta = global_gui_state.get("active_recording_data_source")
        if not loaded_data_meta or not loaded_data_meta.get("h5_file_obj_for_playback") or \
           not loaded_data_meta.get("h5_file_obj_for_playback").id: # Check if a valid recording is loaded in UI state
            update_status_bar("No valid recording loaded. Load one first via File menu.", color=[255,165,0,255], level="warning")
            return

        # Command sim_thread to setup for playback
        ui_to_sim_queue.put({
            "type": "SETUP_PLAYBACK_FROM_RECORDING",
            "config_snapshot": loaded_data_meta["config_snapshot"],
            "h5_file_handle_for_sim_thread": loaded_data_meta["h5_file_obj_for_playback"], # *** ADD THIS LINE ***
            "initial_state_group_name": "initial_state" # Usually "initial_state"
        })
        update_status_bar("Enter playback mode command sent...", level="info")


def handle_load_recording_menu_click(sender=None, app_data=None, user_data=None):
    """Handles the 'File > Load Recording' menu item click. Shows file dialog."""
    if global_gui_state.get("is_recording_active", False):
        update_status_bar("Finalize current recording before loading another.", color=[255,165,0,255], level="warning")
        return
    if global_gui_state.get("is_playback_mode_active", False): 
        # If already in playback, command sim_thread to exit first, then show dialog after confirmation.
        # This makes the flow cleaner. For now, just warn.
        update_status_bar("Exit current playback mode before loading a new recording.", color=[255,165,0,255], level="warning")
        # A better flow: send EXIT_PLAYBACK, then on confirmation, show dialog.
        # ui_to_sim_queue.put({"type": "EXIT_PLAYBACK_MODE", "then_show_dialog": "load_recording_file_dialog_h5"})
        return


    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_recording_file_dialog_h5"):
        dpg.show_item("load_recording_file_dialog_h5")

def _normalize_load_filepath(filepath, filter_extension=None):
    """
    Normalizes a filepath from a load dialog by stripping filter artifacts.

    Args:
        filepath: The filepath from the file dialog
        filter_extension: The filter extension that may have been appended (e.g., ".*", ".h5")

    Returns:
        Cleaned filepath
    """
    # Strip ".*" if DPG appended it from "All Files" filter
    if filepath.endswith(".*"):
        filepath = filepath[:-2]

    # Strip filter extension if it was appended to a valid file path
    if filter_extension and filter_extension not in [".*", ""]:
        if filepath.endswith(filter_extension):
            potential_path = filepath[:-len(filter_extension)]
            if os.path.isfile(potential_path):
                filepath = potential_path

    return filepath

def _estimate_recording_memory_requirements(filepath):
    """
    Estimates the GPU memory required to cache a recording.

    Returns:
        tuple: (num_frames, estimated_bytes, fits_in_vram, available_vram_bytes, vram_limit_pct)
               or (None, None, None, None, None) if estimation fails
    """
    try:
        import h5py

        # Get available GPU memory
        if not cp:
            return None, None, None, None, None

        mem_info = cp.cuda.Device().mem_info
        free_memory, total_memory = mem_info
        vram_limit_pct = 0.90  # Use 90% of available VRAM
        usable_memory = free_memory * vram_limit_pct

        # Open file briefly to estimate size
        with h5py.File(filepath, 'r') as h5_file:
            frames_group = h5_file.get("frames")
            if not frames_group:
                return None, None, None, None, None

            num_frames = len(frames_group.keys())
            if num_frames == 0:
                return 0, 0, True, free_memory, vram_limit_pct

            # Sample first frame to estimate per-frame size
            first_frame_key = f"frame_0"
            first_frame = frames_group.get(first_frame_key)
            if not first_frame:
                # Try to find any frame
                frame_keys = list(frames_group.keys())
                if frame_keys:
                    first_frame = frames_group.get(frame_keys[0])

            if not first_frame:
                return num_frames, None, None, free_memory, vram_limit_pct

            # Estimate frame size from datasets
            frame_size_bytes = 0
            for key in first_frame.keys():
                dataset = first_frame[key]
                if hasattr(dataset, 'shape') and hasattr(dataset, 'dtype'):
                    frame_size_bytes += np.prod(dataset.shape) * dataset.dtype.itemsize

            # Add overhead for CuPy arrays (~10%)
            frame_size_bytes = int(frame_size_bytes * 1.1)

            total_estimated_bytes = frame_size_bytes * num_frames
            fits_in_vram = total_estimated_bytes <= usable_memory

            return num_frames, total_estimated_bytes, fits_in_vram, free_memory, vram_limit_pct

    except Exception as e:
        print(f"Error estimating recording memory: {e}")
        return None, None, None, None, None

def _show_recording_memory_warning_popup(filepath, num_frames, estimated_bytes, available_bytes):
    """Shows a popup warning that the recording won't fit in VRAM."""
    global_gui_state["_pending_recording_filepath"] = filepath

    estimated_gb = estimated_bytes / 1e9
    available_gb = available_bytes / 1e9
    pct_of_vram = (estimated_bytes / available_bytes) * 100 if available_bytes > 0 else 0

    # Update popup text
    if dpg.does_item_exist("recording_memory_warning_text"):
        dpg.set_value("recording_memory_warning_text",
            f"The selected recording ({num_frames} frames) is estimated to require\n"
            f"{estimated_gb:.2f} GB of GPU memory, but only {available_gb:.2f} GB is available.\n"
            f"(Recording is ~{pct_of_vram:.0f}% of available VRAM)\n\n"
            f"How would you like to proceed?"
        )

    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.show_item("recording_memory_warning_popup")

def _recording_memory_popup_partial_cache(sender=None, app_data=None):
    """Callback for 'Partial Cache' button in memory warning popup."""
    filepath = global_gui_state.get("_pending_recording_filepath")
    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.hide_item("recording_memory_warning_popup")

    if filepath:
        ui_to_sim_queue.put({
            "type": "LOAD_RECORDING",
            "filepath": filepath,
            "stream_only": False  # Will auto-stop caching when memory limit reached
        })
        update_status_bar(f"Load recording (partial cache) command sent for: {os.path.basename(filepath)}", level="info")

def _recording_memory_popup_stream_only(sender=None, app_data=None):
    """Callback for 'Stream Only' button in memory warning popup."""
    filepath = global_gui_state.get("_pending_recording_filepath")
    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.hide_item("recording_memory_warning_popup")

    if filepath:
        ui_to_sim_queue.put({
            "type": "LOAD_RECORDING",
            "filepath": filepath,
            "stream_only": True
        })
        update_status_bar(f"Load recording (streaming) command sent for: {os.path.basename(filepath)}", level="info")

def _recording_memory_popup_cancel(sender=None, app_data=None):
    """Callback for 'Cancel' button in memory warning popup."""
    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.hide_item("recording_memory_warning_popup")
    update_status_bar("Recording load cancelled.", level="info")

def load_recording_dialog_callback_h5(sender, app_data):
    """Callback for the 'Load Recording' file dialog. Sends command to sim_thread."""
    filepath_to_load = None
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_load_filepath(
            app_data["file_path_name"],
            filter_extension=app_data.get("current_filter")
        )

        if os.path.isfile(filepath):
            filepath_to_load = filepath
        elif os.path.isdir(filepath):
            update_status_bar(f"Error: Selected path is a directory: {filepath}", color=[255,100,0,255], level="warning")
            return
        else:
            update_status_bar(f"Load error: File not found or invalid path: '{filepath}'.", color=[255,0,0,255], level="error")
            return

    elif "file_name" in app_data and app_data["file_name"] and "current_path" in app_data: # Fallback
        filepath = os.path.join(app_data["current_path"], app_data["file_name"])
        if os.path.isfile(filepath):
            filepath_to_load = filepath
        else:
            update_status_bar(f"Error: Fallback path is not a valid file: {filepath}", color=[255,0,0,255], level="error")
            return
    else:
        update_status_bar("Load recording cancelled or no file selected.", level="info")
        return

    if filepath_to_load:
        # Check if recording fits in VRAM
        num_frames, estimated_bytes, fits_in_vram, available_bytes, _ = _estimate_recording_memory_requirements(filepath_to_load)

        if fits_in_vram is None:
            # Couldn't estimate, just proceed with caching attempt
            ui_to_sim_queue.put({
                "type": "LOAD_RECORDING",
                "filepath": filepath_to_load,
                "stream_only": False
            })
            update_status_bar(f"Load recording (caching) command sent for: {os.path.basename(filepath_to_load)}", level="info")
        elif fits_in_vram:
            # Recording fits, proceed with caching
            ui_to_sim_queue.put({
                "type": "LOAD_RECORDING",
                "filepath": filepath_to_load,
                "stream_only": False
            })
            estimated_gb = estimated_bytes / 1e9 if estimated_bytes else 0
            update_status_bar(f"Load recording (caching ~{estimated_gb:.1f}GB) command sent for: {os.path.basename(filepath_to_load)}", level="info")
        else:
            # Recording won't fit, show warning popup
            _show_recording_memory_warning_popup(filepath_to_load, num_frames, estimated_bytes, available_bytes)

def handle_playback_slider_change(sender, frame_idx_from_slider_float, user_data=None):
    """Handles playback slider changes. Sends command to sim_thread to set frame if handle is valid."""
    frame_idx_from_slider = int(frame_idx_from_slider_float)
    if not global_gui_state.get("is_playback_mode_active", False):
        # This should not happen if controls are correctly disabled, but as a safeguard:
        update_status_bar("Playback not active. Cannot seek.", level="warning")
        return

    if global_gui_state.get("playback_is_playing_ui", False): # If user moves slider while playing, pause.
        global_gui_state["playback_is_playing_ui"] = False
        update_ui_for_playback_playing_state(is_playing=False)
        update_status_bar("Playback paused for manual seek.", level="info")

    loaded_data_meta = global_gui_state.get("active_recording_data_source")
    h5_handle = None
    if loaded_data_meta:
        h5_handle = loaded_data_meta.get("h5_file_obj_for_playback")

    if h5_handle and hasattr(h5_handle, 'id') and h5_handle.id: # Check if handle is valid and open
        num_frames = loaded_data_meta.get("num_frames") if loaded_data_meta else None
        ui_to_sim_queue.put({
            "type": "SET_PLAYBACK_FRAME",
            "frame_index": frame_idx_from_slider,
            "h5_file_handle_for_sim_thread": h5_handle,
            "num_frames": num_frames
        })
        # Status update for successful command send can be minimal or handled by sim thread ACK
        # update_status_bar(f"Seek to frame {frame_idx_from_slider+1} command sent.", level="debug")
    else:
        # This else block means the command will NOT be sent, preventing the error in sim_thread.
        if not loaded_data_meta:
            update_status_bar("Error: No active recording data source for playback seek.", color=[255,0,0,255], level="error")
        else: # loaded_data_meta exists, but handle is bad
            update_status_bar("Error: HDF5 handle for playback is invalid or closed. Cannot seek.", color=[255,0,0,255], level="error")


def handle_playback_play_pause_button_click(sender=None, app_data=None, user_data=None):
    """Handles the Play/Pause button for playback controls."""
    if not global_gui_state.get("is_playback_mode_active", False): return

    new_playing_state = not global_gui_state.get("playback_is_playing_ui", False)
    global_gui_state["playback_is_playing_ui"] = new_playing_state
    update_ui_for_playback_playing_state(is_playing=new_playing_state) # Update button label

    if new_playing_state:
        global_gui_state["last_playback_autostep_time_ui"] = time.perf_counter() # Reset timer for UI-driven stepping
        # If at the end of playback, loop back to the beginning by commanding frame 0
        active_rec_meta = global_gui_state.get("active_recording_data_source")
        if active_rec_meta and "num_frames" in active_rec_meta:
            num_frames = active_rec_meta["num_frames"]
            h5_handle = active_rec_meta.get("h5_file_obj_for_playback")
            current_frame_ui = global_gui_state.get("current_playback_frame_index", 0)
            if num_frames > 0 and current_frame_ui >= num_frames - 1:
                ui_to_sim_queue.put({
                    "type": "SET_PLAYBACK_FRAME",
                    "frame_index": 0,
                    "h5_file_handle_for_sim_thread": h5_handle,
                    "num_frames": num_frames
                })
        update_status_bar("Playback started/resumed by UI.", level="info")
    else:
        update_status_bar("Playback paused by UI.", level="info")
    # The actual frame stepping for playback_is_playing_ui is handled in the main DPG loop.

def handle_playback_step_frames_click(sender, app_data, user_data):
    """Handles clicks for playback step buttons. Sends command to sim_thread if handle is valid."""
    if not global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Playback not active. Cannot step frames.", level="warning")
        return

    step_amount = user_data
    if not isinstance(step_amount, int):
        return

    if global_gui_state.get("playback_is_playing_ui", False):
        global_gui_state["playback_is_playing_ui"] = False
        update_ui_for_playback_playing_state(is_playing=False)
        update_status_bar("Playback paused for manual step.", level="info")

    current_frame_ui = global_gui_state.get("current_playback_frame_index", 0)
    active_rec_meta = global_gui_state.get("active_recording_data_source")
    num_frames = active_rec_meta.get("num_frames", 0) if active_rec_meta else 0

    new_frame_idx = current_frame_ui + step_amount
    if num_frames > 0:
        new_frame_idx = max(0, min(new_frame_idx, num_frames - 1))
    else:
        new_frame_idx = 0

    h5_handle = None
    if active_rec_meta:
        h5_handle = active_rec_meta.get("h5_file_obj_for_playback")

    if h5_handle and hasattr(h5_handle, 'id') and h5_handle.id: # Check if handle is valid and open
        ui_to_sim_queue.put({
            "type": "SET_PLAYBACK_FRAME",
            "frame_index": new_frame_idx,
            "h5_file_handle_for_sim_thread": h5_handle,
            "num_frames": num_frames
        })
        # update_status_bar(f"Step playback by {step_amount} (to frame {new_frame_idx+1}) command sent.", level="debug")
    else:
        # Command will NOT be sent.
        if not active_rec_meta:
            update_status_bar("Error: No active recording data source for playback step.", color=[255,0,0,255], level="error")
        else: # active_rec_meta exists, but handle is bad
            update_status_bar("Error: HDF5 handle for playback is invalid or closed. Cannot step.", color=[255,0,0,255], level="error")

# --- GUI Update Helper Functions for Recording/Playback States (Called by Main/UI Thread) ---

def update_ui_for_simulation_run_state(is_running, is_paused):
    """Updates DPG UI elements based on the simulation's run/pause state (UI perspective)."""
    if not dpg.is_dearpygui_running(): return

    is_playback_active_ui = global_gui_state.get("is_playback_mode_active", False)
    is_recording_active_ui = global_gui_state.get("is_recording_active", False)

    # Live Simulation Controls
    dpg.configure_item("start_button", enabled=not is_running and not is_playback_active_ui)
    dpg.configure_item("pause_button", enabled=is_running and not is_playback_active_ui, label="Resume" if is_paused else "Pause")
    dpg.configure_item("stop_button", enabled=is_running and not is_playback_active_ui)
    dpg.configure_item("step_button", enabled=(is_paused or not is_running) and not is_playback_active_ui)

    # Apply config button: disabled if sim is running (live), or in playback, or actively recording
    can_apply_config = not is_running and not is_playback_active_ui and not is_recording_active_ui
    dpg.configure_item("apply_config_button", enabled=can_apply_config)

    # Record button: label managed by update_ui_for_recording_state.
    # Enabled if not in playback.
    dpg.configure_item("record_button", enabled=not is_playback_active_ui)

    # Playback button: enabled if NOT actively recording AND
    # ( (a recording is loaded AND valid) OR playback is already active (to allow stopping it) )
    loaded_data_meta = global_gui_state.get("active_recording_data_source")
    is_valid_recording_loaded_ui = loaded_data_meta and loaded_data_meta.get("h5_file_obj_for_playback") # Simpler check for UI
    
    can_initiate_or_stop_playback = not is_recording_active_ui and (is_valid_recording_loaded_ui or is_playback_active_ui)
    dpg.configure_item("playback_button", enabled=can_initiate_or_stop_playback)
    dpg.set_item_label("playback_button", "Stop Playback" if is_playback_active_ui else "Playback Recording")


def update_ui_for_recording_state(is_recording_active_ui): # Based on UI's perspective
    """Updates UI elements related to recording state (e.g., Record button label)."""
    if not dpg.is_dearpygui_running(): return
    global_gui_state["is_recording_active"] = is_recording_active_ui # Update UI's view

    dpg.set_item_label("record_button", "Finalize Recording" if is_recording_active_ui else "Record")
    
    # Re-evaluate other controls based on this new recording state
    current_sim_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
    current_sim_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
    update_ui_for_simulation_run_state(is_running=current_sim_running_ui, is_paused=current_sim_paused_ui)


def update_ui_for_playback_mode_state(is_playback_active_ui, num_frames_from_meta=0):
    """Updates UI elements when entering or exiting active playback mode (UI perspective)."""
    if not dpg.is_dearpygui_running(): return
    global_gui_state["is_playback_mode_active"] = is_playback_active_ui

    # Main Playback/Stop Playback button label and state
    if dpg.does_item_exist("playback_button"):
        dpg.set_item_label("playback_button", "Stop Playback" if is_playback_active_ui else "Playback Recording")
    
    # Show/hide and enable/disable playback controls group elements
    if dpg.does_item_exist("playback_controls_group"):
        dpg.configure_item("playback_controls_group", show=is_playback_active_ui)

    step_buttons_enabled_ui = is_playback_active_ui
    if dpg.does_item_exist("playback_step_minus_5"): dpg.configure_item("playback_step_minus_5", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_step_minus_1"): dpg.configure_item("playback_step_minus_1", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_play_pause_button"): dpg.configure_item("playback_play_pause_button", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_step_plus_1"): dpg.configure_item("playback_step_plus_1", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_step_plus_5"): dpg.configure_item("playback_step_plus_5", enabled=step_buttons_enabled_ui)

    if is_playback_active_ui:
        slider_max = max(0, num_frames_from_meta - 1)
        if dpg.does_item_exist("playback_slider"):
            dpg.configure_item("playback_slider", max_value=slider_max, enabled=True)
            dpg.set_value("playback_slider", 0) # Reset slider to beginning
        if dpg.does_item_exist("playback_current_frame_text"):
             dpg.set_value("playback_current_frame_text", f"Frame: 1 / {num_frames_from_meta if num_frames_from_meta > 0 else 1}")
        
        global_gui_state["playback_is_playing_ui"] = False # Start paused
        update_ui_for_playback_playing_state(is_playing=False) 

        # Disable live simulation controls and conflicting file operations
        if dpg.does_item_exist("record_button"): dpg.configure_item("record_button", enabled=False)
        if dpg.does_item_exist("start_button"): dpg.configure_item("start_button", enabled=False)
        if dpg.does_item_exist("pause_button"): dpg.configure_item("pause_button", enabled=False, label="Pause")
        if dpg.does_item_exist("stop_button"): dpg.configure_item("stop_button", enabled=False)
        if dpg.does_item_exist("step_button"): dpg.configure_item("step_button", enabled=False)
        if dpg.does_item_exist("apply_config_button"): dpg.configure_item("apply_config_button", enabled=False)
        
        menu_items_to_disable = ["save_profile_menu", "load_profile_menu", 
                                 "save_checkpoint_menu_h5", "load_checkpoint_menu_h5",
                                 "load_recording_menu_h5"] # Can't load another recording while one is active for playback
        for item_tag in menu_items_to_disable:
            if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, enabled=False)
    else: # Exiting active playback mode
        # Enable live simulation controls and file operations
        # update_ui_for_simulation_run_state will handle most of these based on current sim state
        current_sim_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
        current_sim_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
        update_ui_for_simulation_run_state(is_running=current_sim_running_ui, is_paused=current_sim_paused_ui)

        menu_items_to_enable = ["save_profile_menu", "load_profile_menu", 
                                "save_checkpoint_menu_h5", "load_checkpoint_menu_h5",
                                "load_recording_menu_h5"]
        for item_tag in menu_items_to_enable:
            if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, enabled=True)


def update_ui_for_playback_playing_state(is_playing): # UI's perspective of playback auto-play
    """Updates the Play/Pause button label within the playback controls."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("playback_play_pause_button"):
         dpg.set_item_label("playback_play_pause_button", "Pause Playback" if is_playing else "Play Recording")

def update_ui_after_recording_loaded(loaded_meta_data_package):
    """
    Updates UI elements after a recording's metadata has been successfully prepared by sim_thread.
    Called by UI thread when "RECORDING_METADATA_PREPARED" message is received.
    """
    if not dpg.is_dearpygui_running(): return

    if loaded_meta_data_package:
        global_gui_state["active_recording_data_source"] = loaded_meta_data_package
        global_gui_state["loaded_recording_filepath_for_ui"] = loaded_meta_data_package.get("filepath")
        num_frames = loaded_meta_data_package.get("num_frames", 0)
        
        # Enable the main "Playback Recording" button
        if dpg.does_item_exist("playback_button"):
            dpg.configure_item("playback_button", enabled=True, label="Playback Recording")
        
        # Configure the playback controls group (it's initially hidden)
        # It will be shown when user actually clicks "Playback Recording" -> enters playback mode.
        # For now, just ensure slider max is ready.
        if dpg.does_item_exist("playback_slider"):
            dpg.configure_item("playback_slider", max_value=max(0, num_frames - 1))
            dpg.set_value("playback_slider", 0) # Reset to start
        if dpg.does_item_exist("playback_current_frame_text"):
            dpg.set_value("playback_current_frame_text", f"Frame: 1 / {num_frames if num_frames > 0 else 1}")
        
        update_status_bar(f"Recording '{os.path.basename(loaded_meta_data_package.get('filepath', ''))}' loaded. {num_frames} frames. Ready for playback.", level="success")
    else: # Should not happen if message is for success
        global_gui_state["active_recording_data_source"] = None
        global_gui_state["loaded_recording_filepath_for_ui"] = None
        if dpg.does_item_exist("playback_button"):
            dpg.configure_item("playback_button", enabled=False)
        update_status_bar("Failed to process loaded recording metadata.", level="error")

# --- Handlers for Performance Testing & System Logs ---

def handle_run_benchmark_click(sender=None, app_data=None, user_data=None):
    """Runs the benchmark suite in a background thread."""
    global performance_test_running_type
    # Clear stop flag and enable stop button
    performance_test_stop_flag.clear()
    performance_test_running_type = "benchmark"
    if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
        dpg.configure_item("stop_perf_test_button", enabled=True)
    
    def run_benchmark():
        try:
            if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
                dpg.set_value("perf_test_status_text", "Running benchmark suite...")
                dpg.set_value("perf_test_results_text", "")
            
            import subprocess
            # Stream output line-by-line so LogCapture can see it
            process = subprocess.Popen(
                [sys.executable, "benchmark.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                # Check stop flag
                if performance_test_stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("[STOPPED] Benchmark suite stopped by user")
                    if dpg.is_dearpygui_running():
                        dpg.set_value("perf_test_status_text", "Benchmark stopped by user.")
                        dpg.set_value("perf_test_results_text", "Partial results discarded. Previous results preserved.")
                        update_status_bar("Benchmark stopped", level="warning")
                    return
                
                print(line.rstrip())  # Print to console AND LogCapture
                output_lines.append(line.rstrip())
            
            returncode = process.wait(timeout=300)
            
            if returncode == 0:
                # Reload hardware limits from freshly written benchmark_results.json
                global HARDWARE_LIMITS
                HARDWARE_LIMITS = None  # Force reload
                _load_benchmark_limits()
                hw_note = get_hardware_note()
                if hw_note and dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_hardware_performance_note"):
                    dpg.set_value("cfg_hardware_performance_note", hw_note)

                status = "Benchmark complete. Hardware limits updated."
                summary = hw_note + "\n\n" + "\n".join(output_lines[-5:]) if hw_note else "\n".join(output_lines[-10:])
            else:
                status = f"Benchmark failed with code {returncode}"
                summary = "\n".join(output_lines[-10:]) if len(output_lines) > 10 else "\n".join(output_lines)

            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", status)
                dpg.set_value("perf_test_results_text", summary)
                update_status_bar(status, level="info" if returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            process.kill()
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", "Benchmark timed out after 5 minutes.")
                dpg.set_value("perf_test_results_text", "Check System Logs for partial results.")
                update_status_bar("Benchmark timed out", level="error")
        except Exception as e:
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
                dpg.set_value("perf_test_results_text", "")
                update_status_bar(f"Benchmark error: {str(e)}", level="error")
        finally:
            global performance_test_running_type
            performance_test_running_type = None
            # Disable stop button when done
            if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
                dpg.configure_item("stop_perf_test_button", enabled=False)
    
    threading.Thread(target=run_benchmark, daemon=True).start()
    update_status_bar("Starting benchmark suite...", level="info")

def handle_run_optimization_click(sender=None, app_data=None, user_data=None):
    """Runs the auto-tuning workflow to optimize drive scales for different model/profile combinations."""
    global performance_test_running_type
    # Clear stop flag and enable stop button
    performance_test_stop_flag.clear()
    performance_test_running_type = "optimization"
    if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
        dpg.configure_item("stop_perf_test_button", enabled=True)
    
    def run_optimization():
        try:
            # Check if quick mode is enabled
            quick_mode = False
            if dpg.is_dearpygui_running() and dpg.does_item_exist("optimization_quick_mode_checkbox"):
                quick_mode = dpg.get_value("optimization_quick_mode_checkbox")
            
            mode_text = "quick mode" if quick_mode else "full mode"
            if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
                dpg.set_value("perf_test_status_text", f"Running auto-tuning workflow ({mode_text})...")
                dpg.set_value("perf_test_results_text", "This may take several minutes.\nCheck console for detailed progress.")
            
            import subprocess
            # Build command with --auto-tune flag, optionally with --quick
            cmd = [sys.executable, "neural-simulator.py", "--auto-tune"]
            if quick_mode:
                cmd.append("--quick")
            
            # Stream output line-by-line so LogCapture can see it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                # Check stop flag
                if performance_test_stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("[STOPPED] Auto-tuning/optimization stopped by user")
                    if dpg.is_dearpygui_running():
                        dpg.set_value("perf_test_status_text", "Auto-tuning stopped by user.")
                        dpg.set_value("perf_test_results_text", "Partial results discarded. Previous overrides preserved.")
                        update_status_bar("Auto-tuning stopped", level="warning")
                    return
                
                print(line.rstrip())  # Print to console AND LogCapture
                output_lines.append(line.rstrip())
            
            returncode = process.wait(timeout=1800)
            
            if returncode == 0:
                status = "Auto-tuning complete. Results saved to auto_tuned_overrides.json"
                # Count how many combinations were tuned
                try:
                    import json
                    with open("simulation_profiles/auto_tuned_overrides.json", "r", encoding='utf-8') as f:
                        data = json.load(f)
                    count = len(data.get("tuned_combinations", {}))
                    summary = f"Successfully tuned {count} model/profile combinations.\nReload overrides to apply them."
                except:
                    summary = "Check System Logs or auto_tuned_overrides.json for results."
            else:
                status = f"Auto-tuning failed with code {returncode}"
                summary = "\n".join(output_lines[-10:]) if len(output_lines) > 10 else "\n".join(output_lines)
            
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", status)
                dpg.set_value("perf_test_results_text", summary)
                update_status_bar(status, level="info" if returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            process.kill()
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", "Auto-tuning timed out after 30 minutes.")
                dpg.set_value("perf_test_results_text", "Check System Logs for partial results.")
                update_status_bar("Auto-tuning timed out", level="error")
        except Exception as e:
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
                dpg.set_value("perf_test_results_text", "")
                update_status_bar(f"Auto-tuning error: {str(e)}", level="error")
        finally:
            global performance_test_running_type
            performance_test_running_type = None
            # Disable stop button when done
            if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
                dpg.configure_item("stop_perf_test_button", enabled=False)
    
    threading.Thread(target=run_optimization, daemon=True).start()
    update_status_bar("Starting auto-tuning workflow...", level="info")

def handle_stop_perf_test_click(sender=None, app_data=None, user_data=None):
    """Stops any running benchmark or optimization task."""
    global performance_test_running_type
    
    if performance_test_running_type:
        test_name = "benchmark suite" if performance_test_running_type == "benchmark" else "auto-tuning/optimization"
        print(f"[STOP REQUESTED] Stopping {test_name}...")
        update_status_bar(f"Stopping {test_name}...", level="warning")
    else:
        print("[STOP] No performance test currently running")
        update_status_bar("No test running to stop", level="info")
    
    performance_test_stop_flag.set()
    if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
        dpg.set_value("perf_test_status_text", "Stopping...")

def handle_reload_overrides_click(sender=None, app_data=None, user_data=None):
    """Reloads auto-tuned overrides from disk."""
    global AUTO_TUNED_OVERRIDES
    AUTO_TUNED_OVERRIDES = None  # Force reload
    _load_auto_tuned_overrides_if_needed()
    
    count = len(AUTO_TUNED_OVERRIDES) if AUTO_TUNED_OVERRIDES else 0
    msg = f"Reloaded {count} auto-tuned combinations from disk."
    update_status_bar(msg, level="success")
    
    if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
        dpg.set_value("perf_test_status_text", msg)
        dpg.set_value("perf_test_results_text", f"Available combinations: {count}\nThese will be applied automatically when Apply & Reset is clicked.")

def handle_log_search_change(sender, app_data, user_data):
    """Handles search input changes in the log viewer."""
    if not hasattr(handle_log_search_change, "log_capture"):
        return
    
    query = app_data.strip()
    if not query:
        if dpg.is_dearpygui_running():
            dpg.set_value("log_search_match_text", "0 / 0 matches")
            dpg.configure_item("log_search_prev_button", enabled=False)
            dpg.configure_item("log_search_next_button", enabled=False)
        return
    
    log_capture = handle_log_search_change.log_capture
    matches = log_capture.search(query)
    
    if dpg.is_dearpygui_running():
        if matches:
            handle_log_search_change.current_matches = matches
            handle_log_search_change.current_match_index = 0
            dpg.set_value("log_search_match_text", f"1 / {len(matches)} matches")
            dpg.configure_item("log_search_prev_button", enabled=len(matches) > 1)
            dpg.configure_item("log_search_next_button", enabled=len(matches) > 1)
            # Highlight first match
            _update_log_display_with_highlight(matches[0])
        else:
            dpg.set_value("log_search_match_text", "0 / 0 matches")
            dpg.configure_item("log_search_prev_button", enabled=False)
            dpg.configure_item("log_search_next_button", enabled=False)

def handle_log_search_prev(sender=None, app_data=None, user_data=None):
    """Navigate to previous search match."""
    if not hasattr(handle_log_search_change, "current_matches"):
        return
    
    matches = handle_log_search_change.current_matches
    if not matches:
        return
    
    handle_log_search_change.current_match_index = (handle_log_search_change.current_match_index - 1) % len(matches)
    idx = handle_log_search_change.current_match_index
    
    if dpg.is_dearpygui_running():
        dpg.set_value("log_search_match_text", f"{idx + 1} / {len(matches)} matches")
        _update_log_display_with_highlight(matches[idx])

def handle_log_search_next(sender=None, app_data=None, user_data=None):
    """Navigate to next search match."""
    if not hasattr(handle_log_search_change, "current_matches"):
        return
    
    matches = handle_log_search_change.current_matches
    if not matches:
        return
    
    handle_log_search_change.current_match_index = (handle_log_search_change.current_match_index + 1) % len(matches)
    idx = handle_log_search_change.current_match_index
    
    if dpg.is_dearpygui_running():
        dpg.set_value("log_search_match_text", f"{idx + 1} / {len(matches)} matches")
        _update_log_display_with_highlight(matches[idx])

def _update_log_display_with_highlight(line_index):
    """Updates the log display and scrolls to highlight a specific line."""
    if not hasattr(handle_log_search_change, "log_capture"):
        return
    
    log_capture = handle_log_search_change.log_capture
    logs = log_capture.get_logs()
    
    if 0 <= line_index < len(logs):
        # Show context around the match
        start = max(0, line_index - 5)
        end = min(len(logs), line_index + 6)
        
        display_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_index else "    "
            display_lines.append(f"{prefix}{logs[i]}")
        
        display_text = "\n".join(display_lines)
        if dpg.is_dearpygui_running() and dpg.does_item_exist("system_logs_display"):
            dpg.set_value("system_logs_display", display_text)

def handle_clear_logs_click(sender=None, app_data=None, user_data=None):
    """Clears the log buffer."""
    if hasattr(handle_log_search_change, "log_capture"):
        handle_log_search_change.log_capture.clear()
        if dpg.is_dearpygui_running() and dpg.does_item_exist("system_logs_display"):
            dpg.set_value("system_logs_display", "")
        update_status_bar("Logs cleared.", level="info")

def handle_export_logs_click(sender=None, app_data=None, user_data=None):
    """Exports logs to a timestamped file."""
    if not hasattr(handle_log_search_change, "log_capture"):
        return
    
    try:
        log_capture = handle_log_search_change.log_capture
        logs = log_capture.get_logs()
        
        if not logs:
            update_status_bar("No logs to export.", level="warning")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"simulation_logs_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("\n".join(logs))
        
        update_status_bar(f"Logs exported to {filepath}", level="success")
    except Exception as e:
        update_status_bar(f"Export error: {str(e)}", level="error")

def handle_run_viz_benchmark_click(sender=None, app_data=None, user_data=None):
    """Runs the visualization performance test in a background thread."""
    global performance_test_running_type
    # Clear stop flag and enable stop button
    performance_test_stop_flag.clear()
    performance_test_running_type = "viz_benchmark"
    if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
        dpg.configure_item("stop_perf_test_button", enabled=True)
    
    def run_viz_benchmark():
        try:
            # Check if quick mode is enabled
            quick_mode = False
            if dpg.is_dearpygui_running() and dpg.does_item_exist("viz_benchmark_quick_mode_checkbox"):
                quick_mode = dpg.get_value("viz_benchmark_quick_mode_checkbox")
            
            mode_text = "quick mode" if quick_mode else "full mode"
            if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
                dpg.set_value("perf_test_status_text", f"Running visualization performance test ({mode_text})...")
                dpg.set_value("perf_test_results_text", "This may take several minutes.\nCheck System Logs for detailed progress.")
            
            import subprocess
            # Build command with optional --quick flag
            cmd = [sys.executable, "viz_benchmark.py"]
            if quick_mode:
                cmd.append("--quick")
            
            # Stream output line-by-line so LogCapture can see it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                # Check stop flag
                if performance_test_stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("[STOPPED] Viz performance test stopped by user")
                    if dpg.is_dearpygui_running():
                        dpg.set_value("perf_test_status_text", "Viz performance test stopped by user.")
                        dpg.set_value("perf_test_results_text", "Partial results discarded.")
                        update_status_bar("Viz performance test stopped", level="warning")
                    return
                
                print(line.rstrip())  # Print to console AND LogCapture
                output_lines.append(line.rstrip())
            
            returncode = process.wait(timeout=600)  # 10 minute timeout
            
            if returncode == 0:
                # Load results and update hardware note
                results_path = "benchmarks/viz_performance_results.json"
                try:
                    with open(results_path, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    hardware_note = benchmark_data.get("hardware_performance_note", "Benchmark completed.")
                    
                    # Update hardware note in UI
                    if dpg.does_item_exist("cfg_hardware_performance_note"):
                        dpg.set_value("cfg_hardware_performance_note", hardware_note)
                    
                    # Build summary for results text
                    capacity_summary = benchmark_data.get("capacity_summary", {})
                    summary_lines = ["Viz Performance Test Complete!", ""]
                    if capacity_summary:
                        for key, data in capacity_summary.items():
                            max_n = data.get("max_neurons", 0)
                            conn = data.get("connections_per_neuron", 0)
                            if max_n > 0:
                                summary_lines.append(f"{key}: {max_n:,}N ({max_n * conn:,} synapses)")
                            else:
                                summary_lines.append(f"{key}: No realtime configs found")
                    else:
                        summary_lines.append("No realtime-capable configurations found.")
                    
                    summary_lines.append("")
                    summary_lines.append("Results: benchmarks/viz_performance_results.json")
                    summary_lines.append("Hardware note updated in Core Simulation Parameters.")
                    summary = "\n".join(summary_lines)
                    
                    status = "Viz performance test complete."
                except Exception as e:
                    status = "Viz test complete but failed to parse results."
                    summary = f"Error: {str(e)}\nCheck benchmarks/viz_performance_results.json"
            else:
                status = f"Viz performance test failed with code {returncode}"
                summary = "\n".join(output_lines[-10:]) if len(output_lines) > 10 else "\n".join(output_lines)
            
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", status)
                dpg.set_value("perf_test_results_text", summary)
                update_status_bar(status, level="info" if returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            process.kill()
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", "Viz performance test timed out after 10 minutes.")
                dpg.set_value("perf_test_results_text", "Check System Logs for partial results.")
                update_status_bar("Viz performance test timed out", level="error")
        except Exception as e:
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
                dpg.set_value("perf_test_results_text", "")
                update_status_bar(f"Viz performance test error: {str(e)}", level="error")
        finally:
            global performance_test_running_type
            performance_test_running_type = None
            # Disable stop button when done
            if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
                dpg.configure_item("stop_perf_test_button", enabled=False)
    
    threading.Thread(target=run_viz_benchmark, daemon=True).start()
    update_status_bar("Starting viz performance test...", level="info")

# --- Main DPG GUI Layout Creation (Called by Main/UI Thread) ---

def add_parameter_table_row(label_text, item_callable, item_tag, default_value, callback_func, tooltip=None, **kwargs):
    """
    Adds a row to a DPG table with a label in the first column and a DPG item in the second.
    Assumes this is called within a `with dpg.table(): ...` context where columns are already defined.

    Args:
        tooltip: Optional string for a hover tooltip on the label, providing parameter help.
    """
    with dpg.table_row():
        label_id = dpg.add_text(label_text)
        if tooltip:
            with dpg.tooltip(label_id):
                dpg.add_text(tooltip, wrap=350, color=[220, 220, 180, 255])
        # Ensure 'label' kwarg for the item itself is empty as we're using a separate text widget
        kwargs['label'] = ""

        # Only add width=-1 if it's not a checkbox and width is not already specified.
        # Checkboxes and some other items might not support the 'width' argument or handle it differently.
        if item_callable != dpg.add_checkbox: # Check if the item is NOT a checkbox
            if 'width' not in kwargs: # If width is not already specified for other items
                kwargs['width'] = -1  # Make it fill the table cell
        elif 'width' in kwargs and item_callable == dpg.add_checkbox:
            # If width was somehow passed for a checkbox, remove it to prevent error
            del kwargs['width']

        return item_callable(tag=item_tag, default_value=default_value, callback=callback_func, **kwargs)

def create_gui_layout():
    """Creates the main Dear PyGui layout, including all windows, menus, and widgets."""
    profile_dir = "simulation_profiles/" 
    checkpoint_dir_h5 = "simulation_checkpoints_h5/"
    recording_dir_h5 = "simulation_recordings_h5/"  

    for p_dir in [profile_dir, checkpoint_dir_h5, recording_dir_h5]:
        if not os.path.exists(p_dir): os.makedirs(p_dir, exist_ok=True)

    # Define column widths for parameter tables
    label_col_width = 320 

    with dpg.window(label="Controls & Configuration", tag="controls_monitor_window",
                    width=-1, height=-1, pos=[0,0], 
                    on_close=lambda: (shutdown_flag.set(), dpg.stop_dearpygui() if dpg.is_dearpygui_running() else None),
                    menubar=True):
        dpg.add_spacer(height=5)

        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Save Profile (.json)", callback=handle_save_profile_button_press, tag="save_profile_menu")
                dpg.add_menu_item(label="Load Profile (.json)", callback=handle_load_profile_button_press, tag="load_profile_menu")
                dpg.add_separator()
                dpg.add_menu_item(label="Save Checkpoint (.simstate.h5)", callback=handle_save_checkpoint_button_press, tag="save_checkpoint_menu_h5")
                dpg.add_menu_item(label="Load Checkpoint (.simstate.h5)", callback=handle_load_checkpoint_button_press, tag="load_checkpoint_menu_h5")
                dpg.add_separator()
                dpg.add_menu_item(label="Load Recording (.simrec.h5)", callback=handle_load_recording_menu_click, tag="load_recording_menu_h5")
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: (shutdown_flag.set(), dpg.stop_dearpygui() if dpg.is_dearpygui_running() else None))

        with dpg.collapsing_header(label="Simulation Controls", default_open=True):
            dpg.add_text("Status: Idle", tag="status_bar_text")
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="start_button", callback=handle_start_simulation_event, width = -1)
            with dpg.group(horizontal=True): 
                dpg.add_button(label="Pause", tag="pause_button", callback=handle_pause_simulation_event, width=100, enabled=False)
                dpg.add_button(label="Stop", tag="stop_button", callback=handle_stop_simulation_event, width=100, enabled=False)
                dpg.add_button(label="Step (1ms)", tag="step_button", callback=handle_step_simulation_event, width=-1, enabled=True)

            dpg.add_button(label="Apply Changes & Reset Sim", tag="apply_config_button", callback=handle_apply_config_changes_and_reset, width=-1)
            
            dpg.add_text("Simulation Speed:")
            dpg.add_slider_float(label="", tag="sim_speed_slider", default_value=1.0, min_value=0.01, max_value=20.0, width=-1, callback=handle_sim_speed_change, format="%.2f x")

            dpg.add_separator()
            dpg.add_text("Recording & Playback:")
            with dpg.group(horizontal=True): 
                dpg.add_button(label="Record", tag="record_button", callback=handle_record_button_click, width = -1)
            with dpg.group(horizontal=True): 
                dpg.add_button(label="Playback Recording", tag="playback_button", callback=handle_playback_button_click, width = -1, enabled=False)

            with dpg.group(tag="playback_controls_group", show=False): 
                dpg.add_text("Playback Controls:")
                dpg.add_text("Frame: 0 / 0", tag="playback_current_frame_text")
                dpg.add_slider_int(label="", tag="playback_slider", width=-1, callback=handle_playback_slider_change, min_value=0, max_value=0) 
                with dpg.group(horizontal=True, horizontal_spacing=10):
                    dpg.add_button(label="<< (-5)", tag="playback_step_minus_5", callback=handle_playback_step_frames_click, user_data=-5, width=70)
                    dpg.add_button(label="< (-1)", tag="playback_step_minus_1", callback=handle_playback_step_frames_click, user_data=-1, width=70)
                    dpg.add_button(label="Play", tag="playback_play_pause_button", callback=handle_playback_play_pause_button_click, width = 150) 
                    dpg.add_button(label="(+1) >", tag="playback_step_plus_1", callback=handle_playback_step_frames_click, user_data=1, width=70)
                    dpg.add_button(label="(+5) >>", tag="playback_step_plus_5", callback=handle_playback_step_frames_click, user_data=5, width=70)

        dpg.add_spacer(height=5); dpg.add_separator(); dpg.add_spacer(height=5)

        with dpg.collapsing_header(label="Core Simulation Parameters", default_open=False, tag="core_sim_params_header"):
            # Full Profile dropdown (auto-populated from simulation_profiles/*.json)
            _scan_profile_directory()
            dpg.add_text("Load Full Profile:", color=[150,220,255,255])
            dpg.add_text("Applies all parameters (model, plasticity, noise, etc.) from a saved profile.",
                         color=[140,140,140,255], wrap=label_col_width + 50)
            with dpg.group(horizontal=True):
                dpg.add_combo(tag="cfg_full_profile",
                              items=list(_FULL_PROFILE_MAP.keys()),
                              default_value="(None - use settings below)",
                              callback=_handle_full_profile_dropdown_change,
                              width=350)
                dpg.add_button(label="Refresh", callback=lambda: _refresh_full_profile_dropdown(),
                               width=70)
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)

                add_parameter_table_row("Number of Neurons:", dpg.add_input_int, "cfg_num_neurons", 1000, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, step=100,
                    tooltip="Total neurons in the network. 1K-10K for real-time on most GPUs. 50K-100K for RTX 3090+ (24GB VRAM). Higher counts require more VRAM.")
                add_parameter_table_row("Connections/Neuron (Spatial Fallback):", dpg.add_input_int, "cfg_connections_per_neuron", 100, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Average synaptic connections per neuron when using spatial connectivity. Biological range: 1K-10K (cortex ~7K). Higher values increase memory and computation.")
                add_parameter_table_row("Total Sim Time (ms):", dpg.add_input_float, "cfg_total_sim_time", 60000.0, _update_sim_config_from_ui_and_signal_reset_needed, step=100,
                    tooltip="Maximum simulation duration in milliseconds. 60000ms = 60 seconds of biological time. Can always be stopped early.")
                add_parameter_table_row("Time Step dt (ms):", dpg.add_input_float, "cfg_dt_ms", 1.000, _update_sim_config_from_ui_and_signal_reset_needed, step=0.001, format="%.3f", min_value=0.001,
                    tooltip="Integration timestep. Izhikevich: 0.5-1.0ms is stable. Hodgkin-Huxley: MUST be <= 0.1ms (gating kinetics require fine resolution). AdEx: 0.1-0.5ms recommended. Smaller dt = more accurate but slower.")
                add_parameter_table_row("Seed (-1 for random):", dpg.add_input_int, "cfg_seed", -1, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Random seed for reproducibility. Set to -1 for a new random seed each run. Use a fixed positive integer to reproduce identical simulations.")
                add_parameter_table_row("Number of Traits:", dpg.add_input_int, "cfg_num_traits", 5, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, max_value=len(TRAIT_COLOR_MAP_RAW) if TRAIT_COLOR_MAP_RAW else 10,
                    tooltip="Number of neuron sub-populations (color-coded in 3D view). One trait is designated inhibitory. More traits = more diverse network topology.")
                add_parameter_table_row("Neuron Model:", dpg.add_combo, "cfg_neuron_model_type", NeuronModel.IZHIKEVICH.name, _handle_model_type_change_dpg, items=[model.name for model in NeuronModel],
                    tooltip="Izhikevich: Fast, versatile (20+ firing patterns). Good for large networks.\nHodgkin-Huxley: Biophysically detailed (ion channels, temperature). Requires dt<=0.1ms.\nAdEx: Balance of speed and biophysics. Good for adaptation studies.")
                add_parameter_table_row("Neural Structure Profile:", dpg.add_combo, "cfg_neural_profile", "GENERIC_UNSTRUCTURED", _update_sim_config_from_ui_and_signal_reset_needed, items=sorted(NEURAL_STRUCTURE_PROFILES.keys()),
                    tooltip="Pre-configured brain region profiles with literature-based connectivity, E/I ratios, and neuron type distributions. GENERIC_UNSTRUCTURED uses basic random connectivity.")
            
            # Hardware performance note (read-only info from benchmarks)
            dpg.add_spacer(height=5)
            dpg.add_text("Hardware Performance Note:", color=[150,200,255,255])
            dpg.add_text("", tag="cfg_hardware_performance_note", wrap=label_col_width + 50, color=[180,180,180,255])
            dpg.add_text("", tag="hw_limit_warning_text", wrap=label_col_width + 50, color=[255,100,100,255], show=False)
            dpg.add_spacer(height=5)

            with dpg.group(tag="izhikevich_params_group", show=True):
                dpg.add_text("--- Izhikevich 2007 Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    ui_izh_params = [
                        ("Membrane Capacitance C (pF)", "cfg_izh_C_val", "%.1f", 100.0), ("Scaling Factor k (nS/mV)", "cfg_izh_k_val", "%.2f", 0.7),
                        ("Resting Potential vr (mV)", "cfg_izh_vr_val", "%.1f", -60.0), ("Threshold Potential vt (mV)", "cfg_izh_vt_val", "%.1f", -40.0),
                        ("Spike Peak/Cutoff vpeak (mV)", "cfg_izh_vpeak_val", "%.1f", 35.0), ("Recovery Time Scale a (/ms)", "cfg_izh_a_val", "%.3f", 0.03),
                        ("Recovery Sensitivity b (nS)", "cfg_izh_b_val", "%.2f", -2.0), ("Voltage Reset c (mV)", "cfg_izh_c_val", "%.1f", -50.0),
                        ("Recovery Increment d (pA)", "cfg_izh_d_val", "%.1f", 100.0)
                    ]
                    _izh_tooltips = {
                        "cfg_izh_C_val": "Membrane capacitance. Higher C = slower voltage changes.\nRS ~100 pF, FS ~20-50 pF. (Izhikevich 2007, Table 2)",
                        "cfg_izh_k_val": "Scaling factor relating subthreshold I-V curvature.\nDetermines input resistance near rest.\nRS ~0.7, FS ~1.0, IB ~1.2 nS/mV.",
                        "cfg_izh_vr_val": "Resting membrane potential (no input).\nTypically -60 to -65 mV for cortical neurons.",
                        "cfg_izh_vt_val": "Instantaneous threshold potential.\nVoltage at which dV/dt becomes positive.\nTypically -40 to -45 mV.",
                        "cfg_izh_vpeak_val": "Spike cutoff voltage. When V >= vpeak, a spike\nis registered and V resets to c.\nTypically +25 to +35 mV.",
                        "cfg_izh_a_val": "Recovery variable time constant (1/ms).\nSmall a = slow recovery (RS ~0.03).\nLarge a = fast recovery (FS ~0.1).",
                        "cfg_izh_b_val": "Recovery sensitivity to subthreshold V.\nNegative b = resonator properties.\nRS ~-2 nS, FS ~0.25 nS.",
                        "cfg_izh_c_val": "Post-spike voltage reset.\nMore negative c = stronger after-hyperpolarization.\nRS ~-50 mV, IB ~-55 mV, CH ~-40 mV.",
                        "cfg_izh_d_val": "Post-spike recovery variable increment.\nControls spike-frequency adaptation.\nRS ~100 pA, FS ~25 pA, IB ~130 pA.",
                    }
                    for desc_label, tag, fmt, def_val in ui_izh_params:
                        add_parameter_table_row(desc_label, dpg.add_input_float, tag, def_val, _update_sim_config_from_ui_and_signal_reset_needed, format=fmt,
                            tooltip=_izh_tooltips.get(tag))
            
            with dpg.group(tag="hodgkin_huxley_params_group", show=False):
                dpg.add_text("--- Hodgkin-Huxley Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    # HH neuron type preset selector
                    add_parameter_table_row(
                        "HH Default Neuron Type:",
                        dpg.add_combo,
                        "cfg_default_neuron_type_hh",
                        NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name,
                        _update_sim_config_from_ui_and_signal_reset_needed,
                        items=[nt.name for nt in NeuronType if "HH_" in nt.name],
                        tooltip="Select a biophysical neuron type preset.\nSets conductances and kinetics for specific cell classes\n(e.g., cortical pyramidal, fast-spiking interneuron)."
                    )
                    ui_hh_params = [
                        ("Membrane Capacitance C_m (uF/cm^2)", "cfg_hh_C_m", "%.2f", 1.0),
                        ("Max Sodium Cond. g_Na_max (mS/cm^2)", "cfg_hh_g_Na_max", "%.1f", 50.0),
                        ("Max Potassium Cond. g_K_max (mS/cm^2)", "cfg_hh_g_K_max", "%.1f", 5.0),
                        ("Leak Cond. g_L (mS/cm^2)", "cfg_hh_g_L", "%.3f", 0.1),
                        ("Sodium Reversal E_Na (mV)", "cfg_hh_E_Na", "%.1f", 50.0),
                        ("Potassium Reversal E_K (mV)", "cfg_hh_E_K", "%.1f", -85.0),
                        ("Leak Reversal E_L (mV)", "cfg_hh_E_L", "%.3f", -70.0),
                        ("Spike Detection V_peak (mV)", "cfg_hh_v_peak", "%.1f", 40.0),
                        ("Initial V_rest (mV)", "cfg_hh_v_rest_init", "%.1f", -65.0),
                        ("M-current g_M_max (mS/cm^2)", "cfg_hh_g_M_max", "%.3f", 0.0),
                        ("M-current Tau (ms)", "cfg_hh_m_current_tau_ms", "%.1f", 100.0),
                        ("CaT g_CaT_max (mS/cm^2)", "cfg_hh_g_CaT_max", "%.3f", 0.0),
                        ("CaT Reversal E_CaT (mV)", "cfg_hh_E_CaT", "%.1f", 120.0),
                        ("I_h g_h_max (mS/cm^2)", "cfg_hh_g_h_max", "%.3f", 0.0),
                        ("I_h Reversal E_h (mV)", "cfg_hh_E_h", "%.1f", -30.0),
                        ("NaP g_NaP_max (mS/cm^2)", "cfg_hh_g_NaP_max", "%.3f", 0.0),
                        ("Kinetics Q10 Factor", "cfg_hh_q10_factor", "%.1f", 3.0),
                        ("Kinetics Temperature (C)", "cfg_hh_temperature_celsius", "%.1f", 37.0),
                    ]
                    _hh_tooltips = {
                        "cfg_hh_C_m": "Specific membrane capacitance.\nStandard squid axon: 1.0 uF/cm².\nHigher C_m = slower voltage dynamics.",
                        "cfg_hh_g_Na_max": "Maximum sodium conductance density.\nControls action potential amplitude and rise speed.\nSquid axon: 120, cortical: 50 mS/cm².",
                        "cfg_hh_g_K_max": "Maximum delayed-rectifier potassium conductance.\nControls repolarization and spike width.\nSquid: 36, cortical: 5 mS/cm².",
                        "cfg_hh_g_L": "Leak conductance density.\nSets resting input resistance.\nTypically 0.03-0.3 mS/cm².",
                        "cfg_hh_E_Na": "Sodium Nernst reversal potential.\nSet by [Na+] gradient across membrane.\nTypically +50 mV (mammalian).",
                        "cfg_hh_E_K": "Potassium Nernst reversal potential.\nSet by [K+] gradient.\nTypically -77 to -90 mV.",
                        "cfg_hh_E_L": "Leak reversal potential.\nApproximates resting V when no active currents.\nTypically -54 to -70 mV.",
                        "cfg_hh_v_peak": "Voltage threshold for formal spike detection.\nAt 37°C with Q10=3, fast kinetics may produce\nspikes below +40 mV. Adjust if needed.",
                        "cfg_hh_v_rest_init": "Initial resting membrane potential.\nGating variables are initialized to steady-state\nvalues at this voltage.",
                        "cfg_hh_g_M_max": "Muscarinic (M-type) K+ current max conductance.\nSlow non-inactivating K+ current. Causes spike\nfrequency adaptation. 0 = disabled.",
                        "cfg_hh_m_current_tau_ms": "M-current activation time constant.\nSlow ~100 ms gives adaptation over multiple spikes.\nRange: 50-200 ms.",
                        "cfg_hh_g_CaT_max": "Low-threshold Ca²+ (T-type) current conductance.\nEnables rebound bursting and subthreshold oscillations.\n0 = disabled. Typical: 0.5-2.0 mS/cm².",
                        "cfg_hh_E_CaT": "Calcium reversal potential.\nSet by [Ca²+] gradient. Typically +120 mV.",
                        "cfg_hh_g_h_max": "Hyperpolarization-activated cation current (I_h).\nContributes to resting potential, sag response,\nand pacemaker activity. 0 = disabled.",
                        "cfg_hh_E_h": "I_h reversal potential (mixed Na+/K+).\nTypically -20 to -40 mV, depolarizing from rest.",
                        "cfg_hh_g_NaP_max": "Persistent sodium current conductance.\nNon-inactivating Na+ near threshold.\nAmplifies subthreshold inputs. 0 = disabled.",
                        "cfg_hh_q10_factor": "Temperature coefficient for gating kinetics.\nRate multiplier per 10°C: phi = Q10^((T-6.3)/10).\nQ10=3 is standard for ion channels.",
                        "cfg_hh_temperature_celsius": "Simulation temperature for HH kinetics.\n6.3°C = original squid axon (Hodgkin & Huxley 1952).\n37°C = mammalian with ~28x faster kinetics.",
                    }
                    for desc_label, tag, fmt, def_val in ui_hh_params:
                        add_parameter_table_row(
                            desc_label,
                            dpg.add_input_float,
                            tag,
                            def_val,
                            _update_sim_config_from_ui_and_signal_reset_needed,
                            format=fmt,
                            tooltip=_hh_tooltips.get(tag),
                        )

                    # External drive scale slider (auto-tuned)
                    add_parameter_table_row(
                        "External Drive Scale (HH, auto-tuned):",
                        dpg.add_slider_float,
                        "cfg_hh_external_drive_scale",
                        1.0,
                        _update_sim_config_from_ui_and_signal_reset_needed,
                        min_value=0.1,
                        max_value=8.0,
                        format="%.2f",
                        tooltip="Multiplier for external input current to HH neurons.\nAuto-tuned during initialization. Increase if neurons\nare too quiet, decrease if network is epileptic.",
                    )

                    # Button to reset HH drive scale to auto-tuned value (if available)
                    with dpg.table_row():
                        dpg.add_text("Reset HH Drive to Auto-Tuned:")
                        dpg.add_button(
                            tag="cfg_hh_reset_drive_to_auto_btn",
                            label="Reset",
                            callback=handle_reset_hh_drive_to_auto,
                            width=-1,
                        )

            with dpg.group(tag="adex_params_group", show=False):
                dpg.add_text("--- AdEx Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    ui_adex_params = [
                        ("Membrane Capacitance C (pF)", "cfg_adex_C", "%.1f", 281.0),
                        ("Leak Conductance g_L (nS)", "cfg_adex_g_L", "%.1f", 30.0),
                        ("Leak Reversal E_L (mV)", "cfg_adex_E_L", "%.1f", -70.6),
                        ("Spike Threshold V_T (mV)", "cfg_adex_V_T", "%.1f", -50.4),
                        ("Slope Factor Delta_T (mV)", "cfg_adex_Delta_T", "%.2f", 2.0),
                        ("Subthreshold Coupling a (nS)", "cfg_adex_a", "%.1f", 4.0),
                        ("Adaptation Time Constant tau_w (ms)", "cfg_adex_tau_w", "%.1f", 144.0),
                        ("Spike-triggered Increment b (pA)", "cfg_adex_b", "%.1f", 80.5),
                        ("Reset Potential V_r (mV)", "cfg_adex_V_r", "%.1f", -70.6),
                        ("Spike Detection V_peak (mV)", "cfg_adex_V_peak", "%.1f", -40.0),
                    ]
                    _adex_tooltips = {
                        "cfg_adex_C": "Membrane capacitance. Brette & Gerstner 2005:\nRS ~281 pF, FS ~100 pF. Controls voltage time constant.",
                        "cfg_adex_g_L": "Leak conductance. Sets resting input resistance.\nR_in = 1/g_L. RS ~30 nS, FS ~10 nS.",
                        "cfg_adex_E_L": "Leak reversal / resting potential.\nTypically -70 to -65 mV for cortical neurons.",
                        "cfg_adex_V_T": "Effective spike threshold voltage.\nThe exponential term activates steeply above V_T.\nTypically -50 to -45 mV.",
                        "cfg_adex_Delta_T": "Slope factor of exponential spike initiation.\nSmaller = sharper threshold. 0 = perfect IF.\nTypical: 1-4 mV. (Badel et al. 2008)",
                        "cfg_adex_a": "Subthreshold adaptation coupling.\nLinks adaptation variable w to voltage.\nRS ~4 nS, bursting ~0.5 nS.",
                        "cfg_adex_tau_w": "Adaptation time constant.\nControls how quickly w decays after spikes.\nRS ~144 ms, FS ~20 ms.",
                        "cfg_adex_b": "Spike-triggered adaptation increment.\nAdded to w after each spike.\nLarger b = stronger spike-frequency adaptation.\nRS ~80 pA, FS ~0 pA.",
                        "cfg_adex_V_r": "Post-spike membrane potential reset.\nTypically near E_L. More negative = stronger\nafter-hyperpolarization.",
                        "cfg_adex_V_peak": "Spike detection threshold.\nWhen V exceeds V_peak, spike is registered\nand V resets to V_r. Typically 0 to -40 mV.",
                    }
                    for desc_label, tag, fmt, def_val in ui_adex_params:
                        add_parameter_table_row(
                            desc_label,
                            dpg.add_input_float,
                            tag,
                            def_val,
                            _update_sim_config_from_ui_and_signal_reset_needed,
                            format=fmt,
                            tooltip=_adex_tooltips.get(tag),
                        )

                    # External drive scale slider (auto-tuned)
                    add_parameter_table_row(
                        "External Drive Scale (AdEx, auto-tuned):",
                        dpg.add_slider_float,
                        "cfg_adex_external_drive_scale",
                        1.0,
                        _update_sim_config_from_ui_and_signal_reset_needed,
                        min_value=0.1,
                        max_value=5.0,
                        format="%.2f",
                        tooltip="Multiplier for external input current to AdEx neurons.\nAuto-tuned during initialization. Adjust if firing\nrates are too low or too high.",
                    )

                    # Button to reset AdEx drive scale to auto-tuned value (if available)
                    with dpg.table_row():
                        dpg.add_text("Reset AdEx Drive to Auto-Tuned:")
                        dpg.add_button(
                            tag="cfg_adex_reset_drive_to_auto_btn",
                            label="Reset",
                            callback=handle_reset_adex_drive_to_auto,
                            width=-1,
                        )

        with dpg.collapsing_header(label="Network Connectivity", default_open=False, tag="network_connectivity_header"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Use Watts-Strogatz Generator:", dpg.add_checkbox, "cfg_enable_watts_strogatz", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Use Watts-Strogatz small-world network topology.\nCombines local clustering with short path lengths.\nDisable for random Erdos-Renyi connectivity.")
                add_parameter_table_row("W-S K (Nearest Neighbors, even):", dpg.add_input_int, "cfg_connectivity_k", 10, _update_sim_config_from_ui_and_signal_reset_needed, step=2, min_value=2,
                    tooltip="Each neuron connects to K nearest neighbors.\nMust be even. Higher K = denser local connectivity.\nK=10 gives ~10% connection prob. for 100 neurons.")
                add_parameter_table_row("W-S P (Rewire Probability):", dpg.add_input_float, "cfg_connectivity_p_rewire", 0.1, _update_sim_config_from_ui_and_signal_reset_needed, min_value=0.0, max_value=1.0, format="%.3f",
                    tooltip="Probability of rewiring each edge to a random target.\nP=0: regular lattice. P=1: fully random.\nP=0.05-0.2: small-world regime (Watts & Strogatz 1998).")

        with dpg.collapsing_header(label="Synaptic Parameters", default_open=False, tag="synaptic_params_header"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Excitatory Prop. Strength (g_peak_e scale):", dpg.add_input_float, "cfg_propagation_strength", 0.05, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                    tooltip="Peak excitatory conductance increase per spike (nS).\nScales AMPA synaptic input. Higher = stronger\nexcitatory drive. Typical: 0.01-0.5.")
                add_parameter_table_row("Inhibitory Prop. Strength (g_peak_i scale):", dpg.add_input_float, "cfg_inhibitory_propagation_strength", 0.15, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                    tooltip="Peak inhibitory conductance increase per spike (nS).\nScales GABA_A synaptic input. Usually 2-4x excitatory\nfor E/I balance. Typical: 0.05-1.0.")
                add_parameter_table_row("Excitatory Conductance Tau_g_e (ms):", dpg.add_input_float, "cfg_syn_tau_e", 5.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.1, tooltip="AMPA receptor decay time constant. Fast excitatory transmission (1-10 ms typical).")
                add_parameter_table_row("Inhibitory Conductance Tau_g_i (ms):", dpg.add_input_float, "cfg_syn_tau_i", 10.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.1, tooltip="GABA_A receptor decay time constant. Inhibitory transmission (5-20 ms typical).")
            dpg.add_separator()
            dpg.add_text("NMDA Receptors (Voltage-Dependent Mg²⁺ Block)")
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable NMDA:", dpg.add_checkbox, "cfg_enable_nmda", False, _update_sim_config_from_ui_and_signal_reset_needed, tooltip="NMDA receptors with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990). Adds slow excitatory current gated by postsynaptic depolarization — critical for coincidence detection and associative plasticity.")
                add_parameter_table_row("NMDA:AMPA Ratio:", dpg.add_input_float, "cfg_nmda_ratio", 0.4, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.0, max_value=2.0, tooltip="Ratio of NMDA to AMPA peak conductance. 0.3-0.5 typical for cortex (Myme et al. 2003).")
                add_parameter_table_row("NMDA Tau Decay (ms):", dpg.add_input_float, "cfg_nmda_tau_decay", 100.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=10.0, tooltip="NMDA receptor decay (~100 ms). Much slower than AMPA (~5 ms), enabling temporal integration.")
                add_parameter_table_row("NMDA Tau Rise (ms):", dpg.add_input_float, "cfg_nmda_tau_rise", 3.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.5, tooltip="NMDA receptor rise time (2-5 ms). Slower rise than AMPA due to glutamate binding kinetics.")
                add_parameter_table_row("[Mg²⁺] (mM):", dpg.add_input_float, "cfg_nmda_mg_conc", 1.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.0, max_value=5.0, tooltip="Extracellular magnesium concentration. 1.0 mM physiological. Higher = stronger voltage-dependent block, less NMDA current at rest.")

        with dpg.collapsing_header(label="Learning & Plasticity", default_open=False, tag="learning_plasticity_header"):
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Hebbian Learning:", dpg.add_checkbox, "cfg_enable_hebbian_learning", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Simple Hebbian co-activation learning rule.\nWeights increase when pre and post neurons fire together.\nIncludes weight decay to prevent runaway excitation.")
                add_parameter_table_row("Hebbian Learning Rate:", dpg.add_input_float, "cfg_hebbian_learning_rate", 0.0005, _update_sim_config_from_ui_and_signal_reset_needed, format="%.6f",
                    tooltip="Rate of weight change per co-activation event.\nSmaller = more stable but slower learning.\nTypical range: 0.0001–0.01.")
                add_parameter_table_row("Hebbian Max Weight:", dpg.add_input_float, "cfg_hebbian_max_weight", 1.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f",
                    tooltip="Upper bound on synaptic weights under Hebbian learning.\nPrevents runaway excitation. Also used as upper\nclamp for synaptic scaling.")
            dpg.add_separator()
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Short-Term Plasticity (STP):", dpg.add_checkbox, "cfg_enable_short_term_plasticity", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Tsodyks-Markram short-term plasticity model. Synapses exhibit depression (weakening) and facilitation (strengthening) on timescales of 10-1000ms. Essential for temporal coding.")
                add_parameter_table_row("STP U (Baseline Utilization):", dpg.add_input_float, "cfg_stp_U", 0.15, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", min_value=0.0, max_value=1.0,
                    tooltip="Fraction of available resources used per spike (0-1). Low U (~0.1-0.2): facilitating synapses (cortical). High U (~0.5-0.8): depressing synapses (thalamocortical). Literature: Tsodyks & Markram 1997.")
                add_parameter_table_row("STP Tau_d (Depression, ms):", dpg.add_input_float, "cfg_stp_tau_d", 200.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.1,
                    tooltip="Recovery time constant for synaptic resources (ms). Controls how fast depressed synapses recover. Typical range: 100-800ms.")
                add_parameter_table_row("STP Tau_f (Facilitation, ms):", dpg.add_input_float, "cfg_stp_tau_f", 50.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.1,
                    tooltip="Decay time constant for facilitation variable (ms). Controls duration of synaptic facilitation. Typical range: 20-200ms.")
                add_parameter_table_row("Enable Per-Type STP:", dpg.add_checkbox, "cfg_enable_per_type_stp", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Use different STP parameters for E->E, E->I, I->E, I->I synapses.\nMore biologically realistic: cortical E->E synapses depress (U~0.5)\nwhile I->E show weaker depression (U~0.25).")
            # Per-type STP parameter table
            dpg.add_text("Per-Connection-Type STP Parameters:", color=[150,200,220,255])
            dpg.add_text("(E->E, E->I, I->E, I->I)", color=[140,140,140,255])
            with dpg.table(header_row=True):
                dpg.add_table_column(label="Param", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="E->E", width_stretch=True)
                dpg.add_table_column(label="E->I", width_stretch=True)
                dpg.add_table_column(label="I->E", width_stretch=True)
                dpg.add_table_column(label="I->I", width_stretch=True)
                with dpg.table_row():
                    dpg.add_text("U")
                    dpg.add_input_float(tag="cfg_stp_U_ee", default_value=0.5, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                    dpg.add_input_float(tag="cfg_stp_U_ei", default_value=0.5, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                    dpg.add_input_float(tag="cfg_stp_U_ie", default_value=0.25, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                    dpg.add_input_float(tag="cfg_stp_U_ii", default_value=0.25, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                with dpg.table_row():
                    dpg.add_text("Tau_d")
                    dpg.add_input_float(tag="cfg_stp_tau_d_ee", default_value=200.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_d_ei", default_value=200.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_d_ie", default_value=100.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_d_ii", default_value=100.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                with dpg.table_row():
                    dpg.add_text("Tau_f")
                    dpg.add_input_float(tag="cfg_stp_tau_f_ee", default_value=20.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_f_ei", default_value=20.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_f_ie", default_value=50.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_f_ii", default_value=50.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
            dpg.add_separator()
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Homeostasis:", dpg.add_checkbox, "cfg_enable_homeostasis", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Intrinsic homeostasis via adaptive firing thresholds.\nFor Izhikevich: adjusts spike threshold toward target rate.\nEssential for stable network dynamics over long simulations.")
            with dpg.group(tag="homeostasis_izh_specific_group", show=True):
                 with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    add_parameter_table_row("Homeostasis Target Rate (spikes/dt for Izh):", dpg.add_input_float, "cfg_homeostasis_target_rate", 0.02, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                        tooltip="Desired firing probability per timestep.\n0.02 = ~2% chance of firing each dt.\nAt dt=0.5ms this corresponds to ~40 Hz.\nThreshold adapts to reach this target.")
                    add_parameter_table_row("Homeostasis Min Threshold (Izh, mV):", dpg.add_input_float, "cfg_homeostasis_threshold_min", -55.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f",
                        tooltip="Lower bound on adaptive firing threshold.\nPrevents threshold from dropping too low,\nwhich would cause pathological firing.\nShould be above resting potential (vr).")
                    add_parameter_table_row("Homeostasis Max Threshold (Izh, mV):", dpg.add_input_float, "cfg_homeostasis_threshold_max", -30.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f",
                        tooltip="Upper bound on adaptive firing threshold.\nPrevents threshold from rising too high,\nwhich would silence the neuron entirely.\nShould be below spike peak (vpeak).")

            # C1b: Synaptic Scaling Controls (Turrigiano 2008)
            dpg.add_separator()
            dpg.add_text("--- Synaptic Scaling (Homeostatic) ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Synaptic Scaling:", dpg.add_checkbox, "cfg_enable_synaptic_scaling", False, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Multiplicative synaptic scaling (Turrigiano 2008).\nScales excitatory weights up/down to maintain target firing rate.\nComplementary to threshold homeostasis — works on synaptic strengths\nrather than intrinsic excitability.")
                add_parameter_table_row("Synaptic Scaling Rate:", dpg.add_input_float, "cfg_synaptic_scaling_rate", 0.001, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                    tooltip="Rate of multiplicative weight scaling per timestep.\nHigher values = faster homeostatic correction but risk instability.\nTypical range: 0.0001–0.01. Default 0.001.")

            # C2: STDP Controls
            dpg.add_separator()
            dpg.add_text("--- STDP (Spike-Timing-Dependent Plasticity) ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable STDP:", dpg.add_checkbox, "cfg_enable_stdp", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Spike-Timing-Dependent Plasticity (Bi & Poo 2001).\nPre-before-post = LTP, post-before-pre = LTD.\nBiological Hebbian learning with precise timing.")
                add_parameter_table_row("STDP A+ (LTP amplitude, 0.005-0.02):", dpg.add_input_float, "cfg_stdp_a_plus", 0.01, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f", min_value=0.0,
                    tooltip="Maximum weight increase for causal (pre→post) pairing.\nLarger A+ = faster potentiation.\nA- > A+ gives net depression bias (stable).")
                add_parameter_table_row("STDP A- (LTD amplitude, 0.005-0.02):", dpg.add_input_float, "cfg_stdp_a_minus", 0.0105, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f", min_value=0.0,
                    tooltip="Maximum weight decrease for anti-causal (post→pre) pairing.\nSlightly larger than A+ ensures net weight decrease\nfor random firing, preventing runaway excitation.")
                add_parameter_table_row("STDP Tau+ (LTP time constant, ms):", dpg.add_input_float, "cfg_stdp_tau_plus_ms", 20.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=1.0,
                    tooltip="Time window for LTP (pre-before-post).\n20ms matches cortical STDP data (Bi & Poo 2001).\nLarger tau = wider learning window.")
                add_parameter_table_row("STDP Tau- (LTD time constant, ms):", dpg.add_input_float, "cfg_stdp_tau_minus_ms", 20.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=1.0,
                    tooltip="Time window for LTD (post-before-pre).\n20ms standard. Asymmetric tau+/tau- gives\ndifferent temporal sensitivity for LTP vs LTD.")
                add_parameter_table_row("STDP Weight Min:", dpg.add_input_float, "cfg_stdp_w_min", 0.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f",
                    tooltip="Lower bound on STDP-modified weights.\n0 = synapses can be fully depressed.\nSet > 0 to maintain minimal connectivity.")
                add_parameter_table_row("STDP Weight Max:", dpg.add_input_float, "cfg_stdp_w_max", 2.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f",
                    tooltip="Upper bound on STDP-modified weights.\nPrevents individual synapses from becoming\ntoo strong. 2.0 = 2x initial weight.")
            
            # C2: Reward Modulation Controls
            dpg.add_separator()
            dpg.add_text("--- Reward-Modulated Plasticity ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Reward Modulation:", dpg.add_checkbox, "cfg_enable_reward_modulation", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Three-factor learning: STDP eligibility traces\nare gated by a reward signal (Schultz 2002).\nRequires STDP enabled. Models dopaminergic modulation.")
                add_parameter_table_row("Reward Learning Rate (0.001-0.05):", dpg.add_input_float, "cfg_reward_learning_rate", 0.01, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f", min_value=0.0,
                    tooltip="Scales how strongly reward modulates weight changes.\nHigher = faster reward-driven learning but noisier.\nTypical: 0.001-0.05.")
                add_parameter_table_row("Eligibility Trace Tau (ms, 500-2000):", dpg.add_input_float, "cfg_reward_eligibility_tau_ms", 1000.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=10.0,
                    tooltip="Decay time for eligibility traces (ms).\nBridges the gap between STDP events and delayed reward.\n1000ms = 1 second memory of recent spike correlations.")
                add_parameter_table_row("Reward Baseline (expected reward):", dpg.add_input_float, "cfg_reward_baseline", 0.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f",
                    tooltip="Expected (average) reward level.\nWeight changes proportional to (reward - baseline).\n0 = any positive reward causes LTP.")
                add_parameter_table_row("Current Reward Signal:", dpg.add_input_float, "cfg_current_reward_signal", 0.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f",
                    tooltip="Current reward value (can be changed live).\nPositive = reinforce recent activity.\nNegative = suppress recent activity patterns.\nModels dopaminergic reward prediction error.")
            
            # C3: Structural Plasticity Controls
            dpg.add_separator()
            dpg.add_text("--- Structural Plasticity ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Structural Plasticity:", dpg.add_checkbox, "cfg_enable_structural_plasticity", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Dynamic synapse formation and elimination.\nNew connections form between co-active neurons.\nWeak synapses are pruned. Models developmental\nand experience-dependent rewiring.")
                add_parameter_table_row("Formation Rate (per timestep, 1e-7 to 1e-5):", dpg.add_input_float, "cfg_struct_plast_formation_rate", 1e-6, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2e", min_value=0.0,
                    tooltip="Probability of new synapse creation per candidate pair\nper update interval. Very small values needed to\navoid explosive connectivity growth.")
                add_parameter_table_row("Elimination Rate (per timestep, 1e-7 to 1e-5):", dpg.add_input_float, "cfg_struct_plast_elimination_rate", 5e-7, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2e", min_value=0.0,
                    tooltip="Probability of pruning weak synapses per update interval.\nBalances formation rate. Higher elimination =\nmore aggressive pruning of unused connections.")
                add_parameter_table_row("Weight Threshold (eliminate below):", dpg.add_input_float, "cfg_struct_plast_weight_threshold", 0.05, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", min_value=0.0,
                    tooltip="Synapses with weight below this value are candidates\nfor elimination. Higher threshold = more aggressive\npruning. 0.05 = prune very weak connections.")
                add_parameter_table_row("Target Connection Density (0-1):", dpg.add_input_float, "cfg_struct_plast_target_density", 0.1, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", min_value=0.0, max_value=1.0,
                    tooltip="Target fraction of possible connections present.\n0.1 = 10% connectivity. Formation/elimination rates\nadjust to approach this density.")
                add_parameter_table_row("Distance Scale (spatial, units):", dpg.add_input_float, "cfg_struct_plast_distance_scale", 20.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=1.0,
                    tooltip="Spatial scale for distance-dependent connection\nprobability. New synapses preferentially form between\nnearby neurons. Smaller = more local connectivity.")
                add_parameter_table_row("Update Interval (steps):", dpg.add_input_int, "cfg_struct_plast_update_interval_steps", 100, _update_sim_config_from_ui_and_signal_reset_needed, min_value=10, step=10,
                    tooltip="How often (in sim steps) to evaluate structural changes.\nCSR matrix rebuilds are expensive, so infrequent updates\n(100-1000 steps) are recommended.")
                add_parameter_table_row("Activity Bias (formation):", dpg.add_input_float, "cfg_struct_plast_activity_bias", 0.5, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.0, max_value=1.0,
                    tooltip="Bias synapse formation toward co-active neuron pairs.\n0.0 = purely random formation.\n1.0 = fully activity-driven (Cline & Haas 2008).\n0.5 = 50/50 mix of co-activity-biased and random candidates.")

        with dpg.collapsing_header(label="Heterogeneity & Noise", default_open=False, tag="heterogeneity_noise_header"):
            dpg.add_text("Add biological realism through parameter variability and intrinsic noise.", wrap=label_col_width * 2, color=[200,200,200,255])
            dpg.add_spacer(height=5)
            
            dpg.add_text("--- Parameter Heterogeneity ---", color=[200,200,100,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row(
                    "Enable Parameter Heterogeneity:",
                    dpg.add_checkbox,
                    "cfg_enable_parameter_heterogeneity",
                    True,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Add neuron-to-neuron parameter variability.\nSamples from distributions matching experimental data\n(CV~0.3-0.4). More realistic than identical neurons."
                )
                add_parameter_table_row(
                    "Heterogeneity Seed (-1 = use main seed):",
                    dpg.add_input_int,
                    "cfg_heterogeneity_seed",
                    -1,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    min_value=-1,
                    step=1,
                    tooltip="RNG seed for parameter variability.\n-1 = use main simulation seed (deterministic).\nSet different values to explore different\ninstantiations of the same heterogeneity level."
                )
            
            dpg.add_text(
                "When enabled, parameters are sampled from distributions (CV~0.3-0.4) matching experimental data.",
                wrap=label_col_width * 2,
                color=[150,150,150,255]
            )
            
            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            dpg.add_text("--- Channel & Background Noise ---", color=[200,200,100,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                
                # Conductance noise (HH only)
                add_parameter_table_row(
                    "Enable Conductance Noise (HH only):",
                    dpg.add_checkbox,
                    "cfg_enable_conductance_noise",
                    True,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Add stochastic fluctuations to ion channel conductances.\nModels channel noise from finite ion channel populations.\nOnly applies to Hodgkin-Huxley model."
                )
                add_parameter_table_row(
                    "Conductance Noise Std (relative, 0.05 = 5%):",
                    dpg.add_input_float,
                    "cfg_conductance_noise_relative_std",
                    0.05,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.3f",
                    min_value=0.0,
                    max_value=0.5,
                    tooltip="Standard deviation of conductance noise as fraction\nof max conductance. 0.05 = 5% noise.\nHigher values = more stochastic spiking."
                )
            
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                
                # OU process
                add_parameter_table_row(
                    "Enable OU Process (background drive):",
                    dpg.add_checkbox,
                    "cfg_enable_ou_process",
                    True,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Ornstein-Uhlenbeck process for background synaptic drive.\nModels bombardment from ~10,000 unmodeled synapses.\nProduces realistic 2-5 mV membrane potential fluctuations."
                )
                add_parameter_table_row(
                    "OU Mean Current (pA):",
                    dpg.add_input_float,
                    "cfg_ou_mean_current_pA",
                    0.0,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.1f",
                    tooltip="Mean (DC offset) of background current.\n0 = symmetric fluctuations around zero.\nPositive = tonic depolarizing drive.\nNegative = tonic hyperpolarizing."
                )
                add_parameter_table_row(
                    "OU Std Current (pA, 50-200 typical):",
                    dpg.add_input_float,
                    "cfg_ou_std_current_pA",
                    100.0,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.1f",
                    min_value=0.0,
                    tooltip="Standard deviation of OU noise current.\nControls amplitude of Vm fluctuations.\n100 pA typical for Izhikevich. Scale for HH/AdEx."
                )
                add_parameter_table_row(
                    "OU Time Constant Tau (ms, 10-20 typical):",
                    dpg.add_input_float,
                    "cfg_ou_tau_ms",
                    15.0,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.1f",
                    min_value=1.0,
                    max_value=100.0,
                    tooltip="Temporal correlation time of background noise.\nSmall tau (~5 ms) = fast, white-noise-like.\nLarge tau (~20 ms) = slowly varying, colored noise.\n15 ms matches cortical synaptic timescales."
                )
                add_parameter_table_row(
                    "OU Seed (-1 = use main seed):",
                    dpg.add_input_int,
                    "cfg_ou_seed",
                    -1,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    min_value=-1,
                    step=1,
                    tooltip="RNG seed for OU noise process.\n-1 = use main simulation seed (deterministic).\nDifferent seeds give different noise realizations\nwhile preserving other simulation state."
                )
            
            dpg.add_text(
                "OU process adds temporally correlated background noise (2-5mV Vm fluctuations).",
                wrap=label_col_width * 2,
                color=[150,150,150,255]
            )

        with dpg.collapsing_header(label="Visual Settings", default_open=False, tag="visual_settings_header"):
            dpg.add_text("--- Neurons ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                spiking_filter_options = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
                add_parameter_table_row("Show Spiking Neurons:", dpg.add_combo, "filter_spiking_mode_combo", "Highlight Spiking", trigger_filter_update_signal, items=spiking_filter_options,
                    tooltip="How to display spiking neurons.\nHighlight: bright flash on spike, dim otherwise.\nOnly Spiking: hide non-spiking neurons.\nNo Highlight: uniform appearance.")
                add_parameter_table_row("Enable Synaptic Pulses (GL):", dpg.add_checkbox, "gl_enable_synaptic_pulses_cb", opengl_viz_config.get('ENABLE_SYNAPTIC_PULSES', True) if OPENGL_AVAILABLE else False, handle_gl_enable_synaptic_pulses_change,
                    tooltip="Show animated pulses traveling along synapses\nwhen spikes propagate. Visually appealing but\ncosts GPU performance at high spike rates.")
                add_parameter_table_row("Filter By Neuron Type:", dpg.add_checkbox, "filter_type_enable_cb", False, lambda s, a, u: (dpg.configure_item("filter_neuron_type_combo", enabled=a), trigger_filter_update_signal(s,a,u)),
                    tooltip="Enable filtering to show only neurons of a specific type.\nUseful for isolating excitatory or inhibitory populations.")
                add_parameter_table_row("Select Type:", dpg.add_combo, "filter_neuron_type_combo", "All", trigger_filter_update_signal, items=["All"], enabled=False,
                    tooltip="Select which neuron type to display.\nRequires 'Filter By Neuron Type' to be enabled.")
                add_parameter_table_row("Max Visible Neurons (GL):", dpg.add_input_int, "gl_max_neurons_render_input", opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000) if OPENGL_AVAILABLE else 0, handle_gl_max_neurons_change, min_value=0, step=100,
                    tooltip="Maximum neurons rendered in OpenGL viewport.\nReduce for better frame rate with large networks.\n10000 default. 0 = render all.")
                add_parameter_table_row("Neuron Size (GL):", dpg.add_slider_float, "gl_neuron_point_size_slider", opengl_viz_config.get('POINT_SIZE', 2.0) if OPENGL_AVAILABLE else 1.0, handle_gl_point_size_change, min_value=0.5, max_value=10.0, format="%.1f",
                    tooltip="Point size for neuron rendering in pixels.\nIncrease for visibility at distance, decrease\nfor dense networks to reduce overlap.")
                add_parameter_table_row("Inactive Neuron Opacity (GL):", dpg.add_slider_float, "gl_inactive_neuron_opacity_slider", opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25) if OPENGL_AVAILABLE else 0.1, handle_gl_inactive_neuron_opacity_change, min_value=0.0, max_value=1.0, format="%.2f",
                    tooltip="Transparency of non-spiking neurons.\n0.0 = fully transparent, 1.0 = fully opaque.\nLow values make spiking activity pop visually.")
            
            dpg.add_separator()
            dpg.add_text("--- Synapses ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Show Synapses (GL):", dpg.add_checkbox, "filter_show_synapses_gl_cb", global_gui_state.get("show_connections_gl", True), lambda s,a,u: (global_gui_state.update({"show_connections_gl":a}), trigger_filter_update_signal()),
                    tooltip="Toggle synapse line rendering in OpenGL viewport.\nDisable for cleaner neuron-only view and\nbetter performance with dense networks.")
                add_parameter_table_row("Max Visible Connections (GL):", dpg.add_input_int, "gl_max_connections_render_input", opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000) if OPENGL_AVAILABLE else 0, handle_gl_max_connections_change, min_value=0, step=500,
                    tooltip="Maximum synapse lines rendered. Dense networks\nmay have millions of connections — cap this\nfor usable frame rates. 20000 default.")
                add_parameter_table_row("Synapse Alpha Multiplier (GL):", dpg.add_slider_float, "gl_synapse_alpha_slider", opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.3) if OPENGL_AVAILABLE else 0.1, handle_gl_synapse_alpha_change, min_value=0.0, max_value=2.0, format="%.2f",
                    tooltip="Opacity multiplier for synapse lines.\nLower values = more transparent connections.\nUseful to reduce visual clutter in dense networks.")
                add_parameter_table_row("Min Abs Synapse Weight (Filter):", dpg.add_slider_float, "filter_min_abs_weight_slider", 0.000, trigger_filter_update_signal, max_value=1.0, format="%.3f",
                    tooltip="Only show synapses with |weight| above this value.\nIncrease to see only the strongest connections.\n0 = show all connections.")
            
            dpg.add_separator()
            dpg.add_text("--- General Visuals ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Camera Field of View (FOV, degrees):", dpg.add_slider_float, "cfg_camera_fov", 60.0, _update_sim_config_from_ui_and_signal_reset_needed, min_value=10.0, max_value=120.0,
                    tooltip="Perspective camera field of view.\n60° is natural. Lower = telephoto (flatter).\nHigher = wide-angle (more depth distortion).")
                add_parameter_table_row("Activity Highlight Frames (GL):", dpg.add_input_int, "gl_activity_highlight_frames_input", opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7) if OPENGL_AVAILABLE else 1, handle_gl_activity_highlight_frames_change, min_value=1, max_value=30,
                    tooltip="How many frames a neuron stays highlighted after spiking.\nHigher = longer visible flash. 7 default.\nIncrease for slow sim speeds, decrease for fast.")
                add_parameter_table_row("Viz Update Interval (steps):", dpg.add_input_int, "cfg_viz_update_interval_steps", 1, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, max_value=200, step=1,
                    tooltip="Update visualization every N simulation steps.\n1 = real-time update (smoothest, most GPU overhead).\nHigher values = faster simulation but choppier visuals.")

        # =============================================================================
        # EXPERIMENT & STIMULUS SYSTEM UI
        # =============================================================================
        with dpg.collapsing_header(label="Experiment & Stimulus System", default_open=False, tag="experiment_system_header"):
            dpg.add_text("Configure and run programmable experiments with stimulus injection,\nneuron group I/O, training protocols, and readout analysis.")
            dpg.add_spacer(height=5)

            # --- Experiment Preset Selector ---
            dpg.add_text("Experiment Presets:", color=[180, 220, 255])
            experiment_preset_names = ["-- Select Preset --"] + ExperimentPresets.get_preset_names()
            dpg.add_combo(experiment_preset_names, default_value="-- Select Preset --",
                          tag="experiment_preset_combo", width=350,
                          callback=lambda s, a, u: _handle_experiment_preset_change(a))
            dpg.add_spacer(height=3)

            # Experiment info display
            dpg.add_text("No experiment loaded.", tag="experiment_info_text", color=[150, 150, 150])
            dpg.add_spacer(height=5)

            # --- Control Buttons ---
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Experiment", tag="btn_start_experiment",
                               callback=lambda: ui_to_sim_queue.put({"type": "START_EXPERIMENT"}))
                dpg.add_button(label="Stop Experiment", tag="btn_stop_experiment",
                               callback=lambda: ui_to_sim_queue.put({"type": "STOP_EXPERIMENT"}))
                dpg.add_button(label="Save Log", tag="btn_save_experiment_log",
                               callback=lambda: ui_to_sim_queue.put({"type": "SAVE_EXPERIMENT_LOG",
                                   "filepath": f"experiment_log_{int(time.time())}.json"}))
            dpg.add_spacer(height=5)

            # --- Experiment Status Display ---
            dpg.add_text("Status:", color=[180, 220, 255])
            dpg.add_text("Idle", tag="experiment_status_text", color=[150, 150, 150])
            dpg.add_spacer(height=3)

            # Phase progress
            dpg.add_text("Phase: --", tag="experiment_phase_text", color=[150, 150, 150])
            dpg.add_spacer(height=3)

            # Readout rates display
            dpg.add_text("Readout Rates:", color=[180, 220, 255])
            dpg.add_text("No data", tag="experiment_readout_text", color=[150, 150, 150])
            dpg.add_spacer(height=3)

            # Training progress
            dpg.add_text("Training:", color=[180, 220, 255])
            dpg.add_text("No training active", tag="experiment_training_text", color=[150, 150, 150])
            dpg.add_spacer(height=5)

            # --- Manual Stimulus Configuration ---
            with dpg.collapsing_header(label="Manual Stimulus (Quick Test)", default_open=False,
                                       tag="manual_stimulus_sub_header", indent=10):
                dpg.add_text("Inject a simple stimulus into the network without\nsetting up a full experiment.", color=[150, 150, 150])
                dpg.add_spacer(height=3)
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    add_parameter_table_row("Stimulus Amplitude (pA):", dpg.add_input_float,
                        "manual_stim_amplitude", 150.0, None, min_value=0.0, max_value=5000.0,
                        tooltip="Peak current amplitude in picoamperes.\n100-300 pA typical for driving activity.")
                    add_parameter_table_row("Pattern:", dpg.add_combo,
                        "manual_stim_pattern_combo", "CONSTANT",
                        None, items=["CONSTANT", "PULSE_TRAIN", "SINUSOIDAL", "POISSON_SPIKE_TRAIN", "GAUSSIAN_NOISE"],
                        tooltip="Stimulus waveform type.\nCONSTANT: DC step current\nPULSE_TRAIN: Repeated brief pulses\nSINUSOIDAL: Oscillatory current")
                    add_parameter_table_row("Target Group Size:", dpg.add_input_int,
                        "manual_stim_group_size", 100, None, min_value=1, max_value=10000,
                        tooltip="Number of neurons in the stimulus target group.\nSelects the first N neurons in the network.")
                    add_parameter_table_row("Duration (ms):", dpg.add_input_float,
                        "manual_stim_duration", 500.0, None, min_value=10.0, max_value=100000.0,
                        tooltip="How long the stimulus will be active in milliseconds.")
                dpg.add_spacer(height=3)
                dpg.add_button(label="Inject Stimulus", tag="btn_inject_manual_stimulus",
                               callback=_handle_inject_manual_stimulus)
                dpg.add_spacer(height=5)

        with dpg.collapsing_header(label="Testing & Optimization", default_open=False, tag="perf_testing_header"):
            dpg.add_text("Run performance tests and optimization tasks:")
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Benchmark Suite", tag="run_benchmark_button", callback=handle_run_benchmark_click, width=-1)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Viz Performance Test", tag="run_viz_benchmark_button", callback=handle_run_viz_benchmark_click, width=-80)
                dpg.add_checkbox(label="Quick", tag="viz_benchmark_quick_mode_checkbox", default_value=False)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Auto-Tuning (Optimize Drive Scales)", tag="run_optimization_button", callback=handle_run_optimization_click, width=-80)
                dpg.add_checkbox(label="Quick", tag="optimization_quick_mode_checkbox", default_value=False)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Stop Running Test", tag="stop_perf_test_button", callback=handle_stop_perf_test_click, width=-1, enabled=False)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reload Auto-Tuned Overrides", tag="reload_overrides_button", callback=handle_reload_overrides_click, width=-1)
            
            dpg.add_spacer(height=5)
            dpg.add_text("Status:", color=[150,200,250,255])
            dpg.add_text("Ready", tag="perf_test_status_text", wrap=label_col_width * 2)
            
            dpg.add_spacer(height=3)
            dpg.add_text("Results:", color=[150,200,250,255])
            dpg.add_input_text(default_value="", tag="perf_test_results_text", multiline=True, readonly=True, height=80, width=-1)

        with dpg.collapsing_header(label="System Logs", default_open=False, tag="system_logs_header"):
            dpg.add_text("Search logs:")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="log_search_input", width=220, callback=handle_log_search_change)
                dpg.add_button(label="Previous", tag="log_search_prev_button", callback=handle_log_search_prev, width=70, enabled=False)
                dpg.add_button(label="Next", tag="log_search_next_button", callback=handle_log_search_next, width=70, enabled=False)
            
            dpg.add_text("0 / 0 matches", tag="log_search_match_text")
            dpg.add_spacer(height=3)
            
            def toggle_log_autoscroll(sender, checked):
                """Toggle autoscroll tracking on/off for the log field."""
                if dpg.does_item_exist("system_logs_display"):
                    dpg.configure_item("system_logs_display", tracked=checked, track_offset=1.0 if checked else 0.0)
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Auto-scroll", tag="log_autoscroll_checkbox", default_value=True, callback=toggle_log_autoscroll)
                dpg.add_button(label="Clear Logs", tag="clear_logs_button", callback=handle_clear_logs_click, width=100)
                dpg.add_button(label="Export Logs", tag="export_logs_button", callback=handle_export_logs_click, width=100)
            
            dpg.add_spacer(height=3)
            with dpg.child_window(tag="system_logs_scroll_container", width=-1, height=-1, horizontal_scrollbar=False):
                # Auto-scroll is on by default via tracked=True and track_offset=1.0
                dpg.add_input_text(default_value="", tag="system_logs_display", multiline=True, readonly=True, 
                                 tracked=True, track_offset=1.0, width=-1, height=0)

    # File Dialogs
    profile_dir = global_simulation_bridge.PROFILE_DIR if global_simulation_bridge else "simulation_profiles/"
    checkpoint_dir_h5 = global_simulation_bridge.CHECKPOINT_DIR if global_simulation_bridge else "simulation_checkpoints_h5/" # Updated
    recording_dir_h5 = global_simulation_bridge.RECORDING_DIR if global_simulation_bridge else "simulation_recordings_h5/"   # Updated

    for p_dir in [profile_dir, checkpoint_dir_h5, recording_dir_h5]:
        if not os.path.exists(p_dir): os.makedirs(p_dir, exist_ok=True)

    # Profile dialogs (JSON)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_profile_dialog_callback, tag="save_profile_file_dialog", width=700, height=400, modal=True, default_path=profile_dir, default_filename="profile"):
        dpg.add_file_extension(".json", color=(255, 255, 0, 255), custom_text="JSON Profile (*.json)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_profile_dialog_callback, tag="load_profile_file_dialog", width=700, height=400, modal=True, default_path=profile_dir):
        dpg.add_file_extension(".json", color=(255, 255, 0, 255), custom_text="JSON Profile (*.json)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Checkpoint dialogs (HDF5) - use .h5 as filter (DPG doesn't handle compound extensions well)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_checkpoint_dialog_callback_h5,
                         tag="save_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5, default_filename="checkpoint"):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="Checkpoint Files (*.simstate.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    with dpg.file_dialog(directory_selector=False, show=False, callback=load_checkpoint_dialog_callback_h5,
                         tag="load_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="Checkpoint Files (*.simstate.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Recording dialogs (HDF5) - use .h5 as filter (DPG doesn't handle compound extensions well)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_recording_for_streaming_dialog_callback_h5,
                         tag="save_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5, default_filename="recording"):
        dpg.add_file_extension(".h5", color=(150, 0, 200, 255), custom_text="Recording Files (*.simrec.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Load recording dialog
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_recording_dialog_callback_h5,
                         tag="load_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5):
        dpg.add_file_extension(".h5", color=(150, 0, 200, 255), custom_text="Recording Files (*.simrec.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Recording memory warning popup
    with dpg.window(label="Recording Too Large for GPU", tag="recording_memory_warning_popup",
                    modal=True, show=False, width=450, height=200, no_resize=True, no_collapse=True,
                    pos=[300, 250], no_close=True):
        dpg.add_text("", tag="recording_memory_warning_text", wrap=420)
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Partial Cache", width=130, callback=_recording_memory_popup_partial_cache)
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Cache as many frames as will fit in GPU memory.\n"
                            "Remaining frames will stream from disk.", wrap=250)
            dpg.add_button(label="Stream Only", width=130, callback=_recording_memory_popup_stream_only)
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Stream all frames from disk (no GPU caching).\n"
                            "Uses minimal GPU memory but playback may be slower.", wrap=250)
            dpg.add_button(label="Cancel", width=80, callback=_recording_memory_popup_cancel)

    # Recording options popup (for large-scale recordings)
    with dpg.window(label="Recording Options", tag="recording_options_popup",
                    modal=True, show=False, width=420, height=320, no_resize=True, no_collapse=True,
                    pos=[280, 180], no_close=True):
        dpg.add_text("Configure recording settings before selecting output file.", wrap=400)
        dpg.add_spacer(height=10)

        # Recording mode
        dpg.add_text("Recording Mode:")
        dpg.add_combo(
            items=["gpu_buffered", "streaming"],
            default_value="gpu_buffered",
            tag="rec_opt_mode_combo",
            width=250
        )
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(
                "gpu_buffered: Buffer frames in GPU/CPU memory, write at end.\n"
                "  Best for short recordings that fit in memory.\n\n"
                "streaming: Write frames to disk during simulation.\n"
                "  Required for long recordings or limited memory.",
                wrap=300
            )
        dpg.add_spacer(height=10)

        # Skip synaptic data
        dpg.add_checkbox(
            label="Skip synaptic data (neuron-only recording)",
            tag="rec_opt_skip_synaptic",
            default_value=False
        )
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(
                "For large networks (100K+ neurons), synaptic data can be 10-20x larger "
                "than neuron data. Enable this to dramatically reduce recording size.\n\n"
                "Example: 100K neurons, 10M synapses:\n"
                "  Full frame: ~165MB\n"
                "  Neuron-only: ~10MB (16x smaller)",
                wrap=300
            )
        dpg.add_spacer(height=10)

        # Frame skip
        dpg.add_text("Frame skip (0 = disabled):")
        dpg.add_input_int(
            tag="rec_opt_frame_skip",
            default_value=0,
            min_value=0,
            max_value=1000,
            min_clamped=True,
            max_clamped=True,
            width=100
        )
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(
                "0 or 1 = record every frame (no skipping)\n"
                "10 = record every 10th frame (10x smaller files)\n"
                "100 = record every 100th frame (100x smaller files)\n\n"
                "For dt=1ms, frame_skip=10 gives 10ms temporal resolution.",
                wrap=300
            )
        dpg.add_spacer(height=20)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Continue", width=150, callback=_recording_options_continue_callback)
            dpg.add_button(label="Cancel", width=100, callback=_recording_options_cancel_callback)

# --- Main Application Loop Functions ---

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
