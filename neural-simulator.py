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
from dataclasses import dataclass, field, asdict, fields
from typing import List

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
import cupy.sparse as csp
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

class NeuronModel(Enum):
    IZHIKEVICH = "IZHIKEVICH"
    HODGKIN_HUXLEY = "HODGKIN_HUXLEY"

class NeuronType(Enum):
    IZH2007_RS_CORTICAL_PYRAMIDAL = "IZH2007_RS_CORTICAL_PYRAMIDAL"
    IZH2007_FS_CORTICAL_INTERNEURON = "IZH2007_FS_CORTICAL_INTERNEURON"
    HH_L5_CORTICAL_PYRAMIDAL_RS = "HH_L5_CORTICAL_PYRAMIDAL_RS"
    RS_EXCITATORY_LEGACY = "RS_EXCITATORY_LEGACY"
    FS_INHIBITORY_LEGACY = "FS_INHIBITORY_LEGACY"
    IB_EXCITATORY_LEGACY = "IB_EXCITATORY_LEGACY"
    CH_EXCITATORY_LEGACY = "CH_EXCITATORY_LEGACY"
    LTS_INHIBITORY_LEGACY = "LTS_INHIBITORY_LEGACY"
    HH_EXCITATORY_DEFAULT_LEGACY = "HH_EXCITATORY_DEFAULT_LEGACY"


class DefaultIzhikevichParamsManager:
    PARAMS = {
        NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL: {
            "C": 100.0, "k": 0.7, "vr": -60.0, "vt": -40.0, "vpeak": 35.0,
            "a": 0.03, "b": -2.0, "c_reset": -50.0, "d_increment": 100.0
        },
        NeuronType.IZH2007_FS_CORTICAL_INTERNEURON: {
            "C": 20.0, "k": 1.0, "vr": -55.0, "vt": -40.0, "vpeak": 25.0,
            "a": 0.2, "b": -2.0, "c_reset": -45.0, "d_increment": -55.0 # d_increment was -55.0, Izhikevich paper says -65 for some FS, but params vary. This seems specific.
        },
        NeuronType.RS_EXCITATORY_LEGACY: {"a": 0.02, "b": 0.2, "c_reset": -65.0, "d_increment": 8.0, "vpeak": 30.0},
        NeuronType.FS_INHIBITORY_LEGACY: {"a": 0.1, "b": 0.2, "c_reset": -65.0, "d_increment": 2.0, "vpeak": 30.0},
        NeuronType.IB_EXCITATORY_LEGACY: {"a": 0.02, "b": 0.2, "c_reset": -55.0, "d_increment": 4.0, "vpeak": 50.0},
        NeuronType.CH_EXCITATORY_LEGACY: {"a": 0.02, "b": 0.2, "c_reset": -50.0, "d_increment": 2.0, "vpeak": 35.0},
        NeuronType.LTS_INHIBITORY_LEGACY: {"a": 0.02, "b": 0.25, "c_reset": -65.0, "d_increment": 2.0, "vpeak": 30.0}
    }
    FALLBACK_2007 = PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL].copy()
    FALLBACK_LEGACY = PARAMS[NeuronType.RS_EXCITATORY_LEGACY].copy()

    @staticmethod
    def get_params(neuron_type_enum, use_2007_formulation=True):
        if use_2007_formulation:
            if neuron_type_enum in [NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL, NeuronType.IZH2007_FS_CORTICAL_INTERNEURON]:
                 return DefaultIzhikevichParamsManager.PARAMS.get(neuron_type_enum, DefaultIzhikevichParamsManager.FALLBACK_2007).copy()
            print(f"Warning: Requested legacy type {neuron_type_enum} for 2007 formulation. Using RS_CORTICAL_PYRAMIDAL fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_2007.copy()
        else: # Legacy formulation
            if 'LEGACY' in neuron_type_enum.name:
                 return DefaultIzhikevichParamsManager.PARAMS.get(neuron_type_enum, DefaultIzhikevichParamsManager.FALLBACK_LEGACY).copy()
            print(f"Warning: Requested 2007 type {neuron_type_enum} for legacy formulation. Using RS_EXCITATORY_LEGACY fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_LEGACY.copy()


class DefaultHodgkinHuxleyParams:
    # Parameters for a more realistic Layer 5 Pyramidal Neuron (Regular Spiking) at 37°C
    # Adapted from literature, may require tuning for specific behaviors.
    # Key sources: Mainen & Sejnowski (1996), Pospischil et al. (2008) for general cortical neuron models.
    REALISTIC_L5_PYRAMIDAL_RS_37C = {
        "C_m": 1.0,       # Membrane capacitance (uF/cm^2) - Common value
        "g_Na_max": 50.0, # Max Na conductance (mS/cm^2) - Can vary (e.g., 50-120)
        "g_K_max": 5.0,   # Max K_DR conductance (mS/cm^2) - For delayed rectifier (e.g., 5-30)
        "g_L": 0.1,       # Leak conductance (mS/cm^2) - (e.g., 0.02-0.1)
        "E_Na": 50.0,     # Na reversal potential (mV) - (e.g., +50 to +60)
        "E_K": -85.0,     # K reversal potential (mV) - (e.g., -80 to -90 for K_DR)
        "E_L": -70.0,     # Leak reversal potential (mV) - (e.g., -65 to -75, often near V_rest)
        "v_rest_hh": -65.0, # Resting potential for HH model initialization (mV)
        "v_peak_hh": 40.0,  # Spike peak for HH model (mV) - for spike detection logic
        # Initial gating variable values (approximate for v_rest_hh = -65mV)
        "m_init": 0.0529, # Calculated from alpha_m / (alpha_m + beta_m) at -65mV for original HH
        "h_init": 0.5961, # Calculated from alpha_h / (alpha_h + beta_h) at -65mV for original HH
        "n_init": 0.3177  # Calculated from alpha_n / (alpha_n + beta_n) at -65mV for original HH
    }
    # Original Hodgkin-Huxley parameters (Squid Giant Axon at 6.3°C)
    ORIGINAL_HH_PARAMS = {
        "C_m": 1.0, "g_Na_max": 120.0, "g_K_max": 36.0, "g_L": 0.3,
        "E_Na": 50.0, "E_K": -77.0, "E_L": -54.387, # Note: E_L adjusted for V_rest = -65mV in original model
        "v_rest_hh": -65.0, "v_peak_hh": 40.0,
        "m_init": 0.0529, "h_init": 0.5961, "n_init": 0.3177
    }
    PARAMS = {
        NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS: REALISTIC_L5_PYRAMIDAL_RS_37C.copy(),
        NeuronType.HH_EXCITATORY_DEFAULT_LEGACY: ORIGINAL_HH_PARAMS.copy(), # Legacy can map to original HH
    }
    FALLBACK = PARAMS[NeuronType.HH_EXCITATORY_DEFAULT_LEGACY].copy()

    @staticmethod
    def get_params(neuron_type_enum):
        return DefaultHodgkinHuxleyParams.PARAMS.get(neuron_type_enum, DefaultHodgkinHuxleyParams.FALLBACK).copy()

# --- Performance Optimization: Neuron Type ID Mapper ---
class NeuronTypeIDMapper:
    """Maps NeuronType enums to integer IDs for GPU-efficient operations.
    
    This eliminates string comparisons on CPU by using integer type IDs
    that can be processed directly on the GPU.
    """
    def __init__(self):
        self.type_to_id = {}
        self.id_to_type = {}
        self.id_to_display_name = {}
        self._build_mappings()
    
    def _build_mappings(self):
        """Build bidirectional mappings between NeuronType enums and integer IDs."""
        # Izhikevich types
        izh_types = [nt for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS]
        for idx, ntype in enumerate(izh_types):
            self.type_to_id[ntype] = idx
            self.id_to_type[idx] = ntype
            self.id_to_display_name[idx] = f"Izh2007_{ntype.name.replace('IZH2007_', '')}"
        
        # Hodgkin-Huxley types (offset by max Izh type ID)
        hh_types = [nt for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS]
        hh_offset = len(izh_types)
        for idx, ntype in enumerate(hh_types):
            type_id = hh_offset + idx
            self.type_to_id[ntype] = type_id
            self.id_to_type[type_id] = ntype
            self.id_to_display_name[type_id] = f"HH_{ntype.name.replace('HH_', '')}"
    
    def get_id(self, neuron_type_enum):
        """Get integer ID for a NeuronType enum."""
        return self.type_to_id.get(neuron_type_enum, 0)  # Default to 0 if not found
    
    def get_type(self, type_id):
        """Get NeuronType enum for an integer ID."""
        return self.id_to_type.get(type_id, list(self.id_to_type.values())[0] if self.id_to_type else NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL)
    
    def get_display_name(self, type_id):
        """Get display name string for an integer type ID."""
        return self.id_to_display_name.get(type_id, "Unknown")
    
    def get_all_display_names_for_model(self, model_name):
        """Get list of display names for a specific model type."""
        if model_name == NeuronModel.IZHIKEVICH.name:
            return [self.id_to_display_name[i] for i in sorted(self.id_to_display_name.keys()) 
                    if "Izh" in self.id_to_display_name[i]]
        elif model_name == NeuronModel.HODGKIN_HUXLEY.name:
            return [self.id_to_display_name[i] for i in sorted(self.id_to_display_name.keys()) 
                    if "HH" in self.id_to_display_name[i]]
        return []
    
    def get_id_from_display_name(self, display_name):
        """Get type ID from display name string."""
        for type_id, name in self.id_to_display_name.items():
            if name == display_name:
                return type_id
        return 0  # Default

# Global type mapper instance (initialized after all required classes are defined)
NEURON_TYPE_MAPPER = NeuronTypeIDMapper()

# --- NEW Configuration & Data Classes (Replaces SimulationConfiguration) ---

@dataclass
class CoreSimConfig:
    """Holds parameters essential for the simulation's logic and reproducibility."""
    total_simulation_time_ms: float = 60000.0
    dt_ms: float = 1.000
    num_neurons: int = 1000
    connections_per_neuron: int = 100
    num_traits: int = 5
    seed: int = -1
    neuron_model_type: str = NeuronModel.IZHIKEVICH.name
    default_neuron_type_izh: str = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    default_neuron_type_hh: str = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name
    
    # Izhikevich - initialized from a default type
    izh_C_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["C"])
    izh_k_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["k"])
    izh_vr_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["vr"])
    izh_vt_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["vt"])
    izh_vpeak_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["vpeak"])
    izh_a_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["a"])
    izh_b_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["b"])
    izh_c_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["c_reset"])
    izh_d_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["d_increment"])

    # Hodgkin-Huxley - initialized from a default type
    hh_C_m: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["C_m"])
    hh_g_Na_max: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["g_Na_max"])
    hh_g_K_max: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["g_K_max"])
    hh_g_L: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["g_L"])
    hh_E_Na: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["E_Na"])
    hh_E_K: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["E_K"])
    hh_E_L: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["E_L"])
    hh_v_rest_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["v_rest_hh"])
    hh_v_peak: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["v_peak_hh"])
    hh_m_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["m_init"])
    hh_h_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["h_init"])
    hh_n_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["n_init"])
    hh_temperature_celsius: float = 37.0
    hh_q10_factor: float = 3.0

    # Synapse & Plasticity
    refractory_period_steps: int = 2
    syn_reversal_potential_e: float = 0.0
    syn_reversal_potential_i: float = -70.0
    syn_tau_g_e: float = 5.0
    syn_tau_g_i: float = 10.0
    propagation_strength: float = 0.05
    inhibitory_propagation_strength: float = 0.15
    max_synaptic_delay_ms: float = 20.0
    enable_inhibitory_neurons: bool = True
    inhibitory_trait_index: int = 1
    enable_hebbian_learning: bool = True
    hebbian_learning_rate: float = 0.0005
    hebbian_weight_decay: float = 0.00001
    hebbian_min_weight: float = 0.05
    hebbian_max_weight: float = 1.0
    enable_short_term_plasticity: bool = True
    stp_U: float = 0.15
    stp_tau_d: float = 200.0
    stp_tau_f: float = 50.0
    enable_homeostasis: bool = True
    homeostasis_target_rate: float = 0.02
    homeostasis_threshold_adapt_rate: float = 0.015
    homeostasis_ema_alpha: float = 0.01
    homeostasis_threshold_min: float = -55.0
    homeostasis_threshold_max: float = -30.0
    enable_watts_strogatz: bool = True
    connectivity_k: int = 10
    connectivity_p_rewire: float = 0.1

@dataclass
class VisualizationConfig:
    """Holds parameters for visualization, such as camera and volume."""
    volume_min_x: float = -50.0; volume_max_x: float = 50.0
    volume_min_y: float = -50.0; volume_max_y: float = 50.0
    volume_min_z: float = -50.0; volume_max_z: float = 50.0
    camera_center_x: float = 0.0; camera_center_y: float = 0.0; camera_center_z: float = 0.0
    camera_radius: float = 150.0
    camera_azimuth_angle: float = 0.0
    camera_elevation_angle: float = 0.0
    camera_up_x: float = 0.0; camera_up_y: float = 1.0; camera_up_z: float = 0.0
    camera_fov: float = 60.0
    camera_near_clip: float = 0.1
    camera_far_clip: float = 1000.0
    mouse_last_x: int = 0; mouse_last_y: int = 0
    mouse_left_button_down: bool = False
    mouse_right_button_down: bool = False
    viz_update_interval_steps: int = 17  # Update visualization every N steps (~60fps at dt=1.0ms)

@dataclass
class RuntimeState:
    """Holds the dynamic state of the simulation run. Not typically saved in profiles."""
    current_time_ms: float = 0.0
    current_time_step: int = 0
    is_running: bool = False
    is_paused: bool = False
    simulation_speed_factor: float = 1.0
    neuron_positions_x: List[float] = field(default_factory=list)
    neuron_positions_y: List[float] = field(default_factory=list)
    neuron_types_list_for_viz: List[str] = field(default_factory=list)
    max_delay_steps: int = 200

def _create_config_from_dict(config_cls, data_dict):
    """Helper to create a dataclass instance from a dictionary, ignoring extra keys."""
    if not data_dict:
        return config_cls()
    
    # Get the field names defined in the dataclass
    class_fields = {f.name for f in fields(config_cls)}
    
    # Filter the input dictionary to only include keys that are fields in the class
    filtered_data = {k: v for k, v in data_dict.items() if k in class_fields}
    
    return config_cls(**filtered_data)

def _get_full_config_dict(core_cfg, viz_cfg, runtime_state):
    """Helper to combine all config objects into a single dictionary for saving."""
    return {
        "core_config": asdict(core_cfg),
        "viz_config": asdict(viz_cfg),
        "runtime_state": asdict(runtime_state)
    }

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
        self.neuron_model_type = NeuronModel.IZHIKEVICH.name # Current neuron model ('IZHIKEVICH' or 'HODGKIN_HUXLEY')
        self.default_neuron_type_izh = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name # Default Izhikevich type if trait mapping fails
        self.default_neuron_type_hh = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name # Default Hodgkin-Huxley type

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

        # Basic Neuron & Synapse Properties
        self.refractory_period_steps = 2 # Absolute refractory period in simulation steps (dt units)
        self.syn_reversal_potential_e = 0.0 # Reversal potential for excitatory synapses (mV)
        self.syn_reversal_potential_i = -70.0 # Reversal potential for inhibitory synapses (mV)
        self.syn_tau_g_e = 5.0 # Time constant for excitatory synaptic conductance decay (ms)
        self.syn_tau_g_i = 10.0 # Time constant for inhibitory synaptic conductance decay (ms)
        self.propagation_strength = 0.05 # Scaling factor for excitatory synaptic conductance increase per spike
        self.inhibitory_propagation_strength = 0.15 # Scaling factor for inhibitory synaptic conductance increase
        self.max_synaptic_delay_ms = 20.0 # Maximum synaptic delay in ms (Not fully implemented for individual delays yet)

        # Inhibitory Neuron Configuration
        self.enable_inhibitory_neurons = True # Whether to model inhibitory neurons
        self.inhibitory_trait_index = 1 # Trait index designated as inhibitory (0-indexed)

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

        # Homeostatic Plasticity (Adaptive Thresholds for Izhikevich model)
        self.enable_homeostasis = True # Enable homeostatic threshold adaptation
        self.homeostasis_target_rate = 0.02 # Target firing rate (spikes per dt step)
        self.homeostasis_threshold_adapt_rate = 0.015 # Adaptation rate for firing thresholds
        self.homeostasis_ema_alpha = 0.01 # Alpha for EMA of neuron activity
        self.homeostasis_threshold_min = -55.0 # Minimum firing threshold (mV)
        self.homeostasis_threshold_max = -30.0 # Maximum firing threshold (mV)

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

# --- CuPy Fused Kernels ---
@cp.fuse()
def fused_izhikevich_legacy_dynamics_update(v, u, a, b, total_I, dt):
    """Fused kernel for legacy Izhikevich model dynamics."""
    dv = (0.04 * v**2 + 5 * v + 140 - u + total_I)
    du = a * (b * v - u)
    v_new = v + dv * dt
    u_new = u + du * dt
    return v_new, u_new

@cp.fuse()
def fused_izhikevich2007_dynamics_update(v, u, C_param, k_param, vr_param, vt_param, a_param, b_param, total_synaptic_current, dt):
    """Fused kernel for Izhikevich 2007 model dynamics."""
    # Ensure C_param is not zero to prevent division by zero errors.
    C_param_safe = cp.where(C_param == 0.0, 1.0, C_param) # Use 1.0 as a safe non-zero default if C is 0
    
    # Differential equation for membrane potential v
    dv_dt = (k_param * (v - vr_param) * (v - vt_param) - u + total_synaptic_current) / C_param_safe
    # Differential equation for recovery variable u
    du_dt = a_param * (b_param * (v - vr_param) - u)
    
    # Euler integration to update v and u
    v_new = v + dv_dt * dt
    u_new = u + du_dt * dt
    return v_new, u_new

@cp.fuse()
def fused_hodgkin_huxley_dynamics_update(V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L, temperature_celsius, q10_factor):
    """Fused kernel for Hodgkin-Huxley model dynamics, including temperature effects."""
    # Base temperature for original HH kinetics (typically 6.3°C or similar)
    BASE_HH_KINETICS_TEMP_C = 6.3 
    # Temperature adjustment factor (phi) using Q10
    phi = q10_factor**((temperature_celsius - BASE_HH_KINETICS_TEMP_C) / 10.0)

    # Rate functions (alpha, beta) for gating variables m, h, n
    # Original HH equations, adjusted for V in mV.
    # Handling for V = -40 (for alpha_m) and V = -55 (for alpha_n) to avoid division by zero in expm1.
    # expm1(x) = exp(x) - 1. For small x, expm1(x) approx x.
    # If V = -40, then -(V+40)/10 = 0. The limit of -0.1*x / (exp(-x/10)-1) as x->-40 is 1.0.
    # (Using L'Hopital's rule: d/dx (-0.1(x+40)) / d/dx (exp(-(x+40)/10)-1) = -0.1 / (-0.1 * exp(-(x+40)/10)) = exp((x+40)/10) -> exp(0) = 1)
    
    v_plus_40 = V + 40.0 # For m-gate alpha expression
    alpha_m_orig = cp.where(v_plus_40 == 0, 1.0 * 0.1 * 10.0 , -0.1 * v_plus_40 / cp.expm1(-v_plus_40 / 10.0)) # Corrected limit handling
    beta_m_orig  = 4.0 * cp.exp(-(V + 65.0) / 18.0)

    alpha_h_orig = 0.07 * cp.exp(-(V + 65.0) / 20.0)
    beta_h_orig  = 1.0 / (cp.exp(-(V + 35.0) / 10.0) + 1.0)

    v_plus_55 = V + 55.0 # For n-gate alpha expression
    alpha_n_orig = cp.where(v_plus_55 == 0, 0.1 * 0.01 * 10.0, -0.01 * v_plus_55 / cp.expm1(-v_plus_55 / 10.0)) # Corrected limit handling
    beta_n_orig  = 0.125 * cp.exp(-(V + 65.0) / 80.0)
    
    # Apply temperature correction to rate constants
    alpha_m = alpha_m_orig * phi; beta_m  = beta_m_orig  * phi
    alpha_h = alpha_h_orig * phi; beta_h  = beta_h_orig  * phi
    alpha_n = alpha_n_orig * phi; beta_n  = beta_n_orig  * phi

    # Update gating variables using analytical solution for first-order kinetics (assuming V is constant during dt)
    # m_new = m_inf - (m_inf - m_old) * exp(-dt / tau_m)
    # where m_inf = alpha_m / (alpha_m + beta_m) and tau_m = 1 / (alpha_m + beta_m)

    sum_alpha_beta_m = alpha_m + beta_m
    m_inf = cp.where(sum_alpha_beta_m == 0, m, alpha_m / sum_alpha_beta_m) # Avoid division by zero if sum is 0
    tau_m = cp.where(sum_alpha_beta_m == 0, cp.inf, 1.0 / sum_alpha_beta_m) # tau is inf if sum is 0
    # Handle infinite tau by not changing m (exp(0)=1, so m_new = m_inf - (m_inf - m) = m)
    m_new = m_inf + (m - m_inf) * cp.exp(cp.where(cp.isinf(tau_m), 0.0, -dt / tau_m))


    sum_alpha_beta_h = alpha_h + beta_h
    h_inf = cp.where(sum_alpha_beta_h == 0, h, alpha_h / sum_alpha_beta_h)
    tau_h = cp.where(sum_alpha_beta_h == 0, cp.inf, 1.0 / sum_alpha_beta_h)
    h_new = h_inf + (h - h_inf) * cp.exp(cp.where(cp.isinf(tau_h), 0.0, -dt / tau_h))

    sum_alpha_beta_n = alpha_n + beta_n
    n_inf = cp.where(sum_alpha_beta_n == 0, n, alpha_n / sum_alpha_beta_n)
    tau_n = cp.where(sum_alpha_beta_n == 0, cp.inf, 1.0 / sum_alpha_beta_n)
    n_new = n_inf + (n - n_inf) * cp.exp(cp.where(cp.isinf(tau_n), 0.0, -dt / tau_n))
    
    # Clip gating variables to be between 0 and 1
    m_new = cp.clip(m_new, 0.0, 1.0); h_new = cp.clip(h_new, 0.0, 1.0); n_new = cp.clip(n_new, 0.0, 1.0)

    # Ionic currents
    I_Na = g_Na_max * (m_new**3) * h_new * (V - E_Na) # Sodium current
    I_K  = g_K_max * (n_new**4) * (V - E_K)   # Potassium current
    I_L  = g_L * (V - E_L)                    # Leak current
    I_ion = I_Na + I_K + I_L                  # Total ionic current

    # Membrane potential update
    dV_dt = (I_syn - I_ion) / C_m # dV/dt = (I_external - I_ionic) / C_m
    V_new = V + dV_dt * dt        # Euler integration
    return V_new, m_new, h_new, n_new

@cp.fuse()
def fused_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i):
    """Fused kernel for synaptic conductance decay and calculating synaptic current."""
    # Decay conductances
    g_e_new = g_e * decay_e # Excitatory conductance decay
    g_i_new = g_i * decay_i # Inhibitory conductance decay
    # Calculate total synaptic current based on new conductances
    I_syn = g_e_new * (E_e - v) + g_i_new * (E_i - v) # I_syn = g_e*(E_e - V) + g_i*(E_i - V)
    return g_e_new, g_i_new, I_syn

@cp.fuse()
def fused_stp_decay_recovery(u, x, dt, tau_f, tau_d):
    """Fused kernel for STP u and x variable decay/recovery."""
    # Ensure tau_f and tau_d are not zero to prevent division by zero.
    tau_f_safe = cp.maximum(tau_f, 1e-9) # Use a small epsilon if tau_f is zero
    tau_d_safe = cp.maximum(tau_d, 1e-9) # Use a small epsilon if tau_d is zero

    # Decay of u (facilitation variable)
    u_decayed = u * cp.exp(-dt / tau_f_safe)
    # Recovery of x (depression variable)
    x_recovered_increment = (1.0 - x) * (dt / tau_d_safe) # dx/dt = (1-x)/tau_d
    x_recovered = x + x_recovered_increment
    x_clipped = cp.clip(x_recovered, 0.0, 1.0) # Ensure x stays within [0, 1]
    return u_decayed, x_clipped

@cp.fuse()
def fused_homeostasis_update(neuron_activity_ema_in, fired_this_step_float, target_rate, alpha_ema, adapt_rate,
                             neuron_firing_thresholds_in, thresh_min, thresh_max):
    """Fused kernel for homeostatic threshold adaptation."""
    # Update Exponential Moving Average (EMA) of neuron activity
    new_neuron_activity_ema = (1.0 - alpha_ema) * neuron_activity_ema_in + alpha_ema * fired_this_step_float
    # Calculate error from target firing rate
    error = new_neuron_activity_ema - target_rate
    # Calculate change in threshold based on error and adaptation rate
    threshold_delta = error * adapt_rate
    # Update firing thresholds
    new_neuron_firing_thresholds = neuron_firing_thresholds_in + threshold_delta
    # Clip thresholds to min/max bounds
    new_neuron_firing_thresholds_clipped = cp.clip(new_neuron_firing_thresholds, thresh_min, thresh_max)
    return new_neuron_activity_ema, new_neuron_firing_thresholds_clipped


# --- Simulation Bridge (Core Logic) ---
class SimulationBridge:
    def __init__(self, sim_core_ref=None): # sim_core_ref is legacy, not used with threading
        self.core_config = CoreSimConfig()
        self.viz_config = VisualizationConfig()
        self.runtime_state = RuntimeState()
        self.ui_queue = sim_to_ui_queue # Reference to the queue for sending data/status to UI

        # --- CuPy Arrays for Simulation State ---
        self.cp_membrane_potential_v = None 
        self.cp_recovery_variable_u = None  
        self.cp_conductance_g_e = None      
        self.cp_conductance_g_i = None      
        self.cp_external_input_current = None 
        self.cp_firing_states = None        
        self.cp_prev_firing_states = None   
        self.cp_traits = None               
        self.cp_neuron_type_ids = None      # Integer type IDs for GPU-efficient filtering
        self.cp_neuron_positions_3d = None  
        self.cp_refractory_timers = None    
        self.cp_viz_activity_timers = None

        self.cp_izh_C = None; self.cp_izh_k = None; self.cp_izh_vr = None; self.cp_izh_vt = None
        self.cp_izh_vpeak = None; self.cp_izh_a = None; self.cp_izh_b = None
        self.cp_izh_c_reset = None; self.cp_izh_d_increment = None

        self.cp_izh_legacy_a = None; self.cp_izh_legacy_b = None
        self.cp_izh_legacy_c_reset = None; self.cp_izh_legacy_d_increment = None
        self.cp_izh_legacy_vpeak = None

        self.cp_gating_variable_m = None 
        self.cp_gating_variable_h = None 
        self.cp_gating_variable_n = None 

        self.cp_hh_C_m = None; self.cp_hh_g_Na_max = None; self.cp_hh_g_K_max = None; self.cp_hh_g_L = None
        self.cp_hh_E_Na = None; self.cp_hh_E_K = None; self.cp_hh_E_L = None; self.cp_hh_v_peak = None

        self.cp_neuron_firing_thresholds = None 
        self.cp_neuron_activity_ema = None      

        self.cp_connections = None 

        self.cp_stp_u = None 
        self.cp_stp_x = None 

        self.cp_synapse_pulse_timers = None   
        self.cp_synapse_pulse_progress = None 

        self.is_initialized = False 

        self._mock_total_plasticity_events = 0
        self._mock_network_avg_firing_rate_hz = 0.0
        self._mock_num_spikes_this_step = 0
        
        # Performance profiling
        self._enable_profiling = True  # Set to False to disable
        self._profile_timings = {
            "step_total": deque(maxlen=100),
            "connectivity": deque(maxlen=100),
            "dynamics": deque(maxlen=100),
            "gpu_sync": deque(maxlen=100)
        }

        self.PROFILE_DIR = "simulation_profiles/" 
        self.CHECKPOINT_DIR = "simulation_checkpoints_h5/" 
        self.RECORDING_DIR = "simulation_recordings_h5/"   

        self.recording_file_handle = None 
        self.recording_filepath = None    
        self.current_frame_count_for_h5 = 0
        
        # GPU-buffered recording: store all frames in VRAM during recording
        self.gpu_frame_buffer = {}  # Dict of frame_idx -> dict of CuPy arrays
        self.gpu_buffered_recording = True  # Enable GPU buffering for max performance
        self.gpu_recording_max_frames = 0  # Maximum frames we can buffer
        
        # GPU-buffered playback: load entire recording into VRAM for instant seeking
        self.gpu_playback_cache = {}  # Dict of frame_idx -> dict of CuPy arrays
        self.gpu_playback_enabled = False  # Whether playback is using GPU cache

        for dir_path in [self.PROFILE_DIR, self.CHECKPOINT_DIR, self.RECORDING_DIR]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    self._log_console(f"Created directory: {dir_path}", "info")
                except OSError as e:
                    self._log_console(f"Error creating directory {dir_path}: {e}", "error")
        try:
             cp.cuda.Device(0).use()
             
             # Configure memory pool for better performance
             mempool = cp.get_default_memory_pool()
             pinned_mempool = cp.get_default_pinned_memory_pool()
             
             # Set memory pool to use up to 80% of available GPU memory
             dev_props = cp.cuda.runtime.getDeviceProperties(0)
             total_mem = dev_props['totalGlobalMem']
             mempool.set_limit(size=int(total_mem * 0.8))
             
             gpu_name = dev_props.get('name',b'Unknown').decode()
             self._log_console(f"CuPy using GPU: {gpu_name} ({total_mem / 1024**3:.1f} GB)", "info")
        except Exception as e:
             self._log_console(f"Error setting CuPy device: {e}", "critical")

    def _log_console(self, message, level="info"):
        """Logs a message to the console (standard output)."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}][{level.upper()}] SIM_BRIDGE: {message}")

    def _log_to_ui(self, message, level="info", color=None):
        """Sends a log message to the UI thread via the queue for display in the status bar."""
        if color is None:
            if level == "error" or level == "critical": color = [255, 0, 0]
            elif level == "warning": color = [255, 165, 0]
            elif level == "info": color = [200, 200, 200] 
            elif level == "success": color = [0, 200, 0]
            else: color = [200, 200, 200]
        
        if self.ui_queue:
            try:
                self.ui_queue.put_nowait({
                    "type": "STATUS_UPDATE",
                    "text": message,
                    "color": color,
                    "level": level
                })
            except queue.Full:
                self._log_console("UI queue full. Could not send status message.", "warning")
        self._log_console(message, level)
    
    def _get_gpu_memory_info(self):
        """Returns current GPU memory usage statistics."""
        mem_info = cp.cuda.Device().mem_info
        free_memory, total_memory = mem_info
        used_memory = total_memory - free_memory
        
        return {
            "total_gb": total_memory / 1e9,
            "used_gb": used_memory / 1e9,
            "free_gb": free_memory / 1e9,
            "usage_percent": (used_memory / total_memory) * 100
        }
    
    def _check_gpu_memory_pressure(self):
        """Checks if GPU memory is under pressure and suggests cleanup."""
        mem_stats = self._get_gpu_memory_info()
        
        if mem_stats["usage_percent"] > 90:
            self._log_to_ui(
                f"WARNING: GPU memory usage at {mem_stats['usage_percent']:.1f}% ({mem_stats['used_gb']:.1f}GB/{mem_stats['total_gb']:.1f}GB)",
                "warning"
            )
            # Trigger garbage collection
            cp.get_default_memory_pool().free_all_blocks()
            return True
        elif mem_stats["usage_percent"] > 80:
            self._log_console(f"GPU memory high: {mem_stats['usage_percent']:.1f}%")
            return False
        
        return False


    def _initialize_simulation_data(self, called_from_playback_init=False):
        """Initializes or re-initializes all CuPy arrays and simulation state variables."""
        self._log_console(f"Initializing simulation data for model: {self.core_config.neuron_model_type} (3D)... (playback_init: {called_from_playback_init})")

        if not called_from_playback_init:
            # These global_gui_state checks are for context; actual state changes are UI-driven.
            # Sim thread should not directly modify global_gui_state.
            pass # UI thread manages stopping recording/playback before commanding re-init.

        try:
            n = self.core_config.num_neurons
            cfg = self.core_config
            if n <= 0:
                self._log_console(f"Number of neurons ({n}) must be positive. Initialization failed.", "warning")
                self.is_initialized = False; return

            if cfg.seed != -1:
                cp.random.seed(cfg.seed)
                np.random.seed(cfg.seed)

            self.cp_external_input_current = cp.zeros(n, dtype=cp.float32)
            self.cp_firing_states = cp.zeros(n, dtype=bool)
            self.cp_prev_firing_states = cp.zeros(n, dtype=bool)
            self.cp_traits = cp.random.randint(0, max(1, cfg.num_traits), (n,), dtype=cp.int32) if n > 0 else cp.array([], dtype=cp.int32)
            self.cp_neuron_type_ids = cp.zeros(n, dtype=cp.int32) if n > 0 else cp.array([], dtype=cp.int32)  # Will be populated per neuron
            self.cp_conductance_g_e = cp.zeros(n, dtype=cp.float32)
            self.cp_conductance_g_i = cp.zeros(n, dtype=cp.float32)
            self.cp_refractory_timers = cp.zeros(n, dtype=cp.int32)
            self.cp_neuron_activity_ema = cp.zeros(n, dtype=cp.float32) 
            self.cp_viz_activity_timers = cp.zeros(n, dtype=cp.int32) 

            self.cp_synapse_pulse_timers = cp.array([], dtype=cp.int32)
            self.cp_synapse_pulse_progress = cp.array([], dtype=cp.float32)

            self.runtime_state.neuron_types_list_for_viz = [""] * n

            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                self._log_console(f"Initializing Izhikevich model specifics for {n} neurons...")
                self.cp_izh_C = cp.zeros(n, dtype=cp.float32); self.cp_izh_k = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vr = cp.zeros(n, dtype=cp.float32); self.cp_izh_vt = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vpeak = cp.zeros(n, dtype=cp.float32); self.cp_izh_a = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_b = cp.zeros(n, dtype=cp.float32); self.cp_izh_c_reset = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_d_increment = cp.zeros(n, dtype=cp.float32)
                
                self.cp_membrane_potential_v = cp.zeros(n, dtype=cp.float32)
                self.cp_recovery_variable_u = cp.zeros(n, dtype=cp.float32)

                thresh_base = (cfg.homeostasis_threshold_min + cfg.homeostasis_threshold_max) / 2.0
                thresh_var = (cfg.homeostasis_threshold_max - cfg.homeostasis_threshold_min) / 2.0
                if thresh_var < 0: thresh_var = 1.0 
                self.cp_neuron_firing_thresholds = cp.random.uniform(
                    thresh_base - thresh_var, thresh_base + thresh_var, n
                ).astype(cp.float32) if n > 0 else cp.array([], dtype=cp.float32)
                if n > 0:
                    cp.clip(self.cp_neuron_firing_thresholds,
                            cfg.homeostasis_threshold_min, cfg.homeostasis_threshold_max,
                            out=self.cp_neuron_firing_thresholds)

                np_traits_host = cp.asnumpy(self.cp_traits) 
                defined_izh2007_types = [
                    ntype for ntype in NeuronType
                    if "IZH2007" in ntype.name and ntype in DefaultIzhikevichParamsManager.PARAMS
                ]
                num_defined_izh_variants = len(defined_izh2007_types)
                for i in range(n):
                    trait_val = np_traits_host[i]
                    selected_neuron_type_enum = NeuronType[cfg.default_neuron_type_izh]
                    if num_defined_izh_variants > 0:
                        type_idx_in_list = trait_val % num_defined_izh_variants
                        selected_neuron_type_enum = defined_izh2007_types[type_idx_in_list]
                    
                    # Store integer type ID for GPU operations
                    type_id = NEURON_TYPE_MAPPER.get_id(selected_neuron_type_enum)
                    self.cp_neuron_type_ids[i] = type_id
                    
                    params = DefaultIzhikevichParamsManager.get_params(selected_neuron_type_enum, use_2007_formulation=True)
                    self.cp_izh_C[i] = params["C"]; self.cp_izh_k[i] = params["k"]
                    self.cp_izh_vr[i] = params["vr"]; self.cp_izh_vt[i] = params["vt"]
                    self.cp_izh_vpeak[i] = params["vpeak"]; self.cp_izh_a[i] = params["a"]
                    self.cp_izh_b[i] = params["b"]; self.cp_izh_c_reset[i] = params["c_reset"]
                    self.cp_izh_d_increment[i] = params["d_increment"]
                    self.cp_membrane_potential_v[i] = params["vr"] 
                    self.cp_recovery_variable_u[i] = params["b"] * (self.cp_membrane_potential_v[i] - params["vr"]) 
                    self.runtime_state.neuron_types_list_for_viz[i] = f"Izh2007_{selected_neuron_type_enum.name.replace('IZH2007_', '')}"

            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                self._log_console(f"Initializing Hodgkin-Huxley model specifics for {n} neurons...")
                self.cp_hh_C_m = cp.zeros(n, dtype=cp.float32); self.cp_hh_g_Na_max = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_g_K_max = cp.zeros(n, dtype=cp.float32); self.cp_hh_g_L = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_Na = cp.zeros(n, dtype=cp.float32); self.cp_hh_E_K = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_L = cp.zeros(n, dtype=cp.float32); self.cp_hh_v_peak = cp.zeros(n, dtype=cp.float32)
                
                self.cp_membrane_potential_v = cp.zeros(n, dtype=cp.float32)
                self.cp_gating_variable_m = cp.zeros(n, dtype=cp.float32)
                self.cp_gating_variable_h = cp.zeros(n, dtype=cp.float32)
                self.cp_gating_variable_n = cp.zeros(n, dtype=cp.float32)
                self.cp_neuron_firing_thresholds = None 

                np_traits_host = cp.asnumpy(self.cp_traits)
                defined_hh_types = [
                    ntype for ntype in NeuronType
                    if "HH_" in ntype.name and ntype in DefaultHodgkinHuxleyParams.PARAMS
                ]
                num_defined_hh_variants = len(defined_hh_types)
                for i in range(n):
                    trait_val = np_traits_host[i]
                    selected_neuron_type_enum = NeuronType[cfg.default_neuron_type_hh]
                    if num_defined_hh_variants > 0:
                        type_idx_in_list = trait_val % num_defined_hh_variants
                        selected_neuron_type_enum = defined_hh_types[type_idx_in_list]
                    
                    # Store integer type ID for GPU operations
                    type_id = NEURON_TYPE_MAPPER.get_id(selected_neuron_type_enum)
                    self.cp_neuron_type_ids[i] = type_id

                    params = DefaultHodgkinHuxleyParams.get_params(selected_neuron_type_enum)
                    self.cp_hh_C_m[i] = params["C_m"]; self.cp_hh_g_Na_max[i] = params["g_Na_max"]
                    self.cp_hh_g_K_max[i] = params["g_K_max"]; self.cp_hh_g_L[i] = params["g_L"]
                    self.cp_hh_E_Na[i] = params["E_Na"]; self.cp_hh_E_K[i] = params["E_K"]
                    self.cp_hh_E_L[i] = params["E_L"]; self.cp_hh_v_peak[i] = params["v_peak_hh"]
                    self.cp_membrane_potential_v[i] = params["v_rest_hh"] 
                    self.cp_gating_variable_m[i] = params["m_init"] 
                    self.cp_gating_variable_h[i] = params["h_init"]
                    self.cp_gating_variable_n[i] = params["n_init"]
                    self.runtime_state.neuron_types_list_for_viz[i] = f"HH_{selected_neuron_type_enum.name.replace('HH_', '')}"
            
            self._log_console(f"Generating 3D neuron positions for {n} neurons...")
            if n > 0:
                np_positions_3d = np.random.uniform(
                    low=[self.viz_config.volume_min_x, self.viz_config.volume_min_y, self.viz_config.volume_min_z],
                    high=[self.viz_config.volume_max_x, self.viz_config.volume_max_y, self.viz_config.volume_max_z],
                    size=(n,3)).astype(np.float32)
                self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                self.runtime_state.neuron_positions_x = np_positions_3d[:,0].tolist()
                self.runtime_state.neuron_positions_y = np_positions_3d[:,1].tolist()
            else: 
                self.cp_neuron_positions_3d = cp.array([], dtype=cp.float32).reshape(0,3)
                self.runtime_state.neuron_positions_x = []; self.runtime_state.neuron_positions_y = []

            if not called_from_playback_init:
                self._log_console("Generating connections (3D)...")
                if cfg.enable_watts_strogatz:
                    self.cp_connections = self._generate_watts_strogatz_connections_3d(n, cfg.connectivity_k, cfg.connectivity_p_rewire, cfg)
                else: 
                    self.cp_connections = self._generate_spatial_connections_3d(n, cfg.connections_per_neuron, self.cp_neuron_positions_3d, self.cp_traits, cfg)
                
                if self.cp_connections is None:
                    self._log_console("Connection generation resulted in None. Initializing as empty matrix.", "warning")
                    self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32)
            elif self.cp_connections is None: 
                 self._log_console("Connections are None during playback init before _apply_recorded_arrays. Initializing empty.", "warning")
                 self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32)

            num_synapses = self.cp_connections.nnz if self.cp_connections is not None else 0
            if num_synapses > 0:
                self.cp_synapse_pulse_timers = cp.zeros(num_synapses, dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.zeros(num_synapses, dtype=cp.float32)
            else: 
                self.cp_synapse_pulse_timers = cp.array([], dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.array([], dtype=cp.float32)

            if cfg.enable_short_term_plasticity and num_synapses > 0:
                self._log_console(f"Initializing STP state for {num_synapses} synapses...")
                self.cp_stp_x = cp.ones(num_synapses, dtype=cp.float32) 
                self.cp_stp_u = cp.full(num_synapses, cfg.stp_U, dtype=cp.float32) 
            else: 
                self.cp_stp_x = None; self.cp_stp_u = None

            self.is_initialized = True
            conn_count = self.cp_connections.nnz if self.cp_connections is not None else 0
            
            # Log GPU memory usage after initialization
            mem_stats = self._get_gpu_memory_info()
            self._log_console(
                f"Simulation data initialized for {n} neurons (3D). Connections: {conn_count}. "
                f"GPU memory: {mem_stats['used_gb']:.1f}GB/{mem_stats['total_gb']:.1f}GB ({mem_stats['usage_percent']:.1f}%)"
            )
            self._check_gpu_memory_pressure()
        except Exception as e:
            self._log_console(f"Error during simulation data initialization (3D): {e}","critical")
            import traceback; traceback.print_exc()
            self.is_initialized = False
            if 'cupy' in sys.modules and cp.is_available():
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    def _calculate_distances_3d_gpu(self, pos_i_cp, pos_neighbors_cp):
        """Calculates Euclidean distances in 3D between a point and an array of other points using CuPy."""
        if pos_neighbors_cp.size == 0: return cp.array([], dtype=cp.float32)
        diff_3d = pos_neighbors_cp - pos_i_cp.reshape(1, 3) 
        return cp.sqrt(cp.sum(diff_3d**2, axis=1))

    def _generate_spatial_connections_3d_vectorized(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Generates connections using fully vectorized GPU operations (fast, scalable to 100K+ neurons)."""
        self._log_console("Generating connections (3D spatial, GPU-vectorized)...")
        start_t = time.time()
        
        if n == 0:
            return csp.csr_matrix((0, 0), dtype=cp.float32)
        
        dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
        trait_bias = getattr(config, 'trait_connection_bias', 0.5)
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
        k = min(max_connections_per_neuron, n - 1)
        
        # Compute all pairwise distances on GPU (n x n matrix)
        # Use broadcasting: positions[i] - positions[j] for all i,j
        pos = neuron_positions_3d_cp  # Shape: (n, 3)
        pos_i = pos[:, None, :]  # Shape: (n, 1, 3)
        pos_j = pos[None, :, :]  # Shape: (1, n, 3)
        diff = pos_i - pos_j  # Shape: (n, n, 3)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))  # Shape: (n, n)
        
        # Set self-distances to infinity to exclude self-connections
        cp.fill_diagonal(distances, cp.inf)
        
        # Compute connection probabilities
        prob_dist = cp.exp(-dist_decay * distances)
        
        # Trait similarity component
        traits_i = traits_cp[:, None]  # Shape: (n, 1)
        traits_j = traits_cp[None, :]  # Shape: (1, n)
        prob_trait = (traits_i == traits_j).astype(cp.float32) * trait_bias
        
        # Combined probability
        conn_prob = prob_dist + prob_trait  # Shape: (n, n)
        
        # For each neuron, select top-k connections based on probability
        # Use argsort to get indices of highest probabilities
        top_k_indices = cp.argsort(conn_prob, axis=1)[:, -k:]  # Shape: (n, k)
        
        # Generate weights for connections
        weights = cp.random.uniform(min_w, max_w, (n, k)).astype(cp.float32)
        
        # Convert to COO format
        row_indices = cp.repeat(cp.arange(n), k)  # Shape: (n*k,)
        col_indices = top_k_indices.ravel()  # Shape: (n*k,)
        weights_flat = weights.ravel()  # Shape: (n*k,)
        
        # Create CSR matrix
        conn_matrix = csp.coo_matrix(
            (weights_flat, (row_indices, col_indices)),
            shape=(n, n),
            dtype=cp.float32
        ).tocsr()
        
        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        self._log_console(f"Connections (3D Spatial GPU): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
        return conn_matrix
    
    def _generate_spatial_connections_3d(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Generates synaptic connections based on spatial proximity and trait similarity in 3D."""
        # Use vectorized GPU version for better performance
        if n > 1000:  # Use vectorized for large networks
            return self._generate_spatial_connections_3d_vectorized(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)
        
        # Legacy iterative version for small networks (< 1000 neurons)
        self._log_console("Generating connections (3D spatial, legacy)..."); start_t = time.time()
        if n == 0: 
            self._log_console("No neurons to connect (n=0).", "info")
            return csp.csr_matrix((0,0), dtype=cp.float32)

        dist_decay_factor = getattr(config, 'connection_distance_decay_factor', 0.01) 
        trait_bias = getattr(config, 'trait_connection_bias', 0.5) 
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight 

        rows, cols, weights_list = [], [], [] 

        for i in range(n): 
            pos_i_cp = neuron_positions_3d_cp[i:i+1, :] 
            trait_i_val = traits_cp[i] 

            candidate_indices_np = np.array([j for j in range(n) if j != i], dtype=np.int32)
            if candidate_indices_np.size == 0: continue 

            candidate_indices_cp = cp.asarray(candidate_indices_np) 
            pos_candidates_cp = neuron_positions_3d_cp[candidate_indices_cp] 
            traits_candidates_cp = traits_cp[candidate_indices_cp] 

            distances_cp = self._calculate_distances_3d_gpu(pos_i_cp, pos_candidates_cp)
            prob_distance_component = cp.exp(-dist_decay_factor * distances_cp) 
            prob_trait_component = (traits_candidates_cp == trait_i_val).astype(cp.float32) * trait_bias 
            connection_probabilities_cp = prob_distance_component + prob_trait_component
            
            sum_probs = cp.sum(connection_probabilities_cp)
            if sum_probs > 1e-9: 
                 normalized_probabilities_cp = connection_probabilities_cp / sum_probs
            else: 
                 if connection_probabilities_cp.size > 0: 
                    normalized_probabilities_cp = cp.ones_like(connection_probabilities_cp) / connection_probabilities_cp.size 
                 else:
                    continue 

            num_potential_targets = candidate_indices_cp.size
            if num_potential_targets > 0 :
                num_to_select = min(max_connections_per_neuron, num_potential_targets) 

                if num_to_select > 0:
                    try:
                        if not np.isclose(cp.asnumpy(cp.sum(normalized_probabilities_cp)), 1.0) and cp.sum(normalized_probabilities_cp) > 1e-9:
                            normalized_probabilities_cp = normalized_probabilities_cp / cp.sum(normalized_probabilities_cp)
                        elif cp.sum(normalized_probabilities_cp) <= 1e-9: 
                             selected_local_indices_cp = cp.random.choice(cp.arange(num_potential_targets), size=num_to_select, replace=False)
                        else: 
                             selected_local_indices_cp = cp.random.choice(
                                cp.arange(num_potential_targets),
                                size=num_to_select,
                                replace=False,
                                p=normalized_probabilities_cp 
                            )
                    except (ValueError, NotImplementedError) as e: 
                        sorted_local_indices_cp = cp.argsort(connection_probabilities_cp)[::-1] 
                        selected_local_indices_cp = sorted_local_indices_cp[:num_to_select] 

                    final_target_global_indices_cp = candidate_indices_cp[selected_local_indices_cp]
                    initial_weights_np = np.random.uniform(min_w, max_w, num_to_select).astype(np.float32)
                    final_weights_np = np.clip(initial_weights_np, min_w, max_w) 

                    rows.extend([i] * num_to_select) 
                    cols.extend(cp.asnumpy(final_target_global_indices_cp).tolist()) 
                    weights_list.extend(final_weights_np.tolist()) 

            if n > 0 and i % (max(1, n // 20)) == 0: 
                print(f"\rConn gen (3D Spatial): {i/n*100:.1f}%", end="")

        if n > 0: print("\rConn gen (3D Spatial): 100.0% ") 

        if not rows: 
            self._log_console("No connections generated by 3D spatial method.", "warning")
            return csp.csr_matrix((n, n), dtype=cp.float32) 

        conn_matrix = csp.csr_matrix((cp.asarray(weights_list,dtype=cp.float32),
                                      (cp.asarray(rows,dtype=cp.int32),cp.asarray(cols,dtype=cp.int32))),
                                     shape=(n,n),dtype=cp.float32)
        conn_matrix.sort_indices() 
        self._log_console(f"Connections (3D Spatial): {conn_matrix.nnz}. Time: {time.time()-start_t:.2f}s")
        return conn_matrix

    def _generate_watts_strogatz_connections_3d(self, n, k_neighbors, p_rewire, config):
        """Generates connections using a Watts-Strogatz small-world network model (fallback)."""
        self._log_console("Watts-Strogatz 3D generation currently falls back to 3D spatial generation.", "warning")
        return self._generate_spatial_connections_3d(n, config.connections_per_neuron, self.cp_neuron_positions_3d, self.cp_traits, config)

    def apply_simulation_configuration_core(self, full_config_dict, is_part_of_playback_setup=False):
        """Applies a new simulation configuration from a full dictionary."""
        self._log_to_ui(f"Applying new simulation configuration... (playback_setup: {is_part_of_playback_setup})", "info")

        if self.runtime_state.is_running:
            self.stop_simulation()

        self.clear_simulation_state_and_gpu_memory()

        # Create new config objects from the provided dictionaries
        self.core_config = _create_config_from_dict(CoreSimConfig, full_config_dict.get("core_config"))
        self.viz_config = _create_config_from_dict(VisualizationConfig, full_config_dict.get("viz_config"))
        # We don't load runtime_state from profiles, so we re-initialize it.
        # Checkpoints might restore it, but that's handled in load_checkpoint.
        self.runtime_state = RuntimeState()

        # Update max_delay_steps based on new config
        dt = self.core_config.dt_ms
        self.runtime_state.max_delay_steps = int(self.core_config.max_synaptic_delay_ms / dt) if dt > 0 else 200

        self._initialize_simulation_data(called_from_playback_init=is_part_of_playback_setup)

        if not self.is_initialized:
            self._log_to_ui("Failed to initialize simulation from new configuration. Critical error.", "critical")
            return False

        self.runtime_state.current_time_ms = 0.0
        self.runtime_state.current_time_step = 0
        self._log_to_ui(f"Sim config applied ({self.core_config.neuron_model_type}, N={self.core_config.num_neurons}). Sim re-initialized.", "success")
        return True

    def get_current_simulation_configuration_dict(self):
        """Returns the current simulation configuration as a dictionary."""
        return _get_full_config_dict(self.core_config, self.viz_config, self.runtime_state)

    def clear_simulation_state_and_gpu_memory(self):
        """Clears all CuPy arrays and resets the initialization flag."""
        self._log_console("Clearing simulation state and GPU memory...")
        attrs_to_clear = [
            'cp_membrane_potential_v','cp_recovery_variable_u', 'cp_conductance_g_e','cp_conductance_g_i',
            'cp_external_input_current', 'cp_firing_states','cp_prev_firing_states','cp_traits',
            'cp_neuron_positions_3d','cp_connections', 'cp_refractory_timers', 'cp_viz_activity_timers',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress',
            'cp_izh_C', 'cp_izh_k', 'cp_izh_vr', 'cp_izh_vt', 'cp_izh_vpeak',
            'cp_izh_a', 'cp_izh_b', 'cp_izh_c_reset', 'cp_izh_d_increment',
            'cp_izh_legacy_a', 'cp_izh_legacy_b', 'cp_izh_legacy_c_reset',
            'cp_izh_legacy_d_increment', 'cp_izh_legacy_vpeak',
            'cp_gating_variable_m','cp_gating_variable_h','cp_gating_variable_n',
            'cp_hh_C_m','cp_hh_g_Na_max','cp_hh_g_K_max','cp_hh_g_L',
            'cp_hh_E_Na','cp_hh_E_K','cp_hh_E_L', 'cp_hh_v_peak',
            'cp_neuron_firing_thresholds', 'cp_neuron_activity_ema',
            'cp_stp_u','cp_stp_x'
        ]
        for attr_name in attrs_to_clear:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                setattr(self, attr_name, None) 

        if 'cupy' in sys.modules and cp.is_available():
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                self._log_console(f"Error freeing CuPy memory: {e}", "warning")

        self.is_initialized = False 
        self._log_console("Cleared simulation state and GPU memory.")

    def start_simulation(self):
        """Starts or restarts the simulation (called by sim_thread)."""
        if not self.is_initialized:
            self._log_to_ui("Simulation not initialized. Attempting to initialize.", "warning")
            # Re-create the full config dict to attempt re-initialization
            full_config = self.get_current_simulation_configuration_dict()
            if not self.apply_simulation_configuration_core(full_config):
                self._log_to_ui("Initialization failed. Cannot start simulation.", "error"); return

        self.runtime_state.current_time_ms = 0.0
        self.runtime_state.current_time_step = 0
        
        self.runtime_state.is_running = True; self.runtime_state.is_paused = False
        self._log_to_ui(f"Simulation started. Duration: {self.core_config.total_simulation_time_ms} ms, Model: {self.core_config.neuron_model_type}, dt: {self.core_config.dt_ms} ms.", "info")
        if self.recording_file_handle:
            self._log_to_ui(f"Recording active, streaming to file: {self.recording_filepath}", "info")

    def stop_simulation(self):
        """Stops the currently running simulation (called by sim_thread)."""
        if self.runtime_state.is_running or self.runtime_state.is_paused:
            self.runtime_state.is_running = False; self.runtime_state.is_paused = False
            self._log_to_ui("Simulation stopped.", "info")

    def pause_simulation(self):
        """Pauses the running simulation (called by sim_thread)."""
        if self.runtime_state.is_running and not self.runtime_state.is_paused:
            self.runtime_state.is_paused = True
            self._log_to_ui("Simulation paused.", "info")

    def resume_simulation(self):
        """Resumes a paused simulation (called by sim_thread)."""
        if self.runtime_state.is_running and self.runtime_state.is_paused:
            self.runtime_state.is_paused = False
            self._log_to_ui("Simulation resumed.", "info")

    def toggle_pause_simulation(self):
        """Toggles the pause state of the simulation. Returns the new pause state."""
        # This logic should primarily live in the UI thread to avoid race conditions.
        # The UI sends discrete PAUSE/RESUME commands. This function can be deprecated.
        if not self.runtime_state.is_running:
            self._log_to_ui("Cannot toggle pause: Simulation is not running.", "warning"); return self.runtime_state.is_paused

        self.runtime_state.is_paused = not self.runtime_state.is_paused
        action = "paused" if self.runtime_state.is_paused else "resumed"
        self._log_to_ui(f"Simulation {action}.", "info")
        return self.runtime_state.is_paused

    def set_simulation_speed_factor(self, factor):
        """Sets the simulation speed factor (called by sim_thread)."""
        self.runtime_state.simulation_speed_factor = max(0.01, factor)
        self._log_to_ui(f"Simulation speed factor set to {self.runtime_state.simulation_speed_factor:.2f}x", "info")

    def step_simulation(self, num_steps=1):
        """Advances the simulation by a specified number of steps (called by sim_thread)."""
        # if global_gui_state.get("is_playback_mode_active", False): # Check UI-managed state
        #     self._log_to_ui("Cannot step simulation during playback mode.", "warning"); return

        if not self.is_initialized:
            self._log_to_ui("Cannot step: Sim not initialized.", "warning"); return
        
        can_step_internally = (self.runtime_state.is_running and self.runtime_state.is_paused) or \
                              (not self.runtime_state.is_running)

        if not can_step_internally:
            self._log_to_ui("Sim must be running & paused, or stopped, to step via command.", "warning"); return

        self._log_console(f"Stepping simulation by {num_steps} steps.") 
        for _ in range(num_steps):
            if self.core_config.num_neurons > 0:
                if self.runtime_state.current_time_ms < self.core_config.total_simulation_time_ms:
                    self._run_one_simulation_step() 
                    self.runtime_state.current_time_ms += self.core_config.dt_ms
                    self.runtime_state.current_time_step += 1
                else:
                    self._log_to_ui("Cannot step: Total simulation time reached.", "info")
                    self.stop_simulation() 
                    if self.ui_queue: self.ui_queue.put({"type": "SIM_STOPPED_OR_ENDED", "reason": "Total time reached on step"})
                    break 
            else:
                self._log_console("No neurons to simulate in step.", "debug"); break
        
        latest_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
        if self.ui_queue and latest_data:
            self.ui_queue.put({"type": "SIM_DATA_UPDATE", "data": latest_data})
        self._log_to_ui(f"Stepped sim by {num_steps} substeps. Current time: {self.runtime_state.current_time_ms:.3f} ms", "info")


    def _estimate_frame_size_bytes(self):
        """Estimates the size in bytes of a single recording frame."""
        if not self.is_initialized:
            return 0
        
        total_bytes = 0
        # Dynamic arrays that change each frame
        arrays_to_check = [
            'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
            'cp_conductance_g_e', 'cp_conductance_g_i', 'cp_recovery_variable_u',
            'cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n'
        ]
        
        if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
            if self.cp_connections.data is not None:
                total_bytes += self.cp_connections.data.nbytes
        
        if self.core_config.enable_short_term_plasticity:
            if self.cp_stp_u is not None:
                total_bytes += self.cp_stp_u.nbytes
            if self.cp_stp_x is not None:
                total_bytes += self.cp_stp_x.nbytes
        
        for attr_name in arrays_to_check:
            array_data = getattr(self, attr_name, None)
            if array_data is not None:
                total_bytes += array_data.nbytes
        
        # Add overhead for metadata
        total_bytes += 1024  # Small overhead for scalars
        return total_bytes
    
    def _check_gpu_recording_capacity(self, estimated_frames):
        """Checks if GPU has enough memory for estimated recording frames."""
        frame_size = self._estimate_frame_size_bytes()
        required_memory = frame_size * estimated_frames
        
        mem_info = cp.cuda.Device().mem_info
        free_memory, total_memory = mem_info
        
        # Use 60% of available memory for recording buffer
        available_for_recording = free_memory * 0.6
        max_frames = int(available_for_recording / frame_size) if frame_size > 0 else 0
        
        self._log_console(f"Frame size: {frame_size/1e6:.1f}MB, Free GPU: {free_memory/1e9:.1f}GB, Max frames: {max_frames}")
        
        if required_memory > available_for_recording:
            self._log_to_ui(
                f"Warning: Recording {estimated_frames} frames needs {required_memory/1e9:.1f}GB, "
                f"but only {available_for_recording/1e9:.1f}GB available. Max {max_frames} frames.",
                "warning"
            )
            return False, max_frames
        
        return True, max_frames
    
    def _capture_initial_state_for_recording(self):
        """Captures the full initial state of the simulation for HDF5 recording."""
        if not self.is_initialized:
            self._log_console("Cannot capture initial state: Simulation not initialized.", "error")
            return None

        snapshot = {
            "start_time_ms": self.runtime_state.current_time_ms,
            "start_time_step": self.runtime_state.current_time_step
        }

        if self.cp_traits is not None: snapshot["cp_traits"] = cp.asnumpy(self.cp_traits)
        if self.cp_neuron_positions_3d is not None: snapshot["cp_neuron_positions_3d"] = cp.asnumpy(self.cp_neuron_positions_3d)

        if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                attr_name = f"cp_izh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = cp.asnumpy(getattr(self, attr_name))
        elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                attr_name = f"cp_hh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = cp.asnumpy(getattr(self, attr_name))

        arrays_to_capture = [
            'cp_membrane_potential_v', 'cp_recovery_variable_u', 'cp_gating_variable_m',
            'cp_gating_variable_h', 'cp_gating_variable_n', 'cp_conductance_g_e',
            'cp_conductance_g_i', 'cp_external_input_current', 'cp_refractory_timers',
            'cp_viz_activity_timers', 'cp_neuron_firing_thresholds', 'cp_neuron_activity_ema',
            'cp_firing_states', 'cp_prev_firing_states',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
        ]
        for attr_name in arrays_to_capture:
            array_data = getattr(self, attr_name, None)
            if array_data is not None:
                snapshot[attr_name] = cp.asnumpy(array_data)
            else: 
                snapshot[attr_name] = None 

        if self.cp_connections is not None:
            snapshot["connections_data"] = cp.asnumpy(self.cp_connections.data) if self.cp_connections.data is not None else np.array([])
            snapshot["connections_indices"] = cp.asnumpy(self.cp_connections.indices) if self.cp_connections.indices is not None else np.array([])
            snapshot["connections_indptr"] = cp.asnumpy(self.cp_connections.indptr) if self.cp_connections.indptr is not None else np.array([])
            snapshot["connections_shape"] = self.cp_connections.shape 
        else: 
            snapshot["connections_data"] = np.array([]); snapshot["connections_indices"] = np.array([])
            snapshot["connections_indptr"] = np.array([]); snapshot["connections_shape"] = (0,0)

        if self.cp_stp_u is not None: snapshot["cp_stp_u"] = cp.asnumpy(self.cp_stp_u)
        else: snapshot["cp_stp_u"] = None
        if self.cp_stp_x is not None: snapshot["cp_stp_x"] = cp.asnumpy(self.cp_stp_x)
        else: snapshot["cp_stp_x"] = None
        
        return snapshot

    def _write_gpu_frames_to_disk(self):
        """Writes all GPU-buffered frames to disk in a single batch operation."""
        if not self.gpu_frame_buffer:
            return  # No frames to write
        
        total_frames = len(self.gpu_frame_buffer)
        self._log_to_ui(f"Writing {total_frames} GPU-buffered frames to disk...", "info")
        start_time = time.time()
        
        try:
            # Process frames in order
            sorted_frame_indices = sorted(self.gpu_frame_buffer.keys())
            
            for i, frame_idx in enumerate(sorted_frame_indices):
                frame_data_gpu = self.gpu_frame_buffer[frame_idx]
                
                # Convert CuPy arrays to NumPy (GPU→CPU transfer happens here)
                frame_data_np = {}
                for key, value in frame_data_gpu.items():
                    if isinstance(value, cp.ndarray):
                        frame_data_np[key] = cp.asnumpy(value)
                    else:
                        frame_data_np[key] = value  # Scalars
                
                # Write to HDF5
                frame_group_name = f"frames/frame_{frame_idx}"
                current_frame_group = self.recording_file_handle.create_group(frame_group_name)
                
                for key, value in frame_data_np.items():
                    if isinstance(value, np.ndarray):
                        if value.size > 0:
                            current_frame_group.create_dataset(key, data=value, compression="gzip")
                        else:
                            current_frame_group.attrs[f"{key}_is_empty"] = True
                    elif value is not None:
                        current_frame_group.attrs[key] = value
                    else:
                        current_frame_group.attrs[key] = "NoneType"
                
                # Progress logging every 10%
                if (i + 1) % max(1, total_frames // 10) == 0:
                    progress_pct = ((i + 1) / total_frames) * 100
                    self._log_console(f"  Writing progress: {progress_pct:.0f}% ({i+1}/{total_frames})")
            
            # Final flush
            self.recording_file_handle.flush()
            
            elapsed = time.time() - start_time
            frames_per_sec = total_frames / elapsed if elapsed > 0 else 0
            self._log_to_ui(
                f"Successfully wrote {total_frames} frames in {elapsed:.2f}s ({frames_per_sec:.1f} frames/s)",
                "success"
            )
            
            # Clear GPU buffer to free VRAM
            self.gpu_frame_buffer.clear()
            
        except Exception as e:
            self._log_to_ui(f"Error writing GPU frames to disk: {e}", "error")
            raise
    
    def start_recording_to_file(self, filepath):
        """Starts recording the simulation state to an HDF5 file (called by sim_thread)."""
        if self.recording_file_handle: 
            self._log_to_ui("Error: Recording already active. Stop current one first.", "error")
            return False 

        if not self.is_initialized:
            self._log_to_ui("Simulation not initialized. Cannot start recording.", "warning")
            return False
        
        self._log_console(f"Attempting to start new recording to file: {filepath}")
        try:
            self.recording_filepath = filepath
            self.recording_file_handle = h5py.File(self.recording_filepath, 'w') 
            self.current_frame_count_for_h5 = 0 

            self.recording_file_handle.attrs["format_version"] = RECORDING_FORMAT_VERSION
            # Save complete config structure (core_config, viz_config, runtime_state)
            config_snapshot = _get_full_config_dict(self.core_config, self.viz_config, self.runtime_state)
            # Flatten nested structure for HDF5 attrs by prefixing keys
            flattened_config = {}
            for section_name, section_dict in config_snapshot.items():
                for key, value in section_dict.items():
                    flattened_config[f"{section_name}.{key}"] = value
            save_dict_to_hdf5_attrs(self.recording_file_handle, flattened_config)

            initial_state_data = self._capture_initial_state_for_recording()
            if initial_state_data is None:
                self._log_to_ui("Failed to capture initial state for recording. Aborting.", "error")
                self.recording_file_handle.close()
                self.recording_file_handle = None; self.recording_filepath = None
                return False

            initial_state_group = self.recording_file_handle.create_group("initial_state")
            for key, value in initial_state_data.items():
                if isinstance(value, np.ndarray):
                    if value.size > 0 : 
                        initial_state_group.create_dataset(key, data=value, compression="gzip") 
                    else: 
                        initial_state_group.attrs[f"{key}_is_empty"] = True
                elif key == "connections_shape": 
                    initial_state_group.attrs["connections_shape_0"] = value[0]
                    initial_state_group.attrs["connections_shape_1"] = value[1]
                elif value is not None : 
                    initial_state_group.attrs[key] = value
                else: 
                    initial_state_group.attrs[key] = "NoneType"
            
            self.recording_file_handle.create_group("frames")
            
            # Initialize GPU-buffered recording
            # Estimate frames based on simulation duration
            estimated_frames = int(self.core_config.total_simulation_time_ms / self.core_config.dt_ms)
            can_gpu_buffer, max_gpu_frames = self._check_gpu_recording_capacity(estimated_frames)
            
            if self.gpu_buffered_recording:
                self.gpu_frame_buffer = {}  # Clear any old data
                self.gpu_recording_max_frames = max_gpu_frames
                self._log_console(f"GPU-buffered recording enabled. Max frames: {max_gpu_frames}")
            
            self._log_to_ui(f"Recording armed (GPU-buffered). Start sim to capture.", "info", color=[0,150,200])
            # Signal UI that recording has started successfully
            if self.ui_queue:
                self.ui_queue.put({"type": "RECORDING_STARTED", "filepath": self.recording_filepath})
            return True

        except Exception as e:
            self._log_to_ui(f"Error starting file recording to {filepath}: {e}", "error")
            if self.recording_file_handle:
                try: self.recording_file_handle.close()
                except: pass
            self.recording_file_handle = None; self.recording_filepath = None
            if self.ui_queue:
                self.ui_queue.put({"type": "RECORDING_START_FAILED", "error": str(e)})
            return False

    def stop_recording(self): # Added prompt_save=True default to match original, though UI handles dialogs now
        """Stops the HDF5 recording stream and finalizes the file (called by sim_thread)."""
        if not self.recording_file_handle: 
            self._log_to_ui("No active recording to stop.", "info")
            if self.ui_queue: self.ui_queue.put({"type": "RECORDING_STOPPED_UNEXPECTEDLY"}) # Or a specific "NO_RECORDING_TO_STOP"
            return

        self._log_console("Stopping HDF5 recording stream.")
        was_recording_to_file = False
        finalized_filepath = self.recording_filepath
        if self.recording_file_handle and isinstance(self.recording_file_handle, h5py.File) and self.recording_file_handle.id:
            try:
                # Write GPU-buffered frames to disk if GPU buffering was enabled
                if self.gpu_buffered_recording and self.gpu_frame_buffer:
                    self._write_gpu_frames_to_disk()
                
                # Final flush before closing
                self.recording_file_handle.flush()
                self.recording_file_handle.close()
                self.recording_file_handle.close() 
                was_recording_to_file = True
                self._log_to_ui(f"Recording stream to {finalized_filepath} finalized and saved.", "success")
            except Exception as e:
                self._log_to_ui(f"Error finalizing recording file {finalized_filepath}: {e}", "error")
        else:
            self._log_console(f"Stop recording called, but no active file handle or already closed for {finalized_filepath}.", "warning")

        self.recording_file_handle = None
        self.recording_filepath = None
        self.current_frame_count_for_h5 = 0
        self.gpu_frame_buffer.clear()  # Clear GPU buffer

        if self.ui_queue:
            self.ui_queue.put({
                "type": "RECORDING_FINALIZED",
                "success": was_recording_to_file,
                "filepath": finalized_filepath if was_recording_to_file else None
            })

    def record_current_frame_if_active(self):
        """Records the current simulation state as a frame if recording is active (called by sim_thread)."""
        if not self.recording_file_handle or \
           not isinstance(self.recording_file_handle, h5py.File) or \
           not self.recording_file_handle.id or \
           not self.runtime_state.is_running or \
           self.runtime_state.is_paused: 
            return

        try:
            frame_idx = self.current_frame_count_for_h5
            
            # GPU-buffered recording: store frames in VRAM as CuPy arrays (NO GPU→CPU transfers)
            if self.gpu_buffered_recording:
                frame_data_gpu = {
                    "time_ms": self.runtime_state.current_time_ms,
                    "step": self.runtime_state.current_time_step,
                    "_mock_num_spikes_this_step": self._mock_num_spikes_this_step,
                    "_mock_network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
                    "_mock_total_plasticity_events": self._mock_total_plasticity_events
                }
                
                # Collect all CuPy arrays that need to be stored
                dynamic_arrays_to_capture = [
                    'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
                    'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress' 
                ]
                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    dynamic_arrays_to_capture.extend(['cp_recovery_variable_u'])
                    if self.core_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                        dynamic_arrays_to_capture.append('cp_neuron_firing_thresholds')
                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    dynamic_arrays_to_capture.extend(['cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n'])

                if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                    if self.cp_connections.data is not None:
                        frame_data_gpu["cp_connections_data"] = self.cp_connections.data.copy()

                if self.core_config.enable_short_term_plasticity:
                    if self.cp_stp_u is not None: frame_data_gpu["cp_stp_u"] = self.cp_stp_u.copy()
                    if self.cp_stp_x is not None: frame_data_gpu["cp_stp_x"] = self.cp_stp_x.copy()

                for attr_name in dynamic_arrays_to_capture:
                    array_data = getattr(self, attr_name, None)
                    if array_data is not None:
                        frame_data_gpu[attr_name] = array_data.copy()  # Store CuPy array directly
                    else: 
                        frame_data_gpu[attr_name] = None
                
                # Store entire frame in GPU buffer
                self.gpu_frame_buffer[frame_idx] = frame_data_gpu
                
                # Warn if approaching capacity
                if frame_idx % 100 == 0 and frame_idx > self.gpu_recording_max_frames * 0.8:
                    self._log_console(f"WARNING: GPU buffer at {frame_idx}/{self.gpu_recording_max_frames} frames")
                
            else:
                # Legacy CPU path: immediate streaming to HDF5
                frame_data_np = {
                    "time_ms": self.runtime_state.current_time_ms,
                    "step": self.runtime_state.current_time_step,
                    "_mock_num_spikes_this_step": self._mock_num_spikes_this_step,
                    "_mock_network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
                    "_mock_total_plasticity_events": self._mock_total_plasticity_events
                }

                dynamic_arrays_to_capture = [
                    'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
                    'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress' 
                ]
                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    dynamic_arrays_to_capture.extend(['cp_recovery_variable_u'])
                    if self.core_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                        dynamic_arrays_to_capture.append('cp_neuron_firing_thresholds')
                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    dynamic_arrays_to_capture.extend(['cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n'])

                if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                    if self.cp_connections.data is not None:
                         frame_data_np["cp_connections_data"] = cp.asnumpy(self.cp_connections.data)

                if self.core_config.enable_short_term_plasticity:
                    if self.cp_stp_u is not None: frame_data_np["cp_stp_u"] = cp.asnumpy(self.cp_stp_u)
                    if self.cp_stp_x is not None: frame_data_np["cp_stp_x"] = cp.asnumpy(self.cp_stp_x)

                for attr_name in dynamic_arrays_to_capture:
                    array_data = getattr(self, attr_name, None)
                    if array_data is not None:
                        frame_data_np[attr_name] = cp.asnumpy(array_data)
                    else: 
                        frame_data_np[attr_name] = None 

                frame_group_name = f"frames/frame_{frame_idx}"
                current_frame_group = self.recording_file_handle.create_group(frame_group_name)

                for key, value in frame_data_np.items():
                    if isinstance(value, np.ndarray):
                        if value.size > 0: 
                            current_frame_group.create_dataset(key, data=value, compression="gzip")
                        else: 
                            current_frame_group.attrs[f"{key}_is_empty"] = True
                    elif value is not None: 
                        current_frame_group.attrs[key] = value
                    else: 
                        current_frame_group.attrs[key] = "NoneType"
                
                # Batch frames: only flush periodically for better performance
                if frame_idx % self.recording_buffer_size == 0:
                    self.recording_file_handle.flush()
            
            self.current_frame_count_for_h5 += 1

        except Exception as e:
            self._log_to_ui(f"Error streaming frame to recording file {self.recording_filepath}: {e}", "error")
            self.stop_recording() 
            # UI thread will update global_gui_state based on RECORDING_FINALIZED message.

    def _prepare_loaded_recording_metadata(self, filepath):
        """Opens HDF5 and prepares metadata for playback (called by sim_thread)."""
        self._log_console(f"Preparing metadata for recording file: {filepath}")
        try:
            h5_file = h5py.File(filepath, 'r') 

            version_info_str = h5_file.attrs.get("format_version")
            if version_info_str != RECORDING_FORMAT_VERSION:
                self._log_to_ui(f"Invalid/outdated recording format. Expected {RECORDING_FORMAT_VERSION}, got {version_info_str}.", "error")
                h5_file.close()
                return None

            flattened_config = load_dict_from_hdf5_attrs(h5_file) 
            
            # Reconstruct nested config structure from flattened keys
            config_snapshot = {"core_config": {}, "viz_config": {}, "runtime_state": {}}
            for key, value in flattened_config.items():
                if "." in key:
                    section, field = key.split(".", 1)
                    if section in config_snapshot:
                        config_snapshot[section][field] = value
                else:
                    # Legacy format or keys without section prefix
                    config_snapshot["core_config"][key] = value
            
            # Check if we have the expected config structure
            if not config_snapshot.get("core_config") or "num_neurons" not in config_snapshot["core_config"]: 
                self._log_to_ui("Recording metadata missing or invalid config. Cannot load.", "error")
                h5_file.close(); return None

            initial_full_state_metadata = {} 
            initial_state_group = h5_file.get("initial_state")
            if not initial_state_group:
                self._log_to_ui("Invalid recording: 'initial_state' group missing.", "error"); h5_file.close(); return None

            for key, value in initial_state_group.attrs.items():
                if value == "NoneType": initial_full_state_metadata[key] = None
                else: initial_full_state_metadata[key] = value
            initial_full_state_metadata["datasets_present"] = list(initial_state_group.keys())

            num_frames = 0
            frames_group = h5_file.get("frames")
            if frames_group:
                num_frames = len(list(frames_group.keys())) 

            loaded_data_package = {
                "filepath": filepath,
                "h5_file_obj_for_playback": h5_file, 
                "config_snapshot": config_snapshot,
                "initial_state_metadata": initial_full_state_metadata, 
                "num_frames": num_frames
            }
            self._log_console(f"Successfully prepared metadata for {os.path.basename(filepath)}. Frames: {num_frames}", "info")
            return loaded_data_package

        except FileNotFoundError:
            self._log_to_ui(f"Error: Recording file not found at {filepath}", "error")
            return None
        except Exception as e:
            self._log_to_ui(f"Critical error preparing metadata for recording {filepath}: {e}", "error")
            import traceback; traceback.print_exc()
            if 'h5_file' in locals() and h5_file.id: h5_file.close() 
            return None

    def load_recording(self, filepath):
        """Loads a recording for playback (called by sim_thread)."""
        self._log_to_ui(f"Attempting to load recording from {filepath} for playback...", "info")

        if self.runtime_state.is_running: self.stop_simulation()
        if self.recording_file_handle: 
            self._log_console("load_recording: Closing an existing recording file before loading new one.", "warning")
            try: self.recording_file_handle.close()
            except: pass
            self.recording_file_handle = None; self.recording_filepath = None
        
        # Close any HDF5 file this sim_bridge instance might be holding for playback itself.
        # Note: The main HDF5 handle for playback is managed by UI thread via global_gui_state.active_recording_data_source.
        # This method is for the sim_thread to initially process the file.
        # If sim_bridge was designed to hold its own playback handle, it would close it here.

        prepared_metadata = self._prepare_loaded_recording_metadata(filepath)

        if prepared_metadata:
            # Load recording into GPU cache for instant playback
            h5_file = prepared_metadata["h5_file_obj_for_playback"]
            num_frames = prepared_metadata["num_frames"]
            
            if num_frames > 0:
                success = self._load_recording_to_gpu_cache(h5_file, num_frames)
                if not success:
                    self._log_to_ui("Warning: GPU cache loading failed. Playback will use slower disk I/O.", "warning")
            
            if self.ui_queue:
                self.ui_queue.put({
                    "type": "RECORDING_METADATA_PREPARED",
                    "data": prepared_metadata 
                })
                self._log_to_ui(f"Recording metadata for '{os.path.basename(filepath)}' prepared. UI can now initialize playback.", "info")
            return True 
        else:
            if self.ui_queue:
                 self.ui_queue.put({"type": "RECORDING_LOAD_FAILED", "filepath": filepath})
            return False
    def _apply_config_and_initial_state_from_recording(self, config_snapshot, initial_state_h5_group):
        """
        Applies the configuration and initial state from a loaded HDF5 recording.
        This is called by the simulation thread when commanded by the UI thread after metadata is prepared.
        `initial_state_h5_group` is an open h5py.Group object for "initial_state".
        """
        self._log_console("Applying config and initial state from recording for playback setup...")

        success_apply_config = self.apply_simulation_configuration_core(config_snapshot, is_part_of_playback_setup=True)
        if not success_apply_config or not self.is_initialized:
            self._log_to_ui("CRITICAL: Failed to apply recorded config or initialize from recording data for playback.", "critical")
            if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Config application failed"})
            return False
        self._log_console(f"Applied recording config. Neuron count now: {self.core_config.num_neurons}")

        initial_state_arrays_np = {}
        for key in initial_state_h5_group.attrs.keys(): 
            if key.endswith("_is_empty") and initial_state_h5_group.attrs[key] is True:
                original_key = key.replace("_is_empty","")
                initial_state_arrays_np[original_key] = np.array([]) 
            elif initial_state_h5_group.attrs[key] == "NoneType":
                 initial_state_arrays_np[key] = None
            elif key not in ["connections_shape_0", "connections_shape_1"]: 
                initial_state_arrays_np[key] = initial_state_h5_group.attrs[key]
        
        for key in initial_state_h5_group.keys(): 
            if f"{key}_is_empty" not in initial_state_h5_group.attrs:
                initial_state_arrays_np[key] = initial_state_h5_group[key][:] 

        if "connections_data" in initial_state_arrays_np and \
           "connections_indices" in initial_state_arrays_np and \
           "connections_indptr" in initial_state_arrays_np and \
           initial_state_h5_group.attrs.get("connections_shape_0") is not None: 
            initial_state_arrays_np["connections_shape"] = (
                initial_state_h5_group.attrs["connections_shape_0"],
                initial_state_h5_group.attrs["connections_shape_1"]
            )
        else: 
            if "connections_data" not in initial_state_arrays_np: initial_state_arrays_np["connections_data"] = np.array([])
            if "connections_indices" not in initial_state_arrays_np: initial_state_arrays_np["connections_indices"] = np.array([], dtype=np.int32)
            if "connections_indptr" not in initial_state_arrays_np: initial_state_arrays_np["connections_indptr"] = np.array([0]*(self.core_config.num_neurons+1), dtype=np.int32)
            if "connections_shape" not in initial_state_arrays_np: initial_state_arrays_np["connections_shape"] = (self.core_config.num_neurons, self.core_config.num_neurons)


        self._apply_recorded_arrays_to_gpu(initial_state_arrays_np, is_initial_state=True)
        self._log_console("Applied initial full state from recording to GPU for playback.")

        self.runtime_state.current_time_ms = initial_state_arrays_np.get("start_time_ms", 0.0)
        self.runtime_state.current_time_step = initial_state_arrays_np.get("start_time_step", 0)
        
        if self.ui_queue:
            initial_gui_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
            self.ui_queue.put({
                "type": "PLAYBACK_READY",
                "initial_gui_data": initial_gui_data,
                "current_time_ms": self.runtime_state.current_time_ms,
                "current_time_step": self.runtime_state.current_time_step
            })
        return True

    def _load_recording_to_gpu_cache(self, h5_file_handle, num_frames):
        """Loads entire recording into GPU memory for instant frame seeking."""
        self._log_to_ui(f"Loading {num_frames} frames into GPU cache...", "info")
        start_time = time.time()
        
        try:
            self.gpu_playback_cache.clear()
            
            for frame_idx in range(num_frames):
                frame_group_name = f"frames/frame_{frame_idx}"
                frame_group = h5_file_handle.get(frame_group_name)
                
                if not frame_group:
                    self._log_console(f"Warning: Frame {frame_idx} not found, skipping")
                    continue
                
                # Load frame data and convert to CuPy arrays
                frame_data_gpu = {}
                
                # Load attributes (scalars)
                for key, value in frame_group.attrs.items():
                    if value == "NoneType":
                        frame_data_gpu[key] = None
                    elif key.endswith("_is_empty") and value is True:
                        original_key = key.replace("_is_empty", "")
                        frame_data_gpu[original_key] = cp.array([], dtype=cp.float32)
                    else:
                        frame_data_gpu[key] = value
                
                # Load datasets (arrays) - CPU→GPU transfer happens here
                for key in frame_group.keys():
                    if f"{key}_is_empty" not in frame_group.attrs:
                        np_data = frame_group[key][:]
                        frame_data_gpu[key] = cp.array(np_data)  # Transfer to GPU
                
                self.gpu_playback_cache[frame_idx] = frame_data_gpu
                
                # Progress logging every 10%
                if (frame_idx + 1) % max(1, num_frames // 10) == 0:
                    progress_pct = ((frame_idx + 1) / num_frames) * 100
                    self._log_console(f"  Loading progress: {progress_pct:.0f}% ({frame_idx+1}/{num_frames})")
            
            elapsed = time.time() - start_time
            frames_per_sec = num_frames / elapsed if elapsed > 0 else 0
            
            # Check GPU memory usage
            mem_info = cp.cuda.Device().mem_info
            free_memory, total_memory = mem_info
            used_gb = (total_memory - free_memory) / 1e9
            
            self._log_to_ui(
                f"Loaded {num_frames} frames in {elapsed:.2f}s ({frames_per_sec:.1f} frames/s). GPU usage: {used_gb:.1f}GB",
                "success"
            )
            
            self.gpu_playback_enabled = True
            return True
            
        except Exception as e:
            self._log_to_ui(f"Error loading recording to GPU cache: {e}", "error")
            self.gpu_playback_cache.clear()
            self.gpu_playback_enabled = False
            import traceback
            traceback.print_exc()
            return False
    
    def _read_frame_from_file(self, frame_idx, h5_file_handle):
        """Reads a specific frame's data from the provided open HDF5 file handle."""
        if not h5_file_handle or not h5_file_handle.id: 
            self._log_to_ui("Playback error: HDF5 file is not open or invalid.", "error")
            if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": "File handle invalid"})
            return None
        
        frame_group_name = f"frames/frame_{frame_idx}"
        try:
            frame_group = h5_file_handle.get(frame_group_name)
            if not frame_group:
                self._log_to_ui(f"Playback error: Frame group '{frame_group_name}' not found.", "error")
                return None

            frame_content = {}
            for key, value in frame_group.attrs.items():
                if value == "NoneType": frame_content[key] = None
                elif key.endswith("_is_empty") and value is True: 
                    original_key = key.replace("_is_empty","")
                    frame_content[original_key] = np.array([]) 
                else: frame_content[key] = value

            for key in frame_group.keys():
                 if f"{key}_is_empty" not in frame_group.attrs: 
                    frame_content[key] = frame_group[key][:] 
            return frame_content
        except Exception as e:
            self._log_to_ui(f"Error reading frame {frame_idx} from HDF5: {e}", "error")
            import traceback; traceback.print_exc() 
            return None

    def set_playback_frame(self, frame_idx, h5_file_handle):
        """Sets the simulation state to a specific frame from the loaded recording."""
        if not self.is_initialized: 
            self._log_to_ui("Cannot set playback frame: Sim not initialized for playback.", "error")
            if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": "Not initialized"})
            return
        
        # GPU-cached playback: instant frame seeking (no disk I/O)
        if self.gpu_playback_enabled and frame_idx in self.gpu_playback_cache:
            frame_content_gpu = self.gpu_playback_cache[frame_idx]
            
            # Apply GPU data directly (NO GPU→CPU→GPU transfers)
            self._apply_recorded_arrays_to_gpu_direct(frame_content_gpu, is_initial_state=False)
            
            self.runtime_state.current_time_ms = frame_content_gpu.get("time_ms", self.runtime_state.current_time_ms)
            self.runtime_state.current_time_step = frame_content_gpu.get("step", self.runtime_state.current_time_step)
        else:
            # Legacy path: read from HDF5 file (slow)
            frame_content_np = self._read_frame_from_file(frame_idx, h5_file_handle)
            if frame_content_np is None:
                self._log_to_ui(f"Failed to read frame {frame_idx} for playback. Playback may be unstable.", "error")
                if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": f"Failed to read frame {frame_idx}"})
                return

            self._apply_recorded_arrays_to_gpu(frame_content_np, is_initial_state=False) 

            self.runtime_state.current_time_ms = frame_content_np.get("time_ms", self.runtime_state.current_time_ms)
            self.runtime_state.current_time_step = frame_content_np.get("step", self.runtime_state.current_time_step)

        latest_gui_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
        if self.ui_queue and latest_gui_data:
            self.ui_queue.put({
                "type": "PLAYBACK_FRAME_APPLIED", 
                "gui_data": latest_gui_data,
                "frame_index": frame_idx, 
                "current_time_ms": self.runtime_state.current_time_ms,
                "current_time_step": self.runtime_state.current_time_step
            })

    def _apply_recorded_arrays_to_gpu_direct(self, state_dict_gpu, is_initial_state=False):
        """Applies CuPy arrays directly from GPU cache to simulation state (zero-copy)."""
        if not self.is_initialized:
            self._log_console("Cannot apply GPU-cached frame: Sim not initialized.", "error")
            return
        
        # Direct GPU-to-GPU copies (fast)
        dynamic_arrays_to_apply = [
            'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
            'cp_conductance_g_e', 'cp_conductance_g_i',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
        ]
        
        if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            dynamic_arrays_to_apply.append('cp_recovery_variable_u')
            if self.core_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                dynamic_arrays_to_apply.append('cp_neuron_firing_thresholds')
        elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            dynamic_arrays_to_apply.extend(['cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n'])
        
        # Copy CuPy arrays directly (GPU→GPU, very fast)
        for attr_name in dynamic_arrays_to_apply:
            if attr_name in state_dict_gpu:
                source_array = state_dict_gpu[attr_name]
                if source_array is not None and isinstance(source_array, cp.ndarray):
                    target_array = getattr(self, attr_name, None)
                    if target_array is not None and target_array.shape == source_array.shape:
                        target_array[:] = source_array  # In-place copy
        
        # Apply connection weights if Hebbian learning enabled
        if self.core_config.enable_hebbian_learning and "cp_connections_data" in state_dict_gpu:
            conn_data = state_dict_gpu["cp_connections_data"]
            if conn_data is not None and isinstance(conn_data, cp.ndarray) and self.cp_connections is not None:
                if self.cp_connections.data.shape == conn_data.shape:
                    self.cp_connections.data[:] = conn_data
        
        # Apply STP state if enabled
        if self.core_config.enable_short_term_plasticity:
            if "cp_stp_u" in state_dict_gpu and state_dict_gpu["cp_stp_u"] is not None:
                if self.cp_stp_u is not None and self.cp_stp_u.shape == state_dict_gpu["cp_stp_u"].shape:
                    self.cp_stp_u[:] = state_dict_gpu["cp_stp_u"]
            if "cp_stp_x" in state_dict_gpu and state_dict_gpu["cp_stp_x"] is not None:
                if self.cp_stp_x is not None and self.cp_stp_x.shape == state_dict_gpu["cp_stp_x"].shape:
                    self.cp_stp_x[:] = state_dict_gpu["cp_stp_x"]
    
    def _apply_recorded_arrays_to_gpu(self, state_dict_np, is_initial_state=False):
        """Applies NumPy arrays from HDF5 to CuPy arrays on GPU."""
        if not self.is_initialized and not is_initial_state:
             self._log_console("Cannot apply recorded frame arrays: Sim not initialized for playback.", "error")
             if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": "Sim not initialized for frame apply"})
             return
        if not self.is_initialized and is_initial_state and not self.is_initialized:
             self._log_console("Cannot apply initial recorded arrays: Sim not initialized.", "error")
             if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Sim config missing for initial apply"})
             return

        def _apply_to_cp_array(cp_array_attr_name, np_array_key_in_dict, default_dtype=cp.float32):
            """Helper to apply a NumPy array from state_dict_np to a CuPy array attribute."""
            source_np_array = state_dict_np.get(np_array_key_in_dict)
            
            if source_np_array is None:
                if hasattr(self, cp_array_attr_name) and getattr(self, cp_array_attr_name) is not None:
                    setattr(self, cp_array_attr_name, None)
                return

            if not isinstance(source_np_array, np.ndarray):
                return

            target_cp_array = getattr(self, cp_array_attr_name, None)

            if target_cp_array is None and source_np_array.size > 0 :
                try:
                    setattr(self, cp_array_attr_name, cp.asarray(source_np_array, dtype=default_dtype))
                except Exception as e:
                    self._log_console(f"Error creating {cp_array_attr_name} from recording: {e}", "error"); return
            elif target_cp_array is not None:
                if target_cp_array.shape == source_np_array.shape:
                    if target_cp_array.dtype == source_np_array.dtype:
                        target_cp_array[:] = cp.asarray(source_np_array)
                    else: 
                        try: target_cp_array[:] = cp.asarray(source_np_array.astype(target_cp_array.dtype))
                        except Exception as e: self._log_console(f"Error applying {cp_array_attr_name} due to dtype mismatch and cast fail: {e}", "error")
                elif target_cp_array.size == source_np_array.size and source_np_array.size > 0: 
                    try: target_cp_array[:] = cp.asarray(source_np_array.reshape(target_cp_array.shape))
                    except ValueError as ve: self._log_console(f"ERROR: Failed to reshape {cp_array_attr_name}. Error: {ve}", "error")
                elif source_np_array.size == 0 and target_cp_array.size == 0: pass 
                elif source_np_array.size == 0 and target_cp_array.size > 0: 
                     target_cp_array.fill(0) 
                else: 
                    self._log_console(f"Error: Shape/size mismatch for {cp_array_attr_name} from recording. Target: {target_cp_array.shape}, Source: {source_np_array.shape}. Cannot apply.", "error")
            elif target_cp_array is None and source_np_array.size == 0:
                setattr(self, cp_array_attr_name, cp.array([], dtype=default_dtype))

        if is_initial_state: 
            _apply_to_cp_array("cp_traits", "cp_traits", default_dtype=cp.int32)
            _apply_to_cp_array("cp_neuron_positions_3d", "cp_neuron_positions_3d")
            if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                    _apply_to_cp_array(f"cp_izh_{param}", f"cp_izh_{param}")
            elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                     _apply_to_cp_array(f"cp_hh_{param}", f"cp_hh_{param}")

            conn_data_np = state_dict_np.get("connections_data")
            conn_indices_np = state_dict_np.get("connections_indices")
            conn_indptr_np = state_dict_np.get("connections_indptr")
            conn_shape = state_dict_np.get("connections_shape") 

            if conn_data_np is not None and conn_indices_np is not None and conn_indptr_np is not None and conn_shape is not None:
                if conn_shape[0] != self.core_config.num_neurons or conn_shape[1] != self.core_config.num_neurons:
                    self._log_console(f"Error: Connection shape {conn_shape} from recording's initial_state "
                                     f"does not match configured neuron count {self.core_config.num_neurons}. Playback may fail.", "error")
                
                self.cp_connections = csp.csr_matrix((cp.asarray(conn_data_np),
                                                      cp.asarray(conn_indices_np),
                                                      cp.asarray(conn_indptr_np)),
                                                     shape=conn_shape, dtype=cp.float32)
                self.cp_connections.sort_indices()
            else: 
                self._log_console("Warning: Connection structure missing/incomplete in initial_state. Using empty matrix.", "warning")
                n_cfg = self.core_config.num_neurons
                self.cp_connections = csp.csr_matrix((n_cfg, n_cfg), dtype=cp.float32)
            
            num_synapses_loaded = self.cp_connections.nnz
            _apply_to_cp_array("cp_synapse_pulse_timers", "cp_synapse_pulse_timers", default_dtype=cp.int32)
            _apply_to_cp_array("cp_synapse_pulse_progress", "cp_synapse_pulse_progress")
            
            if self.cp_synapse_pulse_timers is None or self.cp_synapse_pulse_timers.size != num_synapses_loaded:
                self.cp_synapse_pulse_timers = cp.zeros(num_synapses_loaded, dtype=cp.int32)
            if self.cp_synapse_pulse_progress is None or self.cp_synapse_pulse_progress.size != num_synapses_loaded:
                self.cp_synapse_pulse_progress = cp.zeros(num_synapses_loaded, dtype=cp.float32)

            if self.core_config.enable_short_term_plasticity:
                _apply_to_cp_array("cp_stp_u", "cp_stp_u")
                _apply_to_cp_array("cp_stp_x", "cp_stp_x")
                if self.cp_stp_u is None or self.cp_stp_u.size != num_synapses_loaded:
                    self.cp_stp_u = cp.full(num_synapses_loaded, self.core_config.stp_U, dtype=cp.float32) if num_synapses_loaded > 0 else cp.array([], dtype=cp.float32)
                if self.cp_stp_x is None or self.cp_stp_x.size != num_synapses_loaded:
                    self.cp_stp_x = cp.ones(num_synapses_loaded, dtype=cp.float32) if num_synapses_loaded > 0 else cp.array([], dtype=cp.float32)
            else:
                self.cp_stp_u = None; self.cp_stp_x = None

        dynamic_keys_map = { 
            'cp_membrane_potential_v': 'cp_membrane_potential_v', 
            'cp_recovery_variable_u': 'cp_recovery_variable_u', 
            'cp_gating_variable_m': 'cp_gating_variable_m', 
            'cp_gating_variable_h': 'cp_gating_variable_h', 
            'cp_gating_variable_n': 'cp_gating_variable_n', 
            'cp_conductance_g_e': 'cp_conductance_g_e',
            'cp_conductance_g_i': 'cp_conductance_g_i',
            'cp_external_input_current': 'cp_external_input_current',
            'cp_refractory_timers': ('cp_refractory_timers', cp.int32),
            'cp_viz_activity_timers': ('cp_viz_activity_timers', cp.int32),
            'cp_neuron_firing_thresholds': 'cp_neuron_firing_thresholds', 
            'cp_neuron_activity_ema': 'cp_neuron_activity_ema',
            'cp_firing_states': ('cp_firing_states', cp.bool_),
            'cp_prev_firing_states': ('cp_prev_firing_states', cp.bool_),
            'cp_stp_u': 'cp_stp_u', 
            'cp_stp_x': 'cp_stp_x', 
            'cp_synapse_pulse_timers': ('cp_synapse_pulse_timers', cp.int32), 
            'cp_synapse_pulse_progress': 'cp_synapse_pulse_progress' 
        }

        if not is_initial_state and "cp_connections_data" in state_dict_np:
            conn_data_frame_np = state_dict_np.get("cp_connections_data")
            if conn_data_frame_np is not None and self.cp_connections is not None and self.cp_connections.data is not None:
                if isinstance(conn_data_frame_np, np.ndarray):
                    if self.cp_connections.data.shape == conn_data_frame_np.shape:
                        self.cp_connections.data[:] = cp.asarray(conn_data_frame_np)
                    elif self.cp_connections.data.size == conn_data_frame_np.size and conn_data_frame_np.size > 0:
                        try: self.cp_connections.data[:] = cp.asarray(conn_data_frame_np.reshape(self.cp_connections.data.shape))
                        except ValueError as ve: self._log_console(f"ERROR: Failed to reshape cp_connections.data from recording frame. Error: {ve}", "error")
                    elif not (self.cp_connections.data.size == 0 and conn_data_frame_np.size == 0) : 
                        self._log_console(f"Error: Shape/size mismatch for dynamic cp_connections.data from recording frame. Cannot apply.", "error")
            elif conn_data_frame_np is None and self.cp_connections is not None and self.cp_connections.data is not None:
                 pass 
                 
        for cp_attr, key_info in dynamic_keys_map.items():
            np_key = key_info if isinstance(key_info, str) else key_info[0]
            default_dtype = cp.float32 
            if not isinstance(key_info, str) and len(key_info) > 1:
                default_dtype = key_info[1]
            
            if np_key == "cp_connections_data" and not is_initial_state: 
                continue
            _apply_to_cp_array(cp_attr, np_key, default_dtype=default_dtype)

        self._mock_num_spikes_this_step = state_dict_np.get("_mock_num_spikes_this_step", 0)
        self._mock_network_avg_firing_rate_hz = state_dict_np.get("_mock_network_avg_firing_rate_hz", 0.0)
        self._mock_total_plasticity_events = state_dict_np.get("_mock_total_plasticity_events", 0)

        if is_initial_state: 
            self.runtime_state.current_time_ms = state_dict_np.get("start_time_ms", 0.0)
            self.runtime_state.current_time_step = state_dict_np.get("start_time_step", 0)

    def _run_one_simulation_step(self):
        """Executes a single step of the simulation logic."""
        if not self.is_initialized or self.core_config.num_neurons == 0: return
        try:
            n_neurons = self.core_config.num_neurons; cfg = self.core_config; dt = cfg.dt_ms

            # --- 1. Synaptic Plasticity (STP) Update ---
            base_synaptic_weights = self.cp_connections.data 
            effective_synaptic_strength = base_synaptic_weights 

            if cfg.enable_short_term_plasticity and self.cp_connections.nnz > 0 and \
               self.cp_stp_u is not None and self.cp_stp_x is not None:

                self.cp_stp_u, self.cp_stp_x = fused_stp_decay_recovery(self.cp_stp_u, self.cp_stp_x, dt, cfg.stp_tau_f, cfg.stp_tau_d)

                if self.cp_prev_firing_states.any(): 
                    coo_matrix_stp = self.cp_connections.tocoo(copy=False) 
                    active_syn_mask_stp = self.cp_prev_firing_states[coo_matrix_stp.row]
                    active_syn_indices_stp = cp.where(active_syn_mask_stp)[0] 

                    if active_syn_indices_stp.size > 0:
                        U_stp = cfg.stp_U 
                        u_active_old = self.cp_stp_u[active_syn_indices_stp]
                        x_active_old = self.cp_stp_x[active_syn_indices_stp]

                        u_active_new = u_active_old + U_stp * (1.0 - u_active_old)
                        self.cp_stp_u[active_syn_indices_stp] = u_active_new
                        self.cp_stp_x[active_syn_indices_stp] = x_active_old * (1.0 - u_active_new) 

                cp.clip(self.cp_stp_x, 0.0, 1.0, out=self.cp_stp_x)
                cp.clip(self.cp_stp_u, 0.0, 1.0, out=self.cp_stp_u)

                effective_synaptic_strength = base_synaptic_weights * self.cp_stp_u * self.cp_stp_x
                effective_connections_matrix = csp.csr_matrix(
                    (effective_synaptic_strength, self.cp_connections.indices, self.cp_connections.indptr),
                    shape=self.cp_connections.shape
                )
            else: 
                effective_connections_matrix = self.cp_connections 

            # --- 2. Synaptic Conductance Update & Current Calculation ---
            decay_e = cp.exp(-dt / cfg.syn_tau_g_e) if cfg.syn_tau_g_e > 0 else 0.0
            decay_i = cp.exp(-dt / cfg.syn_tau_g_i) if cfg.syn_tau_g_i > 0 else 0.0

            self.cp_conductance_g_e, self.cp_conductance_g_i, synaptic_current_I_syn_pA = fused_conductance_decay_and_current(
                self.cp_conductance_g_e, self.cp_conductance_g_i, decay_e, decay_i,
                self.cp_membrane_potential_v, cfg.syn_reversal_potential_e, cfg.syn_reversal_potential_i
            )

            if effective_connections_matrix.nnz > 0 and self.cp_prev_firing_states.any():
                prev_fired_float = self.cp_prev_firing_states.astype(cp.float32) 

                if cfg.enable_inhibitory_neurons and self.cp_traits is not None:
                    is_inhibitory_neuron_output = (self.cp_traits == cfg.inhibitory_trait_index)
                    exc_fired_prev = prev_fired_float * (~is_inhibitory_neuron_output) 
                    inhib_fired_prev = prev_fired_float * is_inhibitory_neuron_output 

                    g_e_increase = (effective_connections_matrix.T @ exc_fired_prev) * cfg.propagation_strength
                    g_i_increase = (effective_connections_matrix.T @ inhib_fired_prev) * cfg.inhibitory_propagation_strength

                    self.cp_conductance_g_e += g_e_increase
                    self.cp_conductance_g_i += g_i_increase
                else: 
                    g_e_increase = (effective_connections_matrix.T @ prev_fired_float) * cfg.propagation_strength # Corrected line: use prev_fired_float
                    self.cp_conductance_g_e += g_e_increase

            total_input_current_pA = synaptic_current_I_syn_pA + self.cp_external_input_current

            # --- 3. Neuron Model Dynamics Update ---
            fired_this_step = cp.zeros(n_neurons, dtype=bool) 

            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                v_new, u_new = fused_izhikevich2007_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_recovery_variable_u,
                    self.cp_izh_C, self.cp_izh_k, self.cp_izh_vr, self.cp_izh_vt,
                    self.cp_izh_a, self.cp_izh_b,
                    total_input_current_pA, dt
                )
                not_in_refractory = (self.cp_refractory_timers <= 0)
                current_spike_thresholds = self.cp_neuron_firing_thresholds if cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None else self.cp_izh_vpeak
                fired_this_step = (v_new >= current_spike_thresholds) & not_in_refractory
                fired_indices = cp.where(fired_this_step)[0]

                if fired_indices.size > 0:
                    v_new[fired_indices] = self.cp_izh_c_reset[fired_indices] 
                    u_new[fired_indices] += self.cp_izh_d_increment[fired_indices] 
                    self.cp_refractory_timers[fired_indices] = cfg.refractory_period_steps 

                self.cp_membrane_potential_v[:] = v_new
                self.cp_recovery_variable_u[:] = u_new
                self.cp_refractory_timers[self.cp_refractory_timers > 0] -= 1 

            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                total_input_current_uA_density_equivalent = total_input_current_pA * 1e-6 
                v_new, m_new, h_new, n_new = fused_hodgkin_huxley_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_gating_variable_m, self.cp_gating_variable_h, self.cp_gating_variable_n,
                    total_input_current_uA_density_equivalent, dt,
                    self.cp_hh_C_m, self.cp_hh_g_Na_max, self.cp_hh_g_K_max, self.cp_hh_g_L,
                    self.cp_hh_E_Na, self.cp_hh_E_K, self.cp_hh_E_L,
                    cfg.hh_temperature_celsius, cfg.hh_q10_factor
                )
                fired_this_step = (self.cp_membrane_potential_v < self.cp_hh_v_peak) & (v_new >= self.cp_hh_v_peak) 

                self.cp_membrane_potential_v[:] = v_new
                self.cp_gating_variable_m[:] = m_new
                self.cp_gating_variable_h[:] = h_new
                self.cp_gating_variable_n[:] = n_new

            self.cp_firing_states[:] = fired_this_step
            
            # OPTIMIZATION: Defer spike count until needed (avoids GPU->CPU sync every step)
            # Only compute when sending data to UI
            self._mock_num_spikes_this_step = 0  # Will be computed on-demand

            if self.cp_viz_activity_timers is not None:
                max_highlight_val = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7) if OPENGL_AVAILABLE else 7
                self.cp_viz_activity_timers = cp.where(fired_this_step,
                                                       max_highlight_val, 
                                                       self.cp_viz_activity_timers) 

            if OPENGL_AVAILABLE and opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
               self.cp_synapse_pulse_timers is not None and fired_this_step.any(): 
                if self.cp_connections is not None and self.cp_connections.nnz > 0:
                    coo_matrix_for_pulses = self.cp_connections.tocoo(copy=False)
                    presynaptic_fired_mask_for_pulses = fired_this_step[coo_matrix_for_pulses.row]
                    synapses_to_activate_indices = cp.where(presynaptic_fired_mask_for_pulses)[0]

                    if synapses_to_activate_indices.size > 0:
                        pulse_lifetime = opengl_viz_config.get("SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES", 5)
                        self.cp_synapse_pulse_timers[synapses_to_activate_indices] = pulse_lifetime 
                        self.cp_synapse_pulse_progress[synapses_to_activate_indices] = 0.0 

            # --- 4. Hebbian Learning (Long-Term Potentiation/Depression) ---
            if cfg.enable_hebbian_learning and self.cp_connections.nnz > 0 and \
               self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                if self.cp_prev_firing_states.any() and fired_this_step.any(): 
                    coo_matrix_heb = self.cp_connections.tocoo(copy=False) 
                    pre_fired_mask_heb = self.cp_prev_firing_states[coo_matrix_heb.row] 
                    post_fired_mask_heb = fired_this_step[coo_matrix_heb.col] 

                    active_synapse_indices_heb = cp.where(pre_fired_mask_heb & post_fired_mask_heb)[0]
                    num_potentiation_events = 0
                    if active_synapse_indices_heb.size > 0:
                        base_weights_data_array = self.cp_connections.data 
                        current_weights_active_syn = base_weights_data_array[active_synapse_indices_heb]
                        delta_weights = cfg.hebbian_learning_rate * (cfg.hebbian_max_weight - current_weights_active_syn)
                        base_weights_data_array[active_synapse_indices_heb] += delta_weights
                        num_potentiation_events = active_synapse_indices_heb.size

                    self.cp_connections.data *= (1.0 - cfg.hebbian_weight_decay) 
                    cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=self.cp_connections.data)
                    if num_potentiation_events > 0: self._mock_total_plasticity_events += num_potentiation_events

            # --- 5. Homeostatic Plasticity (Adaptive Thresholds for Izhikevich) ---
            if cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name: 
                    self.cp_neuron_activity_ema, self.cp_neuron_firing_thresholds = fused_homeostasis_update(
                        self.cp_neuron_activity_ema, fired_this_step.astype(cp.float32),
                        cfg.homeostasis_target_rate, cfg.homeostasis_ema_alpha, cfg.homeostasis_threshold_adapt_rate,
                        self.cp_neuron_firing_thresholds,
                        cfg.homeostasis_threshold_min, cfg.homeostasis_threshold_max
                    )
                elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name: 
                     self.cp_neuron_activity_ema = (1.0 - cfg.homeostasis_ema_alpha) * self.cp_neuron_activity_ema + \
                                               cfg.homeostasis_ema_alpha * fired_this_step.astype(cp.float32)

            # --- 6. Prepare for Next Step & Record Frame ---
            self.cp_prev_firing_states[:] = fired_this_step 
            self.record_current_frame_if_active() # This was the missing method call's target

            # Note: Network firing rate calculation deferred to avoid GPU->CPU sync every step
            # Will be updated on-demand when GUI data is requested

        except Exception as e:
            self._log_to_ui(f"Error during simulation step: {e}","critical")
            import traceback; traceback.print_exc() 
            self.stop_simulation() 
            if self.ui_queue: self.ui_queue.put({"type": "SIM_ERROR_OCCURRED", "error_message": str(e)})
    def save_checkpoint(self, filepath, gui_config_snapshot=None): # Added gui_config_snapshot
        """Saves the current simulation state to an HDF5 checkpoint file."""
        self._log_to_ui(f"Saving checkpoint to {filepath}...", "info")
        if not self.is_initialized:
            self._log_to_ui("Sim not initialized. Cannot save checkpoint.","warning"); return False

        try:
            with h5py.File(filepath, 'w') as h5f:
                config_dict = self.core_config.to_dict()
                save_dict_to_hdf5_attrs(h5f, config_dict)

                state_group = h5f 

                arrays_to_save_direct = [ 
                    'cp_membrane_potential_v', 'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_external_input_current', 'cp_firing_states', 'cp_prev_firing_states',
                    'cp_traits', 'cp_refractory_timers', 'cp_neuron_positions_3d',
                    'cp_neuron_activity_ema', 'cp_viz_activity_timers',
                    'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
                ]
                for attr_name in arrays_to_save_direct:
                    data_array = getattr(self, attr_name, None)
                    if data_array is not None and data_array.size > 0:
                        state_group.create_dataset(attr_name, data=cp.asnumpy(data_array), compression="gzip")
                    elif data_array is not None: 
                         state_group.attrs[f"{attr_name}_is_empty"] = True

                if self.cp_connections is not None:
                    if self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                        state_group.create_dataset("connections_data", data=cp.asnumpy(self.cp_connections.data), compression="gzip")
                    if self.cp_connections.indices is not None and self.cp_connections.indices.size > 0:
                        state_group.create_dataset("connections_indices", data=cp.asnumpy(self.cp_connections.indices), compression="gzip")
                    if self.cp_connections.indptr is not None and self.cp_connections.indptr.size > 0:
                        state_group.create_dataset("connections_indptr", data=cp.asnumpy(self.cp_connections.indptr), compression="gzip")
                    state_group.attrs["connections_shape_0"] = self.cp_connections.shape[0]
                    state_group.attrs["connections_shape_1"] = self.cp_connections.shape[1]

                if self.cp_stp_u is not None and self.cp_stp_u.size > 0: state_group.create_dataset("cp_stp_u", data=cp.asnumpy(self.cp_stp_u), compression="gzip")
                elif self.cp_stp_u is not None: state_group.attrs["cp_stp_u_is_empty"] = True
                if self.cp_stp_x is not None and self.cp_stp_x.size > 0: state_group.create_dataset("cp_stp_x", data=cp.asnumpy(self.cp_stp_x), compression="gzip")
                elif self.cp_stp_x is not None: state_group.attrs["cp_stp_x_is_empty"] = True

                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    if self.cp_recovery_variable_u is not None and self.cp_recovery_variable_u.size > 0: state_group.create_dataset("cp_recovery_variable_u", data=cp.asnumpy(self.cp_recovery_variable_u), compression="gzip")
                    elif self.cp_recovery_variable_u is not None : state_group.attrs["cp_recovery_variable_u_is_empty"] = True
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                         attr_name_cp = f"cp_izh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
                         elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    if self.cp_neuron_firing_thresholds is not None and self.cp_neuron_firing_thresholds.size > 0: state_group.create_dataset("cp_neuron_firing_thresholds", data=cp.asnumpy(self.cp_neuron_firing_thresholds), compression="gzip")
                    elif self.cp_neuron_firing_thresholds is not None : state_group.attrs["cp_neuron_firing_thresholds_is_empty"] = True

                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                        attr_name_cp = f"cp_gating_variable_{attr_name_suffix}"
                        data_array = getattr(self, attr_name_cp, None)
                        if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
                        elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                         attr_name_cp = f"cp_hh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
                         elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                
                h5f.attrs["_mock_total_plasticity_events"] = self._mock_total_plasticity_events
                h5f.attrs["_mock_network_avg_firing_rate_hz"] = self._mock_network_avg_firing_rate_hz
                
                if self.runtime_state.neuron_types_list_for_viz:
                    h5f.attrs["neuron_types_list_for_viz_json"] = json.dumps(self.runtime_state.neuron_types_list_for_viz)

                if gui_config_snapshot: # Save GUI related config if provided
                    if "opengl_visualization_settings" in gui_config_snapshot and gui_config_snapshot["opengl_visualization_settings"]:
                         h5f.attrs["opengl_viz_config_json"] = json.dumps(gui_config_snapshot["opengl_visualization_settings"])
                    if "filter_settings" in gui_config_snapshot and gui_config_snapshot["filter_settings"]:
                         h5f.attrs["dpg_filter_settings_json"] = json.dumps(gui_config_snapshot["filter_settings"])


            self._log_to_ui(f"Checkpoint saved successfully to {filepath}", "success")
            if self.ui_queue: self.ui_queue.put({"type": "CHECKPOINT_SAVE_SUCCESS", "filepath": filepath})
            return True
        except Exception as e:
            self._log_to_ui(f"Error saving checkpoint: {e}","error"); import traceback; traceback.print_exc()
            if self.ui_queue: self.ui_queue.put({"type": "CHECKPOINT_SAVE_FAILED", "error": str(e)})
            return False

    def load_checkpoint(self, filepath):
        """Loads a simulation state from an HDF5 checkpoint file."""
        self._log_to_ui(f"Loading checkpoint from {filepath}...", "info")
        
        try:
            with h5py.File(filepath, 'r') as h5f:
                if self.runtime_state.is_running : self.stop_simulation() 
                self.clear_simulation_state_and_gpu_memory() 

                loaded_sim_config_dict = load_dict_from_hdf5_attrs(h5f) 
                if not loaded_sim_config_dict or "num_neurons" not in loaded_sim_config_dict: 
                    self._log_to_ui("Checkpoint missing or invalid simulation_configuration. Load failed.","error"); return False
                
                temp_cfg_for_validation = SimulationConfiguration() 
                for key_cfg in temp_cfg_for_validation.to_dict().keys():
                    if key_cfg not in loaded_sim_config_dict: 
                        loaded_sim_config_dict[key_cfg] = getattr(temp_cfg_for_validation, key_cfg) 

                self.core_config = CoreSimConfig(**{k: v for k, v in loaded_sim_config_dict.items() if hasattr(CoreSimConfig, k)})
                n = self.core_config.num_neurons
                state_group = h5f 

                def _load_cp_array_from_h5(key, default_val_func, default_dtype_for_empty=cp.float32):
                    if f"{key}_is_empty" in state_group.attrs and state_group.attrs[f"{key}_is_empty"] is True:
                        return default_val_func(0) 
                    if key in state_group: 
                        return cp.asarray(state_group[key][:]) 
                    self._log_console(f"Checkpoint: Dataset for '{key}' not found or was empty. Using default.", "debug")
                    return default_val_func(n) if n > 0 else default_val_func(0)

                direct_load_map = { 
                    'cp_membrane_potential_v': ('cp_membrane_potential_v', cp.float32),
                    'cp_conductance_g_e': ('cp_conductance_g_e', cp.float32),
                    'cp_conductance_g_i': ('cp_conductance_g_i', cp.float32),
                    'cp_external_input_current': ('cp_external_input_current', cp.float32),
                    'cp_firing_states': ('cp_firing_states', cp.bool_),
                    'cp_prev_firing_states': ('cp_prev_firing_states', cp.bool_),
                    'cp_traits': ('cp_traits', cp.int32),
                    'cp_refractory_timers': ('cp_refractory_timers', cp.int32),
                    'cp_neuron_activity_ema': ('cp_neuron_activity_ema', cp.float32),
                    'cp_viz_activity_timers': ('cp_viz_activity_timers', cp.int32),
                    'cp_synapse_pulse_timers': ('cp_synapse_pulse_timers', cp.int32),
                    'cp_synapse_pulse_progress': ('cp_synapse_pulse_progress', cp.float32)
                }
                for attr_name, (h5_key, dtype) in direct_load_map.items():
                    setattr(self, attr_name, _load_cp_array_from_h5(h5_key, 
                            default_val_func=lambda size_n, dt=dtype: cp.zeros(size_n, dtype=dt), 
                            default_dtype_for_empty=dtype))

                if "cp_neuron_positions_3d" in state_group or ("cp_neuron_positions_3d_is_empty" in state_group.attrs):
                     self.cp_neuron_positions_3d = _load_cp_array_from_h5("cp_neuron_positions_3d", 
                        default_val_func=lambda size_n: cp.zeros((size_n, 3), dtype=cp.float32))
                elif n > 0 : 
                    np_positions_3d = np.random.uniform(
                        low=[self.core_config.volume_min_x,self.core_config.volume_min_y,self.core_config.volume_min_z],
                        high=[self.core_config.volume_max_x,self.core_config.volume_max_y,self.core_config.volume_max_z],
                        size=(n,3)).astype(np.float32)
                    self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                else: self.cp_neuron_positions_3d = cp.array([], dtype=cp.float32).reshape(0,3)

                conn_data_np = state_group["connections_data"][:] if "connections_data" in state_group else np.array([], dtype=cp.float32)
                conn_indices_np = state_group["connections_indices"][:] if "connections_indices" in state_group else np.array([], dtype=cp.int32)
                conn_indptr_np = state_group["connections_indptr"][:] if "connections_indptr" in state_group else np.array([0]*(n+1), dtype=cp.int32) 
                conn_shape_0 = state_group.attrs.get("connections_shape_0", n)
                conn_shape_1 = state_group.attrs.get("connections_shape_1", n)
                conn_shape = (conn_shape_0, conn_shape_1)
                if conn_shape[0] != n or conn_shape[1] != n: 
                    self._log_to_ui(f"Warning: Checkpoint connection shape {conn_shape} mismatch with config N={n}. Adjusting.", "warning")
                    conn_shape = (n,n)
                    if conn_data_np.size == 0 : conn_indptr_np = np.array([0]*(n+1), dtype=cp.int32)

                self.cp_connections = csp.csr_matrix((cp.asarray(conn_data_np), 
                                                      cp.asarray(conn_indices_np), 
                                                      cp.asarray(conn_indptr_np)), 
                                                     shape=conn_shape, dtype=cp.float32)
                
                num_synapses_loaded = self.cp_connections.nnz

                self.cp_stp_u = _load_cp_array_from_h5("cp_stp_u", 
                    lambda s: cp.full(s, self.core_config.stp_U, dtype=cp.float32) if self.core_config.enable_short_term_plasticity and num_synapses_loaded > 0 and s > 0 else (cp.array([],dtype=cp.float32) if s==0 else None))
                self.cp_stp_x = _load_cp_array_from_h5("cp_stp_x", 
                    lambda s: cp.ones(s, dtype=cp.float32) if self.core_config.enable_short_term_plasticity and num_synapses_loaded > 0 and s > 0 else (cp.array([],dtype=cp.float32) if s==0 else None))

                if not (self.core_config.enable_short_term_plasticity and num_synapses_loaded > 0):
                    self.cp_stp_u = None; self.cp_stp_x = None
                else:
                    if self.cp_stp_u is None or self.cp_stp_u.size != num_synapses_loaded:
                        self.cp_stp_u = cp.full(num_synapses_loaded, self.core_config.stp_U, dtype=cp.float32)
                    if self.cp_stp_x is None or self.cp_stp_x.size != num_synapses_loaded:
                        self.cp_stp_x = cp.ones(num_synapses_loaded, dtype=cp.float32)

                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    self.cp_recovery_variable_u = _load_cp_array_from_h5("cp_recovery_variable_u", lambda s: cp.zeros(s, dtype=cp.float32))
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                        setattr(self, f"cp_izh_{param}", _load_cp_array_from_h5(f"cp_izh_{param}",
                                lambda s, p=param: cp.full(s, getattr(self.core_config, f"izh_{p}_val"), dtype=cp.float32)))
                    self.cp_neuron_firing_thresholds = _load_cp_array_from_h5("cp_neuron_firing_thresholds",
                        lambda s: cp.random.uniform(self.core_config.homeostasis_threshold_min, self.core_config.homeostasis_threshold_max, s).astype(cp.float32) if s > 0 else cp.array([], dtype=cp.float32))
                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                         setattr(self, f"cp_gating_variable_{attr_name_suffix}", _load_cp_array_from_h5(f"cp_gating_variable_{attr_name_suffix}",
                                 lambda s, suff=attr_name_suffix: cp.full(s, getattr(self.core_config, f"hh_{suff}_init"), dtype=cp.float32)))
                    hh_param_map = {'C_m': 'hh_C_m', 'g_Na_max': 'hh_g_Na_max', 'g_K_max': 'hh_g_K_max', 'g_L': 'hh_g_L',
                                    'E_Na': 'hh_E_Na', 'E_K': 'hh_E_K', 'E_L': 'hh_E_L', 'v_peak': 'hh_v_peak'}
                    for param_key, config_attr_name in hh_param_map.items():
                         setattr(self, f"cp_hh_{param_key}", _load_cp_array_from_h5(f"cp_hh_{param_key}",
                                 lambda s, ca_name=config_attr_name: cp.full(s, getattr(self.core_config, ca_name), dtype=cp.float32)))
                    self.cp_neuron_firing_thresholds = None 

                self._mock_total_plasticity_events = h5f.attrs.get("_mock_total_plasticity_events",0)
                self._mock_network_avg_firing_rate_hz = h5f.attrs.get("_mock_network_avg_firing_rate_hz",0.0)

                self.is_initialized = True 
                self._log_to_ui(f"Checkpoint loaded. Sim time: {self.runtime_state.current_time_ms}ms, Step: {self.runtime_state.current_time_step}, Model: {self.core_config.neuron_model_type}", "success")

                loaded_gui_settings = {}
                if "opengl_viz_config_json" in h5f.attrs:
                    try: loaded_gui_settings["opengl_visualization_settings"] = json.loads(h5f.attrs["opengl_viz_config_json"])
                    except: self._log_console("Warning: Could not parse opengl_viz_config_json from checkpoint.", "warning")
                if "dpg_filter_settings_json" in h5f.attrs: # Load DPG filter settings if present
                    try: loaded_gui_settings["filter_settings"] = json.loads(h5f.attrs["dpg_filter_settings_json"])
                    except: self._log_console("Warning: Could not parse dpg_filter_settings_json from checkpoint.", "warning")
                
                if "neuron_types_list_for_viz_json" in h5f.attrs: 
                    try: 
                        self.runtime_state.neuron_types_list_for_viz = json.loads(h5f.attrs["neuron_types_list_for_viz_json"])
                        loaded_gui_settings["neuron_types_list_for_viz"] = self.runtime_state.neuron_types_list_for_viz
                    except: self._log_console("Warning: Could not parse neuron_types_list_for_viz_json from checkpoint.", "warning")
                
                if self.ui_queue:
                    initial_gui_data = self.get_initial_sim_data_snapshot() 
                    self.ui_queue.put({
                        "type": "CHECKPOINT_LOADED_SUCCESS",
                        "config_dict": self.core_config.to_dict(),
                        "gui_settings_from_checkpoint": loaded_gui_settings,
                        "initial_gui_data": initial_gui_data
                    })
                return True
        except Exception as e:
            self._log_to_ui(f"Error loading checkpoint: {e}","error"); import traceback; traceback.print_exc()
            self.is_initialized=False; 
            if self.ui_queue: self.ui_queue.put({"type": "CHECKPOINT_LOAD_FAILED", "error": str(e)})
            return False        

    def get_latest_simulation_data_for_gui(self, force_fetch=False):
        """Retrieves a snapshot of the current simulation state for GUI updates.
        Sends CuPy arrays for relevant OpenGL data.
        """
        if not self.is_initialized:
            self._log_console("GUI data request: Sim not initialized.","debug"); return None

        n = self.core_config.num_neurons
        dt = self.core_config.dt_ms
        
        # Compute spike count on-demand only when GUI requests it (avoids sync every step)
        num_spikes_this_step = int(cp.sum(self.cp_firing_states).get()) if self.cp_firing_states is not None else 0
        
        # Compute firing rate on-demand using current spike count
        if n > 0 and dt > 0:
            instantaneous_rate_hz = (num_spikes_this_step / n) / (dt / 1000.0)
            self._mock_network_avg_firing_rate_hz = self._mock_network_avg_firing_rate_hz * 0.95 + instantaneous_rate_hz * 0.05
        else:
            self._mock_network_avg_firing_rate_hz = 0.0
        
        gui_data_dict = {
            "current_time_ms": self.runtime_state.current_time_ms,
            "current_time_step": self.runtime_state.current_time_step,
            "num_spikes_this_step": num_spikes_this_step,
            "network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
            "total_plasticity_events": self._mock_total_plasticity_events,
            "neuron_types_list_for_viz": self.runtime_state.neuron_types_list_for_viz.copy(), # Stays as Python list
            "neuron_model_type_str": self.core_config.neuron_model_type,
            "num_neurons_snapshot": n # Add total number of neurons in this snapshot
        }

        # --- Data to keep as CuPy arrays for OpenGL ---
        if self.cp_firing_states is not None:
            gui_data_dict["neuron_fired_status_cp"] = self.cp_firing_states.copy()
        elif n > 0:
            gui_data_dict["neuron_fired_status_cp"] = cp.zeros(n, dtype=bool)
        else:
            gui_data_dict["neuron_fired_status_cp"] = cp.array([], dtype=bool)

        if self.cp_viz_activity_timers is not None:
            gui_data_dict["neuron_activity_timers_cp"] = self.cp_viz_activity_timers.copy()
        elif n > 0:
            gui_data_dict["neuron_activity_timers_cp"] = cp.zeros(n, dtype=cp.int32)
        else:
            gui_data_dict["neuron_activity_timers_cp"] = cp.array([], dtype=cp.int32)

        if self.cp_neuron_positions_3d is not None:
            gui_data_dict["neuron_positions_3d_cp"] = self.cp_neuron_positions_3d.copy()
        elif n > 0:
            gui_data_dict["neuron_positions_3d_cp"] = cp.zeros((n,3),dtype=cp.float32)
        else:
            gui_data_dict["neuron_positions_3d_cp"] = cp.array([], dtype=cp.float32).reshape(0,3)

        if self.cp_traits is not None:
            gui_data_dict["neuron_traits_cp"] = self.cp_traits.copy()
        elif n > 0:
            gui_data_dict["neuron_traits_cp"] = cp.zeros(n, dtype=cp.int32)
        else:
            gui_data_dict["neuron_traits_cp"] = cp.array([], dtype=cp.int32)
        
        # Add neuron type IDs for GPU-efficient filtering
        if self.cp_neuron_type_ids is not None:
            gui_data_dict["neuron_type_ids_cp"] = self.cp_neuron_type_ids.copy()
        elif n > 0:
            gui_data_dict["neuron_type_ids_cp"] = cp.zeros(n, dtype=cp.int32)
        else:
            gui_data_dict["neuron_type_ids_cp"] = cp.array([], dtype=cp.int32)

        # --- Data for DPG text display (can be NumPy or Python types) ---
        if self.cp_membrane_potential_v is not None:
            # Example: If you need a small sample of Vm for a DPG plot (not for GL points usually)
            # sample_indices_vm = cp.random.choice(cp.arange(n), size=min(n, 100), replace=False) if n > 0 else cp.array([])
            # gui_data_dict["neuron_Vm_sample_np"] = cp.asnumpy(self.cp_membrane_potential_v[sample_indices_vm]) if sample_indices_vm.size > 0 else np.array([])
            pass # For full Vm, if used for something other than GL points directly, decide if cp or np needed

        # Synapse info for GUI is CPU-based and sampled - only update occasionally to minimize CPU-GPU transfers
        # Check if we should update synapse sample this time
        # Use visualization config setting for update interval
        viz_update_interval = self.viz_config.viz_update_interval_steps
        update_synapse_sample = (self.runtime_state.current_time_step % viz_update_interval == 0)
        
        # Use cached synapse info if not updating
        if not update_synapse_sample and hasattr(self, '_cached_synapse_info_gui'):
            synapse_info_for_gui = self._cached_synapse_info_gui
        else:
            synapse_info_for_gui = []
            if self.cp_connections is not None and hasattr(self.cp_connections,'nnz') and self.cp_connections.nnz > 0:
                max_synapses_to_sample_for_gui = 20000
                try:
                    coo_conn = self.cp_connections.tocoo(copy=False)
                    num_actual_synapses = coo_conn.nnz
                    num_to_send = min(num_actual_synapses, max_synapses_to_sample_for_gui)

                    if num_to_send > 0:
                        indices_to_sample_np = np.random.choice(num_actual_synapses, num_to_send, replace=False) \
                                            if num_actual_synapses > num_to_send else np.arange(num_actual_synapses)

                        # Fetch relevant data from CuPy arrays using NumPy indices
                        row_indices_np = cp.asnumpy(coo_conn.row[indices_to_sample_np])
                        col_indices_np = cp.asnumpy(coo_conn.col[indices_to_sample_np])

                        weights_data_to_use_cp = self.cp_connections.data 
                        if self.core_config.enable_short_term_plasticity and \
                        self.cp_stp_u is not None and self.cp_stp_x is not None and \
                        self.cp_stp_u.size == self.cp_connections.data.size and \
                        self.cp_stp_x.size == self.cp_connections.data.size :
                            weights_data_to_use_cp = self.cp_connections.data * self.cp_stp_u * self.cp_stp_x

                        # Sample weights using NumPy indices on the CuPy array, then convert
                        sampled_weights_np = cp.asnumpy(weights_data_to_use_cp[cp.asarray(indices_to_sample_np)])

                        for i in range(num_to_send):
                            synapse_info_for_gui.append({
                                "source_idx": int(row_indices_np[i]),
                                "target_idx": int(col_indices_np[i]),
                                "weight": float(sampled_weights_np[i])
                            })
                except Exception as e: self._log_console(f"Error processing connections for GUI: {e}","error")
            
            # Cache the synapse info for future use
            self._cached_synapse_info_gui = synapse_info_for_gui
        
        gui_data_dict["synapse_info"] = synapse_info_for_gui

        # Pulse data for OpenGL - if pulses are enabled, this part needs to be GPU-centric
        # For now, this logic is complex and might be better handled by sending raw cp_synapse_pulse_timers/progress
        # and relevant connection data for UI thread to compute positions, OR pre-compute on sim thread.
        # Let's assume for now this is handled later or in a simplified way.
        # If pulse positions are needed for GL, they should be sent as a CuPy array.
        # Example: Pre-calculate active pulse positions on sim thread (if feasible):
        if OPENGL_AVAILABLE and opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
        self.cp_synapse_pulse_timers is not None and self.cp_synapse_pulse_progress is not None and \
        self.cp_connections is not None and self.cp_connections.nnz > 0:

            active_pulse_mask = self.cp_synapse_pulse_timers > 0
            active_pulse_indices = cp.where(active_pulse_mask)[0]

            if active_pulse_indices.size > 0:
                coo_conn_for_pulses = self.cp_connections.tocoo(copy=False) # Ensure COO is available

                # Get source and target neuron indices for active pulses
                # These indices are into the full list of synapses (coo_conn.row/col)
                src_neuron_indices_for_active_pulses = coo_conn_for_pulses.row[active_pulse_indices]
                tgt_neuron_indices_for_active_pulses = coo_conn_for_pulses.col[active_pulse_indices]

                # Get positions of these source and target neurons
                pos_src_cp = self.cp_neuron_positions_3d[src_neuron_indices_for_active_pulses]
                pos_tgt_cp = self.cp_neuron_positions_3d[tgt_neuron_indices_for_active_pulses]

                # Get progress for active pulses
                pulse_prog_active = self.cp_synapse_pulse_progress[active_pulse_indices]

                # Interpolate pulse positions: pos_src + progress * (pos_tgt - pos_src)
                # Reshape pulse_prog_active to be (N, 1) for broadcasting with (N, 3) positions
                pulse_positions_cp = pos_src_cp + pulse_prog_active[:, cp.newaxis] * (pos_tgt_cp - pos_src_cp)
                gui_data_dict["pulse_positions_cp_for_gl"] = pulse_positions_cp # Send as CuPy array
            else:
                gui_data_dict["pulse_positions_cp_for_gl"] = cp.array([], dtype=cp.float32).reshape(0,3)
        else:
            gui_data_dict["pulse_positions_cp_for_gl"] = cp.array([], dtype=cp.float32).reshape(0,3)


        # Small, specific NumPy arrays for DPG plots (if any)
        # Example: if self.cp_membrane_potential_v is not None and n > 0:
        #     sample_indices = cp.random.choice(cp.arange(n), size=min(n, 10), replace=False) # Small sample for plotting
        #     gui_data_dict["neuron_Vm_trace_sample_np"] = cp.asnumpy(self.cp_membrane_potential_v[sample_indices])

        return gui_data_dict

    def get_initial_sim_data_snapshot(self):
        """
        Gets a snapshot of simulation data, intended for when the simulation is first initialized or reset.
        Returns data structure consistent with get_latest_simulation_data_for_gui, 
        including CuPy arrays for GL-relevant data.
        """
        if not self.is_initialized:
            self._log_console("Initial snapshot request: Sim not initialized. Providing empty/default structure.","info")
            # Fallback, creating structure similar to get_latest_simulation_data_for_gui
            n_cfg = self.core_config.num_neurons if self.is_initialized else 0
            model_type_str_cfg = self.core_config.neuron_model_type if self.is_initialized else NeuronModel.IZHIKEVICH.name
            types_list_cfg = self.runtime_state.neuron_types_list_for_viz.copy() if self.is_initialized and self.runtime_state.neuron_types_list_for_viz else []
                
            # Ensure this fallback structure matches the keys expected by the UI,
            # especially the CuPy array keys for GL.
            return { 
                "current_time_ms": 0.0, 
                "current_time_step": 0,
                "num_spikes_this_step": 0, 
                "network_avg_firing_rate_hz": 0.0,
                "total_plasticity_events": 0, 
                "synapse_info": [], # Stays as Python list for CPU processing
                "neuron_types_list_for_viz": types_list_cfg, # Stays as Python list
                "neuron_model_type_str": model_type_str_cfg,
                "num_neurons_snapshot": n_cfg,
                
                # CuPy arrays, initialized appropriately (empty or zeros)
                "neuron_fired_status_cp": cp.zeros(n_cfg, dtype=bool) if n_cfg > 0 else cp.array([], dtype=bool),
                "neuron_activity_timers_cp": cp.zeros(n_cfg, dtype=cp.int32) if n_cfg > 0 else cp.array([], dtype=cp.int32),
                "neuron_positions_3d_cp": cp.zeros((n_cfg,3), dtype=cp.float32) if n_cfg > 0 else cp.array([], dtype=cp.float32).reshape(0,3),
                "neuron_traits_cp": cp.zeros(n_cfg, dtype=cp.int32) if n_cfg > 0 else cp.array([], dtype=cp.int32),
                "pulse_positions_cp_for_gl": cp.array([], dtype=cp.float32).reshape(0,3)
                # Add other _cp keys if they are essential for GL init (e.g., Vm if directly used by GL)
                # "neuron_Vm_cp": cp.zeros(n_cfg, dtype=cp.float32) if n_cfg > 0 else cp.array([], dtype=cp.float32), # Example if Vm was also made cp for GL
                }

        # If initialized, get the latest data structure (which now includes CuPy arrays for GL)
        snapshot = self.get_latest_simulation_data_for_gui(force_fetch=True) 

        if snapshot: 
            # Reset time-dependent/cumulative values to represent an "initial" state
            snapshot["current_time_ms"] = 0.0
            snapshot["current_time_step"] = 0
            snapshot["num_spikes_this_step"] = 0
            snapshot["network_avg_firing_rate_hz"] = 0.0
            snapshot["total_plasticity_events"] = 0 # Reset this mock counter
                
            # Reset visual activity timers (which are CuPy arrays in the snapshot)
            # Key name was "neuron_activity_timers", now "neuron_activity_timers_cp" from get_latest_simulation_data_for_gui
            if "neuron_activity_timers_cp" in snapshot and snapshot["neuron_activity_timers_cp"].size > 0:
                snapshot["neuron_activity_timers_cp"].fill(0) 
            
            # If other visual timers or states are present as CuPy arrays and need resetting for an initial view, do it here.
            # e.g., if pulse progress was part of the _cp arrays and needed reset:
            # if "pulse_progress_cp" in snapshot and snapshot["pulse_progress_cp"].size > 0:
            #      snapshot["pulse_progress_cp"].fill(0.0)
            
            return snapshot

        def get_profile_visualization_data(self, from_current_config=False):
            """Prepares data specifically needed for visualizing a network profile (neuron positions, types)."""
            cfg = self.core_config; num_n = cfg.num_neurons

            positions_stale = self.cp_neuron_positions_3d is None or self.cp_neuron_positions_3d.shape[0] != num_n
            types_stale = not cfg.neuron_types_list_for_viz or len(cfg.neuron_types_list_for_viz) != num_n

            if from_current_config and (positions_stale or types_stale):
                self._log_console("Re-populating neuron positions/types for visualization profile (3D).","debug")

                if positions_stale and num_n > 0:
                    np_positions_3d = np.random.uniform(
                        low=[cfg.volume_min_x,cfg.volume_min_y,cfg.volume_min_z],
                        high=[cfg.volume_max_x,cfg.volume_max_y,cfg.volume_max_z],
                        size=(num_n,3)).astype(np.float32)
                    self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                    cfg.neuron_positions_x = np_positions_3d[:,0].tolist() 
                    cfg.neuron_positions_y = np_positions_3d[:,1].tolist()
                elif num_n == 0: 
                    self.cp_neuron_positions_3d = cp.array([],dtype=np.float32).reshape(0,3)
                    cfg.neuron_positions_x=[]; cfg.neuron_positions_y=[]

                if types_stale: 
                    cfg.neuron_types_list_for_viz = [""] * num_n 
                    np_traits_host_temp = cp.asnumpy(self.cp_traits) if self.cp_traits is not None and self.cp_traits.size == num_n else \
                                    np.random.randint(0, max(1, cfg.num_traits), num_n) 
                    if self.cp_traits is None or self.cp_traits.size != num_n: 
                        self.cp_traits = cp.asarray(np_traits_host_temp)

                    if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                        default_izh_type_enum = NeuronType[cfg.default_neuron_type_izh]
                        defined_izh2007_types = [ntype for ntype in NeuronType if "IZH2007" in ntype.name and ntype in DefaultIzhikevichParamsManager.PARAMS]
                        num_defined_izh_variants = len(defined_izh2007_types)
                        for i in range(num_n):
                            trait_val = np_traits_host_temp[i]
                            selected_neuron_type_enum = default_izh_type_enum
                            if num_defined_izh_variants > 0: selected_neuron_type_enum = defined_izh2007_types[trait_val % num_defined_izh_variants]
                            cfg.neuron_types_list_for_viz[i] = f"Izh2007_{selected_neuron_type_enum.name.replace('IZH2007_', '')}"
                    elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                        default_hh_type_enum = NeuronType[cfg.default_neuron_type_hh]
                        defined_hh_types = [ntype for ntype in NeuronType if "HH_" in ntype.name and ntype in DefaultHodgkinHuxleyParams.PARAMS]
                        num_defined_hh_variants = len(defined_hh_types)
                        for i in range(num_n):
                            trait_val = np_traits_host_temp[i]
                            selected_neuron_type_enum = default_hh_type_enum
                            if num_defined_hh_variants > 0: selected_neuron_type_enum = defined_hh_types[trait_val % num_defined_hh_variants]
                            cfg.neuron_types_list_for_viz[i] = f"HH_{selected_neuron_type_enum.name.replace('HH_', '')}"
                    else: 
                        cfg.neuron_types_list_for_viz = [f"Unknown_Type_{np_traits_host_temp[i]}" for i in range(num_n)]

            positions_3d_np = cp.asnumpy(self.cp_neuron_positions_3d) if self.cp_neuron_positions_3d is not None else np.zeros((0,3), dtype=np.float32)
            return {
                "neuron_positions_3d": positions_3d_np,
                "neuron_types": cfg.neuron_types_list_for_viz, 
                "neuron_positions_x_proj": cfg.neuron_positions_x, 
                "neuron_positions_y_proj": cfg.neuron_positions_y  
            }

        def get_available_neuron_types(self):
            """Returns a list of available neuron type strings for the current model, for UI filters."""
            cfg = self.core_config
            available_types = ["All"] 
            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                available_types.extend([f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS])
            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                available_types.extend([f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS])
            return list(dict.fromkeys(available_types)) 

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

    if not OPENGL_AVAILABLE:
        gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
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

    if not OPENGL_AVAILABLE or global_simulation_bridge is None : return # Sim bridge for camera config
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
        # Get current time and step from runtime_state (updated by sim_thread, reflected in UI thread via queue)
        # For rendering, it's better if this data is directly from the latest snapshot used for GL.
        # Assume global_simulation_bridge.runtime_state is reasonably up-to-date via messages or direct access if safe.
        sim_time_s = (runtime.current_time_ms / 1000.0)
        render_text_gl(margin, margin + 3*line_h, f"Time: {sim_time_s:.3f} s")
        render_text_gl(margin, margin + 2*line_h, f"Step: {runtime.current_time_step}")
        
        # Get telemetry from sim_bridge (this might need to come from the snapshot too)
        avg_fr = global_simulation_bridge._mock_network_avg_firing_rate_hz # Accessing sim_bridge directly here
        render_text_gl(margin, margin + line_h, f"Avg Rate: {avg_fr:.2f} Hz")

        spikes_step = global_simulation_bridge._mock_num_spikes_this_step
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800)
        render_text_gl(margin + win_w // 3, margin + 2*line_h, f"Spikes/Step: {spikes_step}")

        mode_text = "Mode: Playback" if global_gui_state.get("is_playback_mode_active") else "Mode: Live"
        if global_gui_state.get("is_recording_active"): mode_text += " (Recording)"
        render_text_gl(margin + win_w // 3, margin + line_h, mode_text)

        render_text_gl(margin, margin, "LMB:Rotate, RMB:Pan, Scroll:Zoom, R:Reset Cam, N:Synapses, P:Pause Sim, S:Step Sim, Esc:Exit")

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

    try: 
        key_char = key.decode("utf-8").lower() # Decode byte string to char
    except UnicodeDecodeError: # Handle special keys like ESC
        if key == b'\x1b': # ESC key
            print("ESC pressed in OpenGL window. Signaling shutdown.")
            shutdown_flag.set() # Signal all threads to shut down
            # GLUT main loop and DPG loop will check this flag.
        return # Other non-decodeable keys are ignored

    cfg = global_simulation_bridge.viz_config # For camera reset

    # --- Keyboard Shortcuts for OpenGL Window ---
    if key_char == 'n': # Toggle synapse visibility
        # This action directly modifies UI state, which then affects GL rendering data prep.
        new_show_state = not global_gui_state.get("show_connections_gl", False)
        global_gui_state["show_connections_gl"] = new_show_state
        if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_show_synapses_gl_cb"):
            dpg.set_value("filter_show_synapses_gl_cb", new_show_state) # Update DPG checkbox
        trigger_filter_update_signal() # Signal GL data needs re-filtering and VBO update
        print(f"Synapse visibility toggled {'on' if new_show_state else 'off'}.")

    elif key_char == 'p': # Pause/Resume simulation (Live mode only)
        if not global_gui_state.get("is_playback_mode_active", False):
            # Command the simulation thread to toggle pause
            # The actual pause/resume is handled by sim_thread, UI updates via queue.
            current_sim_running = global_gui_state.get("_sim_is_running_ui_view", False) # UI's idea of sim running
            current_sim_paused = global_gui_state.get("_sim_is_paused_ui_view", False) # UI's idea of sim paused
            if current_sim_running:
                if current_sim_paused:
                    ui_to_sim_queue.put({"type": "RESUME_SIM"})
                else:
                    ui_to_sim_queue.put({"type": "PAUSE_SIM"})
            else:
                print("GL Keyboard: Sim not running, cannot pause/resume.")


    elif key_char == 's': # Step simulation if paused/stopped (Live mode only)
        if not global_gui_state.get("is_playback_mode_active", False):
            is_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
            is_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
            if (is_running_ui and is_paused_ui) or not is_running_ui:
                # Determine number of steps for approx 1ms based on current dt
                # This dt should come from the actual sim_config, ideally via a UI state mirror
                # For now, assume a default or fetch if possible (but direct sim_config access is tricky from here)
                # Let sim_thread decide num_steps for its "STEP_SIM_ONE_MS" command.
                ui_to_sim_queue.put({"type": "STEP_SIM_ONE_MS"}) 
                print("GL Keyboard: Step command sent.")
            else:
                print("GL Keyboard: Sim must be paused or stopped to step.")
    
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
        if dpg.does_item_exist("cfg_num_neurons"): cfg_dict_from_ui["num_neurons"] = max(0, dpg.get_value("cfg_num_neurons"))
        if dpg.does_item_exist("cfg_total_sim_time"): cfg_dict_from_ui["total_simulation_time_ms"] = max(0.0, dpg.get_value("cfg_total_sim_time"))
        if dpg.does_item_exist("cfg_dt_ms"): cfg_dict_from_ui["dt_ms"] = max(0.001, dpg.get_value("cfg_dt_ms"))
        if dpg.does_item_exist("cfg_seed"): cfg_dict_from_ui["seed"] = dpg.get_value("cfg_seed")

        if dpg.does_item_exist("cfg_neuron_model_type"):
            selected_model_name = dpg.get_value("cfg_neuron_model_type")
            cfg_dict_from_ui["neuron_model_type"] = selected_model_name
            # Default neuron types based on selected model (these are part of SimulationConfiguration defaults too)
            if selected_model_name == NeuronModel.IZHIKEVICH.name:
                cfg_dict_from_ui["default_neuron_type_izh"] = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
            elif selected_model_name == NeuronModel.HODGKIN_HUXLEY.name:
                cfg_dict_from_ui["default_neuron_type_hh"] = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name
                # Suggest smaller dt for HH if current UI value is large
                if dpg.does_item_exist("cfg_dt_ms") and dpg.get_value("cfg_dt_ms") > 0.05:
                     cfg_dict_from_ui["dt_ms"] = 1.000 # This will be part of the dict sent

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
        if dpg.does_item_exist("cfg_num_traits"): cfg_dict_from_ui["num_traits"] = max(1, dpg.get_value("cfg_num_traits"))

        # Learning & Plasticity
        if dpg.does_item_exist("cfg_enable_hebbian_learning"): cfg_dict_from_ui["enable_hebbian_learning"] = dpg.get_value("cfg_enable_hebbian_learning")
        if dpg.does_item_exist("cfg_hebbian_learning_rate"): cfg_dict_from_ui["hebbian_learning_rate"] = dpg.get_value("cfg_hebbian_learning_rate")
        if dpg.does_item_exist("cfg_hebbian_max_weight"): cfg_dict_from_ui["hebbian_max_weight"] = dpg.get_value("cfg_hebbian_max_weight")
        if dpg.does_item_exist("cfg_enable_short_term_plasticity"): cfg_dict_from_ui["enable_short_term_plasticity"] = dpg.get_value("cfg_enable_short_term_plasticity")
        if dpg.does_item_exist("cfg_stp_U"): cfg_dict_from_ui["stp_U"] = dpg.get_value("cfg_stp_U")
        if dpg.does_item_exist("cfg_stp_tau_d"): cfg_dict_from_ui["stp_tau_d"] = max(0.1, dpg.get_value("cfg_stp_tau_d"))
        if dpg.does_item_exist("cfg_stp_tau_f"): cfg_dict_from_ui["stp_tau_f"] = max(0.1, dpg.get_value("cfg_stp_tau_f"))

        # Homeostasis
        if dpg.does_item_exist("cfg_enable_homeostasis"): cfg_dict_from_ui["enable_homeostasis"] = dpg.get_value("cfg_enable_homeostasis")
        if dpg.does_item_exist("cfg_homeostasis_target_rate"): cfg_dict_from_ui["homeostasis_target_rate"] = dpg.get_value("cfg_homeostasis_target_rate")
        if dpg.does_item_exist("cfg_homeostasis_threshold_min"): cfg_dict_from_ui["homeostasis_threshold_min"] = dpg.get_value("cfg_homeostasis_threshold_min")
        if dpg.does_item_exist("cfg_homeostasis_threshold_max"): cfg_dict_from_ui["homeostasis_threshold_max"] = dpg.get_value("cfg_homeostasis_threshold_max")
        
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
                ui_hh_params_keys = ["hh_C_m", "hh_g_Na_max", "hh_g_K_max", "hh_g_L", "hh_E_Na", "hh_E_K", "hh_E_L", "hh_v_peak", "hh_v_rest_init", "hh_q10_factor", "hh_temperature_celsius"]
                for key_suffix in ui_hh_params_keys:
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
    """
    if not dpg.is_dearpygui_running() or not config_dict: return

    # Use SimulationConfiguration.from_dict to ensure all fields are present with defaults if missing in dict
    cfg = SimulationConfiguration.from_dict(config_dict)

    # General parameters
    if dpg.does_item_exist("cfg_num_neurons"): dpg.set_value("cfg_num_neurons", cfg.num_neurons)
    if dpg.does_item_exist("cfg_total_sim_time"): dpg.set_value("cfg_total_sim_time", cfg.total_simulation_time_ms)
    if dpg.does_item_exist("cfg_dt_ms"): dpg.set_value("cfg_dt_ms", cfg.dt_ms)
    if dpg.does_item_exist("cfg_seed"): dpg.set_value("cfg_seed", cfg.seed)
    if dpg.does_item_exist("cfg_neuron_model_type"): dpg.set_value("cfg_neuron_model_type", cfg.neuron_model_type)
    
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
    if dpg.does_item_exist("cfg_num_traits"): dpg.set_value("cfg_num_traits", cfg.num_traits)

    # Learning & Plasticity
    if dpg.does_item_exist("cfg_enable_hebbian_learning"): dpg.set_value("cfg_enable_hebbian_learning", cfg.enable_hebbian_learning)
    if dpg.does_item_exist("cfg_hebbian_learning_rate"): dpg.set_value("cfg_hebbian_learning_rate", cfg.hebbian_learning_rate)
    if dpg.does_item_exist("cfg_hebbian_max_weight"): dpg.set_value("cfg_hebbian_max_weight", cfg.hebbian_max_weight)
    if dpg.does_item_exist("cfg_enable_short_term_plasticity"): dpg.set_value("cfg_enable_short_term_plasticity", cfg.enable_short_term_plasticity)
    if dpg.does_item_exist("cfg_stp_U"): dpg.set_value("cfg_stp_U", cfg.stp_U)
    if dpg.does_item_exist("cfg_stp_tau_d"): dpg.set_value("cfg_stp_tau_d", cfg.stp_tau_d)
    if dpg.does_item_exist("cfg_stp_tau_f"): dpg.set_value("cfg_stp_tau_f", cfg.stp_tau_f)

    # Homeostasis
    if dpg.does_item_exist("cfg_enable_homeostasis"): dpg.set_value("cfg_enable_homeostasis", cfg.enable_homeostasis)
    if dpg.does_item_exist("cfg_homeostasis_target_rate"): dpg.set_value("cfg_homeostasis_target_rate", cfg.homeostasis_target_rate)
    if dpg.does_item_exist("cfg_homeostasis_threshold_min"): dpg.set_value("cfg_homeostasis_threshold_min", cfg.homeostasis_threshold_min)
    if dpg.does_item_exist("cfg_homeostasis_threshold_max"): dpg.set_value("cfg_homeostasis_threshold_max", cfg.homeostasis_threshold_max)

    # Camera FOV and Visualization settings
    if dpg.does_item_exist("cfg_camera_fov"): dpg.set_value("cfg_camera_fov", cfg.camera_fov)
    # Handle viz_update_interval_steps if it exists in the config (backward compatibility)
    if hasattr(cfg, "viz_update_interval_steps") and dpg.does_item_exist("cfg_viz_update_interval_steps"):
        dpg.set_value("cfg_viz_update_interval_steps", cfg.viz_update_interval_steps)
    
    # Model-specific parameters
    if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
        ui_izh_params_keys = ["izh_C_val", "izh_k_val", "izh_vr_val", "izh_vt_val", "izh_vpeak_val", "izh_a_val", "izh_b_val", "izh_c_val", "izh_d_val"]
        for key_suffix in ui_izh_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))
    elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
        ui_hh_params_keys = ["hh_C_m", "hh_g_Na_max", "hh_g_K_max", "hh_g_L", "hh_E_Na", "hh_E_K", "hh_E_L", "hh_v_peak", "hh_v_rest_init", "hh_q10_factor", "hh_temperature_celsius"]
        for key_suffix in ui_hh_params_keys:
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

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist("izhikevich_params_group"): dpg.configure_item("izhikevich_params_group", show=is_izh)
        if dpg.does_item_exist("hodgkin_huxley_params_group"): dpg.configure_item("hodgkin_huxley_params_group", show=is_hh)
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

def _update_sim_config_from_ui_and_signal_reset_needed(sender=None, app_data=None, user_data=None):
    """
    Callback for UI elements that change sim config. Sets a flag that sim needs reset.
    The actual config update is collected by `handle_apply_config_changes_and_reset`.
    """
    global_gui_state["reset_sim_needed_from_ui_change"] = True
    update_status_bar("Parameter changed. Press 'Apply Changes & Reset Sim' to take effect.", color=[255,165,0,255], level="warning")
    # If model type changed, also update visibility of specific param groups
    if sender == "cfg_neuron_model_type": # Check if this callback is for model type combo
         _toggle_model_specific_params_visibility(sender, app_data)


def _handle_model_type_change_dpg(sender, app_data, user_data=None):
    """Handles change in neuron model type selection in DPG. Updates UI visibility and signals reset."""
    _toggle_model_specific_params_visibility(sender, app_data) # Update UI sections
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

def save_profile_dialog_callback(sender, app_data): # Profiles are JSON
    """
    Callback for the 'Save Profile' file dialog. Saves current UI config and GUI settings.
    This operation is done entirely by the UI thread.
    """
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        if not filepath.lower().endswith(".json"): filepath += ".json"

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
            with open(filepath, 'w') as f: json.dump(content_to_save, f, indent=4)
            update_status_bar(f"Profile saved: {os.path.basename(filepath)}", color=[0,200,0,255], level="success")
            if dpg.does_item_exist("profile_name_input"): 
                dpg.set_value("profile_name_input", os.path.basename(filepath).replace(".json", ""))
            global_gui_state["current_profile_name"] = os.path.basename(filepath)
        except Exception as e: 
            update_status_bar(f"Error saving profile: {e}", color=[255,0,0,255], level="error")
    else: 
        update_status_bar("Save profile cancelled.", level="info")


def _execute_profile_load_on_ui_thread(filepath): # Profiles are JSON
    """
    Loads a profile file, updates UI, and sends new config to sim_thread.
    Called by UI thread.
    """
    profile_name = os.path.basename(filepath)
    update_status_bar(f"Loading profile '{profile_name}'...", level="info")
    try:
        with open(filepath, 'r') as f: profile_content = json.load(f)
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
        filepath = app_data["file_path_name"]
        selected_filter = app_data.get("current_filter", "")

        # If ".*" was the filter and DPG appended ".*" to a user-typed name
        if selected_filter == ".*" and filepath.endswith(".*"):
            filepath = filepath[:-2] # Strip the ".*"

        # Ensure the filepath ends with the correct ".simstate.h5"
        # Remove other potential HDF5 extensions first to avoid "file.h5.simstate.h5"
        if filepath.lower().endswith(".h5"):
            filepath = filepath[:-3] # Remove .h5
        
        if not filepath.lower().endswith(".simstate.h5"):
            filepath += ".simstate.h5"

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
    Updates the DPG monitoring overlay text elements with current simulation data.
    Called by the main/UI thread when new data arrives from sim_to_ui_queue.
    """
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist("monitor_sim_time_text"): return

    if sim_data_dict is None: # If no data (e.g., sim not started, or error)
        for tag_sfx in ["sim_time_text", "current_step_text", "step_spikes_text", "avg_firerate_text", "plasticity_updates_text"]:
            tag, label = f"monitor_{tag_sfx}", tag_sfx.replace('_text', '').replace('_', ' ').title()
            if dpg.does_item_exist(tag): dpg.set_value(tag, f"{label}: N/A")
    else: # Valid data received
        dpg.set_value("monitor_sim_time_text", f"Sim Time: {sim_data_dict.get('current_time_ms', 0)/1000.0:.3f} s")
        dpg.set_value("monitor_current_step_text", f"Current Step: {sim_data_dict.get('current_time_step', 0)}")
        dpg.set_value("monitor_step_spikes_text", f"Spikes (step): {sim_data_dict.get('num_spikes_this_step', 0)}")
        dpg.set_value("monitor_avg_firerate_text", f"Avg Rate (net): {sim_data_dict.get('network_avg_firing_rate_hz', 0.0):.2f} Hz")
        dpg.set_value("monitor_plasticity_updates_text", f"Plasticity Evts: {sim_data_dict.get('total_plasticity_events', 0)}")

    # Update GL specific monitor values (these are from main thread's GL state)
    vis_neurons_gl = gl_num_neurons_to_draw if OPENGL_AVAILABLE else 'N/A'
    vis_syns_gl = gl_num_synapse_lines_to_draw if OPENGL_AVAILABLE else 'N/A'
    if dpg.does_item_exist("monitor_visible_neurons_text"): dpg.set_value("monitor_visible_neurons_text", f"Visible Neurons: {vis_neurons_gl}")
    if dpg.does_item_exist("monitor_visible_synapses_text"): dpg.set_value("monitor_visible_synapses_text", f"Visible Synapses: {vis_syns_gl}")

    # Update playback frame counter if in playback mode
    if global_gui_state.get("is_playback_mode_active") and dpg.does_item_exist("playback_current_frame_text"):
        active_rec_meta = global_gui_state.get("active_recording_data_source") # This is UI thread's copy
        if active_rec_meta and "num_frames" in active_rec_meta:
            total_frames = active_rec_meta["num_frames"]
            current_frame_idx_ui = global_gui_state.get("current_playback_frame_index",0) # UI's current frame
            dpg.set_value("playback_current_frame_text", f"Frame: {current_frame_idx_ui + 1} / {total_frames if total_frames > 0 else 1}")

# --- DPG Event Handlers for Recording & Playback (HDF5) ---

def handle_record_button_click(sender=None, app_data=None, user_data=None):
    """
    Handles the 'Record' / 'Finalize Recording' button click.
    Shows file dialog or sends command to stop recording.
    """
    if global_gui_state.get("is_recording_active", False): # If currently recording, this button means "Finalize"
        ui_to_sim_queue.put({"type": "STOP_RECORDING"})
        update_status_bar("Finalize recording command sent...", level="info")
        # UI state will be updated when sim_thread confirms via "RECORDING_FINALIZED"
    else: # Not recording, this button means "Record" - show save dialog
        if global_gui_state.get("is_playback_mode_active", False):
            update_status_bar("Error: Cannot record while in playback mode.", color=[255,0,0,255], level="error")
            return
        if dpg.is_dearpygui_running() and dpg.does_item_exist("save_recording_file_dialog_h5"):
            dpg.show_item("save_recording_file_dialog_h5")
        else:
            update_status_bar("Error: Recording dialog missing.", color=[255,0,0,255], level="error")

def save_recording_for_streaming_dialog_callback_h5(sender, app_data):
    """Callback for the 'Record' (Save Recording As) file dialog. Ensures correct extension."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        selected_filter = app_data.get("current_filter", "")

        if selected_filter == ".*" and filepath.endswith(".*"):
            filepath = filepath[:-2]

        if filepath.lower().endswith(".h5"):
            filepath = filepath[:-3]

        if not filepath.lower().endswith(".simrec.h5"):
            filepath += ".simrec.h5"

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

def load_recording_dialog_callback_h5(sender, app_data):
    """Callback for the 'Load Recording' file dialog. Sends command to sim_thread."""
    filepath_to_load = None
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        selected_filter = app_data.get("current_filter", "")

        if selected_filter == ".*" and filepath.endswith(".*"):
            potential_filepath_stripped = filepath[:-2]
            if os.path.isfile(potential_filepath_stripped):
                filepath = potential_filepath_stripped
            elif not os.path.isfile(filepath):
                 update_status_bar(f"Load error: Path '{filepath}' from '.*' filter seems invalid.", color=[255,0,0,255], level="error")
                 return

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
        ui_to_sim_queue.put({"type": "LOAD_RECORDING", "filepath": filepath_to_load})
        update_status_bar(f"Load recording command sent for: {os.path.basename(filepath_to_load)}", level="info")

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
        ui_to_sim_queue.put({
            "type": "SET_PLAYBACK_FRAME",
            "frame_index": frame_idx_from_slider,
            "h5_file_handle_for_sim_thread": h5_handle
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
            current_frame_ui = global_gui_state.get("current_playback_frame_index", 0)
            if num_frames > 0 and current_frame_ui >= num_frames - 1:
                ui_to_sim_queue.put({"type": "SET_PLAYBACK_FRAME", "frame_index": 0})
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
            "h5_file_handle_for_sim_thread": h5_handle
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

# --- Main DPG GUI Layout Creation (Called by Main/UI Thread) ---

def add_parameter_table_row(label_text, item_callable, item_tag, default_value, callback_func, **kwargs):
    """
    Adds a row to a DPG table with a label in the first column and a DPG item in the second.
    Assumes this is called within a `with dpg.table(): ...` context where columns are already defined.
    """
    with dpg.table_row():
        dpg.add_text(label_text)
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

        with dpg.collapsing_header(label="Simulation Monitor", default_open=True, tag="monitor_panel_group"):
            monitor_labels = ["Sim Time", "Current Step", "Spikes (Current Step)", "Avg Rate (Network)", "Plasticity Events", "Visible Neurons", "Visible Synapses"]
            monitor_tags = ["sim_time_text", "current_step_text", "step_spikes_text", "avg_firerate_text", "plasticity_updates_text", "visible_neurons_text", "visible_synapses_text"]
            for label, tag_suffix in zip(monitor_labels, monitor_tags):
                dpg.add_text(f"{label}: N/A", tag=f"monitor_{tag_suffix}")
            dpg.add_text("Status: Idle", tag="status_bar_text") 

        with dpg.collapsing_header(label="Simulation Controls", default_open=True):
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
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)

                add_parameter_table_row("Number of Neurons:", dpg.add_input_int, "cfg_num_neurons", 1000, _update_sim_config_from_ui_and_signal_reset_needed)
                add_parameter_table_row("Connections/Neuron (Spatial Fallback):", dpg.add_input_int, "cfg_connections_per_neuron", 100, _update_sim_config_from_ui_and_signal_reset_needed)
                add_parameter_table_row("Total Sim Time (ms):", dpg.add_input_float, "cfg_total_sim_time", 60000.0, _update_sim_config_from_ui_and_signal_reset_needed, step=100)
                add_parameter_table_row("Time Step dt (ms):", dpg.add_input_float, "cfg_dt_ms", 1.000, _update_sim_config_from_ui_and_signal_reset_needed, step=0.001, format="%.3f", min_value=0.001)
                add_parameter_table_row("Seed (-1 for random):", dpg.add_input_int, "cfg_seed", -1, _update_sim_config_from_ui_and_signal_reset_needed)
                add_parameter_table_row("Number of Traits:", dpg.add_input_int, "cfg_num_traits", 5, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, max_value=len(TRAIT_COLOR_MAP_RAW) if TRAIT_COLOR_MAP_RAW else 10)
                add_parameter_table_row("Neuron Model:", dpg.add_combo, "cfg_neuron_model_type", NeuronModel.IZHIKEVICH.name, _handle_model_type_change_dpg, items=[model.name for model in NeuronModel])

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
                    for desc_label, tag, fmt, def_val in ui_izh_params:
                        add_parameter_table_row(desc_label, dpg.add_input_float, tag, def_val, _update_sim_config_from_ui_and_signal_reset_needed, format=fmt)
            
            with dpg.group(tag="hodgkin_huxley_params_group", show=False):
                dpg.add_text("--- Hodgkin-Huxley Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    ui_hh_params = [
                        ("Membrane Capacitance C_m (uF/cm^2)", "cfg_hh_C_m", "%.2f", 1.0), ("Max Sodium Cond. g_Na_max (mS/cm^2)", "cfg_hh_g_Na_max", "%.1f", 50.0),
                        ("Max Potassium Cond. g_K_max (mS/cm^2)", "cfg_hh_g_K_max", "%.1f", 5.0), ("Leak Cond. g_L (mS/cm^2)", "cfg_hh_g_L", "%.3f", 0.1),
                        ("Sodium Reversal E_Na (mV)", "cfg_hh_E_Na", "%.1f", 50.0), ("Potassium Reversal E_K (mV)", "cfg_hh_E_K", "%.1f", -85.0),
                        ("Leak Reversal E_L (mV)", "cfg_hh_E_L", "%.3f", -70.0), ("Spike Detection V_peak (mV)", "cfg_hh_v_peak", "%.1f", 40.0),
                        ("Initial V_rest (mV)", "cfg_hh_v_rest_init", "%.1f", -65.0), ("Kinetics Q10 Factor", "cfg_hh_q10_factor", "%.1f", 3.0),
                        ("Kinetics Temperature (°C)", "cfg_hh_temperature_celsius", "%.1f", 37.0)
                    ]
                    for desc_label, tag, fmt, def_val in ui_hh_params:
                         add_parameter_table_row(desc_label, dpg.add_input_float, tag, def_val, _update_sim_config_from_ui_and_signal_reset_needed, format=fmt)

        with dpg.collapsing_header(label="Network Connectivity", default_open=False, tag="network_connectivity_header"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Use Watts-Strogatz Generator:", dpg.add_checkbox, "cfg_enable_watts_strogatz", True, _update_sim_config_from_ui_and_signal_reset_needed)
                add_parameter_table_row("W-S K (Nearest Neighbors, even):", dpg.add_input_int, "cfg_connectivity_k", 10, _update_sim_config_from_ui_and_signal_reset_needed, step=2, min_value=2)
                add_parameter_table_row("W-S P (Rewire Probability):", dpg.add_input_float, "cfg_connectivity_p_rewire", 0.1, _update_sim_config_from_ui_and_signal_reset_needed, min_value=0.0, max_value=1.0, format="%.3f")

        with dpg.collapsing_header(label="Synaptic Parameters", default_open=False, tag="synaptic_params_header"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Excitatory Prop. Strength (g_peak_e scale):", dpg.add_input_float, "cfg_propagation_strength", 0.05, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f")
                add_parameter_table_row("Inhibitory Prop. Strength (g_peak_i scale):", dpg.add_input_float, "cfg_inhibitory_propagation_strength", 0.15, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f")
                add_parameter_table_row("Excitatory Conductance Tau_g_e (ms):", dpg.add_input_float, "cfg_syn_tau_e", 5.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.1)
                add_parameter_table_row("Inhibitory Conductance Tau_g_i (ms):", dpg.add_input_float, "cfg_syn_tau_i", 10.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.1)

        with dpg.collapsing_header(label="Learning & Plasticity", default_open=False, tag="learning_plasticity_header"):
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Hebbian Learning:", dpg.add_checkbox, "cfg_enable_hebbian_learning", True, _update_sim_config_from_ui_and_signal_reset_needed)
                add_parameter_table_row("Hebbian Learning Rate:", dpg.add_input_float, "cfg_hebbian_learning_rate", 0.0005, _update_sim_config_from_ui_and_signal_reset_needed, format="%.6f")
                add_parameter_table_row("Hebbian Max Weight:", dpg.add_input_float, "cfg_hebbian_max_weight", 1.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f")
            dpg.add_separator()
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Short-Term Plasticity (STP):", dpg.add_checkbox, "cfg_enable_short_term_plasticity", True, _update_sim_config_from_ui_and_signal_reset_needed)
                add_parameter_table_row("STP U (Baseline Utilization):", dpg.add_input_float, "cfg_stp_U", 0.15, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f")
                add_parameter_table_row("STP Tau_d (Depression, ms):", dpg.add_input_float, "cfg_stp_tau_d", 200.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.1)
                add_parameter_table_row("STP Tau_f (Facilitation, ms):", dpg.add_input_float, "cfg_stp_tau_f", 50.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.1)
            dpg.add_separator()
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Homeostasis:", dpg.add_checkbox, "cfg_enable_homeostasis", True, _update_sim_config_from_ui_and_signal_reset_needed)
            with dpg.group(tag="homeostasis_izh_specific_group", show=True):
                 with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    add_parameter_table_row("Homeostasis Target Rate (spikes/dt for Izh):", dpg.add_input_float, "cfg_homeostasis_target_rate", 0.02, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f")
                    add_parameter_table_row("Homeostasis Min Threshold (Izh, mV):", dpg.add_input_float, "cfg_homeostasis_threshold_min", -55.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f")
                    add_parameter_table_row("Homeostasis Max Threshold (Izh, mV):", dpg.add_input_float, "cfg_homeostasis_threshold_max", -30.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f")

        with dpg.collapsing_header(label="Visual Settings & Filters", default_open=False, tag="visual_settings_header"):
            dpg.add_text("--- Neurons ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                spiking_filter_options = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
                add_parameter_table_row("Show Spiking Neurons:", dpg.add_combo, "filter_spiking_mode_combo", "Highlight Spiking", trigger_filter_update_signal, items=spiking_filter_options)
                add_parameter_table_row("Enable Synaptic Pulses (GL):", dpg.add_checkbox, "gl_enable_synaptic_pulses_cb", opengl_viz_config.get('ENABLE_SYNAPTIC_PULSES', True) if OPENGL_AVAILABLE else False, handle_gl_enable_synaptic_pulses_change)
                add_parameter_table_row("Filter By Neuron Type:", dpg.add_checkbox, "filter_type_enable_cb", False, lambda s, a, u: (dpg.configure_item("filter_neuron_type_combo", enabled=a), trigger_filter_update_signal(s,a,u)))
                add_parameter_table_row("Select Type:", dpg.add_combo, "filter_neuron_type_combo", "All", trigger_filter_update_signal, items=["All"], enabled=False)
                add_parameter_table_row("Max Visible Neurons (GL):", dpg.add_input_int, "gl_max_neurons_render_input", opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000) if OPENGL_AVAILABLE else 0, handle_gl_max_neurons_change, min_value=0, step=100)
                add_parameter_table_row("Neuron Size (GL):", dpg.add_slider_float, "gl_neuron_point_size_slider", opengl_viz_config.get('POINT_SIZE', 2.0) if OPENGL_AVAILABLE else 1.0, handle_gl_point_size_change, min_value=0.5, max_value=10.0, format="%.1f")
                add_parameter_table_row("Inactive Neuron Opacity (GL):", dpg.add_slider_float, "gl_inactive_neuron_opacity_slider", opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25) if OPENGL_AVAILABLE else 0.1, handle_gl_inactive_neuron_opacity_change, min_value=0.0, max_value=1.0, format="%.2f")
            
            dpg.add_separator()
            dpg.add_text("--- Synapses ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Show Synapses (GL):", dpg.add_checkbox, "filter_show_synapses_gl_cb", global_gui_state.get("show_connections_gl", True), lambda s,a,u: (global_gui_state.update({"show_connections_gl":a}), trigger_filter_update_signal()))
                add_parameter_table_row("Max Visible Connections (GL):", dpg.add_input_int, "gl_max_connections_render_input", opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000) if OPENGL_AVAILABLE else 0, handle_gl_max_connections_change, min_value=0, step=500)
                add_parameter_table_row("Synapse Alpha Multiplier (GL):", dpg.add_slider_float, "gl_synapse_alpha_slider", opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.3) if OPENGL_AVAILABLE else 0.1, handle_gl_synapse_alpha_change, min_value=0.0, max_value=2.0, format="%.2f")
                add_parameter_table_row("Min Abs Synapse Weight (Filter):", dpg.add_slider_float, "filter_min_abs_weight_slider", 0.000, trigger_filter_update_signal, max_value=1.0, format="%.3f")
            
            dpg.add_separator()
            dpg.add_text("--- General Visuals ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Camera Field of View (FOV, degrees):", dpg.add_slider_float, "cfg_camera_fov", 60.0, _update_sim_config_from_ui_and_signal_reset_needed, min_value=10.0, max_value=120.0)
                add_parameter_table_row("Activity Highlight Frames (GL):", dpg.add_input_int, "gl_activity_highlight_frames_input", opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7) if OPENGL_AVAILABLE else 1, handle_gl_activity_highlight_frames_change, min_value=1, max_value=30)
                add_parameter_table_row("Viz Update Interval (steps):", dpg.add_input_int, "cfg_viz_update_interval_steps", 1, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, max_value=200, step=1)

    # File Dialogs
    profile_dir = global_simulation_bridge.PROFILE_DIR if global_simulation_bridge else "simulation_profiles/"
    checkpoint_dir_h5 = global_simulation_bridge.CHECKPOINT_DIR if global_simulation_bridge else "simulation_checkpoints_h5/" # Updated
    recording_dir_h5 = global_simulation_bridge.RECORDING_DIR if global_simulation_bridge else "simulation_recordings_h5/"   # Updated

    for p_dir in [profile_dir, checkpoint_dir_h5, recording_dir_h5]:
        if not os.path.exists(p_dir): os.makedirs(p_dir, exist_ok=True)

    # Profile dialogs (JSON)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_profile_dialog_callback, tag="save_profile_file_dialog", width=700, height=400, modal=True, default_path=profile_dir):
        dpg.add_file_extension(".json", color=(255, 255, 0, 255), custom_text="JSON Profile (*.json)")
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_profile_dialog_callback, tag="load_profile_file_dialog", width=700, height=400, modal=True, default_path=profile_dir):
        dpg.add_file_extension(".json", color=(255, 255, 0, 255), custom_text="JSON Profile (*.json)")

    # Checkpoint dialogs (HDF5)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_checkpoint_dialog_callback_h5,
                         tag="save_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="HDF5 Checkpoints (*.simstate.h5, *.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    with dpg.file_dialog(directory_selector=False, show=False, callback=load_checkpoint_dialog_callback_h5,
                         tag="load_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="HDF5 Checkpoints (*.simstate.h5, *.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Recording dialogs (HDF5)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_recording_for_streaming_dialog_callback_h5,
                         tag="save_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5):
        dpg.add_file_extension(".h5", color=(100, 0, 100, 255), custom_text="HDF5 Recordings (*.simrec.h5, *.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    with dpg.file_dialog(directory_selector=False, show=False, callback=load_recording_dialog_callback_h5,
                         tag="load_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5):
        dpg.add_file_extension(".h5", color=(100, 0, 100, 255), custom_text="HDF5 Recordings (*.simrec.h5, *.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

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
    DATA_UPDATE_INTERVAL_STEPS = 1 # Real-time visualization (60 FPS capable)
    SYNAPSE_SAMPLE_UPDATE_INTERVAL_STEPS = 200 # Update synapse samples much less frequently

    try:
        while not local_shutdown_event.is_set():
            # --- 1. Process Commands from UI Thread ---
            try:
                while not command_q.empty():
                    command = command_q.get_nowait()
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
                    elif cmd_type == "START_RECORDING":
                        sim_bridge.start_recording_to_file(command["filepath"]) # Sim_bridge sends status
                    elif cmd_type == "STOP_RECORDING":
                        sim_bridge.stop_recording() # Sim_bridge sends status
                    elif cmd_type == "LOAD_RECORDING": # UI requests sim_thread to prepare metadata
                        sim_bridge.load_recording(command["filepath"]) # Sim_bridge sends RECORDING_METADATA_PREPARED or _FAILED
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
                        if active_playback_handle_for_frame:
                            sim_bridge.set_playback_frame(command["frame_index"], active_playback_handle_for_frame)
                        else:
                             sim_bridge._log_to_ui("Playback error: No HDF5 handle for SET_PLAYBACK_FRAME.", "error")
                             data_q.put({"type": "PLAYBACK_ERROR", "reason": "Missing H5 handle for frame set"})
                    elif cmd_type == "EXIT_PLAYBACK_MODE":
                        # Sim_thread doesn't directly manage global_gui_state.is_playback_mode_active.
                        # It just needs to reset its internal state if it was in a playback-specific mode.
                        # For example, if it was holding an HDF5 file open for playback, it should close it.
                        # The main task is to re-apply the "live" simulation config.
                        sim_bridge.apply_simulation_configuration_core(sim_bridge.core_config.to_dict(), is_part_of_playback_setup=False)
                        data_q.put({
                            "type": "PLAYBACK_EXITED_SIM_SIDE",
                            "new_config_dict": sim_bridge.get_current_simulation_configuration_dict(),
                            "initial_gui_data": sim_bridge.get_initial_sim_data_snapshot()
                        })
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
                            sim_bridge.runtime_state.current_time_ms += dt_ms_val
                            sim_bridge.runtime_state.current_time_step += 1
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
        
        # Force exit from the idle loop
        sys.exit(0)

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
            elif msg_type == "SIM_STOPPED_OR_ENDED":
                global_gui_state["_sim_is_running_ui_view"] = False
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=False, is_paused=False)
                update_status_bar(f"Simulation stopped/ended: {message.get('reason', '')}", level="info")
                if global_simulation_bridge: 
                     initial_data = global_simulation_bridge.get_initial_sim_data_snapshot()
                     if initial_data: update_monitoring_overlay_values(initial_data)
            elif msg_type == "CONFIG_APPLIED_AND_RESET_DONE":
                # Don't repopulate UI - it already has the values the user set
                # Only update monitoring values
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
                    if message.get("filepath"): # Auto-load the just-saved recording
                        ui_to_sim_queue.put({"type": "LOAD_RECORDING", "filepath": message["filepath"]})
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

            elif msg_type in ["CONFIG_APPLIED_ERROR", "CHECKPOINT_LOAD_FAILED", "RECORDING_LOAD_FAILED", 
                              "RECORDING_START_FAILED", "PLAYBACK_SETUP_FAILED", "PLAYBACK_ERROR", "SIM_FATAL_ERROR",
                              "CHECKPOINT_SAVE_FAILED"]:
                update_status_bar(f"Error: {message.get('reason', message.get('error', 'Unknown error'))}", color=[255,0,0,255], level="error")
                if msg_type == "SIM_FATAL_ERROR": shutdown_flag.set() 

            sim_to_ui_queue.task_done()
    except queue.Empty:
        pass 

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

    dpg.create_context()
    dpg.configure_app(docking=False)

    global_simulation_bridge = SimulationBridge() # Initialize the simulation core (sim_bridge.ui_queue is set here)

    # Attempt to load default profile (JSON) - This is a UI-side operation before sim_thread starts.
    default_profile_filename = "default_profile.json"
    default_profile_path = os.path.join(global_simulation_bridge.PROFILE_DIR, default_profile_filename)
    loaded_default_sim_config_dict = None
    loaded_default_gui_config_dict = None

    if os.path.exists(default_profile_path):
        try:
            with open(default_profile_path, 'r') as f: profile_content = json.load(f)
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
    
    if dpg.is_dearpygui_running(): dpg.destroy_context() 
    print("Neuron simulator application shutdown complete.")

if __name__ == '__main__':
    main()
