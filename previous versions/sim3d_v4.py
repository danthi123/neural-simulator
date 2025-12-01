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
            "a": 0.2, "b": -2.0, "c_reset": -45.0, "d_increment": -55.0
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
        else:
            if 'LEGACY' in neuron_type_enum.name:
                 return DefaultIzhikevichParamsManager.PARAMS.get(neuron_type_enum, DefaultIzhikevichParamsManager.FALLBACK_LEGACY).copy()
            print(f"Warning: Requested 2007 type {neuron_type_enum} for legacy formulation. Using RS_EXCITATORY_LEGACY fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_LEGACY.copy()


class DefaultHodgkinHuxleyParams:
    REALISTIC_L5_PYRAMIDAL_RS_37C = {
        "C_m": 1.0, "g_Na_max": 50.0, "g_K_max": 5.0, "g_L": 0.1,
        "E_Na": 50.0, "E_K": -85.0, "E_L": -70.0,
        "v_rest_hh": -65.0, "v_peak_hh": 40.0,
        "m_init": 0.0529, "h_init": 0.5961, "n_init": 0.3177
    }
    ORIGINAL_HH_PARAMS = {
        "C_m": 1.0, "g_Na_max": 120.0, "g_K_max": 36.0, "g_L": 0.3,
        "E_Na": 50.0, "E_K": -77.0, "E_L": -54.387,
        "v_rest_hh": -65.0, "v_peak_hh": 40.0,
        "m_init": 0.0529, "h_init": 0.5961, "n_init": 0.3177
    }
    PARAMS = {
        NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS: REALISTIC_L5_PYRAMIDAL_RS_37C.copy(),
        NeuronType.HH_EXCITATORY_DEFAULT_LEGACY: ORIGINAL_HH_PARAMS.copy(),
    }
    FALLBACK = PARAMS[NeuronType.HH_EXCITATORY_DEFAULT_LEGACY].copy()

    @staticmethod
    def get_params(neuron_type_enum):
        return DefaultHodgkinHuxleyParams.PARAMS.get(neuron_type_enum, DefaultHodgkinHuxleyParams.FALLBACK).copy()


class SimulationConfiguration:
    def __init__(self):
        self.total_simulation_time_ms = 1000.0
        self.dt_ms = 1.000
        self.num_neurons = 1000
        self.connections_per_neuron = 100
        self.num_traits = 5
        self.seed = -1
        self.neuron_model_type = NeuronModel.IZHIKEVICH.name
        self.default_neuron_type_izh = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
        self.default_neuron_type_hh = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name

        rs_params_2007 = DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]
        self.izh_C_val = rs_params_2007["C"]
        self.izh_k_val = rs_params_2007["k"]
        self.izh_vr_val = rs_params_2007["vr"]
        self.izh_vt_val = rs_params_2007["vt"]
        self.izh_vpeak_val = rs_params_2007["vpeak"]
        self.izh_a_val = rs_params_2007["a"]
        self.izh_b_val = rs_params_2007["b"]
        self.izh_c_val = rs_params_2007["c_reset"]
        self.izh_d_val = rs_params_2007["d_increment"]

        self.lif_v_rest = -65.0
        self.initial_firing_threshold = -40.0
        self.initial_threshold_variation = 1.0

        hh_defaults = DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]
        self.hh_C_m = hh_defaults["C_m"]
        self.hh_g_Na_max = hh_defaults["g_Na_max"]
        self.hh_g_K_max = hh_defaults["g_K_max"]
        self.hh_g_L = hh_defaults["g_L"]
        self.hh_E_Na = hh_defaults["E_Na"]
        self.hh_E_K = hh_defaults["E_K"]
        self.hh_E_L = hh_defaults["E_L"]
        self.hh_v_rest_init = hh_defaults["v_rest_hh"]
        self.hh_v_peak = hh_defaults["v_peak_hh"]
        self.hh_m_init = hh_defaults["m_init"]
        self.hh_h_init = hh_defaults["h_init"]
        self.hh_n_init = hh_defaults["n_init"]
        self.hh_temperature_celsius = 37.0
        self.hh_q10_factor = 3.0

        self.refractory_period_steps = 2
        self.syn_reversal_potential_e = 0.0
        self.syn_reversal_potential_i = -70.0
        self.syn_tau_g_e = 5.0
        self.syn_tau_g_i = 10.0
        self.propagation_strength = 0.05
        self.inhibitory_propagation_strength = 0.15
        self.max_synaptic_delay_ms = 20.0

        self.enable_inhibitory_neurons = True
        self.inhibitory_trait_index = 1

        self.enable_hebbian_learning = True
        self.hebbian_learning_rate = 0.0005
        self.hebbian_weight_decay = 0.00001
        self.hebbian_min_weight = 0.05
        self.hebbian_max_weight = 1.0

        self.enable_short_term_plasticity = True
        self.stp_U = 0.15
        self.stp_tau_d = 200.0
        self.stp_tau_f = 50.0

        self.enable_homeostasis = True
        self.homeostasis_target_rate = 0.02
        self.homeostasis_threshold_adapt_rate = 0.015
        self.homeostasis_ema_alpha = 0.01
        self.homeostasis_threshold_min = -55.0
        self.homeostasis_threshold_max = -30.0

        self.enable_watts_strogatz = True
        self.connectivity_k = 10
        self.connectivity_p_rewire = 0.1

        self.current_time_ms = 0.0
        self.current_time_step = 0
        self.is_running = False
        self.is_paused = False
        self.simulation_speed_factor = 1.0

        self.network_definition_dict = {"neuron_groups": [], "connections": []}
        self.neuron_positions_x = []
        self.neuron_positions_y = []
        self.neuron_types_list_for_viz = []
        self.max_delay_steps = int(self.max_synaptic_delay_ms / self.dt_ms) if self.dt_ms > 0 else 200

        self.volume_min_x = -50.0; self.volume_max_x = 50.0
        self.volume_min_y = -50.0; self.volume_max_y = 50.0
        self.volume_min_z = -50.0; self.volume_max_z = 50.0
        self.camera_eye_x = 0.0; self.camera_eye_y = 0.0; self.camera_eye_z = 150.0
        self.camera_center_x = 0.0; self.camera_center_y = 0.0; self.camera_center_z = 0.0
        self.camera_up_x = 0.0; self.camera_up_y = 1.0; self.camera_up_z = 0.0
        self.camera_fov = 60.0
        self.camera_near_clip = 0.1; self.camera_far_clip = 1000.0
        self.mouse_last_x = 0; self.mouse_last_y = 0
        self.mouse_left_button_down = False; self.mouse_right_button_down = False
        self.camera_azimuth_angle = 0.0; self.camera_elevation_angle = 0.0; self.camera_radius = 150.0

    def reset_simulation_time_and_counters(self):
        self.current_time_ms = 0.0
        self.current_time_step = 0

    def to_dict(self):
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
        if isinstance(data.get('neuron_model_type'), Enum):
            data['neuron_model_type'] = data['neuron_model_type'].name
        if isinstance(data.get('default_neuron_type_izh'), Enum):
            data['default_neuron_type_izh'] = data['default_neuron_type_izh'].name
        if isinstance(data.get('default_neuron_type_hh'), Enum):
            data['default_neuron_type_hh'] = data['default_neuron_type_hh'].name
        return data

    @classmethod
    def from_dict(cls, data):
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key == 'neuron_model_type' and isinstance(value, str):
                    try: setattr(config, key, NeuronModel[value].name)
                    except KeyError: setattr(config, key, cls().neuron_model_type)
                elif key == 'default_neuron_type_izh' and isinstance(value, str):
                    try: setattr(config, key, NeuronType[value].name)
                    except KeyError: setattr(config, key, cls().default_neuron_type_izh)
                elif key == 'default_neuron_type_hh' and isinstance(value, str):
                    try: setattr(config, key, NeuronType[value].name)
                    except KeyError: setattr(config, key, cls().default_neuron_type_hh)
                else:
                    setattr(config, key, value)

        config.dt_ms = getattr(config, 'dt_ms', 0.1)
        if config.dt_ms is None or config.dt_ms <= 0: config.dt_ms = 0.1
        config.max_delay_steps = int(config.max_synaptic_delay_ms / config.dt_ms) if config.dt_ms > 0 else 200

        default_instance_for_fallback = cls()
        izh_2007_params_keys = [
            'izh_C_val', 'izh_k_val', 'izh_vr_val', 'izh_vt_val', 'izh_vpeak_val',
            'izh_a_val', 'izh_b_val', 'izh_c_val', 'izh_d_val'
        ]
        for param_key in izh_2007_params_keys:
            if not hasattr(config, param_key) or getattr(config, param_key) is None:
                 setattr(config, param_key, getattr(default_instance_for_fallback, param_key))

        config.hh_temperature_celsius = getattr(config, 'hh_temperature_celsius', 37.0)
        config.hh_q10_factor = getattr(config, 'hh_q10_factor', 3.0)

        new_3d_cam_params = [
            'volume_min_x', 'volume_max_x', 'volume_min_y', 'volume_max_y', 'volume_min_z', 'volume_max_z',
            'camera_eye_x', 'camera_eye_y', 'camera_eye_z',
            'camera_center_x', 'camera_center_y', 'camera_center_z',
            'camera_up_x', 'camera_up_y', 'camera_up_z',
            'camera_fov', 'camera_near_clip', 'camera_far_clip',
            'mouse_last_x', 'mouse_last_y', 'mouse_left_button_down', 'mouse_right_button_down',
            'camera_azimuth_angle', 'camera_elevation_angle', 'camera_radius'
        ]
        for param_key in new_3d_cam_params:
            if not hasattr(config, param_key) or getattr(config, param_key) is None:
                 setattr(config, param_key, getattr(default_instance_for_fallback, param_key))
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
    dv = (0.04 * v**2 + 5 * v + 140 - u + total_I)
    du = a * (b * v - u)
    v_new = v + dv * dt
    u_new = u + du * dt
    return v_new, u_new

@cp.fuse()
def fused_izhikevich2007_dynamics_update(v, u, C_param, k_param, vr_param, vt_param, a_param, b_param, total_synaptic_current, dt):
    C_param_safe = cp.where(C_param == 0.0, 1.0, C_param)
    dv_dt = (k_param * (v - vr_param) * (v - vt_param) - u + total_synaptic_current) / C_param_safe
    du_dt = a_param * (b_param * (v - vr_param) - u)
    v_new = v + dv_dt * dt
    u_new = u + du_dt * dt
    return v_new, u_new

@cp.fuse()
def fused_hodgkin_huxley_dynamics_update(V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L, temperature_celsius, q10_factor):
    BASE_HH_KINETICS_TEMP_C = 6.3
    phi = q10_factor**((temperature_celsius - BASE_HH_KINETICS_TEMP_C) / 10.0)

    v_plus_40 = V + 40.0
    alpha_m_orig = cp.where(v_plus_40 == 0, 1.0 * 0.1 * 10.0 , -0.1 * v_plus_40 / cp.expm1(-v_plus_40 / 10.0))
    beta_m_orig  = 4.0 * cp.exp(-(V + 65.0) / 18.0)

    alpha_h_orig = 0.07 * cp.exp(-(V + 65.0) / 20.0)
    beta_h_orig  = 1.0 / (cp.exp(-(V + 35.0) / 10.0) + 1.0)

    v_plus_55 = V + 55.0
    alpha_n_orig = cp.where(v_plus_55 == 0, 0.1 * 0.01 * 10.0, -0.01 * v_plus_55 / cp.expm1(-v_plus_55 / 10.0))
    beta_n_orig  = 0.125 * cp.exp(-(V + 65.0) / 80.0)

    alpha_m = alpha_m_orig * phi; beta_m  = beta_m_orig  * phi
    alpha_h = alpha_h_orig * phi; beta_h  = beta_h_orig  * phi
    alpha_n = alpha_n_orig * phi; beta_n  = beta_n_orig  * phi

    sum_alpha_beta_m = alpha_m + beta_m
    m_inf = cp.where(sum_alpha_beta_m == 0, m, alpha_m / sum_alpha_beta_m)
    tau_m = cp.where(sum_alpha_beta_m == 0, cp.inf, 1.0 / sum_alpha_beta_m)
    m_new = m_inf + (m - m_inf) * cp.exp(cp.where(cp.isinf(tau_m), 0.0, -dt / tau_m))

    sum_alpha_beta_h = alpha_h + beta_h
    h_inf = cp.where(sum_alpha_beta_h == 0, h, alpha_h / sum_alpha_beta_h)
    tau_h = cp.where(sum_alpha_beta_h == 0, cp.inf, 1.0 / sum_alpha_beta_h)
    h_new = h_inf + (h - h_inf) * cp.exp(cp.where(cp.isinf(tau_h), 0.0, -dt / tau_h))

    sum_alpha_beta_n = alpha_n + beta_n
    n_inf = cp.where(sum_alpha_beta_n == 0, n, alpha_n / sum_alpha_beta_n)
    tau_n = cp.where(sum_alpha_beta_n == 0, cp.inf, 1.0 / sum_alpha_beta_n)
    n_new = n_inf + (n - n_inf) * cp.exp(cp.where(cp.isinf(tau_n), 0.0, -dt / tau_n))

    m_new = cp.clip(m_new, 0.0, 1.0); h_new = cp.clip(h_new, 0.0, 1.0); n_new = cp.clip(n_new, 0.0, 1.0)

    I_Na = g_Na_max * (m_new**3) * h_new * (V - E_Na)
    I_K  = g_K_max * (n_new**4) * (V - E_K)
    I_L  = g_L * (V - E_L)
    I_ion = I_Na + I_K + I_L

    dV_dt = (I_syn - I_ion) / C_m
    V_new = V + dV_dt * dt
    return V_new, m_new, h_new, n_new

@cp.fuse()
def fused_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i):
    g_e_new = g_e * decay_e
    g_i_new = g_i * decay_i
    I_syn = g_e_new * (E_e - v) + g_i_new * (E_i - v)
    return g_e_new, g_i_new, I_syn

@cp.fuse()
def fused_stp_decay_recovery(u, x, dt, tau_f, tau_d):
    tau_f_safe = cp.maximum(tau_f, 1e-9)
    tau_d_safe = cp.maximum(tau_d, 1e-9)
    u_decayed = u * cp.exp(-dt / tau_f_safe)
    x_recovered_increment = (1.0 - x) * (dt / tau_d_safe)
    x_recovered = x + x_recovered_increment
    x_clipped = cp.clip(x_recovered, 0.0, 1.0)
    return u_decayed, x_clipped

@cp.fuse()
def fused_homeostasis_update(neuron_activity_ema_in, fired_this_step_float, target_rate, alpha_ema, adapt_rate,
                             neuron_firing_thresholds_in, thresh_min, thresh_max):
    new_neuron_activity_ema = (1.0 - alpha_ema) * neuron_activity_ema_in + alpha_ema * fired_this_step_float
    error = new_neuron_activity_ema - target_rate
    threshold_delta = error * adapt_rate
    new_neuron_firing_thresholds = neuron_firing_thresholds_in + threshold_delta
    new_neuron_firing_thresholds_clipped = cp.clip(new_neuron_firing_thresholds, thresh_min, thresh_max)
    return new_neuron_activity_ema, new_neuron_firing_thresholds_clipped


# --- Simulation Bridge (Core Logic) ---
class SimulationBridge:
    def __init__(self, sim_core_ref=None):
        self.sim_core_ref = sim_core_ref
        self.sim_config = SimulationConfiguration()

        # --- CuPy Arrays for Simulation State ---
        self.cp_membrane_potential_v = None
        self.cp_recovery_variable_u = None
        self.cp_conductance_g_e = None
        self.cp_conductance_g_i = None
        self.cp_external_input_current = None
        self.cp_firing_states = None
        self.cp_prev_firing_states = None
        self.cp_traits = None
        self.cp_neuron_positions_3d = None
        self.cp_refractory_timers = None
        self.cp_viz_activity_timers = None

        self.cp_izh_C = None
        self.cp_izh_k = None
        self.cp_izh_vr = None
        self.cp_izh_vt = None
        self.cp_izh_vpeak = None
        self.cp_izh_a = None
        self.cp_izh_b = None
        self.cp_izh_c_reset = None
        self.cp_izh_d_increment = None

        self.cp_izh_legacy_a = None
        self.cp_izh_legacy_b = None
        self.cp_izh_legacy_c_reset = None
        self.cp_izh_legacy_d_increment = None
        self.cp_izh_legacy_vpeak = None

        self.cp_gating_variable_m = None
        self.cp_gating_variable_h = None
        self.cp_gating_variable_n = None

        self.cp_hh_C_m = None
        self.cp_hh_g_Na_max = None
        self.cp_hh_g_K_max = None
        self.cp_hh_g_L = None
        self.cp_hh_E_Na = None
        self.cp_hh_E_K = None
        self.cp_hh_E_L = None
        self.cp_hh_v_peak = None

        self.cp_neuron_firing_thresholds = None
        self.cp_neuron_activity_ema = None

        self.cp_connections = None # This will remain a CuPy sparse matrix

        self.cp_stp_u = None
        self.cp_stp_x = None

        self.cp_synapse_pulse_timers = None
        self.cp_synapse_pulse_progress = None

        self.is_initialized = False

        self._mock_total_plasticity_events = 0
        self._mock_network_avg_firing_rate_hz = 0.0
        self._mock_num_spikes_this_step = 0

        self.PROFILE_DIR = "simulation_profiles/" # Stays .json
        self.CHECKPOINT_DIR = "simulation_checkpoints_h5/" # Changed to _h5
        self.RECORDING_DIR = "simulation_recordings_h5/" # Changed to _h5

        # --- Attributes for Streamed Recording & Playback ---
        self.recording_file_handle = None # Will be an h5py.File object
        self.recording_filepath = None
        self.current_frame_count_for_h5 = 0 # For naming frame groups in HDF5

        for dir_path in [self.PROFILE_DIR, self.CHECKPOINT_DIR, self.RECORDING_DIR]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    print(f"[INFO] SIM_BRIDGE_INIT: Created directory: {dir_path}")
                except OSError as e:
                    print(f"[ERROR] SIM_BRIDGE_INIT: Error creating directory {dir_path}: {e}")
        try:
             cp.cuda.Device(0).use()
             dev_props = cp.cuda.runtime.getDeviceProperties(0)
             gpu_name = dev_props.get('name',b'Unknown').decode()
             self.log_message(f"CuPy using GPU: {gpu_name}")
        except Exception as e:
             self.log_message(f"Error setting CuPy device: {e}", "critical")

    def log_message(self, message, level="info"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}][{level.upper()}] SIM_BRIDGE: {message}")

    def _initialize_simulation_data(self, called_from_playback_init=False):
        self.log_message(f"Initializing simulation data for model: {self.sim_config.neuron_model_type} (3D)... (called_from_playback_init: {called_from_playback_init})")

        if not called_from_playback_init:
            if global_gui_state.get("is_recording", False):
                self.log_message("Sim re-initialization (non-playback): Stopping active file recording.", "warning")
                self.stop_recording(prompt_save=False)

            if global_gui_state.get("is_playback_mode", False):
                self.log_message("_initialize_simulation_data: External call while in playback. Forcing exit.", "critical")
                self.exit_playback_mode()

        try:
            n = self.sim_config.num_neurons
            cfg = self.sim_config
            if n <= 0:
                self.log_message(f"Number of neurons ({n}) must be positive. Initialization failed.", "warning")
                self.is_initialized = False; return

            if cfg.seed != -1:
                cp.random.seed(cfg.seed)
                np.random.seed(cfg.seed)

            self.cp_external_input_current = cp.zeros(n, dtype=cp.float32)
            self.cp_firing_states = cp.zeros(n, dtype=bool)
            self.cp_prev_firing_states = cp.zeros(n, dtype=bool)
            # Corrected: Pass size as a tuple (n,) for cp.random.randint
            self.cp_traits = cp.random.randint(0, max(1, cfg.num_traits), (n,), dtype=cp.int32) if n > 0 else cp.array([], dtype=cp.int32)
            self.cp_conductance_g_e = cp.zeros(n, dtype=cp.float32)
            self.cp_conductance_g_i = cp.zeros(n, dtype=cp.float32)
            self.cp_refractory_timers = cp.zeros(n, dtype=cp.int32)
            self.cp_neuron_activity_ema = cp.zeros(n, dtype=cp.float32)
            self.cp_viz_activity_timers = cp.zeros(n, dtype=cp.int32)

            self.cp_synapse_pulse_timers = cp.array([], dtype=cp.int32)
            self.cp_synapse_pulse_progress = cp.array([], dtype=cp.float32)

            cfg.neuron_types_list_for_viz = [""] * n

            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                self.log_message(f"Initializing Izhikevich model specifics for {n} neurons...")
                self.cp_izh_C = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_k = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vr = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vt = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vpeak = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_a = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_b = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_c_reset = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_d_increment = cp.zeros(n, dtype=cp.float32)
                self.cp_membrane_potential_v = cp.zeros(n, dtype=cp.float32)
                self.cp_recovery_variable_u = cp.zeros(n, dtype=cp.float32)

                thresh_base = (cfg.homeostasis_threshold_min + cfg.homeostasis_threshold_max) / 2.0
                thresh_var = (cfg.homeostasis_threshold_max - cfg.homeostasis_threshold_min) / 2.0
                if thresh_var < 0: thresh_var = 1.0
                self.cp_neuron_firing_thresholds = cp.random.uniform(
                    thresh_base - thresh_var, thresh_base + thresh_var, n # size n is acceptable for uniform
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

                    params = DefaultIzhikevichParamsManager.get_params(selected_neuron_type_enum, use_2007_formulation=True)
                    self.cp_izh_C[i] = params["C"]; self.cp_izh_k[i] = params["k"]
                    self.cp_izh_vr[i] = params["vr"]; self.cp_izh_vt[i] = params["vt"]
                    self.cp_izh_vpeak[i] = params["vpeak"]; self.cp_izh_a[i] = params["a"]
                    self.cp_izh_b[i] = params["b"]; self.cp_izh_c_reset[i] = params["c_reset"]
                    self.cp_izh_d_increment[i] = params["d_increment"]
                    self.cp_membrane_potential_v[i] = params["vr"]
                    self.cp_recovery_variable_u[i] = params["b"] * (self.cp_membrane_potential_v[i] - params["vr"])
                    cfg.neuron_types_list_for_viz[i] = f"Izh2007_{selected_neuron_type_enum.name.replace('IZH2007_', '')}"

            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                self.log_message(f"Initializing Hodgkin-Huxley model specifics for {n} neurons...")
                self.cp_hh_C_m = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_g_Na_max = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_g_K_max = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_g_L = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_Na = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_K = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_L = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_v_peak = cp.zeros(n, dtype=cp.float32)
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

                    params = DefaultHodgkinHuxleyParams.get_params(selected_neuron_type_enum)
                    self.cp_hh_C_m[i] = params["C_m"]; self.cp_hh_g_Na_max[i] = params["g_Na_max"]
                    self.cp_hh_g_K_max[i] = params["g_K_max"]; self.cp_hh_g_L[i] = params["g_L"]
                    self.cp_hh_E_Na[i] = params["E_Na"]; self.cp_hh_E_K[i] = params["E_K"]
                    self.cp_hh_E_L[i] = params["E_L"]; self.cp_hh_v_peak[i] = params["v_peak_hh"]
                    self.cp_membrane_potential_v[i] = params["v_rest_hh"]
                    self.cp_gating_variable_m[i] = params["m_init"]
                    self.cp_gating_variable_h[i] = params["h_init"]
                    self.cp_gating_variable_n[i] = params["n_init"]
                    cfg.neuron_types_list_for_viz[i] = f"HH_{selected_neuron_type_enum.name.replace('HH_', '')}"

            self.log_message(f"Generating 3D neuron positions for {n} neurons...")
            if n > 0:
                np_positions_3d = np.random.uniform(
                    low=[cfg.volume_min_x, cfg.volume_min_y, cfg.volume_min_z],
                    high=[cfg.volume_max_x, cfg.volume_max_y, cfg.volume_max_z],
                    size=(n,3)).astype(np.float32)
                self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                cfg.neuron_positions_x = np_positions_3d[:,0].tolist()
                cfg.neuron_positions_y = np_positions_3d[:,1].tolist()
            else:
                self.cp_neuron_positions_3d = cp.array([], dtype=cp.float32).reshape(0,3)
                cfg.neuron_positions_x = []; cfg.neuron_positions_y = []

            if not called_from_playback_init:
                self.log_message("Generating connections (3D)...")
                if cfg.enable_watts_strogatz:
                    self.cp_connections = self._generate_watts_strogatz_connections_3d(n, cfg.connectivity_k, cfg.connectivity_p_rewire, cfg)
                else:
                    self.cp_connections = self._generate_spatial_connections_3d(n, cfg.connections_per_neuron, self.cp_neuron_positions_3d, self.cp_traits, cfg)

                if self.cp_connections is None: # Should not happen if n > 0 and generation functions are robust
                    self.log_message("Connection generation resulted in None. Ensuring empty matrix for n=0.", "warning")
                    self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32)
            elif self.cp_connections is None: # During playback init, connections should be loaded by _apply_recorded_arrays
                 self.log_message("Warning: Connections are None during playback init before _apply_recorded_arrays. Initializing empty.", "warning")
                 self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32)


            if self.cp_connections is not None and self.cp_connections.nnz > 0:
                num_synapses = self.cp_connections.nnz
                self.cp_synapse_pulse_timers = cp.zeros(num_synapses, dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.zeros(num_synapses, dtype=cp.float32)
            else:
                self.cp_synapse_pulse_timers = cp.array([], dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.array([], dtype=cp.float32)

            num_synapses_for_stp = self.cp_connections.nnz if self.cp_connections is not None else 0
            if cfg.enable_short_term_plasticity and num_synapses_for_stp > 0:
                self.log_message(f"Initializing STP state for {num_synapses_for_stp} synapses...")
                self.cp_stp_x = cp.ones(num_synapses_for_stp, dtype=cp.float32)
                self.cp_stp_u = cp.full(num_synapses_for_stp, cfg.stp_U, dtype=cp.float32)
            else:
                self.cp_stp_x = None; self.cp_stp_u = None

            self.is_initialized = True
            conn_count = self.cp_connections.nnz if self.cp_connections is not None else 0
            self.log_message(f"Simulation data initialized for {n} neurons (3D). Connections: {conn_count}")
        except Exception as e:
            self.log_message(f"Error during simulation data initialization (3D): {e}","critical")
            import traceback; traceback.print_exc()
            self.is_initialized = False
            if 'cupy' in sys.modules and cp.is_available():
                cp.get_default_memory_pool().free_all_blocks()

    def _calculate_distances_3d_gpu(self, pos_i_cp, pos_neighbors_cp):
        if pos_neighbors_cp.size == 0: return cp.array([], dtype=cp.float32)
        diff_3d = pos_neighbors_cp - pos_i_cp.reshape(1, 3)
        return cp.sqrt(cp.sum(diff_3d**2, axis=1))

    def _generate_spatial_connections_3d(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        self.log_message("Generating connections (3D spatial)..."); start_t = time.time()
        if n == 0: self.log_message("No neurons to connect (n=0).", "info"); return csp.csr_matrix((0,0), dtype=cp.float32)

        dist_decay_factor = getattr(config, 'connection_distance_decay_factor', 0.01)
        trait_bias = getattr(config, 'trait_connection_bias', 0.5)
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight

        rows, cols, weights_list = [], [], []

        for i in range(n):
            pos_i_cp = neuron_positions_3d_cp[i:i+1, :] # Keep as 2D for broadcasting
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
            # Normalize probabilities if sum is not zero
            sum_probs = cp.sum(connection_probabilities_cp)
            if sum_probs > 1e-9: # Avoid division by zero
                 normalized_probabilities_cp = connection_probabilities_cp / sum_probs
            else: # If all probabilities are zero (e.g., all candidates too far), skip or handle
                 # For now, if sum_probs is effectively zero, choice will likely fail or pick uniformly if p is None.
                 # We can explicitly skip or choose randomly without p if that's desired.
                 # If we proceed with zero probabilities, cp.random.choice will raise error if p doesn't sum to 1.
                 # So, if sum_probs is zero, we might want to select a few random candidates without probability weighting.
                 if connection_probabilities_cp.size > 0: # Check if there are candidates at all
                    normalized_probabilities_cp = cp.ones_like(connection_probabilities_cp) / connection_probabilities_cp.size # Uniform
                 else:
                    continue # No candidates, skip

            num_potential_targets = candidate_indices_cp.size
            if num_potential_targets > 0 :
                num_to_select = min(max_connections_per_neuron, num_potential_targets)

                if num_to_select > 0:
                    try:
                        # Ensure probabilities sum to 1 for cp.random.choice
                        if not np.isclose(cp.asnumpy(cp.sum(normalized_probabilities_cp)), 1.0) and cp.sum(normalized_probabilities_cp) > 1e-9:
                            normalized_probabilities_cp = normalized_probabilities_cp / cp.sum(normalized_probabilities_cp)
                        elif cp.sum(normalized_probabilities_cp) <= 1e-9: # All probs are zero
                             # Fallback to uniform random choice if all probabilities are zero
                             selected_local_indices_cp = cp.random.choice(cp.arange(num_potential_targets), size=num_to_select, replace=False)
                        else:
                             selected_local_indices_cp = cp.random.choice(
                                cp.arange(num_potential_targets),
                                size=num_to_select,
                                replace=False,
                                p=normalized_probabilities_cp
                            )
                    except (ValueError, NotImplementedError) as e: # Fallback if probabilities are tricky or choice fails
                        # print(f"Warning: cp.random.choice failed with p (sum={cp.sum(normalized_probabilities_cp)}): {e}. Using argsort fallback.")
                        sorted_local_indices_cp = cp.argsort(connection_probabilities_cp)[::-1] # Sort by original probs
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
            self.log_message("No connections generated by 3D spatial method.", "warning")
            return csp.csr_matrix((n, n), dtype=cp.float32)

        conn_matrix = csp.csr_matrix((cp.asarray(weights_list,dtype=cp.float32),
                                      (cp.asarray(rows,dtype=cp.int32),cp.asarray(cols,dtype=cp.int32))),
                                     shape=(n,n),dtype=cp.float32)
        conn_matrix.sort_indices()
        self.log_message(f"Connections (3D Spatial): {conn_matrix.nnz}. Time: {time.time()-start_t:.2f}s")
        return conn_matrix

    def _generate_watts_strogatz_connections_3d(self, n, k_neighbors, p_rewire, config):
        self.log_message("Watts-Strogatz 3D generation fallback to 3D spatial.", "warning")
        return self._generate_spatial_connections_3d(n, config.connections_per_neuron, self.cp_neuron_positions_3d, self.cp_traits, config)

    def apply_simulation_configuration_core(self, sim_cfg_dict, is_part_of_playback_setup=False):
        self.log_message(f"Applying new simulation configuration... (is_part_of_playback_setup: {is_part_of_playback_setup})")

        if self.sim_config.is_running:
            self.stop_simulation()

        if not is_part_of_playback_setup:
            if global_gui_state.get("is_recording", False):
                self.log_message("Config change: Stopping active file recording.", "info")
                self.stop_recording(prompt_save=True)

            if global_gui_state.get("is_playback_mode", False):
                self.log_message("Config change is forcing exit from playback mode.", "info")
                self.exit_playback_mode()
                pass 

        self.clear_simulation_state_and_gpu_memory()
        self.sim_config = SimulationConfiguration.from_dict(sim_cfg_dict)

        self._initialize_simulation_data(called_from_playback_init=is_part_of_playback_setup)

        if not self.is_initialized:
            self.log_message("Failed to initialize simulation from new configuration. Critical error.", "critical")
            return False

        self.sim_config.reset_simulation_time_and_counters()
        self.log_message(f"Simulation configuration applied ({self.sim_config.neuron_model_type}, N={self.sim_config.num_neurons}). Sim re-initialized.")
        return True

    def get_current_simulation_configuration_dict(self):
        return self.sim_config.to_dict()

    def clear_simulation_state_and_gpu_memory(self):
        self.log_message("Clearing simulation state and GPU memory...")
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
                self.log_message(f"Error freeing CuPy memory: {e}", "warning")

        self.is_initialized = False
        self.log_message("Cleared simulation state and GPU memory.")

    def start_simulation(self):
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot start simulation during playback mode.", "warning"); return

        if not self.is_initialized:
            self.log_message("Simulation not initialized. Attempting to initialize with current config.", "warning")
            if not self.apply_simulation_configuration_core(self.sim_config.to_dict()):
                self.log_message("Initialization failed. Cannot start simulation.", "error"); return

        self.sim_config.reset_simulation_time_and_counters()

        if self.recording_file_handle and not global_gui_state.get("is_recording", False):
            self.log_message("Warning: An old recording file handle was unexpectedly open. Closing it.", "warning")
            try:
                if isinstance(self.recording_file_handle, h5py.File) and self.recording_file_handle.id:
                    self.recording_file_handle.close()
            except Exception as e:
                self.log_message(f"Error closing unexpected old file handle: {e}", "error")
            self.recording_file_handle = None
            self.recording_filepath = None


        self.sim_config.is_running = True; self.sim_config.is_paused = False
        self.log_message(f"Simulation started. Duration: {self.sim_config.total_simulation_time_ms} ms, Model: {self.sim_config.neuron_model_type}, dt: {self.sim_config.dt_ms} ms.")
        if global_gui_state.get("is_recording", False):
            self.log_message(f"Recording active, streaming to file: {self.recording_filepath}")

    def stop_simulation(self):
        if self.sim_config.is_running or self.sim_config.is_paused:
            self.sim_config.is_running = False; self.sim_config.is_paused = False
            self.log_message("Simulation stopped.")
            if global_gui_state.get("is_recording", False):
                self.log_message("Recording paused (simulation stopped). Finalize recording from UI or continue sim.", "info")
                if dpg.is_dearpygui_running() and dpg.does_item_exist("apply_config_status_text"):
                     if global_gui_state.get("unsaved_recording_exists", False): # unsaved_recording_exists now means file is open
                        dpg.set_value("apply_config_status_text", "Sim stopped. Recording to file is active.")

    def pause_simulation(self):
        if self.sim_config.is_running and not self.sim_config.is_paused:
            self.sim_config.is_paused = True; self.log_message("Simulation paused.")
            if global_gui_state.get("is_recording", False):
                self.log_message("Recording paused (simulation paused).", "info")

    def resume_simulation(self):
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot resume simulation during playback mode.", "warning"); return
        if self.sim_config.is_running and self.sim_config.is_paused:
            self.sim_config.is_paused = False; self.log_message("Simulation resumed.")
            if global_gui_state.get("is_recording", False):
                self.log_message("Recording resumed (simulation resumed).", "info")

    def toggle_pause_simulation(self):
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot toggle simulation pause during playback mode.", "warning")
            return self.sim_config.is_paused

        if not self.sim_config.is_running:
            self.log_message("Cannot toggle pause: Simulation is not running.", "warning"); return self.sim_config.is_paused

        self.sim_config.is_paused = not self.sim_config.is_paused
        self.log_message(f"Simulation {'paused' if self.sim_config.is_paused else 'resumed'}.")
        if global_gui_state.get("is_recording", False):
            self.log_message(f"Recording {'paused' if self.sim_config.is_paused else 'resumed'}.", "info")
        return self.sim_config.is_paused

    def set_simulation_speed_factor(self, factor):
        self.sim_config.simulation_speed_factor = max(0.01, factor)

    def step_simulation(self, num_steps=1):
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot step simulation during playback mode.", "warning"); return

        if not self.is_initialized:
            self.log_message("Cannot step: Sim not initialized.", "warning"); return
        if self.sim_config.is_running and not self.sim_config.is_paused:
            self.log_message("Cannot step: Sim running. Pause first.", "warning"); return

        self.log_message(f"Stepping simulation by {num_steps} steps.")
        for _ in range(num_steps):
            if self.sim_config.num_neurons > 0:
                self._run_one_simulation_step()
                self.sim_config.current_time_ms += self.sim_config.dt_ms
                self.sim_config.current_time_step += 1
            else:
                self.log_message("No neurons to simulate in step.", "debug"); break

    def _capture_initial_state_for_recording(self):
        if not self.is_initialized:
            self.log_message("Cannot capture initial state: Simulation not initialized.", "error")
            return None

        snapshot = {
            "start_time_ms": self.sim_config.current_time_ms,
            "start_time_step": self.sim_config.current_time_step
            # Scalar values like these will be stored as attributes in HDF5
        }

        # These will become datasets in HDF5
        if self.cp_traits is not None: snapshot["cp_traits"] = cp.asnumpy(self.cp_traits)
        if self.cp_neuron_positions_3d is not None: snapshot["cp_neuron_positions_3d"] = cp.asnumpy(self.cp_neuron_positions_3d)

        if self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                attr_name = f"cp_izh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = cp.asnumpy(getattr(self, attr_name))
        elif self.sim_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
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
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                snapshot[attr_name] = cp.asnumpy(getattr(self, attr_name))
            else:
                snapshot[attr_name] = None # Will be handled during HDF5 saving


        if self.cp_connections is not None:
            # For HDF5, we'll store these components separately
            snapshot["connections_data"] = cp.asnumpy(self.cp_connections.data) if self.cp_connections.data is not None else np.array([])
            snapshot["connections_indices"] = cp.asnumpy(self.cp_connections.indices) if self.cp_connections.indices is not None else np.array([])
            snapshot["connections_indptr"] = cp.asnumpy(self.cp_connections.indptr) if self.cp_connections.indptr is not None else np.array([])
            snapshot["connections_shape"] = self.cp_connections.shape # Tuple, will be stored as attributes

        if self.cp_stp_u is not None: snapshot["cp_stp_u"] = cp.asnumpy(self.cp_stp_u)
        if self.cp_stp_x is not None: snapshot["cp_stp_x"] = cp.asnumpy(self.cp_stp_x)

        return snapshot

    def start_recording_to_file(self, filepath):
        if global_gui_state.get("is_recording", False):
            self.log_message("Already recording. Stop current one first.", "warning")
            return False
        if not self.is_initialized:
            self.log_message("Simulation not initialized. Cannot start recording.", "warning")
            if dpg.is_dearpygui_running(): update_status_bar("Error: Sim not initialized for recording.")
            return False
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot start recording during playback mode.", "warning")
            if dpg.is_dearpygui_running(): update_status_bar("Error: Cannot record in playback mode.")
            return False

        if self.recording_file_handle: # Check if it's an h5py.File and try to close
            try:
                if isinstance(self.recording_file_handle, h5py.File):
                    self.log_message(f"Warning: Previous recording file {self.recording_filepath} was still open. Closing it.", "warning")
                    self.recording_file_handle.close()
            except Exception as e_close:
                self.log_message(f"Error closing previous recording file: {e_close}", "error")
        self.recording_file_handle = None
        self.recording_filepath = None
        self.current_frame_count_for_h5 = 0


        self.log_message(f"Attempting to start new recording to file: {filepath}")
        try:
            self.recording_filepath = filepath
            self.recording_file_handle = h5py.File(self.recording_filepath, 'w') # Open HDF5 file in write mode

            # 1. Write format version as a root attribute
            self.recording_file_handle.attrs["format_version"] = RECORDING_FORMAT_VERSION

            # 2. Write simulation configuration snapshot as root attributes
            config_snapshot = self.sim_config.to_dict()
            save_dict_to_hdf5_attrs(self.recording_file_handle, config_snapshot) # Using the helper

            # 3. Write initial full state into a group
            initial_state_data = self._capture_initial_state_for_recording()
            if initial_state_data is None:
                self.log_message("Failed to capture initial state for recording. Aborting.", "error")
                self.recording_file_handle.close()
                self.recording_file_handle = None
                self.recording_filepath = None
                if dpg.is_dearpygui_running(): update_status_bar("Error: Failed to capture initial state for recording.")
                return False

            initial_state_group = self.recording_file_handle.create_group("initial_state")
            for key, value in initial_state_data.items():
                if isinstance(value, np.ndarray):
                    if value.size > 0 : # Only save non-empty arrays
                        initial_state_group.create_dataset(key, data=value)
                    else: # Store a marker for empty arrays if needed, or just skip
                        initial_state_group.attrs[f"{key}_is_empty"] = True

                elif key == "connections_shape": # Store shape as attributes
                    initial_state_group.attrs["connections_shape_0"] = value[0]
                    initial_state_group.attrs["connections_shape_1"] = value[1]
                elif value is not None : # Store scalar attributes
                    initial_state_group.attrs[key] = value
                else: # value is None
                    initial_state_group.attrs[key] = "NoneType"


            # Create a group for frames
            self.recording_file_handle.create_group("frames")
            self.current_frame_count_for_h5 = 0

            global_gui_state["is_recording"] = True
            global_gui_state["unsaved_recording_exists"] = True
            self.log_message(f"Recording armed. Streaming to HDF5: {self.recording_filepath}. Start simulation to capture frames.")
            if dpg.is_dearpygui_running():
                update_ui_for_recording_state(is_recording=True)
                update_status_bar(f"Recording to {os.path.basename(filepath)}. Run sim to capture.", color=[0,150,200])
            return True

        except Exception as e:
            self.log_message(f"Error starting file recording to {filepath}: {e}", "error")
            if self.recording_file_handle:
                try: self.recording_file_handle.close()
                except: pass
            self.recording_file_handle = None
            self.recording_filepath = None
            global_gui_state["is_recording"] = False
            global_gui_state["unsaved_recording_exists"] = False
            if dpg.is_dearpygui_running():
                update_ui_for_recording_state(is_recording=False)
                update_status_bar(f"Error starting recording: {e}", color=[255,0,0])
            return False

    def stop_recording(self, prompt_save=True):
        if not global_gui_state.get("is_recording", False):
            return

        self.log_message("Stopping HDF5 recording stream.")
        was_recording_to_file = False
        finalized_filepath = self.recording_filepath

        if self.recording_file_handle and isinstance(self.recording_file_handle, h5py.File) and self.recording_file_handle.id:
            try:
                self.recording_file_handle.close()
                was_recording_to_file = True
                self.log_message(f"Recording stream to {finalized_filepath} finalized.")
            except Exception as e:
                self.log_message(f"Error finalizing recording file {finalized_filepath}: {e}", "error")
        else:
            self.log_message(f"Stop recording called, but no active file handle or already closed for {finalized_filepath}.", "warning")

        self.recording_file_handle = None
        self.recording_filepath = None
        self.current_frame_count_for_h5 = 0

        global_gui_state["is_recording"] = False
        global_gui_state["unsaved_recording_exists"] = False

        if dpg.is_dearpygui_running():
            update_ui_for_recording_state(is_recording=False) # Update record button label
            if was_recording_to_file and finalized_filepath and os.path.exists(finalized_filepath):
                update_status_bar(f"Recording saved: {os.path.basename(finalized_filepath)}. Preparing for view...", color=[0,200,0])
                # Prepare metadata for the just-saved recording
                if self._prepare_loaded_recording_metadata(finalized_filepath):
                    # Now also apply its initial state for immediate viewing
                    if self._apply_config_and_initial_state_from_recording(global_gui_state["loaded_recording_data"]):
                        update_status_bar(f"Recording '{os.path.basename(finalized_filepath)}' loaded and initialized.", color=[0,200,0])
                        update_ui_for_loaded_recording(can_playback=True) # Show playback controls
                    else:
                        update_status_bar(f"Recording saved, but error initializing view: {os.path.basename(finalized_filepath)}", color=[255,100,0])
                        update_ui_for_loaded_recording(can_playback=False)
                else: 
                     update_status_bar(f"Recording saved, but error preparing metadata: {os.path.basename(finalized_filepath)}", color=[255,100,0])
                     update_ui_for_loaded_recording(can_playback=False)
            elif was_recording_to_file and finalized_filepath:
                 update_status_bar(f"Recording finalized, but file {os.path.basename(finalized_filepath)} not found.", color=[255,0,0])
                 update_ui_for_loaded_recording(can_playback=False)
            else:
                update_status_bar("Recording stopped.")

    def record_current_frame_if_active(self):
        if not global_gui_state.get("is_recording", False) or \
           not self.sim_config.is_running or \
           self.sim_config.is_paused or \
           self.recording_file_handle is None or \
           not isinstance(self.recording_file_handle, h5py.File) or \
           not self.recording_file_handle.id: # Check if file is open
            return

        try:
            frame_data_np = {
                # Scalar values will be attributes of the frame group
                "time_ms": self.sim_config.current_time_ms,
                "step": self.sim_config.current_time_step,
                "_mock_num_spikes_this_step": self._mock_num_spikes_this_step,
                "_mock_network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
                "_mock_total_plasticity_events": self._mock_total_plasticity_events
            }

            # NumPy arrays will be datasets within the frame group
            dynamic_arrays_to_capture = [
                'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
                'cp_conductance_g_e', 'cp_conductance_g_i',
                'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
            ]
            if self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                dynamic_arrays_to_capture.extend(['cp_recovery_variable_u'])
                if self.sim_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                    dynamic_arrays_to_capture.append('cp_neuron_firing_thresholds')
            elif self.sim_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                dynamic_arrays_to_capture.extend(['cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n'])

            if self.sim_config.enable_hebbian_learning and self.cp_connections is not None:
                if self.cp_connections.data is not None:
                     # Storing full connection data per frame can be large.
                     # Consider only if weights change frequently and are critical per frame.
                     # For now, let's assume it's needed if Hebbian learning is on.
                     frame_data_np["cp_connections_data"] = cp.asnumpy(self.cp_connections.data)

            if self.sim_config.enable_short_term_plasticity:
                if self.cp_stp_u is not None: frame_data_np["cp_stp_u"] = cp.asnumpy(self.cp_stp_u)
                if self.cp_stp_x is not None: frame_data_np["cp_stp_x"] = cp.asnumpy(self.cp_stp_x)

            for attr_name in dynamic_arrays_to_capture:
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    frame_data_np[attr_name] = cp.asnumpy(getattr(self, attr_name))
                else:
                    frame_data_np[attr_name] = None # Will be handled during HDF5 saving

            # Create a new group for the current frame
            frame_group_name = f"frames/frame_{self.current_frame_count_for_h5}"
            current_frame_group = self.recording_file_handle.create_group(frame_group_name)

            for key, value in frame_data_np.items():
                if isinstance(value, np.ndarray):
                    if value.size > 0: # Only save non-empty arrays
                        current_frame_group.create_dataset(key, data=value)
                    else:
                        current_frame_group.attrs[f"{key}_is_empty"] = True
                elif value is not None: # Scalar value, store as attribute
                    current_frame_group.attrs[key] = value
                else: # value is None
                    current_frame_group.attrs[key] = "NoneType"


            self.current_frame_count_for_h5 += 1
            self.recording_file_handle.flush() # Ensure data is written

        except Exception as e:
            self.log_message(f"Error streaming frame to recording file {self.recording_filepath}: {e}", "error")
            self.stop_recording(prompt_save=False) # Stop recording on stream error

    def _apply_config_and_initial_state_from_recording(self, loaded_data_meta):
        """
        Applies the configuration and initial state from loaded recording metadata.
        Sets the system to be in playback mode.
        """
        if not loaded_data_meta:
            self.log_message("Cannot apply config and initial state: No loaded data metadata.", "error")
            return False

        config_snapshot = loaded_data_meta.get("config_snapshot")
        if not config_snapshot:
            self.log_message("Error: Recording metadata missing simulation_config_snapshot.", "error")
            return False

        success_apply_config = self.apply_simulation_configuration_core(config_snapshot, is_part_of_playback_setup=True)
        if not success_apply_config or not self.is_initialized:
            self.log_message("CRITICAL: Failed to apply recorded config or initialize from recording data.", "error")
            self.apply_simulation_configuration_core(SimulationConfiguration().to_dict(), is_part_of_playback_setup=False) 
            return False
        self.log_message(f"Applied recording config. Neuron count now: {self.sim_config.num_neurons}")

        initial_state = loaded_data_meta.get("initial_full_state")
        if not initial_state:
            self.log_message("Error: Recording metadata missing initial_full_state.", "error")
            return False

        self._apply_recorded_arrays_to_gpu(initial_state, is_initial_state=True)
        self.log_message("Applied initial full state from recording.")

        self.sim_config.current_time_ms = initial_state.get("start_time_ms", 0.0)
        self.sim_config.current_time_step = initial_state.get("start_time_step", 0)

        # --- Key changes for immediate playback readiness ---
        global_gui_state["is_playback_mode"] = True # Enter playback mode conceptually
        global_gui_state["active_recording_data_source"] = loaded_data_meta # Ensure this is set
        global_gui_state["current_playback_frame_index"] = 0
        global_gui_state["playback_is_playing"] = False # Start paused

        num_frames = loaded_data_meta.get("num_frames", 0)

        if dpg.is_dearpygui_running():
            # Update main playback button and disable conflicting UI
            update_ui_for_playback_state(is_playback=True, num_frames=num_frames) 
            # Ensure playback controls group is visible and slider is set
            update_ui_for_loaded_recording(can_playback=True) 

            if dpg.does_item_exist("playback_slider"):
                dpg.set_value("playback_slider", 0)
                slider_max = max(0, num_frames - 1)
                dpg.configure_item("playback_slider", max_value=slider_max)
            if dpg.does_item_exist("playback_current_frame_text"):
                 dpg.set_value("playback_current_frame_text", f"Frame: 1 / {num_frames if num_frames > 0 else 1}")

        if OPENGL_AVAILABLE and glut.glutGetWindow() != 0:
            trigger_filter_update_signal()
            glut.glutPostRedisplay()

        latest_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
        if latest_data:
            update_monitoring_overlay_values(latest_data)

        return True

    def _run_one_simulation_step(self):
        if not self.is_initialized or self.sim_config.num_neurons == 0: return
        try:
            n_neurons = self.sim_config.num_neurons; cfg = self.sim_config; dt = cfg.dt_ms

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
                    g_e_increase = (effective_connections_matrix.T @ exc_fired_prev) * cfg.propagation_strength
                    self.cp_conductance_g_e += g_e_increase

            total_input_current_pA = synaptic_current_I_syn_pA + self.cp_external_input_current
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
                total_input_current_uA_density_equivalent = total_input_current_pA * 1e-6 # pA to uA/cm^2 (assuming cm^2 area for density)
                v_new, m_new, h_new, n_new = fused_hodgkin_huxley_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_gating_variable_m, self.cp_gating_variable_h, self.cp_gating_variable_n,
                    total_input_current_uA_density_equivalent, dt,
                    self.cp_hh_C_m, self.cp_hh_g_Na_max, self.cp_hh_g_K_max, self.cp_hh_g_L,
                    self.cp_hh_E_Na, self.cp_hh_E_K, self.cp_hh_E_L,
                    cfg.hh_temperature_celsius, cfg.hh_q10_factor
                )
                fired_this_step = (self.cp_membrane_potential_v < self.cp_hh_v_peak) & (v_new >= self.cp_hh_v_peak) # Crossing threshold upwards
                self.cp_membrane_potential_v[:] = v_new
                self.cp_gating_variable_m[:] = m_new
                self.cp_gating_variable_h[:] = h_new
                self.cp_gating_variable_n[:] = n_new

            self.cp_firing_states[:] = fired_this_step
            self._mock_num_spikes_this_step = int(cp.sum(fired_this_step).get())

            if self.cp_viz_activity_timers is not None:
                max_highlight_val = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
                self.cp_viz_activity_timers = cp.where(fired_this_step,
                                                       max_highlight_val,
                                                       self.cp_viz_activity_timers)

            if opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
               self.cp_synapse_pulse_timers is not None and fired_this_step.any():
                if self.cp_connections is not None and self.cp_connections.nnz > 0:
                    coo_matrix_for_pulses = self.cp_connections.tocoo(copy=False)
                    presynaptic_fired_mask_for_pulses = fired_this_step[coo_matrix_for_pulses.row]
                    synapses_to_activate_indices = cp.where(presynaptic_fired_mask_for_pulses)[0]

                    if synapses_to_activate_indices.size > 0:
                        pulse_lifetime = opengl_viz_config.get("SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES", 5)
                        self.cp_synapse_pulse_timers[synapses_to_activate_indices] = pulse_lifetime
                        self.cp_synapse_pulse_progress[synapses_to_activate_indices] = 0.0


            if cfg.enable_hebbian_learning and self.cp_connections.nnz > 0 and \
               self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                if self.cp_prev_firing_states.any() and fired_this_step.any():
                    coo_matrix_heb = self.cp_connections.tocoo(copy=False) # Avoid copying if possible
                    pre_fired_mask_heb = self.cp_prev_firing_states[coo_matrix_heb.row]
                    post_fired_mask_heb = fired_this_step[coo_matrix_heb.col]
                    active_synapse_indices_heb = cp.where(pre_fired_mask_heb & post_fired_mask_heb)[0]
                    num_potentiation_events = 0
                    if active_synapse_indices_heb.size > 0:
                        base_weights_data_array = self.cp_connections.data # Direct reference
                        current_weights_active_syn = base_weights_data_array[active_synapse_indices_heb]
                        delta_weights = cfg.hebbian_learning_rate * (cfg.hebbian_max_weight - current_weights_active_syn)
                        base_weights_data_array[active_synapse_indices_heb] += delta_weights
                        num_potentiation_events = active_synapse_indices_heb.size

                    self.cp_connections.data *= (1.0 - cfg.hebbian_weight_decay) # Apply decay
                    cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=self.cp_connections.data) # Clip
                    if num_potentiation_events > 0: self._mock_total_plasticity_events += num_potentiation_events

            if cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name: # Homeostasis primarily for Izhikevich in this model
                    self.cp_neuron_activity_ema, self.cp_neuron_firing_thresholds = fused_homeostasis_update(
                        self.cp_neuron_activity_ema, fired_this_step.astype(cp.float32),
                        cfg.homeostasis_target_rate, cfg.homeostasis_ema_alpha, cfg.homeostasis_threshold_adapt_rate,
                        self.cp_neuron_firing_thresholds,
                        cfg.homeostasis_threshold_min, cfg.homeostasis_threshold_max
                    )
                elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name: # Simpler EMA update for HH if needed
                     self.cp_neuron_activity_ema = (1.0 - cfg.homeostasis_ema_alpha) * self.cp_neuron_activity_ema + \
                                               cfg.homeostasis_ema_alpha * fired_this_step.astype(cp.float32)


            self.cp_prev_firing_states[:] = fired_this_step
            self.record_current_frame_if_active() # Streams frame to HDF5 file if recording

            if n_neurons > 0 and dt > 0:
                instantaneous_rate_hz = (self._mock_num_spikes_this_step / n_neurons) / (dt / 1000.0)
                self._mock_network_avg_firing_rate_hz = self._mock_network_avg_firing_rate_hz * 0.95 + instantaneous_rate_hz * 0.05
            else: self._mock_network_avg_firing_rate_hz = 0.0

        except Exception as e:
            self.log_message(f"Error during simulation step: {e}","critical")
            import traceback; traceback.print_exc(); self.stop_simulation()

    # --- Playback Methods (HDF5) ---
    def _prepare_loaded_recording_metadata(self, filepath):
        self.log_message(f"Preparing metadata for recording file: {filepath}")
        try:
            h5_file = h5py.File(filepath, 'r')

            # 1. Read format version
            version_info_str = h5_file.attrs.get("format_version")
            if version_info_str != RECORDING_FORMAT_VERSION:
                self.log_message(f"Invalid or outdated recording file format. Expected {RECORDING_FORMAT_VERSION}, got {version_info_str}.", "error")
                h5_file.close()
                return False

            # 2. Read simulation configuration snapshot
            config_snapshot = load_dict_from_hdf5_attrs(h5_file) # Load from root attributes

            # 3. Read initial full state
            initial_full_state = {}
            initial_state_group = h5_file.get("initial_state")
            if not initial_state_group:
                self.log_message("Invalid recording: 'initial_state' group missing.", "error"); h5_file.close(); return False

            for key in initial_state_group.attrs.keys(): # Load attributes first (scalars, None markers, shapes)
                if key.endswith("_is_empty") and initial_state_group.attrs[key] is True:
                    original_key = key.replace("_is_empty","")
                    initial_full_state[original_key] = np.array([])
                elif initial_state_group.attrs[key] == "NoneType":
                     initial_full_state[key] = None
                elif key not in ["connections_shape_0", "connections_shape_1"]: # Avoid double-adding shape components
                    initial_full_state[key] = initial_state_group.attrs[key]


            for key in initial_state_group.keys(): # Load datasets
                if f"{key}_is_empty" not in initial_state_group.attrs: # Don't load if marked as empty
                    initial_full_state[key] = initial_state_group[key][:]

            # Reconstruct connections sparse matrix parts if saved
            if "connections_data" in initial_full_state and \
               "connections_indices" in initial_full_state and \
               "connections_indptr" in initial_full_state and \
               initial_state_group.attrs.get("connections_shape_0") is not None and \
               initial_state_group.attrs.get("connections_shape_1") is not None:
                # These are already numpy arrays from the loop above.
                # The shape is stored as attributes.
                initial_full_state["connections_shape"] = (
                    initial_state_group.attrs["connections_shape_0"],
                    initial_state_group.attrs["connections_shape_1"]
                )
            else: # Ensure keys exist even if not fully formed, for consistency in _apply_recorded_arrays
                if "connections_data" not in initial_full_state: initial_full_state["connections_data"] = np.array([])
                if "connections_indices" not in initial_full_state: initial_full_state["connections_indices"] = np.array([])
                if "connections_indptr" not in initial_full_state: initial_full_state["connections_indptr"] = np.array([])
                if "connections_shape" not in initial_full_state: initial_full_state["connections_shape"] = (0,0)


            # 4. Get frame count
            num_frames = 0
            frames_group = h5_file.get("frames")
            if frames_group:
                num_frames = len(list(frames_group.keys()))


            global_gui_state["loaded_recording_data"] = {
                "filepath": filepath,
                "h5_file_obj_for_playback": h5_file, # Keep file open
                "config_snapshot": config_snapshot,
                "initial_full_state": initial_full_state,
                "num_frames": num_frames
            }
            self.log_message(f"Successfully prepared metadata for {os.path.basename(filepath)}. Frames found: {num_frames}", "info")
            return True

        except FileNotFoundError:
            self.log_message(f"Error: Recording file not found at {filepath}", "error")
            global_gui_state["loaded_recording_data"] = None
            return False
        except Exception as e:
            self.log_message(f"Critical error preparing metadata for recording {filepath}: {e}", "error")
            import traceback; traceback.print_exc()
            if 'h5_file' in locals() and h5_file.id: h5_file.close()
            global_gui_state["loaded_recording_data"] = None
            return False

    def load_recording(self, filepath):
        self.log_message(f"Attempting to load recording from {filepath} for streamed playback...")

        self.stop_simulation() # Stop any live simulation
        if global_gui_state.get("is_recording", False):
            self.log_message("Loading recording: Stopping current active recording stream.", "info")
            self.stop_recording(prompt_save=False) 

        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Loading new recording: Exiting current playback mode.", "info")
            self.exit_playback_mode() 

        if self._prepare_loaded_recording_metadata(filepath): # This opens HDF5 and prepares metadata
            global_gui_state["unsaved_recording_exists"] = False
            
            # Immediately apply the recording's config and initial state
            if self._apply_config_and_initial_state_from_recording(global_gui_state["loaded_recording_data"]):
                if dpg.is_dearpygui_running():
                    update_status_bar(f"Recording loaded & initialized: {os.path.basename(filepath)}.", color=[0,200,0])
                    update_ui_for_loaded_recording(can_playback=True) # This will show playback controls
                return True
            else:
                # Failed to apply initial state, clear loaded data
                if global_gui_state["loaded_recording_data"] and global_gui_state["loaded_recording_data"].get("h5_file_obj_for_playback"):
                    try:
                        global_gui_state["loaded_recording_data"]["h5_file_obj_for_playback"].close()
                    except Exception: pass
                global_gui_state["loaded_recording_data"] = None
                if dpg.is_dearpygui_running():
                    update_status_bar(f"Error initializing from recording: {os.path.basename(filepath)}", color=[255,0,0])
                    update_ui_for_loaded_recording(can_playback=False)
                return False
        else: # _prepare_loaded_recording_metadata failed
            if dpg.is_dearpygui_running(): 
                update_status_bar(f"Error loading recording: {os.path.basename(filepath)}", color=[255,0,0])
            global_gui_state["loaded_recording_data"] = None 
            update_ui_for_loaded_recording(can_playback=False)
            return False

    def enter_playback_mode(self):
        loaded_data_meta = global_gui_state.get("loaded_recording_data")

        if not loaded_data_meta or \
           not loaded_data_meta.get("h5_file_obj_for_playback") or \
           not loaded_data_meta.get("h5_file_obj_for_playback").id or \
           loaded_data_meta.get("num_frames", 0) == 0:
            self.log_message("No valid recording loaded. Cannot enter playback mode.", "warning")
            if dpg.is_dearpygui_running(): update_status_bar("Load a valid recording first.")
            return

        # If not already in playback mode (e.g., user clicked "Playback" after loading a recording
        # and then exiting playback previously), re-initialize the playback mode state.
        if not global_gui_state.get("is_playback_mode", False):
            self.log_message("Re-entering playback mode for already loaded recording...")
            # Re-apply config and initial state to ensure sim is correctly set up for this recording
            if not self._apply_config_and_initial_state_from_recording(loaded_data_meta):
                self.log_message("Failed to re-initialize for playback. Aborting.", "error")
                # Attempt to clear playback state
                global_gui_state["is_playback_mode"] = False
                if dpg.is_dearpygui_running():
                    update_ui_for_playback_state(is_playback=False)
                    update_ui_for_loaded_recording(can_playback=False) # Hide controls if init failed
                return
        else:
            # Already in playback mode, likely just toggling play/pause or already set up.
            # Ensure UI reflects this active playback state.
            self.log_message("Already in playback mode. Ensuring UI is consistent.")

        # Ensure the simulation is not running its own logic
        self.stop_simulation() 

        global_gui_state["is_playback_mode"] = True # Explicitly set/confirm
        global_gui_state["playback_is_playing"] = False # Ensure it starts paused when "Playback" button is hit
        global_gui_state["current_playback_frame_index"] = 0 # Reset to frame 0 on explicit playback start
        self.set_playback_frame(0, update_slider_gui=True) # Update slider and visuals to frame 0

        if dpg.is_dearpygui_running():
            num_frames = loaded_data_meta.get("num_frames", 0)
            update_ui_for_playback_state(is_playback=True, num_frames=num_frames)
            update_status_bar("Playback mode active. Use controls.")

        global_gui_state["filters_changed"] = True 
        if OPENGL_AVAILABLE and glut.glutGetWindow() != 0:
            glut.glutPostRedisplay()

    def _read_frame_from_file(self, frame_idx):
        loaded_data_meta = global_gui_state.get("active_recording_data_source")
        if not loaded_data_meta or "h5_file_obj_for_playback" not in loaded_data_meta:
            self.log_message("Playback error: No file object in active recording source.", "error")
            return None

        h5_file = loaded_data_meta["h5_file_obj_for_playback"]
        if not h5_file or not h5_file.id: # Check if file is open
            self.log_message("Playback error: File is not open.", "error")
            self.exit_playback_mode()
            return None

        num_frames = loaded_data_meta.get("num_frames", 0)
        if not (0 <= frame_idx < num_frames):
            self.log_message(f"Playback error: Frame index {frame_idx} out of bounds (0-{num_frames-1}).", "error")
            return None

        frame_group_name = f"frames/frame_{frame_idx}"
        try:
            frame_group = h5_file.get(frame_group_name)
            if not frame_group:
                self.log_message(f"Playback error: Frame group '{frame_group_name}' not found.", "error")
                return None

            frame_content = {}
            # Load attributes (scalar values)
            for key, value in frame_group.attrs.items():
                if value == "NoneType": frame_content[key] = None
                elif key.endswith("_is_empty") and value is True:
                    original_key = key.replace("_is_empty","")
                    frame_content[original_key] = np.array([])
                else: frame_content[key] = value

            # Load datasets (numpy arrays)
            for key in frame_group.keys():
                 if f"{key}_is_empty" not in frame_group.attrs: # Don't load if marked as empty
                    frame_content[key] = frame_group[key][:]
            return frame_content
        except Exception as e:
            self.log_message(f"Error reading frame {frame_idx} from {loaded_data_meta['filepath']}: {e}", "error")
            return None


    def set_playback_frame(self, frame_idx, update_slider_gui=False):
        if not global_gui_state.get("is_playback_mode", False) or \
           global_gui_state.get("active_recording_data_source") is None:
            return

        loaded_data_meta = global_gui_state.get("active_recording_data_source")
        num_frames = loaded_data_meta.get("num_frames", 0)

        if num_frames == 0:
            self.log_message("Cannot set playback frame: Recording has no frames.", "warning")
            if dpg.is_dearpygui_running() and dpg.does_item_exist("playback_current_frame_text"):
                dpg.set_value("playback_current_frame_text", "Frame: 0 / 0 (No frames)")
            return

        clamped_frame_idx = max(0, min(frame_idx, num_frames - 1))

        frame_content = self._read_frame_from_file(clamped_frame_idx)
        if frame_content is None:
            self.log_message(f"Failed to read frame {clamped_frame_idx} for playback. Playback may be unstable.", "error")
            return

        global_gui_state["current_playback_frame_index"] = clamped_frame_idx
        self._apply_recorded_arrays_to_gpu(frame_content, is_initial_state=False)

        self.sim_config.current_time_ms = frame_content.get("time_ms", self.sim_config.current_time_ms)
        self.sim_config.current_time_step = frame_content.get("step", self.sim_config.current_time_step)

        if update_slider_gui and dpg.is_dearpygui_running() and dpg.does_item_exist("playback_slider"):
            if dpg.get_value("playback_slider") != clamped_frame_idx:
                 dpg.set_value("playback_slider", clamped_frame_idx)

        global_gui_state["filters_changed"] = True

    def step_playback(self, direction=1):
        if not global_gui_state.get("is_playback_mode", False) or \
           global_gui_state.get("active_recording_data_source") is None:
            return

        current_idx = global_gui_state.get("current_playback_frame_index", 0)
        loaded_data_meta = global_gui_state.get("active_recording_data_source")
        num_frames = loaded_data_meta.get("num_frames", 0)

        if num_frames == 0: return

        new_idx = current_idx + direction

        if 0 <= new_idx < num_frames:
            self.set_playback_frame(new_idx, update_slider_gui=True)
        elif new_idx >= num_frames:
            self.set_playback_frame(num_frames - 1, update_slider_gui=True)
            if global_gui_state.get("playback_is_playing", False):
                global_gui_state["playback_is_playing"] = False
                if dpg.is_dearpygui_running(): update_ui_for_playback_playing_state(is_playing=False)
        elif new_idx < 0:
            self.set_playback_frame(0, update_slider_gui=True)
            if global_gui_state.get("playback_is_playing", False):
                global_gui_state["playback_is_playing"] = False
                if dpg.is_dearpygui_running(): update_ui_for_playback_playing_state(is_playing=False)

    def exit_playback_mode(self):
        is_currently_in_playback = global_gui_state.get("is_playback_mode", False)
        if not is_currently_in_playback:
            # If a file was loaded but playback mode wasn't fully entered or was reset, still try to close any open HDF5 file.
            loaded_data_meta = global_gui_state.get("loaded_recording_data")
            if loaded_data_meta and loaded_data_meta.get("h5_file_obj_for_playback") and \
               loaded_data_meta.get("h5_file_obj_for_playback").id:
                try:
                    loaded_data_meta["h5_file_obj_for_playback"].close()
                    self.log_message("Closed file associated with loaded recording (not in active playback).")
                except Exception as e:
                    self.log_message(f"Error closing file on partial playback exit: {e}", "error")
                loaded_data_meta["h5_file_obj_for_playback"] = None
            return

        self.log_message("Exiting playback mode...")
        global_gui_state["is_playback_mode"] = False
        global_gui_state["playback_is_playing"] = False

        loaded_data_meta = global_gui_state.get("active_recording_data_source") # or loaded_recording_data
        if loaded_data_meta and "h5_file_obj_for_playback" in loaded_data_meta:
            h5_file = loaded_data_meta["h5_file_obj_for_playback"]
            if h5_file and h5_file.id: 
                try:
                    h5_file.close()
                    self.log_message("Closed file from playback mode.")
                except Exception as e:
                    self.log_message(f"Error closing file on playback exit: {e}", "error")
            loaded_data_meta["h5_file_obj_for_playback"] = None 

        if dpg.is_dearpygui_running():
            update_ui_for_playback_state(is_playback=False) # This updates main button, enables live controls
            # update_ui_for_loaded_recording will determine if playback controls remain visible
            is_still_loaded = global_gui_state.get("loaded_recording_data") is not None
            update_ui_for_loaded_recording(can_playback=is_still_loaded)

        self.log_message("Resetting simulation to current profile/config after HDF5 playback.")
        # Apply the last known "live" config, not the recording's config.

        self.apply_simulation_configuration_core(self.sim_config.to_dict(), is_part_of_playback_setup=False)

        if dpg.is_dearpygui_running():
            handle_stop_simulation_event() 
            update_status_bar("Exited playback. Live simulation mode.")

    def _apply_recorded_arrays_to_gpu(self, state_dict_np, is_initial_state=False):
        if not self.is_initialized:
            self.log_message("Cannot apply recorded arrays: Sim not initialized for playback.", "error")
            return

        def _apply_to_cp_array(cp_array_attr_name, np_array_key_in_dict):
            if np_array_key_in_dict in state_dict_np and state_dict_np[np_array_key_in_dict] is not None:
                target_cp_array = getattr(self, cp_array_attr_name, None)
                source_np_array = state_dict_np[np_array_key_in_dict]

                if not isinstance(source_np_array, np.ndarray): # Should be numpy array from HDF5
                    self.log_message(f"Warning: Source for {cp_array_attr_name} is not a NumPy array (type: {type(source_np_array)}). Skipping.", "warning")
                    return

                if target_cp_array is None and source_np_array.size > 0 : # If target doesn't exist but source has data
                    self.log_message(f"Warning: Target CuPy array {cp_array_attr_name} is None. Creating from recording.", "debug")
                    try:
                        setattr(self, cp_array_attr_name, cp.asarray(source_np_array))
                        target_cp_array = getattr(self, cp_array_attr_name) # Refresh reference
                    except Exception as e:
                        self.log_message(f"Error creating {cp_array_attr_name} from recording: {e}", "error"); return

                if target_cp_array is not None:
                    if target_cp_array.shape == source_np_array.shape:
                        if target_cp_array.dtype == source_np_array.dtype:
                            target_cp_array[:] = cp.asarray(source_np_array)
                        else: # Dtype mismatch, attempt cast
                            try: target_cp_array[:] = cp.asarray(source_np_array.astype(target_cp_array.dtype))
                            except Exception as e: self.log_message(f"Error applying {cp_array_attr_name} due to dtype mismatch and cast fail: {e}", "error")
                    elif target_cp_array.size == source_np_array.size and source_np_array.size > 0: # Same size, different shape
                        self.log_message(f"Warning: Shape mismatch for {cp_array_attr_name} during playback. Reshaping. Target: {target_cp_array.shape}, Source: {source_np_array.shape}", "debug")
                        try: target_cp_array[:] = cp.asarray(source_np_array.reshape(target_cp_array.shape))
                        except ValueError as ve: self.log_message(f"ERROR: Failed to reshape {cp_array_attr_name}. Error: {ve}", "error")
                    elif source_np_array.size == 0 and target_cp_array.size == 0: pass # Both empty
                    elif source_np_array.size == 0 and target_cp_array.size > 0: # Source is empty, clear target
                         target_cp_array.fill(0) # Or appropriate default
                    else: # Mismatch that can't be easily resolved
                        self.log_message(f"Error: Shape/size mismatch for {cp_array_attr_name} from recording. Target: {target_cp_array.shape}, Source: {source_np_array.shape}. Cannot apply.", "error")
            elif np_array_key_in_dict in state_dict_np and state_dict_np[np_array_key_in_dict] is None:
                 # If source was explicitly None, ensure target is also None or empty
                 target_cp_array = getattr(self, cp_array_attr_name, None)
                 if target_cp_array is not None:
                     if target_cp_array.size > 0: # If target exists and has data, maybe clear it or set to None
                         self.log_message(f"Info: Source for {cp_array_attr_name} is None in recording. Clearing/resetting target.", "debug")
                         # setattr(self, cp_array_attr_name, cp.array([], dtype=target_cp_array.dtype) if target_cp_array.ndim == 1 else cp.zeros_like(target_cp_array))
                         # Or, more simply, if the array is expected to be optional:
                         setattr(self, cp_array_attr_name, None)


        if is_initial_state:
            _apply_to_cp_array("cp_traits", "cp_traits")
            _apply_to_cp_array("cp_neuron_positions_3d", "cp_neuron_positions_3d")
            if self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                    _apply_to_cp_array(f"cp_izh_{param}", f"cp_izh_{param}")
            elif self.sim_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                     _apply_to_cp_array(f"cp_hh_{param}", f"cp_hh_{param}")

            if "connections_data" in state_dict_np and \
               "connections_indices" in state_dict_np and \
               "connections_indptr" in state_dict_np and \
               "connections_shape" in state_dict_np:

                conn_data_np = state_dict_np["connections_data"]
                conn_indices_np = state_dict_np["connections_indices"]
                conn_indptr_np = state_dict_np["connections_indptr"]
                conn_shape = state_dict_np["connections_shape"] # This should be a tuple from HDF5 attrs

                if conn_shape[0] != self.sim_config.num_neurons or conn_shape[1] != self.sim_config.num_neurons:
                    self.log_message(f"Error: Connection shape {conn_shape} from recording initial_state "
                                     f"does not match configured neuron count {self.sim_config.num_neurons}. Playback may fail.", "error")

                if conn_data_np.size > 0 or conn_indices_np.size > 0 or conn_indptr_np.size > 0 : #Only if there's actual data
                    self.cp_connections = csp.csr_matrix((cp.asarray(conn_data_np),
                                                          cp.asarray(conn_indices_np),
                                                          cp.asarray(conn_indptr_np)),
                                                         shape=conn_shape)
                    self.cp_connections.sort_indices()
                    self.log_message(f"Playback: Applied connection structure. NNZ: {self.cp_connections.nnz}", "debug")
                else: # Empty connection data
                    self.cp_connections = csp.csr_matrix(conn_shape, dtype=cp.float32)
                    self.log_message(f"Playback: Applied empty connection structure. Shape: {conn_shape}", "debug")


                num_synapses_from_recording = self.cp_connections.nnz

                # Initialize pulse timers and STP arrays based on loaded connections
                self.cp_synapse_pulse_timers = cp.asarray(state_dict_np.get("cp_synapse_pulse_timers", cp.zeros(num_synapses_from_recording, dtype=cp.int32)))
                self.cp_synapse_pulse_progress = cp.asarray(state_dict_np.get("cp_synapse_pulse_progress", cp.zeros(num_synapses_from_recording, dtype=cp.float32)))

                if self.sim_config.enable_short_term_plasticity and num_synapses_from_recording > 0:
                    self.cp_stp_u = cp.asarray(state_dict_np.get("cp_stp_u", cp.full(num_synapses_from_recording, self.sim_config.stp_U, dtype=cp.float32)))
                    self.cp_stp_x = cp.asarray(state_dict_np.get("cp_stp_x", cp.ones(num_synapses_from_recording, dtype=cp.float32)))
                else:
                    self.cp_stp_u = None; self.cp_stp_x = None
            else:
                self.log_message("Warning: Connection structure missing or incomplete in initial_state. Using defaults.", "warning")
                # Default init for connections and related arrays if not in HDF5
                if self.cp_connections is None: self.cp_connections = csp.csr_matrix((self.sim_config.num_neurons, self.sim_config.num_neurons), dtype=cp.float32)
                num_syn = self.cp_connections.nnz
                self.cp_synapse_pulse_timers = cp.zeros(num_syn, dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.zeros(num_syn, dtype=cp.float32)
                if self.sim_config.enable_short_term_plasticity and num_syn > 0:
                    self.cp_stp_u = cp.full(num_syn, self.sim_config.stp_U, dtype=cp.float32)
                    self.cp_stp_x = cp.ones(num_syn, dtype=cp.float32)
                else: self.cp_stp_u = None; self.cp_stp_x = None


        dynamic_keys_map = {
            'cp_membrane_potential_v': 'cp_membrane_potential_v', 'cp_recovery_variable_u': 'cp_recovery_variable_u',
            'cp_gating_variable_m': 'cp_gating_variable_m', 'cp_gating_variable_h': 'cp_gating_variable_h',
            'cp_gating_variable_n': 'cp_gating_variable_n', 'cp_conductance_g_e': 'cp_conductance_g_e',
            'cp_conductance_g_i': 'cp_conductance_g_i', 'cp_external_input_current': 'cp_external_input_current',
            'cp_refractory_timers': 'cp_refractory_timers', 'cp_viz_activity_timers': 'cp_viz_activity_timers',
            'cp_neuron_firing_thresholds': 'cp_neuron_firing_thresholds', 'cp_neuron_activity_ema': 'cp_neuron_activity_ema',
            'cp_firing_states': 'cp_firing_states', 'cp_prev_firing_states': 'cp_prev_firing_states',
            'cp_stp_u': 'cp_stp_u', 'cp_stp_x': 'cp_stp_x',
            'cp_synapse_pulse_timers': 'cp_synapse_pulse_timers',
            'cp_synapse_pulse_progress': 'cp_synapse_pulse_progress'
        }

        if not is_initial_state and "cp_connections_data" in state_dict_np: # For per-frame connection data updates
            if self.cp_connections is not None and self.cp_connections.data is not None and \
               state_dict_np["cp_connections_data"] is not None:
                source_np_array = state_dict_np["cp_connections_data"]
                if isinstance(source_np_array, np.ndarray):
                    if self.cp_connections.data.shape == source_np_array.shape:
                        self.cp_connections.data[:] = cp.asarray(source_np_array)
                    elif self.cp_connections.data.size == source_np_array.size and source_np_array.size > 0:
                        try: self.cp_connections.data[:] = cp.asarray(source_np_array.reshape(self.cp_connections.data.shape))
                        except ValueError as ve: self.log_message(f"ERROR: Failed to reshape cp_connections.data from recording frame. Error: {ve}", "error")
                    elif not (self.cp_connections.data.size == 0 and source_np_array.size == 0) :
                        self.log_message(f"Error: Shape/size mismatch for dynamic cp_connections.data from recording frame. Cannot apply.", "error")

        for cp_attr, np_key in dynamic_keys_map.items():
            if np_key == "cp_connections_data" and not is_initial_state: # Handled above for frames
                continue
            _apply_to_cp_array(cp_attr, np_key)

        self._mock_num_spikes_this_step = state_dict_np.get("_mock_num_spikes_this_step", 0)
        self._mock_network_avg_firing_rate_hz = state_dict_np.get("_mock_network_avg_firing_rate_hz", 0.0)
        self._mock_total_plasticity_events = state_dict_np.get("_mock_total_plasticity_events", 0)

        if is_initial_state: # Set initial time from the recording's initial state attributes
            self.sim_config.current_time_ms = state_dict_np.get("start_time_ms", 0.0)
            self.sim_config.current_time_step = state_dict_np.get("start_time_step", 0)

    def get_latest_simulation_data_for_gui(self, force_fetch=False):
        if not global_gui_state.get("is_playback_mode", False):
            if not self.sim_config.is_running and not self.sim_config.is_paused and not force_fetch:
                return None

        if not self.is_initialized:
            self.log_message("GUI data request: Sim not initialized.","debug"); return None

        n = self.sim_config.num_neurons
        vm_np, fired_np = np.array([]), np.array([])
        neuron_positions_3d_np = np.array([])
        neuron_activity_timers_np = np.array([])

        if self.cp_membrane_potential_v is not None and self.cp_firing_states is not None:
            vm_np = cp.asnumpy(self.cp_membrane_potential_v)
            fired_np = cp.asnumpy(self.cp_firing_states)
        elif n > 0: # Fallback if arrays not initialized but n > 0
            vm_np = np.full(n, self.sim_config.lif_v_rest if self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name else self.sim_config.hh_v_rest_init, dtype=np.float32)
            fired_np = np.zeros(n, dtype=bool)

        if self.cp_neuron_positions_3d is not None:
            neuron_positions_3d_np = cp.asnumpy(self.cp_neuron_positions_3d)
        elif n > 0:
            neuron_positions_3d_np = np.zeros((n,3),dtype=np.float32)

        if self.cp_viz_activity_timers is not None:
            neuron_activity_timers_np = cp.asnumpy(self.cp_viz_activity_timers)
        elif n > 0:
            neuron_activity_timers_np = np.zeros(n, dtype=np.int32)

        synapse_info_for_gui = []
        if self.cp_connections is not None and hasattr(self.cp_connections,'nnz') and self.cp_connections.nnz > 0:
            max_synapses_to_visualize = opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000) if OPENGL_AVAILABLE else 7500
            try:
                coo_conn = self.cp_connections.tocoo(copy=False)
                num_actual_synapses = coo_conn.nnz
                num_to_show = min(num_actual_synapses, max_synapses_to_visualize)

                if num_to_show > 0:
                    indices_to_sample = np.random.choice(num_actual_synapses, num_to_show, replace=False) if num_actual_synapses > num_to_show else np.arange(num_actual_synapses)

                    row_indices_np = cp.asnumpy(coo_conn.row[indices_to_sample])
                    col_indices_np = cp.asnumpy(coo_conn.col[indices_to_sample])
                    
                    current_weights_for_viz_data = self.cp_connections.data
                    if not global_gui_state.get("is_playback_mode", False) and \
                       self.sim_config.enable_short_term_plasticity and \
                       self.cp_stp_u is not None and self.cp_stp_x is not None and \
                       self.cp_stp_u.size == self.cp_connections.data.size and \
                       self.cp_stp_x.size == self.cp_connections.data.size :
                         current_weights_for_viz_data = self.cp_connections.data * self.cp_stp_u * self.cp_stp_x
                    
                    sampled_weights_np = cp.asnumpy(current_weights_for_viz_data[indices_to_sample])


                    for i in range(num_to_show):
                        synapse_info_for_gui.append({
                            "source_idx": int(row_indices_np[i]),
                            "target_idx": int(col_indices_np[i]),
                            "weight": float(sampled_weights_np[i])
                        })
            except Exception as e: self.log_message(f"Error processing connections for GUI: {e}","error")

        gui_data_dict = {
            "current_time_ms": self.sim_config.current_time_ms,
            "current_time_step": self.sim_config.current_time_step,
            "neuron_Vm_sample": vm_np,
            "neuron_fired_status": fired_np,
            "neuron_activity_timers": neuron_activity_timers_np,
            "neuron_positions_3d": neuron_positions_3d_np,
            "num_spikes_this_step": self._mock_num_spikes_this_step,
            "network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
            "total_plasticity_events": self._mock_total_plasticity_events,
            "synapse_info": synapse_info_for_gui
        }

        if self.sim_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            if self.cp_gating_variable_m is not None: gui_data_dict["neuron_m_sample"] = cp.asnumpy(self.cp_gating_variable_m)
            if self.cp_gating_variable_h is not None: gui_data_dict["neuron_h_sample"] = cp.asnumpy(self.cp_gating_variable_h)
            if self.cp_gating_variable_n is not None: gui_data_dict["neuron_n_sample"] = cp.asnumpy(self.cp_gating_variable_n)
        elif self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            if self.cp_recovery_variable_u is not None: gui_data_dict["neuron_u_sample"] = cp.asnumpy(self.cp_recovery_variable_u)
            if self.cp_neuron_firing_thresholds is not None: gui_data_dict["neuron_thresholds_sample"] = cp.asnumpy(self.cp_neuron_firing_thresholds)
        return gui_data_dict

    def get_initial_sim_data_snapshot(self):
        if not self.is_initialized:
            self.log_message("Initial snapshot request: Sim not initialized. Attempting default init.","info")
            self._initialize_simulation_data(called_from_playback_init=False) # Ensure it's not playback init
            if not self.is_initialized:
                self.log_message("Initialization failed for initial snapshot.","error")
                return { # Return a minimal structure
                    "current_time_ms":0.0, "current_time_step":0,
                    "neuron_Vm_sample":np.array([]), "neuron_fired_status":np.array([]),
                    "neuron_activity_timers": np.array([]),
                    "neuron_positions_3d":np.array([]),
                    "num_spikes_this_step":0, "network_avg_firing_rate_hz":0.0,
                    "total_plasticity_events":0, "synapse_info":[]
                }

        snapshot = self.get_latest_simulation_data_for_gui(force_fetch=True)

        if snapshot: # Reset time-dependent fields for a true "initial" snapshot
            if not global_gui_state.get("is_playback_mode", False): # Only if not in playback
                snapshot.update({
                    "current_time_ms":0.0,
                    "current_time_step":0,
                    "num_spikes_this_step":0,
                    "network_avg_firing_rate_hz":0.0,
                    "total_plasticity_events":0
                })
                if "neuron_activity_timers" in snapshot and snapshot["neuron_activity_timers"].size > 0:
                     snapshot["neuron_activity_timers"].fill(0) # Reset activity timers
        return snapshot

    def get_profile_visualization_data(self, from_current_config=False):
        cfg = self.sim_config; num_n = cfg.num_neurons

        positions_stale = self.cp_neuron_positions_3d is None or self.cp_neuron_positions_3d.shape[0] != num_n
        types_stale = not cfg.neuron_types_list_for_viz or len(cfg.neuron_types_list_for_viz) != num_n

        if from_current_config and (positions_stale or types_stale):
            self.log_message("Re-populating neuron positions/types for visualization profile (3D).","debug")

            if positions_stale and num_n > 0:
                np_positions_3d = np.random.uniform(
                    low=[cfg.volume_min_x,cfg.volume_min_y,cfg.volume_min_z],
                    high=[cfg.volume_max_x,cfg.volume_max_y,cfg.volume_max_z],
                    size=(num_n,3)).astype(np.float32)
                self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                cfg.neuron_positions_x = np_positions_3d[:,0].tolist()
                cfg.neuron_positions_y = np_positions_3d[:,1].tolist()
            elif num_n == 0: # Ensure empty arrays for n=0
                self.cp_neuron_positions_3d = cp.array([],dtype=np.float32).reshape(0,3)
                cfg.neuron_positions_x=[]; cfg.neuron_positions_y=[]

            if types_stale: # Re-generate types list
                cfg.neuron_types_list_for_viz = [""] * num_n
                np_traits_host_temp = cp.asnumpy(self.cp_traits) if self.cp_traits is not None and self.cp_traits.size == num_n else \
                                 np.random.randint(0, max(1, cfg.num_traits), num_n)
                if self.cp_traits is None or self.cp_traits.size != num_n: # Ensure cp_traits is consistent
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
                else: # Fallback for unknown model types
                    cfg.neuron_types_list_for_viz = [f"Unknown_Type_{np_traits_host_temp[i]}" for i in range(num_n)]

        positions_3d_np = cp.asnumpy(self.cp_neuron_positions_3d) if self.cp_neuron_positions_3d is not None else np.zeros((0,3), dtype=np.float32)
        return {
            "neuron_positions_3d": positions_3d_np,
            "neuron_types": cfg.neuron_types_list_for_viz, # Ensure this is populated
            "neuron_positions_x_proj": cfg.neuron_positions_x, # Ensure this is populated
            "neuron_positions_y_proj": cfg.neuron_positions_y  # Ensure this is populated
        }

    def get_available_neuron_types(self):
        cfg = self.sim_config
        available_types = ["All"] # "All" is always an option
        if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            available_types.extend([f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS])
        elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            available_types.extend([f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS])
        # Add other model types here if they have specific named variants
        return list(dict.fromkeys(available_types)) # Remove duplicates if any, preserve order

    def save_checkpoint(self, filepath):
        self.log_message(f"Saving checkpoint to {filepath}...")
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot save checkpoint during playback mode. Exit playback first.", "warning")
            if dpg.is_dearpygui_running(): update_status_bar("Error: Exit playback to save checkpoint.")
            return False
        if not self.is_initialized:
            self.log_message("Sim not initialized. Cannot save checkpoint.","warning"); return False

        try:
            with h5py.File(filepath, 'w') as h5f:
                # Save simulation_configuration as root attributes
                config_dict = self.sim_config.to_dict()
                save_dict_to_hdf5_attrs(h5f, config_dict)

                # Save main simulation state arrays as datasets in the root or a 'state' group
                state_group = h5f # Or h5f.create_group("simulation_state")

                arrays_to_save_direct = [ # Attributes that are directly CuPy arrays
                    'cp_membrane_potential_v', 'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_external_input_current', 'cp_firing_states', 'cp_prev_firing_states',
                    'cp_traits', 'cp_refractory_timers', 'cp_neuron_positions_3d',
                    'cp_neuron_activity_ema', 'cp_viz_activity_timers',
                    'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
                ]
                for attr_name in arrays_to_save_direct:
                    data_array = getattr(self, attr_name, None)
                    if data_array is not None and data_array.size > 0:
                        state_group.create_dataset(attr_name, data=cp.asnumpy(data_array))
                    elif data_array is not None: # Empty array
                         state_group.attrs[f"{attr_name}_is_empty"] = True


                if self.cp_connections is not None:
                    if self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                        state_group.create_dataset("connections_data", data=cp.asnumpy(self.cp_connections.data))
                    if self.cp_connections.indices is not None and self.cp_connections.indices.size > 0:
                        state_group.create_dataset("connections_indices", data=cp.asnumpy(self.cp_connections.indices))
                    if self.cp_connections.indptr is not None and self.cp_connections.indptr.size > 0:
                        state_group.create_dataset("connections_indptr", data=cp.asnumpy(self.cp_connections.indptr))
                    state_group.attrs["connections_shape_0"] = self.cp_connections.shape[0]
                    state_group.attrs["connections_shape_1"] = self.cp_connections.shape[1]

                if self.cp_stp_u is not None and self.cp_stp_u.size > 0: state_group.create_dataset("cp_stp_u", data=cp.asnumpy(self.cp_stp_u))
                if self.cp_stp_x is not None and self.cp_stp_x.size > 0: state_group.create_dataset("cp_stp_x", data=cp.asnumpy(self.cp_stp_x))

                if self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    if self.cp_recovery_variable_u is not None and self.cp_recovery_variable_u.size > 0: state_group.create_dataset("cp_recovery_variable_u", data=cp.asnumpy(self.cp_recovery_variable_u))
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                         attr_name_cp = f"cp_izh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array))
                    if self.cp_neuron_firing_thresholds is not None and self.cp_neuron_firing_thresholds.size > 0: state_group.create_dataset("cp_neuron_firing_thresholds", data=cp.asnumpy(self.cp_neuron_firing_thresholds))
                elif self.sim_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                        attr_name_cp = f"cp_gating_variable_{attr_name_suffix}"
                        data_array = getattr(self, attr_name_cp, None)
                        if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array))
                    for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                         attr_name_cp = f"cp_hh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array))

                # Save mock telemetry and minimal GUI state as attributes
                h5f.attrs["_mock_total_plasticity_events"] = self._mock_total_plasticity_events
                h5f.attrs["_mock_network_avg_firing_rate_hz"] = self._mock_network_avg_firing_rate_hz
                if global_viz_data_cache.get("neuron_types"): # Save neuron_types_list_for_viz if available
                    h5f.attrs["neuron_types_list_for_viz_json"] = json.dumps(global_viz_data_cache["neuron_types"])

                save_dict_to_hdf5_attrs(h5f.attrs, {"opengl_viz_config": opengl_viz_config.copy() if OPENGL_AVAILABLE else {}})
                save_dict_to_hdf5_attrs(h5f.attrs, {"global_gui_state_filters": {"show_connections_gl": global_gui_state.get("show_connections_gl", False)}})


            self.log_message(f"Checkpoint saved successfully to {filepath}"); return True
        except Exception as e:
            self.log_message(f"Error saving checkpoint: {e}","error"); import traceback; traceback.print_exc(); return False

    def load_checkpoint(self, filepath):
        self.log_message(f"Loading checkpoint from {filepath}...")
        if global_gui_state.get("is_recording", False):
            self.log_message("Cannot load checkpoint while recording. Stop recording first.", "warning")
            if dpg.is_dearpygui_running(): update_status_bar("Error: Stop recording to load checkpoint.")
            return False
        if global_gui_state.get("is_playback_mode", False):
            self.log_message("Cannot load checkpoint during playback. Exit playback first.", "warning")
            if dpg.is_dearpygui_running(): update_status_bar("Error: Exit playback to load checkpoint.")
            return False

        try:
            with h5py.File(filepath, 'r') as h5f:
                self.stop_simulation() # Stop current sim before clearing
                self.clear_simulation_state_and_gpu_memory()

                loaded_sim_config_dict = load_dict_from_hdf5_attrs(h5f) # Load from root attributes
                if not loaded_sim_config_dict: # Basic check
                    self.log_message("Checkpoint missing simulation_configuration attributes. Load failed.","error"); return False
                
                # Ensure all expected keys are present or defaulted before creating SimulationConfiguration
                # This step might need more robust handling if config structure changes significantly
                temp_cfg_for_validation = SimulationConfiguration()
                for key_cfg in temp_cfg_for_validation.to_dict().keys():
                    if key_cfg not in loaded_sim_config_dict:
                        loaded_sim_config_dict[key_cfg] = getattr(temp_cfg_for_validation, key_cfg)


                self.sim_config = SimulationConfiguration.from_dict(loaded_sim_config_dict)
                n = self.sim_config.num_neurons
                state_group = h5f # Assuming data is in root, or use h5f.get("simulation_state")

                def _load_cp_array_from_h5(key, default_val_func=lambda size_n: cp.zeros(size_n, dtype=cp.float32)):
                    if f"{key}_is_empty" in state_group.attrs and state_group.attrs[f"{key}_is_empty"] is True:
                        return default_val_func(0) # Return empty array of correct type/shape if possible
                    if key in state_group:
                        return cp.asarray(state_group[key][:])
                    # If key not present and not marked empty, means it wasn't saved (e.g. optional array)
                    # or it's an older checkpoint. Fallback to default.
                    self.log_message(f"checkpoint: Dataset for '{key}' not found or was empty. Using default.", "debug")
                    return default_val_func(n) if n > 0 else default_val_func(0)


                arrays_to_load_direct = [
                    'cp_membrane_potential_v', 'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_external_input_current', 'cp_firing_states', 'cp_prev_firing_states',
                    'cp_traits', 'cp_refractory_timers', 'cp_neuron_activity_ema',
                    'cp_viz_activity_timers', 'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
                ]
                for attr_name in arrays_to_load_direct:
                    default_dtype = bool if 'firing_states' in attr_name else \
                                    (cp.int32 if 'timers' in attr_name or 'traits' in attr_name else cp.float32)
                    setattr(self, attr_name, _load_cp_array_from_h5(attr_name,
                            default_val_func=lambda size_n, dt=default_dtype: cp.zeros(size_n, dtype=dt) ))

                # neuron_positions_3d might not exist in older checkpoints, handle gracefully
                if "cp_neuron_positions_3d" in state_group:
                     self.cp_neuron_positions_3d = _load_cp_array_from_h5("cp_neuron_positions_3d")
                elif n > 0 : # If not in checkpoint, generate random ones
                    np_positions_3d = np.random.uniform(
                        low=[self.sim_config.volume_min_x,self.sim_config.volume_min_y,self.sim_config.volume_min_z],
                        high=[self.sim_config.volume_max_x,self.sim_config.volume_max_y,self.sim_config.volume_max_z],
                        size=(n,3)).astype(np.float32)
                    self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                else: self.cp_neuron_positions_3d = cp.array([], dtype=cp.float32).reshape(0,3)


                if "connections_data" in state_group and \
                   "connections_indices" in state_group and \
                   "connections_indptr" in state_group and \
                   "connections_shape_0" in state_group.attrs and \
                   "connections_shape_1" in state_group.attrs:
                    conn_data_np = state_group["connections_data"][:]
                    conn_indices_np = state_group["connections_indices"][:]
                    conn_indptr_np = state_group["connections_indptr"][:]
                    conn_shape = (state_group.attrs["connections_shape_0"], state_group.attrs["connections_shape_1"])
                    self.cp_connections = csp.csr_matrix((cp.asarray(conn_data_np), cp.asarray(conn_indices_np), cp.asarray(conn_indptr_np)), shape=conn_shape)
                elif n > 0: self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32) # Fallback
                else: self.cp_connections = csp.csr_matrix((0,0), dtype=cp.float32)

                num_synapses_loaded = self.cp_connections.nnz if self.cp_connections is not None else 0

                self.cp_stp_u = _load_cp_array_from_h5("cp_stp_u",
                    lambda s: cp.full(s, self.sim_config.stp_U, dtype=cp.float32) if self.sim_config.enable_short_term_plasticity and num_synapses_loaded > 0 and s > 0 else None)
                self.cp_stp_x = _load_cp_array_from_h5("cp_stp_x",
                    lambda s: cp.ones(s, dtype=cp.float32) if self.sim_config.enable_short_term_plasticity and num_synapses_loaded > 0 and s > 0 else None)
                # Ensure None if they ended up as empty arrays but should be None
                if self.cp_stp_u is not None and self.cp_stp_u.size == 0 and not (self.sim_config.enable_short_term_plasticity and num_synapses_loaded > 0) : self.cp_stp_u = None
                if self.cp_stp_x is not None and self.cp_stp_x.size == 0 and not (self.sim_config.enable_short_term_plasticity and num_synapses_loaded > 0) : self.cp_stp_x = None


                if self.sim_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    self.cp_recovery_variable_u = _load_cp_array_from_h5("cp_recovery_variable_u")
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                        setattr(self, f"cp_izh_{param}", _load_cp_array_from_h5(f"cp_izh_{param}",
                                lambda s, p=param: cp.full(s, getattr(self.sim_config, f"izh_{p}_val"), dtype=cp.float32)))
                    self.cp_neuron_firing_thresholds = _load_cp_array_from_h5("cp_neuron_firing_thresholds",
                        lambda s: cp.random.uniform(self.sim_config.homeostasis_threshold_min, self.sim_config.homeostasis_threshold_max, s).astype(cp.float32) if s > 0 else cp.array([], dtype=cp.float32))
                elif self.sim_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                         setattr(self, f"cp_gating_variable_{attr_name_suffix}", _load_cp_array_from_h5(f"cp_gating_variable_{attr_name_suffix}",
                                 lambda s, suff=attr_name_suffix: cp.full(s, getattr(self.sim_config, f"hh_{suff}_init"), dtype=cp.float32)))
                    hh_param_map = {'C_m': 'hh_C_m', 'g_Na_max': 'hh_g_Na_max', 'g_K_max': 'hh_g_K_max', 'g_L': 'hh_g_L',
                                    'E_Na': 'hh_E_Na', 'E_K': 'hh_E_K', 'E_L': 'hh_E_L', 'v_peak': 'hh_v_peak'}
                    for param_key, config_attr_name in hh_param_map.items():
                         setattr(self, f"cp_hh_{param_key}", _load_cp_array_from_h5(f"cp_hh_{param_key}",
                                 lambda s, ca_name=config_attr_name: cp.full(s, getattr(self.sim_config, ca_name), dtype=cp.float32)))
                    self.cp_neuron_firing_thresholds = None # Not used for HH in this model's homeostasis

                self._mock_total_plasticity_events = h5f.attrs.get("_mock_total_plasticity_events",0)
                self._mock_network_avg_firing_rate_hz = h5f.attrs.get("_mock_network_avg_firing_rate_hz",0.0)

                self.is_initialized = True # Set after all data is loaded
                self.log_message(f"Checkpoint loaded. Sim time: {self.sim_config.current_time_ms}ms, Step: {self.sim_config.current_time_step}, Model: {self.sim_config.neuron_model_type}")

                # Load and apply GUI/Viz settings from checkpoint
                if "neuron_types_list_for_viz_json" in h5f.attrs:
                    try:
                        self.sim_config.neuron_types_list_for_viz = json.loads(h5f.attrs["neuron_types_list_for_viz_json"])
                        global_viz_data_cache["neuron_types"] = self.sim_config.neuron_types_list_for_viz
                    except json.JSONDecodeError:
                         self.log_message("Warning: Could not parse neuron_types_list_for_viz_json from checkpoint.", "warning")
                         # Fallback to re-generating if needed
                         profile_viz_data_temp = self.get_profile_visualization_data(from_current_config=True)
                         if profile_viz_data_temp and "neuron_types" in profile_viz_data_temp:
                             global_viz_data_cache["neuron_types"] = profile_viz_data_temp["neuron_types"]
                             self.sim_config.neuron_types_list_for_viz = profile_viz_data_temp["neuron_types"]

                if self.cp_neuron_positions_3d is not None and n > 0:
                    np_pos_3d = cp.asnumpy(self.cp_neuron_positions_3d)
                    global_viz_data_cache["neuron_positions_x"]=np_pos_3d[:,0].tolist()
                    global_viz_data_cache["neuron_positions_y"]=np_pos_3d[:,1].tolist()
                elif n == 0:
                    global_viz_data_cache["neuron_positions_x"]=[]
                    global_viz_data_cache["neuron_positions_y"]=[]


                if OPENGL_AVAILABLE:
                    loaded_gl_config_dict = load_dict_from_hdf5_attrs(h5f.attrs) # Helper loads all attrs
                    if "opengl_viz_config" in loaded_gl_config_dict and isinstance(loaded_gl_config_dict["opengl_viz_config"], dict):
                         opengl_viz_config.update(loaded_gl_config_dict["opengl_viz_config"])
                if "global_gui_state_filters" in load_dict_from_hdf5_attrs(h5f.attrs):
                    gui_filters_loaded = load_dict_from_hdf5_attrs(h5f.attrs)["global_gui_state_filters"]
                    if isinstance(gui_filters_loaded, dict):
                         global_gui_state["show_connections_gl"] = gui_filters_loaded.get("show_connections_gl",False)


            return True
        except Exception as e:
            self.log_message(f"Error loading checkpoint: {e}","error"); import traceback; traceback.print_exc()
            self.is_initialized=False; return False

# --- Global Variables & GUI State (Must be defined before use in SimulationBridge or DPG) ---
global_simulation_bridge = None
global_gui_state = {
    "filters_changed": False,
    "current_profile_name": "default_profile.json", # Profiles remain .json
    "_was_running_last_frame": False,
    "show_connections_gl": True,
    "neuron_filter_mode_gl": 0, # 0: All, 1: Spiking, 2: By Type (example, not fully used by this logic)
    "_dt_warning_logged": False,
    "reset_sim_needed_from_ui_change": False,

    "is_recording": False,
    "is_playback_mode": False,
    "current_playback_frame_index": 0,
    "loaded_recording_data": None, # For HDF5: {filepath, h5_file_obj_for_playback, config_snapshot, initial_full_state, num_frames}
    "active_recording_data_source": None,
    "unsaved_recording_exists": False, # True if HDF5 recording file is open and being written to
    "playback_slider_max_value": 0,
    "playback_is_playing": False,
    "last_playback_autostep_time": 0.0,
    "playback_fps": 30.0
}
global_viz_data_cache = {
    "neuron_positions_x": [],
    "neuron_positions_y": [],
    "neuron_types": [], # This will be critical for filtering
    "last_visible_neuron_indices": [],
    "last_visible_synapse_indices": []
}

# --- Shutdown Flag & Other Top-Level Globals ---
shutdown_flag = threading.Event()
last_sim_update_time_dpg = 0.0

# --- OpenGL Specific Globals & Config ---
gl_neuron_pos_vbo = None
gl_neuron_color_vbo = None
gl_synapse_vertices_vbo = None
gl_pulse_vertices_vbo = None
gl_num_neurons_to_draw = 0
gl_num_synapse_lines_to_draw = 0
gl_num_pulses_to_draw = 0

gl_neuron_pos_np = np.array([], dtype=np.float32)
gl_neuron_colors_np = np.array([], dtype=np.float32)
gl_connection_vertices_np = np.array([], dtype=np.float32)
gl_pulse_vertices_np = np.array([], dtype=np.float32)


if OPENGL_AVAILABLE:
    gl_data_lock = threading.Lock()
    opengl_viz_config = {
        "WINDOW_WIDTH": 800, "WINDOW_HEIGHT": 600,
        "POINT_SIZE": 3.0,
        "MAX_NEURONS_TO_RENDER": 1000000, # Increased default
        "MAX_CONNECTIONS_TO_RENDER": 10000000, # Increased default
        "INACTIVE_NEURON_OPACITY": 0.25,
        "FIRING_NEURON_COLOR": [1.0, 1.0, 0.0, 1.0], # R,G,B,A
        "ACTIVITY_HIGHLIGHT_FRAMES": 7, # How many frames a neuron stays highlighted after firing
        "FOOTER_HEIGHT_PIXELS": 75,
        "SYNAPSE_ALPHA_MODIFIER": 0.50, # Multiplier for base synapse alpha
        "SYNAPSE_BASE_COLOR": [0.4, 0.4, 0.5], # Base RGB for synapses
        "CAMERA_PAN_SPEED_FACTOR": 0.1,
        "CAMERA_ROTATE_SPEED_FACTOR": 0.005,
        "CAMERA_ZOOM_SPEED_FACTOR": 20.0,
        "ENABLE_SYNAPTIC_PULSES": True,
        "SYNAPTIC_PULSE_COLOR": [0.7, 0.9, 1.0, 0.9], # RGBA
        "SYNAPTIC_PULSE_SIZE": 3.0,
        "SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES": 5, # Frames a pulse point lasts
    }
    TRAIT_COLOR_MAP = [ # RGBA, A is base opacity when inactive
        [0.8, 0.2, 0.2, 0.85], [0.2, 0.8, 0.2, 0.85], [0.2, 0.2, 0.8, 0.85],
        [0.8, 0.8, 0.2, 0.85], [0.8, 0.2, 0.8, 0.85], [0.2, 0.8, 0.8, 0.85],
        [1.0, 0.5, 0.0, 0.85], [0.5, 0.2, 0.8, 0.85], [0.1, 0.5, 0.5, 0.85],
        [0.7, 0.7, 0.7, 0.85]
    ]
    glut.glut_window_id = None # type: ignore
else:
    gl_data_lock = None # type: ignore
    opengl_viz_config = {}
    TRAIT_COLOR_MAP = [] # type: ignore


# --- OpenGL Visualization Functions ---
def init_gl():
    if not OPENGL_AVAILABLE: return
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo

    glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0, 0, 0, 0); # Black background
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0));
    glEnable(GL_DEPTH_TEST)

    try:
        vbo_ids = glGenBuffers(4)
        if not isinstance(vbo_ids, (list, tuple, np.ndarray)) or len(vbo_ids) < 4 :
            if isinstance(vbo_ids, int) and vbo_ids > 0: # Fallback for some GL bindings
                 gl_neuron_pos_vbo = vbo_ids
                 gl_neuron_color_vbo = glGenBuffers(1)
                 gl_synapse_vertices_vbo = glGenBuffers(1)
                 gl_pulse_vertices_vbo = glGenBuffers(1)
            else: raise ValueError("glGenBuffers did not return expected VBO IDs.")
        else:
            gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo = vbo_ids[0], vbo_ids[1], vbo_ids[2], vbo_ids[3]
    except Exception as e:
        print(f"Error: glGenBuffers failed: {e}. OpenGL visualization will likely fail.")
        gl_neuron_pos_vbo = 0; gl_neuron_color_vbo = 0; gl_synapse_vertices_vbo = 0; gl_pulse_vertices_vbo = 0; return


def reshape_gl_window(width, height):
    if not OPENGL_AVAILABLE or height <= 0 or not global_simulation_bridge: return
    cfg = global_simulation_bridge.sim_config

    opengl_viz_config['WINDOW_WIDTH'] = width; opengl_viz_config['WINDOW_HEIGHT'] = height

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(cfg.camera_fov, float(width) / float(height), cfg.camera_near_clip, cfg.camera_far_clip)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()

def render_text_gl(x, y, text, font=glut.GLUT_BITMAP_9_BY_15):
    if not OPENGL_AVAILABLE: return
    try:
        current_win = glut.glutGetWindow();
        if current_win == 0: return # No current GL context

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800); win_h = opengl_viz_config.get('WINDOW_HEIGHT', 600)
        gluOrtho2D(0, win_w, 0, win_h) # Set up 2D orthographic projection

        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glColor3f(0.9, 0.9, 0.9); # Text color
        glDisable(GL_DEPTH_TEST) # Render text on top

        glRasterPos2i(int(x), int(y)) # Position the text
        for character in text:
            glut.glutBitmapCharacter(font, ord(character))

        glEnable(GL_DEPTH_TEST); glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
    except Exception as e:
        print(f"[ERROR] OpenGL render_text_gl: {e}")


def get_color_for_trait(trait_index, activity_timer_value, is_currently_spiking, neuron_model_name_str, neuron_type_str=""):
    """Determines neuron color based on trait, activity, spiking status, and filter mode."""
    max_highlight_frames = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
    firing_rgb_config = opengl_viz_config.get("FIRING_NEURON_COLOR", [1.0, 1.0, 0.0, 1.0])
    firing_rgb = firing_rgb_config[0:3]
    base_firing_alpha = firing_rgb_config[3]
    default_inactive_alpha = opengl_viz_config.get("INACTIVE_NEURON_OPACITY", 0.25)

    base_color_rgb = [0.5, 0.5, 0.5]; base_alpha = default_inactive_alpha
    if TRAIT_COLOR_MAP and len(TRAIT_COLOR_MAP) > 0:
        color_def_from_map = TRAIT_COLOR_MAP[trait_index % len(TRAIT_COLOR_MAP)]
        base_color_rgb = color_def_from_map[0:3]
        base_alpha = color_def_from_map[3] if len(color_def_from_map) > 3 else default_inactive_alpha

    final_color_rgba = list(base_color_rgb) + [base_alpha] # Default

    filters_dict = get_current_filter_settings_from_gui() 
    spiking_mode = filters_dict.get("spiking_mode", "Highlight Spiking")

    if spiking_mode == "No Spiking Highlight":
        # No special coloring, just base trait color.
        return final_color_rgba 

    if is_currently_spiking:
        # If "Highlight Spiking" or "Show Only Spiking" and it's currently spiking.
        final_color_rgba = list(firing_rgb) + [base_firing_alpha]
    elif spiking_mode == "Highlight Spiking" and activity_timer_value > 0:
        # "Highlight Spiking" mode: not currently spiking, but has a residual timer.
        decay_ratio = max(0.0, min(1.0, float(activity_timer_value) / max_highlight_frames))
        dimmed_firing_alpha = base_firing_alpha * decay_ratio * 0.6 
        dimmed_firing_alpha = max(dimmed_firing_alpha, base_alpha * 0.8, 0.05) 
        dimmed_firing_alpha = min(base_firing_alpha * 0.8, dimmed_firing_alpha) 
        final_color_rgba = list(firing_rgb) + [dimmed_firing_alpha]
    # In "Show Only Spiking" mode, if not is_currently_spiking, it wouldn't be drawn anyway.
    # If "No Spiking Highlight", it returns base color earlier.

    return final_color_rgba

def update_gl_data():
    """Prepares neuron, synapse, and pulse data for OpenGL rendering by updating VBOs."""
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw
    global gl_neuron_pos_np, gl_neuron_colors_np, gl_connection_vertices_np, gl_pulse_vertices_np
    global global_simulation_bridge, global_viz_data_cache, opengl_viz_config, global_gui_state

    if not OPENGL_AVAILABLE or not global_simulation_bridge or not global_simulation_bridge.is_initialized:
        if gl_data_lock:
            with gl_data_lock: gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
        else: gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
        return

    sim_data_snapshot = global_simulation_bridge.get_latest_simulation_data_for_gui(force_fetch=True)
    if sim_data_snapshot is None:
        if gl_data_lock:
            with gl_data_lock: gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
        else: gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
        return

    neuron_fired_np = sim_data_snapshot["neuron_fired_status"]
    neuron_activity_timers_np = sim_data_snapshot.get("neuron_activity_timers")
    all_neuron_positions_3d_np = sim_data_snapshot["neuron_positions_3d"]
    all_neuron_types_str_list = global_simulation_bridge.sim_config.neuron_types_list_for_viz

    # Ensure neuron types list is populated if it's empty or mismatched
    if not all_neuron_types_str_list or len(all_neuron_types_str_list) != global_simulation_bridge.sim_config.num_neurons:
        viz_profile_data = global_simulation_bridge.get_profile_visualization_data(from_current_config=True)
        all_neuron_types_str_list = viz_profile_data.get("neuron_types", [])
        global_simulation_bridge.sim_config.neuron_types_list_for_viz = all_neuron_types_str_list
        global_viz_data_cache["neuron_types"] = all_neuron_types_str_list # Update cache

    current_filters = get_current_filter_settings_from_gui()
    all_indices = np.arange(global_simulation_bridge.sim_config.num_neurons)
    # Apply filters to get indices of neurons to draw
    visible_neuron_indices_np = np.array(apply_neuron_filters_to_indices(
        all_indices, neuron_fired_np, all_neuron_types_str_list, current_filters
    ), dtype=np.int32)

    max_render_neurons = opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000)
    if len(visible_neuron_indices_np) > max_render_neurons:
        chosen_neuron_indices_np = np.random.choice(visible_neuron_indices_np, size=max_render_neurons, replace=False)
    else: chosen_neuron_indices_np = visible_neuron_indices_np
    current_num_neurons_to_draw = len(chosen_neuron_indices_np)

    gl_neuron_pos_np_temp = np.array([], dtype=np.float32).reshape(0,3)
    gl_neuron_colors_np_temp = np.array([], dtype=np.float32).reshape(0,4)

    if current_num_neurons_to_draw > 0 and \
       all_neuron_positions_3d_np is not None and \
       all_neuron_positions_3d_np.ndim == 2 and \
       all_neuron_positions_3d_np.shape[0] == global_simulation_bridge.sim_config.num_neurons and \
       all_neuron_positions_3d_np.shape[1] == 3:

        gl_neuron_pos_np_temp = all_neuron_positions_3d_np[chosen_neuron_indices_np]
        gl_neuron_colors_np_temp = np.zeros((current_num_neurons_to_draw, 4), dtype=np.float32)
        neuron_traits_cp = global_simulation_bridge.cp_traits
        neuron_traits_np = cp.asnumpy(neuron_traits_cp) if neuron_traits_cp is not None and neuron_traits_cp.size == global_simulation_bridge.sim_config.num_neurons else \
                           np.zeros(global_simulation_bridge.sim_config.num_neurons, dtype=np.int32) # Fallback
        model_name_str = global_simulation_bridge.sim_config.neuron_model_type
        for i, global_idx in enumerate(chosen_neuron_indices_np):
            trait_val = neuron_traits_np[global_idx]
            neuron_type_str_val = all_neuron_types_str_list[global_idx] if global_idx < len(all_neuron_types_str_list) else ""

            is_currently_spiking = False
            if neuron_fired_np is not None and global_idx < len(neuron_fired_np):
                is_currently_spiking = neuron_fired_np[global_idx]

            activity_timer_val = 0
            if neuron_activity_timers_np is not None and global_idx < len(neuron_activity_timers_np):
                activity_timer_val = neuron_activity_timers_np[global_idx]
            gl_neuron_colors_np_temp[i] = get_color_for_trait(
                trait_val,
                activity_timer_val,
                is_currently_spiking,
                model_name_str,
                neuron_type_str_val
            )

    gl_connection_vertices_np_temp = np.array([], dtype=np.float32).reshape(0,3)
    current_num_synapse_lines_to_draw = 0
    if global_gui_state.get("show_connections_gl", False) and sim_data_snapshot["synapse_info"]:
        all_synapse_data_list = sim_data_snapshot["synapse_info"] # This is already sampled
        visible_synapse_indices_in_sample = apply_synapse_filters_to_indices(all_synapse_data_list, current_filters)
        conn_verts_list_3d = []
        chosen_global_indices_set = set(chosen_neuron_indices_np) # Use the currently visible neurons
        for local_syn_idx in visible_synapse_indices_in_sample:
            syn_data = all_synapse_data_list[local_syn_idx]
            src_global_idx, tgt_global_idx = syn_data["source_idx"], syn_data["target_idx"]
            # Check if both source and target neurons are in the set of neurons being drawn
            if src_global_idx in chosen_global_indices_set and tgt_global_idx in chosen_global_indices_set:
                if all_neuron_positions_3d_np is not None and \
                   src_global_idx < len(all_neuron_positions_3d_np) and \
                   tgt_global_idx < len(all_neuron_positions_3d_np):
                    pos_src_3d_np = all_neuron_positions_3d_np[src_global_idx]
                    pos_tgt_3d_np = all_neuron_positions_3d_np[tgt_global_idx]
                    conn_verts_list_3d.extend([pos_src_3d_np[0], pos_src_3d_np[1], pos_src_3d_np[2]])
                    conn_verts_list_3d.extend([pos_tgt_3d_np[0], pos_tgt_3d_np[1], pos_tgt_3d_np[2]])
                    current_num_synapse_lines_to_draw += 1
        if conn_verts_list_3d:
            gl_connection_vertices_np_temp = np.array(conn_verts_list_3d, dtype=np.float32).reshape(-1, 3)

    gl_pulse_vertices_np_temp = np.array([], dtype=np.float32).reshape(0,3)
    current_num_pulses_to_draw = 0
    if opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
       global_simulation_bridge.cp_synapse_pulse_timers is not None and \
       global_simulation_bridge.cp_synapse_pulse_timers.size > 0 and \
       global_simulation_bridge.cp_connections is not None and \
       global_simulation_bridge.cp_connections.nnz > 0 and \
       all_neuron_positions_3d_np is not None and \
       all_neuron_positions_3d_np.shape[0] == global_simulation_bridge.sim_config.num_neurons:

        active_pulse_indices_cp = cp.where(global_simulation_bridge.cp_synapse_pulse_timers > 0)[0]
        if active_pulse_indices_cp.size > 0:
            coo_conn = global_simulation_bridge.cp_connections.tocoo(copy=False) # Avoid copy if possible
            # Ensure indices are valid for coo_conn.row (which has size equal to nnz)
            valid_pulse_indices_mask = active_pulse_indices_cp < coo_conn.row.size
            active_pulse_indices_cp = active_pulse_indices_cp[valid_pulse_indices_mask]

            if active_pulse_indices_cp.size > 0:
                src_neuron_indices_cp = coo_conn.row[active_pulse_indices_cp]
                tgt_neuron_indices_cp = coo_conn.col[active_pulse_indices_cp]

                # Filter pulses where src/tgt neurons are among those being drawn
                # This requires mapping global neuron indices to the chosen_neuron_indices_np
                # For simplicity here, we'll draw all active pulses if their neurons exist,
                # visibility filtering of pulses themselves is not implemented yet.
                max_neuron_idx = all_neuron_positions_3d_np.shape[0] -1
                valid_src_tgt_mask = (src_neuron_indices_cp <= max_neuron_idx) & (tgt_neuron_indices_cp <= max_neuron_idx)

                src_neuron_indices_cp = src_neuron_indices_cp[valid_src_tgt_mask]
                tgt_neuron_indices_cp = tgt_neuron_indices_cp[valid_src_tgt_mask]
                active_pulse_indices_cp_final = active_pulse_indices_cp[valid_src_tgt_mask] # Use the same mask

                if src_neuron_indices_cp.size > 0:
                    neuron_pos_all_cp = cp.asarray(all_neuron_positions_3d_np) # Ensure it's a CuPy array
                    src_pos_cp = neuron_pos_all_cp[src_neuron_indices_cp]
                    tgt_pos_cp = neuron_pos_all_cp[tgt_neuron_indices_cp]

                    pulse_progress_cp = global_simulation_bridge.cp_synapse_pulse_progress[active_pulse_indices_cp_final]

                    pulse_positions_cp = src_pos_cp + pulse_progress_cp[:, cp.newaxis] * (tgt_pos_cp - src_pos_cp)
                    gl_pulse_vertices_np_temp = cp.asnumpy(pulse_positions_cp)
                    current_num_pulses_to_draw = gl_pulse_vertices_np_temp.shape[0]


    # Decrement timers for visualization effects (only in live sim mode)
    if not global_gui_state.get("is_playback_mode", False):
        if global_simulation_bridge and global_simulation_bridge.cp_viz_activity_timers is not None \
           and global_simulation_bridge.cp_viz_activity_timers.size > 0:
            needs_decrement_mask = global_simulation_bridge.cp_viz_activity_timers > 0
            global_simulation_bridge.cp_viz_activity_timers = cp.where(
                needs_decrement_mask,
                global_simulation_bridge.cp_viz_activity_timers - 1,
                global_simulation_bridge.cp_viz_activity_timers
            )

    # Update pulse timers and progress (always, as it's visual decay)
    if opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
       global_simulation_bridge and \
       global_simulation_bridge.cp_synapse_pulse_timers is not None and \
       global_simulation_bridge.cp_synapse_pulse_timers.size > 0:

        active_pulse_mask = global_simulation_bridge.cp_synapse_pulse_timers > 0
        global_simulation_bridge.cp_synapse_pulse_timers = cp.where(
            active_pulse_mask,
            global_simulation_bridge.cp_synapse_pulse_timers - 1,
            global_simulation_bridge.cp_synapse_pulse_timers
        )
        pulse_lifetime = opengl_viz_config.get("SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES", 5)
        if pulse_lifetime > 0:
            progress_increment = 1.0 / pulse_lifetime
            global_simulation_bridge.cp_synapse_pulse_progress = cp.where(
                active_pulse_mask,
                cp.clip(global_simulation_bridge.cp_synapse_pulse_progress + progress_increment, 0.0, 1.0),
                global_simulation_bridge.cp_synapse_pulse_progress
            )
            # Reset progress for pulses that just became inactive
            just_became_inactive_mask = (global_simulation_bridge.cp_synapse_pulse_timers == 0) & \
                                        (global_simulation_bridge.cp_synapse_pulse_progress > 0) # Progress was > 0
            global_simulation_bridge.cp_synapse_pulse_progress = cp.where(
                just_became_inactive_mask,
                0.0, # Reset progress to 0
                global_simulation_bridge.cp_synapse_pulse_progress
            )


    # Safely update global GL data under lock
    if gl_data_lock:
        with gl_data_lock:
            gl_num_neurons_to_draw = current_num_neurons_to_draw
            gl_neuron_pos_np = gl_neuron_pos_np_temp
            gl_neuron_colors_np = gl_neuron_colors_np_temp
            if gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0: # Check VBO ID
                glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_pos_vbo)
                glBufferData(GL_ARRAY_BUFFER, gl_neuron_pos_np.nbytes, gl_neuron_pos_np, GL_DYNAMIC_DRAW)
            if gl_neuron_color_vbo is not None and gl_neuron_color_vbo > 0: # Check VBO ID
                glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_color_vbo)
                glBufferData(GL_ARRAY_BUFFER, gl_neuron_colors_np.nbytes, gl_neuron_colors_np, GL_DYNAMIC_DRAW)

            gl_num_synapse_lines_to_draw = current_num_synapse_lines_to_draw
            gl_connection_vertices_np = gl_connection_vertices_np_temp
            if gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0: # Check VBO ID
                glBindBuffer(GL_ARRAY_BUFFER, gl_synapse_vertices_vbo)
                glBufferData(GL_ARRAY_BUFFER, gl_connection_vertices_np.nbytes, gl_connection_vertices_np, GL_DYNAMIC_DRAW)

            gl_num_pulses_to_draw = current_num_pulses_to_draw
            gl_pulse_vertices_np = gl_pulse_vertices_np_temp
            if gl_pulse_vertices_vbo is not None and gl_pulse_vertices_vbo > 0: # Check VBO ID
                glBindBuffer(GL_ARRAY_BUFFER, gl_pulse_vertices_vbo)
                glBufferData(GL_ARRAY_BUFFER, gl_pulse_vertices_np.nbytes, gl_pulse_vertices_np, GL_DYNAMIC_DRAW)

            if gl_neuron_pos_vbo is not None : glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind
    else: # No lock, direct update (should not happen if OPENGL_AVAILABLE is true)
            gl_num_neurons_to_draw = current_num_neurons_to_draw
            gl_neuron_pos_np = gl_neuron_pos_np_temp
            gl_neuron_colors_np = gl_neuron_colors_np_temp
            gl_num_synapse_lines_to_draw = current_num_synapse_lines_to_draw
            gl_connection_vertices_np = gl_connection_vertices_np_temp
            gl_num_pulses_to_draw = current_num_pulses_to_draw
            gl_pulse_vertices_np = gl_pulse_vertices_np_temp

    global_viz_data_cache["last_visible_neuron_indices"] = chosen_neuron_indices_np.tolist()
    # global_viz_data_cache["last_visible_synapse_indices"] = ... # This needs careful thought if synapses are also filtered by index

def render_scene_gl():
    global opengl_viz_config, global_gui_state, glut_window_id # type: ignore
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw

    if not OPENGL_AVAILABLE or not global_simulation_bridge: return
    try:
        current_win = glut.glutGetWindow()
        if glut.glut_window_id is not None and current_win != glut.glut_window_id and current_win != 0: # type: ignore
            glut.glutSetWindow(glut.glut_window_id) # type: ignore
        elif current_win == 0: # No window context
            return
    except Exception as e: # Catch potential errors if GLUT context is lost
        # print(f"GLUT context error in render_scene_gl: {e}")
        return

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0))

    cfg = global_simulation_bridge.sim_config
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Calculate camera eye position based on spherical coordinates (azimuth, elevation, radius)
    eye_x = cfg.camera_center_x + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.sin(cfg.camera_azimuth_angle)
    eye_y = cfg.camera_center_y + cfg.camera_radius * math.sin(cfg.camera_elevation_angle)
    eye_z = cfg.camera_center_z + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.cos(cfg.camera_azimuth_angle)

    gluLookAt(eye_x, eye_y, eye_z, # Eye position
              cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z, # Look-at point
              cfg.camera_up_x, cfg.camera_up_y, cfg.camera_up_z) # Up vector

    if gl_data_lock:
        with gl_data_lock:
            # Render Synapses
            if global_gui_state.get("show_connections_gl", False) and gl_num_synapse_lines_to_draw > 0 and \
               gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0:
                base_syn_color = opengl_viz_config.get('SYNAPSE_BASE_COLOR', [0.3,0.3,0.4])
                alpha_mod = opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.5)
                final_alpha = np.clip(0.15 * alpha_mod, 0.02, 0.5) # Ensure some visibility but not too opaque
                glColor4f(base_syn_color[0], base_syn_color[1], base_syn_color[2], final_alpha)
                glLineWidth(0.5)

                glBindBuffer(GL_ARRAY_BUFFER, gl_synapse_vertices_vbo)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, None)
                glDrawArrays(GL_LINES, 0, gl_num_synapse_lines_to_draw * 2) # Each line has 2 vertices
                glDisableClientState(GL_VERTEX_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

            # Render Neurons
            if gl_num_neurons_to_draw > 0 and \
               gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 and \
               gl_neuron_color_vbo is not None and gl_neuron_color_vbo > 0:

                glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_pos_vbo)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, None)

                glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_color_vbo)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(4, GL_FLOAT, 0, None) # 4 components (RGBA)

                glDrawArrays(GL_POINTS, 0, gl_num_neurons_to_draw)

                glDisableClientState(GL_COLOR_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

            # Render Synaptic Pulses
            if opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
               gl_num_pulses_to_draw > 0 and \
               gl_pulse_vertices_vbo is not None and gl_pulse_vertices_vbo > 0:

                pulse_color_rgba = opengl_viz_config.get("SYNAPTIC_PULSE_COLOR", [0.7, 0.9, 1.0, 0.9])
                glColor4fv(pulse_color_rgba)
                glPointSize(opengl_viz_config.get("SYNAPTIC_PULSE_SIZE", 3.0))

                glBindBuffer(GL_ARRAY_BUFFER, gl_pulse_vertices_vbo)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, None)
                glDrawArrays(GL_POINTS, 0, gl_num_pulses_to_draw)
                glDisableClientState(GL_VERTEX_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

                glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)) # Reset point size
    else: # Should not happen if OPENGL_AVAILABLE
        pass

    # Render Footer Text
    footer_h = opengl_viz_config.get('FOOTER_HEIGHT_PIXELS', 75)
    if footer_h > 0:
        line_h, margin = 15, 10
        sim_time_s = (global_simulation_bridge.sim_config.current_time_ms / 1000.0)
        render_text_gl(margin, margin + 3*line_h, f"Time: {sim_time_s:.3f} s")
        current_step = global_simulation_bridge.sim_config.current_time_step
        render_text_gl(margin, margin + 2*line_h, f"Step: {current_step}")
        avg_fr = global_simulation_bridge._mock_network_avg_firing_rate_hz
        render_text_gl(margin, margin + line_h, f"Avg Rate: {avg_fr:.2f} Hz")

        spikes_step = global_simulation_bridge._mock_num_spikes_this_step
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800)
        render_text_gl(margin + win_w // 3, margin + 2*line_h, f"Spikes/Step: {spikes_step}")

        mode_text = "Mode: Playback" if global_gui_state.get("is_playback_mode") else "Mode: Live"
        if global_gui_state.get("is_recording"): mode_text += " (Recording)"
        render_text_gl(margin + win_w // 3, margin + line_h, mode_text)

        render_text_gl(margin, margin, "LMB:Rotate, RMB:Pan, Scroll:Zoom, R:Reset Cam, N:Synapses, P:Pause, S:Step, Esc:Exit")

    glut.glutSwapBuffers()

def mouse_button_func_gl(button, state, x, y):
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.sim_config
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

    cfg.mouse_last_x = x; cfg.mouse_last_y = y
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay()

def mouse_motion_func_gl(x, y):
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.sim_config
    dx = x - cfg.mouse_last_x; dy = y - cfg.mouse_last_y

    rotate_speed = opengl_viz_config.get("CAMERA_ROTATE_SPEED_FACTOR", 0.005)
    pan_speed_config = opengl_viz_config.get("CAMERA_PAN_SPEED_FACTOR", 0.1)

    if cfg.mouse_left_button_down: # Rotate
        cfg.camera_azimuth_angle -= dx * rotate_speed
        cfg.camera_elevation_angle -= dy * rotate_speed
        # Clamp elevation to avoid flipping
        cfg.camera_elevation_angle = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, cfg.camera_elevation_angle))
    elif cfg.mouse_right_button_down: # Pan
        # Calculate camera's local right and up vectors
        eye = np.array([
            cfg.camera_center_x + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.sin(cfg.camera_azimuth_angle),
            cfg.camera_center_y + cfg.camera_radius * math.sin(cfg.camera_elevation_angle),
            cfg.camera_center_z + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.cos(cfg.camera_azimuth_angle)
        ]);
        center = np.array([cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z])
        up_world = np.array([cfg.camera_up_x, cfg.camera_up_y, cfg.camera_up_z]) # World up

        forward = center - eye; forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6: forward /= forward_norm
        else: forward = np.array([0,0,-1]) # Default forward if eye is at center

        right = np.cross(forward, up_world); right_norm = np.linalg.norm(right)
        if right_norm > 1e-6: right /= right_norm
        else: # Handle cases where forward is aligned with world up (gimbal lock like)
            if abs(forward[1]) > 0.99 : right = np.array([1,0,0]) # If looking straight up/down, right is along X
            else: # Recalculate right if forward is not aligned with Y, but cross product was zero
                right_temp = np.cross(forward, np.array([0,1,0])); right_norm_temp = np.linalg.norm(right_temp)
                right = right_temp/right_norm_temp if right_norm_temp > 1e-6 else np.array([1,0,0])


        cam_up = np.cross(right, forward) # Camera's local up vector

        pan_scale = pan_speed_config * (cfg.camera_radius / 150.0) # Scale pan speed with zoom
        pan_vector_x = -dx * right * pan_scale # Pan left/right
        pan_vector_y = dy * cam_up * pan_scale  # Pan up/down

        cfg.camera_center_x += (pan_vector_x[0] + pan_vector_y[0])
        cfg.camera_center_y += (pan_vector_x[1] + pan_vector_y[1])
        cfg.camera_center_z += (pan_vector_x[2] + pan_vector_y[2])

    cfg.mouse_last_x = x; cfg.mouse_last_y = y
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay()

def keyboard_func_gl(key, x, y):
    global global_gui_state, global_simulation_bridge, shutdown_flag
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.sim_config

    try: key_char = key.decode("utf-8").lower()
    except UnicodeDecodeError:
        if key == b'\x1b': # ESC key
            print("ESC pressed in OpenGL window. Shutting down.")
            shutdown_flag.set()
            if OPENGL_AVAILABLE and glut.glutGetWindow() != 0 :
                 try: glut.glutLeaveMainLoop()
                 except: pass # Ignore errors if already leaving
        return

    if key_char == 'n': # Toggle synapse visibility
        global_gui_state["show_connections_gl"] = not global_gui_state.get("show_connections_gl", False)
        if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_show_synapses_gl_cb"):
            dpg.set_value("filter_show_synapses_gl_cb", global_gui_state["show_connections_gl"])
        trigger_filter_update_signal() # Signal GL data update
    elif key_char == 'p': handle_pause_simulation_event() # Pause/Resume sim
    elif key_char == 's': # Step sim if paused/stopped (and not in playback)
        if (cfg.is_paused or not cfg.is_running) and not global_gui_state.get("is_playback_mode"):
            handle_step_simulation_event()
    elif key_char == 'r': # Reset camera
        cfg.camera_azimuth_angle = 0.0; cfg.camera_elevation_angle = 0.0
        cfg.camera_radius = 150.0; cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = 0.0, 0.0, 0.0
        if glut.glutGetWindow() != 0: glut.glutPostRedisplay()

# --- DPG GUI Helper Functions ---
def trigger_filter_update_signal(sender=None, app_data=None, user_data=None):
    """Sets a flag indicating that visualization filters have changed and GL data needs update."""
    global global_gui_state
    global_gui_state["filters_changed"] = True

def get_current_filter_settings_from_gui():
    """Retrieves current filter settings from DPG UI elements."""
    settings = {
        "spiking_mode": "Highlight Spiking", # Default value
        "type_filter_enabled": False,
        "selected_neuron_type": "All", 
        "min_abs_weight": 0.01 
    }
    if dpg.is_dearpygui_running(): 
        if dpg.does_item_exist("filter_spiking_mode_combo"): # New combo box
            settings["spiking_mode"] = dpg.get_value("filter_spiking_mode_combo")
        if dpg.does_item_exist("filter_type_enable_cb"):
            settings["type_filter_enabled"] = dpg.get_value("filter_type_enable_cb")
        if dpg.does_item_exist("filter_neuron_type_combo"):
            settings["selected_neuron_type"] = dpg.get_value("filter_neuron_type_combo")
        if dpg.does_item_exist("filter_min_abs_weight_slider"):
            settings["min_abs_weight"] = dpg.get_value("filter_min_abs_weight_slider")
    return settings

def apply_neuron_filters_to_indices(all_indices, fired_status_np, neuron_types_list_str, filters_dict):
    """Applies filters to a list of neuron indices."""
    visible_indices = list(all_indices) # Start with all neurons

    # Filter by spiking status
    spiking_mode = filters_dict.get("spiking_mode", "Highlight Spiking")

    if spiking_mode == "Show Only Spiking":
        if fired_status_np is not None and fired_status_np.size == len(all_indices):
            visible_indices = [i for i in visible_indices if fired_status_np[i]]

    # Filter by neuron type
    if filters_dict.get("type_filter_enabled", False):
        selected_type_str = filters_dict.get("selected_neuron_type", "All")
        if selected_type_str != "All" and neuron_types_list_str is not None and len(neuron_types_list_str) == len(all_indices):
            visible_indices = [i for i in visible_indices if i < len(neuron_types_list_str) and neuron_types_list_str[i] == selected_type_str]
    
    return visible_indices

def apply_synapse_filters_to_indices(all_synapse_data_list, filters_dict):
    """Applies filters to a list of synapse data dictionaries."""
    if not global_gui_state.get("show_connections_gl", False): return [] # If connections are globally hidden

    visible_syn_indices = []
    min_abs_w = filters_dict.get("min_abs_weight", 0.01)
    for i, syn_data in enumerate(all_synapse_data_list):
        if abs(syn_data.get("weight", 0.0)) >= min_abs_w:
            visible_syn_indices.append(i)
    return visible_syn_indices

def update_status_bar(message, color=[200,200,200]):
    """Updates the text and color of the DPG status bar."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("status_bar_text"):
        dpg.set_value("status_bar_text", message)
        dpg.configure_item("status_bar_text", color=color)

# --- DPG GUI Element Creation & Event Handlers ---

def _update_sim_config_from_ui(update_model_specific=True):
    """Updates the global_simulation_bridge.sim_config object from DPG UI elements."""
    if not global_simulation_bridge or not global_simulation_bridge.sim_config:
        print("Warning: Sim bridge or config not available in _update_sim_config_from_ui")
        return
    cfg = global_simulation_bridge.sim_config
    global_gui_state["reset_sim_needed_from_ui_change"] = True # Flag that sim needs reset

    # General parameters
    if dpg.does_item_exist("cfg_num_neurons"): cfg.num_neurons = max(0, dpg.get_value("cfg_num_neurons"))
    if dpg.does_item_exist("cfg_total_sim_time"): cfg.total_simulation_time_ms = max(0.0, dpg.get_value("cfg_total_sim_time"))
    if dpg.does_item_exist("cfg_dt_ms"): cfg.dt_ms = max(0.001, dpg.get_value("cfg_dt_ms")) # Ensure dt is positive
    if dpg.does_item_exist("cfg_seed"): cfg.seed = dpg.get_value("cfg_seed")

    if dpg.does_item_exist("cfg_neuron_model_type"):
        selected_model_name = dpg.get_value("cfg_neuron_model_type")
        if selected_model_name != cfg.neuron_model_type: # If model type changed
            cfg.neuron_model_type = selected_model_name
            # Adjust dt for HH model if it's too large
            if selected_model_name == NeuronModel.HODGKIN_HUXLEY.name and (cfg.dt_ms is None or cfg.dt_ms > 0.05) :
                cfg.dt_ms = 0.025 # Suggest a smaller dt for HH
                if dpg.does_item_exist("cfg_dt_ms"): dpg.set_value("cfg_dt_ms", cfg.dt_ms)
            # Set default neuron types when model changes
            if selected_model_name == NeuronModel.IZHIKEVICH.name:
                cfg.default_neuron_type_izh = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
            elif selected_model_name == NeuronModel.HODGKIN_HUXLEY.name:
                cfg.default_neuron_type_hh = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name

    # Connectivity
    if dpg.does_item_exist("cfg_enable_watts_strogatz"): cfg.enable_watts_strogatz = dpg.get_value("cfg_enable_watts_strogatz")
    if dpg.does_item_exist("cfg_connectivity_k"): cfg.connectivity_k = max(2, dpg.get_value("cfg_connectivity_k"))
    if dpg.does_item_exist("cfg_connectivity_p_rewire"): cfg.connectivity_p_rewire = dpg.get_value("cfg_connectivity_p_rewire")
    if dpg.does_item_exist("cfg_connections_per_neuron"): cfg.connections_per_neuron = max(0, dpg.get_value("cfg_connections_per_neuron"))

    # Synaptic parameters
    if dpg.does_item_exist("cfg_propagation_strength"): cfg.propagation_strength = dpg.get_value("cfg_propagation_strength")
    if dpg.does_item_exist("cfg_inhibitory_propagation_strength"): cfg.inhibitory_propagation_strength = dpg.get_value("cfg_inhibitory_propagation_strength")
    if dpg.does_item_exist("cfg_syn_tau_e"): cfg.syn_tau_g_e = max(0.1, dpg.get_value("cfg_syn_tau_e"))
    if dpg.does_item_exist("cfg_syn_tau_i"): cfg.syn_tau_g_i = max(0.1, dpg.get_value("cfg_syn_tau_i"))

    if dpg.does_item_exist("cfg_num_traits"): cfg.num_traits = max(1, dpg.get_value("cfg_num_traits"))

    # Learning & Plasticity
    if dpg.does_item_exist("cfg_enable_hebbian_learning"): cfg.enable_hebbian_learning = dpg.get_value("cfg_enable_hebbian_learning")
    if dpg.does_item_exist("cfg_hebbian_learning_rate"): cfg.hebbian_learning_rate = dpg.get_value("cfg_hebbian_learning_rate")
    if dpg.does_item_exist("cfg_hebbian_max_weight"): cfg.hebbian_max_weight = dpg.get_value("cfg_hebbian_max_weight")
    if dpg.does_item_exist("cfg_enable_short_term_plasticity"): cfg.enable_short_term_plasticity = dpg.get_value("cfg_enable_short_term_plasticity")
    if dpg.does_item_exist("cfg_stp_U"): cfg.stp_U = dpg.get_value("cfg_stp_U")
    if dpg.does_item_exist("cfg_stp_tau_d"): cfg.stp_tau_d = max(0.1, dpg.get_value("cfg_stp_tau_d"))
    if dpg.does_item_exist("cfg_stp_tau_f"): cfg.stp_tau_f = max(0.1, dpg.get_value("cfg_stp_tau_f"))

    # Homeostasis
    if dpg.does_item_exist("cfg_enable_homeostasis"): cfg.enable_homeostasis = dpg.get_value("cfg_enable_homeostasis")
    if dpg.does_item_exist("cfg_homeostasis_target_rate"): cfg.homeostasis_target_rate = dpg.get_value("cfg_homeostasis_target_rate")
    if dpg.does_item_exist("cfg_homeostasis_threshold_min"): cfg.homeostasis_threshold_min = dpg.get_value("cfg_homeostasis_threshold_min")
    if dpg.does_item_exist("cfg_homeostasis_threshold_max"): cfg.homeostasis_threshold_max = dpg.get_value("cfg_homeostasis_threshold_max")
    
    if dpg.does_item_exist("cfg_camera_fov"): cfg.camera_fov = dpg.get_value("cfg_camera_fov")


    if update_model_specific:
        if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            if dpg.does_item_exist("cfg_izh_C_val"): cfg.izh_C_val = dpg.get_value("cfg_izh_C_val")
            if dpg.does_item_exist("cfg_izh_k_val"): cfg.izh_k_val = dpg.get_value("cfg_izh_k_val")
            if dpg.does_item_exist("cfg_izh_vr_val"): cfg.izh_vr_val = dpg.get_value("cfg_izh_vr_val")
            if dpg.does_item_exist("cfg_izh_vt_val"): cfg.izh_vt_val = dpg.get_value("cfg_izh_vt_val")
            if dpg.does_item_exist("cfg_izh_vpeak_val"): cfg.izh_vpeak_val = dpg.get_value("cfg_izh_vpeak_val")
            if dpg.does_item_exist("cfg_izh_a_val"): cfg.izh_a_val = dpg.get_value("cfg_izh_a_val")
            if dpg.does_item_exist("cfg_izh_b_val"): cfg.izh_b_val = dpg.get_value("cfg_izh_b_val")
            if dpg.does_item_exist("cfg_izh_c_val"): cfg.izh_c_val = dpg.get_value("cfg_izh_c_val")
            if dpg.does_item_exist("cfg_izh_d_val"): cfg.izh_d_val = dpg.get_value("cfg_izh_d_val")
        elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            if dpg.does_item_exist("cfg_hh_C_m"): cfg.hh_C_m = dpg.get_value("cfg_hh_C_m")
            if dpg.does_item_exist("cfg_hh_g_Na_max"): cfg.hh_g_Na_max = dpg.get_value("cfg_hh_g_Na_max")
            if dpg.does_item_exist("cfg_hh_g_K_max"): cfg.hh_g_K_max = dpg.get_value("cfg_hh_g_K_max")
            if dpg.does_item_exist("cfg_hh_g_L"): cfg.hh_g_L = dpg.get_value("cfg_hh_g_L")
            if dpg.does_item_exist("cfg_hh_E_Na"): cfg.hh_E_Na = dpg.get_value("cfg_hh_E_Na")
            if dpg.does_item_exist("cfg_hh_E_K"): cfg.hh_E_K = dpg.get_value("cfg_hh_E_K")
            if dpg.does_item_exist("cfg_hh_E_L"): cfg.hh_E_L = dpg.get_value("cfg_hh_E_L")
            if dpg.does_item_exist("cfg_hh_v_peak"): cfg.hh_v_peak = dpg.get_value("cfg_hh_v_peak")
            if dpg.does_item_exist("cfg_hh_v_rest_init"): cfg.hh_v_rest_init = dpg.get_value("cfg_hh_v_rest_init")
            if dpg.does_item_exist("cfg_hh_q10_factor"): cfg.hh_q10_factor = dpg.get_value("cfg_hh_q10_factor")
            if dpg.does_item_exist("cfg_hh_temperature_celsius"): cfg.hh_temperature_celsius = dpg.get_value("cfg_hh_temperature_celsius")

    if dpg.does_item_exist("sim_speed_slider"): cfg.simulation_speed_factor = dpg.get_value("sim_speed_slider")

    # Update max_delay_steps based on dt
    if cfg.dt_ms > 0: cfg.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    else: cfg.max_delay_steps = 200 # Fallback if dt is invalid

    update_status_bar("Parameters changed. Press 'Apply Changes & Reset Sim' to take effect.", color=[255,165,0])

def _populate_ui_from_sim_config():
    """Populates DPG UI from sim_config and other global states."""
    if not global_simulation_bridge or not global_simulation_bridge.sim_config: return
    cfg = global_simulation_bridge.sim_config

    ui_mappings = [
        ("cfg_num_neurons", "num_neurons", 100),
        ("cfg_total_sim_time", "total_simulation_time_ms", 2000.0),
        ("cfg_dt_ms", "dt_ms", 0.1),
        ("cfg_seed", "seed", -1),
        ("cfg_neuron_model_type", "neuron_model_type", NeuronModel.IZHIKEVICH.name),
        ("cfg_enable_watts_strogatz", "enable_watts_strogatz", True),
        ("cfg_connectivity_k", "connectivity_k", 10),
        ("cfg_connectivity_p_rewire", "connectivity_p_rewire", 0.1),
        ("cfg_connections_per_neuron", "connections_per_neuron", 50),
        ("cfg_num_traits", "num_traits", 2),
        ("cfg_propagation_strength", "propagation_strength", 0.05),
        ("cfg_inhibitory_propagation_strength", "inhibitory_propagation_strength", 0.15),
        ("cfg_syn_tau_e", "syn_tau_g_e", 5.0),
        ("cfg_syn_tau_i", "syn_tau_g_i", 10.0),
        ("cfg_enable_hebbian_learning", "enable_hebbian_learning", True),
        ("cfg_hebbian_learning_rate", "hebbian_learning_rate", 0.0005),
        ("cfg_hebbian_max_weight", "hebbian_max_weight", 1.0),
        ("cfg_enable_short_term_plasticity", "enable_short_term_plasticity", True),
        ("cfg_stp_U", "stp_U", 0.15),
        ("cfg_stp_tau_d", "stp_tau_d", 200.0),
        ("cfg_stp_tau_f", "stp_tau_f", 50.0),
        ("cfg_enable_homeostasis", "enable_homeostasis", True),
        ("cfg_homeostasis_target_rate", "homeostasis_target_rate", 0.02),
        ("cfg_homeostasis_threshold_min", "homeostasis_threshold_min", -55.0),
        ("cfg_homeostasis_threshold_max", "homeostasis_threshold_max", -30.0),
        ("cfg_izh_C_val", "izh_C_val", 100.0), ("cfg_izh_k_val", "izh_k_val", 0.7),
        ("cfg_izh_vr_val", "izh_vr_val", -60.0), ("cfg_izh_vt_val", "izh_vt_val", -40.0),
        ("cfg_izh_vpeak_val", "izh_vpeak_val", 35.0), ("cfg_izh_a_val", "izh_a_val", 0.03),
        ("cfg_izh_b_val", "izh_b_val", -2.0), ("cfg_izh_c_val", "izh_c_val", -50.0),
        ("cfg_izh_d_val", "izh_d_val", 100.0),
        ("cfg_hh_C_m", "hh_C_m", 1.0), ("cfg_hh_g_Na_max", "hh_g_Na_max", 120.0),
        ("cfg_hh_g_K_max", "hh_g_K_max", 36.0), ("cfg_hh_g_L", "hh_g_L", 0.3),
        ("cfg_hh_E_Na", "hh_E_Na", 50.0), ("cfg_hh_E_K", "hh_E_K", -77.0),
        ("cfg_hh_E_L", "hh_E_L", -54.387), ("cfg_hh_v_peak", "hh_v_peak", 40.0),
        ("cfg_hh_v_rest_init", "hh_v_rest_init", -65.0),
        ("cfg_hh_q10_factor", "hh_q10_factor", 3.0),
        ("cfg_hh_temperature_celsius", "hh_temperature_celsius", 37.0),
        ("sim_speed_slider", "simulation_speed_factor", 1.0),
        ("cfg_camera_fov", "camera_fov", 60.0)
    ]
    for tag, attr, default in ui_mappings:
        if dpg.does_item_exist(tag):
            value_to_set = getattr(cfg, attr, default)
            if value_to_set is None: value_to_set = default 
            dpg.set_value(tag, value_to_set)

    _toggle_model_specific_params_visibility(None, cfg.neuron_model_type)

    if OPENGL_AVAILABLE:
        gl_ui_map = [
            ("gl_neuron_point_size_slider", 'POINT_SIZE', 2.0),
            ("gl_synapse_alpha_slider", 'SYNAPSE_ALPHA_MODIFIER', 0.3),
            ("gl_max_neurons_render_input", 'MAX_NEURONS_TO_RENDER', 10000),
            ("gl_max_connections_render_input", 'MAX_CONNECTIONS_TO_RENDER', 20000),
            ("gl_inactive_neuron_opacity_slider", 'INACTIVE_NEURON_OPACITY', 0.25),
            ("gl_activity_highlight_frames_input", 'ACTIVITY_HIGHLIGHT_FRAMES', 7),
            ("gl_enable_synaptic_pulses_cb", 'ENABLE_SYNAPTIC_PULSES', True)
        ]
        for tag, key, default in gl_ui_map:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, opengl_viz_config.get(key, default))
    
    if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_spiking_mode_combo"):
        # If a profile was loaded, apply_gui_configuration_core would have set this.
        # This line ensures a default if no profile set it, or if this func is called independently.
        current_value = dpg.get_value("filter_spiking_mode_combo")
        if not current_value: # If it's empty or None (e.g. initial setup)
             dpg.set_value("filter_spiking_mode_combo", "Highlight Spiking")

    update_status_bar("") 
    global_gui_state["reset_sim_needed_from_ui_change"] = False

def _toggle_model_specific_params_visibility(sender, app_data, user_data=None):
    """Shows/hides UI groups for Izhikevich or Hodgkin-Huxley parameters based on selection."""
    selected_model_name = app_data # This is the string name of the model

    is_izh = selected_model_name == NeuronModel.IZHIKEVICH.name
    is_hh = selected_model_name == NeuronModel.HODGKIN_HUXLEY.name

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist("izhikevich_params_group"): dpg.configure_item("izhikevich_params_group", show=is_izh)
        if dpg.does_item_exist("hodgkin_huxley_params_group"): dpg.configure_item("hodgkin_huxley_params_group", show=is_hh)
        if dpg.does_item_exist("homeostasis_izh_specific_group"): dpg.configure_item("homeostasis_izh_specific_group", show=is_izh) # Homeostasis threshold only for Izh

        # Update neuron type filter combo based on selected model
        if global_simulation_bridge and dpg.does_item_exist("filter_neuron_type_combo"):
            # Temporarily set sim_config's model type to get correct types
            original_model_in_cfg = global_simulation_bridge.sim_config.neuron_model_type
            global_simulation_bridge.sim_config.neuron_model_type = selected_model_name
            available_types_for_filter = global_simulation_bridge.get_available_neuron_types()
            global_simulation_bridge.sim_config.neuron_model_type = original_model_in_cfg # Restore

            current_filter_value = dpg.get_value("filter_neuron_type_combo")
            dpg.configure_item("filter_neuron_type_combo", items=available_types_for_filter)
            if current_filter_value in available_types_for_filter:
                dpg.set_value("filter_neuron_type_combo", current_filter_value)
            elif "All" in available_types_for_filter: # Default to "All" if available
                dpg.set_value("filter_neuron_type_combo", "All")
            elif available_types_for_filter: # Otherwise, first available type
                dpg.set_value("filter_neuron_type_combo", available_types_for_filter[0])
            else: # No types available
                dpg.set_value("filter_neuron_type_combo", "")


def handle_gl_point_size_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE: opengl_viz_config['POINT_SIZE'] = app_data; trigger_filter_update_signal()
def handle_gl_synapse_alpha_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE: opengl_viz_config['SYNAPSE_ALPHA_MODIFIER'] = app_data; trigger_filter_update_signal()
def handle_gl_activity_highlight_frames_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_frames = int(app_data)
            if new_frames >= 1: opengl_viz_config['ACTIVITY_HIGHLIGHT_FRAMES'] = new_frames
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender): # Reset to current if invalid
                dpg.set_value(sender, opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7))
        except ValueError: # If input is not an int
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


def _update_sim_config_from_ui_and_signal_reset_needed(sender=None, app_data=None, user_data=None):
    """Convenience function to call _update_sim_config_from_ui, ensuring reset flag is set."""
    _update_sim_config_from_ui() # This already sets the reset_sim_needed_from_ui_change flag

def _handle_model_type_change(sender, app_data, user_data=None):
    """Handles change in neuron model type selection."""
    _toggle_model_specific_params_visibility(sender, app_data)
    _update_sim_config_from_ui_and_signal_reset_needed() # Signal that sim needs reset

def handle_start_simulation_event(sender=None, app_data=None, user_data=None):
    if global_simulation_bridge:
        if global_gui_state.get("is_playback_mode", False):
            update_status_bar("Error: Cannot start simulation in playback mode.", color=[255,0,0])
            return
        if global_gui_state.get("reset_sim_needed_from_ui_change", False): # Check if changes need applying
            update_status_bar("Apply changes before starting!", color=[255,100,100])
            return

        global_simulation_bridge.start_simulation()
        if dpg.is_dearpygui_running():
            update_ui_for_simulation_run_state(is_running=True, is_paused=False)
            update_status_bar("Simulation started.")
        global_gui_state['_was_running_last_frame'] = False # Reset for sim loop logic

def handle_stop_simulation_event(sender=None, app_data=None, user_data=None):
    if global_simulation_bridge:
        is_currently_recording = global_gui_state.get("is_recording", False)
        was_unsaved = global_gui_state.get("unsaved_recording_exists", False)

        global_simulation_bridge.stop_simulation() # This pauses the sim logic

        if dpg.is_dearpygui_running():
            update_ui_for_simulation_run_state(is_running=False, is_paused=False)
            if is_currently_recording and was_unsaved:
                 update_status_bar("Sim stopped. Recording to is active. Finalize from Record button.", color=[255,165,0])
            else:
                 update_status_bar("Simulation stopped.")

        # Refresh monitor with initial/reset values
        initial_data_snapshot = global_simulation_bridge.get_initial_sim_data_snapshot()
        if initial_data_snapshot:
            update_monitoring_overlay_values(initial_data_snapshot)
        if OPENGL_AVAILABLE: trigger_filter_update_signal() # Update GL view

def handle_pause_simulation_event(sender=None, app_data=None, user_data=None):
    if global_simulation_bridge and global_simulation_bridge.sim_config.is_running: # Can only pause if running
        is_paused = global_simulation_bridge.toggle_pause_simulation()
        if dpg.is_dearpygui_running():
            update_ui_for_simulation_run_state(is_running=True, is_paused=is_paused)
            update_status_bar(f"Simulation {'paused' if is_paused else 'resumed'}.")


def handle_step_simulation_event(sender=None, app_data=None, user_data=None):
    if global_simulation_bridge:
        if global_gui_state.get("is_playback_mode", False):
            update_status_bar("Error: Cannot step live sim in playback mode.", color=[255,0,0])
            return

        # Allow stepping if sim is paused OR if it's not running at all (for initial steps)
        can_step = (global_simulation_bridge.sim_config.is_running and global_simulation_bridge.sim_config.is_paused) or \
                   (not global_simulation_bridge.sim_config.is_running)

        if can_step:
            if global_gui_state.get("reset_sim_needed_from_ui_change", False):
                update_status_bar("Apply changes before stepping!", color=[255,100,100])
                return

            dt_ms_val = global_simulation_bridge.sim_config.dt_ms
            if dt_ms_val is None or dt_ms_val <= 0: dt_ms_val = 0.1 # Fallback
            steps_for_1ms_approx = max(1, int(round(1.0 / dt_ms_val))) # Step roughly 1ms of sim time

            global_simulation_bridge.step_simulation(num_steps=steps_for_1ms_approx)
            trigger_filter_update_signal() # Update visuals
            latest_data = global_simulation_bridge.get_latest_simulation_data_for_gui(force_fetch=True)
            if latest_data: update_monitoring_overlay_values(latest_data) # Update monitor
            update_status_bar(f"Stepped simulation by {steps_for_1ms_approx} substeps (~1ms).")
        else:
            update_status_bar("Sim must be paused or stopped to step.", color=[255,165,0])


def handle_apply_config_changes_and_reset(sender=None, app_data=None, user_data=None, from_reset_button=False):
    update_status_bar("Applying configuration and resetting simulation...", color=[200,200,0])

    if global_simulation_bridge:
        if not from_reset_button: # If called from general UI change, ensure config is freshest
            _update_sim_config_from_ui(update_model_specific=True)

        success = global_simulation_bridge.apply_simulation_configuration_core(global_simulation_bridge.sim_config.to_dict())

        if success:
            update_status_bar("Configuration applied. Simulation has been reset.", color=[0,200,0])
            _populate_ui_from_sim_config() # Refresh UI with applied config
            handle_stop_simulation_event() # Ensure sim is in a stopped state UI-wise

            # Refresh visualization data based on new config
            profile_viz_data = global_simulation_bridge.get_profile_visualization_data(from_current_config=True)
            if profile_viz_data and "neuron_types" in profile_viz_data :
                global_viz_data_cache["neuron_types"] = profile_viz_data["neuron_types"]
                global_simulation_bridge.sim_config.neuron_types_list_for_viz = profile_viz_data["neuron_types"]

            initial_data_snapshot = global_simulation_bridge.get_initial_sim_data_snapshot()
            if initial_data_snapshot: update_monitoring_overlay_values(initial_data_snapshot)

            if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_neuron_type_combo"):
                 available_types = global_simulation_bridge.get_available_neuron_types()
                 dpg.configure_item("filter_neuron_type_combo", items=available_types)
                 dpg.set_value("filter_neuron_type_combo", "All" if "All" in available_types else (available_types[0] if available_types else ""))

            if OPENGL_AVAILABLE: trigger_filter_update_signal()
            global_gui_state["reset_sim_needed_from_ui_change"] = False # Reset the flag
        else:
            update_status_bar("Error applying configuration. Please check parameters.", color=[255,0,0])
    else:
        update_status_bar("Sim bridge not available.", color=[255,0,0])


def handle_sim_speed_change(sender, app_data, user_data):
    if global_simulation_bridge:
        global_simulation_bridge.set_simulation_speed_factor(app_data)

def get_profile_files(profile_directory): # Profiles are still JSON
    try:
        if os.path.exists(profile_directory) and os.path.isdir(profile_directory):
            return sorted([f for f in os.listdir(profile_directory) if f.endswith(".json") and os.path.isfile(os.path.join(profile_directory, f))])
    except Exception as e: print(f"Error listing profile directory '{profile_directory}': {e}")
    return []

def get_hdf5_files(directory, extension): # Helper for .simrec.h5 and .simstate.h5
    try:
        if os.path.exists(directory) and os.path.isdir(directory):
            return sorted([f for f in os.listdir(directory) if f.endswith(extension) and os.path.isfile(os.path.join(directory, f))])
    except Exception as e: print(f"Error listing directory '{directory}' for '{extension}': {e}")
    return []


def handle_save_profile_button_press(sender=None, app_data=None, user_data=None): # Profiles are JSON
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_profile_file_dialog"):
        _update_sim_config_from_ui(update_model_specific=True) # Ensure current UI state is in config
        dpg.show_item("save_profile_file_dialog")

def handle_load_profile_button_press(sender=None, app_data=None, user_data=None): # Profiles are JSON
    if global_gui_state.get("is_recording", False) or global_gui_state.get("is_playback_mode", False):
        update_status_bar("Stop recording/playback before loading a profile.", color=[255,165,0])
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_profile_file_dialog"):
        dpg.show_item("load_profile_file_dialog")

def save_profile_dialog_callback(sender, app_data): # Profiles are JSON
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        if not filepath.lower().endswith(".json"): filepath += ".json"

        sim_config_dict_to_save = global_simulation_bridge.sim_config.to_dict()
        # Remove runtime state from saved profile
        keys_to_remove = ["neuron_positions_x", "neuron_positions_y", "neuron_types_list_for_viz",
                          "current_time_ms", "current_time_step", "is_running", "is_paused", "max_delay_steps"]
        for key in keys_to_remove:
            if key in sim_config_dict_to_save: del sim_config_dict_to_save[key]

        gui_settings_to_save = get_current_gui_configuration_dict() # Get current GUI settings
        content_to_save = {"simulation_configuration": sim_config_dict_to_save, "gui_configuration": gui_settings_to_save}

        try:
            with open(filepath, 'w') as f: json.dump(content_to_save, f, indent=4)
            update_status_bar(f"Profile saved: {os.path.basename(filepath)}", color=[0,200,0])
            if dpg.does_item_exist("profile_name_input"): # Update UI field if exists
                dpg.set_value("profile_name_input", os.path.basename(filepath).replace(".json", ""))
            global_gui_state["current_profile_name"] = os.path.basename(filepath)
        except Exception as e: update_status_bar(f"Error saving profile: {e}", color=[255,0,0])
    else: update_status_bar("Save profile cancelled.")


def _execute_profile_load(filepath): # Profiles are JSON
    profile_name = os.path.basename(filepath)
    update_status_bar(f"Loading profile '{profile_name}'...", color=[200,200,0])
    if global_simulation_bridge:
        try:
            with open(filepath, 'r') as f: profile_content = json.load(f)
            sim_cfg_data = profile_content.get("simulation_configuration")
            gui_cfg_data = profile_content.get("gui_configuration")

            if sim_cfg_data:
                if global_simulation_bridge.apply_simulation_configuration_core(sim_cfg_data): # This resets the sim
                    _populate_ui_from_sim_config() # Update UI with loaded sim config
                    if gui_cfg_data: apply_gui_configuration_core(gui_cfg_data) # Apply GUI settings

                    update_status_bar(f"Profile '{profile_name}' loaded. Sim reset.", color=[0,200,0])
                    if dpg.does_item_exist("profile_name_input"):
                        dpg.set_value("profile_name_input", profile_name.replace(".json", ""))
                    global_gui_state["current_profile_name"] = profile_name

                    handle_stop_simulation_event() # Ensure sim is stopped and UI reflects this
                    if OPENGL_AVAILABLE: trigger_filter_update_signal() # Update GL view
                    global_gui_state["reset_sim_needed_from_ui_change"] = False
                else: update_status_bar("Error: Invalid sim config in profile.", color=[255,0,0])
            else: update_status_bar("Error: Profile missing 'simulation_configuration'.", color=[255,0,0])
        except Exception as e:
            update_status_bar(f"Error loading profile: {e}", color=[255,0,0]); import traceback; traceback.print_exc()

def load_profile_dialog_callback(sender, app_data): # Profiles are JSON
    if "file_path_name" in app_data and app_data["file_path_name"]:
        _execute_profile_load(app_data["file_path_name"])
    else: update_status_bar("Load profile cancelled.")

def handle_save_checkpoint_button_press(sender, app_data, user_data): # Checkpoints are HDF5
    if global_gui_state.get("is_playback_mode", False):
        update_status_bar("Error: Cannot save checkpoint in playback mode.", color=[255,0,0])
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_checkpoint_file_dialog_h5"):
        dpg.show_item("save_checkpoint_file_dialog_h5")

def save_checkpoint_dialog_callback_h5(sender, app_data): # Checkpoints are HDF5
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        if not filepath.lower().endswith(".simstate.h5"): filepath += ".simstate.h5"
        if global_simulation_bridge.save_checkpoint(filepath): # save_checkpoint is now HDF5 aware
            update_status_bar(f"Checkpoint saved: {os.path.basename(filepath)}", color=[0,200,0])
        else: update_status_bar("Error saving checkpoint.", color=[255,0,0])
    else: update_status_bar("Save checkpoint cancelled.")

def handle_load_checkpoint_button_press(sender, app_data, user_data): # Checkpoints are HDF5
    if global_gui_state.get("is_recording", False) or global_gui_state.get("is_playback_mode", False):
        update_status_bar("Stop recording/playback before loading a checkpoint.", color=[255,165,0])
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_checkpoint_file_dialog_h5"):
        dpg.show_item("load_checkpoint_file_dialog_h5")

def load_checkpoint_dialog_callback_h5(sender, app_data): # Checkpoints are HDF5
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        if global_simulation_bridge.load_checkpoint(filepath): # load_checkpoint is HDF5 aware
            _populate_ui_from_sim_config() # Update UI from loaded config
            # Apply GUI settings that might have been part of the checkpoint (minimal for now)
            gui_cfg_from_chkpt = {
                "filter_settings": global_gui_state.get("global_gui_state_filters", {}), # Loaded by load_checkpoint
                "opengl_visualization_settings": opengl_viz_config.copy() # Loaded by load_checkpoint
            }
            apply_gui_configuration_core(gui_cfg_from_chkpt)

            update_status_bar(f"Checkpoint '{os.path.basename(filepath)}' loaded.", color=[0,200,0])
            if dpg.does_item_exist("profile_name_input"): # Update profile name field to reflect loaded state
                dpg.set_value("profile_name_input", f"state_{os.path.basename(filepath).replace('.simstate.h5','')}")
            global_gui_state["current_profile_name"] = f"state_{os.path.basename(filepath)}"

            handle_stop_simulation_event() # Ensure UI reflects stopped sim state
            if OPENGL_AVAILABLE: trigger_filter_update_signal()
            global_gui_state["reset_sim_needed_from_ui_change"] = False
        else: update_status_bar("Error loading checkpoint.", color=[255,0,0])
    else: update_status_bar("Load checkpoint cancelled.")

def get_current_gui_configuration_dict():
    """Gets current GUI settings, including filters and OpenGL viz config."""
    dpg_filters = get_current_filter_settings_from_gui() # This now includes "spiking_mode"
    # "spiking_only" is no longer directly used from checkbox.
    dpg_filters["show_synapses_cb"] = global_gui_state.get("show_connections_gl", False)

    current_gl_config = opengl_viz_config.copy() if OPENGL_AVAILABLE else {}
    if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_camera_fov") and global_simulation_bridge:
         current_gl_config["CAMERA_FOV_DPG_Snapshot"] = global_simulation_bridge.sim_config.camera_fov

    return {"filter_settings": dpg_filters, "opengl_visualization_settings": current_gl_config}

def apply_gui_configuration_core(gui_cfg_dict):
    """Applies a dictionary of GUI settings to the DPG UI."""
    if not gui_cfg_dict or not dpg.is_dearpygui_running(): return False

    filter_settings = gui_cfg_dict.get("filter_settings", {})
    if dpg.does_item_exist("filter_spiking_mode_combo"): # Updated for combo box
        dpg.set_value("filter_spiking_mode_combo", filter_settings.get("spiking_mode", "Highlight Spiking"))

    type_filter_enabled = filter_settings.get("type_filter_enabled", False)
    if dpg.does_item_exist("filter_type_enable_cb"): dpg.set_value("filter_type_enable_cb", type_filter_enabled)
    if dpg.does_item_exist("filter_neuron_type_combo"):
        dpg.configure_item("filter_neuron_type_combo", enabled=type_filter_enabled) 
        if global_simulation_bridge: 
            available_types = global_simulation_bridge.get_available_neuron_types()
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
        if loaded_gl_settings: opengl_viz_config.update(loaded_gl_settings) 

        gl_settings_to_apply = [
            ("gl_neuron_point_size_slider", 'POINT_SIZE', 2.0),
            ("gl_synapse_alpha_slider", 'SYNAPSE_ALPHA_MODIFIER', 0.3),
            ("gl_max_neurons_render_input", 'MAX_NEURONS_TO_RENDER', 10000),
            ("gl_max_connections_render_input", 'MAX_CONNECTIONS_TO_RENDER', 20000),
            ("gl_inactive_neuron_opacity_slider", 'INACTIVE_NEURON_OPACITY', 0.25),
            ("gl_activity_highlight_frames_input", 'ACTIVITY_HIGHLIGHT_FRAMES', 7),
            ("gl_enable_synaptic_pulses_cb", 'ENABLE_SYNAPTIC_PULSES', True)
        ]
        for tag, key, default in gl_settings_to_apply:
            if dpg.does_item_exist(tag): dpg.set_value(tag, opengl_viz_config.get(key, default))
        
        if global_simulation_bridge and dpg.does_item_exist("cfg_camera_fov") and "CAMERA_FOV_DPG_Snapshot" in opengl_viz_config:
            dpg.set_value("cfg_camera_fov", opengl_viz_config["CAMERA_FOV_DPG_Snapshot"])
        elif global_simulation_bridge and dpg.does_item_exist("cfg_camera_fov"): 
             dpg.set_value("cfg_camera_fov", global_simulation_bridge.sim_config.camera_fov)

    trigger_filter_update_signal(); return True

def update_monitoring_overlay_values(sim_data_dict):
    """Updates the monitoring overlay text elements with current simulation data."""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist("monitor_sim_time_text"): return

    if sim_data_dict is None: # If no data, show N/A
        for tag_sfx in ["sim_time_text", "current_step_text", "step_spikes_text", "avg_firerate_text", "plasticity_updates_text"]:
            tag, label = f"monitor_{tag_sfx}", tag_sfx.replace('_text', '').replace('_', ' ').title()
            if dpg.does_item_exist(tag): dpg.set_value(tag, f"{label}: N/A")
    else:
        dpg.set_value("monitor_sim_time_text", f"Sim Time: {sim_data_dict.get('current_time_ms', 0)/1000.0:.3f} s")
        dpg.set_value("monitor_current_step_text", f"Current Step: {sim_data_dict.get('current_time_step', 0)}")
        dpg.set_value("monitor_step_spikes_text", f"Spikes (step): {sim_data_dict.get('num_spikes_this_step', 0)}")
        dpg.set_value("monitor_avg_firerate_text", f"Avg Rate (net): {sim_data_dict.get('network_avg_firing_rate_hz', 0.0):.2f} Hz")
        dpg.set_value("monitor_plasticity_updates_text", f"Plasticity Evts: {sim_data_dict.get('total_plasticity_events', 0)}")

    # Update GL specific monitor values
    vis_neurons_gl = gl_num_neurons_to_draw if OPENGL_AVAILABLE else 'N/A'
    vis_syns_gl = gl_num_synapse_lines_to_draw if OPENGL_AVAILABLE else 'N/A'
    if dpg.does_item_exist("monitor_visible_neurons_text"): dpg.set_value("monitor_visible_neurons_text", f"Visible Neurons: {vis_neurons_gl}")
    if dpg.does_item_exist("monitor_visible_synapses_text"): dpg.set_value("monitor_visible_synapses_text", f"Visible Synapses: {vis_syns_gl}")

    # Update playback frame counter if in playback mode
    if global_gui_state.get("is_playback_mode") and dpg.does_item_exist("playback_current_frame_text"):
        active_rec_meta = global_gui_state.get("active_recording_data_source")
        if active_rec_meta and "num_frames" in active_rec_meta:
            total_frames = active_rec_meta["num_frames"]
            current_frame_idx = global_gui_state.get("current_playback_frame_index",0)
            dpg.set_value("playback_current_frame_text", f"Frame: {current_frame_idx + 1} / {total_frames}")

# --- DPG Event Handlers for Recording & Playback (HDF5) ---

def handle_record_button_click(sender=None, app_data=None, user_data=None):
    """Handles the 'Record' / 'Finalize Recording' button click."""
    if not global_simulation_bridge: return

    if global_gui_state.get("is_recording", False): # If currently recording, stop it
        global_simulation_bridge.stop_recording() # This will also handle UI updates
    else: # If not recording, prompt for file and start
        if global_gui_state.get("is_playback_mode", False):
            update_status_bar("Error: Cannot record while in playback mode.", color=[255,0,0])
            return
        if dpg.is_dearpygui_running() and dpg.does_item_exist("save_recording_file_dialog_h5"):
            dpg.show_item("save_recording_file_dialog_h5")
        else:
            global_simulation_bridge.log_message("Error: Recording file dialog not found.", "error")
            update_status_bar("Error: Recording dialog missing.", color=[255,0,0])

def save_recording_for_streaming_dialog_callback_h5(sender, app_data):
    """Callback for the 'Record' file dialog."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        if not filepath.lower().endswith(".simrec.h5"): filepath += ".simrec.h5"

        if global_simulation_bridge:
            global_simulation_bridge.start_recording_to_file(filepath) # This handles its own UI updates
    elif dpg.is_dearpygui_running():
        update_status_bar("Recording setup cancelled.")


def handle_playback_button_click(sender=None, app_data=None, user_data=None):
    """Handles the 'Playback' / 'Stop Playback' button click."""
    if not global_simulation_bridge: return

    if global_gui_state.get("is_playback_mode", False):
        global_simulation_bridge.exit_playback_mode()
    else:
        if global_gui_state.get("is_recording", False) and global_gui_state.get("unsaved_recording_exists", False) :
            update_status_bar("Error: Finalize active recording before entering playback mode.", color=[255,165,0])
            return
        # Check if there's valid loaded HDF5 recording data
        loaded_data = global_gui_state.get("loaded_recording_data")
        if not loaded_data or not loaded_data.get("h5_file_obj_for_playback") or not loaded_data.get("h5_file_obj_for_playback").id:
            update_status_bar("No valid recording loaded for playback. Load one first.", color=[255,165,0])
            return
        global_simulation_bridge.enter_playback_mode()


def handle_load_recording_menu_click(sender=None, app_data=None, user_data=None):
    """Handles the 'File > Load Recording' menu item click."""
    if global_gui_state.get("is_recording", False) and global_gui_state.get("unsaved_recording_exists", False):
        update_status_bar("Finalize (stop) current recording before loading another.", color=[255,165,0])
        return
    if global_gui_state.get("is_playback_mode", False): # If already in playback, exit first
        global_simulation_bridge.exit_playback_mode()
        # Add a small delay or ensure UI updates before showing dialog if needed, though usually DPG handles it.
        # update_status_bar("Exited current playback. Ready to load new HDF5 recording.", color=[200,200,0])


    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_recording_file_dialog_h5"):
        dpg.show_item("load_recording_file_dialog_h5")

def load_recording_dialog_callback_h5(sender, app_data):
    """Callback for the 'Load Recording' file dialog."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        if global_simulation_bridge:
            global_simulation_bridge.load_recording(filepath) # This handles UI updates internally
    elif dpg.is_dearpygui_running():
        update_status_bar("Load recording cancelled.")

def handle_playback_slider_change(sender, frame_idx_from_slider, user_data=None):
    if global_simulation_bridge and global_gui_state.get("is_playback_mode", False):
        if global_gui_state.get("playback_is_playing", False): # If user moves slider while playing, pause.
            global_gui_state["playback_is_playing"] = False
            if dpg.is_dearpygui_running(): update_ui_for_playback_playing_state(is_playing=False)

        global_simulation_bridge.set_playback_frame(int(frame_idx_from_slider), update_slider_gui=False) # No need to update slider from itself

def handle_playback_play_pause_button_click(sender=None, app_data=None, user_data=None):
    if not global_simulation_bridge or not global_gui_state.get("is_playback_mode", False): return

    global_gui_state["playback_is_playing"] = not global_gui_state.get("playback_is_playing", False)

    if global_gui_state["playback_is_playing"]:
        global_gui_state["last_playback_autostep_time"] = time.perf_counter()
        # If at the end of playback, loop back to the beginning
        active_rec_meta = global_gui_state.get("active_recording_data_source")
        if active_rec_meta and "num_frames" in active_rec_meta:
            num_frames = active_rec_meta["num_frames"]
            if num_frames > 0 and global_gui_state.get("current_playback_frame_index", 0) >= num_frames - 1:
                global_simulation_bridge.set_playback_frame(0, update_slider_gui=True)

    if dpg.is_dearpygui_running():
        update_ui_for_playback_playing_state(is_playing=global_gui_state["playback_is_playing"])
        status_msg = "Playback started..." if global_gui_state["playback_is_playing"] else "Playback paused."
        update_status_bar(status_msg)

def handle_playback_step_frames_click(sender, app_data, user_data):
    """Handles clicks for playback step buttons (-5, -1, +1, +5 frames)."""
    if not global_simulation_bridge or not global_gui_state.get("is_playback_mode", False):
        return

    step_amount = user_data # user_data will be -5, -1, 1, or 5
    if isinstance(step_amount, int):
        # If playback is currently playing, pause it before stepping manually
        if global_gui_state.get("playback_is_playing", False):
            global_gui_state["playback_is_playing"] = False
            if dpg.is_dearpygui_running():
                update_ui_for_playback_playing_state(is_playing=False)
                update_status_bar("Playback paused for manual step.")
        
        global_simulation_bridge.step_playback(step_amount)

def handle_playback_stop_button_click(sender=None, app_data=None, user_data=None): # This is the main "Stop Playback" button
    if global_simulation_bridge and global_gui_state.get("is_playback_mode", False):
        global_simulation_bridge.exit_playback_mode()

# --- GUI Update Helper Functions for Recording/Playback States ---

def update_ui_for_simulation_run_state(is_running, is_paused):
    """Updates DPG UI elements based on the simulation's run/pause state."""
    if not dpg.is_dearpygui_running(): return

    is_playback = global_gui_state.get("is_playback_mode", False)
    # is_recording_active_file_open: True if an HDF5 file is currently open for writing
    is_recording_active_file_open = global_gui_state.get("is_recording", False) and global_gui_state.get("unsaved_recording_exists", False)

    dpg.configure_item("start_button", enabled=not is_running and not is_playback)
    dpg.configure_item("pause_button", enabled=is_running and not is_playback, label="Resume" if is_paused else "Pause")
    dpg.configure_item("stop_button", enabled=is_running and not is_playback) # Stop live sim
    dpg.configure_item("step_button", enabled=(is_paused or not is_running) and not is_playback)

    # Apply config button: disabled if sim is running, or in playback, or actively recording to an open file
    can_apply_config = not is_running and not is_playback and not is_recording_active_file_open
    dpg.configure_item("apply_config_button", enabled=can_apply_config)

    # Record button: label managed by update_ui_for_recording_state.
    # Enabled if not in playback. If recording is active, it means "Finalize Recording".
    dpg.configure_item("record_button", enabled=not is_playback)

    # Playback button: enabled if NOT actively recording to an open file AND
    # ( (a recording is loaded AND valid) OR playback is already active (to allow stopping it) )
    loaded_data_meta = global_gui_state.get("loaded_recording_data")
    is_valid_recording_loaded = loaded_data_meta and \
                                loaded_data_meta.get("h5_file_obj_for_playback") and \
                                loaded_data_meta.get("h5_file_obj_for_playback").id and \
                                loaded_data_meta.get("num_frames", 0) > 0

    can_initiate_playback = not is_recording_active_file_open and (is_valid_recording_loaded or is_playback)
    dpg.configure_item("playback_button", enabled=can_initiate_playback)
    dpg.set_item_label("playback_button", "Stop Playback" if is_playback else "Playback Recording")


def update_ui_for_recording_state(is_recording): # is_recording means HDF5 file stream is active
    """Updates UI elements related to recording state (e.g., Record button label)."""
    if not dpg.is_dearpygui_running(): return

    dpg.set_item_label("record_button", "Finalize Recording" if is_recording else "Record")

    if global_simulation_bridge: # Re-evaluate other controls based on recording state
        update_ui_for_simulation_run_state(
            is_running=global_simulation_bridge.sim_config.is_running,
            is_paused=global_simulation_bridge.sim_config.is_paused
        )

def update_ui_for_playback_state(is_playback, num_frames=0): # num_frames is less critical here now
    """Updates UI elements when entering or exiting active playback mode."""
    if not dpg.is_dearpygui_running(): return

    # Main Playback/Stop Playback button label and state
    if dpg.does_item_exist("playback_button"):
        dpg.set_item_label("playback_button", "Stop Playback" if is_playback else "Playback Recording")
        if is_playback:
            dpg.configure_item("playback_button", enabled=True)

    # Enable/disable step buttons and play/pause within the playback_controls_group
    step_buttons_active = is_playback
    if dpg.does_item_exist("playback_step_minus_5"): dpg.configure_item("playback_step_minus_5", enabled=step_buttons_active)
    if dpg.does_item_exist("playback_step_minus_1"): dpg.configure_item("playback_step_minus_1", enabled=step_buttons_active)
    if dpg.does_item_exist("playback_play_pause_button"): dpg.configure_item("playback_play_pause_button", enabled=step_buttons_active)
    if dpg.does_item_exist("playback_step_plus_1"): dpg.configure_item("playback_step_plus_1", enabled=step_buttons_active)
    if dpg.does_item_exist("playback_step_plus_5"): dpg.configure_item("playback_step_plus_5", enabled=step_buttons_active)

    if is_playback:
        # If entering active playback, ensure slider and frame text are set for frame 0
        loaded_data_meta = global_gui_state.get("active_recording_data_source")
        actual_num_frames = loaded_data_meta.get("num_frames", 0) if loaded_data_meta else 0 # Use num_frames passed if no meta
        
        slider_max = max(0, actual_num_frames - 1)
        global_gui_state["playback_slider_max_value"] = slider_max # Store for reference
        
        if dpg.does_item_exist("playback_slider"):
            dpg.configure_item("playback_slider", max_value=slider_max, enabled=True)
            dpg.set_value("playback_slider", 0) # Reset slider to beginning
        if dpg.does_item_exist("playback_current_frame_text"):
             dpg.set_value("playback_current_frame_text", f"Frame: 1 / {actual_num_frames if actual_num_frames > 0 else 1}")

        update_ui_for_playback_playing_state(is_playing=False) # Ensure playback starts paused

        # Disable live simulation controls and conflicting file operations during active playback
        dpg.configure_item("record_button", enabled=False)
        dpg.configure_item("start_button", enabled=False)
        dpg.configure_item("pause_button", enabled=False, label="Pause") # Keep label consistent
        dpg.configure_item("stop_button", enabled=False)
        dpg.configure_item("step_button", enabled=False)
        dpg.configure_item("apply_config_button", enabled=False)
        
        if dpg.does_item_exist("save_profile_menu"): dpg.configure_item("save_profile_menu", enabled=False)
        if dpg.does_item_exist("load_profile_menu"): dpg.configure_item("load_profile_menu", enabled=False)
        if dpg.does_item_exist("save_checkpoint_menu_h5"): dpg.configure_item("save_checkpoint_menu_h5", enabled=False)
        if dpg.does_item_exist("load_checkpoint_menu_h5"): dpg.configure_item("load_checkpoint_menu_h5", enabled=False)
        if dpg.does_item_exist("load_recording_menu_h5"): dpg.configure_item("load_recording_menu_h5", enabled=False)

    else: # Exiting active playback mode
        is_recording_loaded = global_gui_state.get("loaded_recording_data") is not None and \
                              global_gui_state.get("loaded_recording_data", {}).get("h5_file_obj_for_playback") is not None and \
                              global_gui_state.get("loaded_recording_data", {}).get("h5_file_obj_for_playback").id
        
        update_ui_for_loaded_recording(can_playback=is_recording_loaded)
        
        # Ensure the main simulation controls are reset to a sensible default (usually "stopped" state)
        update_ui_for_simulation_run_state(is_running=False, is_paused=False)

        # Re-enable file operations that were disabled during playback
        if dpg.does_item_exist("save_profile_menu"): dpg.configure_item("save_profile_menu", enabled=True)
        if dpg.does_item_exist("load_profile_menu"): dpg.configure_item("load_profile_menu", enabled=True)
        if dpg.does_item_exist("save_checkpoint_menu_h5"): dpg.configure_item("save_checkpoint_menu_h5", enabled=True)
        if dpg.does_item_exist("load_checkpoint_menu_h5"): dpg.configure_item("load_checkpoint_menu_h5", enabled=True)
        if dpg.does_item_exist("load_recording_menu_h5"): dpg.configure_item("load_recording_menu_h5", enabled=True)

def update_ui_for_playback_playing_state(is_playing):
    """Updates the Play/Pause button label within the playback controls."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("playback_play_pause_button"):
         dpg.set_item_label("playback_play_pause_button", "Pause Playback" if is_playing else "Play Recording")

def update_ui_for_loaded_recording(can_playback):
    """
    Shows/hides playback controls and updates related UI based on whether a valid recording is loaded.
    """
    if not dpg.is_dearpygui_running(): return

    is_rec_file_open = global_gui_state.get("is_recording", False) and global_gui_state.get("unsaved_recording_exists", False)

    # Main Playback button state
    if dpg.does_item_exist("playback_button"):
        # Enable if a recording can be played AND not currently recording to a file.
        dpg.configure_item("playback_button", enabled=(can_playback and not is_rec_file_open))
        # Label is managed by update_ui_for_playback_state when active playback starts/stops.
        # If not in active playback, but a recording is loaded, it should say "Playback..."
        if not global_gui_state.get("is_playback_mode", False) and can_playback:
            dpg.set_item_label("playback_button", "Playback Recording")
        elif not can_playback: # No recording loaded
             dpg.set_item_label("playback_button", "Playback Recording")

    # Visibility of the playback controls group
    if dpg.does_item_exist("playback_controls_group"):
        dpg.configure_item("playback_controls_group", show=can_playback)

    if can_playback:
        loaded_data_meta = global_gui_state.get("loaded_recording_data") # Should be valid if can_playback is true
        actual_num_frames = loaded_data_meta.get("num_frames", 0) if loaded_data_meta else 0
        slider_max = max(0, actual_num_frames - 1)
        # current_playback_frame_index should be 0 if just loaded
        current_frame_idx = global_gui_state.get("current_playback_frame_index", 0) 

        if dpg.does_item_exist("playback_slider"):
            dpg.configure_item("playback_slider", max_value=slider_max, enabled=True)
            dpg.set_value("playback_slider", current_frame_idx) 
        if dpg.does_item_exist("playback_current_frame_text"):
            dpg.set_value("playback_current_frame_text", f"Frame: {current_frame_idx + 1} / {actual_num_frames if actual_num_frames > 0 else 1}")

        # Enable step buttons and play/pause if playback controls are shown
        step_buttons_enabled = True 
        if dpg.does_item_exist("playback_step_minus_5"): dpg.configure_item("playback_step_minus_5", enabled=step_buttons_enabled)
        if dpg.does_item_exist("playback_step_minus_1"): dpg.configure_item("playback_step_minus_1", enabled=step_buttons_enabled)
        if dpg.does_item_exist("playback_play_pause_button"): 
            dpg.configure_item("playback_play_pause_button", enabled=step_buttons_enabled)
            # Ensure play/pause label is correct if we just made controls visible
            update_ui_for_playback_playing_state(is_playing=global_gui_state.get("playback_is_playing", False))
        if dpg.does_item_exist("playback_step_plus_1"): dpg.configure_item("playback_step_plus_1", enabled=step_buttons_enabled)
        if dpg.does_item_exist("playback_step_plus_5"): dpg.configure_item("playback_step_plus_5", enabled=step_buttons_enabled)
    else: # No recording loaded, ensure playback controls are hidden and relevant buttons disabled
        if dpg.does_item_exist("playback_slider"): dpg.configure_item("playback_slider", enabled=False)
        if dpg.does_item_exist("playback_play_pause_button"): dpg.configure_item("playback_play_pause_button", enabled=False)
        # Step buttons are already handled by the show=False on the group.

# --- Main DPG GUI Layout Creation ---
def create_gui_layout():
    global global_simulation_bridge # Used to get directory paths

    with dpg.window(label="Controls & Configuration", tag="controls_monitor_window",
                    width=-1, height=-1, pos=[0,0], # Fill available space
                    on_close=lambda: (shutdown_flag.set(), dpg.stop_dearpygui() if dpg.is_dearpygui_running() else None), # Handle window close
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
            dpg.add_text("Status: Idle", tag="status_bar_text") # General status updates

        with dpg.collapsing_header(label="Simulation Controls", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="start_button", callback=handle_start_simulation_event, width = -1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Pause", tag="pause_button", callback=handle_pause_simulation_event, width=100, enabled=False)
                dpg.add_button(label="Stop", tag="stop_button", callback=handle_stop_simulation_event, width=100, enabled=False)
                dpg.add_button(label="Step (1ms)", tag="step_button", callback=handle_step_simulation_event, width=100, enabled=True)

            dpg.add_button(label="Apply Changes & Reset Sim", tag="apply_config_button", callback=handle_apply_config_changes_and_reset, width=-1)
            dpg.add_text("Simulation Speed:")
            dpg.add_slider_float(label="", tag="sim_speed_slider", default_value=1.0, min_value=0.01, max_value=20.0, width=-1, callback=handle_sim_speed_change, format="%.2f x")

            dpg.add_separator()
            dpg.add_text("Recording & Playback:")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Record", tag="record_button", callback=handle_record_button_click, width = -1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Playback Recording", tag="playback_button", callback=handle_playback_button_click, width = -1)

            with dpg.group(tag="playback_controls_group", show=False): # Hidden by default
                dpg.add_text("Playback Controls:")
                dpg.add_text("Frame: 0 / 0", tag="playback_current_frame_text")
                dpg.add_slider_int(label="", tag="playback_slider", width=-1, callback=handle_playback_slider_change, min_value=0, max_value=0) # Max value updated dynamically
            with dpg.group(horizontal=True):
                dpg.add_button(label="<< -5", tag="playback_step_minus_5", callback=handle_playback_step_frames_click, user_data=-5, width=70)
                dpg.add_button(label="< -1", tag="playback_step_minus_1", callback=handle_playback_step_frames_click, user_data=-1, width=70)
                dpg.add_button(label="Play/Pause", tag="playback_play_pause_button", callback=handle_playback_play_pause_button_click, width = 200) 
                dpg.add_button(label="+1 >", tag="playback_step_plus_1", callback=handle_playback_step_frames_click, user_data=1, width=70)
                dpg.add_button(label="+5 >>", tag="playback_step_plus_5", callback=handle_playback_step_frames_click, user_data=5, width=70)

        dpg.add_spacer(height=5); dpg.add_separator(); dpg.add_spacer(height=5)

        with dpg.collapsing_header(label="Core Simulation Parameters", default_open=False, tag="core_sim_params_header"):
            def add_labeled_input(label_text, item_callable, **kwargs): # Helper for cleaner layout
                dpg.add_text(label_text)
                return item_callable(label="", **kwargs) # Pass kwargs like tag, default_value, etc.

            add_labeled_input("Number of Neurons:", dpg.add_input_int, tag="cfg_num_neurons", default_value=100, width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Connections/Neuron (Spatial Fallback):", dpg.add_input_int, tag="cfg_connections_per_neuron", default_value=50, width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Total Sim Time (ms):", dpg.add_input_float, tag="cfg_total_sim_time", default_value=2000.0, width=-1, step=100, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Time Step dt (ms):", dpg.add_input_float, tag="cfg_dt_ms", default_value=0.1, width=-1, step=0.001, format="%.3f", min_value=0.001, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Seed (-1 for random):", dpg.add_input_int, tag="cfg_seed", default_value=-1, width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Number of Traits:", dpg.add_input_int, tag="cfg_num_traits", default_value=2, min_value=1, max_value=len(TRAIT_COLOR_MAP) if TRAIT_COLOR_MAP else 10, width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)

            add_labeled_input("Neuron Model:", dpg.add_combo, items=[model.name for model in NeuronModel], tag="cfg_neuron_model_type", default_value=NeuronModel.IZHIKEVICH.name, width=-1, callback=_handle_model_type_change)

            with dpg.group(tag="izhikevich_params_group", show=True): # Default show for Izhikevich
                dpg.add_text("--- Izhikevich 2007 Model Parameters ---", color=[200,200,100])
                ui_izh_params = [
                    ("Membrane Capacitance C (pF)", "cfg_izh_C_val", "%.1f"),
                    ("Scaling Factor k (nS/mV)", "cfg_izh_k_val", "%.2f"),
                    ("Resting Potential vr (mV)", "cfg_izh_vr_val", "%.1f"),
                    ("Threshold Potential vt (mV)", "cfg_izh_vt_val", "%.1f"),
                    ("Spike Peak/Cutoff vpeak (mV)", "cfg_izh_vpeak_val", "%.1f"),
                    ("Recovery Time Scale a (/ms)", "cfg_izh_a_val", "%.3f"),
                    ("Recovery Sensitivity b (nS)", "cfg_izh_b_val", "%.2f"),
                    ("Voltage Reset c (mV)", "cfg_izh_c_val", "%.1f"),
                    ("Recovery Increment d (pA)", "cfg_izh_d_val", "%.1f")
                ]
                for desc_label, tag, fmt in ui_izh_params:
                    add_labeled_input(desc_label, dpg.add_input_float, tag=tag, width=-1, format=fmt, callback=_update_sim_config_from_ui_and_signal_reset_needed)

            with dpg.group(tag="hodgkin_huxley_params_group", show=False): # Hidden by default
                dpg.add_text("--- Hodgkin-Huxley Model Parameters (Global Defaults) ---", color=[200,200,100])
                ui_hh_params = [
                    ("Membrane Capacitance C_m (uF/cm^2)", "cfg_hh_C_m", "%.2f"),
                    ("Max Sodium Cond. g_Na_max (mS/cm^2)", "cfg_hh_g_Na_max", "%.1f"),
                    ("Max Potassium Cond. g_K_max (mS/cm^2)", "cfg_hh_g_K_max", "%.1f"),
                    ("Leak Cond. g_L (mS/cm^2)", "cfg_hh_g_L", "%.3f"),
                    ("Sodium Reversal E_Na (mV)", "cfg_hh_E_Na", "%.1f"),
                    ("Potassium Reversal E_K (mV)", "cfg_hh_E_K", "%.1f"),
                    ("Leak Reversal E_L (mV)", "cfg_hh_E_L", "%.3f"),
                    ("Spike Detection V_peak (mV)", "cfg_hh_v_peak", "%.1f"),
                    ("Initial V_rest (mV)", "cfg_hh_v_rest_init", "%.1f"),
                    ("Kinetics Q10 Factor", "cfg_hh_q10_factor", "%.1f"),
                    ("Kinetics Temperature (°C)", "cfg_hh_temperature_celsius", "%.1f")
                ]
                for desc_label, tag, fmt in ui_hh_params:
                     add_labeled_input(desc_label, dpg.add_input_float, tag=tag, width=-1, format=fmt, callback=_update_sim_config_from_ui_and_signal_reset_needed)

        with dpg.collapsing_header(label="Network Connectivity", default_open=False, tag="network_connectivity_header"):
            add_labeled_input("Use Watts-Strogatz Generator:", dpg.add_checkbox, tag="cfg_enable_watts_strogatz", default_value=True, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("W-S K (Nearest Neighbors, even):", dpg.add_input_int, tag="cfg_connectivity_k", default_value=10, width=-1, step=2, min_value=2, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("W-S P (Rewire Probability):", dpg.add_input_float, tag="cfg_connectivity_p_rewire", default_value=0.1, width=-1, min_value=0.0, max_value=1.0, format="%.3f", callback=_update_sim_config_from_ui_and_signal_reset_needed)

        with dpg.collapsing_header(label="Synaptic Parameters", default_open=False, tag="synaptic_params_header"):
            add_labeled_input("Excitatory Propagation Strength (g_peak_e scale):", dpg.add_input_float, tag="cfg_propagation_strength", default_value=0.05, format="%.4f", width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Inhibitory Propagation Strength (g_peak_i scale):", dpg.add_input_float, tag="cfg_inhibitory_propagation_strength", default_value=0.15, format="%.4f", width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Excitatory Conductance Tau_g_e (ms):", dpg.add_input_float, tag="cfg_syn_tau_e", default_value=5.0, format="%.2f", width=-1, min_value=0.1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Inhibitory Conductance Tau_g_i (ms):", dpg.add_input_float, tag="cfg_syn_tau_i", default_value=10.0, format="%.2f", width=-1, min_value=0.1, callback=_update_sim_config_from_ui_and_signal_reset_needed)

        with dpg.collapsing_header(label="Learning & Plasticity", default_open=False, tag="learning_plasticity_header"):
            add_labeled_input("Enable Hebbian Learning:", dpg.add_checkbox, tag="cfg_enable_hebbian_learning", default_value=True, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Hebbian Learning Rate:", dpg.add_input_float, tag="cfg_hebbian_learning_rate", default_value=0.0005, format="%.6f", width=-1, callback=_update_sim_config_from_ui) # No reset needed for rate change
            add_labeled_input("Hebbian Max Weight:", dpg.add_input_float, tag="cfg_hebbian_max_weight", default_value=1.0, format="%.2f", width=-1, callback=_update_sim_config_from_ui) # No reset
            dpg.add_separator()
            add_labeled_input("Enable Short-Term Plasticity (STP):", dpg.add_checkbox, tag="cfg_enable_short_term_plasticity", default_value=True, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("STP U (Baseline Utilization):", dpg.add_input_float, tag="cfg_stp_U", default_value=0.15, format="%.3f", width=-1, callback=_update_sim_config_from_ui) # No reset
            add_labeled_input("STP Tau_d (Depression, ms):", dpg.add_input_float, tag="cfg_stp_tau_d", default_value=200.0, format="%.1f", width=-1, callback=_update_sim_config_from_ui) # No reset
            add_labeled_input("STP Tau_f (Facilitation, ms):", dpg.add_input_float, tag="cfg_stp_tau_f", default_value=50.0, format="%.1f", width=-1, callback=_update_sim_config_from_ui) # No reset
            dpg.add_separator()
            add_labeled_input("Enable Homeostasis:", dpg.add_checkbox, tag="cfg_enable_homeostasis", default_value=True, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            with dpg.group(tag="homeostasis_izh_specific_group", show=True): # Only for Izhikevich
                add_labeled_input("Homeostasis Target Rate (spikes/step for Izh):", dpg.add_input_float, tag="cfg_homeostasis_target_rate", default_value=0.02, format="%.4f", width=-1, callback=_update_sim_config_from_ui) # No reset
                add_labeled_input("Homeostasis Min Threshold (Izh, mV):", dpg.add_input_float, tag="cfg_homeostasis_threshold_min", default_value=-55.0, format="%.1f", width=-1, callback=_update_sim_config_from_ui) # No reset
                add_labeled_input("Homeostasis Max Threshold (Izh, mV):", dpg.add_input_float, tag="cfg_homeostasis_threshold_max", default_value=-30.0, format="%.1f", width=-1, callback=_update_sim_config_from_ui) # No reset

        with dpg.collapsing_header(label="Visual Settings & Filters", default_open=False, tag="visual_settings_header"):
            dpg.add_text("--- Neurons ---", color=[150,200,250])
            spiking_filter_options = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
            add_labeled_input("Show Spiking Neurons:", dpg.add_combo, items=spiking_filter_options, tag="filter_spiking_mode_combo", default_value="Highlight Spiking", width=-1, callback=trigger_filter_update_signal)
            add_labeled_input("Enable Synaptic Pulses:", dpg.add_checkbox, tag="gl_enable_synaptic_pulses_cb", default_value=opengl_viz_config.get('ENABLE_SYNAPTIC_PULSES', True), callback=handle_gl_enable_synaptic_pulses_change)
            add_labeled_input("Filter By Neuron Type:", dpg.add_checkbox, tag="filter_type_enable_cb", callback=lambda s, a, u: (dpg.configure_item("filter_neuron_type_combo", enabled=a), trigger_filter_update_signal(s,a,u)))
            add_labeled_input("Select Type:", dpg.add_combo, tag="filter_neuron_type_combo", default_value="All", width=-1, enabled=False, callback=trigger_filter_update_signal)
            add_labeled_input("Max Visible Neurons:", dpg.add_input_int, tag="gl_max_neurons_render_input", default_value=opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000), width=-1, min_value=0, step=100, callback=handle_gl_max_neurons_change)
            add_labeled_input("Neuron Size:", dpg.add_slider_float, tag="gl_neuron_point_size_slider", default_value=opengl_viz_config.get('POINT_SIZE', 2.0), min_value=0.5, max_value=10.0, width=-1, callback=handle_gl_point_size_change, format="%.1f")
            add_labeled_input("Inactive Neuron Opacity:", dpg.add_slider_float, tag="gl_inactive_neuron_opacity_slider", default_value=opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25), width=-1, min_value=0.0, max_value=1.0, format="%.2f", callback=handle_gl_inactive_neuron_opacity_change)
            dpg.add_separator()
            dpg.add_text("--- Synapses ---", color=[150,200,250])
            add_labeled_input("Show Synapses:", dpg.add_checkbox, tag="filter_show_synapses_gl_cb", default_value=global_gui_state.get("show_connections_gl", True), callback=lambda s,a,u: (global_gui_state.update({"show_connections_gl":a}), trigger_filter_update_signal()))
            add_labeled_input("Max Visible Connections:", dpg.add_input_int, tag="gl_max_connections_render_input", default_value=opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000), width=-1, min_value=0, step=500, callback=handle_gl_max_connections_change)
            add_labeled_input("Synapse Alpha Multiplier:", dpg.add_slider_float, tag="gl_synapse_alpha_slider", default_value=opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.3), min_value=0.0, max_value=2.0, width=-1, callback=handle_gl_synapse_alpha_change, format="%.2f")
            add_labeled_input("Min Abs Synapse Weight:", dpg.add_slider_float, tag="filter_min_abs_weight_slider", default_value=0.01, max_value=1.0, width=-1, callback=trigger_filter_update_signal, format="%.3f")
            dpg.add_separator()
            dpg.add_text("--- General ---", color=[150,200,250])
            add_labeled_input("Camera Field of View (FOV, degrees):", dpg.add_slider_float, tag="cfg_camera_fov", default_value=60.0, min_value=10.0, max_value=120.0, width=-1, callback=_update_sim_config_from_ui_and_signal_reset_needed)
            add_labeled_input("Activity Highlight Frames:", dpg.add_input_int, tag="gl_activity_highlight_frames_input", default_value=opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7), width=-1, min_value=1, max_value=30, callback=handle_gl_activity_highlight_frames_change)

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
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_checkpoint_dialog_callback_h5, tag="save_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="HDF5 Files (*.h5)")
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_checkpoint_dialog_callback_h5, tag="load_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="HDF5 Files (*.h5)")

    # Recording dialogs (HDF5)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_recording_for_streaming_dialog_callback_h5, tag="save_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5):
        dpg.add_file_extension(".h5", color=(100, 0, 100, 255), custom_text="HDF5 Files (*.simrec.h5)")
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_recording_dialog_callback_h5, tag="load_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5):
        dpg.add_file_extension(".h5", color=(100, 0, 100, 255), custom_text="HDF5 Files (*.simrec.h5)")

# --- Main Application Loop (Idle Function for GLUT and DPG Rendering) ---
def idle_and_dpg_render_func():
    global global_simulation_bridge, global_gui_state, last_sim_update_time_dpg, shutdown_flag

    if shutdown_flag.is_set():
        if OPENGL_AVAILABLE and glut.glutGetWindow() != 0:
            try: glut.glutLeaveMainLoop()
            except: pass # Ignore errors if already leaving
        if dpg.is_dearpygui_running(): dpg.stop_dearpygui()
        return

    if not dpg.is_dearpygui_running(): # If DPG window closed, initiate shutdown
        if OPENGL_AVAILABLE and glut.glutGetWindow() != 0 and not shutdown_flag.is_set():
            shutdown_flag.set();
            try: glut.glutLeaveMainLoop()
            except: pass
        return

    current_perf_time = time.perf_counter()
    sim_steps_taken_this_frame = 0
    dt_ms_val = 0.1 # Default, will be updated
    refresh_needed_for_gui_or_gl = False

    if global_gui_state.get("is_playback_mode", False):
        if global_gui_state.get("playback_is_playing", False):
            time_since_last_step = current_perf_time - global_gui_state.get("last_playback_autostep_time", 0.0)
            playback_interval = 1.0 / global_gui_state.get("playback_fps", 30.0)

            if time_since_last_step >= playback_interval:
                if global_simulation_bridge: global_simulation_bridge.step_playback(1)
                global_gui_state["last_playback_autostep_time"] = current_perf_time
                refresh_needed_for_gui_or_gl = True # Data changed, need refresh

        if global_gui_state.get("filters_changed", False): # If filters changed during playback
            refresh_needed_for_gui_or_gl = True

    else: # Live simulation mode
        if global_simulation_bridge and global_simulation_bridge.is_initialized:
            sim_is_active = global_simulation_bridge.sim_config.is_running and \
                            not global_simulation_bridge.sim_config.is_paused
            dt_ms_val = global_simulation_bridge.sim_config.dt_ms
            if dt_ms_val is None or dt_ms_val <= 0: dt_ms_val = 0.1 # Safety fallback

            if sim_is_active:
                if not global_gui_state.get('_was_running_last_frame', False): # If just started/resumed
                    last_sim_update_time_dpg = current_perf_time # Reset timer

                elapsed_real_time_s = current_perf_time - last_sim_update_time_dpg
                sim_time_advance_ms = elapsed_real_time_s * 1000.0 * global_simulation_bridge.sim_config.simulation_speed_factor

                if dt_ms_val > 0:
                    num_steps_to_run = int(sim_time_advance_ms / dt_ms_val)
                    if num_steps_to_run > 0:
                        max_steps_per_frame = 500 # Limit steps per render frame to keep UI responsive
                        num_steps_to_run = min(num_steps_to_run, max_steps_per_frame)

                        for _ in range(num_steps_to_run):
                            if global_simulation_bridge.sim_config.current_time_ms < global_simulation_bridge.sim_config.total_simulation_time_ms:
                                global_simulation_bridge._run_one_simulation_step() # This now records to HDF5 if active
                                global_simulation_bridge.sim_config.current_time_ms += dt_ms_val
                                global_simulation_bridge.sim_config.current_time_step += 1
                                sim_steps_taken_this_frame += 1
                            else: # Total simulation time reached
                                global_simulation_bridge.log_message("Total simulation time reached.", "info")
                                handle_stop_simulation_event() # This updates UI
                                break
                        # Adjust last_sim_update_time based on simulated time processed
                        last_sim_update_time_dpg += (num_steps_to_run * dt_ms_val) / \
                                                    (global_simulation_bridge.sim_config.simulation_speed_factor * 1000.0
                                                     if global_simulation_bridge.sim_config.simulation_speed_factor > 0 else 1000.0)
                elif not global_gui_state.get('_dt_warning_logged', False): # Log warning for invalid dt once
                         global_simulation_bridge.log_message(f"Sim dt_ms is {dt_ms_val}, cannot advance simulation.", "warning")
                         global_gui_state['_dt_warning_logged'] = True

            global_gui_state['_was_running_last_frame'] = sim_is_active
            if dt_ms_val > 0 : global_gui_state['_dt_warning_logged'] = False # Reset warning if dt becomes valid
            if sim_steps_taken_this_frame > 0: refresh_needed_for_gui_or_gl = True

    # Refresh GUI monitor and OpenGL data if needed
    if refresh_needed_for_gui_or_gl or global_gui_state.get("filters_changed", False):
        if global_simulation_bridge and global_simulation_bridge.is_initialized:
            monitor_snapshot = global_simulation_bridge.get_latest_simulation_data_for_gui(force_fetch=True)
            if monitor_snapshot: update_monitoring_overlay_values(monitor_snapshot)
            if OPENGL_AVAILABLE: update_gl_data() # Update VBOs for OpenGL
        global_gui_state["filters_changed"] = False # Reset flag

    if dpg.is_dearpygui_running(): dpg.render_dearpygui_frame()

    if OPENGL_AVAILABLE and glut.glutGetWindow() != 0:
        try:
            current_win = glut.glutGetWindow()
            if glut.glut_window_id is not None and current_win != glut.glut_window_id: glut.glutSetWindow(glut.glut_window_id) # type: ignore
            glut.glutPostRedisplay()
        except Exception: pass # Ignore if context is lost

def main():
    global global_simulation_bridge, global_gui_state, global_viz_data_cache, opengl_viz_config
    global glut_window_id, last_sim_update_time_dpg, shutdown_flag # type: ignore

    dpg.create_context()
    dpg.configure_app(docking=False) # Docking can be enabled if desired

    global_simulation_bridge = SimulationBridge() # Initialize the simulation core

    # Attempt to load default profile (still JSON)
    default_profile_filename = "default_profile.json"
    default_profile_path = os.path.join(global_simulation_bridge.PROFILE_DIR, default_profile_filename)
    loaded_default_successfully = False; gui_cfg_from_default = None
    if os.path.exists(default_profile_path):
        try:
            with open(default_profile_path, 'r') as f: profile_content = json.load(f)
            sim_cfg_data = profile_content.get("simulation_configuration")
            if sim_cfg_data:
                if global_simulation_bridge.apply_simulation_configuration_core(sim_cfg_data):
                    global_gui_state["current_profile_name"] = default_profile_filename
                    gui_cfg_from_default = profile_content.get("gui_configuration")
                    loaded_default_successfully = True
        except Exception as e:
            global_simulation_bridge.log_message(f"Error loading default profile '{default_profile_path}': {e}", "warning")

    if not loaded_default_successfully: # Fallback to internal defaults if profile load fails
        global_simulation_bridge.log_message("Using basic default internal configuration.", "info")
        if not global_simulation_bridge.apply_simulation_configuration_core(global_simulation_bridge.sim_config.to_dict()):
             global_simulation_bridge.log_message("Critical error: Failed to initialize with internal defaults.", "critical"); return
        global_gui_state["current_profile_name"] = "unsaved_internal_defaults.json"

    # DPG Viewport setup
    dpg_viewport_width = 700 # Width for the DPG control panel
    dpg_viewport_height = int(SCREEN_HEIGHT * 0.90) if SCREEN_HEIGHT > 0 else 760 # Adjust height based on screen
    dpg.create_viewport(title="Neuron Simulator Controls (DPG)",
                        width=dpg_viewport_width, height=dpg_viewport_height,
                        x_pos=0, y_pos=20) # Position on the left

    create_gui_layout() # Create all DPG widgets
    dpg.set_primary_window("controls_monitor_window", True)

    _populate_ui_from_sim_config() # Populate UI with current (possibly default-loaded) config
    if gui_cfg_from_default: apply_gui_configuration_core(gui_cfg_from_default) # Apply GUI settings from profile

    if dpg.does_item_exist("profile_name_input"): # Show current profile name in UI
        dpg.set_value("profile_name_input", global_gui_state["current_profile_name"].replace(".json", ""))

    if dpg.does_item_exist("filter_neuron_type_combo"): # Populate neuron type filter
        available_types = global_simulation_bridge.get_available_neuron_types()
        dpg.configure_item("filter_neuron_type_combo", items=available_types)
        dpg.set_value("filter_neuron_type_combo", "All" if "All" in available_types else (available_types[0] if available_types else ""))

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Initialize visualization data cache
    profile_viz_data = global_simulation_bridge.get_profile_visualization_data(from_current_config=True)
    if profile_viz_data and "neuron_types" in profile_viz_data:
        global_viz_data_cache["neuron_types"] = profile_viz_data["neuron_types"]
        global_simulation_bridge.sim_config.neuron_types_list_for_viz = profile_viz_data["neuron_types"]

    initial_monitor_snapshot = global_simulation_bridge.get_initial_sim_data_snapshot()
    if initial_monitor_snapshot: update_monitoring_overlay_values(initial_monitor_snapshot)

    # Set initial UI states for simulation controls, recording, and playback
    update_ui_for_simulation_run_state(is_running=False, is_paused=False)
    update_ui_for_recording_state(is_recording=False)
    update_ui_for_playback_state(is_playback=False)
    update_ui_for_loaded_recording(can_playback=global_gui_state.get("loaded_recording_data") is not None)

    global_gui_state["reset_sim_needed_from_ui_change"] = False # Clear reset flag
    last_sim_update_time_dpg = time.perf_counter() # Initialize timer for sim loop

    if OPENGL_AVAILABLE:
        glut.glutInit(sys.argv if hasattr(sys, "argv") and sys.argv else ["sim3d.py"])
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)

        # Calculate GL window size and position based on DPG window
        gl_win_width = SCREEN_WIDTH - dpg_viewport_width - 30 if SCREEN_WIDTH > dpg_viewport_width + 30 else 600
        gl_win_height = dpg_viewport_height; gl_win_width = max(400, gl_win_width); gl_win_height = max(300, gl_win_height)
        gl_win_x_pos = dpg_viewport_width + 10 # Position to the right of DPG window

        glut.glutInitWindowPosition(gl_win_x_pos, 20); glut.glutInitWindowSize(gl_win_width, gl_win_height)
        try: glut.glut_window_id = glut.glutCreateWindow(b"3D Network Visualization (OpenGL)") # type: ignore
        except TypeError: glut.glut_window_id = glut.glutCreateWindow("3D Network Visualization (OpenGL)") # type: ignore

        opengl_viz_config['WINDOW_WIDTH'] = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        opengl_viz_config['WINDOW_HEIGHT'] = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)

        init_gl(); # Initialize OpenGL state
        glut.glutDisplayFunc(render_scene_gl); # Set display callback
        glut.glutReshapeFunc(reshape_gl_window) # Set reshape callback
        glut.glutKeyboardFunc(keyboard_func_gl); # Set keyboard callback
        glut.glutMouseFunc(mouse_button_func_gl) # Set mouse button callback
        glut.glutMotionFunc(mouse_motion_func_gl); # Set mouse motion callback
        glut.glutIdleFunc(idle_and_dpg_render_func) # Set idle callback (main loop)

        update_gl_data(); # Initial population of GL data
        print("Starting GLUT main loop...")
        try: glut.glutMainLoop()
        except Exception as e: print(f"Exception during GLUT main loop: {e}")
        finally:
            print("Exited GLUT main loop."); shutdown_flag.set()
            if dpg.is_dearpygui_running(): dpg.stop_dearpygui()
    else: # No OpenGL, run DPG only
        print("OpenGL not available. Running DPG controls only.")
        while dpg.is_dearpygui_running() and not shutdown_flag.is_set():
            idle_and_dpg_render_func(); # Call the DPG part of the loop
            time.sleep(0.005) # Prevent busy-waiting
        if dpg.is_dearpygui_running(): dpg.stop_dearpygui()

    if dpg.is_dearpygui_running(): dpg.destroy_context() # Clean up DPG
    print("Neuron simulator application shutdown.")

if __name__ == '__main__':
    main()