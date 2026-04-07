"""DearPyGUI callback functions for the neural simulator UI.

Contains all event handlers, config sync functions, file dialogs,
recording/playback UI state management, and experiment system callbacks.

Module-level references are set by init_callbacks() from neural-simulator.py.
"""

import os
import json
import time
import threading
import numpy as np

import dearpygui.dearpygui as dpg

# These are set by init_callbacks()
global_simulation_bridge = None
global_gui_state = None
global_viz_data_cache = None
opengl_viz_config = None
OPENGL_AVAILABLE = False
TRAIT_COLOR_MAP_RAW = None
shutdown_flag = None
ui_to_sim_queue = None
sim_to_ui_queue = None
_FULL_PROFILE_MAP = {}

# Imported types - set by init_callbacks()
SimulationConfiguration = None
NeuronModel = None
NeuronType = None
NeuronGroupRole = None
ExperimentPhaseType = None
DefaultHodgkinHuxleyParams = None
DefaultIzhikevichParamsManager = None
NEURON_TYPE_MAPPER = None
CoreSimConfig = None
VisualizationConfig = None
RuntimeState = None
GPUConfig = None
_create_config_from_dict = None
_get_full_config_dict = None
NEURAL_STRUCTURE_PROFILES = None
get_compatible_hh_type_names_for_profile = None
get_auto_tuned_overrides_for_combo = None
check_config_against_limits = None
get_hardware_limits_for_model = None
ExperimentConfig = None
ExperimentPresets = None
StimulusPattern = None
StimulusChannel = None
NeuronGroup = None
ExperimentPhase = None
ReadoutConfig = None
TrainingConfig = None
StimulusPatternType = None
experiment_config_to_dict = None
experiment_config_from_dict = None
performance_test_stop_flag = None
BENCHMARK_RESULTS_PATH = None
performance_test_running_type = None  # Track which test is running: "benchmark" or "optimization"
AUTO_TUNED_OVERRIDES = None  # Lazy-loaded mapping (used by reload_overrides callback)
HARDWARE_LIMITS = None  # Lazy-loaded benchmark limits dict
_load_auto_tuned_overrides_if_needed = None  # Function ref from main module
_load_benchmark_limits = None  # Function ref from main module
get_hardware_note = None  # Function ref from main module


def init_callbacks(**kwargs):
    """Initialize module-level references. Called once from neural-simulator.py."""
    g = globals()
    for key, value in kwargs.items():
        if key in g:
            g[key] = value


# Import filter functions from viz
from viz.renderer import (trigger_filter_update_signal, get_current_filter_settings_from_gui)


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
    # Profile dir is at project root, not relative to this file (which is in ui/)
    profile_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation_profiles")
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

