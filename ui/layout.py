"""DearPyGUI layout definition for the neural simulator UI.

Contains create_gui_layout() and the add_parameter_table_row() helper.
All widget creation code lives here.

References to callback functions and shared state are imported from ui.callbacks.
"""

import os
import dearpygui.dearpygui as dpg

# Re-import everything needed from callbacks (which holds the shared state refs)
from ui.callbacks import (
    # Shared state
    global_gui_state, opengl_viz_config, OPENGL_AVAILABLE,
    TRAIT_COLOR_MAP_RAW, shutdown_flag,
    # Types
    NeuronModel, NeuronType, DefaultIzhikevichParamsManager, DefaultHodgkinHuxleyParams,
    NEURAL_STRUCTURE_PROFILES,
    StimulusPatternType,
    # Callbacks
    update_status_bar,
    handle_start_simulation_event, handle_stop_simulation_event,
    handle_pause_simulation_event, handle_step_simulation_event,
    handle_apply_config_changes_and_reset, handle_sim_speed_change,
    handle_record_button_click, handle_playback_button_click,
    handle_playback_slider_change, handle_playback_play_pause_button_click,
    handle_playback_step_frames_click,
    handle_save_profile_button_press, handle_load_profile_button_press,
    save_profile_dialog_callback, load_profile_dialog_callback,
    handle_save_checkpoint_button_press, save_checkpoint_dialog_callback_h5,
    handle_load_checkpoint_button_press, load_checkpoint_dialog_callback_h5,
    handle_load_recording_menu_click, load_recording_dialog_callback_h5,
    save_recording_for_streaming_dialog_callback_h5,
    _handle_model_type_change_dpg,
    _update_sim_config_from_ui_and_signal_reset_needed,
    handle_gl_point_size_change, handle_gl_synapse_alpha_change,
    handle_gl_activity_highlight_frames_change,
    handle_gl_max_neurons_change, handle_gl_max_connections_change,
    handle_gl_inactive_neuron_opacity_change,
    handle_gl_enable_synaptic_pulses_change,
    handle_reset_hh_drive_to_auto, handle_reset_adex_drive_to_auto,
    _handle_experiment_preset_change, _handle_inject_manual_stimulus,
    _scan_profile_directory, _FULL_PROFILE_MAP,
    _handle_full_profile_dropdown_change, _refresh_full_profile_dropdown,
    _recording_options_continue_callback, _recording_options_cancel_callback,
    handle_run_benchmark_click, handle_run_optimization_click,
    handle_stop_perf_test_click, handle_reload_overrides_click,
    handle_run_viz_benchmark_click,
    handle_log_search_change, handle_log_search_prev, handle_log_search_next,
    handle_clear_logs_click, handle_export_logs_click,
    trigger_filter_update_signal,
    get_profile_files, get_hdf5_files,
)


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
    """Creates the main Dear PyGui layout, including all windows, menus, and widgets.

    Returns:
        dict: A dict with 'inspector_update' callable for refreshing the neuron inspector.
    """
    _inspector_update_fn = None
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

        # --- Live Monitoring Plots ---
        with dpg.collapsing_header(label="Live Monitoring", default_open=False, tag="live_monitoring_header"):
            dpg.add_text("Real-time simulation data plots", color=[150, 150, 150])

        dpg.add_spacer(height=5); dpg.add_separator(); dpg.add_spacer(height=5)

        # --- Neuron Inspector ---
        from ui.inspector import create_inspector_panel
        _inspector_update_fn = create_inspector_panel("controls_monitor_window")

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

    return {"inspector_update": _inspector_update_fn}

