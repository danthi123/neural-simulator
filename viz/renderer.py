"""OpenGL rendering, VBO management, neuron coloring, and filter functions.

This module contains the core visualization pipeline:
  - init_gl(): OpenGL state initialization and VBO generation
  - update_gl_data(): Prepares neuron/synapse/pulse data for rendering (VBO updates)
  - render_scene_gl(): Main GLUT display callback
  - fast_vbo_update(): Optimized CuPy->VBO transfer
  - get_color_for_trait(): Per-neuron color based on trait, activity, and filters
  - Filter functions for neurons and synapses

Module-level state is initialized by the main entry point via set_shared_state().
"""

import time
import math
import threading
import numpy as np

try:
    from OpenGL.GL import *
    import OpenGL.GLUT as glut
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None

from viz.overlays import render_text_gl

# ---------------------------------------------------------------------------
# Module-level references set by set_shared_state() from neural-simulator.py
# ---------------------------------------------------------------------------
global_simulation_bridge = None
global_gui_state = None
global_viz_data_cache = None
opengl_viz_config = None
TRAIT_COLOR_MAP_RAW = None
TRAIT_COLOR_MAP_GPU = None
shutdown_flag = None
ui_to_sim_queue = None
# update_ui_for_simulation_run_state callback (set by UI module)
_update_ui_for_simulation_run_state = None
# update_status_bar callback
_update_status_bar = None

# NeuronTypeIDMapper reference
NEURON_TYPE_MAPPER = None

# ---------------------------------------------------------------------------
# OpenGL VBO globals
# ---------------------------------------------------------------------------
gl_neuron_pos_vbo = None
gl_neuron_color_vbo = None
gl_synapse_vertices_vbo = None
gl_pulse_vertices_vbo = None

gl_num_neurons_to_draw = 0
gl_num_synapse_lines_to_draw = 0
gl_num_pulses_to_draw = 0

# Frame rate limiting for smooth 60 FPS
gl_last_render_time = 0.0
gl_target_frame_time = 1.0 / 60.0

# FPS counter tracking
gl_frame_times = []
gl_fps_update_interval = 0.5
gl_last_fps_update_time = 0.0
gl_current_fps = 0.0

# CuPy arrays holding data ready for VBO buffering
gl_neuron_pos_cp = None
gl_neuron_colors_cp = None
gl_connection_vertices_cp = None
gl_pulse_vertices_cp = None

# CUDA-OpenGL interop flag
cuda_gl_interop_enabled = False

# Frame counter for VBO update skipping
_gl_frame_counter = 0

# GLUT window ID
glut_window_id = None


def set_shared_state(sim_bridge, gui_state, viz_data_cache, viz_config,
                     trait_color_map_raw, trait_color_map_gpu,
                     shutdown_evt, ui_sim_queue, neuron_type_mapper,
                     update_run_state_fn, update_status_fn):
    """Called once from neural-simulator.py main() to inject shared references."""
    global global_simulation_bridge, global_gui_state, global_viz_data_cache
    global opengl_viz_config, TRAIT_COLOR_MAP_RAW, TRAIT_COLOR_MAP_GPU
    global shutdown_flag, ui_to_sim_queue, NEURON_TYPE_MAPPER
    global _update_ui_for_simulation_run_state, _update_status_bar
    global gl_neuron_pos_cp, gl_neuron_colors_cp, gl_connection_vertices_cp, gl_pulse_vertices_cp

    global_simulation_bridge = sim_bridge
    global_gui_state = gui_state
    global_viz_data_cache = viz_data_cache
    opengl_viz_config = viz_config
    TRAIT_COLOR_MAP_RAW = trait_color_map_raw
    TRAIT_COLOR_MAP_GPU = trait_color_map_gpu
    shutdown_flag = shutdown_evt
    ui_to_sim_queue = ui_sim_queue
    NEURON_TYPE_MAPPER = neuron_type_mapper
    _update_ui_for_simulation_run_state = update_run_state_fn
    _update_status_bar = update_status_fn

    if cp is not None:
        gl_neuron_pos_cp = cp.array([], dtype=cp.float32).reshape(0, 3)
        gl_neuron_colors_cp = cp.array([], dtype=cp.float32).reshape(0, 4)
        gl_connection_vertices_cp = cp.array([], dtype=cp.float32).reshape(0, 3)
        gl_pulse_vertices_cp = cp.array([], dtype=cp.float32).reshape(0, 3)


def set_glut_window_id(wid):
    global glut_window_id
    glut_window_id = wid


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def trigger_filter_update_signal(sender=None, app_data=None, user_data=None):
    """Sets a flag indicating that visualization filters have changed and GL data needs update."""
    if global_gui_state is not None:
        global_gui_state["filters_changed"] = True


def get_current_filter_settings_from_gui():
    """Retrieves current filter settings from DPG UI elements. Called by main/UI thread."""
    settings = {
        "spiking_mode": "Highlight Spiking",
        "type_filter_enabled": False,
        "selected_neuron_type": "All",
        "min_abs_weight": 0.01
    }
    if dpg is not None and dpg.is_dearpygui_running():
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
    if all_indices.size == 0:
        return []

    visible_mask = np.ones(all_indices.size, dtype=bool)

    spiking_mode = filters_dict.get("spiking_mode", "Highlight Spiking")
    if spiking_mode == "Show Only Spiking":
        if fired_status_np is not None and fired_status_np.shape == visible_mask.shape:
            visible_mask &= fired_status_np
        else:
            if fired_status_np is not None:
                print(f"Warning: fired_status_np shape mismatch in filter. Expected {visible_mask.shape}, got {fired_status_np.shape}")

    if filters_dict.get("type_filter_enabled", False):
        selected_type_str = filters_dict.get("selected_neuron_type", "All")
        if selected_type_str != "All" and neuron_types_list_str is not None and len(neuron_types_list_str) == all_indices.size:
            type_mask = np.array([neuron_types_list_str[i] == selected_type_str for i in all_indices], dtype=bool)
            visible_mask &= type_mask
        elif selected_type_str != "All":
            if neuron_types_list_str is not None:
                print(f"Warning: neuron_types_list_str length mismatch in filter. Expected {all_indices.size}, got {len(neuron_types_list_str)}")

    return all_indices[visible_mask]


def apply_synapse_filters_to_indices(all_synapse_data_list, filters_dict):
    """
    Applies filters to a list of synapse data dictionaries to determine visibility.
    Returns a list of indices (into all_synapse_data_list) of visible synapses.
    """
    if global_gui_state is None or not global_gui_state.get("show_connections_gl", False):
        return []

    visible_syn_indices = []
    min_abs_w = filters_dict.get("min_abs_weight", 0.01)
    for i, syn_data in enumerate(all_synapse_data_list):
        if abs(syn_data.get("weight", 0.0)) >= min_abs_w:
            visible_syn_indices.append(i)
    return visible_syn_indices


# ---------------------------------------------------------------------------
# Neuron coloring
# ---------------------------------------------------------------------------

def get_color_for_trait(trait_index, activity_timer_value, is_currently_spiking, neuron_model_name_str, neuron_type_str=""):
    """
    Determines neuron color based on trait, activity, spiking status, and filter mode.
    Called by the main thread during GL data preparation.
    """
    max_highlight_frames = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
    firing_rgb_config = opengl_viz_config.get("FIRING_NEURON_COLOR", [1.0, 1.0, 0.0, 1.0])
    firing_rgb = firing_rgb_config[0:3]
    base_firing_alpha = firing_rgb_config[3]
    default_inactive_alpha = opengl_viz_config.get("INACTIVE_NEURON_OPACITY", 0.25)

    base_color_rgb = [0.5, 0.5, 0.5]
    base_alpha = default_inactive_alpha
    if TRAIT_COLOR_MAP_RAW and len(TRAIT_COLOR_MAP_RAW) > 0:
        color_def_from_map = TRAIT_COLOR_MAP_RAW[trait_index % len(TRAIT_COLOR_MAP_RAW)]
        base_color_rgb = color_def_from_map[0:3]
        base_alpha = color_def_from_map[3] if len(color_def_from_map) > 3 else default_inactive_alpha

    final_color_rgba = list(base_color_rgb) + [base_alpha]

    filters_dict = get_current_filter_settings_from_gui()
    spiking_mode_filter = filters_dict.get("spiking_mode", "Highlight Spiking")

    if spiking_mode_filter == "No Spiking Highlight":
        return final_color_rgba

    if is_currently_spiking:
        final_color_rgba = list(firing_rgb) + [base_firing_alpha]
    elif spiking_mode_filter == "Highlight Spiking" and activity_timer_value > 0:
        decay_ratio = max(0.0, min(1.0, float(activity_timer_value) / max_highlight_frames))
        dimmed_firing_alpha = base_firing_alpha * decay_ratio * 0.6
        dimmed_firing_alpha = max(dimmed_firing_alpha, base_alpha * 0.8, 0.05)
        dimmed_firing_alpha = min(base_firing_alpha * 0.8, dimmed_firing_alpha)
        final_color_rgba = list(firing_rgb) + [dimmed_firing_alpha]

    return final_color_rgba


# ---------------------------------------------------------------------------
# VBO transfer
# ---------------------------------------------------------------------------

def fast_vbo_update(vbo_id, cupy_array):
    """Optimized VBO update using pinned memory for faster transfers."""
    if cupy_array.size == 0:
        return

    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)

    try:
        if not cupy_array.flags.c_contiguous:
            cupy_array = cp.ascontiguousarray(cupy_array)

        np_array = cp.asnumpy(cupy_array, order='C')
        glBufferData(GL_ARRAY_BUFFER, np_array.nbytes, np_array, GL_DYNAMIC_DRAW)
    except Exception as e:
        print(f"[VBO Update] Error: {e}")


# ---------------------------------------------------------------------------
# OpenGL initialization
# ---------------------------------------------------------------------------

def init_gl():
    """Initializes OpenGL state. Called by the main thread."""
    if not OPENGL_AVAILABLE:
        return
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global cuda_gl_interop_enabled

    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0, 0, 0, 0)
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0))
    glEnable(GL_DEPTH_TEST)

    try:
        vbo_ids = glGenBuffers(4)
        if not isinstance(vbo_ids, (list, tuple, np.ndarray)) or len(vbo_ids) < 4:
            if isinstance(vbo_ids, int) and vbo_ids > 0:
                gl_neuron_pos_vbo = vbo_ids
                gl_neuron_color_vbo = glGenBuffers(1)
                gl_synapse_vertices_vbo = glGenBuffers(1)
                gl_pulse_vertices_vbo = glGenBuffers(1)
            else:
                raise ValueError("glGenBuffers did not return expected VBO IDs.")
        else:
            gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo = (
                vbo_ids[0], vbo_ids[1], vbo_ids[2], vbo_ids[3]
            )

        try:
            from cuda import cudart
            cuda_gl_interop_enabled = True
            print("[CUDA-GL Interop] Enabled for zero-copy GPU->OpenGL transfers")
        except ImportError:
            cuda_gl_interop_enabled = False
            print("[CUDA-GL Interop] Not available (cuda-python not installed). Using GPU->CPU->GPU path.")

    except Exception as e:
        print(f"Error: glGenBuffers failed: {e}. OpenGL visualization will likely fail.")
        gl_neuron_pos_vbo = 0
        gl_neuron_color_vbo = 0
        gl_synapse_vertices_vbo = 0
        gl_pulse_vertices_vbo = 0
        cuda_gl_interop_enabled = False
        return


# ---------------------------------------------------------------------------
# update_gl_data
# ---------------------------------------------------------------------------

def update_gl_data():
    """
    Prepares neuron, synapse, and pulse data for OpenGL rendering by updating VBOs.
    This function is called by the main/UI thread. It gets data from global_viz_data_cache.gl_render_data_buffer,
    which is populated by the simulation thread with CuPy arrays for GL data.
    """
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw
    global gl_neuron_pos_cp, gl_neuron_colors_cp, gl_connection_vertices_cp, gl_pulse_vertices_cp
    global _gl_frame_counter

    if not OPENGL_AVAILABLE:
        gl_num_neurons_to_draw = 0
        gl_num_synapse_lines_to_draw = 0
        gl_num_pulses_to_draw = 0
        return

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
        if sim_data_snapshot is None and not global_gui_state.get("filters_changed", False):
            return

    # --- Extract CuPy arrays and other data from snapshot ---
    neuron_fired_cp = sim_data_snapshot.get("neuron_fired_status_cp", cp.array([], dtype=bool))
    neuron_activity_timers_cp = sim_data_snapshot.get("neuron_activity_timers_cp", cp.array([], dtype=cp.int32))
    all_neuron_positions_3d_cp = sim_data_snapshot.get("neuron_positions_3d_cp", cp.array([], dtype=cp.float32).reshape(0, 3))
    all_neuron_traits_cp = sim_data_snapshot.get("neuron_traits_cp", cp.array([], dtype=cp.int32))
    all_neuron_type_ids_cp = sim_data_snapshot.get("neuron_type_ids_cp", cp.array([], dtype=cp.int32))

    all_neuron_types_str_list_cpu = sim_data_snapshot.get("neuron_types_list_for_viz", [])
    model_name_str = sim_data_snapshot.get("neuron_model_type_str", "IZHIKEVICH")
    num_neurons_in_snapshot = sim_data_snapshot.get("num_neurons_snapshot", 0)

    if all_neuron_positions_3d_cp.shape[0] != num_neurons_in_snapshot:
        all_neuron_positions_3d_cp = cp.zeros((num_neurons_in_snapshot, 3), dtype=cp.float32)
    if neuron_fired_cp.size != num_neurons_in_snapshot:
        neuron_fired_cp = cp.zeros(num_neurons_in_snapshot, dtype=bool)
    if neuron_activity_timers_cp.size != num_neurons_in_snapshot:
        neuron_activity_timers_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32)
    if all_neuron_traits_cp.size != num_neurons_in_snapshot:
        all_neuron_traits_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32)
    if all_neuron_type_ids_cp.size != num_neurons_in_snapshot:
        all_neuron_type_ids_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32)
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
        if selected_type_str_cpu != "All" and NEURON_TYPE_MAPPER is not None:
            selected_type_id = NEURON_TYPE_MAPPER.get_id_from_display_name(selected_type_str_cpu)
            type_mask_cp = (all_neuron_type_ids_cp == selected_type_id)
            visible_mask_cp &= type_mask_cp

    visible_neuron_indices_cp = all_indices_cp[visible_mask_cp]

    max_render_neurons = opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 100000)
    if visible_neuron_indices_cp.size > max_render_neurons:
        chosen_neuron_indices_cp = cp.random.choice(visible_neuron_indices_cp, size=max_render_neurons, replace=False)
    else:
        chosen_neuron_indices_cp = visible_neuron_indices_cp

    current_num_neurons_to_draw = chosen_neuron_indices_cp.size

    temp_gl_neuron_pos_cp = cp.array([], dtype=cp.float32).reshape(0, 3)
    temp_gl_neuron_colors_cp = cp.array([], dtype=cp.float32).reshape(0, 4)

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

        if TRAIT_COLOR_MAP_GPU is not None and TRAIT_COLOR_MAP_GPU.ndim == 2 and TRAIT_COLOR_MAP_GPU.shape[1] == 4:
            temp_gl_neuron_colors_cp = TRAIT_COLOR_MAP_GPU[chosen_traits % TRAIT_COLOR_MAP_GPU.shape[0]]
        else:
            temp_gl_neuron_colors_cp = cp.full(
                (current_num_neurons_to_draw, 4),
                cp.array([0.5, 0.5, 0.5, default_inactive_alpha_gpu], dtype=cp.float32),
                dtype=cp.float32
            )

        if spiking_mode_filter != "No Spiking Highlight":
            spiking_mask = chosen_is_spiking
            if cp.any(spiking_mask):
                temp_gl_neuron_colors_cp[spiking_mask, 0:3] = firing_rgb_gpu
                temp_gl_neuron_colors_cp[spiking_mask, 3] = base_firing_alpha_gpu

            if spiking_mode_filter == "Highlight Spiking":
                active_timer_mask = (~chosen_is_spiking) & (chosen_activity_timers > 0)
                if cp.any(active_timer_mask):
                    decay_ratio = cp.clip(chosen_activity_timers[active_timer_mask].astype(cp.float32) / max_highlight_frames_val, 0.0, 1.0)
                    base_alpha_for_active_timer = temp_gl_neuron_colors_cp[active_timer_mask, 3].copy()
                    dimmed_firing_alpha = base_firing_alpha_gpu * decay_ratio * 0.6
                    dimmed_firing_alpha = cp.maximum(dimmed_firing_alpha, base_alpha_for_active_timer * 0.8)
                    dimmed_firing_alpha = cp.maximum(dimmed_firing_alpha, 0.05)
                    dimmed_firing_alpha = cp.minimum(dimmed_firing_alpha, base_firing_alpha_gpu * 0.9)
                    temp_gl_neuron_colors_cp[active_timer_mask, 0:3] = firing_rgb_gpu
                    temp_gl_neuron_colors_cp[active_timer_mask, 3] = dimmed_firing_alpha

    # --- Synapse Data (GPU-accelerated filtering using cp.isin) ---
    temp_gl_connection_vertices_cp = cp.array([], dtype=cp.float32).reshape(0, 3)
    current_num_synapse_lines_to_draw = 0
    if global_gui_state.get("show_connections_gl", False) and "synapse_info" in sim_data_snapshot:
        all_synapse_data_list_cpu = sim_data_snapshot["synapse_info"]

        if all_synapse_data_list_cpu:
            src_indices_all = np.array([syn["source_idx"] for syn in all_synapse_data_list_cpu], dtype=np.int32)
            tgt_indices_all = np.array([syn["target_idx"] for syn in all_synapse_data_list_cpu], dtype=np.int32)
            weights_all = np.array([syn["weight"] for syn in all_synapse_data_list_cpu], dtype=np.float32)

            src_indices_all_cp = cp.asarray(src_indices_all)
            tgt_indices_all_cp = cp.asarray(tgt_indices_all)
            weights_all_cp = cp.asarray(weights_all)

            src_visible_mask = cp.isin(src_indices_all_cp, chosen_neuron_indices_cp)
            tgt_visible_mask = cp.isin(tgt_indices_all_cp, chosen_neuron_indices_cp)

            min_abs_w = current_filters.get("min_abs_weight", 0.01)
            weight_mask = cp.abs(weights_all_cp) >= min_abs_w

            synapse_visible_mask = src_visible_mask & tgt_visible_mask & weight_mask
            visible_synapse_indices_cp = cp.where(synapse_visible_mask)[0]

            if visible_synapse_indices_cp.size > 0:
                visible_src_indices = src_indices_all_cp[visible_synapse_indices_cp]
                visible_tgt_indices = tgt_indices_all_cp[visible_synapse_indices_cp]

                pos_src_all_cp = all_neuron_positions_3d_cp[visible_src_indices]
                pos_tgt_all_cp = all_neuron_positions_3d_cp[visible_tgt_indices]

                temp_gl_connection_vertices_cp = cp.empty((visible_synapse_indices_cp.size * 2, 3), dtype=cp.float32)
                temp_gl_connection_vertices_cp[0::2] = pos_src_all_cp
                temp_gl_connection_vertices_cp[1::2] = pos_tgt_all_cp
                current_num_synapse_lines_to_draw = visible_synapse_indices_cp.size

    # --- Synaptic Pulse Data ---
    temp_gl_pulse_vertices_cp = sim_data_snapshot.get("pulse_positions_cp_for_gl", cp.array([], dtype=cp.float32).reshape(0, 3))
    current_num_pulses_to_draw = temp_gl_pulse_vertices_cp.shape[0]

    # --- Update global GL CuPy arrays and VBOs ---
    gl_num_neurons_to_draw = current_num_neurons_to_draw
    gl_neuron_pos_cp = temp_gl_neuron_pos_cp
    gl_neuron_colors_cp = temp_gl_neuron_colors_cp

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

    if gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0:
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    global_gui_state["filters_changed"] = False


# ---------------------------------------------------------------------------
# render_scene_gl
# ---------------------------------------------------------------------------

def render_scene_gl():
    """Main OpenGL rendering function. Called by GLUT display callback in the main thread."""
    global gl_frame_times, gl_last_fps_update_time, gl_current_fps

    if not OPENGL_AVAILABLE or global_simulation_bridge is None:
        return

    current_time = time.perf_counter()
    if len(gl_frame_times) > 0:
        gl_frame_times.append(current_time)
        if len(gl_frame_times) > 60:
            gl_frame_times.pop(0)
    else:
        gl_frame_times.append(current_time)

    if current_time - gl_last_fps_update_time >= gl_fps_update_interval:
        if len(gl_frame_times) >= 2:
            time_span = gl_frame_times[-1] - gl_frame_times[0]
            if time_span > 0:
                gl_current_fps = (len(gl_frame_times) - 1) / time_span
            gl_last_fps_update_time = current_time

    try:
        current_win = glut.glutGetWindow()
        if glut_window_id is not None and current_win != glut_window_id and current_win != 0:
            glut.glutSetWindow(glut_window_id)
        elif current_win == 0:
            return
    except Exception:
        return

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0))

    viz_cfg = global_simulation_bridge.viz_config
    runtime = global_simulation_bridge.runtime_state
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    eye_x = viz_cfg.camera_center_x + viz_cfg.camera_radius * math.cos(viz_cfg.camera_elevation_angle) * math.sin(viz_cfg.camera_azimuth_angle)
    eye_y = viz_cfg.camera_center_y + viz_cfg.camera_radius * math.sin(viz_cfg.camera_elevation_angle)
    eye_z = viz_cfg.camera_center_z + viz_cfg.camera_radius * math.cos(viz_cfg.camera_elevation_angle) * math.cos(viz_cfg.camera_azimuth_angle)

    gluLookAt(eye_x, eye_y, eye_z,
              viz_cfg.camera_center_x, viz_cfg.camera_center_y, viz_cfg.camera_center_z,
              viz_cfg.camera_up_x, viz_cfg.camera_up_y, viz_cfg.camera_up_z)

    # Render Synapses
    if global_gui_state.get("show_connections_gl", False) and gl_num_synapse_lines_to_draw > 0 and \
       gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0:
        base_syn_color = opengl_viz_config.get('SYNAPSE_BASE_COLOR', [0.3, 0.3, 0.4])
        alpha_mod = opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.75)
        final_alpha = np.clip(0.15 * alpha_mod, 0.02, 0.5)
        glColor4f(base_syn_color[0], base_syn_color[1], base_syn_color[2], final_alpha)
        glLineWidth(0.5)

        glBindBuffer(GL_ARRAY_BUFFER, gl_synapse_vertices_vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINES, 0, gl_num_synapse_lines_to_draw * 2)
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
        glColorPointer(4, GL_FLOAT, 0, None)

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

        glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0))

    # Render Footer Text Overlay
    footer_h = opengl_viz_config.get('FOOTER_HEIGHT_PIXELS', 75)
    if footer_h > 0:
        line_h, margin = 15, 10
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800)

        sim_time_s = (runtime.current_time_ms / 1000.0)

        avg_fr = global_simulation_bridge._mock_network_avg_firing_rate_hz
        spikes_step = global_simulation_bridge._mock_num_spikes_this_step
        plasticity_events = global_simulation_bridge._mock_total_plasticity_events

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
        if global_gui_state.get("is_recording_active"):
            mode_text += " (Rec)"

        render_text_gl(margin, margin + 4 * line_h, f"Time: {sim_time_s:.3f}s")
        render_text_gl(margin + win_w // 3, margin + 4 * line_h, f"Spikes: {spikes_step}")
        render_text_gl(margin + 2 * win_w // 3, margin + 4 * line_h, fps_text)

        render_text_gl(margin, margin + 3 * line_h, f"Step: {runtime.current_time_step}")
        render_text_gl(margin + win_w // 3, margin + 3 * line_h, f"Rate: {avg_fr:.2f} Hz")
        render_text_gl(margin + 2 * win_w // 3, margin + 3 * line_h, f"Mode: {mode_text}")

        render_text_gl(margin, margin + 2 * line_h, f"Plasticity: {plasticity_events}")
        render_text_gl(margin + win_w // 3, margin + 2 * line_h, f"Vis.Neurons: {gl_num_neurons_to_draw}")
        render_text_gl(margin + 2 * win_w // 3, margin + 2 * line_h, f"Vis.Syns: {gl_num_synapse_lines_to_draw}")

        render_text_gl(margin, margin, "LMB:Rotate, RMB:Pan, Scroll:Zoom, R:Reset, S:Synapses, N:Neurons, Space:Pause/Resume, Esc:Exit")

    glut.glutSwapBuffers()
