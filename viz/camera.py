"""Camera controls for the OpenGL visualization window.

Handles mouse (rotation, pan, zoom) and keyboard callbacks for GLUT,
plus the reshape callback for window resizing.
"""

import math
import numpy as np

try:
    from OpenGL.GL import *
    import OpenGL.GLUT as glut
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None

# Module-level references (set by renderer.set_shared_state via init below)
global_simulation_bridge = None
global_gui_state = None
opengl_viz_config = None
shutdown_flag = None
ui_to_sim_queue = None
_trigger_filter_update_signal = None
_update_ui_for_simulation_run_state = None

# Click-detection state for neuron picking (click = down+up at ~same position)
_mouse_down_pos = None  # (x, y) at mouse-down, or None
_CLICK_DRAG_THRESHOLD = 5  # pixels; movement beyond this is a drag, not a click

# UI state reference for selected neurons (set via set_ui_state)
_ui_state = None


def set_ui_state(ui_state):
    """Inject UIState reference for neuron selection tracking."""
    global _ui_state
    _ui_state = ui_state


def set_shared_state(sim_bridge, gui_state, viz_config, shutdown_evt,
                     ui_sim_queue, filter_fn, update_run_state_fn):
    """Inject shared references from neural-simulator.py."""
    global global_simulation_bridge, global_gui_state, opengl_viz_config
    global shutdown_flag, ui_to_sim_queue
    global _trigger_filter_update_signal, _update_ui_for_simulation_run_state
    global_simulation_bridge = sim_bridge
    global_gui_state = gui_state
    opengl_viz_config = viz_config
    shutdown_flag = shutdown_evt
    ui_to_sim_queue = ui_sim_queue
    _trigger_filter_update_signal = filter_fn
    _update_ui_for_simulation_run_state = update_run_state_fn


def reshape_gl_window(width, height):
    """Handles OpenGL window reshape events. Called by GLUT in the main thread."""
    if not OPENGL_AVAILABLE or height <= 0 or global_simulation_bridge is None:
        return
    viz_cfg = global_simulation_bridge.viz_config

    opengl_viz_config['WINDOW_WIDTH'] = width
    opengl_viz_config['WINDOW_HEIGHT'] = height

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(viz_cfg.camera_fov, float(width) / float(height),
                   viz_cfg.camera_near_clip, viz_cfg.camera_far_clip)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def mouse_button_func_gl(button, state, x, y):
    """Handles mouse button events for OpenGL window (camera control). Called by GLUT."""
    global _mouse_down_pos
    if not global_simulation_bridge:
        return
    cfg = global_simulation_bridge.viz_config
    zoom_speed = opengl_viz_config.get("CAMERA_ZOOM_SPEED_FACTOR", 20.0)

    if button == glut.GLUT_LEFT_BUTTON:
        if state == glut.GLUT_DOWN:
            _mouse_down_pos = (x, y)
            cfg.mouse_left_button_down = True
        else:
            cfg.mouse_left_button_down = False
            # Detect click (not drag): mouse-up near mouse-down position
            if _mouse_down_pos is not None:
                dx = abs(x - _mouse_down_pos[0])
                dy = abs(y - _mouse_down_pos[1])
                if dx <= _CLICK_DRAG_THRESHOLD and dy <= _CLICK_DRAG_THRESHOLD:
                    print(f"[Camera] Click detected at ({x},{y}), calling picker...")
                    _handle_neuron_pick(x, y)
                _mouse_down_pos = None
    elif button == glut.GLUT_RIGHT_BUTTON:
        cfg.mouse_right_button_down = (state == glut.GLUT_DOWN)
    elif button == 3:  # Scroll up (zoom in)
        if state == glut.GLUT_UP:
            return
        cfg.camera_radius = max(cfg.camera_near_clip * 2, cfg.camera_radius - zoom_speed)
    elif button == 4:  # Scroll down (zoom out)
        if state == glut.GLUT_UP:
            return
        cfg.camera_radius += zoom_speed
        cfg.camera_radius = min(cfg.camera_radius, cfg.camera_far_clip * 0.8)

    cfg.mouse_last_x = x
    cfg.mouse_last_y = y
    if glut.glutGetWindow() != 0:
        glut.glutPostRedisplay()


def _handle_neuron_pick(x, y):
    """Perform color-based neuron picking at screen coordinates (x, y)."""
    if not global_simulation_bridge or _ui_state is None:
        print(f"[Picker] Early return: bridge={global_simulation_bridge is not None}, ui_state={_ui_state is not None}")
        return

    try:
        import cupy as cp
        from viz.picker import pick_neuron_at

        bridge = global_simulation_bridge
        if bridge.cp_neuron_positions_3d is None or bridge.cp_neuron_positions_3d.shape[0] == 0:
            return

        positions_np = cp.asnumpy(bridge.cp_neuron_positions_3d)
        num_neurons = bridge.core_config.num_neurons
        viz_cfg = bridge.viz_config

        idx = pick_neuron_at(x, y, positions_np, num_neurons, viz_cfg)

        if idx >= 0:
            _ui_state.set("selected_neurons", {idx})
            print(f"Selected neuron #{idx}")
        else:
            _ui_state.set("selected_neurons", set())

    except Exception as e:
        print(f"Neuron pick error: {e}")


def mouse_motion_func_gl(x, y):
    """Handles mouse motion events for OpenGL window (camera control). Called by GLUT."""
    if not global_simulation_bridge:
        return
    cfg = global_simulation_bridge.viz_config
    dx = x - cfg.mouse_last_x
    dy = y - cfg.mouse_last_y

    rotate_speed = opengl_viz_config.get("CAMERA_ROTATE_SPEED_FACTOR", 0.005)
    pan_speed_config = opengl_viz_config.get("CAMERA_PAN_SPEED_FACTOR", 0.1)

    if cfg.mouse_left_button_down:  # Rotate camera
        cfg.camera_azimuth_angle -= dx * rotate_speed
        cfg.camera_elevation_angle -= dy * rotate_speed
        cfg.camera_elevation_angle = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, cfg.camera_elevation_angle))
    elif cfg.mouse_right_button_down:  # Pan camera
        eye_calc_x = cfg.camera_center_x + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.sin(cfg.camera_azimuth_angle)
        eye_calc_y = cfg.camera_center_y + cfg.camera_radius * math.sin(cfg.camera_elevation_angle)
        eye_calc_z = cfg.camera_center_z + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.cos(cfg.camera_azimuth_angle)
        eye = np.array([eye_calc_x, eye_calc_y, eye_calc_z])

        center = np.array([cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z])
        up_world = np.array([cfg.camera_up_x, cfg.camera_up_y, cfg.camera_up_z])

        forward = center - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward /= forward_norm
        else:
            forward = np.array([0, 0, -1])

        right = np.cross(forward, up_world)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right /= right_norm
        else:
            if abs(forward[1]) > 0.99:
                right = np.array([1, 0, 0])
            else:
                right_temp = np.cross(forward, np.array([0, 1, 0]))
                right_norm_temp = np.linalg.norm(right_temp)
                right = right_temp / right_norm_temp if right_norm_temp > 1e-6 else np.array([1, 0, 0])

        cam_up = np.cross(right, forward)

        pan_scale = pan_speed_config * (cfg.camera_radius / 150.0)
        pan_vector_x = -dx * right * pan_scale
        pan_vector_y = dy * cam_up * pan_scale

        new_center = center + pan_vector_x + pan_vector_y
        cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = new_center[0], new_center[1], new_center[2]

    cfg.mouse_last_x = x
    cfg.mouse_last_y = y
    if glut.glutGetWindow() != 0:
        glut.glutPostRedisplay()


def keyboard_func_gl(key, x, y):
    """Handles keyboard events for the OpenGL window. Called by GLUT."""
    if global_simulation_bridge is None:
        return

    if key == b'\x1b':  # ESC key
        print("ESC pressed in OpenGL window. Signaling shutdown.")
        shutdown_flag.set()
        return

    try:
        key_char = key.decode("utf-8").lower()
    except UnicodeDecodeError:
        return

    cfg = global_simulation_bridge.viz_config

    if key_char == 's':  # Toggle synapse visibility
        new_show_state = not global_gui_state.get("show_connections_gl", False)
        global_gui_state["show_connections_gl"] = new_show_state
        if dpg is not None and dpg.is_dearpygui_running() and dpg.does_item_exist("filter_show_synapses_gl_cb"):
            dpg.set_value("filter_show_synapses_gl_cb", new_show_state)
        if _trigger_filter_update_signal:
            _trigger_filter_update_signal()
        print(f"Synapse visibility toggled {'on' if new_show_state else 'off'}.")

    elif key_char == 'n':  # Cycle through neuron spiking display modes
        if dpg is not None and dpg.is_dearpygui_running() and dpg.does_item_exist("filter_spiking_mode_combo"):
            modes = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
            current_mode = dpg.get_value("filter_spiking_mode_combo")
            try:
                current_idx = modes.index(current_mode)
                next_idx = (current_idx + 1) % len(modes)
            except ValueError:
                next_idx = 0
            new_mode = modes[next_idx]
            dpg.set_value("filter_spiking_mode_combo", new_mode)
            if _trigger_filter_update_signal:
                _trigger_filter_update_signal()
            print(f"Neuron display mode: {new_mode}")

    elif key_char == ' ':  # Space: Pause/Resume or Start simulation
        if not global_gui_state.get("is_playback_mode_active", False):
            current_sim_running = global_gui_state.get("_sim_is_running_ui_view", False)
            current_sim_paused = global_gui_state.get("_sim_is_paused_ui_view", False)

            if not current_sim_running:
                ui_to_sim_queue.put({"type": "START_SIM"})
                global_gui_state["_sim_is_running_ui_view"] = True
                global_gui_state["_sim_is_paused_ui_view"] = False
                if _update_ui_for_simulation_run_state:
                    _update_ui_for_simulation_run_state(is_running=True, is_paused=False)
                print("GL Keyboard: Starting simulation.")
            elif current_sim_paused:
                ui_to_sim_queue.put({"type": "RESUME_SIM"})
                global_gui_state["_sim_is_paused_ui_view"] = False
                if _update_ui_for_simulation_run_state:
                    _update_ui_for_simulation_run_state(is_running=True, is_paused=False)
                print("GL Keyboard: Resuming simulation.")
            else:
                ui_to_sim_queue.put({"type": "PAUSE_SIM"})
                global_gui_state["_sim_is_paused_ui_view"] = True
                if _update_ui_for_simulation_run_state:
                    _update_ui_for_simulation_run_state(is_running=True, is_paused=True)
                print("GL Keyboard: Pausing simulation.")

    elif key_char == 'r':  # Reset camera position
        cfg.camera_azimuth_angle = 0.0
        cfg.camera_elevation_angle = 0.0
        cfg.camera_radius = 150.0
        cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = 0.0, 0.0, 0.0
        if glut.glutGetWindow() != 0:
            glut.glutPostRedisplay()
        print("Camera reset.")

    if glut.glutGetWindow() != 0:
        glut.glutPostRedisplay()
