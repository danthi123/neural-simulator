"""Enhanced experiment dashboard with phase timeline and controls.

Replaces the minimal experiment UI with a richer control panel featuring:
- Visual phase timeline (colored blocks per phase type)
- Phase detail table (name, type, duration, channels, reps)
- Enhanced status display with readout rates and training progress
- Custom experiment config save/load buttons
"""
import json
import time

try:
    import dearpygui.dearpygui as dpg
    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False


PHASE_COLORS = {
    "BASELINE": (128, 128, 128, 200),
    "STIMULUS": (70, 130, 255, 200),
    "TRAINING": (70, 200, 100, 200),
    "TESTING": (255, 180, 50, 200),
    "REST": (100, 100, 100, 150),
}


def create_experiment_dashboard(parent, preset_names, callbacks_module=None):
    """Create the experiment dashboard inside *parent*.

    Uses the same DPG widget tags as the old minimal experiment UI so that
    existing callbacks in ``ui.callbacks`` and ``neural-simulator.py``
    continue to work without modification.

    Args:
        parent: DPG parent container tag.
        preset_names: List of preset display names for the combo box.
        callbacks_module: Reference to ``ui.callbacks`` module for queue access.

    Returns:
        Callable ``update(engine, status)`` to refresh the dashboard each frame.
    """
    if not DPG_AVAILABLE:
        return lambda engine, status: None

    # --- Preset selector (keeps the original tag) ---
    dpg.add_text("Experiment Preset:", parent=parent, color=[180, 220, 255])
    dpg.add_combo(preset_names, default_value=preset_names[0] if preset_names else "",
                  tag="experiment_preset_combo", width=-1, parent=parent,
                  callback=lambda s, a, u: _on_preset_change(a, callbacks_module))
    dpg.add_spacer(height=3, parent=parent)

    # --- Experiment info (keeps original tag for neural-simulator.py compat) ---
    dpg.add_text("No experiment loaded.", tag="experiment_info_text",
                 color=[150, 150, 150], parent=parent)
    dpg.add_separator(parent=parent)

    # --- Phase timeline (drawlist) ---
    dpg.add_text("Phase Timeline:", parent=parent, color=[180, 220, 255])
    dpg.add_drawlist(width=-1, height=50, tag="phase_timeline_drawlist", parent=parent)
    dpg.add_separator(parent=parent)

    # --- Phase detail table ---
    with dpg.table(tag="phase_table", header_row=True, parent=parent,
                   borders_innerH=True, borders_outerH=True,
                   borders_innerV=True, borders_outerV=True):
        dpg.add_table_column(label="Phase")
        dpg.add_table_column(label="Type")
        dpg.add_table_column(label="Duration")
        dpg.add_table_column(label="Channels")
        dpg.add_table_column(label="Reps")

    dpg.add_separator(parent=parent)

    # --- Control buttons (keeps original tags) ---
    with dpg.group(horizontal=True, parent=parent):
        dpg.add_button(label="Start Experiment", tag="btn_start_experiment", width=120,
                       callback=lambda: _start_experiment(callbacks_module))
        dpg.add_button(label="Stop Experiment", tag="btn_stop_experiment", width=120,
                       callback=lambda: _stop_experiment(callbacks_module))
        dpg.add_button(label="Save Log", tag="btn_save_experiment_log", width=80,
                       callback=lambda: _save_log(callbacks_module))
    dpg.add_spacer(height=5, parent=parent)

    # --- Status display (keeps original tags for callbacks compat) ---
    dpg.add_text("Status:", color=[180, 220, 255], parent=parent)
    dpg.add_text("Idle", tag="experiment_status_text", color=[150, 150, 150], parent=parent)
    dpg.add_spacer(height=3, parent=parent)

    dpg.add_text("Phase: --", tag="experiment_phase_text", color=[150, 150, 150], parent=parent)
    dpg.add_spacer(height=3, parent=parent)

    dpg.add_text("Readout Rates:", color=[180, 220, 255], parent=parent)
    dpg.add_text("No data", tag="experiment_readout_text", color=[150, 150, 150], parent=parent)
    dpg.add_spacer(height=3, parent=parent)

    dpg.add_text("Training:", color=[180, 220, 255], parent=parent)
    dpg.add_text("No training active", tag="experiment_training_text",
                 color=[150, 150, 150], parent=parent)
    dpg.add_separator(parent=parent)

    # --- Save / Load custom experiment config ---
    with dpg.group(horizontal=True, parent=parent):
        dpg.add_button(label="Save Config", tag="exp_save_config_btn", width=100,
                       callback=lambda: _save_config(callbacks_module))
        dpg.add_button(label="Load Config", tag="exp_load_config_btn", width=100,
                       callback=lambda: _load_config(callbacks_module))
    dpg.add_spacer(height=5, parent=parent)

    # -----------------------------------------------------------------
    # Update function returned to caller
    # -----------------------------------------------------------------
    def update(engine, status):
        """Refresh the dashboard widgets from *status* dict each frame."""
        if engine is None:
            return
        try:
            if status:
                is_running = status.get("is_running", False)
                phase_name = status.get("current_phase_name", "")
                phase_idx = status.get("current_phase_idx", 0)
                total_phases = status.get("total_phases", 0)

                # The old callbacks already set experiment_status_text etc.
                # We layer on richer per-frame info for the dashboard widgets.

                if is_running and phase_name and total_phases > 0:
                    pct = (phase_idx / max(total_phases, 1)) * 100
                    _update_timeline_marker(phase_idx, total_phases)
        except Exception:
            pass

    return update


# ── Populate helpers (called when an experiment config is loaded) ─────────

def populate_experiment_info(config):
    """Fill the dashboard with loaded experiment config info.

    Called from the main app after LOAD_EXPERIMENT_PRESET completes.
    """
    if not DPG_AVAILABLE or config is None:
        return
    try:
        total_ms = sum(p.duration_ms * p.num_repetitions for p in config.phases)
        dpg.set_value("experiment_info_text",
                      f"{config.name} — {total_ms/1000:.1f}s total, {len(config.phases)} phases")
        dpg.configure_item("experiment_info_text", color=[100, 255, 100])

        _populate_phase_table(config.phases)
        _draw_phase_timeline(config.phases, total_ms)
    except Exception:
        pass


def _populate_phase_table(phases):
    """Fill the phase detail table."""
    # Clear existing rows
    for child in dpg.get_item_children("phase_table", 1) or []:
        dpg.delete_item(child)

    for phase in phases:
        with dpg.table_row(parent="phase_table"):
            dpg.add_text(phase.name)
            dpg.add_text(phase.phase_type)
            dpg.add_text(f"{phase.duration_ms:.0f}ms")
            channels = phase.active_channels if phase.active_channels else []
            dpg.add_text(", ".join(channels) if channels else "-")
            dpg.add_text(str(phase.num_repetitions))


def _draw_phase_timeline(phases, total_ms):
    """Draw colored phase blocks on the drawlist timeline."""
    if total_ms <= 0:
        return

    drawlist = "phase_timeline_drawlist"
    dpg.delete_item(drawlist, children_only=True)

    width = dpg.get_item_width(drawlist) or 400
    height = 50
    x = 0.0

    for phase in phases:
        phase_total = phase.duration_ms * phase.num_repetitions
        w = max(2.0, (phase_total / total_ms) * width)
        color = PHASE_COLORS.get(phase.phase_type, (128, 128, 128, 200))

        dpg.draw_rectangle((x, 2), (x + w, height - 2), fill=color, parent=drawlist)

        # Label if block is wide enough for text
        if w > 30:
            label = phase.name[:int(w / 7)]
            dpg.draw_text((x + 3, 15), label, size=12,
                          color=(255, 255, 255, 255), parent=drawlist)

        x += w


def _update_timeline_marker(current_phase_idx, total_phases):
    """Draw a vertical marker on the timeline showing current phase position."""
    try:
        drawlist = "phase_timeline_drawlist"
        width = dpg.get_item_width(drawlist) or 400
        height = 50

        # Remove any previous marker (tagged)
        if dpg.does_item_exist("__timeline_marker"):
            dpg.delete_item("__timeline_marker")

        if total_phases <= 0:
            return
        marker_x = (current_phase_idx / total_phases) * width
        dpg.draw_line((marker_x, 0), (marker_x, height),
                      color=(255, 50, 50, 255), thickness=2,
                      parent=drawlist, tag="__timeline_marker")
    except Exception:
        pass


# ── Button callbacks ─────────────────────────────────────────────────────

def _on_preset_change(preset_name, callbacks_module):
    if callbacks_module and hasattr(callbacks_module, '_handle_experiment_preset_change'):
        callbacks_module._handle_experiment_preset_change(preset_name)


def _start_experiment(callbacks_module):
    if callbacks_module and hasattr(callbacks_module, 'ui_to_sim_queue'):
        callbacks_module.ui_to_sim_queue.put({"type": "START_EXPERIMENT"})


def _stop_experiment(callbacks_module):
    if callbacks_module and hasattr(callbacks_module, 'ui_to_sim_queue'):
        callbacks_module.ui_to_sim_queue.put({"type": "STOP_EXPERIMENT"})


def _save_log(callbacks_module):
    if callbacks_module and hasattr(callbacks_module, 'ui_to_sim_queue'):
        callbacks_module.ui_to_sim_queue.put({
            "type": "SAVE_EXPERIMENT_LOG",
            "filepath": f"experiment_log_{int(time.time())}.json"
        })


def _save_config(callbacks_module):
    """Placeholder for file-dialog based experiment config save."""
    pass  # TODO: file dialog for saving custom config


def _load_config(callbacks_module):
    """Placeholder for file-dialog based experiment config load."""
    pass  # TODO: file dialog for loading custom config
