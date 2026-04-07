"""Neuron inspection panel showing details of selected neurons."""

import numpy as np

try:
    import dearpygui.dearpygui as dpg
    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False


def create_inspector_panel(parent):
    """Create the neuron inspector UI section.

    Returns:
        callable: update(ui_state, sim_bridge) to refresh inspector data.
    """
    if not DPG_AVAILABLE:
        return lambda ui_state, sim_bridge: None

    with dpg.collapsing_header(label="Neuron Inspector", parent=parent,
                                default_open=False, tag="inspector_header"):
        dpg.add_text("Click a neuron in 3D view to inspect", tag="inspector_hint",
                     color=[150, 150, 150])
        dpg.add_separator()

        # Identity
        dpg.add_text("", tag="inspector_identity")
        dpg.add_text("", tag="inspector_trait")
        dpg.add_text("", tag="inspector_group")
        dpg.add_separator()

        # Live state
        dpg.add_text("State:", color=[100, 200, 255])
        dpg.add_text("", tag="inspector_voltage")
        dpg.add_text("", tag="inspector_rate")
        dpg.add_text("", tag="inspector_last_spike")
        dpg.add_separator()

        # Connectivity
        dpg.add_text("Connectivity:", color=[100, 200, 255])
        dpg.add_text("", tag="inspector_connections")
        dpg.add_text("", tag="inspector_weights")

    def update(ui_state, sim_bridge):
        """Update inspector with selected neuron data."""
        if ui_state is None or not ui_state.selected_neurons or sim_bridge is None:
            return

        idx = min(ui_state.selected_neurons)  # Show first selected
        n = sim_bridge.core_config.num_neurons
        if idx < 0 or idx >= n:
            return

        try:
            import cupy as cp

            # Identity
            trait = -1
            if sim_bridge.cp_traits is not None and idx < len(sim_bridge.cp_traits):
                trait = int(sim_bridge.cp_traits[idx].get())
            inh_idx = getattr(sim_bridge.core_config, 'inhibitory_trait_index', 1)
            trait_name = "Inhibitory" if trait == inh_idx else "Excitatory"
            dpg.set_value("inspector_identity", f"Neuron #{idx}")
            dpg.set_value("inspector_trait", f"Type: {trait_name} (trait {trait})")

            # Group membership
            group_name = "none"
            if hasattr(sim_bridge, 'experiment_engine') and sim_bridge.experiment_engine is not None:
                gm = getattr(sim_bridge.experiment_engine, 'group_manager', None)
                if gm is not None:
                    for g in getattr(gm, 'groups', []):
                        if g.index_start <= idx < g.index_end:
                            group_name = f"{g.name} ({g.role})"
                            break
            dpg.set_value("inspector_group", f"Group: {group_name}")

            # Voltage
            v = 0.0
            if sim_bridge.cp_membrane_potential_v is not None and idx < len(sim_bridge.cp_membrane_potential_v):
                v = float(sim_bridge.cp_membrane_potential_v[idx].get())
            dpg.set_value("inspector_voltage", f"Membrane potential: {v:.1f} mV")

            # Firing rate placeholder
            dpg.set_value("inspector_rate", "")

            # Last spike time
            if sim_bridge.cp_last_spike_time is not None and idx < len(sim_bridge.cp_last_spike_time):
                last_spike = float(sim_bridge.cp_last_spike_time[idx].get())
                current_t = sim_bridge.runtime_state.current_time_ms
                dpg.set_value("inspector_last_spike",
                              f"Last spike: {last_spike:.1f} ms (t={current_t:.0f})")
            else:
                dpg.set_value("inspector_last_spike", "Last spike: N/A")

            # Connectivity
            if sim_bridge.cp_connections is not None and sim_bridge.cp_connections.nnz > 0:
                coo = sim_bridge._get_cached_coo()
                if coo is not None:
                    row_np = cp.asnumpy(coo.row)
                    col_np = cp.asnumpy(coo.col)
                    outgoing = int(np.sum(row_np == idx))
                    incoming = int(np.sum(col_np == idx))
                    dpg.set_value("inspector_connections",
                                  f"Synapses: {incoming} in / {outgoing} out")

                    # Mean outgoing weight
                    out_mask = row_np == idx
                    if np.any(out_mask):
                        out_indices = np.where(out_mask)[0]
                        out_weights = cp.asnumpy(coo.data[cp.array(out_indices)])
                        dpg.set_value("inspector_weights",
                                      f"Mean out weight: {out_weights.mean():.4f}")
                    else:
                        dpg.set_value("inspector_weights", "Mean out weight: N/A")
                else:
                    dpg.set_value("inspector_connections", "Synapses: N/A")
                    dpg.set_value("inspector_weights", "")
            else:
                dpg.set_value("inspector_connections", "Synapses: none")
                dpg.set_value("inspector_weights", "")

        except Exception as e:
            dpg.set_value("inspector_identity", f"Error: {str(e)[:60]}")

    return update
