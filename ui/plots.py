"""Live DearPyGUI plots for real-time simulation monitoring."""
import numpy as np
from collections import deque

try:
    import dearpygui.dearpygui as dpg
    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False

# Fixed subsample size for raster plot
_RASTER_SUBSAMPLE_N = 100  # 50 excitatory + 50 inhibitory target
_RASTER_TIME_WINDOW_S = 5.0  # Default visible time window


def create_raster_plot(parent, data_bus, tag_prefix="raster"):
    """Create a spike raster plot. Returns update function."""
    if not DPG_AVAILABLE:
        return lambda: None

    plot_tag = f"{tag_prefix}_plot"
    series_tag = f"{tag_prefix}_series"

    with dpg.collapsing_header(label="Spike Raster", parent=parent, default_open=False):
        with dpg.plot(label="Spike Raster", height=200, width=-1, tag=plot_tag):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=f"{tag_prefix}_x")
            with dpg.plot_axis(dpg.mvYAxis, label="Neuron (subsampled)", tag=f"{tag_prefix}_y"):
                dpg.add_scatter_series([], [], tag=series_tag)
        dpg.set_axis_limits(f"{tag_prefix}_y", 0, _RASTER_SUBSAMPLE_N)

    spike_buffer = deque(maxlen=5000)

    # Fixed set of subsampled neuron indices (built lazily on first spike event)
    _sampled_indices = {}  # global_idx -> display_idx (0..99)
    _initialized = [False]

    def _build_subsample(num_neurons):
        """Build a fixed set of 100 subsampled neuron indices."""
        if num_neurons <= _RASTER_SUBSAMPLE_N:
            for i in range(num_neurons):
                _sampled_indices[i] = i
        else:
            # Take evenly spaced neurons: first 50 from lower half, last 50 from upper half
            half = num_neurons // 2
            exc_indices = np.linspace(0, half - 1, 50, dtype=int)
            inh_indices = np.linspace(half, num_neurons - 1, 50, dtype=int)
            all_indices = np.concatenate([exc_indices, inh_indices])
            for display_idx, global_idx in enumerate(all_indices):
                _sampled_indices[int(global_idx)] = display_idx
        _initialized[0] = True

    def on_spike_event(data):
        spike_buffer.append(data)
        if not _initialized[0]:
            indices = data.get("neuron_indices")
            if indices is not None and len(indices) > 0:
                # Estimate total neuron count from max index seen
                max_idx = int(np.max(indices))
                _build_subsample(max(max_idx + 1, 1000))

    if data_bus:
        data_bus.subscribe("spike_events", on_spike_event)

    def update():
        if not spike_buffer:
            return
        times, neurons = [], []
        for evt in spike_buffer:
            t = evt["time_ms"] / 1000.0
            for idx in evt["neuron_indices"]:
                idx_int = int(idx)
                if idx_int in _sampled_indices:
                    times.append(t)
                    neurons.append(_sampled_indices[idx_int])
        if times:
            dpg.set_value(series_tag, [times, neurons])
            # Auto-fit x-axis to recent window
            t_max = max(times)
            dpg.set_axis_limits(f"{tag_prefix}_x", t_max - _RASTER_TIME_WINDOW_S, t_max)

    return update


def create_firing_rate_plot(parent, data_bus, tag_prefix="rate"):
    """Create a population firing rate trace. Returns update function."""
    if not DPG_AVAILABLE:
        return lambda: None

    series_tag = f"{tag_prefix}_series"

    with dpg.collapsing_header(label="Population Firing Rate", parent=parent, default_open=False):
        with dpg.plot(label="Firing Rate", height=150, width=-1, tag=f"{tag_prefix}_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=f"{tag_prefix}_x")
            with dpg.plot_axis(dpg.mvYAxis, label="Rate (Hz)", tag=f"{tag_prefix}_y"):
                dpg.add_line_series([], [], label="Population", tag=series_tag)
        # Set sensible initial axis limits
        dpg.set_axis_limits(f"{tag_prefix}_x", 0, 10)
        dpg.set_axis_limits(f"{tag_prefix}_y", 0, 20)

    times = deque(maxlen=10000)
    rates = deque(maxlen=10000)

    def on_rate(data):
        times.append(data["time_ms"] / 1000.0)
        rates.append(data["rate_hz"])

    if data_bus:
        data_bus.subscribe("firing_rates", on_rate)

    def update():
        if not times:
            return
        t_list = list(times)
        r_list = list(rates)
        dpg.set_value(series_tag, [t_list, r_list])
        if t_list:
            t_max = t_list[-1]
            dpg.set_axis_limits(f"{tag_prefix}_x", t_max - 10.0, t_max)
            # Auto-expand Y if rates exceed current limit
            r_max = max(r_list) if r_list else 0
            y_limit = max(20, r_max * 1.2)
            dpg.set_axis_limits(f"{tag_prefix}_y", 0, y_limit)

    return update


def create_weight_histogram(parent, data_bus, tag_prefix="whist"):
    """Create a weight distribution histogram. Returns update function."""
    if not DPG_AVAILABLE:
        return lambda: None

    series_tag = f"{tag_prefix}_series"

    with dpg.collapsing_header(label="Weight Distribution", parent=parent, default_open=False):
        with dpg.plot(label="Synaptic Weights", height=170, width=-1, tag=f"{tag_prefix}_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Count", tag=f"{tag_prefix}_x")
            with dpg.plot_axis(dpg.mvYAxis, label="Weight", tag=f"{tag_prefix}_y"):
                dpg.add_bar_series([], [], tag=series_tag, weight=0.06, horizontal=True)
            dpg.set_axis_limits(f"{tag_prefix}_y", 0.0, 2.0)

    def update():
        history = data_bus.get_history("weights", 1) if data_bus else []
        if not history:
            return
        weights = history[-1].get("weights")
        if weights is None or len(weights) == 0:
            return
        counts, edges = np.histogram(weights, bins=25, range=(0, 2))
        centers = (edges[:-1] + edges[1:]) / 2
        # Horizontal bars: x=counts (bar length), y=weight bin centers (bar position)
        dpg.set_value(series_tag, [centers.tolist(), counts.tolist()])

    return update


def create_live_plots(parent_tag, data_bus, plot_manager):
    """Create all live monitoring plots and register them with the plot manager.

    Call this after the DPG context and layout are created, when the data_bus is available.
    """
    if not DPG_AVAILABLE:
        return

    raster_update = create_raster_plot(parent_tag, data_bus)
    rate_update = create_firing_rate_plot(parent_tag, data_bus)
    weight_update = create_weight_histogram(parent_tag, data_bus)

    if plot_manager:
        plot_manager.register_plot("raster", raster_update, update_interval_ms=100)
        plot_manager.register_plot("firing_rate", rate_update, update_interval_ms=100)
        plot_manager.register_plot("weight_hist", weight_update, update_interval_ms=1000)
