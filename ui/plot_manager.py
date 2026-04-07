"""Plot management: creation, synchronized updates, performance budgeting."""
import time

try:
    import dearpygui.dearpygui as dpg
    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False


class PlotManager:
    """Manages live DearPyGUI plots with staggered updates."""

    def __init__(self, data_bus=None):
        self.data_bus = data_bus
        self._plots = {}  # name -> PlotConfig
        self._update_index = 0
        self._paused = False
        self.time_window_s = 10.0  # Default visible time window

    def register_plot(self, name, update_fn, update_interval_ms=100):
        """Register a plot with its update function."""
        self._plots[name] = {
            "update_fn": update_fn,
            "interval_ms": update_interval_ms,
            "last_update": 0,
        }

    def update_all(self):
        """Called each UI frame. Staggers plot updates for performance."""
        if self._paused or not self._plots:
            return
        now = time.time() * 1000
        for name, cfg in self._plots.items():
            if now - cfg["last_update"] >= cfg["interval_ms"]:
                try:
                    cfg["update_fn"]()
                    cfg["last_update"] = now
                except Exception:
                    pass

    def set_paused(self, paused):
        self._paused = paused
