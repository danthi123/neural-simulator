"""Typed, observable UI state management."""


class UIState:
    """Central UI state with change notification."""

    def __init__(self):
        self.selected_neurons = set()  # Neuron indices selected in 3D view
        self.active_experiment = None  # Current ExperimentConfig
        self.experiment_engine = None  # Current ExperimentEngine ref
        self.sweep_state = None       # Current sweep progress/results
        self.plot_visibility = {}     # {plot_name: bool}
        self.is_paused = False        # Plot update pause
        self._callbacks = {}          # {field_name: [callback]}

    def set(self, field, value):
        """Set a field and notify subscribers."""
        old = getattr(self, field, None)
        setattr(self, field, value)
        for cb in self._callbacks.get(field, []):
            try:
                cb(field, old, value)
            except Exception:
                pass

    def on_change(self, field, callback):
        """Register callback for field changes. callback(field, old_val, new_val)."""
        self._callbacks.setdefault(field, []).append(callback)
