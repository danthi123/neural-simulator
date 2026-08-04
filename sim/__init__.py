"""Core neural-simulator package with import-order-independent public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SimulationBridge": ("sim.bridge", "SimulationBridge"),
    "CoreSimConfig": ("sim.config", "CoreSimConfig"),
    "VisualizationConfig": ("sim.config", "VisualizationConfig"),
    "RuntimeState": ("sim.config", "RuntimeState"),
    "GPUConfig": ("sim.config", "GPUConfig"),
    "NeuronModel": ("sim.enums", "NeuronModel"),
    "NeuronType": ("sim.enums", "NeuronType"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
