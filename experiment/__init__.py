"""Experiment system package with lightweight, lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ExperimentEngine": ("experiment.engine", "ExperimentEngine"),
    "ExperimentPresets": ("experiment.presets", "ExperimentPresets"),
    "ReadoutEngine": ("experiment.readout", "ReadoutEngine"),
    "TrainingProtocolEngine": ("experiment.training", "TrainingProtocolEngine"),
    "StimulusManager": ("experiment.stimulus", "StimulusManager"),
    "NeuronGroupManager": ("experiment.groups", "NeuronGroupManager"),
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
