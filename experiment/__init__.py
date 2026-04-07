"""Experiment system package.

Provides programmable stimulus injection, I/O neuron group management,
training protocols, readout/analysis, and multi-phase experiment execution.
"""

from experiment.engine import ExperimentEngine
from experiment.presets import ExperimentPresets
from experiment.readout import ReadoutEngine
from experiment.training import TrainingProtocolEngine
from experiment.stimulus import StimulusManager
from experiment.groups import NeuronGroupManager
