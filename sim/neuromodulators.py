"""Neuromodulator subsystem (Session E.1).

Models hormones / neuromodulators as declarative entities with concentration
dynamics and configurable effects on neuronal/synaptic state. Replaces the
one-off `current_reward_signal` and shelved `cp_synaptic_gain_modulator`
mechanisms with a real framework.

See:
- docs/plans/2026-04-24-neuromodulator-subsystem.md
- research/findings/2026-04-24-session-d-part-a.md §4 (motivation: silent-motor trap)
- research/findings/2026-04-24-session-c.md §4 (why eligibility-only modulation fails)

Concentration semantics: scalar per modulator (global broadcast). Each step
the concentration decays toward `baseline` with `decay_tau_ms`, then any
production rules add to it. Effects are applied to bridge state via target
configurations.

Backward compatibility: opt-in via `core_config.enable_neuromodulator_subsystem`.
When disabled (default), no code paths in this module run; legacy reward
modulation continues unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class ModulatorTarget:
    """How a modulator's concentration affects bridge state.

    target_type:
        "synaptic_gain"
            Multiplies effective_synaptic_strength.
            effect = 1.0 + sensitivity * (conc - baseline)
        "plasticity_rate"
            Multiplies STDP amplitudes and reward_learning_rate.
            effect = 1.0 + sensitivity * (conc - baseline)
        "excitability_drive"
            Adds current to membrane drive (pA).
            effect = sensitivity * (conc - baseline)

    scope: which neurons / synapses are affected.
        "all"               every neuron / synapse
        "trait:<idx>"       neurons whose cp_traits == idx
        "group:<name>"      neurons registered with the experiment engine
                            under that group name
        "plastic_only"      synapses with cp_synapse_plastic_mask == True
                            (synaptic_gain & plasticity_rate only)

    sensitivity: scaling factor in the effect formulas above.
        0.0 disables the target without removing it.
    """

    target_type: str
    scope: str = "all"
    sensitivity: float = 1.0


@dataclass
class ProductionRule:
    """How bridge state drives modulator concentration.

    rule_type:
        "manual"
            Concentration is only changed externally (via set_concentration).
            Useful for experiments and testing.
        "from_reward"
            On each step, adds sensitivity * (current_reward_signal - baseline)
            to the modulator concentration. Models phasic dopamine.
        "from_error_persistence"
            Tracks an EMA of |reward_error| over `window_ms`. When the EMA
            exceeds `threshold`, produces sensitivity * (ema - threshold) *
            (dt/1000) per step. Models tonic noradrenaline rising under
            sustained negative-reward stress.
        "from_novelty"
            (Reserved for future ACh; emits 0 for now.)

    sensitivity, threshold, window_ms: tunable per rule.
    """

    rule_type: str
    sensitivity: float = 1.0
    threshold: float = 0.5
    window_ms: float = 500.0


@dataclass
class NeuromodulatorConfig:
    """Declarative description of one hormone / neuromodulator.

    The simulator owns the concentration dynamics; the user only supplies
    the parameters and the receptor targets / production rules.
    """

    name: str
    baseline: float = 0.0
    decay_tau_ms: float = 500.0
    concentration_min: float = 0.0
    concentration_max: float = 5.0
    targets: List[ModulatorTarget] = field(default_factory=list)
    production_rules: List[ProductionRule] = field(default_factory=list)
