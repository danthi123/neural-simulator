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


class NeuromodulatorManager:
    """Owns per-modulator concentration state and applies effects each step.

    Lifecycle:
        mgr = NeuromodulatorManager(configs, dt_ms)
        mgr.initialize(n_neurons, cp_module)            # called once after bridge has cp + n
        mgr.set_group_indices({"motor": [...], ...})    # optional, for group:NAME scopes
        # per simulation step:
        mgr.step(bridge)                                # decay + production
        # query effects to apply:
        mgr.compute_synaptic_gain_multiplier()
        mgr.compute_plasticity_rate_multiplier()
        mgr.compute_excitability_drive_pA()             # scalar
        mgr.compute_excitability_drive_per_neuron(cp_traits=..., group_indices=...)
    """

    def __init__(self, configs: Sequence[NeuromodulatorConfig], dt_ms: float):
        self._configs = list(configs)
        self.dt_ms = float(dt_ms)
        self._concentrations: dict[str, float] = {}
        self._cp = None
        self._n_neurons = 0
        # Per-rule running state (e.g. EMA of reward error)
        self._rule_state: dict[str, dict] = {}
        # Optional cached group indices: {group_name: list[int]}
        self._group_indices: dict[str, list[int]] = {}

    def initialize(self, n_neurons: int, cp_module) -> None:
        self._cp = cp_module
        self._n_neurons = int(n_neurons)
        self._concentrations = {c.name: float(c.baseline) for c in self._configs}
        self._rule_state = {c.name: {"err_ema": 0.0} for c in self._configs}

    def get_concentration(self, name: str) -> float:
        return self._concentrations[name]

    def set_concentration(self, name: str, value: float) -> None:
        """Manually set a concentration, clipped to the modulator's bounds.

        Useful for tests, manual probes, and 'manual' production rules.
        """
        cfg = self._config_by_name(name)
        v = max(cfg.concentration_min, min(cfg.concentration_max, float(value)))
        self._concentrations[name] = v

    def modulator_names(self) -> List[str]:
        return list(self._concentrations.keys())

    def _config_by_name(self, name: str) -> NeuromodulatorConfig:
        for c in self._configs:
            if c.name == name:
                return c
        raise KeyError(name)

    def step(self, bridge) -> None:
        """One simulation step: decay each concentration toward baseline,
        then add production-rule contributions, then clip.

        bridge can be None for unit tests that only exercise decay (no
        production rules will fire without bridge state).
        """
        for cfg in self._configs:
            conc = self._concentrations[cfg.name]

            # Exponential decay toward baseline.
            decay_factor = math.exp(-self.dt_ms / max(cfg.decay_tau_ms, 1e-9))
            conc = cfg.baseline + (conc - cfg.baseline) * decay_factor

            # Production rules -- implemented in subsequent tasks.
            for rule in cfg.production_rules:
                conc += self._compute_production(rule, cfg, bridge)

            # Clip to bounds.
            conc = max(cfg.concentration_min, min(cfg.concentration_max, conc))
            self._concentrations[cfg.name] = conc

    def _compute_production(self, rule: ProductionRule,
                             cfg: NeuromodulatorConfig, bridge) -> float:
        """Compute production contribution for one rule.

        Returns the additive concentration contribution for this step,
        BEFORE clipping (which step() applies after summing all rules).
        """
        rt = rule.rule_type
        if rt == "manual":
            return 0.0

        if rt == "from_reward":
            if bridge is None or not hasattr(bridge, "core_config"):
                return 0.0
            cc = bridge.core_config
            reward = float(getattr(cc, "current_reward_signal", 0.0))
            baseline = float(getattr(cc, "reward_baseline", 0.0))
            return rule.sensitivity * (reward - baseline)

        if rt == "from_error_persistence":
            # Implemented in Task 5
            return 0.0

        if rt == "from_novelty":
            # Reserved for future ACh
            return 0.0

        # Unknown rule type: silently no-op rather than crash. Future rules
        # are forward-compatible.
        return 0.0
