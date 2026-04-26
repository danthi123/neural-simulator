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
        "from_surprise"
            Tracks an EMA of recent reward over `window_ms`. Produces
            sensitivity * (|RPE| - threshold) when |RPE| > threshold, where
            RPE = current_reward - prior_ema. Phasic on unexpected outcomes,
            silent during expected ones. Models NE-like fast meta-modulation
            (Schultz 1997 reward-prediction-error encoding).
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

    # ----- Group registration (for scope="group:NAME" targets) -----

    def set_group_indices(self, group_dict: dict) -> None:
        """Register neuron groups so target scopes like 'group:motor' work.

        group_dict: {group_name: list_of_int_indices}.
        """
        self._group_indices = {
            str(name): [int(i) for i in indices]
            for name, indices in group_dict.items()
        }

    # ----- Effect computation -----

    def compute_synaptic_gain_multiplier(self) -> float:
        """Aggregate synaptic_gain effects across all modulators (scope=all).

        Per-trait / per-group / per-synapse scoping is not supported on this
        path -- callers needing those should compute per-synapse effects
        explicitly. Returns a non-negative scalar; clamped at 0 so transmission
        cannot be reversed by extreme negative concentrations.

        Effect formula per modulator: 1 + sensitivity * (conc - baseline).
        Multiplicative across modulators.
        """
        multiplier = 1.0
        for cfg in self._configs:
            for tgt in cfg.targets:
                if tgt.target_type != "synaptic_gain":
                    continue
                if tgt.scope != "all":
                    continue
                conc = self._concentrations[cfg.name]
                multiplier *= 1.0 + tgt.sensitivity * (conc - cfg.baseline)
        return float(max(0.0, multiplier))

    def compute_plasticity_rate_multiplier(self) -> float:
        """Aggregate plasticity_rate effects across all modulators (scope=all).

        Returns a non-negative scalar to multiply STDP amplitudes / reward
        learning rate. Same formula as synaptic_gain.
        """
        multiplier = 1.0
        for cfg in self._configs:
            for tgt in cfg.targets:
                if tgt.target_type != "plasticity_rate":
                    continue
                if tgt.scope != "all":
                    continue
                conc = self._concentrations[cfg.name]
                multiplier *= 1.0 + tgt.sensitivity * (conc - cfg.baseline)
        return float(max(0.0, multiplier))

    def compute_excitability_drive_pA(self) -> float:
        """Scalar additive drive (pA) from all scope=all excitability_drive targets.

        Sum across modulators, additive (not multiplicative).
        """
        drive = 0.0
        for cfg in self._configs:
            for tgt in cfg.targets:
                if tgt.target_type != "excitability_drive":
                    continue
                if tgt.scope != "all":
                    continue
                conc = self._concentrations[cfg.name]
                drive += tgt.sensitivity * (conc - cfg.baseline)
        return float(drive)

    def compute_excitability_drive_per_neuron(self, cp_traits=None):
        """Per-neuron additive drive array, honoring trait:N and group:NAME scopes.

        Returns:
            None if no per-neuron-scoped excitability_drive targets exist
            (caller can skip applying it).
            Otherwise, a cupy float32 array of shape (n_neurons,) summing
            contributions from all matching targets.
        """
        if self._cp is None:
            return None
        cp = self._cp

        drive = None  # Allocate lazily
        for cfg in self._configs:
            for tgt in cfg.targets:
                if tgt.target_type != "excitability_drive":
                    continue
                if tgt.scope == "all":
                    continue  # handled by compute_excitability_drive_pA
                conc = self._concentrations[cfg.name]
                value = float(tgt.sensitivity * (conc - cfg.baseline))
                if abs(value) < 1e-12:
                    continue

                if drive is None:
                    drive = cp.zeros(self._n_neurons, dtype=cp.float32)

                if tgt.scope.startswith("trait:") and cp_traits is not None:
                    try:
                        idx = int(tgt.scope.split(":", 1)[1])
                    except ValueError:
                        continue
                    drive = drive + cp.where(
                        cp_traits == idx,
                        cp.float32(value),
                        cp.float32(0.0),
                    )
                elif tgt.scope.startswith("group:"):
                    gname = tgt.scope.split(":", 1)[1]
                    indices = self._group_indices.get(gname)
                    if not indices:
                        continue
                    idx_arr = cp.asarray(indices, dtype=cp.int32)
                    mask = cp.zeros(self._n_neurons, dtype=cp.bool_)
                    mask[idx_arr] = True
                    drive = drive + cp.where(
                        mask,
                        cp.float32(value),
                        cp.float32(0.0),
                    )

        return drive

    # ----- Production rule helpers -----

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
            if bridge is None or not hasattr(bridge, "core_config"):
                return 0.0
            cc = bridge.core_config
            reward = float(getattr(cc, "current_reward_signal", 0.0))
            baseline = float(getattr(cc, "reward_baseline", 0.0))
            err = abs(reward - baseline)

            # Update EMA of |error| with time-constant matching window_ms.
            state = self._rule_state[cfg.name]
            ema_alpha = self.dt_ms / max(rule.window_ms, 1e-9)
            ema = state.get("err_ema", 0.0)
            ema = ema + ema_alpha * (err - ema)
            state["err_ema"] = ema

            # Produce iff sustained EMA exceeds threshold. Per-step
            # production scales with (ema - threshold) * sensitivity * dt/1000
            # so equilibrium concentration is balanced against decay.
            if ema > rule.threshold:
                return rule.sensitivity * (ema - rule.threshold) * (self.dt_ms / 1000.0)
            return 0.0

        if rt == "from_surprise":
            # NE-like phasic firing on unexpected reward (RPE).
            # Tracks EMA of recent reward; fires when |reward - ema| > threshold.
            if bridge is None or not hasattr(bridge, "core_config"):
                return 0.0
            cc = bridge.core_config
            reward = float(getattr(cc, "current_reward_signal", 0.0))

            state = self._rule_state[cfg.name]
            ema = state.get("reward_ema", 0.0)
            # window_ms is the EMA tau for the reward expectation
            decay = math.exp(-self.dt_ms / max(rule.window_ms, 1e-9))

            # Compute RPE BEFORE updating ema (= surprise relative to prior expectation)
            rpe = reward - ema
            surprise = abs(rpe)

            # Update EMA
            state["reward_ema"] = decay * ema + (1 - decay) * reward

            # Phasic production iff surprise exceeds threshold
            if surprise > rule.threshold:
                return rule.sensitivity * (surprise - rule.threshold)
            return 0.0

        if rt == "from_novelty":
            # Reserved for future ACh
            return 0.0

        # Unknown rule type: silently no-op rather than crash. Future rules
        # are forward-compatible.
        return 0.0
