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
        "plasticity_gate"
            Drives a per-pathway plasticity gate. The gate name must match
            a `RegionPathway.plasticity_gate` set during construction.
            effect = clip(0, 1, sensitivity * (conc - baseline))
            Use scope to identify the gate: "gate:<name>"
            Biological grounding: critical-period closure via PV
            interneuron maturation, DA-gated corticostriatal plasticity,
            ACh-gated cortical attention plasticity. The neuromodulator
            concentration becomes the actual plasticity gate value.
        "plasticity_window_gate"
            (Wired in Cluster B.3 Task 2.) Inverse of plasticity_gate:
            HIGH concentration BLOCKS plasticity, LOW concentration
            (pause) PERMITS it. Models BG TANs / cholinergic gating of
            corticostriatal LTP. For Task 1 this target is data only —
            the existing compute_* methods silently skip it (forward-
            compatibility pattern), but it must parse cleanly into the
            ACh default config so Task 2 can wire the bridge effect
            without re-shaping the data.

    scope: which neurons / synapses are affected.
        "all"               every neuron / synapse
        "trait:<idx>"       neurons whose cp_traits == idx
        "group:<name>"      neurons registered with the experiment engine
                            under that group name
        "plastic_only"      synapses with cp_synapse_plastic_mask == True
                            (synaptic_gain & plasticity_rate only)
        "gate:<name>"       plasticity-gate by name (plasticity_gate target only)

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
        "pause_on_reward"
            On each step, adds sensitivity * (|current_reward_signal| -
            threshold) when |reward| > threshold; otherwise emits 0.
            With negative sensitivity (e.g., -2.0), salient reward events
            drive concentration DOWN below baseline. Combined with a
            tonic baseline > 0 and natural decay back to baseline, this
            models the BG TAN "pause then recover" pattern: tonic ACh
            release suppresses corticostriatal plasticity, brief pauses
            on salient events open plasticity windows. Threshold acts
            as a salience floor — small fluctuations don't trigger pauses.
        "from_novelty"
            (Reserved for future ACh; emits 0 for now.)
        "from_region_firing"
            Reads mean firing rate across `source_regions` (using bridge's
            cp_firing_states + region_manager.indices) and produces
            `sensitivity * (rate_ema - threshold)` per step when above
            threshold, else 0. Uses `window_ms` as the EMA tau for the
            rate estimate. Models neuropeptide co-release: D1 MSNs firing
            cause dynorphin/SP release; D2 MSNs firing cause enkephalin
            release (PBR-160 ch 16 McGinty). Requires `bridge.region_manager`
            and `bridge.cp_firing_states` to be available; if either is
            missing, emits 0 (graceful no-op).

    sensitivity, threshold, window_ms: tunable per rule.
    source_regions: optional List[str] of region names; only used by
        rule_type="from_region_firing".
    """

    rule_type: str
    sensitivity: float = 1.0
    threshold: float = 0.5
    window_ms: float = 500.0
    source_regions: List[str] = field(default_factory=list)


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

    def compute_plasticity_window_gate_multiplier(self) -> float:
        """Aggregate plasticity_window_gate effects across all modulators (scope=all).

        Inverse of plasticity_rate / plasticity_gate: HIGH concentration BLOCKS
        plasticity, LOW concentration PERMITS it. Models BG TANs / cholinergic
        gating of corticostriatal LTP — tonic ACh release suppresses
        plasticity, brief pauses on salient events open transient plasticity
        windows.

        Effect formula per modulator: clip(1 - conc/baseline, 0, 1).
            ACh at baseline -> gate = 0 (plasticity blocked)
            ACh = 0 (full pause) -> gate = 1 (plasticity permitted)
            ACh > baseline (overshoot) -> gate clamped to 0

        Multiple modulators with this target combine multiplicatively, matching
        the precedent of compute_synaptic_gain_multiplier.

        Returns 1.0 when:
            - subsystem disabled / manager not initialized (caller default),
            - no modulator declares a plasticity_window_gate target,
            - the modulator's baseline is 0 (no tonic level to escape).
        i.e. the no-op default is "fully permitted" so existing flagship
        configurations are bit-identical when ACh is not registered.

        Only scope="all" is honored, matching the existing plasticity_rate
        target's restriction. Other scopes are silently skipped.
        """
        multiplier = 1.0
        any_target_found = False
        for cfg in self._configs:
            for tgt in cfg.targets:
                if tgt.target_type != "plasticity_window_gate":
                    continue
                if tgt.scope != "all":
                    continue
                any_target_found = True
                baseline = float(cfg.baseline)
                if baseline <= 0.0:
                    # No tonic level => no suppression to lift => permitted.
                    continue
                conc = self._concentrations[cfg.name]
                gate = 1.0 - (conc / baseline)
                # Clip to [0, 1] -- overshoot ACh blocks plasticity, full pause
                # permits it. Sensitivity intentionally unused: the gate is
                # purely a function of conc/baseline, matching the spec.
                gate = max(0.0, min(1.0, gate))
                multiplier *= gate
        if not any_target_found:
            return 1.0
        return float(multiplier)

    def compute_plasticity_gate_values(self) -> dict:
        """Compute plasticity gate values driven by neuromodulator concentrations.

        Returns a dict {gate_name: value} for each plasticity_gate target
        with scope='gate:<name>'. The bridge calls this each step (or
        periodically) to update its per-pathway plasticity gates.

        Effect formula: clip(0, 1, sensitivity * (conc - baseline))
        Multiple modulators targeting the same gate are summed before
        clipping. A gate driven by NM conc=1, baseline=0, sensitivity=1
        will have gain=1 (full plasticity) when conc=1; gain=0 (frozen)
        when conc=0.

        Biological grounding: developmental NMs (slow ramp), critical-period
        gating (fast on, slow off), DA-gated corticostriatal LTP (phasic),
        ACh-gated attention plasticity (transient).
        """
        gate_contributions: dict = {}
        for cfg in self._configs:
            for tgt in cfg.targets:
                if tgt.target_type != "plasticity_gate":
                    continue
                if not tgt.scope.startswith("gate:"):
                    continue
                gate_name = tgt.scope.split(":", 1)[1]
                conc = self._concentrations[cfg.name]
                contribution = tgt.sensitivity * (conc - cfg.baseline)
                gate_contributions[gate_name] = (
                    gate_contributions.get(gate_name, 0.0) + contribution
                )
        # Clip to [0, 1]
        return {name: float(max(0.0, min(1.0, val)))
                for name, val in gate_contributions.items()}

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

        if rt == "pause_on_reward":
            # ACh-style: drive concentration DOWN on |reward| > threshold.
            # With negative sensitivity, salient events suppress the modulator
            # below its tonic baseline; decay then pulls it back. Models BG
            # TANs pausing briefly on reward / novelty (Aosaki et al. 1994,
            # Morris et al. 2004).
            if bridge is None or not hasattr(bridge, "core_config"):
                return 0.0
            cc = bridge.core_config
            reward = float(getattr(cc, "current_reward_signal", 0.0))
            magnitude = abs(reward)
            if magnitude > rule.threshold:
                return rule.sensitivity * (magnitude - rule.threshold)
            return 0.0

        if rt == "from_novelty":
            # Reserved for future ACh
            return 0.0

        if rt == "from_region_firing":
            # R3.6 (2026-04-29): neuropeptide co-release driven by D1/D2 MSN
            # firing. Reads mean firing rate across `source_regions`, maintains
            # an EMA over `window_ms`, and produces sensitivity * (ema - threshold)
            # when ema > threshold. PBR-160 ch 16 McGinty: D1 -> dynorphin/SP,
            # D2 -> enkephalin co-release.
            if bridge is None or not rule.source_regions:
                return 0.0
            rm = getattr(bridge, "region_manager", None)
            firing = getattr(bridge, "cp_firing_states", None)
            if rm is None or firing is None or self._cp is None:
                return 0.0
            # Compute current mean firing fraction across source regions
            try:
                indices = []
                for region_name in rule.source_regions:
                    region_idx = rm.indices(region_name)
                    if region_idx is None or len(region_idx) == 0:
                        continue
                    indices.extend(list(region_idx))
                if not indices:
                    return 0.0
                idx_cp = self._cp.asarray(indices, dtype=self._cp.int32)
                # mean fraction firing this step (0.0 - 1.0)
                rate = float(self._cp.mean(firing[idx_cp].astype(self._cp.float32)).get())
            except Exception:
                return 0.0
            # EMA over window_ms (mirrors from_error_persistence pattern)
            state = self._rule_state[cfg.name]
            ema_alpha = self.dt_ms / max(rule.window_ms, 1e-9)
            ema = state.get("rate_ema", 0.0)
            ema = ema + ema_alpha * (rate - ema)
            state["rate_ema"] = ema
            if ema > rule.threshold:
                return rule.sensitivity * (ema - rule.threshold) * (self.dt_ms / 1000.0)
            return 0.0

        # Unknown rule type: silently no-op rather than crash. Future rules
        # are forward-compatible.
        return 0.0


# ----- Default config helpers -----


def _default_acetylcholine_config() -> NeuromodulatorConfig:
    """Default acetylcholine (ACh) neuromodulator config for BG TANs.

    Models tonically active cholinergic interneurons that pause briefly on
    salient events (reward, novelty), opening a transient corticostriatal
    plasticity window. See Cluster B.3 plan
    (`docs/plans/2026-04-28-cluster-b3-tans-implementation.md`).

    Defaults:
        baseline = 1.0           # tonic ACh release ("plasticity off")
        decay_tau_ms = 500       # ~half-second pause/recover time scale
        sensitivity = -2.0       # |reward| drives concentration DOWN
        threshold = 0.0          # any non-zero reward triggers pause

    Targets:
        plasticity_window_gate (scope=all)
            Wired in Task 2 of the Cluster B.3 plan; for Task 1 this is
            data only and the existing compute_* methods silently skip it.

    The runner is expected to enable the neuromodulator subsystem and
    register this config when `--enable-tans` is set (Task 3).
    """
    return NeuromodulatorConfig(
        name="acetylcholine",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            ModulatorTarget(target_type="plasticity_window_gate", scope="all"),
        ],
        production_rules=[
            ProductionRule(
                rule_type="pause_on_reward",
                sensitivity=-2.0,
                threshold=0.0,
            ),
        ],
    )


# ----- R3.6 (2026-04-29): D1/D2 neuropeptide co-release configs -----
# PBR-160 ch 16 (McGinty pp 273-280):
# - D1 MSNs co-release dynorphin + substance P with GABA. Dynorphin -> KOR
#   on Glu/DA terminals (suppresses release; homeostatic brake). Substance P
#   -> NK-1 on cholinergic interneurons (raises ACh). Net: D1 firing closes
#   a Glu/DA auto-regulatory loop AND drives ACh up.
# - D2 MSNs co-release enkephalin with GABA. Enkephalin -> DOR on cholinergic
#   interneurons (raises DA, lowers ACh). Net: D2 firing increases DA
#   and lowers ACh — opposite of D1.
# All three are opt-in (registered when --enable-bg-neuropeptides is set).
# Defaults assume the BG cascade exposes str_D1_{N,E,S,W} and
# str_D2_{N,E,S,W} regions (true under build_bg_brain_regions).


def _default_dynorphin_config() -> NeuromodulatorConfig:
    """Dynorphin: D1-driven, kappa-opioid suppressive of cortex->striatum
    plasticity. Sensitivity -0.4 modulates plasticity_rate downward when
    D1 firing rate EMA exceeds the threshold (homeostatic brake)."""
    return NeuromodulatorConfig(
        name="dynorphin",
        baseline=0.0,
        decay_tau_ms=2000.0,  # peptide neuromodulators decay slowly
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            ModulatorTarget(
                target_type="plasticity_rate", scope="all",
                sensitivity=-0.4,
            ),
        ],
        production_rules=[
            ProductionRule(
                rule_type="from_region_firing",
                sensitivity=2.0,
                threshold=0.05,  # ema firing-fraction threshold (~5%)
                window_ms=500.0,
                source_regions=[
                    "str_D1_N", "str_D1_E", "str_D1_S", "str_D1_W",
                ],
            ),
        ],
    )


def _default_substance_p_config() -> NeuromodulatorConfig:
    """Substance P: D1-driven, NK-1 receptor on TANs raises ACh. Sensitivity
    +0.5 boosts plasticity_window_gate's effective baseline (i.e., raises
    ACh further when D1 fires).

    Note: in our framework this is implemented via excitability_drive on
    the ACh-modulated population. With ACh modulator already wired
    (--enable-tans), we avoid double-modulation by routing SP -> ACh via
    excitability_drive(scope=all)."""
    return NeuromodulatorConfig(
        name="substance_p",
        baseline=0.0,
        decay_tau_ms=1500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            ModulatorTarget(
                target_type="excitability_drive", scope="all",
                sensitivity=20.0,  # pA depolarization at unit concentration
            ),
        ],
        production_rules=[
            ProductionRule(
                rule_type="from_region_firing",
                sensitivity=2.0,
                threshold=0.05,
                window_ms=500.0,
                source_regions=[
                    "str_D1_N", "str_D1_E", "str_D1_S", "str_D1_W",
                ],
            ),
        ],
    )


def _default_enkephalin_config() -> NeuromodulatorConfig:
    """Enkephalin: D2-driven, DOR receptor effects: raises DA and lowers
    ACh. Modeled as plasticity_rate boost (mirroring DA's effect on
    cortex->striatum LTP). Per McGinty Fig 5 (p 280), this is the
    indirect-pathway counterbalance to dynorphin's D1 brake."""
    return NeuromodulatorConfig(
        name="enkephalin",
        baseline=0.0,
        decay_tau_ms=2000.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            ModulatorTarget(
                target_type="plasticity_rate", scope="all",
                sensitivity=+0.3,  # opposite sign from dynorphin
            ),
        ],
        production_rules=[
            ProductionRule(
                rule_type="from_region_firing",
                sensitivity=2.0,
                threshold=0.05,
                window_ms=500.0,
                source_regions=[
                    "str_D2_N", "str_D2_E", "str_D2_S", "str_D2_W",
                ],
            ),
        ],
    )
