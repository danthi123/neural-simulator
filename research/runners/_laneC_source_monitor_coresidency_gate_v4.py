"""Adaptive-inhibition successor for co-resident episodic source monitoring.

V4 inherits the v2 circuit and replaces its fixed FS-to-rival-source GABA-A
weights with default-off Vogels-style inhibitory STDP. Seed 600 is wiring
smoke only. Formal calibration requires the exact tuple (601, 607); later
partitions remain mechanically locked.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    ACC_GATE,
    ACC_REGION,
    APFC_GATE,
    APFC_SOURCE,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_AFFERENT_GATE,
    SOURCE_LEARNING_GATE,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    _dominant_source,
    _source_margin,
    make_episode_patterns,
)
from research.runners._laneC_source_monitor_coresidency_gate_v2 import (
    MIN_ATTRIBUTION_FRACTION,
    MIN_SOURCE_MARGIN,
    SOURCE_COMPETITION_GATE,
    SOURCE_INTERNEURON,
    SourceMonitorConfigV2,
    SourceMonitorCoresidencyGateV2,
)
from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.regions import RegionPathway
from tools.lab import attributable_to


SMOKE_SEED = 600
CALIBRATION_SEEDS = (601, 607)
DEVELOPMENT_SEEDS = (613, 617, 619)
HELD_OUT_SEEDS = (631, 641, 643)
OPEN_PHASES = ("calibration",)

INHIBITORY_LEARNING_GATE = "source_competition_inhibitory_learning"
ISTDP_TAU_MS = 20.0
ISTDP_TARGET_RATE_PER_STEP = 0.02
ISTDP_ETA = 0.001
ISTDP_WEIGHT_MIN = 0.0
ISTDP_WEIGHT_MAX = 6.0
ISTDP_INITIAL_WEIGHT = 3.0
INHIBITORY_REHEARSAL_STEPS = 5000
SMOKE_PAIR_STEPS = 80


@dataclass(frozen=True)
class SourceMonitorConfigV4(SourceMonitorConfigV2):
    """Frozen v2 operating point plus local inhibitory plasticity."""

    inhibitory_stdp_tau_ms: float = ISTDP_TAU_MS
    inhibitory_stdp_target_rate_per_step: float = ISTDP_TARGET_RATE_PER_STEP
    inhibitory_stdp_eta: float = ISTDP_ETA
    inhibitory_stdp_w_min: float = ISTDP_WEIGHT_MIN
    inhibitory_stdp_w_max: float = ISTDP_WEIGHT_MAX
    inhibitory_rehearsal_steps: int = INHIBITORY_REHEARSAL_STEPS


def validate_individual_seed(seed: int, phase: str) -> int:
    """Validate one internal worker seed while keeping reserved phases closed."""

    seed = int(seed)
    if phase not in OPEN_PHASES:
        raise ValueError(
            f"phase {phase!r} is not open; development={DEVELOPMENT_SEEDS} and "
            f"held-out={HELD_OUT_SEEDS} remain locked"
        )
    if seed not in CALIBRATION_SEEDS:
        raise ValueError(
            f"seed {seed} is not a v4 calibration seed; allowed={CALIBRATION_SEEDS}"
        )
    return seed


def validate_phase_seeds(phase: str, seeds: Sequence[int]) -> tuple[int, ...]:
    """Require the complete ordered calibration partition."""

    checked = tuple(validate_individual_seed(seed, phase) for seed in seeds)
    if checked != CALIBRATION_SEEDS:
        raise ValueError(
            "v4 calibration requires the exact ordered seed tuple "
            f"{CALIBRATION_SEEDS}; received={checked}"
        )
    return checked


class SourceMonitorCoresidencyGateV4(SourceMonitorCoresidencyGateV2):
    """V2 source circuit with plastic FS-to-rival GABA-A inhibition."""

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigV4 | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigV4)
            else SourceMonitorConfigV4(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        self.bridge.set_plasticity_gate(INHIBITORY_LEARNING_GATE, 0.0)

    def _validate_config(self) -> None:
        super()._validate_config()
        c = self.config
        v2 = SourceMonitorConfigV2()
        for field_name in SourceMonitorConfigV2.__dataclass_fields__:
            expected = getattr(v2, field_name)
            if field_name == "interneuron_to_rival_weight":
                expected = ISTDP_INITIAL_WEIGHT
            if getattr(c, field_name) != expected:
                raise ValueError(
                    f"v4 freezes inherited v2 field {field_name!r} at {expected!r}"
                )
        frozen = {
            "inhibitory_stdp_tau_ms": ISTDP_TAU_MS,
            "inhibitory_stdp_target_rate_per_step": ISTDP_TARGET_RATE_PER_STEP,
            "inhibitory_stdp_eta": ISTDP_ETA,
            "inhibitory_stdp_w_min": ISTDP_WEIGHT_MIN,
            "inhibitory_stdp_w_max": ISTDP_WEIGHT_MAX,
            "inhibitory_rehearsal_steps": INHIBITORY_REHEARSAL_STEPS,
        }
        for field_name, expected in frozen.items():
            if getattr(c, field_name) != expected:
                raise ValueError(f"v4 freezes {field_name!r} at {expected!r}")

    def _build_bridge(self) -> SimulationBridge:
        c = self.config
        regions = [self._region(EPISODE_REGION, c.n_episode)]
        for source in SOURCES:
            regions.extend(
                [
                    self._region(SOURCE_AFFERENT[source], c.n_source_afferent),
                    self._region(SOURCE_MEMORY[source], c.n_source_memory),
                    self._fs_region(SOURCE_INTERNEURON[source], c.n_source_interneuron),
                    self._region(APFC_SOURCE[source], c.n_apfc),
                ]
            )
        regions.append(self._region(ACC_REGION, c.n_acc))

        pathways: list[RegionPathway] = []
        for source in SOURCES:
            pathways.extend(
                [
                    RegionPathway(
                        from_region=EPISODE_REGION,
                        to_region=SOURCE_MEMORY[source],
                        density=1.0,
                        weight_mean=0.0,
                        weight_jitter=0.0,
                        plastic=True,
                        plasticity_gate=SOURCE_LEARNING_GATE,
                        transmission_gate=SOURCE_RECALL_GATE,
                    ),
                    RegionPathway(
                        from_region=SOURCE_AFFERENT[source],
                        to_region=SOURCE_MEMORY[source],
                        density=1.0,
                        weight_mean=float(c.source_afferent_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_AFFERENT_GATE,
                    ),
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=SOURCE_INTERNEURON[source],
                        density=1.0,
                        weight_mean=float(c.source_to_interneuron_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_COMPETITION_GATE,
                    ),
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=APFC_SOURCE[source],
                        density=1.0,
                        weight_mean=float(c.source_to_apfc_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=APFC_GATE,
                    ),
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=ACC_REGION,
                        density=1.0,
                        weight_mean=float(c.source_to_acc_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=ACC_GATE,
                    ),
                ]
            )
            for rival in SOURCES:
                if rival == source:
                    continue
                pathways.append(
                    RegionPathway(
                        from_region=SOURCE_INTERNEURON[source],
                        to_region=SOURCE_MEMORY[rival],
                        density=1.0,
                        weight_mean=float(c.interneuron_to_rival_weight),
                        weight_jitter=0.0,
                        plastic=True,
                        plasticity_gate=INHIBITORY_LEARNING_GATE,
                        transmission_gate=SOURCE_COMPETITION_GATE,
                        receptor="gaba_a",
                    )
                )

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = regions
        cfg.region_pathways = pathways
        cfg.seed = self.seed
        cfg.dt_ms = 0.5
        cfg.enable_parameter_heterogeneity = False
        cfg.enable_stdp = False
        cfg.enable_inhibitory_stdp = True
        cfg.inhibitory_stdp_tau_ms = float(c.inhibitory_stdp_tau_ms)
        cfg.inhibitory_stdp_target_rate_per_step = float(
            c.inhibitory_stdp_target_rate_per_step
        )
        cfg.inhibitory_stdp_eta = float(c.inhibitory_stdp_eta)
        cfg.inhibitory_stdp_w_min = float(c.inhibitory_stdp_w_min)
        cfg.inhibitory_stdp_w_max = float(c.inhibitory_stdp_w_max)
        cfg.enable_hebbian_learning = True
        cfg.hebbian_symmetric = True
        cfg.hebbian_learning_rate = float(c.hebbian_learning_rate)
        cfg.hebbian_max_weight = float(c.hebbian_max_weight)
        cfg.hebbian_min_weight = 0.0
        cfg.hebbian_weight_decay = 0.0
        cfg.enable_reward_modulation = False
        cfg.enable_structural_plasticity = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_homeostasis = False
        cfg.ou_std_current_pA = 0.0
        cfg.fast_spike_reset = True
        bridge = SimulationBridge(
            core_config=cfg,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=GPUConfig(),
        )
        bridge._initialize_simulation_data(called_from_playback_init=False)
        return bridge

    def inhibitory_synapse_indices(self) -> np.ndarray:
        return np.asarray(
            self.bridge._plasticity_gate_to_synapses[INHIBITORY_LEARNING_GATE],
            dtype=np.int64,
        )

    def inhibitory_weight_vector(self) -> np.ndarray:
        weights = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        return weights[self.inhibitory_synapse_indices()].copy()

    def excitatory_weight_vector(self) -> np.ndarray:
        weights = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        mask = np.ones(weights.size, dtype=bool)
        mask[self.inhibitory_synapse_indices()] = False
        return weights[mask].copy()

    def threshold_vector(self) -> np.ndarray:
        return np.asarray(
            to_host(self.bridge.cp_neuron_firing_thresholds), dtype=np.float64
        ).copy()

    def inhibitory_route_means(self) -> dict[str, dict[str, float]]:
        coo = self.bridge.cp_connections.tocoo(copy=False)
        rows = np.asarray(to_host(coo.row), dtype=np.int64)
        cols = np.asarray(to_host(coo.col), dtype=np.int64)
        weights = np.asarray(to_host(coo.data), dtype=np.float64)
        eligible = np.zeros(weights.size, dtype=bool)
        eligible[self.inhibitory_synapse_indices()] = True
        means: dict[str, dict[str, float]] = {}
        for source in SOURCES:
            means[source] = {}
            source_rows = np.isin(rows, self._competition_indices[source])
            for rival in SOURCES:
                if rival == source:
                    continue
                route = eligible & source_rows & np.isin(
                    cols, self._source_memory_indices[rival]
                )
                means[source][rival] = float(weights[route].mean())
        return means

    def rehearse_inhibitory_competition(
        self,
        episode_patterns: Sequence[Sequence[int]],
        *,
        learning_enabled: bool,
    ) -> dict:
        """Run balanced episode rehearsal with only inhibitory plasticity open."""

        patterns = tuple(episode_patterns)
        if len(patterns) != 4:
            raise ValueError("v4 rehearsal requires three single-source and one mixed episode")
        c = self.config
        block_steps = int(c.training_steps) + int(c.rest_steps)
        cycles = int(
            math.ceil(c.inhibitory_rehearsal_steps / (len(patterns) * block_steps))
        )
        before_i = self.inhibitory_weight_vector()
        before_e = self.excitatory_weight_vector()
        before_t = self.threshold_vector()
        prior_hebbian = bool(self.bridge.core_config.enable_hebbian_learning)
        self.bridge.core_config.enable_hebbian_learning = False
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        try:
            for _ in range(cycles):
                for pattern in patterns:
                    self.bridge.set_plasticity_gate(
                        INHIBITORY_LEARNING_GATE, 1.0 if learning_enabled else 0.0
                    )
                    self._drive(self._episode_global_indices(pattern))
                    for _ in range(int(c.training_steps)):
                        self.bridge._run_one_simulation_step()
                    self.bridge.set_plasticity_gate(INHIBITORY_LEARNING_GATE, 0.0)
                    self._rest()
        finally:
            self.bridge.set_plasticity_gate(INHIBITORY_LEARNING_GATE, 0.0)
            self.bridge.core_config.enable_hebbian_learning = prior_hebbian
            self.bridge.cp_external_input_current[:] = 0.0
        after_i = self.inhibitory_weight_vector()
        after_e = self.excitatory_weight_vector()
        after_t = self.threshold_vector()
        return {
            "learning_enabled": bool(learning_enabled),
            "cycles": cycles,
            "executed_steps": cycles * len(patterns) * block_steps,
            "inhibitory_weight_delta_l1": float(np.abs(after_i - before_i).sum()),
            "excitatory_weights_unchanged": bool(np.array_equal(before_e, after_e)),
            "thresholds_unchanged": bool(np.array_equal(before_t, after_t)),
        }


def _train_standard_sources(
    gate: SourceMonitorCoresidencyGateV4,
    patterns: Sequence[Sequence[int]],
) -> None:
    gate.experience(patterns[0], visual_activity=True)
    gate.experience(patterns[1], auditory_activity=True)
    gate.experience(patterns[2], corollary_discharge=True)
    gate.experience(patterns[3], visual_activity=True, auditory_activity=True)


def _recall_standard(
    gate: SourceMonitorCoresidencyGateV4,
    patterns: Sequence[Sequence[int]],
) -> dict[str, dict]:
    return {
        "seen": gate.recall(patterns[0]),
        "heard": gate.recall(patterns[1]),
        "self_generated": gate.recall(patterns[2]),
    }


def _rival_spike_burden(records: Mapping[str, Mapping]) -> float:
    return float(
        sum(
            sum(
                count
                for source, count in records[target]["source_spikes"].items()
                if source != target
            )
            for target in SOURCES
        )
    )


def _all_finite(value) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(np.isfinite(value))
    return True


def adaptive_inhibition_assessment(
    intact_records: Mapping[str, Mapping],
    learning_lesion_records: Mapping[str, Mapping],
    expression_lesion_records: Mapping[str, Mapping],
) -> dict:
    margins = {
        source: _source_margin(intact_records[source], source) for source in SOURCES
    }
    lesion_margins = {
        source: _source_margin(learning_lesion_records[source], source)
        for source in SOURCES
    }
    losses = {
        source: max(0.0, lesion_margins[source] - margins[source]) for source in SOURCES
    }
    surplus = {
        source: max(0.0, lesion_margins[source] - MIN_SOURCE_MARGIN)
        for source in SOURCES
    }
    spike_quantum = 1.0 / (SourceMonitorConfigV4().read_steps * SourceMonitorConfigV4().n_source_memory)
    intact_burden = _rival_spike_burden(intact_records)
    learning_lesion_burden = _rival_spike_burden(learning_lesion_records)
    expression_lesion_burden = _rival_spike_burden(expression_lesion_records)
    return {
        "components": {
            "all_source_margins_meet_fixed_floor": bool(
                min(margins.values()) >= MIN_SOURCE_MARGIN
            ),
            "weakest_margin_beats_learning_lesion_by_one_spike_quantum": bool(
                min(margins.values()) >= min(lesion_margins.values()) + spike_quantum
            ),
            "bounded_inhibitory_tradeoffs_protect_floor": bool(
                all(losses[source] <= surplus[source] for source in SOURCES)
            ),
            "adaptive_inhibition_reduces_rival_spike_burden": bool(
                intact_burden < learning_lesion_burden
                and intact_burden < expression_lesion_burden
            ),
        },
        "margins": margins,
        "learning_lesion_margins": lesion_margins,
        "losses": losses,
        "surplus": surplus,
        "spike_quantum": spike_quantum,
        "rival_spike_burden": {
            "intact": intact_burden,
            "learning_lesion": learning_lesion_burden,
            "expression_lesion": expression_lesion_burden,
        },
    }


def evaluate_calibration_seed(seed: int) -> dict:
    """Run one internal v4 calibration worker and all preregistered arms."""

    seed = validate_individual_seed(seed, "calibration")
    c = SourceMonitorConfigV4()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()
    intact = SourceMonitorCoresidencyGateV4(seed=seed, config=c)
    learning_lesion = SourceMonitorCoresidencyGateV4(seed=seed, config=c)
    initial = intact.weight_summary()
    initial_i = intact.inhibitory_weight_vector()
    initial_t = intact.threshold_vector()
    initial_arms_match = bool(
        np.array_equal(initial_i, learning_lesion.inhibitory_weight_vector())
        and np.array_equal(
            intact.excitatory_weight_vector(),
            learning_lesion.excitatory_weight_vector(),
        )
        and np.array_equal(initial_t, learning_lesion.threshold_vector())
    )
    _train_standard_sources(intact, patterns)
    _train_standard_sources(learning_lesion, patterns)
    learned = intact.weight_summary()
    learned_weights_match = np.array_equal(
        intact.excitatory_weight_vector(), learning_lesion.excitatory_weight_vector()
    )
    intact_rehearsal = intact.rehearse_inhibitory_competition(
        patterns[:4], learning_enabled=True
    )
    lesion_rehearsal = learning_lesion.rehearse_inhibitory_competition(
        patterns[:4], learning_enabled=False
    )

    intact_records = _recall_standard(intact, patterns)
    learning_lesion_records = _recall_standard(learning_lesion, patterns)
    mixed = intact.recall(patterns[3])
    unseen = intact.recall(patterns[4])
    source_lesion = intact.recall(patterns[0], source_path_lesion=True)
    acc_lesion = intact.recall(patterns[0], acc_lesion=True)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        expression_lesion_records = _recall_standard(intact, patterns)
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    swapped = SourceMonitorCoresidencyGateV4(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swapped.rehearse_inhibitory_competition(
        (patterns[0], patterns[1], patterns[0], patterns[1]),
        learning_enabled=True,
    )
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV4(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(
        patterns[0], visual_activity=True, learning_enabled=False
    )
    off_after = learning_off.weight_summary()
    off_recall = learning_off.recall(patterns[0])

    assessment = adaptive_inhibition_assessment(
        intact_records, learning_lesion_records, expression_lesion_records
    )
    seen = intact_records["seen"]
    seen_total = float(sum(seen["source_spikes"].values()))
    source_lesion_total = float(sum(source_lesion["source_spikes"].values()))
    source_fraction = attributable_to(
        "source recall pathway", seen_total, source_lesion_total
    )
    acc_fraction = attributable_to(
        "source-to-ACC pathway", seen["acc_spikes"], acc_lesion["acc_spikes"]
    )
    final_i = intact.inhibitory_weight_vector()
    lesion_i = learning_lesion.inhibitory_weight_vector()
    final_t = intact.threshold_vector()
    expression_seen = expression_lesion_records["seen"]
    competition_intact_spikes = float(sum(seen["competition_spikes"].values()))
    competition_expression_lesion_spikes = float(
        sum(expression_seen["competition_spikes"].values())
    )
    recall_parameters = list(inspect.signature(intact.recall).parameters)
    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    expected_regions.update(SOURCE_INTERNEURON.values())
    components = {
        "learned_routes_start_zero": bool(initial["l1"] == 0.0),
        "experience_changes_synaptic_weights": bool(learned["l1"] > initial["l1"]),
        **assessment["components"],
        "seen_source_recalled": bool(_dominant_source(seen) == "seen"),
        "heard_source_recalled": bool(
            _dominant_source(intact_records["heard"]) == "heard"
        ),
        "self_source_recalled": bool(
            _dominant_source(intact_records["self_generated"]) == "self_generated"
        ),
        "source_swap_follows_afferent_activity": bool(
            _dominant_source(swap_zero) == "heard"
            and _dominant_source(swap_one) == "seen"
        ),
        "mixed_source_reinstates_both": bool(
            mixed["source_spikes"]["seen"] > 0.0
            and mixed["source_spikes"]["heard"] > 0.0
        ),
        "source_path_attribution_meets_fixed_floor": bool(
            source_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "source_path_lesion_collapses_recall": bool(
            source_lesion_total <= 0.10 * max(seen_total, 1.0)
        ),
        "acc_path_attribution_meets_fixed_floor": bool(
            acc_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "acc_lesion_preserves_source_and_silences_acc": bool(
            sum(acc_lesion["source_spikes"].values()) >= 0.90 * seen_total
            and acc_lesion["acc_spikes"] == 0.0
        ),
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0 and seen["acc_spikes"] > 0.0
        ),
        "competition_circuit_is_active_and_lesionable": bool(
            competition_intact_spikes > 0.0
            and competition_expression_lesion_spikes == 0.0
        ),
        "learning_off_keeps_weights_zero": bool(
            off_initial["l1"] == 0.0 and off_after["l1"] == 0.0
        ),
        "learning_off_has_no_source_recall": bool(
            sum(off_recall["source_spikes"].values()) == 0.0
        ),
        "unseen_episode_has_no_source_recall": bool(
            sum(unseen["source_spikes"].values()) == 0.0
        ),
        "inhibitory_weights_change_only_in_intact": bool(
            not np.array_equal(initial_i, final_i)
            and np.array_equal(initial_i, lesion_i)
        ),
        "episode_weights_match_before_inhibitory_rehearsal": bool(
            learned_weights_match
        ),
        "rehearsal_preserves_excitatory_weights_and_thresholds": bool(
            intact_rehearsal["excitatory_weights_unchanged"]
            and lesion_rehearsal["excitatory_weights_unchanged"]
            and intact_rehearsal["thresholds_unchanged"]
            and lesion_rehearsal["thresholds_unchanged"]
            and np.array_equal(initial_t, final_t)
        ),
    }
    preconditions = {
        "matched_arms_begin_identical": initial_arms_match,
        "expected_regions_share_one_bridge": expected_regions.issubset(region_names),
        "recall_accepts_no_source_metadata": bool(
            recall_parameters
            == ["self", "episode_pattern", "source_path_lesion", "acc_lesion"]
        ),
        "episode_patterns_are_disjoint_and_fit_population": bool(
            len(set().union(*(set(pattern.tolist()) for pattern in patterns)))
            == len(patterns) * c.episode_pattern_size
        ),
        "inhibitory_pathway_gate_reaches_synapses": bool(initial_i.size > 0),
        "learning_lesion_keeps_inhibitory_weights_fixed": bool(
            np.array_equal(initial_i, lesion_i)
        ),
        "inhibitory_rehearsal_keeps_noninhibitory_state_fixed": bool(
            intact_rehearsal["excitatory_weights_unchanged"]
            and lesion_rehearsal["excitatory_weights_unchanged"]
            and intact_rehearsal["thresholds_unchanged"]
            and lesion_rehearsal["thresholds_unchanged"]
        ),
        "rehearsal_budget_reached": bool(
            intact_rehearsal["executed_steps"] >= INHIBITORY_REHEARSAL_STEPS
            and lesion_rehearsal["executed_steps"] >= INHIBITORY_REHEARSAL_STEPS
        ),
        "all_scored_values_are_finite": bool(
            _all_finite(assessment)
            and _all_finite(intact_records)
            and _all_finite(learning_lesion_records)
            and _all_finite(expression_lesion_records)
        ),
    }
    status = (
        "UNDEFINED"
        if not all(preconditions.values())
        else "CALIBRATION_PASS" if all(components.values()) else "CALIBRATION_FAIL"
    )
    return {
        "seed": seed,
        "phase": "calibration",
        "status": status,
        "preconditions": preconditions,
        "components": components,
        "assessment": assessment,
        "rehearsal": {"intact": intact_rehearsal, "learning_lesion": lesion_rehearsal},
        "records": {
            "intact": intact_records,
            "learning_lesion": learning_lesion_records,
            "expression_lesion": expression_lesion_records,
            "mixed": mixed,
            "unseen": unseen,
            "source_path_lesion": source_lesion,
            "acc_lesion": acc_lesion,
            "learning_off": off_recall,
        },
        "weights": {
            "initial_inhibitory": initial_i.tolist(),
            "final_inhibitory": final_i.tolist(),
            "learning_lesion_inhibitory": lesion_i.tolist(),
        },
        "config": asdict(c),
        "interface_guards": {
            "recall_parameters": recall_parameters,
            "no_source_argument_at_inference": "source" not in recall_parameters,
            "host_gain_normalization": False,
            "host_response_decision": False,
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def _apply_smoke_pair(
    gate: SourceMonitorCoresidencyGateV4,
    *,
    inhibitory_source: str,
    coactive_rival: str,
    learning_enabled: bool,
) -> dict:
    """Apply a deterministic synthetic-spike wiring diagnostic, not evidence."""

    before_i = gate.inhibitory_route_means()
    before_e = gate.excitatory_weight_vector()
    before_t = gate.threshold_vector()
    fired = np.zeros(gate.bridge.cp_firing_states.shape[0], dtype=bool)
    fired[gate._competition_indices[inhibitory_source]] = True
    fired[gate._source_memory_indices[coactive_rival]] = True
    xp, _ = get_backend()
    gate.bridge.set_plasticity_gate(
        INHIBITORY_LEARNING_GATE, 1.0 if learning_enabled else 0.0
    )
    try:
        for _ in range(SMOKE_PAIR_STEPS):
            gate.bridge._apply_inhibitory_stdp(
                xp.asarray(fired), plasticity_gated=True
            )
    finally:
        gate.bridge.set_plasticity_gate(INHIBITORY_LEARNING_GATE, 0.0)
    after_i = gate.inhibitory_route_means()
    after_e = gate.excitatory_weight_vector()
    after_t = gate.threshold_vector()
    silent_rival = next(
        source
        for source in SOURCES
        if source not in (inhibitory_source, coactive_rival)
    )
    return {
        "inhibitory_source": inhibitory_source,
        "coactive_rival": coactive_rival,
        "silent_rival": silent_rival,
        "learning_enabled": bool(learning_enabled),
        "coactive_delta": (
            after_i[inhibitory_source][coactive_rival]
            - before_i[inhibitory_source][coactive_rival]
        ),
        "silent_delta": (
            after_i[inhibitory_source][silent_rival]
            - before_i[inhibitory_source][silent_rival]
        ),
        "inhibitory_weights_before": before_i,
        "inhibitory_weights_after": after_i,
        "excitatory_weights_unchanged": bool(np.array_equal(before_e, after_e)),
        "thresholds_unchanged": bool(np.array_equal(before_t, after_t)),
    }


def run_smoke(seed: int = SMOKE_SEED) -> dict:
    """Run only the preregistered non-scientific activity-routing smoke."""

    if int(seed) != SMOKE_SEED:
        raise ValueError(f"smoke seed must be exactly {SMOKE_SEED}")
    heard = _apply_smoke_pair(
        SourceMonitorCoresidencyGateV4(seed=seed),
        inhibitory_source="seen",
        coactive_rival="heard",
        learning_enabled=True,
    )
    self_generated = _apply_smoke_pair(
        SourceMonitorCoresidencyGateV4(seed=seed),
        inhibitory_source="seen",
        coactive_rival="self_generated",
        learning_enabled=True,
    )
    lesion = _apply_smoke_pair(
        SourceMonitorCoresidencyGateV4(seed=seed),
        inhibitory_source="seen",
        coactive_rival="heard",
        learning_enabled=False,
    )
    checks = {
        "coactive_rival_changes_more_than_silent_rival": bool(
            heard["coactive_delta"] > heard["silent_delta"]
        ),
        "activity_swap_moves_larger_change_to_new_rival": bool(
            heard["coactive_delta"] > heard["silent_delta"]
            and self_generated["coactive_delta"] > self_generated["silent_delta"]
        ),
        "learning_lesion_is_bit_identical": bool(
            lesion["inhibitory_weights_before"] == lesion["inhibitory_weights_after"]
        ),
        "excitatory_weights_unchanged": bool(
            heard["excitatory_weights_unchanged"]
            and self_generated["excitatory_weights_unchanged"]
            and lesion["excitatory_weights_unchanged"]
        ),
        "thresholds_unchanged": bool(
            heard["thresholds_unchanged"]
            and self_generated["thresholds_unchanged"]
            and lesion["thresholds_unchanged"]
        ),
    }
    return {
        "seed": int(seed),
        "phase": "smoke",
        "status": "SMOKE_PASS" if all(checks.values()) else "SMOKE_FAIL",
        "scientific_verdict": None,
        "checks": checks,
        "records": {
            "heard_coactive": heard,
            "self_generated_coactive": self_generated,
            "learning_lesion": lesion,
        },
        "warning": (
            "Synthetic population spikes test rule scope and activity routing only. "
            "This smoke result is not scientific evidence and cannot open development."
        ),
    }


def run_calibration(seeds: Sequence[int]) -> dict:
    checked = validate_phase_seeds("calibration", seeds)
    rows = [evaluate_calibration_seed(seed) for seed in checked]
    if any(row["status"] == "UNDEFINED" for row in rows):
        status = "UNDEFINED"
    elif all(row["status"] == "CALIBRATION_PASS" for row in rows):
        status = "CALIBRATION_PASS"
    else:
        status = "CALIBRATION_FAIL"
    return {"phase": "calibration", "seeds": list(checked), "status": status, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source-monitor v4 smoke or calibration.")
    parser.add_argument("--phase", choices=("smoke", "calibration"), default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.phase == "smoke":
        if args.seeds not in (None, [SMOKE_SEED]):
            raise ValueError(f"smoke requires seed {SMOKE_SEED}")
        result = run_smoke()
    else:
        result = run_calibration(args.seeds or CALIBRATION_SEEDS)
    print(f"[source-monitor-coresidency-v4] {result['status']}", flush=True)
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[source-monitor-coresidency-v4] wrote {out_path}", flush=True)
    return 0 if result["status"] in ("SMOKE_PASS", "CALIBRATION_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
