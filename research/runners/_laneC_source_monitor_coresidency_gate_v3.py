"""Calibration-only v3 gate for co-resident episodic source monitoring.

V3 preserves v2's learned source routes and local fast-spiking competition,
then gives only the source-memory populations cell-autonomous intrinsic
excitability homeostasis.  A matched arm receives identical training and
balanced rehearsal while its local threshold updates are frozen.  The fixed
acceptance rule protects the inherited absolute margin floor while asking the
stabilizer to improve the weakest source representation.

Seed 220 is reserved for non-scientific construction and dynamics smoke tests.
Only formal calibration seeds 232 and 233 are open. Development and held-out
seeds are named but mechanically rejected. The runner never receives source
metadata during recall and does not choose language or behavior.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    ACC_GATE,
    ACC_REGION,
    APFC_SOURCE,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_LEARNING_GATE,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    SourceMonitorConfig,
    _dominant_source,
    _source_margin,
    make_episode_patterns,
)
from research.runners._laneC_source_monitor_coresidency_gate_v2 import (
    APFC_GATE,
    MIN_ATTRIBUTION_FRACTION,
    MIN_SOURCE_MARGIN,
    SOURCE_AFFERENT_GATE,
    SOURCE_COMPETITION_GATE,
    SOURCE_INTERNEURON,
    SourceMonitorConfigV2,
    SourceMonitorCoresidencyGateV2,
)
from sim.backend import to_host
from sim.regions import BrainRegion
from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict


SMOKE_SEED = 220
CALIBRATION_SEEDS = (232, 233)
DEVELOPMENT_SEEDS = (234, 235, 330)
HELD_OUT_SEEDS = (331, 332, 333)
OPEN_PHASES = ("calibration",)

HOMEOSTASIS_TARGET_RATE = 0.02
HOMEOSTASIS_EMA_ALPHA = 0.0002
HOMEOSTASIS_ADAPT_RATE = 0.0005
HOMEOSTASIS_THRESHOLD_MIN = -55.0
HOMEOSTASIS_THRESHOLD_MAX = -30.0
HOMEOSTASIS_SETTLING_STEPS = 5000


@dataclass(frozen=True)
class SourceMonitorConfigV3(SourceMonitorConfigV2):
    """Frozen v2 operating point plus canonical local homeostasis."""

    source_homeostasis_target_rate: float = HOMEOSTASIS_TARGET_RATE
    source_homeostasis_ema_alpha: float = HOMEOSTASIS_EMA_ALPHA
    source_homeostasis_adapt_rate: float = HOMEOSTASIS_ADAPT_RATE
    source_homeostasis_threshold_min: float = HOMEOSTASIS_THRESHOLD_MIN
    source_homeostasis_threshold_max: float = HOMEOSTASIS_THRESHOLD_MAX
    source_homeostasis_settling_steps: int = HOMEOSTASIS_SETTLING_STEPS


def bounded_tradeoff_assessment(
    margins: Mapping[str, float],
    lesion_margins: Mapping[str, float],
    *,
    floor: float = MIN_SOURCE_MARGIN,
) -> dict:
    """Apply the preregistered floor-derived competition tradeoff rule."""

    if set(margins) != set(SOURCES) or set(lesion_margins) != set(SOURCES):
        raise ValueError(f"margins must contain exactly {SOURCES}")
    intact = {source: float(margins[source]) for source in SOURCES}
    lesion = {source: float(lesion_margins[source]) for source in SOURCES}
    losses = {
        source: max(0.0, lesion[source] - intact[source]) for source in SOURCES
    }
    spendable_surplus = {
        source: max(0.0, lesion[source] - float(floor)) for source in SOURCES
    }
    return {
        "components": {
            "all_source_margins_meet_fixed_floor": bool(
                min(intact.values()) >= float(floor)
            ),
            "bounded_homeostasis_tradeoffs_protect_floor": bool(
                all(
                    losses[source] <= spendable_surplus[source]
                    for source in SOURCES
                )
            ),
            "homeostasis_strictly_improves_weakest_source_margin": bool(
                min(intact.values()) > min(lesion.values())
            ),
        },
        "losses": losses,
        "spendable_surplus": spendable_surplus,
        "margin_gains": {
            source: intact[source] - lesion[source] for source in SOURCES
        },
        "minimum_intact_margin": min(intact.values()),
        "minimum_lesion_margin": min(lesion.values()),
    }


class SourceMonitorCoresidencyGateV3(SourceMonitorCoresidencyGateV2):
    """V2 circuit with local source-memory threshold homeostasis."""

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigV3 | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigV3)
            else SourceMonitorConfigV3(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        if self.bridge.cp_homeostasis_neuron_mask is None:
            raise RuntimeError("source-memory homeostasis mask was not initialized")
        self._source_homeostasis_mask = (
            self.bridge.cp_homeostasis_neuron_mask.copy()
        )
        self.set_local_homeostasis_updates(False)

    def _validate_config(self) -> None:
        super()._validate_config()
        c = self.config
        v2 = SourceMonitorConfigV2()
        for field_name in SourceMonitorConfigV2.__dataclass_fields__:
            if getattr(c, field_name) != getattr(v2, field_name):
                raise ValueError(
                    f"v3 freezes inherited v2 field {field_name!r} at "
                    f"{getattr(v2, field_name)!r}"
                )
        frozen = {
            "source_homeostasis_target_rate": HOMEOSTASIS_TARGET_RATE,
            "source_homeostasis_ema_alpha": HOMEOSTASIS_EMA_ALPHA,
            "source_homeostasis_adapt_rate": HOMEOSTASIS_ADAPT_RATE,
            "source_homeostasis_threshold_min": HOMEOSTASIS_THRESHOLD_MIN,
            "source_homeostasis_threshold_max": HOMEOSTASIS_THRESHOLD_MAX,
            "source_homeostasis_settling_steps": HOMEOSTASIS_SETTLING_STEPS,
        }
        for field_name, expected in frozen.items():
            if getattr(c, field_name) != expected:
                raise ValueError(
                    f"v3 freezes {field_name!r} at {expected!r}"
                )

    @staticmethod
    def _region(name: str, n_neurons: int) -> BrainRegion:
        region = SourceMonitorCoresidencyGateV2._region(name, n_neurons)
        if name in SOURCE_MEMORY.values():
            return replace(region, enable_homeostasis=True)
        return region

    def _build_bridge(self):
        bridge = super()._build_bridge()
        c = self.config
        cfg = bridge.core_config
        cfg.homeostasis_target_rate = float(c.source_homeostasis_target_rate)
        cfg.homeostasis_ema_alpha = float(c.source_homeostasis_ema_alpha)
        cfg.homeostasis_threshold_adapt_rate = float(
            c.source_homeostasis_adapt_rate
        )
        cfg.homeostasis_threshold_min = float(
            c.source_homeostasis_threshold_min
        )
        cfg.homeostasis_threshold_max = float(
            c.source_homeostasis_threshold_max
        )
        return bridge

    def expected_homeostasis_mask(self) -> np.ndarray:
        expected = np.zeros(self.bridge.cp_firing_states.shape[0], dtype=bool)
        for source in SOURCES:
            expected[self._source_memory_indices[source]] = True
        return expected

    def homeostasis_mask(self) -> np.ndarray:
        return np.asarray(
            to_host(self._source_homeostasis_mask), dtype=bool
        )

    def set_local_homeostasis_updates(self, enabled: bool) -> None:
        mask = self._source_homeostasis_mask.copy()
        if not enabled:
            mask[:] = False
        self.bridge.cp_homeostasis_update_neuron_mask = mask

    def source_threshold_vector(self) -> np.ndarray:
        thresholds = np.asarray(
            to_host(self.bridge.cp_neuron_firing_thresholds), dtype=np.float64
        )
        return np.concatenate(
            [thresholds[self._source_memory_indices[source]] for source in SOURCES]
        )

    def learned_weight_vector(self) -> np.ndarray:
        weights = np.asarray(
            to_host(self.bridge.cp_connections.data), dtype=np.float64
        )
        return weights[self._learned_synapse_indices()].copy()

    def settle_homeostasis(
        self,
        episode_patterns: Sequence[Sequence[int]],
        *,
        updates_enabled: bool,
    ) -> dict:
        """Run one EMA time constant of balanced, learning-off rehearsal."""

        patterns = tuple(episode_patterns)
        if not patterns:
            raise ValueError("homeostatic settling requires an episode pattern")
        c = self.config
        block_steps = int(c.training_steps) + int(c.rest_steps)
        cycles = int(
            math.ceil(
                int(c.source_homeostasis_settling_steps)
                / float(len(patterns) * block_steps)
            )
        )
        before_thresholds = self.source_threshold_vector()
        before_weights = self.learned_weight_vector()
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        self.set_local_homeostasis_updates(updates_enabled)
        try:
            for _ in range(cycles):
                for pattern in patterns:
                    episode_global = self._episode_global_indices(pattern)
                    self._drive(episode_global)
                    for _ in range(int(c.training_steps)):
                        self.bridge._run_one_simulation_step()
                    self._rest()
        finally:
            self.set_local_homeostasis_updates(False)
            self.bridge.cp_external_input_current[:] = 0.0
        after_thresholds = self.source_threshold_vector()
        after_weights = self.learned_weight_vector()
        threshold_delta = after_thresholds - before_thresholds
        return {
            "updates_enabled": bool(updates_enabled),
            "patterns_per_cycle": len(patterns),
            "cycles": cycles,
            "executed_steps": cycles * len(patterns) * block_steps,
            "threshold_before_mean": float(before_thresholds.mean()),
            "threshold_after_mean": float(after_thresholds.mean()),
            "threshold_delta_l1": float(np.abs(threshold_delta).sum()),
            "threshold_delta_max_abs": float(np.abs(threshold_delta).max()),
            "learned_weights_unchanged": bool(
                np.array_equal(before_weights, after_weights)
            ),
        }


def validate_phase_seed(seed: int, phase: str) -> int:
    """Mechanically keep development and held-out v3 seeds closed."""

    seed = int(seed)
    if phase not in OPEN_PHASES:
        raise ValueError(
            f"phase {phase!r} is not open; development seeds "
            f"{DEVELOPMENT_SEEDS} and held-out seeds {HELD_OUT_SEEDS} remain locked"
        )
    if seed not in CALIBRATION_SEEDS:
        raise ValueError(
            f"seed {seed} is not a v3 calibration seed; allowed={CALIBRATION_SEEDS}, "
            f"development_reserved={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    return seed


def _train_standard_sources(
    gate: SourceMonitorCoresidencyGateV3,
    patterns: Sequence[Sequence[int]],
) -> None:
    gate.experience(patterns[0], visual_activity=True)
    gate.experience(patterns[1], auditory_activity=True)
    gate.experience(patterns[2], corollary_discharge=True)
    gate.experience(patterns[3], visual_activity=True, auditory_activity=True)


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfigV3 | None = None,
    *,
    phase: str = "calibration",
) -> dict:
    """Run one v3 calibration seed and every preregistered control."""

    seed = validate_phase_seed(seed, phase)
    c = config or SourceMonitorConfigV3()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGateV3(seed=seed, config=c)
    homeostasis_lesion = SourceMonitorCoresidencyGateV3(seed=seed, config=c)
    initial = intact.weight_summary()
    initial_thresholds = intact.source_threshold_vector()
    lesion_initial_thresholds = homeostasis_lesion.source_threshold_vector()
    _train_standard_sources(intact, patterns)
    _train_standard_sources(homeostasis_lesion, patterns)
    learned = intact.weight_summary()
    weights_match_before_settling = np.array_equal(
        intact.learned_weight_vector(),
        homeostasis_lesion.learned_weight_vector(),
    )
    intact_settling = intact.settle_homeostasis(
        patterns[:3], updates_enabled=True
    )
    lesion_settling = homeostasis_lesion.settle_homeostasis(
        patterns[:3], updates_enabled=False
    )
    weights_match_after_settling = np.array_equal(
        intact.learned_weight_vector(),
        homeostasis_lesion.learned_weight_vector(),
    )

    seen = intact.recall(patterns[0])
    heard = intact.recall(patterns[1])
    self_generated = intact.recall(patterns[2])
    mixed = intact.recall(patterns[3])
    unseen = intact.recall(patterns[4])
    source_lesion = intact.recall(patterns[0], source_path_lesion=True)
    acc_lesion = intact.recall(patterns[0], acc_lesion=True)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        competition_lesions = {
            "seen": intact.recall(patterns[0]),
            "heard": intact.recall(patterns[1]),
            "self_generated": intact.recall(patterns[2]),
        }
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    homeostasis_lesion_records = {
        "seen": homeostasis_lesion.recall(patterns[0]),
        "heard": homeostasis_lesion.recall(patterns[1]),
        "self_generated": homeostasis_lesion.recall(patterns[2]),
    }

    swapped = SourceMonitorCoresidencyGateV3(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swapped_settling = swapped.settle_homeostasis(
        patterns[:2], updates_enabled=True
    )
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV3(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(
        patterns[0], visual_activity=True, learning_enabled=False
    )
    off_after = learning_off.weight_summary()
    learning_off_settling = learning_off.settle_homeostasis(
        patterns[:1], updates_enabled=True
    )
    off_recall = learning_off.recall(patterns[0])

    margins = {
        "seen": _source_margin(seen, "seen"),
        "heard": _source_margin(heard, "heard"),
        "self_generated": _source_margin(self_generated, "self_generated"),
    }
    lesion_margins = {
        source: _source_margin(homeostasis_lesion_records[source], source)
        for source in SOURCES
    }
    tradeoff = bounded_tradeoff_assessment(margins, lesion_margins)

    intact_source_total = float(sum(seen["source_spikes"].values()))
    lesioned_source_total = float(sum(source_lesion["source_spikes"].values()))
    source_path_fraction = attributable_to(
        "source recall pathway", intact_source_total, lesioned_source_total
    )
    acc_path_fraction = attributable_to(
        "source-to-ACC pathway",
        float(seen["acc_spikes"]),
        float(acc_lesion["acc_spikes"]),
    )
    competition_intact_spikes = float(sum(seen["competition_spikes"].values()))
    competition_lesion_spikes = float(
        sum(competition_lesions["seen"]["competition_spikes"].values())
    )
    mask_is_local = np.array_equal(
        intact.homeostasis_mask(), intact.expected_homeostasis_mask()
    )
    thresholds_initially_match = np.array_equal(
        initial_thresholds, lesion_initial_thresholds
    )

    components = {
        "learned_routes_start_zero": bool(initial["l1"] == 0.0),
        "experience_changes_synaptic_weights": bool(learned["l1"] > initial["l1"]),
        "seen_source_recalled": bool(
            _dominant_source(seen) == "seen" and margins["seen"] > 0.0
        ),
        "heard_source_recalled": bool(
            _dominant_source(heard) == "heard" and margins["heard"] > 0.0
        ),
        "self_source_recalled": bool(
            _dominant_source(self_generated) == "self_generated"
            and margins["self_generated"] > 0.0
        ),
        **tradeoff["components"],
        "source_swap_follows_afferent_activity": bool(
            _dominant_source(swap_zero) == "heard"
            and _dominant_source(swap_one) == "seen"
        ),
        "mixed_source_reinstates_both": bool(
            mixed["source_spikes"]["seen"] > 0.0
            and mixed["source_spikes"]["heard"] > 0.0
        ),
        "source_path_lesion_collapses_recall": bool(
            lesioned_source_total <= 0.10 * max(intact_source_total, 1.0)
        ),
        "source_path_attribution_meets_fixed_floor": bool(
            source_path_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "acc_lesion_preserves_source_and_silences_acc": bool(
            sum(acc_lesion["source_spikes"].values())
            >= 0.90 * intact_source_total
            and acc_lesion["acc_spikes"] == 0.0
        ),
        "acc_path_attribution_meets_fixed_floor": bool(
            acc_path_fraction >= MIN_ATTRIBUTION_FRACTION
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
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0
            and seen["acc_spikes"] > 0.0
        ),
        "competition_circuit_is_active_and_lesionable": bool(
            competition_intact_spikes > 0.0 and competition_lesion_spikes == 0.0
        ),
        "homeostasis_mask_is_source_local": bool(mask_is_local),
        "homeostasis_thresholds_change_and_lesion_stays_fixed": bool(
            intact_settling["threshold_delta_l1"] > 0.0
            and lesion_settling["threshold_delta_l1"] == 0.0
        ),
        "matched_arms_keep_identical_learned_weights": bool(
            weights_match_before_settling
            and weights_match_after_settling
            and intact_settling["learned_weights_unchanged"]
            and lesion_settling["learned_weights_unchanged"]
        ),
    }

    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    expected_regions.update(SOURCE_INTERNEURON.values())
    recall_parameters = list(
        inspect.signature(SourceMonitorCoresidencyGateV3.recall).parameters
    )
    v2 = SourceMonitorConfigV2()
    inherited_v2_frozen = all(
        getattr(c, field_name) == getattr(v2, field_name)
        for field_name in SourceMonitorConfigV2.__dataclass_fields__
    )

    earned = Verdict("source-monitor co-residency v3 calibration")
    earned.require(
        "episode, source, competition, aPFC, and ACC populations share one bridge",
        expected_regions.issubset(region_names),
        expect=True,
    )
    earned.require(
        "recall accepts episode activity without source metadata",
        recall_parameters
        == ["self", "episode_pattern", "source_path_lesion", "acc_lesion"],
        expect=True,
    )
    earned.require(
        "all inherited v2 operating-point fields are frozen",
        inherited_v2_frozen,
        expect=True,
    )
    earned.require(
        "episode patterns are disjoint and fit the declared population",
        len(set().union(*(set(pattern.tolist()) for pattern in patterns)))
        == len(patterns) * c.episode_pattern_size,
        expect=True,
    )
    earned.require(
        "homeostasis mask contains only source-memory neurons",
        mask_is_local,
        expect=True,
    )
    earned.require(
        "matched arm starts from identical source thresholds",
        thresholds_initially_match,
        expect=True,
    )
    earned.require(
        "matched arms retain identical learned source weights",
        weights_match_before_settling and weights_match_after_settling,
        expect=True,
    )
    earned.knob(
        "source homeostasis target rate",
        requested=c.source_homeostasis_target_rate,
        applied=intact.bridge.core_config.homeostasis_target_rate,
    )
    earned.knob(
        "source homeostasis EMA alpha",
        requested=c.source_homeostasis_ema_alpha,
        applied=intact.bridge.core_config.homeostasis_ema_alpha,
    )
    earned.knob(
        "source homeostasis threshold adaptation",
        requested=c.source_homeostasis_adapt_rate,
        applied=intact.bridge.core_config.homeostasis_threshold_adapt_rate,
    )
    earned.reaches(
        "episode-to-source transmission lesion",
        before=intact_source_total,
        after=lesioned_source_total,
    )
    earned.reaches(
        "source-to-ACC transmission lesion",
        before=float(seen["acc_spikes"]),
        after=float(acc_lesion["acc_spikes"]),
    )
    earned.reaches(
        "source-pool competition lesion",
        before=competition_intact_spikes,
        after=competition_lesion_spikes,
    )
    earned.reaches(
        "source-local intrinsic homeostasis",
        before=0.0,
        after=float(intact_settling["threshold_delta_l1"]),
    )
    earned.disabled(
        "STDP, reward modulation, synaptic scaling, short-term plasticity, and structural plasticity",
        why=(
            "v3 isolates zero-initialized Hebbian source association, fixed local "
            "GABA-A competition, and source-region intrinsic homeostasis"
        ),
    )
    decided = earned.decide(go=all(components.values()), verbose=False)
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else "CALIBRATION_PASS" if decided["go"] else "CALIBRATION_FAIL"
    )

    return {
        "seed": seed,
        "phase": phase,
        "status": status,
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "components": components,
        "metrics": {
            "seen_margin": margins["seen"],
            "heard_margin": margins["heard"],
            "self_generated_margin": margins["self_generated"],
            "minimum_source_margin": min(margins.values()),
            "homeostasis_lesion_margins": lesion_margins,
            "homeostasis_margin_gains": tradeoff["margin_gains"],
            "homeostasis_margin_losses": tradeoff["losses"],
            "spendable_margin_surplus": tradeoff["spendable_surplus"],
            "minimum_homeostasis_lesion_margin": tradeoff[
                "minimum_lesion_margin"
            ],
            "intact_source_spikes": intact_source_total,
            "source_lesion_spikes": lesioned_source_total,
            "intact_acc_spikes": float(seen["acc_spikes"]),
            "acc_lesion_spikes": float(acc_lesion["acc_spikes"]),
            "mixed_seen_spikes": float(mixed["source_spikes"]["seen"]),
            "mixed_heard_spikes": float(mixed["source_spikes"]["heard"]),
            "competition_intact_spikes": competition_intact_spikes,
            "competition_lesion_spikes": competition_lesion_spikes,
        },
        "attribution": {
            "source_recall_path": {
                "intact_source_spikes": intact_source_total,
                "lesioned_source_spikes": lesioned_source_total,
                "lesion_delta": intact_source_total - lesioned_source_total,
                "attributable_to": SOURCE_RECALL_GATE,
                "attributable_fraction": source_path_fraction,
            },
            "acc_output_path": {
                "intact_acc_spikes": float(seen["acc_spikes"]),
                "lesioned_acc_spikes": float(acc_lesion["acc_spikes"]),
                "lesion_delta": float(seen["acc_spikes"] - acc_lesion["acc_spikes"]),
                "attributable_to": ACC_GATE,
                "attributable_fraction": acc_path_fraction,
            },
        },
        "records": {
            "seen": seen,
            "heard": heard,
            "self_generated": self_generated,
            "mixed": mixed,
            "unseen": unseen,
            "source_path_lesion": source_lesion,
            "acc_lesion": acc_lesion,
            "competition_lesions": competition_lesions,
            "homeostasis_lesions": homeostasis_lesion_records,
            "swap_pattern_zero": swap_zero,
            "swap_pattern_one": swap_one,
            "learning_off": off_recall,
        },
        "weights": {
            "initial": initial,
            "learned": learned,
            "learning_off_initial": off_initial,
            "learning_off_after": off_after,
            "matched_before_settling": bool(weights_match_before_settling),
            "matched_after_settling": bool(weights_match_after_settling),
        },
        "homeostasis": {
            "intact": intact_settling,
            "lesion": lesion_settling,
            "swapped": swapped_settling,
            "learning_off": learning_off_settling,
            "mask_is_source_local": bool(mask_is_local),
            "initial_thresholds_match": bool(thresholds_initially_match),
        },
        "config": asdict(c),
        "interface_guards": {
            "recall_parameters": recall_parameters,
            "no_source_argument_at_inference": "source"
            not in inspect.signature(SourceMonitorCoresidencyGateV3.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_gain_normalization": False,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
            "homeostasis_uses_local_firing_ema": True,
            "source_region_mask_is_hand_selected": True,
            "homeostasis_schedule_is_host_timed": True,
        },
        "seed_policy": {
            "non_scientific_smoke": SMOKE_SEED,
            "open_phases": list(OPEN_PHASES),
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out": list(HELD_OUT_SEEDS),
        },
        "fixed_criteria": {
            "minimum_source_margin": MIN_SOURCE_MARGIN,
            "minimum_attribution_fraction": MIN_ATTRIBUTION_FRACTION,
            "bounded_loss": "max(0, lesion_margin - intact_margin)",
            "spendable_surplus": "max(0, lesion_margin - minimum_source_margin)",
            "weakest_margin_must_strictly_improve": True,
            "unseen_source_spikes": 0,
            "learning_off_source_spikes": 0,
            "all_structural_and_lesion_components_required": True,
        },
        "honest_scope": (
            "V3 adds region-scoped intrinsic threshold homeostasis to v2's learned source circuit and "
            "fixed GABA-A competition. Sparse episode activity, physical source-afferent identity, "
            "hand-selected homeostasis regions, externally separated learning/rehearsal/measurement, "
            "and competition suppression during rest remain scaffolds. Spike counts, margins, and "
            "thresholds are host-read for evaluation only. No language, confidence, response policy, "
            "natural episode allocation, or complete self-model is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one calibration seed for source-monitor co-residency v3."
    )
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument(
        "--phase",
        choices=("calibration", "development", "held-out"),
        default="calibration",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(args.seed, phase=args.phase)
    print(
        "[source-monitor-coresidency-v3] "
        f"seed={row['seed']} status={row['status']} metrics={row['metrics']} "
        f"components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency-v3] wrote {out_path}", flush=True)
    return 0 if row["status"] == "CALIBRATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
